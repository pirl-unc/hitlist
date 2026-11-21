"""Multi-modality, context-rich, multi-species T-cell immunogenicity bundles (#944, #947).

Extends the pinned v1 ``hitlist-human-tcell-training-v1`` contract without
breaking backward compatibility:

- Routes cell-free TCR-pMHC biophysical assays (SPR, BLI, 3D structure, KD,
  kon/koff) out of cellular immunogenicity tracks into ``biophysics.parquet``.
- Exposes granular biological modalities (``pmhc_multimer``, ``th1_cytokine``,
  ``th2_cytokine``, ``th17_cytokine``, ``treg_regulatory_cytokine``,
  ``cytotoxicity``, ``degranulation``, ``proliferation``, ``activation_aim``).
- Exposes normalized ordinal intensity, subject-level binomial counts
  ``(n_subjects_positive, n_subjects_tested)``, quantitative readouts, priming
  and disease context, ``effector_repertoire_type`` (``endogenous_polyclonal``
  vs ``tcr_clone_or_transduced``), ``antigen_delivery_mode``, ``apc_lineage``,
  ``antigen_category``, and coordinate-verified ``wt_peptide`` for CEDAR
  somatic neoantigens.
- Exposes both strict experimental single-allele ``mhc_allele`` and unified
  condition-level ``mhc_allele_set`` (with provenance, cardinality,
  ``mhc_species``, ``host_species``, and ``is_chimeric_mhc_host``) across human
  and non-human species.
- Supports ``source_verification_mode="reconcile"`` so live refreshed IEDB/CEDAR
  ZIPs record structured provenance instead of aborting on byte-pin drift.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
from collections import Counter, defaultdict
from collections.abc import Collection, Iterator
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from .curation_yaml import load_curation_yaml
from .mhc_genotypes import infer_host_species, resolve_condition_mhc_genotype
from .tcell_assays import (
    CATALOG_PATH,
    SOURCE_VERIFICATION_MODES,
    _iter_assays_with_cells,
)
from .tcell_peptides import CANONICAL_PEPTIDE, HUMAN_TAXON_IRI, QUALITATIVE_RESULTS
from .version import __version__

FORMAT = "hitlist-tcell-immunogenicity-v2"
TASK = "multi_modality_tcell_immunogenicity"

_REQUIRED_PEPTIDE_FIELDS = (
    "Epitope | Source Organism IRI",
    "Epitope | Object Type",
    "Epitope | Name",
    "Epitope | Modified residues",
    "Epitope | Modifications",
    "Assay Antigen | Epitope Relation",
    "Assay Antigen | Name",
    "Assay Antigen | Source Organism IRI",
)

_OPTIONAL_CONTEXT_FIELDS = (
    "Epitope | Source Organism",
    "Epitope | Starting Position",
    "Epitope | Ending Position",
    "Epitope | Related Object",
    "Assay Antigen | Source Organism",
    "Assay | Response Frequency (%)",
    "Host | MHC Present",
    "1st in vivo Process | Process Type",
    "1st in vivo Process | Disease",
    "Effector Cell | TCR Name",
)

_CEDAR_OPTIONAL_FIELDS = ("Epitope | Mutation",)

_ORDINAL_RANK: dict[str, tuple[str, int | None]] = {
    "Negative": ("Negative", 0),
    "Positive-Low": ("Positive-Low", 1),
    "Positive-Intermediate": ("Positive-Intermediate", 2),
    "Positive-High": ("Positive-High", 3),
    "Positive": ("Positive-Ungraded", None),
}

_TH1_CYTOKINES: tuple[tuple[str, str], ...] = (
    ("ifng", "IFNG"),
    ("ifn-g", "IFNG"),
    ("ifn-gamma", "IFNG"),
    ("interferon-gamma", "IFNG"),
    ("tnfa", "TNFA"),
    ("tnf-a", "TNFA"),
    ("tnf-alpha", "TNFA"),
    ("tnf ", "TNFA"),
    ("il-2 ", "IL2"),
    ("il2 ", "IL2"),
    ("granzyme b", "GZMB"),
    ("granzyme", "GZMB"),
    ("perforin", "PRF1"),
    ("gm-csf", "CSF2"),
    ("mip-1", "CCL3_4"),
)

_TH2_CYTOKINES: tuple[tuple[str, str], ...] = (
    ("il-4", "IL4"),
    ("il4", "IL4"),
    ("il-5", "IL5"),
    ("il5", "IL5"),
    ("il-13", "IL13"),
    ("il13", "IL13"),
)

_TH17_CYTOKINES: tuple[tuple[str, str], ...] = (
    ("il-17a", "IL17A"),
    ("il-17f", "IL17F"),
    ("il-17", "IL17"),
    ("il17", "IL17"),
    ("il-21", "IL21"),
    ("il-22", "IL22"),
)

_TREG_CYTOKINES: tuple[tuple[str, str], ...] = (
    ("il-10", "IL10"),
    ("il10", "IL10"),
    ("tgfb", "TGFB"),
    ("tgf-b", "TGFB"),
    ("tgf-beta", "TGFB"),
)

_MUTATION_SUB_RE = re.compile(r"^(?:p\.)?([ACDEFGHIKLMNPQRSTVWY])(\d+)([ACDEFGHIKLMNPQRSTVWY])$")

_CLONE_TOKENS = (
    "t cell clone",
    "t-cell clone",
    "clone",
    "cloned",
    "transduced",
    "jurkat",
    "hybridoma",
    "tcr-transduced",
    "tcr transduced",
    "tcr engineered",
)

IMMUNOGENICITY_COLUMNS = (
    "example_id",
    "example_unit",
    "task",
    "target_value",
    "ordinal_intensity",
    "ordinal_intensity_rank",
    "source_qualitative_result",
    "label_consistency_status",
    "peptide",
    "wt_peptide",
    "wt_peptide_provenance",
    "cedar_mutation_raw",
    "antigen_format",
    "antigen_delivery_mode",
    "antigen_category",
    "epitope_source_organism",
    "epitope_source_organism_iri",
    "assay_family",
    "immunogenicity_modality",
    "cytokine_or_readout",
    "assay_method",
    "response_measured",
    "quantitative_value",
    "quantitative_measurement",
    "measurement_inequality",
    "measurement_units",
    "n_subjects_tested",
    "n_subjects_positive",
    "reported_response_frequency_pct",
    "in_vivo_process_type",
    "in_vivo_process_type_raw",
    "disease_context",
    "in_vitro_process_type",
    "in_vitro_process_type_raw",
    "effector_repertoire_type",
    "effector_tcr_name",
    "effector_cell_name",
    "effector_culture_condition",
    "apc_lineage",
    "apc_name",
    "apc_culture_condition",
    "mhc_allele",
    "mhc_allele_set",
    "mhc_allele_set_size",
    "mhc_allele_cardinality",
    "mhc_allele_set_provenance",
    "reported_mhc_name",
    "reported_mhc_class",
    "restriction_evidence_code",
    "mhc_assignment_status",
    "is_predicted_restriction",
    "mhc_species",
    "host_species",
    "is_chimeric_mhc_host",
    "host_name",
    "source_assay_id",
    "source_dataset",
    "source_reference_id",
    "reference_pmid",
    "study_group_id",
    "cedar_present",
)

_BOOL_COLS = frozenset(
    {"target_value", "is_predicted_restriction", "is_chimeric_mhc_host", "cedar_present"}
)
_INT_COLS = frozenset(
    {
        "ordinal_intensity_rank",
        "n_subjects_tested",
        "n_subjects_positive",
        "mhc_allele_set_size",
    }
)
_FLOAT_COLS = frozenset({"quantitative_value", "reported_response_frequency_pct"})

IMMUNOGENICITY_SCHEMA = pa.schema(
    [
        (
            name,
            pa.bool_()
            if name in _BOOL_COLS
            else pa.int64()
            if name in _INT_COLS
            else pa.float64()
            if name in _FLOAT_COLS
            else pa.string(),
        )
        for name in IMMUNOGENICITY_COLUMNS
    ]
)

EXCLUDED_SCHEMA = pa.schema(
    [
        ("source_assay_id", pa.string()),
        ("source_reference_id", pa.string()),
        ("assay_family", pa.string()),
        ("reason", pa.string()),
    ]
)

BIOPHYSICS_SCHEMA = pa.schema(
    [
        ("source_assay_id", pa.string()),
        ("source_reference_id", pa.string()),
        ("reference_pmid", pa.string()),
        ("peptide", pa.string()),
        ("reported_mhc_name", pa.string()),
        ("mhc_species", pa.string()),
        ("assay_method", pa.string()),
        ("response_measured", pa.string()),
        ("qualitative_measurement", pa.string()),
        ("quantitative_value", pa.float64()),
        ("quantitative_measurement", pa.string()),
        ("measurement_inequality", pa.string()),
        ("measurement_units", pa.string()),
        ("effector_tcr_name", pa.string()),
        ("cedar_present", pa.bool_()),
    ]
)


def classify_immunogenicity_modality(
    assay_family_name: str, assay_method: str, response_measured: str
) -> tuple[str, str]:
    """Return ``(immunogenicity_modality, cytokine_or_readout)`` for an assay."""
    if assay_family_name == "tcr_pmhc_biophysics":
        return "tcr_pmhc_biophysics", (response_measured or assay_method or "biophysics").strip()
    if assay_family_name == "pmhc_multimer":
        return "pmhc_multimer", "multimer_binding"
    resp = (response_measured or "").strip().casefold()
    meth = (assay_method or "").strip().casefold()
    if assay_family_name == "cytotoxicity" or "cytotox" in resp or "51cr" in meth:
        return "cytotoxicity", "cytotoxicity"
    if "degranulation" in resp or "cd107" in resp or "lamp-1" in resp:
        return "degranulation", "CD107A"
    if "proliferation" in resp or "thymidine" in meth or "cfse" in meth or "brdu" in meth:
        return "proliferation", "proliferation"
    if (
        "activation" in resp
        or "aim" in meth
        or any(marker in resp for marker in ("cd69", "cd137", "4-1bb", "ox40", "cd154", "cd40l"))
    ):
        return "activation_aim", "activation"
    if resp.endswith(" release") or assay_family_name == "soluble_factor_release":
        padded = f"{resp} "
        for token, canonical in _TH1_CYTOKINES:
            if token in padded:
                return "th1_cytokine", canonical
        for token, canonical in _TH2_CYTOKINES:
            if token in padded:
                return "th2_cytokine", canonical
        for token, canonical in _TH17_CYTOKINES:
            if token in padded:
                return "th17_cytokine", canonical
        for token, canonical in _TREG_CYTOKINES:
            if token in padded:
                return "treg_regulatory_cytokine", canonical
        return "other_cytokine", resp.removesuffix(" release").strip() or "cytokine"
    return "other_cellular_tcell", resp or "other"


def classify_in_vivo_process(process_type: str, disease: str) -> str:
    """Normalize ``1st in vivo Process | Process Type`` + disease context."""
    proc = (process_type or "").strip().casefold()
    dis = (disease or "").strip().casefold()
    if not proc and not dis:
        return "unreported"
    if "no immunization" in proc or "unprimed" in proc or "naive" in proc:
        return "no_immunization"
    if "prophylactic vaccination" in proc or "vaccin" in proc:
        return "vaccination"
    if "cancer" in proc or "tumor" in proc or "carcinoma" in dis or "melanoma" in dis:
        return "cancer"
    if "infectious disease" in proc or "infection" in proc:
        return "infectious_disease"
    if "allergy" in proc or "allergic" in proc:
        return "allergy"
    if "autoimmun" in proc or "autoimmun" in dis or "diabetes" in dis:
        return "autoimmunity"
    if "environmental exposure" in proc:
        return "environmental_exposure"
    if "administration in vivo" in proc:
        return "administration_in_vivo"
    return "other_in_vivo"


def classify_in_vitro_process(in_vitro_process: str) -> str:
    """Normalize ``In vitro Process | Process Type`` into priming vs recall vs ex-vivo."""
    proc = (in_vitro_process or "").strip().casefold()
    if not proc:
        return "ex_vivo_direct"
    if "primary induction" in proc or "primary priming" in proc:
        return "primary_induction_in_vitro"
    if "restimulation" in proc:
        return "restimulation_in_vitro"
    return "other_in_vitro"


def classify_effector_repertoire(
    effector_tcr_name: str, effector_cell_name: str, effector_culture_condition: str
) -> str:
    """Separate endogenous polyclonal repertoires from TCR clones/transduced lines (#947)."""
    if (effector_tcr_name or "").strip():
        return "tcr_clone_or_transduced"
    combined = f"{effector_cell_name or ''} {effector_culture_condition or ''}".casefold()
    if any(token in combined for token in _CLONE_TOKENS):
        return "tcr_clone_or_transduced"
    return "endogenous_polyclonal"


def classify_antigen_delivery_mode(
    antigen_object_type: str, epitope_relation: str, apc_culture_condition: str
) -> str:
    """Distinguish exogenous synthetic peptide pulsing from endogenous antigen processing."""
    obj = (antigen_object_type or "").strip().casefold()
    rel = (epitope_relation or "").strip().casefold()
    apc_cond = (apc_culture_condition or "").strip().casefold()
    if any(
        token in obj or token in apc_cond
        for token in (
            "minigene",
            "mrna",
            "dna",
            "transfect",
            "transduc",
            "infected",
            "tumor cell",
            "endogenous",
        )
    ):
        return "endogenous_processed"
    if obj == "linear peptide" and rel in {"epitope", "analog", ""}:
        return "exogenous_peptide"
    if obj in {"protein", "organism", "whole protein", "peptide pool", "cell", "lysate"} or rel in {
        "source antigen",
        "source organism",
    }:
        return "whole_protein_or_organism"
    if obj == "linear peptide":
        return "exogenous_peptide"
    return "other"


def classify_apc_lineage(apc_name: str) -> str:
    """Normalize ``Antigen Presenting Cell | Name`` into controlled APC lineages."""
    name = (apc_name or "").strip().casefold()
    if not name:
        return "unreported"
    if "pbmc" in name or "peripheral blood mononuclear" in name:
        return "pbmc"
    if "splenocyte" in name or "spleen" in name:
        return "splenocyte"
    if "ebv" in name or "lcl" in name or "721.221" in name or "jy" in name:
        return "ebv_lcl"
    if "dendritic" in name or re.search(r"\b(mo)?dc\b", name):
        return "dc"
    if "macrophage" in name:
        return "macrophage"
    if "monocyte" in name:
        return "monocyte"
    if re.search(r"\bt2\b", name):
        return "t2"
    if "k562" in name or "k-562" in name:
        return "k562"
    if "c1r" in name:
        return "c1r"
    if "b cell" in name or "b-cell" in name or "b lymphocyte" in name or "raji" in name:
        return "b_cell"
    if any(
        tok in name
        for tok in ("tumor", "melanoma", "carcinoma", "hela", "a549", "el4", "el-4", "p815")
    ):
        return "tumor_cell"
    return "other"


def classify_antigen_category(
    *,
    cedar_present: bool,
    cedar_mutation: str,
    epitope_origin_iri: str,
    epitope_organism: str,
    in_vivo_category: str,
    disease_context: str,
) -> str:
    """Classify antigen biological category (neoantigen, cancer, pathogen, allergen, self)."""
    if (cedar_mutation or "").strip():
        return "mutated_neoantigen"
    org = (epitope_organism or "").strip().casefold()
    dis = (disease_context or "").strip().casefold()
    is_human_origin = epitope_origin_iri == HUMAN_TAXON_IRI or "homo sapiens" in org
    if cedar_present:
        if not is_human_origin and any(
            v in org for v in ("virus", "hpv", "ebv", "hbv", "hcv", "merkel", "herpes")
        ):
            return "viral_oncoantigen"
        return "cancer_associated"
    if in_vivo_category == "allergy" or "allerg" in dis:
        return "allergen"
    if in_vivo_category == "autoimmunity" or (is_human_origin and "autoimmun" in dis):
        return "autoantigen"
    if is_human_origin:
        if in_vivo_category == "cancer":
            return "cancer_associated"
        return "human_self_unmutated"
    if org or epitope_origin_iri:
        return "pathogen_or_nonhuman"
    return "other"


def _derive_wt_peptide(
    peptide: str,
    *,
    cedar_mutation: str,
    start_pos: str,
    end_pos: str,
    related_object: str,
) -> tuple[str, str]:
    """Derive wild-type counterpart strictly when explicitly reported or coordinate-verified."""
    related = (related_object or "").strip().upper()
    if related and related != peptide and CANONICAL_PEPTIDE.fullmatch(related):
        return related, "reported_related_object"
    mut = (cedar_mutation or "").strip()
    if mut and start_pos.isdecimal() and end_pos.isdecimal():
        match = _MUTATION_SUB_RE.fullmatch(mut)
        if match:
            ref_aa, pos_str, alt_aa = match.groups()
            pos = int(pos_str)
            start = int(start_pos)
            end = int(end_pos)
            idx = pos - start
            if (
                start <= pos <= end
                and end - start + 1 == len(peptide)
                and 0 <= idx < len(peptide)
                and peptide[idx] == alt_aa
                and ref_aa != alt_aa
            ):
                wt = peptide[:idx] + ref_aa + peptide[idx + 1 :]
                return wt, "reconstructed_from_cedar_mutation_coords"
    return "", ""


def _parse_optional_int(value: str) -> tuple[int | None, bool]:
    if not value:
        return None, True
    if not value.isascii() or not value.isdecimal():
        return None, False
    number = int(value)
    if number > 2**63 - 1:
        return None, False
    return number, True


def _parse_optional_float(value: str) -> float | None:
    if not value:
        return None
    cleaned = value.strip().lstrip("<>=").strip()
    try:
        parsed = float(cleaned)
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def _file_digest(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return {"sha256": digest.hexdigest(), "size_bytes": path.stat().st_size}


def _flush(writer, path: Path, rows: list[dict], schema: pa.Schema):
    if writer is None:
        writer = pq.ParquetWriter(path, schema, compression="zstd")
    writer.write_table(pa.Table.from_pylist(rows, schema=schema))
    rows.clear()
    return writer


def _species_matches(
    species_filter: str | Collection[str] | None, *, host_species: str, mhc_species: str
) -> bool:
    if species_filter is None or species_filter == "all":
        return True
    if isinstance(species_filter, str):
        targets = {species_filter.strip().casefold()}
    else:
        targets = {str(s).strip().casefold() for s in species_filter if str(s).strip()}
    if not targets or "all" in targets:
        return True
    if "human" in targets:
        targets.add("homo sapiens")
    if "mouse" in targets:
        targets.add("mus musculus")
    candidates = {s.casefold() for s in (host_species, mhc_species) if s}
    return bool(candidates & targets)


def iter_tcell_immunogenicity_assays(
    *,
    species_filter: str | Collection[str] | None = "all",
    include_biophysics: bool = False,
    iedb_path=None,
    cedar_path=None,
    data_root=None,
    catalog_path=None,
    source_verification_mode: str = "pinned",
    reconciliation_out: dict | None = None,
) -> Iterator[dict]:
    """Yield enriched, multi-modality, multi-species T-cell immunogenicity rows."""
    for metadata, cells in _iter_assays_with_cells(
        iedb_path=iedb_path,
        cedar_path=cedar_path,
        data_root=data_root,
        catalog_path=catalog_path,
        extra_fields=_REQUIRED_PEPTIDE_FIELDS,
        optional_fields=_OPTIONAL_CONTEXT_FIELDS,
        cedar_extra_fields=_CEDAR_OPTIONAL_FIELDS,
        source_verification_mode=source_verification_mode,
        reconciliation_out=reconciliation_out,
    ):
        family = metadata["assay_family"]
        if family == "tcr_pmhc_biophysics" and not include_biophysics:
            continue
        peptide = cells["Epitope | Name"]
        if (
            cells["Epitope | Object Type"] != "Linear peptide"
            or CANONICAL_PEPTIDE.fullmatch(peptide) is None
            or cells["Epitope | Modified residues"]
            or cells["Epitope | Modifications"]
            or metadata["qualitative_measurement"] not in QUALITATIVE_RESULTS
        ):
            continue

        resolved_mhc = resolve_condition_mhc_genotype(
            reported_mhc_name=cells["MHC Restriction | Name"],
            restriction_evidence_code=metadata["restriction_evidence_code"],
            reported_mhc_class=metadata["reported_mhc_class"],
            host_mhc_types=cells.get("Host | MHC Present", ""),
            apc_name=metadata["apc_name"],
            host_name=metadata["host_name"],
        )
        host_species = resolved_mhc.host_species or infer_host_species(metadata["host_name"])
        if not _species_matches(
            species_filter, host_species=host_species, mhc_species=resolved_mhc.mhc_species
        ):
            continue

        n_tested, tested_ok = _parse_optional_int(metadata["reported_n_subjects_tested"])
        n_positive, pos_ok = _parse_optional_int(metadata["reported_n_subjects_positive"])
        target_value = metadata["qualitative_measurement"] != "Negative"
        if not tested_ok or not pos_ok:
            continue
        if n_tested is not None and n_positive is not None and n_positive > n_tested:
            continue
        if n_positive is not None and target_value != (n_positive > 0):
            continue

        ordinal_label, ordinal_rank = _ORDINAL_RANK[metadata["qualitative_measurement"]]
        modality, readout = classify_immunogenicity_modality(
            family, metadata["assay_method"], metadata["response_measured"]
        )
        in_vivo_raw = cells.get("1st in vivo Process | Process Type", "")
        disease_raw = cells.get("1st in vivo Process | Disease", "")
        in_vivo_norm = classify_in_vivo_process(in_vivo_raw, disease_raw)
        in_vitro_raw = metadata["in_vitro_process_type"]
        in_vitro_norm = classify_in_vitro_process(in_vitro_raw)
        tcr_name = cells.get("Effector Cell | TCR Name", "")
        repertoire_type = classify_effector_repertoire(
            tcr_name, metadata["effector_cell_name"], metadata["effector_culture_condition"]
        )
        delivery_mode = classify_antigen_delivery_mode(
            metadata["antigen_object_type"],
            cells["Assay Antigen | Epitope Relation"],
            metadata["apc_culture_condition"],
        )
        apc_lineage = classify_apc_lineage(metadata["apc_name"])
        cedar_mutation = cells.get("CEDAR::Epitope | Mutation", "")
        wt_peptide, wt_prov = _derive_wt_peptide(
            peptide,
            cedar_mutation=cedar_mutation,
            start_pos=cells.get("Epitope | Starting Position", ""),
            end_pos=cells.get("Epitope | Ending Position", ""),
            related_object=cells.get("Epitope | Related Object", ""),
        )
        antigen_category = classify_antigen_category(
            cedar_present=metadata["cedar_present"],
            cedar_mutation=cedar_mutation,
            epitope_origin_iri=cells["Epitope | Source Organism IRI"],
            epitope_organism=cells.get("Epitope | Source Organism", ""),
            in_vivo_category=in_vivo_norm,
            disease_context=disease_raw,
        )
        mhc_status = (
            "single_allele_experimental_restriction_evidence"
            if resolved_mhc.experimental_mhc_allele is not None
            else "predicted_restriction_only"
            if resolved_mhc.is_predicted_restriction
            else resolved_mhc.mhc_allele_set_provenance
        )
        yield {
            "example_id": f"iedb:tcell:{metadata['source_assay_id']}",
            "example_unit": "assay",
            "task": TASK,
            "target_value": target_value,
            "ordinal_intensity": ordinal_label,
            "ordinal_intensity_rank": ordinal_rank,
            "source_qualitative_result": metadata["qualitative_measurement"],
            "label_consistency_status": (
                "agrees_with_reported_subject_count"
                if n_positive is not None
                else "subject_count_not_reported"
            ),
            "peptide": peptide,
            "wt_peptide": wt_peptide,
            "wt_peptide_provenance": wt_prov,
            "cedar_mutation_raw": cedar_mutation,
            "antigen_format": (
                "direct_linear_peptide"
                if metadata["antigen_object_type"] == "Linear peptide"
                and cells["Assay Antigen | Epitope Relation"] == "Epitope"
                and cells["Assay Antigen | Name"] == peptide
                else "indirect_or_processed_antigen"
            ),
            "antigen_delivery_mode": delivery_mode,
            "antigen_category": antigen_category,
            "epitope_source_organism": cells.get("Epitope | Source Organism", ""),
            "epitope_source_organism_iri": cells["Epitope | Source Organism IRI"],
            "assay_family": family,
            "immunogenicity_modality": modality,
            "cytokine_or_readout": readout,
            "assay_method": metadata["assay_method"],
            "response_measured": metadata["response_measured"],
            "quantitative_value": _parse_optional_float(metadata["quantitative_measurement"]),
            "quantitative_measurement": metadata["quantitative_measurement"],
            "measurement_inequality": metadata["measurement_inequality"],
            "measurement_units": metadata["measurement_units"],
            "n_subjects_tested": n_tested,
            "n_subjects_positive": n_positive,
            "reported_response_frequency_pct": _parse_optional_float(
                cells.get("Assay | Response Frequency (%)", "")
            ),
            "in_vivo_process_type": in_vivo_norm,
            "in_vivo_process_type_raw": in_vivo_raw,
            "disease_context": disease_raw,
            "in_vitro_process_type": in_vitro_norm,
            "in_vitro_process_type_raw": in_vitro_raw,
            "effector_repertoire_type": repertoire_type,
            "effector_tcr_name": tcr_name,
            "effector_cell_name": metadata["effector_cell_name"],
            "effector_culture_condition": metadata["effector_culture_condition"],
            "apc_lineage": apc_lineage,
            "apc_name": metadata["apc_name"],
            "apc_culture_condition": metadata["apc_culture_condition"],
            "mhc_allele": resolved_mhc.experimental_mhc_allele,
            "mhc_allele_set": resolved_mhc.mhc_allele_set,
            "mhc_allele_set_size": resolved_mhc.mhc_allele_set_size,
            "mhc_allele_cardinality": resolved_mhc.mhc_allele_cardinality,
            "mhc_allele_set_provenance": resolved_mhc.mhc_allele_set_provenance,
            "reported_mhc_name": cells["MHC Restriction | Name"],
            "reported_mhc_class": metadata["reported_mhc_class"],
            "restriction_evidence_code": metadata["restriction_evidence_code"],
            "mhc_assignment_status": mhc_status,
            "is_predicted_restriction": resolved_mhc.is_predicted_restriction,
            "mhc_species": resolved_mhc.mhc_species,
            "host_species": host_species,
            "is_chimeric_mhc_host": resolved_mhc.is_chimeric_mhc_host,
            "host_name": metadata["host_name"],
            "source_assay_id": metadata["source_assay_id"],
            "source_dataset": "IEDB",
            "source_reference_id": metadata["source_reference_id"],
            "reference_pmid": metadata["reference_pmid"],
            "study_group_id": metadata["source_reference_id"],
            "cedar_present": metadata["cedar_present"],
        }


def write_tcell_immunogenicity_bundle(
    output_dir,
    *,
    species_filter: str | Collection[str] | None = "all",
    iedb_path=None,
    cedar_path=None,
    data_root=None,
    catalog_path=None,
    batch_size: int = 25000,
    source_verification_mode: str = "pinned",
) -> dict:
    """Write multi-modality cellular immunogenicity + separate TCR-pMHC biophysics tables."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if source_verification_mode not in SOURCE_VERIFICATION_MODES:
        raise ValueError(
            f"source_verification_mode must be one of {sorted(SOURCE_VERIFICATION_MODES)}"
        )
    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    catalog = load_curation_yaml(catalog_path or CATALOG_PATH)
    reconciliation: dict = {}

    with tempfile.TemporaryDirectory(prefix=f".{destination.name}-", dir=destination.parent) as tmp:
        staging = Path(tmp)
        training_path = staging / "training.parquet"
        biophysics_path = staging / "biophysics.parquet"
        excluded_path = staging / "excluded.parquet"
        training_writer = biophysics_writer = excluded_writer = None
        training_rows: list[dict] = []
        biophysics_rows: list[dict] = []
        excluded_rows: list[dict] = []

        counts = Counter()
        modality_counts: dict[str, Counter] = defaultdict(Counter)
        species_counts = Counter()
        cardinality_counts = Counter()
        repertoire_counts = Counter()
        exclusion_counts = Counter()
        studies: set[str] = set()

        try:
            for metadata, cells in _iter_assays_with_cells(
                iedb_path=iedb_path,
                cedar_path=cedar_path,
                data_root=data_root,
                catalog_path=catalog_path,
                extra_fields=_REQUIRED_PEPTIDE_FIELDS,
                optional_fields=_OPTIONAL_CONTEXT_FIELDS,
                cedar_extra_fields=_CEDAR_OPTIONAL_FIELDS,
                source_verification_mode=source_verification_mode,
                reconciliation_out=reconciliation,
            ):
                counts["n_source_assays"] += 1
                family = metadata["assay_family"]
                peptide = cells["Epitope | Name"]

                if family == "tcr_pmhc_biophysics":
                    resolved_bio = resolve_condition_mhc_genotype(
                        reported_mhc_name=cells["MHC Restriction | Name"],
                        restriction_evidence_code=metadata["restriction_evidence_code"],
                        reported_mhc_class=metadata["reported_mhc_class"],
                        host_name=metadata["host_name"],
                    )
                    biophysics_rows.append(
                        {
                            "source_assay_id": metadata["source_assay_id"],
                            "source_reference_id": metadata["source_reference_id"],
                            "reference_pmid": metadata["reference_pmid"],
                            "peptide": peptide,
                            "reported_mhc_name": cells["MHC Restriction | Name"],
                            "mhc_species": resolved_bio.mhc_species,
                            "assay_method": metadata["assay_method"],
                            "response_measured": metadata["response_measured"],
                            "qualitative_measurement": metadata["qualitative_measurement"],
                            "quantitative_value": _parse_optional_float(
                                metadata["quantitative_measurement"]
                            ),
                            "quantitative_measurement": metadata["quantitative_measurement"],
                            "measurement_inequality": metadata["measurement_inequality"],
                            "measurement_units": metadata["measurement_units"],
                            "effector_tcr_name": cells.get("Effector Cell | TCR Name", ""),
                            "cedar_present": metadata["cedar_present"],
                        }
                    )
                    counts["n_biophysics_assays"] += 1
                    excluded_rows.append(
                        {
                            "source_assay_id": metadata["source_assay_id"],
                            "source_reference_id": metadata["source_reference_id"],
                            "assay_family": family,
                            "reason": "non_cellular_tcr_pmhc_biophysics",
                        }
                    )
                    exclusion_counts["non_cellular_tcr_pmhc_biophysics"] += 1
                    if len(biophysics_rows) >= batch_size:
                        biophysics_writer = _flush(
                            biophysics_writer, biophysics_path, biophysics_rows, BIOPHYSICS_SCHEMA
                        )
                    if len(excluded_rows) >= batch_size:
                        excluded_writer = _flush(
                            excluded_writer, excluded_path, excluded_rows, EXCLUDED_SCHEMA
                        )
                    continue

                reason = None
                if cells["Epitope | Object Type"] != "Linear peptide":
                    reason = "epitope_not_linear_peptide"
                elif CANONICAL_PEPTIDE.fullmatch(peptide) is None:
                    reason = "noncanonical_or_unknown_sequence"
                elif cells["Epitope | Modified residues"] or cells["Epitope | Modifications"]:
                    reason = "modified_epitope"
                elif metadata["qualitative_measurement"] not in QUALITATIVE_RESULTS:
                    reason = "qualitative_result_unavailable"

                resolved_mhc = None
                host_species = ""
                if reason is None:
                    resolved_mhc = resolve_condition_mhc_genotype(
                        reported_mhc_name=cells["MHC Restriction | Name"],
                        restriction_evidence_code=metadata["restriction_evidence_code"],
                        reported_mhc_class=metadata["reported_mhc_class"],
                        host_mhc_types=cells.get("Host | MHC Present", ""),
                        apc_name=metadata["apc_name"],
                        host_name=metadata["host_name"],
                    )
                    host_species = resolved_mhc.host_species or infer_host_species(
                        metadata["host_name"]
                    )
                    if not _species_matches(
                        species_filter,
                        host_species=host_species,
                        mhc_species=resolved_mhc.mhc_species,
                    ):
                        reason = "filtered_by_species"

                n_tested = n_positive = None
                target_value = metadata["qualitative_measurement"] != "Negative"
                if reason is None:
                    n_tested, tested_ok = _parse_optional_int(
                        metadata["reported_n_subjects_tested"]
                    )
                    n_positive, pos_ok = _parse_optional_int(
                        metadata["reported_n_subjects_positive"]
                    )
                    if not tested_ok or not pos_ok:
                        reason = "invalid_reported_subject_count"
                    elif n_tested is not None and n_positive is not None and n_positive > n_tested:
                        reason = "positive_subjects_exceed_tested"
                    elif n_positive is not None and target_value != (n_positive > 0):
                        reason = "qualitative_subject_count_disagreement"

                if reason is not None:
                    excluded_rows.append(
                        {
                            "source_assay_id": metadata["source_assay_id"],
                            "source_reference_id": metadata["source_reference_id"],
                            "assay_family": family,
                            "reason": reason,
                        }
                    )
                    exclusion_counts[reason] += 1
                    if len(excluded_rows) >= batch_size:
                        excluded_writer = _flush(
                            excluded_writer, excluded_path, excluded_rows, EXCLUDED_SCHEMA
                        )
                    continue

                assert resolved_mhc is not None
                ordinal_label, ordinal_rank = _ORDINAL_RANK[metadata["qualitative_measurement"]]
                modality, readout = classify_immunogenicity_modality(
                    family, metadata["assay_method"], metadata["response_measured"]
                )
                in_vivo_raw = cells.get("1st in vivo Process | Process Type", "")
                disease_raw = cells.get("1st in vivo Process | Disease", "")
                in_vivo_norm = classify_in_vivo_process(in_vivo_raw, disease_raw)
                in_vitro_raw = metadata["in_vitro_process_type"]
                in_vitro_norm = classify_in_vitro_process(in_vitro_raw)
                tcr_name = cells.get("Effector Cell | TCR Name", "")
                repertoire_type = classify_effector_repertoire(
                    tcr_name, metadata["effector_cell_name"], metadata["effector_culture_condition"]
                )
                delivery_mode = classify_antigen_delivery_mode(
                    metadata["antigen_object_type"],
                    cells["Assay Antigen | Epitope Relation"],
                    metadata["apc_culture_condition"],
                )
                apc_lineage = classify_apc_lineage(metadata["apc_name"])
                cedar_mutation = cells.get("CEDAR::Epitope | Mutation", "")
                wt_peptide, wt_prov = _derive_wt_peptide(
                    peptide,
                    cedar_mutation=cedar_mutation,
                    start_pos=cells.get("Epitope | Starting Position", ""),
                    end_pos=cells.get("Epitope | Ending Position", ""),
                    related_object=cells.get("Epitope | Related Object", ""),
                )
                antigen_category = classify_antigen_category(
                    cedar_present=metadata["cedar_present"],
                    cedar_mutation=cedar_mutation,
                    epitope_origin_iri=cells["Epitope | Source Organism IRI"],
                    epitope_organism=cells.get("Epitope | Source Organism", ""),
                    in_vivo_category=in_vivo_norm,
                    disease_context=disease_raw,
                )
                mhc_status = (
                    "single_allele_experimental_restriction_evidence"
                    if resolved_mhc.experimental_mhc_allele is not None
                    else "predicted_restriction_only"
                    if resolved_mhc.is_predicted_restriction
                    else resolved_mhc.mhc_allele_set_provenance
                )
                example = {
                    "example_id": f"iedb:tcell:{metadata['source_assay_id']}",
                    "example_unit": "assay",
                    "task": TASK,
                    "target_value": target_value,
                    "ordinal_intensity": ordinal_label,
                    "ordinal_intensity_rank": ordinal_rank,
                    "source_qualitative_result": metadata["qualitative_measurement"],
                    "label_consistency_status": (
                        "agrees_with_reported_subject_count"
                        if n_positive is not None
                        else "subject_count_not_reported"
                    ),
                    "peptide": peptide,
                    "wt_peptide": wt_peptide,
                    "wt_peptide_provenance": wt_prov,
                    "cedar_mutation_raw": cedar_mutation,
                    "antigen_format": (
                        "direct_linear_peptide"
                        if metadata["antigen_object_type"] == "Linear peptide"
                        and cells["Assay Antigen | Epitope Relation"] == "Epitope"
                        and cells["Assay Antigen | Name"] == peptide
                        else "indirect_or_processed_antigen"
                    ),
                    "antigen_delivery_mode": delivery_mode,
                    "antigen_category": antigen_category,
                    "epitope_source_organism": cells.get("Epitope | Source Organism", ""),
                    "epitope_source_organism_iri": cells["Epitope | Source Organism IRI"],
                    "assay_family": family,
                    "immunogenicity_modality": modality,
                    "cytokine_or_readout": readout,
                    "assay_method": metadata["assay_method"],
                    "response_measured": metadata["response_measured"],
                    "quantitative_value": _parse_optional_float(
                        metadata["quantitative_measurement"]
                    ),
                    "quantitative_measurement": metadata["quantitative_measurement"],
                    "measurement_inequality": metadata["measurement_inequality"],
                    "measurement_units": metadata["measurement_units"],
                    "n_subjects_tested": n_tested,
                    "n_subjects_positive": n_positive,
                    "reported_response_frequency_pct": _parse_optional_float(
                        cells.get("Assay | Response Frequency (%)", "")
                    ),
                    "in_vivo_process_type": in_vivo_norm,
                    "in_vivo_process_type_raw": in_vivo_raw,
                    "disease_context": disease_raw,
                    "in_vitro_process_type": in_vitro_norm,
                    "in_vitro_process_type_raw": in_vitro_raw,
                    "effector_repertoire_type": repertoire_type,
                    "effector_tcr_name": tcr_name,
                    "effector_cell_name": metadata["effector_cell_name"],
                    "effector_culture_condition": metadata["effector_culture_condition"],
                    "apc_lineage": apc_lineage,
                    "apc_name": metadata["apc_name"],
                    "apc_culture_condition": metadata["apc_culture_condition"],
                    "mhc_allele": resolved_mhc.experimental_mhc_allele,
                    "mhc_allele_set": resolved_mhc.mhc_allele_set,
                    "mhc_allele_set_size": resolved_mhc.mhc_allele_set_size,
                    "mhc_allele_cardinality": resolved_mhc.mhc_allele_cardinality,
                    "mhc_allele_set_provenance": resolved_mhc.mhc_allele_set_provenance,
                    "reported_mhc_name": cells["MHC Restriction | Name"],
                    "reported_mhc_class": metadata["reported_mhc_class"],
                    "restriction_evidence_code": metadata["restriction_evidence_code"],
                    "mhc_assignment_status": mhc_status,
                    "is_predicted_restriction": resolved_mhc.is_predicted_restriction,
                    "mhc_species": resolved_mhc.mhc_species,
                    "host_species": host_species,
                    "is_chimeric_mhc_host": resolved_mhc.is_chimeric_mhc_host,
                    "host_name": metadata["host_name"],
                    "source_assay_id": metadata["source_assay_id"],
                    "source_dataset": "IEDB",
                    "source_reference_id": metadata["source_reference_id"],
                    "reference_pmid": metadata["reference_pmid"],
                    "study_group_id": metadata["source_reference_id"],
                    "cedar_present": metadata["cedar_present"],
                }
                training_rows.append(example)
                counts["n_training_examples"] += 1
                counts["n_positive_examples"] += int(target_value)
                counts["n_examples_with_experimental_mhc"] += int(
                    resolved_mhc.experimental_mhc_allele is not None
                )
                counts["n_examples_with_mhc_allele_set"] += int(
                    resolved_mhc.mhc_allele_set_size > 0
                )
                counts["n_chimeric_mhc_host_examples"] += int(resolved_mhc.is_chimeric_mhc_host)
                modality_counts[modality]["n_examples"] += 1
                modality_counts[modality]["n_positive_examples"] += int(target_value)
                species_counts[host_species or "unknown"] += 1
                cardinality_counts[resolved_mhc.mhc_allele_cardinality] += 1
                repertoire_counts[repertoire_type] += 1
                studies.add(example["study_group_id"])
                if len(training_rows) >= batch_size:
                    training_writer = _flush(
                        training_writer, training_path, training_rows, IMMUNOGENICITY_SCHEMA
                    )

            if training_rows or training_writer is None:
                training_writer = _flush(
                    training_writer, training_path, training_rows, IMMUNOGENICITY_SCHEMA
                )
            if biophysics_rows or biophysics_writer is None:
                biophysics_writer = _flush(
                    biophysics_writer, biophysics_path, biophysics_rows, BIOPHYSICS_SCHEMA
                )
            if excluded_rows or excluded_writer is None:
                excluded_writer = _flush(
                    excluded_writer, excluded_path, excluded_rows, EXCLUDED_SCHEMA
                )
        finally:
            for writer in (training_writer, biophysics_writer, excluded_writer):
                if writer is not None:
                    writer.close()

        files = {
            name: _file_digest(staging / name)
            for name in ("training.parquet", "biophysics.parquet", "excluded.parquet")
        }
        manifest = {
            "format": FORMAT,
            "schema_version": 2,
            "hitlist_version": __version__,
            "source_observed": catalog["observed"],
            "sources": (
                reconciliation["actual_sources"]
                if source_verification_mode == "reconcile"
                else catalog["files"]
            ),
            "source_verification_mode": source_verification_mode,
            "species_filter": (
                species_filter
                if isinstance(species_filter, str) or species_filter is None
                else sorted(species_filter)
            ),
            "files": files,
            "n_source_assays": counts["n_source_assays"],
            "n_training_examples": counts["n_training_examples"],
            "n_positive_examples": counts["n_positive_examples"],
            "n_negative_examples": counts["n_training_examples"] - counts["n_positive_examples"],
            "n_biophysics_assays": counts["n_biophysics_assays"],
            "n_excluded_assays": sum(exclusion_counts.values()),
            "exclusion_counts": dict(sorted(exclusion_counts.items())),
            "n_examples_with_experimental_mhc": counts["n_examples_with_experimental_mhc"],
            "n_examples_with_mhc_allele_set": counts["n_examples_with_mhc_allele_set"],
            "n_chimeric_mhc_host_examples": counts["n_chimeric_mhc_host_examples"],
            "modality_coverage": {
                name: {
                    "n_examples": value["n_examples"],
                    "n_positive_examples": value["n_positive_examples"],
                    "n_negative_examples": value["n_examples"] - value["n_positive_examples"],
                }
                for name, value in sorted(modality_counts.items())
            },
            "host_species_coverage": dict(sorted(species_counts.items())),
            "mhc_cardinality_coverage": dict(sorted(cardinality_counts.items())),
            "repertoire_coverage": dict(sorted(repertoire_counts.items())),
            "n_study_groups": len(studies),
            "target_policy": TASK,
            "biophysics_policy": "separated_from_cellular_immunogenicity_training_track",
        }
        if source_verification_mode == "reconcile":
            manifest["source_reconciliation"] = reconciliation
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        os.replace(staging, destination)
    return manifest


def verify_tcell_immunogenicity_bundle(directory, *, catalog_path=None) -> dict:
    """Verify file hashes, schemas, row counts, and source provenance of a v2 bundle."""
    root = Path(directory)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("format") != FORMAT or manifest.get("schema_version") != 2:
        raise ValueError("Unknown T-cell immunogenicity bundle format")
    catalog = load_curation_yaml(catalog_path or CATALOG_PATH)
    mode = manifest.get("source_verification_mode", "pinned")
    if mode == "pinned":
        if manifest.get("sources") != catalog["files"]:
            raise ValueError("T-cell immunogenicity source pins differ from catalog")
    elif mode == "reconcile":
        reconciliation = manifest.get("source_reconciliation")
        if not isinstance(reconciliation, dict) or manifest.get("sources") != reconciliation.get(
            "actual_sources"
        ):
            raise ValueError(
                "Reconciled T-cell immunogenicity bundle missing source_reconciliation"
            )
    else:
        raise ValueError(f"Unknown source_verification_mode: {mode}")

    expected_files = {"training.parquet", "biophysics.parquet", "excluded.parquet"}
    if set(manifest.get("files", {})) != expected_files:
        raise ValueError("T-cell immunogenicity bundle file inventory differs")
    for name, expected in manifest["files"].items():
        if _file_digest(root / name) != expected:
            raise ValueError(f"T-cell immunogenicity bundle file hash differs: {name}")

    training = pq.read_table(root / "training.parquet")
    biophysics = pq.read_table(root / "biophysics.parquet")
    excluded = pq.read_table(root / "excluded.parquet")
    if (
        training.schema != IMMUNOGENICITY_SCHEMA
        or biophysics.schema != BIOPHYSICS_SCHEMA
        or excluded.schema != EXCLUDED_SCHEMA
    ):
        raise ValueError("T-cell immunogenicity bundle schema differs")
    if (
        training.num_rows != manifest["n_training_examples"]
        or biophysics.num_rows != manifest["n_biophysics_assays"]
        or excluded.num_rows != manifest["n_excluded_assays"]
    ):
        raise ValueError("T-cell immunogenicity bundle row count differs")
    if training.num_rows + excluded.num_rows != manifest["n_source_assays"]:
        raise ValueError("T-cell immunogenicity routing omitted source assays")
    if any(fam == "tcr_pmhc_biophysics" for fam in training.column("assay_family").to_pylist()):
        raise ValueError("Non-cellular biophysics rows leaked into cellular training.parquet")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--species", default="all")
    parser.add_argument("--iedb", type=Path)
    parser.add_argument("--cedar", type=Path)
    parser.add_argument(
        "--verification-mode",
        choices=sorted(SOURCE_VERIFICATION_MODES),
        default="pinned",
    )
    args = parser.parse_args()
    manifest = write_tcell_immunogenicity_bundle(
        args.output_dir,
        species_filter=args.species,
        iedb_path=args.iedb,
        cedar_path=args.cedar,
        source_verification_mode=args.verification_mode,
    )
    print(
        json.dumps(
            {
                key: manifest[key]
                for key in (
                    "n_training_examples",
                    "n_biophysics_assays",
                    "n_examples_with_mhc_allele_set",
                )
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
