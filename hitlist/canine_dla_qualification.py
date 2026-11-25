"""Qualification contracts, vital-organ blacklist guards and offline canine DLA adapters.

Covers:
- #674: Normal heart/brain/lung canine DLA presentation blacklist policy & audit guard
- #676: Lang 2026 splenic DLA-DR (PMID 42752862) supplement provenance & offline MS loader
- #677: Miyamae 2026 ten-allele DLA-88 preprint (10.64898/2026.10.06.756904, cc_no) offline loader
- #666: OncoRef-facing bulk protein-abundance summary guard keeping bulk MS separate from DLA

No restricted or unacquired source payloads are bundled or downloaded automatically.
"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from .provenance import file_digest

CONTRACT_PATH = (
    Path(__file__).parent / "data" / "evidence_revisions" / "canine_dla_atlas_qualification.json"
)
_AA_RE = re.compile(r"^[ACDEFGHIKLMNPQRSTVWY]+$")
_CANINE_SPECIES = frozenset({"canis lupus familiaris", "canis familiaris", "canis lupus", "dog"})
_VALID_MIYAMAE_MEASURED = frozenset({"measured_eluted_ligand"})
_NON_MEASURED_MIYAMAE = frozenset(
    {
        "predicted_binding",
        "motif_neighbour_extrapolation",
        "candidate_selection",
        "normal_expression_screen",
    }
)
_VALID_LANG_MS = frozenset({"measured_ms_eluted_ligand"})
_NON_MS_LANG = frozenset(
    {
        "motif_prediction",
        "protein_count_summary",
        "competition_binding",
        "rt_qpcr_primers",
    }
)


def canine_dla_atlas_qualification() -> dict[str, Any]:
    """Return a validated deep copy of the canine DLA qualification contract."""
    data = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1:
        raise ValueError(
            f"Unsupported canine DLA qualification schema: {data.get('schema_version')}"
        )
    policy = data.get("vital_organ_blacklist_policy", {})
    if policy.get("current_coverage_status") == "missing" and policy.get(
        "validated_empty_blacklist"
    ):
        raise ValueError(
            "Missing canine vital-organ coverage cannot claim validated_empty_blacklist"
        )
    return copy.deepcopy(data)


def canine_dla_source_contract(source_id: str) -> dict[str, Any]:
    """Return the validated qualification contract for a named canine DLA/bulk source."""
    sources = canine_dla_atlas_qualification()["sources"]
    if source_id not in sources:
        raise ValueError(f"Unknown canine DLA source {source_id!r}; available: {sorted(sources)}")
    return copy.deepcopy(sources[source_id])


def _clean_str(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def _parse_bool(value: Any, *, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return default
    text = str(value).strip().casefold()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n", ""}:
        return False
    raise ValueError(f"Cannot interpret boolean value {value!r}")


def _canonical_vital_tissue(raw_tissue: str, aliases: dict[str, str]) -> str:
    token = raw_tissue.strip().casefold()
    return aliases.get(token, "")


def _is_canine_label(value: str) -> bool:
    return value.strip().casefold() in _CANINE_SPECIES


def evaluate_canine_vital_organ_blacklist(
    observations: pd.DataFrame | None = None,
    *,
    donor_aliases: dict[str, str] | None = None,
    min_distinct_dogs: int = 2,
) -> dict[str, Any]:
    """Evaluate #674's canine nonmalignant heart/brain/lung DLA blacklist policy.

    Rules enforced:
    - Exact observed DLA-presented ligands from Canis lupus familiaris MHC only.
    - Nonmalignant primary heart, brain or lung tissue only (spleen, tumors, cell
      lines, engineered hosts, unresolved pools, bulk proteomics, predictions,
      competition assays, nested windows and human HLA are excluded).
    - Deduplicates multiple runs/acquisitions, multiple vital tissues from the same
      animal, and republished/aliased donor IDs (via ``donor_aliases``) so each
      resolved individual dog counts at most once per peptide.
    - Requires ``>= min_distinct_dogs`` resolved dogs independent of allele.
    - When no qualifying dogs exist, emits ``coverage_status="missing"`` and
      ``validated_empty_blacklist=False`` (never an experimentally validated empty
      blacklist).
    """
    if type(min_distinct_dogs) is not int or min_distinct_dogs < 1:
        raise ValueError("min_distinct_dogs must be a positive integer")

    contract = canine_dla_atlas_qualification()
    policy = contract["vital_organ_blacklist_policy"]
    tissue_aliases: dict[str, str] = policy["tissue_aliases"]
    qualifying_tissues = tuple(policy["qualifying_tissues"])
    alias_map = {k.strip(): v.strip() for k, v in (donor_aliases or {}).items()}

    empty_blacklist = pd.DataFrame(
        columns=[
            "peptide",
            "n_dogs",
            "dog_ids",
            "tissues",
            "n_observations",
            "alleles_observed",
            "source_ids",
        ]
    )
    empty_audit = pd.DataFrame(
        columns=[
            "row_index",
            "peptide",
            "canonical_dog_id",
            "canonical_tissue",
            "qualifies",
            "exclusion_reason",
        ]
    )

    if observations is None or len(observations) == 0:
        return {
            "policy_issue": 674,
            "coverage_status": "missing",
            "validated_empty_blacklist": False,
            "min_distinct_dogs": min_distinct_dogs,
            "n_input_rows": 0,
            "n_qualifying_rows": 0,
            "n_qualifying_dogs": 0,
            "n_blacklisted_peptides": 0,
            "n_dogs_by_tissue": dict.fromkeys(qualifying_tissues, 0),
            "n_excluded_by_reason": {},
            "blacklist_df": empty_blacklist,
            "audit_df": empty_audit,
        }

    if not isinstance(observations, pd.DataFrame):
        raise TypeError("observations must be a pandas DataFrame or None")

    audit_rows: list[dict[str, Any]] = []
    qualifying_records: list[dict[str, Any]] = []
    excluded_counts: dict[str, int] = {}

    for row_idx, row in observations.iterrows():
        peptide = _clean_str(row.get("peptide")).upper()
        evidence_kind = _clean_str(row.get("evidence_kind")).casefold()
        match_kind = (_clean_str(row.get("match_kind")) or "exact").casefold()
        mhc_species = _clean_str(row.get("mhc_species")) or _clean_str(row.get("species"))
        tissue_raw = _clean_str(row.get("tissue")) or _clean_str(row.get("source_tissue"))
        tissue_status = _clean_str(row.get("tissue_status")).casefold()
        is_cell_line = _parse_bool(row.get("is_cell_line"), default=False)
        donor_status = _clean_str(row.get("donor_status")).casefold()
        raw_donor_id = _clean_str(row.get("donor_id"))
        canonical_dog_id = alias_map.get(raw_donor_id, raw_donor_id)
        canonical_tissue = _canonical_vital_tissue(tissue_raw, tissue_aliases)
        source_id = _clean_str(row.get("source_id"))
        allele = _clean_str(row.get("mhc_restriction")) or _clean_str(row.get("dla_allele"))

        reason = ""
        if not peptide or not _AA_RE.fullmatch(peptide):
            reason = "invalid_peptide_sequence"
        elif match_kind != "exact":
            reason = "non_exact_or_nested_window"
        elif evidence_kind in {
            "bulk",
            "bulk_proteomics",
            "digest_ms",
            "protein_count_summary",
            "normal_expression_screen",
        }:
            reason = "bulk_or_non_presentation_evidence"
        elif evidence_kind in {
            "binding_prediction",
            "predicted_binding",
            "motif_prediction",
            "motif_neighbour_extrapolation",
            "competition_binding",
            "candidate_selection",
        }:
            reason = "prediction_or_binding_not_elution"
        elif evidence_kind not in {
            "dla_elution",
            "primary_tissue_dla_elution",
            "measured_eluted_ligand",
            "measured_ms_eluted_ligand",
            "ms_elution",
        }:
            reason = f"unsupported_evidence_kind:{evidence_kind or 'missing'}"
        elif not _is_canine_label(mhc_species):
            reason = "non_canine_mhc_species"
        elif is_cell_line or tissue_status in {"cell_line", "engineered_cell_line"}:
            reason = "cell_line_excluded"
        elif tissue_status != "nonmalignant":
            reason = "malignant_or_unresolved_tissue_status"
        elif donor_status != "resolved_individual" or not canonical_dog_id:
            reason = "unresolved_or_pooled_donor"
        elif canonical_tissue not in qualifying_tissues:
            reason = "non_vital_organ_tissue"

        qualifies = reason == ""
        audit_rows.append(
            {
                "row_index": row_idx,
                "peptide": peptide,
                "canonical_dog_id": canonical_dog_id if qualifies else "",
                "canonical_tissue": canonical_tissue,
                "qualifies": qualifies,
                "exclusion_reason": reason,
            }
        )
        if qualifies:
            qualifying_records.append(
                {
                    "peptide": peptide,
                    "dog_id": canonical_dog_id,
                    "tissue": canonical_tissue,
                    "allele": allele,
                    "source_id": source_id,
                }
            )
        else:
            excluded_counts[reason] = excluded_counts.get(reason, 0) + 1

    audit_df = pd.DataFrame(audit_rows)

    if not qualifying_records:
        return {
            "policy_issue": 674,
            "coverage_status": "missing",
            "validated_empty_blacklist": False,
            "min_distinct_dogs": min_distinct_dogs,
            "n_input_rows": len(observations),
            "n_qualifying_rows": 0,
            "n_qualifying_dogs": 0,
            "n_blacklisted_peptides": 0,
            "n_dogs_by_tissue": dict.fromkeys(qualifying_tissues, 0),
            "n_excluded_by_reason": dict(sorted(excluded_counts.items())),
            "blacklist_df": empty_blacklist,
            "audit_df": audit_df,
        }

    qual_df = pd.DataFrame(qualifying_records)
    all_dogs = sorted(qual_df["dog_id"].unique())
    n_dogs_by_tissue = {
        tissue: int(qual_df.loc[qual_df["tissue"] == tissue, "dog_id"].nunique())
        for tissue in qualifying_tissues
    }

    blacklist_rows: list[dict[str, Any]] = []
    for peptide, grp in qual_df.groupby("peptide", sort=True):
        dogs = sorted(grp["dog_id"].unique())
        if len(dogs) >= min_distinct_dogs:
            tissues = sorted(grp["tissue"].unique())
            alleles = sorted(a for a in grp["allele"].unique() if a)
            sources = sorted(s for s in grp["source_id"].unique() if s)
            blacklist_rows.append(
                {
                    "peptide": peptide,
                    "n_dogs": len(dogs),
                    "dog_ids": ";".join(dogs),
                    "tissues": ";".join(tissues),
                    "n_observations": len(grp),
                    "alleles_observed": ";".join(alleles),
                    "source_ids": ";".join(sources),
                }
            )

    blacklist_df = pd.DataFrame(blacklist_rows) if blacklist_rows else empty_blacklist

    all_tissues_covered = all(
        n_dogs_by_tissue[tissue] >= min_distinct_dogs for tissue in qualifying_tissues
    )
    if len(all_dogs) >= min_distinct_dogs and all_tissues_covered:
        coverage_status = "established"
    else:
        coverage_status = "partial"

    return {
        "policy_issue": 674,
        "coverage_status": coverage_status,
        "validated_empty_blacklist": bool(
            coverage_status == "established" and len(blacklist_df) == 0
        ),
        "min_distinct_dogs": min_distinct_dogs,
        "n_input_rows": len(observations),
        "n_qualifying_rows": len(qualifying_records),
        "n_qualifying_dogs": len(all_dogs),
        "n_blacklisted_peptides": len(blacklist_df),
        "n_dogs_by_tissue": n_dogs_by_tissue,
        "n_excluded_by_reason": dict(sorted(excluded_counts.items())),
        "blacklist_df": blacklist_df,
        "audit_df": audit_df,
    }


def _read_local_tabular(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Local canine DLA source file not found: {path}")
    sep = "\t" if path.suffix.casefold() in {".tsv", ".txt"} else ","
    return pd.read_csv(path, sep=sep, dtype=str, keep_default_na=False)


def load_miyamae_2026_dla88_ligands(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
    include_non_measured: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Access-aware offline loader for Miyamae 2026 ten-allele DLA-88 tables (#677).

    Respects bioRxiv ``license=cc_no`` (local file only, never downloads).
    Preserves parental line, construct, residual endogenous MHC, acquisition ID,
    modifications, protein ambiguity and FDR/confidence; separates measured eluted
    ligands from predictions, motif-neighbour extrapolations and expression screens;
    and retains distinct allele/population observations rather than collapsing them.
    """
    contract = canine_dla_source_contract("miyamae_2026_dla88")
    source_path = Path(path)
    digest = file_digest(source_path)
    if expected_sha256 is not None and digest["sha256"] != expected_sha256:
        raise ValueError(
            f"SHA-256 mismatch for {source_path}: got {digest['sha256']}, expected {expected_sha256}"
        )

    raw = _read_local_tabular(source_path)
    required = contract["required_columns"]
    missing_cols = [c for c in required if c not in raw.columns]
    if missing_cols:
        raise ValueError(f"Miyamae 2026 table missing required columns: {missing_cols}")

    records: list[dict[str, Any]] = []
    n_excluded_non_measured = 0
    for idx, row in raw.iterrows():
        evidence_kind = _clean_str(row["evidence_kind"]).casefold()
        if evidence_kind not in (_VALID_MIYAMAE_MEASURED | _NON_MEASURED_MIYAMAE):
            raise ValueError(
                f"Unsupported Miyamae 2026 evidence_kind at row {idx + 1}: {evidence_kind!r}"
            )
        is_measured = evidence_kind in _VALID_MIYAMAE_MEASURED
        if not is_measured and not include_non_measured:
            n_excluded_non_measured += 1
            continue
        peptide = _clean_str(row["peptide"]).upper()
        if not peptide or not _AA_RE.fullmatch(peptide):
            raise ValueError(f"Invalid amino-acid sequence at row {idx + 1}: {row['peptide']!r}")
        dla_allele = _clean_str(row["dla_allele"])
        parental_line = _clean_str(row["parental_line"])
        construct_id = _clean_str(row["construct_id"])
        if not dla_allele or not parental_line or not construct_id:
            raise ValueError(f"Missing allele/cell-context metadata at row {idx + 1}")
        records.append(
            {
                "source_id": contract["source_id"],
                "source_row_index": idx + 1,
                "peptide": peptide,
                "peptide_length": len(peptide),
                "dla_allele": dla_allele,
                "mhc_class": "I",
                "mhc_species": contract["mhc_species"],
                "parental_line": parental_line,
                "construct_id": construct_id,
                "residual_endogenous_mhc": _clean_str(row.get("residual_endogenous_mhc"))
                or "unknown",
                "acquisition_id": _clean_str(row.get("acquisition_id")) or "unknown",
                "modifications_raw": _clean_str(row.get("modifications_raw")),
                "source_proteins_raw": _clean_str(row.get("source_proteins_raw")),
                "confidence_or_fdr": _clean_str(row.get("confidence_or_fdr")) or "unknown",
                "search_reference": _clean_str(row.get("search_reference")) or "unknown",
                "evidence_kind": evidence_kind,
                "is_measured_dla_ligand": is_measured,
                "is_cell_line": True,
                "tissue_status": "malignant_cell_line",
                "donor_status": "cell_line_not_donor",
                "satisfies_vital_organ_blacklist": False,
            }
        )

    df = pd.DataFrame(records)
    measured_df = df[df["is_measured_dla_ligand"]] if not df.empty else df
    summary = {
        "source_id": contract["source_id"],
        "tracked_issue": 677,
        "license": contract["license"],
        "redistributable": False,
        "input_file": str(source_path),
        "input_sha256": digest["sha256"],
        "input_size_bytes": digest["size_bytes"],
        "counting_unit": "allele_population_observation_row",
        "n_input_rows": len(raw),
        "n_observations": len(df),
        "n_measured_observations": len(measured_df),
        "n_excluded_non_measured_rows": int(n_excluded_non_measured),
        "n_unique_peptides": int(measured_df["peptide"].nunique()) if not measured_df.empty else 0,
        "n_alleles": int(measured_df["dla_allele"].nunique()) if not measured_df.empty else 0,
        "satisfies_vital_organ_blacklist": False,
    }
    return df, summary


def load_lang_2026_splenic_dla_dr_ligands(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
    include_non_ms: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Access-aware offline loader for Lang 2026 splenic DLA-DR MS exports (#676).

    Joins verified Table S2 donor metadata (SHA-256 ``c107a758...``) while keeping
    unreported donor health, treatment and DRA typing explicitly ``"unknown"``.
    Rejects non-MS supplement tables (Tables S3/S4 motif predictions, Tables S5/S6
    protein counts, rabies competition binding assays) from naturally presented MS
    observations, preserves all accepted MS observations separately from any motif
    subset (``in_motif_subset``), and enforces ``satisfies_vital_organ_blacklist=False``
    because spleen does not satisfy #674's heart/brain/lung requirement.
    """
    contract = canine_dla_source_contract("lang_2026_splenic_dla_dr")
    donors = contract["verified_table_s2_donors"]
    source_path = Path(path)
    digest = file_digest(source_path)
    if expected_sha256 is not None and digest["sha256"] != expected_sha256:
        raise ValueError(
            f"SHA-256 mismatch for {source_path}: got {digest['sha256']}, expected {expected_sha256}"
        )

    raw = _read_local_tabular(source_path)
    required = contract["required_ms_columns"]
    missing_cols = [c for c in required if c not in raw.columns]
    if missing_cols:
        raise ValueError(f"Lang 2026 MS table missing required columns: {missing_cols}")

    records: list[dict[str, Any]] = []
    n_excluded_non_ms = 0
    for idx, row in raw.iterrows():
        evidence_kind = _clean_str(row["evidence_kind"]).casefold()
        if evidence_kind not in (_VALID_LANG_MS | _NON_MS_LANG):
            raise ValueError(
                f"Unsupported Lang 2026 evidence_kind at row {idx + 1}: {evidence_kind!r}"
            )
        is_ms = evidence_kind in _VALID_LANG_MS
        if not is_ms and not include_non_ms:
            n_excluded_non_ms += 1
            continue
        peptide = _clean_str(row["peptide"]).upper()
        if is_ms and (not peptide or not _AA_RE.fullmatch(peptide)):
            raise ValueError(f"Invalid MS peptide sequence at row {idx + 1}: {row['peptide']!r}")
        sample_id = _clean_str(row["sample_id"])
        if sample_id not in donors:
            raise ValueError(
                f"Unknown Lang 2026 Table S2 sample_id {sample_id!r} at row {idx + 1}; "
                f"expected one of {sorted(donors)}"
            )
        donor = donors[sample_id]
        in_motif_subset = _parse_bool(row.get("in_motif_subset"), default=False)
        records.append(
            {
                "source_id": contract["source_id"],
                "source_record_id": _clean_str(row["source_record_id"]),
                "source_row_index": idx + 1,
                "sample_id": sample_id,
                "donor_id": donor["donor_id"],
                "donor_status": "resolved_individual",
                "age_years": donor["age_years"],
                "reported_sex": donor["reported_sex"],
                "breed": donor["breed"],
                "haplotype_drb1_dqa1_dqb1": donor["haplotype_drb1_dqa1_dqb1"],
                "dra_genotype": donor["dra_genotype"],
                "health_status": donor["health_status"],
                "treatment_status": donor["treatment_status"],
                "tissue": "spleen",
                "tissue_status": "unknown",
                "is_cell_line": False,
                "peptide": peptide,
                "peptide_length": len(peptide),
                "mhc_class": "II",
                "mhc_locus": "DLA-DR",
                "mhc_species": contract["mhc_species"],
                "source_organism": _clean_str(row.get("source_organism"))
                or "Canis lupus familiaris",
                "ip_antibody": contract["ip_antibody"],
                "modifications_raw": _clean_str(row.get("modifications_raw")),
                "source_proteins_raw": _clean_str(row.get("source_proteins_raw")),
                "confidence_or_fdr": _clean_str(row.get("confidence_or_fdr")) or "unknown",
                "search_reference": _clean_str(row.get("search_reference")) or "unknown",
                "evidence_kind": evidence_kind,
                "is_ms_observation": is_ms,
                "in_motif_subset": in_motif_subset,
                "satisfies_vital_organ_blacklist": False,
            }
        )

    df = pd.DataFrame(records)
    ms_df = df[df["is_ms_observation"]] if not df.empty else df
    summary = {
        "source_id": contract["source_id"],
        "tracked_issue": 676,
        "pmid": contract["pmid"],
        "doi": contract["doi"],
        "table_s2_sha256": contract["reviewed_supplement_archive"]["content_inventory"][1][
            "sha256"
        ],
        "input_file": str(source_path),
        "input_sha256": digest["sha256"],
        "input_size_bytes": digest["size_bytes"],
        "n_input_rows": len(raw),
        "n_observations": len(df),
        "n_ms_observations": len(ms_df),
        "n_motif_subset_observations": int(ms_df["in_motif_subset"].sum())
        if not ms_df.empty
        else 0,
        "n_excluded_non_ms_rows": int(n_excluded_non_ms),
        "n_unique_peptides": int(ms_df["peptide"].nunique()) if not ms_df.empty else 0,
        "n_donors": int(ms_df["donor_id"].nunique()) if not ms_df.empty else 0,
        "tissue": "spleen",
        "satisfies_vital_organ_blacklist": False,
    }
    return df, summary


def summarize_canine_bulk_protein_evidence(
    directory: str | Path,
    *,
    tissue: str | None = None,
    max_rows: int = 250000,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Expose OncoRef-ready canine bulk protein abundance facts with explicit scope (#666).

    Preserves stable source IDs, native abundance/iBAQ units, finite/missing/nonfinite
    uncertainty statuses, pool/developmental/health metadata, and explicitly marks
    ``provides_dla_presentation_blacklist=False`` so bulk proteomics can never satisfy #674.
    """
    from .canine_bulk import load_bulk_evidence, verify_bulk_evidence

    root = Path(directory)
    manifest = verify_bulk_evidence(root)
    source = manifest["source"]
    adapter = source["adapter"]

    if manifest["schema_version"] == 1:
        frame = load_bulk_evidence(root, table="proteins", tissue=tissue, max_rows=max_rows)
    else:
        raise ValueError(
            "summarize_canine_bulk_protein_evidence expects a schema-1 canine bulk protein bundle; "
            "use open_canine_peptide_evidence() for schema-2 peptide/protein-group bundles."
        )

    out = frame.copy()
    out["provides_dla_presentation_blacklist"] = False
    out["presentation_coverage"] = manifest["presentation_coverage"]

    n_proteins = int(out["protein_id"].nunique()) if "protein_id" in out and not out.empty else 0
    n_samples = int(out["sample_id"].nunique()) if "sample_id" in out and not out.empty else 0
    summary = {
        "tracked_issue": 666,
        "source_id": source["source_id"],
        "source_version": source["version"],
        "adapter": adapter,
        "source_taxon": source["source_taxon"],
        "study": source["study"],
        "reference_role": source["reference_role"],
        "mapping_background_complete": bool(source["mapping_background_complete"]),
        "presentation_coverage": manifest["presentation_coverage"],
        "provides_dla_presentation_blacklist": False,
        "n_rows": len(out),
        "n_proteins": n_proteins,
        "n_samples": n_samples,
        "donor_statuses": sorted(out["donor_status"].dropna().unique().tolist())
        if "donor_status" in out
        else [],
    }
    return out, summary
