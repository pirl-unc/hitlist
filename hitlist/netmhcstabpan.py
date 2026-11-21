# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Source-qualified NetMHCstabpan 1.0 pMHC-I stability ingestion (#875).

Qualifies the published NetMHCstabpan 1.0 ``Stability.txt`` archive
(Rasmussen et al. 2016, PMID 27402703 / PMC4976001; 28,166 9-mer measurements
across 75 HLA-A/B constructs measured via the Harndahl et al. 2011 125I-b2m
scintillation proximity dissociation assay, PMID 21044632) for Hitlist's
binding/observations indexes and model-training interfaces:

- Acquires ``Stability.txt`` on demand via ``datacache`` with pinned SHA-256
  and byte-size verification rather than bundling non-CC-licensed data in the
  wheel/sdist.
- Preserves source URL, SHA-256, 1-based physical line locator, literal
  reported MHC construct, literal ``Thalf`` string/value in hours, and
  unreported-qualifier provenance.
- Distinguishes literal ``0.00``-hour SPA nondetection/subthreshold records,
  within-window ``(0, 48]``-hour decay fits, and ``> 48``-hour extrapolated
  half-lives without inventing positive floors or artificial ``> 48`` censors.
- Preserves explicit mutant construct identities (``HLA-B*14:01(C67S)``,
  ``HLA-B*14:02(C67S)``, ``HLA-B*39:06(C67S)`` -> canonical ``mhcgnomes``
  mutant alleles; see ``pirl-unc/mhcgnomes#268``), never stripping mutations
  into wild-type HLA-B molecules.
- Separates confirmed DTU IEDB direct-submission duplicates
  (``reference/1028282``..``1028294``, ``submission_id`` ``1000631``..``1000641``)
  from cross-study ``(allele, peptide)`` / value overlap warnings.
"""

from __future__ import annotations

import json
import math
import re
import shutil
from collections import Counter
from pathlib import Path

import mhcgnomes
import numpy as np
import pandas as pd

from .assays import assay_annotations
from .curation import (
    classify_allele_resolution,
    classify_ms_row,
    expand_allele_set,
    is_non_peptide_ligand,
    resolve_allele_identity,
    resolve_mhc_annotation,
    restriction_evidence_for_row,
)
from .managed_assets import check_cache_path
from .peptide_modifications import parse_peptide_modifications
from .provenance import file_digest

REVISION = "netmhcstabpan_1_0_stability_v1"
DATASET_NAME = "netmhcstabpan"
CONTRACT_PATH = (
    Path(__file__).parent / "data" / "evidence_revisions" / "netmhcstabpan_stability.json"
)
_MIN_FREE_BYTES = 1024**3
_MUTANT_TOKEN_RE = re.compile(r"\([A-Z]\d+[A-Z]\)|\bmutant\b", re.IGNORECASE)

NETMHCSTABPAN_PROVENANCE_COLUMNS: tuple[str, ...] = (
    "source_url",
    "source_sha256",
    "pinned_source_verified",
    "source_line_number",
    "source_record_id",
    "reported_mhc_construct",
    "construct_resolution_status",
    "is_mutant_mhc",
    "raw_stability_hours",
    "raw_stability_unit",
    "raw_qualifier",
    "qualifier_provenance",
    "measurement_origin_status",
    "stability_value_status",
    "censoring_evidence_status",
    "peptide_chemistry_status",
    "iedb_overlap_status",
    "iedb_overlap_assay_iris",
    "iedb_overlap_reference_iris",
    "wildtype_base_allele_pmhc_in_iedb",
)

IEDB_OVERLAP_STATUSES: tuple[str, ...] = (
    "not_checked",
    "no_iedb_half_life_overlap",
    "confirmed_dtu_iedb_submission_duplicate",
    "confirmed_dtu_iedb_submission_value_conflict",
    "cross_reference_exact_value_overlap_warning",
    "cross_reference_pmhc_different_value_overlap",
    "unresolved_construct_not_matched",
)

_TIME_TO_HOURS: dict[str, float] = {
    "h": 1.0,
    "hr": 1.0,
    "hrs": 1.0,
    "hour": 1.0,
    "hours": 1.0,
    "min": 1.0 / 60.0,
    "mins": 1.0 / 60.0,
    "minute": 1.0 / 60.0,
    "minutes": 1.0 / 60.0,
    "s": 1.0 / 3600.0,
    "sec": 1.0 / 3600.0,
    "secs": 1.0 / 3600.0,
    "second": 1.0 / 3600.0,
    "seconds": 1.0 / 3600.0,
}


def _is_parsed_mutant(parsed: object) -> bool:
    if parsed is None:
        return False
    if bool(getattr(parsed, "mutations", ())):
        return True
    alpha = getattr(parsed, "alpha", None)
    beta = getattr(parsed, "beta", None)
    return bool(getattr(alpha, "mutations", ())) or bool(getattr(beta, "mutations", ()))


def netmhcstabpan_source_contract() -> dict:
    """Return a fresh, validated copy of the NetMHCstabpan 1.0 source contract."""
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    if contract.get("schema_version") != 1 or contract.get("revision") != REVISION:
        raise ValueError("Unsupported NetMHCstabpan stability contract revision")
    source = contract.get("source", {})
    if (
        source.get("dataset") != DATASET_NAME
        or not isinstance(source.get("sha256"), str)
        or len(source["sha256"]) != 64
        or type(source.get("size_bytes")) is not int
        or source["size_bytes"] <= 0
    ):
        raise ValueError("Invalid NetMHCstabpan source identity in contract")
    expected = contract.get("expected_reconciliation", {})
    table2 = contract.get("table2_construct_counts", {})
    if (
        len(table2) != expected.get("n_constructs")
        or sum(table2.values()) != expected.get("n_observations")
        or expected.get("n_positive_rows", 0) + expected.get("n_zero_rows", 0)
        != expected.get("n_observations")
    ):
        raise ValueError("NetMHCstabpan contract Table 2 reconciliation counts disagree")
    for raw_construct, spec in contract.get("curated_mutant_constructs", {}).items():
        canonical = spec.get("canonical_mhc", "")
        parsed = mhcgnomes.parse(canonical, raise_on_error=False)
        if parsed is None or parsed.to_string() != canonical or not _is_parsed_mutant(parsed):
            raise ValueError(
                f"Curated mutant construct {raw_construct!r} -> {canonical!r} is not a canonical mhcgnomes mutant allele"
            )
        # If upstream mhcgnomes#268 adds native parenthesized mutant parsing,
        # require exact agreement with the curated canonical spelling.
        native = mhcgnomes.parse(raw_construct, raise_on_error=False)
        if native is not None and native.to_string() != canonical:
            raise ValueError(
                f"Native mhcgnomes parse of {raw_construct!r} ({native.to_string()!r}) disagrees with contract {canonical!r}"
            )
    return contract


def fetch_netmhcstabpan_source(
    cache_directory: str | Path | None = None,
    *,
    force: bool = False,
    verbose: bool = True,
) -> Path:
    """Fetch checksum-pinned NetMHCstabpan ``Stability.txt`` via ``datacache``.

    When ``cache_directory`` is omitted, delegates to :func:`hitlist.downloads.fetch`
    so the file lands in the standard registered cache path resolved by
    :func:`load_netmhcstabpan_stability` and :func:`scan_netmhcstabpan_stability`.
    """
    from .downloads import download_to_file, fetch

    if cache_directory is None:
        return fetch(DATASET_NAME, force=force)

    contract = netmhcstabpan_source_contract()
    src = contract["source"]
    directory = Path(cache_directory)
    check_cache_path(directory, directory)
    directory.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(directory).free < _MIN_FREE_BYTES:
        raise OSError("NetMHCstabpan source download requires at least 1 GiB free disk reserve")
    dest = directory / src["filename"]
    check_cache_path(directory, dest)
    return download_to_file(
        src["source_url"],
        dest,
        label=DATASET_NAME,
        force=force,
        verbose=verbose,
        expected_sha256=src["sha256"],
        expected_size=src["size_bytes"],
    )


def _resolve_construct(
    raw_construct: str, curated_mutants: dict[str, dict]
) -> tuple[str, str, bool]:
    """Return ``(canonical_or_raw_mhc, construct_resolution_status, is_mutant_mhc)``."""
    cleaned = raw_construct.strip()
    if cleaned in curated_mutants:
        return curated_mutants[cleaned]["canonical_mhc"], "curated_mutant_construct", True
    parsed = mhcgnomes.parse(cleaned, raise_on_error=False)
    if parsed is not None and type(parsed).__name__ in {"Allele", "Pair"}:
        serialized = parsed.to_string()
        if classify_allele_resolution(serialized) == "four_digit":
            is_mut = _is_parsed_mutant(parsed)
            status = "canonical_mutant_construct" if is_mut else "canonical_wildtype_allele"
            return serialized, status, is_mut
    return cleaned, "unresolved_construct", bool(_MUTANT_TOKEN_RE.search(cleaned))


def scan_netmhcstabpan_stability(
    path: str | Path | None = None,
    *,
    provenance=None,
    verify_pinned_digest: bool = True,
    classify_source: bool = True,
) -> pd.DataFrame:
    """Scan NetMHCstabpan ``Stability.txt`` into a binding-index-compatible DataFrame.

    Parameters
    ----------
    path
        Local path to ``Stability.txt``. When ``None``, resolves the dataset
        registered under ``"netmhcstabpan"`` via :func:`hitlist.downloads.get_path`.
    provenance
        Optional :class:`~hitlist.provenance.ContributorCollector` used during
        :func:`~hitlist.builder.build_observations` to capture line-level source
        contributor records.
    verify_pinned_digest
        When ``True`` (default), enforces the exact byte count, SHA-256, and
        75-construct ``PMC4976001`` Table 2 row-count census before returning.
        Set ``False`` only for synthetic unit-test fixtures; ``pinned_source_verified``
        records ``False`` on every resulting row.
    classify_source
        Populate ``classify_ms_row`` source context flags for schema parity with
        :func:`hitlist.scanner.scan` (default ``True``).
    """
    if path is None:
        from .downloads import get_path

        source_path = get_path(DATASET_NAME)
    else:
        source_path = Path(path)
    if not source_path.is_file():
        raise FileNotFoundError(f"NetMHCstabpan Stability.txt not found: {source_path}")

    contract = netmhcstabpan_source_contract()
    src_meta = contract["source"]
    protocol_meta = contract["assay_protocol"]
    curated_mutants = contract["curated_mutant_constructs"]
    expected_digest = {"sha256": src_meta["sha256"], "size_bytes": src_meta["size_bytes"]}

    before = file_digest(source_path)
    if verify_pinned_digest and before != expected_digest:
        raise ValueError(
            f"NetMHCstabpan Stability.txt checksum/size mismatch at {source_path}: "
            f"got {before}, expected {expected_digest}"
        )
    raw_text = source_path.read_bytes().decode("utf-8")
    after = file_digest(source_path)
    if after != before:
        raise ValueError(f"NetMHCstabpan Stability.txt changed during read: {source_path}")

    lines = raw_text.splitlines()
    if not lines:
        raise ValueError(f"Empty NetMHCstabpan file: {source_path}")
    header = lines[0]
    if verify_pinned_digest:
        if header != src_meta["header"]:
            raise ValueError(
                f"Unexpected NetMHCstabpan header {header!r}; expected {src_meta['header']!r}"
            )
    elif header.split() != ["HLA", "Pep", "Thalf"]:
        raise ValueError(
            f"Unexpected NetMHCstabpan header {header!r}; expected 'HLA\\tPep\\tThalf'"
        )

    pinned_verified = before == expected_digest
    pmid = int(src_meta["pmid"])
    source_url = src_meta["source_url"]
    reference_iri = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
    reference_title = src_meta["reference_title"]
    assay_method = protocol_meta["assay_method"]
    response_measured = protocol_meta["response_measured"]
    reported_units = protocol_meta["reported_units"]
    window_hours = float(protocol_meta["reading_window_hours"])
    assay_comments = (
        f"NetMHCstabpan 1.0 Stability.txt (PMID {pmid} / {src_meta['pmcid']}); "
        f"125I-b2m scintillation proximity dissociation assay at 37C "
        f"(PMID {protocol_meta['protocol_pmid']})."
    )

    if provenance is not None:
        provenance.register_source(
            DATASET_NAME,
            source_path,
            description=f"{reference_title} ({source_url})",
        )

    pos_assay_annot = assay_annotations("Positive", assay_comments, assay_method, response_measured)
    neg_assay_annot = assay_annotations("Negative", assay_comments, assay_method, response_measured)

    construct_cache: dict[str, dict] = {}
    seen_pairs: set[tuple[str, str]] = set()
    construct_counts: Counter[str] = Counter()
    n_positive = 0
    n_zero = 0
    n_beyond_48h = 0
    rows: list[dict] = []

    for logical_row, (line_number, line) in enumerate(enumerate(lines[1:], start=2), start=1):
        if not line.strip():
            raise ValueError(
                f"Blank data line in NetMHCstabpan Stability.txt at line {line_number}"
            )
        parts = line.split()
        if len(parts) != 3:
            raise ValueError(
                f"Malformed NetMHCstabpan row at line {line_number} (expected 3 fields, got {len(parts)}): {line!r}"
            )
        raw_hla, raw_pep, raw_thalf = parts
        pair_key = (raw_hla, raw_pep)
        if pair_key in seen_pairs:
            raise ValueError(
                f"Duplicate (HLA, Pep) row {pair_key!r} in NetMHCstabpan Stability.txt at line {line_number}"
            )
        seen_pairs.add(pair_key)

        try:
            val_hours = float(raw_thalf)
        except ValueError as err:
            raise ValueError(
                f"Non-numeric Thalf {raw_thalf!r} in NetMHCstabpan Stability.txt at line {line_number}"
            ) from err
        if not math.isfinite(val_hours) or val_hours < 0.0:
            raise ValueError(
                f"Invalid negative or non-finite Thalf {raw_thalf!r} at line {line_number}"
            )

        construct_counts[raw_hla] += 1
        cmeta = construct_cache.get(raw_hla)
        if cmeta is None:
            mhc_res_input, c_status, is_mut = _resolve_construct(raw_hla, curated_mutants)
            mhc_annot = resolve_mhc_annotation(mhc_res_input, "I", "Homo sapiens")
            mhc_res = mhc_annot.restriction
            non_pep = is_non_peptide_ligand(mhc_res)
            src_flags = (
                classify_ms_row(
                    "",
                    "",
                    "",
                    "",
                    "",
                    pmid,
                    mhc_restriction=mhc_res,
                    mhc_species_context="Homo sapiens",
                    submission_id="",
                    assay_comments=assay_comments,
                )
                if classify_source
                else {}
            )
            res_ev_pos = restriction_evidence_for_row(
                mhc_res,
                pmid=pmid,
                submission_id="",
                is_binding_assay=True,
                is_monoallelic=bool(src_flags.get("is_monoallelic", False)),
                process_type="",
                disease="",
                culture_condition="",
                source_tissue="",
                cell_name="",
                assay_comments=assay_comments,
                assay_method=assay_method,
                response_measured=response_measured,
                qualitative_measurement="Positive",
            )
            res_ev_neg = restriction_evidence_for_row(
                mhc_res,
                pmid=pmid,
                submission_id="",
                is_binding_assay=True,
                is_monoallelic=bool(src_flags.get("is_monoallelic", False)),
                process_type="",
                disease="",
                culture_condition="",
                source_tissue="",
                cell_name="",
                assay_comments=assay_comments,
                assay_method=assay_method,
                response_measured=response_measured,
                qualitative_measurement="Negative",
            )
            allele_set, set_prov, set_size = expand_allele_set(
                mhc_res,
                "",
                pmid,
                mhc_annot.mhc_class,
                frozenset(),
                species_context="Homo sapiens",
            )
            cmeta = {
                "mhc_fields": mhc_annot.as_record_fields(),
                "construct_resolution_status": c_status,
                "is_mutant_mhc": is_mut,
                "is_non_peptide_ligand": non_pep,
                "src_flags": src_flags,
                "res_ev_pos": res_ev_pos,
                "res_ev_neg": res_ev_neg,
                "mhc_allele_set": allele_set,
                "mhc_allele_provenance": set_prov,
                "mhc_allele_set_size": set_size,
            }
            construct_cache[raw_hla] = cmeta

        bare_peptide, peptide_modifications_str, has_ptm, peptide_extended = (
            parse_peptide_modifications(raw_pep)
        )
        if not bare_peptide:
            raise ValueError(f"Empty peptide in NetMHCstabpan Stability.txt at line {line_number}")
        if has_ptm or peptide_modifications_str:
            chem_status = "explicit_modification_annotated"
        elif "C" in bare_peptide:
            chem_status = "synthetic_unannotated_plain_sequence_with_cys"
        else:
            chem_status = "synthetic_unannotated_plain_sequence"

        if val_hours == 0.0:
            n_zero += 1
            qual = "Negative"
            val_status = "zero_hour_nondetection_or_subthreshold"
            censor_ev = "unresolved_assay_floor_zero_reported"
            assay_rec = neg_assay_annot
            res_ev = cmeta["res_ev_neg"]
        else:
            n_positive += 1
            qual = "Positive"
            assay_rec = pos_assay_annot
            res_ev = cmeta["res_ev_pos"]
            if val_hours > window_hours:
                n_beyond_48h += 1
                val_status = "positive_extrapolated_beyond_48h_window"
                censor_ev = "extrapolated_beyond_48h_assay_window_unreported_censor"
            else:
                val_status = "positive_within_48h_window"
                censor_ev = "point_estimate_within_48h_assay_window"

        source_record_id = f"netmhcstabpan-1.0:Stability.txt:line:{line_number}"
        assay_iri = f"{source_url}#line={line_number}"

        prov_id = ""
        if provenance is not None:
            raw_fields = {
                "HLA": raw_hla,
                "Pep": raw_pep,
                "Thalf": raw_thalf,
                "line_number": line_number,
                "pmid": str(pmid),
                "source_url": source_url,
                "source_sha256": before["sha256"],
            }
            rec_id = provenance.record(
                DATASET_NAME, logical_row, raw_fields, [raw_hla, raw_pep, raw_thalf]
            )
            prov_id = provenance.observe(rec_id)

        record = {
            "peptide": bare_peptide,
            "peptide_modifications": peptide_modifications_str,
            "has_ptm": has_ptm,
            "peptide_extended": peptide_extended,
            **cmeta["mhc_fields"],
            "assay_iri": assay_iri,
            "reference_iri": reference_iri,
            "pmid": pmid,
            "submission_id": "",
            "reference_title": reference_title,
            "source_organism": "",
            "species": "",
            "host": "Homo sapiens",
            "host_age": "",
            "host_mhc_types": "",
            "process_type": "",
            "disease": "",
            "disease_stage": "",
            "source_tissue": "",
            "cell_name": "",
            "culture_condition": "",
            "antigen_processing_comments": "",
            "assay_comments": assay_comments,
            "qualitative_measurement": qual,
            "assay_method": assay_method,
            "response_measured": response_measured,
            "measurement_units": reported_units,
            "measurement_inequality": "",
            "quantitative_measurement": raw_thalf,
            "quantitative_value": val_hours,
            **assay_rec,
            "is_non_peptide_ligand": cmeta["is_non_peptide_ligand"],
            **cmeta["src_flags"],
            "restriction_evidence": res_ev,
            "mhc_allele_set": cmeta["mhc_allele_set"],
            "mhc_allele_provenance": cmeta["mhc_allele_provenance"],
            "mhc_allele_set_size": cmeta["mhc_allele_set_size"],
            "attributed_sample_label": "",
            "source": DATASET_NAME,
            "source_url": source_url,
            "source_sha256": before["sha256"],
            "pinned_source_verified": pinned_verified,
            "source_line_number": line_number,
            "source_record_id": source_record_id,
            "reported_mhc_construct": raw_hla,
            "construct_resolution_status": cmeta["construct_resolution_status"],
            "is_mutant_mhc": cmeta["is_mutant_mhc"],
            "raw_stability_hours": val_hours,
            "raw_stability_unit": reported_units,
            "raw_qualifier": "",
            "qualifier_provenance": "unreported_in_source_table",
            "measurement_origin_status": "experimental_screen_or_inhouse_archive",
            "stability_value_status": val_status,
            "censoring_evidence_status": censor_ev,
            "peptide_chemistry_status": chem_status,
            "iedb_overlap_status": "not_checked",
            "iedb_overlap_assay_iris": "",
            "iedb_overlap_reference_iris": "",
            "wildtype_base_allele_pmhc_in_iedb": False,
        }
        if provenance is not None:
            record["provenance_id"] = prov_id
        rows.append(record)

    if verify_pinned_digest:
        exp = contract["expected_reconciliation"]
        if (
            len(rows) != exp["n_observations"]
            or len(construct_counts) != exp["n_constructs"]
            or n_positive != exp["n_positive_rows"]
            or n_zero != exp["n_zero_rows"]
            or n_beyond_48h != exp["n_beyond_48h_rows"]
            or dict(construct_counts) != contract["table2_construct_counts"]
        ):
            raise ValueError(
                "NetMHCstabpan Stability.txt row/construct reconciliation disagreed with PMC4976001 Table 2"
            )

    return pd.DataFrame(rows)


def _strip_iri_origin(iri: str) -> str:
    return re.sub(r"^https?://[^/]+", "", str(iri or "").strip())


def reconcile_netmhcstabpan_with_iedb(
    stab_df: pd.DataFrame,
    binding_df: pd.DataFrame,
    *,
    value_tolerance_hours: float = 0.015,
) -> pd.DataFrame:
    """Annotate NetMHCstabpan rows with assay-lineage vs cross-reference IEDB overlap provenance.

    Confirmed duplicates require matching one of DTU's 10 direct IEDB SPA stability
    submissions (``reference/1028282``..``1028294``, ``submission_id`` ``1000631``..``1000641``)
    within ``value_tolerance_hours`` (default ``0.015`` h = ``0.9`` min, accommodating
    two-decimal hour rounding in ``Stability.txt``). Matches against other IEDB
    references are preserved as explicit overlap warnings, and mutant constructs
    never equate to their wild-type base alleles.
    """
    if value_tolerance_hours < 0 or not math.isfinite(value_tolerance_hours):
        raise ValueError("value_tolerance_hours must be finite and nonnegative")
    result = stab_df.copy()
    if result.empty:
        for col, default in (
            ("iedb_overlap_status", pd.Series(dtype="string")),
            ("iedb_overlap_assay_iris", pd.Series(dtype="string")),
            ("iedb_overlap_reference_iris", pd.Series(dtype="string")),
            ("wildtype_base_allele_pmhc_in_iedb", pd.Series(dtype=bool)),
        ):
            result[col] = default
        return result

    contract = netmhcstabpan_source_contract()
    dtu_ref_paths = {
        _strip_iri_origin(entry["reference_iri"])
        for entry in contract["dtu_iedb_stability_submissions"]
    }
    dtu_sub_ids = {
        str(entry["submission_id"]) for entry in contract["dtu_iedb_stability_submissions"]
    }
    mutant_base_map = {
        raw: spec["wildtype_base_allele"]
        for raw, spec in contract["curated_mutant_constructs"].items()
    }

    if binding_df is None or binding_df.empty:
        unres = (
            result["construct_resolution_status"].astype("string").eq("unresolved_construct")
            if "construct_resolution_status" in result.columns
            else pd.Series(False, index=result.index)
        )
        result["iedb_overlap_status"] = np.where(
            unres, "unresolved_construct_not_matched", "no_iedb_half_life_overlap"
        )
        result["iedb_overlap_assay_iris"] = ""
        result["iedb_overlap_reference_iris"] = ""
        result["wildtype_base_allele_pmhc_in_iedb"] = False
        return result

    # Exclude any already-ingested netmhcstabpan rows so self-reconciliation never self-matches.
    bind = binding_df
    if "source" in bind.columns:
        bind = bind[~bind["source"].astype("string").fillna("").eq(DATASET_NAME)]
    if "response_measured" in bind.columns:
        hl_mask = (
            bind["response_measured"]
            .astype("string")
            .fillna("")
            .str.strip()
            .str.casefold()
            .isin(["half life", "half-life"])
        )
        bind = bind[hl_mask]

    iedb_by_pmhc: dict[tuple[str, str], list[tuple[bool, float, str, str, str]]] = {}
    if not bind.empty:
        b_rest = bind["mhc_restriction"].astype("string").fillna("").str.strip()
        b_id_map = {raw: resolve_allele_identity(raw) for raw in b_rest.unique()}
        b_ids = b_rest.map(b_id_map)
        b_peps = bind["peptide"].astype("string").fillna("").str.strip().str.upper()
        b_vals = (
            pd.to_numeric(bind["quantitative_value"], errors="coerce")
            if "quantitative_value" in bind.columns
            else pd.Series(np.nan, index=bind.index, dtype="float64")
        )
        b_units = (
            bind["measurement_units"].astype("string").fillna("").str.strip()
            if "measurement_units" in bind.columns
            else pd.Series("", index=bind.index, dtype="string")
        )
        b_hours = b_vals * b_units.map(_TIME_TO_HOURS)
        b_assay = (
            bind["assay_iri"].astype("string").fillna("").str.strip()
            if "assay_iri" in bind.columns
            else pd.Series("", index=bind.index, dtype="string")
        )
        b_ref = (
            bind["reference_iri"].astype("string").fillna("").str.strip()
            if "reference_iri" in bind.columns
            else pd.Series("", index=bind.index, dtype="string")
        )
        b_sub = (
            bind["submission_id"].astype("string").fillna("").str.strip()
            if "submission_id" in bind.columns
            else pd.Series("", index=bind.index, dtype="string")
        )
        b_prov = (
            bind["provenance_id"].astype("string").fillna("").str.strip()
            if "provenance_id" in bind.columns
            else pd.Series("", index=bind.index, dtype="string")
        )
        for aid, pep, vh, airi, riri, sub_id, pid in zip(
            b_ids, b_peps, b_hours, b_assay, b_ref, b_sub, b_prov
        ):
            if not aid or not pep:
                continue
            is_dtu = (_strip_iri_origin(riri) in dtu_ref_paths) or (sub_id in dtu_sub_ids)
            iedb_by_pmhc.setdefault((aid, pep), []).append(
                (is_dtu, float(vh) if pd.notna(vh) else np.nan, airi, riri, pid)
            )

    s_rest = result["mhc_restriction"].astype("string").fillna("").str.strip()
    s_id_map = {raw: resolve_allele_identity(raw) for raw in s_rest.unique()}
    s_ids = s_rest.map(s_id_map)
    s_peps = result["peptide"].astype("string").fillna("").str.strip().str.upper()
    s_hours = pd.to_numeric(
        result.get("raw_stability_hours", result.get("quantitative_value")), errors="coerce"
    )
    s_cstatus = (
        result["construct_resolution_status"].astype("string").fillna("")
        if "construct_resolution_status" in result.columns
        else pd.Series("", index=result.index, dtype="string")
    )
    s_reported = (
        result["reported_mhc_construct"].astype("string").fillna("")
        if "reported_mhc_construct" in result.columns
        else s_rest
    )

    statuses: list[str] = []
    assay_iris_out: list[str] = []
    ref_iris_out: list[str] = []
    wt_in_iedb_out: list[bool] = []

    for aid, pep, sh, cstat, rep_construct in zip(s_ids, s_peps, s_hours, s_cstatus, s_reported):
        wt_base = mutant_base_map.get(rep_construct, "")
        wt_in_iedb_out.append(bool(wt_base and (wt_base, pep) in iedb_by_pmhc))
        if cstat == "unresolved_construct" or not aid:
            statuses.append("unresolved_construct_not_matched")
            assay_iris_out.append("")
            ref_iris_out.append("")
            continue
        hits = iedb_by_pmhc.get((aid, pep))
        if not hits:
            statuses.append("no_iedb_half_life_overlap")
            assay_iris_out.append("")
            ref_iris_out.append("")
            continue

        assay_iris_out.append(";".join(sorted({h[2] for h in hits if h[2]})))
        ref_iris_out.append(";".join(sorted({h[3] for h in hits if h[3]})))

        dtu_hits = [h for h in hits if h[0]]
        if dtu_hits:
            if any(
                math.isfinite(h[1]) and abs(h[1] - float(sh)) <= value_tolerance_hours
                for h in dtu_hits
            ):
                statuses.append("confirmed_dtu_iedb_submission_duplicate")
            else:
                statuses.append("confirmed_dtu_iedb_submission_value_conflict")
        else:
            if any(
                math.isfinite(h[1]) and abs(h[1] - float(sh)) <= value_tolerance_hours for h in hits
            ):
                statuses.append("cross_reference_exact_value_overlap_warning")
            else:
                statuses.append("cross_reference_pmhc_different_value_overlap")

    result["iedb_overlap_status"] = pd.Series(statuses, index=result.index, dtype="string")
    result["iedb_overlap_assay_iris"] = pd.Series(
        assay_iris_out, index=result.index, dtype="string"
    )
    result["iedb_overlap_reference_iris"] = pd.Series(
        ref_iris_out, index=result.index, dtype="string"
    )
    result["wildtype_base_allele_pmhc_in_iedb"] = pd.Series(
        wt_in_iedb_out, index=result.index, dtype=bool
    )
    return result


def deduplicate_netmhcstabpan_against_binding(
    stab_df: pd.DataFrame,
    binding_df: pd.DataFrame,
    *,
    provenance=None,
    value_tolerance_hours: float = 0.015,
) -> tuple[pd.DataFrame, dict]:
    """Reconcile against ``binding_df`` and drop only confirmed DTU IEDB submission duplicates.

    When ``provenance`` is provided, redirects every dropped NetMHCstabpan
    ``provenance_id`` to the matching DTU IEDB submission ``provenance_id``
    via ``supplementary_overlap`` so ``observation_contributors.parquet``
    links ``PMID 27402703`` onto those IEDB records. Cross-study pMHC/value
    overlaps are retained with their explicit warning status.
    """
    reconciled = reconcile_netmhcstabpan_with_iedb(
        stab_df, binding_df, value_tolerance_hours=value_tolerance_hours
    )
    status_counts = {
        status: int((reconciled["iedb_overlap_status"] == status).sum())
        for status in IEDB_OVERLAP_STATUSES
        if int((reconciled["iedb_overlap_status"] == status).sum()) > 0
    }
    dup_mask = reconciled["iedb_overlap_status"].eq("confirmed_dtu_iedb_submission_duplicate")
    n_redirect_links = 0

    if (
        dup_mask.any()
        and provenance is not None
        and binding_df is not None
        and not binding_df.empty
        and "provenance_id" in reconciled.columns
        and "provenance_id" in binding_df.columns
    ):
        contract = netmhcstabpan_source_contract()
        dtu_ref_paths = {
            _strip_iri_origin(entry["reference_iri"])
            for entry in contract["dtu_iedb_stability_submissions"]
        }
        dtu_sub_ids = {
            str(entry["submission_id"]) for entry in contract["dtu_iedb_stability_submissions"]
        }
        bind = binding_df
        if "source" in bind.columns:
            bind = bind[~bind["source"].astype("string").fillna("").eq(DATASET_NAME)]
        if "response_measured" in bind.columns:
            bind = bind[
                bind["response_measured"]
                .astype("string")
                .fillna("")
                .str.strip()
                .str.casefold()
                .isin(["half life", "half-life"])
            ]
        r_paths = (
            bind["reference_iri"].astype("string").fillna("").map(_strip_iri_origin)
            if "reference_iri" in bind.columns
            else pd.Series("", index=bind.index)
        )
        s_ids_col = (
            bind["submission_id"].astype("string").fillna("").str.strip()
            if "submission_id" in bind.columns
            else pd.Series("", index=bind.index)
        )
        dtu_bind = bind[r_paths.isin(dtu_ref_paths) | s_ids_col.isin(dtu_sub_ids)]
        target_by_pmhc: dict[tuple[str, str], list[str]] = {}
        if not dtu_bind.empty:
            b_rest = dtu_bind["mhc_restriction"].astype("string").fillna("").str.strip()
            b_id_map = {raw: resolve_allele_identity(raw) for raw in b_rest.unique()}
            for aid, pep, pid in zip(
                b_rest.map(b_id_map),
                dtu_bind["peptide"].astype("string").fillna("").str.strip().str.upper(),
                dtu_bind["provenance_id"].astype("string").fillna("").str.strip(),
            ):
                if aid and pep and pid:
                    target_by_pmhc.setdefault((aid, pep), []).append(pid)

        dups = reconciled.loc[dup_mask]
        d_rest = dups["mhc_restriction"].astype("string").fillna("").str.strip()
        d_id_map = {raw: resolve_allele_identity(raw) for raw in d_rest.unique()}
        for aid, pep, src_pid in zip(
            d_rest.map(d_id_map),
            dups["peptide"].astype("string").fillna("").str.strip().str.upper(),
            dups["provenance_id"].astype("string").fillna("").str.strip(),
        ):
            if not src_pid:
                continue
            for tgt_pid in dict.fromkeys(target_by_pmhc.get((aid, pep), ())):
                if src_pid != tgt_pid:
                    provenance.redirect(src_pid, tgt_pid, "supplementary_overlap")
                    n_redirect_links += 1

    retained = reconciled.loc[~dup_mask].reset_index(drop=True)
    summary = {
        "n_input_rows": len(reconciled),
        "n_retained_rows": len(retained),
        "n_dropped_confirmed_dtu_duplicates": int(dup_mask.sum()),
        "n_provenance_redirect_links": n_redirect_links,
        "by_iedb_overlap_status": status_counts,
    }
    return retained, summary


def load_netmhcstabpan_stability(
    path: str | Path | None = None,
    *,
    binding_df: pd.DataFrame | None = None,
    deduplicate_confirmed_iedb: bool = False,
    allow_download: bool = False,
    verify_pinned_digest: bool = True,
) -> pd.DataFrame:
    """Load source-qualified NetMHCstabpan ``Stability.txt`` rows directly.

    Convenience entry point for downstream consumers (such as Presto ``#138``)
    to load and optionally reconcile/deduplicate NetMHCstabpan half-lives
    against an existing ``binding.parquet`` frame without rebuilding the entire
    observations corpus.
    """
    resolved_path = path
    if resolved_path is None:
        from .downloads import get_path

        try:
            resolved_path = get_path(DATASET_NAME)
        except (KeyError, FileNotFoundError):
            if not allow_download:
                raise
            resolved_path = fetch_netmhcstabpan_source()

    df = scan_netmhcstabpan_stability(
        resolved_path, verify_pinned_digest=verify_pinned_digest, classify_source=True
    )
    if binding_df is not None:
        if deduplicate_confirmed_iedb:
            df, _ = deduplicate_netmhcstabpan_against_binding(df, binding_df)
        else:
            df = reconcile_netmhcstabpan_with_iedb(df, binding_df)
    return df
