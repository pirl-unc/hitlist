"""Independent ERAP1 biochemical substrate/product assay inventory & qualification (#809).

Implements #809:
- Pins verified source availability and audit receipts for Maben et al. 2021 (DOI
  10.1038/s41467-021-25564-w, CC BY 4.0 Source Data ZIP SHA-256 ``a9d7bcba...``), Chang et al.
  2005 (PMID 16286653 / PMC1287962, PDF SI only), and Evnouchidou et al. 2008 (PMID 18987748,
  graphs/aggregates only).
- Keeps biochemical endpoint families strictly separate (initial rate vs endpoint fraction vs
  kinetic parameter vs intermediate-product time course vs small-molecule reporter vs structural),
  preserves missing cells distinct from numeric zeros, retains values >100 without clipping into
  probabilities, forbids expanding mixed-library aggregates into individual substrate measurements,
  and marks figure digitizations distinct from deposited replicate tables.
"""

from __future__ import annotations

import hashlib
import json
import math
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd

_DATA_PATH = (
    Path(__file__).parent / "data" / "evidence_revisions" / "erap1_biochemical_inventory.json"
)


def erap1_biochemical_inventory_contract() -> dict[str, Any]:
    """Return the pinned independent ERAP1 biochemical assay inventory contract (#809)."""
    return json.loads(_DATA_PATH.read_text(encoding="utf-8"))


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_maben_2021_erap1_source(
    dest_dir: str | Path | None = None,
    *,
    force: bool = False,
    offline: bool | None = None,
) -> Path:
    """Download and SHA-256-verify the Maben 2021 ERAP1 Source Data ZIP (479,402 bytes)."""
    from .downloads import data_dir, download_to_file

    asset = erap1_biochemical_inventory_contract()["sources"]["maben_2021"]["archive_asset"]
    target_dir = (
        Path(dest_dir) if dest_dir is not None else data_dir() / "maben_2021_erap1_source_data"
    )
    target_dir.mkdir(parents=True, exist_ok=True)
    return download_to_file(
        asset["url"],
        target_dir / asset["filename"],
        label="Maben2021 ERAP1 Source Data",
        force=force,
        expected_sha256=asset["sha256"],
        expected_size=int(asset["size_bytes"]),
        offline=offline,
    )


def parse_maben_2021_archive(
    zip_path: str | Path,
    *,
    allow_unverified_checksum: bool = False,
) -> dict[str, Any]:
    """Inspect Maben 2021 Source Data ZIP workbooks while enforcing audited curation guards."""
    import openpyxl

    contract = erap1_biochemical_inventory_contract()["sources"]["maben_2021"]
    asset = contract["archive_asset"]
    path = Path(zip_path)
    actual_sha = _sha256_file(path)
    if not allow_unverified_checksum and actual_sha != asset["sha256"]:
        raise ValueError(
            f"SHA-256 mismatch for {path.name}: expected {asset['sha256']}, got {actual_sha}"
        )

    workbook_inventory: dict[str, dict[str, Any]] = {}
    with zipfile.ZipFile(path, "r") as zf:
        for member_name in sorted(zf.namelist()):
            if not member_name.endswith(".xlsx") or member_name.startswith("__MACOSX/"):
                continue
            with zf.open(member_name) as stream:
                wb = openpyxl.load_workbook(stream, read_only=True, data_only=True)
                sheets_summary: dict[str, dict[str, int]] = {}
                for sheet_name in wb.sheetnames:
                    ws = wb[sheet_name]
                    n_pop = 0
                    n_num = 0
                    for row in ws.iter_rows(values_only=True):
                        for val in row:
                            if val is not None and val != "":
                                n_pop += 1
                                if isinstance(val, (int, float)) and not isinstance(val, bool):
                                    n_num += 1
                    sheets_summary[sheet_name] = {
                        "n_populated_cells": n_pop,
                        "n_numeric_cells": n_num,
                    }
                wb.close()
                workbook_inventory[Path(member_name).name] = {
                    "n_sheets": len(sheets_summary),
                    "sheets": sheets_summary,
                }

    return {
        "study_id": "maben_2021",
        "source_asset": path.name,
        "source_sha256": actual_sha,
        "workbooks": workbook_inventory,
        "n_total_sheets": sum(w["n_sheets"] for w in workbook_inventory.values()),
        "passes_sequence_diverse_specificity_training_gate": False,
        "fig2i_duplicate_concentration_guard": contract["audit_summary"][
            "fig2i_duplicate_concentration_warning"
        ],
        "isolated_biochemical_not_apc_concentration": True,
    }


def qualify_erap1_biochemical_records(
    df: pd.DataFrame,
    *,
    study_id: str,
    endpoint_family: str,
    source_kind: str = "deposited_numeric_table",
) -> pd.DataFrame:
    """Attach #809 semantic guards to independent ERAP1 biochemical substrate/product records.

    Required column: ``substrate_sequence``.
    Preserves missing vs zero values, retains raw numeric values > 100 without clipping into
    probabilities, forbids expanding mixed-library aggregates into individual substrate labels,
    and flags figure digitizations separately from deposited numeric replicate tables.
    """
    contract = erap1_biochemical_inventory_contract()
    allowed_endpoints = set(contract["allowed_endpoint_families"])
    allowed_sources = set(contract["allowed_source_kinds"])
    if endpoint_family not in allowed_endpoints:
        raise ValueError(
            f"Unknown endpoint_family {endpoint_family!r}; expected one of {sorted(allowed_endpoints)}"
        )
    if source_kind not in allowed_sources:
        raise ValueError(
            f"Unknown source_kind {source_kind!r}; expected one of {sorted(allowed_sources)}"
        )
    if "substrate_sequence" not in df.columns:
        raise ValueError("Input DataFrame must contain a 'substrate_sequence' column")

    rows: list[dict[str, Any]] = []
    for row_num, (_, rec) in enumerate(df.iterrows(), start=2):
        raw_val = rec.get("value")
        parsed_val: float | None = None
        if raw_val is not None and not pd.isna(raw_val):
            try:
                cand = float(raw_val)
                if math.isfinite(cand):
                    parsed_val = cand
            except (TypeError, ValueError):
                parsed_val = None

        if parsed_val is None:
            value_state = "missing"
        elif parsed_val == 0.0:
            value_state = "zero"
        elif parsed_val > 100.0 and endpoint_family == "endpoint_fraction":
            # Explicitly retain values > 100 unclipped (#809 Fig S8d guard).
            value_state = "positive_above_100_unclipped"
        elif parsed_val > 0.0:
            value_state = "positive"
        else:
            value_state = "negative_numeric"

        raw_mixed = rec.get("is_mixed_library_aggregate", False)
        is_mixed = False if pd.isna(raw_mixed) else bool(raw_mixed)
        raw_mhc = rec.get("mhc_present", False)
        mhc_present = False if pd.isna(raw_mhc) else bool(raw_mhc)

        if endpoint_family in {"structural_not_trimming", "small_molecule_reporter"}:
            eligibility = "excluded_non_peptide_trimming_endpoint"
        elif is_mixed:
            eligibility = "excluded_unresolved_mixed_library_aggregate"
        elif source_kind == "digitized_figure":
            eligibility = "secondary_digitized_figure_not_independent_replicates"
        elif value_state == "missing":
            eligibility = "excluded_missing_value"
        else:
            eligibility = "qualified_isolated_biochemical_record"

        rows.append(
            {
                "study_id": study_id,
                "source_row": row_num,
                "substrate_sequence": str(rec["substrate_sequence"]).strip(),
                "product_sequence": (
                    ""
                    if pd.isna(rec.get("product_sequence"))
                    else str(rec.get("product_sequence")).strip()
                ),
                "enzyme_species": (
                    "Homo sapiens"
                    if pd.isna(rec.get("enzyme_species"))
                    else str(rec.get("enzyme_species")).strip()
                ),
                "endpoint_family": endpoint_family,
                "source_kind": source_kind,
                "value": parsed_val,
                "value_state": value_state,
                "value_clipped_to_probability": False,
                "unit": "" if pd.isna(rec.get("unit")) else str(rec.get("unit")).strip(),
                "is_mixed_library_aggregate": is_mixed,
                "mhc_present": mhc_present,
                "figure_points_are_independent_replicates": False,
                "isolated_biochemical_not_apc_concentration": True,
                "qualification_disposition": eligibility,
            }
        )
    return pd.DataFrame(rows)
