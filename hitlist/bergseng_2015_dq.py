"""Source-qualified adapter for Bergseng 2015 HLA-DQ run-level ion peak volumes (#859).

Source: Bergseng et al. (2015), *Immunogenetics*, PMID 25502872, PMC4297300,
DOI 10.1007/s00251-014-0819-9 (CC-BY 4.0), Supplemental Table 1
(``251_2014_819_MOESM3_ESM.xlsx``).

Preserves:
- All 12,712 peptide rows and all 54 peptide-by-run MS1 ion peak volumes with
  exact Excel cell coordinates (``Sheet1!L3`` etc.) and original headers.
- Raw MS1 ion peak volumes separately from per-run sum-normalized relative
  fractions (0-1 scale); neither is an absolute concentration.
- Explicit zeros (521,368 cells) separately from missing/unmeasured values (0)
  and positive values (165,080), without treating zeros as nonbinders or every
  nonzero as an independent MS/MS identification (3-min RT alignment enabled).
- Full 9-group / 18-experiment / 54-technical-run hierarchy and Table 1 DNA
  typing, preserving ``9064 AMALA`` ``DQA1*05:01/05:03`` ambiguity and leaving
  header group ``RN`` explicitly unresolved (never mapped to ``CD114`` by
  elimination).
- Qualification flags for the 134 rows with ``PEP > 0.1``, variable-modification
  untracked peptidoform scope, and unspecified low-signal experiment exclusions.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import openpyxl
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .provenance import file_digest
from .version import __version__

PACKAGE_FAMILY = "bergseng_2015_dq"
SCHEMA_VERSION = 1
REVISION = "bergseng2015_dq_ion_volumes_v1"
CONTRACT_PATH = Path(__file__).parent / "data" / "evidence_revisions" / "bergseng_2015_dq.json"

_VOL_HEADER_RE = re.compile(r"^Ion peak volume ([A-Za-z0-9]+)_(\d+)([a-z])$")


def _excel_col_letter(idx_1based: int) -> str:
    letters: list[str] = []
    n = idx_1based
    while n > 0:
        n, rem = divmod(n - 1, 26)
        letters.append(chr(ord("A") + rem))
    return "".join(reversed(letters))


def bergseng_2015_source_contract() -> dict[str, Any]:
    """Return a fresh copy of the pinned Bergseng 2015 (#859) source contract."""
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    if contract.get("revision") != REVISION:
        raise ValueError(f"Unsupported Bergseng 2015 contract revision: {contract.get('revision')}")
    return contract


def _verify_workbook_bytes(path: Path, asset: dict[str, Any]) -> dict[str, Any]:
    raw = path.read_bytes()
    size = len(raw)
    sha256 = hashlib.sha256(raw).hexdigest()
    md5 = hashlib.md5(raw).hexdigest()
    if size != asset["size_bytes"]:
        raise ValueError(
            f"Bergseng 2015 workbook size mismatch: got {size}, expected {asset['size_bytes']}"
        )
    if sha256 != asset["sha256"]:
        raise ValueError(
            f"Bergseng 2015 workbook sha256 mismatch: got {sha256}, expected {asset['sha256']}"
        )
    if md5 != asset["md5"]:
        raise ValueError(f"Bergseng 2015 workbook md5 mismatch: got {md5}, expected {asset['md5']}")
    return {"size_bytes": size, "sha256": sha256, "md5": md5}


def fetch_bergseng_2015_source(
    cache_directory: str | Path | None = None,
    *,
    force: bool = False,
    verbose: bool = True,
) -> Path:
    """Download and checksum-verify ``251_2014_819_MOESM3_ESM.xlsx`` from EuropePMC OA."""
    from .downloads import data_dir, download_to_file

    contract = bergseng_2015_source_contract()
    asset = contract["asset"]
    root = Path(cache_directory) if cache_directory is not None else data_dir() / "bergseng_2015_dq"
    root.mkdir(parents=True, exist_ok=True)
    target = root / asset["filename"]
    if target.exists() and not force:
        _verify_workbook_bytes(target, asset)
        return target

    with tempfile.TemporaryDirectory(prefix=".bergseng-fetch-", dir=root) as tmp:
        archive_path = Path(tmp) / "PMC4297300_supplementaryFiles.zip"
        download_to_file(
            asset["archive_url"],
            archive_path,
            force=True,
            verbose=verbose,
        )
        with zipfile.ZipFile(archive_path) as zf:
            if asset["archive_member"] not in zf.namelist():
                raise ValueError(
                    f"Missing {asset['archive_member']} in PMC4297300 supplementary archive"
                )
            member_bytes = zf.read(asset["archive_member"])
        staged = Path(tmp) / asset["filename"]
        staged.write_bytes(member_bytes)
        _verify_workbook_bytes(staged, asset)
        os.replace(staged, target)
    return target


def parse_bergseng_2015_workbook(
    workbook_path: str | Path,
    *,
    verify_pin: bool = True,
) -> dict[str, Any]:
    """Parse ``251_2014_819_MOESM3_ESM.xlsx`` into normalized peptides, runs, and measurements.

    Set ``verify_pin=False`` only for synthetic unit-test fixtures; real source
    materialization always verifies the exact byte size, MD5, and SHA-256.
    """
    path = Path(workbook_path)
    contract = bergseng_2015_source_contract()
    asset = contract["asset"]
    proto = contract["protocol_provenance"]
    groups = contract["header_groups"]
    if verify_pin:
        _verify_workbook_bytes(path, asset)

    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        sheet_name = asset["sheet_name"]
        if sheet_name not in wb.sheetnames:
            raise ValueError(f"Expected sheet {sheet_name!r}, found {wb.sheetnames}")
        ws = wb[sheet_name]
        all_rows = list(ws.iter_rows(values_only=True))
    finally:
        wb.close()

    header_idx = asset["header_row_1based"] - 1
    if len(all_rows) <= header_idx:
        raise ValueError("Bergseng 2015 workbook missing header row")
    headers = [str(h).strip() if h is not None else "" for h in all_rows[header_idx]]
    data_rows = all_rows[asset["first_data_row_1based"] - 1 :]

    required_meta = (
        "Peptide number",
        "Sequence",
        "Length",
        "Mass",
        "Proteins",
        "Leading razor protein",
        "Unique (Groups)",
        "Unique (Proteins)",
        "Charges",
        "PEP",
        "Score",
        "Reverse",
        "Contaminant",
    )
    missing_cols = [c for c in required_meta if c not in headers]
    if missing_cols:
        raise ValueError(f"Missing required Bergseng 2015 columns: {missing_cols}")
    col_pos = {name: headers.index(name) for name in headers if name}

    run_cols: list[tuple[int, str, str, str, int, str]] = []
    for idx, header in enumerate(headers):
        match = _VOL_HEADER_RE.fullmatch(header)
        if match:
            grp, exp_str, tech = match.groups()
            if grp not in groups:
                raise ValueError(f"Unexpected Bergseng 2015 header group: {grp!r}")
            run_cols.append((idx, _excel_col_letter(idx + 1), header, grp, int(exp_str), tech))
    if not run_cols:
        raise ValueError("No 'Ion peak volume ...' columns found in Bergseng 2015 workbook")

    # First pass: compute exact per-run sums & build peptide table.
    run_totals = {col_idx: 0.0 for col_idx, *_ in run_cols}
    run_pos_counts = {col_idx: 0 for col_idx, *_ in run_cols}
    run_zero_counts = {col_idx: 0 for col_idx, *_ in run_cols}

    peptide_records: list[dict[str, Any]] = []
    n_pep_gt_01 = 0
    n_rev = 0
    n_contam = 0

    for offset, row in enumerate(data_rows):
        excel_row = asset["first_data_row_1based"] + offset
        seq = str(row[col_pos["Sequence"]] or "").strip()
        if not seq:
            continue
        pep_num = int(row[col_pos["Peptide number"]])
        pep_val = float(row[col_pos["PEP"]]) if row[col_pos["PEP"]] is not None else math.nan
        score_val = float(row[col_pos["Score"]]) if row[col_pos["Score"]] is not None else math.nan
        mass_val = float(row[col_pos["Mass"]]) if row[col_pos["Mass"]] is not None else math.nan
        exceeds_pep = bool(math.isfinite(pep_val) and pep_val > proto["methods_stated_max_pep"])
        rev_flag = str(row[col_pos["Reverse"]] or "").strip()
        contam_flag = str(row[col_pos["Contaminant"]] or "").strip()
        if exceeds_pep:
            n_pep_gt_01 += 1
        if rev_flag:
            n_rev += 1
        if contam_flag:
            n_contam += 1

        peptide_records.append(
            {
                "source_row_1based": excel_row,
                "peptide_number": pep_num,
                "peptide": seq,
                "peptide_length": int(row[col_pos["Length"]]),
                "mass": mass_val,
                "proteins_raw": str(row[col_pos["Proteins"]] or "").strip(),
                "leading_razor_protein": str(row[col_pos["Leading razor protein"]] or "").strip(),
                "unique_groups": str(row[col_pos["Unique (Groups)"]] or "").strip(),
                "unique_proteins": str(row[col_pos["Unique (Proteins)"]] or "").strip(),
                "charges_raw": str(row[col_pos["Charges"]] or "").strip(),
                "pep": pep_val,
                "score": score_val,
                "reverse_flag": rev_flag,
                "contaminant_flag": contam_flag,
                "exceeds_methods_pep_0_1": exceeds_pep,
                "peptidoform_status": proto["peptidoform_status"],
                "source_dataset": PACKAGE_FAMILY,
                "source_pmid": str(contract["pmid"]),
                "source_file": asset["filename"],
            }
        )

        for col_idx, *_ in run_cols:
            raw_v = row[col_idx] if col_idx < len(row) else None
            if raw_v is None or raw_v == "":
                continue
            val = float(raw_v)
            if val > 0.0:
                run_totals[col_idx] += val
                run_pos_counts[col_idx] += 1
            elif val == 0.0:
                run_zero_counts[col_idx] += 1

    # Build run-level metadata table preserving Table 1 typing & unresolved RN alias.
    run_records: list[dict[str, Any]] = []
    run_id_by_col: dict[int, str] = {}
    for col_idx, col_letter, header, grp, exp_num, tech in run_cols:
        g_meta = groups[grp]
        run_id = f"{grp}_{exp_num}{tech}"
        run_id_by_col[col_idx] = run_id
        run_records.append(
            {
                "run_id": run_id,
                "source_column_letter": col_letter,
                "source_header": header,
                "header_group": grp,
                "experiment_number": exp_num,
                "technical_replicate": tech,
                "cell_line_name": g_meta["table1_cell_line"],
                "ihwg_id": g_meta["ihwg_id"],
                "table1_join_status": g_meta["table1_join_status"],
                "dq_serotype": g_meta["dq_serotype"],
                "reported_dqa1": g_meta["reported_dqa1"],
                "reported_dqb1": g_meta["reported_dqb1"],
                "mhc_allele_set": g_meta["mhc_allele_set"],
                "mhc_allele_cardinality": g_meta["mhc_allele_cardinality"],
                "typing_resolution": g_meta["typing_resolution"],
                "ip_antibody": proto["ip_antibody"],
                "total_ion_peak_volume": run_totals[col_idx],
                "n_positive_peptides": run_pos_counts[col_idx],
                "n_explicit_zero_peptides": run_zero_counts[col_idx],
                "low_signal_exclusion_status": proto["low_signal_exclusion_status"],
                "cell_lineage": "ebv_lcl",
                "malignancy_status": "immortalized_nonmalignant",
                "tumor_type": "none",
                "study_design_scope": contract["study_design_scope"],
            }
        )

    # Second pass: build peptide-by-run measurements table.
    meas_records: list[dict[str, Any]] = []
    n_pos = n_zero = n_missing = 0
    for offset, row in enumerate(data_rows):
        excel_row = asset["first_data_row_1based"] + offset
        seq = str(row[col_pos["Sequence"]] or "").strip()
        if not seq:
            continue
        pep_num = int(row[col_pos["Peptide number"]])
        pep_val = float(row[col_pos["PEP"]]) if row[col_pos["PEP"]] is not None else math.nan
        exceeds_pep = bool(math.isfinite(pep_val) and pep_val > proto["methods_stated_max_pep"])
        for col_idx, col_letter, header, grp, exp_num, tech in run_cols:
            run_id = run_id_by_col[col_idx]
            raw_v = row[col_idx] if col_idx < len(row) else None
            if raw_v is None or raw_v == "":
                raw_float = math.nan
                norm_float = math.nan
                state = "missing"
                n_missing += 1
            else:
                raw_float = float(raw_v)
                total = run_totals[col_idx]
                norm_float = (raw_float / total) if total > 0.0 else math.nan
                if raw_float > 0.0:
                    state = "positive"
                    n_pos += 1
                elif raw_float == 0.0:
                    state = "explicit_zero"
                    n_zero += 1
                else:
                    raise ValueError(
                        f"Negative ion peak volume at {sheet_name}!{col_letter}{excel_row}: {raw_float}"
                    )
            meas_records.append(
                {
                    "measurement_id": f"bergseng2015:p{pep_num}:{run_id}",
                    "peptide_number": pep_num,
                    "peptide": seq,
                    "run_id": run_id,
                    "header_group": grp,
                    "experiment_number": exp_num,
                    "technical_replicate": tech,
                    "source_cell_locator": f"{sheet_name}!{col_letter}{excel_row}",
                    "source_header": header,
                    "measurement_family": "ms1_ion_peak_volume",
                    "raw_ion_peak_volume": raw_float,
                    "raw_unit": "ion_peak_volume_au",
                    "normalized_relative_volume": norm_float,
                    "normalized_unit": "fraction_of_run_total_ion_peak_volume",
                    "value_state": state,
                    "identification_transfer_caveat": proto["identification_transfer_caveat"],
                    "exceeds_methods_pep_0_1": exceeds_pep,
                    "is_absolute_concentration": False,
                    "evidence_grain": "peptide_by_technical_run_separate_from_iedb_aggregate",
                }
            )

    peptides_df = pd.DataFrame(peptide_records)
    runs_df = pd.DataFrame(run_records)
    measurements_df = pd.DataFrame(meas_records)

    groups_seen = sorted({r["header_group"] for r in run_records})
    reconciliation = {
        "n_peptides": len(peptides_df),
        "n_distinct_sequences": int(peptides_df["peptide"].nunique())
        if not peptides_df.empty
        else 0,
        "n_runs": len(runs_df),
        "n_experiments": len({(r["header_group"], r["experiment_number"]) for r in run_records}),
        "n_header_groups": len(groups_seen),
        "n_resolved_header_groups": sum(
            groups[g]["table1_join_status"] != "unresolved_alias" for g in groups_seen
        ),
        "n_unresolved_header_groups": sum(
            groups[g]["table1_join_status"] == "unresolved_alias" for g in groups_seen
        ),
        "n_measurements": len(measurements_df),
        "n_positive_measurements": n_pos,
        "n_explicit_zero_measurements": n_zero,
        "n_missing_measurements": n_missing,
        "n_pep_gt_0_1_peptides": n_pep_gt_01,
        "n_reverse_flagged_peptides": n_rev,
        "n_contaminant_flagged_peptides": n_contam,
    }
    if verify_pin and reconciliation != contract["expected_reconciliation"]:
        raise ValueError(
            f"Bergseng 2015 reconciliation mismatch:\n"
            f"  got:      {reconciliation}\n"
            f"  expected: {contract['expected_reconciliation']}"
        )
    return {
        "peptides": peptides_df,
        "runs": runs_df,
        "measurements": measurements_df,
        "reconciliation": reconciliation,
        "contract": contract,
    }


def write_bergseng_2015_dq_package(
    output_dir: str | Path,
    *,
    workbook_path: str | Path | None = None,
    cache_directory: str | Path | None = None,
    verify_pin: bool = True,
) -> dict[str, Any]:
    """Write a checksum-verified Bergseng 2015 DQ Parquet package atomically."""
    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    source_path = (
        Path(workbook_path)
        if workbook_path is not None
        else fetch_bergseng_2015_source(cache_directory=cache_directory)
    )
    parsed = parse_bergseng_2015_workbook(source_path, verify_pin=verify_pin)
    with tempfile.TemporaryDirectory(prefix=f".{destination.name}-", dir=destination.parent) as tmp:
        staging = Path(tmp)
        for name in ("peptides", "runs", "measurements"):
            table = pa.Table.from_pandas(parsed[name], preserve_index=False)
            pq.write_table(table, staging / f"{name}.parquet", compression="zstd")
        (staging / "contract.json").write_text(
            json.dumps(parsed["contract"], indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        files = {
            rel: file_digest(staging / rel)
            for rel in (
                "peptides.parquet",
                "runs.parquet",
                "measurements.parquet",
                "contract.json",
            )
        }
        manifest = {
            "package_family": PACKAGE_FAMILY,
            "schema_version": SCHEMA_VERSION,
            "revision": REVISION,
            "hitlist_version": __version__,
            "source_verified_pin": verify_pin,
            "source_file_digest": file_digest(source_path),
            "reconciliation": parsed["reconciliation"],
            "files": files,
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(staging, destination)
    return manifest


def verify_bergseng_2015_dq_package(directory: str | Path) -> dict[str, Any]:
    """Verify file hashes, contract identity, and row-count conservation of a package."""
    root = Path(directory)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if (
        manifest.get("package_family") != PACKAGE_FAMILY
        or manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("revision") != REVISION
    ):
        raise ValueError("Unsupported Bergseng 2015 DQ package manifest")
    expected_files = {
        "peptides.parquet",
        "runs.parquet",
        "measurements.parquet",
        "contract.json",
    }
    if set(manifest.get("files", {})) != expected_files:
        raise ValueError("Bergseng 2015 DQ package file inventory differs")
    for rel, expected in manifest["files"].items():
        if file_digest(root / rel) != expected:
            raise ValueError(f"Bergseng 2015 DQ package file hash differs: {rel}")
    contract = bergseng_2015_source_contract()
    if json.loads((root / "contract.json").read_text(encoding="utf-8")) != contract:
        raise ValueError("Bergseng 2015 DQ package contract differs from packaged contract")
    rec = manifest["reconciliation"]
    if pq.read_table(root / "peptides.parquet").num_rows != rec["n_peptides"]:
        raise ValueError("Peptides row count mismatch")
    if pq.read_table(root / "runs.parquet").num_rows != rec["n_runs"]:
        raise ValueError("Runs row count mismatch")
    if pq.read_table(root / "measurements.parquet").num_rows != rec["n_measurements"]:
        raise ValueError("Measurements row count mismatch")
    if (
        rec["n_positive_measurements"]
        + rec["n_explicit_zero_measurements"]
        + rec["n_missing_measurements"]
        != rec["n_measurements"]
    ):
        raise ValueError("Measurement value-state counts do not sum to n_measurements")
    return manifest
