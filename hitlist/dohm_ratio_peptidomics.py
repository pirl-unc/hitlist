"""Source-qualified adapter for DO:DM ratio MHC-II immunopeptidomics (#860).

Source: *Tuning DO:DM Ratios Modulates MHC Class II Immunopeptidomes*,
PMID 35085787, PMC10329146, DOI 10.1016/j.mcpro.2022.100204, PRIDE PXD024392
(CC-BY 4.0), Dataset S3 (``PMC10329146.1/mmc4.xlsx``).

Preserves:
- Exact source coordinates, inline modified peptide strings, flanking residues,
  confidence scores (``Xcorr``, ``10lgP``, ``ALC``, ``Q_value``, ``PEP``), and
  all parent protein alternatives from ``Supp_data_MHC_peptides``.
- Exact 36-run MS1 peak areas (6 conditions x 2 biological x 3 technical
  replicates) from ``Supp_data_MHC_quant`` keeping missing cells explicitly
  ``missing`` (never zero or negative).
- Exact ``HLA-DRA*01:01/DRB1*04:01`` pair, clone identities (``G8``, ``1C3``,
  ``2D7``, ``2D7_KO``), relative DO:DM ratio caveats, and separate ``DOKO``
  comparison tracking.
- Strict namespace separation rejecting computational ``Affinity(nM)``
  (NetMHCIIpan), PLAtEAU imputed cores, and whole-cell TMT protein quantities
  from measured peptide MS1 / binding tables.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Any

import openpyxl
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .provenance import file_digest
from .version import __version__

PACKAGE_FAMILY = "dodm_ratio_35085787"
SCHEMA_VERSION = 1
REVISION = "dodm_ratio_35085787_v1"
CONTRACT_PATH = Path(__file__).parent / "data" / "evidence_revisions" / "dohm_ratio_35085787.json"

_QUANT_COL_RE = re.compile(r"^(DR4|DR4DM|DR4DMDO\+|DR4DMDO\+\+|DR4DMDO\+\+\+|DOKO)_rep(\d+)_(\d+)$")
_MOD_TOKEN_RE = re.compile(r"\[[^\]]+\]")

DISALLOWED_MEASURED_BINDING_NAMESPACES = frozenset(
    {"predicted_netmhc_affinity", "plateau_imputed_cores", "whole_cell_tmt_proteome"}
)


def _excel_col_letter(idx_1based: int) -> str:
    letters: list[str] = []
    n = idx_1based
    while n > 0:
        n, rem = divmod(n - 1, 26)
        letters.append(chr(ord("A") + rem))
    return "".join(reversed(letters))


def dodm_ratio_source_contract() -> dict[str, Any]:
    """Return a fresh copy of the pinned PMID 35085787 / PXD024392 (#860) source contract."""
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    if contract.get("revision") != REVISION:
        raise ValueError(f"Unsupported DO:DM contract revision: {contract.get('revision')}")
    return contract


def _verify_asset(path: Path, asset: dict[str, Any]) -> dict[str, Any]:
    raw = path.read_bytes()
    size = len(raw)
    sha256 = hashlib.sha256(raw).hexdigest()
    if size != asset["size_bytes"]:
        raise ValueError(
            f"DO:DM Dataset S3 size mismatch: got {size}, expected {asset['size_bytes']}"
        )
    if sha256 != asset["sha256"]:
        raise ValueError(
            f"DO:DM Dataset S3 sha256 mismatch: got {sha256}, expected {asset['sha256']}"
        )
    return {"size_bytes": size, "sha256": sha256}


def fetch_dodm_ratio_source(
    cache_directory: str | Path | None = None,
    *,
    force: bool = False,
    verbose: bool = True,
) -> Path:
    """Download and checksum-verify ``PMC10329146.1/mmc4.xlsx`` from PMC Open Data S3."""
    from .downloads import data_dir, download_to_file

    contract = dodm_ratio_source_contract()
    asset = contract["asset"]
    root = (
        Path(cache_directory) if cache_directory is not None else data_dir() / "dodm_ratio_35085787"
    )
    root.mkdir(parents=True, exist_ok=True)
    target = root / asset["filename"]
    if target.exists() and not force:
        _verify_asset(target, asset)
        return target
    download_to_file(
        asset["source_url"],
        target,
        force=force,
        verbose=verbose,
        expected_sha256=asset["sha256"],
        expected_size=asset["size_bytes"],
    )
    _verify_asset(target, asset)
    return target


def guard_dodm_evidence_namespace(namespace: str, *, allow_derived: bool = False) -> str:
    """Reject computational predictions / imputed cores from measured MS1/binding paths (#860)."""
    ns = str(namespace).strip()
    if ns == "predicted_netmhc_affinity":
        raise ValueError(
            "Table S3 Affinity(nM) is a NetMHCIIpan prediction and must never enter "
            "measured MHC binding or MS1 peptide tables (#860)."
        )
    if not allow_derived and ns in DISALLOWED_MEASURED_BINDING_NAMESPACES:
        raise ValueError(
            f"Evidence namespace {ns!r} must remain separate from measured MS1 peptide quantities."
        )
    return ns


def parse_dodm_ratio_workbook(
    workbook_path: str | Path,
    *,
    verify_pin: bool = True,
) -> dict[str, Any]:
    """Parse Dataset S3 (``mmc4.xlsx``) into identifications, runs, and MS1 quantity tables."""
    path = Path(workbook_path)
    contract = dodm_ratio_source_contract()
    asset = contract["asset"]
    proto = contract["protocol_provenance"]
    conds = contract["conditions"]
    if verify_pin:
        _verify_asset(path, asset)

    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        id_sheet = asset["identifications_sheet"]
        q_sheet = asset["quantities_sheet"]
        for req_sheet in (id_sheet, q_sheet):
            if req_sheet not in wb.sheetnames:
                raise ValueError(f"Missing required Dataset S3 sheet: {req_sheet!r}")
        id_rows = list(wb[id_sheet].iter_rows(values_only=True))
        q_rows = list(wb[q_sheet].iter_rows(values_only=True))
    finally:
        wb.close()

    if not id_rows or not q_rows:
        raise ValueError("Empty Dataset S3 sheet")

    id_headers = [str(h).strip() if h is not None else "" for h in id_rows[0]]
    # Note: 'Peptide' appears twice in Supp_data_MHC_peptides:
    # column 1 (0-based) is modified peptide without flanks, column 31 is flanked peptide.
    first_pep_idx = id_headers.index("Peptide")
    flanked_pep_idx = (
        len(id_headers) - 1 - id_headers[::-1].index("Peptide")
        if id_headers.count("Peptide") > 1
        else first_pep_idx
    )
    id_pos = {name: id_headers.index(name) for name in id_headers if name}

    id_records: list[dict[str, Any]] = []
    n_mod_ids = 0
    for row_1based, row in enumerate(id_rows[1:], start=2):
        sample_id = str(row[id_pos["Sample ID"]] or "").strip()
        mod_seq = str(row[first_pep_idx] or "").strip()
        if not sample_id and not mod_seq:
            continue
        flanked_seq = str(row[flanked_pep_idx] or "").strip() if flanked_pep_idx < len(row) else ""
        unmod_seq = _MOD_TOKEN_RE.sub("", mod_seq)
        has_ptm = bool(_MOD_TOKEN_RE.search(mod_seq))
        if has_ptm:
            n_mod_ids += 1
        id_records.append(
            {
                "source_row_1based": row_1based,
                "source_cell_locator": f"{id_sheet}!A{row_1based}",
                "sample_id": sample_id,
                "peptide_modified": mod_seq,
                "peptide_flanked": flanked_seq,
                "peptide": unmod_seq,
                "has_ptm": has_ptm,
                "proteins_raw": str(row[id_pos["Proteins"]] or "").strip(),
                "spec_id": str(row[id_pos["SpecID"]] or "").strip(),
                "scan_number": str(row[id_pos["Scanf"]] or "").strip(),
                "xcorr": (
                    float(row[id_pos["Xcorr"]])
                    if "Xcorr" in id_pos and row[id_pos["Xcorr"]] is not None
                    else math.nan
                ),
                "ten_lg_p": (
                    float(row[id_pos["10lgP"]])
                    if "10lgP" in id_pos and row[id_pos["10lgP"]] is not None
                    else math.nan
                ),
                "alc": (
                    float(row[id_pos["ALC"]])
                    if "ALC" in id_pos and row[id_pos["ALC"]] is not None
                    else math.nan
                ),
                "q_value": (
                    float(row[id_pos["Q_value"]])
                    if "Q_value" in id_pos and row[id_pos["Q_value"]] is not None
                    else math.nan
                ),
                "pep_score": (
                    float(row[id_pos["PEP_sore"]])
                    if "PEP_sore" in id_pos and row[id_pos["PEP_sore"]] is not None
                    else math.nan
                ),
                "mhc_allele_set": proto["mhc_allele_set"],
                "ip_antibody": proto["ip_antibody"],
                "evidence_namespace": "peptide_identifications",
            }
        )

    q_headers = [str(h).strip() if h is not None else "" for h in q_rows[0]]
    run_cols: list[tuple[int, str, str, str, int, int]] = []
    for idx, header in enumerate(q_headers):
        match = _QUANT_COL_RE.fullmatch(header)
        if match:
            cond_key, bio_str, tech_str = match.groups()
            run_cols.append(
                (idx, _excel_col_letter(idx + 1), header, cond_key, int(bio_str), int(tech_str))
            )
    if not run_cols:
        raise ValueError("No DO:DM MS1 run columns found in Supp_data_MHC_quant")

    run_records: list[dict[str, Any]] = []
    for _idx, col_letter, header, cond_key, bio_rep, tech_rep in run_cols:
        cmeta = conds[cond_key]
        run_records.append(
            {
                "run_id": header,
                "source_column_letter": col_letter,
                "source_header": header,
                "condition_key": cond_key,
                "clone_name": cmeta["clone_name"],
                "parent_clone": cmeta["parent_clone"],
                "biological_replicate": bio_rep,
                "technical_replicate": tech_rep,
                "dm_present": bool(cmeta["dm_present"]),
                "do_present": bool(cmeta["do_present"]),
                "do_dm_level": cmeta["do_dm_level"],
                "is_doko_comparison": bool(cmeta["is_doko_comparison"]),
                "baseline_engineering": cmeta["baseline_engineering"],
                "host_cell_line": proto["host_cell_line"],
                "mhc_class": proto["mhc_class"],
                "mhc_allele_set": proto["mhc_allele_set"],
                "ip_antibody": proto["ip_antibody"],
                "ratio_gradient_caveat": proto["ratio_gradient_caveat"],
            }
        )

    q_pos = {
        name: q_headers.index(name)
        for name in ("Accession", "Proteins", "Peptide")
        if name in q_headers
    }
    ms1_records: list[dict[str, Any]] = []
    n_quant_peps = 0
    n_pos = n_zero = n_missing = 0
    for row_1based, row in enumerate(q_rows[1:], start=2):
        pep_raw = str(row[q_pos["Peptide"]] or "").strip()
        if not pep_raw:
            continue
        n_quant_peps += 1
        acc = str(row[q_pos["Accession"]] or "").strip()
        prots = str(row[q_pos["Proteins"]] or "").strip()
        unmod = _MOD_TOKEN_RE.sub("", pep_raw)
        has_ptm = bool(_MOD_TOKEN_RE.search(pep_raw))
        for col_idx, col_letter, header, cond_key, bio_rep, tech_rep in run_cols:
            raw_v = row[col_idx] if col_idx < len(row) else None
            if raw_v is None or raw_v == "":
                val = math.nan
                state = "missing"
                n_missing += 1
            else:
                val = float(raw_v)
                if val > 0.0:
                    state = "positive"
                    n_pos += 1
                elif val == 0.0:
                    state = "explicit_zero"
                    n_zero += 1
                else:
                    raise ValueError(
                        f"Negative MS1 area at {q_sheet}!{col_letter}{row_1based}: {val}"
                    )
            cmeta = conds[cond_key]
            ms1_records.append(
                {
                    "measurement_id": f"dodm35085787:r{row_1based}:{header}",
                    "source_row_1based": row_1based,
                    "source_cell_locator": f"{q_sheet}!{col_letter}{row_1based}",
                    "accession_raw": acc,
                    "proteins_raw": prots,
                    "peptide_raw": pep_raw,
                    "peptide": unmod,
                    "has_ptm": has_ptm,
                    "run_id": header,
                    "condition_key": cond_key,
                    "clone_name": cmeta["clone_name"],
                    "biological_replicate": bio_rep,
                    "technical_replicate": tech_rep,
                    "dm_present": bool(cmeta["dm_present"]),
                    "do_present": bool(cmeta["do_present"]),
                    "do_dm_level": cmeta["do_dm_level"],
                    "is_doko_comparison": bool(cmeta["is_doko_comparison"]),
                    "mhc_allele_set": proto["mhc_allele_set"],
                    "raw_ms1_area": val,
                    "raw_unit": "ms1_peak_area_au",
                    "value_state": state,
                    "is_imputed": False,
                    "is_absolute_concentration": False,
                    "evidence_namespace": "peptide_ms1_area",
                }
            )

    reconciliation = {
        "n_identifications": len(id_records),
        "n_modified_identifications": n_mod_ids,
        "n_quant_peptides": n_quant_peps,
        "n_runs": len(run_records),
        "n_conditions": len({r["condition_key"] for r in run_records}),
        "n_ms1_measurements": len(ms1_records),
        "n_positive_ms1_measurements": n_pos,
        "n_zero_ms1_measurements": n_zero,
        "n_missing_ms1_measurements": n_missing,
    }
    if verify_pin and reconciliation != contract["expected_reconciliation"]:
        raise ValueError(
            f"DO:DM Dataset S3 reconciliation mismatch:\n"
            f"  got:      {reconciliation}\n"
            f"  expected: {contract['expected_reconciliation']}"
        )
    return {
        "identifications": pd.DataFrame(id_records),
        "runs": pd.DataFrame(run_records),
        "ms1_quantities": pd.DataFrame(ms1_records),
        "reconciliation": reconciliation,
        "contract": contract,
    }


def write_dodm_ratio_package(
    output_dir: str | Path,
    *,
    workbook_path: str | Path | None = None,
    cache_directory: str | Path | None = None,
    verify_pin: bool = True,
) -> dict[str, Any]:
    """Write a checksum-verified DO:DM ratio peptidomics Parquet package atomically."""
    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    source_path = (
        Path(workbook_path)
        if workbook_path is not None
        else fetch_dodm_ratio_source(cache_directory=cache_directory)
    )
    parsed = parse_dodm_ratio_workbook(source_path, verify_pin=verify_pin)
    with tempfile.TemporaryDirectory(prefix=f".{destination.name}-", dir=destination.parent) as tmp:
        staging = Path(tmp)
        for name in ("identifications", "runs", "ms1_quantities"):
            table = pa.Table.from_pandas(parsed[name], preserve_index=False)
            pq.write_table(table, staging / f"{name}.parquet", compression="zstd")
        (staging / "contract.json").write_text(
            json.dumps(parsed["contract"], indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        files = {
            rel: file_digest(staging / rel)
            for rel in (
                "identifications.parquet",
                "runs.parquet",
                "ms1_quantities.parquet",
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


def verify_dodm_ratio_package(directory: str | Path) -> dict[str, Any]:
    """Verify file hashes, contract identity, and row conservation of a DO:DM package."""
    root = Path(directory)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if (
        manifest.get("package_family") != PACKAGE_FAMILY
        or manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("revision") != REVISION
    ):
        raise ValueError("Unsupported DO:DM ratio package manifest")
    expected_files = {
        "identifications.parquet",
        "runs.parquet",
        "ms1_quantities.parquet",
        "contract.json",
    }
    if set(manifest.get("files", {})) != expected_files:
        raise ValueError("DO:DM ratio package file inventory differs")
    for rel, expected in manifest["files"].items():
        if file_digest(root / rel) != expected:
            raise ValueError(f"DO:DM ratio package file hash differs: {rel}")
    contract = dodm_ratio_source_contract()
    if json.loads((root / "contract.json").read_text(encoding="utf-8")) != contract:
        raise ValueError("DO:DM ratio package contract differs from packaged contract")
    rec = manifest["reconciliation"]
    if pq.read_table(root / "identifications.parquet").num_rows != rec["n_identifications"]:
        raise ValueError("Identifications row count mismatch")
    if pq.read_table(root / "runs.parquet").num_rows != rec["n_runs"]:
        raise ValueError("Runs row count mismatch")
    if pq.read_table(root / "ms1_quantities.parquet").num_rows != rec["n_ms1_measurements"]:
        raise ValueError("MS1 quantities row count mismatch")
    return manifest
