"""Verified nominal-dose source packages through the shared evidence store."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from .dilution_observer_compiler import compile_dilution_source
from .dilution_observer_schema import SCHEMAS
from .evidence_snapshot import snapshot_file
from .provenance import file_digest

PACKAGE_SCHEMA_VERSION = 28
CATALOG_SCHEMA_VERSION = 37
PACKAGE_FAMILY = "dilution_observer"
REVISION = "pxd024809_dilution_observer_v1"
ROOT = Path("dilution_observer")
CONTRACT = Path(__file__).parent / "data/evidence_revisions/dilution_observer.json"


def _contract():
    contract = json.loads(CONTRACT.read_text())
    names = [a["filename"] for a in contract["assets"]]
    if len(names) != len(set(names)) or any(
        not name or Path(name).name != name or name in {".", ".."} or "\\" in name for name in names
    ):
        raise ValueError("Dilution asset names must be unique safe basenames")
    return contract


def _inventory(directory):
    return {
        str(p.relative_to(directory)): file_digest(p)
        for p in sorted(directory.rglob("*"))
        if p.is_file() and p != directory / "manifest.json"
    }


def _space(directory):
    if shutil.disk_usage(directory).free < 1024**3:
        raise OSError("Dilution-source work requires 1 GiB free reserve")


def verify_dilution_observer_package(directory):
    """Replay every typed row from pinned original bytes, including ambiguity."""
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    expected = (PACKAGE_SCHEMA_VERSION, CATALOG_SCHEMA_VERSION, PACKAGE_FAMILY, REVISION)
    if (
        tuple(
            manifest.get(k)
            for k in ("schema_version", "catalog_schema_version", "package_family", "revision")
        )
        != expected
    ):
        raise ValueError("Unsupported dilution-observer package")
    if _inventory(directory) != manifest.get("files"):
        raise ValueError("Dilution source inventory or hashes differ")
    contract = _contract()
    if (
        json.loads((directory / ROOT / "contract.json").read_text()) != contract
        or manifest.get("source_registry") != contract["source_registry"]
    ):
        raise ValueError("Dilution source contract differs")
    assets = directory / ROOT / "source_assets"
    for asset in contract["assets"]:
        if file_digest(assets / asset["filename"]) != {
            k: asset[k] for k in ("sha256", "size_bytes")
        }:
            raise ValueError("Pinned dilution source differs")
    _space(directory.parent)
    scratch = Path(tempfile.mkdtemp(prefix="hitlist-dilution-verify-", dir=directory.parent))
    try:
        counts = compile_dilution_source(assets, scratch / "typed", contract)
        if counts != manifest.get("reconciliation"):
            raise ValueError("Dilution source row counts differ")
        for name, schema in SCHEMAS.items():
            actual = pq.ParquetFile(directory / ROOT / "typed" / f"{name}.parquet")
            reference = pq.ParquetFile(scratch / "typed" / f"{name}.parquet")
            if not actual.schema_arrow.equals(schema):
                raise ValueError(f"Dilution typed schema differs: {name}")
            for left, right in zip(reference.iter_batches(), actual.iter_batches(), strict=True):
                if not left.equals(right):
                    raise ValueError(f"Dilution original-source replay differs: {name}")
    except Exception:
        (scratch / "failure-context.json").write_text(
            json.dumps({"package": str(directory)}) + "\n"
        )
        raise
    else:
        shutil.rmtree(scratch)
    return manifest


def write_dilution_observer_package(source_directory, destination, *, assets=None):
    """Snapshot admitted originals, optionally located through a basename/path map.

    ``assets`` changes file locations only, never the pinned source contract.
    No destination is overwritten and no source is linked through a shared inode.
    """
    source_directory, destination = Path(source_directory), Path(destination)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    contract = _contract()
    overrides = {} if assets is None else dict(assets)
    if set(overrides) - {a["filename"] for a in contract["assets"]}:
        raise ValueError("Unknown dilution asset location override")
    destination.parent.mkdir(parents=True, exist_ok=True)
    _space(destination.parent)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    try:
        for asset in contract["assets"]:
            source = Path(overrides.get(asset["filename"], source_directory / asset["filename"]))
            if file_digest(source) != {k: asset[k] for k in ("sha256", "size_bytes")}:
                raise ValueError(f"Pinned dilution source differs: {asset['filename']}")
            _space(staging)
            snapshot_file(source, staging / ROOT / "source_assets" / asset["filename"])
        (staging / ROOT / "contract.json").write_text(
            json.dumps(contract, indent=2, sort_keys=True) + "\n"
        )
        counts = compile_dilution_source(
            staging / ROOT / "source_assets", staging / ROOT / "typed", contract
        )
        from .version import __version__

        manifest = {
            "schema_version": PACKAGE_SCHEMA_VERSION,
            "catalog_schema_version": CATALOG_SCHEMA_VERSION,
            "package_family": PACKAGE_FAMILY,
            "revision": REVISION,
            "hitlist_version": __version__,
            "source_registry": contract["source_registry"],
            "reconciliation": counts,
            "files": _inventory(staging),
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        verify_dilution_observer_package(staging)
        staging.rename(destination)
        return destination / "manifest.json"
    except Exception:
        (staging / "failure-context.json").write_text(
            json.dumps({"source": str(source_directory), "destination": str(destination)}) + "\n"
        )
        raise


def dilution_observer_applicability_contract() -> dict[str, Any]:
    """Return the PXD024809 nominal-dose applicability and diagnostic recipe contract (#845).

    Encodes the verified source/method boundaries and forbids unsupported transfer claims:
    - Nine observed post-isolation heavy standards (90 standard x run slots, 87 exact-feature
      positives admitted, 1 missing-feature-ID positive and 2 literal zeros excluded); the
      tenth intended standard is absent from the master library and exports (NOT 10 negatives).
    - DIA1 (10 x 15 m/z, 400-550 m/z) and DIA2 (3 x 34 m/z, 550-650 m/z) are distinct
      acquisition methods, never exchangeable replicates.
    - Retains the Table 1 / PRIDE 5x twofold (0.2 -> 0.0125 fmol/uL) versus article Results
      1:10 dose-grid conflict and distinguishes nominal solution concentration from certified
      on-column amount.
    - Explicitly prohibits general sequence-encoder transfer, native non-MHC MS1 transfer,
      unconditional detection claims, or APC intracellular concentration transfer.
    """
    contract = _contract()
    return {
        "source_dataset": contract["source_dataset"],
        "pmc_id": "PMC12406853",
        "doi": contract["source_registry"]["PXD024809"]["doi"],
        "project_url": contract["source_registry"]["PXD024809"]["project_url"],
        "article_url": contract["source_registry"]["PXD024809"]["article_url"],
        "allowed_diagnostic_scope": (
            "conditional_nominal_dose_sensitivity_on_selected_known_standard_panel_only"
        ),
        "n_intended_standards": int(contract["n_nominal_standards"]),
        "n_observed_standards": 9,
        "n_absent_standards": 1,
        "n_nominal_levels": int(contract["n_nominal_levels"]),
        "n_dia_methods": 2,
        "n_expected_standard_slots": 90,
        "n_expected_admitted_slots": 87,
        "n_expected_excluded_missing_feature_id_positive": 1,
        "n_expected_excluded_literal_zero": 2,
        "absent_standard_is_measured_negative": False,
        "dose_grid_conflict_present": True,
        "dose_grid_conflict_summary": (
            "Table 1 and PRIDE protocol specify five twofold levels (0.2 to 0.0125 fmol/uL), "
            "whereas the article Results text states 1:10 dilution; retain both explicitly."
        ),
        "nominal_solution_concentration_is_certified_on_column_amount": False,
        "dia1_and_dia2_exchangeable_replicates": False,
        "library_conditioned_on_all_levels": True,
        "unconditional_detection_claim_allowed": False,
        "general_sequence_transfer_allowed": False,
        "apc_concentration_transfer_allowed": False,
        "native_non_mhc_ms1_transfer_allowed": False,
        "spike_stage": "after_HLA_isolation",
        "reference_diagnostic_benchmark": {
            "cv_scheme": "five_whole_dose_leave_level_out_folds_holding_DIA1_and_DIA2_together",
            "metric": "equal_group_log_signal_rmse",
            "constant_rmse": 0.665914,
            "fixed_unit_slope_rmse": 0.334620,
            "learned_shared_slope_rmse": 0.340878,
            "method_slopes_rmse": 0.343272,
            "preferred_model": "fixed_unit_slope",
        },
        "n_pinned_assets": len(contract["assets"]),
        "total_pinned_size_bytes": sum(int(a["size_bytes"]) for a in contract["assets"]),
    }


def fetch_dilution_observer_sources(
    dest_dir: str | Path | None = None,
    *,
    force: bool = False,
    include_library: bool = True,
    offline: bool | None = None,
) -> Path:
    """Download and SHA-256-verify all pinned PXD024809 dilution-observer source assets (#845)."""
    from .downloads import data_dir, download_to_file

    contract = _contract()
    target_dir = (
        Path(dest_dir) if dest_dir is not None else data_dir() / "pxd024809_dilution_observer"
    )
    target_dir.mkdir(parents=True, exist_ok=True)
    for asset in contract["assets"]:
        if not include_library and asset["role"] == "master_query_library":
            continue
        url = asset["source_url"]
        if url.startswith("https://github.com/") and "/blob/" in url:
            url = url.replace("https://github.com/", "https://raw.githubusercontent.com/").replace(
                "/blob/", "/"
            )
        download_to_file(
            url,
            target_dir / asset["filename"],
            label=f"PXD024809/{asset['filename']}",
            force=force,
            expected_sha256=asset["sha256"],
            expected_size=int(asset["size_bytes"]),
            offline=offline,
        )
    return target_dir


def _load_dilution_tables(
    package_or_tables: str | Path | dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    required = (
        "nominal_standards",
        "standard_memberships",
        "aligned_quantities",
        "feature_links",
        "acquisitions",
        "nominal_levels",
    )
    if isinstance(package_or_tables, dict):
        missing = [k for k in required if k not in package_or_tables]
        if missing:
            raise ValueError(f"Missing required dilution observer tables: {missing}")
        return {k: package_or_tables[k].copy() for k in required}
    root = Path(package_or_tables)
    typed_dir = root / ROOT / "typed" if (root / ROOT / "typed").is_dir() else root
    return {name: pd.read_parquet(typed_dir / f"{name}.parquet") for name in required}


def dilution_observer_nominal_dose_recipe(
    package_or_tables: str | Path | dict[str, pd.DataFrame],
) -> dict[str, Any]:
    """Replay the registered PXD024809 conditional nominal-dose diagnostic recipe (#845).

    Preserves the full 10-standard intended panel, keeps any absent standard separate from
    observations (never generating negative rows), retains all standard x run slots with
    explicit admission/exclusion dispositions, separates DIA1 vs DIA2 acquisition methods,
    and assigns whole-dose leave-level-out development folds holding both methods together.
    """
    tables = _load_dilution_tables(package_or_tables)
    applicability = dilution_observer_applicability_contract()

    standards = tables["nominal_standards"]
    memberships = tables["standard_memberships"]
    aligned = tables["aligned_quantities"]
    links = tables["feature_links"]
    acquisitions = tables["acquisitions"]
    levels = tables["nominal_levels"]

    in_lib = memberships[
        memberships["library_precursor_id"].notna()
        & (memberships["library_precursor_id"].str.len() > 0)
        & (memberships["membership_status"] == "target")
    ].copy()
    observed_std_ids = set(in_lib["standard_id"].tolist())
    absent_standards = standards[~standards["standard_id"].isin(observed_std_ids)].reset_index(
        drop=True
    )

    std_slots = in_lib[["standard_id", "library_precursor_id", "membership_status"]].merge(
        standards[
            [
                "standard_id",
                "peptide_sequence",
                "reported_standard_form",
                "expected_modified_sequence",
                "nominal_status",
            ]
        ],
        on="standard_id",
        how="left",
    )
    std_slots = std_slots.merge(
        aligned[
            [
                "measurement_id",
                "library_precursor_id",
                "acquisition_id",
                "native_feature_id",
                "quantitative_raw",
                "quantitative_value",
                "value_status",
                "measurement_unit",
                "dilution_m_score_raw",
                "dilution_m_score",
            ]
        ],
        on="library_precursor_id",
        how="inner",
    )
    std_slots = std_slots.merge(
        links[
            [
                "measurement_id",
                "identification_id",
                "feature_link_status",
                "form_agrees",
                "charge_agrees",
                "precursor_agrees",
                "target_agrees",
            ]
        ],
        on="measurement_id",
        how="left",
    )
    std_slots = std_slots.merge(
        acquisitions[
            [
                "acquisition_id",
                "raw_file",
                "source_run_label",
                "nominal_level_id",
                "dia_index_from_name",
                "method_link_precision",
            ]
        ],
        on="acquisition_id",
        how="left",
    )
    std_slots = std_slots.merge(
        levels[
            [
                "nominal_level_id",
                "sample_label_raw",
                "dilution_factor_raw",
                "nominal_concentration_raw",
                "nominal_concentration",
            ]
        ],
        on="nominal_level_id",
        how="left",
    )

    # Deterministic whole-dose leave-level-out fold index ordered by descending nominal dose.
    ordered_levels = (
        levels[["nominal_level_id", "nominal_concentration"]]
        .drop_duplicates("nominal_level_id")
        .sort_values(
            ["nominal_concentration", "nominal_level_id"],
            ascending=[False, True],
            na_position="last",
        )
        .reset_index(drop=True)
    )
    level_to_fold = {row.nominal_level_id: int(idx) for idx, row in ordered_levels.iterrows()}
    std_slots["leave_level_out_fold"] = std_slots["nominal_level_id"].map(level_to_fold)
    std_slots["dia_method"] = std_slots["dia_index_from_name"].map(
        lambda x: f"DIA{int(x)}" if pd.notna(x) else "unknown"
    )

    dispositions: list[str] = []
    reasons: list[str] = []
    for row in std_slots.itertuples(index=False):
        vstatus = str(row.value_status)
        fstatus = str(row.feature_link_status) if pd.notna(row.feature_link_status) else "missing"
        exact_match = (
            fstatus in {"matched_identity", "exact_unique_match", "exact_match"}
            and bool(row.form_agrees)
            and bool(row.charge_agrees)
            and bool(row.precursor_agrees)
            and bool(row.target_agrees)
        )
        missing_feature = (
            pd.isna(row.native_feature_id)
            or str(row.native_feature_id).strip() in {"", "NA", "NaN", "nan", "null"}
            or "missing" in fstatus.lower()
        )
        if vstatus == "positive" and exact_match:
            dispositions.append("admitted")
            reasons.append("positive_exact_feature_match")
        elif vstatus == "positive" and missing_feature:
            dispositions.append("excluded")
            reasons.append("excluded_missing_feature_id_positive")
        elif vstatus == "zero":
            dispositions.append("excluded")
            reasons.append("excluded_literal_zero")
        elif vstatus == "missing":
            dispositions.append("excluded")
            reasons.append("excluded_missing_value")
        else:
            dispositions.append("excluded")
            reasons.append("excluded_nonexact_feature_link")

    std_slots["recipe_disposition"] = dispositions
    std_slots["recipe_reason"] = reasons

    admitted = std_slots[std_slots["recipe_disposition"] == "admitted"].reset_index(drop=True)
    excluded = std_slots[std_slots["recipe_disposition"] == "excluded"].reset_index(drop=True)

    summary = {
        "source_dataset": applicability["source_dataset"],
        "allowed_diagnostic_scope": applicability["allowed_diagnostic_scope"],
        "n_intended_standards": len(standards),
        "n_observed_standards": len(observed_std_ids),
        "n_absent_standards": len(absent_standards),
        "n_total_standard_slots": len(std_slots),
        "n_admitted_slots": len(admitted),
        "n_excluded_slots": len(excluded),
        "n_excluded_missing_feature_id_positive": int(
            (std_slots["recipe_reason"] == "excluded_missing_feature_id_positive").sum()
        ),
        "n_excluded_literal_zero": int(
            (std_slots["recipe_reason"] == "excluded_literal_zero").sum()
        ),
        "n_leave_level_out_folds": int(std_slots["leave_level_out_fold"].nunique(dropna=True)),
        "dia_methods_present": sorted(std_slots["dia_method"].dropna().unique().tolist()),
        "absent_standard_is_measured_negative": False,
        "dose_grid_conflict_present": True,
        "general_sequence_transfer_allowed": False,
        "apc_concentration_transfer_allowed": False,
    }
    return {
        "applicability": applicability,
        "admitted": admitted,
        "excluded": excluded,
        "all_slots": std_slots,
        "absent_standards": absent_standards,
        "summary": summary,
    }
