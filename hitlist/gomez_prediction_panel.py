"""Immutable NetMHCpan prediction panels linked to the Gomez-Zepeda deposit.

The publisher's prediction rows are retained as source measurements.  The
restriction projection is a separate relation: it makes the existing curated
allele explicit as a predicted label without rewriting the measured MHC rows,
and keeps unknown rows unknown when no named restriction was curated.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import shutil
import tempfile
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from .evidence_snapshot import snapshot_file
from .gomez_arm_projection import verify_gomez_arm_projection_package
from .gomez_source_lineage import CELL_FILES, _derived_index
from .provenance import file_digest

PACKAGE_SCHEMA_VERSION = 13
CATALOG_SCHEMA_VERSION = 18
REVISION = "gomez_zepeda_2024_prediction_panels_v1"
PARENT_SCHEMA_VERSION = 12
PARENT_REVISION = "gomez_zepeda_2024_arm_projection_v1"
PARENT_MANIFEST = Path("manifests/gomez_arm_projection.json")
ROOT = Path("gomez_prediction")
S5_ARCHIVE = ROOT / "MOESM8.zip"
S7_ARCHIVE = ROOT / "MOESM10.zip"
PREDICTIONS = ROOT / "prediction_rows.parquet"
PROJECTION = ROOT / "restriction_projection.parquet"
REVIEW = ROOT / "review.json"

SOURCE_ASSETS = (
    {
        "phase": "s5_prediction_panel",
        "asset": "MOESM8.zip",
        "archive": S5_ARCHIVE,
        "member": "S5_mhcpred_all_06154.csv",
        "archive_sha256": "4e483930dc81c900fe5dcec3d0d89b56120060fd9e8e850713c04eefd2adedc5",
        "member_sha256": "b2af685d16d007e76f660b725acd4f00a51d8206927ef5ae99991eefb1f38a4c",
        "n_rows": 252_043,
    },
    {
        "phase": "s7_prediction_panel",
        "asset": "MOESM10.zip",
        "archive": S7_ARCHIVE,
        "member": "S7_mhcpred_all_06133b.csv",
        "archive_sha256": "9f5d521156bd618052a47861202e6e1c3ba1ee44bfea460a22c65210ec3f489d",
        "member_sha256": "40216c7689c443f73596fcc19db5ee2c7bbf293665b645ab95ca9ce74cc9930d",
        "n_rows": 142_909,
    },
)
CELL_TO_FILE = dict(CELL_FILES)
CELL_TO_FILE["SKMEL37"] = CELL_TO_FILE["SK-MEL-37"]
PREFERRED_PHASE = {
    "JY": "s7_prediction_panel",
    "Raji": "s7_prediction_panel",
    "HeLa": "s5_prediction_panel",
    "SK-MEL-37": "s5_prediction_panel",
    "SKMEL37": "s5_prediction_panel",
    "Plasma": "s5_prediction_panel",
}
PREDICTION_COLUMNS = (
    "source_prediction_id",
    "source_row_index",
    "source_phase",
    "source_asset_id",
    "source_asset_sha256",
    "source_member_sha256",
    "source_pmid",
    "source_dataset",
    "cell_line_name",
    "cell_line_name_raw",
    "peptide_sequence",
    "allele_raw",
    "allele_normalized",
    "cell_line_allele_scope",
    "prediction_model",
    "binder",
    "el_score_raw",
    "el_score",
    "el_rank_raw",
    "el_rank",
    "aff_score_raw",
    "aff_score",
    "aff_rank_raw",
    "aff_rank",
    "aff_nm_raw",
    "aff_nm",
    "source_fields_json",
    "derived_observation_id",
    "prediction_link_status",
)
PROJECTION_COLUMNS = (
    "derived_observation_id",
    "supplementary_file",
    "peptide_sequence",
    "material_id",
    "native_mhc_restriction",
    "native_restriction_evidence",
    "projected_restriction_evidence",
    "projection_status",
    "projection_reason",
    "selected_allele_raw",
    "selected_allele_normalized",
    "prediction_panel_scope",
    "selected_prediction_ids_json",
    "alternative_prediction_ids_json",
    "n_prediction_rows",
    "n_selected_prediction_rows",
    "n_alternative_prediction_rows",
)
FLOAT_COLUMNS = {"el_score", "el_rank", "aff_score", "aff_rank", "aff_nm"}
INTEGER_COLUMNS = {
    "source_row_index",
    "n_prediction_rows",
    "n_selected_prediction_rows",
    "n_alternative_prediction_rows",
}


def normalize_allele(value):
    """Normalize the two publisher spellings without lowering resolution."""
    value = (value or "").strip()
    match = re.fullmatch(r"(?:HLA-)?([ABC])\*(\d{2}):(\d{2})", value)
    if match:
        return f"HLA-{match.group(1)}*{match.group(2)}:{match.group(3)}"
    match = re.fullmatch(r"HLA-([ABC])(\d{2})(\d{2})", value)
    if match:
        return f"HLA-{match.group(1)}*{match.group(2)}:{match.group(3)}"
    return value


def _number(value):
    return None if value in (None, "") else float(value)


def _schema(columns):
    return pa.schema(
        [
            (
                name,
                pa.int64()
                if name in INTEGER_COLUMNS
                else pa.float64()
                if name in FLOAT_COLUMNS
                else pa.string(),
            )
            for name in columns
        ]
    )


def _source_archive(path, source):
    if file_digest(path)["sha256"] != source["archive_sha256"]:
        raise ValueError(f"Changed Gomez-Zepeda prediction archive: {source['asset']}")
    archive = zipfile.ZipFile(path)
    if archive.namelist() != [source["member"]]:
        archive.close()
        raise ValueError("Unexpected Gomez-Zepeda prediction ZIP members")
    digest = hashlib.sha256()
    with archive.open(source["member"]) as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != source["member_sha256"]:
        archive.close()
        raise ValueError(f"Changed Gomez-Zepeda prediction CSV member: {source['member']}")
    return archive


def _iter_prediction_rows(directory, derived):
    for source in SOURCE_ASSETS:
        with (
            _source_archive(directory / source["archive"], source) as archive,
            archive.open(source["member"]) as handle,
        ):
            reader = csv.DictReader(io.TextIOWrapper(handle, encoding="utf-8-sig", newline=""))
            expected = {
                "Cell.Line",
                "Allele",
                "Sequence",
                "EL_score",
                "EL_Rank",
                "Aff_Score",
                "Aff_Rank",
                "Aff_nM",
                "Binder",
            }
            if set(reader.fieldnames or ()) != expected | (
                {"Cell.Line_allele"} if source["phase"] == "s7_prediction_panel" else set()
            ):
                raise ValueError(f"Unexpected {source['member']} header")
            for row_index, raw in enumerate(reader, 1):
                if None in raw or any(value is None for value in raw.values()):
                    raise ValueError("Malformed Gomez-Zepeda prediction row")
                cell = raw["Cell.Line"]
                normalized_cell = CELL_TO_FILE.get(cell)
                if normalized_cell is None:
                    raise ValueError(f"Unknown Gomez-Zepeda prediction cell line: {cell}")
                peptide = raw["Sequence"]
                derived_id = derived.get((normalized_cell, peptide))
                preferred = PREFERRED_PHASE[cell]
                status = "unlinked_prediction_sequence"
                if derived_id:
                    status = (
                        "linked_preferred_panel"
                        if source["phase"] == preferred
                        else "linked_fallback_panel"
                    )
                yield {
                    "source_prediction_id": f"gomez:prediction:{source['phase'].split('_')[0]}:{row_index}",
                    "source_row_index": row_index,
                    "source_phase": source["phase"],
                    "source_asset_id": source["asset"],
                    "source_asset_sha256": source["archive_sha256"],
                    "source_member_sha256": source["member_sha256"],
                    "source_pmid": "38480730",
                    "source_dataset": "gomez_zepeda_2024_prediction_panels",
                    "cell_line_name": cell if cell != "SKMEL37" else "SK-MEL-37",
                    "cell_line_name_raw": cell,
                    "peptide_sequence": peptide,
                    "allele_raw": raw["Allele"],
                    "allele_normalized": normalize_allele(raw["Allele"]),
                    "cell_line_allele_scope": raw.get("Cell.Line_allele") or None,
                    "prediction_model": "NetMHCpan 4.1",
                    "binder": raw["Binder"],
                    "el_score_raw": raw["EL_score"],
                    "el_score": _number(raw["EL_score"]),
                    "el_rank_raw": raw["EL_Rank"],
                    "el_rank": _number(raw["EL_Rank"]),
                    "aff_score_raw": raw["Aff_Score"],
                    "aff_score": _number(raw["Aff_Score"]),
                    "aff_rank_raw": raw["Aff_Rank"],
                    "aff_rank": _number(raw["Aff_Rank"]),
                    "aff_nm_raw": raw["Aff_nM"],
                    "aff_nm": _number(raw["Aff_nM"]),
                    "source_fields_json": json.dumps(
                        raw, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                    ),
                    "derived_observation_id": derived_id,
                    "prediction_link_status": status,
                }


def _native_restrictions(directory):
    table = ds.dataset(directory / "native/observations.parquet").to_table(
        columns=[
            "provenance_id",
            "peptide",
            "supplementary_file",
            "mhc_restriction",
            "restriction_evidence",
        ],
        filter=ds.field("pmid") == 38480730,
    )
    result = {}
    for row in table.to_pylist():
        key = (row["supplementary_file"], row["peptide"])
        if key in result:
            raise ValueError(f"Duplicate Gomez-Zepeda native restriction row: {key}")
        result[key] = row
    return result


def _projection_rows(directory, predictions):
    native = _native_restrictions(directory)
    by_derived = defaultdict(list)
    for row in predictions:
        if row["derived_observation_id"]:
            by_derived[row["derived_observation_id"]].append(row)
    rows = []
    named = unknown = 0
    for item in sorted(native.values(), key=lambda row: row["provenance_id"]):
        derived_id = item["provenance_id"]
        linked = by_derived.get(derived_id, [])
        material = next(
            (name for name, file in CELL_FILES.items() if file == item["supplementary_file"]), None
        )
        if material is None:
            raise ValueError(f"Unknown Gomez supplementary file: {item['supplementary_file']}")
        preferred_phase = PREFERRED_PHASE[material]
        preferred = [row for row in linked if row["source_phase"] == preferred_phase]
        if material == "JY" and not any(row["binder"] in {"SB", "WB"} for row in preferred):
            preferred = [row for row in linked if row["source_phase"] == "s5_prediction_panel"]
        binder_rows = [row for row in preferred if row["binder"] in {"SB", "WB"}]
        selected_raw = item["mhc_restriction"] or ""
        selected = normalize_allele(selected_raw)
        selected_rows = (
            [row for row in binder_rows if row["allele_normalized"] == selected] if selected else []
        )
        alternative_rows = [row for row in binder_rows if row not in selected_rows]
        if selected_raw:
            named += 1
            status = "predicted_named"
            evidence = "predicted"
            reason = (
                "selected_source_allele_row"
                if selected_rows
                else "named_native_restriction_preserved_without_matching_prediction_row"
            )
        else:
            unknown += 1
            status = "unknown_no_named_prediction"
            evidence = "unknown"
            reason = "native_restriction_unresolved"
        rows.append(
            {
                "derived_observation_id": derived_id,
                "supplementary_file": item["supplementary_file"],
                "peptide_sequence": item["peptide"],
                "material_id": material,
                "native_mhc_restriction": selected_raw,
                "native_restriction_evidence": item["restriction_evidence"],
                "projected_restriction_evidence": evidence,
                "projection_status": status,
                "projection_reason": reason,
                "selected_allele_raw": selected_raw,
                "selected_allele_normalized": selected if selected_raw else "",
                "prediction_panel_scope": preferred_phase
                if preferred
                else "no_linked_prediction_rows",
                "selected_prediction_ids_json": json.dumps(
                    sorted(row["source_prediction_id"] for row in selected_rows),
                    separators=(",", ":"),
                ),
                "alternative_prediction_ids_json": json.dumps(
                    sorted(row["source_prediction_id"] for row in alternative_rows),
                    separators=(",", ":"),
                ),
                "n_prediction_rows": len(binder_rows),
                "n_selected_prediction_rows": len(selected_rows),
                "n_alternative_prediction_rows": len(alternative_rows),
            }
        )
    if len(rows) != 68_552 or named != 50_761 or unknown != 17_791:
        raise ValueError(
            f"Unexpected Gomez prediction projection counts: {len(rows)}, {named}, {unknown}"
        )
    return rows


def _write_table(path, rows, columns):
    pq.write_table(pa.Table.from_pylist(rows, schema=_schema(columns)), path, compression="zstd")


def _manifest_files(staging):
    return {
        str(path.relative_to(staging)): file_digest(path)
        for path in sorted(staging.rglob("*"))
        if path.is_file() and path != staging / "manifest.json"
    }


def verify_gomez_prediction_panel_package(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if (
        manifest.get("schema_version") != PACKAGE_SCHEMA_VERSION
        or manifest.get("catalog_schema_version") != CATALOG_SCHEMA_VERSION
        or manifest.get("revision") != REVISION
        or manifest.get("parent_kind") != "gomez_arm_projection"
        or manifest.get("parent_manifest_path") != str(PARENT_MANIFEST)
    ):
        raise ValueError("Unsupported Gomez-Zepeda prediction-panel package")
    actual = {
        str(path.relative_to(directory))
        for path in directory.rglob("*")
        if path.is_file() and path != directory / "manifest.json"
    }
    if set(manifest.get("files", {})) != actual:
        raise ValueError("Gomez-Zepeda prediction-panel inventory mismatch")
    for name, digest in manifest["files"].items():
        if file_digest(directory / name)["sha256"] != digest["sha256"]:
            raise ValueError(f"Changed prediction-panel artifact: {name}")
    parent_manifest = directory / PARENT_MANIFEST
    parent = verify_gomez_arm_projection_package(
        directory, manifest_path=PARENT_MANIFEST, allow_additional_files=True
    )
    if (
        parent.get("schema_version") != PARENT_SCHEMA_VERSION
        or parent.get("revision") != PARENT_REVISION
    ):
        raise ValueError("Prediction panel predecessor is not schema-12 arm projection")
    if file_digest(parent_manifest)["sha256"] != manifest.get("parent_manifest_sha256"):
        raise ValueError("Prediction panel predecessor changed")
    derived = _derived_index(directory)
    predictions = list(_iter_prediction_rows(directory, derived))
    if len(predictions) != sum(source["n_rows"] for source in SOURCE_ASSETS):
        raise ValueError("Prediction row count changed")
    actual_predictions = pq.read_table(directory / PREDICTIONS)
    expected_predictions = pa.Table.from_pylist(predictions, schema=_schema(PREDICTION_COLUMNS))
    if not actual_predictions.equals(expected_predictions, check_metadata=False):
        raise ValueError("Prediction parquet row count changed")
    projection = _projection_rows(directory, predictions)
    actual_projection = pq.read_table(directory / PROJECTION)
    expected_projection = pa.Table.from_pylist(projection, schema=_schema(PROJECTION_COLUMNS))
    if not actual_projection.equals(expected_projection, check_metadata=False):
        raise ValueError("Restriction projection changed")
    reconciliation = manifest.get("reconciliation", {})
    if reconciliation.get("n_prediction_rows") != len(predictions):
        raise ValueError("Prediction reconciliation changed")
    if reconciliation.get("n_projection_rows") != len(projection):
        raise ValueError("Projection reconciliation changed")
    return manifest


def write_gomez_prediction_panel_package(predecessor_package, archives_directory, destination):
    """Append both prediction panels and their explicit restriction relation."""
    predecessor_package = Path(predecessor_package)
    archives_directory = Path(archives_directory)
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    parent = verify_gomez_arm_projection_package(predecessor_package)
    if (
        parent.get("schema_version") != PARENT_SCHEMA_VERSION
        or parent.get("revision") != PARENT_REVISION
    ):
        raise ValueError("Gomez prediction panels require schema-12 arm projection predecessor")
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    try:
        for name in parent["files"]:
            snapshot_file(predecessor_package / name, staging / name)
        snapshot_file(predecessor_package / "manifest.json", staging / PARENT_MANIFEST)
        (staging / ROOT).mkdir(parents=True, exist_ok=True)
        for source in SOURCE_ASSETS:
            snapshot_file(archives_directory / source["asset"], staging / source["archive"])
        derived = _derived_index(staging)
        predictions = list(_iter_prediction_rows(staging, derived))
        projection = _projection_rows(staging, predictions)
        _write_table(staging / PREDICTIONS, predictions, PREDICTION_COLUMNS)
        _write_table(staging / PROJECTION, projection, PROJECTION_COLUMNS)
        named = sum(row["projection_status"] == "predicted_named" for row in projection)
        review = {
            "revision": REVISION,
            "article_pmid": "38480730",
            "prediction_model": "NetMHCpan 4.1",
            "source_assets": [
                {
                    "phase": source["phase"],
                    "archive": source["asset"],
                    "archive_sha256": source["archive_sha256"],
                    "member": source["member"],
                    "member_sha256": source["member_sha256"],
                    "n_rows": source["n_rows"],
                }
                for source in SOURCE_ASSETS
            ],
            "projection_contract": {
                "named_native_restrictions": named,
                "unknown_native_restrictions": len(projection) - named,
                "predicted_label_is_not_measured_assay": True,
                "native_assay_rows_rewritten": False,
                "s7_preferred_for_jy": True,
            },
            "no_rows_dropped": True,
        }
        (staging / REVIEW).write_text(json.dumps(review, indent=2, sort_keys=True) + "\n")
        from .version import __version__

        manifest = {
            "schema_version": PACKAGE_SCHEMA_VERSION,
            "catalog_schema_version": CATALOG_SCHEMA_VERSION,
            "hitlist_version": __version__,
            "revision": REVISION,
            "parent_kind": "gomez_arm_projection",
            "parent_manifest_path": str(PARENT_MANIFEST),
            "parent_manifest_sha256": file_digest(staging / PARENT_MANIFEST)["sha256"],
            "adjudication_manifest_path": parent["adjudication_manifest_path"],
            "source_reference": parent["source_reference"],
            "reconciliation": {
                "n_prediction_rows": len(predictions),
                "n_projection_rows": len(projection),
                "n_named_projection_rows": named,
                "n_unknown_projection_rows": len(projection) - named,
                "by_prediction_link_status": dict(
                    sorted(Counter(row["prediction_link_status"] for row in predictions).items())
                ),
            },
            "files": _manifest_files(staging),
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        verify_gomez_prediction_panel_package(staging)
        staging.rename(destination)
        return destination / "manifest.json"
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
