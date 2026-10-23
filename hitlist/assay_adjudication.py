"""Source-reviewed assay interpretations appended without changing measured rows."""

from __future__ import annotations

import json
import re
import shutil
import tempfile
from importlib.resources import files
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from .curation_yaml import load_curation_yaml
from .evidence_snapshot import snapshot_file
from .provenance import file_digest
from .reference_evidence import _safe_relative, verify_reference_evidence_package

PACKAGE_SCHEMA_VERSION = 8
CATALOG_SCHEMA_VERSION = 10
REVIEW_REVISION = "tcell_multimer_2026_10_23"
REVIEW_SHA256 = "0cc78c8f956abccda03053e474019d6c8b1a762b45373c469891f9a933a81f11"
REVIEW_SOURCE = files("hitlist").joinpath("data/assay_adjudications.yaml")
REVIEW_ASSET = Path("tcell/adjudication/reviews.yaml")
REVIEW_TABLE = Path("tcell/adjudication/reviews.parquet")
REVIEW_ASSAY_IDS = frozenset(
    {
        "1646138",
        "2445272",
        "12892501",
        "12892550",
        "12892559",
        "12892569",
        "12892606",
        "12892609",
        "12892626",
    }
)
SOURCE_KEYS = {
    "source_pmid",
    "publication_url",
    "source_asset_url",
    "source_asset_sha256",
    "source_asset_status",
    "source_locator",
}
ROW_KEYS = {
    "source_assay_id",
    "source_key",
    "source_peptide_code",
    "source_row_locator",
    "expected_source_reference_id",
    "expected_qualitative_result_raw",
    "expected_n_subjects_tested_raw",
    "expected_n_subjects_positive_raw",
    "expected_peptide_sequence",
    "expected_reported_mhc_raw",
    "expected_effector_cell_name",
    "review_status",
    "interpretation_code",
    "reviewed_endpoint_scope",
    "reviewed_population_scope",
    "reviewed_preparation_scope",
    "reviewed_reagent",
    "unresolved_question",
    "training_effect",
}
REVIEW_SCHEMA = pa.schema(
    [
        pa.field(name, pa.string())
        for name in (
            "claim_id",
            "source_assay_id",
            "source_dataset",
            "assay_family",
            "review_revision",
            "source_pmid",
            "publication_url",
            "source_asset_url",
            "source_asset_sha256",
            "source_asset_status",
            "source_locator",
            *sorted(ROW_KEYS - {"source_key", "source_assay_id"}),
        )
    ]
)


def load_assay_adjudications(path=None):
    """Read the pinned review ledger, rejecting duplicate or incomplete claims."""
    source = REVIEW_SOURCE if path is None else Path(path)
    if file_digest(source)["sha256"] != REVIEW_SHA256:
        raise ValueError("Unrecognized assay adjudication input bytes")
    document = load_curation_yaml(source)
    if not isinstance(document, dict) or set(document) != {
        "schema_version",
        "revision",
        "sources",
        "reviews",
    }:
        raise ValueError("Invalid assay adjudication document")
    if document["schema_version"] != 1 or document["revision"] != REVIEW_REVISION:
        raise ValueError("Unsupported assay adjudication revision")
    sources = document["sources"]
    if not isinstance(sources, dict) or set(sources) != {
        "kotzin_1999",
        "gapvac_2018",
        "hombrink_2015",
    }:
        raise ValueError("Invalid reviewed publication registry")
    for key, item in sources.items():
        if not isinstance(item, dict) or set(item) != SOURCE_KEYS:
            raise ValueError(f"Invalid reviewed source: {key}")
        if any(
            not isinstance(value, str) or not value
            for name, value in item.items()
            if name != "source_asset_sha256"
        ):
            raise ValueError(f"Empty reviewed source field: {key}")
        pinned = item["source_asset_status"] == "sha256_pinned"
        if pinned != bool(item["source_asset_sha256"]):
            raise ValueError(f"Invalid reviewed source asset status: {key}")
        if pinned and not re.fullmatch(r"[0-9a-f]{64}", item["source_asset_sha256"]):
            raise ValueError(f"Invalid reviewed source hash: {key}")
        if item["source_asset_status"] not in {"sha256_pinned", "linked_unpinned"}:
            raise ValueError(f"Unknown reviewed source asset status: {key}")
    reviews = document["reviews"]
    if not isinstance(reviews, list) or len(reviews) != len(REVIEW_ASSAY_IDS):
        raise ValueError("Assay adjudication review coverage changed")
    rows = []
    for review in reviews:
        if not isinstance(review, dict) or set(review) != ROW_KEYS:
            raise ValueError("Invalid assay adjudication row")
        if any(not isinstance(value, str) or not value for value in review.values()):
            raise ValueError("Empty assay adjudication field")
        if review["source_key"] not in sources:
            raise ValueError("Unknown assay adjudication source")
        if review["review_status"] not in {"partially_resolved", "unresolved"}:
            raise ValueError("Unsupported assay adjudication status")
        if review["training_effect"] != "retain_excluded":
            raise ValueError("Assay adjudication cannot silently admit a training label")
        source_item = sources[review["source_key"]]
        row = {
            "claim_id": f"iedb:tcell:{review['source_assay_id']}:review:{REVIEW_REVISION}",
            "source_dataset": "IEDB",
            "assay_family": "pmhc_multimer",
            "review_revision": REVIEW_REVISION,
            **source_item,
            **{key: value for key, value in review.items() if key != "source_key"},
        }
        rows.append(row)
    if {row["source_assay_id"] for row in rows} != REVIEW_ASSAY_IDS:
        raise ValueError("Unknown or duplicate reviewed assay ID")
    if len({row["claim_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate assay adjudication claim")
    return sorted(rows, key=lambda row: int(row["source_assay_id"]))


def _indexed_rows(directory, name, columns, assay_ids):
    table = pq.read_table(
        directory / "tcell" / "typed" / f"{name}.parquet",
        columns=columns,
        filters=[("source_assay_id", "in", sorted(assay_ids))],
    )
    rows = table.to_pylist()
    indexed = {row["source_assay_id"]: row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError(f"Duplicate source assay in {name}")
    return indexed


def _validate_source_anchors(directory, reviews):
    """Require the reviewed raw claims and existing exclusion to match exactly."""
    directory = Path(directory)
    assay_ids = {row["source_assay_id"] for row in reviews}
    if len(assay_ids) != len(reviews):
        raise ValueError("Duplicate reviewed assay ID")
    assays = _indexed_rows(
        directory,
        "assays",
        [
            "source_assay_id",
            "assay_family",
            "source_reference_id",
            "reference_pmid",
            "effector_cell_name",
        ],
        assay_ids,
    )
    outcomes = _indexed_rows(
        directory,
        "outcomes",
        [
            "source_assay_id",
            "qualitative_result_raw",
            "n_subjects_tested_raw",
            "n_subjects_positive_raw",
        ],
        assay_ids,
    )
    antigens = _indexed_rows(
        directory,
        "antigen_claims",
        ["source_assay_id", "epitope_name_raw", "reported_mhc_name_raw"],
        assay_ids,
    )
    eligibility = _indexed_rows(
        directory,
        "eligibility",
        [
            "source_assay_id",
            "task",
            "training_status",
            "training_exclusion_reason",
            "direct_peptide_candidate",
        ],
        assay_ids,
    )
    contributors = pq.read_table(
        directory / "tcell/typed/source_contributors.parquet",
        columns=["source_assay_id", "source_dataset", "source_record_id"],
        filters=[("source_assay_id", "in", sorted(assay_ids))],
    ).to_pylist()
    contributor_ids = {
        row["source_assay_id"] for row in contributors if row["source_dataset"] == "IEDB"
    }
    training = pq.read_table(
        directory / "tcell/typed/training_examples.parquet",
        columns=["source_assay_id"],
        filters=[("source_assay_id", "in", sorted(assay_ids))],
    )
    if training.num_rows:
        raise ValueError("Reviewed excluded assay appears in training examples")
    for row in reviews:
        assay_id = row["source_assay_id"]
        assay = assays.get(assay_id)
        if assay is None or (
            assay["assay_family"] != row["assay_family"]
            or assay["source_reference_id"] != row["expected_source_reference_id"]
            or assay["reference_pmid"] != row["source_pmid"]
            or assay["effector_cell_name"] != row["expected_effector_cell_name"]
        ):
            raise ValueError(f"Reviewed assay source changed: {assay_id}")
        outcome = outcomes.get(assay_id)
        if outcome is None or any(
            outcome[column] != row[expected]
            for column, expected in (
                ("qualitative_result_raw", "expected_qualitative_result_raw"),
                ("n_subjects_tested_raw", "expected_n_subjects_tested_raw"),
                ("n_subjects_positive_raw", "expected_n_subjects_positive_raw"),
            )
        ):
            raise ValueError(f"Reviewed source outcome changed: {assay_id}")
        antigen = antigens.get(assay_id)
        if antigen is None or (
            antigen["epitope_name_raw"] != row["expected_peptide_sequence"]
            or antigen["reported_mhc_name_raw"] != row["expected_reported_mhc_raw"]
        ):
            raise ValueError(f"Reviewed antigen or restriction claim changed: {assay_id}")
        eligible = eligibility.get(assay_id)
        if eligible is None or (
            eligible["task"] != "assay_specific_pmhc_multimer_cell_binding"
            or eligible["training_status"] != "excluded"
            or eligible["training_exclusion_reason"] != "qualitative_subject_count_disagreement"
            or eligible["direct_peptide_candidate"] is not True
        ):
            raise ValueError(f"Reviewed assay eligibility changed: {assay_id}")
        if assay_id not in contributor_ids:
            raise ValueError(f"Reviewed assay has no IEDB source contributor: {assay_id}")


def _reconciliation(directory):
    typed = Path(directory) / "tcell/typed"
    assays = pq.ParquetFile(typed / "assays.parquet").metadata.num_rows
    training = pq.ParquetFile(typed / "training_examples.parquet").metadata.num_rows
    eligibility = pq.read_table(
        typed / "eligibility.parquet",
        columns=["task", "direct_peptide_candidate", "training_status"],
    )
    multimer = [
        row
        for row in eligibility.to_pylist()
        if row["task"] == "assay_specific_pmhc_multimer_cell_binding"
        and row["direct_peptide_candidate"]
    ]
    result = {
        "n_tcell_assays": assays,
        "n_training_examples": training,
        "n_multimer_candidates": len(multimer),
        "n_multimer_accepted": sum(row["training_status"] == "accepted" for row in multimer),
        "n_multimer_excluded": sum(row["training_status"] == "excluded" for row in multimer),
    }
    if result != {
        "n_tcell_assays": 578_529,
        "n_training_examples": 55_844,
        "n_multimer_candidates": 8_432,
        "n_multimer_accepted": 8_423,
        "n_multimer_excluded": 9,
    }:
        raise ValueError("T-cell assay or training baseline changed")
    return result


def verify_assay_adjudication_package(directory, *, manifest_path="manifest.json"):
    """Verify the complete successor, parent chain and review-to-assay anchors."""
    directory = Path(directory)
    location = _safe_relative(manifest_path)
    manifest = json.loads((directory / location).read_text())
    if (manifest.get("schema_version"), manifest.get("catalog_schema_version")) != (
        PACKAGE_SCHEMA_VERSION,
        CATALOG_SCHEMA_VERSION,
    ):
        raise ValueError("Unsupported assay adjudication package schema")
    if location == Path("manifest.json"):
        inventory = {
            str(path.relative_to(directory))
            for path in directory.rglob("*")
            if path.is_file() and path != directory / "manifest.json"
        }
        if set(manifest["files"]) != inventory:
            raise ValueError("Assay adjudication package file inventory mismatch")
    for relative, expected in manifest["files"].items():
        if file_digest(directory / _safe_relative(relative)) != expected:
            raise ValueError(f"Assay adjudication artifact changed: {relative}")
    parent_path = "manifests/reference.json"
    if manifest["files"][parent_path]["sha256"] != manifest["parent_manifest_sha256"]:
        raise ValueError("Assay adjudication parent manifest changed")
    parent = verify_reference_evidence_package(directory, manifest_path=parent_path)
    if parent["catalog_schema_version"] != 9:
        raise ValueError("Unexpected assay adjudication predecessor catalog")
    if manifest.get("review_revision") != REVIEW_REVISION:
        raise ValueError("Assay adjudication review revision changed")
    if manifest.get("source_review_sha256") != REVIEW_SHA256:
        raise ValueError("Assay adjudication review receipt changed")
    reviews = load_assay_adjudications(directory / REVIEW_ASSET)
    parquet = pq.ParquetFile(directory / REVIEW_TABLE)
    if parquet.schema_arrow != REVIEW_SCHEMA or parquet.metadata.num_rows != len(reviews):
        raise ValueError("Assay adjudication typed relation changed")
    if pq.read_table(directory / REVIEW_TABLE).to_pylist() != reviews:
        raise ValueError("Assay adjudication rows differ from pinned review input")
    _validate_source_anchors(directory, reviews)
    if manifest.get("reconciliation") != {**_reconciliation(directory), "n_assay_reviews": 9}:
        raise ValueError("Assay adjudication reconciliation changed")
    return manifest


def write_assay_adjudication_package(reference_package, destination):
    """Append nine source-linked interpretations to an independent frozen copy."""
    reference_package, destination = Path(reference_package), Path(destination)
    if destination.exists():
        raise FileExistsError(f"Assay adjudication package already exists: {destination}")
    parent = verify_reference_evidence_package(reference_package)
    reviews = load_assay_adjudications()
    _validate_source_anchors(reference_package, reviews)
    reconciliation = {**_reconciliation(reference_package), "n_assay_reviews": len(reviews)}
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    try:
        for relative in parent["files"]:
            snapshot_file(reference_package / _safe_relative(relative), staging / relative)
        snapshot_file(reference_package / "manifest.json", staging / "manifests/reference.json")
        snapshot_file(REVIEW_SOURCE, staging / REVIEW_ASSET)
        table_path = staging / REVIEW_TABLE
        table_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(
            pa.Table.from_pylist(reviews, schema=REVIEW_SCHEMA), table_path, compression="zstd"
        )
        files_manifest = {
            str(path.relative_to(staging)): file_digest(path)
            for path in sorted(staging.rglob("*"))
            if path.is_file()
        }
        from .version import __version__

        manifest = {
            "schema_version": PACKAGE_SCHEMA_VERSION,
            "catalog_schema_version": CATALOG_SCHEMA_VERSION,
            "hitlist_version": __version__,
            "parent_manifest_sha256": files_manifest["manifests/reference.json"]["sha256"],
            "review_revision": REVIEW_REVISION,
            "source_review_sha256": REVIEW_SHA256,
            "reconciliation": reconciliation,
            "files": files_manifest,
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        verify_assay_adjudication_package(staging)
        staging.rename(destination)
        return destination / "manifest.json"
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
