"""Assay-level source reviews do not alter measured labels or assay weight."""

from __future__ import annotations

import json
import shutil

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from hitlist import assay_adjudication as review
from hitlist.assay_adjudication import _validate_source_anchors, load_assay_adjudications
from hitlist.evidence_bindings import materialized_bindings
from hitlist.evidence_store import EvidenceStore
from hitlist.provenance import file_digest


def _write(directory, name, rows):
    path = directory / "tcell" / "typed" / f"{name}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def _source_fixture(tmp_path, *, n_subjects_positive="3", training_status="excluded"):
    _write(
        tmp_path,
        "assays",
        [
            {
                "source_assay_id": "1646138",
                "assay_family": "pmhc_multimer",
                "source_reference_id": "1014650",
                "reference_pmid": "10618411",
                "effector_cell_name": "T cell CD4+",
            }
        ],
    )
    _write(
        tmp_path,
        "outcomes",
        [
            {
                "source_assay_id": "1646138",
                "qualitative_result_raw": "Negative",
                "n_subjects_tested_raw": "20",
                "n_subjects_positive_raw": n_subjects_positive,
            }
        ],
    )
    _write(
        tmp_path,
        "antigen_claims",
        [
            {
                "source_assay_id": "1646138",
                "epitope_name_raw": "RSFTLASSETGVG",
                "reported_mhc_name_raw": "HLA-DRB1*04:01",
            }
        ],
    )
    _write(
        tmp_path,
        "eligibility",
        [
            {
                "source_assay_id": "1646138",
                "task": "assay_specific_pmhc_multimer_cell_binding",
                "training_status": training_status,
                "training_exclusion_reason": "qualitative_subject_count_disagreement",
                "direct_peptide_candidate": True,
            }
        ],
    )
    _write(
        tmp_path,
        "source_contributors",
        [
            {
                "source_assay_id": "1646138",
                "source_dataset": "IEDB",
                "source_record_id": "iedb:tcell:r107153",
            }
        ],
    )
    _write(tmp_path, "training_examples", [{"source_assay_id": "other"}])


def test_review_source_anchors_reject_changed_counts_and_training_admission(tmp_path):
    reviews = load_assay_adjudications()
    assert len(reviews) == 9
    assert {row["training_effect"] for row in reviews} == {"retain_excluded"}
    assert {row["review_status"] for row in reviews} == {"partially_resolved", "unresolved"}
    row = next(row for row in reviews if row["source_assay_id"] == "1646138")
    _source_fixture(tmp_path)
    _validate_source_anchors(tmp_path, [row])

    _source_fixture(tmp_path, n_subjects_positive="0")
    with pytest.raises(ValueError, match="source outcome"):
        _validate_source_anchors(tmp_path, [row])

    _source_fixture(tmp_path, training_status="accepted")
    with pytest.raises(ValueError, match="eligibility"):
        _validate_source_anchors(tmp_path, [row])


def test_review_verifier_accepts_embedded_manifest_without_claiming_outer_files(
    tmp_path, monkeypatch
):
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    parent_path = manifests / "reference.json"
    parent_path.write_text('{"catalog_schema_version": 9}\n')
    review_dir = tmp_path / "tcell" / "adjudication"
    review_dir.mkdir(parents=True)
    yaml_path = review_dir / "reviews.yaml"
    shutil.copyfile(review.REVIEW_SOURCE, yaml_path)
    parquet_path = review_dir / "reviews.parquet"
    pq.write_table(
        pa.Table.from_pylist(load_assay_adjudications(), schema=review.REVIEW_SCHEMA),
        parquet_path,
    )
    files = {
        str(path.relative_to(tmp_path)): file_digest(path)
        for path in (parent_path, yaml_path, parquet_path)
    }
    manifest = {
        "schema_version": 8,
        "catalog_schema_version": 10,
        "parent_manifest_sha256": files["manifests/reference.json"]["sha256"],
        "review_revision": review.REVIEW_REVISION,
        "source_review_sha256": review.REVIEW_SHA256,
        "reconciliation": {"n_assay_reviews": 9},
        "files": files,
    }
    (manifests / "adjudication.json").write_text(json.dumps(manifest))
    (tmp_path / "external.txt").write_text("owned by the outer successor")
    monkeypatch.setattr(
        review,
        "verify_reference_evidence_package",
        lambda directory, *, manifest_path: {"catalog_schema_version": 9},
    )
    monkeypatch.setattr(review, "_validate_source_anchors", lambda directory, rows: None)
    monkeypatch.setattr(review, "_reconciliation", lambda directory: {})
    assert (
        review.verify_assay_adjudication_package(
            tmp_path, manifest_path="manifests/adjudication.json"
        )
        == manifest
    )


def test_review_relation_requires_explicit_presence():
    assert "assay_adjudication" not in materialized_bindings(6)
    assert "external_reference_sequence" not in materialized_bindings(6)
    assert "external_reference_sequence" in materialized_bindings(7)
    assert "assay_adjudication" not in materialized_bindings(7)
    assert "assay_adjudication" not in materialized_bindings(8)
    assert "assay_adjudication" in materialized_bindings(8, include_adjudications=True)
    assert "assay_adjudication" in materialized_bindings(9, include_adjudications=True)


def test_one_review_joins_to_one_assay_without_multiplying_it(tmp_path):
    (tmp_path / "manifest.json").write_text('{"schema_version": 8}\n')
    _write(tmp_path, "assays", [{"source_assay_id": "1646138", "assay_family": "pmhc_multimer"}])
    path = tmp_path / "tcell" / "adjudication" / "reviews.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "source_assay_id": "1646138",
                    "claim_id": "review:1646138",
                    "review_status": "partially_resolved",
                }
            ]
        ),
        path,
    )
    store = EvidenceStore.__new__(EvidenceStore)
    store.directory = tmp_path
    store.manifest = {"schema_version": 8}
    bindings = materialized_bindings(8, include_adjudications=True)
    store.bindings = {name: bindings[name] for name in ("tcell_availability", "assay_adjudication")}
    store.schemas = {
        name: pq.ParquetFile(tmp_path / binding[0].materialized_artifact).schema_arrow
        for name, binding in store.bindings.items()
    }
    result = store.query(
        "tcell_availability",
        select=["source_assay_id", "adjudication.review_status"],
        joins=["adjudication"],
    )
    assert result.table.to_pylist() == [
        {
            "source_assay_id": "1646138",
            "adjudication.review_status": "partially_resolved",
        }
    ]


def test_successor_registry_keeps_parent_reference_and_review_sources(tmp_path):
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    values = {
        "mhc": {"source_registry": {}},
        "context": {"source_registry": {"corrected_human_cohort": {}}},
        "supporting": {"source_registry": {"bulk_v4": {}, "line_expression_v4": {}}},
        "tcell": {"source_registry": {}},
        "external": {"sources": {"pxd057588": {"asset_receipt_sha256": "receipt"}}},
        "reference": {"reference": {"presto_commit": "pinned"}},
    }
    for name, value in values.items():
        (manifests / f"{name}.json").write_text(json.dumps(value))
    store = EvidenceStore.__new__(EvidenceStore)
    store.directory = tmp_path
    store.manifest = {
        "schema_version": 8,
        "review_revision": "tcell_multimer_2026_10_23",
        "source_review_sha256": "review_hash",
        "reconciliation": {"n_assay_reviews": 9},
    }
    registry = store._source_registry()
    assert registry["pxd057588_mouse_reference_2026_03"] == {"presto_commit": "pinned"}
    assert registry["tcell_multimer_review_2026_10_23"]["n_assay_reviews"] == 9
