"""Offline regression and contract tests for #680 (lineage holdout gate) and #678 (frozen release gate)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import hitlist
from hitlist.lineage import attach_lineage, lineage_registry_sha256, load_lineage
from hitlist.split_audit import audit_splits
from hitlist.training_release_gate import (
    assign_lineage_grouped_splits,
    audit_training_release_diff,
    build_training_release_freshness_receipt,
    evaluate_training_release_gate,
    training_release_gate_contract,
)


def _registry_contract_frame() -> pd.DataFrame:
    """Source-backed observations spanning Abelin/Sarkizova, Bassani/Liepe, and canine Lola."""
    rows = pd.DataFrame(
        {
            "evidence_row_id": [
                "ms:abelin:a0201",
                "ms:sarkizova:a0201",
                "ms:abelin:a0101",
                "ms:sarkizova:a0101",
                "ms:bassani:fib",
                "ms:liepe:fib",
                "ms:miyamae:lola_h58a",
                "ms:miyamae:lola_bb76",
                "ms:miyamae:bogey_bb76",
                "ms:miyamae:lily_h58a",
            ],
            "pmid": [
                28228285,
                31844290,
                28228285,
                31844290,
                25576301,
                27846572,
                42199926,
                42199926,
                42199926,
                42199926,
            ],
            "condition_id": [
                "721_221_hla_a_02_01",
                "721_221_hla_a_02_01",
                "721_221_hla_a_01_01",
                "721_221_hla_a_01_01",
                "fib_primary_fibroblasts",
                "primary_fibroblasts",
                "lola_h58a",
                "lola_bb7_6",
                "bogey_bb7_6",
                "lily_h58a",
            ],
            "sample_attribution": "curated_sample_label",
            "evidence_kind": "ms",
            "peptide": [
                "SLYNTVATL",
                "SLYNTVATL",
                "YLEPGPVTA",
                "YLEPGPVTA",
                "GILGFVFTL",
                "GILGFVFTL",
                "KLQCVDLHV",
                "KLQCVDLHV",
                "NLVPMVATV",
                "GLCTLVAML",
            ],
            "mhc_restriction": [
                "HLA-A*02:01",
                "HLA-A*02:01",
                "HLA-A*01:01",
                "HLA-A*01:01",
                "HLA-A*02:01",
                "HLA-A*02:01",
                "DLA-88*003:02",
                "DLA-88*003:02",
                "DLA-88*012:01",
                "DLA-88*501:01",
            ],
            "has_peptide_level_allele": True,
        }
    )
    return attach_lineage(rows, load_lineage())


def test_public_exports_and_contract_integrity():
    for symbol in (
        "lineage_registry_sha256",
        "training_release_gate_contract",
        "build_training_release_freshness_receipt",
        "audit_training_release_diff",
        "assign_lineage_grouped_splits",
        "evaluate_training_release_gate",
    ):
        with pytest.warns(DeprecationWarning):
            assert hasattr(hitlist, symbol), f"Missing deprecated shim export: {symbol}"

    contract = training_release_gate_contract()
    assert contract["schema_version"] == 1
    assert set(contract["tracked_issues"]) == {678, 680}
    baseline = contract["historical_baseline"]
    assert baseline["release_tag"] == "human-cohort-corrected-v1"
    assert (
        baseline["native_manifest_sha256"]
        == "b180eba804532f633a6d61bb350584dbe24f8df4b4b96f02c6ec492c60144372"
    )
    assert baseline["hitlist_version"] == "1.78.0"

    # Every referenced contract file must exist in hitlist/data/evidence_revisions/.
    revisions_dir = Path(hitlist.__file__).parent / "data" / "evidence_revisions"
    for entry in contract["integrated_evidence_modules"]:
        if entry.get("contract_file"):
            assert (revisions_dir / entry["contract_file"]).exists(), entry["contract_file"]

    # Every source-backed reuse link must match the shipped specimen_lineage.yaml registry.
    registry = load_lineage()
    entity_ids = {e["identity_id"] for e in registry["entities"]}
    for link in contract["source_backed_reuse_links"]:
        expected = link["expected_n_shared_origins"]
        if expected > 0:
            matched = [eid for eid in entity_ids if eid.startswith(link["identity_prefix"])]
            assert len(matched) == expected, link["link_id"]


def test_split_audit_freezes_registry_hash_and_per_status_coverage():
    registry = load_lineage()
    expected_sha = lineage_registry_sha256(registry)
    assert len(expected_sha) == 64

    frame = _registry_contract_frame()
    # Intentionally split Abelin (28228285) vs Sarkizova (31844290) and Lola H58A vs BB7.6
    train = frame.loc[frame["evidence_row_id"].isin(["ms:abelin:a0201", "ms:miyamae:lola_h58a"])]
    test = frame.loc[frame["evidence_row_id"].isin(["ms:sarkizova:a0201", "ms:miyamae:lola_bb76"])]

    report = audit_splits({"train": train, "test": test}, policy="independent_experiments")
    assert report["verdict"] == "fail"
    assert report["lineage_registry_sha256"] == expected_sha
    assert report["lineage_registry_summary"]["n_entities"] == len(registry["entities"])
    assert report["lineage_registry_summary"]["n_contexts"] == len(registry["contexts"])

    # Observation-weighted per-status counts must be explicit and sum cleanly.
    train_origin_cov = report["partitions"]["train"]["coverage"]["experimental_origin"]
    assert train_origin_cov["n_resolved_observations"] == 2
    assert train_origin_cov["n_unknown_observations"] == 0
    assert train_origin_cov["n_pooled_observations"] == 0
    assert train_origin_cov["n_ambiguous_observations"] == 0
    assert train_origin_cov["coverage_fraction"] == pytest.approx(1.0)

    train_donor_cov = report["partitions"]["train"]["coverage"]["donor"]
    assert train_donor_cov["n_resolved_observations"] == 1
    assert train_donor_cov["n_unknown_observations"] == 1
    assert train_donor_cov["coverage_fraction"] == pytest.approx(0.5)

    # Cross-partition reuse must catch BOTH Abelin/Sarkizova origin reuse AND Lola donor/specimen reuse!
    reuse_by_axis = {entry["axis"]: entry for entry in report["cross_partition_reuse"]}
    assert (
        "origin:MSV000080527:721_221_hla_a_02_01"
        in reuse_by_axis["experimental_origin"]["shared_values"]
    )
    assert "donor:pmid:42199926:lola" in reuse_by_axis["donor"]["shared_values"]
    assert "specimen:pmid:42199926:lola" in reuse_by_axis["specimen"]["shared_values"]


def test_assign_lineage_grouped_splits_keeps_originals_and_reanalyses_together():
    frame = _registry_contract_frame()
    assigned, summary = assign_lineage_grouped_splits(
        frame,
        fractions={"train": 0.6, "val": 0.2, "test": 0.2},
        group_axes=("experimental_origin", "specimen", "donor"),
        policy="independent_experiments",
    )
    by_id = assigned.set_index("evidence_row_id")["split_partition"].to_dict()

    # Abelin/Sarkizova reused origins must stay together
    assert by_id["ms:abelin:a0201"] == by_id["ms:sarkizova:a0201"]
    assert by_id["ms:abelin:a0101"] == by_id["ms:sarkizova:a0101"]
    # Bassani-Sternberg/Liepe fibroblast reuse must stay together
    assert by_id["ms:bassani:fib"] == by_id["ms:liepe:fib"]
    # Canine Lola dual IPs share donor + specimen and must stay together
    assert by_id["ms:miyamae:lola_h58a"] == by_id["ms:miyamae:lola_bb76"]

    assert summary["lineage_registry_sha256"] == lineage_registry_sha256()
    assert summary["n_observations"] == len(frame)
    assert summary["n_reviewed_lineage_observations"] == len(frame)
    assert summary["n_unresolved_fallback_observations"] == 0
    assert summary["audit"]["verdict"] == "pass"
    # Zero cross-partition reuse on observation or any biological lineage axis
    bio_reuse = [
        entry
        for entry in summary["audit"]["cross_partition_reuse"]
        if entry["axis"]
        in {"observation", "experimental_origin", "specimen", "donor", "acquisition"}
    ]
    assert bio_reuse == []


def test_unknown_lineage_never_fabricates_ids_and_stays_inconclusive():
    """#680: missing biological identities must yield inconclusive, never pass."""
    frame = pd.DataFrame(
        {
            "evidence_row_id": [
                "ms:atlas:donor1:obs1",
                "ms:atlas:donor2:obs2",
                "ms:unknown:studyA:obs3",
                "ms:unknown:studyB:obs4",
            ],
            "pmid": [33858848, 33858848, 99990001, 99990002],
            "condition_id": ["donor_1", "donor_2", "arm_a", "arm_b"],
            "sample_attribution": "curated_sample_label",
            "evidence_kind": "ms",
            "peptide": ["AAAAAAAAA", "BBBBBBBBB", "CCCCCCCCC", "DDDDDDDDD"],
            "mhc_restriction": "HLA-A*02:01",
            "has_peptide_level_allele": True,
        }
    )
    assigned, summary = assign_lineage_grouped_splits(
        frame,
        fractions={"train": 0.5, "test": 0.5},
        policy="independent_experiments",
    )
    assert set(assigned["experimental_origin_status"]) == {"unknown"}
    assert set(assigned["specimen_status"]) == {"unknown"}
    assert set(assigned["acquisition_status"]) == {"unknown"}
    assert set(assigned["lineage_group_basis"]) == {"unresolved_fallback_pmid"}
    assert summary["n_unresolved_fallback_observations"] == 4
    assert summary["audit"]["verdict"] == "inconclusive"

    partitions = {
        name: assigned.loc[assigned["split_partition"] == name]
        for name in sorted(assigned["split_partition"].unique())
    }
    gate = evaluate_training_release_gate(partitions, policy="independent_experiments")
    assert gate["lineage_verdict"] == "inconclusive"
    assert gate["overall_gate_status"] == "inconclusive_lineage"
    assert "lineage_inconclusive:independent_experiments" in gate["blocking_reasons"]
    assert gate["cohort_coverage"]["experimental_origin"]["n_unknown_observations"] == 4
    assert gate["cohort_coverage"]["experimental_origin"]["coverage_fraction"] == 0.0


def test_freshness_receipt_rejects_historical_baseline_overwrite_and_pins_contracts():
    with pytest.raises(ValueError, match="historical release tag"):
        build_training_release_freshness_receipt(release_tag="human-cohort-corrected-v1")
    with pytest.raises(ValueError, match="historical release tag"):
        build_training_release_freshness_receipt(release_tag="ci-corpus-v4")

    receipt = build_training_release_freshness_receipt(
        release_tag="human-cohort-corrected-v2",
        upstream_snapshots={
            "iedb": {
                "sha256": "a" * 64,
                "snapshot_date": "2026-11-25",
                "size_bytes": 123456,
            }
        },
    )
    assert receipt["release_tag"] == "human-cohort-corrected-v2"
    assert receipt["replaces_historical_baseline"] is False
    assert receipt["claims_predictive_improvement"] is False
    assert receipt["lineage_registry_sha256"] == lineage_registry_sha256()
    assert receipt["n_upstream_snapshots"] == 1
    assert receipt["n_integrated_evidence_modules"] >= 9
    assert "peptidomicsdb_qualification.json" in receipt["evidence_contracts"]
    assert "tcrvdb_adapter_contract.json" in receipt["evidence_contracts"]
    assert "canine_dla_atlas_qualification.json" in receipt["evidence_contracts"]
    for contract_entry in receipt["evidence_contracts"].values():
        assert len(contract_entry["sha256"]) == 64


def test_audit_training_release_diff_distinguishes_annotation_vs_source_updates():
    previous = pd.DataFrame(
        {
            "evidence_row_id": ["obs:1", "obs:2", "obs:3"],
            "peptide": ["SLYNTVATL", "GILGFVFTL", "NLVPMVATV"],
            "measurement_value": [12.5, 50.0, 500.0],
            "measurement_units": ["nM", "nM", "nM"],
            "source_tissue": ["", "Blood", "Skin"],
            "condition_id": ["", "c2", "c3"],
        }
    )

    # Case 1: Annotation-only curation refresh (like #314 blank tissue fill + condition curation)
    annotation_refresh = previous.copy()
    annotation_refresh.loc[0, "source_tissue"] = "Pancreas"
    annotation_refresh.loc[0, "condition_id"] = "pdac_primary"
    diff_ann = audit_training_release_diff(previous, annotation_refresh)
    assert diff_ann["original_ids_preserved"] is True
    assert diff_ann["measurement_invariants_preserved"] is True
    assert diff_ann["n_retained_ids"] == 3
    assert diff_ann["n_added_ids"] == 0
    assert diff_ann["n_removed_ids"] == 0
    assert diff_ann["n_annotation_only_changed_rows"] == 1
    assert diff_ann["n_source_acquisition_updated_rows"] == 0
    assert diff_ann["n_unchanged_rows"] == 2
    assert diff_ann["change_classification"] == "annotation_only_refresh"
    assert diff_ann["changed_fields"]["source_tissue"] == {
        "category": "annotation_curation",
        "n_changed_rows": 1,
    }

    # Case 2: Source acquisition update (new row added + measured value mutated -> flagged!)
    source_refresh = pd.concat(
        [
            previous,
            pd.DataFrame(
                {
                    "evidence_row_id": ["obs:4"],
                    "peptide": ["YLEPGPVTA"],
                    "measurement_value": [25.0],
                    "measurement_units": ["nM"],
                    "source_tissue": ["Skin"],
                    "condition_id": ["melanoma_1"],
                }
            ),
        ],
        ignore_index=True,
    )
    source_refresh.loc[1, "measurement_value"] = 55.0
    diff_src = audit_training_release_diff(previous, source_refresh)
    assert diff_src["n_added_ids"] == 1
    assert diff_src["n_source_acquisition_updated_rows"] == 1
    assert diff_src["measurement_invariants_preserved"] is False
    assert diff_src["measurement_invariant_violations"] == {"measurement_value": 1}
    assert diff_src["change_classification"] == "source_acquisition_refresh"


def test_evaluate_training_release_gate_end_to_end_qualified_and_blocked():
    frame = _registry_contract_frame()
    assigned, _ = assign_lineage_grouped_splits(
        frame,
        fractions={"train": 0.6, "test": 0.4},
        policy="independent_experiments",
    )
    partitions = {
        name: assigned.loc[assigned["split_partition"] == name]
        for name in sorted(assigned["split_partition"].unique())
    }
    receipt = build_training_release_freshness_receipt(release_tag="human-cohort-corrected-v2")
    diff_ok = audit_training_release_diff(
        frame[["evidence_row_id", "peptide", "condition_id"]],
        frame[["evidence_row_id", "peptide", "condition_id"]],
    )

    gate_ok = evaluate_training_release_gate(
        partitions,
        policy="independent_experiments",
        freshness_receipt=receipt,
        diff_audit=diff_ok,
    )
    assert gate_ok["overall_gate_status"] == "qualified"
    assert gate_ok["blocking_reasons"] == []
    assert gate_ok["claims_predictive_improvement"] is False

    # Stale receipt or mismatched registry hash must block release integrity
    stale_receipt = dict(receipt)
    stale_receipt["cache_freshness"] = {"observations": "stale"}
    gate_stale = evaluate_training_release_gate(
        partitions,
        policy="independent_experiments",
        freshness_receipt=stale_receipt,
        diff_audit=diff_ok,
    )
    assert gate_stale["overall_gate_status"] == "blocked_release_integrity"
    assert "stale_observation_index:stale" in gate_stale["blocking_reasons"]


def test_split_audit_reports_pooled_and_ambiguous_status_counts():
    registry = {
        "schema_version": 1,
        "entities": [
            {
                "identity_id": "s1",
                "kind": "specimen",
                "namespace": "test",
                "reported_id": "s1",
                "evidence_reference": "ref",
                "evidence_note": "note",
            },
            {
                "identity_id": "s2",
                "kind": "specimen",
                "namespace": "test",
                "reported_id": "s2",
                "evidence_reference": "ref",
                "evidence_note": "note",
            },
        ],
        "aliases": [],
        "contexts": [
            {
                "pmid": 1,
                "condition_id": "resolved_arm",
                "evidence_reference": "ref",
                "specimen": {"status": "resolved", "ids": ["s1"]},
            },
            {
                "pmid": 2,
                "condition_id": "pooled_arm",
                "evidence_reference": "ref",
                "specimen": {"status": "pooled", "ids": ["s1", "s2"]},
            },
            {
                "pmid": 3,
                "condition_id": "ambiguous_arm",
                "evidence_reference": "ref",
                "specimen": {"status": "ambiguous", "ids": ["s2"]},
            },
        ],
    }
    rows = attach_lineage(
        pd.DataFrame(
            {
                "evidence_row_id": ["obs:1", "obs:2", "obs:3", "obs:4"],
                "pmid": [1, 2, 3, 4],
                "condition_id": ["resolved_arm", "pooled_arm", "ambiguous_arm", "unknown_arm"],
                "sample_attribution": "curated_sample_label",
                "evidence_kind": "ms",
                "peptide": ["AAAA", "BBBB", "CCCC", "DDDD"],
                "mhc_restriction": "HLA-A*02:01",
                "has_peptide_level_allele": True,
            }
        ),
        registry,
    )
    report = audit_splits(
        {"train": rows.iloc[:2], "test": rows.iloc[2:]},
        policy="specimen_disjoint",
        registry=registry,
    )
    train_cov = report["partitions"]["train"]["coverage"]["specimen"]
    test_cov = report["partitions"]["test"]["coverage"]["specimen"]
    assert train_cov["n_resolved_observations"] == 1
    assert train_cov["n_pooled_observations"] == 1
    assert train_cov["n_ambiguous_observations"] == 0
    assert train_cov["n_unknown_observations"] == 0
    assert test_cov["n_resolved_observations"] == 0
    assert test_cov["n_pooled_observations"] == 0
    assert test_cov["n_ambiguous_observations"] == 1
    assert test_cov["n_unknown_observations"] == 1
    # Pooled s1/s2 in train overlaps ambiguous s2 in test -> fail!
    assert report["verdict"] == "fail"
    assert json.dumps(report)  # Fully JSON-serializable


def test_freshness_receipt_extracts_verified_bundle_metadata(
    tmp_path, monkeypatch, _isolated_curation_root
):
    from hitlist import builder, downloads, supplement
    from hitlist.parquet_io import atomic_write_parquet
    from hitlist.training_bundle import write_training_bundle
    from tests.test_provenance import _row
    from tests.test_scanner import _write_tiny_iedb_csv

    indexes = tmp_path / "indexes"
    (_isolated_curation_root / "pmid_overrides.yaml").write_text("[]\n")
    monkeypatch.setattr(downloads, "_override_data_dir", indexes)
    source = tmp_path / "iedb.csv"
    _write_tiny_iedb_csv(
        source,
        [
            _row("http://iedb.org/assay/10"),
            _row("http://iedb.org/assay/11", "independent"),
        ],
    )
    downloads.register("iedb", source)
    monkeypatch.setattr(supplement, "scan_supplementary", lambda **kwargs: pd.DataFrame())

    def _empty(path):
        frame = pd.DataFrame()
        atomic_write_parquet(frame, path)
        return frame

    monkeypatch.setattr(
        builder,
        "build_bulk_proteomics",
        lambda **kw: _empty(builder._bulk_proteomics_path()),
    )
    monkeypatch.setattr(
        builder,
        "build_line_expression",
        lambda **kw: _empty(builder._line_expression_path()),
    )
    builder.build_observations(build_mappings=False)

    bundle_dir = tmp_path / "bundle_v2"
    write_training_bundle(bundle_dir, include_evidence="ms")
    receipt = build_training_release_freshness_receipt(
        release_tag="human-cohort-corrected-v2",
        bundle_dir=bundle_dir,
    )
    assert receipt["verified_bundle"]["n_observations"] == 2
    assert receipt["verified_bundle"]["n_rows"] == 2
    assert len(receipt["verified_bundle"]["manifest_sha256"]) == 64
    assert receipt["cache_freshness"]["observations"] == "current"
    assert "iedb" in receipt["upstream_snapshots"]
    assert receipt["upstream_snapshots"]["iedb"]["sha256"]
