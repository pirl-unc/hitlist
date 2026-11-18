import copy
import json

import pandas as pd
import pytest

from hitlist.lineage import (
    APPLICABILITY_COLUMNS,
    attach_lineage,
    attach_lineage_applicability,
    axis_applicability_status,
    classify_assay_preparation,
    load_lineage,
    validate_lineage,
)
from hitlist.split_audit import audit_splits


@pytest.fixture
def registry():
    entities = []
    for kind, ids in {
        "specimen": ["s1", "s2", "renamed-s1"],
        "donor": ["d1"],
        "experimental_origin": ["o1", "o2"],
        "acquisition": ["a1"],
    }.items():
        for value in ids:
            entities.append(
                {
                    "identity_id": value,
                    "kind": kind,
                    "namespace": "fixture:study",
                    "reported_id": value,
                    "evidence_reference": "fixture:table-S1",
                    "evidence_note": "Explicit sample inventory and reuse statement",
                }
            )
    contexts = []
    for pmid, specimen, origin in [(1, "s1", "o1"), (2, "s2", "o2"), (3, "renamed-s1", "o1")]:
        contexts.append(
            {
                "pmid": pmid,
                "condition_id": "arm",
                "evidence_reference": "fixture:S1",
                "specimen": {"ids": [specimen], "status": "resolved"},
                "donor": {"ids": ["d1"], "status": "resolved"},
                "experimental_origin": {"ids": [origin], "status": "resolved"},
            }
        )
    contexts.append(
        {
            "pmid": 4,
            "condition_id": "arm",
            "evidence_reference": "fixture:S1",
            "specimen": {"ids": ["s1", "s2"], "status": "pooled"},
        }
    )
    return {
        "schema_version": 1,
        "entities": entities,
        "contexts": contexts,
        "aliases": [
            {
                "identity_id": "renamed-s1",
                "canonical_id": "s1",
                "evidence_reference": "fixture:S2",
                "evidence_note": "The paper explicitly names the earlier specimen",
            }
        ],
    }


def _observations(pmids, registry, peptides=None):
    return attach_lineage(
        pd.DataFrame(
            {
                "pmid": pmids,
                "condition_id": "arm",
                "sample_attribution": "curated_sample_label",
                "evidence_kind": "ms",
                "evidence_row_id": [f"ms:assay:{p}" for p in pmids],
                "peptide": peptides or [f"PEPTIDE{p}" for p in pmids],
                "mhc_restriction": "HLA-A*02:01",
                "has_peptide_level_allele": True,
            }
        ),
        registry,
    )


def test_specimens_are_separate_from_donors_and_peptides(registry):
    rows = _observations([1, 2], registry, ["SLYNTVATL"] * 2)
    partitions = {"train": rows.iloc[:1], "test": rows.iloc[1:]}
    assert (
        audit_splits(partitions, policy="specimen_disjoint", registry=registry)["verdict"] == "pass"
    )
    assert audit_splits(partitions, policy="donor_disjoint", registry=registry)["verdict"] == "fail"
    assert (
        audit_splits(partitions, policy="peptide_disjoint", registry=registry)["verdict"] == "fail"
    )
    assert audit_splits(partitions, policy="pmhc_disjoint", registry=registry)["verdict"] == "fail"
    report = audit_splits(partitions, policy="independent_experiments", registry=registry)
    assert report["verdict"] == "pass"
    assert report["pairs"][0]["overlaps"]["pmhc"]["n_shared_values"] == 1


def test_reviewed_aliases_and_mapping_alternatives_cross_partitions(registry):
    rows = _observations([1, 3], registry)
    partitions = {"train": pd.concat([rows.iloc[:1]] * 3), "test": rows.iloc[1:]}
    report = audit_splits(partitions, policy="specimen_disjoint", registry=registry)
    assert report["verdict"] == "fail"
    assert report["partitions"]["train"]["n_observations"] == 1
    assert report["partitions"]["train"]["n_mapping_alternative_rows"] == 2
    assert report["pairs"][0]["overlaps"]["specimen"]["shared_values"] == ["s1"]
    duplicated = {"train": rows.iloc[:1], "test": rows.iloc[:1]}
    assert (
        audit_splits(duplicated, policy="independent_experiments", registry=registry)["verdict"]
        == "fail"
    )


def test_unknown_and_pooled_never_certify_independence(registry):
    rows = _observations([2, 4, 5], registry)
    pooled = {"train": rows.iloc[:1], "test": rows.iloc[1:2]}
    # Known pooled membership is still overlap; pooled is not an independent specimen.
    assert audit_splits(pooled, policy="specimen_disjoint", registry=registry)["verdict"] == "fail"
    unknown = {"train": rows.iloc[:1], "test": rows.iloc[2:]}
    report = audit_splits(unknown, policy="specimen_disjoint", registry=registry)
    assert report["verdict"] == "inconclusive"
    assert report["partitions"]["test"]["coverage"]["specimen"]["n_unresolved_observations"] == 1
    assert audit_splits(unknown, policy="whole_study", registry=registry)["verdict"] == "pass"
    # Matching labels and cell lines cannot fill the absent registry context.
    assert rows.iloc[2].specimen_ids == "[]"


def test_missing_columns_and_unresolved_mhc_are_inconclusive(registry):
    frames = {
        "train": pd.DataFrame({"peptide": ["AAA"]}),
        "test": pd.DataFrame({"peptide": ["BBB"]}),
    }
    report = audit_splits(frames, policy="independent_experiments", registry=registry)
    assert report["verdict"] == "inconclusive"
    assert not report["partitions"]["test"]["coverage"]["specimen"]["available"]
    for frame in frames.values():
        frame["mhc_restriction"] = "HLA class I"
    assert (
        audit_splits(frames, policy="pmhc_disjoint", registry=registry)["verdict"] == "inconclusive"
    )


def test_ambiguous_arm_cannot_inherit_resolved_specimen(registry):
    rows = _observations([1], registry)
    rows["sample_attribution"] = "group_ambiguous"
    result = attach_lineage(rows, registry)
    assert result.iloc[0].specimen_ids == "[]"
    assert result.iloc[0].lineage_context_id == ""


def test_registry_rejects_unsupported_aliases_and_kinds(registry):
    broken = copy.deepcopy(registry)
    broken["aliases"][0]["canonical_id"] = "d1"
    with pytest.raises(ValueError, match="alias"):
        validate_lineage(broken)
    broken = copy.deepcopy(registry)
    broken["contexts"][0]["donor"]["ids"] = ["s1"]
    with pytest.raises(ValueError, match="donor identity"):
        validate_lineage(broken)


def test_shipped_origin_links_have_precise_contexts_without_specimen_inference():
    registry = load_lineage()
    rows = pd.DataFrame(
        {
            "pmid": [28228285, 31844290, 25576301, 27846572],
            "condition_id": ["721_221_hla_a_02_01"] * 2
            + ["fib_primary_fibroblasts", "primary_fibroblasts"],
            "sample_attribution": "allele_exact",
            "evidence_kind": "ms",
        }
    )
    rows = attach_lineage(rows, registry)
    assert rows.experimental_origin_ids.iloc[0] == rows.experimental_origin_ids.iloc[1]
    assert rows.experimental_origin_ids.iloc[2] == rows.experimental_origin_ids.iloc[3]
    assert all(json.loads(value) for value in rows.experimental_origin_ids)
    assert set(rows.specimen_status) == {"unknown"}
    assert set(rows.donor_status) == {"unknown"}
    assert len([c for c in registry["contexts"] if c["pmid"] == 31844290]) == 16


def test_contributor_uncertainty_and_original_publications_are_audited(registry):
    rows = _observations([1, 2], registry)
    rows["contributor_resolution"] = ["captured", "overlap_unresolved"]
    rows["contributor_pmids"] = ['["1", "99"]', '["2", "99"]']
    rows["contributor_study_status"] = "resolved"
    partitions = {"train": rows.iloc[:1], "test": rows.iloc[1:]}
    report = audit_splits(partitions, policy="independent_experiments", registry=registry)
    assert report["verdict"] == "inconclusive"
    assert report["pairs"][0]["overlaps"]["study"]["shared_values"] == ["99"]
    assert audit_splits(partitions, policy="whole_study", registry=registry)["verdict"] == "fail"


def test_donor_sets_are_not_exact_pmhcs(registry):
    rows = _observations([1, 2], registry)
    rows["mhc_restriction"] = "HLA-A*02:01;HLA-A*03:01"
    partitions = {"train": rows.iloc[:1], "test": rows.iloc[1:]}
    report = audit_splits(partitions, policy="pmhc_disjoint", registry=registry)
    assert report["verdict"] == "inconclusive"
    assert report["pairs"][0]["overlaps"]["pmhc"]["n_shared_values"] == 0


def test_assay_identity_normalizes_database_copies(registry):
    rows = _observations([1, 2], registry)
    rows["evidence_row_id"] = [
        "ms:http://www.iedb.org/assay/123",
        "ms:https://cedar.iedb.org/assay/123",
    ]
    report = audit_splits(
        {"train": rows.iloc[:1], "test": rows.iloc[1:]},
        policy="independent_experiments",
        registry=registry,
    )
    assert report["verdict"] == "fail"
    assert report["pairs"][0]["overlaps"]["observation"]["n_shared_values"] == 1


def test_classify_assay_preparation_distinguishes_preparations_and_conflicts():
    assert classify_assay_preparation("purified MHC/competitive/radioactivity", "binding") == (
        "purified_mhc",
        "explicit_purified_mhc_method",
    )
    assert classify_assay_preparation("cellular MHC/competitive/fluorescence", "binding") == (
        "cellular_mhc",
        "explicit_cellular_mhc_method",
    )
    assert classify_assay_preparation("lysate MHC/direct/radioactivity", "binding") == (
        "lysate_mhc",
        "explicit_lysate_mhc_method",
    )
    assert classify_assay_preparation("mass spectrometry", "ms") == (
        "cell_associated_ms",
        "native_elution_evidence",
    )
    # Numeric assay phrasing without an explicit preparation clause never implies purified MHC.
    assert classify_assay_preparation("competitive radioactivity IC50", "binding") == (
        "unknown",
        "unspecified_assay_preparation",
    )
    assert classify_assay_preparation("not purified MHC clause", "binding") == (
        "unknown",
        "unspecified_assay_preparation",
    )
    assert classify_assay_preparation(
        "purified MHC; cellular MHC/competitive/fluorescence", "binding"
    ) == ("unknown", "conflicting_assay_preparation")
    assert classify_assay_preparation("purified MHC/direct/mass spectrometry", "ms") == (
        "unknown",
        "conflicting_assay_preparation",
    )
    with pytest.raises(ValueError, match="Unknown lineage axis"):
        axis_applicability_status("invalid_axis", "unknown", "purified_mhc")
    with pytest.raises(ValueError, match="Unknown assay preparation"):
        axis_applicability_status("donor", "unknown", "invented_prep")


def test_attach_lineage_applicability_preserves_native_status_and_handles_generated_rows(registry):
    empty = attach_lineage_applicability(pd.DataFrame())
    assert set(APPLICABILITY_COLUMNS) <= set(empty.columns)

    frame = pd.DataFrame(
        {
            "pmid": [900, 901, 1, 902, 903, 904],
            "condition_id": ["", "", "arm", "", "", ""],
            "sample_attribution": ["", "", "curated_sample_label", "", "", ""],
            "evidence_kind": ["binding", "binding", "ms", "binding", "ms", "binding"],
            "assay_method": [
                "purified MHC/competitive/radioactivity",
                "cellular MHC/competitive/fluorescence",
                "mass spectrometry",
                "",
                "",
                "",
            ],
            "generated_id": ["", "", "", "gen:purified", "gen:cell", "gen:mixed"],
            "parent_assay_contexts": [
                "",
                "",
                "",
                json.dumps(
                    [{"method": "purified MHC/direct/fluorescence", "evidence_kind": "binding"}]
                ),
                json.dumps(
                    [
                        {"method": "cellular MHC/direct/fluorescence", "evidence_kind": "binding"},
                        {"method": "mass spectrometry", "evidence_kind": "ms"},
                    ]
                ),
                json.dumps(
                    [
                        {"method": "purified MHC/direct/fluorescence", "evidence_kind": "binding"},
                        {"method": "cellular MHC/direct/fluorescence", "evidence_kind": "binding"},
                    ]
                ),
            ],
        }
    )
    enriched = attach_lineage(frame, registry, include_applicability=True)
    # Native <axis>_status values are never rewritten by applicability.
    assert list(enriched["donor_status"]) == [
        "unknown",
        "unknown",
        "resolved",
        "unknown",
        "unknown",
        "unknown",
    ]
    assert list(enriched["assay_preparation"]) == [
        "purified_mhc",
        "cellular_mhc",
        "cell_associated_ms",
        "purified_mhc",
        "generated_cellular_consensus",
        "unknown",
    ]
    assert list(enriched["donor_applicability"]) == [
        "not_applicable",
        "applicable_unresolved",
        "applicable_resolved",
        "not_applicable",
        "applicable_unresolved",
        "applicability_unknown",
    ]
    # Experimental origin and acquisition remain applicable even for purified recombinant MHC.
    assert enriched.loc[0, "experimental_origin_applicability"] == "applicable_unresolved"
    assert enriched.loc[0, "acquisition_applicability"] == "applicable_unresolved"

    bad_parent = frame.iloc[[3]].copy()
    bad_parent["parent_assay_contexts"] = '[{"method": "purified MHC"}]'
    with pytest.raises(ValueError, match="parent_assay_contexts"):
        attach_lineage_applicability(bad_parent)


def test_split_audit_reports_applicability_denominators_without_false_pass(registry):
    purified = attach_lineage(
        pd.DataFrame(
            {
                "pmid": [101, 102],
                "condition_id": ["", ""],
                "sample_attribution": ["", ""],
                "evidence_kind": ["binding", "binding"],
                "assay_method": [
                    "purified MHC/competitive/radioactivity",
                    "purified MHC/direct/fluorescence",
                ],
                "evidence_row_id": [
                    "binding:http://iedb.org/assay/101",
                    "binding:http://iedb.org/assay/102",
                ],
                "peptide": ["AAAAAAAAA", "BBBBBBBBB"],
                "mhc_restriction": ["HLA-A*02:01", "HLA-A*03:01"],
                "has_peptide_level_allele": [True, True],
            }
        ),
        registry,
        include_applicability=True,
    )
    report = audit_splits(
        {"train": purified.iloc[:1], "test": purified.iloc[1:]},
        policy="donor_disjoint",
        registry=registry,
    )
    # Non-applicable donor axes must NEVER turn an unreviewed split into "pass".
    assert report["verdict"] == "inconclusive"
    assert report["partitions"]["train"]["by_assay_preparation"] == {"purified_mhc": 1}
    donor_cov = report["partitions"]["train"]["coverage"]["donor"]
    assert donor_cov["n_unresolved_observations"] == 1
    assert donor_cov["n_not_applicable_observations"] == 1
    assert donor_cov["n_applicable_observations"] == 0
    assert report["pairs"][0]["overlaps"]["donor"]["applicability_status"] == "all_not_applicable"
    assert (
        report["pairs"][0]["overlaps"]["experimental_origin"]["applicability_status"]
        == "applicable_unresolved"
    )

    resolved_ms = _observations([1], registry, ["CCCCCCCCC"])
    resolved_ms["assay_method"] = "mass spectrometry"
    mixed_report = audit_splits(
        {
            "train": pd.concat([resolved_ms, purified.iloc[:1]], ignore_index=True),
            "test": purified.iloc[1:],
        },
        policy="donor_disjoint",
        registry=registry,
    )
    assert mixed_report["verdict"] == "inconclusive"
    assert (
        mixed_report["pairs"][0]["overlaps"]["donor"]["applicability_status"]
        == "disjoint_where_applicable"
    )

    cellular_unresolved = purified.iloc[1:].copy()
    cellular_unresolved["assay_method"] = "cellular MHC/competitive/fluorescence"
    cellular_unresolved = attach_lineage_applicability(cellular_unresolved)
    cell_report = audit_splits(
        {"train": resolved_ms, "test": cellular_unresolved},
        policy="donor_disjoint",
        registry=registry,
    )
    assert (
        cell_report["partitions"]["test"]["coverage"]["donor"][
            "n_applicable_unresolved_observations"
        ]
        == 1
    )
    assert (
        cell_report["pairs"][0]["overlaps"]["donor"]["applicability_status"]
        == "applicable_unresolved"
    )
