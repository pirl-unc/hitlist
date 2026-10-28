"""Cohort context must remain linked to source assays and curated claims."""

import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from hitlist.context_evidence import (
    COHORT_ARTIFACTS,
    COHORT_EXTRA_FIELDS,
    IDENTITY_FIELDS,
    verify_context_evidence_package,
    write_context_evidence_package,
)
from hitlist.mhc_evidence import NATIVE_FIELDS, write_mhc_evidence_package
from hitlist.provenance import file_digest
from tests.test_mhc_evidence import _source


def _cohort_fixture(
    tmp_path,
    *,
    bad_arm=False,
    duplicate_id=False,
    extra_field=False,
    class_pool=False,
    augment_source=None,
):
    source = _source(tmp_path)
    if augment_source is not None:
        augment_source(source)
    mhc_package = tmp_path / "mhc"
    write_mhc_evidence_package(source, mhc_package)
    cohort = tmp_path / "cohort"
    cohort.mkdir()
    original = [
        pq.read_table(source / "observations.parquet").to_pylist()[0],
        pq.read_table(source / "binding.parquet").to_pylist()[0],
    ]
    training = []
    for index, row in enumerate(original):
        augmented = {**row, **dict.fromkeys(COHORT_EXTRA_FIELDS, "")}
        augmented.update(
            {
                "evidence_kind": "ms" if index == 0 else "binding",
                "provenance_status": "source_record",
                "evidence_source_id": row["assay_iri"],
                "evidence_row_id": f"test:{index}",
                "condition_id": "missing_arm"
                if bad_arm and index == 0
                else "test_arm"
                if index == 0
                else "",
                "sample_attribution": ("class_pool" if class_pool else "curated_sample_label")
                if index == 0
                else "",
                "condition_knockout_genes": "TAP1" if index == 0 else "",
                "condition_infection": "test exposure" if index == 0 else "",
                "condition_combination": "simultaneous" if index == 0 else "",
                "cell_type": "test cell" if index == 0 else "",
                "instrument": "arm instrument" if index == 0 else "study instrument",
                "experimental_origin_ids": "[]",
                "experimental_origin_status": "unknown",
                "specimen_ids": "[]",
                "specimen_status": "unknown",
                "donor_ids": "[]",
                "donor_status": "unknown",
                "acquisition_ids": "[]",
                "acquisition_status": "unknown",
            }
        )
        if extra_field:
            augmented["unexpected_context"] = "unreviewed"
        training.append(augmented)
    if duplicate_id:
        training[1]["provenance_id"] = training[0]["provenance_id"]
    assert list(training[0]) == [*NATIVE_FIELDS, *COHORT_EXTRA_FIELDS] + (
        ["unexpected_context"] if extra_field else []
    )
    pq.write_table(pa.Table.from_pylist(training), cohort / "training.parquet")
    identities = []
    for row in training:
        identity = {field: row.get(field) for field in IDENTITY_FIELDS}
        identity.update(
            {
                "contributor_resolution": "captured",
                "contributor_pmids": '["12345678"]',
                "contributor_study_status": "resolved",
            }
        )
        identities.append(identity)
    pq.write_table(pa.Table.from_pylist(identities), cohort / "identities.parquet")
    contributors = pq.read_table(source / "observation_contributors.parquet")
    contributors = contributors.filter(
        pa.compute.is_in(
            contributors["provenance_id"],
            value_set=pa.array([original[0]["provenance_id"], original[1]["provenance_id"]]),
        )
    )
    pq.write_table(contributors, cohort / "contributors.parquet")
    (cohort / "lineage.json").write_text(
        '{"schema_version": 1, "entities": [], "aliases": [], "contexts": []}\n'
    )
    old_yaml = tmp_path / "historical.yaml"
    old_yaml.write_text(
        """- pmid: 12345678
  instrument: study instrument
  ms_samples:
  - condition_id: test_arm
    sample_label: test arm
    condition: edited and exposed
    condition_status: annotated
    condition_evidence: curated_text
    condition_combination: simultaneous
    condition_knockout_genes: TAP1
    condition_infection: test exposure
    mhc: HLA-A*02:01
    mhc_basis: sample_typing
    mhc_genotype: HLA-A*01:01
    mhc_genotype_cell: test cell
    mhc_genotype_source: test citation
    instrument: arm instrument
  - condition_id: test_control
    sample_label: control arm
    condition: control
    condition_status: annotated
    condition_evidence: curated_text
    condition_control: untreated
    condition_control_for: test_arm
"""
    )
    current_yaml = tmp_path / "current.yaml"
    current_yaml.write_text(
        old_yaml.read_text().replace(
            "    condition_control: untreated\n",
            "    condition_control: untreated\n    condition_knockout_genes: none\n",
        )
    )
    lineage = tmp_path / "lineage.yaml"
    lineage.write_text("schema_version: 1\nentities: []\ncontexts: []\n")
    mhc_manifest = json.loads((mhc_package / "manifest.json").read_text())
    artifacts = {
        name: {"file": filename, **file_digest(cohort / filename)}
        for name, filename in COHORT_ARTIFACTS.items()
    }
    inputs = {
        name: mhc_manifest["files"][f"native/{name}"]
        for name in (
            "observations.parquet",
            "binding.parquet",
            "other_assays.parquet",
            "observation_contributors.parquet",
        )
    }
    inputs["curation:pmid_overrides.yaml"] = file_digest(old_yaml)
    inputs["specimen_lineage.yaml"] = file_digest(lineage)
    (cohort / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "n_rows": len(training),
                "artifacts": artifacts,
                "inputs": {"files": inputs},
                "coverage": {"n_contributor_links": contributors.num_rows},
            }
        )
    )
    return mhc_package, cohort, old_yaml, current_yaml, lineage


@pytest.mark.parametrize("historical_mhc", [False, True])
def test_context_package_keeps_native_sources_and_separate_claim_roles(tmp_path, historical_mhc):
    mhc, cohort, old, current, lineage = _cohort_fixture(tmp_path)
    if historical_mhc:
        # This source fixture has identical typed bytes in both MHC versions.
        # Exercise successor creation and recursive verification with the old
        # parent envelope as well as the current required-core envelope.
        path = mhc / "manifest.json"
        parent = json.loads(path.read_text())
        parent.update(schema_version=1, catalog_schema_version=2, hitlist_version="2.0.42")
        del parent["positive_ms_core_contract"]
        path.write_text(json.dumps(parent))
    target = tmp_path / "context"
    write_context_evidence_package(
        mhc,
        cohort,
        target,
        historical_curation_path=old,
        current_curation_path=current,
        lineage_file=lineage,
    )
    manifest = verify_context_evidence_package(target)
    counts = manifest["reconciliation"]
    assert counts["n_mhc_assays"] == 5
    assert counts["n_assay_context_rows"] == 2
    assert counts["n_contributor_links"] == 3
    assert counts["n_condition_link_rows"] == 1
    assert counts["n_curated_arms_per_revision"] == 2
    assert counts["n_curated_studies_per_revision"] == 1
    assert counts["n_arm_control_links"] == 2
    assert counts["n_curation_amendments"] == 1
    assert manifest["source_registry"]["corrected_human_cohort"]["license_status"] == ("unverified")
    assert file_digest(target / "cohort_native" / "training.parquet") == file_digest(
        cohort / "training.parquet"
    )
    assert file_digest(target / "curation" / "pmid_overrides_1_78.yaml") == file_digest(old)
    context = pq.read_table(target / "typed" / "assay_context.parquet").to_pylist()
    assert context[0]["arm_id"] == "pmid:12345678:arm:test_arm"
    assert context[0]["arm_link_status"] == "attributed_arm"
    assert context[1]["arm_id"] is None
    assert context[0]["donor_status"] == "unknown"
    factors = pq.read_table(target / "typed" / "arm_factors.parquet").to_pylist()
    assert {x["factor_type"] for x in factors if x["arm_id"].endswith("test_arm")} >= {
        "knockout_genes",
        "infection",
    }
    assert any(x["claim_status"] == "explicit_absence" for x in factors)
    assert {x["evidence_basis"] for x in factors} == {"curated_text"}
    mhc_claims = pq.read_table(target / "typed" / "arm_mhc_claims.parquet").to_pylist()
    assert {x["role"] for x in mhc_claims} == {
        "presenting_sample_candidates",
        "cellular_mhc_genotype",
    }
    protocols = pq.read_table(target / "typed" / "protocol_claims.parquet").to_pylist()
    assert {x["source_scope"] for x in protocols} == {"study_default", "arm_override"}
    controls = pq.read_table(target / "typed" / "arm_control_links.parquet").to_pylist()
    assert {x["comparison_arm_id"] for x in controls} == {"pmid:12345678:arm:test_arm"}
    assert not any(x["role"] == "peptide_mhc_restriction" for x in mhc_claims)


def test_context_package_survives_mutated_predecessor_and_cohort_inputs(tmp_path):
    mhc, cohort, old, current, lineage = _cohort_fixture(tmp_path)
    target = tmp_path / "context"
    write_context_evidence_package(
        mhc,
        cohort,
        target,
        historical_curation_path=old,
        current_curation_path=current,
        lineage_file=lineage,
    )
    original_mhc = (target / "native" / "binding.parquet").read_bytes()
    original_cohort = (target / "cohort_native" / "training.parquet").read_bytes()

    with (mhc / "native" / "binding.parquet").open("ab") as stream:
        stream.write(b"later predecessor edit")
    with (cohort / "training.parquet").open("ab") as stream:
        stream.write(b"later cohort edit")

    assert (target / "native" / "binding.parquet").read_bytes() == original_mhc
    assert (target / "cohort_native" / "training.parquet").read_bytes() == original_cohort
    assert (target / "native" / "binding.parquet").stat().st_ino != (
        mhc / "native" / "binding.parquet"
    ).stat().st_ino
    assert (target / "cohort_native" / "training.parquet").stat().st_ino != (
        cohort / "training.parquet"
    ).stat().st_ino
    verify_context_evidence_package(target)


@pytest.mark.parametrize(
    "variation, message",
    [
        ({"bad_arm": True}, "unregistered historical arm"),
        ({"duplicate_id": True}, "Duplicate cohort_ids observation ID"),
        ({"extra_field": True}, "Unreviewed cohort training schema"),
    ],
)
def test_context_package_rejects_unreviewed_or_unlinked_source(tmp_path, variation, message):
    mhc, cohort, old, current, lineage = _cohort_fixture(tmp_path, **variation)
    target = tmp_path / "context"
    with pytest.raises(ValueError, match=message):
        write_context_evidence_package(
            mhc,
            cohort,
            target,
            historical_curation_path=old,
            current_curation_path=current,
            lineage_file=lineage,
        )
    assert not target.exists()


def test_context_package_detects_source_tampering(tmp_path):
    mhc, cohort, old, current, lineage = _cohort_fixture(tmp_path)
    with (cohort / "training.parquet").open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="Cohort artifact mismatch"):
        write_context_evidence_package(
            mhc,
            cohort,
            tmp_path / "context",
            historical_curation_path=old,
            current_curation_path=current,
            lineage_file=lineage,
        )


def test_class_pool_link_stays_coarse_and_package_detects_tampering(tmp_path):
    mhc, cohort, old, current, lineage = _cohort_fixture(tmp_path, class_pool=True)
    target = tmp_path / "context"
    write_context_evidence_package(
        mhc,
        cohort,
        target,
        historical_curation_path=old,
        current_curation_path=current,
        lineage_file=lineage,
    )
    context = pq.read_table(target / "typed" / "assay_context.parquet").to_pylist()
    assert context[0]["arm_link_status"] == "class_pool_context"
    with (target / "typed" / "arm_factors.parquet").open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="artifact mismatch"):
        verify_context_evidence_package(target)


def test_historical_context_manifest_verifies_inside_successor_package(tmp_path):
    mhc, cohort, old, current, lineage = _cohort_fixture(tmp_path)
    target = tmp_path / "context"
    write_context_evidence_package(
        mhc,
        cohort,
        target,
        historical_curation_path=old,
        current_curation_path=current,
        lineage_file=lineage,
    )
    nested = target / "manifests" / "context.json"
    nested.write_bytes((target / "manifest.json").read_bytes())
    (target / "typed" / "later_source.txt").write_text("new source")
    assert verify_context_evidence_package(target, manifest_path="manifests/context.json") == (
        json.loads(nested.read_text())
    )
    with pytest.raises(ValueError, match="Invalid context package manifest path"):
        verify_context_evidence_package(target, manifest_path="../context/manifest.json")
