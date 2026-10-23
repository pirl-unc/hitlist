"""Original result and membership contracts keep experimental phases distinct."""

import hashlib
import zipfile

import pytest

from hitlist import gomez_source_lineage as gomez
from hitlist.evidence_bindings import materialized_bindings
from hitlist.evidence_package import package_family
from hitlist.provenance import file_digest


def test_source_identity_and_membership_do_not_collapse_conditions():
    row = {
        "Cell.Line": "JY",
        "Condition": "JY_WT",
        "BioReplicate": "JY_WT_1",
        "Sample": "JY_WT_1_1",
        "Source_File_ID": "run1.d",
        "Sequence": "AAAAAAAAA",
    }
    source = {"id": "s5_spike_experiment"}
    key = gomez._membership_key(source, row)
    assert gomez._membership_id(key) == gomez._membership_id(key)
    assert gomez._membership_id(key) != gomez._membership_id(
        gomez._membership_key(source, {**row, "Condition": "JY_S"})
    )
    assert gomez._membership_id(key) != gomez._membership_id(
        gomez._membership_key({"id": "s3_method_comparison"}, row)
    )
    index = {("gomez_zepeda_2024_jy.csv", "AAAAAAAAA"): "observation:1"}
    assert gomez._derived_link(row, index) == (
        "observation:1",
        "linked_exact_sequence_and_cell_source",
    )
    assert gomez._derived_link({**row, "Sequence": "A"}, index) == (
        None,
        "outside_derived_length_8_15",
    )
    with pytest.raises(ValueError, match="missing derived"):
        gomez._derived_link({**row, "Sequence": "BBBBBBBBB"}, index)


def test_original_zip_and_member_both_need_pinned_bytes(tmp_path):
    path = tmp_path / "source.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("original.csv", "Sequence,Area\nAAAAAAAAA,0\n")
    source = {
        "archive": "source.zip",
        "member": "original.csv",
        "archive_sha256": file_digest(path)["sha256"],
        "member_sha256": hashlib.sha256(b"Sequence,Area\nAAAAAAAAA,0\n").hexdigest(),
    }
    with gomez._source_reader(path, source) as archive:
        assert archive.namelist() == ["original.csv"]
    with pytest.raises(ValueError, match="CSV member"):
        gomez._source_reader(path, {**source, "member_sha256": "bad"})
    with pytest.raises(ValueError, match="archive"):
        gomez._source_reader(path, {**source, "archive_sha256": "bad"})


def test_successor_declares_both_new_grains_without_hiding_predecessor():
    old = materialized_bindings(10, appended_source_ids=["pxd009092", "pxd004886"])
    new = materialized_bindings(11, appended_source_ids=["pxd009092", "pxd004886"])
    assert set(new) == set(old) | {"gomez_source_result", "gomez_peptide_membership"}
    envelope = {
        "schema_version": 11,
        "parent_kind": "binding_review",
        "revision": gomez.REVISION,
        "adjudication_manifest_path": "manifests/adjudication.json",
    }
    assert package_family(envelope) == "gomez_source"
