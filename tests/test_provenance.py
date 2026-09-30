"""Contributor retention across real scanner/build deduplication boundaries."""

import json

import pandas as pd
import pytest

from hitlist.builder import _drop_duplicate_iris, _drop_supplementary_duplicates
from hitlist.provenance import ContributorCollector, file_digest, load_contributors
from hitlist.scanner import scan
from hitlist.supplement import scan_supplementary
from tests.test_scanner import _write_tiny_iedb_csv


def _row(assay, sample="original sample"):
    row = [""] * 27
    row[0], row[1], row[2] = assay, "ref:1", "99999999"
    row[5], row[8], row[10] = "SLYNTVATL", "Homo sapiens", "Cellular MHC ligand presentation"
    row[17], row[19], row[20] = sample, "HLA-A*02:01", "I"
    row[22] = "mass spectrometry"
    return row


def test_chained_dedup_retains_sources_without_reweighting(tmp_path, monkeypatch):
    iedb, cedar = tmp_path / "iedb.csv", tmp_path / "cedar.csv"
    _write_tiny_iedb_csv(
        iedb,
        [
            _row("http://iedb.org/assay/1"),
            _row("http://iedb.org/assay/1", "copy metadata"),
            _row("http://iedb.org/assay/2", "independent sample"),
        ],
    )
    _write_tiny_iedb_csv(cedar, [_row("https://cedar.iedb.org/assay/1")])
    supp = tmp_path / "sample.csv"
    supp.write_text("peptide,mhc_restriction\nSLYNTVATL,HLA-A*02:01\nSLYNTVATL,HLA-A*02:01\n")
    monkeypatch.setattr("hitlist.supplement._SUPP_DIR", tmp_path)
    monkeypatch.setattr(
        "hitlist.supplement.load_supplementary_manifest",
        lambda: [{"file": "sample.csv", "pmid": 99999999, "source": "Table S1, extracted rows"}],
    )
    with ContributorCollector() as collector:
        frames = [
            scan(iedb_path=iedb, mhc_species=None, provenance=collector),
            scan(cedar_path=cedar, mhc_species=None, provenance=collector),
        ]
        combined = pd.concat(frames, ignore_index=True)
        obs = _drop_duplicate_iris(combined, "MS", provenance=collector)
        supplemental = scan_supplementary(provenance=collector)
        assert len(supplemental) == 1
        retained = _drop_supplementary_duplicates(supplemental, obs, provenance=collector)
        assert retained.empty
        assert len(obs) == 2
        baseline = pd.concat(
            [scan(iedb_path=iedb, mhc_species=None), scan(cedar_path=cedar, mhc_species=None)],
            ignore_index=True,
        )
        pd.testing.assert_frame_equal(
            obs.drop(columns="provenance_id"), _drop_duplicate_iris(baseline, "MS")
        )
        path = tmp_path / "contributors.parquet"
        metadata = collector.write([obs], path)
    links = pd.read_parquet(path)
    assert links.source_record_id.nunique() == 6
    counts = links.groupby("provenance_id").size().to_dict()
    assert sorted(counts.values()) == [3, 5]
    assert set(links[links.source_dataset == "supplement:sample.csv"].relationship_status) == {
        "overlap_unresolved"
    }
    original = links[(links.source_dataset == "iedb") & (links.source_row == 2)].iloc[0]
    assert json.loads(original.original_fields)["cell_name"] == "copy metadata"
    assert "assay_copy" in json.loads(original.relationships)
    assert metadata["sources"]["iedb"]["sha256"] == file_digest(iedb)["sha256"]
    assert metadata["n_contributor_links"] == 8


def test_donor_expansion_keeps_separate_nodes(tmp_path, monkeypatch):
    path = tmp_path / "iedb.csv"
    _write_tiny_iedb_csv(path, [_row("http://iedb.org/assay/1")])
    monkeypatch.setattr("hitlist.curation.peptide_attribution_applies_to_row", lambda *a: True)
    monkeypatch.setattr(
        "hitlist.curation.attribute_peptide_to_per_sample_typings",
        lambda *a: (
            ("donor A", frozenset({"HLA-A*02:01"})),
            ("donor B", frozenset({"HLA-A*02:01"})),
        ),
    )
    with ContributorCollector() as collector:
        obs = scan(iedb_path=path, mhc_species=None, provenance=collector)
        assert len(obs) == 2
        collector.write([obs], tmp_path / "contributors.parquet")
    links = pd.read_parquet(tmp_path / "contributors.parquet")
    assert links.provenance_id.nunique() == 2
    assert links.source_record_id.nunique() == 1
    assert set(links.attributed_sample_label) == {"donor A", "donor B"}


def test_load_legacy_and_inconsistent_provenance(tmp_path, monkeypatch):
    monkeypatch.setattr("hitlist.downloads._override_data_dir", tmp_path)
    assert load_contributors().empty
    with pytest.raises(ValueError, match="metadata missing"):
        load_contributors(["observation:record:iedb:row:1"])


def test_changed_input_aborts_sidecar_publication(tmp_path):
    source = tmp_path / "source.csv"
    source.write_text("original")
    path = tmp_path / "contributors.parquet"
    with ContributorCollector() as collector:
        collector.register_source("iedb", source)
        source.write_text("replaced")
        with pytest.raises(ValueError, match="Source changed"):
            collector.write([], path)
    assert not path.exists()
    assert not path.with_suffix(".parquet.partial").exists()


def test_blank_identifiers_are_not_duplicate_evidence(tmp_path):
    source = tmp_path / "iedb.csv"
    rows = [_row("", "sample A"), _row("", "sample B"), _row("", "sample C")]
    for row in rows:
        row[1] = ""
    rows[-1][5] = "SIINFEKL"
    _write_tiny_iedb_csv(source, rows)
    with ContributorCollector() as collector:
        observations = scan(iedb_path=source, mhc_species=None, provenance=collector)
        retained = _drop_duplicate_iris(observations, "MS", provenance=collector)
        assert len(retained) == 3
        collector.write([retained], tmp_path / "contributors.parquet")
    links = pd.read_parquet(tmp_path / "contributors.parquet")
    assert len(links) == 3
    assert links.provenance_id.nunique() == 3
    assert set(links.relationships) == {'["retained"]'}
