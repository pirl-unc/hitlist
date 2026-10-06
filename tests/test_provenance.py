"""Contributor retention across real scanner/build deduplication boundaries."""

import errno
import json
import random
import shutil
import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from hitlist.builder import _drop_duplicate_iris, _drop_supplementary_duplicates
from hitlist.provenance import (
    CONTRIBUTOR_COLUMNS,
    RELATIONS,
    ContributorCollector,
    ProvenanceStorageError,
    file_digest,
    load_contributors,
)
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


def test_scratch_cap_is_enforced_and_preserves_published_output(tmp_path):
    output = tmp_path / "contributors.parquet"
    output.write_bytes(b"previous published artifact")
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    rng = random.Random(643)
    budget = 128 * 1024
    with (
        pytest.raises(ProvenanceStorageError, match="HITLIST_PROVENANCE_MAX_GB"),
        ContributorCollector(scratch_dir=scratch, max_scratch_bytes=budget) as collector,
    ):
        for row_number in range(1000):
            # Incompressible input must still respect the storage cap.
            payload = rng.randbytes(2048).hex()
            collector.record("source", row_number, {"payload": payload}, [payload])
            assert sum(p.stat().st_size for p in scratch.rglob("*") if p.is_file()) <= budget
        pytest.fail("The collector exceeded its configured budget without stopping")
    assert output.read_bytes() == b"previous published artifact"
    assert not list(scratch.iterdir())


def test_large_source_payloads_are_lossless_without_uncompressed_scratch(tmp_path):
    fields = {"pmid": "123", "note": 'é\nquoted "source" ' * 10000}
    values = [fields["note"], "", "extra unmapped source column"]
    output = tmp_path / "contributors.parquet"
    with ContributorCollector(max_scratch_bytes=256 * 1024) as collector:
        roots = [
            collector.observe(collector.record("source", i, fields, values)) for i in range(20)
        ]
        collector.write([pd.DataFrame({"provenance_id": roots})], output)
    links = pd.read_parquet(output)
    assert len(links) == 20
    assert set(links.original_fields) == {json.dumps(fields, sort_keys=True)}
    assert set(links.source_row_values) == {json.dumps(values)}


@pytest.mark.parametrize("batch_size", [1, 7, 256])
def test_batched_ancestry_matches_independent_graph_oracle(tmp_path, batch_size):
    """Paths with different flags/labels survive; identical paths deduplicate."""
    rng = random.Random(643)
    output = tmp_path / "contributors.parquet"
    with ContributorCollector() as collector:
        collector._ROOT_BATCH_SIZE = batch_size
        records = [collector.record("x", i, {"pmid": str(i)}, [str(i)]) for i in range(12)]
        roots = []
        edges = []
        for record in records[:8]:
            for label in ("", "donor é"):
                root = collector.observe(record, label)
                roots.append(root)
                edges.append((record, root, 0, label))
        nodes = records + roots
        for _ in range(50):
            source, target = rng.sample(nodes, 2)
            relation = rng.choice(list(RELATIONS))
            collector.redirect(source, target, relation)
            edges.append((source, target, RELATIONS[relation], ""))
        # Duplicate edge and a deliberately retained root with a cycle.
        collector.redirect(records[0], records[1], "assay_copy")
        collector.redirect(records[0], records[1], "assay_copy")
        collector.redirect(records[1], records[0], "database_copy")
        edges.extend([(records[0], records[1], 1, "")] * 2 + [(records[1], records[0], 2, "")])
        retained = [*roots[::2], roots[1]]
        collector.write([pd.DataFrame({"provenance_id": retained})], output)

    expected = set()
    for root in retained:
        pending = [(root, 0, "")]
        visited = set()
        while pending:
            state = pending.pop()
            if state in visited:
                continue
            visited.add(state)
            node, flags, label = state
            if node in records:
                expected.add((root, node, label, flags))
            pending.extend(
                (source, flags | relation, edge_label or label)
                for source, target, relation, edge_label in edges
                if target == node
            )
    links = pd.read_parquet(output)
    actual = [
        (
            row.provenance_id,
            row.source_record_id,
            row.attributed_sample_label,
            sum(RELATIONS.get(value, 0) for value in json.loads(row.relationships)),
        )
        for row in links.itertuples()
    ]
    assert actual == sorted(expected)
    assert list(links) == CONTRIBUTOR_COLUMNS
    for row in links.itertuples():
        assert row.original_fields == json.dumps({"pmid": str(row.source_row)})
        assert row.source_row_values == json.dumps([str(row.source_row)])


def test_traversal_has_no_hidden_sqlite_work_files(tmp_path):
    """The scratch cap must cover graph work, including a large fan-in root."""
    with ContributorCollector() as collector:
        collector._ROOT_BATCH_SIZE = 2
        roots = []
        for i in range(30):
            record = collector.record("x", i, {}, [])
            roots.append(collector.observe(record))
            if i:
                collector.redirect(record, roots[0], "reference_overlap")
        statements = set()
        collector.db.set_trace_callback(statements.add)
        collector.write([pd.DataFrame({"provenance_id": roots})], tmp_path / "out.parquet")
        collector.db.set_trace_callback(None)
        for sql in statements:
            if sql.lstrip().split()[0] not in {"SELECT", "INSERT", "DELETE"}:
                continue
            bytecode = collector.db.execute("EXPLAIN " + sql).fetchall()
            assert not {row[1] for row in bytecode} & {
                "OpenEphemeral",
                "SorterOpen",
                "OpenAutoindex",
            }, sql
        scratch_files = list(Path(collector._temporary.name).iterdir())
        assert [path.name for path in scratch_files] == ["records.sqlite"]


def test_flattening_limit_cleans_partial_and_keeps_old_artifact(tmp_path):
    output = tmp_path / "contributors.parquet"
    output.write_bytes(b"published")
    with (
        pytest.raises(ProvenanceStorageError),
        ContributorCollector(scratch_dir=tmp_path) as collector,
    ):
        roots = []
        for i in range(100):
            record = collector.record("x", i, {}, [])
            roots.append(collector.observe(record))
            if i:
                collector.redirect(record, roots[0], "reference_overlap")
        collector.db.commit()
        # Simulate reaching a nearly exhausted budget after source capture.
        pages = collector.db.execute("PRAGMA page_count").fetchone()[0]
        collector.db.execute(f"PRAGMA max_page_count={pages + 1}")
        collector.write([pd.DataFrame({"provenance_id": roots})], output)
    assert output.read_bytes() == b"published"
    assert not output.with_suffix(".parquet.partial").exists()
    assert not list(tmp_path.glob("hitlist-contributors-*"))


def test_space_preflight_and_configuration(tmp_path, monkeypatch):
    scratch = tmp_path / "configured"
    monkeypatch.setenv("HITLIST_PROVENANCE_SCRATCH_DIR", str(scratch))
    monkeypatch.setenv("HITLIST_PROVENANCE_MAX_GB", "0.25")
    monkeypatch.setenv("HITLIST_PROVENANCE_MIN_FREE_GB", "0.125")
    with ContributorCollector() as collector:
        assert Path(collector._temporary.name).parent == scratch
        assert collector.max_scratch_bytes == 256 * 1024**2
        assert collector.min_free_bytes == 128 * 1024**2
    monkeypatch.setattr(
        "hitlist.provenance.shutil.disk_usage", lambda _: shutil._ntuple_diskusage(100, 99, 1)
    )
    with pytest.raises(ProvenanceStorageError, match=r"only.*available"), ContributorCollector():
        pytest.fail("No scratch work may start below the free-space reserve")
    assert not list(scratch.iterdir())


def test_output_filesystem_space_is_checked(tmp_path, monkeypatch):
    scratch = tmp_path / "scratch"
    output = tmp_path / "output" / "contributors.parquet"
    output.parent.mkdir()
    output.write_bytes(b"published")
    original_usage = shutil.disk_usage
    with (
        pytest.raises(ProvenanceStorageError, match="output"),
        ContributorCollector(scratch_dir=scratch) as collector,
    ):
        monkeypatch.setattr(
            "hitlist.provenance.shutil.disk_usage",
            lambda path: (
                shutil._ntuple_diskusage(100, 100, 0)
                if Path(path) == output.parent
                else original_usage(path)
            ),
        )
        collector.write([], output)
    assert output.read_bytes() == b"published"
    assert not list(scratch.iterdir())


@pytest.mark.parametrize(
    "error", [OSError(errno.ENOSPC, "full"), sqlite3.OperationalError("database or disk is full")]
)
def test_capacity_errors_during_export_are_actionable(tmp_path, monkeypatch, error):
    output = tmp_path / "out.parquet"
    output.write_bytes(b"published")

    def fail(self):
        raise error

    monkeypatch.setattr(ContributorCollector, "_iter_links", fail)
    with (
        pytest.raises(ProvenanceStorageError, match="HITLIST_PROVENANCE_SCRATCH_DIR"),
        ContributorCollector() as collector,
    ):
        collector.write([], output)
    assert output.read_bytes() == b"published"
    assert not output.with_suffix(".parquet.partial").exists()


def test_empty_contributors_preserve_schema(tmp_path):
    output = tmp_path / "out.parquet"
    with ContributorCollector() as collector:
        metadata = collector.write([], output)
    assert metadata["n_contributor_links"] == 0
    assert list(pd.read_parquet(output)) == CONTRIBUTOR_COLUMNS


def test_failure_after_publication_is_not_misreported_as_preserving_old_output(tmp_path):
    output = tmp_path / "out.parquet"
    later_failure = OSError(errno.ENOSPC, "later observation write failed")
    with pytest.raises(OSError) as caught, ContributorCollector() as collector:
        collector.write([], output)
        raise later_failure
    assert caught.value is later_failure
    assert output.exists()


def test_unrelated_build_failure_is_not_misreported_as_provenance_capacity():
    mapping_failure = OSError(errno.ENOSPC, "mapping sidecar volume is full")
    with pytest.raises(OSError) as caught, ContributorCollector():
        raise mapping_failure
    assert caught.value is mapping_failure


def test_setup_failure_removes_owned_scratch(tmp_path):
    with (
        pytest.raises(ProvenanceStorageError),
        ContributorCollector(scratch_dir=tmp_path, max_scratch_bytes=4096),
    ):
        pytest.fail("A single page cannot hold the scratch schema")
    assert not list(tmp_path.iterdir())


def test_long_sql_notices_free_space_loss(tmp_path, monkeypatch):
    output = tmp_path / "out.parquet"
    output.write_bytes(b"published")
    with (
        pytest.raises(ProvenanceStorageError, match="available") as caught,
        ContributorCollector(scratch_dir=tmp_path) as collector,
    ):
        root = collector.observe(collector.record("x", 0, {}, []))
        for i in range(1, 1000):
            collector.redirect(collector.record("x", i, {}, []), root, "reference_overlap")
        collector.db.commit()

        # Free-space loss during SQL must interrupt that statement, not wait
        # until another output batch returns to Python.
        def progress():
            monkeypatch.setattr(
                "hitlist.provenance.shutil.disk_usage",
                lambda _: shutil._ntuple_diskusage(100, 100, 0),
            )
            collector._next_space_check = 0
            return collector._progress()

        collector.db.set_progress_handler(progress, 100)
        collector.write([pd.DataFrame({"provenance_id": [root]})], output)
    assert isinstance(caught.value.__cause__, sqlite3.OperationalError)
    assert output.read_bytes() == b"published"
    assert not output.with_suffix(".parquet.partial").exists()
    assert not list(tmp_path.glob("hitlist-contributors-*"))
