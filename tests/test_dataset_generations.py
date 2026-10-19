"""Complete publication and reader isolation, using real-format source records."""

import json
from pathlib import Path

import pandas as pd
import pytest

from hitlist import builder, downloads, mappings, provenance
from hitlist.parquet_io import atomic_write_parquet
from tests.test_build_smoke import _write_synthetic_iedb

_REAL_BUILD_MAPPINGS = mappings.build_peptide_mappings


@pytest.fixture
def tiny_dataset(tmp_path, monkeypatch, _isolated_curation_root):
    from hitlist import supplement

    (_isolated_curation_root / "pmid_overrides.yaml").write_text("[]\n")
    source = tmp_path / "iedb.csv"
    _write_synthetic_iedb(source)
    downloads.register("iedb", source)
    monkeypatch.setattr(supplement, "scan_supplementary", lambda **kw: pd.DataFrame())

    def empty_index(path):
        frame = pd.DataFrame()
        atomic_write_parquet(frame, path)
        return frame

    monkeypatch.setattr(
        builder, "build_bulk_proteomics", lambda **kw: empty_index(builder._bulk_proteomics_path())
    )
    monkeypatch.setattr(
        builder, "build_line_expression", lambda **kw: empty_index(builder._line_expression_path())
    )

    def tiny_mappings(**kwargs):
        path = mappings.mappings_path()
        frame = kwargs.get("obs_override")
        if frame is None:
            frame = pd.read_parquet(builder._observations_path())
        atomic_write_parquet(frame[["peptide"]], path)
        mappings.mappings_meta_path().write_text(
            json.dumps({"observations": mappings._obs_fingerprint()})
        )
        return path

    monkeypatch.setattr(mappings, "build_peptide_mappings", tiny_mappings)
    return source


def _published_paths():
    return [
        builder._observations_path(),
        builder._binding_path(),
        builder._other_assays_path(),
        provenance.contributors_path(),
        mappings.mappings_path(),
        mappings.mappings_meta_path(),
        builder._bulk_proteomics_path(),
        builder._line_expression_path(),
        builder._meta_path(),
    ]


@pytest.mark.parametrize(
    "boundary",
    [
        "contributors",
        "observations.parquet",
        "binding.parquet",
        "other_assays.parquet",
        "bulk",
        "expression",
        "metadata",
    ],
)
def test_interrupted_build_preserves_complete_previous_dataset(tiny_dataset, monkeypatch, boundary):
    builder.build_observations()
    old = {path: path.read_bytes() for path in _published_paths()}
    tiny_dataset.write_text(tiny_dataset.read_text().replace("33858848", "99858848"))

    def fail():
        raise OSError("injected storage interruption")

    if boundary == "contributors":
        original = provenance.ContributorCollector.write

        def write(self, *args, **kwargs):
            original(self, *args, **kwargs)
            fail()

        monkeypatch.setattr(provenance.ContributorCollector, "write", write)
    elif boundary in {"bulk", "expression"}:
        name = "build_bulk_proteomics" if boundary == "bulk" else "build_line_expression"
        original = getattr(builder, name)

        def write(*args, **kwargs):
            original(*args, **kwargs)
            fail()

        monkeypatch.setattr(builder, name, write)
    elif boundary == "metadata":
        original = Path.write_text

        def write(path, *args, **kwargs):
            result = original(path, *args, **kwargs)
            if path.name == "observations_meta.json":
                fail()
            return result

        monkeypatch.setattr(Path, "write_text", write)
    else:
        original = builder.atomic_write_parquet

        def write(frame, path, **kwargs):
            original(frame, path, **kwargs)
            if path.name == boundary:
                fail()

        monkeypatch.setattr(builder, "atomic_write_parquet", write)

    with pytest.raises(OSError, match="injected storage interruption"):
        builder.build_observations(force=True)
    assert {path: path.read_bytes() for path in old} == old
    assert _published_paths() == list(old)
    assert provenance._contributor_contract(verify_hashes=True)


@pytest.mark.parametrize("boundary", ["mappings", "validation", "readme", "publish"])
def test_late_failures_never_activate_partial_generation(tiny_dataset, monkeypatch, boundary):
    from datacache.bundle_store import BundleStore

    from hitlist import dataset

    first = builder.build_observations()
    old = {path: path.read_bytes() for path in _published_paths()}

    def fail(*args, **kwargs):
        raise OSError("injected publication failure")

    if boundary == "mappings":
        original = mappings.build_peptide_mappings

        def failed_mappings(**kwargs):
            original(**kwargs)
            fail()

        monkeypatch.setattr(mappings, "build_peptide_mappings", failed_mappings)
    elif boundary == "validation":
        monkeypatch.setattr(dataset, "_validate_generation", fail)
    elif boundary == "readme":
        monkeypatch.setattr(dataset, "_write_readme", fail)
    else:
        monkeypatch.setattr(BundleStore, "publish", fail)
    with pytest.raises(OSError, match="injected publication failure"):
        builder.build_observations(force=True)
    assert builder._observations_path() == first
    assert {path: path.read_bytes() for path in old} == old
    assert provenance._contributor_contract(verify_hashes=True)


def test_reader_pin_spans_publication_and_paths_remain_valid(tiny_dataset):
    from concurrent.futures import ThreadPoolExecutor

    from datacache import inspect_materialization

    from hitlist import dataset_snapshot
    from hitlist.observations import load_observations

    first = builder.build_observations()
    old = {path: path.read_bytes() for path in _published_paths()}
    tiny_dataset.write_text(tiny_dataset.read_text().replace("33858848", "99858848"))
    with dataset_snapshot() as pinned:
        assert pinned == first.parent
        before = load_observations(columns=["pmid", "provenance_id"])
        second = builder.build_observations(force=True)
        assert second != first
        pd.testing.assert_frame_equal(before, load_observations(columns=["pmid", "provenance_id"]))
        assert _published_paths() == list(old)
        # Independent readers choose the new generation even while this
        # reader still holds the old one; the pin isn't process-global state.
        with ThreadPoolExecutor(max_workers=1) as executor:
            assert executor.submit(builder._observations_path).result() == second
    assert builder._observations_path() == second
    assert {path: path.read_bytes() for path in old} == old
    assert 99858848 in set(load_observations(columns=["pmid"]).pmid)
    state = inspect_materialization(downloads.data_dir() / "dataset")
    assert state.status == "available"
    assert len(state.files) == 10
    metadata = builder._cache_meta()
    assert metadata["provenance"]["sources"]["iedb"]["path"] == str(tiny_dataset)
    assert builder._cache_is_valid(builder._source_paths())
    assert ".staging" not in json.dumps(metadata)


def test_cache_hit_and_optional_mappings_do_not_mix_generations(tiny_dataset, monkeypatch):
    from hitlist.dataset import dataset_snapshot

    first = builder.build_observations()
    second = builder.build_observations(force=True, build_mappings=False)
    assert not mappings.mappings_path().exists()
    assert (first.parent / "peptide_mappings.parquet").exists()

    def no_build(*args, **kwargs):
        pytest.fail("Current observations should not rebuild or allocate provenance scratch")

    monkeypatch.setattr(provenance, "ContributorCollector", no_build)
    with dataset_snapshot():
        assert builder.build_observations(build_mappings=False) == second
    assert not mappings.mappings_path().exists()


def test_legacy_migration_preserves_flat_artifacts(tiny_dataset, tmp_path, monkeypatch):
    import shutil

    from hitlist import dataset_snapshot

    built = builder.build_observations().parent
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    for path in built.iterdir():
        if not path.name.startswith("."):
            shutil.copy2(path, legacy / path.name)
    monkeypatch.setattr(downloads, "_override_data_dir", legacy)
    downloads.register("iedb", tiny_dataset)
    old = {path: path.read_bytes() for path in legacy.iterdir() if path.is_file()}
    with dataset_snapshot() as pinned:
        assert pinned == legacy
        published = builder.build_observations(force=True)
        assert builder._observations_path() == legacy / "observations.parquet"
        assert published.parent != legacy
    assert builder._observations_path() == published
    assert {path: path.read_bytes() for path in old} == old


def test_corrupt_generation_does_not_fall_back_and_can_rebuild(tiny_dataset):
    from hitlist.observations import load_observations, observations_cache_is_current

    first = builder.build_observations()
    first.write_bytes(b"corrupt")
    # A valid-looking flat file must never mask a broken published generation.
    pd.DataFrame({"peptide": ["FALLBACK"]}).to_parquet(
        downloads.data_dir() / "observations.parquet"
    )
    with pytest.raises(ValueError, match="Invalid Hitlist dataset"):
        load_observations()
    assert not observations_cache_is_current()
    second = builder.build_observations(force=True)
    assert second != first
    assert provenance._contributor_contract(verify_hashes=True)


def test_source_changed_during_build_is_rejected(tiny_dataset, monkeypatch):
    first = builder.build_observations()
    original = builder.build_line_expression

    def change_source(**kwargs):
        result = original(**kwargs)
        tiny_dataset.write_text(tiny_dataset.read_text() + "\n")
        return result

    monkeypatch.setattr(builder, "build_line_expression", change_source)
    with pytest.raises(RuntimeError, match="inputs changed during the build"):
        builder.build_observations(force=True)
    assert builder._observations_path() == first


def test_standalone_sidecar_update_preserves_prior_generation(tiny_dataset):
    from datacache import inspect_materialization

    from hitlist.line_expression import write_line_expression_index

    first = builder.build_observations()
    old = {path: path.read_bytes() for path in _published_paths()}
    bulk = pd.DataFrame({"granularity": ["protein", "peptide"], "marker": [1, 2]})
    out = builder._write_bulk_index(bulk)
    assert out.parent != first.parent
    pd.testing.assert_frame_equal(pd.read_parquet(out), bulk)
    assert builder._cache_meta()["n_bulk_rows"] == 2
    assert builder._cache_meta()["n_bulk_protein_rows"] == 1
    assert {path: path.read_bytes() for path in old} == old
    assert builder._cache_is_valid(builder._source_paths())
    # A direct current-path write also publishes instead of mutating a bundle.
    current_expression = builder._line_expression_path()
    expression = write_line_expression_index(pd.DataFrame(), current_expression)
    assert expression.parent != current_expression.parent
    assert inspect_materialization(downloads.data_dir() / "dataset").status == "available"
    with pytest.raises(ValueError, match="immutable"):
        write_line_expression_index(pd.DataFrame(), first.parent / "line_expression.parquet")


def test_quiet_paths_do_not_create_a_cache(tmp_path, monkeypatch):
    from hitlist import dataset_snapshot

    root = tmp_path / "absent"
    monkeypatch.setattr(downloads, "_override_data_dir", root)
    with dataset_snapshot() as directory:
        assert directory == root
        assert builder._observations_path() == root / "observations.parquet"
    assert not root.exists()


def test_real_mapping_only_build_publishes_a_complete_generation(tiny_dataset, monkeypatch):
    first = builder.build_observations(build_mappings=False)
    old = first.read_bytes()
    monkeypatch.setattr(mappings, "build_peptide_mappings", _REAL_BUILD_MAPPINGS)
    # An unknown reference exercises the actual writer without downloading a
    # whole proteome. Unmapped counts remain explicit in mapping metadata.
    monkeypatch.setattr(downloads, "lookup_proteome", lambda *a, **kw: None)
    monkeypatch.setattr(builder, "_collect_pmid_extra_proteomes", lambda: {})
    path = mappings.build_peptide_mappings(fetch_missing=False, verbose=False)
    assert path.parent != first.parent
    assert first.read_bytes() == old
    assert builder._observations_path().read_bytes() == old
    assert mappings.mappings_cache_is_current(fetch_missing=False)
    assert builder._cache_is_valid(builder._source_paths())
    assert builder._cache_meta()["with_mappings"] is True
    assert json.loads(mappings.mappings_meta_path().read_text())["unmapped_organisms"]
    # A current standalone mapping build is quiet and does not publish again.
    assert mappings.build_peptide_mappings(fetch_missing=False, verbose=False) == path
    old_mapping = path.read_bytes()
    # Overrides are only valid inside the full builder, even on a cache hit.
    with pytest.raises(ValueError, match="Frame overrides require"):
        mappings.build_peptide_mappings(
            obs_override=pd.read_parquet(builder._observations_path()),
            fetch_missing=False,
            verbose=False,
        )
    assert path.read_bytes() == old_mapping


def test_sidecar_update_rejects_an_intervening_generation(tiny_dataset, monkeypatch):
    from hitlist import dataset

    builder.build_observations()
    original = dataset.publish_dataset
    intervening = []

    def publish(*args, **kwargs):
        if not intervening:
            intervening.append(True)
            latest = builder._write_bulk_index(pd.DataFrame({"granularity": ["protein"]}))
            intervening.append(latest)
        return original(*args, **kwargs)

    monkeypatch.setattr(dataset, "publish_dataset", publish)
    with pytest.raises(RuntimeError, match="generation changed before update"):
        builder._write_bulk_index(pd.DataFrame())
    assert builder._bulk_proteomics_path() == intervening[1]
    assert builder._cache_meta()["n_bulk_rows"] == 1


def test_explicit_expression_output_is_independent_of_a_broken_default_cache(
    tiny_dataset, tmp_path
):
    from hitlist.line_expression import write_line_expression_index

    builder.build_observations().write_bytes(b"corrupt")
    destination = tmp_path / "independent-expression.parquet"
    assert write_line_expression_index(pd.DataFrame(), destination) == destination
    assert destination.exists()
