"""A spawned build must use the caller's configured cached proteomes (#543)."""

import concurrent.futures
import json
import multiprocessing

import pandas as pd
import pytest

from hitlist import downloads, mappings, proteome


@pytest.mark.parametrize("fallback_has_proteomes", [False, True])
def test_spawned_build_preserves_configured_cache(tmp_path, monkeypatch, fallback_has_proteomes):
    configured = tmp_path / "configured"
    fallback = tmp_path / "fallback"
    configured.mkdir()
    fallback.mkdir()
    index_cache = tmp_path / "indexes"
    monkeypatch.setenv("HITLIST_DATA_DIR", str(fallback))
    monkeypatch.setattr(downloads, "_override_data_dir", configured)
    monkeypatch.setattr(proteome, "_PROTEOME_INDEX_DISK_CACHE_DIR", index_cache)
    monkeypatch.setenv("HITLIST_PROTEOME_INDEX_CACHE_GB", "1")

    labels = [f"Synthetic {i}" for i in range(4)]
    entries = {
        label: {"kind": "uniprot", "proteome_id": f"UP_CACHE_TEST_{i}"}
        for i, label in enumerate(labels)
    }
    for directory, protein_prefix in ((configured, "CORRECT"), (fallback, "WRONG")):
        if directory == fallback and not fallback_has_proteomes:
            continue
        manifest = {"proteomes": {}}
        for i, label in enumerate(labels):
            fasta = directory / f"{i}.fasta"
            fasta.write_text(f">sp|{protein_prefix}{i}|TEST GN=TEST\nMPEPTIDEKACDEFGHIK\n")
            manifest["proteomes"][label] = {**entries[label], "path": str(fasta)}
        (directory / "manifest.json").write_text(json.dumps(manifest))

    monkeypatch.setattr(downloads, "lookup_proteome", lambda organism, **_: entries[organism])
    monkeypatch.setattr("hitlist.builder._collect_pmid_extra_proteomes", lambda: {})
    observations = pd.DataFrame(
        {"peptide": "PEPTIDEK", "source_organism": labels, "mhc_species": "", "pmid": 0}
    )
    real_executor = concurrent.futures.ProcessPoolExecutor

    def spawned_executor(**kwargs):
        return real_executor(mp_context=multiprocessing.get_context("spawn"), **kwargs)

    monkeypatch.setattr(concurrent.futures, "ProcessPoolExecutor", spawned_executor)
    frames = []
    metadata = []
    for n_workers in (1, 2):
        monkeypatch.setenv("HITLIST_BUILD_WORKERS", str(n_workers))
        proteome.clear_disk_cache()
        path = mappings.build_peptide_mappings(
            obs_override=observations, fetch_missing=False, force=True, verbose=False
        )
        frames.append(pd.read_parquet(path))
        metadata.append(json.loads(mappings.mappings_meta_path().read_text()))
        assert len(list(index_cache.glob("*.pkl"))) == 4

    assert list(frames[0]["protein_id"]) == [f"sp|CORRECT{i}|TEST" for i in range(4)]
    pd.testing.assert_frame_equal(frames[1], frames[0])
    assert metadata[1]["per_proteome"] == metadata[0]["per_proteome"]
    assert metadata[1]["unavailable_proteomes"] == []
    assert metadata[1]["n_rows"] == metadata[0]["n_rows"] == 4
    assert downloads.data_dir() == configured
    assert not (fallback / "peptide_mappings.parquet").exists()
