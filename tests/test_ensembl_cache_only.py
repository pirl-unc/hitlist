"""Real PyEnsembl lazy APIs must not fetch absent optional references."""

import pandas as pd
import pyensembl
import pytest

from hitlist.builder import _fill_gene_names_via_ensembl
from hitlist.mappings import _build_species_index
from hitlist.proteome import ProteomeIndex, clear_fasta_index_cache


@pytest.fixture
def download_attempts(monkeypatch):
    from pyensembl.download_cache import DownloadCache

    attempts = []

    def record(*args, **kwargs):
        attempts.append((args, kwargs))
        raise RuntimeError("unexpected reference download")

    monkeypatch.setattr(DownloadCache, "_fetch", record)
    return attempts


@pytest.mark.parametrize("partial", [False, True])
def test_missing_ensembl_is_skipped_without_download(
    tmp_path, monkeypatch, download_attempts, partial
):
    data = pyensembl.EnsemblRelease(112, cache_directory_path=str(tmp_path))
    if partial:
        from pathlib import Path

        first = Path(data.required_local_files()[0])
        first.parent.mkdir(parents=True, exist_ok=True)
        first.write_text("partial reference\n")
    assert not data.required_local_files_exist()
    monkeypatch.setattr(pyensembl, "EnsemblRelease", lambda *args, **kwargs: data)

    with pytest.raises(FileNotFoundError, match="not installed locally"):
        ProteomeIndex.from_ensembl(fetch_missing=False, verbose=False)
    assert (
        _build_species_index(
            "Homo sapiens",
            release=112,
            use_uniprot=False,
            entry={"kind": "ensembl", "species": "human"},
            fetch_missing=False,
            verbose=False,
        )
        is None
    )
    original = pd.DataFrame({"gene_id": ["ENSG1"], "gene_name": [""]})
    pd.testing.assert_frame_equal(
        _fill_gene_names_via_ensembl(original.copy(), releases=(112,)), original
    )
    assert download_attempts == []

    # The recorder catches the original failure path, even when callers catch
    # its exception. Explicit direct callers retain their default fetch policy.
    with pytest.raises(RuntimeError, match="unexpected reference download"):
        ProteomeIndex.from_ensembl(verbose=False)
    assert download_attempts


def test_local_ensembl_sources_support_mapping_and_symbol_fill(
    tmp_path, monkeypatch, download_attempts
):
    gtf = tmp_path / "test.gtf"
    attrs = (
        'gene_id "ENSG1"; gene_name "TEST"; gene_biotype "protein_coding"; '
        'transcript_id "ENST1"; transcript_biotype "protein_coding"; protein_id "ENSP1";'
    )
    gtf.write_text(
        "".join(
            f"1\ttest\t{feature}\t1\t18\t.\t+\t.\t{attrs}\n"
            for feature in ("gene", "transcript", "exon", "CDS")
        )
    )
    fasta = tmp_path / "test.fa"
    fasta.write_text(">ENSP1\nACDACD\n")
    data = pyensembl.Genome(
        reference_name="test",
        annotation_name="test",
        annotation_version=1,
        gtf_path_or_url=str(gtf),
        protein_fasta_paths_or_urls=[str(fasta)],
        cache_directory_path=str(tmp_path / "cache"),
    )
    data.index()
    assert data.required_local_files_exist()
    monkeypatch.setattr(pyensembl, "EnsemblRelease", lambda *args, **kwargs: data)
    monkeypatch.setattr("hitlist.proteome._resolve_disk_cache_max_gb", lambda: 0)
    clear_fasta_index_cache()
    try:
        index = _build_species_index(
            "Homo sapiens",
            release=112,
            use_uniprot=False,
            entry={"kind": "ensembl", "species": "human"},
            fetch_missing=False,
            verbose=False,
            lengths=(3,),
        )
        assert index is not None
        assert index.lookup("ACD") == [("ENSP1", 0), ("ENSP1", 3)]
        result = _fill_gene_names_via_ensembl(
            pd.DataFrame({"gene_id": ["ENSG1"], "gene_name": [""]}), releases=(112,)
        )
        assert result["gene_name"].tolist() == ["TEST"]
        assert download_attempts == []
    finally:
        clear_fasta_index_cache()
