"""Tests for hitlist.mappings — peptide_mappings.parquet contract.

Issue #141 added ``transcript_id`` and ``is_canonical_transcript`` as
first-class columns and exposed matching filters on
``load_peptide_mappings``.  These tests pin down the schema (uniform
across FASTA + Ensembl backends) and the filter pushdown.
"""

from __future__ import annotations

import pandas as pd
import pytest

from hitlist.mappings import (
    _MAPPING_COLUMNS,
    _flanking_rows_to_mapping_rows,
    _proteome_group_key,
    load_peptide_mappings,
)


def test_mapping_columns_contract_includes_transcript_fields():
    """The canonical mapping schema MUST include the new transcript fields."""
    cols = set(_MAPPING_COLUMNS)
    assert "transcript_id" in cols
    assert "is_canonical_transcript" in cols
    # Existing fields stay.
    for legacy in ("peptide", "protein_id", "gene_name", "gene_id", "position"):
        assert legacy in cols


def test_flanking_rows_to_mapping_rows_carries_transcript_columns():
    """When a transcript-aware ProteomeIndex feeds map_peptides → output,
    ``_flanking_rows_to_mapping_rows`` must propagate transcript_id and
    is_canonical_transcript onto the long-form mapping frame.
    """
    flanking = pd.DataFrame(
        {
            "peptide": ["ABCDEFGHI"],
            "protein_id": ["ENSP00000001"],
            "gene_name": ["TP53"],
            "gene_id": ["ENSG00000141510"],
            "transcript_id": ["ENST00000269305"],
            "is_canonical_transcript": [True],
            "position": [42],
            "n_flank": ["NNNNN"],
            "c_flank": ["CCCCC"],
        }
    )
    out = _flanking_rows_to_mapping_rows(
        flanking, proteome_label="Homo sapiens", proteome_source="ensembl"
    )
    assert list(out.columns) == list(_MAPPING_COLUMNS)
    row = out.iloc[0]
    assert row["transcript_id"] == "ENST00000269305"
    assert row["is_canonical_transcript"] is True or row["is_canonical_transcript"] == True  # noqa: E712


def test_flanking_rows_to_mapping_rows_legacy_input_safe_defaults():
    """Older fixtures / FASTA-only proteome indexes don't supply the new
    columns; the converter must fill safe defaults so the parquet schema
    stays uniform regardless of backend.
    """
    flanking = pd.DataFrame(
        {
            "peptide": ["ZZZZZ"],
            "protein_id": ["sp|P|A"],
            "gene_name": ["GENE"],
            "gene_id": [""],
            "position": [0],
            "n_flank": [""],
            "c_flank": [""],
        }
    )
    out = _flanking_rows_to_mapping_rows(flanking, proteome_label="custom", proteome_source="fasta")
    assert "transcript_id" in out.columns
    assert "is_canonical_transcript" in out.columns
    assert out.iloc[0]["transcript_id"] == ""
    assert bool(out.iloc[0]["is_canonical_transcript"]) is False


def test_flanking_rows_to_mapping_rows_empty_input_emits_full_schema():
    out = _flanking_rows_to_mapping_rows(pd.DataFrame(), proteome_label="x", proteome_source="x")
    assert list(out.columns) == list(_MAPPING_COLUMNS)
    assert len(out) == 0


def test_load_peptide_mappings_transcript_id_filter(tmp_path, monkeypatch):
    """``load_peptide_mappings(transcript_id=...)`` must push down the
    filter to pyarrow and return only the matching ENST rows.
    """
    rows = pd.DataFrame(
        {
            "peptide": ["AAA", "AAA", "BBB"],
            "protein_id": ["ENSP1", "ENSP2", "ENSP3"],
            "gene_name": ["TP53", "TP53", "MYC"],
            "gene_id": ["ENSG_TP53", "ENSG_TP53", "ENSG_MYC"],
            "transcript_id": ["ENST_T1", "ENST_T2", "ENST_T3"],
            "is_canonical_transcript": [True, False, True],
            "position": [1, 2, 3],
            "n_flank": ["", "", ""],
            "c_flank": ["", "", ""],
            "proteome": ["Homo sapiens"] * 3,
            "proteome_source": ["ensembl"] * 3,
        }
    )
    p = tmp_path / "peptide_mappings.parquet"
    rows.to_parquet(p, index=False)
    monkeypatch.setattr("hitlist.mappings.mappings_path", lambda: p)

    sub = load_peptide_mappings(transcript_id="ENST_T2")
    assert list(sub["transcript_id"]) == ["ENST_T2"]
    assert list(sub["protein_id"]) == ["ENSP2"]


def test_load_peptide_mappings_is_canonical_filter(tmp_path, monkeypatch):
    """``is_canonical_transcript=True`` returns only the canonical rows."""
    rows = pd.DataFrame(
        {
            "peptide": ["AAA", "AAA", "BBB"],
            "protein_id": ["ENSP1", "ENSP2", "ENSP3"],
            "gene_name": ["TP53", "TP53", "MYC"],
            "gene_id": ["ENSG_TP53", "ENSG_TP53", "ENSG_MYC"],
            "transcript_id": ["ENST_T1", "ENST_T2", "ENST_T3"],
            "is_canonical_transcript": [True, False, True],
            "position": [1, 2, 3],
            "n_flank": ["", "", ""],
            "c_flank": ["", "", ""],
            "proteome": ["Homo sapiens"] * 3,
            "proteome_source": ["ensembl"] * 3,
        }
    )
    p = tmp_path / "peptide_mappings.parquet"
    rows.to_parquet(p, index=False)
    monkeypatch.setattr("hitlist.mappings.mappings_path", lambda: p)

    canon = load_peptide_mappings(is_canonical_transcript=True)
    assert set(canon["transcript_id"]) == {"ENST_T1", "ENST_T3"}
    non_canon = load_peptide_mappings(is_canonical_transcript=False)
    assert list(non_canon["transcript_id"]) == ["ENST_T2"]


def test_load_peptide_mappings_missing_file_raises(tmp_path, monkeypatch):
    monkeypatch.setattr("hitlist.mappings.mappings_path", lambda: tmp_path / "nonexistent.parquet")
    with pytest.raises(FileNotFoundError, match="not built"):
        load_peptide_mappings(peptide="AAA")


# ── #107 / v1.30.6: build-order clusters same-FASTA canonicals ─────────


def test_proteome_group_key_uniprot_same_proteome_id_clusters():
    """v1.30.6 / #107: two canonicals with the same UniProt proteome_id
    share a group key, so sorting by it lands them adjacently and the
    second one's ``from_fasta`` call hits the LRU cache. Strain-variant
    canonicals (multiple LCMV / SARS-CoV-2 / EBV strains) all share one
    underlying FASTA via ``proteome_id``."""
    e1 = {
        "kind": "uniprot",
        "proteome_id": "UP000111111",
        "canonical_species": "Strain A",
    }
    e2 = {
        "kind": "uniprot",
        "proteome_id": "UP000111111",
        "canonical_species": "Strain Z",
    }
    e3 = {
        "kind": "uniprot",
        "proteome_id": "UP000999999",
        "canonical_species": "Other species",
    }
    assert _proteome_group_key(e1) == _proteome_group_key(e2)
    assert _proteome_group_key(e1) != _proteome_group_key(e3)


def test_proteome_group_key_orders_ensembl_before_uniprot_before_other():
    """Ensembl bucket sorts before uniprot, uniprot before unrecognised,
    so the build log clusters the human pass first, then FASTA-backed
    species, then any tail of unmatchable canonicals."""
    e_ensembl = {"kind": "ensembl", "species": "human"}
    e_uniprot = {"kind": "uniprot", "proteome_id": "UP000005640"}
    e_other = {"kind": "", "canonical_species": "Mystery sp."}
    assert _proteome_group_key(e_ensembl) < _proteome_group_key(e_uniprot)
    assert _proteome_group_key(e_uniprot) < _proteome_group_key(e_other)


def test_proteome_group_key_sorts_canonicals_by_fasta_adjacency():
    """End-to-end: a list of mixed canonicals sorted by
    ``(group_key, canonical)`` puts same-FASTA canonicals next to each
    other instead of scattering them by alphabetic name."""
    canonical_to_entry = {
        "Apple virus alpha": {
            "kind": "uniprot",
            "proteome_id": "UP000ZZZ001",
            "canonical_species": "Apple virus alpha",
        },
        # Same FASTA as "Apple virus alpha" — alphabetically far apart but
        # should cluster after sorting.
        "Zebra virus omega": {
            "kind": "uniprot",
            "proteome_id": "UP000ZZZ001",
            "canonical_species": "Zebra virus omega",
        },
        "Banana virus beta": {
            "kind": "uniprot",
            "proteome_id": "UP000ZZZ002",
            "canonical_species": "Banana virus beta",
        },
    }
    canonicals = list(canonical_to_entry.keys())
    ordered = sorted(
        canonicals,
        key=lambda c: (_proteome_group_key(canonical_to_entry[c]), c),
    )
    # The two ZZZ001 entries are now adjacent; the ZZZ002 is separate.
    apple_idx = ordered.index("Apple virus alpha")
    zebra_idx = ordered.index("Zebra virus omega")
    banana_idx = ordered.index("Banana virus beta")
    assert abs(apple_idx - zebra_idx) == 1, ordered
    # Banana (different proteome_id) is not sandwiched between them.
    assert not (apple_idx < banana_idx < zebra_idx), ordered
