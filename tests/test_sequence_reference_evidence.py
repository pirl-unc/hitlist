"""Reference identities and literal occurrences are not measured origins."""

import gzip
import hashlib

import pyarrow.parquet as pq
import pytest

from hitlist.sequence_reference_evidence import (
    read_reference,
    reference_relations,
    verify_reference_relations,
    write_reference_relations,
)


def _dat(tmp_path):
    path = tmp_path / "human.dat.gz"
    records = [
        ("P1", ["OLD"], "A" * 7 + "BCDEFGH"),
        ("P2", ["OLD"], "BCDEFGZZ"),
        ("P1-2", [], "A" * 7 + "BCDEFGH"),
    ]
    with gzip.open(path, "wt") as stream:
        for accession, aliases, sequence in records:
            stream.write(f"ID   {accession}_HUMAN Reviewed; {len(sequence)} AA.\n")
            stream.write("AC   " + "; ".join([accession, *aliases]) + ";\n")
            stream.write("DT   01-JAN-2023, sequence version 1.\nOX   NCBI_TaxID=9606;\n")
            stream.write(f"SQ   SEQUENCE   {len(sequence)} AA;\n     {sequence}\n//\n")
    return path


def _peptides():
    def row(key, seq, listed, leading, start=1, before="-", after="A"):
        return {
            "peptide_row_id": key,
            "sequence_raw": seq,
            "parent_labels_raw": listed,
            "leading_parent_raw": leading,
            "start_1based_reported": start,
            "end_1based_reported": start + len(seq) - 1,
            "n_flank_raw": before,
            "c_flank_raw": after,
        }

    return [
        row("p0", "AAAAAA", "OLD", "OLD"),
        row("p1", "BCDEFG", "P1-9", "P1"),
        row("p2", "YYYYYY", "P1", "P1"),
        row("p3", "BCDEFG", "P2", "P2", after="Z"),
    ]


def test_alias_splits_repeats_sequences_and_coordinate_conflicts(tmp_path):
    records = list(read_reference(_dat(tmp_path), "uniprot_dat_gzip"))
    out = reference_relations("historical", records, _peptides())
    occurrences = out["peptide_reference_occurrences"]
    expected = {
        ("p0", "P1", 1),
        ("p0", "P1", 2),
        ("p0", "P1-2", 1),
        ("p0", "P1-2", 2),
        ("p1", "P1", 8),
        ("p1", "P1-2", 8),
        ("p1", "P2", 1),
        ("p3", "P1", 8),
        ("p3", "P1-2", 8),
        ("p3", "P2", 1),
    }
    assert {
        (r["peptide_row_id"], r["reference_record_id"].split(":")[1], r["start_1based"])
        for r in occurrences
    } == expected
    assert all(r["end_1based"] - r["start_1based"] == 5 for r in occurrences)
    leading_only = [
        r
        for r in occurrences
        if r["peptide_row_id"] == "p1" and r["reference_record_id"] == "historical:P1"
    ]
    assert leading_only[0]["source_leading_resolved"]
    assert not leading_only[0]["source_listed_resolved"]
    assert len(out["peptide_reference_coverage"]) == 4
    assert len(out["reference_proteins"]) == 3
    p0 = next(r for r in out["peptide_reference_coverage"] if r["peptide_row_id"] == "p0")
    assert p0["n_reference_records"] == 2 and p0["n_reference_sequences"] == 1
    assert p0["n_source_resolved_records"] == 2 and p0["n_source_matched_records"] == 1
    assert p0["leading_coordinate_flanks_match"]
    claims = out["source_parent_claims"]
    assert next(r for r in claims if r["source_accession_raw"] == "OLD")["n_candidate_records"] == 2
    assert (
        next(r for r in claims if r["source_accession_raw"] == "P1-9")["resolution_status"]
        == "missing_id"
    )
    negative = [
        r
        for r in out["source_parent_resolutions"]
        if r["occurrence_status"] == "resolved_sequence_absent"
    ]
    assert len(negative) == 2  # OLD resolves to P2 but AAAAAA is absent; YYYYYY absent in P1.
    coverage = {r["peptide_row_id"]: r for r in out["peptide_reference_coverage"]}
    assert coverage["p1"]["leading_coordinate_conflict"]
    assert coverage["p2"]["mapping_status"] == "no_exact_occurrence"
    assert (
        coverage["p0"]["shared_sequence_component_id"]
        == coverage["p1"]["shared_sequence_component_id"]
        == coverage["p3"]["shared_sequence_component_id"]
    )
    assert (
        coverage["p2"]["shared_sequence_component_id"]
        != coverage["p0"]["shared_sequence_component_id"]
    )


def test_fasta_does_not_inherit_historical_aliases(tmp_path):
    path = tmp_path / "human.fasta"
    path.write_text(
        ">sp|P1|P1_HUMAN OX=9606 SV=2\nAAAAAAABCDEFGH\n>sp|P1-2|P1_HUMAN OX=9606\nAAAAAAABCDEFGH\n"
    )
    out = reference_relations("modern", list(read_reference(path, "uniprot_fasta")), _peptides())
    assert (
        next(r for r in out["source_parent_claims"] if r["source_accession_raw"] == "OLD")[
            "resolution_status"
        ]
        == "missing_id"
    )
    assert {r["accession_role"] for r in out["reference_accessions"]} == {"primary"}
    assert out["reference_proteins"][0]["sequence_version"] == 2
    assert out["reference_proteins"][1]["sequence_version"] is None


def test_reference_files_are_hash_bound_and_quantity_files_untouched(tmp_path):
    (tmp_path / "typed").mkdir()
    (tmp_path / "assets").mkdir()
    path = _dat(tmp_path / "assets")
    quantity = tmp_path / "typed/quantity_results.parquet"
    quantity.write_bytes(b"immutable-quantity-fixture")
    spec = {
        "reference_id": "fixture",
        "asset_path": "human.dat.gz",
        "asset_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "asset_size_bytes": path.stat().st_size,
        "source_url": "fixture",
        "reference_format": "uniprot_dat_gzip",
        "generation_label": "test",
        "reference_role": "later_model_reference",
        "organism_scope": "human",
        "license": "fixture",
        "original_reference_status": "unavailable",
    }
    result = write_reference_relations(tmp_path, [spec], _peptides())
    verify_reference_relations(tmp_path, result)
    assert quantity.read_bytes() == b"immutable-quantity-fixture"
    assert pq.read_table(tmp_path / "typed/reference_proteins.parquet").num_rows == 3
    with pytest.raises(FileExistsError, match="already exist"):
        write_reference_relations(tmp_path, [spec], _peptides())
    verify_reference_relations(tmp_path, result)
    fresh = tmp_path / "bad-reference"
    (fresh / "assets").mkdir(parents=True)
    (fresh / "typed").mkdir()
    (fresh / "assets/human.dat.gz").write_bytes(path.read_bytes())
    bad = {**spec, "asset_sha256": "0" * 64}
    with pytest.raises(ValueError, match="hash or size"):
        write_reference_relations(fresh, [bad], _peptides())
    assert not list((fresh / "typed").iterdir())


def test_duplicate_identities_and_malformed_references_fail(tmp_path):
    proteins = list(read_reference(_dat(tmp_path), "uniprot_dat_gzip"))
    with pytest.raises(ValueError, match="Duplicate primary"):
        reference_relations("x", [*proteins, proteins[0]], _peptides())
    with pytest.raises(ValueError, match="Duplicate peptide"):
        reference_relations("x", proteins, [*_peptides(), _peptides()[0]])
    with pytest.raises(ValueError, match="Unsupported explicit"):
        list(read_reference(tmp_path / "missing", "guessed_format"))
    with pytest.raises(ValueError, match="literal peptide"):
        reference_relations("x", proteins, [{**_peptides()[0], "sequence_raw": "aaa"}])
    path = tmp_path / "invalid.fasta"
    path.write_text("SEQ\n")
    with pytest.raises(ValueError, match="before FASTA header"):
        list(read_reference(path, "uniprot_fasta"))
    path = tmp_path / "truncated.dat.gz"
    with gzip.open(path, "wt") as stream:
        stream.write("ID   unfinished\n")
    with pytest.raises(ValueError, match="Unterminated"):
        list(read_reference(path, "uniprot_dat_gzip"))
