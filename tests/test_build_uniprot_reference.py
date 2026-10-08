"""Independent tiny archives exercise the maintainer's streaming extractor."""

import gzip
import hashlib
import io
import tarfile

import pytest
from scripts import build_uniprot_reference as build


def archive_bytes(files):
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w:gz") as archive:
        for name, data in files.items():
            member = tarfile.TarInfo(name)
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))
    return output.getvalue()


def extract(monkeypatch, data, *, reviewed, expected_md5=None):
    monkeypatch.setattr(build.urllib.request, "urlopen", lambda *a, **kw: io.BytesIO(data))
    output = io.BytesIO()
    writer = build.FastaWriter(output)
    receipt = build.extract_archive(
        "https://example.test/archive.gz",
        {"size_bytes": len(data), "md5": expected_md5 or hashlib.md5(data).hexdigest()},
        writer,
        9606,
        reviewed=reviewed,
    )
    return output.getvalue(), writer, receipt


def test_reviewed_archive_filters_taxon_and_streams_nested_gzip(monkeypatch):
    data = archive_bytes(
        {
            "../ignored.txt": b"ignored, never extracted",
            "dir/uniprot_sprot.fasta.gz": gzip.compress(
                b">sp|P1|ONE_HUMAN protein\nAAK\n>sp|P2|TWO_MOUSE protein\nDDK\n", mtime=0
            ),
            "dir/uniprot_sprot_varsplic.fasta": b">sp|P1-2|ONE_HUMAN Isoform 2\nCCK\n>sp|P2-2|TWO_MOUSE Isoform 2\nEEK\n",
        }
    )
    output, writer, receipt = extract(monkeypatch, data, reviewed=True)
    assert output == b">sp|P1|ONE_HUMAN protein\nAAK\n>sp|P1-2|ONE_HUMAN Isoform 2\nCCK\n"
    assert writer.counts == {
        "n_reviewed_canonical_sequences": 1,
        "n_reviewed_isoform_sequences": 1,
        "n_unreviewed_canonical_sequences": 0,
    }
    assert receipt["sha256"] == hashlib.sha256(data).hexdigest()
    assert len(receipt["members"]) == 2


DAT = b"""ID   ONE_HUMAN Unreviewed; 6 AA.
AC   A0AAA1;
DT   14-OCT-2015, sequence version 2.
DE   SubName: Full=Example;
GN   Name=GENE;
OS   Homo sapiens (Human).
OX   NCBI_TaxID=9606;
SQ   SEQUENCE   6 AA;
     AAK CCK
//
"""


def test_dat_taxonomy_filter_preserves_accession_gene_and_sequence_version(monkeypatch):
    mouse = DAT.replace(b"9606", b"10090").replace(b"ONE_HUMAN", b"ONE_MOUSE")
    data = archive_bytes({"uniprot_trembl.dat": mouse + DAT})
    output, writer, _ = extract(monkeypatch, data, reviewed=False)
    assert (
        output
        == b">tr|A0AAA1|ONE_HUMAN Example OS=Homo sapiens (Human) OX=9606 GN=GENE SV=2\nAAKCCK\n"
    )
    assert writer.counts["n_unreviewed_canonical_sequences"] == 1


def test_record_boundaries_cross_read_chunks(monkeypatch):
    monkeypatch.setattr(build, "CHUNK_BYTES", 7)
    assert list(build.records(io.BytesIO(DAT + DAT), b"\n//\n")) == [DAT[:-4], DAT[:-4]]
    assert list(build.fasta_records(io.BytesIO(b">a\nAA\nK\n>b\nCCK\n"))) == [
        ("a", "AAK"),
        ("b", "CCK"),
    ]


def test_bad_official_hash_rejects_extraction(monkeypatch):
    data = archive_bytes({"uniprot_trembl.dat": DAT})
    with pytest.raises(ValueError, match="MD5 mismatch"):
        extract(monkeypatch, data, reviewed=False, expected_md5="0" * 32)


def test_missing_expected_archive_member_fails(monkeypatch):
    data = archive_bytes({"uniprot_sprot.fasta": b">sp|P1|ONE_HUMAN\nAAK\n"})
    with pytest.raises(ValueError, match="layout"):
        extract(monkeypatch, data, reviewed=True)


def test_resource_and_duplicate_guards(monkeypatch):
    monkeypatch.setattr(build, "MAX_RECORD_BYTES", 5)
    with pytest.raises(ValueError, match="Oversized"):
        list(build.records(io.BytesIO(b"123456"), b"\n"))
    writer = build.FastaWriter(io.BytesIO())
    writer.write("sp|P1|ONE_HUMAN", "AAK", "n_reviewed_canonical_sequences")
    with pytest.raises(ValueError, match="Duplicate"):
        writer.write("sp|P1|ONE_HUMAN", "AAK", "n_reviewed_canonical_sequences")
    monkeypatch.setattr(build, "MAX_OUTPUT_BYTES", 1)
    with pytest.raises(ValueError, match="output budget"):
        writer.write("sp|P2|TWO_HUMAN", "CCK", "n_reviewed_canonical_sequences")
    with pytest.raises(ValueError, match="published size"):
        build.HashedReader(io.BytesIO(b"XX"), 1).read(2)


def test_rest_requires_matching_release_headers_and_complete_canonical_count(monkeypatch):
    fasta = b">sp|P1|ONE_HUMAN\nAAK\n>sp|P1-2|ONE_HUMAN\nCCK\n>tr|Q1|TWO_HUMAN\nDDK\n"
    expected = "2"
    release = "2026_03"

    def respond(url, **kwargs):
        result = io.BytesIO(gzip.compress(fasta, mtime=0) if "/stream?" in url else b"Entry\nP1\n")
        result.headers = {"X-UniProt-Release": release, "X-Total-Results": expected}
        return result

    monkeypatch.setattr(build.urllib.request, "urlopen", respond)
    writer = build.FastaWriter(io.BytesIO())
    receipt = build.extract_rest("2026_03", writer)[0]
    assert receipt["sha256"] == hashlib.sha256(gzip.compress(fasta, mtime=0)).hexdigest()
    assert sum(writer.counts.values()) == 3
    expected = "3"
    with pytest.raises(ValueError, match="Incomplete"):
        build.extract_rest("2026_03", build.FastaWriter(io.BytesIO()))
    release = "2026_04"
    with pytest.raises(ValueError, match="server returned"):
        build.extract_rest("2026_03", build.FastaWriter(io.BytesIO()))
