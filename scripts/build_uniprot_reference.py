"""Stream a taxon's FASTA from an official historical UniProt release.

Maintainer tool: large all-species archives are read once, never stored or
extracted. Only the bounded taxon FASTA and a source receipt are written.
The installed Hitlist client downloads this compact, checksum-pinned result.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import tarfile
import tempfile
import time
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

CHUNK_BYTES = 2**20
MAX_RECORD_BYTES = 4 * 2**20
MAX_OUTPUT_BYTES = 256 * 2**20


class HashedReader:
    def __init__(self, source, expected_size):
        self.source = source
        self.expected_size = expected_size
        self.size_bytes = 0
        self.sha256 = hashlib.sha256()
        self.md5 = hashlib.md5()
        self.started = time.monotonic()
        self.last_report = self.started

    def read(self, size=-1):
        if size < 0:
            raise ValueError("Unbounded archive read")
        value = self.source.read(size)
        self.size_bytes += len(value)
        if self.size_bytes > self.expected_size:
            raise ValueError("Archive exceeds published size")
        self.sha256.update(value)
        self.md5.update(value)
        if time.monotonic() - self.last_report > 30:
            print(f"Archive: {self.size_bytes:,}/{self.expected_size:,} bytes", flush=True)
            self.last_report = time.monotonic()
        return value


def records(source, separator):
    """Split large streams in C-sized chunks while bounding an individual record."""
    pending = b""
    while chunk := source.read(CHUNK_BYTES):
        parts = (pending + chunk).split(separator)
        pending = parts.pop()
        for part in parts:
            if len(part) > MAX_RECORD_BYTES:
                raise ValueError("Oversized sequence record")
            yield part
        if len(pending) > MAX_RECORD_BYTES:
            raise ValueError("Oversized sequence record")
    if pending.strip():
        yield pending


def fasta_records(source):
    for record in records(source, b"\n>"):
        record = record.lstrip(b">")
        header, _, sequence = record.partition(b"\n")
        yield header.decode("utf-8"), b"".join(sequence.split()).decode("ascii")


def dat_fasta(record, taxonomy_id):
    # Historical TrEMBL OX lines can include evidence annotations before the
    # semicolon, with their evidence list continuing onto additional OX lines.
    # Match the complete numeric source taxon; OH organism-host lines and taxon
    # prefixes must never admit a record to the selected reference.
    taxon = re.search(rb"(?m)^OX   NCBI_TaxID=(\d+)(?=;|[ \t]+\{)", record)
    if taxon is None or int(taxon.group(1)) != taxonomy_id:
        return None
    text = record.decode("utf-8")
    accession = re.search(r"\nAC   (\w+);", text).group(1)
    identifier = re.match(r"ID   (\S+)", text).group(1)
    version = re.search(r"\nDT   .*sequence version (\d+)\.", text).group(1)
    species = re.search(r"\nOS   (.*)", text).group(1).rstrip(".")
    name = re.search(r"\nDE   (?:RecName|SubName): Full=(.*?);", text)
    gene = re.search(r"\nGN   Name=([^; {]+)", text)
    sequence = "".join(text.split("\nSQ   ", 1)[1].split("\n", 1)[1].split())
    header = f"tr|{accession}|{identifier} {name.group(1) if name else identifier} OS={species} OX={taxonomy_id}"
    if gene:
        header += f" GN={gene.group(1)}"
    header += f" SV={version}"
    return header, sequence


class FastaWriter:
    def __init__(self, handle):
        self.handle = handle
        self.size_bytes = 0
        self.sha256 = hashlib.sha256()
        self.accessions = set()
        self.counts = {
            "n_reviewed_canonical_sequences": 0,
            "n_reviewed_isoform_sequences": 0,
            "n_unreviewed_canonical_sequences": 0,
        }

    def write(self, header, sequence, kind):
        accession = header.split("|")[1]
        if accession in self.accessions:
            raise ValueError(f"Duplicate accession: {accession}")
        if not sequence or re.fullmatch("[A-Z]+", sequence) is None:
            raise ValueError(f"Invalid sequence: {accession}")
        self.accessions.add(accession)
        data = (
            ">"
            + header
            + "\n"
            + "\n".join(sequence[i : i + 60] for i in range(0, len(sequence), 60))
            + "\n"
        ).encode()
        self.size_bytes += len(data)
        if self.size_bytes > MAX_OUTPUT_BYTES:
            raise ValueError("Human FASTA exceeds output budget")
        self.handle.write(data)
        self.sha256.update(data)
        self.counts[kind] += 1


def extract_archive(url, expected, writer, taxonomy_id, *, reviewed):
    print(f"Reading {url}", flush=True)
    members = []
    with urllib.request.urlopen(url, timeout=60) as response:
        reader = HashedReader(response, expected["size_bytes"])
        # tarfile streaming mode retains member headers. Clear that list after
        # every iteration; sequence records and buffers also have fixed bounds.
        with (
            gzip.GzipFile(fileobj=reader) as expanded,
            tarfile.open(fileobj=expanded, mode="r|", bufsize=CHUNK_BYTES) as archive,
        ):
            for n_members, member in enumerate(archive):
                archive.members.clear()
                name = Path(member.name).name
                if n_members < 30:
                    print(f"Member {member.name}: {member.size:,} bytes", flush=True)
                if not member.isfile():
                    continue
                plain_name = name.removesuffix(".gz")
                selected = plain_name in (
                    ("uniprot_sprot.fasta", "uniprot_sprot_varsplic.fasta")
                    if reviewed
                    else ("uniprot_trembl.dat",)
                )
                if not selected:
                    continue
                source = archive.extractfile(member)
                if name.endswith(".gz"):
                    source = gzip.GzipFile(fileobj=source)
                members.append({"name": member.name, "size_bytes": member.size})
                if reviewed:
                    # Archived 2015 FASTA headers predate OX=. UniProt's entry
                    # identifier suffix HUMAN is the taxon code for Homo sapiens.
                    if taxonomy_id != 9606:
                        raise ValueError(
                            "Historical FASTA taxon-code selection currently supports human only"
                        )
                    kind = (
                        "n_reviewed_isoform_sequences"
                        if "varsplic" in name
                        else "n_reviewed_canonical_sequences"
                    )
                    for header, sequence in fasta_records(source):
                        if re.match(r"sp\|[^|]+\|\S+_HUMAN(?:\s|$)", header):
                            writer.write(header, sequence, kind)
                else:
                    for record in records(source, b"\n//\n"):
                        result = dat_fasta(record, taxonomy_id)
                        if result is not None:
                            writer.write(*result, "n_unreviewed_canonical_sequences")
                source.close()
            # Consume gzip trailers and any tar padding before accepting hashes.
            while expanded.read(CHUNK_BYTES):
                pass
        while reader.read(CHUNK_BYTES):
            pass
    if reader.size_bytes != expected["size_bytes"] or reader.md5.hexdigest() != expected["md5"]:
        raise ValueError(f"Official archive size/MD5 mismatch: {url}")
    if len(members) != (2 if reviewed else 1):
        raise ValueError(f"Unexpected archive layout: {members}")
    return {"url": url, **expected, "sha256": reader.sha256.hexdigest(), "members": members}


def _require_release(response, release):
    actual = response.headers.get("X-UniProt-Release")
    if actual != release:
        raise ValueError(f"Requested UniProt {release}, server returned {actual}")


def extract_rest(release, writer):
    base = "https://rest.uniprot.org/uniprotkb/"
    query = "query=%28organism_id%3A9606%29"
    count_url = base + "search?" + query + "&format=tsv&fields=accession&size=1"
    with urllib.request.urlopen(count_url, timeout=60) as response:
        _require_release(response, release)
        expected_canonical = int(response.headers["X-Total-Results"])
        count_receipt = {"url": count_url, "headers": dict(response.headers)}
        response.read(2**20)
    url = base + "stream?" + query + "&format=fasta&includeIsoform=true&compressed=true"
    with urllib.request.urlopen(url, timeout=60) as response:
        _require_release(response, release)
        headers = dict(response.headers)
        reader = HashedReader(response, MAX_OUTPUT_BYTES)
        for header, sequence in fasta_records(gzip.GzipFile(fileobj=reader)):
            # Current isoform headers omit OX=, but canonical entries carry it.
            if not re.match(r"(?:sp|tr)\|[^|]+\|\S+_HUMAN(?:\s|$)", header):
                raise ValueError(f"Unexpected human FASTA header: {header}")
            accession = header.split("|")[1]
            kind = (
                "n_reviewed_isoform_sequences"
                if "-" in accession
                else "n_reviewed_canonical_sequences"
                if header.startswith("sp|")
                else "n_unreviewed_canonical_sequences"
            )
            if kind == "n_reviewed_isoform_sequences" and not header.startswith("sp|"):
                raise ValueError("Unexpected unreviewed isoform")
            writer.write(header, sequence, kind)
    canonical = (
        writer.counts["n_reviewed_canonical_sequences"]
        + writer.counts["n_unreviewed_canonical_sequences"]
    )
    if canonical != expected_canonical or writer.counts["n_reviewed_isoform_sequences"] == 0:
        raise ValueError(
            f"Incomplete REST human reference: {writer.counts}; expected {expected_canonical} canonical entries"
        )
    return [
        {
            "url": url,
            "headers": headers,
            "size_bytes": reader.size_bytes,
            "sha256": reader.sha256.hexdigest(),
            "canonical_count_check": count_receipt,
        }
    ]


def main():
    builder_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--release", default="2015_10")
    parser.add_argument("--source", choices=["archive", "rest"], default="archive")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    source_metadata = None
    if re.fullmatch(r"\d{4}_\d{2}", args.release) is None:
        parser.error("--release must be an explicit YYYY_NN release")
    if args.source == "archive":
        if args.release != "2015_10":
            parser.error("Historical extraction currently validates 2015_10 counts only")
        base = f"https://ftp.uniprot.org/pub/databases/uniprot/previous_releases/release-{args.release}/knowledgebase/"
        with urllib.request.urlopen(base + "RELEASE.metalink", timeout=60) as response:
            metalink = response.read(2**20)
        source_metadata = {
            "url": base + "RELEASE.metalink",
            "sha256": hashlib.sha256(metalink).hexdigest(),
        }
        tree = ET.fromstring(metalink)
        for element in tree.iter():
            if element.tag.rsplit("}", 1)[-1] == "file":
                fields = {child.tag.rsplit("}", 1)[-1]: child for child in element.iter()}
                files[element.attrib["name"]] = {
                    "size_bytes": int(fields["size"].text),
                    "md5": fields["hash"].text,
                }
    output = args.output_dir / f"uniprot-human-{args.release}.fasta"
    if output.exists():
        raise FileExistsError(output)
    staging = tempfile.NamedTemporaryFile(dir=args.output_dir, prefix=".uniprot-", delete=False)  # noqa: SIM115 -- closed by the with below, cleaned by finally
    partial = Path(staging.name)
    try:
        with staging as handle:
            writer = FastaWriter(handle)
            if args.source == "rest":
                sources = extract_rest(args.release, writer)
            else:
                sources = []
                for name, reviewed in [
                    (f"uniprot_sprot-only{args.release}.tar.gz", True),
                    (f"knowledgebase{args.release}.tar.gz", False),
                ]:
                    sources.append(
                        extract_archive(base + name, files[name], writer, 9606, reviewed=reviewed)
                    )
        if args.source == "archive" and (
            writer.counts["n_reviewed_canonical_sequences"] != 20196
            or writer.counts["n_unreviewed_canonical_sequences"] != 128790
        ):
            raise ValueError(
                f"Counts differ from official human release statistics: {writer.counts}"
            )
        if writer.counts["n_reviewed_isoform_sequences"] == 0:
            raise ValueError("No reviewed isoforms found")
        receipt = {
            "schema_version": 1,
            "release": args.release,
            "taxonomy_id": 9606,
            "selection": "All human Swiss-Prot and TrEMBL canonical entries plus reviewed isoforms",
            "source_metadata": source_metadata,
            "sources": sources,
            "filename": output.name,
            "size_bytes": writer.size_bytes,
            "sha256": writer.sha256.hexdigest(),
            "sequence_counts": writer.counts,
            "n_sequences": sum(writer.counts.values()),
            "builder_sha256": builder_sha256,
            "license": "CC-BY-4.0",
            "license_url": "https://www.uniprot.org/help/license",
            "attribution": "The UniProt Consortium; human-only selection and FASTA conversion by Hitlist",
        }
        if output.exists():
            raise FileExistsError(output)
        partial.rename(output)
        output.with_suffix(".receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        print(json.dumps(receipt, indent=2), flush=True)
    finally:
        partial.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
