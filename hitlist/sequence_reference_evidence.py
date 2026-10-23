"""Reference-scoped exact occurrences without assigning measured protein origin.

Primary records, secondary accession claims, full sequences, source candidates
and sequence occurrences have separate identities. No accession normalization,
I/L substitution, approximate match or cross-reference alias transfer is used.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def _schema(text=(), integers=(), flags=()):
    return pa.schema(
        [(x, pa.string()) for x in text]
        + [(x, pa.int64()) for x in integers]
        + [(x, pa.bool_()) for x in flags]
    )


SCHEMAS = {
    "sequence_references": _schema(
        (
            "reference_id",
            "asset_path",
            "asset_sha256",
            "source_url",
            "reference_format",
            "generation_label",
            "reference_role",
            "organism_scope",
            "license",
            "original_reference_status",
        ),
        ("asset_size_bytes",),
    ),
    "reference_proteins": _schema(
        (
            "reference_record_id",
            "reference_id",
            "accession",
            "sequence",
            "sequence_sha256",
            "header_raw",
            "metadata_json",
        ),
        ("sequence_length", "sequence_version"),
    ),
    "reference_accessions": _schema(
        ("reference_id", "reference_record_id", "accession_raw", "accession_role"),
    ),
    "peptide_reference_occurrences": _schema(
        (
            "occurrence_id",
            "peptide_row_id",
            "reference_id",
            "reference_record_id",
            "n_flank",
            "c_flank",
        ),
        ("start_1based", "end_1based"),
        ("source_listed_resolved", "source_leading_resolved"),
    ),
    "source_parent_claims": _schema(
        ("claim_id", "peptide_row_id", "reference_id", "source_accession_raw", "resolution_status"),
        ("n_candidate_records",),
        ("listed_in_source", "source_leading"),
    ),
    "source_parent_resolutions": _schema(
        (
            "claim_id",
            "reference_record_id",
            "accession_role",
            "occurrence_status",
            "coordinate_status",
        ),
        ("n_exact_occurrences",),
        (
            "coordinate_in_range",
            "coordinate_peptide_equal",
            "coordinate_left_equal",
            "coordinate_right_equal",
        ),
    ),
    "peptide_reference_coverage": _schema(
        ("peptide_row_id", "reference_id", "mapping_status", "shared_sequence_component_id"),
        (
            "n_source_claims",
            "n_missing_claims",
            "n_source_resolved_records",
            "n_source_matched_records",
            "n_reference_records",
            "n_reference_sequences",
            "n_occurrences",
            "n_additional_records",
        ),
        (
            "leading_has_occurrence",
            "leading_coordinate_match",
            "leading_coordinate_flanks_match",
            "leading_coordinate_conflict",
        ),
    ),
}


def _record(accession, sequence, *, aliases=(), header=None, metadata=None, version=None):
    if not accession or not sequence or not re.fullmatch("[A-Z]+", sequence):
        raise ValueError("Invalid literal reference sequence or accession")
    aliases = list(aliases)
    if len({accession, *aliases}) != 1 + len(aliases):
        raise ValueError("Duplicate accession in one reference record")
    return {
        "accession": accession,
        "sequence": sequence,
        "aliases": aliases,
        "sequence_sha256": hashlib.sha256(sequence.encode()).hexdigest(),
        "header_raw": header,
        "metadata": metadata or {},
        "sequence_version": version,
    }


def read_reference(path, reference_format):
    """Read literal reviewed UniProt FASTA or DAT with reference-local aliases."""
    path = Path(path)
    if reference_format == "uniprot_fasta":
        header, sequence = None, []

        def finish():
            parts = header.split("|", 2)
            if len(parts) != 3 or parts[0] != "sp":
                raise ValueError("Expected reviewed UniProt FASTA header")
            match = re.search(r"\bSV=(\d+)", header)
            taxonomy = re.search(r"\bOX=(\d+)", header)
            return _record(
                parts[1],
                "".join(sequence),
                header=header,
                metadata={
                    "entry": parts[2].split()[0],
                    "taxonomy": taxonomy.group(1) if taxonomy else None,
                },
                version=int(match.group(1)) if match else None,
            )

        with path.open() as stream:
            for raw in stream:
                line = raw.strip()
                if not line:
                    continue
                if line.startswith(">"):
                    if header is not None:
                        yield finish()
                    header, sequence = line[1:], []
                elif header is None:
                    raise ValueError("Sequence before FASTA header")
                else:
                    sequence.append(line)
        if header is not None:
            yield finish()
    elif reference_format == "uniprot_dat_gzip":
        tags, sequence, in_sequence = defaultdict(list), [], False
        with gzip.open(path, "rt") as stream:
            for raw in stream:
                line = raw.rstrip("\n")
                if line == "//":
                    accessions = " ".join(tags["AC"]).replace(";", " ").split()
                    size = re.search(r"(\d+) AA", " ".join(tags["SQ"]))
                    version = [
                        re.search(r"sequence version (\d+)", x).group(1)
                        for x in tags["DT"]
                        if "sequence version " in x
                    ]
                    seq = "".join(sequence)
                    if (
                        not accessions
                        or size is None
                        or len(seq) != int(size.group(1))
                        or len(version) != 1
                    ):
                        raise ValueError("Malformed UniProt DAT record")
                    yield _record(
                        accessions[0],
                        seq,
                        aliases=accessions[1:],
                        metadata={
                            "id_lines": tags["ID"],
                            "dates": tags["DT"],
                            "taxonomy": tags["OX"],
                        },
                        version=int(version[0]),
                    )
                    tags, sequence, in_sequence = defaultdict(list), [], False
                elif in_sequence:
                    sequence.append("".join(line.split()))
                else:
                    tags[line[:2]].append(line[5:])
                    in_sequence = line.startswith("SQ")
        if tags or sequence:
            raise ValueError("Unterminated UniProt DAT record")
    else:
        raise ValueError("Unsupported explicit reference format")


def exact_occurrences(proteins, peptides):
    """Map every supplied identity, retaining all overlapping exact positions."""
    if not peptides or any(not re.fullmatch("[A-Z]+", seq) for seq in peptides.values()):
        raise ValueError("Expected nonempty literal peptide sequences")
    width = min(6, min(map(len, peptides.values())))
    index = defaultdict(list)
    for key, sequence in peptides.items():
        index[sequence[:width]].append((key, sequence))
    found = defaultdict(list)
    for accession, record in proteins.items():
        sequence = record["sequence"]
        for start in range(len(sequence) - width + 1):
            for key, peptide in index.get(sequence[start : start + width], ()):
                if sequence.startswith(peptide, start):
                    found[key].append((accession, start + 1))
    return found


def _components(peptides, found, proteins):
    roots = {key: key for key in peptides}

    def root(key):
        while key != roots[key]:
            roots[key] = roots[roots[key]]
            key = roots[key]
        return key

    owners = {}
    for key in sorted(peptides):
        for accession, _ in found[key]:
            checksum = proteins[accession]["sequence_sha256"]
            if checksum in owners:
                a, b = root(key), root(owners[checksum])
                roots[max(a, b)] = min(a, b)
            else:
                owners[checksum] = key
    return {key: root(key) for key in peptides}


def _coordinate(peptide, protein, leading):
    if not leading:
        return "not_leading_source_claim", (None,) * 4
    start, end = peptide.get("start_1based_reported"), peptide.get("end_1based_reported")
    if start is None or end is None:
        return "source_coordinates_unreported", (None,) * 4
    sequence = protein["sequence"]
    in_range = 1 <= start <= end <= len(sequence)
    matches = (
        in_range,
        in_range and sequence[start - 1 : end] == peptide["sequence_raw"],
        in_range and peptide["n_flank_raw"] == (sequence[start - 2] if start > 1 else "-"),
        in_range and peptide["c_flank_raw"] == (sequence[end] if end < len(sequence) else "-"),
    )
    status = (
        "out_of_range"
        if not in_range
        else (
            "peptide_mismatch"
            if not matches[1]
            else ("peptide_and_flanks_match" if all(matches) else "peptide_match_flank_mismatch")
        )
    )
    return status, matches


def reference_relations(reference_id, proteins, peptide_rows):
    """Return separate relation rows; quantity observations are never joined here."""
    peptides = {p["peptide_row_id"]: p for p in peptide_rows}
    if len(peptides) != len(peptide_rows) or not reference_id:
        raise ValueError("Duplicate peptide identity or empty reference ID")
    records, aliases = {}, defaultdict(set)
    for protein in proteins:
        accession = protein["accession"]
        if accession in records:
            raise ValueError("Duplicate primary reference accession")
        records[accession] = protein
        for alias in [accession, *protein["aliases"]]:
            aliases[alias].add(accession)
    if not records:
        raise ValueError("Empty sequence reference")
    found = exact_occurrences(records, {k: p["sequence_raw"] for k, p in peptides.items()})
    components = _components(peptides, found, records)
    out = {name: [] for name in SCHEMAS if name != "sequence_references"}
    for accession, protein in records.items():
        rid = f"{reference_id}:{accession}"
        out["reference_proteins"].append(
            {
                "reference_record_id": rid,
                "reference_id": reference_id,
                "accession": accession,
                "sequence": protein["sequence"],
                "sequence_sha256": protein["sequence_sha256"],
                "header_raw": protein["header_raw"],
                "metadata_json": json.dumps(protein["metadata"], sort_keys=True),
                "sequence_length": len(protein["sequence"]),
                "sequence_version": protein["sequence_version"],
            }
        )
        for alias in [accession, *protein["aliases"]]:
            out["reference_accessions"].append(
                {
                    "reference_id": reference_id,
                    "reference_record_id": rid,
                    "accession_raw": alias,
                    "accession_role": "primary" if alias == accession else "recorded_secondary",
                }
            )
    for key, peptide in peptides.items():
        leading = peptide["leading_parent_raw"]
        listed = set(filter(None, peptide["parent_labels_raw"].split(";")))
        claims = sorted(listed | ({leading} if leading else set()))
        listed_resolved = set().union(*(aliases.get(c, set()) for c in listed))
        resolved, coordinates, leading_matches = set(), [], False
        by_accession = defaultdict(list)
        for accession, start in found[key]:
            by_accession[accession].append(start)
        for claim in claims:
            cid = f"{key}:reference:{reference_id}:source_parent:{claim}"
            candidates = sorted(aliases.get(claim, ()))
            resolved.update(candidates)
            out["source_parent_claims"].append(
                {
                    "claim_id": cid,
                    "peptide_row_id": key,
                    "reference_id": reference_id,
                    "source_accession_raw": claim,
                    "resolution_status": "resolved" if candidates else "missing_id",
                    "n_candidate_records": len(candidates),
                    "listed_in_source": claim in listed,
                    "source_leading": claim == leading,
                }
            )
            for accession in candidates:
                status, comparison = _coordinate(peptide, records[accession], claim == leading)
                if claim == leading:
                    coordinates.append(comparison)
                    leading_matches |= bool(by_accession[accession])
                out["source_parent_resolutions"].append(
                    {
                        "claim_id": cid,
                        "reference_record_id": f"{reference_id}:{accession}",
                        "accession_role": "primary" if accession == claim else "recorded_secondary",
                        "occurrence_status": "exact_occurrence"
                        if by_accession[accession]
                        else "resolved_sequence_absent",
                        "n_exact_occurrences": len(by_accession[accession]),
                        "coordinate_status": status,
                        **dict(
                            zip(
                                (
                                    "coordinate_in_range",
                                    "coordinate_peptide_equal",
                                    "coordinate_left_equal",
                                    "coordinate_right_equal",
                                ),
                                comparison,
                            )
                        ),
                    }
                )
        for accession, start in found[key]:
            seq = records[accession]["sequence"]
            end = start + len(peptide["sequence_raw"]) - 1
            out["peptide_reference_occurrences"].append(
                {
                    "occurrence_id": f"{key}:{reference_id}:{accession}:{start}",
                    "peptide_row_id": key,
                    "reference_id": reference_id,
                    "reference_record_id": f"{reference_id}:{accession}",
                    "start_1based": start,
                    "end_1based": end,
                    "n_flank": seq[start - 2] if start > 1 else "-",
                    "c_flank": seq[end] if end < len(seq) else "-",
                    "source_listed_resolved": accession in listed_resolved,
                    "source_leading_resolved": accession in aliases.get(leading, set()),
                }
            )
        matched = set(by_accession)
        # Only observed occurrences belong in matched; absent resolution lookups create empty lists.
        matched = {a for a in matched if by_accession[a]}
        measured_coords = [c for c in coordinates if c[0] is not None]
        out["peptide_reference_coverage"].append(
            {
                "peptide_row_id": key,
                "reference_id": reference_id,
                "mapping_status": "exact_occurrence" if matched else "no_exact_occurrence",
                "shared_sequence_component_id": f"{reference_id}:component:{components[key]}",
                "n_source_claims": len(claims),
                "n_missing_claims": sum(not aliases.get(c) for c in claims),
                "n_source_resolved_records": len(resolved),
                "n_source_matched_records": len(resolved & matched),
                "n_reference_records": len(matched),
                "n_reference_sequences": len({records[a]["sequence_sha256"] for a in matched}),
                "n_occurrences": len(found[key]),
                "n_additional_records": len(matched - resolved),
                "leading_has_occurrence": leading_matches,
                "leading_coordinate_match": any(c[1] for c in measured_coords),
                "leading_coordinate_flanks_match": any(all(c) for c in measured_coords),
                "leading_coordinate_conflict": bool(measured_coords)
                and not any(c[1] for c in measured_coords),
            }
        )
    return out


def write_reference_relations(source_dir, specifications, peptide_rows):
    source_dir = Path(source_dir)
    if (
        len({s["reference_id"] for s in specifications}) != len(specifications)
        or not specifications
    ):
        raise ValueError("Duplicate or missing reference specifications")
    if any((source_dir / "typed" / f"{name}.parquet").exists() for name in SCHEMAS):
        raise FileExistsError("Reference relations already exist")
    prepared = []
    for spec in specifications:
        relative = Path(spec["asset_path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Invalid reference asset path")
        path = source_dir / "assets" / relative
        checksum = hashlib.sha256(path.read_bytes()).hexdigest()
        if checksum != spec["asset_sha256"] or path.stat().st_size != spec["asset_size_bytes"]:
            raise ValueError("Reference asset hash or size mismatch")
        record = {k: spec[k] for k in SCHEMAS["sequence_references"].names}
        prepared.append((spec, path, record))
    writers = {
        name: pq.ParquetWriter(source_dir / "typed" / f"{name}.parquet", schema, compression="zstd")
        for name, schema in SCHEMAS.items()
    }
    counts = defaultdict(int)
    per_reference = []
    try:
        for spec, path, record in prepared:
            relations = reference_relations(
                spec["reference_id"],
                list(read_reference(path, spec["reference_format"])),
                peptide_rows,
            )
            relations["sequence_references"] = [record]
            for name, rows in relations.items():
                writers[name].write_table(pa.Table.from_pylist(rows, schema=SCHEMAS[name]))
                counts[name] += len(rows)
            per_reference.append(
                {
                    "reference_id": spec["reference_id"],
                    **{f"n_{name}_rows": len(rows) for name, rows in relations.items()},
                }
            )
    finally:
        for writer in writers.values():
            writer.close()
    return {"reference_counts": per_reference, "reference_relation_rows": dict(counts)}


def verify_reference_relations(source_dir, reconciliation):
    for name, schema in SCHEMAS.items():
        file = pq.ParquetFile(Path(source_dir) / "typed" / f"{name}.parquet")
        if (
            file.schema_arrow != schema
            or file.metadata.num_rows != reconciliation["reference_relation_rows"][name]
        ):
            raise ValueError(f"Reference relation mismatch: {name}")
