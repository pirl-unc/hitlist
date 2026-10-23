"""Append the pinned PXD057588 mouse-reference audit as typed evidence.

The source peptide row, assay result, numeric cell, reference sequence and
exact parent occurrence remain separate grains. This generation copies every
predecessor byte into a new directory and never expands a measured row.
"""

from __future__ import annotations

import csv
import gzip
import json
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from .provenance import file_digest
from .supporting_numeric_evidence import verify_supporting_numeric_evidence_package

PACKAGE_SCHEMA_VERSION = 7
CATALOG_SCHEMA_VERSION = 9
PRESTO_COMMIT = "24f1d2b"
PRESTO_REPORT = (
    "pirl-unc/presto/experiments/2026-10-22_1914_codex_pxd057588-shared-sequence-admission"
)
OUTPUT_RECEIPT_SHA256 = "a6599a24c01f484660b91f227a57f68f6904605fbce8389e3b096c0e13abfcf6"
INDEPENDENT_REPLAY_SHA256 = "4c6ad0aa3e88f391195532519422fe71fe6927edc819d406862488a6cb80ea27"
REFERENCE_SHA256 = "41bc0fa6cb9118ba95f330a27b18c39559e9397b6d99d1e73cc094869c99787a"
ROOT = Path("external/pxd057588/reference")
TYPED = Path("external/pxd057588/typed")
EXPECTED_ROWS = {
    "reference_source_mappings": 35_772,
    "reference_sequences": 30_587,
    "reference_occurrences": 52_277,
    "reference_parents": 25_758,
    "reference_parent_support": 26_449,
    "reference_result_context": 309_126,
}


def _schema(*items):
    return pa.schema([pa.field(name, dtype) for name, dtype in items])


SCHEMAS = {
    "reference_source_mappings": _schema(
        ("peptide_row_id", pa.string()),
        ("sequence_id", pa.string()),
        ("sequence_raw", pa.string()),
        ("component", pa.string()),
        ("table_number", pa.int64()),
        ("row_1based", pa.int64()),
        ("asset_path", pa.string()),
        ("sequence_cell", pa.string()),
        ("parent_label_cell", pa.string()),
        ("parent_label_raw", pa.string()),
        ("parent_mapping_status", pa.string()),
        ("source_parent_status", pa.string()),
        ("source_parent_positions_start1", pa.list_(pa.int64())),
        ("additional_parent_identities", pa.int64()),
        ("reference_identity_count", pa.int64()),
        ("reference_full_sequence_count", pa.int64()),
        ("reference_occurrence_count", pa.int64()),
        ("measured_pool", pa.string()),
        ("measured_population", pa.string()),
        ("partner_population", pa.string()),
        ("material_link_precision", pa.string()),
        ("selection_scope", pa.string()),
    ),
    "reference_sequences": _schema(
        ("sequence_id", pa.string()),
        ("sequence", pa.string()),
        ("length", pa.int64()),
        ("reference_identity_count", pa.int64()),
        ("reference_full_sequence_count", pa.int64()),
        ("reference_occurrence_count", pa.int64()),
        ("component", pa.string()),
    ),
    "reference_occurrences": _schema(
        ("sequence_id", pa.string()),
        ("peptide_sequence", pa.string()),
        ("component", pa.string()),
        ("accession", pa.string()),
        ("start1", pa.int64()),
        ("end1", pa.int64()),
        ("left", pa.string()),
        ("right", pa.string()),
        ("parent_sequence_sha256", pa.string()),
    ),
    "reference_parents": _schema(
        ("accession", pa.string()),
        ("entry", pa.string()),
        ("length", pa.int64()),
        ("sequence_sha256", pa.string()),
        ("sequence", pa.string()),
        ("taxonomy", pa.list_(pa.string())),
        ("aliases", pa.list_(pa.string())),
        ("reference_status", pa.string()),
    ),
    "reference_parent_support": _schema(
        ("definition", pa.string()),
        ("parent_key", pa.string()),
        ("table9_rows", pa.list_(pa.string())),
        ("table11_rows", pa.list_(pa.string())),
        ("n_table9_rows", pa.int64()),
        ("n_table11_rows", pa.int64()),
    ),
    "reference_species_roles": _schema(
        ("claim_id", pa.string()),
        ("source_dataset", pa.string()),
        ("population_role", pa.string()),
        ("population_label", pa.string()),
        ("organism_taxon_id", pa.string()),
        ("organism_name", pa.string()),
        ("source_scope", pa.string()),
        ("source_locator", pa.string()),
        ("role_basis", pa.string()),
    ),
    "reference_result_context": _schema(
        ("result_id", pa.string()),
        ("peptide_row_id", pa.string()),
        ("source_dataset", pa.string()),
        ("sequence_id", pa.string()),
        ("component", pa.string()),
        ("source_parent_status", pa.string()),
        ("reference_identity_count", pa.int64()),
        ("reference_full_sequence_count", pa.int64()),
        ("reference_occurrence_count", pa.int64()),
        ("table_number", pa.int64()),
        ("measured_pool", pa.string()),
        ("measured_population", pa.string()),
        ("partner_population", pa.string()),
        ("material_link_precision", pa.string()),
        ("specimen_status", pa.string()),
        ("organism_taxon_id", pa.string()),
        ("organism_source_scope", pa.string()),
    ),
}


def _copy(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if sys.platform == "darwin":
        # APFS copy-on-write gives an independent inode and preserves the old
        # byte stream when either generation is later edited.
        subprocess.run(["cp", "-c", str(source), str(target)], check=True)
    else:
        shutil.copy2(source, target)
    a, b = source.stat(), target.stat()
    if a.st_dev == b.st_dev and a.st_ino == b.st_ino:
        raise ValueError(f"Reference package shares an input inode: {source}")


def _jsonl(path):
    with gzip.open(path, "rt") as stream:
        for line in stream:
            yield json.loads(line)


def _csv(path):
    with gzip.open(path, "rt", newline="") as stream:
        yield from csv.DictReader(stream)


def _write(path, rows, schema):
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path, compression="zstd")


def _source_receipt(directory):
    receipt = directory / ROOT / "output_hashes.json"
    if file_digest(receipt)["sha256"] != OUTPUT_RECEIPT_SHA256:
        raise ValueError("Unrecognized reference mapping output receipt")
    hashes = json.loads(receipt.read_text())
    for name, expected in hashes.items():
        if file_digest(directory / ROOT / name)["sha256"] != expected:
            raise ValueError(f"Reference mapping source bytes changed: {name}")
    if file_digest(directory / ROOT / "reference.fasta")["sha256"] != REFERENCE_SHA256:
        raise ValueError("Mouse reference FASTA changed")
    if (
        file_digest(directory / ROOT / "independent-verification.json")["sha256"]
        != INDEPENDENT_REPLAY_SHA256
    ):
        raise ValueError("Independent mouse-reference replay receipt changed")
    independent = json.loads((directory / ROOT / "independent-verification.json").read_text())
    if not independent.get("passed") or independent.get("complete_scan_positions") != 52_277:
        raise ValueError("Independent mouse-reference replay did not pass")
    return hashes


def _project_species(directory):
    source = directory / "external/pxd057588/assets/PXD057588/project.json"
    project = json.loads(source.read_text())
    organism = [
        item
        for item in project["sampleAttributes"]
        if item.get("key", {}).get("accession") == "OBI:0100026"
    ]
    if len(organism) != 1 or len(organism[0]["value"]) != 1:
        raise ValueError("PXD057588 project organism claim changed")
    value = organism[0]["value"][0]
    if value.get("accession") != "NEWT:10090":
        raise ValueError("PXD057588 mouse organism claim changed")
    return value["accession"].removeprefix("NEWT:"), value["name"]


def _build_typed(directory):
    raw = directory / ROOT
    typed = directory / TYPED
    taxon, organism = _project_species(directory)
    source_rows = list(_jsonl(raw / "source_mapping.jsonl.gz"))
    source_by_id = {row["peptide_row_id"]: row for row in source_rows}
    if len(source_by_id) != len(source_rows) or len(source_rows) != 35_772:
        raise ValueError("Reference source row identities or count changed")
    original = pq.read_table(
        directory / "external/pxd057588/typed/peptide_rows.parquet",
        columns=["peptide_row_id", "sequence_raw", "parent_label_raw", "table_number"],
    ).to_pylist()
    if len(original) != len(source_rows):
        raise ValueError("Reference source and original peptide row counts differ")
    for row in original:
        mapped = source_by_id.get(row["peptide_row_id"])
        if mapped is None or any(mapped[key] != row[key] for key in row if key != "peptide_row_id"):
            raise ValueError(f"Reference source row differs from original: {row['peptide_row_id']}")
    _write(
        typed / "reference_source_mappings.parquet",
        source_rows,
        SCHEMAS["reference_source_mappings"],
    )

    sequences = list(_csv(raw / "sequences.csv.gz"))
    sequence_by_id = {row["sequence_id"]: row for row in sequences}
    if len(sequence_by_id) != 30_587:
        raise ValueError("Reference sequence count or identities changed")
    for row in sequences:
        for name in (
            "length",
            "reference_identity_count",
            "reference_full_sequence_count",
            "reference_occurrence_count",
        ):
            row[name] = int(row[name])
        if len(row["sequence"]) != row["length"]:
            raise ValueError("Reference sequence length changed")
    for row in source_rows:
        sequence = sequence_by_id.get(row["sequence_id"])
        if sequence is None or (row["sequence_raw"], row["component"]) != (
            sequence["sequence"],
            sequence["component"],
        ):
            raise ValueError("Source row reference sequence link changed")
    _write(typed / "reference_sequences.parquet", sequences, SCHEMAS["reference_sequences"])

    matched = {row["accession"]: row for row in _jsonl(raw / "matched_parents.jsonl.gz")}
    if len(matched) != 10_497:
        raise ValueError("Matched reference parent identities changed")
    parents = []
    for row in _jsonl(raw / "reference_entries.jsonl.gz"):
        if "10090" not in row["taxonomy"]:
            raise ValueError("Non-mouse identity entered the pinned mouse reference")
        match = matched.get(row["accession"])
        if match is not None and (
            match["sequence_sha256"] != row["sequence_sha256"] or match["length"] != row["length"]
        ):
            raise ValueError("Matched parent metadata changed")
        parents.append(
            {
                "accession": row["accession"],
                "entry": row["entry"],
                "length": row["length"],
                "sequence_sha256": row["sequence_sha256"],
                "sequence": match["sequence"] if match else None,
                "taxonomy": row["taxonomy"],
                "aliases": row["aliases"],
                "reference_status": "matched_sequence" if match else "reference_entry_unmatched",
            }
        )
    if len(parents) != 25_758 or len({row["accession"] for row in parents}) != len(parents):
        raise ValueError("Reference parent identity inventory changed")
    _write(typed / "reference_parents.parquet", parents, SCHEMAS["reference_parents"])

    occurrences = []
    for row in _csv(raw / "matches.csv.gz"):
        sequence = sequence_by_id[row["sequence_id"]]
        parent = matched[row["accession"]]
        start, end = int(row["start1"]), int(row["end1"])
        protein = parent["sequence"]
        if (
            protein[start - 1 : end] != sequence["sequence"]
            or row["left"] != (protein[start - 2] if start > 1 else "-")
            or row["right"] != (protein[end] if end < len(protein) else "-")
            or row["parent_sequence_sha256"] != parent["sequence_sha256"]
        ):
            raise ValueError("Exact mouse-reference position or flank changed")
        occurrences.append(
            {
                **row,
                "peptide_sequence": sequence["sequence"],
                "component": sequence["component"],
                "start1": start,
                "end1": end,
            }
        )
    if len(occurrences) != 52_277:
        raise ValueError("Exact reference occurrence count changed")
    if len({(row["sequence_id"], row["accession"], row["start1"]) for row in occurrences}) != len(
        occurrences
    ):
        raise ValueError("Duplicate exact reference occurrence identity")
    _write(typed / "reference_occurrences.parquet", occurrences, SCHEMAS["reference_occurrences"])

    support = []
    for row in _jsonl(raw / "mhc_digest_parent_support.jsonl.gz"):
        if any(source_by_id[key]["table_number"] != 9 for key in row["table9_rows"]):
            raise ValueError("Parent ligand support crossed source tables")
        if any(source_by_id[key]["table_number"] != 11 for key in row["table11_rows"]):
            raise ValueError("Parent digest support crossed source tables")
        support.append(
            {
                **row,
                "n_table9_rows": len(row["table9_rows"]),
                "n_table11_rows": len(row["table11_rows"]),
            }
        )
    if len(support) != 26_449:
        raise ValueError("Reference parent support count changed")
    if len({(row["definition"], row["parent_key"]) for row in support}) != len(support):
        raise ValueError("Duplicate reference parent support identity")
    _write(typed / "reference_parent_support.parquet", support, SCHEMAS["reference_parent_support"])

    roles = sorted(
        {
            (role, row[field])
            for row in source_rows
            for role, field in (
                ("measured_population", "measured_population"),
                ("partner_population", "partner_population"),
            )
            if row[field] is not None
        }
    )
    species = [
        {
            "claim_id": f"PXD057588:organism:{role}:{label}",
            "source_dataset": "PXD057588",
            "population_role": role,
            "population_label": label,
            "organism_taxon_id": taxon,
            "organism_name": organism,
            "source_scope": "project_sample_attribute",
            "source_locator": "PXD057588/project.json:sampleAttributes[OBI:0100026]",
            "role_basis": "project_organism_and_source_table_population_role",
        }
        for role, label in roles
    ]
    _write(typed / "reference_species_roles.parquet", species, SCHEMAS["reference_species_roles"])

    result_path = directory / "external/pxd057588/typed/quantity_results.parquet"
    context_path = typed / "reference_result_context.parquet"
    writer = pq.ParquetWriter(context_path, SCHEMAS["reference_result_context"], compression="zstd")
    n_results = 0
    seen = set()
    try:
        for batch in pq.ParquetFile(result_path).iter_batches(batch_size=16_384):
            output = []
            for row in batch.to_pylist():
                if row["result_id"] in seen:
                    raise ValueError("Duplicate external assay result ID")
                seen.add(row["result_id"])
                source = source_by_id[row["peptide_row_id"]]
                if any(
                    row[key] != source[key]
                    for key in ("table_number", "measured_pool", "material_link_precision")
                ):
                    raise ValueError("Assay result and source peptide context differ")
                output.append(
                    {
                        "result_id": row["result_id"],
                        "peptide_row_id": row["peptide_row_id"],
                        "source_dataset": "PXD057588",
                        "sequence_id": source["sequence_id"],
                        "component": source["component"],
                        "source_parent_status": source["source_parent_status"],
                        "reference_identity_count": source["reference_identity_count"],
                        "reference_full_sequence_count": source["reference_full_sequence_count"],
                        "reference_occurrence_count": source["reference_occurrence_count"],
                        "table_number": row["table_number"],
                        "measured_pool": row["measured_pool"],
                        "measured_population": source["measured_population"],
                        "partner_population": source["partner_population"],
                        "material_link_precision": row["material_link_precision"],
                        "specimen_status": "unresolved",
                        "organism_taxon_id": taxon,
                        "organism_source_scope": "project_sample_attribute",
                    }
                )
            writer.write_table(
                pa.Table.from_pylist(output, schema=SCHEMAS["reference_result_context"])
            )
            n_results += len(output)
    finally:
        writer.close()
    if n_results != 309_126:
        raise ValueError("External result count changed while linking reference context")
    return {
        "typed_rows": {
            **EXPECTED_ROWS,
            "reference_species_roles": len(species),
        },
        "source_parent_status": dict(Counter(row["source_parent_status"] for row in source_rows)),
        "organism_taxon_id": taxon,
        "n_original_external_results": n_results,
        "n_original_external_measurements": pq.ParquetFile(
            directory / "external/pxd057588/typed/quantity_measurements.parquet"
        ).metadata.num_rows,
    }


def _safe_relative(value):
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"Invalid reference package artifact path: {value}")
    return path


def _verify_typed_relations(directory):
    """Recheck source links and exact spans without trusting the new manifest."""
    typed = directory / TYPED
    source_rows = pq.read_table(
        typed / "reference_source_mappings.parquet",
        columns=["peptide_row_id", "sequence_id", "sequence_raw", "component", "table_number"],
    ).to_pylist()
    source = {row["peptide_row_id"]: row for row in source_rows}
    original = pq.read_table(
        directory / "external/pxd057588/typed/peptide_rows.parquet",
        columns=["peptide_row_id", "sequence_raw", "table_number"],
    ).to_pylist()
    if len(source) != len(original):
        raise ValueError("Reference source mapping lost or duplicated a peptide row")
    for row in original:
        mapped = source.get(row["peptide_row_id"])
        if mapped is None or any(mapped[key] != row[key] for key in row if key != "peptide_row_id"):
            raise ValueError("Reference source mapping changed an original peptide row")
    sequences = {
        row["sequence_id"]: row
        for row in pq.read_table(
            typed / "reference_sequences.parquet", columns=["sequence_id", "sequence", "component"]
        ).to_pylist()
    }
    for row in source_rows:
        sequence = sequences.get(row["sequence_id"])
        if sequence is None or (sequence["sequence"], sequence["component"]) != (
            row["sequence_raw"],
            row["component"],
        ):
            raise ValueError("Reference source-to-sequence relation changed")
    parents = {
        row["accession"]: row
        for row in pq.read_table(
            typed / "reference_parents.parquet",
            columns=["accession", "sequence", "sequence_sha256"],
        ).to_pylist()
    }
    for batch in pq.ParquetFile(typed / "reference_occurrences.parquet").iter_batches():
        for row in batch.to_pylist():
            sequence = sequences.get(row["sequence_id"])
            parent = parents.get(row["accession"])
            if sequence is None or parent is None or parent["sequence"] is None:
                raise ValueError("Reference occurrence lost a sequence or parent")
            protein = parent["sequence"]
            start, end = row["start1"], row["end1"]
            if (
                protein[start - 1 : end] != sequence["sequence"]
                or row["peptide_sequence"] != sequence["sequence"]
                or row["component"] != sequence["component"]
                or row["left"] != (protein[start - 2] if start > 1 else "-")
                or row["right"] != (protein[end] if end < len(protein) else "-")
                or row["parent_sequence_sha256"] != parent["sequence_sha256"]
            ):
                raise ValueError("Typed exact reference occurrence changed")
    context = {
        row["result_id"]: row
        for row in pq.read_table(
            typed / "reference_result_context.parquet",
            columns=["result_id", "peptide_row_id", "sequence_id", "component", "specimen_status"],
        ).to_pylist()
    }
    if len(context) != EXPECTED_ROWS["reference_result_context"]:
        raise ValueError("Reference result context key count changed")
    results = pq.read_table(
        directory / "external/pxd057588/typed/quantity_results.parquet",
        columns=["result_id", "peptide_row_id"],
    ).to_pylist()
    for row in results:
        item = context.get(row["result_id"])
        mapped = source.get(row["peptide_row_id"])
        if (
            item is None
            or mapped is None
            or item["peptide_row_id"] != row["peptide_row_id"]
            or item["sequence_id"] != mapped["sequence_id"]
            or item["component"] != mapped["component"]
            or item["specimen_status"] != "unresolved"
        ):
            raise ValueError("Reference result context changed source identity or specimen status")
    for batch in pq.ParquetFile(
        directory / "external/pxd057588/typed/quantity_measurements.parquet"
    ).iter_batches(columns=["result_id"]):
        if any(result_id not in context for result_id in batch[0].to_pylist()):
            raise ValueError("Numeric measurement lost reference result context")


def verify_reference_evidence_package(directory, *, manifest_path="manifest.json"):
    """Verify every byte, predecessor and new typed-grain reconciliation."""
    directory = Path(directory)
    location = _safe_relative(manifest_path)
    manifest = json.loads((directory / location).read_text())
    if (
        manifest.get("schema_version") != PACKAGE_SCHEMA_VERSION
        or manifest.get("catalog_schema_version") != CATALOG_SCHEMA_VERSION
    ):
        raise ValueError("Unsupported reference evidence package schema")
    files = manifest["files"]
    if location == Path("manifest.json"):
        actual = {
            str(path.relative_to(directory))
            for path in directory.rglob("*")
            if path.is_file() and path != directory / "manifest.json"
        }
        if set(files) != actual:
            raise ValueError("Reference package file inventory mismatch")
    for relative, digest in files.items():
        if file_digest(directory / _safe_relative(relative)) != digest:
            raise ValueError(f"Reference package artifact mismatch: {relative}")
    parent_path = "manifests/numeric.json"
    if files[parent_path]["sha256"] != manifest["parent_manifest_sha256"]:
        raise ValueError("Reference predecessor manifest mismatch")
    parent = verify_supporting_numeric_evidence_package(directory, manifest_path=parent_path)
    if parent["catalog_schema_version"] != 7:
        raise ValueError("Unexpected numeric predecessor catalog")
    reference = manifest.get("reference", {})
    if (
        reference.get("presto_commit") != PRESTO_COMMIT
        or reference.get("output_receipt_sha256") != OUTPUT_RECEIPT_SHA256
        or reference.get("reference_fasta_sha256") != REFERENCE_SHA256
        or reference.get("taxonomy_id") != "10090"
        or reference.get("source_rights")
        != {
            "uniprot_reference": "CC BY 4.0",
            "publisher_supplements": "CC BY 4.0",
            "pride_project_metadata": "CC0",
        }
    ):
        raise ValueError("Reference source identity or rights metadata changed")
    _source_receipt(directory)
    reconciliation = manifest["reconciliation"]
    if reconciliation["typed_rows"] != {**EXPECTED_ROWS, "reference_species_roles": 5}:
        raise ValueError("Reference typed-grain reconciliation changed")
    for name, expected in EXPECTED_ROWS.items():
        parquet = pq.ParquetFile(directory / TYPED / f"{name}.parquet")
        if parquet.schema_arrow != SCHEMAS[name] or parquet.metadata.num_rows != expected:
            raise ValueError(f"Reference typed relation mismatch: {name}")
    components = Counter(
        component
        for component in pq.read_table(
            directory / TYPED / "reference_sequences.parquet", columns=["component"]
        )["component"].to_pylist()
        if component
    )
    if len(components) != 5_374 or max(components.values()) != 862:
        raise ValueError("Reference component grouping changed")
    name = "reference_species_roles"
    parquet = pq.ParquetFile(directory / TYPED / f"{name}.parquet")
    if (
        parquet.schema_arrow != SCHEMAS[name]
        or parquet.metadata.num_rows != reconciliation["typed_rows"][name]
        or reconciliation["organism_taxon_id"] != _project_species(directory)[0]
    ):
        raise ValueError("Reference species role claims changed")
    if reconciliation["source_parent_status"] != {
        "exact_sequence_match": 35_747,
        "missing_id": 24,
        "resolved_id_peptide_absent": 1,
    }:
        raise ValueError("Reference source-parent status reconciliation changed")
    if (
        reconciliation["n_original_external_results"] != 309_126
        or reconciliation["n_original_external_measurements"] != 509_850
    ):
        raise ValueError("External assay or numeric source count changed")
    _verify_typed_relations(directory)
    return manifest


def write_reference_evidence_package(numeric_package, results_dir, reference_fasta, destination):
    """Snapshot the numeric generation and append exact mouse-reference evidence."""
    numeric_package, results_dir = Path(numeric_package), Path(results_dir)
    reference_fasta, destination = Path(reference_fasta), Path(destination)
    if destination.exists():
        raise FileExistsError(f"Reference evidence package already exists: {destination}")
    parent = verify_supporting_numeric_evidence_package(numeric_package)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    try:
        for relative in parent["files"]:
            _copy(numeric_package / _safe_relative(relative), staging / relative)
        _copy(numeric_package / "manifest.json", staging / "manifests/numeric.json")
        if file_digest(results_dir / "output_hashes.json")["sha256"] != OUTPUT_RECEIPT_SHA256:
            raise ValueError("Unrecognized shared-reference output receipt")
        hashes = json.loads((results_dir / "output_hashes.json").read_text())
        for name in (*hashes, "output_hashes.json"):
            _copy(results_dir / name, staging / ROOT / name)
        _copy(
            results_dir / "independent-verification.json",
            staging / ROOT / "independent-verification.json",
        )
        _copy(reference_fasta, staging / ROOT / "reference.fasta")
        _source_receipt(staging)
        reconciliation = _build_typed(staging)
        files = {
            str(path.relative_to(staging)): file_digest(path)
            for path in sorted(staging.rglob("*"))
            if path.is_file()
        }
        from .version import __version__

        manifest = {
            "schema_version": PACKAGE_SCHEMA_VERSION,
            "catalog_schema_version": CATALOG_SCHEMA_VERSION,
            "hitlist_version": __version__,
            "parent_manifest_sha256": files["manifests/numeric.json"]["sha256"],
            "reference": {
                "presto_commit": PRESTO_COMMIT,
                "report": PRESTO_REPORT,
                "output_receipt_sha256": OUTPUT_RECEIPT_SHA256,
                "reference_fasta_sha256": REFERENCE_SHA256,
                "taxonomy_id": "10090",
                "release": "UniProt 2026_03 reviewed mouse including isoforms",
                "source_rights": {
                    "uniprot_reference": "CC BY 4.0",
                    "publisher_supplements": "CC BY 4.0",
                    "pride_project_metadata": "CC0",
                },
                "interpretation": "exact candidate mappings; not original search database or specimen pairing",
            },
            "reconciliation": reconciliation,
            "files": files,
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        verify_reference_evidence_package(staging)
        staging.rename(destination)
        return destination / "manifest.json"
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
