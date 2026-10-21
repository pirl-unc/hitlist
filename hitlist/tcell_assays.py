"""Pinned, row-level IEDB/CEDAR T-cell assay metadata.

This intentionally omits antigen identities, sequences, source organisms and
allele names. Reported assay outcomes remain source values, not model labels.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import tempfile
import zipfile
from collections import Counter
from collections.abc import Iterator
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from datacache import install_bundle

from .curation_yaml import load_curation_yaml
from .downloads import data_dir

CATALOG_PATH = Path(__file__).parent / "data" / "tcell_audit_sources.yaml"
IEDB_NAME = "iedb_tcell_full_v3.zip"
CEDAR_NAME = "cedar_tcell_full_v3.zip"
MEMBER = "tcell_full_v3.csv"
HLA_TWO_FIELD = re.compile(r"^HLA-[A-Z0-9]+\*\d{2,3}:\d{2,3}$")

# Exact source cells selected for the metadata export. No epitope or antigen
# identity, organism, sequence, or allele name is projected from the source.
SOURCE_FIELDS = {
    "reference_pmid": "Reference | PMID",
    "assay_method": "Assay | Method",
    "response_measured": "Assay | Response measured",
    "qualitative_measurement": "Assay | Qualitative Measurement",
    "quantitative_measurement": "Assay | Quantitative measurement",
    "measurement_inequality": "Assay | Measurement Inequality",
    "measurement_units": "Assay | Units",
    "reported_n_subjects_tested": "Assay | Number of Subjects Tested",
    "reported_n_subjects_positive": "Assay | Number of Subjects Positive",
    "antigen_object_type": "Assay Antigen | Object Type",
    "host_name": "Host | Name",
    "in_vitro_process_type": "In vitro Process | Process Type",
    "in_vitro_responder_cell": "in vitro Responder Cell | Name",
    "in_vitro_stimulator_cell": "in vitro Stimulator Cell | Name",
    "effector_cell_name": "Effector Cell | Name",
    "effector_culture_condition": "Effector Cell | Culture Condition",
    "apc_name": "Antigen Presenting Cell | Name",
    "apc_culture_condition": "Antigen Presenting Cell | Culture Condition",
    "restriction_evidence_code": "MHC Restriction | Evidence Code",
    "reported_mhc_class": "MHC Restriction | Class",
}
_PARITY_FIELDS = (*SOURCE_FIELDS.values(), "MHC Restriction | Name")
EXPORT_COLUMNS = (
    "source_assay_id",
    "source_reference_id",
    *SOURCE_FIELDS,
    "assay_family",
    "restriction_name_resolution",
    "cedar_present",
    "specimen_identity_status",
    "receptor_identity_status",
)
EXPORT_SCHEMA = pa.schema(
    [(name, pa.bool_() if name == "cedar_present" else pa.string()) for name in EXPORT_COLUMNS]
)


def assay_family(method: str, response: str) -> str:
    """Route method/readout metadata without assigning a training target."""
    method, response = method.casefold(), response.casefold()
    if method == "multimer/tetramer":
        return "pmhc_multimer"
    if response == "cytotoxicity":
        return "cytotoxicity"
    if response in {"activation", "degranulation", "proliferation"}:
        return "activation_or_proliferation"
    if response.endswith(" release"):
        return "soluble_factor_release"
    return "other_tcell_assay"


def tcell_assay_sources(*, data_root=None):
    """Acquire the exact public ZIP bytes counted in the availability audit."""
    catalog = load_curation_yaml(CATALOG_PATH)
    return install_bundle(
        Path(data_root or data_dir()) / "functional-evidence/tcell-availability-2026-10-20",
        catalog["files"],
        download_options={"timeout": 120, "show_progress": False},
    )


def _check_pin(path: Path, spec: dict):
    if path.stat().st_size != spec["size"]:
        raise ValueError(f"Source size differs from pinned bytes: {path.name}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    if digest.hexdigest() != spec["sha256"]:
        raise ValueError(f"Source hash differs from pinned bytes: {path.name}")


def _source_rows(path: Path, *, source: str) -> Iterator[tuple[str, str, dict[str, str]]]:
    """Stream validated two-header rows, projected to the allowlisted cells."""
    with zipfile.ZipFile(path) as archive:
        if archive.namelist() != [MEMBER]:
            raise ValueError(f"Unexpected T-cell archive members: {path.name}")
        with archive.open(MEMBER) as raw:
            reader = csv.reader(io.TextIOWrapper(raw, encoding="utf-8-sig", newline=""))
            try:
                major, minor = next(reader), next(reader)
            except StopIteration as exc:
                raise ValueError(f"Missing T-cell double header: {path.name}") from exc
            headers = [f"{a} | {b}" for a, b in zip(major, minor)]
            if len(major) != len(minor) or len(headers) != len(set(headers)):
                raise ValueError(f"Invalid T-cell double header: {path.name}")
            assay_header = f"Assay ID | {source} IRI"
            reference_header = f"Reference | {source} IRI"
            required = {assay_header, reference_header, *_PARITY_FIELDS}
            missing = required - set(headers)
            if missing:
                raise ValueError(f"Missing T-cell columns: {sorted(missing)}")
            positions = {name: headers.index(name) for name in required}
            seen = set()
            for row_number, row in enumerate(reader, start=1):
                if len(row) != len(headers):
                    raise ValueError(f"Malformed T-cell row {row_number}: {path.name}")
                assay_iri = row[positions[assay_header]].strip()
                reference_iri = row[positions[reference_header]].strip()
                if not assay_iri or not reference_iri:
                    raise ValueError(f"Missing assay or reference ID on row {row_number}")
                assay_id = assay_iri.rsplit("/", 1)[-1]
                reference_id = reference_iri.rsplit("/", 1)[-1]
                if assay_id in seen:
                    raise ValueError(f"Repeated assay ID: {assay_id}")
                seen.add(assay_id)
                cells = {name: row[position].strip() for name, position in positions.items()}
                yield assay_id, reference_id, cells


def _parity_digest(cells: dict[str, str]) -> bytes:
    values = [cells[name] for name in _PARITY_FIELDS]
    return hashlib.blake2b(
        json.dumps(values, ensure_ascii=False, separators=(",", ":")).encode(), digest_size=16
    ).digest()


def iter_tcell_assay_metadata(
    *, iedb_path=None, cedar_path=None, data_root=None, catalog_path=None
) -> Iterator[dict]:
    """Yield one row per pinned IEDB assay, with checked CEDAR membership.

    Supply both local ZIP paths to avoid a download; otherwise Hitlist fetches
    the pinned public files. Both hashes are checked before any row is yielded.
    The CEDAR projection must match IEDB for every shared assay ID.
    """
    if (iedb_path is None) != (cedar_path is None):
        raise ValueError("Provide both iedb_path and cedar_path, or neither")
    catalog = load_curation_yaml(catalog_path or CATALOG_PATH)
    paths = (
        {IEDB_NAME: Path(iedb_path), CEDAR_NAME: Path(cedar_path)}
        if iedb_path is not None
        else tcell_assay_sources(data_root=data_root)
    )
    for name in (IEDB_NAME, CEDAR_NAME):
        _check_pin(Path(paths[name]), catalog["files"][name])
    cedar = {
        assay_id: (reference_id, _parity_digest(cells))
        for assay_id, reference_id, cells in _source_rows(Path(paths[CEDAR_NAME]), source="CEDAR")
    }
    unmatched = set(cedar)
    for assay_id, reference_id, cells in _source_rows(Path(paths[IEDB_NAME]), source="IEDB"):
        in_cedar = assay_id in cedar
        if in_cedar:
            if cedar[assay_id] != (reference_id, _parity_digest(cells)):
                raise ValueError(f"IEDB/CEDAR assay metadata differs: {assay_id}")
            unmatched.remove(assay_id)
        restriction_name = cells["MHC Restriction | Name"]
        resolution = (
            "missing"
            if not restriction_name
            else "single_hla_two_field_name"
            if HLA_TWO_FIELD.fullmatch(restriction_name)
            else "other_reported_name"
        )
        row = {
            "source_assay_id": assay_id,
            "source_reference_id": reference_id,
            **{output: cells[source] for output, source in SOURCE_FIELDS.items()},
            "assay_family": assay_family(
                cells["Assay | Method"], cells["Assay | Response measured"]
            ),
            "restriction_name_resolution": resolution,
            "cedar_present": in_cedar,
            "specimen_identity_status": "not_in_export",
            "receptor_identity_status": "not_in_export",
        }
        yield row
    if unmatched:
        raise ValueError(f"CEDAR assay IDs absent from IEDB: {len(unmatched)}")


def write_tcell_assay_metadata(
    output_dir,
    *,
    iedb_path=None,
    cedar_path=None,
    data_root=None,
    catalog_path=None,
    batch_size=25000,
) -> dict:
    """Write a local, versioned Parquet table and provenance JSON atomically."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    catalog = load_curation_yaml(catalog_path or CATALOG_PATH)
    with tempfile.TemporaryDirectory(prefix=f".{destination.name}-", dir=destination.parent) as tmp:
        staging = Path(tmp)
        table_path = staging / "assays.parquet"
        counts = Counter()
        writer = None
        batch = []
        try:
            for row in iter_tcell_assay_metadata(
                iedb_path=iedb_path,
                cedar_path=cedar_path,
                data_root=data_root,
                catalog_path=catalog_path,
            ):
                batch.append(row)
                counts["n_assays"] += 1
                counts["n_cedar_assays"] += int(row["cedar_present"])
                if len(batch) >= batch_size:
                    writer = _write_batch(writer, table_path, batch)
                    batch.clear()
            if batch or writer is None:
                writer = _write_batch(writer, table_path, batch)
        finally:
            if writer is not None:
                writer.close()
        digest = hashlib.sha256()
        with table_path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1 << 20), b""):
                digest.update(chunk)
        manifest = {
            "schema_version": 1,
            "source_observed": catalog["observed"],
            "sources": catalog["files"],
            "n_assays": counts["n_assays"],
            "n_cedar_assays": counts["n_cedar_assays"],
            "parquet_sha256": digest.hexdigest(),
            "content_policy": "assay_metadata_without_antigen_or_allele_identity",
            "label_policy": "reported_assay_outcome_not_recast_as_binding_or_ms",
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        os.replace(staging, destination)
    return manifest


def _write_batch(writer, path: Path, rows: list[dict]):
    if writer is None:
        writer = pq.ParquetWriter(path, EXPORT_SCHEMA, compression="zstd")
    writer.write_table(pa.Table.from_pylist(rows, schema=EXPORT_SCHEMA))
    return writer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--iedb", type=Path)
    parser.add_argument("--cedar", type=Path)
    args = parser.parse_args()
    manifest = write_tcell_assay_metadata(
        args.output_dir, iedb_path=args.iedb, cedar_path=args.cedar
    )
    print(
        json.dumps({key: manifest[key] for key in ("n_assays", "n_cedar_assays")}, sort_keys=True)
    )


if __name__ == "__main__":
    main()
