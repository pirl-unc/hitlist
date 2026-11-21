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
SOURCE_VERIFICATION_MODES = frozenset({"pinned", "reconcile"})
_RECONCILIATION_SAMPLE_LIMIT = 100

_BIOPHYSICS_METHODS = frozenset(
    {
        "surface plasmon resonance",
        "surface plasmon resonance (spr)",
        "spr",
        "biolayer interferometry",
        "bio-layer interferometry",
        "bli",
        "x-ray crystallography",
        "3d structure",
        "isothermal titration calorimetry",
        "itc",
        "microscale thermophoresis",
        "mst",
        "nuclear magnetic resonance",
        "nmr",
        "cryo-em",
        "cryo-electron microscopy",
        "analytical ultracentrifugation",
    }
)
_BIOPHYSICS_RESPONSES = frozenset(
    {
        "3d structure",
        "dissociation constant kd",
        "association constant ka",
        "off rate",
        "on rate",
        "kd",
        "ka",
        "koff",
        "kon",
        "half life",
        "dissociation constant",
        "association constant",
        "binding constant",
        "enthalpy",
        "entropy",
        "gibbs free energy",
    }
)

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


def assay_family(method: str, response: str, *, separate_biophysics: bool = True) -> str:
    """Route method/readout metadata without assigning a training target.

    When ``separate_biophysics=True`` (the default), cell-free recombinant
    TCR-pMHC biophysical binding assays (SPR, BLI, 3D structure / X-ray
    crystallography, ITC, KD, kon/koff) route to ``"tcr_pmhc_biophysics"``
    instead of ``"other_tcell_assay"`` (#947).
    """
    method_cf, response_cf = method.strip().casefold(), response.strip().casefold()
    if separate_biophysics and (
        method_cf in _BIOPHYSICS_METHODS
        or any(
            token in method_cf
            for token in (
                "surface plasmon resonance",
                "biolayer interferometry",
                "bio-layer interferometry",
                "x-ray crystallography",
                "isothermal titration calorimetry",
                "microscale thermophoresis",
            )
        )
    ):
        return "tcr_pmhc_biophysics"
    if method_cf == "multimer/tetramer":
        return "pmhc_multimer"
    if separate_biophysics and response_cf in _BIOPHYSICS_RESPONSES:
        return "tcr_pmhc_biophysics"
    if response_cf == "cytotoxicity":
        return "cytotoxicity"
    if response_cf in {"activation", "degranulation", "proliferation"}:
        return "activation_or_proliferation"
    if response_cf.endswith(" release"):
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


def _measure_source(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return {"sha256": digest.hexdigest(), "size": path.stat().st_size}


def _check_pin(path: Path, spec: dict):
    if path.stat().st_size != spec["size"]:
        raise ValueError(f"Source size differs from pinned bytes: {path.name}")
    measured = _measure_source(path)
    if measured["sha256"] != spec["sha256"]:
        raise ValueError(f"Source hash differs from pinned bytes: {path.name}")
    return measured


def _source_rows(
    path: Path,
    *,
    source: str,
    extra_fields: tuple[str, ...] = (),
    optional_fields: tuple[str, ...] = (),
) -> Iterator[tuple[str, str, dict[str, str]]]:
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
            required = {assay_header, reference_header, *_PARITY_FIELDS, *extra_fields}
            missing = required - set(headers)
            if missing:
                raise ValueError(f"Missing T-cell columns: {sorted(missing)}")
            header_index = {name: idx for idx, name in enumerate(headers)}
            positions = {name: header_index[name] for name in required}
            opt_positions = {
                name: header_index[name] for name in optional_fields if name in header_index
            }
            opt_missing = [name for name in optional_fields if name not in header_index]
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
                for name, position in opt_positions.items():
                    cells[name] = row[position].strip()
                for name in opt_missing:
                    cells[name] = ""
                yield assay_id, reference_id, cells


def _parity_digest(cells: dict[str, str], extra_fields: tuple[str, ...] = ()) -> bytes:
    values = [cells[name] for name in (*_PARITY_FIELDS, *extra_fields)]
    return hashlib.blake2b(
        json.dumps(values, ensure_ascii=False, separators=(",", ":")).encode(), digest_size=16
    ).digest()


def iter_tcell_assay_metadata(
    *,
    iedb_path=None,
    cedar_path=None,
    data_root=None,
    catalog_path=None,
    source_verification_mode: str = "pinned",
) -> Iterator[dict]:
    """Yield one row per IEDB assay, with checked or reconciled CEDAR membership.

    Supply both local ZIP paths to avoid a download; otherwise Hitlist fetches
    the pinned public files. In ``source_verification_mode="pinned"`` (default),
    both hashes are checked against the catalog before any row is yielded and
    shared assay IDs must agree byte-for-byte. In ``"reconcile"`` mode (#947),
    live refreshed ZIPs and asynchronous IEDB/CEDAR parity diffs are recorded
    as structured provenance rather than aborting iteration.
    """
    for row, _ in _iter_assays_with_cells(
        iedb_path=iedb_path,
        cedar_path=cedar_path,
        data_root=data_root,
        catalog_path=catalog_path,
        source_verification_mode=source_verification_mode,
    ):
        yield row


def _iter_assays_with_cells(
    *,
    iedb_path=None,
    cedar_path=None,
    data_root=None,
    catalog_path=None,
    extra_fields: tuple[str, ...] = (),
    optional_fields: tuple[str, ...] = (),
    cedar_extra_fields: tuple[str, ...] = (),
    source_verification_mode: str = "pinned",
    reconciliation_out: dict | None = None,
) -> Iterator[tuple[dict, dict[str, str]]]:
    """Share pinned or reconciled source validation with row projections."""
    if (iedb_path is None) != (cedar_path is None):
        raise ValueError("Provide both iedb_path and cedar_path, or neither")
    if source_verification_mode not in SOURCE_VERIFICATION_MODES:
        raise ValueError(
            f"source_verification_mode must be one of {sorted(SOURCE_VERIFICATION_MODES)}"
        )
    catalog = load_curation_yaml(catalog_path or CATALOG_PATH)
    paths = (
        {IEDB_NAME: Path(iedb_path), CEDAR_NAME: Path(cedar_path)}
        if iedb_path is not None
        else tcell_assay_sources(data_root=data_root)
    )
    actual_sources: dict[str, dict] = {}
    pin_status: dict[str, str] = {}
    for name in (IEDB_NAME, CEDAR_NAME):
        path = Path(paths[name])
        spec = catalog["files"][name]
        if source_verification_mode == "pinned":
            measured = _check_pin(path, spec)
            pin_status[name] = "matched"
        else:
            measured = _measure_source(path)
            matched = measured["size"] == spec["size"] and measured["sha256"] == spec["sha256"]
            pin_status[name] = "matched" if matched else "drifted"
        actual_sources[name] = {
            "url": spec.get("url", ""),
            "sha256": measured["sha256"],
            "size": measured["size"],
        }

    cedar: dict[str, tuple[str, bytes, dict[str, str]]] = {}
    for assay_id, reference_id, cells in _source_rows(
        Path(paths[CEDAR_NAME]),
        source="CEDAR",
        extra_fields=extra_fields,
        optional_fields=cedar_extra_fields,
    ):
        extras = {field: cells.get(field, "") for field in cedar_extra_fields}
        cedar[assay_id] = (reference_id, _parity_digest(cells, extra_fields), extras)

    unmatched = set(cedar)
    n_parity_diffs = 0
    parity_diff_sample: list[str] = []
    for assay_id, reference_id, cells in _source_rows(
        Path(paths[IEDB_NAME]),
        source="IEDB",
        extra_fields=extra_fields,
        optional_fields=optional_fields,
    ):
        in_cedar = assay_id in cedar
        if in_cedar:
            cedar_ref, cedar_digest, cedar_extras = cedar[assay_id]
            if (cedar_ref, cedar_digest) != (reference_id, _parity_digest(cells, extra_fields)):
                if source_verification_mode == "pinned":
                    raise ValueError(f"IEDB/CEDAR assay metadata differs: {assay_id}")
                n_parity_diffs += 1
                if len(parity_diff_sample) < _RECONCILIATION_SAMPLE_LIMIT:
                    parity_diff_sample.append(assay_id)
            unmatched.remove(assay_id)
            for field, val in cedar_extras.items():
                cells[f"CEDAR::{field}"] = val
        else:
            for field in cedar_extra_fields:
                cells[f"CEDAR::{field}"] = ""
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
        yield row, cells
    if unmatched and source_verification_mode == "pinned":
        raise ValueError(f"CEDAR assay IDs absent from IEDB: {len(unmatched)}")
    if reconciliation_out is not None:
        reconciliation_out.clear()
        reconciliation_out.update(
            {
                "source_verification_mode": source_verification_mode,
                "pin_status": pin_status,
                "actual_sources": actual_sources,
                "pinned_sources": catalog["files"],
                "n_parity_diffs": n_parity_diffs,
                "parity_diff_assay_ids_sample": parity_diff_sample,
                "n_cedar_only_assays": len(unmatched),
                "cedar_only_assay_ids_sample": sorted(unmatched)[:_RECONCILIATION_SAMPLE_LIMIT],
            }
        )


def write_tcell_assay_metadata(
    output_dir,
    *,
    iedb_path=None,
    cedar_path=None,
    data_root=None,
    catalog_path=None,
    batch_size=25000,
    source_verification_mode: str = "pinned",
) -> dict:
    """Write a local, versioned Parquet table and provenance JSON atomically."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    catalog = load_curation_yaml(catalog_path or CATALOG_PATH)
    reconciliation: dict = {}
    with tempfile.TemporaryDirectory(prefix=f".{destination.name}-", dir=destination.parent) as tmp:
        staging = Path(tmp)
        table_path = staging / "assays.parquet"
        counts = Counter()
        writer = None
        batch = []
        try:
            for row, _ in _iter_assays_with_cells(
                iedb_path=iedb_path,
                cedar_path=cedar_path,
                data_root=data_root,
                catalog_path=catalog_path,
                source_verification_mode=source_verification_mode,
                reconciliation_out=reconciliation,
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
            "sources": (
                reconciliation["actual_sources"]
                if source_verification_mode == "reconcile"
                else catalog["files"]
            ),
            "source_verification_mode": source_verification_mode,
            "n_assays": counts["n_assays"],
            "n_cedar_assays": counts["n_cedar_assays"],
            "parquet_sha256": digest.hexdigest(),
            "content_policy": "assay_metadata_without_antigen_or_allele_identity",
            "label_policy": "reported_assay_outcome_not_recast_as_binding_or_ms",
        }
        if source_verification_mode == "reconcile":
            manifest["source_reconciliation"] = reconciliation
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
    parser.add_argument(
        "--verification-mode",
        choices=sorted(SOURCE_VERIFICATION_MODES),
        default="pinned",
    )
    args = parser.parse_args()
    manifest = write_tcell_assay_metadata(
        args.output_dir,
        iedb_path=args.iedb,
        cedar_path=args.cedar,
        source_verification_mode=args.verification_mode,
    )
    print(
        json.dumps({key: manifest[key] for key in ("n_assays", "n_cedar_assays")}, sort_keys=True)
    )


if __name__ == "__main__":
    main()
