"""Source-faithful Goldberg2025 MAPP evidence, without training admission.

Raw intensity is a search-result quantity, not intracellular molarity. Original
and publisher views have distinct identities; sample labels never create physical
specimen/aliquot links. The pooled publisher table is not a supervised cohort.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import re
import shutil
import tempfile
import zipfile
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq

from .evidence_snapshot import snapshot_file
from .mapp_evidence import _schema, _Sink
from .material_context import MATERIAL_CONTEXT_SCHEMA, MATERIAL_LINK_SCHEMA, validate_material_graph
from .provenance import file_digest

PACKAGE_SCHEMA_VERSION = 25
CATALOG_SCHEMA_VERSION = 31
PACKAGE_FAMILY = "goldberg_mapp"
REVISION = "goldberg2025_mapp_original_v1"
ROOT = Path("goldberg_mapp")
CONTRACT = Path(__file__).parent / "data/evidence_revisions/goldberg_mapp.json"
ASSET_RECEIPT = Path(__file__).parent / "data/evidence_revisions/goldberg_mapp_assets.json"
SUPPORT_FILES = {"contract.json": CONTRACT}

SCHEMAS = {
    "assets": _schema(
        ("source_asset_id", "source_file", "source_asset_url", "source_asset_role", "asset_sha256"),
        integers=("size_bytes",),
    ),
    "source_records": _schema(
        ("source_record_id", "source_file", "source_locator", "source_row_json"),
        integers=("source_row_index",),
    ),
    "acquisitions": _schema(
        (
            "acquisition_id",
            "context_id",
            "source_record_id",
            "raw_file",
            "assay_name_raw",
            "sample_label_raw",
            "cell_line_name",
            "measured_pool",
            "condition_status",
            "specimen_id",
            "aliquot_id",
            "material_link_precision",
            "component_state_status",
        ),
    ),
    "groups": _schema(
        (
            "group_id",
            "source_record_id",
            "native_id",
            "parent_label_raw",
            "fasta_headers_raw",
            "source_peptide_ids_raw",
            "source_evidence_ids_raw",
            "reverse_raw",
            "contaminant_raw",
        ),
    ),
    "references": _schema(
        (
            "reference_record_id",
            "source_record_id",
            "accession_raw",
            "header_raw",
            "organism_taxon_id",
            "taxonomy_status",
            "reference_sequence",
            "reference_status",
        ),
        integers=("source_slot_index",),
    ),
    "peptides": _schema(
        (
            "peptide_id",
            "source_record_id",
            "native_id",
            "peptide_sequence",
            "parent_label_raw",
            "source_group_ids_raw",
            "source_evidence_ids_raw",
            "source_form_ids_raw",
            "start_raw",
            "end_raw",
            "flank_n_raw",
            "flank_c_raw",
            "reverse_raw",
            "contaminant_raw",
            "source_organism_status",
            "confidence_raw",
            "confidence_status",
        ),
        numbers=("posterior_error_probability",),
    ),
    "forms": _schema(
        (
            "form_id",
            "source_record_id",
            "native_id",
            "peptide_id",
            "peptide_sequence",
            "parent_label_raw",
            "peptide_modifications_raw",
            "source_group_ids_raw",
            "source_evidence_ids_raw",
            "reverse_raw",
            "contaminant_raw",
            "confidence_raw",
        ),
    ),
    "identifications": _schema(
        (
            "identification_id",
            "source_record_id",
            "native_id",
            "peptide_id",
            "form_id",
            "acquisition_id",
            "experiment_raw",
            "peptide_sequence",
            "parent_label_raw",
            "peptide_modifications_raw",
            "modified_sequence_raw",
            "identification_type",
            "source_group_ids_raw",
            "reverse_raw",
            "contaminant_raw",
            "confidence_raw",
            "confidence_status",
        ),
        numbers=("posterior_error_probability",),
    ),
    "quantities": _schema(
        (
            "measurement_id",
            "source_record_id",
            "observation_id",
            "peptide_id",
            "acquisition_id",
            "source_field",
            "source_cell",
            "quantity_kind",
            "quantitative_raw",
            "value_status",
            "measurement_unit",
            "quantity_scale",
            "identification_type",
            "experiment_raw",
            "acquisition_link_status",
        ),
        numbers=("quantitative_value",),
    ),
    "assertions": _schema(
        (
            "claim_id",
            "source_record_id",
            "subject_id",
            "source_scope",
            "assertion_predicate",
            "assertion_value",
            "assertion_status",
            "source_locator",
        ),
    ),
    "publisher_cells": _schema(
        (
            "cell_id",
            "source_record_id",
            "source_file",
            "sheet_name",
            "source_cell",
            "cell_type_raw",
            "cell_style_raw",
            "cell_value_raw",
            "cell_value_resolved",
            "cell_formula_raw",
        ),
        integers=("source_row_index", "source_column_index"),
    ),
    "publisher_quantities": _schema(
        (
            "measurement_id",
            "source_record_id",
            "source_file",
            "sheet_name",
            "source_cell",
            "source_field",
            "peptide_sequence",
            "parent_label_raw",
            "native_id",
            "measured_pool",
            "quantity_kind",
            "quantitative_raw",
            "value_status",
            "quantity_scale",
            "measurement_unit",
        ),
        numbers=("quantitative_value",),
    ),
    "publisher_links": _schema(
        (
            "claim_id",
            "source_record_id",
            "source_measurement_id",
            "publisher_measurement_id",
            "peptide_id",
            "link_status",
            "assertion_predicate",
            "source_locator",
        ),
    ),
    "pooled_slots": _schema(
        (
            "slot_id",
            "source_record_id",
            "source_cell",
            "slot_raw",
            "sample_label_raw",
            "quantitative_raw",
            "value_status",
            "quantity_scale",
            "measurement_unit",
            "identity_status",
            "acquisition_id",
            "specimen_id",
            "aliquot_id",
        ),
        integers=("source_slot_index",),
        numbers=("quantitative_value",),
    ),
    "material_contexts": MATERIAL_CONTEXT_SCHEMA,
    "material_links": MATERIAL_LINK_SCHEMA,
}


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _id(kind, *values):
    return "goldberg2025:" + kind + ":" + hashlib.sha256(_json(values).encode()).hexdigest()


def _number(value):
    if value is None or value == "":
        return None, "missing"
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None, "invalid_literal"
    if not math.isfinite(result):
        return None, "nonfinite"
    return result, "positive" if result > 0 else "zero" if result == 0 else "negative"


def _confidence(value):
    result, status = _number(value)
    if result is not None:
        status = "reported_probability" if 0 <= result <= 1 else "outside_probability_range"
    return result, status


def _tsv(handle):
    """Retain ordered columns; reject ambiguous field lookup, not duplicate data."""
    previous = csv.field_size_limit()
    csv.field_size_limit(max(previous, 16 * 1024**2))
    try:
        reader = csv.reader(handle, delimiter="\t")
        headers = next(reader)
        for number, values in enumerate(reader, 2):
            if len(values) != len(headers):
                raise ValueError(f"Source TSV row width differs at row {number}")
            yield number, headers, values
    finally:
        csv.field_size_limit(previous)


def _fields(headers, values):
    if len(set(headers)) != len(headers):
        raise ValueError("Duplicate source headers require positional interpretation")
    return dict(zip(headers, values, strict=True))


class _Compiler:
    def __init__(self, assets, destination, contract):
        self.assets, self.contract = assets, contract
        destination.mkdir(parents=True)
        self.sinks = {name: _Sink(destination, name, schema=s) for name, s in SCHEMAS.items()}
        self.acquisitions = {}
        self.experiments = {}
        self.peptides = {}
        self.forms = {}
        self.peptide_quantities = {}
        self.taxa = defaultdict(set)
        self.contexts, self.links = [], []

    def record(self, source_file, locator, row_index, headers, values):
        key = _id("row", source_file, locator)
        self.sinks["source_records"].add(
            {
                "source_record_id": key,
                "source_file": source_file,
                "source_locator": locator,
                "source_row_index": row_index,
                "source_row_json": _json({"headers": headers, "values": values}),
            }
        )
        return key

    def assertion(self, record, subject, scope, predicate, value, locator, status="reported"):
        self.sinks["assertions"].add(
            {
                "claim_id": _id("assertion", record, subject, predicate, locator),
                "source_record_id": record,
                "subject_id": subject,
                "source_scope": scope,
                "assertion_predicate": predicate,
                "assertion_value": value,
                "assertion_status": status,
                "source_locator": locator,
            }
        )

    def original_rows(self, archive, name):
        with archive.open(name) as handle:
            for index, headers, values in _tsv(io.TextIOWrapper(handle, encoding="utf-8-sig")):
                record = self.record(
                    self.contract["archive"], f"{name}:row:{index}", index, headers, values
                )
                yield record, _fields(headers, values)

    def context(self, key, kind, label, locator):
        self.contexts.append(
            {
                "context_id": key,
                "context_kind": kind,
                "sample_label_raw": label,
                "condition_id": None,
                "condition_text": label,
                "condition_status": "source_label_only",
                "specimen_id": None,
                "aliquot_id": None,
                "material_link_precision": {
                    "source_project": "project_context",
                    "study_arm": "study_arm",
                    "source_acquisition": "source_acquisition",
                }[kind],
                "context_status": "reported_context_not_physical_sample_identity",
                "source_locator": locator,
                "notes": "No specimen or aliquot identity inferred",
            }
        )

    def link(self, source, target, precision, locator):
        self.links.append(
            {
                "claim_id": _id("material_link", source, target),
                "source_context_id": source,
                "target_context_id": target,
                "assertion_predicate": "source_context_membership",
                "assertion_status": "reported",
                "material_link_precision": precision,
                "source_locator": locator,
                "notes": "Not an exact specimen/aliquot pairing",
            }
        )

    def deposition(self):
        from .goldberg_workbooks import workbook_rows

        project = _id("project", "PXD052314")
        self.context(project, "source_project", "PXD052314", self.contract["deposition"])
        arms = set()
        for sheet, index, cells in workbook_rows(self.assets / self.contract["deposition"]):
            if index == 1:
                headers = [c["cell_value_resolved"] for c in cells]
                continue
            values = [c["cell_value_resolved"] for c in cells]
            # The deposition table has a fixed five-column rectangle; omitted XML
            # cells must not shift its native field positions.
            values = [
                next((c["cell_value_resolved"] for c in cells if c["source_column_index"] == i), "")
                for i in range(1, len(headers) + 1)
            ]
            row = _fields(headers, values)
            raw = row["Raw File"]
            if not raw:
                continue
            locator = f"{sheet}!row:{index}"
            record = self.record(self.contract["deposition"], locator, index, headers, values)
            if raw in self.acquisitions:
                raise ValueError("Duplicate deposition raw-file identity")
            acquisition = _id("acquisition", raw)
            arm = _id("arm", row["Experiment"], row["Sample Name"])
            if arm not in arms:
                self.context(
                    arm, "study_arm", row["Experiment"] + ":" + row["Sample Name"], locator
                )
                self.link(arm, project, "project_context", locator)
                arms.add(arm)
            self.context(acquisition, "source_acquisition", raw, locator)
            self.link(acquisition, arm, "study_arm", locator)
            baseline = bool(re.fullmatch(r"Ctrl4?_[123]", row["Sample Name"]))
            data = {
                "acquisition_id": acquisition,
                "context_id": acquisition,
                "source_record_id": record,
                "raw_file": raw,
                "assay_name_raw": row["Experiment"],
                "sample_label_raw": row["Sample Name"],
                "cell_line_name": row["Cell Type"],
                "measured_pool": self.contract["assay_pools"].get(row["Experiment"], "unresolved"),
                "condition_status": "source_control_label" if baseline else "other_source_label",
                "specimen_id": None,
                "aliquot_id": None,
                "material_link_precision": "source_acquisition",
                "component_state_status": "unreported_not_assumed_wildtype",
            }
            self.acquisitions[raw] = data
            self.sinks["acquisitions"].add(data)
            for field, value in row.items():
                self.assertion(
                    record, acquisition, "original_deposition_acquisition", field, value, locator
                )

    def metadata(self):
        for name in ("project_metadata", "file_metadata"):
            filename = self.contract[name]
            data = json.loads((self.assets / filename).read_text())
            self.record(filename, "json_root", 1, ["json"], [data])
        filename = self.contract["sdrf"]
        with (self.assets / filename).open() as handle:
            for index, headers, values in _tsv(handle):
                row = _fields(headers, values)
                locator = f"row:{index}"
                record = self.record(filename, locator, index, headers, values)
                raw = row["comment[data file]"].removesuffix(".raw")
                subject = self.acquisitions.get(raw, {}).get("acquisition_id", record)
                status = (
                    "secondary_assertion_exact_run_match"
                    if raw in self.acquisitions
                    else "unmatched_secondary_raw_name"
                )
                for field, value in zip(headers, values, strict=True):
                    self.assertion(record, subject, "community_sdrf", field, value, locator, status)

    def summary(self, archive):
        for record, row in self.original_rows(archive, "parameters.txt"):
            self.assertion(
                record,
                _id("project", "PXD052314"),
                "original_search_parameters",
                row["Parameter"],
                row["Value"],
                "parameters.txt",
            )
        for record, row in self.original_rows(archive, "summary.txt"):
            if not row["Experiment"]:
                continue  # Retained aggregate row, not another acquisition.
            raw = row["Raw file"]
            if raw not in self.acquisitions:
                raise ValueError("Original summary acquisition absent from deposition")
            if row["Experiment"] in self.experiments:
                raise ValueError("Experiment label does not identify exactly one acquisition")
            self.experiments[row["Experiment"]] = self.acquisitions[raw]["acquisition_id"]
            for field, value in row.items():
                self.assertion(
                    record,
                    self.experiments[row["Experiment"]],
                    "original_search_run",
                    field,
                    value,
                    "summary.txt",
                )

    def groups(self, archive):
        seen = set()
        for record, row in self.original_rows(archive, "proteinGroups.txt"):
            if row["id"] in seen:
                raise ValueError("Duplicate original protein group ID")
            seen.add(row["id"])
            self.sinks["groups"].add(
                {
                    "group_id": _id("group", row["id"]),
                    "source_record_id": record,
                    "native_id": row["id"],
                    "parent_label_raw": row["Protein IDs"],
                    "fasta_headers_raw": row["Fasta headers"],
                    "source_peptide_ids_raw": row["Peptide IDs"],
                    "source_evidence_ids_raw": row["Evidence IDs"],
                    "reverse_raw": row["Reverse"],
                    "contaminant_raw": row["Potential contaminant"],
                }
            )
            for slot, header in enumerate(row["Fasta headers"].split(";"), 1):
                parts = header.split("|")
                accession = parts[1] if len(parts) >= 3 else None
                taxa = re.findall(r"\bOX=(\d+)(?=\s)", header)
                taxon = taxa[0] if len(taxa) == 1 else None
                if accession and taxon:
                    self.taxa[accession].add(taxon)
                self.sinks["references"].add(
                    {
                        "reference_record_id": _id("header", record, slot),
                        "source_record_id": record,
                        "source_slot_index": slot,
                        "accession_raw": accession,
                        "header_raw": header,
                        "organism_taxon_id": taxon,
                        "taxonomy_status": "source_header_assertion"
                        if taxon
                        else "missing_or_truncated_or_ambiguous",
                        "reference_sequence": None,
                        "reference_status": "full_search_reference_not_deposited",
                    }
                )

    def organism(self, parents):
        taxa = set()
        unresolved = False
        for accession in parents.split(";"):
            known = self.taxa.get(accession, set())
            unresolved |= not known
            taxa.update(known)
        if unresolved or not taxa:
            return "mixed_or_unresolved"
        if taxa == {"9606"}:
            return "source_human_only"
        return "mixed_or_unresolved" if "9606" in taxa else "source_nonhuman_only"

    def quantity(
        self,
        record,
        observation,
        peptide,
        field,
        raw,
        kind,
        experiment,
        identification,
        acquisition=None,
    ):
        value, status = _number(raw)
        if acquisition is None and experiment:
            acquisition = self.experiments.get(experiment)
        key = _id("quantity", record, field)
        self.sinks["quantities"].add(
            {
                "measurement_id": key,
                "source_record_id": record,
                "observation_id": observation,
                "peptide_id": peptide,
                "acquisition_id": acquisition,
                "source_field": field,
                "source_cell": field,
                "quantity_kind": kind,
                "quantitative_raw": raw,
                "quantitative_value": value,
                "value_status": status,
                "measurement_unit": "search_intensity",
                "quantity_scale": "linear_source_reported",
                "identification_type": identification,
                "experiment_raw": experiment,
                "acquisition_link_status": "exact_source_experiment_or_raw_file"
                if acquisition
                else "aggregate_or_unresolved_no_acquisition_inferred",
            }
        )
        return key, value, status

    def original_entities(self, archive, member, relation, kind):
        seen = set()
        for record, row in self.original_rows(archive, member):
            native = row["id"]
            if native in seen:
                raise ValueError(f"Duplicate original {kind} ID")
            seen.add(native)
            key = _id(kind, native)
            peptide = key if kind == "peptide" else _id("peptide", row["Peptide ID"])
            if kind != "peptide" and row["Peptide ID"] not in self.peptides:
                raise ValueError("Source peptide ID does not resolve")
            if (
                kind != "peptide"
                and self.peptides[row["Peptide ID"]]["peptide_sequence"] != row["Sequence"]
            ):
                raise ValueError("Linked source peptide sequence differs")
            confidence, confidence_status = _confidence(row.get("PEP", ""))
            data = {
                f"{kind}_id": key,
                "native_id": native,
                "source_record_id": record,
                "peptide_sequence": row["Sequence"],
                "parent_label_raw": row["Proteins"],
                "source_group_ids_raw": row["Protein group IDs"],
                "reverse_raw": row["Reverse"],
                "contaminant_raw": row["Potential contaminant"],
                "confidence_raw": row.get("PEP", ""),
            }
            if kind == "peptide":
                data.update(
                    source_evidence_ids_raw=row["Evidence IDs"],
                    source_form_ids_raw=row["Mod. peptide IDs"],
                    start_raw=row["Start position"],
                    end_raw=row["End position"],
                    flank_n_raw=row["Amino acid before"],
                    flank_c_raw=row["Amino acid after"],
                    source_organism_status=self.organism(row["Proteins"]),
                    posterior_error_probability=confidence,
                    confidence_status=confidence_status,
                )
                self.peptides[native] = data
            elif kind == "form":
                self.forms[native] = row["Peptide ID"]
                data.update(
                    peptide_id=peptide,
                    peptide_modifications_raw=row["Modifications"],
                    source_evidence_ids_raw=row["Evidence IDs"],
                )
            else:
                if self.forms.get(row["Mod. peptide ID"]) != row["Peptide ID"]:
                    raise ValueError("Source evidence form does not resolve to its peptide")
                acquisition = self.acquisitions.get(row["Raw file"], {}).get("acquisition_id")
                if not acquisition:
                    raise ValueError("Evidence raw-file identity does not resolve")
                data.update(
                    peptide_id=peptide,
                    form_id=_id("form", row["Mod. peptide ID"]),
                    acquisition_id=acquisition,
                    experiment_raw=row["Experiment"],
                    peptide_modifications_raw=row["Modifications"],
                    modified_sequence_raw=row["Modified sequence"],
                    identification_type=row["Type"],
                    posterior_error_probability=confidence,
                    confidence_status=confidence_status,
                )
            self.sinks[relation].add(data)
            for field, raw in row.items():
                if field == "Intensity" or field.startswith("Intensity "):
                    experiment = field.removeprefix("Intensity ") if field != "Intensity" else ""
                    identification = row.get(
                        "Identification type " + experiment, row.get("Type", "")
                    )
                    result = self.quantity(
                        record,
                        key,
                        peptide,
                        field,
                        raw,
                        kind + "_intensity",
                        experiment,
                        identification,
                        data.get("acquisition_id"),
                    )
                    if kind == "peptide":
                        self.peptide_quantities[(native, field)] = result

    def run(self):
        from .goldberg_workbooks import compile_publisher

        try:
            for asset in self.contract["assets"]:
                self.sinks["assets"].add(
                    {
                        "source_asset_id": _id("asset", asset["filename"]),
                        "source_file": asset["filename"],
                        "source_asset_url": asset["source_url"],
                        "source_asset_role": asset["role"],
                        "asset_sha256": asset["sha256"],
                        "size_bytes": asset["size_bytes"],
                    }
                )
            self.deposition()
            self.metadata()
            with zipfile.ZipFile(self.assets / self.contract["archive"]) as archive:
                self.summary(archive)
                self.groups(archive)
                self.original_entities(archive, "peptides.txt", "peptides", "peptide")
                self.original_entities(archive, "modificationSpecificPeptides.txt", "forms", "form")
                self.original_entities(archive, "evidence.txt", "identifications", "identification")
            compile_publisher(self)
            validate_material_graph(self.contexts, self.links)
            for name, rows in (
                ("material_contexts", self.contexts),
                ("material_links", self.links),
            ):
                for row in rows:
                    self.sinks[name].add(row)
            return {"n_" + name + "_rows": sink.n_rows for name, sink in self.sinks.items()}
        finally:
            for sink in self.sinks.values():
                sink.close()


def _compile(assets, destination, contract):
    return _Compiler(assets, destination, contract).run()


def _verify_typed_tables(assets, typed, contract, reconciliation):
    """Replay direct typed relations from frozen source assets.

    Both the standalone source receipt and the external-source adapter use this
    verifier. Keeping the replay independent of either package envelope means
    neither form can hide a changed typed value behind a refreshed inventory.
    """
    with tempfile.TemporaryDirectory(prefix="hitlist-goldberg-verify-") as temporary:
        replay = Path(temporary) / "typed"
        counts = _compile(assets, replay, contract)
        if counts != reconciliation:
            raise ValueError("Goldberg source reconciliation differs")
        for name, schema in SCHEMAS.items():
            expected = pq.ParquetFile(replay / f"{name}.parquet")
            actual = pq.ParquetFile(Path(typed) / f"{name}.parquet")
            if not actual.schema_arrow.equals(schema):
                raise ValueError(f"Goldberg schema differs: {name}")
            for left, right in zip(expected.iter_batches(), actual.iter_batches(), strict=True):
                if not left.equals(right):
                    raise ValueError(f"Goldberg typed rows differ from original: {name}")


def _adapter_contract(source_dir):
    """Read a generic external-source envelope without weakening its contract."""
    source_dir = Path(source_dir)
    contract = json.loads((source_dir / "contracts/contract.json").read_text())
    receipt = json.loads((source_dir / "contracts/source_assets.json").read_text())
    expected = {
        asset["filename"]: {"sha256": asset["sha256"], "size": asset["size_bytes"]}
        for asset in contract["assets"]
    }
    if receipt.get("assets") != expected:
        raise ValueError("Goldberg adapter receipt disagrees with source contract")
    return contract


def _inventory(directory):
    return {
        str(p.relative_to(directory)): file_digest(p)
        for p in sorted(directory.rglob("*"))
        if p.is_file() and p != directory / "manifest.json"
    }


def verify_goldberg_mapp_package(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if (
        manifest.get("schema_version"),
        manifest.get("catalog_schema_version"),
        manifest.get("package_family"),
        manifest.get("revision"),
    ) != (PACKAGE_SCHEMA_VERSION, CATALOG_SCHEMA_VERSION, PACKAGE_FAMILY, REVISION):
        raise ValueError("Unsupported Goldberg MAPP package")
    if _inventory(directory) != manifest.get("files"):
        raise ValueError("Goldberg inventory or hashes differ")
    contract = json.loads(CONTRACT.read_text())
    if json.loads((directory / ROOT / "contract.json").read_text()) != contract:
        raise ValueError("Goldberg source contract differs")
    if manifest.get("source_registry") != contract["source_registry"]:
        raise ValueError("Goldberg source registry differs")
    for asset in contract["assets"]:
        if file_digest(directory / ROOT / "source_assets" / asset["filename"]) != {
            k: asset[k] for k in ("sha256", "size_bytes")
        }:
            raise ValueError("Goldberg original source mismatch")
    _verify_typed_tables(
        directory / ROOT / "source_assets",
        directory / ROOT / "typed",
        contract,
        manifest.get("reconciliation"),
    )
    return manifest


def write_goldberg_mapp_package(source_directory, destination):
    """Snapshot pinned originals and materialize evidence; do not select a cohort."""
    source_directory, destination = Path(source_directory), Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    contract = json.loads(CONTRACT.read_text())
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    try:
        for asset in contract["assets"]:
            source = source_directory / asset["filename"]
            if file_digest(source) != {k: asset[k] for k in ("sha256", "size_bytes")}:
                raise ValueError(f"Goldberg source mismatch: {asset['filename']}")
            snapshot_file(source, staging / ROOT / "source_assets" / asset["filename"])
        (staging / ROOT / "contract.json").write_text(_json(contract) + "\n")
        counts = _compile(staging / ROOT / "source_assets", staging / ROOT / "typed", contract)
        from .version import __version__

        manifest = {
            "schema_version": PACKAGE_SCHEMA_VERSION,
            "catalog_schema_version": CATALOG_SCHEMA_VERSION,
            "package_family": PACKAGE_FAMILY,
            "revision": REVISION,
            "hitlist_version": __version__,
            "source_registry": contract["source_registry"],
            "reconciliation": counts,
            "files": _inventory(staging),
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        verify_goldberg_mapp_package(staging)
        staging.rename(destination)
        return destination / "manifest.json"
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


class GoldbergMAPPAdapter:
    """Expose Goldberg source preservation through the shared adapter protocol.

    This adapter does not replace the standalone ``goldberg_mapp`` package. It
    writes the same source-faithful relations beneath an external successor so
    the canonical MAPP projector can consume them alongside other sources.
    """

    source_id = "goldberg2025"
    schema_version = 1
    asset_receipt_path = ASSET_RECEIPT
    support_files = SUPPORT_FILES

    def write(self, source_dir):
        source_dir = Path(source_dir)
        contract = _adapter_contract(source_dir)
        return _compile(source_dir / "assets", source_dir / "typed", contract)

    def verify(self, source_dir, reconciliation):
        source_dir = Path(source_dir)
        _verify_typed_tables(
            source_dir / "assets",
            source_dir / "typed",
            _adapter_contract(source_dir),
            reconciliation,
        )

    def registry(self, reconciliation):
        contract = json.loads(CONTRACT.read_text())
        registry = contract["source_registry"][self.source_id]
        return {
            "source_id": self.source_id,
            "dataset_accession": registry["project_accession"],
            "organism_scope": "A549 source cell-line label; source reference origin unresolved",
            "measured_pools": list(contract["assay_pools"].values()),
            "measurement_grain": "original_MaxQuant_quantity_or_publisher_cell",
            "training_admission": registry["training_status"],
            "quantity_artifact": "typed/quantities.parquet",
            "material_context_artifact": "typed/material_contexts.parquet",
            "material_link_artifact": "typed/material_links.parquet",
            "rights": {"source": registry["license_status"]},
            "n_primary_peptide_rows": reconciliation["n_peptides_rows"],
            "limitations": [
                "captured_pool_not_free_concentration",
                "source_sample_labels_not_physical_specimen_identity",
                "unresolved_specimens_and_aliquots",
                "publisher_views_are_distinct_assays",
                "pooled_slots_have_unresolved_run_and_quantity_semantics",
                "full_search_FASTA_absent",
            ],
        }
