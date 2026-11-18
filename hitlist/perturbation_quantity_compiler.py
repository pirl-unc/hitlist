"""Compile pinned reporter workbooks without inferring missing measurements."""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from dataclasses import asdict
from decimal import Decimal, InvalidOperation
from pathlib import Path
from urllib.parse import quote

import pyarrow as pa
import pyarrow.parquet as pq

from .material_context import validate_material_graph
from .nontryptic_workbooks import read_source_workbook
from .perturbation_quantity_schema import RELATIONS, SCHEMAS


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def identity(*parts):
    return ":".join(quote(str(part), safe="") for part in parts)


def numeric(raw, *, signed=False):
    if raw in (None, ""):
        return None, "not_reported"
    try:
        decimal = Decimal(raw)
        value = float(decimal)
    except (InvalidOperation, ValueError, OverflowError):
        return None, "invalid_numeric"
    if (
        not decimal.is_finite()
        or not math.isfinite(value)
        or (decimal != 0 and value == 0)
        or (value < 0 and not signed)
    ):
        return None, "invalid_numeric"
    return value, "reported_zero" if value == 0 else "reported_finite"


def peptide_form(raw):
    """The admitted source uses bracketed phosphorylated residues, not ProForma."""
    tokens = re.findall(r"\[([STY])\]|([ACDEFGHIKLMNPQRSTVWY])", raw)
    reconstructed = "".join(f"[{modified}]" if modified else plain for modified, plain in tokens)
    if not tokens or reconstructed != raw:
        raise ValueError(f"Unsupported source peptide form: {raw!r}")
    sequence = "".join(modified or plain for modified, plain in tokens)
    modifications = [
        {"position": i, "residue": modified, "modification": "phosphorylation"}
        for i, (modified, _) in enumerate(tokens, 1)
        if modified
    ]
    return sequence, modifications


class TableSink:
    """Bound row buffers; avoid retaining all measurement/cell dictionaries."""

    def __init__(self, destination):
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=False)
        self.writers = {
            name: pq.ParquetWriter(destination / f"{name}.parquet", schema)
            for name, schema in SCHEMAS.items()
        }
        self.rows = defaultdict(list)
        self.counts = {f"n_{name}_rows": 0 for name in SCHEMAS}
        self.identities = {name: set() for name in SCHEMAS}

    def add(self, name, row):
        if row.keys() - set(SCHEMAS[name].names):
            raise ValueError(f"Unknown {name} fields")
        primary_key = RELATIONS[name][2]
        key = row.get(primary_key)
        if not key or key in self.identities[name]:
            raise ValueError(f"Missing or duplicate {name} identity: {key}")
        self.identities[name].add(key)
        self.rows[name].append(row)
        self.counts[f"n_{name}_rows"] += 1
        if len(self.rows[name]) >= 1024:
            self.flush(name)

    def flush(self, name):
        if self.rows[name]:
            self.writers[name].write_table(
                pa.Table.from_pylist(self.rows[name], schema=SCHEMAS[name])
            )
            self.rows[name].clear()

    def close(self):
        try:
            for name in SCHEMAS:
                self.flush(name)
        finally:
            for writer in self.writers.values():
                writer.close()


class Compiler:
    def __init__(self, contract, sink):
        self.contract, self.sink = contract, sink
        self.project = identity(contract["source_dataset"], "project")
        self.contexts, self.links = [], []
        self.n_normalized_measurements = 0
        self.n_excluded_aggregate_rows = 0

    def ident(self, kind, *parts):
        return identity(self.contract["source_dataset"], kind, *parts)

    def context(self, context_id, kind, label, locator, condition_id=None):
        self.contexts.append(
            {
                "context_id": context_id,
                "context_kind": kind,
                "sample_label_raw": label,
                "condition_id": condition_id,
                "condition_status": "source_scoped",
                "specimen_id": None,
                "aliquot_id": None,
                "material_link_precision": "project_context"
                if kind == "source_project"
                else "study_arm",
                "context_status": "source_label_not_physical_material_identity",
                "source_locator": locator,
                "notes": "Physical specimen identity and cross-plex reuse are unresolved.",
            }
        )

    def link(self, source, target, predicate, locator):
        self.links.append(
            {
                "claim_id": self.ident("context_link", source, target, predicate),
                "source_context_id": source,
                "target_context_id": target,
                "assertion_predicate": predicate,
                "assertion_status": "source_reported",
                "material_link_precision": "study_arm",
                "source_locator": locator,
            }
        )

    def protocol(self):
        article = self.contract["article_url"]
        self.context(self.project, "source_project", self.contract["source_dataset"], article)
        treatments = {item["treated_condition_id"]: item for item in self.contract["comparisons"]}
        for arm in self.contract["conditions"]:
            key = arm["condition_id"]
            context_id = self.ident("condition", key)
            treatment = treatments.get(key)
            self.context(context_id, "study_arm", arm["condition"], article, key)
            self.link(context_id, self.project, "condition_in_source_project", article)
            self.sink.add(
                "conditions",
                {
                    "condition_id": key,
                    "context_id": context_id,
                    "cell_line_name": arm["sample_group"],
                    "condition_text": arm["condition"],
                    "perturbation_name": treatment["treatment"] if treatment else "DMSO vehicle",
                    "treatment_dose": treatment["dose"] if treatment else None,
                    "dose_unit": treatment["dose_unit"] if treatment else None,
                    "treatment_duration_hours": 72.0,
                    "cellular_mhc_genotype": arm["mhc"],
                    "mhc_genotype_basis": "curated_deposited_sample_genotype_not_peptide_restriction",
                    "source_locator": article + ";pmid_overrides.yaml#32488085/" + key,
                },
            )
        for predicate, value, section in self.contract["protocol_assertions"]:
            self.sink.add(
                "protocol_assertions",
                {
                    "claim_id": self.ident("protocol", predicate),
                    "context_id": self.project,
                    "assertion_predicate": predicate,
                    "assertion_value": value,
                    "assertion_status": "curation_policy"
                    if section in {"source qualification", "curation policy"}
                    else "reviewed_primary_source",
                    "source_scope": "study_protocol_not_individual_PSM_receipt",
                    "source_locator": article + "#" + section,
                },
            )

    @staticmethod
    def value(sheet, coordinate):
        cell = sheet.cells.get(coordinate)
        return cell.value_literal if cell is not None else None

    def row_id(self, filename, sheet, row):
        return self.ident("row", filename, sheet, row)

    def capture(self, asset, sheet, comparison):
        filename = asset["filename"]
        sheet_id = self.ident("sheet", filename, sheet.name)
        locator = filename + "#" + sheet.name
        self.sink.add(
            "source_sheets",
            {
                "sheet_id": sheet_id,
                "source_asset_id": self.ident("asset", filename),
                "sheet_name": sheet.name,
                "sheet_state": sheet.state,
                "sheet_dimension": sheet.dimension,
                "merged_ranges_json": encoded(sheet.merged_ranges),
                "source_locator": locator,
            },
        )
        rows = defaultdict(list)
        for coordinate, cell in sheet.cells.items():
            rows[int(re.search(r"[0-9]+$", coordinate)[0])].append(cell)
        for rownum, cells in sorted(rows.items()):
            sequence = self.value(sheet, f"A{rownum}")
            disposition = "source_metadata"
            if not any(cell.value_present or cell.has_formula for cell in cells):
                disposition = "explicit_blank_row"
            if comparison and rownum > comparison["header_row"] and sequence:
                disposition = "excluded_aggregate" if sequence == "Grand Total" else "peptidoform"
            elif sheet.name == "3.9 SKMEL5_1_AUC" and rownum > 7 and sequence:
                disposition = "pooled_auc_not_condition_specific"
            if disposition == "excluded_aggregate":
                self.n_excluded_aggregate_rows += 1
            self.sink.add(
                "source_records",
                {
                    "source_record_id": self.row_id(filename, sheet.name, rownum),
                    "sheet_id": sheet_id,
                    "source_row_ordinal": rownum,
                    "n_source_cells": len(cells),
                    "source_cells_json": encoded([asdict(cell) for cell in cells]),
                    "row_disposition": disposition,
                    "source_locator": locator + "!" + str(rownum),
                },
            )
        return rows

    def comparison(self, asset, sheet, spec, row_numbers):
        filename = asset["filename"]
        header = spec["header_row"]
        for column, expected in spec["expected_headers"].items():
            if self.value(sheet, f"{column}{header}") != str(expected):
                raise ValueError(f"Qualified source header changed: {sheet.name}!{column}{header}")
        plex = self.ident("plex", filename, sheet.name)
        locator = filename + "#" + sheet.name
        self.sink.add(
            "plexes",
            {
                "plex_id": plex,
                "sheet_id": self.ident("sheet", filename, sheet.name),
                "cell_line_name": spec["cell_line_name"],
                "control_condition_id": spec["control_condition_id"],
                "treated_condition_id": spec["treated_condition_id"],
                "dose_qualification": spec["dose_qualification"],
                "normalization_scope": "source_hipMHC_correction_separate_within_conditions_for_large_shifts",
                "source_dataset": self.contract["source_dataset"],
                "source_reference_id": self.contract["source_reference_id"],
                "source_locator": locator,
            },
        )
        samples = []
        for slot in range(6):
            role = "control" if slot < 3 else "treated"
            channel = f"TMT-{126 + slot}"
            sample_id = self.ident("sample", filename, sheet.name, channel)
            samples.append(sample_id)
            condition = spec[role + "_condition_id"]
            self.context(sample_id, "source_sample", channel, locator, condition)
            self.link(
                sample_id, self.ident("condition", condition), "channel_in_condition", locator
            )
            self.sink.add(
                "samples",
                {
                    "sample_id": sample_id,
                    "plex_id": plex,
                    "condition_id": condition,
                    "context_id": sample_id,
                    "reporter_channel": channel,
                    "arm_role": role,
                    "source_slot_ordinal": slot % 3 + 1,
                    "replicate_type": "reported_biological_unpaired_specimen_identity_unresolved",
                    "specimen_id": None,
                    "aliquot_id": None,
                    "source_locator": locator,
                },
            )
        n_forms, forms_seen = 0, set()
        for rownum in sorted(row_numbers):
            raw = self.value(sheet, f"A{rownum}")
            if rownum <= header or not raw or raw == "Grand Total":
                continue
            sequence, modifications = peptide_form(raw)
            if raw in forms_seen:
                raise ValueError("Repeated molecular form within a source comparison")
            forms_seen.add(raw)
            n_forms += 1
            form_id = self.ident("form", filename, sheet.name, rownum)
            record = self.row_id(filename, sheet.name, rownum)
            length = self.value(sheet, f"{spec['length_column']}{rownum}")
            if length is not None and Decimal(length) != len(sequence):
                raise ValueError(f"Source peptide length disagrees: {locator}!{rownum}")
            self.sink.add(
                "forms",
                {
                    "form_id": form_id,
                    "source_record_id": record,
                    "plex_id": plex,
                    "peptide_sequence": sequence,
                    "modified_sequence_raw": raw,
                    "modifications_json": encoded(modifications),
                    "form_status": "source_phosphorylated"
                    if modifications
                    else "source_unmodified",
                    "protein_label_raw": self.value(sheet, f"B{rownum}"),
                    "gene_symbol_raw": self.value(sheet, f"{spec['gene_column']}{rownum}"),
                    "reported_length_raw": length,
                    "peptide_length": len(sequence),
                },
            )
            for quantity in spec["quantities"]:
                kind = quantity["quantity_kind"]
                for column, slot in zip(quantity["columns"], quantity["slots"], strict=True):
                    coordinate = f"{column}{rownum}"
                    raw_value = self.value(sheet, coordinate)
                    value, status = numeric(raw_value)
                    self.sink.add(
                        "measurements",
                        {
                            "measurement_id": self.ident(
                                "measurement", filename, sheet.name, coordinate
                            ),
                            "form_id": form_id,
                            "sample_id": samples[slot],
                            "plex_id": plex,
                            "source_record_id": record,
                            "quantity_kind": kind,
                            "quantitative_raw": raw_value,
                            "quantitative_value": value,
                            "value_status": status,
                            "measurement_unit": "relative_reporter_intensity",
                            "source_column": column,
                            "source_header_raw": self.value(sheet, f"{column}{header}"),
                            "source_locator": locator + "!" + coordinate,
                        },
                    )
                    if kind == "hipmhc_normalized_reporter":
                        self.n_normalized_measurements += 1
            for statistic, column in spec["statistics"].items():
                coordinate = f"{column}{rownum}"
                raw_value = self.value(sheet, coordinate)
                value, status = numeric(raw_value, signed=statistic == "log2_fold_change")
                self.sink.add(
                    "reported_summaries",
                    {
                        "summary_id": self.ident("summary", filename, sheet.name, coordinate),
                        "form_id": form_id,
                        "plex_id": plex,
                        "source_record_id": record,
                        "statistic_name": statistic,
                        "quantity_kind": "mean_adjusted_reporter"
                        if statistic.startswith("mean_adjusted_")
                        else "hipmhc_normalized_reporter",
                        "quantitative_raw": raw_value,
                        "quantitative_value": value,
                        "value_status": status,
                        "measurement_unit": "relative_reporter_intensity"
                        if statistic.startswith(("mean_", "sd_"))
                        and not statistic.startswith("mean_adjusted_")
                        else "dimensionless",
                        "source_column": column,
                        "source_locator": locator + "!" + coordinate,
                    },
                )
        if n_forms != spec["n_forms"]:
            raise ValueError(f"Source form census changed: {sheet.name}")

    def filemap(self, asset, sheet, spec):
        filename = asset["filename"]
        for block in spec["blocks"]:
            start = block["start_row"]
            if (
                self.value(sheet, f"{spec['line_column']}{start}") != block["line_raw"]
                or self.value(sheet, f"{spec['condition_column']}{start}") != block["condition_raw"]
                or Decimal(self.value(sheet, f"{spec['table_column']}{start}"))
                != Decimal(block["table_raw"])
            ):
                raise ValueError("Qualified file-map block changed")
            target = block["comparison_sheet"]
            for rownum in block["rows"]:
                coordinate = f"{spec['filename_column']}{rownum}"
                raw_file = self.value(sheet, coordinate)
                if not raw_file:
                    raise ValueError("Source file-map entry is missing")
                self.sink.add(
                    "file_links",
                    {
                        "claim_id": self.ident("file_link", filename, sheet.name, rownum),
                        "plex_id": self.ident("plex", filename, target) if target else None,
                        "source_file": raw_file,
                        "source_file_type": self.value(sheet, f"{spec['type_column']}{rownum}"),
                        "source_record_id": self.row_id(filename, sheet.name, rownum),
                        "source_table_label": self.value(sheet, f"{spec['table_column']}{start}"),
                        "filemap_qualification": "pooled_auc_not_condition_specific"
                        if target is None
                        else "line_and_dose_qualified_obsolete_table_number_preserved"
                        if asset["role"] == "data3"
                        else "source_filemap_agrees_with_comparison",
                        "source_locator": filename + "#" + sheet.name + "!" + coordinate,
                    },
                )

    def run(self, source_directory):
        self.protocol()
        for asset in self.contract["assets"]:
            filename = asset["filename"]
            self.sink.add(
                "assets",
                {
                    "source_asset_id": self.ident("asset", filename),
                    "source_file": filename,
                    "source_asset_url": asset["source_url"],
                    "asset_sha256": asset["sha256"],
                    "size_bytes": asset["size_bytes"],
                    "license_status": asset["license_status"],
                },
            )
            comparisons = {
                c["sheet"]: c for c in self.contract["comparisons"] if c["asset"] == filename
            }
            filemaps = {c["sheet"]: c for c in self.contract["filemaps"] if c["asset"] == filename}
            sheets = read_source_workbook(Path(source_directory) / filename)
            if not (comparisons.keys() | filemaps.keys()) <= {s.name for s in sheets}:
                raise ValueError("Qualified source worksheet missing")
            for sheet in sheets:
                rows = self.capture(asset, sheet, comparisons.get(sheet.name))
                if sheet.name in comparisons:
                    self.comparison(asset, sheet, comparisons[sheet.name], rows)
                if sheet.name in filemaps:
                    self.filemap(asset, sheet, filemaps[sheet.name])
            del sheets
        validate_material_graph(self.contexts, self.links)
        for row in self.contexts:
            self.sink.add("material_contexts", row)
        for row in self.links:
            self.sink.add("material_links", row)
        return {
            **self.sink.counts,
            "n_normalized_measurements": self.n_normalized_measurements,
            "n_excluded_aggregate_rows": self.n_excluded_aggregate_rows,
        }


def write_perturbation_tables(source_directory, destination, contract):
    sink = TableSink(destination)
    try:
        return Compiler(contract, sink).run(source_directory)
    finally:
        sink.close()
