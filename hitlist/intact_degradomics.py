"""PXD058121 original mass signals, candidate assignments and qualified context.

No source row is automatically a supervised concentration or detection label.
Raw observations, repeated views and assignment alternatives remain distinct.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

from .evidence_snapshot import snapshot_file
from .intact_degradomics_schema import SCHEMAS
from .intact_degradomics_workbook import workbook_rows
from .mapp_evidence import _Sink
from .material_context import validate_material_graph
from .provenance import file_digest

PACKAGE_SCHEMA_VERSION = 26
CATALOG_SCHEMA_VERSION = 32
PACKAGE_FAMILY = "intact_degradomics"
REVISION = "pxd058121_intact_degradomics_v1"
ROOT = Path("intact_degradomics")
CONTRACT = Path(__file__).parent / "data/evidence_revisions/intact_degradomics.json"


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _id(kind, *values):
    return "pxd058121:" + kind + ":" + hashlib.sha256(_json(values).encode()).hexdigest()


def _number(raw):
    if raw is None or raw == "":
        return None, "missing"
    try:
        value = float(raw)
    except (ValueError, TypeError):
        return None, "invalid_literal"
    if not math.isfinite(value):
        return None, "nonfinite"
    return value, "positive" if value > 0 else "zero" if value == 0 else "negative"


def _integer(raw):
    value, _ = _number(raw)
    return int(value) if value is not None and value.is_integer() else None


def _native_id(raw):
    if raw is not None and re.fullmatch(r"[0-9]+", raw):
        return str(int(raw))
    return raw


def _sample_tail(label):
    """Explicit source-label grammar, not inferred biological sample identity."""
    if label is None:
        return None
    match = re.fullmatch(r"(Brain|Liver|Spleen)(?:_20S_aSyn)?_([0-9]+min)_([123])", label)
    if match:
        return f"AlphaSyn_{match[1]}20S_{match[2]}_R{match[3]}"
    match = re.fullmatch(r"([0-9]+min)_Rep([123])", label)
    if match:
        return f"AlphaSyn_20S_{match[1]}_R{match[2]}"
    for prefix, replacement in (
        ("20S_AlphaSyn_", "AlphaSyn_20S_"),
        ("aSyn+20S_", "AlphaSyn_20S_"),
        ("aSyn_", "AlphaSyn_"),
    ):
        if label.startswith(prefix):
            label = replacement + label[len(prefix) :]
            break
    return re.sub(r"_([123])$", r"_R\1", label)


def _run_properties(tail):
    match = re.fullmatch(
        r"(?:(AlphaSyn)_)?(?:(Brain|Liver|Spleen)?(20S)_)?([0-9]+)min_R([123])", tail
    )
    if not match:
        return {"measured_pool": "ancillary_acquisition_unresolved"}
    substrate, organ, enzyme, minutes, rep = match.groups()
    pool = (
        "purified_20S_substrate_and_products"
        if substrate and enzyme
        else ("substrate_only_control" if substrate else "proteasome_only_control")
    )
    return {
        "sample_label_raw": tail,
        "organ_label": organ or ("Liver" if enzyme else None),
        "time_minutes": float(minutes),
        "replicate_label_raw": rep,
        "replicate_status": "source_label_biological_technical_independence_unresolved",
        "measured_pool": pool,
        "substrate_species": "Homo sapiens" if substrate else None,
        "enzyme_species": "Mus musculus" if enzyme else None,
    }


class _Compiler:
    def __init__(self, assets, destination, contract):
        self.assets, self.contract = assets, contract
        destination.mkdir(parents=True)
        self.sinks = {k: _Sink(destination, k, schema=s) for k, s in SCHEMAS.items()}
        self.contexts, self.material_links = [], []
        self.raw_tails, self.acquisitions = {}, {}
        self.views, self.inputs, self.assignments, self.quantities = {}, {}, [], {}
        self.publisher = []
        self.n_nonempty_cells = 0

    def record(self, filename, locator, values, *, sheet=None, row=None):
        key = _id("source", filename, locator)
        self.sinks["source_records"].add(
            {
                "source_record_id": key,
                "source_file": filename,
                "sheet_name": sheet,
                "source_locator": locator,
                "source_row_index": row,
                "source_row_json": _json(values),
            }
        )
        return key

    def claim(self, source_record, subject, predicate, value, status, locator, scope="source"):
        self.sinks["assertions"].add(
            {
                "claim_id": _id("assertion", subject, predicate, locator),
                "source_record_id": source_record,
                "subject_id": subject,
                "source_scope": scope,
                "assertion_predicate": predicate,
                "assertion_value": value,
                "assertion_status": status,
                "source_locator": locator,
            }
        )

    def reconcile(self, record, subject, target, predicate, status, locator, details):
        self.sinks["reconciliations"].add(
            {
                "claim_id": _id("reconcile", subject, target, predicate),
                "source_record_id": record,
                "source_subject_id": subject,
                "target_subject_id": target,
                "assertion_predicate": predicate,
                "assertion_status": status,
                "source_locator": locator,
                "details_json": _json(details),
            }
        )

    def context(self, key, kind, label, locator):
        precision = "project_context" if kind == "source_project" else "source_acquisition"
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
                "material_link_precision": precision,
                "context_status": "reported_not_inferred_physical_identity",
                "source_locator": locator,
                "notes": "Organ sources reported from one animal; no replicate-to-specimen inference",
            }
        )

    def metadata(self):
        project = self.contract["project_metadata"]
        record = self.record(project, "project", json.loads((self.assets / project).read_text()))
        self.project_context = _id("context", "project")
        self.context(self.project_context, "source_project", "PXD058121", project)
        for claim in self.contract["assertions"]:
            self.claim(
                record,
                self.project_context,
                claim["predicate"],
                claim["value"],
                claim["status"],
                claim["locator"],
                "pinned_source_interpretation",
            )
        for filename in self.contract["file_metadata"]:
            for i, item in enumerate(json.loads((self.assets / filename).read_text())):
                record = self.record(filename, f"item:{i}", item)
                if item.get("fileCategory", {}).get("value") != "RAW":
                    continue
                raw = item["fileName"]
                tail = re.sub(r"^(?:(?:F|SF)[0-9]+_)+", "", raw).removesuffix(".RAW")
                if tail in self.raw_tails:
                    raise ValueError("Raw suffix is not unique; source mapping needs qualification")
                acquisition = _id("acquisition", raw)
                context = _id("context", raw)
                self.context(context, "source_acquisition", raw, f"{filename}:item:{i}")
                self.material_links.append(
                    {
                        "claim_id": _id("material_link", context),
                        "source_context_id": context,
                        "target_context_id": self.project_context,
                        "assertion_predicate": "source_project_membership",
                        "assertion_status": "reported",
                        "material_link_precision": "project_context",
                        "source_locator": f"{filename}:item:{i}",
                        "notes": "File identity does not establish a reaction, animal or aliquot ID",
                    }
                )
                row = {
                    "acquisition_id": acquisition,
                    "source_record_id": record,
                    "context_id": context,
                    "raw_file": raw,
                    "source_locator": f"{filename}:item:{i}",
                    "source_scope": "original_file_inventory_and_literal_label",
                    "material_link_precision": "source_acquisition",
                    "specimen_id": None,
                    "aliquot_id": None,
                    **_run_properties(tail),
                }
                self.acquisitions[acquisition] = row
                self.raw_tails[tail] = acquisition
                self.sinks["acquisitions"].add(row)

    def acquisition(self, label):
        return self.raw_tails.get(_sample_tail(label))

    def quantity(self, record, view, observation, cell, field, kind, acquisition, sequence=None):
        value, status = _number(cell.get("cell_value_resolved"))
        if cell.get("cell_type_raw") in {"e", "b", "d"}:
            value, status = None, "source_error_or_nonquantity_type"
        if cell.get("formula_attributes_json") is not None:
            status = (
                "cached_formula_value" if value is not None else "formula_without_numeric_cache"
            )
        key = _id("quantity", record, cell["source_cell"], kind)
        info = self.views[view]
        row = {
            "measurement_id": key,
            "source_record_id": record,
            "view_id": view,
            "observation_id": observation,
            "acquisition_id": acquisition,
            "source_file": info["source_file"],
            "sheet_name": info["sheet_name"],
            "source_cell": cell["source_cell"],
            "source_field": field,
            "quantity_kind": kind,
            "quantitative_raw": cell.get("cell_value_resolved"),
            "quantitative_value": value,
            "value_status": status,
            "quantity_scale": "source_linear_signal_not_molar",
            "measurement_unit": "source_arbitrary_intensity",
            "peptide_sequence": sequence,
            "measured_pool": self.acquisitions.get(acquisition, {}).get(
                "measured_pool", "unresolved"
            ),
            "source_view_role": info["source_view_role"],
        }
        self.quantities[key] = row
        self.sinks["quantities"].add(row)
        return key

    @staticmethod
    def cell(cells, column, row):
        if column in cells:
            return cells[column]
        # Missing source field is explicit, not a zero or fabricated XML cell.
        n, letters = column, ""
        while n:
            n, digit = divmod(n - 1, 26)
            letters = chr(65 + digit) + letters
        return {"source_cell": letters + str(row), "cell_value_resolved": None}

    def search_row(self, record, view, row_index, cells, headers):
        info = self.views[view]
        acquisition = info["acquisition_id"]
        missing_ids = info["layout_status"] == "input_ids_absent_masses_A_B_intensity_C"
        shifted = info["layout_status"] == "extra_H_identifier_assignments_shift_after_G"

        def value(col):
            return self.cell(cells, col, row_index).get("cell_value_resolved")

        if value(1) is not None:
            identity = _id("input_mass", record)
            native = None if missing_ids else value(1)
            shift = -1 if missing_ids else 0
            measurement = self.quantity(
                record,
                view,
                identity,
                self.cell(cells, 4 + shift, row_index),
                "Sum Intensity",
                "input_mass_sum_intensity",
                acquisition,
            )
            input_row = {
                "input_mass_id": identity,
                "source_record_id": record,
                "view_id": view,
                "acquisition_id": acquisition,
                "native_mass_id_raw": native,
                "native_mass_id_normalized": _native_id(native),
                "input_identity_status": "absent_not_inferred" if missing_ids else "reported",
                "measurement_id": measurement,
                "mass_y_raw": value(2 + shift),
                "mass_adjusted_raw": value(3 + shift),
                "mass_y": _number(value(2 + shift))[0],
                "mass_adjusted": _number(value(3 + shift))[0],
            }
            self.inputs[identity] = input_row
            self.sinks["input_masses"].add(input_row)
        columns = {
            label: column + int(shifted and column >= 8)
            for column, label in headers.items()
            if label
        }
        if "Sequence" not in columns or value(columns["Sequence"]) is None:
            return

        def get(name):
            return value(columns[name]) if name in columns else None

        identity = _id("assignment", record)
        sequence = get("Sequence")
        start, end = _integer(get("Start AA")), _integer(get("End AA"))
        parent = self.contract["parent_sequence"]
        valid_span = start is not None and end is not None and 1 <= start <= end <= len(parent)
        span_status = (
            "matches_source_substrate"
            if valid_span and parent[start - 1 : end] == sequence
            else "conflicting"
        )
        measurement = self.quantity(
            record,
            view,
            identity,
            self.cell(cells, columns["Intensity"], row_index),
            "Intensity",
            "assignment_intensity",
            acquisition,
            sequence,
        )
        native = get("Mass ID")
        count = _integer(get("MultiMass Assignment") or get("MultiMass Ass"))
        self.assignments.append(
            {
                "assignment_id": identity,
                "source_record_id": record,
                "view_id": view,
                "acquisition_id": acquisition,
                "native_mass_id_raw": native,
                "native_mass_id_normalized": _native_id(native),
                "peptide_sequence": sequence,
                "parent_accession": self.contract["parent_accession"],
                "parent_sequence": parent,
                "fragment_type": get("Frag Type"),
                "observed_mass_raw": get("Observed Mass"),
                "theoretical_mass_raw": get("Theoretical Mass"),
                "observed_mass": _number(get("Observed Mass"))[0],
                "theoretical_mass": _number(get("Theoretical Mass"))[0],
                "mass_error_ppm": _number(get("Error"))[0],
                "start_1based": start,
                "end_1based": end,
                "modifications_json": _json(
                    {k: get(k) for k in ("Localized Mod", "Unlocalized Mod", "Term Mod")}
                ),
                "measurement_id": measurement,
                "n_assignments_reported": count,
                "assignment_status": "unique_mass_match"
                if count == 1
                else "multiple_or_unresolved",
                "span_status": span_status,
                "confidence_status": "source_mass_match_not_general_MS_MS_confirmation",
                "source_extra_identifier": value(8) if shifted else None,
            }
        )

    def supplement_row(self, record, view, row_index, cells, headers):
        info = self.views[view]
        values = {
            label: self.cell(cells, column, row_index).get("cell_value_resolved")
            for column, label in headers.items()
            if label
        }
        if not any(x is not None for x in values.values()) or info["sheet_name"] in {
            "Content",
            "Sheet1",
        }:
            return
        identity = _id("supplement", record)
        label = values.get("Sample ID")
        if label is None:
            match = re.search(r"_Rep([123])$", info["sheet_name"])
            label = "60min_Rep" + match[1] if match else None
        acquisition = self.acquisition(label)
        sequence = values.get("Sequence")
        row = {
            "observation_id": identity,
            "source_record_id": record,
            "view_id": view,
            "acquisition_id": acquisition,
            "sample_label_raw": label,
            "source_view_role": info["source_view_role"],
            "peptide_sequence": sequence,
            "fragment_type": values.get("Frag Type"),
            "observed_mass_raw": values.get("Observed Mass"),
            "theoretical_mass_raw": values.get("Theoretical Mass"),
            "observed_mass": _number(values.get("Observed Mass"))[0],
            "theoretical_mass": _number(values.get("Theoretical Mass"))[0],
            "start_1based": _integer(values.get("Start AA")),
            "end_1based": _integer(values.get("End AA")),
            "charge_state": _number(values.get("Charge State"))[0],
            "fit_percent": _number(values.get("Fit %"))[0],
            "mass_error_ppm": _number(values.get("Error"))[0],
            "row_interpretation_json": _json(values),
        }
        self.sinks["supplement_rows"].add(row)
        measurements = []
        for col, field in headers.items():
            if not field or "intensity" not in field.lower():
                continue
            linked_acquisition = acquisition
            if field.startswith("Intensity_Rep"):
                linked_acquisition = self.acquisition(info["sheet_name"] + "_R" + field[-1])
            kind = (
                "charge_normalized_intensity"
                if "Charge Normalized" in field
                else "supplement_intensity"
            )
            measurements.append(
                self.quantity(
                    record,
                    view,
                    identity,
                    self.cell(cells, col, row_index),
                    field,
                    kind,
                    linked_acquisition,
                    sequence,
                )
            )
        if info["source_view_role"] in {"kinetics_publisher", "organs_publisher"}:
            self.publisher.append((row, measurements))

    def workbooks(self):
        books = {**self.contract["search_workbooks"], **self.contract["supplement_workbooks"]}
        for filename, role in books.items():
            headers = {}
            for sheet, row_index, original_cells in workbook_rows(self.assets / filename):
                record = self.record(
                    filename, f"{sheet}:row:{row_index}", original_cells, sheet=sheet, row=row_index
                )
                view = _id("view", filename, sheet)
                if row_index == 1:
                    headers[sheet] = {
                        c["source_column_index"]: c["cell_value_resolved"] for c in original_cells
                    }
                    layout = self.contract["layout_exceptions"].get(
                        filename + ":" + sheet, "original_header"
                    )
                    info = {
                        "view_id": view,
                        "source_record_id": record,
                        "source_file": filename,
                        "sheet_name": sheet,
                        "acquisition_id": self.acquisition(sheet),
                        "source_view_role": role,
                        "layout_status": layout,
                        "source_locator": f"{filename}:{sheet}",
                        "interpretation_json": _json(
                            {"original_headers": headers[sheet], "layout": layout}
                        ),
                    }
                    self.views[view] = info
                    self.sinks["source_views"].add(info)
                if view not in self.views:
                    raise ValueError("Worksheet does not have a source header row")
                for cell in original_cells:
                    self.n_nonempty_cells += (
                        cell["cell_value_resolved"] is not None
                        or cell["formula_attributes_json"] is not None
                    )
                    self.sinks["source_cells"].add(
                        {
                            "cell_id": _id("cell", record, cell["source_cell"]),
                            "source_record_id": record,
                            "source_file": filename,
                            "sheet_name": sheet,
                            **cell,
                        }
                    )
                if row_index == 1:
                    continue
                cells = {c["source_column_index"]: c for c in original_cells}
                if filename in self.contract["search_workbooks"]:
                    self.search_row(record, view, row_index, cells, headers[sheet])
                else:
                    self.supplement_row(record, view, row_index, cells, headers[sheet])

    def assignment_links(self):
        inputs = defaultdict(list)
        for row in self.inputs.values():
            if row["native_mass_id_normalized"] is not None:
                inputs[row["view_id"], row["native_mass_id_normalized"]].append(row)
        multiplicity = Counter(
            (r["view_id"], r["native_mass_id_normalized"]) for r in self.assignments
        )
        for row in self.assignments:
            key = row["view_id"], row["native_mass_id_normalized"]
            row["n_assignments_replayed"] = multiplicity[key] if key[1] is not None else None
            matches = inputs.get(key, [])
            row["input_mass_id"] = matches[0]["input_mass_id"] if len(matches) == 1 else None
            if (
                self.views[row["view_id"]]["layout_status"]
                == "input_ids_absent_masses_A_B_intensity_C"
            ):
                status = "input_ids_not_supplied_no_inferred_link"
            elif not matches:
                status = "input_id_not_found"
            elif len(matches) > 1:
                status = "ambiguous_input_id"
            else:
                left = self.quantities[row["measurement_id"]]
                right = self.quantities[matches[0]["measurement_id"]]
                status = (
                    "input_id_and_intensity_agree"
                    if (left["quantitative_value"], left["value_status"])
                    == (right["quantitative_value"], right["value_status"])
                    else "input_intensity_conflict"
                )
            row["mass_link_status"] = status
            self.sinks["assignments"].add(row)
            if row["n_assignments_reported"] != row["n_assignments_replayed"]:
                self.claim(
                    row["source_record_id"],
                    row["assignment_id"],
                    "assignment_count_replay",
                    _json(
                        {
                            "reported": row["n_assignments_reported"],
                            "replayed": row["n_assignments_replayed"],
                        }
                    ),
                    "conflicting",
                    self.views[row["view_id"]]["source_locator"],
                )

    def publisher_links(self):
        index = defaultdict(list)
        for assignment in self.assignments:
            info = self.views[assignment["view_id"]]
            key = (
                info["source_view_role"],
                assignment["acquisition_id"],
                assignment["peptide_sequence"],
                assignment["start_1based"],
                assignment["end_1based"],
                assignment["fragment_type"],
            )
            index[key].append(assignment)
        for row, measurements in self.publisher:
            role = {"kinetics_publisher": "kinetics", "organs_publisher": "organs"}[
                row["source_view_role"]
            ]
            fragment = {
                "N-term": "B Fragment",
                "C-term": "Y Fragment",
                "Internal": "BY Int Fragment",
            }.get(row["fragment_type"], row["fragment_type"])
            candidates = index.get(
                (
                    role,
                    row["acquisition_id"],
                    row["peptide_sequence"],
                    row["start_1based"],
                    row["end_1based"],
                    fragment,
                ),
                [],
            )
            # An unresolved acquisition must not match unrelated unresolved rows.
            if row["acquisition_id"] is None:
                candidates = []
            structural_candidates = candidates
            mass_matches = [a for a in candidates if a["observed_mass"] == row["observed_mass"]]
            if len(candidates) > 1 and len(mass_matches) == 1:
                candidates = mass_matches
            status, target = "unresolved_identity", None
            details = {
                "structural_candidates": [a["assignment_id"] for a in structural_candidates],
                "resolved_candidates": [a["assignment_id"] for a in candidates],
                "source_measurement_ids": measurements,
                "intensity_used_to_choose_identity": False,
            }
            if len(candidates) == 1:
                original = candidates[0]
                target = original["assignment_id"]
                differences = [
                    k
                    for k in ("observed_mass", "theoretical_mass", "mass_error_ppm")
                    if row[k] != original[k]
                ]
                original_quantity = self.quantities[original["measurement_id"]]
                equal = len(measurements) == 1 and all(
                    self.quantities[measurements[0]][k] == original_quantity[k]
                    for k in ("quantitative_value", "value_status")
                )
                if not equal:
                    differences.append("intensity")
                details.update(
                    mismatched_fields=differences,
                    intensity_equal=equal,
                    target_measurement_id=original["measurement_id"],
                )
                status = (
                    "identity_and_values_agree"
                    if not differences
                    else "identity_matches_values_conflict"
                )
            self.reconcile(
                row["source_record_id"],
                row["observation_id"],
                target,
                "publisher_assignment_reconciliation",
                status,
                self.views[row["view_id"]]["source_locator"],
                details,
            )

    def repeated_views(self):
        inputs_by_view, assignments_by_view = defaultdict(list), defaultdict(list)
        for row in self.inputs.values():
            inputs_by_view[row["view_id"]].append(
                [row[k] for k in ("native_mass_id_normalized", "mass_y", "mass_adjusted")]
                + [self.quantities[row["measurement_id"]]["quantitative_value"]]
            )
        for row in self.assignments:
            assignments_by_view[row["view_id"]].append(
                [
                    row[k]
                    for k in (
                        "native_mass_id_normalized",
                        "peptide_sequence",
                        "start_1based",
                        "end_1based",
                        "fragment_type",
                        "observed_mass",
                        "theoretical_mass",
                        "mass_error_ppm",
                        "n_assignments_reported",
                        "modifications_json",
                    )
                ]
                + [self.quantities[row["measurement_id"]]["quantitative_value"]]
            )
        groups = defaultdict(list)
        for row in self.views.values():
            if (
                row["source_view_role"] in {"controls_and_endpoints", "kinetics", "organs"}
                and row["acquisition_id"]
            ):
                groups[row["acquisition_id"]].append(row)
        for views in groups.values():
            for left_index, left in enumerate(views):
                for right in views[left_index + 1 :]:

                    def compare(data, left=left, right=right):
                        return Counter(map(_json, data[left["view_id"]])) == Counter(
                            map(_json, data[right["view_id"]])
                        )

                    inputs_equal, assignments_equal = (
                        compare(inputs_by_view),
                        compare(assignments_by_view),
                    )
                    self.reconcile(
                        left["source_record_id"],
                        left["view_id"],
                        right["view_id"],
                        "repeated_acquisition_source_views",
                        "same_values"
                        if inputs_equal and assignments_equal
                        else "conflicting_values",
                        left["source_locator"],
                        {
                            "input_masses_equal": inputs_equal,
                            "assignments_equal": assignments_equal,
                            "not_independent_measurements": True,
                        },
                    )

    def run(self):
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
                        "license_status": asset["license_status"],
                    }
                )
            self.metadata()
            self.workbooks()
            self.assignment_links()
            self.publisher_links()
            self.repeated_views()
            validate_material_graph(self.contexts, self.material_links)
            for table, records in (
                ("material_contexts", self.contexts),
                ("material_links", self.material_links),
            ):
                for row in records:
                    self.sinks[table].add(row)
            return {
                **{"n_" + name + "_rows": sink.n_rows for name, sink in self.sinks.items()},
                "n_populated_source_cells": self.n_nonempty_cells,
            }
        finally:
            for sink in self.sinks.values():
                sink.close()


def _compile(assets, destination, contract):
    return _Compiler(assets, destination, contract).run()


def _inventory(directory):
    return {
        str(p.relative_to(directory)): file_digest(p)
        for p in sorted(directory.rglob("*"))
        if p.is_file() and p != directory / "manifest.json"
    }


def verify_intact_degradomics_package(directory):
    """Verify hashes, pinned source claims, schemas and deterministic original replay."""
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if (
        manifest.get("schema_version"),
        manifest.get("catalog_schema_version"),
        manifest.get("package_family"),
        manifest.get("revision"),
    ) != (PACKAGE_SCHEMA_VERSION, CATALOG_SCHEMA_VERSION, PACKAGE_FAMILY, REVISION):
        raise ValueError("Unsupported intact-degradomics package")
    if _inventory(directory) != manifest.get("files"):
        raise ValueError("Intact-degradomics inventory or hashes differ")
    contract = json.loads(CONTRACT.read_text())
    if json.loads((directory / ROOT / "contract.json").read_text()) != contract:
        raise ValueError("Intact-degradomics source contract differs")
    if manifest.get("source_registry") != contract["source_registry"]:
        raise ValueError("Intact-degradomics source registry differs")
    for asset in contract["assets"]:
        if file_digest(directory / ROOT / "source_assets" / asset["filename"]) != {
            k: asset[k] for k in ("sha256", "size_bytes")
        }:
            raise ValueError("Intact-degradomics original source mismatch")
    temporary = Path(tempfile.mkdtemp(prefix="hitlist-intact-verify-"))
    try:
        counts = _compile(directory / ROOT / "source_assets", temporary / "typed", contract)
        if counts != manifest.get("reconciliation"):
            raise ValueError("Intact-degradomics reconciliation differs")
        for name, schema in SCHEMAS.items():
            actual = pq.ParquetFile(directory / ROOT / "typed" / f"{name}.parquet")
            expected = pq.ParquetFile(temporary / "typed" / f"{name}.parquet")
            if not actual.schema_arrow.equals(schema):
                raise ValueError(f"Intact-degradomics schema differs: {name}")
            for left, right in zip(expected.iter_batches(), actual.iter_batches(), strict=True):
                if not left.equals(right):
                    raise ValueError(f"Intact-degradomics source replay differs: {name}")
    except Exception:
        # Preserve the failed replay instead of deleting evidence needed for diagnosis.
        (temporary / "failure-context.json").write_text(_json({"package": str(directory)}) + "\n")
        raise
    else:
        shutil.rmtree(temporary)
    return manifest


def write_intact_degradomics_package(source_directory, destination):
    """Snapshot exact original assets; retain every source row, no training selection."""
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
                raise ValueError(f"Intact-degradomics original source differs: {asset['filename']}")
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
        verify_intact_degradomics_package(staging)
        staging.rename(destination)
        return destination / "manifest.json"
    except Exception:
        (staging / "failure-context.json").write_text(
            _json({"source": str(source_directory), "destination": str(destination)}) + "\n"
        )
        raise
