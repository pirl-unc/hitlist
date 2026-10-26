"""Compile original Iso18 source claims without manufacturing training labels."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation, localcontext
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from .material_context import validate_material_graph
from .nontryptic_calibration_schema import RELATIONS, SCHEMAS
from .nontryptic_workbooks import read_source_workbook
from .provenance import file_digest


def identity(kind, *parts):
    value = json.dumps(parts, ensure_ascii=True, separators=(",", ":"))
    return "pxd024917:" + kind + ":" + hashlib.sha256(value.encode()).hexdigest()


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def numeric(raw):
    if raw in (None, ""):
        return None, "not_reported"
    try:
        value = Decimal(raw)
        parsed = float(value)
    except (InvalidOperation, ValueError, OverflowError):
        return None, "invalid_numeric"
    if (
        not value.is_finite()
        or not math.isfinite(parsed)
        or value < 0
        or (value != 0 and parsed == 0)
    ):
        return None, "invalid_numeric"
    return parsed, "reported_zero" if value == 0 else "reported_positive"


def form_pattern(text, *, parenthesized=False):
    pattern = r"\(([A-Z])\)|([A-Z])" if parenthesized else r"([A-Z])(?:\[\+([0-9]+)\])?"
    tokens = list(re.finditer(pattern, text))
    if not tokens or "".join(m[0] for m in tokens) != text:
        raise ValueError(f"Unsupported reported peptide form: {text!r}")
    if parenthesized:
        sequence = "".join(m[1] or m[2] for m in tokens)
        positions = [i for i, m in enumerate(tokens, 1) if m[1]]
        shift = None
    else:
        sequence = "".join(m[1] for m in tokens)
        positions = [i for i, m in enumerate(tokens, 1) if m[2]]
        shift = sum(int(m[2]) for m in tokens if m[2])
    return sequence, positions, shift


class Compiler:
    def __init__(self, source_directory, contract):
        self.source = Path(source_directory)
        self.contract = contract
        self.tables = {name: [] for name in SCHEMAS}
        self.books = {}
        self.assets = {}
        self.standards = {}
        self.amounts = {}
        self.groups = {}
        self.exports = {}
        self.forms_by_export = defaultdict(list)

    def add(self, name, row):
        unknown = row.keys() - set(SCHEMAS[name].names)
        if unknown:
            raise ValueError(f"Unknown {name} fields: {sorted(unknown)}")
        self.tables[name].append(row)
        return row

    def address(self, role, sheet, coordinate):
        filename = self.contract["workbooks"][role]
        return f"{filename}#{sheet}!{coordinate}"

    def sheet(self, role, name):
        return self.books[role][name]

    def cell(self, role, sheet, coordinate):
        return self.sheet(role, sheet).cells.get(coordinate)

    def value(self, role, sheet, coordinate):
        cell = self.cell(role, sheet, coordinate)
        return cell.value_literal if cell else None

    def measured(self, role, sheet, coordinate):
        cell = self.cell(role, sheet, coordinate)
        raw = cell.value_literal if cell else None
        if cell and cell.has_formula:
            return raw, None, "formula_not_measurement"
        if cell and cell.excel_type != "n" and raw not in (None, ""):
            return raw, None, "non_numeric_cell"
        value, status = numeric(raw)
        return raw, value, status

    def rows(self, role, sheet, *, start):
        return sorted(
            {int(re.search(r"[0-9]+$", p)[0]) for p in self.sheet(role, sheet).cells}
            - set(range(1, start))
        )

    def asset_id(self, role):
        return identity("asset", self.contract["workbooks"][role])

    def require_standard(self, sequence):
        if sequence not in self.standards:
            raise ValueError(f"Peptide absent from the reported panel: {sequence}")
        return self.standards[sequence]

    def sources(self):
        names = [a["filename"] for a in self.contract["assets"]]
        if len(names) != len(set(names)) or any(
            Path(n).name != n or n in {"", ".", ".."} or "\\" in n for n in names
        ):
            raise ValueError("Source assets require unique safe basenames")
        for asset in self.contract["assets"]:
            filename = asset["filename"]
            if file_digest(self.source / filename) != {
                k: asset[k] for k in ("sha256", "size_bytes")
            }:
                raise ValueError(f"Pinned source asset differs: {filename}")
            self.assets[filename] = asset
            self.add(
                "assets",
                {
                    "source_asset_id": identity("asset", filename),
                    "source_file": filename,
                    "source_asset_url": asset["source_url"],
                    "source_asset_role": asset["role"],
                    "asset_sha256": asset["sha256"],
                    "size_bytes": asset["size_bytes"],
                    "license_status": asset["license_status"],
                },
            )
        for role, filename in self.contract["workbooks"].items():
            if filename not in self.assets:
                raise ValueError("Workbook not in pinned asset contract")
            sheets = read_source_workbook(self.source / filename)
            self.books[role] = {s.name: s for s in sheets}
            for sheet in sheets:
                sheet_id = identity("sheet", filename, sheet.name)
                self.add(
                    "source_sheets",
                    {
                        "sheet_id": sheet_id,
                        "source_asset_id": identity("asset", filename),
                        "sheet_name": sheet.name,
                        "sheet_state": sheet.state,
                        "sheet_dimension": sheet.dimension,
                        "merged_ranges_json": encoded(sheet.merged_ranges),
                        "source_locator": f"{filename}#{sheet.name}",
                    },
                )
                for coordinate, cell in sheet.cells.items():
                    self.add(
                        "source_cells",
                        {
                            "cell_id": identity("cell", filename, sheet.name, coordinate),
                            "sheet_id": sheet_id,
                            "cell_ref": coordinate,
                            "excel_type": cell.excel_type,
                            "value_literal": cell.value_literal,
                            "raw_cached_value": cell.raw_cached_value,
                            "value_present": cell.value_present,
                            "has_formula": cell.has_formula,
                            "formula": cell.formula,
                            "cell_attributes_json": encoded(cell.attributes),
                            "formula_attributes_json": encoded(cell.formula_attributes),
                            "xml_element": cell.source_xml,
                            "source_locator": self.address(role, sheet.name, coordinate),
                        },
                    )

    def panel(self):
        sheet = "DatasetS1"
        if [self.value("panel", sheet, c + "3") for c in "BCDEF"] != [
            "Peptide",
            "Gene",
            "Annotated Gene",
            "Length",
            "Predicted affinity [nM]",
        ]:
            raise ValueError("Panel headers differ")
        for row in self.rows("panel", sheet, start=4):
            sequence = self.value("panel", sheet, f"B{row}")
            if sequence is None:
                continue
            if not re.fullmatch("[ACDEFGHIKLMNPQRSTVWY]+", sequence) or sequence in self.standards:
                raise ValueError("Invalid or duplicate panel peptide")
            length_raw, length, length_status = self.measured("panel", sheet, f"E{row}")
            if length_status == "reported_positive" and length.is_integer():
                length_consistency = (
                    "agrees_with_sequence" if length == len(sequence) else "conflicts_with_sequence"
                )
            else:
                length_consistency = "no_valid_reported_length"
            raw, affinity, status = self.measured("panel", sheet, f"F{row}")
            standard = identity("standard", sequence)
            self.standards[sequence] = standard
            self.add(
                "standards",
                {
                    "standard_id": standard,
                    "peptide_sequence": sequence,
                    "gene_symbol_raw": self.value("panel", sheet, f"C{row}"),
                    "annotated_gene_raw": self.value("panel", sheet, f"D{row}"),
                    "peptide_length": len(sequence),
                    "reported_length_raw": length_raw,
                    "reported_length_value": length,
                    "reported_length_status": length_status,
                    "length_consistency_status": length_consistency,
                    "predicted_affinity_raw": raw,
                    "predicted_affinity_nm": affinity,
                    "prediction_status": status + "_prediction_not_assay",
                    "source_asset_id": self.asset_id("panel"),
                    "source_locator": self.address("panel", sheet, f"B{row}:F{row}"),
                },
            )

    def isotope_forms(self):
        residues = {}
        for row in self.rows("isotopes", "DatasetS6A", start=5):
            residue = self.value("isotopes", "DatasetS6A", f"A{row}")
            if residue is None:
                continue
            raw, value, status = self.measured("isotopes", "DatasetS6A", f"C{row}")
            if (
                not re.fullmatch("[A-Z]", residue)
                or residue in residues
                or status != "reported_positive"
            ):
                raise ValueError("Invalid or duplicate S6A residue assertion")
            residues[residue] = Decimal(raw)
            locator = self.address("isotopes", "DatasetS6A", f"A{row}:C{row}")
            self.add(
                "isotope_assertions",
                {
                    "claim_id": identity("isotope", locator),
                    "residue_symbol": residue,
                    "isotope_atoms_raw": self.value("isotopes", "DatasetS6A", f"B{row}"),
                    "mass_offset_raw": raw,
                    "mass_offset_da": value,
                    "source_asset_id": self.asset_id("isotopes"),
                    "source_locator": locator,
                },
            )
        for row in self.rows("isotopes", "DatasetS6B", start=4):
            label = self.value("isotopes", "DatasetS6B", f"A{row}")
            if label not in {"1H", "2H", "3H", "4H"}:
                continue
            text = self.value("isotopes", "DatasetS6B", f"B{row}")
            sequence, positions, _ = form_pattern(text, parenthesized=True)
            raw, shift, status = self.measured("isotopes", "DatasetS6B", f"D{row}")
            if status != "reported_positive":
                raise ValueError("Invalid S6B nominal mass shift")
            total = None
            consistency = "missing_S6A_residue_assertion"
            if all(sequence[i - 1] in residues for i in positions):
                total = sum((residues[sequence[i - 1]] for i in positions), Decimal(0))
                consistency = (
                    "within_0.15_Da"
                    if abs(total - Decimal(raw)) <= Decimal("0.15")
                    else "conflicting_source_assertions"
                )
            locator = self.address("isotopes", "DatasetS6B", f"A{row}:H{row}")
            self.add(
                "reported_forms",
                {
                    "form_id": identity("form", locator),
                    "standard_id": self.require_standard(sequence),
                    "peptide_sequence": sequence,
                    "isotopologue_label": label,
                    "reported_form": text,
                    "labeled_positions_json": encoded(positions),
                    "nominal_mass_shift_raw": raw,
                    "nominal_mass_shift_da": shift,
                    "heavy_amino_acid_raw": self.value("isotopes", "DatasetS6B", f"E{row}"),
                    "s6a_consistency_status": consistency,
                    "s6a_sum_raw": str(total) if total is not None else None,
                    "canonical_isotope_status": "unresolved_reported_claim_only",
                    "source_asset_id": self.asset_id("isotopes"),
                    "source_locator": locator,
                },
            )

    def contexts(self):
        for label in ("project", "direct", "embedded"):
            self.add(
                "material_contexts",
                {
                    "context_id": identity("context", label),
                    "context_kind": "source_project" if label == "project" else "study_arm",
                    "sample_label_raw": self.contract["source_dataset"]
                    if label == "project"
                    else label,
                    "condition_id": None,
                    "condition_text": None,
                    "condition_status": "protocol_context_not_perturbation",
                    "specimen_id": None,
                    "aliquot_id": None,
                    "material_link_precision": "project_context"
                    if label == "project"
                    else "study_arm",
                    "context_status": "reported_observation_design_no_physical_sample_identity",
                    "source_locator": self.contract["context_locators"][label],
                    "notes": "Calibration standard response, not native APC peptide availability",
                },
            )
            if label != "project":
                self.add(
                    "material_links",
                    {
                        "claim_id": identity("context_link", label),
                        "source_context_id": identity("context", label),
                        "target_context_id": identity("context", "project"),
                        "assertion_predicate": "study_arm_in_project",
                        "assertion_status": "source_reported",
                        "material_link_precision": "project_context",
                        "source_locator": self.contract["context_locators"][label],
                        "notes": None,
                    },
                )
        for index, claim in enumerate(self.contract["protocol_assertions"]):
            if claim["source_file"] not in self.assets or claim["context"] not in {
                "project",
                "direct",
                "embedded",
            }:
                raise ValueError("Protocol assertion lacks admitted source/context")
            self.add(
                "protocol_assertions",
                {
                    "claim_id": identity("protocol_assertion", index, claim),
                    "context_id": identity("context", claim["context"]),
                    "assertion_predicate": claim["predicate"],
                    "assertion_value": claim["value_raw"],
                    "assertion_status": claim["status"],
                    "source_scope": claim["scope"],
                    "measurement_unit": claim.get("unit"),
                    "source_asset_id": identity("asset", claim["source_file"]),
                    "source_locator": claim["source_locator"],
                    "evidence_text": claim.get("evidence_text"),
                },
            )

    def responses(self):
        for sheet, context, isotope in [
            ("DatasetS3A", "direct", "1H"),
            ("DatasetS3B", "embedded", "2H"),
        ]:
            if [self.value("responses", sheet, c + "3").lower() for c in "ABC"] != [
                "peptide",
                "fmol",
                "summed intensity",
            ]:
                raise ValueError("Response table headers differ")
            for ordinal, column in enumerate("CDE", 1):
                self.add(
                    "source_slots",
                    {
                        "slot_id": identity("slot", sheet, column),
                        "context_id": identity("context", context),
                        "sheet_id": identity(
                            "sheet", self.contract["workbooks"]["responses"], sheet
                        ),
                        "source_column": column,
                        "source_slot_ordinal": ordinal,
                        "acquisition_id": None,
                        "specimen_id": None,
                        "aliquot_id": None,
                        "material_link_precision": "unresolved",
                        "source_locator": self.address("responses", sheet, column),
                    },
                )
            current = None
            for row in self.rows("responses", sheet, start=4):
                raw, dose, status = self.measured("responses", sheet, f"B{row}")
                if status == "not_reported":
                    if any(self.value("responses", sheet, f"{c}{row}") is not None for c in "CDE"):
                        raise ValueError("Response has no nominal dose")
                    continue
                if status != "reported_positive":
                    raise ValueError("Nominal dose is not a positive numeric source value")
                label = self.value("responses", sheet, f"A{row}")
                if label is not None:
                    current = label
                if current is None:
                    raise ValueError("Response has no peptide block")
                sequence, _, _ = form_pattern(current)
                standard = self.require_standard(sequence)
                amount_id = identity("amount", sheet, row)
                self.amounts[amount_id] = self.add(
                    "nominal_amounts",
                    {
                        "nominal_amount_id": amount_id,
                        "standard_id": standard,
                        "context_id": identity("context", context),
                        "reported_form_label": current,
                        "isotopologue_label": isotope,
                        "nominal_amount_raw": raw,
                        "nominal_amount_fmol": dose,
                        "measurement_unit": "fmol",
                        "nominal_status": "source_reported_not_independently_certified",
                        "source_asset_id": self.asset_id("responses"),
                        "source_locator": self.address("responses", sheet, f"B{row}"),
                    },
                )
                for column in "CDE":
                    coordinate = f"{column}{row}"
                    signal, value, signal_status = self.measured("responses", sheet, coordinate)
                    cell = self.cell("responses", sheet, coordinate)
                    self.add(
                        "measurements",
                        {
                            "measurement_id": identity("measurement", sheet, coordinate),
                            "nominal_amount_id": amount_id,
                            "slot_id": identity("slot", sheet, column),
                            "standard_id": standard,
                            "context_id": identity("context", context),
                            "quantitative_raw": signal,
                            "quantitative_value": value,
                            "value_status": signal_status,
                            "measurement_unit": "source_arbitrary_summed_intensity",
                            "observation_kind": "reported_targeted_product_ion_sum",
                            "aggregation_status": "reported_aggregate_not_fully_reconstructed",
                            "cell_id": identity(
                                "cell", self.contract["workbooks"]["responses"], sheet, coordinate
                            )
                            if cell
                            else None,
                            "source_locator": self.address("responses", sheet, coordinate),
                        },
                    )
        sheet = "DatasetS3C"
        for row in self.rows("responses", sheet, start=3):
            sequence = self.value("responses", sheet, f"B{row}")
            if sequence is None:
                continue
            standard = self.require_standard(sequence)
            for column in "CDEFGH":
                header = self.value("responses", sheet, column + "2")
                match = re.fullmatch(r"([0-9.]+) (amol|fmol)", header or "")
                if match is None:
                    raise ValueError("Transition-count dose header differs")
                dose = Decimal(match[1]) * (Decimal("0.001") if match[2] == "amol" else 1)
                raw, value, status = self.measured("responses", sheet, f"{column}{row}")
                if value is not None and value != int(value):
                    raise ValueError("Transition count is fractional")
                self.add(
                    "transition_counts",
                    {
                        "measurement_id": identity("transition_count", row, column),
                        "standard_id": standard,
                        "nominal_amount_raw": match[1],
                        "amount_unit_raw": match[2],
                        "nominal_amount_fmol": float(dose),
                        "quantitative_raw": raw,
                        "quantitative_value": value,
                        "value_status": status,
                        "measurement_unit": "selected_transition_count",
                        "source_asset_id": self.asset_id("responses"),
                        "source_locator": self.address("responses", sheet, f"{column}{row}"),
                    },
                )

    def export_form(self, role, sheet, row, sequence, text):
        key = role, text
        if key in self.exports:
            return self.exports[key]
        parsed, positions, shift = form_pattern(text)
        if parsed != sequence:
            raise ValueError("Export peptide and modified sequence disagree")
        form_id = identity("export_form", self.contract["workbooks"][role], text)
        self.exports[key] = form_id
        self.add(
            "export_forms",
            {
                "export_form_id": form_id,
                "standard_id": self.require_standard(sequence),
                "peptide_sequence": sequence,
                "modified_sequence_raw": text,
                "labeled_positions_json": encoded(positions),
                "nominal_mass_shift_raw": str(shift),
                "canonical_isotope_status": "unresolved_reported_claim_only",
                "source_asset_id": self.asset_id(role),
                "source_locator": self.address(role, sheet, f"B{row}"),
            },
        )
        candidates = [
            form
            for form in self.tables["reported_forms"]
            if form["peptide_sequence"] == sequence
            and form["labeled_positions_json"] == encoded(positions)
            and Decimal(form["nominal_mass_shift_raw"]) == shift
        ]
        self.forms_by_export[form_id] = candidates
        for candidate in candidates or [None]:
            self.add(
                "form_correspondences",
                {
                    "claim_id": identity(
                        "form_correspondence", form_id, candidate["form_id"] if candidate else None
                    ),
                    "export_form_id": form_id,
                    "form_id": candidate["form_id"] if candidate else None,
                    "n_form_candidates": len(candidates),
                    "correspondence_status": "unique_reported_correspondence"
                    if len(candidates) == 1
                    else "ambiguous"
                    if candidates
                    else "no_reported_correspondence",
                    "correspondence_basis": "sequence_explicit_label_positions_and_nominal_total_shift_not_atoms",
                },
            )
        return form_id

    def transitions(self):
        expected = [
            "Peptide",
            "Modified Sequence",
            "Replicate",
            "Precursor Mz",
            "Precursor Charge",
            "Product Mz",
            "Product Charge",
            "Fragment Ion",
            "Retention Time",
            "Area",
            "Peak Rank",
            "Concat",
        ]
        for role, context in [("direct_export", "direct"), ("embedded_export", "embedded")]:
            if len(self.books[role]) != 1:
                raise ValueError("Transition export must have one worksheet")
            sheet = next(iter(self.books[role]))
            if [self.value(role, sheet, c + "1") for c in "ABCDEFGHIJKL"] != expected:
                raise ValueError("Transition export headers differ")
            if context == "direct" and [
                self.value(role, sheet, f"N{row}") for row in range(1, 6)
            ] != ["Replicate ", "0.1 = 100 amol", "1 = 1 fmol", "10 = 10 fmol", "100 = 100 fmol"]:
                raise ValueError("DirectSpike dose legend differs")
            identities = Counter()
            row_keys = {}
            for row in self.rows(role, sheet, start=2):
                values = {c: self.value(role, sheet, f"{c}{row}") for c in "ABCDEFGHIJKL"}
                if all(v is None for v in values.values()):
                    continue
                if any(values[c] is None for c in "ABCDEFGH"):
                    raise ValueError("Transition identity is incomplete")
                sequence, text, label = (values[c] for c in "ABC")
                parsed, _, _ = form_pattern(text)
                if parsed != sequence:
                    raise ValueError("Export peptide and modified sequence disagree")
                standard = self.require_standard(sequence)
                form_id = self.export_form(role, sheet, row, sequence, text)
                pattern = (
                    r"(0\.1|1|10|100)_([123])"
                    if context == "direct"
                    else r"([0-9.]+)(amol|fmol)_([123])"
                )
                match = re.fullmatch(pattern, label)
                if match is None or numeric(match[1])[1] != "reported_positive":
                    raise ValueError("Unsupported nominal amount/replicate label")
                unit = "fmol" if context == "direct" else match[2]
                dose = Decimal(match[1]) * (Decimal("0.001") if unit == "amol" else 1)
                group_id = identity(
                    "transition_group", self.contract["workbooks"][role], text, label
                )
                if group_id not in self.groups:
                    self.groups[group_id] = self.add(
                        "transition_groups",
                        {
                            "transition_group_id": group_id,
                            "export_form_id": form_id,
                            "standard_id": standard,
                            "context_id": identity("context", context),
                            "source_run_label": label,
                            "nominal_amount_raw": match[1],
                            "amount_unit_raw": unit,
                            "nominal_amount_fmol": float(dose),
                            "replicate_suffix_ordinal": int(
                                match[2] if context == "direct" else match[3]
                            ),
                            "acquisition_id": None,
                            "material_link_precision": "unresolved",
                            "source_asset_id": self.asset_id(role),
                            "source_locator": self.address(role, sheet, f"A{row}:C{row}"),
                        },
                    )
                raw, value, status = self.measured(role, sheet, f"J{row}")
                identity_key = tuple(values[c] for c in "ABCDEFGH")
                identities[identity_key] += 1
                transition_id = identity("transition", self.contract["workbooks"][role], sheet, row)
                row_keys[transition_id] = identity_key
                self.add(
                    "transitions",
                    {
                        "transition_id": transition_id,
                        "transition_group_id": group_id,
                        "precursor_mz_raw": values["D"],
                        "precursor_charge_raw": values["E"],
                        "product_mz_raw": values["F"],
                        "product_charge_raw": values["G"],
                        "fragment_ion_raw": values["H"],
                        "retention_time_raw": values["I"],
                        "quantitative_raw": raw,
                        "quantitative_value": value,
                        "value_status": status,
                        "peak_rank_raw": values["K"],
                        "concat_raw": values["L"],
                        "concat_status": "not_reported"
                        if values["L"] in (None, "")
                        else "consistent"
                        if values["L"] == values["H"] + "_" + values["G"]
                        else "conflicting",
                        "signal_kind": "precursor" if values["H"] == "precursor" else "product_ion",
                        "measurement_unit": "export_peak_area",
                        "n_duplicate_identity_rows": None,
                        "source_asset_id": self.asset_id(role),
                        "source_locator": self.address(role, sheet, f"A{row}:L{row}"),
                    },
                )
            for transition in self.tables["transitions"]:
                if transition["transition_id"] in row_keys:
                    transition["n_duplicate_identity_rows"] = identities[
                        row_keys[transition["transition_id"]]
                    ]

    def aggregates_and_candidates(self):
        grouped = defaultdict(list)
        for transition in self.tables["transitions"]:
            grouped[transition["transition_group_id"]].append(transition)
        for group_id, transitions in grouped.items():
            for mode in ("product_only", "all_transitions"):
                included = [
                    t
                    for t in transitions
                    if mode == "all_transitions" or t["signal_kind"] == "product_ion"
                ]
                states = Counter(t["value_status"] for t in included)
                n_missing = states["not_reported"]
                n_invalid = sum(
                    n
                    for state, n in states.items()
                    if state not in {"not_reported", "reported_positive", "reported_zero"}
                )
                raw = None
                if included and not (n_missing or n_invalid):
                    values = [Decimal(t["quantitative_raw"]) for t in included]
                    with localcontext() as context:
                        context.prec = (
                            max(v.adjusted() for v in values)
                            - min(v.as_tuple().exponent for v in values)
                            + len(str(len(values)))
                            + 4
                        )
                        raw = str(sum(values, Decimal(0)))
                value, status = numeric(raw)
                if not included:
                    status = "no_transitions_of_requested_kind"
                elif n_missing or n_invalid:
                    status = "incomplete_sum"
                elif any(t["n_duplicate_identity_rows"] > 1 for t in included):
                    status = "duplicate_identity_sum_unqualified"
                self.add(
                    "aggregate_signals",
                    {
                        "aggregate_id": identity("aggregate", group_id, mode),
                        "transition_group_id": group_id,
                        "aggregation_mode": mode,
                        "quantitative_raw": raw,
                        "quantitative_value": value,
                        "value_status": status,
                        "measurement_unit": "derived_export_peak_area_sum",
                        "source_transition_ids_json": encoded(
                            [t["transition_id"] for t in included]
                        ),
                        "n_transition_rows": len(included),
                        "n_missing_areas": n_missing,
                        "n_invalid_areas": n_invalid,
                        "n_zero_areas": states["reported_zero"],
                    },
                )
        slots = {s["slot_id"]: s for s in self.tables["source_slots"]}
        for measurement in self.tables["measurements"]:
            amount = self.amounts[measurement["nominal_amount_id"]]
            candidates = []
            for group in self.groups.values():
                if (
                    group["standard_id"] != amount["standard_id"]
                    or group["context_id"] != amount["context_id"]
                ):
                    continue
                dose = Decimal(group["nominal_amount_raw"]) * (
                    Decimal("0.001") if group["amount_unit_raw"] == "amol" else 1
                )
                forms = self.forms_by_export[group["export_form_id"]]
                if dose == Decimal(amount["nominal_amount_raw"]) and any(
                    form["isotopologue_label"] == amount["isotopologue_label"] for form in forms
                ):
                    candidates.append(group)
            for group in candidates or [None]:
                self.add(
                    "measurement_candidates",
                    {
                        "claim_id": identity(
                            "measurement_candidate",
                            measurement["measurement_id"],
                            group["transition_group_id"] if group else None,
                        ),
                        "measurement_id": measurement["measurement_id"],
                        "transition_group_id": group["transition_group_id"] if group else None,
                        "n_group_candidates": len(candidates),
                        "correspondence_status": "form_dose_candidate_only"
                        if group
                        else "no_form_dose_candidate",
                        "correspondence_basis": "reported_form_and_nominal_amount_no_acquisition_assignment",
                        "ordinal_suffix_equal": slots[measurement["slot_id"]]["source_slot_ordinal"]
                        == group["replicate_suffix_ordinal"]
                        if group
                        else None,
                        "acquisition_join_established": False,
                    },
                )

    def compile(self):
        self.sources()
        self.panel()
        self.isotope_forms()
        self.contexts()
        self.responses()
        self.transitions()
        self.aggregates_and_candidates()
        validate_material_graph(self.tables["material_contexts"], self.tables["material_links"])
        for name, rows in self.tables.items():
            key = RELATIONS[name][2]
            keys = [row[key] for row in rows]
            if len(set(keys)) != len(keys) or any(not key for key in keys):
                raise ValueError(f"Missing or duplicate typed identity: {name}")
        return {
            name: pa.Table.from_pylist(rows, schema=SCHEMAS[name])
            for name, rows in self.tables.items()
        }


def compile_nontryptic_source(source_directory, contract):
    """Return source-faithful typed tables; callers pin the reviewed contract."""
    return Compiler(source_directory, contract).compile()


def write_nontryptic_tables(source_directory, destination, contract):
    """Write a new typed directory without replacing any prior source generation."""
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    tables = compile_nontryptic_source(source_directory, contract)
    destination.mkdir(parents=True, exist_ok=False)
    for name, table in tables.items():
        pq.write_table(table, destination / f"{name}.parquet", compression="zstd")
    return {"n_" + name + "_rows": table.num_rows for name, table in tables.items()}
