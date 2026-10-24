"""Literal OOXML projection for Goldberg's original publisher workbooks.

Excel date styles never become inferred gene names. Values, styles and formulas
remain distinct. Repeated pooled labels remain ordered slots, not dictionary keys.
"""

from __future__ import annotations

import io
import posixpath
import re
import zipfile
from xml.etree import ElementTree as ET

from .goldberg_mapp import _id, _number

NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
REL_NS = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"


def _column(coordinate):
    letters = re.match(r"[A-Z]+", coordinate)
    if not letters:
        raise ValueError("Invalid workbook cell coordinate")
    result = 0
    for char in letters[0]:
        result = result * 26 + ord(char) - ord("A") + 1
    return result


def workbook_rows(path_or_buffer):
    """Yield sparse literal cells by row; preserve XML serials and styles."""
    with zipfile.ZipFile(path_or_buffer) as book:
        strings = []
        if "xl/sharedStrings.xml" in book.namelist():
            with book.open("xl/sharedStrings.xml") as stream:
                for _, node in ET.iterparse(stream, events=("end",)):
                    if node.tag == NS + "si":
                        strings.append("".join(n.text or "" for n in node.iter(NS + "t")))
                        node.clear()
        relationships = {
            node.get("Id"): posixpath.normpath(posixpath.join("xl", node.get("Target"))).lstrip("/")
            for node in ET.fromstring(book.read("xl/_rels/workbook.xml.rels"))
        }
        workbook = ET.fromstring(book.read("xl/workbook.xml"))
        for sheet in workbook.find(NS + "sheets"):
            member = relationships[sheet.get(REL_NS + "id")]
            with book.open(member) as stream:
                root = None
                for event, node in ET.iterparse(stream, events=("start", "end")):
                    if root is None:
                        root = node
                    if event != "end" or node.tag != NS + "row":
                        continue
                    index = int(node.get("r"))
                    cells = []
                    for cell in node:
                        if cell.tag != NS + "c":
                            continue
                        raw = cell.findtext(NS + "v")
                        kind = cell.get("t", "n")
                        if kind == "s":
                            resolved = strings[int(raw)]
                        elif kind == "inlineStr":
                            resolved = "".join(n.text or "" for n in cell.iter(NS + "t"))
                        else:
                            resolved = raw or ""
                        cells.append(
                            {
                                "source_cell": cell.get("r"),
                                "cell_type_raw": kind,
                                "cell_style_raw": cell.get("s"),
                                "cell_value_raw": raw,
                                "cell_value_resolved": resolved,
                                "cell_formula_raw": cell.findtext(NS + "f"),
                                "source_row_index": index,
                                "source_column_index": _column(cell.get("r")),
                            }
                        )
                    yield sheet.get("name"), index, cells
                    node.clear()
                    # Clearing the worksheet's sheetData children bounds memory
                    # without retaining hundreds of thousands of empty row nodes.
                    data = root.find(NS + "sheetData")
                    if data is not None:
                        data.clear()


def _slots(compiler, record, cell):
    raw = cell["cell_value_resolved"]
    for index, slot in enumerate(raw.split(";"), 1):
        label, separator, value_raw = slot.partition(":")
        value, status = _number(value_raw)
        if not separator:
            status = "invalid_slot_no_separator"
        compiler.sinks["pooled_slots"].add(
            {
                "slot_id": _id("pooled_slot", record, index),
                "source_record_id": record,
                "source_cell": cell["source_cell"],
                "source_slot_index": index,
                "slot_raw": slot,
                "sample_label_raw": label,
                "quantitative_raw": value_raw,
                "quantitative_value": value,
                "value_status": status,
                "quantity_scale": "unresolved",
                "measurement_unit": "unreported",
                "identity_status": "unresolved_label_not_run_or_physical_sample",
                "acquisition_id": None,
                "specimen_id": None,
                "aliquot_id": None,
            }
        )


def _quantity_kind(field):
    lower = field.lower()
    if "intensity" not in lower and "median" not in lower:
        return None
    if "log10" in lower:
        return "publisher_derived_log10", "log10"
    if "median" in lower:
        return "publisher_derived_median", "source_reported_unqualified"
    if field.startswith("LFQ intensity "):
        return "publisher_LFQ_intensity", "linear"
    if field.startswith("Intensity ") or field == "Intensity":
        return "publisher_raw_intensity", "linear"
    return "publisher_other_intensity", "source_reported_unqualified"


def _publisher_row(compiler, source, sheet, record, cells, headers, pool):
    values = {headers.get(c["source_column_index"], ""): c["cell_value_resolved"] for c in cells}
    sequence = values.get("Sequence", values.get("Peptide Sequence"))
    parent = values.get("Proteins", values.get("Protein IDs", values.get("Protein ID")))
    native = values.get("id")
    original = compiler.peptides.get(native) if sheet == "MAPP_peptidomics" else None
    if sheet == "MAPP_peptidomics":
        status = (
            "exact_native_id_and_sequence"
            if original and original["peptide_sequence"] == sequence
            else "unresolved_or_sequence_conflict"
        )
        compiler.sinks["publisher_links"].add(
            {
                "claim_id": _id("publisher_row_link", record),
                "source_record_id": record,
                "peptide_id": original["peptide_id"] if original else None,
                "link_status": status,
                "assertion_predicate": "publisher_row_original_identity",
                "source_locator": f"{source}:{sheet}",
            }
        )
    for cell in cells:
        field = headers.get(cell["source_column_index"], "")
        if sheet == "MAPP database" and field == "Treatment":
            _slots(compiler, record, cell)
        kind = _quantity_kind(field)
        if kind is None:
            continue
        value, status = _number(cell["cell_value_resolved"])
        if cell["cell_formula_raw"] is not None:
            status = (
                "formula_cached_value_unverified"
                if value is not None
                else "formula_without_cached_value"
            )
        if cell["cell_type_raw"] in ("e", "b"):
            value, status = None, "invalid_quantity_cell_type"
        measurement = _id("publisher_quantity", record, cell["source_cell"])
        compiler.sinks["publisher_quantities"].add(
            {
                "measurement_id": measurement,
                "source_record_id": record,
                "source_file": source,
                "sheet_name": sheet,
                "source_cell": cell["source_cell"],
                "source_field": field,
                "peptide_sequence": sequence,
                "parent_label_raw": parent,
                "native_id": native,
                "measured_pool": pool,
                "quantity_kind": kind[0],
                "quantitative_raw": cell["cell_value_resolved"],
                "quantitative_value": value,
                "value_status": status,
                "quantity_scale": kind[1],
                "measurement_unit": "publisher_source_signal_not_molarity",
            }
        )
        if sheet != "MAPP_peptidomics" or kind[0] != "publisher_raw_intensity":
            continue
        source_quantity = compiler.peptide_quantities.get((native, field))
        if not original or original["peptide_sequence"] != sequence:
            agreement = "unresolved_row_identity"
        elif source_quantity is None:
            agreement = "unresolved_exact_column_name"
        elif (value, status) == source_quantity[1:]:
            agreement = "exact_source_value_agreement"
        else:
            agreement = "source_value_conflict"
        compiler.sinks["publisher_links"].add(
            {
                "claim_id": _id("publisher_quantity_link", measurement),
                "source_record_id": record,
                "source_measurement_id": source_quantity[0] if source_quantity else None,
                "publisher_measurement_id": measurement,
                "peptide_id": original["peptide_id"] if original else None,
                "link_status": agreement,
                "assertion_predicate": "publisher_original_raw_intensity",
                "source_locator": f"{source}:{sheet}!{cell['source_cell']}",
            }
        )


def compile_publisher(compiler):
    with zipfile.ZipFile(compiler.assets / compiler.contract["publisher"]) as archive:
        for block in compiler.contract["publisher_workbooks"]:
            source = compiler.contract["publisher"] + "!" + block["member"]
            headers = {}
            for sheet, index, cells in workbook_rows(io.BytesIO(archive.read(block["member"]))):
                locator = f"{sheet}!row:{index}"
                record = compiler.record(
                    source,
                    locator,
                    index,
                    [c["source_cell"] for c in cells],
                    [c["cell_value_resolved"] for c in cells],
                )
                for cell in cells:
                    compiler.sinks["publisher_cells"].add(
                        {
                            "cell_id": _id("publisher_cell", source, sheet, cell["source_cell"]),
                            "source_record_id": record,
                            "source_file": source,
                            "sheet_name": sheet,
                            **cell,
                        }
                    )
                if index == 1:
                    headers[sheet] = {
                        c["source_column_index"]: c["cell_value_resolved"] for c in cells
                    }
                    continue
                pool = block["sheet_pools"].get(sheet)
                if pool:
                    _publisher_row(compiler, source, sheet, record, cells, headers[sheet], pool)
