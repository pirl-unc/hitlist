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
_SAMPLE_LABEL = r"(?:Ctrl|Infect)4?_[1-9][0-9]*"
_RUN_LABEL = re.compile(rf"(?P<run>[0-9]+)_(?P<label>.+_{_SAMPLE_LABEL})$")
_SAMPLE_SUFFIX = re.compile(rf"(?P<sample>{_SAMPLE_LABEL})$")
_READ_ME_CONDITION = re.compile(
    r"A549 (?P<role>control|S\.E\. infection)(?: for)? "
    r"(?P<hours>[1-9][0-9]*) hours?, bio replicate (?P<replicate>[1-9][0-9]*)$"
)


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


def _row_values(cells):
    return {cell["source_column_index"]: cell for cell in cells}


def _source_run_tokens(raw_file):
    """Return unambiguous four-digit literal tokens from a deposited raw name."""
    return set(re.findall(r"(?<![0-9])[0-9]{4}(?![0-9])", raw_file))


def _header_run_label(compiler, field, experiment):
    """Resolve a Table5 column label only through source-reported identities.

    The original MAPP summary has a full experiment label for its own rows.
    The other two publisher views carry a literal raw-file run token and the
    deposited sample-label suffix. Neither route uses a run order or a
    replicate number as an identity.
    """
    direct = [
        (label, acquisition_id)
        for label, acquisition_id in compiler.experiments.items()
        if field.endswith(label)
        and compiler.acquisitions_by_id[acquisition_id]["assay_name_raw"] == experiment
    ]
    if len(direct) == 1:
        label, acquisition_id = direct[0]
        return label, acquisition_id, "exact_original_search_run_and_deposition_acquisition"
    if len(direct) > 1:
        raise ValueError("Publisher header matches multiple original search labels")

    match = _RUN_LABEL.search(field)
    if not match:
        return None
    run_label = match.group(0)
    sample = _SAMPLE_SUFFIX.search(run_label)
    if not sample:
        return None
    candidates = [
        row
        for row in compiler.acquisitions.values()
        if row["assay_name_raw"] == experiment
        and row["sample_label_raw"] == sample.group("sample")
        and match.group("run") in _source_run_tokens(row["raw_file"])
    ]
    if len(candidates) != 1:
        return None
    acquisition_id = candidates[0]["acquisition_id"]
    status = "exact_deposition_raw_token_and_sample_label"
    if compiler.search_run_labels[acquisition_id]:
        status = "exact_deposition_raw_token_and_sample_label_conflicts_original_search_run"
    return run_label, acquisition_id, status


def _publisher_sample_label(run_label):
    match = _RUN_LABEL.fullmatch(run_label)
    return match.group("label") if match else run_label


def _readme_condition(value):
    """Parse only the stable literal wording in the source ReadMe legend."""
    match = _READ_ME_CONDITION.fullmatch(value)
    if not match:
        return ()
    return (
        ("reported_condition_role", match.group("role")),
        ("reported_exposure_duration_hours", match.group("hours")),
        ("reported_biological_replicate", match.group("replicate")),
    )


def _publisher_condition_assertions(compiler, headers, legends):
    """Attach Table5 legend facts through exact header/deposition identities.

    This records the publisher evidence at the acquisition grain. The shared
    source-sample label context remains deliberately nonphysical, and no
    mapping is made for Table3's pooled slots or other workbook sheets.
    """
    for (sheet, run_label), header in headers.items():
        legend_rows = legends.get((sheet, _publisher_sample_label(run_label)), ())
        # The ReadMe contains several unrelated legends. Only an unambiguous
        # matching literal label can contribute a condition claim; all other
        # workbook cells remain in publisher_cells without an invented join.
        if len(legend_rows) != 1:
            continue
        legend = legend_rows[0]
        record, label_cell, description_cell, description = legend
        acquisition_id = header["acquisition_id"]
        locator = f"{sheet}!{label_cell}"
        status = "exact_publisher_header_to_deposition_acquisition"
        compiler.assertion(
            record,
            acquisition_id,
            "publisher_table5_readme_via_exact_header_and_deposition_raw",
            "publisher_sample_label",
            _publisher_sample_label(run_label),
            locator,
            status,
        )
        compiler.assertion(
            record,
            acquisition_id,
            "publisher_table5_readme_via_exact_header_and_deposition_raw",
            "reported_condition_description",
            description,
            f"{sheet}!{description_cell}",
            status,
        )
        for predicate, value in _readme_condition(description):
            compiler.assertion(
                record,
                acquisition_id,
                "publisher_table5_readme_via_exact_header_and_deposition_raw",
                predicate,
                value,
                f"{sheet}!{description_cell}",
                "literal_parse_of_publisher_readme_description",
            )


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
    views = compiler.contract.get("publisher_condition_views", {})
    headers_by_label = {}
    legends = {}
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
                    experiment = views.get(sheet)
                    if experiment:
                        for cell in cells:
                            resolved = _header_run_label(
                                compiler, cell["cell_value_resolved"], experiment
                            )
                            if resolved is None:
                                continue
                            run_label, acquisition_id, status = resolved
                            key = (sheet, run_label)
                            previous = headers_by_label.get(key)
                            if previous and previous["acquisition_id"] != acquisition_id:
                                raise ValueError(
                                    "Publisher run label maps to multiple deposited acquisitions"
                                )
                            if previous is None:
                                headers_by_label[key] = {
                                    "acquisition_id": acquisition_id,
                                    "status": status,
                                }
                                compiler.assertion(
                                    record,
                                    acquisition_id,
                                    "publisher_table5_column_header",
                                    "publisher_run_label",
                                    run_label,
                                    f"{sheet}!{cell['source_cell']}",
                                    status,
                                )
                    continue
                if sheet == "ReadMe":
                    values = _row_values(cells)
                    label, description = values.get(1), values.get(2)
                    if label is not None and description is not None:
                        label_value = label["cell_value_resolved"]
                        for condition_sheet in views:
                            key = (condition_sheet, label_value)
                            legends.setdefault(key, []).append(
                                (
                                    record,
                                    label["source_cell"],
                                    description["source_cell"],
                                    description["cell_value_resolved"],
                                )
                            )
                pool = block["sheet_pools"].get(sheet)
                if pool:
                    _publisher_row(compiler, source, sheet, record, cells, headers[sheet], pool)
    _publisher_condition_assertions(compiler, headers_by_label, legends)
