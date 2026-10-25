"""Sparse original OOXML cells without Excel date conversion or recalculation."""

from __future__ import annotations

import json
import posixpath
import re
import zipfile
from xml.etree import ElementTree as ET

NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
REL = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"


def column_index(coordinate):
    match = re.fullmatch(r"([A-Z]+)([0-9]+)", coordinate)
    if match is None:
        raise ValueError("Invalid original worksheet coordinate")
    result = 0
    for char in match[1]:
        result = result * 26 + ord(char) - 64
    return result


def workbook_rows(path):
    """Retain all explicit cells, raw formula attributes and cached values."""
    with zipfile.ZipFile(path) as archive:
        strings = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            strings = ["".join(t.text or "" for t in n.iter(NS + "t")) for n in root]
        relationships = {
            n.get("Id"): n.get("Target")
            for n in ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        }
        book = ET.fromstring(archive.read("xl/workbook.xml"))
        for sheet in book.find(NS + "sheets"):
            target = relationships[sheet.get(REL)]
            target = target.lstrip("/") if target.startswith("/") else posixpath.join("xl", target)
            with archive.open(posixpath.normpath(target)) as stream:
                for _, row in ET.iterparse(stream, events=("end",)):
                    if row.tag != NS + "row":
                        continue
                    cells = []
                    for cell in row.findall(NS + "c"):
                        kind, raw = cell.get("t", "n"), cell.findtext(NS + "v")
                        if kind == "s" and raw is not None:
                            resolved = strings[int(raw)]
                        elif kind == "inlineStr":
                            inline = cell.find(NS + "is")
                            resolved = (
                                None
                                if inline is None
                                else "".join(t.text or "" for t in inline.iter(NS + "t"))
                            )
                        else:
                            resolved = raw
                        formula = cell.find(NS + "f")
                        cells.append(
                            {
                                "source_cell": cell.get("r"),
                                "cell_type_raw": kind,
                                "cell_style_raw": cell.get("s"),
                                "cell_value_raw": raw,
                                "cell_value_resolved": resolved,
                                "cell_formula_raw": None if formula is None else formula.text,
                                "formula_attributes_json": None
                                if formula is None
                                else json.dumps(
                                    dict(formula.attrib), sort_keys=True, separators=(",", ":")
                                ),
                                "source_row_index": int(row.get("r")),
                                "source_column_index": column_index(cell.get("r")),
                            }
                        )
                    yield sheet.get("name"), int(row.get("r")), cells
                    row.clear()
