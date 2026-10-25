"""Read calibration workbooks without interpreting amounts, identities or formulas.

Numeric literals deliberately remain strings. This prevents source identifiers or
decimal quantities from being silently rounded before their meaning is established.
The original XLSX asset remains authoritative for lexical XML and workbook styles.
"""

from __future__ import annotations

import posixpath
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile

NS = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
RELATION_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
RELATION_ID = f"{{{RELATION_NS}}}id"
PACKAGE_RELATION_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
MAX_EXPANDED_BYTES = 64 * 1024**2


@dataclass(frozen=True)
class SourceCell:
    coordinate: str
    excel_type: str
    value_literal: str | None
    raw_cached_value: str | None
    value_present: bool
    has_formula: bool
    formula: str | None
    attributes: dict[str, str]
    formula_attributes: dict[str, str]
    source_xml: str


@dataclass(frozen=True)
class SourceSheet:
    name: str
    sheet_id: str
    relationship_id: str
    source_member: str
    state: str
    dimension: str | None
    merged_ranges: tuple[str, ...]
    cells: dict[str, SourceCell]


def _read_xml(archive, member, expected_tag):
    root = ET.fromstring(archive.read(member))
    if root.tag != expected_tag:
        raise ValueError(f"Unexpected XML root in {member}: {root.tag!r}")
    return root


def _one_child(node, tag, *, required=False):
    children = node.findall(f"s:{tag}", NS)
    if len(children) > 1 or (required and not children):
        raise ValueError(f"Expected {'exactly' if required else 'at most'} one {tag} element")
    return children[0] if children else None


def _display_text(node):
    """Displayed text excludes optional phonetic annotations, retained in XML."""
    parts = []
    for child in node:
        if child.tag == f"{{{NS['s']}}}t":
            parts.append(child.text or "")
        elif child.tag == f"{{{NS['s']}}}r":
            parts.extend(part.text or "" for part in child.findall("s:t", NS))
    return "".join(parts)


def _cell(node, strings):
    coordinate = node.attrib.get("r", "")
    if not re.fullmatch(r"[A-Z]+[1-9][0-9]*", coordinate):
        raise ValueError(f"Invalid source cell coordinate: {coordinate!r}")
    kind = node.attrib.get("t", "n")
    if kind not in {"n", "s", "inlineStr", "b", "str", "e", "d"}:
        raise ValueError(f"Unsupported Excel cell type: {kind!r}")
    value_node, inline_node = _one_child(node, "v"), _one_child(node, "is")
    formula = _one_child(node, "f")
    if (kind == "inlineStr" and value_node is not None) or (
        kind != "inlineStr" and inline_node is not None
    ):
        raise ValueError(f"Conflicting source cell value encoding at {coordinate}")
    raw = value_node.text if value_node is not None else None
    if kind == "inlineStr":
        literal = _display_text(inline_node) if inline_node is not None else None
        present = inline_node is not None
    elif kind == "s" and raw is not None:
        if not re.fullmatch(r"[0-9]+", raw) or int(raw) >= len(strings):
            raise ValueError(f"Invalid shared-string reference at {coordinate}")
        literal, present = strings[int(raw)], True
    else:
        literal, present = raw, value_node is not None
    return SourceCell(
        coordinate=coordinate,
        excel_type=kind,
        value_literal=literal,
        raw_cached_value=raw,
        value_present=present,
        has_formula=formula is not None,
        formula=formula.text if formula is not None else None,
        attributes=dict(node.attrib),
        formula_attributes=dict(formula.attrib) if formula is not None else {},
        source_xml=ET.tostring(node, encoding="unicode"),
    )


def read_source_workbook(path: str | Path) -> tuple[SourceSheet, ...]:
    """Return source sheets/cells, including explicitly stored empty cells.

    Missing coordinates are absent from ``cells``. ``value_present`` distinguishes an
    empty cached-value element from no value element. Formulas are never evaluated.
    No external relationships, filesystem extraction or source mutation is performed.
    """
    with ZipFile(path) as archive:
        members = archive.infolist()
        if len({item.filename for item in members}) != len(members):
            raise ValueError("Ambiguous duplicate ZIP members")
        if sum(item.file_size for item in members) > MAX_EXPANDED_BYTES:
            raise ValueError("Expanded source workbook exceeds size limit")
        strings = (
            [
                _display_text(node)
                for node in _read_xml(archive, "xl/sharedStrings.xml", f"{{{NS['s']}}}sst").findall(
                    "s:si", NS
                )
            ]
            if "xl/sharedStrings.xml" in archive.namelist()
            else []
        )
        relations = {}
        for node in _read_xml(
            archive,
            "xl/_rels/workbook.xml.rels",
            f"{{{PACKAGE_RELATION_NS}}}Relationships",
        ):
            if node.tag != f"{{{PACKAGE_RELATION_NS}}}Relationship":
                raise ValueError("Unexpected workbook relationship element")
            key = node.attrib["Id"]
            if key in relations:
                raise ValueError("Ambiguous duplicate workbook relationships")
            relations[key] = node.attrib
        sheets, names, sheet_ids, source_targets = [], set(), set(), set()
        workbook = _read_xml(archive, "xl/workbook.xml", f"{{{NS['s']}}}workbook")
        sheet_container = _one_child(workbook, "sheets", required=True)
        for node in sheet_container.findall("s:sheet", NS):
            name = node.attrib["name"]
            if name in names:
                raise ValueError("Ambiguous duplicate worksheet names")
            names.add(name)
            sheet_id = node.attrib["sheetId"]
            if sheet_id in sheet_ids:
                raise ValueError("Ambiguous duplicate worksheet IDs")
            sheet_ids.add(sheet_id)
            relationship_id = node.attrib[RELATION_ID]
            relation = relations[relationship_id]
            if relation.get("Type") != f"{RELATION_NS}/worksheet":
                raise ValueError(f"Unsupported worksheet relationship type for {name!r}")
            target = relation["Target"]
            if relation.get("TargetMode") == "External" or ":" in target or "\\" in target:
                raise ValueError("External or invalid worksheet target")
            member = posixpath.normpath(
                target.lstrip("/") if target.startswith("/") else posixpath.join("xl", target)
            )
            if not member.startswith("xl/") or member in source_targets:
                raise ValueError("Escaping or duplicate worksheet target")
            source_targets.add(member)
            worksheet = _read_xml(archive, member, f"{{{NS['s']}}}worksheet")
            sheet_data = _one_child(worksheet, "sheetData", required=True)
            cells = {}
            for element in sheet_data.findall("s:row/s:c", NS):
                cell = _cell(element, strings)
                if cell.coordinate in cells:
                    raise ValueError("Ambiguous duplicate source cell coordinates")
                cells[cell.coordinate] = cell
            dimension = _one_child(worksheet, "dimension")
            merge_cells = _one_child(worksheet, "mergeCells")
            sheets.append(
                SourceSheet(
                    name=name,
                    sheet_id=sheet_id,
                    relationship_id=relationship_id,
                    source_member=member,
                    state=node.attrib.get("state", "visible"),
                    dimension=dimension.attrib.get("ref") if dimension is not None else None,
                    merged_ranges=tuple(
                        item.attrib["ref"]
                        for item in (
                            merge_cells.findall("s:mergeCell", NS)
                            if merge_cells is not None
                            else []
                        )
                    ),
                    cells=cells,
                )
            )
    return tuple(sheets)
