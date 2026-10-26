"""Source-format tests only; no biological or calibration training examples."""

from pathlib import Path
from zipfile import ZipFile

import pytest

from hitlist import nontryptic_workbooks as source

NS = source.NS["s"]
REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"


def workbook(
    tmp_path,
    cells,
    *,
    shared=None,
    target="worksheets/sheet1.xml",
    external=False,
    extra_sheets="",
    extra_relations="",
):
    path = tmp_path / "fixture.xlsx"
    with ZipFile(path, "w") as archive:
        archive.writestr(
            "xl/workbook.xml",
            f'<workbook xmlns="{NS}" xmlns:r="{REL}"><sheets>'
            '<sheet name="Source" sheetId="1" state="hidden" r:id="rId1"/>'
            f"{extra_sheets}</sheets></workbook>",
        )
        mode = ' TargetMode="External"' if external else ""
        archive.writestr(
            "xl/_rels/workbook.xml.rels",
            f'<Relationships xmlns="{source.PACKAGE_RELATION_NS}">'
            f'<Relationship Id="rId1" Type="{REL}/worksheet" Target="{target}"{mode}/>'
            f"{extra_relations}</Relationships>",
        )
        archive.writestr(
            "xl/worksheets/sheet1.xml",
            f'<worksheet xmlns="{NS}"><dimension ref="A1:Z9"/><sheetData>'
            f'<row r="1">{cells}</row></sheetData>'
            '<mergeCells><mergeCell ref="C3:E3"/></mergeCells></worksheet>',
        )
        if shared is not None:
            archive.writestr("xl/sharedStrings.xml", f'<sst xmlns="{NS}">{shared}</sst>')
    return path


def rewrite_member(path, member, transform):
    """Change one tiny synthetic fixture member without duplicate ZIP entries."""
    with ZipFile(path) as archive:
        parts = {name: archive.read(name) for name in archive.namelist()}
    parts[member] = transform(parts[member].decode()).encode()
    with ZipFile(path, "w") as archive:
        for name, content in parts.items():
            archive.writestr(name, content)
    return path


def test_nontryptic_workbooks_preserve_values_and_missingness(tmp_path):
    path = workbook(
        tmp_path,
        '<c r="A1"><v>9007199254740993</v></c>'
        '<c r="B1"><v>0.10000000000000001</v></c>'
        '<c r="C1"><v>0</v></c><c r="D1" s="4"/>'
        '<c r="E1"><v/></c><c r="F1" t="inlineStr"><is/></c>'
        '<c r="G1" t="b"><v>0</v></c><c r="H1" t="e"><v>#N/A</v></c>'
        '<c r="I1" t="d"><v>2021-01-02T00:00:00</v></c>',
    )
    original = path.read_bytes()
    (sheet,) = source.read_source_workbook(path)
    assert (sheet.name, sheet.state, sheet.dimension, sheet.merged_ranges) == (
        "Source",
        "hidden",
        "A1:Z9",
        ("C3:E3",),
    )
    assert sheet.cells["A1"].value_literal == "9007199254740993"
    assert sheet.cells["B1"].value_literal == "0.10000000000000001"
    assert sheet.cells["C1"].value_literal == "0"
    assert not sheet.cells["D1"].value_present
    assert sheet.cells["D1"].attributes["s"] == "4"
    assert sheet.cells["E1"].value_present and sheet.cells["E1"].value_literal is None
    assert sheet.cells["F1"].value_present and sheet.cells["F1"].value_literal == ""
    assert sheet.cells["G1"].excel_type == "b" and sheet.cells["G1"].value_literal == "0"
    assert sheet.cells["H1"].excel_type == "e" and sheet.cells["H1"].value_literal == "#N/A"
    assert sheet.cells["I1"].excel_type == "d"
    assert "J1" not in sheet.cells
    assert path.read_bytes() == original


def test_nontryptic_workbooks_rich_text_and_formula_caches(tmp_path):
    path = workbook(
        tmp_path,
        '<c r="A1" t="s"><v>0</v></c>'
        '<c r="B1" t="inlineStr"><is><t>V</t><r><t>[+6]</t></r>'
        "<rPh><t>phonetic</t></rPh></is></c>"
        '<c r="C1"><f t="shared" si="0">1/0</f><v>12.3</v></c>'
        '<c r="D1"><f t="shared" si="0"/></c>',
        shared="<si><r><t>ALD</t></r><r><t>GG</t></r><rPh><t>annotation</t></rPh></si>",
    )
    (sheet,) = source.read_source_workbook(path)
    assert sheet.cells["A1"].value_literal == "ALDGG"
    assert sheet.cells["A1"].raw_cached_value == "0"
    assert sheet.cells["B1"].value_literal == "V[+6]"
    assert "phonetic" in sheet.cells["B1"].source_xml
    assert sheet.cells["C1"].formula == "1/0"
    assert sheet.cells["C1"].value_literal == "12.3"
    assert sheet.cells["D1"].has_formula and sheet.cells["D1"].formula is None
    assert sheet.cells["D1"].formula_attributes == {"t": "shared", "si": "0"}
    assert not sheet.cells["D1"].value_present


@pytest.mark.parametrize(
    "cells,match",
    [
        ('<c r="A1"/><c r="A1"/>', "duplicate source cell"),
        ('<c r="A0"/>', "Invalid source cell"),
        ('<c r="A1" t="mystery"/>', "Unsupported Excel"),
        ('<c r="A1" t="s"><v>-1</v></c>', "shared-string reference"),
        ('<c r="A1" t="s"><v>8</v></c>', "shared-string reference"),
        ('<c r="A1"><v>1</v><v>2</v></c>', "one v element"),
        ('<c r="A1"><f>1</f><f>2</f><v>1</v></c>', "one f element"),
        ('<c r="A1" t="inlineStr"><is/><is/></c>', "one is element"),
        ('<c r="A1" t="inlineStr"><v>1</v><is/></c>', "Conflicting source cell"),
        ('<c r="A1"><is><t>x</t></is></c>', "Conflicting source cell"),
    ],
)
def test_nontryptic_workbooks_reject_ambiguous_cell_encodings(tmp_path, cells, match):
    with pytest.raises(ValueError, match=match):
        source.read_source_workbook(workbook(tmp_path, cells, shared="<si><t>x</t></si>"))


@pytest.mark.parametrize(
    "target,external",
    [
        ("../../outside.xml", False),
        ("https://example.com/sheet.xml", True),
        ("worksheets\\sheet1.xml", False),
    ],
)
def test_nontryptic_workbooks_reject_external_or_escaping_targets(tmp_path, target, external):
    with pytest.raises(ValueError, match="worksheet target"):
        source.read_source_workbook(workbook(tmp_path, "", target=target, external=external))


def test_nontryptic_workbooks_reject_duplicate_sheet_relationships(tmp_path):
    path = workbook(tmp_path, "", extra_relations='<Relationship Id="rId1" Target="x.xml"/>')
    with pytest.raises(ValueError, match="duplicate workbook relationships"):
        source.read_source_workbook(path)


def test_nontryptic_workbooks_reject_duplicate_sheet_names_and_targets(tmp_path):
    path = workbook(tmp_path, "", extra_sheets='<sheet name="Source" sheetId="2" r:id="rId1"/>')
    with pytest.raises(ValueError, match="duplicate worksheet names"):
        source.read_source_workbook(path)
    path = workbook(tmp_path, "", extra_sheets='<sheet name="Other" sheetId="2" r:id="rId1"/>')
    with pytest.raises(ValueError, match="duplicate worksheet target"):
        source.read_source_workbook(path)
    path = workbook(tmp_path, "", extra_sheets='<sheet name="Other" sheetId="1" r:id="rId1"/>')
    with pytest.raises(ValueError, match="duplicate worksheet IDs"):
        source.read_source_workbook(path)


def test_nontryptic_workbooks_limit_expanded_archives(tmp_path, monkeypatch):
    path = workbook(tmp_path, '<c r="A1"><v>42</v></c>')
    monkeypatch.setattr(source, "MAX_EXPANDED_BYTES", 1)
    with pytest.raises(ValueError, match="size limit"):
        source.read_source_workbook(path)


def test_nontryptic_workbooks_reject_duplicate_zip_members(tmp_path):
    path = workbook(tmp_path, "")
    with ZipFile(path, "a") as archive, pytest.warns(UserWarning, match="Duplicate name"):
        archive.writestr("xl/worksheets/sheet1.xml", "<worksheet/>")
    with pytest.raises(ValueError, match="duplicate ZIP members"):
        source.read_source_workbook(Path(path))


@pytest.mark.parametrize(
    "member,root",
    [
        ("xl/workbook.xml", "workbook"),
        ("xl/sharedStrings.xml", "sst"),
        ("xl/worksheets/sheet1.xml", "worksheet"),
        ("xl/_rels/workbook.xml.rels", "Relationships"),
    ],
)
def test_nontryptic_workbooks_reject_wrong_document_roots(tmp_path, member, root):
    path = workbook(tmp_path, "", shared="<si><t>x</t></si>")
    rewrite_member(path, member, lambda text: text.replace(root, "wrongRoot"))
    with pytest.raises(ValueError, match="Unexpected XML root"):
        source.read_source_workbook(path)


@pytest.mark.parametrize(
    "member,container",
    [("xl/workbook.xml", "sheets"), ("xl/worksheets/sheet1.xml", "sheetData")],
)
@pytest.mark.parametrize("change", ["duplicate", "missing"])
def test_nontryptic_workbooks_reject_ambiguous_containers(tmp_path, member, container, change):
    path = workbook(tmp_path, "")
    if change == "duplicate":
        rewrite_member(
            path,
            member,
            lambda text: text.replace(f"</{container}>", f"</{container}><{container}/>"),
        )
    else:
        rewrite_member(path, member, lambda text: text.replace(container, "unknownContainer"))
    with pytest.raises(ValueError, match=f"exactly one {container} element"):
        source.read_source_workbook(path)


def test_nontryptic_workbooks_reject_non_worksheet_relationship(tmp_path):
    path = workbook(tmp_path, "")
    rewrite_member(
        path,
        "xl/_rels/workbook.xml.rels",
        lambda text: text.replace(f"{REL}/worksheet", f"{REL}/chart"),
    )
    with pytest.raises(ValueError, match="Unsupported worksheet relationship type"):
        source.read_source_workbook(path)


def test_nontryptic_workbooks_empty_sheet_and_absolute_target(tmp_path):
    (sheet,) = source.read_source_workbook(
        workbook(tmp_path, "", target="/xl/worksheets/sheet1.xml")
    )
    assert sheet.cells == {}
    assert sheet.source_member == "xl/worksheets/sheet1.xml"
