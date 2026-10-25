"""Synthetic software fixtures for source layout, ambiguity and public query fidelity."""

import io
import json
import zipfile

import openpyxl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from hitlist import intact_degradomics as source
from hitlist.evidence_package import package_family
from hitlist.evidence_store import EvidenceStore
from hitlist.intact_degradomics_workbook import workbook_rows
from hitlist.material_context import links_with_minimum_precision
from hitlist.provenance import file_digest

HEADERS = [
    "Mass ID",
    "Input Monoisotopic Mass y-ions",
    "Input Monoisotopic Mass b-/by-ions -H2O",
    "Sum Intensity",
    None,
    None,
    "Mass ID",
    "Frag_number_ClipsMS",
    "Frag Type",
    "Localized Mod",
    "Unlocalized Mod",
    "Term Mod",
    "Observed Mass",
    "Theoretical Mass",
    "Start AA",
    "End AA",
    "Error",
    "Sequence",
    "Intensity",
    "Formula",
    "MultiMass Assignment",
]
PUBLISHED = [
    "Sample ID",
    "Frag Type",
    "Observed Mass",
    "Theoretical Mass",
    "Start AA",
    "End AA",
    "Error",
    "Sequence",
    "Intensity",
    "Formula",
]


def search_rows(
    *, shift=False, missing_ids=False, abbreviated=False, intensity=10, two_masses=False
):
    headers = HEADERS.copy()
    if abbreviated:
        headers[-1] = "MultiMass Ass"
    result = [headers]
    for i in range(1, 3 if two_masses else 2):
        observed = 483.0 + i - 1
        left = [i, observed + 18.010565, observed, intensity]
        if missing_ids:
            left = [*left[1:], None]
        right = [
            str(i),
            i,
            "B Fragment",
            0,
            "None",
            0,
            observed,
            observed,
            1,
            4,
            0,
            "ACDE",
            intensity,
            "source formula",
            1,
        ]
        if shift:
            right.insert(1, f"R1_40min_{i}")
        result.append([*left, None, None, *right])
    return result


def write_workbook(path, sheets):
    workbook = openpyxl.Workbook()
    workbook.remove(workbook.active)
    for name, rows in sheets.items():
        sheet = workbook.create_sheet(name)
        for row in rows:
            sheet.append(row)
    workbook.save(path)


@pytest.fixture
def original(tmp_path, monkeypatch):
    assets = tmp_path / "assets"
    assets.mkdir()
    contract = json.loads(source.CONTRACT.read_text())
    contract["parent_sequence"] = "ACDEFGHIK"
    contract["assertions"] = [
        {
            "predicate": "replicate_independence",
            "value": "synthetic fixture unknown",
            "status": "unresolved",
            "locator": "fixture",
        }
    ]
    first, second, third = contract["search_workbooks"]
    write_workbook(assets / first, {"20S_AlphaSyn_0min_R1": search_rows()})
    write_workbook(
        assets / second,
        {
            "20S_AlphaSyn_0min_R1": search_rows(),
            "20S_AlphaSyn_40min_R1": search_rows(shift=True, intensity=20),
            "20S_AlphaSyn_100min_R1": search_rows(two_masses=True),
            "20S_AlphaSyn_120min_R3": search_rows(abbreviated=True),
        },
    )
    write_workbook(
        assets / third, {"Brain_20S_aSyn_0min_2": search_rows(missing_ids=True, intensity=30)}
    )
    supplements = contract["supplement_workbooks"]
    for name in supplements:
        write_workbook(assets / name, {"Content": [["Table content"], ["fixture"]]})
    write_workbook(
        assets / "pnas.2419607122.sd01.xlsx",
        {"mz_FF90_Rep1": [["Charge State", "Charge Normalized Intensity", "Fit %"], [2, 40, 95]]},
    )
    write_workbook(
        assets / "pnas.2419607122.sd02.xlsx",
        {
            "aSyn+20S_0min": [
                ["Sequence", "Intensity_Rep1", "Intensity_Rep2", "Intensity_Rep3"],
                ["ACDE", 10, 0, "#N/A"],
            ]
        },
    )
    write_workbook(
        assets / "pnas.2419607122.sd04.xlsx",
        {
            "Kinetics": [
                PUBLISHED,
                ["0min_Rep1", "N-term", 483, 483, 1, 4, 0, "ACDE", 10, "source formula"],
                ["40min_Rep1", "N-term", 483, 483, 1, 4, 0, "ACDE", 20, "source formula"],
                ["100min_Rep1", "N-term", 484, 484, 1, 4, 0, "ACDE", 999, "source formula"],
            ]
        },
    )
    write_workbook(
        assets / "pnas.2419607122.sd05.xlsx",
        {
            "Organs": [
                PUBLISHED,
                ["Brain_0min_2", "N-term", 483, 483, 1, 4, 0, "ACDE", 30, "source formula"],
            ]
        },
    )
    raw_tails = [
        "AlphaSyn_20S_0min_R1",
        "AlphaSyn_20S_40min_R1",
        "AlphaSyn_20S_100min_R1",
        "AlphaSyn_20S_120min_R3",
        "AlphaSyn_Brain20S_0min_R2",
        "AlphaSyn_20S_60min_R1",
    ]
    inventory = [
        {"fileName": "F1_F2_SF3_" + tail + ".RAW", "fileCategory": {"value": "RAW"}}
        for tail in raw_tails
    ]
    (assets / "project.json").write_text('{"source": "synthetic software fixture"}')
    (assets / "files-project-route.json").write_text(json.dumps(inventory))
    (assets / "files-page1.json").write_text("[]")
    contract["assets"] = [
        {
            "filename": p.name,
            "source_url": "https://example.invalid/" + p.name,
            "role": "synthetic_software_fixture",
            "license_status": "test_fixture",
            **file_digest(p),
        }
        for p in sorted(assets.iterdir())
    ]
    contract["source_registry"] = {"pxd058121": {"source_role": "synthetic_software_fixture"}}
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps(contract))
    monkeypatch.setattr(source, "CONTRACT", contract_path)
    return assets, tmp_path / "package", contract


def build(original):
    assets, package, _ = original
    source.write_intact_degradomics_package(assets, package)
    return EvidenceStore.open(package)


def query(store, name, **kwargs):
    return store.query("intact_" + name, limit=None, **kwargs).table.to_pylist()


def test_source_layout_and_missing_ids_remain_qualified(original):
    store = build(original)
    assert package_family(store.manifest) == "intact_degradomics"
    assignments = query(store, "assignments")
    assert len(assignments) == 7
    assert all(x["span_status"] == "matches_source_substrate" for x in assignments)
    shifted = [x for x in assignments if x["source_extra_identifier"]]
    assert len(shifted) == 1 and shifted[0]["source_extra_identifier"] == "R1_40min_1"
    missing = [
        x for x in assignments if x["mass_link_status"] == "input_ids_not_supplied_no_inferred_link"
    ]
    assert len(missing) == 1 and missing[0]["input_mass_id"] is None
    masses = query(store, "input_masses", where={"input_identity_status": "absent_not_inferred"})
    assert len(masses) == 1 and masses[0]["native_mass_id_raw"] is None
    joined = query(
        store,
        "assignments",
        select=["peptide_sequence", "quantity.quantitative_value"],
        joins=["quantity"],
    )
    assert len(joined) == 7 and any(x["quantity.quantitative_value"] == 30 for x in joined)
    linked = query(
        store,
        "assignments",
        select=["assignment_id", "acquisition.raw_file", "input_mass.native_mass_id_raw"],
        joins=["acquisition", "input_mass"],
    )
    assert len(linked) == 7 and all(x["acquisition.raw_file"] for x in linked)
    assert sum(x["input_mass.native_mass_id_raw"] is None for x in linked) == 1


def test_views_and_publisher_conflicts_do_not_choose_by_quantity(original):
    store = build(original)
    comparisons = query(store, "reconciliations")
    reuse = [
        x for x in comparisons if x["assertion_predicate"] == "repeated_acquisition_source_views"
    ]
    assert len(reuse) == 1 and reuse[0]["assertion_status"] == "same_values"
    publisher = [
        x for x in comparisons if x["assertion_predicate"] == "publisher_assignment_reconciliation"
    ]
    assert len(publisher) == 4
    conflicts = [
        x for x in publisher if x["assertion_status"] == "identity_matches_values_conflict"
    ]
    assert len(conflicts) == 1
    detail = json.loads(conflicts[0]["details_json"])
    assert len(detail["structural_candidates"]) == 2 and len(detail["resolved_candidates"]) == 1
    assert detail["mismatched_fields"] == ["intensity"]
    assert detail["intensity_used_to_choose_identity"] is False
    assert any(x["quantitative_value"] == 999 for x in query(store, "quantities"))


def test_zero_error_and_unknown_material_identity_are_not_filled(original):
    store = build(original)
    quantities = query(store, "quantities")
    assert any(x["quantitative_value"] == 0 and x["value_status"] == "zero" for x in quantities)
    assert any(
        x["quantitative_raw"] == "#N/A" and x["quantitative_value"] is None for x in quantities
    )
    contexts = query(store, "material_contexts")
    assert all(x["specimen_id"] is None and x["aliquot_id"] is None for x in contexts)
    assert links_with_minimum_precision(query(store, "material_links"), "exact_specimen") == ()
    acquisitions = query(store, "acquisitions")
    assert len(acquisitions) == 6 and not any("Spleen" in x["raw_file"] for x in acquisitions)
    assert (
        store.catalog()["source_registry"]["pxd058121"]["source_role"]
        == "synthetic_software_fixture"
    )


def test_replay_detects_changed_quantity_even_with_refreshed_manifest(original):
    store = build(original)
    package = store.directory
    path = package / source.ROOT / "typed/quantities.parquet"
    table = pq.read_table(path)
    rows = table.to_pylist()
    rows[0]["quantitative_value"] += 1
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), path)
    manifest = json.loads((package / "manifest.json").read_text())
    manifest["files"] = source._inventory(package)
    (package / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="source replay differs"):
        EvidenceStore.open(package)


def test_pinned_original_hash_and_no_overwrite(original):
    assets, package, _ = original
    store = build(original)
    with pytest.raises(FileExistsError):
        source.write_intact_degradomics_package(assets, package)
    (assets / "project.json").write_text("{}")
    with pytest.raises(ValueError, match="original source differs"):
        source.write_intact_degradomics_package(assets, package.parent / "changed")
    assert store.manifest["reconciliation"]["n_quantities_rows"] > 0


def test_literal_shared_formulas_and_cached_values(tmp_path):
    path = tmp_path / "shared.xlsx"
    write_workbook(path, {"Source": [["Mass", "Adjusted"], [500, 482], [600, 582]]})
    output = io.BytesIO()
    with zipfile.ZipFile(path) as original, zipfile.ZipFile(output, "w") as changed:
        for name in original.namelist():
            data = original.read(name)
            if name == "xl/worksheets/sheet1.xml":
                data = data.replace(
                    b'<c r="B2" t="n"><v>482</v></c>',
                    b'<c r="B2" t="n"><f t="shared" ref="B2:B3" si="0">A2-18</f><v>482</v></c>',
                )
                data = data.replace(
                    b'<c r="B3" t="n"><v>582</v></c>',
                    b'<c r="B3" t="n"><f t="shared" si="0"/><v>582</v></c>',
                )
            changed.writestr(name, data)
    path.write_bytes(output.getvalue())
    cells = [
        c for _, _, row in workbook_rows(path) for c in row if c["source_cell"] in {"B2", "B3"}
    ]
    assert len(cells) == 2
    assert cells[0]["cell_formula_raw"] == "A2-18"
    assert cells[1]["cell_formula_raw"] is None and cells[1]["cell_value_raw"] == "582"
    assert json.loads(cells[1]["formula_attributes_json"]) == {"t": "shared", "si": "0"}
