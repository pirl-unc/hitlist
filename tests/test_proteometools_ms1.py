"""Small synthetic file fixtures test curation semantics, never biological accuracy."""

import csv
import io
import json
import zipfile
from copy import deepcopy

import openpyxl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from hitlist import proteometools_ms1 as source
from hitlist import proteometools_response as response
from hitlist.evidence_recipe import RecipeError, materialize_recipe, verify_recipe_package
from hitlist.evidence_store import EvidenceStore
from hitlist.provenance import file_digest


def tsv(headers, rows):
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, delimiter="\t")
    writer.writerow(headers)
    writer.writerows(rows)
    return stream.getvalue()


def repin(directory, receipts):
    return [{**r, **file_digest(directory / r["filename"])} for r in receipts]


def change_member(directory, receipt, name, mutate):
    path = directory / receipt["filename"]
    with zipfile.ZipFile(path) as archive:
        contents = {name: archive.read(name) for name in archive.namelist()}
    contents[name] = mutate(contents[name].decode()).encode()
    with zipfile.ZipFile(path, "w") as archive:
        for name, value in contents.items():
            archive.writestr(name, value)


@pytest.fixture
def original(tmp_path, monkeypatch):
    directory = tmp_path / "original"
    directory.mkdir()
    (directory / "article-2021.xml").write_text("<article>test fixture only</article>")
    (directory / "PXD021013-sdrf.tsv").write_text(
        "source name\tmodification\tmodification\ncell\tCAM\tOxidation\n"
    )
    book = openpyxl.Workbook()
    book.remove(book.active)
    for sheet_name in source.MAPPING_SHEETS:
        sheet = book.create_sheet(sheet_name)
        sheet.append(["Pool name", "Sequence"])
        if sheet_name == "HLA Class I":
            for pool in ("TUM_HLA_1", "TUM_HLA_2"):
                for seq in ("ACDE", "AAAA", "CCCC", "DDDD", "EEEE", "GGGG", "YYYY"):
                    sheet.append([pool, seq])
        if sheet_name == "HLA Class II":
            sheet.append(["TUM_HLA2_1", "WWWW"])
    book.save(directory / "article-2021-planned-peptides.xlsx")
    index, receipts = [], []
    for number, pool in enumerate(("TUM_HLA_1", "TUM_HLA_2")):
        name = pool + "_01_01_DDA.zip"
        filename = f"archive-{number}.zip"
        raw = f"raw-{number}"
        index.extend(
            [
                [str(number), raw + ".raw", source.BASE_URL + raw + ".raw", "RAW", "-"],
                [str(number + 10), name, source.BASE_URL + name, "SEARCH", str(number)],
            ]
        )
        headers = [
            "Sequence",
            "id",
            "Intensity",
            "Intensity experiment",
            "PEP",
            "Reverse",
            "Potential contaminant",
            "Proteins",
        ]
        rows = [
            [seq, str(i), intensity, intensity, ".001", "", "+" if seq == "EEEE" else "", pool]
            for i, (seq, intensity) in enumerate(
                [
                    ("ACDE", "100"),
                    ("AAAA", "10000"),
                    ("CCCC", "0"),
                    ("DDDD", ""),
                    ("EEEE", "40"),
                    ("FFFF", "1000"),
                    ("GGGG", "7"),
                ]
            )
        ]
        forms = [
            [*row, row[1], "Oxidation (M)" if row[0] == "GGGG" else "Unmodified", "2;3", raw]
            for row in rows
        ]
        with zipfile.ZipFile(directory / filename, "w") as archive:
            archive.writestr("peptides.txt", tsv(headers, rows))
            archive.writestr(
                "modificationSpecificPeptides.txt",
                tsv([*headers, "Peptide ID", "Modifications", "Charges", "Raw file"], forms),
            )
            archive.writestr(
                "summary.txt",
                tsv(
                    ["Raw file", "Experiment", "Enzyme", "Enzyme mode", "Variable modifications"],
                    [[raw, "experiment", "", "Unspecific", "Oxidation (M)"]],
                ),
            )
            archive.writestr(
                "parameters.txt",
                tsv(
                    ["Parameter", "Value"],
                    [["Version", "1.5.3.30"], ["Fixed modifications", "Carbamidomethyl (C)"]],
                ),
            )
            archive.writestr("unprojected-original.txt", "retained original bytes")
        receipts.append({"filename": filename, "source_name": name})
    (directory / "PXD021013-README.txt").write_text(
        tsv(["ID", "NAME", "URI", "TYPE", "MAPPINGS"], index)
    )
    core = tmp_path / "core.json"
    specs = [
        {
            "filename": p.name,
            "role": "fixture",
            "source_url": "https://example.org/fixture/" + p.name,
            **file_digest(p),
        }
        for p in sorted(directory.iterdir())
        if p.suffix != ".zip"
    ]
    core.write_text(json.dumps({"assets": specs}))
    monkeypatch.setattr(source, "CORE", core)
    return directory, repin(directory, receipts)


def build(original, tmp_path):
    directory, receipts = original
    package = tmp_path / "package"
    source.write_proteometools_ms1_package(directory, package, archives=receipts)
    return package


def read_tables(package):
    return {
        name: pq.read_table(package / source.ROOT / "typed" / f"{name}.parquet").to_pylist()
        for name in response.TABLE_BINDINGS
    }


def rehash(package):
    path = package / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["files"] = {
        str(p.relative_to(package)): file_digest(p)
        for p in package.rglob("*")
        if p.is_file() and p != path
    }
    path.write_text(json.dumps(manifest))


def test_roundtrip_context_positional_metadata_and_quantities(original, tmp_path):
    package = build(original, tmp_path)
    store = EvidenceStore.open(package)
    description = store.catalog()
    assert (
        description["source_registry"]["pxd021013"]["source_role"] == "non_MHC_synthesis_pool_MS1"
    )
    assert set(store.bindings) == {"proteometools_" + name for name in source.SCHEMAS}
    assert any(
        r["contract"]["recipe_id"] == response.RECIPE_ID and r["status"] == "materialized"
        for r in description["registered_recipes"]
    )
    records = pq.read_table(package / source.ROOT / "typed/source_records.parquet").to_pylist()
    sdrf = [r for r in records if r["source_role"] == "SDRF"]
    assert json.loads(sdrf[0]["source_row_json"]) == {
        "headers": ["source name", "modification", "modification"],
        "cells": ["cell", "CAM", "Oxidation"],
    }
    rows = store.query(
        "proteometools_forms",
        select=["peptide_sequence", "value_status", "quantitative_raw"],
        limit=None,
    ).table.to_pylist()
    assert len(rows) == 14
    assert {r["value_status"] for r in rows} == {"positive", "zero", "missing"}
    assert {r["quantitative_raw"] for r in rows if r["peptide_sequence"] == "DDDD"} == {""}
    assert "cell" not in source.CONTEXT["material_type"]
    assert (
        pq.read_table(package / source.ROOT / "typed/runs.parquet").to_pylist()[0][
            "fixed_modifications_raw"
        ]
        == "Carbamidomethyl (C)"
    )
    with pytest.raises(FileExistsError):
        source.write_proteometools_ms1_package(original[0], package, archives=original[1])


def test_recipe_exclusions_centering_absence_and_groups(original, tmp_path):
    package = build(original, tmp_path)
    tables = read_tables(package)
    result = response.compile_response(tables)
    assert result["summary"]["n_admitted_forms"] == 4
    assert result["summary"]["n_dependence_components"] == 1
    assert sorted(r["target_centered_log10_signal"] for r in result["examples"]) == [
        -1.0,
        -1.0,
        1.0,
        1.0,
    ]
    assert {r["sequence"] for r in result["examples"]} == {"ACDE", "AAAA"}
    assert result["summary"]["planned_status_counts"]["unobserved_unknown_cause_not_negative"] == 2
    assert all(
        r["status"] == "unobserved_unknown_cause_not_negative"
        for r in result["planned_dispositions"]
        if r["sequence"] == "YYYY"
    )
    assert all(r["fixed_modifications_raw"] == "Carbamidomethyl (C)" for r in result["examples"])
    # Dependence groups do not change when quantities change or all quality gates fail.
    altered = deepcopy(tables)
    for row in altered["forms"]:
        row["pep"] = 0.9
    empty = response.compile_response(altered)
    assert not empty["examples"]
    assert empty["dependence_groups"] == result["dependence_groups"]
    # Every ambiguous unmodified form is excluded, not silently summed or picked.
    duplicate = deepcopy(tables["forms"][0])
    duplicate["measurement_id"] += ":second"
    tables["forms"].append(duplicate)
    ambiguous = response.compile_response(tables)
    assert ambiguous["summary"]["reason_counts"]["ambiguous_multiple_unmodified_forms"] == 2
    assert ambiguous["summary"]["n_admitted_forms"] == 3


def test_registered_materialization_and_rehashed_target_tamper(original, tmp_path):
    package = build(original, tmp_path)
    output = tmp_path / "recipe"
    recipe = {"schema_version": 1, "name": "fixture", "registered_recipe": response.RECIPE_ID}
    materialize_recipe(EvidenceStore.open(package), recipe, output)
    manifest = verify_recipe_package(output, source_package=package)
    assert manifest["coverage"]["n_admitted_forms"] == 4
    path = output / "examples.json"
    examples = json.loads(path.read_text())
    examples[0]["target_centered_log10_signal"] += 1
    path.write_text(json.dumps(examples))
    rehash(output)
    with pytest.raises(RecipeError, match="replay differs"):
        verify_recipe_package(output, source_package=package)


def test_source_replay_rejects_rehashed_typed_tamper(original, tmp_path):
    package = build(original, tmp_path)
    path = package / source.ROOT / "typed/forms.parquet"
    table = pq.read_table(path)
    rows = table.to_pylist()
    rows[0]["quantitative_value"] += 5
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), path)
    rehash(package)
    with pytest.raises(ValueError, match="typed rows differ"):
        source.verify_proteometools_ms1_package(package)


@pytest.mark.parametrize(
    "member,old,new,error",
    [
        ("summary.txt", "raw-0", "wrong-raw", "Summary RAW"),
        ("peptides.txt", "100\t100", "100\t200", "intensity disagree"),
        ("modificationSpecificPeptides.txt", "ACDE", "WWWW", "identity disagrees"),
    ],
)
def test_malformed_source_joins_fail_atomically(original, tmp_path, member, old, new, error):
    directory, receipts = original
    change_member(directory, receipts[0], member, lambda text: text.replace(old, new, 1))
    receipts = repin(directory, receipts)
    output = tmp_path / "rejected"
    with pytest.raises(ValueError, match=error):
        source.write_proteometools_ms1_package(directory, output, archives=receipts)
    assert not output.exists()
    assert not list(tmp_path.glob(".rejected-*"))


def test_subset_extension_and_receipt_rejection(original, tmp_path):
    directory, receipts = original
    output = tmp_path / "subset"
    source.write_proteometools_ms1_package(directory, output, archives=receipts[:1])
    manifest = source.verify_proteometools_ms1_package(output)
    assert manifest["reconciliation"]["n_runs_rows"] == 1
    full = build(original, tmp_path)
    assert source.verify_proteometools_ms1_package(full)["reconciliation"]["n_runs_rows"] == 2
    invalid = deepcopy(receipts)
    invalid[0]["filename"] = "../bad.zip"
    with pytest.raises(ValueError, match="Unsafe"):
        source.write_proteometools_ms1_package(directory, tmp_path / "unsafe", archives=invalid)
    invalid = deepcopy(receipts)
    invalid[0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="source mismatch"):
        source.write_proteometools_ms1_package(directory, tmp_path / "wrong", archives=invalid)


def test_summary_aggregates_are_preserved_but_not_runs(original, tmp_path):
    directory, receipts = original
    change_member(
        directory,
        receipts[0],
        "summary.txt",
        lambda text: text + "experiment\t\t\t\t\nTotal\t\t\t\t\n",
    )
    amended = (directory, repin(directory, receipts))
    package = build(amended, tmp_path)
    assert len(read_tables(package)["runs"]) == 2
    records = pq.read_table(package / source.ROOT / "typed/source_records.parquet").to_pylist()
    assert sum(row["source_role"] == "summary.txt" for row in records) == 4
    change_member(
        directory,
        receipts[0],
        "summary.txt",
        lambda text: text + "another-run\texperiment2\t\tUnspecific\t\n",
    )
    with pytest.raises(ValueError, match="Unsupported extra run"):
        source.write_proteometools_ms1_package(
            directory, tmp_path / "multi", archives=repin(directory, receipts)
        )


def test_repeated_parameters_preserve_rows_and_reject_required_conflicts(original, tmp_path):
    directory, receipts = original
    change_member(
        directory,
        receipts[0],
        "parameters.txt",
        lambda text: (
            text
            + "Decoy mode\trevert\nDecoy mode\trevert\nFixed modifications\tCarbamidomethyl (C)\n"
        ),
    )
    package = build((directory, repin(directory, receipts)), tmp_path)
    records = pq.read_table(package / source.ROOT / "typed/source_records.parquet").to_pylist()
    assert sum(row["source_role"] == "parameters.txt" for row in records) == 7
    change_member(
        directory, receipts[0], "parameters.txt", lambda text: text + "Fixed modifications\tOther\n"
    )
    with pytest.raises(ValueError, match="Conflicting required search parameter"):
        source.write_proteometools_ms1_package(
            directory, tmp_path / "conflict", archives=repin(directory, receipts)
        )


def test_shared_off_planned_background_does_not_merge_target_groups(original, tmp_path):
    tables = read_tables(build(original, tmp_path))
    # Same reported background remains, but the intended panels are disjoint.
    for row in tables["planned_products"]:
        if row["pool_name"] == "TUM_HLA_2":
            row["peptide_sequence"] = "WWWW"
    result = response.compile_response(tables)
    assert result["summary"]["n_dependence_components"] == 2
    assert result["summary"]["n_admitted_forms"] == 2
    assert all(row["pool_name"] == "TUM_HLA_1" for row in result["examples"])
