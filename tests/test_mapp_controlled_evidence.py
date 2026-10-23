"""Controlled panels preserve observed quantities and distinguish context axes."""

import csv
import json

import pyarrow.parquet as pq
import pytest

from hitlist.mapp_controlled_evidence import (
    channel_context,
    match_quantity_columns,
    write_controlled_tables,
)


def _write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _fixture(root):
    (root / "contracts").mkdir()
    (root / "contracts/contract.json").write_text(json.dumps({"assertions": []}))
    for project in ("PXD009508", "PXD007263"):
        if project == "PXD009508":
            labels = [
                f"{1511 + i}_7181-300a_{capture}_{treatment}{rep}"
                for i, (capture, treatment, rep) in enumerate(
                    (c, t, r)
                    for c in ("alpha6", "iso", "gap")
                    for t in ("ut", "epoxo")
                    for r in (1, 2, 3)
                )
            ]
            columns = [
                "Intensity "
                + label.rsplit("_", 1)[0]
                + "-"
                + label.rsplit("_", 1)[1][:-1]
                + "_"
                + label[-1]
                for label in labels
            ]
        else:
            labels = [
                f"{667 + i}_TNF-TP-0-6_{hour}_{rep}"
                for i, (rep, hour) in enumerate((r, h) for r in (1, 2, 3) for h in (0, 1, 2, 6))
            ]
            columns = ["Intensity " + label for label in labels]
        directory = root / "assets" / project / "txt"
        _write(
            directory / "summary.txt",
            [
                {"Raw file": f"raw_{index}", "Experiment": label, "Enzyme mode": "Unspecific"}
                for index, label in enumerate(labels)
            ]
            + [{"Raw file": "Total", "Experiment": "", "Enzyme mode": ""}],
        )
        peptide = {
            "id": "0",
            "Sequence": "ACDEFGH",
            "Proteins": "P1",
            "Leading razor protein": "P1",
            "Protein group IDs": "0",
            "Amino acid before": "-",
            "Amino acid after": "I",
            "Start position": "1",
            "End position": "7",
            "PEP": "0.001",
            "Reverse": "",
            "Potential contaminant": "",
            "Evidence IDs": ";".join(str(i) for i in range(len(labels))),
            **{column: str(index) for index, column in enumerate(columns)},
        }
        _write(directory / "peptides.txt", [peptide])
        _write(
            directory / "proteinGroups.txt",
            [
                {
                    "id": "0",
                    "Protein IDs": "P1",
                    "Peptide IDs": "0",
                    "Reverse": "",
                    "Potential contaminant": "",
                }
            ],
        )
        _write(
            directory / "evidence.txt",
            [
                {
                    "id": str(index),
                    "Peptide ID": "0",
                    "Experiment": label,
                    "Sequence": "ACDEFGH",
                    "Raw file": f"raw_{index}",
                    "Protein group IDs": "0",
                    "Intensity": str(index) if index else "",
                    "Type": "MULTI-MSMS" if index else "MSMS",
                    "MS/MS Count": "1",
                    "Modified sequence": "_ACDEFGH_",
                    "Modifications": "Unmodified",
                    "PEP": "NaN" if index == 1 else "0.001",
                    "Reverse": "",
                    "Potential contaminant": "",
                }
                for index, label in enumerate(labels)
            ],
        )
        _write(directory / "parameters.txt", [{"Parameter": "Protein FDR", "Value": "1"}])
    return root


def test_context_does_not_invent_genetic_loss_or_pairing():
    value = channel_context("PXD009508", "1514_7181-300a_alpha6_epoxo1")
    assert value["treatment_kind"] == "pharmacological"
    assert value["treatment_dose"] == 1 and value["duration_hours"] == 4
    assert value["capture_target_gene"] is None
    assert value["specimen_id"] is value["aliquot_id"] is value["pair_id"] is None
    assert value["capture_role"] == "candidate_proteasome_capture"
    control = channel_context("PXD009508", "1520_7181-300a_iso_epoxo1")
    assert control["capture_role"] == "capture_control"
    assert control["treatment_kind"] == "pharmacological"
    time = channel_context("PXD007263", "669_TNF-TP-0-6_2_1")
    assert time["time_hours"] == 2
    assert time["treatment_label_raw"] == "TNF" and time["treatment_name"] is None


def test_aliases_are_complete_identifier_claims_not_specimen_links():
    source = "1511_7181-300a_alpha6_ut1"
    column = "Intensity 1511_7181-300a_alpha6-ut_1"
    assert match_quantity_columns([column], [source])[column] == (
        source,
        "unique_full_identifier_without_hyphen_underscore",
    )
    with pytest.raises(ValueError, match="Ambiguous"):
        match_quantity_columns([column], [source, source.replace("_ut1", "_ut_1")])
    with pytest.raises(ValueError, match="Incomplete"):
        match_quantity_columns(["Intensity a"], ["a", "b"])


def test_all_runs_quantities_raw_claims_and_missing_values_survive(tmp_path):
    root = _fixture(tmp_path)
    result = write_controlled_tables(root)
    assert result["relation_rows"]["channels"] == 30
    assert result["relation_rows"]["quantity_results"] == 30
    assert result["relation_rows"]["quantity_measurements"] == 60
    aliases = pq.read_table(root / "typed/label_aliases.parquet").to_pylist()
    assert sum(row["match_rule"] != "literal_equal" for row in aliases) == 18
    values = pq.read_table(root / "typed/quantity_measurements.parquet").to_pylist()
    zeros = [row for row in values if row["value"] == 0]
    assert len(zeros) == 2
    assert all(row["value_status"] == "reported_zero_not_absence" for row in zeros)
    unknown = [row for row in values if row["value"] is None]
    assert len(unknown) == 2
    assert all(row["value_status"] == "all_evidence_intensities_unreported" for row in unknown)
    assert {row["source_dataset"] for row in values} == {"PXD009508", "PXD007263"}
    assertions = pq.read_table(root / "typed/source_assertions.parquet").to_pylist()
    assert all(row["predicate"] == "Protein FDR" and row["value_raw"] == "1" for row in assertions)


def test_unreplayable_reported_quantity_fails_before_admission(tmp_path):
    root = _fixture(tmp_path)
    path = root / "assets/PXD009508/txt/peptides.txt"
    with path.open() as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    first = next(key for key in rows[0] if key.startswith("Intensity "))
    rows[0][first] = "100"
    _write(path, rows)
    with pytest.raises(ValueError, match="does not replay"):
        write_controlled_tables(root)


def test_optional_source_bindings_are_explicit():
    from hitlist.evidence_bindings import materialized_bindings

    assert "processing_peptide" not in materialized_bindings(9)
    bindings = materialized_bindings(9, appended_source_ids=["wolflevy2018_controlled"])
    assert "processing_peptide" in bindings and "processing_channel" in bindings
    assert "mapp_peptide" not in bindings


def _recipe_tables(root):
    import pyarrow as pa

    from hitlist.mapp_controlled_recipe import TABLE_BINDINGS
    from hitlist.sequence_reference_evidence import SCHEMAS as REFERENCE_SCHEMAS

    write_controlled_tables(root)
    for name in ("reference_proteins", "peptide_reference_occurrences"):
        pq.write_table(
            pa.Table.from_pylist([], schema=REFERENCE_SCHEMAS[name]),
            root / "typed" / f"{name}.parquet",
        )
    return {
        name: pq.read_table(root / "typed" / f"{name}.parquet").to_pylist()
        for name in TABLE_BINDINGS
    }


def test_recipe_excludes_zeros_and_keeps_identical_sequences_in_one_split(tmp_path):
    from hitlist.mapp_controlled_recipe import compile_controlled

    tables = _recipe_tables(_fixture(tmp_path))
    compiled = compile_controlled(tables)
    assert compiled["summary"]["n_source_results"] == 30
    assert compiled["summary"]["n_examples"] == 28
    assert compiled["summary"]["n_excluded_results"] == 2
    assert len({row["component_id"] for row in compiled["components"]}) == 1
    assert len({row["split"] for row in compiled["components"]}) == 1
    assert all(row["reported_intensity"] > 0 for row in compiled["examples"])
    assert all(row["context"]["pair_id"] is None for row in compiled["examples"])
    original = compiled["components"]
    for row in tables["quantity_measurements"]:
        if row["value"] is not None:
            row["value"] += 10
        if row["recomputed_value"] is not None:
            row["recomputed_value"] += 10
    assert compile_controlled(tables)["components"] == original


def test_recipe_connects_shared_parent_sequence_across_reference_ids(tmp_path):
    from hitlist.mapp_controlled_recipe import compile_controlled

    tables = _recipe_tables(_fixture(tmp_path))
    left, right = tables["peptide_rows"]
    right["sequence_raw"] = "ACDEFGI"
    tables["reference_proteins"] = [
        {"reference_record_id": "old:P1", "sequence_sha256": "same_full_sequence"},
        {"reference_record_id": "new:P1", "sequence_sha256": "same_full_sequence"},
    ]
    tables["peptide_reference_occurrences"] = [
        {"peptide_row_id": left["peptide_row_id"], "reference_record_id": "old:P1"},
        {"peptide_row_id": right["peptide_row_id"], "reference_record_id": "new:P1"},
    ]
    result = compile_controlled(tables)
    assert len({row["component_id"] for row in result["components"]}) == 1


def test_recipe_roundtrip_rejects_changed_target_even_with_new_file_hash(tmp_path):
    from hitlist.evidence_bindings import materialized_bindings
    from hitlist.evidence_recipe import RecipeError, verify_recipe_package
    from hitlist.evidence_store import EvidenceStore
    from hitlist.mapp_controlled_recipe import RECIPE_ID, TABLE_BINDINGS
    from hitlist.provenance import file_digest

    source = tmp_path / "package"
    panel = source / "external/wolflevy2018_controlled"
    panel.mkdir(parents=True)
    _recipe_tables(_fixture(panel))
    files = {
        str(path.relative_to(source)): file_digest(path)
        for path in source.rglob("*")
        if path.is_file()
    }
    envelope = {
        "schema_version": 9,
        "files": files,
        "appended_sources": {"wolflevy2018_controlled": {}},
    }
    (source / "manifest.json").write_text(json.dumps(envelope))
    store = EvidenceStore.__new__(EvidenceStore)
    store.directory, store.manifest = source, envelope
    bindings = materialized_bindings(9, appended_source_ids=["wolflevy2018_controlled"])
    store.bindings = {name: bindings[name] for name in TABLE_BINDINGS.values()}
    store.schemas = {
        name: pq.ParquetFile(source / binding[0].materialized_artifact).schema_arrow
        for name, binding in store.bindings.items()
    }
    destination = tmp_path / "recipe"
    store.materialize(
        {"schema_version": 1, "name": "fixture", "registered_recipe": RECIPE_ID}, destination
    )
    verify_recipe_package(destination, source_package=source)
    path = destination / "examples.json"
    examples = json.loads(path.read_text())
    examples[0]["target_log10_intensity"] += 1
    path.write_text(json.dumps(examples))
    manifest_path = destination / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"]["examples.json"] = file_digest(path)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(RecipeError, match="derived contents changed: examples"):
        verify_recipe_package(destination, source_package=source)
