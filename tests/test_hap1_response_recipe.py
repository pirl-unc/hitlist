"""Response semantics and source-link isolation, using artificial software fixtures."""

import hashlib
import json
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from hitlist import hap1_quantitative as source_schema
from hitlist import hap1_response_recipe as response
from hitlist.evidence_recipe import RecipeError, materialize_recipe, verify_recipe_package
from hitlist.provenance import file_digest


@pytest.fixture
def typed(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    tables = {name: [] for name in response.TABLES}
    conditions = [response.WT, response.B2M, "hap1_tap1_ko"]
    for condition in conditions:
        for modality in ["elution", "proteomics"]:
            context = f"{condition}:{modality}"
            tables["contexts"].append({"context_id": context, "condition_id": condition})
            for replicate in [1, 2, 3]:
                tables["samples"].append(
                    {"sample_id": f"{context}:{replicate}", "context_id": context}
                )
    sequences = {"P1": "MACDEFGHIK", "P2": "MKLMNP", "P3": "MWYV", "P4": "MQRSTV"}
    parents = {}
    for name, sequence in sequences.items():
        digest = hashlib.sha256(sequence.encode()).hexdigest()
        parents[name] = digest
        tables["references"].append(
            {
                "reference_record_id": name,
                "sequence_sha256": digest,
                "reference_sequence": sequence,
                "accession_raw": name,
                "reference_role": "human_target",
            }
        )
        tables["source_records"].append(
            {
                "source_record_id": "protein:" + name,
                "source_role": "deposited_protein_report",
                "source_row_json": json.dumps(
                    {"Protein": name, "Indistinguishable Proteins": "P2" if name == "P1" else ""}
                ),
            }
        )
    # P2 belongs to two source groups: preserve, do not select one.
    peptides = {"ACDE": "P1", "DEFG": "P1", "KLMN": "P2", "WY": "P3", "QRST": "P4"}
    for peptide, name in peptides.items():
        start = sequences[name].index(peptide) + 1
        tables["occurrences"].append(
            {
                "occurrence_id": peptide,
                "source_record_id": peptide,
                "peptide_sequence": peptide,
                "reference_record_id": name,
                "parent_label_raw": name,
                "start_1based": start,
                "end_1based": start + len(peptide) - 1,
            }
        )

    def cell(name, condition, replicate, value, protein=False):
        modality = "proteomics" if protein else "elution"
        raw = str(value) if value is not None else "NA"
        tables["quantities"].append(
            {
                "measurement_id": f"{name}:{condition}:{replicate}",
                "source_record_id": name,
                "sample_id": f"{condition}:{modality}:{replicate}",
                "context_id": f"{condition}:{modality}",
                "peptide_sequence": None if protein else name,
                "parent_label_raw": name if protein else peptides[name],
                "quantity_kind": "protein_MaxLFQ_intensity"
                if protein
                else "normalized_peptide_intensity",
                "quantitative_value": value,
                "quantitative_raw": raw,
                "value_status": "source_missing_NA"
                if value is None
                else "source_zero"
                if value == 0
                else "source_positive",
            }
        )

    for peptide in peptides:
        for replicate in [1, 2]:
            cell(peptide, response.WT, replicate, 10.0 if replicate == 1 else 1000.0)
            cell(peptide, "hap1_tap1_ko", replicate, 1000.0 if replicate == 1 else 100.0)
        cell(peptide, "hap1_tap1_ko", 3, None if peptide != "DEFG" else 0.0)
    cell("WY", response.B2M, 1, 1.0)
    for parent in sequences:
        for replicate in [1, 2]:
            cell(parent, response.WT, replicate, 100.0, True)
            cell(
                parent,
                "hap1_tap1_ko",
                replicate,
                1000.0 if parent != "P4" or replicate == 1 else None,
                True,
            )
    for name, rows in tables.items():
        pq.write_table(
            pa.Table.from_pylist(rows, schema=source_schema.SCHEMAS[name]), root / f"{name}.parquet"
        )
    return root, tables, parents


def _read(root, name):
    return pq.read_table(root / f"{name}.parquet").to_pylist()


def test_response_denominator_missingness_background_precursor_and_components(typed, tmp_path):
    source, tables, parents = typed
    output = tmp_path / "output"
    summary = response.compile_responses(source, output)
    rows = _read(output, "responses")
    assert len(rows) == 8  # four peptides, two positive KO cells; no WT cross products
    selected = [r for r in rows if r["peptide_sequence"] == "ACDE"]
    assert [r["target_log10_ratio"] for r in selected] == [1.0, 0.0]
    assert all(len(json.loads(r["wt_measurement_ids_json"])) == 2 for r in selected)
    assert all(r["precursor_log10_ratio"] == 1.0 for r in selected)
    assert all(
        r["precursor_status"] == "incomplete"
        for r in rows
        if r["peptide_sequence"] in {"KLMN", "QRST"}
    )
    assert all(r["peptide_sequence"] != "WY" for r in rows)
    dispositions = _read(output, "dispositions")
    assert len(dispositions) == len(tables["quantities"]) == summary["n_source_quantity_cells"]
    assert len({r["measurement_id"] for r in dispositions}) == len(dispositions)
    assert {r["reason"] for r in dispositions} >= {
        "source_zero",
        "source_missing_NA",
        "positive_B2M_background_sequence",
    }
    components = {r["peptide_sequence"]: r for r in _read(output, "components")}
    assert (
        components["ACDE"]["component_id"]
        == components["DEFG"]["component_id"]
        == components["KLMN"]["component_id"]
    )
    precursor = {r["parent_id"]: r for r in _read(output, "precursor_inputs")}
    assert precursor[parents["P2"]]["status"] == "ambiguous_protein_groups"
    assert precursor[parents["P4"]]["status"] == "insufficient_positive_protein_replicates"


def test_nonhuman_alternative_defers_instead_of_selecting_best_parent(typed, tmp_path):
    source, tables, _ = typed
    sequence = "ACDEW"
    tables["references"].append(
        {
            "reference_record_id": "decoy",
            "sequence_sha256": hashlib.sha256(sequence.encode()).hexdigest(),
            "reference_sequence": sequence,
            "accession_raw": "rev_P9",
            "reference_role": "reverse_decoy",
        }
    )
    tables["occurrences"].append(
        {
            "occurrence_id": "alt",
            "source_record_id": "ACDE",
            "peptide_sequence": "ACDE",
            "reference_record_id": "decoy",
            "parent_label_raw": "P1",
            "start_1based": 1,
            "end_1based": 4,
        }
    )
    for name in ["references", "occurrences"]:
        pq.write_table(
            pa.Table.from_pylist(tables[name], schema=source_schema.SCHEMAS[name]),
            source / f"{name}.parquet",
        )
    output = tmp_path / "output"
    response.compile_responses(source, output)
    assert not any(r["peptide_sequence"] == "ACDE" for r in _read(output, "responses"))
    assert any(
        r["reason"] == "unsupported_candidate_parent_groups" for r in _read(output, "dispositions")
    )


def test_registered_roundtrip_and_derived_tampering(typed, tmp_path):
    source, _, _ = typed
    package = tmp_path / "package"
    target = package / "hap1/typed"
    target.mkdir(parents=True)
    import shutil

    for path in source.iterdir():
        shutil.copyfile(path, target / path.name)
    (package / "hap1/contract.json").write_text('{"original":true}\n')
    manifest = {
        "revision": source_schema.REVISION,
        "schema_version": 20,
        "files": {
            str(p.relative_to(package)): file_digest(p) for p in package.rglob("*") if p.is_file()
        },
    }
    (package / "manifest.json").write_text(json.dumps(manifest))
    store = SimpleNamespace(
        directory=package,
        manifest=manifest,
        _software_version=lambda: "test",
        _manifest_hash=lambda: file_digest(package / "manifest.json")["sha256"],
    )
    recipe = {"schema_version": 1, "name": "fixture", "registered_recipe": response.RECIPE_ID}
    output = tmp_path / "recipe"
    materialize_recipe(store, recipe, output)
    verify_recipe_package(output, source_package=package)
    rows = _read(output / "derived", "responses")
    rows[0]["target_log10_ratio"] += 1
    changed = output / "derived/responses.parquet"
    pq.write_table(pa.Table.from_pylist(rows, schema=response.SCHEMAS["responses"]), changed)
    # Rehashing the outer receipt must not hide incorrect derived arithmetic.
    receipt = json.loads((output / "manifest.json").read_text())
    receipt["files"]["derived/responses.parquet"] = file_digest(changed)
    (output / "manifest.json").write_text(json.dumps(receipt))
    with pytest.raises(RecipeError, match="derived rows changed"):
        verify_recipe_package(output, source_package=package)


def test_invalid_recipe_rejected():
    with pytest.raises(RecipeError, match="Invalid HAP1"):
        response._validate({"registered_recipe": response.RECIPE_ID})
