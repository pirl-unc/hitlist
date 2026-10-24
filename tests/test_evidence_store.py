"""Adversarial fixtures for grain-safe v2 evidence requests."""

import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from hitlist.evidence_bindings import materialized_bindings
from hitlist.evidence_recipe import RecipeError, _read_recipe, verify_recipe_package
from hitlist.evidence_store import (
    EvidenceFieldError,
    EvidenceRelationshipError,
    EvidenceStore,
)


def _store(tmp_path, rows):
    bindings = materialized_bindings()
    schemas = {}
    for name, values in rows.items():
        path = tmp_path / bindings[name][0].materialized_artifact
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(values), path)
        schemas[name] = pq.ParquetFile(path).schema_arrow
    (tmp_path / "manifest.json").write_text(json.dumps({"schema_version": 6}))
    store = EvidenceStore.__new__(EvidenceStore)
    store.directory = tmp_path
    store.manifest = {"schema_version": 6}
    store.bindings = {name: bindings[name] for name in rows}
    store.schemas = schemas
    return store


def test_recipe_yaml_rejects_duplicate_keys(tmp_path):
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text("schema_version: 1\nschema_version: 2\n")
    with pytest.raises(yaml.constructor.ConstructorError, match="duplicate key"):
        _read_recipe(recipe)


@pytest.mark.parametrize("keys", [[None], [None, "ctx"]])
def test_optional_context_join_preserves_null_rows_and_never_matches_null_identity(tmp_path, keys):
    store = _store(
        tmp_path / "source",
        {
            "hap1_precursors": [
                {"source_record_id": str(i), "context_id": key} for i, key in enumerate(keys)
            ],
            "hap1_contexts": [
                {"context_id": None, "condition_knockout_genes": "must_not_match"},
                {"context_id": "ctx", "condition_knockout_genes": "TAP1"},
            ],
        },
    )
    result = store.query(
        "hap1_precursors",
        select=["source_record_id", "context.condition_knockout_genes"],
        joins=["context"],
        limit=None,
    )
    assert result.table["context.condition_knockout_genes"].to_pylist() == [
        None if key is None else "TAP1" for key in keys
    ]
    assert result.table.schema.field("context.condition_knockout_genes").type == pa.string()


def test_required_context_join_rejects_null_source_key(tmp_path):
    store = _store(
        tmp_path / "source",
        {
            "hap1_quantities": [{"measurement_id": "m", "context_id": None}],
            "hap1_contexts": [{"context_id": "ctx", "condition_knockout_genes": "TAP1"}],
        },
    )
    with pytest.raises(EvidenceRelationshipError, match="missing"):
        store.query(
            "hap1_quantities", select=["context.condition_knockout_genes"], joins=["context"]
        )


def test_mhc_partition_and_measurement_do_not_convert_negative_ms(tmp_path):
    store = _store(
        tmp_path,
        {
            "mhc_positive_ms": [
                {"observation_id": "positive", "evidence_type": "mhc_positive_ms"},
                {"observation_id": "negative", "evidence_type": "mhc_other"},
            ],
            "mhc_other": [
                {"observation_id": "positive", "evidence_type": "mhc_positive_ms"},
                {"observation_id": "negative", "evidence_type": "mhc_other"},
            ],
            "mhc_reported_measurement": [
                {"observation_id": "positive", "qualitative_result": "Positive"},
                {"observation_id": "negative", "qualitative_result": "Negative"},
            ],
        },
    )
    fields = ["observation_id", "reported_measurement.qualitative_result"]
    positive = store.query(
        "mhc_positive_ms", select=fields, joins=["reported_measurement"], limit=None
    )
    other = store.query("mhc_other", select=fields, joins=["reported_measurement"], limit=None)
    assert positive.table.to_pylist() == [
        {"observation_id": "positive", "reported_measurement.qualitative_result": "Positive"}
    ]
    assert other.table.to_pylist() == [
        {"observation_id": "negative", "reported_measurement.qualitative_result": "Negative"}
    ]


def test_join_requires_name_and_rejects_broken_one_to_one(tmp_path):
    store = _store(
        tmp_path,
        {
            "mhc_positive_ms": [{"observation_id": "a", "evidence_type": "mhc_positive_ms"}],
            "mhc_reported_measurement": [
                {"observation_id": "a", "qualitative_result": "Positive"},
                {"observation_id": "a", "qualitative_result": "Negative"},
            ],
        },
    )
    with pytest.raises(EvidenceRelationshipError, match="Name relationship"):
        store.query("mhc_positive_ms", select=["reported_measurement.qualitative_result"])
    with pytest.raises(EvidenceRelationshipError, match="multiple target rows"):
        store.query(
            "mhc_positive_ms",
            select=["reported_measurement.qualitative_result"],
            joins=["reported_measurement"],
        )


def test_source_numeric_representations_are_distinct_measurements(tmp_path):
    store = _store(
        tmp_path,
        {
            "bulk_ms_protein": [
                {"observation_id": "protein:1", "grain": "protein", "protein_id": "P1"}
            ],
            "bulk_numeric_measurement": [
                {
                    "observation_id": "protein:1",
                    "measurement_id": "protein:1:log2",
                    "quantity_scale": "log2_intensity",
                    "quantitative_value": 3.0,
                },
                {
                    "observation_id": "protein:1",
                    "measurement_id": "protein:1:percentile",
                    "quantity_scale": "abundance_percentile",
                    "quantitative_value": 0.8,
                },
            ],
        },
    )
    values = store.query(
        "bulk_numeric_measurement",
        select=["measurement_id", "quantity_scale", "quantitative_value"],
        limit=None,
    ).table.to_pylist()
    assert [value["quantity_scale"] for value in values] == [
        "log2_intensity",
        "abundance_percentile",
    ]
    with pytest.raises(EvidenceRelationshipError, match="many cardinality"):
        store.query(
            "bulk_ms_protein",
            select=["numeric_measurements.quantitative_value"],
            joins=["numeric_measurements"],
        )


def test_project_instrument_is_not_result_instrument(tmp_path):
    store = _store(
        tmp_path,
        {
            "external_mhc_peptide_quantity": [
                {
                    "result_id": "r1",
                    "peptide_row_id": "p1",
                    "channel_id": "c1",
                    "table_number": 7,
                    "endpoint": "observed_source_intensity",
                }
            ],
            "external_context_assertion": [
                {
                    "assertion_id": "project:1",
                    "predicate": "instrument",
                    "value_raw": "Orbitrap Exploris 480",
                    "scope": "project_not_individual_run",
                }
            ],
        },
    )
    with pytest.raises(EvidenceFieldError, match="unavailable"):
        store.query("external_mhc_peptide_quantity", select=["instrument"])
    claim = store.query(
        "external_context_assertion",
        select=["assertion_predicate", "assertion_value", "source_scope"],
    ).table.to_pylist()[0]
    assert claim == {
        "assertion_predicate": "instrument",
        "assertion_value": "Orbitrap Exploris 480",
        "source_scope": "project_not_individual_run",
    }


def test_mouse_reference_context_keeps_measurements_and_roles_distinct(tmp_path):
    store = _store(
        tmp_path / "source",
        {
            "external_quantity_measurement": [
                {
                    "measurement_id": "m9",
                    "result_id": "r9",
                    "representation": "raw_tmt_intensity",
                    "reported_value": 4.0,
                    "unit": "source_intensity_units",
                    "quantity_scale": "relative_tmt_signal",
                    "value_status": "source_reported",
                },
                {
                    "measurement_id": "m10",
                    "result_id": "r10",
                    "representation": "raw_tmt_intensity",
                    "reported_value": 5.0,
                    "unit": "source_intensity_units",
                    "quantity_scale": "relative_tmt_signal",
                    "value_status": "source_reported",
                },
            ],
            "external_reference_result_context": [
                {
                    "result_id": "r9",
                    "peptide_row_id": "p9",
                    "component": "shared-parent-component",
                    "source_parent_status": "exact_sequence_match",
                    "measured_population": "macrophage",
                    "partner_population": "tumor_cell",
                    "material_link_precision": "assay_channel_unresolved_specimen",
                    "specimen_status": "unresolved",
                    "organism_taxon_id": "10090",
                },
                {
                    "result_id": "r10",
                    "peptide_row_id": "p10",
                    "component": "shared-parent-component",
                    "source_parent_status": "missing_id",
                    "measured_population": "macrophage",
                    "partner_population": "mhc_i_loss_tumor_cell",
                    "material_link_precision": "assay_channel_unresolved_specimen",
                    "specimen_status": "unresolved",
                    "organism_taxon_id": "10090",
                },
            ],
        },
    )
    rows = store.query(
        "external_quantity_measurement",
        select=[
            "measurement_id",
            "reference_context.measured_population",
            "reference_context.partner_population",
            "reference_context.source_parent_status",
            "reference_context.specimen_status",
        ],
        joins=["reference_context"],
    ).table.to_pylist()
    assert len(rows) == 2
    assert [row["reference_context.measured_population"] for row in rows] == [
        "macrophage",
        "macrophage",
    ]
    assert rows[1]["reference_context.partner_population"] == "mhc_i_loss_tumor_cell"
    assert rows[1]["reference_context.source_parent_status"] == "missing_id"
    assert all(row["reference_context.specimen_status"] == "unresolved" for row in rows)
    recipe = {
        "schema_version": 1,
        "name": "reported_mouse_signal",
        "mapping_policy": "no_expansion",
        "sources": [
            {
                "evidence_type": "external_quantity_measurement",
                "row_id": "measurement_id",
                "select": ["reference_context.source_record_id"],
                "joins": ["reference_context"],
                "where": {"representation": "raw_tmt_intensity"},
                "target": {
                    "field": "reported_value",
                    "mode": "numeric",
                    "unit_field": "measurement_unit",
                    "allowed_units": ["source_intensity_units"],
                    "scale_field": "quantity_scale",
                    "allowed_scales": ["relative_tmt_signal"],
                    "status_field": "value_status",
                    "allowed_statuses": ["source_reported"],
                },
                "context_precision": {
                    "field": "reference_context.material_link_precision",
                    "allowed": ["assay_channel_unresolved_specimen"],
                    "on_mismatch": "ambiguous",
                },
                "grouping": {
                    "fields": ["reference_context.component"],
                    "on_missing": "mark_uncertain",
                },
            }
        ],
    }
    output = tmp_path / "training"
    store.materialize(recipe, output)
    base = output / "sources" / "external_quantity_measurement"
    assert pq.read_table(base / "accepted.parquet").num_rows == 2
    assert {
        row["source_record_id"] for row in pq.read_table(base / "contributors.parquet").to_pylist()
    } == {
        "p9",
        "p10",
    }
    assert (
        len({row["split_group_id"] for row in pq.read_table(base / "lineage.parquet").to_pylist()})
        == 1
    )


def test_tcell_availability_is_not_a_peptide_label(tmp_path):
    store = _store(
        tmp_path,
        {
            "tcell_availability": [
                {
                    "source_assay_id": "123",
                    "assay_family": "soluble_factor_release",
                    "qualitative_measurement": "Positive",
                }
            ],
            "tcell_functional_peptide": [
                {
                    "source_assay_id": "123",
                    "task": "assay_specific_tcell_qualitative_response",
                    "peptide": "LLDVTAAV",
                    "source_qualitative_result": "Positive",
                }
            ],
        },
    )
    with pytest.raises(EvidenceFieldError, match="Unknown field"):
        store.query("tcell_availability", select=["peptide_sequence"])
    assert store.query(
        "tcell_functional_peptide", select=["peptide_sequence", "qualitative_result"]
    ).table.to_pylist() == [{"peptide_sequence": "LLDVTAAV", "qualitative_result": "Positive"}]


def test_recipe_retains_accepted_excluded_and_ambiguous_candidates(tmp_path):
    store = _store(
        tmp_path / "source",
        {
            "mhc_positive_ms": [
                {
                    "observation_id": "a",
                    "evidence_type": "mhc_positive_ms",
                    "peptide_sequence": "LLDVTAAV",
                    "source_dataset": "iedb",
                },
                {
                    "observation_id": "b",
                    "evidence_type": "mhc_positive_ms",
                    "peptide_sequence": "VVVVVVVV",
                    "source_dataset": "iedb",
                },
                {
                    "observation_id": "c",
                    "evidence_type": "mhc_positive_ms",
                    "peptide_sequence": None,
                    "source_dataset": "cedar",
                },
            ],
            "mhc_reported_measurement": [
                {"observation_id": "a", "qualitative_result": "Positive"},
                {"observation_id": "b", "qualitative_result": "Positive-Low"},
                {"observation_id": "c", "qualitative_result": "Positive"},
            ],
            "source_contributor": [
                {
                    "provenance_id": "a",
                    "source_record_id": "iedb:a",
                    "source_dataset": "iedb",
                    "original_pmid": "1",
                    "source_row": 1,
                    "relationships": '["retained"]',
                },
                {
                    "provenance_id": "a",
                    "source_record_id": "cedar:a",
                    "source_dataset": "cedar",
                    "original_pmid": "1",
                    "source_row": 2,
                    "relationships": '["database_copy"]',
                },
                {
                    "provenance_id": "b",
                    "source_record_id": "iedb:b",
                    "source_dataset": "iedb",
                    "original_pmid": "1",
                    "source_row": 3,
                    "relationships": '["retained"]',
                },
                {
                    "provenance_id": "c",
                    "source_record_id": "iedb:c",
                    "source_dataset": "iedb",
                    "original_pmid": "1",
                    "source_row": 4,
                    "relationships": '["retained"]',
                },
            ],
        },
    )
    recipe = {
        "schema_version": 1,
        "name": "strict_positive_ligand",
        "mapping_policy": "no_expansion",
        "sources": [
            {
                "evidence_type": "mhc_positive_ms",
                "row_id": "observation_id",
                "select": ["peptide_sequence", "source_dataset"],
                "joins": ["reported_measurement"],
                "required_fields": ["peptide_sequence"],
                "target": {
                    "field": "reported_measurement.qualitative_result",
                    "mode": "mapped",
                    "dtype": "boolean",
                    "map": {"Positive": True},
                },
                "grouping": {"fields": ["source_dataset"], "on_missing": "mark_uncertain"},
            }
        ],
    }
    output = tmp_path / "training"
    store.materialize(recipe, output)
    manifest = verify_recipe_package(output, source_package=store.directory)
    report = manifest["coverage"]["mhc_positive_ms"]
    assert (
        report["n_candidates"],
        report["n_accepted"],
        report["n_excluded"],
        report["n_ambiguous"],
    ) == (
        3,
        1,
        1,
        1,
    )
    assert report["reason_counts"] == {
        "missing_required:peptide_sequence": 1,
        "unmapped_target": 1,
    }
    base = output / "sources" / "mhc_positive_ms"
    accepted = pq.read_table(base / "accepted.parquet").to_pylist()
    assert [(row["candidate_id"], row["target_value"]) for row in accepted] == [
        ("mhc_positive_ms:a", True)
    ]
    contributors = pq.read_table(base / "contributors.parquet").to_pylist()
    assert len(contributors) == 4
    assert {
        row["source_dataset"] for row in contributors if row["candidate_id"] == "mhc_positive_ms:a"
    } == {"iedb", "cedar"}
    assert pq.read_table(base / "lineage.parquet").num_rows == 3


def test_recipe_includes_direct_source_lineage_without_explicit_selection(tmp_path):
    store = _store(
        tmp_path / "source",
        {
            "bulk_ms_peptide": [
                {
                    "grain": "peptide",
                    "source_record_id": "bulk:row:7",
                    "source_dataset": "bulk",
                    "peptide_sequence": "LLDVTAAV",
                }
            ]
        },
    )
    recipe = {
        "schema_version": 1,
        "name": "bulk_peptide_inventory",
        "mapping_policy": "no_expansion",
        "sources": [
            {
                "evidence_type": "bulk_ms_peptide",
                "row_id": "source_record_id",
                "select": ["peptide_sequence"],
                "target": {"field": "peptide_sequence", "mode": "category"},
                "grouping": {"fields": ["source_dataset"], "on_missing": "mark_uncertain"},
            }
        ],
    }
    output = tmp_path / "training"
    store.materialize(recipe, output)
    base = output / "sources" / "bulk_ms_peptide"
    assert pq.read_table(base / "accepted.parquet").to_pylist()[0]["source_dataset"] == "bulk"
    assert (
        pq.read_table(base / "contributors.parquet").to_pylist()[0]["source_record_id"]
        == "bulk:row:7"
    )


def test_recipe_refuses_duplicate_candidate_identity(tmp_path):
    store = _store(
        tmp_path / "source",
        {
            "bulk_numeric_measurement": [
                {
                    "measurement_id": "duplicate",
                    "quantitative_value": 1.0,
                    "quantity_scale": "tpm",
                    "value_status": "source_numeric_cell",
                },
                {
                    "measurement_id": "duplicate",
                    "quantitative_value": 2.0,
                    "quantity_scale": "tpm",
                    "value_status": "source_numeric_cell",
                },
            ]
        },
    )
    recipe = {
        "schema_version": 1,
        "name": "duplicate_guard",
        "mapping_policy": "no_expansion",
        "sources": [
            {
                "evidence_type": "bulk_numeric_measurement",
                "row_id": "measurement_id",
                "target": {
                    "field": "quantitative_value",
                    "mode": "numeric",
                    "scale_field": "quantity_scale",
                    "allowed_scales": ["tpm"],
                    "status_field": "value_status",
                    "allowed_statuses": ["source_numeric_cell"],
                },
                "grouping": {"fields": ["quantity_scale"], "on_missing": "mark_uncertain"},
            }
        ],
    }
    with pytest.raises(RecipeError, match="Duplicate candidate identity"):
        store.materialize(recipe, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_numeric_recipe_marks_censoring_and_unknown_units_ambiguous(tmp_path):
    store = _store(
        tmp_path / "source",
        {
            "mhc_reported_measurement": [
                {
                    "measurement_id": "exact",
                    "observation_id": "assay:a",
                    "quantitative_value": 50.0,
                    "measurement_unit": "nM",
                    "measurement_inequality": "=",
                    "quantitative_status": "reported",
                },
                {
                    "measurement_id": "censored",
                    "observation_id": "assay:b",
                    "quantitative_value": 100.0,
                    "measurement_unit": "nM",
                    "measurement_inequality": "<",
                    "quantitative_status": "reported",
                },
                {
                    "measurement_id": "unitless",
                    "observation_id": "assay:c",
                    "quantitative_value": 200.0,
                    "measurement_unit": None,
                    "measurement_inequality": "=",
                    "quantitative_status": "reported",
                },
            ]
        },
    )
    recipe = {
        "schema_version": 1,
        "name": "exact_affinity",
        "mapping_policy": "no_expansion",
        "sources": [
            {
                "evidence_type": "mhc_reported_measurement",
                "row_id": "measurement_id",
                "target": {
                    "field": "quantitative_value",
                    "mode": "numeric",
                    "unit_field": "measurement_unit",
                    "allowed_units": ["nM"],
                    "inequality_field": "measurement_inequality",
                    "allowed_inequalities": ["="],
                    "status_field": "quantitative_status",
                    "allowed_statuses": ["reported"],
                },
                "grouping": {"fields": ["observation_id"], "on_missing": "exclude"},
            }
        ],
    }
    output = tmp_path / "output"
    store.materialize(recipe, output)
    report = verify_recipe_package(output)["coverage"]["mhc_reported_measurement"]
    assert (report["n_accepted"], report["n_ambiguous"], report["n_excluded"]) == (
        1,
        2,
        0,
    )
    assert report["reason_counts"] == {
        "inequality_not_allowed": 1,
        "unit_not_allowed": 1,
    }


def test_numeric_unit_conversion_preserves_censored_values_as_ambiguous(tmp_path):
    store = _store(
        tmp_path / "source",
        {
            "mhc_reported_measurement": [
                {
                    "measurement_id": "exact",
                    "observation_id": "a",
                    "quantitative_value": 0.5,
                    "measurement_unit": "uM",
                    "measurement_inequality": "=",
                    "quantitative_status": "reported",
                },
                {
                    "measurement_id": "censored",
                    "observation_id": "b",
                    "quantitative_value": 0.5,
                    "measurement_unit": "uM",
                    "measurement_inequality": ">",
                    "quantitative_status": "reported",
                },
            ]
        },
    )
    target = {
        "field": "quantitative_value",
        "mode": "numeric",
        "unit_field": "measurement_unit",
        "allowed_units": ["uM"],
        "unit_conversion": {"output_unit": "nM", "factor_by_unit": {"uM": 1000.0}},
        "inequality_field": "measurement_inequality",
        "allowed_inequalities": ["="],
        "status_field": "quantitative_status",
        "allowed_statuses": ["reported"],
    }
    recipe = {
        "schema_version": 1,
        "name": "exact_binding_nM",
        "mapping_policy": "no_expansion",
        "sources": [
            {
                "evidence_type": "mhc_reported_measurement",
                "row_id": "measurement_id",
                "target": target,
                "grouping": {"fields": ["observation_id"], "on_missing": "mark_uncertain"},
            }
        ],
    }
    output = tmp_path / "training"
    store.materialize(recipe, output)
    base = output / "sources" / "mhc_reported_measurement"
    accepted = pq.read_table(base / "accepted.parquet").to_pylist()
    ambiguous = pq.read_table(base / "ambiguous.parquet").to_pylist()
    assert [(row["target_value"], row["target_unit"]) for row in accepted] == [(500.0, "nM")]
    assert ambiguous[0]["reason_codes"] == ["inequality_not_allowed"]
    assert ambiguous[0]["measurement_inequality"] == ">"
    target["allowed_inequalities"] = [">"]
    with pytest.raises(RecipeError, match="Censored inequalities"):
        store.materialize(recipe, tmp_path / "invalid")
