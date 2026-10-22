"""Semantic invariants of the v2 evidence catalog, without a built corpus."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from hitlist.evidence_catalog import (
    BASELINES,
    describe_catalog,
    describe_evidence,
    describe_task,
    list_evidence,
    list_tasks,
)


def test_catalog_covers_current_data_grains_without_double_counting():
    observations = set(list_evidence(kind="observation"))
    projections = set(list_evidence(kind="projection"))
    relations = set(list_evidence(kind="relation"))
    assert {"mhc_positive_ms", "mhc_binding", "mhc_other", "tcell_availability"} <= observations
    assert {"bulk_ms_peptide", "bulk_ms_protein", "line_expression_gene"} <= observations
    assert {"tcell_functional_peptide", "tcell_multimer_peptide"} <= projections
    assert {"peptide_protein_occurrence", "source_contributor", "reviewed_lineage"} <= relations

    functional = describe_evidence("tcell_functional_peptide")
    multimer = describe_evidence("tcell_multimer_peptide")
    availability = describe_evidence("tcell_availability")
    assert functional["source_population"] == multimer["source_population"]
    assert functional["source_population"] == availability["source_population"]
    assert "peptide_sequence" not in {field["name"] for field in availability["fields"]}
    assert len(availability["assay_families"]) == 5
    assert set(functional["assay_families"]).isdisjoint(multimer["assay_families"])
    assert "pmhc_multimer" not in availability["endpoints"]

    bulk = [item for item in describe_catalog()["evidence"] if item["family"] == "bulk_proteomics"]
    assert {item["native_selector"] for item in bulk} == {
        "granularity=peptide",
        "granularity=protein",
    }
    assert len({item["native_artifact"] for item in bulk}) == 1
    mhc = describe_evidence("mhc_positive_ms")
    assert mhc["status"] == "v2_materialized"
    assert mhc["materialized_artifact"] == "typed/mhc_observations.parquet"
    assert describe_evidence("source_contributor")["materialized_artifact"] == (
        "native/observation_contributors.parquet"
    )


def test_labels_and_biological_state_have_distinct_roles():
    fields = {
        field["name"]: field for item in describe_catalog()["evidence"] for field in item["fields"]
    }
    assert fields["cytokine_exposure"]["role"] == "biological_context"
    assert fields["gene_perturbation"]["role"] == "biological_context"
    assert fields["assay_readout"]["role"] == "protocol"
    assert fields["instrument"]["role"] == "protocol"
    assert fields["qualitative_result"]["role"] == "target"
    assert fields["candidate_mhc_set"]["name"] != fields["peptide_mhc_restriction"]["name"]
    assert fields["reported_mhc"]["name"] != fields["tested_mhc"]["name"]
    assert fields["is_potential_contaminant"]["role"] == "annotation"
    assert "explicit_absence" in fields["cytokine_exposure"]["null_states"]
    assert "explicit_absence" not in fields["donor_id"]["null_states"]


def test_index_requirements_are_weaker_than_task_requirements():
    binding = describe_evidence("mhc_binding")
    quantitative = describe_task("mhc_quantitative_binding")
    assert "quantitative_value" not in binding["index_required"]
    assert "quantitative_value" in quantitative["required"]
    assert quantitative["evidence"] == ("mhc_binding",)

    ms = describe_task("mhc_positive_presentation")
    functional = describe_task("tcell_functional_response")
    multimer = describe_task("tcell_multimer_cell_binding")
    assert ms["evidence"] == ("mhc_positive_ms",)
    assert "outcome_status" in ms["required"]
    assert {"is_non_peptide_ligand", "is_potential_contaminant"} <= set(ms["required"])
    assert "qualitative_result" not in ms["required"]  # curated MS tables may leave it blank
    assert functional["evidence"] != multimer["evidence"]
    assert all(describe_task(name)["status"] == "v2_contract_only" for name in list_tasks())


def test_strict_description_and_deterministic_json():
    with pytest.raises(KeyError, match="Unknown evidence type"):
        describe_evidence("unknown")
    with pytest.raises(KeyError, match="Unknown task"):
        describe_task("unknown")
    with pytest.raises(ValueError, match="Fields unavailable"):
        describe_evidence("tcell_availability", fields=("peptide_sequence",))
    with pytest.raises(ValueError, match="Duplicate requested fields"):
        describe_evidence("mhc_binding", fields=("tested_mhc", "tested_mhc"))
    with pytest.raises(ValueError, match="Unknown evidence kind"):
        list_evidence(kind="unknown")

    selected = describe_evidence("mhc_binding", fields=("tested_mhc", "measurement_unit"))
    assert [field["name"] for field in selected["fields"]] == [
        "tested_mhc",
        "measurement_unit",
    ]
    first = json.dumps(describe_catalog(), sort_keys=True)
    assert first == json.dumps(describe_catalog(), sort_keys=True)


def test_baselines_have_explicit_generation_and_count_units():
    baselines = {item.population: item for item in BASELINES}
    assert baselines["mhc_positive_ms"].count_unit == "assay_rows"
    assert baselines["tcell_2026_10_20"].count_unit == "distinct_iedb_assay_ids"
    assert baselines["tcell_2026_10_20"].n_rows == 578_529
    assert (
        baselines["tcell_functional_peptide"].n_rows + baselines["tcell_multimer_peptide"].n_rows
        == 55_844
    )
    assert baselines["bulk_v4"].count_unit == "measurement_rows"
    assert all(item.generation for item in BASELINES)


def test_catalog_import_does_not_build_or_download(tmp_path):
    data_root = tmp_path / "unused-data"
    env = os.environ.copy()
    env["HITLIST_DATA_DIR"] = str(data_root)
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            "from hitlist.evidence_catalog import describe_catalog; "
            "print(describe_catalog()['schema_version'])",
        ],
        env=env,
        check=True,
        capture_output=True,
        text=True,
        cwd=Path(__file__).parents[1],
    )
    assert probe.stdout.strip() == "2"
    assert not data_root.exists()
