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
    assert {"external_mhc_peptide_quantity", "external_digest_peptide_quantity"} <= observations
    assert {"tcell_functional_peptide", "tcell_multimer_peptide"} <= projections
    assert {"peptide_protein_occurrence", "source_contributor", "reviewed_lineage"} <= relations
    assert {
        "tcell_reported_outcome",
        "tcell_antigen_claim",
        "tcell_source_contributor",
        "tcell_cedar_mutation_claim",
        "tcell_eligibility",
    } <= relations
    assert {"external_source_cell", "external_panel_sequence", "external_derivation_input"} <= (
        relations
    )

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
    for name in (
        "assay_context",
        "curated_study",
        "curated_material_arm",
        "arm_control_link",
        "arm_factor",
        "arm_mhc_claim",
        "protocol_claim",
        "curation_amendment",
        "reviewed_lineage",
        "bulk_ms_peptide",
        "bulk_ms_protein",
        "line_expression_gene",
        "peptide_protein_occurrence",
        "peptide_mapping_coverage",
        "bulk_protocol_scope",
        "line_expression_source",
        "line_anchor_claim",
        "external_mhc_peptide_quantity",
        "external_digest_peptide_quantity",
        "external_source_cell",
        "external_panel_sequence",
        "external_assay_channel",
        "external_contrast",
        "external_derivation_input",
        "external_context_assertion",
        "external_sdrf_cell",
        "external_deposited_file",
        "external_source_asset",
        "tcell_availability",
        "tcell_functional_peptide",
        "tcell_multimer_peptide",
        "tcell_reported_outcome",
        "tcell_antigen_claim",
        "tcell_source_contributor",
        "tcell_cedar_mutation_claim",
        "tcell_eligibility",
    ):
        assert describe_evidence(name)["status"] == "v2_materialized"
    assert describe_evidence("bulk_ms_peptide")["endpoints"] == ("bulk_peptide_detection",)
    assert describe_evidence("line_expression_gene")["index_required_any_of"] == (
        ("gene_id", "gene_name"),
    )
    assert describe_evidence("line_expression_transcript")["status"] == "v2_contract_only"
    external = describe_evidence("external_mhc_peptide_quantity")
    digest = describe_evidence("external_digest_peptide_quantity")
    assert external["source_population"] == digest["source_population"] == "pxd057588"
    assert external["endpoints"] != digest["endpoints"]
    assert digest["native_selector"] == "table=11"
    assert functional["materialized_artifact"] == multimer["materialized_artifact"]
    assert functional["native_selector"] != multimer["native_selector"]
    assert describe_evidence("tcell_cedar_mutation_claim")["materialized_artifact"] == (
        "tcell/typed/cedar_mutation_claims.parquet"
    )
    for candidate in (
        "external_proteasome_fragment_quantity",
        "external_paired_protein_quantity",
        "external_calibrated_target_quantity",
    ):
        contract = describe_evidence(candidate)
        assert contract["status"] == "v2_contract_only"
        assert contract["materialized_artifact"] == ""


def test_labels_and_biological_state_have_distinct_roles():
    fields = {
        field["name"]: field for item in describe_catalog()["evidence"] for field in item["fields"]
    }
    assert fields["cytokine_exposure"]["role"] == "biological_context"
    assert fields["gene_perturbation"]["role"] == "biological_context"
    assert fields["assay_readout"]["role"] == "protocol"
    assert fields["instrument"]["role"] == "protocol"
    assert fields["arm_id"]["role"] == "biological_context"
    assert fields["sample_attribution"]["role"] == "provenance"
    assert fields["material_link_status"]["role"] == "relation"
    assert fields["search_enzyme"]["role"] == "protocol"
    assert fields["quantity_scale"]["role"] == "target"
    assert fields["measured_pool"]["role"] == "target"
    assert fields["partner_population"]["role"] == "biological_context"
    assert fields["normalization_policy"]["role"] == "protocol"
    assert fields["source_cell"]["role"] == "provenance"
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
    assert describe_task("gene_expression")["required_any_of"] == (("gene_id", "gene_name"),)
    assert "material_id" not in describe_task("bulk_peptide_detection")["required"]
    assert (
        "normalization_universe"
        in describe_task("external_digest_peptide_relative_quantity")["required"]
    )
    assert "material_link_precision" in describe_task("external_mhc_relative_quantity")["required"]
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
    assert baselines["bulk_v4"].count_unit == "source_result_rows"
    assert baselines["peptide_protein_occurrence"].n_rows == 6_205_013
    assert baselines["peptide_mapping_coverage"].n_rows == 1_506_664
    assert baselines["pxd057588"].n_rows == 35_772
    assert baselines["external_mhc_peptide_quantity"].n_rows == 83_136
    assert baselines["external_digest_peptide_quantity"].n_rows == 225_990
    assert baselines["external_source_cell"].n_rows == 851_830
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
    assert probe.stdout.strip() == "6"
    assert not data_root.exists()
