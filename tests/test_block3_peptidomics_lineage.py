"""Offline tests for Block 3 quantitative peptidomics & specimen-lineage guards.

Covers:
- #811: PeptidomicsDB qualification, capture-vs-compartment hazards (PPD017/PPD070/PPD016),
        and PEAKS reanalysis provenance
- #726: HAP1 MixMHCpred 2.2 selection-covariate contract & non-filtering 1:1 join
- #817: Nanaware 2024 quantitative grain, accession-token normalization, and DC channel guard
- #705: HLA Ligand Atlas SDRF/inventory specimen-lineage & QC-inclusion audit guard
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

import hitlist
from hitlist.atlas import atlas_specimen_lineage_contract, audit_atlas_sdrf_specimen_lineage
from hitlist.downloads import available_dataset_catalog
from hitlist.hap1_selection_covariates import (
    attach_hap1_selection_covariates,
    hap1_selection_covariates_contract,
    validate_hap1_covariate_metadata,
)
from hitlist.lymph_composition import (
    audit_nanaware_accession_token_normalization,
    guard_nanaware_quantity_aggregation,
    nanaware_quantitative_grain_contract,
)
from hitlist.peptidomicsdb_qualification import (
    guard_peptidomicsdb_reanalysis_rows,
    peptidomicsdb_source_contract,
    qualify_peptidomicsdb_samples,
)


def test_peptidomicsdb_contract_and_capture_vs_compartment_hazards():
    contract = peptidomicsdb_source_contract()
    assert contract["tracked_issue"] == 811
    assert contract["doi"] == "10.1093/nar/gkag1032"
    assert contract["current_n_qualified_native_apc_cohorts"] == 0
    assert contract["ooxml_verified_nonempty_values"] == 63761
    assert (
        contract["metadata_workbooks"]["sample-download.xlsx"]["sha256"]
        == "739fd96bf13f90988cc00548b0250d640a0f97bbf25ecb737b7d89fc165b1f39"
    )

    samples = pd.DataFrame(
        [
            # Hazard 1: PPD017/PXD021177 sample PPD0170004 tagged MHC-I + Intracellular (PMID 34211107 HLA-IP)
            {
                "PeptidomicsDB project ID": "PPD017",
                "PeptidomicsDB sample ID": "PPD0170004",
                "Original project accession": "PXD021177",
                "Raw file": "JY_whole_cell.raw",
                "Condition": "Whole cell",
                "Species": "Homo sapiens",
                "Tissue": "JY",
                "Peptidomics category": "MHC-I immunopeptidomics; Intracellular peptidomics",
                "Enzyme": "None",
            },
            # Hazard 2: PPD070/PXD009801 generically tagged Peptidomics (PMID 24715725 dynamic SILAC HLA-I)
            {
                "PeptidomicsDB project ID": "PPD070",
                "PeptidomicsDB sample ID": "PPD0700001",
                "Original project accession": "PXD009801",
                "Raw file": "U937_silac.raw",
                "Condition": "SILAC",
                "Species": "Homo sapiens",
                "Tissue": "U937; RPMI; JY",
                "Peptidomics category": "Peptidomics",
                "Enzyme": "None",
            },
            # Solely Intracellular-tagged enteroendocrine cells -> excluded non-APC
            {
                "PeptidomicsDB project ID": "PPD016",
                "PeptidomicsDB sample ID": "PPD0160001",
                "Original project accession": "PXD033369",
                "Raw file": "entero_01.raw",
                "Condition": "Control",
                "Species": "Mus musculus",
                "Tissue": "Enteroendocrine cells",
                "Peptidomics category": "Intracellular peptidomics",
                "Enzyme": "None",
            },
            # Unreviewed generic project with Enzyme=None -> deferred (Enzyme=None != native extraction!)
            {
                "PeptidomicsDB project ID": "PPD999",
                "PeptidomicsDB sample ID": "PPD9990001",
                "Original project accession": "PXD999999",
                "Raw file": "mystery.raw",
                "Condition": "Baseline",
                "Species": "Homo sapiens",
                "Tissue": "PBMC",
                "Peptidomics category": "Peptidomics",
                "Enzyme": "None",
            },
        ]
    )

    qdf, summary = qualify_peptidomicsdb_samples(samples)
    assert summary["n_rows"] == 4
    assert summary["n_eligible_rows"] == 0
    assert summary["n_excluded_rows"] == 3
    assert summary["n_deferred_rows"] == 1
    assert summary["establishes_specimen_independence"] is False
    assert summary["establishes_cellular_concentration"] is False

    by_id = qdf.set_index("sample_id")
    # PPD0170004 excluded with HLA-IP provenance preserved separately from raw category
    assert by_id.loc["PPD0170004", "disposition"] == "excluded"
    assert (
        by_id.loc["PPD0170004", "disposition_reason"]
        == "capture_vs_compartment_hazard_hla_ip_not_free_intracellular"
    )
    assert by_id.loc["PPD0170004", "molecular_capture_enrichment"] == "hla_immunoaffinity"
    assert (
        by_id.loc["PPD0170004", "source_category_raw"]
        == "MHC-I immunopeptidomics; Intracellular peptidomics"
    )

    # PPD0700001 excluded despite generic Peptidomics category
    assert by_id.loc["PPD0700001", "disposition"] == "excluded"
    assert (
        by_id.loc["PPD0700001", "disposition_reason"]
        == "generic_peptidomics_category_hides_hla_i_dynamic_silac"
    )

    # PPD0160001 excluded as enteroendocrine non-APC
    assert by_id.loc["PPD0160001", "disposition"] == "excluded"
    assert by_id.loc["PPD0160001", "disposition_reason"] == "non_apc_enteroendocrine_cells"

    # Unreviewed generic project with Enzyme=None remains deferred AND post_extraction_digestion=unknown!
    assert by_id.loc["PPD9990001", "disposition"] == "deferred"
    assert by_id.loc["PPD9990001", "ms_search_enzyme"] == "None"
    assert by_id.loc["PPD9990001", "post_extraction_digestion"] == "unknown"

    # Catalog symmetry check
    cat = available_dataset_catalog(category="perturbation_and_calibration")
    assert "peptidomicsdb_metadata" in cat
    assert cat["peptidomicsdb_metadata"]["group"] == "peptidomics_qualification"


def test_peptidomicsdb_reanalysis_guard_preserves_methods_and_missing_vs_zero():
    rows = pd.DataFrame(
        [
            {
                "project_id": "PPD001",
                "sample_id": "PPD0010001",
                "peptide": "GVYDGREHTV",
                "peaks_version": "PEAKS Studio 11",
                "identification_method": "DB Search",
                "origin_alternatives_raw": "P43358;Q99999",
                "quantity_raw": "12345.6",
            },
            {
                "project_id": "PPD001",
                "sample_id": "PPD0010001",
                "peptide": "SIINFEKL",
                "peaks_version": "PEAKS Studio 11",
                "identification_method": "DeepNovo",
                "origin_alternatives_raw": "",
                "quantity_raw": "0",
            },
            {
                "project_id": "PPD002",
                "sample_id": "PPD0020001",
                "peptide": "GILGFVFTL",
                "peaks_version": "PEAKS Studio 11",
                "identification_method": "Sequence Variants",
                "origin_alternatives_raw": "P03485",
                "quantity_raw": "",
            },
        ]
    )
    out, summary = guard_peptidomicsdb_reanalysis_rows(rows)
    assert summary["n_rows"] == 3
    assert summary["cross_project_intensity_comparable"] is False
    assert summary["multi_project_comparison_warning"] is True
    assert summary["n_by_identification_method"] == {
        "DB Search": 1,
        "DeepNovo": 1,
        "Sequence Variants": 1,
    }
    assert summary["n_by_quantity_value_status"] == {
        "missing": 1,
        "observed_positive": 1,
        "observed_zero": 1,
    }
    assert not out["cross_project_intensity_comparable"].any()

    bad = rows.copy()
    bad.loc[0, "identification_method"] = "Collapsed"
    with pytest.raises(ValueError, match="Unsupported PEAKS identification_method"):
        guard_peptidomicsdb_reanalysis_rows(bad)


def test_hap1_selection_covariates_non_filtering_join_and_provenance():
    contract = hap1_selection_covariates_contract()
    assert contract["tracked_issue"] == 726
    assert contract["original_paper_outputs_status"] == "unrecovered"
    assert contract["hap1_genotype"] == ["HLA-A*02:01", "HLA-B*40:01", "HLA-C*03:04"]

    meta = {
        "derivation_origin": "newly_computed_pinned_derivation",
        "predictor_name": "MixMHCpred",
        "predictor_version": "2.2",
        "model_config": "default_class1_v2.2",
        "score_type": "percent_rank",
        "score_direction": "lower_is_better",
        "rank_reference_population": "random_human_peptides_2.2",
        "assignment_rule": "minimum_percent_rank_across_hap1_alleles",
        "tie_rule": "retain_all_tied_alleles_unassigned",
        "threshold": "2.0",
        "input_sha256": "a" * 64,
        "software_sha256": "b" * 64,
        "model_sha256": "c" * 64,
        "license_and_access": "academic_noncommercial_local_binary",
        "training_overlap_caveat": "MixMHCpred 2.2 trained on public HLA-I ligands; potential overlap with HAP1 peptides",
    }
    validate_hap1_covariate_metadata(meta)

    with pytest.raises(ValueError, match="retroactively attributed"):
        validate_hap1_covariate_metadata({**meta, "attributed_to_original_paper": True})

    response_df = pd.DataFrame(
        [
            {
                "peptide_sequence": "GILGFVFTL",
                "split": "train",
                "parent_origins": "P1;P2",
                "log10_ratio": -0.2,
            },
            {
                "peptide_sequence": "TIEDPEPTD",
                "split": "val",
                "parent_origins": "P3",
                "log10_ratio": 0.1,
            },
            # Unsupported 15-mer missing from predictor output -> MUST NOT be filtered or imputed!
            {
                "peptide_sequence": "AAAAAAAAAAAAAAA",
                "split": "test",
                "parent_origins": "P4",
                "log10_ratio": 0.4,
            },
        ]
    )
    covariate_df = pd.DataFrame(
        [
            {"peptide_sequence": "GILGFVFTL", "allele": "HLA-A*02:01", "score": 0.05, "rank": 0.05},
            {"peptide_sequence": "GILGFVFTL", "allele": "HLA-B*40:01", "score": 12.0, "rank": 12.0},
            {"peptide_sequence": "GILGFVFTL", "allele": "HLA-C*03:04", "score": 8.5, "rank": 8.5},
            {"peptide_sequence": "TIEDPEPTD", "allele": "HLA-A*02:01", "score": 1.0, "rank": 1.0},
            {"peptide_sequence": "TIEDPEPTD", "allele": "HLA-B*40:01", "score": 1.0, "rank": 1.0},
            {"peptide_sequence": "TIEDPEPTD", "allele": "HLA-C*03:04", "score": 9.0, "rank": 9.0},
        ]
    )

    joined, audit = attach_hap1_selection_covariates(response_df, covariate_df, metadata=meta)
    assert audit["preserves_all_response_rows"] is True
    assert audit["applies_binding_eligibility_filter"] is False
    assert audit["imputes_missing_scores"] is False
    assert audit["n_response_rows"] == 3
    assert audit["n_covered_rows"] == 2
    assert audit["n_missing_or_unsupported_rows"] == 1
    assert audit["n_tied_rows"] == 1

    # Splits and parent origins untouched
    assert list(joined["split"]) == ["train", "val", "test"]
    assert list(joined["parent_origins"]) == ["P1;P2", "P3", "P4"]

    # Row 0: clear winner + ALL 3 candidate alleles preserved in JSON
    assert joined.loc[0, "inferred_best_allele"] == "HLA-A*02:01"
    assert bool(joined.loc[0, "inferred_best_allele_is_tie"]) is False
    all_scores = json.loads(joined.loc[0, "selection_covariate_all_alleles_json"])
    assert set(all_scores) == {"HLA-A*02:01", "HLA-B*40:01", "HLA-C*03:04"}
    assert all_scores["HLA-B*40:01"]["score"] == 12.0

    # Row 1: exact tie preserved explicitly
    assert bool(joined.loc[1, "inferred_best_allele_is_tie"]) is True
    assert joined.loc[1, "inferred_best_allele"] == ""
    assert joined.loc[1, "inferred_best_allele_tie_set"] == "HLA-A*02:01;HLA-B*40:01"

    # Row 2: missing/unsupported NOT dropped and NOT imputed
    assert joined.loc[2, "selection_covariate_status"] == "missing_or_unsupported"
    assert pd.isna(joined.loc[2, "inferred_best_score"])

    # Omitting non-best candidate alleles must raise!
    incomplete = covariate_df[covariate_df["allele"] == "HLA-A*02:01"]
    with pytest.raises(ValueError, match="omits non-best candidate alleles"):
        attach_hap1_selection_covariates(response_df, incomplete, metadata=meta)


def test_nanaware_quantitative_grain_and_accession_normalization_guards():
    contract = nanaware_quantitative_grain_contract()
    assert contract["tracked_issue"] == 817
    assert contract["safe_summed_total_peptide_quantity_established"] is False
    assert contract["dc_declared_replicates"] == {"cervical": 3, "mesenteric": 6}
    assert contract["dc_exported_quantity_channels"] == {"cervical": 3, "mesenteric": 3}
    assert contract["dc_channel_to_replicate_correspondence"] == "unresolved"

    ok = guard_nanaware_quantity_aggregation("representative_charge_feature")
    assert ok["allowed"] is True

    with pytest.raises(ValueError, match="disallowed"):
        guard_nanaware_quantity_aggregation("sum_across_charge_features")
    with pytest.raises(ValueError, match="disallowed"):
        guard_nanaware_quantity_aggregation("pool_db_and_library_as_replicates")
    with pytest.raises(ValueError, match="remains unresolved"):
        guard_nanaware_quantity_aggregation(
            "representative_charge_feature",
            dc_channel_map={"ch1": "cervical_1"},
        )

    norm = audit_nanaware_accession_token_normalization(
        raw_labels=["sp|P02768|ALBU_HUMAN", "P01857|IGHG1_HUMAN", "Q99999"],
        reference_accessions=["P02768", "sp|P01857|IGHG1_HUMAN", "tr|P01857|ALT"],
    )
    assert norm["preserves_original_raw_labels"] is True
    assert norm["substitutes_external_modern_references"] is False
    assert norm["n_exact_raw_matches"] == 0
    assert norm["n_normalized_token_matches"] == 2
    assert norm["n_missing_deposited_references"] == 1
    assert norm["rows"][0]["raw_label"] == "sp|P02768|ALBU_HUMAN"
    assert norm["rows"][0]["candidate_deposited_accessions"] == ["P02768"]
    assert norm["rows"][1]["candidate_deposited_accessions"] == [
        "sp|P01857|IGHG1_HUMAN",
        "tr|P01857|ALT",
    ]


def test_atlas_specimen_lineage_contract_and_audit_guard():
    contract = atlas_specimen_lineage_contract()
    assert contract["tracked_issue"] == 705
    assert contract["pinned_sources"]["PXD019643_community_annotated.sdrf.tsv"]["n_rows"] == 1468
    assert contract["n_multi_source_name_primary_groups"] == 5
    assert contract["qc_inclusion_status"] == "unknown"
    assert contract["observation_specimen_status"] == "unknown"
    assert contract["acquisition_independence_verdict"] == "inconclusive"

    sdrf_df = pd.DataFrame(
        [
            {
                "source name": "AUT01-DN17-Heart",
                "comment[data file]": "run1.mzML",
                "donor": "AUT01-DN17",
                "tissue": "Heart",
                "hla_class": "HLA-I",
            },
            {
                "source name": "AUT01-DN17-Duo-A",
                "comment[data file]": "run2.mzML",
                "donor": "AUT01-DN17",
                "tissue": "Duodenum",
                "hla_class": "HLA-II",
            },
            {
                "source name": "AUT01-DN17-Duo-B",
                "comment[data file]": "run3.mzML",
                "donor": "AUT01-DN17",
                "tissue": "duodenum ",
                "hla_class": "HLA-II",
            },
        ]
    )
    inv_df = pd.DataFrame(
        [
            {"fileName": "run1.mzML", "fileCategory": "PEAK"},
            {"fileName": "run2.mzML", "fileCategory": "PEAK"},
            {"fileName": "run3.mzML", "fileCategory": "PEAK"},
            {"fileName": "run1.raw", "fileCategory": "RAW"},
        ]
    )
    groups_df = pd.DataFrame(
        [
            {
                "donor": "AUT01-DN17",
                "tissue": "Heart",
                "hla_class": "HLA-I",
                "n_observations": 1000,
            },
            {
                "donor": "AUT01-DN17",
                "tissue": "Duodenum",
                "hla_class": "HLA-II",
                "n_observations": 500,
            },
        ]
    )
    audit = audit_atlas_sdrf_specimen_lineage(sdrf_df, inv_df, groups_df)
    assert audit["n_exact_unique_peak_matches"] == 3
    assert audit["n_unmatched_peak_files"] == 0
    assert audit["n_multi_source_name_groups"] == 1
    assert audit["n_multi_source_name_observations"] == 500
    assert audit["n_observations_qc_resolved"] == 0
    assert audit["qc_inclusion_status"] == "unknown"
    assert audit["observation_specimen_status"] == "unknown"
    assert audit["acquisition_independence_verdict"] == "inconclusive"

    # Public lazy exports resolve cleanly
    assert hitlist.peptidomicsdb_source_contract is peptidomicsdb_source_contract
    assert hitlist.qualify_peptidomicsdb_samples is qualify_peptidomicsdb_samples
    assert hitlist.attach_hap1_selection_covariates is attach_hap1_selection_covariates
    assert hitlist.nanaware_quantitative_grain_contract is nanaware_quantitative_grain_contract
    assert hitlist.atlas_specimen_lineage_contract is atlas_specimen_lineage_contract
