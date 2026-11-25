"""Offline tests for canine DLA qualification contracts, vital-organ guard and adapters."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

import hitlist
from hitlist.canine_dla_qualification import (
    canine_dla_atlas_qualification,
    canine_dla_source_contract,
    evaluate_canine_vital_organ_blacklist,
    load_lang_2026_splenic_dla_dr_ligands,
    load_miyamae_2026_dla88_ligands,
    summarize_canine_bulk_protein_evidence,
)
from hitlist.downloads import available_dataset_catalog


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_qualification_contract_and_catalog_symmetry():
    contract = canine_dla_atlas_qualification()
    assert contract["schema_version"] == 1
    assert set(contract["tracked_issues"]) == {666, 674, 676, 677}
    policy = contract["vital_organ_blacklist_policy"]
    assert policy["current_coverage_status"] == "missing"
    assert policy["validated_empty_blacklist"] is False
    assert policy["qualifying_tissues"] == ["heart", "brain", "lung"]
    assert (
        policy["min_distinct_dogs"]
        if "min_distinct_dogs" in policy
        else policy["min_distinct_resolved_dogs"] == 2
    )
    assert len(policy["source_audit_ledger"]) >= 10

    miyamae = canine_dla_source_contract("miyamae_2026_dla88")
    assert miyamae["license"] == "cc_no"
    assert miyamae["redistributable"] is False
    assert miyamae["satisfies_vital_organ_blacklist"] is False

    lang = canine_dla_source_contract("lang_2026_splenic_dla_dr")
    assert lang["pmid"] == "42752862"
    assert lang["reviewed_supplement_archive"]["contains_ms_peptide_sequences"] is False
    assert lang["reviewed_supplement_archive"]["size_bytes"] == 1866144
    assert (
        lang["reviewed_supplement_archive"]["sha256"]
        == "d79d05543f88a31cd70fd6ee68aaaffbfb72ab2bfd4d56c078a4f130b9c6c4d5"
    )
    assert set(lang["verified_table_s2_donors"]) == {"1", "2"}
    for donor in lang["verified_table_s2_donors"].values():
        assert donor["dra_genotype"] == "unknown"
        assert donor["health_status"] == "unknown"
        assert donor["treatment_status"] == "unknown"

    with pytest.raises(ValueError, match="Unknown canine DLA source"):
        canine_dla_source_contract("nonexistent_source")

    catalog = available_dataset_catalog(category="canine_and_cross_species")
    assert "miyamae_2026_dla88" in catalog
    assert "lang_2026_splenic_dla_dr" in catalog
    assert catalog["miyamae_2026_dla88"]["group"] == "canine_dla"
    assert catalog["lang_2026_splenic_dla_dr"]["group"] == "canine_dla"

    # Public lazy exports resolve cleanly
    assert hitlist.evaluate_canine_vital_organ_blacklist is evaluate_canine_vital_organ_blacklist
    assert hitlist.load_miyamae_2026_dla88_ligands is load_miyamae_2026_dla88_ligands
    assert hitlist.load_lang_2026_splenic_dla_dr_ligands is load_lang_2026_splenic_dla_dr_ligands
    assert hitlist.summarize_canine_bulk_protein_evidence is summarize_canine_bulk_protein_evidence


def test_vital_organ_blacklist_missing_coverage_never_validated_empty():
    empty_result = evaluate_canine_vital_organ_blacklist(None)
    assert empty_result["coverage_status"] == "missing"
    assert empty_result["validated_empty_blacklist"] is False
    assert empty_result["n_qualifying_dogs"] == 0
    assert empty_result["n_blacklisted_peptides"] == 0

    # Non-qualifying rows (tumors, cell lines, pools, bulk proteomics, spleen, predictions,
    # nested windows, human HLA) must all be rejected and keep coverage_status="missing".
    non_qualifying = pd.DataFrame(
        [
            {
                "peptide": "KLQALVEEV",
                "evidence_kind": "bulk_proteomics",
                "mhc_species": "Canis lupus familiaris",
                "tissue": "heart",
                "tissue_status": "nonmalignant",
                "is_cell_line": False,
                "donor_status": "pooled",
                "donor_id": "pool_1",
            },
            {
                "peptide": "KLQALVEEV",
                "evidence_kind": "dla_elution",
                "mhc_species": "Canis lupus familiaris",
                "tissue": "heart",
                "tissue_status": "nonmalignant",
                "is_cell_line": False,
                "donor_status": "pooled",
                "donor_id": "pool_1",
            },
            {
                "peptide": "KLQALVEEV",
                "evidence_kind": "dla_elution",
                "mhc_species": "Canis lupus familiaris",
                "tissue": "spleen",
                "tissue_status": "nonmalignant",
                "is_cell_line": False,
                "donor_status": "resolved_individual",
                "donor_id": "lang_donor_1",
            },
            {
                "peptide": "KLQALVEEV",
                "evidence_kind": "dla_elution",
                "mhc_species": "Canis lupus familiaris",
                "tissue": "heart",
                "tissue_status": "malignant",
                "is_cell_line": False,
                "donor_status": "resolved_individual",
                "donor_id": "tumor_dog_1",
            },
            {
                "peptide": "KLQALVEEV",
                "evidence_kind": "dla_elution",
                "mhc_species": "Canis lupus familiaris",
                "tissue": "heart",
                "tissue_status": "nonmalignant",
                "is_cell_line": True,
                "donor_status": "resolved_individual",
                "donor_id": "cell_line_1",
            },
            {
                "peptide": "KLQALVEEV",
                "evidence_kind": "binding_prediction",
                "mhc_species": "Canis lupus familiaris",
                "tissue": "heart",
                "tissue_status": "nonmalignant",
                "is_cell_line": False,
                "donor_status": "resolved_individual",
                "donor_id": "dog_pred",
            },
            {
                "peptide": "KLQALVEEV",
                "evidence_kind": "dla_elution",
                "match_kind": "nested_window",
                "mhc_species": "Canis lupus familiaris",
                "tissue": "heart",
                "tissue_status": "nonmalignant",
                "is_cell_line": False,
                "donor_status": "resolved_individual",
                "donor_id": "dog_nest",
            },
            {
                "peptide": "KLQALVEEV",
                "evidence_kind": "dla_elution",
                "mhc_species": "Homo sapiens",
                "tissue": "heart",
                "tissue_status": "nonmalignant",
                "is_cell_line": False,
                "donor_status": "resolved_individual",
                "donor_id": "human_1",
            },
        ]
    )
    res = evaluate_canine_vital_organ_blacklist(non_qualifying)
    assert res["coverage_status"] == "missing"
    assert res["validated_empty_blacklist"] is False
    assert res["n_qualifying_rows"] == 0
    assert res["n_excluded_by_reason"] == {
        "bulk_or_non_presentation_evidence": 1,
        "cell_line_excluded": 1,
        "malignant_or_unresolved_tissue_status": 1,
        "non_canine_mhc_species": 1,
        "non_exact_or_nested_window": 1,
        "non_vital_organ_tissue": 1,
        "prediction_or_binding_not_elution": 1,
        "unresolved_or_pooled_donor": 1,
    }


def test_vital_organ_blacklist_deduplicates_runs_tissues_and_donor_aliases():
    obs = pd.DataFrame(
        [
            # Same dog across two runs, two tissues (heart + lung), and a republished alias -> MUST count as 1 dog!
            {
                "peptide": "SINGLEDAG",
                "evidence_kind": "primary_tissue_dla_elution",
                "mhc_species": "Canis lupus familiaris",
                "tissue": "heart",
                "tissue_status": "nonmalignant",
                "is_cell_line": False,
                "donor_status": "resolved_individual",
                "donor_id": "dog_A",
                "mhc_restriction": "DLA-88*003:02",
                "source_id": "study_1",
            },
            {
                "peptide": "SINGLEDAG",
                "evidence_kind": "primary_tissue_dla_elution",
                "mhc_species": "Canis lupus familiaris",
                "tissue": "lung",
                "tissue_status": "nonmalignant",
                "is_cell_line": False,
                "donor_status": "resolved_individual",
                "donor_id": "dog_A_repub",
                "mhc_restriction": "DLA-88*003:02",
                "source_id": "study_2_reanalysis",
            },
            # Two distinct dogs across different alleles and vital organs (cerebellum->brain + heart) -> BLACKLISTED!
            {
                "peptide": "SHAREDPEP",
                "evidence_kind": "primary_tissue_dla_elution",
                "mhc_species": "Canis lupus familiaris",
                "tissue": "cerebellum",
                "tissue_status": "nonmalignant",
                "is_cell_line": False,
                "donor_status": "resolved_individual",
                "donor_id": "dog_A",
                "mhc_restriction": "DLA-88*003:02",
                "source_id": "study_1",
            },
            {
                "peptide": "SHAREDPEP",
                "evidence_kind": "primary_tissue_dla_elution",
                "mhc_species": "Canis lupus familiaris",
                "tissue": "heart",
                "tissue_status": "nonmalignant",
                "is_cell_line": False,
                "donor_status": "resolved_individual",
                "donor_id": "dog_B",
                "mhc_restriction": "DLA-88*501:01",
                "source_id": "study_1",
            },
        ]
    )
    res = evaluate_canine_vital_organ_blacklist(
        obs,
        donor_aliases={"dog_A_repub": "dog_A"},
        min_distinct_dogs=2,
    )
    # Only heart/brain/lung have 2/1/1 dogs respectively -> partial (not established across all 3 organs yet)
    assert res["coverage_status"] == "partial"
    assert res["n_qualifying_dogs"] == 2
    assert res["n_dogs_by_tissue"] == {"heart": 2, "brain": 1, "lung": 1}
    assert res["n_blacklisted_peptides"] == 1
    row = res["blacklist_df"].iloc[0]
    assert row["peptide"] == "SHAREDPEP"
    assert row["n_dogs"] == 2
    assert row["dog_ids"] == "dog_A;dog_B"
    assert row["tissues"] == "brain;heart"
    assert row["alleles_observed"] == "DLA-88*003:02;DLA-88*501:01"


def test_miyamae_2026_offline_loader_separates_measured_from_predictions(tmp_path):
    table = tmp_path / "miyamae_fixture.tsv"
    pd.DataFrame(
        [
            {
                "peptide": "KLQALVEEV",
                "dla_allele": "DLA-88*501:01",
                "parental_line": "CMeC-1",
                "construct_id": "CMeC-1_DLA88_50101",
                "residual_endogenous_mhc": "DLA-88_KO_pending_verification",
                "acquisition_id": "run_01",
                "modifications_raw": "",
                "source_proteins_raw": "XP_001;XP_002",
                "confidence_or_fdr": "FDR<=0.01",
                "search_reference": "CanFam4_RefSeq",
                "evidence_kind": "measured_eluted_ligand",
            },
            # Same peptide observed on a second allele population -> must stay separate!
            {
                "peptide": "KLQALVEEV",
                "dla_allele": "DLA-88*012:01",
                "parental_line": "CMeC-1",
                "construct_id": "CMeC-1_DLA88_01201",
                "residual_endogenous_mhc": "DLA-88_KO_pending_verification",
                "acquisition_id": "run_02",
                "modifications_raw": "M(Ox)",
                "source_proteins_raw": "XP_001",
                "confidence_or_fdr": "FDR<=0.01",
                "search_reference": "CanFam4_RefSeq",
                "evidence_kind": "measured_eluted_ligand",
            },
            {
                "peptide": "AALVEEVKL",
                "dla_allele": "DLA-88*999:01",
                "parental_line": "CMeC-1",
                "construct_id": "in_silico",
                "residual_endogenous_mhc": "n/a",
                "acquisition_id": "",
                "modifications_raw": "",
                "source_proteins_raw": "XP_003",
                "confidence_or_fdr": "",
                "search_reference": "CanFam4_RefSeq",
                "evidence_kind": "motif_neighbour_extrapolation",
            },
            {
                "peptide": "VEEVKLAAL",
                "dla_allele": "DLA-88*501:01",
                "parental_line": "CMeC-1",
                "construct_id": "normal_rna_screen",
                "residual_endogenous_mhc": "",
                "acquisition_id": "",
                "modifications_raw": "",
                "source_proteins_raw": "XP_004",
                "confidence_or_fdr": "",
                "search_reference": "",
                "evidence_kind": "normal_expression_screen",
            },
        ]
    ).to_csv(table, sep="\t", index=False)

    df, summary = load_miyamae_2026_dla88_ligands(table, expected_sha256=_sha256(table))
    assert summary["license"] == "cc_no"
    assert summary["redistributable"] is False
    assert summary["n_input_rows"] == 4
    assert summary["n_measured_observations"] == 2
    assert summary["n_excluded_non_measured_rows"] == 2
    assert summary["n_unique_peptides"] == 1
    assert summary["n_alleles"] == 2
    assert summary["satisfies_vital_organ_blacklist"] is False
    assert len(df) == 2
    assert df["is_cell_line"].all()
    assert not df["satisfies_vital_organ_blacklist"].any()

    # Feeding Miyamae rows into #674 blacklist guard must yield missing coverage!
    guard = evaluate_canine_vital_organ_blacklist(df)
    assert guard["coverage_status"] == "missing"
    assert guard["n_qualifying_rows"] == 0

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_miyamae_2026_dla88_ligands(table, expected_sha256="0" * 64)


def test_lang_2026_offline_loader_joins_table_s2_and_never_satisfies_674(tmp_path):
    table = tmp_path / "lang_ms_fixture.tsv"
    pd.DataFrame(
        [
            {
                "peptide": "GRFSVVAAQKPELG",
                "sample_id": "1",
                "source_record_id": "lang_s1_row1",
                "evidence_kind": "measured_ms_eluted_ligand",
                "in_motif_subset": "true",
                "modifications_raw": "",
                "source_proteins_raw": "A0A8I3N123;A0A8I3N456",
                "confidence_or_fdr": "q=0.005",
            },
            {
                "peptide": "GRFSVVAAQKPELG",
                "sample_id": "2",
                "source_record_id": "lang_s2_row1",
                "evidence_kind": "measured_ms_eluted_ligand",
                "in_motif_subset": "false",
                "modifications_raw": "",
                "source_proteins_raw": "A0A8I3N123",
                "confidence_or_fdr": "q=0.009",
            },
            {
                "peptide": "RABIESPEP",
                "sample_id": "1",
                "source_record_id": "s3_pred_1",
                "evidence_kind": "motif_prediction",
                "in_motif_subset": "true",
                "modifications_raw": "",
                "source_proteins_raw": "RABV_G",
                "confidence_or_fdr": "",
            },
            {
                "peptide": "LIB_PEP_01",
                "sample_id": "2",
                "source_record_id": "comp_1",
                "evidence_kind": "competition_binding",
                "in_motif_subset": "false",
                "modifications_raw": "",
                "source_proteins_raw": "RABV_N",
                "confidence_or_fdr": "",
            },
        ]
    ).to_csv(table, sep="\t", index=False)

    df, summary = load_lang_2026_splenic_dla_dr_ligands(table, expected_sha256=_sha256(table))
    assert summary["n_input_rows"] == 4
    assert summary["n_ms_observations"] == 2
    assert summary["n_motif_subset_observations"] == 1
    assert summary["n_excluded_non_ms_rows"] == 2
    assert summary["n_donors"] == 2
    assert summary["tissue"] == "spleen"
    assert summary["satisfies_vital_organ_blacklist"] is False

    # Verified Table S2 donor metadata joined while unknowns stay explicit
    assert set(df["breed"]) == {"Standard Poodle", "Golden Retriever"}
    assert (df["dra_genotype"] == "unknown").all()
    assert (df["health_status"] == "unknown").all()
    assert (df["treatment_status"] == "unknown").all()

    # Two resolved spleen donors MUST NOT satisfy #674 heart/brain/lung blacklist!
    guard = evaluate_canine_vital_organ_blacklist(df)
    assert guard["coverage_status"] == "missing"
    assert guard["n_qualifying_rows"] == 0
    assert guard["n_excluded_by_reason"] == {"malignant_or_unresolved_tissue_status": 2}

    # Even if tissue_status were later verified nonmalignant, spleen is still excluded!
    df_nonmalignant = df.copy()
    df_nonmalignant["tissue_status"] = "nonmalignant"
    guard2 = evaluate_canine_vital_organ_blacklist(df_nonmalignant)
    assert guard2["coverage_status"] == "missing"
    assert guard2["n_excluded_by_reason"] == {"non_vital_organ_tissue": 2}


def test_summarize_canine_bulk_protein_evidence_guard(tmp_path, monkeypatch):
    # Minimal verified schema-1 Pfizer fixture exercising summarize_canine_bulk_protein_evidence
    from hitlist import canine_bulk

    fasta = tmp_path / "ref.fasta"
    fasta.write_text(">sp|P00001|ACTB_CANLF Actin\nMDDDIAALVVDNGSGMCK\n")
    table = tmp_path / "pfizer.tsv"
    table.write_text(
        "BioReplicate\tCondition\tProtein\tProteinGroup\tRun\tMixture\tChannel\tTechRepMixture\t"
        "Abundance\tIBAQ\trIBAQ\tsumIBAQ\n"
        "Pool_Heart_1\tHeart\tP00001\tPG1\tRun1\tMix1\t126\t1\t100.5\t50.2\t0.1\t50.2\n"
        "Pool_Heart_1\tHeart\tP00002\tPG2\tRun1\tMix1\t126\t1\tNA\tNaN\tNA\tNA\n"
    )
    catalog = canine_bulk.canine_bulk_catalog()
    src = catalog["sources"]["pfizer_2026_03"]
    src["assets"]["reference"] = {
        "filename": "ref.fasta",
        "url": "https://example.invalid/ref.fasta",
        "size_bytes": fasta.stat().st_size,
        "sha256": _sha256(fasta),
    }
    src["assets"]["table"] = {
        "filename": "pfizer.tsv",
        "url": "https://example.invalid/pfizer.tsv",
        "size_bytes": table.stat().st_size,
        "sha256": _sha256(table),
    }
    monkeypatch.setattr(canine_bulk, "canine_bulk_catalog", lambda: catalog)

    bundle = tmp_path / "bundle"
    canine_bulk.curate_canine_bulk(
        "pfizer_2026_03",
        bundle,
        local_assets={"table": table, "reference": fasta},
    )
    df, summary = summarize_canine_bulk_protein_evidence(bundle, tissue="Heart")
    assert summary["tracked_issue"] == 666
    assert summary["source_id"] == "pfizer_2026_03"
    assert summary["provides_dla_presentation_blacklist"] is False
    assert summary["presentation_coverage"] == "missing"
    assert summary["n_rows"] == 2
    assert summary["n_proteins"] == 2
    assert summary["donor_statuses"] == ["pooled"]
    assert not df["provides_dla_presentation_blacklist"].any()
    assert not df["is_ms_observation"].any()

    # Ensure manifest JSON is untouched and verifiable
    assert json.loads((bundle / "manifest.json").read_text())["presentation_coverage"] == "missing"
