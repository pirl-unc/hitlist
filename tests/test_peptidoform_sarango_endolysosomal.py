"""Tests for peptidoform provenance (#896), Sarango 2022 PXD024417 (#858), and Oztemiz 2026 (#812)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from hitlist.curation import load_pmid_overrides
from hitlist.peptidoform_provenance import (
    get_peptidoform_preparation_contract,
    guard_affinity_aggregation,
    join_psms_to_supplement_rows,
    peptidoform_preparation_contracts,
    summarize_peptidoform_eligibility,
)
from hitlist.sarango_2022_pxd024417 import (
    parse_sarango_2022_sdrf,
    qualify_sarango_2022_measurements,
    sarango_2022_source_contract,
    summarize_sarango_2022_sdrf,
)


def test_peptidoform_896_contracts_and_no_blanket_has_ptm_false() -> None:
    contracts = peptidoform_preparation_contracts()
    assert contracts["policy_rules"]["blanket_fill_has_ptm_false_allowed"] is False
    assert contracts["policy_rules"]["all_uppercase_implies_unmodified"] is False
    assert (
        contracts["policy_rules"]["silent_aggregation_of_modified_into_unmodified_affinity_allowed"]
        is False
    )

    strazar = get_peptidoform_preparation_contract(37301199)
    assert strazar["massive_accession"] == "MSV000089225"
    assert strazar["preparation_chemistry"]["alkylation_reagent"] == "iodoacetamide"
    assert (
        strazar["preparation_chemistry"]["default_disposition_without_psm_join"]
        == "unresolved_form"
    )

    abelin = get_peptidoform_preparation_contract(31495665)
    assert abelin["massive_accession"] == "MSV000083991"

    # Without PSM join, bare sequences MUST remain unresolved_form (has_ptm is None/NaN, never False!)
    sup_df = pd.DataFrame(
        [
            {
                "peptide": "AAKVVLLPEK",
                "mhc_restriction": "HLA-DQA1*01:02/DQB1*06:02",
                "dataset": "internal_aff_clean1and2",
            },
            {
                "peptide": "CCKVVLLPEK",
                "mhc_restriction": "HLA-DQA1*01:02/DQB1*06:02",
                "dataset": "iedb_reused",
            },
        ]
    )
    joined_no_psm = join_psms_to_supplement_rows(sup_df, None, pmid=37301199)
    assert (joined_no_psm["peptidoform_disposition"] == "unresolved_form").all()
    assert joined_no_psm["has_ptm"].isna().all()
    summary = summarize_peptidoform_eligibility(joined_no_psm)
    assert summary["n_independent_current_study_rows"] == 1
    assert summary["n_excluded_reused_or_decoy_rows"] == 1
    assert summary["n_unresolved_form_rows"] == 2
    assert summary["n_eligible_unmodified_independent_rows"] == 0


def test_peptidoform_896_psm_join_dispositions_and_affinity_guard() -> None:
    sup_df = pd.DataFrame(
        [
            {
                "peptide": "PEPTIDEONE",
                "mhc_restriction": "HLA-DRB1*04:01",
                "dataset": "internal_aff_clean1and2",
            },
            {
                "peptide": "PEPTIDECYS",
                "mhc_restriction": "HLA-DRB1*04:01",
                "dataset": "internal_aff_clean1and2",
            },
            {
                "peptide": "PEPTIDEMET",
                "mhc_restriction": "HLA-DRB1*04:01",
                "dataset": "internal_aff_clean1and2",
            },
            {
                "peptide": "PEPTIDEMIX",
                "mhc_restriction": "HLA-DRB1*04:01",
                "dataset": "internal_aff_clean1and2",
            },
            {
                "peptide": "PEPTIDENOPSM",
                "mhc_restriction": "HLA-DRB1*04:01",
                "dataset": "internal_aff_clean1and2",
            },
        ]
    )
    psm_df = pd.DataFrame(
        [
            {
                "peptide": "PEPTIDEONE",
                "mhc_restriction": "HLA-DRB1*04:01",
                "modified_sequence": "PEPTIDEONE",
                "ptm_annotation": "",
                "source_file": "run1.raw",
                "run_id": "R1",
                "scan_number": 100,
            },
            {
                "peptide": "PEPTIDECYS",
                "mhc_restriction": "HLA-DRB1*04:01",
                "modified_sequence": "PEPTIDEC[+57.0215]YS",
                "ptm_annotation": "Carbamidomethyl",
                "source_file": "run1.raw",
                "run_id": "R1",
                "scan_number": 101,
            },
            {
                "peptide": "PEPTIDEMET",
                "mhc_restriction": "HLA-DRB1*04:01",
                "modified_sequence": "PEPTIDEM[Oxidation]ET",
                "ptm_annotation": "Oxidation",
                "source_file": "run1.raw",
                "run_id": "R1",
                "scan_number": 102,
            },
            {
                "peptide": "PEPTIDEMIX",
                "mhc_restriction": "HLA-DRB1*04:01",
                "modified_sequence": "PEPTIDEMIX",
                "ptm_annotation": "",
                "source_file": "run1.raw",
                "run_id": "R1",
                "scan_number": 103,
            },
            {
                "peptide": "PEPTIDEMIX",
                "mhc_restriction": "HLA-DRB1*04:01",
                "modified_sequence": "PEPTIDEM[Oxidation]IX",
                "ptm_annotation": "Oxidation",
                "source_file": "run2.raw",
                "run_id": "R2",
                "scan_number": 104,
            },
        ]
    )
    joined = join_psms_to_supplement_rows(sup_df, psm_df, pmid=37301199)
    by_pep = joined.set_index("peptide")

    assert by_pep.loc["PEPTIDEONE", "peptidoform_disposition"] == "unmodified"
    assert bool(by_pep.loc["PEPTIDEONE", "eligible_unmodified_ms_observation"]) is True

    # Prep-only carbamidomethylation keeps biological PTM distinct from prep derivatization
    assert by_pep.loc["PEPTIDECYS", "peptidoform_disposition"] == "preparation_derivatized_only"
    assert bool(by_pep.loc["PEPTIDECYS", "has_ptm"]) is True
    assert bool(by_pep.loc["PEPTIDECYS", "has_biological_ptm"]) is False

    assert by_pep.loc["PEPTIDEMET", "peptidoform_disposition"] == "modified"
    assert bool(by_pep.loc["PEPTIDEMET", "has_biological_ptm"]) is True

    # Multi-form peptides stay explicitly ambiguous with both forms preserved
    assert by_pep.loc["PEPTIDEMIX", "peptidoform_disposition"] == "multi_form_ambiguous"
    assert by_pep.loc["PEPTIDEMIX", "n_distinct_forms"] == 2
    assert "PEPTIDEM[Oxidation]IX" in by_pep.loc["PEPTIDEMIX", "observed_modified_sequences"]

    assert by_pep.loc["PEPTIDENOPSM", "peptidoform_disposition"] == "unresolved_form"

    # Guard refuses silent aggregation of modified/ambiguous/unresolved rows into unmodified affinity
    with pytest.raises(ValueError, match="Cannot silently aggregate"):
        guard_affinity_aggregation(joined)
    ok_only = guard_affinity_aggregation(joined[joined["peptidoform_disposition"] == "unmodified"])
    assert len(ok_only) == 1


def test_sarango_858_contract_sdrf_boundaries_and_note_rewording(tmp_path: Path) -> None:
    contract = sarango_2022_source_contract()
    assert contract["pmid"] == 36215666
    assert contract["pride_inventory"]["n_total_files"] == 81
    assert contract["pride_inventory"]["n_raw_files"] == 40
    assert contract["pride_inventory"]["n_msf_files"] == 40
    assert (
        contract["secondary_community_sdrf_typing_policy"]["promote_to_primary_genotype"] is False
    )
    assert (
        contract["secondary_community_sdrf_typing_policy"]["restore_withdrawn_class_i_alleles"]
        is False
    )
    assert contract["secondary_community_sdrf_typing_policy"]["manufacture_dra_allele"] is False

    # Acceptance criterion #5: pmid_overrides.yaml rewords blanket reduction to repertoire alteration
    sarango_override = load_pmid_overrides()[36215666]
    note = sarango_override["note"]
    assert "reduces the MHC-II immunopeptidome" not in note
    assert "alteration of the MHC-II immunopeptidome repertoire" in note

    # Build minimal synthetic SDRF mirroring exact PXD024417 columns and verify strict boundaries
    sdrf_path = tmp_path / "PXD024417_community_annotated.sdrf.tsv"
    pd.DataFrame(
        [
            {
                "source name": "HeLa-CIITA_Mock1_classI",
                "characteristics[cell line]": "HeLa-CIITA",
                "characteristics[cellosaurus accession]": "CVCL_0030",
                "characteristics[biological replicate]": "1",
                "characteristics[mhc protein complex]": "MHC class I protein complex",
                "characteristics[antibody enrichment]": "W6/32",
                "characteristics[mhc typing]": "HLA-A*68:02;HLA-B*15:03;HLA-C*12:03",
                "characteristics[mhc typing method]": "not available",
                "assay name": "run_Mock1_classI_rep1",
                "comment[technical replicate]": "1",
                "comment[data file]": "Hela_Mock1_W632_replicate1.raw",
                "comment[sample preparation batch]": "batch_190220",
                "comment[acquisition date]": "2019-02-23T04:30:59",
                "comment[modification parameters]": "NT=Oxidation;MT=variable;TA=M;AC=UNIMOD:35",
                "comment[alkylation reagent]": "not applicable",
            },
            {
                "source name": "HeLa-CIITA_siT6BP_classII",
                "characteristics[cell line]": "HeLa-CIITA",
                "characteristics[cellosaurus accession]": "CVCL_0030",
                "characteristics[biological replicate]": "1",
                "characteristics[mhc protein complex]": "MHC class II protein complex",
                "characteristics[antibody enrichment]": "Tue39",
                "characteristics[mhc typing]": "HLA-DRB1*01:02;HLA-DQA1*01:02;HLA-DQB1*05:01",
                "characteristics[mhc typing method]": "not available",
                "assay name": "run_siT6BP_classII_rep1",
                "comment[technical replicate]": "1",
                "comment[data file]": "Hela_siT6BP_Tue39_replicate1.raw",
                "comment[sample preparation batch]": "batch_190220",
                "comment[acquisition date]": "2019-02-24T04:30:59",
                "comment[modification parameters]": "NT=Oxidation;MT=variable;TA=M;AC=UNIMOD:35",
                "comment[alkylation reagent]": "not applicable",
            },
        ]
    ).to_csv(sdrf_path, sep="\t", index=False)

    # Corrupted/synthetic checksum rejected by default
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        parse_sarango_2022_sdrf(sdrf_path, allow_unverified_checksum=False)

    sdrf = parse_sarango_2022_sdrf(sdrf_path, allow_unverified_checksum=True)
    summary = summarize_sarango_2022_sdrf(sdrf)
    assert summary["any_secondary_typing_promoted"] is False
    assert summary["any_class_i_primary_genotype_restored"] is False
    c2_row = sdrf[sdrf["mhc_class"] == "II"].iloc[0]
    assert c2_row["primary_verified_mhc_genotype"] == "HLA-DRB1*01:02"
    assert "DRA" not in str(c2_row["primary_verified_mhc_genotype"])
    assert (
        c2_row["biological_replicate_qualification"]
        == "unresolved_second_biological_sirna_experiment_in_pride"
    )

    meas = pd.DataFrame(
        [
            {
                "raw_filename": "Hela_siT6BP_Tue39_replicate1.raw",
                "peptide": "KPVSKMRMATPLLMQAL",
                "intensity": 12345.0,
            }
        ]
    )
    qual = qualify_sarango_2022_measurements(
        meas, sdrf, evidence_family="pan_mhc_peptide_ms_intensity"
    )
    assert qual.iloc[0]["condition_knockdown_genes"] == "TAX1BP1"
    assert bool(qual.iloc[0]["is_plateau_core_summary"]) is False
    assert bool(qual.iloc[0]["missing_detection_is_measured_negative"]) is False

    # Catalog registration check
    from hitlist.downloads import available_datasets, preflight

    assert "sarango-2022-pxd024417-sdrf" in set(available_datasets())
    assert preflight("sarango-2022-pxd024417-sdrf")["name"] == "sarango-2022-pxd024417-sdrf"


def test_oztemiz_812_contract_via_study_qualifications() -> None:
    from hitlist.study_qualifications import load_study_contract

    contract = load_study_contract("oztemiz_2026_endolysosomal")
    assert contract["pmid"] == 42846314
    assert contract["pmcid"] == "PMC13643861"
    assert contract["training_data_admitted"] is False
    assert contract["biological_system_and_assay_scope"]["is_intact_cell_concentration"] is False
