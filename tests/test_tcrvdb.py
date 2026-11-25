"""Tests for #722: access-aware TCRvdb paired-receptor functional evidence adapter."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import hitlist
from hitlist import downloads
from hitlist.tcrvdb import (
    NEAR_DUPLICATE_GROUP_COLUMNS,
    TCRVDB_ADAPTER_COLUMNS,
    audit_tcrvdb_leakage_and_coverage,
    build_tcrvdb_near_duplicate_groups,
    export_tcrvdb_local_bundle,
    load_tcrvdb_functional_evidence,
    tcrvdb_source_contract,
    verify_tcrvdb_local_bundle,
)


def _write_synthetic_tcrvdb_csv(path: Path) -> Path:
    rows = [
        {
            "name": "TCR_SYN_001",
            "epitope": "YLQPRTFLL",
            "mhc.a": "HLA-A*02:01",
            "TRAV_IMGT": "TRAV12-1*01",
            "TRAJ_IMGT": "TRAJ43*01",
            "CDR3a": "CVVNIGDMRF",
            "TRBV_IMGT": "TRBV7-9*01",
            "TRBJ_IMGT": "TRBJ2-1*01",
            "CDR3b": "CASSPDIEQFF",
            "TRAV_IMGT_allele_collapsed": "TRAV12-1",
            "TRBV_IMGT_allele_collapsed": "TRBV7-9",
            "alpha_full_aa": "MKSLRVLLVILWLQLSWVWSQQKEVEQNSGPLSVPEGAIASLNCTYSDRGSQSFFWYRQYSGKSPELIMFIYSNGDKEDGRFTAQLNKASQYVSLLIRDSQPSDSATYLCVVNIGDMRFGAGTRLTVKP",
            "beta_full_aa": "MGTSLLCWMALCLLGADHADTGVSQNPRHKITKRGQNVTFRCDPISEHNRLYWYRQTLGQGPEFLTYFQNEAQLEKSRLLSDRFSAERPKGSFSTLEIQRTEQGDSAMYLCASSPDIEQFFGPGTRLTVL",
            "alpha_sequence_provenance": "experimentally_observed_full_length",
            "beta_sequence_provenance": "experimentally_observed_full_length",
            "reconstruction_method": "",
            "validation_status": "validating",
            "qc_status": "pass",
            "expression_status": "high",
            "n_replicates": "3",
            "baseMean": "1420.5",
            "log2FoldChange": "3.8",
            "padj": "1e-6",
            "vdjdb.score": "3",
            "source_pmid": "33521695",
            "source_study": "StudyA",
        },
        # Near-duplicate (1 aa Hamming in CDR3b) with reconstructed full chain (must NEVER be labeled observed!)
        {
            "name": "TCR_SYN_002",
            "epitope": "YLQPRTFLL",
            "mhc.a": "HLA-A*02:01",
            "TRAV_IMGT": "TRAV12-1*01",
            "TRAJ_IMGT": "TRAJ43*01",
            "CDR3a": "CVVNIGDMRF",
            "TRBV_IMGT": "TRBV7-9*01",
            "TRBJ_IMGT": "TRBJ2-1*01",
            "CDR3b": "CASSPDAEQFF",
            "TRAV_IMGT_allele_collapsed": "TRAV12-1",
            "TRBV_IMGT_allele_collapsed": "TRBV7-9",
            "alpha_full_aa": "MKSLRVLLVILWLQLSWVWSQQKEVEQNSGPLSVPEGAIASLNCTYSDRGSQSFFWYRQYSGKSPELIMFIYSNGDKEDGRFTAQLNKASQYVSLLIRDSQPSDSATYLCVVNIGDMRFGAGTRLTVKP",
            "beta_full_aa": "MGTSLLCWMALCLLGADHADTGVSQNPRHKITKRGQNVTFRCDPISEHNRLYWYRQTLGQGPEFLTYFQNEAQLEKSRLLSDRFSAERPKGSFSTLEIQRTEQGDSAMYLCASSPDAEQFFGPGTRLTVL",
            "alpha_sequence_provenance": "experimentally_observed_full_length",
            "beta_sequence_provenance": "experimentally_observed_full_length",
            "reconstruction_method": "TCRtoolbox_stitchr",
            "validation_status": "non-validating",
            "qc_status": "pass",
            "expression_status": "normal",
            "n_replicates": "3",
            "baseMean": "980.0",
            "log2FoldChange": "-0.2",
            "padj": "0.85",
            "vdjdb.score": "2",
            "source_pmid": "34000001",
            "source_study": "StudyB",
        },
        # CDR3/V/J-only annotation, indeterminate
        {
            "name": "TCR_SYN_003",
            "epitope": "GLCTLVAML",
            "mhc.a": "HLA-A*02:01",
            "TRAV_IMGT": "TRAV5*01",
            "TRAJ_IMGT": "TRAJ31*01",
            "CDR3a": "CAEDNNARLMF",
            "TRBV_IMGT": "TRBV20-1*01",
            "TRBJ_IMGT": "TRBJ1-2*01",
            "CDR3b": "CSARDGTGNGYTF",
            "TRAV_IMGT_allele_collapsed": "TRAV5",
            "TRBV_IMGT_allele_collapsed": "TRBV20-1",
            "alpha_full_aa": "",
            "beta_full_aa": "",
            "alpha_sequence_provenance": "",
            "beta_sequence_provenance": "",
            "reconstruction_method": "",
            "validation_status": "ambiguous",
            "qc_status": "pass",
            "expression_status": "normal",
            "n_replicates": "2",
            "baseMean": "310.0",
            "log2FoldChange": "0.9",
            "padj": "0.08",
            "vdjdb.score": "1",
            "source_pmid": "28423320",
            "source_study": "StudyC",
        },
        # Failed QC row (must stay failed_qc even if raw label says non-validating)
        {
            "name": "TCR_SYN_004",
            "epitope": "GILGFVFTL",
            "mhc.a": "HLA-A*02:01",
            "TRAV_IMGT": "TRAV27*01",
            "TRAJ_IMGT": "TRAJ42*01",
            "CDR3a": "CAGSQGNLIF",
            "TRBV_IMGT": "TRBV19*01",
            "TRBJ_IMGT": "TRBJ2-7*01",
            "CDR3b": "CASSIRSSYEQYF",
            "TRAV_IMGT_allele_collapsed": "TRAV27",
            "TRBV_IMGT_allele_collapsed": "TRBV19",
            "alpha_full_aa": "",
            "beta_full_aa": "",
            "alpha_sequence_provenance": "",
            "beta_sequence_provenance": "",
            "reconstruction_method": "",
            "validation_status": "non-validating",
            "qc_status": "qc_fail",
            "expression_status": "undetected",
            "n_replicates": "2",
            "baseMean": "3.0",
            "log2FoldChange": "",
            "padj": "",
            "vdjdb.score": "3",
            "source_pmid": "28423320",
            "source_study": "StudyC",
        },
        # Unassayed row
        {
            "name": "TCR_SYN_005",
            "epitope": "NLVPMVATV",
            "mhc.a": "HLA-A*02:01",
            "TRAV_IMGT": "TRAV24*01",
            "TRAJ_IMGT": "TRAJ49*01",
            "CDR3a": "CARNTGNQFYF",
            "TRBV_IMGT": "TRBV6-5*01",
            "TRBJ_IMGT": "TRBJ1-2*01",
            "CDR3b": "CASSPQTGASYGYTF",
            "TRAV_IMGT_allele_collapsed": "TRAV24",
            "TRBV_IMGT_allele_collapsed": "TRBV6-5",
            "alpha_full_aa": "",
            "beta_full_aa": "",
            "alpha_sequence_provenance": "",
            "beta_sequence_provenance": "",
            "reconstruction_method": "",
            "validation_status": "",
            "qc_status": "",
            "expression_status": "",
            "n_replicates": "",
            "baseMean": "",
            "log2FoldChange": "",
            "padj": "",
            "vdjdb.score": "0",
            "source_pmid": "29000001",
            "source_study": "StudyD",
        },
    ]
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_tcrvdb_contract_and_manual_dataset_registration():
    contract = tcrvdb_source_contract()
    assert contract["access_terms"]["redistribution_allowed"] is False
    assert contract["access_terms"]["public_package_embedding_prohibited"] is True
    assert contract["access_terms"]["vdjdb_score_is_ground_truth"] is False
    assert contract["policies"]["treat_non_validation_as_universal_nonbinding"] is False
    assert contract["policies"]["allow_pairing_unrelated_chains_by_epitope"] is False

    # Registered in hitlist manual catalog & schema conventions
    catalog = hitlist.available_dataset_catalog(category="tcell_and_atlas")
    assert "tcrvdb" in catalog
    assert catalog["tcrvdb"]["kind"] == "manual download"
    assert "tcr_paired_functional" in hitlist.ASSAY_SCHEMA_CONVENTIONS

    # Without authorized local file, raises clear access-controlled error
    with pytest.raises(FileNotFoundError, match=r"tcrvdb\.nkiai\.nl"):
        load_tcrvdb_functional_evidence()


def test_tcrvdb_adapter_preserves_pairing_provenance_five_states_and_leakage(tmp_path: Path):
    csv_path = _write_synthetic_tcrvdb_csv(tmp_path / "2026-10-01_TCRvdb_synthetic.csv")
    df = load_tcrvdb_functional_evidence(csv_path)
    assert tuple(df.columns) == TCRVDB_ADAPTER_COLUMNS
    assert len(df) == 5
    assert (df["evidence_modality"] == "tcr_paired_functional").all()
    assert (df["pairing_status"] == "paired_alpha_beta_same_receptor").all()

    # Five distinct validation states preserved verbatim
    assert df["validation_status"].tolist() == [
        "validated",
        "tested_not_validating",
        "indeterminate",
        "failed_qc",
        "unassayed",
    ]
    # Non-validation is explicitly assay-scoped, never universal molecular non-binding
    assert (
        df.loc[1, "non_validation_scope"] == "assay_scoped_non_validation_not_universal_nonbinding"
    )
    assert df.loc[0, "non_validation_scope"] == "not_applicable"

    # Reconstruction provenance guard: row 1 has reconstruction_method set, so even though
    # raw input claimed experimentally_observed_full_length, adapter demotes it to reconstructed!
    assert df.loc[0, "alpha_sequence_provenance"] == "experimentally_observed_full_length"
    assert df.loc[1, "alpha_sequence_provenance"] == "reconstructed_from_vj_cdr3"
    assert df.loc[1, "beta_sequence_provenance"] == "reconstructed_from_vj_cdr3"
    assert df.loc[2, "alpha_sequence_provenance"] == "cdr3_vj_only"

    # Near-duplicate grouping clusters TCR_SYN_001 & TCR_SYN_002 (1 aa diff in CDR3b on YLQPRTFLL)
    assert df.loc[0, "near_duplicate_group_id"] == df.loc[1, "near_duplicate_group_id"]
    groups = build_tcrvdb_near_duplicate_groups(df)
    assert tuple(groups.columns) == NEAR_DUPLICATE_GROUP_COLUMNS
    assert len(groups) == 4
    ylq_grp = groups[groups["n_receptors"] == 2].iloc[0]
    assert bool(ylq_grp["has_conflicting_validation"]) is True
    assert ylq_grp["n_distinct_studies"] == 2

    audit = audit_tcrvdb_leakage_and_coverage(df)
    assert audit["n_rows"] == 5
    assert audit["n_unique_receptors"] == 5
    assert audit["n_multi_receptor_near_duplicate_groups"] == 1
    assert audit["n_conflicting_validation_groups"] == 1
    assert audit["arbitrary_epitope_generalization_claimed"] is False
    assert audit["redistribution_allowed"] is False

    # Registered local file resolution via hitlist.downloads.register("tcrvdb", ...)
    downloads.register("tcrvdb", csv_path)
    df_reg = load_tcrvdb_functional_evidence()
    assert len(df_reg) == 5


def test_tcrvdb_rejects_unpaired_chains_and_digest_mismatch(tmp_path: Path):
    bad_csv = tmp_path / "unpaired.csv"
    pd.DataFrame(
        [
            {
                "name": "BETA_ONLY",
                "epitope": "YLQPRTFLL",
                "mhc.a": "HLA-A*02:01",
                "TRAV_IMGT": "",
                "TRAJ_IMGT": "",
                "CDR3a": "",
                "TRBV_IMGT": "TRBV7-9*01",
                "TRBJ_IMGT": "TRBJ2-1*01",
                "CDR3b": "CASSPDIEQFF",
                "validation_status": "validating",
            }
        ]
    ).to_csv(bad_csv, index=False)
    with pytest.raises(ValueError, match="paired alpha\\+beta"):
        load_tcrvdb_functional_evidence(bad_csv)

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_tcrvdb_functional_evidence(bad_csv, expected_sha256="0" * 64)


def test_tcrvdb_local_bundle_export_and_verify_roundtrip(tmp_path: Path):
    csv_path = _write_synthetic_tcrvdb_csv(tmp_path / "2026-10-01_TCRvdb_synthetic.csv")
    bundle_dir = tmp_path / "tcrvdb_local_bundle"
    receipt_path = export_tcrvdb_local_bundle(csv_path, bundle_dir)
    assert receipt_path.exists()
    verified = verify_tcrvdb_local_bundle(bundle_dir)
    assert verified["access_terms"]["redistribution_allowed"] is False
    assert verified["coverage"]["n_rows"] == 5

    # Tampering fails verification
    (bundle_dir / "paired_receptors.parquet").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="digest mismatch"):
        verify_tcrvdb_local_bundle(bundle_dir)

    # Receipt JSON is clean and self-describing
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["source_file"]["filename"] == "2026-10-01_TCRvdb_synthetic.csv"
