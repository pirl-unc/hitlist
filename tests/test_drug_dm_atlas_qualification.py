"""Regressions for #853 (Abelin MAPTAC DM arms), #855 (Atlas prediction provenance),
#901 (DLBCL supplement recovery), and #902 (Flucloxacillin adducts & conditions)."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import pytest

from hitlist.curation import load_pmid_overrides, restriction_evidence_for_row
from hitlist.export import (
    _statement_maps,
    expand_condition_memberships,
    generate_ms_samples_table,
    generate_observations_table,
    generate_training_table,
)
from hitlist.study_qualifications import load_study_contract
from hitlist.supplement import load_supplementary_manifest, scan_supplementary

_SUPP_DIR = Path(__file__).resolve().parents[1] / "hitlist" / "data" / "supplementary"


# ---------------------------------------------------------------------------
# #853: Abelin 2019 MAPTAC Class-II HLA-DM arm recovery
# ---------------------------------------------------------------------------


def test_abelin_dm_contract_and_manifest_replace_merged_csv():
    contract = load_study_contract("abelin_2019_dm")
    assert contract["pmid"] == 31495665
    assert (
        contract["workbook_sha256"]
        == "78b551cecd395d8650c9696ebdb34f78066f351b83f27d043e0e57880420adbe"
    )
    assert (
        contract["biological_interpretation"]["dm_minus_meaning"]
        == "no_added_hla_dm_cotransfection"
    )
    assert (
        contract["biological_interpretation"]["endogenous_dm_status"]
        == "unknown_not_measured_knockout"
    )

    derived = contract["derived_asset"]
    asset_path = _SUPP_DIR / derived["file"]
    assert asset_path.is_file()
    assert hashlib.sha256(asset_path.read_bytes()).hexdigest() == derived["sha256"]
    assert asset_path.stat().st_size == derived["size_bytes"]

    # Ensure supplementary.yaml references only the condition-resolved file (never double-counts merged CSV).
    abelin_entries = [e for e in load_supplementary_manifest() if e["pmid"] == 31495665]
    abelin_files = {e["file"] for e in abelin_entries}
    assert "abelin_2019_maptac_class2_dm_arms.csv.gz" in abelin_files
    assert "abelin_2019_maptac_class2.csv" not in abelin_files
    assert derived["n_paired_dm_rows"] == 13_317
    assert derived["n_paired_dm_shared_peptides"] == 2_133
    assert derived["n_paired_dm_union_rows"] == 11_184


@pytest.mark.integration
def test_abelin_dm_full_asset_preserves_heterodimers_and_shared_peptides():
    contract = load_study_contract("abelin_2019_dm")
    derived = contract["derived_asset"]
    asset_path = _SUPP_DIR / derived["file"]
    df = pd.read_csv(asset_path, dtype=str).fillna("")
    assert len(df) == 45_636
    assert set(df.columns) == {
        "peptide",
        "mhc_class",
        "mhc_restriction",
        "attributed_sample_label",
        "source_column",
        "source_row",
    }
    # Complete DP and DQ heterodimer pairs must be preserved verbatim (no beta-only truncation).
    assert "HLA-DPA1*01:03/DPB1*06:01" in set(df["mhc_restriction"])
    assert "HLA-DQA1*01:02/DQB1*06:04" in set(df["mhc_restriction"])
    assert "HLA-DPB1*06:01" not in set(df["mhc_restriction"])
    assert "HLA-DQB1*06:04" not in set(df["mhc_restriction"])
    counts = df["attributed_sample_label"].value_counts().to_dict()
    assert counts == contract["arm_row_counts"]


def test_abelin_dm_shared_peptide_survives_ingestion_export_and_unified_contrast_recipe(
    tmp_path, monkeypatch
):
    """Same peptide observed in both dm- and dm+ must retain two distinct rows
    through scan_supplementary -> generate_training_table -> prepare_hla_dm_contrast_data."""
    fixture_csv = tmp_path / "abelin_fixture.csv"
    pd.DataFrame(
        [
            {
                "peptide": "SHAREDDMPEPTIDE",
                "mhc_class": "II",
                "mhc_restriction": "HLA-DPA1*01:03/DPB1*06:01",
                "attributed_sample_label": "MAPTAC-HLA-DPB1*06:01/DPA1*01:03 (dm-)",
                "source_column": "MAPTAC_DPB1*06:01/DPA1*01:03_dm-",
                "source_row": "2",
            },
            {
                "peptide": "SHAREDDMPEPTIDE",
                "mhc_class": "II",
                "mhc_restriction": "HLA-DPA1*01:03/DPB1*06:01",
                "attributed_sample_label": "MAPTAC-HLA-DPB1*06:01/DPA1*01:03 (dm+)",
                "source_column": "MAPTAC_DPB1*06:01/DPA1*01:03_dm+",
                "source_row": "3",
            },
            {
                "peptide": "DMPLUSONLYPEP",
                "mhc_class": "II",
                "mhc_restriction": "HLA-DQA1*01:02/DQB1*06:04",
                "attributed_sample_label": "MAPTAC-HLA-DQB1*06:04/DQA1*01:02 (dm+)",
                "source_column": "MAPTAC_DQB1*06:04/DQA1*01:02_dm+",
                "source_row": "4",
            },
        ]
    ).to_csv(fixture_csv, index=False)

    base_entry = next(
        e
        for e in load_supplementary_manifest()
        if e["file"] == "abelin_2019_maptac_class2_dm_arms.csv.gz"
    )
    fixture_entry = {k: v for k, v in base_entry.items() if k not in {"sha256", "size_bytes"}}
    fixture_entry["file"] = fixture_csv.name

    scanned = scan_supplementary(entries=[fixture_entry], directory=tmp_path, allow_download=False)
    assert len(scanned) == 3
    assert scanned["assay_iri"].nunique() == 3

    obs_path = tmp_path / "observations.parquet"
    scanned.assign(source="supplement").to_parquet(obs_path, index=False)
    monkeypatch.setattr("hitlist.observations.observations_path", lambda: obs_path)

    exported = generate_training_table(include_evidence="ms")
    assert len(exported) == 3
    assert exported["evidence_row_id"].nunique() == 3
    assert set(exported["condition_id"]) == {
        "maptac_hla_dpb1_06_01_dpa1_01_03_dm",
        "maptac_hla_dpb1_06_01_dpa1_01_03_dm_2",
        "maptac_hla_dqb1_06_04_dqa1_01_02_dm_2",
    }
    assert set(exported["condition_baseline_engineering"]) == {"MAPTAC_tagged_HLA_transfection"}


# ---------------------------------------------------------------------------
# #855: HLA Ligand Atlas prediction-derived allele-association provenance
# ---------------------------------------------------------------------------


def test_hla_ligand_atlas_restriction_evidence_marks_single_allele_ms_predicted_only():
    contract = load_study_contract("hla_ligand_atlas_33858848")
    assert contract["pmid"] == 33858848
    assert contract["pmcid"] == "PMC8054196"
    assert (
        contract["ms_detection_vs_restriction_contract"][
            "independent_experimental_binding_evidence"
        ]
        is False
    )

    # Single-allele named Class-II DP/DQ/DR and Class-I MS rows must be marked predicted.
    for allele in ("HLA-DPA1*01:03/DPB1*04:01", "HLA-DRB1*07:01", "HLA-A*11:01"):
        assert (
            restriction_evidence_for_row(
                allele,
                pmid=33858848,
                assay_method="cellular MHC/mass spectrometry",
                response_measured="ligand presentation",
            )
            == "predicted"
        )

    # Multi-allele donor candidate sets and class-only rows must remain unknown (never invented).
    assert (
        restriction_evidence_for_row(
            "HLA-DPA1*01:03/DPB1*02:01;HLA-DRB1*15:01",
            pmid=33858848,
            assay_method="cellular MHC/mass spectrometry",
            response_measured="ligand presentation",
        )
        == "unknown"
    )
    assert (
        restriction_evidence_for_row(
            "HLA class II",
            pmid=33858848,
            assay_method="cellular MHC/mass spectrometry",
            response_measured="ligand presentation",
        )
        == "unknown"
    )

    # A separately measured binding assay must NOT be blanket-relabeled as predicted.
    assert (
        restriction_evidence_for_row(
            "HLA-DPA1*01:03/DPB1*04:01",
            pmid=33858848,
            is_binding_assay=True,
            assay_method="purified MHC/competitive/fluorescence",
            response_measured="half maximal inhibitory concentration (IC50)",
        )
        == "experimental"
    )


# ---------------------------------------------------------------------------
# #902: Flucloxacillin condition membership and adduct views
# ---------------------------------------------------------------------------


def test_flucloxacillin_statement_map_and_modification_aware_views(tmp_path, monkeypatch):
    contract = load_study_contract("flucloxacillin_33633747")
    assert contract["pmid"] == 33633747
    assert len(contract["confirmed_flx_covalent_adducts"]) == 3

    obs_rows = pd.DataFrame(
        [
            # Same base sequence KAAKLKEKY across unmodified, Table 1 confirmed K4 adduct,
            # and IEDB alternative-site Filler Assay K6 row: must NEVER collapse!
            {
                "peptide": "KAAKLKEKY",
                "peptide_modifications": "",
                "has_ptm": False,
                "pmid": 33633747,
                "mhc_class": "I",
                "mhc_restriction": "HLA-B*57:01",
                "cell_name": "721.221-B*57:01",
                "assay_comments": "The epitope was eluted from the following experimental conditions(s): FLX-Treated.",
                "source": "iedb",
                "is_binding_assay": False,
                "assay_method": "cellular MHC/mass spectrometry",
                "response_measured": "ligand presentation",
                "qualitative_measurement": "Positive",
            },
            {
                "peptide": "KAAKLKEKY",
                "peptide_modifications": "MCM:K4",
                "has_ptm": True,
                "pmid": 33633747,
                "mhc_class": "I",
                "mhc_restriction": "HLA-B*57:01",
                "cell_name": "721.221-B*57:01",
                "assay_comments": "",
                "source": "iedb",
                "is_binding_assay": False,
                "assay_method": "cellular MHC/mass spectrometry",
                "response_measured": "ligand presentation",
                "qualitative_measurement": "Positive",
            },
            {
                "peptide": "KAAKLKEKY",
                "peptide_modifications": "MCM:K6",
                "has_ptm": True,
                "pmid": 33633747,
                "mhc_class": "I",
                "mhc_restriction": "HLA-B*57:01",
                "cell_name": "721.221-B*57:01",
                "assay_comments": "Filler Assay - delete.",
                "source": "iedb",
                "is_binding_assay": False,
                "assay_method": "cellular MHC/mass spectrometry",
                "response_measured": "ligand presentation",
                "qualitative_measurement": "Positive",
            },
            # Shared untreated + FLX row
            {
                "peptide": "SHAREDFLXPEP",
                "peptide_modifications": "",
                "has_ptm": False,
                "pmid": 33633747,
                "mhc_class": "I",
                "mhc_restriction": "HLA-B*57:01",
                "cell_name": "721.221-B*57:01",
                "assay_comments": "The epitope was eluted from the following experimental conditions(s): Untreated; FLX-Treated.",
                "source": "iedb",
                "is_binding_assay": False,
                "assay_method": "cellular MHC/mass spectrometry",
                "response_measured": "ligand presentation",
                "qualitative_measurement": "Positive",
            },
            # Untreated-only endogenous PTM row
            {
                "peptide": "KSRLKNPK",
                "peptide_modifications": "CITR:R3",
                "has_ptm": True,
                "pmid": 33633747,
                "mhc_class": "I",
                "mhc_restriction": "HLA-B*57:01",
                "cell_name": "721.221-B*57:01",
                "assay_comments": "The epitope was eluted from the following experimental conditions(s): Untreated.",
                "source": "iedb",
                "is_binding_assay": False,
                "assay_method": "cellular MHC/mass spectrometry",
                "response_measured": "ligand presentation",
                "qualitative_measurement": "Positive",
            },
            # Unknown localization MCM adduct (no site)
            {
                "peptide": "UNKNOWNLOCPEP",
                "peptide_modifications": "MCM",
                "has_ptm": True,
                "pmid": 33633747,
                "mhc_class": "I",
                "mhc_restriction": "HLA-B*57:01",
                "cell_name": "721.221-B*57:01",
                "assay_comments": "The epitope was eluted from the following experimental conditions(s): FLX-Treated.",
                "source": "iedb",
                "is_binding_assay": False,
                "assay_method": "cellular MHC/mass spectrometry",
                "response_measured": "ligand presentation",
                "qualitative_measurement": "Positive",
            },
        ]
    )
    obs_path = tmp_path / "obs_flx.parquet"
    obs_rows.to_parquet(obs_path, index=False)
    monkeypatch.setattr("hitlist.observations.observations_path", lambda: obs_path)

    exported = generate_observations_table(exclude_non_peptide_ligand=False)
    by_form = exported.set_index(["peptide", "peptide_modifications"])
    assert by_form.loc[("KAAKLKEKY", ""), "condition_id"] == "cells_flucloxacillin"
    assert by_form.loc[("SHAREDFLXPEP", ""), "condition_id"] == ""
    assert (
        by_form.loc[("SHAREDFLXPEP", ""), "supported_condition_ids"]
        == "cells_flucloxacillin;cells_untreated"
    )
    assert by_form.loc[("KSRLKNPK", "CITR:R3"), "condition_id"] == "cells_untreated"
    assert by_form.loc[("KAAKLKEKY", "MCM:K6"), "condition_id"] == ""

    edges = expand_condition_memberships(exported)
    shared_edges = edges[edges["peptide"] == "SHAREDFLXPEP"]
    assert set(shared_edges["condition_id"]) == {"cells_flucloxacillin", "cells_untreated"}


def test_dlbcl_supplement_contract_pins_db_untreated():
    contract = load_study_contract("dlbcl_35561310")
    assert contract["pmid"] == 35561310
    assert contract["quantity_availability"]["contains_peptide_intensity_or_area"] is False
    assert contract["workbooks"]["table_3_class_i"]["sheet_row_counts"]["DB_untreated"] == 7
    assert contract["workbooks"]["table_5_class_ii"]["sheet_row_counts"]["DB_untreated"] == 20


@pytest.mark.integration
def test_real_corpus_flucloxacillin_and_atlas_reconciliation():
    """Verify exact real-corpus counts for #902 (33633747) and #855 (33858848)."""
    import pyarrow.dataset as ds

    from hitlist.observations import is_built, observations_path

    if not is_built():
        pytest.skip("Observations corpus not built")

    dset = ds.dataset(observations_path(), format="parquet")

    # 1. Flucloxacillin (#902): exact statement-map counts over all 2,496 rows
    smap = _statement_maps(load_pmid_overrides(), generate_ms_samples_table())[33633747]
    flx = dset.to_table(
        columns=["assay_comments", "peptide_modifications"], filter=ds.field("pmid") == 33633747
    ).to_pandas()
    assert len(flx) == 2_496
    n_flx_only = n_unt_only = n_shared = n_unmapped = 0
    for stmt, cnt in flx["assay_comments"].value_counts(dropna=False).items():
        allowed = smap.allowed("" if pd.isna(stmt) else str(stmt))
        if allowed is None:
            n_unmapped += int(cnt)
        elif set(allowed) == {"cells_flucloxacillin"}:
            n_flx_only += int(cnt)
        elif set(allowed) == {"cells_untreated"}:
            n_unt_only += int(cnt)
        elif set(allowed) == {"cells_untreated", "cells_flucloxacillin"}:
            n_shared += int(cnt)
    assert n_flx_only == 1_142
    assert n_unt_only == 226
    assert n_shared == 1_122
    assert n_unmapped == 6  # 3 "Filler Assay - delete." + 3 blank sHLA rows

    # 2. HLA Ligand Atlas (#855): replay restriction_evidence_for_row on all 901,613 rows
    atlas = dset.to_table(
        columns=["mhc_class", "mhc_restriction", "assay_method", "response_measured"],
        filter=ds.field("pmid") == 33858848,
    ).to_pandas()
    assert len(atlas) == 901_613
    replayed = [
        restriction_evidence_for_row(
            str(r), pmid=33858848, assay_method=str(m), response_measured=str(resp)
        )
        for r, m, resp in zip(
            atlas["mhc_restriction"],
            atlas["assay_method"],
            atlas["response_measured"],
            strict=True,
        )
    ]
    atlas["replayed"] = replayed
    counts = atlas.groupby(["mhc_class", "replayed"], observed=True).size().to_dict()
    assert counts[("I", "predicted")] == 203_992
    assert counts[("I", "unknown")] == 202_686
    assert counts[("II", "predicted")] == 161_050
    assert counts[("II", "unknown")] == 333_885
