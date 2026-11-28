"""Tests for thin zero-DSL ML dataset loaders and overlap-audit helpers (#993)."""

from __future__ import annotations

import pandas as pd
import pytest

import hitlist
from hitlist import split_audit
from hitlist.parquet_io import atomic_write_parquet


def test_load_binding_dataset_clean_and_unit_standardization(tmp_path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    binding = pd.DataFrame(
        {
            "peptide": [
                "GILGFVFTL",
                "NLVPMVATV",
                "SLYNTVATL",
                "KLQCVDLHV",
                "YLEPGPVTA",
                "FMYSDFHFI",
                "GILGFVFTL",  # exact duplicate of row 0
            ],
            "mhc_restriction": [
                "HLA-A*02:01",
                "HLA-A*02:01",
                "HLA-A*02:01",
                "HLA-A*02",  # ambiguous resolution
                "HLA-A*02:01",
                "HLA-A*02:01",
                "HLA-A*02:01",
            ],
            "mhc_species": ["Homo sapiens"] * 7,
            "mhc_class": ["I"] * 7,
            "allele_resolution": [
                "four_digit",
                "four_digit",
                "four_digit",
                "two_digit",
                "four_digit",
                "four_digit",
                "four_digit",
            ],
            "response_measured": [
                "half maximal inhibitory concentration (IC50)",
                "dissociation constant KD",
                "qualitative binding",
                "half maximal inhibitory concentration (IC50)",
                "half maximal inhibitory concentration (IC50)",
                "half maximal inhibitory concentration (IC50)",
                "half maximal inhibitory concentration (IC50)",
            ],
            "quantitative_value": [12.5, 0.5, float("nan"), 100.0, 50.0, 200.0, 12.5],
            "measurement_units": ["nM", "uM", "", "nM", "nM", "nM", "nM"],
            "measurement_inequality": ["=", ">", "", "=", "=", "=", "="],
            "qualitative_measurement": [
                "Positive-High",
                "Positive",
                "Positive-Low",
                "Positive",
                "Positive",
                "Positive",
                "Positive-High",
            ],
            "has_ptm": [False, False, False, False, True, False, False],
            "is_non_peptide_ligand": [False, False, False, False, False, True, False],
            "is_binding_assay": [True] * 7,
            "pmid": [1001, 1002, 1003, 1004, 1005, 1006, 1001],
            "source": ["iedb"] * 7,
        }
    )
    atomic_write_parquet(binding, bundle / "binding.parquet")

    df = hitlist.load_binding_dataset(bundle=bundle, mhc_class="I", species="Homo sapiens")
    assert list(df.columns) == [
        "peptide",
        "allele",
        "mhc_species",
        "mhc_class",
        "measurement_kind",
        "value_nm",
        "inequality",
        "qualitative",
        "pmid",
        "source",
    ]
    # Drops two_digit (KLQCVDLHV), PTM (YLEPGPVTA), non-peptide (FMYSDFHFI), and exact dup.
    assert df["peptide"].tolist() == ["GILGFVFTL", "NLVPMVATV", "SLYNTVATL"]
    assert df.loc[0, "measurement_kind"] == "ic50"
    assert df.loc[0, "value_nm"] == pytest.approx(12.5)
    assert df.loc[0, "inequality"] == "="
    # 0.5 uM converted linearly to 500 nM
    assert df.loc[1, "measurement_kind"] == "kd"
    assert df.loc[1, "value_nm"] == pytest.approx(500.0)
    assert df.loc[1, "inequality"] == ">"
    # Qualitative fallback retained with NaN value_nm
    assert df.loc[2, "measurement_kind"] == "qualitative_binding"
    assert pd.isna(df.loc[2, "value_nm"])
    assert df.loc[2, "qualitative"] == "Positive-Low"

    quant_only = hitlist.load_binding_dataset(
        bundle=bundle, mhc_class="I", include_qualitative=False
    )
    assert quant_only["peptide"].tolist() == ["GILGFVFTL", "NLVPMVATV"]


def test_load_ms_dataset_clean_monoallelic_and_host_flanks(tmp_path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    obs = pd.DataFrame(
        {
            "peptide": [
                "GILGFVFTL",
                "GILGFVFTL",  # duplicate within same cohort
                "NLVPMVATV",  # multiallelic
                "SLYNTVATL",  # PTM
                "KLQCVDLHV",  # contaminant
                "YLEPGPVTA",  # cattle cohort
            ],
            "mhc_class": ["I"] * 6,
            "mhc_species": [
                "Homo sapiens",
                "Homo sapiens",
                "Homo sapiens",
                "Homo sapiens",
                "Homo sapiens",
                "Bos taurus",
            ],
            "source_organism": [
                "Homo sapiens",
                "Homo sapiens",
                "Homo sapiens",
                "Homo sapiens",
                "Homo sapiens",
                "Bos taurus",
            ],
            "mhc_allele_set": [
                "HLA-A*02:01",
                "HLA-A*02:01",
                "HLA-A*02:01;HLA-B*07:02",
                "HLA-A*02:01",
                "HLA-A*02:01",
                "BoLA-1*023:01",
            ],
            "is_monoallelic": [True, True, False, True, True, True],
            "has_ptm": [False, False, False, True, False, False],
            "is_non_peptide_ligand": [False] * 6,
            "is_potential_contaminant": [False, False, False, False, True, False],
            "pmid": [28228285, 28228285, 28228285, 28228285, 28228285, 36423003],
            "provenance_id": [f"prov:{i}" for i in range(6)],
            "source": ["iedb"] * 6,
        }
    )
    pf = pd.DataFrame(
        {
            "peptide": ["GILGFVFTL", "YLEPGPVTA"],
            "proteome_id": ["Homo sapiens", "Homo sapiens"],  # Bos taurus NOT mapped
            "n_flank": ["AAAAA", "CCCCC"],
            "c_flank": ["TTTTT", "GGGGG"],
            "protein_id": ["ENSP1", "ENSP2"],
            "gene_name": ["M1", "M2"],
            "gene_id": ["ENSG1", "ENSG2"],
            "transcript_id": ["ENST1", "ENST2"],
            "is_canonical_transcript": [True, True],
            "position": [10, 20],
            "n_mappings": [1, 1],
        }
    )
    atomic_write_parquet(obs, bundle / "observations.parquet")
    atomic_write_parquet(pf, bundle / "primary_flanks.parquet")

    ms = hitlist.load_ms_dataset(bundle=bundle, mhc_class="I", species="Homo sapiens")
    assert ms["peptide"].tolist() == ["GILGFVFTL", "NLVPMVATV"]
    assert ms["sample_id"].str.startswith("pmid28228285:").all()

    mono = hitlist.load_ms_dataset(
        bundle=bundle,
        mhc_class="I",
        species="Homo sapiens",
        monoallelic_only=True,
        with_flanks=True,
    )
    assert mono["peptide"].tolist() == ["GILGFVFTL"]
    assert mono.iloc[0]["n_flank"] == "AAAAA"
    assert mono.iloc[0]["c_flank"] == "TTTTT"
    assert mono.iloc[0]["cohort_flank_coverage"] == pytest.approx(1.0)

    # All-species with min_flank_coverage=0.5 drops the 0%-flank Cattle cohort automatically.
    flanked_all = hitlist.load_ms_dataset(
        bundle=bundle,
        mhc_class="I",
        monoallelic_only=True,
        with_flanks=True,
        min_flank_coverage=0.5,
    )
    assert flanked_all["mhc_species"].tolist() == ["Homo sapiens"]


def test_audit_overlap_and_filter_peptide_disjoint():
    df = pd.DataFrame(
        {
            "peptide": ["GILGFVFTL", "NLVPMVATV", "SLYNTVATL", "KLQCVDLHV", "YLEPGPVTA"],
            "cohort": ["dog_iedb", "dog_iedb", "dog_new", "dog_new", "dog_new"],
            "pmid": [36423003, 36423003, 42199926, 42199926, 42199926],
        }
    )
    refs = {
        "mhcflurry_binding": {"GILGFVFTL", "SLYNTVATL"},
        "mhcflurry_ms": {"NLVPMVATV"},
    }

    overall = hitlist.audit_overlap(df, refs)
    assert len(overall) == 1
    assert overall.iloc[0]["n_unique_peptides"] == 5
    assert overall.iloc[0]["n_overlapping_peptides"] == 3
    assert overall.iloc[0]["n_disjoint_peptides"] == 2
    assert overall.iloc[0]["overlap_fraction"] == pytest.approx(0.6)
    assert overall.iloc[0]["n_overlap_mhcflurry_binding_peptides"] == 2
    assert overall.iloc[0]["n_overlap_mhcflurry_ms_peptides"] == 1

    per_cohort = hitlist.audit_overlap(df, refs, group_by="cohort")
    # Sorted by overlap_fraction descending: dog_iedb (1.0) before dog_new (1/3).
    assert per_cohort["cohort"].tolist() == ["dog_iedb", "dog_new"]
    assert per_cohort.iloc[0]["overlap_fraction"] == pytest.approx(1.0)
    assert per_cohort.iloc[1]["overlap_fraction"] == pytest.approx(1.0 / 3.0)

    disjoint = split_audit.filter_peptide_disjoint(df, refs)
    assert disjoint["peptide"].tolist() == ["KLQCVDLHV", "YLEPGPVTA"]
