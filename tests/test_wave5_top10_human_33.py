"""Tests for Wave 5 top 10 uncurated 2021-2026 large human studies (#33, 1.19M rows)."""

from __future__ import annotations

import pytest

from hitlist.curation import classify_ms_row, load_pmid_overrides, ms_excluded_pmids

WAVE5_HUMAN_PMIDS = (
    42010059,
    36185575,
    35239002,
    42653321,
    39827205,
    38684663,
    34391887,
    42155283,
    40689854,
    34260243,
)


@pytest.mark.parametrize("pmid", WAVE5_HUMAN_PMIDS)
def test_wave5_human_pmids_curated_and_not_excluded(pmid: int) -> None:
    overrides = load_pmid_overrides()
    excluded = ms_excluded_pmids()
    assert pmid in overrides
    assert pmid not in excluded
    assert overrides[pmid].get("ms_samples")


def test_bai_2026_reclassifies_paracancer_lung_as_adjacent_not_healthy() -> None:
    paracancer = classify_ms_row(
        process_type="No immunization",
        disease="",
        culture_condition="Direct Ex Vivo",
        source_tissue="lung",
        cell_name="cell found in tissue",
        pmid=42010059,
        mhc_restriction="HLA-A*02:01",
        assay_comments="The epitope was eluted using an anti-HLA-A/B/C kit, BZNEO-H-I-10, from distal paracancer lung tissue.",
    )
    assert paracancer["src_adjacent_to_tumor"] is True
    assert paracancer["src_healthy_tissue"] is False

    ipf = classify_ms_row(
        process_type="No immunization",
        disease="idiopathic pulmonary fibrosis",
        culture_condition="Direct Ex Vivo",
        source_tissue="lung",
        cell_name="cell found in tissue",
        pmid=42010059,
        mhc_restriction="HLA-A*02:01",
        assay_comments="The epitope was eluted using an anti-HLA-A/B/C kit, BZNEO-H-I-10, from lung tissue of a patient with pulmonary fibrosis.",
    )
    assert ipf["src_healthy_tissue"] is False
    assert ipf["src_adjacent_to_tumor"] is False


def test_kessler_2025_and_leddy_2025_dcs_are_activated_apc_not_healthy() -> None:
    for pmid in (39827205, 40689854):
        res = classify_ms_row(
            process_type="No immunization",
            disease="",
            culture_condition="Direct Ex Vivo",
            source_tissue="blood",
            cell_name="dendritic cell",
            pmid=pmid,
            mhc_restriction="HLA-A*02:01",
        )
        assert res["src_activated_apc"] is True
        assert res["src_healthy_tissue"] is False


def test_li_2026_reclassifies_jesthom_and_schu_as_ebv_lcl_not_cancer() -> None:
    for cell in ("JESTHOM cell (B cell)", "SCHU cell (B cell)"):
        res = classify_ms_row(
            process_type="No immunization",
            disease="",
            culture_condition="Cell Line / Clone",
            source_tissue="blood",
            cell_name=cell,
            pmid=42155283,
            mhc_restriction="HLA-DR",
        )
        assert res["src_ebv_lcl"] is True
        assert res["src_cancer"] is False
        assert res["src_healthy_tissue"] is False


def test_de_beijer_2022_separates_hcc_hbv_and_healthy_hepatocytes() -> None:
    hcc = classify_ms_row(
        process_type="Occurrence of cancer",
        disease="hepatocellular carcinoma",
        culture_condition="Direct Ex Vivo",
        source_tissue="liver",
        cell_name="hepatocyte",
        pmid=36185575,
        mhc_restriction="HLA-A*02:01",
    )
    assert hcc["src_cancer"] is True
    assert hcc["src_healthy_tissue"] is False

    hbv = classify_ms_row(
        process_type="Occurrence of infectious disease",
        disease="hepatitis B",
        culture_condition="Direct Ex Vivo",
        source_tissue="liver",
        cell_name="hepatocyte",
        pmid=36185575,
        mhc_restriction="HLA-A*02:01",
    )
    assert hbv["src_cancer"] is False
    assert hbv["src_healthy_tissue"] is False

    healthy = classify_ms_row(
        process_type="No immunization",
        disease="healthy",
        culture_condition="Direct Ex Vivo",
        source_tissue="liver",
        cell_name="hepatocyte",
        pmid=36185575,
        mhc_restriction="HLA-A*02:01",
    )
    assert healthy["src_healthy_tissue"] is True
    assert healthy["src_cancer"] is False


def test_fugmann_2017_dss_colitis_does_not_leak_into_healthy() -> None:
    colitis = classify_ms_row(
        process_type="No immunization",
        disease="ulcerative colitis",
        culture_condition="Direct Ex Vivo",
        source_tissue="Lymph Node",
        cell_name="Lymph node cells",
        pmid=28011936,
        mhc_restriction="H2-b class II",
    )
    assert colitis["src_healthy_tissue"] is False

    healthy = classify_ms_row(
        process_type="No immunization",
        disease="",
        culture_condition="Direct Ex Vivo",
        source_tissue="Lymph Node",
        cell_name="Lymph node cells",
        pmid=28011936,
        mhc_restriction="H2-b class II",
    )
    assert healthy["src_healthy_tissue"] is True
