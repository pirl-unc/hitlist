"""Tests for Wave 6 top 15 remaining uncurated human studies (#33, 181K rows)."""

from __future__ import annotations

import pytest

from hitlist.curation import classify_ms_row, load_pmid_overrides, ms_excluded_pmids

WAVE6_HUMAN_PMIDS = (
    40906332,
    41440022,
    41191777,
    42228098,
    36656713,
    39617188,
    42069060,
    40684438,
    40466865,
    40051755,
    41534530,
    37759700,
    42273698,
    42735508,
    40498839,
)


@pytest.mark.parametrize("pmid", WAVE6_HUMAN_PMIDS)
def test_wave6_human_pmids_curated_and_not_excluded(pmid: int) -> None:
    overrides = load_pmid_overrides()
    excluded = ms_excluded_pmids()
    assert pmid in overrides
    assert pmid not in excluded
    assert overrides[pmid].get("ms_samples")


def test_engineered_monoallelic_ebv_lcl_hosts_recover_flags() -> None:
    meyer = classify_ms_row(
        process_type="No immunization",
        disease="",
        culture_condition="Engineered",
        source_tissue="blood",
        cell_name="LCL721.221 cell (B cell)",
        pmid=36656713,
        mhc_restriction="HLA-A*02:01",
    )
    assert meyer["src_ebv_lcl"] is True
    assert meyer["src_cell_line"] is True
    assert meyer["src_cancer"] is False
    assert meyer["is_monoallelic"] is True
    assert meyer["monoallelic_host"] == "721.221"

    cobos = classify_ms_row(
        process_type="No immunization",
        disease="",
        culture_condition="Engineered",
        source_tissue="blood",
        cell_name="HMy2.C1R cell (B cell)",
        pmid=40466865,
        mhc_restriction="HLA-B*14:03",
    )
    assert cobos["src_ebv_lcl"] is True
    assert cobos["src_cell_line"] is True
    assert cobos["src_cancer"] is False
    assert cobos["is_monoallelic"] is True
    assert cobos["monoallelic_host"] == "C1R"


def test_leddy_2025_and_miranda_2025_phagocytes_are_activated_apc() -> None:
    for pmid, cell in ((41191777, "dendritic cell"), (39617188, "macrophage")):
        res = classify_ms_row(
            process_type="No immunization",
            disease="",
            culture_condition="Direct Ex Vivo",
            source_tissue="blood",
            cell_name=cell,
            pmid=pmid,
            mhc_restriction="HLA-DRB1*07:01",
        )
        assert res["src_activated_apc"] is True
        assert res["src_healthy_tissue"] is False


def test_dorvash_2026_separates_panc1_cancer_from_c1r_ebv_lcl() -> None:
    panc1 = classify_ms_row(
        process_type="No immunization",
        disease="",
        culture_condition="Cell Line / Clone",
        source_tissue="pancreas",
        cell_name="PANC-1 cell (epithelial cell)",
        pmid=42228098,
        mhc_restriction="HLA-A*02:01;HLA-A*11:01;HLA-B*38:01;HLA-C*12:03",
    )
    assert panc1["src_cancer"] is True
    assert panc1["src_ebv_lcl"] is False

    c1r = classify_ms_row(
        process_type="No immunization",
        disease="",
        culture_condition="Cell Line / Clone",
        source_tissue="blood",
        cell_name="HMy2.C1R cell (B cell)",
        pmid=42228098,
        mhc_restriction="HLA-A*02:01;HLA-B*35:03;HLA-C*04:01",
    )
    assert c1r["src_ebv_lcl"] is True
    assert c1r["src_cancer"] is False


def test_croce_2025_jurkat_j76_is_cellular_cancer_line_not_excluded() -> None:
    res = classify_ms_row(
        process_type="Occurrence of cancer",
        disease="childhood T-cell acute lymphoblastic leukemia",
        culture_condition="Cell Line / Clone",
        source_tissue="blood",
        cell_name="Jurkat cell (T cell)",
        pmid=40498839,
        mhc_restriction="HLA-A*02:01;HLA-A*03:01;HLA-B*07:02;HLA-B*35:01;HLA-C*04:01;HLA-C*07:02",
    )
    assert res["src_cancer"] is True
    assert res["src_cell_line"] is True
    assert res["src_healthy_tissue"] is False


def test_wang_2026_dr15_separates_ebv_lcl_thymus_and_ms_cns() -> None:
    thymus = classify_ms_row(
        process_type="No immunization",
        disease="",
        culture_condition="Direct Ex Vivo",
        source_tissue="thymus",
        cell_name="cell found in tissue",
        pmid=41534530,
        mhc_restriction="HLA-DRB5*01:01",
    )
    assert thymus["src_healthy_thymus"] is True
    assert thymus["src_healthy_tissue"] is False

    ms_cns = classify_ms_row(
        process_type="Occurrence of autoimmune disease",
        disease="multiple sclerosis",
        culture_condition="Direct Ex Vivo",
        source_tissue="central nervous system (CNS)",
        cell_name="cell found in tissue",
        pmid=41534530,
        mhc_restriction="HLA-DRB1*15:01",
    )
    assert ms_cns["src_healthy_tissue"] is False
    assert ms_cns["src_ex_vivo"] is True
