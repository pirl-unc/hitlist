"""Tests for Wave 3 large murine immunopeptidome curation (#13)."""

from __future__ import annotations

import pytest

from hitlist.curation import classify_ms_row, load_pmid_overrides, ms_excluded_pmids

WAVE3_MOUSE_PMIDS = (
    25428506,
    29615400,
    26495903,
    31219806,
    37828378,
    30084848,
    35820727,
    28011936,
    33995376,
    27802388,
    32152506,
    35984892,
    25605945,
    31599954,
    33725478,
    36210013,
    34203471,
)


@pytest.mark.parametrize("pmid", WAVE3_MOUSE_PMIDS)
def test_wave3_mouse_pmid_curated_and_not_excluded(pmid: int) -> None:
    overrides = load_pmid_overrides()
    excluded = ms_excluded_pmids()
    assert pmid in overrides
    assert pmid not in excluded
    entry = overrides[pmid]
    assert entry.get("ms_samples"), f"PMID {pmid} missing ms_samples"


def test_schuster_2018_separates_19_healthy_organs_from_cancer_lines() -> None:
    healthy_spleen = classify_ms_row(
        process_type="No immunization",
        disease="",
        culture_condition="Direct Ex Vivo",
        source_tissue="spleen",
        cell_name="splenocyte",
        pmid=30084848,
        mhc_restriction="H2-D*b",
    )
    assert healthy_spleen["src_healthy_tissue"] is True
    assert healthy_spleen["src_cancer"] is False

    el4_lymphoma = classify_ms_row(
        process_type="Occurrence of cancer",
        disease="lymphoma",
        culture_condition="Cell Line / Clone",
        source_tissue="lymphoid",
        cell_name="EL4 cell (T cell)",
        pmid=30084848,
        mhc_restriction="H2-D*b",
    )
    assert el4_lymphoma["src_cancer"] is True
    assert el4_lymphoma["src_healthy_tissue"] is False


def test_borg_2021_min6n8_insulinoma_overrides_bad_ebv_lcl_tag() -> None:
    """IEDB mislabeled MIN6N8 insulinoma as EBV-LCL; curation restores src_cancer=True."""
    res = classify_ms_row(
        process_type="Occurrence of cancer",
        disease="insulinoma",
        culture_condition="Cell Line / Clone (EBV transformed, B-LCL)",
        source_tissue="Pancreas",
        cell_name="Pancreatic beta cell",
        pmid=34203471,
        mhc_restriction="H2-K*b",
    )
    assert res["src_cancer"] is True
    assert res["src_healthy_tissue"] is False


def test_devette_2018_separates_721_221_ebv_lcl_from_hela() -> None:
    hela = classify_ms_row(
        process_type="No immunization",
        disease="",
        culture_condition="Cell Line / Clone",
        source_tissue="Uterine Cervix",
        cell_name="HeLa cells-Epithelial cell",
        pmid=29615400,
        mhc_restriction="H2-D*q",
    )
    assert hela["src_cancer"] is True

    b721 = classify_ms_row(
        process_type="No immunization",
        disease="",
        culture_condition="Cell Line / Clone",
        source_tissue="Lymphoid",
        cell_name="B cell",
        pmid=29615400,
        mhc_restriction="H2-D*q",
        assay_comments="Epitope eluted from soluble H2-Dq from 721 cells.",
    )
    assert b721["src_ebv_lcl"] is True
    assert b721["src_cancer"] is False
    assert b721["src_healthy_tissue"] is False


def test_karunakaran_2017_salmonella_bmdcs_are_activated_apc() -> None:
    res = classify_ms_row(
        process_type="No immunization",
        disease="",
        culture_condition="Direct Ex Vivo",
        source_tissue="Bone Marrow",
        cell_name="Dendritic cell",
        pmid=27802388,
        mhc_restriction="H2-K*b",
    )
    assert res["src_activated_apc"] is True
    assert res["src_healthy_tissue"] is False


def test_fessenden_2022_separates_bmdcs_from_b16f10_melanoma() -> None:
    bmdc = classify_ms_row(
        process_type="No immunization",
        disease="",
        culture_condition="Direct Ex Vivo",
        source_tissue="Bone Marrow",
        cell_name="Dendritic cell",
        pmid=35820727,
        mhc_restriction="H2-b class I",
    )
    assert bmdc["src_activated_apc"] is True
    assert bmdc["src_healthy_tissue"] is False

    b16 = classify_ms_row(
        process_type="Occurrence of cancer",
        disease="experimental melanoma",
        culture_condition="Cell Line / Clone",
        source_tissue="Skin",
        cell_name="B16-F10-Epithelial cell",
        pmid=35820727,
        mhc_restriction="H2-b class I",
    )
    assert b16["src_cancer"] is True
    assert b16["src_healthy_tissue"] is False
