"""Tests for Wave 4 remaining non-human + 2024-2026 large mouse curation (#14, #13, #980)."""

from __future__ import annotations

import pytest

from hitlist.curation import classify_ms_row, load_pmid_overrides, ms_excluded_pmids

WAVE4_CELLULAR_PMIDS = (
    40505327,
    35453564,
    39147027,
    42809615,
    41540279,
    41315082,
)

WAVE4_EXCLUDED_PMIDS = (
    18687473,
    20519644,
    32269076,
    40412526,
)


@pytest.mark.parametrize("pmid", WAVE4_CELLULAR_PMIDS)
def test_wave4_cellular_pmids_curated(pmid: int) -> None:
    overrides = load_pmid_overrides()
    excluded = ms_excluded_pmids()
    assert pmid in overrides
    assert pmid not in excluded
    assert overrides[pmid].get("ms_samples")


@pytest.mark.parametrize("pmid", WAVE4_EXCLUDED_PMIDS)
def test_wave4_non_peptide_or_refolding_pmids_excluded(pmid: int) -> None:
    overrides = load_pmid_overrides()
    excluded = ms_excluded_pmids()
    assert pmid in overrides
    assert pmid in excluded
    assert overrides[pmid]["exclude_from_ms"] is True


def test_wave4_infected_bmdcs_and_bmdms_classified_activated_apc() -> None:
    for pmid, cell, restr in (
        (42809615, "dendritic cell", "H2-AA*b/AB*b"),
        (41540279, "macrophage", "H2 class II"),
        (40505327, "macrophage", "BoLA-DR"),
    ):
        res = classify_ms_row(
            process_type="No immunization",
            disease="",
            culture_condition="Direct Ex Vivo",
            source_tissue="bone marrow",
            cell_name=cell,
            pmid=pmid,
            mhc_restriction=restr,
        )
        assert res["src_activated_apc"] is True, f"PMID {pmid} should be activated_apc"
        assert res["src_healthy_tissue"] is False, f"PMID {pmid} must not leak into healthy_tissue"


def test_srivastava_2026_rules_separate_healthy_human_from_t1d_and_nod_dcs() -> None:
    healthy = classify_ms_row(
        process_type="No immunization",
        disease="healthy",
        culture_condition="Direct Ex Vivo",
        source_tissue="blood",
        cell_name="PBMC",
        pmid=41315082,
        mhc_restriction="HLA-DR",
    )
    assert healthy["src_healthy_tissue"] is True

    t1d = classify_ms_row(
        process_type="No immunization",
        disease="type 1 diabetes mellitus",
        culture_condition="Direct Ex Vivo",
        source_tissue="blood",
        cell_name="PBMC",
        pmid=41315082,
        mhc_restriction="HLA-DR",
    )
    assert t1d["src_healthy_tissue"] is False

    nod_dc = classify_ms_row(
        process_type="No immunization",
        disease="",
        culture_condition="Direct Ex Vivo",
        source_tissue="spleen",
        cell_name="dendritic cell",
        pmid=41315082,
        mhc_restriction="H2-AA*d/AB*g7",
    )
    assert nod_dc["src_activated_apc"] is True
    assert nod_dc["src_healthy_tissue"] is False
