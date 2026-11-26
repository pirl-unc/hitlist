"""Tests for Wave 2 non-model vertebrate & non-human curation (#14, #980)."""

from __future__ import annotations

import mhcgnomes
import pytest

from hitlist.curation import classify_ms_row, load_pmid_overrides, ms_excluded_pmids

CELLULAR_NONHUMAN_PMIDS = {
    33789985: ("Bos taurus", "noncancer_cell_line"),
    30239163: ("Canis lupus familiaris", "cell_line"),
    33430316: ("Sus scrofa", "healthy"),
    28315936: ("Equus caballus", "cell_line"),
    26399241: ("Equus caballus", "cell_line"),
    25860507: ("Gallus gallus", "healthy"),
    16879355: ("Gallus gallus", "noncancer_cell_line"),
    15650875: ("Gallus gallus", "noncancer_cell_line"),
    12648094: ("Bos taurus", "noncancer_cell_line"),
}

EXCLUDED_NONHUMAN_PMIDS = {
    32894211: "Sus scrofa",
    35623661: "Anas platyrhynchos",
    30341185: "Gallus gallus",
    18083574: "Gallus gallus",
    31776204: "Xenopus laevis",
    33078460: "Ailuropoda melanoleuca",
}


def test_fisch_2021_bola_dr_rules_separate_pbmc_ova_from_theileria_lines() -> None:
    """PMID 33789985 rules match MHC Restriction to classify PBMC_OVA as healthy
    and Theileria-transformed bovine cell lines as cell_line."""
    overrides = load_pmid_overrides()
    entry = overrides[33789985]
    assert entry["species"] == "Bos taurus"
    assert entry["ip_antibody"] == "IL-A21"
    assert len(entry["ms_samples"]) == 11

    pbmc_ova = classify_ms_row(
        process_type="No immunization",
        disease="",
        culture_condition="Direct Ex Vivo",
        source_tissue="blood",
        cell_name="PBMC",
        pmid=33789985,
        mhc_restriction="Bota-DRB3*001:01;Bota-DRB3*010:01",
    )
    assert pbmc_ova["src_healthy_tissue"] is True
    assert pbmc_ova["src_cell_line"] is False
    assert pbmc_ova["src_cancer"] is False

    theileria_line = classify_ms_row(
        process_type="No immunization",
        disease="",
        culture_condition="Direct Ex Vivo",
        source_tissue="blood",
        cell_name="PBMC",
        pmid=33789985,
        mhc_restriction="Bota-DRB3*020:02",
    )
    assert theileria_line["src_cell_line"] is True
    assert theileria_line["src_cancer"] is False
    assert theileria_line["src_ex_vivo"] is False
    assert theileria_line["src_healthy_tissue"] is False


def test_ebv_lcl_override_clears_direct_ex_vivo_healthy_flag() -> None:
    """An ebv_lcl override must not leak into src_healthy_tissue when IEDB tags Direct Ex Vivo."""
    overrides = load_pmid_overrides()
    ebv_pmid = next(
        pmid
        for pmid, entry in overrides.items()
        if isinstance(pmid, int) and entry.get("override") == "ebv_lcl" and not entry.get("rules")
    )
    res = classify_ms_row(
        process_type="No immunization",
        disease="",
        culture_condition="Direct Ex Vivo",
        source_tissue="blood",
        cell_name="B cell",
        pmid=ebv_pmid,
    )
    assert res["src_ebv_lcl"] is True
    assert res["src_cell_line"] is True
    assert res["src_ex_vivo"] is False
    assert res["src_healthy_tissue"] is False


@pytest.mark.parametrize(("pmid", "expected"), sorted(CELLULAR_NONHUMAN_PMIDS.items()))
def test_curated_nonhuman_cellular_ms_pmids(pmid: int, expected: tuple[str, str]) -> None:
    species, expected_override = expected
    overrides = load_pmid_overrides()
    excluded = ms_excluded_pmids()

    assert pmid in overrides
    assert pmid not in excluded
    entry = overrides[pmid]
    assert entry["species"] == species
    assert entry["override"] == expected_override
    samples = entry.get("ms_samples") or []
    assert len(samples) >= 1
    for sample in samples:
        mhc_str = sample.get("mhc", "")
        assert mhc_str, f"PMID {pmid} sample {sample.get('sample_label')} missing mhc"
        # Single haplotype/class string like "Gaga-B19 class II" parses whole;
        # space-separated allele lists parse per token.
        parsed_whole = mhcgnomes.parse(mhc_str, raise_on_error=False)
        if parsed_whole is None:
            for token in mhc_str.split():
                assert mhcgnomes.parse(token, raise_on_error=False) is not None, (
                    f"PMID {pmid} unparseable MHC token: {token}"
                )


@pytest.mark.parametrize(("pmid", "expected_species"), sorted(EXCLUDED_NONHUMAN_PMIDS.items()))
def test_nonhuman_non_ms_studies_excluded(pmid: int, expected_species: str) -> None:
    overrides = load_pmid_overrides()
    excluded = ms_excluded_pmids()

    assert pmid in overrides
    assert pmid in excluded
    assert overrides[pmid]["species"] == expected_species
    assert overrides[pmid]["exclude_from_ms"] is True
