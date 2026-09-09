"""Source-verified facts for studies whose curation the #436 audit disproved.

Each assertion below corresponds to a statement in the primary source or in
the authors' deposit, cited in the test that makes it.  They exist because
the previous curation was self-consistent and wrong: it named a cell line
the paper never mentions, collapsed a knockout axis, called an
IFN-gamma-treated engineered derivative an untreated parental line, and
inverted a citation.  Nothing but a test rereads a paper.
"""

from __future__ import annotations

import pytest

from hitlist.curation import load_pmid_overrides
from hitlist.export import generate_ms_samples_table


def _samples(pmid: int) -> list[dict]:
    return load_pmid_overrides()[pmid]["ms_samples"]


def _labels(pmid: int) -> list[str]:
    return [s["sample_label"] for s in _samples(pmid)]


def _entry_text(pmid: int) -> str:
    return str(load_pmid_overrides()[pmid])


# ── PMID 34497125 — Stopfer 2021, PNAS (10.1073/pnas.2111173118) ────────────


def test_stopfer_2021_is_skmel5_plus_binimetinib():
    """The paper names SKMEL5 and binimetinib and neither A375 nor trametinib.

    *"a multiplexed, discovery immunopeptidomics analysis of BRAF mutant
    melanoma SKMEL5 cells treated with binimetinib for 72 h"*, against a
    DMSO control.  PXD024917 deposits `SKMEL5_Binimetinib_100nM_HFX_PSMs.txt`
    and `SKMEL5_Transitions_DMSO_MEKi.xlsx`.
    """
    labels = _labels(34497125)
    assert any("SKMEL5" in label and "DMSO" in label for label in labels)
    assert any("SKMEL5" in label and "binimetinib" in label for label in labels)
    text = _entry_text(34497125)
    # The corrective note has to name what was wrong, so only the sample and
    # condition fields are checked for the retracted terms.
    for sample in _samples(34497125):
        assert "A375" not in sample["sample_label"]
        assert "trametinib" not in sample["sample_label"]
        assert "trametinib" not in sample["condition"]
    assert "hla_alleles" not in load_pmid_overrides()[34497125]
    assert "HLA-B*44:03" not in text, "A375's genotype must not survive anywhere"


def test_stopfer_2021_claims_no_skmel5_genotype():
    """The paper never states SKMEL5's typing, so neither may the curation.

    It says only that the Iso18 targets were predicted HLA-A*02:01 binders
    and that all 10 biopsies were HLA-A2*01+ — a selection criterion, not a
    genotype.  ``cell_lines.yaml`` carries SK-MEL-5 (CVCL_0527) with no HLA
    typing either, so an imprecise ``mhc`` is the truthful answer.
    """
    assert {s["mhc"] for s in _samples(34497125)} == {"HLA class I"}


# ── PMID 34129938 — Pollock 2021, MCP (10.1016/j.mcpro.2021.100108) ────────


def test_pollock_2021_has_four_idadpgkg_arms_on_an_ifn_gamma_background():
    """Methods give four treatments, all on 20 ng/ml mouse IFN-gamma.

    IFN-gamma alone 51 h ("control"); IFN-gamma 48 h then 1 uM dTAG-13 3 h
    ("dTAG"); IFN-gamma + 1 ug/ml doxycycline 51 h ("dox"); IFN-gamma + dox
    48 h then dTAG-13 3 h ("both").  The line is the piggyBAC-transfected
    idAdpgkG derivative, not parental MC38.
    """
    arms = [s for s in _samples(34129938) if "idAdpgkG" in s["sample_label"]]
    assert len(arms) == 4
    assert all("IFN-gamma" in s["condition"] for s in arms)
    assert sum("dTAG-13" in s["condition"] for s in arms) == 2
    assert sum("doxycycline" in s["condition"] for s in arms) == 2


def test_pollock_2021_ifn_gamma_arms_are_flagged_as_perturbed():
    """An IFN-gamma background is an antigen-processing perturbation.

    The single "unperturbed MC38" entry this replaces asserted the opposite
    for every quantified arm in the paper.
    """
    samples = generate_ms_samples_table()
    arms = samples[(samples["pmid"] == 34129938) & samples["sample_label"].str.contains("idAdpgkG")]
    assert len(arms) == 4
    assert (arms["apm_perturbed"] == "true").all()
    assert arms["apm_genes_perturbed"].str.contains("ifn_gamma").all()
    assert (arms["is_control_arm"] == "false").all()


# ── PMID 39438697 — Leddy 2025, Nat Protoc (10.1038/s41596-024-01076-x) ────


def test_leddy_2025_thp1_arms_match_the_authors_conditions_table():
    """`conditions_table_TAP.csv` names four THP-1 conditions.

    `080623_THP_SQ_WT_mock`, `..._TAP1_KO_mock`, `..._WT_H37Rv`,
    `..._TAP1_KO_H37Rv` — a 2x2 of TAP1 genotype by infection that the
    previous two arms collapsed.
    """
    thp1 = [s for s in _samples(39438697) if s["sample_label"].startswith("THP-1")]
    assert len(thp1) == 4
    assert sum("TAP1 knockout" in s["sample_label"] for s in thp1) == 2
    assert sum("H37Rv" in s["sample_label"] for s in thp1) == 2


def test_leddy_2025_tap1_knockout_arms_are_flagged():
    """A TAP1 knockout must reach the APM gene flags, not just the label."""
    samples = generate_ms_samples_table()
    ko = samples[
        (samples["pmid"] == 39438697) & samples["sample_label"].str.contains("TAP1 knockout")
    ]
    assert len(ko) == 2
    assert ko["apm_genes_perturbed"].str.contains("tap1").all()


def test_leddy_2025_includes_hmdms_and_pulsed_splenocytes():
    """Two sample types the previous curation missed or mislabeled.

    Figures 4-5 use primary human monocyte-derived macrophages, Mtb- and
    mock-infected.  Figure 6's splenocytes are *pulsed with Alg8 peptide*;
    they were curated as unperturbed.
    """
    labels = _labels(39438697)
    assert sum("monocyte-derived macrophages" in label for label in labels) == 2
    splenocytes = [s for s in _samples(39438697) if "splenocytes" in s["sample_label"]]
    assert len(splenocytes) == 1
    assert "Alg8" in splenocytes[0]["condition"]
    assert splenocytes[0]["condition"] != "unperturbed"


# ── PMID 27846572 — Liepe 2016, Science (10.1126/science.aaf4384) ──────────


def test_liepe_2016_samples_are_the_four_the_paper_uses():
    """GR-LCL, C1R, T2, primary fibroblasts — and nothing else.

    The paper mentions neither JY nor HeLa, both of which were curated as
    samples, and the corpus carries no row for either.  T2, the TAP-deficient
    control, was missing entirely.
    """
    labels = _labels(27846572)
    assert len(labels) == 4
    assert not any("JY" in label for label in labels)
    assert not any("HeLa" in label for label in labels)
    assert any(label.startswith("T2") for label in labels)


@pytest.mark.parametrize(
    ("fragment", "allele"),
    [
        # All 11,733 GR-LCL rows carry this four-digit restriction; the
        # curation had it at two-digit resolution.
        ("GR-LCL", "HLA-B*27:05"),
        # All 3,206 C1R rows carry B*40:02, which the curated genotype omitted.
        ("C1R", "HLA-B*40:02"),
        # All 111 T2 rows.
        ("T2", "HLA-A*02:01"),
    ],
)
def test_liepe_2016_genotypes_match_the_evidence(fragment, allele):
    """Curated genotypes are taken from the rows, not from a registry."""
    matched = [s for s in _samples(27846572) if fragment in s["sample_label"]]
    assert len(matched) == 1
    assert allele in matched[0]["mhc"]


def test_liepe_2016_c1r_note_cites_caron_not_bassani_sternberg():
    """The citation was inverted.

    The paper cites its C1R immunopeptidome to reference 5, *Caron et al.,
    eLife 4 (2015)*, and the primary fibroblasts to reference 6,
    *Bassani-Sternberg et al. (2015)*.  The old note attributed C1R to
    Bassani-Sternberg.
    """
    c1r = next(s for s in _samples(27846572) if "C1R" in s["sample_label"])
    assert "Caron" in c1r["note"]
    assert "data sourced from Bassani-Sternberg" not in c1r["note"]


def test_liepe_2016_t2_rows_are_not_attributed_to_a_phantom_sample(full_observations_df):
    """The removed JY sample was not inert — it was taking T2's evidence.

    JY was curated with ``HLA-A*02:01 HLA-B*07:02 HLA-C*07:02``.  T2's 111
    rows carry ``HLA-A*02:01``, so the join matched them to JY by exact
    allele — the highest-confidence attribution tier — and labeled a
    TAP-deficient hybridoma as an EBV-LCL.  A curated sample the paper
    never mentions is not a harmless extra row; it competes for real
    evidence.
    """
    sub = full_observations_df[full_observations_df["pmid"] == 27846572]
    if sub.empty:
        pytest.skip("Liepe 2016 not present in this build")
    labels = sub["sample_label"].astype(str)
    assert not labels.str.contains("JY").any()
    assert not labels.str.contains("HeLa").any()
    t2 = sub[labels.str.startswith("T2")]
    assert len(t2) == 111
    assert (t2["sample_attribution"].astype(str) == "allele_exact").all()


# ── The #450 primary-source pilot ───────────────────────────────────────────
#
# Four studies read against the paper to check that the flat condition columns
# can carry what a source actually states.  Two of the four disproved existing
# curation, which is the point of reading them: both errors were plausible,
# and neither was visible to any consistency check.


def _condition(pmid: int, condition_id: str) -> dict:
    return next(s for s in _samples(pmid) if s.get("condition_id") == condition_id)


def test_pilot_records_cite_where_in_the_source_the_facts_are():
    """A `primary_source` claim without a locator is an unverifiable assertion."""
    verified = [s for pmid in (31530632, 30833945, 34497125, 40113210) for s in _samples(pmid)]
    assert len(verified) == 19
    for sample in verified:
        assert sample["condition_evidence"] == "primary_source"
        reference = sample["condition_reference"]
        assert reference.startswith("PMC"), reference
        assert len(reference) > 40, f"{reference!r} names no section"


def test_lorente_2019_separates_the_background_from_the_knockout():
    """C1R's own ERAP1 haplotype is not something this study did to it.

    *"C1R cells are ERAP2-positive and express the ERAP1 variant Hap8"*
    (Experimental Procedures, "Cell Lines"), and the KO leaves it alone —
    *"their expression of ERAP1 remained unaltered"*.  So Hap8 is the
    material's background on both arms, while ERAP2 is the intervention on
    one.  Collapsing the two would read as an ERAP1 perturbation.
    """
    wt = _condition(31530632, "c1r_hla_b_40_02_wt")
    ko = _condition(31530632, "c1r_hla_b_40_02_erap2_ko")
    assert wt["condition_background"] == ko["condition_background"] == "ERAP1_hap8"
    assert ko["condition_knockout_genes"] == "ERAP2"
    # ERAP2-positive is stated, so absence here is a claim, not a silence.
    assert wt["condition_knockout_genes"] == "none"
    assert wt["condition_control_for"] == "c1r_hla_b_40_02_erap2_ko"
    # A transfectant on an HLA-low host; the paper never calls it mono-allelic.
    assert wt["condition_mhc_context"] == "mhc_transfectant"


def test_javitt_2019_keeps_both_cytokines_without_splitting_them():
    """Only the combination was profiled, so neither effect is separable.

    *"A549 cells were stimulated with TNF-alpha and IFN-gamma (denoted as T+I) or left
    untreated (UT)"* — there is no single-cytokine MS arm anywhere in the
    paper, so the two factors belong in one row and nothing may infer either
    one alone.
    """
    arms = _samples(30833945)
    assert len(arms) == 2
    treated = _condition(30833945, "a549_lung_cancer_tnfa_ifng")
    assert treated["condition_cytokines"] == "IFNG;TNF"
    assert treated["condition_combination"] == "simultaneous"
    untreated = _condition(30833945, "a549_lung_cancer_untreated")
    assert not untreated.get("condition_cytokines")
    assert untreated["condition_control_for"] == "a549_lung_cancer_tnfa_ifng"


def test_javitt_2019_a549_genotype_matches_the_paper_and_the_rows():
    """`HLA-B*07:02` was a transcription error for `HLA-B*18:01`.

    The paper names B*18:01 twice — *"both HLA-B4403 and -B1801 bind mainly
    to peptides with a glutamic acid at the second position"* — and never
    mentions B*07:02.  IEDB agrees: this PMID's rows carry
    `HLA-A*25:01;HLA-A*30:01;HLA-B*18:01;HLA-B*44:03;HLA-C*12:03`, 1,120 of
    them restricted to B*18:01 alone and none to B*07:02.  A curated genotype
    no row carries is a claim about nothing (#436), and it mattered here: the
    paper's central result is a B*18:01/B*44:03 P2-glutamate motif, while
    B*07:02 is a P2-proline binder.
    """
    for pmid in (30833945, 34171305):
        for sample in _samples(pmid):
            mhc = str(sample.get("mhc") or "")
            if "HLA-A*25:01" in mhc:
                assert "HLA-B*18:01" in mhc, f"PMID {pmid}: {mhc}"
                assert "HLA-B*07:02" not in mhc, f"PMID {pmid}: {mhc}"


def test_stopfer_2021_biopsies_are_frozen_and_of_unstated_treatment():
    """Two assertions the paper does not support, both plausible.

    *"Tumor samples were collected, snap frozen, and stored at -80 C prior
    to analysis"* (Materials and Methods) — the curation said "fresh", a word
    that appears nowhere in the paper or its SI.  And the paper says nothing
    at all about prior or concurrent therapy for these metastatic-melanoma
    patients, so `condition_control: untreated` asserted a fact no source
    states.  An unreported condition must not become an untreated one.
    """
    biopsies = _condition(34497125, "patient_melanoma_biopsies")
    assert biopsies["condition_material"] == "frozen"
    assert "snap-frozen" in biopsies["condition"]
    assert "fresh" not in biopsies["condition"]
    assert not biopsies.get("condition_control")


def test_stopfer_2021_vehicle_is_both_a_vehicle_and_a_comparator():
    """*"treated with binimetinib ... or DMSO as a vehicle control for 72 hrs"*.

    The vehicle exposure and the control role are different facts and both
    survive: DMSO is what the cells saw, `vehicle` is what the arm is for.
    """
    dmso = _condition(34497125, "skmel5_melanoma_dmso_control")
    assert dmso["condition_drugs"] == "DMSO"
    assert dmso["condition_control"] == "vehicle"
    assert dmso["condition_control_for"] == "skmel5_melanoma_binimetinib_meki"


def test_shapiro_2025_each_arm_carries_only_its_own_knockout():
    """*"each with a KO of a single gene"* — 11 KOs plus wildtype, no doubles.

    The study-wide union is what #353 stopped reaching the APM flags; the
    same union must never reach these columns either.  The wildtype arm is
    the paper's comparator for all eleven, which `condition_control_for`
    records once rather than eleven implicit times.
    """
    arms = _samples(40113210)
    assert len(arms) == 12
    knockouts = {
        s["condition_id"]: s.get("condition_knockout_genes", "")
        for s in arms
        if s["condition_id"] != "hap1_wildtype"
    }
    assert len(knockouts) == 11
    for condition_id, genes in knockouts.items():
        assert ";" not in genes, f"{condition_id} claims more than one knockout"
        assert genes, f"{condition_id} claims none"
    assert sorted(knockouts.values()) == sorted(
        ["B2M", "TAP1", "TAP2", "TAPBP", "IRF2", "PDIA3", "ERAP1", "GANAB", "SPPL3", "CANX", "CALR"]
    )
    wildtype = _condition(40113210, "hap1_wildtype")
    assert wildtype["condition_knockout_genes"] == "none"
    assert wildtype["condition_control_for"].split(";") == sorted(knockouts)
