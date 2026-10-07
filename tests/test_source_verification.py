"""Source-verified curation regressions (#558, #559, #567).

Each block below is pinned to a primary source that was read for the fix — the
paper's own figure or Methods text, or the IPD-MHC release — so a later edit
that contradicts it fails here rather than in a corpus rebuild.

The attribution checks write a synthetic ``observations.parquet`` whose rows
reproduce the deposited restriction and elution statement verbatim, and run the
real exporter against the real curated YAML.  They need no built index.
"""

import pandas as pd
import pytest

from hitlist.curation import load_pmid_overrides
from hitlist.export import generate_ms_samples_table, generate_observations_table
from tests.deposited_statements import GBM_STATEMENT

ATTRIBUTION_FIELDS = [
    "sample_label",
    "condition_id",
    "sample_attribution",
    "sample_match_type",
    "sample_mhc",
]


def _write_observations(tmp_path, monkeypatch, rows):
    """Publish ``rows`` as the observations index the exporter reads."""
    from hitlist import observations

    path = tmp_path / "observations.parquet"
    defaults = {
        "mhc_class": "I",
        "mhc_species": "Homo sapiens",
        "source": "iedb",
        "is_binding_assay": False,
        "cell_name": "",
        "source_tissue": "",
        "antigen_processing_comments": "",
        "assay_comments": "",
    }
    pd.DataFrame([{**defaults, **row} for row in rows]).assign(
        assay_method="mass spectrometry"
    ).to_parquet(path, index=False)
    monkeypatch.setattr(observations, "observations_path", lambda: path)
    return path


def _assert_attribution_matches(selected, complete, peptides):
    """A filter must change neither the values nor their dtypes (#532).

    ``.astype(str)`` on both sides was the whole comparison, which hides the
    divergence this guards: ``sample_attribution`` and its neighbours are
    declared categoricals, and a filtered export that rebuilt one as a plain
    object column -- or with a different category set -- compared equal after
    the cast while reading differently to every consumer.
    """
    left = selected.loc[peptides, ATTRIBUTION_FIELDS]
    right = complete.loc[peptides, ATTRIBUTION_FIELDS]
    pd.testing.assert_frame_equal(left.astype(str), right.astype(str))
    assert list(left.dtypes.astype(str)) == list(right.dtypes.astype(str))


# ── #558  Abelin 2017 mono-allelic B721.221 transfectants ────────────────────
#
# Figure 1D of PMC5405381 names the 16 alleles this study profiled.  Read off
# that figure, top to bottom.  A*11:01 and B*07:02 are NOT among them: they
# belong to the Sarkizova 95-allele panel, and carrying them here left the
# deposit's 1,901 A*02:04 and 1,071 B*44:02 rows with no arm to reach.
ABELIN_FIGURE_1D_ALLELES = (
    "HLA-A*02:01",
    "HLA-A*01:01",
    "HLA-A*03:01",
    "HLA-A*24:02",
    "HLA-B*44:02",
    "HLA-B*35:01",
    "HLA-B*51:01",
    "HLA-B*44:03",
    "HLA-B*57:01",
    "HLA-A*29:02",
    "HLA-A*31:01",
    "HLA-A*68:02",
    "HLA-A*02:03",
    "HLA-A*02:07",
    "HLA-A*02:04",
    "HLA-B*54:01",
)

ABELIN_VALIDATION_LABELS = (
    "validation cell lines (HCC1937, HCT116, HeLa)",
    "validation primary fibroblasts",
    "validation PBMCs",
)


def _abelin_entry():
    return load_pmid_overrides()[28228285]


def _abelin_transfectant_arms():
    return [s for s in _abelin_entry()["ms_samples"] if s["sample_label"].startswith("721.221-")]


def _abelin_statement(allele):
    """The deposited elution statement for one transfectant, verbatim.

    IEDB writes the allele without the ``*`` separator and misspells
    "transfected"; both are reproduced exactly, because the statement is the
    join key the exporter matches on.
    """
    return f"The epitope was eluted from B721.221 cells tranfected with {allele.replace('*', '')}."


def test_abelin_roster_is_the_sixteen_alleles_of_figure_1d():
    entry = _abelin_entry()
    assert set(entry["hla_alleles"]["profiled"]) == set(ABELIN_FIGURE_1D_ALLELES)
    arms = {s["mhc"] for s in _abelin_transfectant_arms()}
    assert arms == set(ABELIN_FIGURE_1D_ALLELES)
    assert len(arms) == 16


@pytest.mark.parametrize("absent", ["HLA-A*11:01", "HLA-B*07:02"])
def test_abelin_roster_excludes_the_sarkizova_panel_alleles(absent):
    entry = _abelin_entry()
    assert absent not in entry["hla_alleles"]["profiled"]
    assert absent not in {s["mhc"] for s in _abelin_transfectant_arms()}


@pytest.mark.parametrize(
    "one, other",
    [("HLA-A*02:03", "HLA-A*02:04"), ("HLA-B*44:02", "HLA-B*44:03")],
)
def test_abelin_keeps_the_neighbouring_transfectants_distinct(one, other):
    """Four separate cell lines, never two spellings of two (#558)."""
    by_allele = {s["mhc"]: s for s in _abelin_transfectant_arms()}
    assert by_allele[one]["condition_id"] != by_allele[other]["condition_id"]
    assert by_allele[one]["sample_label"] != by_allele[other]["sample_label"]


def test_abelin_validation_datasets_are_not_attribution_candidates():
    """They are earlier publications' data, so no deposited row is theirs."""
    samples = generate_ms_samples_table()
    abelin = samples[samples.pmid.eq(28228285)].set_index("sample_label")
    for label in ABELIN_VALIDATION_LABELS:
        assert abelin.loc[label, "profiled"] == "false"
    from hitlist.export import _observation_eligible_samples

    eligible = _observation_eligible_samples(samples)
    eligible = eligible[eligible.pmid.eq(28228285)]
    assert set(eligible.sample_label) == {s["sample_label"] for s in _abelin_transfectant_arms()}


def test_every_abelin_transfectant_statement_reaches_its_own_arm(tmp_path, monkeypatch):
    """Per-row regression over all 16 deposited statements."""
    rows = [
        {
            "peptide": "AAAAAAAA" + chr(ord("A") + i),
            "pmid": 28228285,
            "mhc_restriction": allele,
            "cell_name": "B cell",
            "assay_comments": _abelin_statement(allele),
        }
        for i, allele in enumerate(sorted(ABELIN_FIGURE_1D_ALLELES))
    ]
    _write_observations(tmp_path, monkeypatch, rows)
    result = generate_observations_table().set_index("mhc_restriction")
    for allele in ABELIN_FIGURE_1D_ALLELES:
        row = result.loc[allele]
        assert row["sample_label"] == f"721.221-{allele}"
        assert row["sample_mhc"] == allele
        assert row["sample_attribution"] == "allele_exact"
        assert row["sample_match_type"] == "allele_match"
    assert not set(result["sample_label"]) & set(ABELIN_VALIDATION_LABELS)


@pytest.mark.parametrize("conflicting", ["HLA-A*02:04", "HLA-B*44:02"])
def test_abelin_transfectant_row_never_lands_on_a_class_only_validation_sample(
    tmp_path, monkeypatch, conflicting
):
    """The defect #558 reported: a row whose statement names a B721.221
    transfectant was absorbed by the class-only ``HLA class I`` validation
    sample, which had no allele to contradict it."""
    _write_observations(
        tmp_path,
        monkeypatch,
        [
            {
                "peptide": "SIINFEKLL",
                "pmid": 28228285,
                "mhc_restriction": conflicting,
                "cell_name": "B cell",
                "assay_comments": _abelin_statement(conflicting),
            }
        ],
    )
    result = generate_observations_table()
    assert result.sample_label.tolist() == [f"721.221-{conflicting}"]
    assert result.sample_mhc.tolist() == [conflicting]
    assert result.mhc_restriction.tolist() == [conflicting]


@pytest.mark.parametrize(
    "query",
    [
        {"peptide": "AAAAAAAAD"},
        {"mhc_allele": "HLA-A*02:04"},
        {"source": "iedb"},
    ],
)
def test_abelin_attribution_survives_a_filtered_query(tmp_path, monkeypatch, query):
    """A filter must not change a deposited row's sample identity (#532)."""
    rows = [
        {
            "peptide": "AAAAAAAA" + chr(ord("A") + i),
            "pmid": 28228285,
            "mhc_restriction": allele,
            "cell_name": "B cell",
            "assay_comments": _abelin_statement(allele),
        }
        for i, allele in enumerate(sorted(ABELIN_FIGURE_1D_ALLELES))
    ]
    _write_observations(tmp_path, monkeypatch, rows)
    complete = generate_observations_table().set_index("peptide")
    selected = generate_observations_table(**query).set_index("peptide")
    assert "AAAAAAAAD" in selected.index
    _assert_attribution_matches(selected, complete, ["AAAAAAAAD"])
    assert complete.loc["AAAAAAAAD", "sample_label"] == "721.221-HLA-A*02:04"


# ── #559  Sherman 2008 chicken BF2 nomenclature ──────────────────────────────
#
# The paper names its two alleles BF2*2101 and BF2*1301 and gives their GenBank
# accessions ("BF2*1301 (AF013494) and BF2*2101 (AF013493)").  IPD-MHC names the
# same two sequences Gaga-BF2*021:01:01 (CHICKEN08580, cross-references
# AF013493) and Gaga-BF2*004:01:01 (CHICKEN08568, cross-references AF013494) --
# so the old haplotype-based names and the new sequence-based ones are related
# by curated identity, not by padding or stripping a zero.  B21 happens to map
# to allele group 021; B13 maps to 004, because B13's BF2 is B4's.
#
# IEDB deposits one allele under each convention, so each arm is curated under
# the name its own rows carry.
SHERMAN_DEPOSITED = {
    # restriction -> (curated sample mhc, the paper's name, deposited statement)
    "Gaga-BF2*021:01": (
        "Gaga-BF2*021:01",
        "BF2*2101",
        "The epitope was eluted from BF2*2101 from transfected RP9 cells.",
    ),
    "Gaga-BF2*13:01": (
        "Gaga-BF2*13:01",
        "BF2*1301",
        "The epitope was eluted from BF2*1301 from transfected RP9 cells.",
    ),
}


def _sherman_arms():
    return {s["mhc"]: s for s in load_pmid_overrides()[18612635]["ms_samples"]}


def test_sherman_arms_are_named_as_their_deposited_rows_are():
    """Each arm's curated value is the deposit's own string, character for
    character, so the join does not rest on how mhcgnomes spells either name
    today -- pirl-unc/mhcgnomes#199 could change that (#584 review 4). The
    paper's own names stay on the labels and in the note."""
    arms = _sherman_arms()
    assert set(arms) == set(SHERMAN_DEPOSITED)
    for restriction, (curated, paper_name, _) in SHERMAN_DEPOSITED.items():
        assert curated == restriction
        assert paper_name in arms[curated]["sample_label"]
        assert arms[curated]["mhc_basis"] == "selected_restriction"


def test_sherman_curation_invents_no_allele_by_padding_or_stripping():
    """No field-width rule is applied to either arm.

    Asserting mhcgnomes' current spellings would pin a floor with no ceiling:
    pirl-unc/mhcgnomes#199 proposes curated aliases that would legitimately
    change them. What must hold whatever that issue does is the curation's own
    claim -- that neither arm is named by transforming the other's digits.
    ``Gaga-BF2*013:01`` is what padding BF2*1301 would produce and IPD-MHC has
    no such allele (B13's BF2 sequence is B4's, so its IPD name is
    ``Gaga-BF2*004:01``); ``Gaga-BF2*21:01`` is what stripping the B21 arm's
    leading zero would leave.
    """
    import mhcgnomes

    arms = set(_sherman_arms())
    assert "Gaga-BF2*013:01" not in arms
    assert "Gaga-BF2*21:01" not in arms
    # Two arms, two alleles, however the parser spells them today or after #199.
    assert len({mhcgnomes.parse(mhc).to_string() for mhc in arms}) == 2
    # And the IPD identity of each is on the record for a consumer reconciling
    # the deposit's two conventions.
    for mhc, accession, ipd_name in (
        ("Gaga-BF2*021:01", "AF013493", "Gaga-BF2*021:01"),
        ("Gaga-BF2*13:01", "AF013494", "Gaga-BF2*004:01"),
    ):
        note = _sherman_arms()[mhc]["note"]
        assert accession in note
        assert ipd_name in note


@pytest.mark.parametrize("restriction", sorted(SHERMAN_DEPOSITED))
def test_sherman_rows_reach_their_arm_with_their_restriction_intact(
    tmp_path, monkeypatch, restriction
):
    curated, _, statement = SHERMAN_DEPOSITED[restriction]
    _write_observations(
        tmp_path,
        monkeypatch,
        [
            {
                "peptide": "SIINFEKLL",
                "pmid": 18612635,
                "mhc_restriction": restriction,
                "mhc_species": "Gallus gallus",
                "cell_name": "B cell",
                "assay_comments": statement,
            }
        ],
    )
    result = generate_observations_table()
    assert result.mhc_restriction.tolist() == [restriction]
    assert result.sample_mhc.tolist() == [curated]
    assert result.sample_attribution.tolist() == ["allele_exact"]
    assert result.sample_match_type.tolist() == ["allele_match"]
    assert result.arm_resolution.tolist() == ["resolved"]
    assert "RP9 transduced with" in result.sample_label.iloc[0]


def test_sherman_arms_stay_separate_under_a_filtered_query(tmp_path, monkeypatch):
    _write_observations(
        tmp_path,
        monkeypatch,
        [
            {
                "peptide": peptide,
                "pmid": 18612635,
                "mhc_restriction": restriction,
                "mhc_species": "Gallus gallus",
                "cell_name": "B cell",
                "assay_comments": SHERMAN_DEPOSITED[restriction][2],
            }
            for peptide, restriction in [
                ("AAAAAAAAA", "Gaga-BF2*021:01"),
                ("LLLLLLLLL", "Gaga-BF2*13:01"),
            ]
        ],
    )
    complete = generate_observations_table().set_index("peptide")
    assert complete.loc["AAAAAAAAA", "condition_id"] != complete.loc["LLLLLLLLL", "condition_id"]
    for peptide in ("AAAAAAAAA", "LLLLLLLLL"):
        selected = generate_observations_table(peptide=peptide).set_index("peptide")
        _assert_attribution_matches(selected, complete, [peptide])


# ── #567  PMID 33592498 parental (CIITA-negative) class-II arms ──────────────
#
# Every lysate, parental included, went through both affinity plates -- "The
# lysates were loaded first through the HLA-I affinity plate and then through
# the HLA-II affinity plate by gravity at 4 °C" -- and the parental runs
# yielded class-II identifications the authors read as mostly background:
# "As GBM cells do not naturally express HLA-II molecules, only 165, 651, and 83
# peptides were identified in HROG02, HROG17, and RA cells, respectively." IEDB
# deposits 585 of those rows under the three parental-only statements.
GBM_PARENTAL_CLASS_II = {
    # deposited line spelling -> (condition_id, sample_label)
    "HRGO02 cells": ("hrog02_parental_class_ii", "HROG02 parental (class II)"),
    "HROG17 cells": ("hrog17_parental_class_ii", "HROG17 parental (class II)"),
    "RA cells": ("ra_parental_class_ii", "RA parental (class II)"),
}


def _gbm_entry():
    return load_pmid_overrides()[33592498]


def test_gbm_parental_class_ii_arms_are_curated():
    arms = {s["condition_id"]: s for s in _gbm_entry()["ms_samples"]}
    for condition_id, label in GBM_PARENTAL_CLASS_II.values():
        arm = arms[condition_id]
        assert arm["sample_label"] == label
        assert arm["mhc_class"] == "II"
        assert arm["condition"] == "unperturbed"
        assert arm["condition_control"] == "untreated"
        # Pan-HLA-II pull-down, so the candidates are the line's class-II typing.
        assert arm["ip_antibody"] == "HB245/IVA12"
        assert arm["mhc_basis"] == "sample_typing"
        # The transduced twin of the same line offers the same candidates: same
        # cells, same antibody. What differs is the condition, not the typing.
        transduced = arms[condition_id.replace("parental", "ciita_transduced")]
        assert arm["mhc"] == transduced["mhc"]
        # And the authors' own reading of these peptides travels with the arm.
        assert "background level of potential contaminants" in arm["note"]
        assert "165, 651, and 83 peptides" in arm["note"]


@pytest.mark.parametrize("line", sorted(GBM_PARENTAL_CLASS_II))
def test_gbm_parental_statement_names_an_arm_in_each_class(line):
    targets = _gbm_entry()["elution_condition_ids"][GBM_STATEMENT.format(line)]
    class_ii = GBM_PARENTAL_CLASS_II[line][0]
    assert class_ii in targets
    assert class_ii.replace("_class_ii", "_class_i") in targets
    assert not [t for t in targets if "ciita" in t]


@pytest.mark.parametrize(
    "line, restriction",
    [
        # One allele typed in that line only, and one shared with another line:
        # before #567 the first reached a CIITA-transduced arm until the
        # statement veto refused it, and the second reached no arm at all.
        ("HRGO02 cells", "HLA-DRB1*07:01"),
        ("HRGO02 cells", "HLA-DRB4*01:03"),
        ("HROG17 cells", "HLA-DRB1*01:02"),
        ("HROG17 cells", "HLA-DPA1*01:03/DPB1*04:01"),
        ("RA cells", "HLA-DRB1*08:01"),
        ("RA cells", "HLA-DRB4*01:03"),
    ],
)
def test_gbm_parental_class_ii_row_reaches_the_sample_it_was_eluted_from(
    tmp_path, monkeypatch, line, restriction
):
    condition_id, label = GBM_PARENTAL_CLASS_II[line]
    _write_observations(
        tmp_path,
        monkeypatch,
        [
            {
                "peptide": "PEPTIDEKLM",
                "pmid": 33592498,
                "mhc_restriction": restriction,
                "mhc_class": "II",
                "assay_comments": GBM_STATEMENT.format(line),
            }
        ],
    )
    result = generate_observations_table()
    assert result.sample_label.tolist() == [label]
    assert result.condition_id.tolist() == [condition_id]
    assert result.sample_attribution.tolist() == ["elution_conditions"]
    # The arm is unperturbed, and the row is kept as evidence with the
    # authors' caveat attached -- it is not excluded.
    assert result.condition_transduction.tolist() == ["none"]
    assert "potential contaminants" in result.sample_note.iloc[0]


@pytest.mark.parametrize("line", sorted(GBM_PARENTAL_CLASS_II))
def test_gbm_paired_statement_attributes_class_ii_to_the_transduced_arm(
    tmp_path, monkeypatch, line
):
    """A class-II peptide seen under both conditions is presented in one.

    The parental lines do not express HLA-II -- "they do not express HLA-DR,
    HLA-DP, and HLA-DQ molecules" -- and the paper reads their class-II
    identifications as background, so a peptide deposited under both
    conditions is attributed to the arm that presented it rather than made
    ambiguous between the two. Class I on the same statements is the other
    way round: both conditions really do present, so those rows stay
    ambiguous by evidence.
    """
    both = f"{line}, {line} treated with CIITA"
    _write_observations(
        tmp_path,
        monkeypatch,
        [
            {
                "peptide": "PEPTIDEKLM",
                "pmid": 33592498,
                "mhc_restriction": "HLA-DPA1*01:03/DPB1*04:01",
                "mhc_class": "II",
                "assay_comments": GBM_STATEMENT.format(both),
            }
        ],
    )
    result = generate_observations_table()
    expected = GBM_PARENTAL_CLASS_II[line][0].replace("parental", "ciita_transduced")
    assert result.condition_id.tolist() == [expected]
    assert result.condition_transduction.tolist() == ["CIITA"]
    # The parental co-detection travels with the arm as a caveat.
    assert "both the parental and the CIITA-treated" in result.sample_note.iloc[0]


def test_gbm_parental_class_ii_attribution_survives_a_filtered_query(tmp_path, monkeypatch):
    rows = [
        {
            "peptide": "PEPTIDEKL" + chr(ord("A") + i),
            "pmid": 33592498,
            "mhc_restriction": "HLA-DRB4*01:03",
            "mhc_class": "II",
            "assay_comments": GBM_STATEMENT.format(line),
        }
        for i, line in enumerate(["HRGO02 cells", "RA cells"])
    ]
    _write_observations(tmp_path, monkeypatch, rows)
    complete = generate_observations_table().set_index("peptide")
    assert complete["sample_label"].tolist() == [
        "HROG02 parental (class II)",
        "RA parental (class II)",
    ]
    for peptide in complete.index:
        selected = generate_observations_table(peptide=peptide).set_index("peptide")
        _assert_attribution_matches(selected, complete, [peptide])
