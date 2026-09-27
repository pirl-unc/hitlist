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
    pd.DataFrame([{**defaults, **row} for row in rows]).to_parquet(path, index=False)
    monkeypatch.setattr(observations, "observations_path", lambda: path)
    return path


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
    pd.testing.assert_frame_equal(
        selected.loc[["AAAAAAAAD"], ATTRIBUTION_FIELDS].astype(str),
        complete.loc[["AAAAAAAAD"], ATTRIBUTION_FIELDS].astype(str),
    )
    assert complete.loc["AAAAAAAAD", "sample_label"] == "721.221-HLA-A*02:04"
