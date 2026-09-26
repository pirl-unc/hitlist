"""Source-specific treatment statements must resolve exactly one arm (#512)."""

import pandas as pd
import pytest

from hitlist.curation import load_pmid_overrides
from hitlist.export import _select_by_elution_conditions, generate_observations_table

SINGLE_CONDITIONS = [
    ("1uM CDK4/6i", "palbociclib_1um"),
    ("10uM CDK4/6i", "palbociclib_10um"),
    ("10ng/mL IFNg", "ifng"),
]
MULTIPLE_CONDITIONS = [
    "1uM CDK4/6i and 10uM CDK4/6i",
    "1uM CDK4/6i and 10ng/mL IFNg",
    "10uM CDK4/6i and 10ng/mL IFNg",
    "1uM CDK4/6i, 10uM CDK4/6i, and 10ng/mL IFNg",
]


def _statement(treatments):
    return f"The epitope was eluted from cells treated with {treatments}."


@pytest.mark.parametrize("line", ["SK-MEL-2", "SK-MEL-5", "SK-MEL-28", "IPC-298"])
@pytest.mark.parametrize("treatment, suffix", SINGLE_CONDITIONS)
def test_stopfer_exact_deposited_condition_selects_one_arm(monkeypatch, line, treatment, suffix):
    # PMID 32488085, Figures 4/6 and Supplementary Data 3/5: separate arms.
    monkeypatch.setattr(
        "hitlist.observations.load_observations",
        lambda **kwargs: pd.DataFrame(
            [
                {
                    "peptide": "AAAAAAAAA",
                    "pmid": 32488085,
                    "mhc_restriction": "HLA class I",
                    "mhc_class": "I",
                    "mhc_species": "Homo sapiens",
                    "cell_name": observed_line + "-Melanocyte",
                    "assay_comments": _statement(treatment),
                    "is_binding_assay": False,
                    "source": "iedb",
                }
                for observed_line in ["SK-MEL-2", "SK-MEL-5", "SK-MEL-28", "IPC-298"]
            ]
        ),
    )
    result = generate_observations_table(exclude_non_peptide_ligand=False)
    row = result[result.cell_name == line + "-Melanocyte"].iloc[0]
    assert row.condition_id == line.lower().replace("-", "_") + "_" + suffix
    assert row.sample_attribution == "elution_conditions"


@pytest.mark.parametrize("treatment", [*MULTIPLE_CONDITIONS, "100uM CDK4/6i"])
def test_mixed_or_unregistered_dose_does_not_select_even_one_remaining_candidate(treatment):
    mapping = load_pmid_overrides()[32488085].get("elution_condition_ids", {})
    candidate = ("high", "palbociclib", {"condition_id": "sk_mel_5_palbociclib_10um"})
    assert (
        _select_by_elution_conditions(
            [candidate], _statement(treatment), curated_condition_ids=mapping
        )
        is None
    )


def test_curated_statement_does_not_apply_to_unregistered_studies():
    candidate = ("high", "palbociclib", {"condition_id": "sk_mel_5_palbociclib_10um"})
    assert _select_by_elution_conditions([candidate], _statement("10uM CDK4/6i")) is None


def test_curated_conditions_apply_after_an_ambiguous_exact_allele_join(monkeypatch):
    monkeypatch.setattr(
        "hitlist.observations.load_observations",
        lambda **kwargs: pd.DataFrame(
            [
                {
                    "peptide": "AAAAAAAAA",
                    "pmid": 32488085,
                    "mhc_restriction": "HLA-A*11:01",
                    "mhc_class": "I",
                    "mhc_species": "Homo sapiens",
                    "cell_name": line + "-Melanocyte",
                    "assay_comments": _statement("10uM CDK4/6i"),
                    "is_binding_assay": False,
                    "source": "iedb",
                }
                for line in ["SK-MEL-5", "SK-MEL-28"]
            ]
        ),
    )
    result = generate_observations_table(exclude_non_peptide_ligand=False)
    assert result.condition_id.tolist() == [
        "sk_mel_5_palbociclib_10um",
        "sk_mel_28_palbociclib_10um",
    ]
    assert set(result.sample_attribution) == {"elution_conditions"}
    assert set(result.sample_match_type) == {"allele_match"}


@pytest.mark.parametrize(
    "mapping",
    [
        [],
        {"": ["treated"]},
        {" statement ": ["treated"]},
        {"statement": "treated"},
        {"statement": []},
        {"statement": [None]},
        {"statement": ["treated", "treated"]},
        {"statement": ["misspelled"]},
        {"statement": ["not_profiled"]},
    ],
)
def test_elution_mapping_rejects_unsupported_or_ambiguous_ids(tmp_path, monkeypatch, mapping):
    import yaml

    from hitlist import curation

    path = tmp_path / "pmid_overrides.yaml"
    path.write_text(
        yaml.safe_dump(
            [
                {
                    "pmid": 99999512,
                    "elution_condition_ids": mapping,
                    "ms_samples": [
                        {
                            "sample_label": "treated",
                            "condition_id": "treated",
                            "condition_status": "unreported",
                        },
                        {
                            "sample_label": "not profiled",
                            "condition_id": "not_profiled",
                            "condition_status": "unreported",
                            "profiled": False,
                        },
                    ],
                }
            ]
        )
    )
    real_path = curation._data_path
    monkeypatch.setattr(
        curation,
        "_data_path",
        lambda filename: str(path) if filename == "pmid_overrides.yaml" else real_path(filename),
    )
    curation.load_pmid_overrides.cache_clear()
    try:
        with pytest.raises(ValueError, match="elution"):
            curation.load_pmid_overrides()
    finally:
        curation.load_pmid_overrides.cache_clear()


# ── PMID 33592498: the GBM lines share class-II alleles (#565) ──


GBM_STATEMENT = "The epitope was eluted from the following conditions: {}."


@pytest.mark.parametrize(
    "deposited_line, condition_prefix, restriction",
    [
        # HLA-DPA1*01:03/DPB1*04:01 is typed in all three lines, so the key is
        # ambiguous for every one of them and only the statement can resolve it.
        ("HRGO02", "hrog02", "HLA-DPA1*01:03/DPB1*04:01"),
        ("HROG17", "hrog17", "HLA-DPA1*01:03/DPB1*04:01"),
        ("RA", "ra", "HLA-DPA1*01:03/DPB1*04:01"),
        # HLA-DRB4*01:03 is typed in HROG02 and RA only -- the original case
        # this map was curated for. HROG17 is deliberately absent: it carries
        # DRB3*02:02, not DRB4, so asking it to claim a DRB4 ligand would be
        # asking for an arm whose own candidate list excludes the restriction.
        ("HRGO02", "hrog02", "HLA-DRB4*01:03"),
        ("RA", "ra", "HLA-DRB4*01:03"),
    ],
)
def test_gbm_ciita_statement_picks_its_own_line_on_a_shared_class_ii_allele(
    monkeypatch, deposited_line, condition_prefix, restriction
):
    """A class-II allele shared between these lines leaves the allele key
    ambiguous; the deposited statement names the line and resolves it. Token
    scoring cannot -- "RA" is two characters and the scorer wants three.
    Without the map the tie first-picks HROG02 for RA's rows.
    """
    monkeypatch.setattr(
        "hitlist.observations.load_observations",
        lambda **kwargs: pd.DataFrame(
            [
                {
                    "peptide": "AAAAAAAAAAAAAAA",
                    "pmid": 33592498,
                    "mhc_restriction": restriction,
                    "mhc_class": "II",
                    "mhc_species": "Homo sapiens",
                    "cell_name": "Glial cell",
                    "assay_comments": GBM_STATEMENT.format(
                        deposited_line + " cells treated with CIITA"
                    ),
                    "is_binding_assay": False,
                    "source": "iedb",
                }
            ]
        ),
    )
    row = generate_observations_table(exclude_non_peptide_ligand=False).iloc[0]
    assert row.condition_id == condition_prefix + "_ciita_transduced_class_ii"
    assert row.sample_attribution == "elution_conditions"
    assert row.sample_match_type == "allele_match"
    # An arm may only be claimed for a restriction its own candidates contain.
    # Without this the class-pool stage can rescue a row onto an arm that was
    # never typed for the allele, and it still reports ``allele_match``.
    assert restriction in row.sample_mhc.split()


def test_parental_statement_never_claims_the_transduced_arm(monkeypatch):
    """The parental lines do not express class II -- that is what the CIITA
    transduction is for -- so a class-II peptide deposited under a
    parental-only statement is the authors' own background call. An allele
    typed in exactly one line makes the (pmid, allele) key unique, which skips
    the tie-break where the statement map is read, and the row would be handed
    the transduced arm as ``allele_exact``: 492 rows of asserted provenance the
    deposit contradicts (#565/#567).
    """
    monkeypatch.setattr(
        "hitlist.observations.load_observations",
        lambda **kwargs: pd.DataFrame(
            [
                {
                    "peptide": "AAAAAAAAAAAAAAA",
                    "pmid": 33592498,
                    # Typed in HROG17 alone, so the allele key is unique.
                    "mhc_restriction": "HLA-DPB1*11:01",
                    "mhc_class": "II",
                    "mhc_species": "Homo sapiens",
                    "cell_name": "Glial cell",
                    "assay_comments": GBM_STATEMENT.format("HROG17 cells"),
                    "is_binding_assay": False,
                    "source": "iedb",
                }
            ]
        ),
    )
    row = generate_observations_table(exclude_non_peptide_ligand=False).iloc[0]
    assert row.sample_label == ""
    assert row.sample_attribution == "elution_conditions_excluded"
    assert row.condition_transduction == ""
    # Its sample_mhc is a union across arms, so the join-provenance column says
    # so rather than claiming allele_match over a pooled candidate set. (The
    # predict guard that once motivated this reads `sample_mhc_origin` now --
    # #564 -- so this assertion stands on `sample_match_type`'s own documented
    # meaning, and the relabel is gated on the row having actually taken the
    # pool rather than on the study merely having one.)
    assert row.sample_match_type == "pmid_class_pool"
    assert row.sample_mhc_origin == "class_pool"
    # Study-origin metadata is a property of the deposit, not of any arm, so
    # it survives a row that reaches no arm (#373). Consensus across the
    # study's class-II arms would not do: they are all CIITA-transduced, so it
    # would hand back the transduction claim the veto just refused.
    assert row.arm_resolution == "curation_gap"
    assert (row.effective_override, row.effective_override_origin) == ("cell_line", "study")


def test_every_deposited_gbm_statement_is_curated():
    """The map is keyed on exact deposited text, so a statement it misses
    silently falls back to token scoring. IEDB carries nine for this study:
    each line parental, each line CIITA-induced, and each line's pair."""
    mapping = load_pmid_overrides()[33592498]["elution_condition_ids"]
    expected = {
        GBM_STATEMENT.format(text)
        for line in ("HRGO02", "HROG17", "RA")
        for text in (
            f"{line} cells",
            f"{line} cells treated with CIITA",
            f"{line} cells, {line} cells treated with CIITA",
        )
    }
    assert set(mapping) == expected
    # A statement naming both arms must keep naming both, so the class-I rows
    # it covers stay ambiguous rather than being handed to one arm.
    both = mapping[GBM_STATEMENT.format("RA cells, RA cells treated with CIITA")]
    assert {"ra_parental_class_i", "ra_ciita_transduced_class_i"} <= set(both)


# ── sample_mhc_origin: where a row's candidates came from (#564) ──


@pytest.mark.parametrize(
    "samples, restriction, expected_origin",
    [
        # One arm, its own candidates: the row carries that arm's list.
        (
            [{"sample_label": "solo", "mhc": "HLA-A*02:01", "mhc_class": "I"}],
            "HLA-A*02:01",
            "sample",
        ),
        # Two arms, a class-only row: no arm's candidates apply, so the class
        # pool fills sample_mhc and the origin says so.
        (
            [
                {"sample_label": "armA", "mhc": "HLA-A*02:01", "mhc_class": "I"},
                {"sample_label": "armB", "mhc": "HLA-B*07:02", "mhc_class": "I"},
            ],
            "HLA class I",
            "class_pool",
        ),
    ],
)
def test_sample_mhc_origin_distinguishes_a_pool_fill_from_an_arm(
    monkeypatch, samples, restriction, expected_origin
):
    """``sample_match_type`` cannot answer this: it reports whether the study
    *has* a class pool, not whether this row took it. That conflation excluded
    14,532 class-only rows from reassignment -- every one of them carrying its
    own arm's candidate list, and none carrying a union (#564).
    """
    entries = {777: {"ms_samples": samples}}
    monkeypatch.setattr("hitlist.export.load_pmid_overrides", lambda: entries)
    monkeypatch.setattr("hitlist.curation.load_pmid_overrides", lambda: entries)
    monkeypatch.setattr(
        "hitlist.observations.load_observations",
        lambda **kwargs: pd.DataFrame(
            [
                {
                    "peptide": "AAAAAAAAA",
                    "pmid": 777,
                    "mhc_restriction": restriction,
                    "mhc_class": "I",
                    "mhc_species": "Homo sapiens",
                    "cell_name": "",
                    "assay_comments": "",
                    "is_binding_assay": False,
                    "source": "iedb",
                }
            ]
        ),
    )
    row = generate_observations_table(exclude_non_peptide_ligand=False).iloc[0]
    assert row.sample_mhc_origin == expected_origin
    assert bool(row.sample_mhc) is True


def test_sample_mhc_origin_is_blank_when_there_are_no_candidates(monkeypatch):
    """Blank means no candidates reached the row at all, which is different
    from a pooled union and from an arm's own list."""
    entries = {778: {"ms_samples": [{"sample_label": "x", "mhc": "", "mhc_class": "I"}]}}
    monkeypatch.setattr("hitlist.export.load_pmid_overrides", lambda: entries)
    monkeypatch.setattr("hitlist.curation.load_pmid_overrides", lambda: entries)
    monkeypatch.setattr(
        "hitlist.observations.load_observations",
        lambda **kwargs: pd.DataFrame(
            [
                {
                    "peptide": "AAAAAAAAA",
                    "pmid": 778,
                    "mhc_restriction": "HLA class I",
                    "mhc_class": "I",
                    "mhc_species": "Homo sapiens",
                    "cell_name": "",
                    "assay_comments": "",
                    "is_binding_assay": False,
                    "source": "iedb",
                }
            ]
        ),
    )
    row = generate_observations_table(exclude_non_peptide_ligand=False).iloc[0]
    assert row.sample_mhc == ""
    assert row.sample_mhc_origin == ""
