"""Source-specific treatment statements must resolve exactly one arm (#512)."""

import pandas as pd
import pytest

from hitlist.curation import load_pmid_overrides
from hitlist.export import _select_by_elution_conditions, generate_observations_table
from tests.deposited_statements import GBM_STATEMENT

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
    """A class-II peptide deposited under a parental-only statement came off
    the parental sample, whatever the authors think it is.

    An allele typed in exactly one line makes the (pmid, allele) key unique,
    which skips the tie-break where the statement map is read, so before #565
    the row was handed the transduced arm as ``allele_exact``: 492 rows of
    asserted provenance the deposit contradicts. #565 refused that with the
    statement veto but had no arm to offer instead; #567 curates the parental
    class-II arms, so the row now reaches the sample it was eluted from and
    carries the authors' background assessment in ``sample_note`` rather than
    reaching nothing. The guarantee in the name is what both versions assert.
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
    assert row.sample_label == "HROG17 parental (class II)"
    assert row.condition_id == "hrog17_parental_class_ii"
    assert row.sample_attribution == "elution_conditions"
    # Not the transduced twin, which is the whole point: the arm this row
    # reaches states that no transduction was applied to it.
    assert "ciita" not in row.condition_id
    assert row.condition_transduction == "none"
    # It reached a named arm, so its candidates are that arm's own typing
    # rather than a pooled union across arms.
    assert row.sample_match_type == "allele_match"
    assert row.sample_mhc_origin == "sample"
    assert "HLA-DPA1*01:03/DPB1*11:01" in row.sample_mhc.split()
    # The authors' own reading of these peptides travels with the arm; it is
    # recorded, not acted on.
    assert "potential contaminants" in row.sample_note
    assert row.arm_resolution == "multi_arm_evidence"
    assert (row.effective_override, row.effective_override_origin) == ("cell_line", "study")


def test_statement_and_allele_disagreement_claims_no_arm(monkeypatch):
    """Statement and allele pointing at different lines resolves to neither.

    The deposited statement names RA while the restriction is typed in HROG17
    alone. The statement is the per-row evidence, so the RA arms it names are
    the only ones allowed, and none of them carries this allele; the HROG17
    arms the allele key offers are exactly the ones the statement rules out.
    Neither side may be preferred, so the row reaches no arm.
    """
    monkeypatch.setattr(
        "hitlist.observations.load_observations",
        lambda **kwargs: pd.DataFrame(
            [
                {
                    "peptide": "AAAAAAAAAAAAAAA",
                    "pmid": 33592498,
                    "mhc_restriction": "HLA-DPB1*11:01",
                    "mhc_class": "II",
                    "mhc_species": "Homo sapiens",
                    "cell_name": "Glial cell",
                    "assay_comments": GBM_STATEMENT.format("RA cells"),
                    "is_binding_assay": False,
                    "source": "iedb",
                }
            ]
        ),
    )
    row = generate_observations_table(exclude_non_peptide_ligand=False).iloc[0]
    assert row.sample_label == ""
    assert row.condition_id == ""
    assert row.sample_attribution == "elution_conditions_excluded"
    # None of the columns that carry a typing claim may survive either. Curating
    # a second class-II arm per line made every single-line allele key ambiguous,
    # so this row reached ``_consensus_meta``, which blanks ``condition_id`` and
    # keeps everything the two excluded arms agree on -- HROG17's typing, cell and
    # candidate list, reported as ``allele_match`` from a line the statement rules
    # out (#584 review). The veto has to bite before that, not after it.
    assert row.sample_mhc_origin == "class_pool"
    assert row.mhc_basis == ""
    assert row.sample_match_type == "pmid_class_pool"
    assert row.mhc_genotype_cell == ""
    assert "HLA-DPA1*01:03/DPB1*11:01" in row.sample_mhc.split()
    assert "HLA-DPA1*01:03/DPB1*19:01" in row.sample_mhc.split()
    # Study-origin metadata is a property of the deposit, not of any arm, so
    # it survives a row that reaches no arm (#373).
    assert (row.effective_override, row.effective_override_origin) == ("cell_line", "study")
    assert row.arm_resolution == "multi_arm_evidence"


def _vetoable_overrides():
    """A minimal study whose statement map can contradict an allele key.

    The veto needs a shape no real curated study has to keep handy: one
    statement that names arm A, a second that names arm B, and a row whose
    unique (pmid, allele) key points at B while its statement is A's. Using a
    synthetic study keeps the guard covered whatever the corpus's curation
    state -- PMID 33592498 exercised it until #567 gave its parental class-II
    rows an arm of their own, and a guard whose only test rides on one study's
    open curation gap stops being tested the moment that gap is closed.
    """
    return {
        99999567: {
            "elution_condition_ids": {
                "The epitope was eluted from Alpha cells.": ["alpha"],
                "The epitope was eluted from Beta cells.": ["beta"],
            },
            "ms_samples": [
                {
                    "sample_label": "Alpha cells",
                    "condition_id": "alpha",
                    "mhc": "HLA-DRB1*01:01",
                    "mhc_class": "II",
                    "condition": "unperturbed",
                },
                {
                    "sample_label": "Beta cells",
                    "condition_id": "beta",
                    "mhc": "HLA-DRB1*04:01",
                    "mhc_class": "II",
                    "condition": "unperturbed",
                },
            ],
        }
    }


def test_statement_vetoes_an_arm_the_allele_key_would_have_claimed(monkeypatch):
    """An arm the statement excludes is a collision, not a match (#565).

    ``HLA-DRB1*04:01`` is typed in Beta alone, so the allele key is unique and
    skips the tie-break where the map is read -- the row would be handed Beta
    as ``allele_exact`` although its own statement names Alpha.
    """
    monkeypatch.setattr("hitlist.export.load_pmid_overrides", _vetoable_overrides)
    monkeypatch.setattr(
        "hitlist.observations.load_observations",
        lambda **kwargs: pd.DataFrame(
            [
                {
                    "peptide": "AAAAAAAAAAAAAAA",
                    "pmid": 99999567,
                    "mhc_restriction": "HLA-DRB1*04:01",
                    "mhc_class": "II",
                    "mhc_species": "Homo sapiens",
                    "cell_name": "",
                    "assay_comments": "The epitope was eluted from Alpha cells.",
                    "is_binding_assay": False,
                    "source": "iedb",
                }
            ]
        ),
    )
    row = generate_observations_table(exclude_non_peptide_ligand=False).iloc[0]
    assert row.sample_attribution == "elution_conditions_excluded"
    assert row.condition_id == ""
    assert row.sample_label != "Beta cells"
    # And no arm at all, not even the one the statement allows (#581). The
    # veto's consensus used to be taken over the allowed arms, so a statement
    # naming exactly one arm of the row's class handed that arm's whole record
    # back -- ``sample_label``, ``sample_mhc``, the cellular typing -- to a row
    # ``sample_attribution`` says reached none. It is the study's arms of this
    # class that are consensused now, so only what they agree on survives.
    assert row.sample_label == ""
    assert row.sample_mhc_origin == "class_pool"
    assert row.mhc_basis == ""
    assert set(row.sample_mhc.split()) == {"HLA-DRB1*01:01", "HLA-DRB1*04:01"}
    # The restriction itself is untouched -- only the arm claim is refused.
    assert row.mhc_restriction == "HLA-DRB1*04:01"


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


# ── One resolution for every statement-excluded row (#584 review 2) ──────────
#
# Three stages can refuse a row -- the allele stage, the class-pool stage, and
# the assigned-arm check that catches a unique allele key or a curated label --
# and each used to build its own fallback metadata. They disagreed, so what a
# vetoed row reported depended on which stage happened to notice it: an
# excluded arm's label (#581), its typing, ``allele_match`` over a candidate
# list the deposit rules out. These pin the single record all of them produce.

_STMT = "The epitope was eluted from {}."


def _arm(label, condition_id, mhc, mhc_class):
    return {
        "sample_label": label,
        "condition_id": condition_id,
        "mhc": mhc,
        "mhc_class": mhc_class,
        "condition": "unperturbed",
    }


#: (name, ms_samples, statement map, observation row). Every case is a row the
#: deposited statement excludes, reached by a different route.
VETO_SHAPES = [
    (
        # The row's class has exactly one arm, so "consensus" over the arms the
        # statement allows was that arm's whole record -- #581, via the allele
        # stage this time.
        "single_arm_in_class",
        [
            _arm("Alpha cells", "alpha", "HLA-A*02:01", "I"),
            _arm("Beta cells", "beta", "HLA-DRB1*04:01", "II"),
        ],
        {_STMT.format("Alpha cells"): ["alpha"], _STMT.format("Beta cells"): ["beta"]},
        {"mhc_restriction": "HLA-DRB1*04:01", "mhc_class": "II"},
    ),
    (
        # An arm the map never names, surviving narrowing alone on the
        # class-pool path: the statement's own arm is class I, so no arm it
        # names is a candidate at all.
        "lone_unmapped_arm_class_pool",
        [
            _arm("Alpha cells", "alpha", "HLA-A*02:01", "I"),
            _arm("Beta cells", "beta", "HLA-DRB1*04:01", "II"),
            _arm("Gamma cells", "gamma", "HLA-DRB1*07:01", "II"),
        ],
        {_STMT.format("Alpha cells"): ["alpha"], _STMT.format("Beta cells"): ["beta"]},
        {"mhc_restriction": "HLA class II", "mhc_class": "II"},
    ),
    (
        # The same lone unmapped arm on the allele path, where its allele makes
        # a unique key that never reaches the narrowing at all.
        "lone_unmapped_arm_allele_key",
        [
            _arm("Alpha cells", "alpha", "HLA-A*02:01", "I"),
            _arm("Beta cells", "beta", "HLA-DRB1*04:01", "II"),
            _arm("Gamma cells", "gamma", "HLA-DRB1*07:01", "II"),
        ],
        {_STMT.format("Alpha cells"): ["alpha"], _STMT.format("Beta cells"): ["beta"]},
        {"mhc_restriction": "HLA-DRB1*07:01", "mhc_class": "II"},
    ),
    (
        # A class-II allele deposited under mhc_class "I", so the row's class
        # has no pool to fall back on. ``sample_mhc`` is legitimately blank;
        # the provenance columns still have to say the arm was refused.
        "no_class_pool_for_the_rows_class",
        [
            _arm("Alpha cells", "alpha", "HLA-DRB1*01:01", "II"),
            _arm("Beta cells", "beta", "HLA-DRB1*04:01", "II"),
        ],
        {_STMT.format("Alpha cells"): ["alpha"], _STMT.format("Beta cells"): ["beta"]},
        {"mhc_restriction": "HLA-DRB1*04:01", "mhc_class": "I"},
    ),
    (
        # Two excluded arms that agree on their typing: consensus keeps what
        # every candidate shares, so agreement handed back the typing the veto
        # had just refused.
        "excluded_arms_sharing_a_typing",
        [
            _arm("Alpha cells", "alpha", "HLA-A*02:01", "I"),
            _arm("Beta one", "beta1", "HLA-DRB1*04:01", "II"),
            _arm("Beta two", "beta2", "HLA-DRB1*04:01", "II"),
        ],
        {_STMT.format("Alpha cells"): ["alpha"], _STMT.format("Beta cells"): ["beta1", "beta2"]},
        {"mhc_restriction": "HLA-DRB1*04:01", "mhc_class": "II"},
    ),
]


@pytest.mark.parametrize(
    "name, arms, condition_map, row",
    VETO_SHAPES,
    ids=[shape[0] for shape in VETO_SHAPES],
)
def test_every_statement_excluded_row_reports_the_same_record(
    monkeypatch, name, arms, condition_map, row
):
    """Whichever stage refuses it, a vetoed row reads identically."""
    pmid = 99999584
    monkeypatch.setattr(
        "hitlist.export.load_pmid_overrides",
        lambda: {
            pmid: {
                "override": "cell_line",
                "arm_resolution": "curation_gap",
                "elution_condition_ids": condition_map,
                "ms_samples": arms,
            }
        },
    )
    monkeypatch.setattr(
        "hitlist.observations.load_observations",
        lambda **kwargs: pd.DataFrame(
            [
                {
                    "peptide": "AAAAAAAAAAAAAAA",
                    "pmid": pmid,
                    "mhc_species": "Homo sapiens",
                    "cell_name": "",
                    "source_tissue": "",
                    "antigen_processing_comments": "",
                    # Names the class-I arm, which this class-II row can never
                    # have come from.
                    "assay_comments": _STMT.format("Alpha cells"),
                    "is_binding_assay": False,
                    "source": "iedb",
                    **row,
                }
            ]
        ),
    )
    result = generate_observations_table(exclude_non_peptide_ligand=False).iloc[0]

    # No arm, and never half of one: a label without a condition_id is the
    # shape #581 reported.
    assert result.sample_label == ""
    assert result.sample_group == ""
    assert result.condition_id == ""
    assert result.sample_attribution == "elution_conditions_excluded"
    # Never allele_match: whatever its allele matched is an arm the deposit
    # excludes, so the candidates it reports are the class pool's.
    assert result.sample_match_type == "pmid_class_pool"
    assert result.sample_mhc_origin == "class_pool"
    # The typing of an excluded arm never survives, not even when several of
    # them agree on it.
    assert result.mhc_basis == ""
    assert result.mhc_genotype == ""
    assert result.mhc_genotype_cell == ""
    # Study-level facts do survive -- they belong to the deposit, not to an arm
    # (#373) -- including when the row's class has no pool at all.
    assert result.arm_resolution == "curation_gap"
    assert (result.effective_override, result.effective_override_origin) == (
        "cell_line",
        "study",
    )
    # And the candidates are the class pool's union, or nothing when the row's
    # class has none.
    expected_pool = sorted({a["mhc"] for a in arms if a["mhc_class"] == row["mhc_class"]})
    assert sorted(result.sample_mhc.split()) == expected_pool
