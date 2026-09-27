import pandas as pd
import pytest

from hitlist.curation import class_i_prediction_scope


def _scorable(mhc_field):
    return list(class_i_prediction_scope(mhc_field).scorable)


def test_predict_mhcflurry_scores_more_than_six_unique_pairs(monkeypatch):
    """#488: Class1PresentationPredictor.predict()'s `alleles` argument only
    accepts a flat list of <=6 allele strings (one shared genotype tried
    against every peptide) or a dict of sample_name -> alleles paired with
    `sample_names`. A real query scores more than 6 (peptide, allele) pairs
    almost immediately, so a regression back to the flat one-allele-list-
    per-row form (which mhcflurry silently treats as a >6-allele genotype)
    must fail here, not the first time someone runs a real query."""
    mhcflurry = pytest.importorskip("mhcflurry")

    class _FakeAffinityPredictor:
        supported_peptide_lengths = (5, 15)

    class _FakePresentationPredictor:
        """Mimics just enough of the real API surface/validation to catch
        a regression to the unsupported list-of-single-element-lists shape,
        without needing mhcflurry's real (large, downloaded) model weights."""

        affinity_predictor = _FakeAffinityPredictor()

        @classmethod
        def load(cls):
            return cls()

        def predict(self, peptides, alleles, sample_names=None, verbose=0):
            assert all(5 <= len(p) <= 15 for p in peptides), (
                "caller must filter by supported length"
            )
            if not isinstance(alleles, dict):
                if len(alleles) > 6:
                    raise ValueError("When alleles is a list, it must have at most 6 elements.")
                raise AssertionError("expected the dict + sample_names form, not a flat list")
            assert sample_names is not None and len(sample_names) == len(peptides)
            return pd.DataFrame(
                [
                    {
                        "peptide": pep,
                        "sample_name": sn,
                        "best_allele": alleles[sn][0],
                        "affinity": 100.0,
                        "presentation_percentile": 5.0,
                    }
                    for pep, sn in zip(peptides, sample_names)
                ]
            )

    monkeypatch.setattr(mhcflurry, "Class1PresentationPredictor", _FakePresentationPredictor)

    from hitlist.predict import _predict_mhcflurry

    pairs = pd.DataFrame(
        {
            "peptide": [f"PEPTIDE{i}" for i in range(8)] + ["TOOLONGAPEPTIDESEQ"],
            "allele": [f"HLA-A*{i:02d}:01" for i in range(8)] + ["HLA-A*09:01"],
        }
    )
    out = _predict_mhcflurry(pairs)
    assert len(out) == 9
    # #488: one atypical-length peptide (18-mer, outside MHCflurry's 5-15
    # supported range) must come back NaN, not abort scoring the other 8.
    assert out["affinity_nM"].iloc[:8].tolist() == [100.0] * 8
    assert pd.isna(out["affinity_nM"].iloc[8])
    assert out["presentation_percentile"].iloc[:8].tolist() == [5.0] * 8
    assert pd.isna(out["presentation_percentile"].iloc[8])


def test_class_i_alleles_parses_space_separated():
    scope = class_i_prediction_scope("HLA-A*02:01 HLA-B*07:02 HLA-C*07:01")
    assert list(scope.scorable) == ["HLA-A*02:01", "HLA-B*07:02", "HLA-C*07:01"]
    assert scope.unscorable == ()
    assert scope.is_eligible


def test_class_i_alleles_skips_class_ii():
    scope = class_i_prediction_scope("HLA-A*02:01 HLA-DRB1*15:01 HLA-DPB1*04:01")
    # Class-II candidates are a different class, not an unscorable class-I one:
    # they must not hold back a class-I prediction.
    assert list(scope.scorable) == ["HLA-A*02:01"]
    assert scope.unscorable == ()
    assert scope.is_eligible


@pytest.mark.parametrize(
    "sentinel,expected_unscorable",
    [
        ("", ()),
        ("unknown", ()),
        (None, ()),
        # A class-II sentinel is a different candidate space, not an
        # unresolved class-I one.
        ("HLA class II", ()),
        # A class-I sentinel *is* a class-I designation the source reported:
        # it says class-I material is present and untyped, which is exactly an
        # unresolved candidate space.  Alone it scores nothing either way; the
        # value matters in the mixed case below.
        ("HLA class I", ("human class I",)),
    ],
)
def test_class_i_alleles_empty_on_sentinels(sentinel, expected_unscorable):
    scope = class_i_prediction_scope(sentinel)
    assert scope.scorable == (), f"{sentinel!r} must yield no scorable candidate"
    assert scope.unscorable == expected_unscorable, f"{sentinel!r} unscorable"
    assert not scope.is_eligible, f"{sentinel!r} must not be eligible"


def test_class_i_candidates_keep_precision_and_accept_curated_separators():
    assert _scorable("A*02:01; HLA-B*07:02 HLA-DRB1*15:01") == ["HLA-A*02:01", "HLA-B*07:02"]
    # A serotype is not an allele-level candidate, and a class token names
    # nothing: neither yields predictor input.
    assert _scorable("HLA-A2 HLA class I") == []


def test_class_i_candidates_exclude_nonhuman_and_nonclassical_molecules():
    """#574: a class-I molecule the backends cannot score blocks the context.

    ``H2-K*b`` and ``Mamu-A*01`` are class-I alleles this module's wiring cannot
    score -- it hands MHCflurry's human models the string, and
    ``_netmhcpan_allele_arg`` would spell the mouse allele ``H2-Kb`` rather than
    netMHCpan's ``H-2-Kb``.  ``HLA-E``/``HLA-G`` are class ``Ib``: genuine
    class-I presenting molecules that no wired predictor models.  All four leave
    the class-I candidate space unresolved, so they belong in ``unscorable``.
    Only the class-II allele is irrelevant to a class-I prediction.
    """
    scope = class_i_prediction_scope(
        "HLA-A*02:01 H2-K*b Mamu-A*01 HLA-E*01:01 HLA-G*01:01 HLA-DRB1*15:01"
    )
    assert list(scope.scorable) == ["HLA-A*02:01"]
    # ``Mamu-A*01`` is a retired name; the parser resolves it to the current
    # ``Mamu-A1*001``, which is the designation that would reach a backend.
    assert list(scope.unscorable) == [
        "H2-K*b",
        "HLA-E*01:01",
        "HLA-G*01:01",
        "Mamu-A1*001",
    ]
    assert not scope.is_eligible


@pytest.mark.parametrize(
    "field,expected_unscorable",
    [
        # A serotype reaches the boundary through `SampleMhcCandidates.
        # serotypes`, not `exact`, so a partition built only from `exact`
        # never saw it and silently predicted on the remainder.
        ("HLA-A2 HLA-B*07:02", ("HLA-A2",)),
        ("HLA-A3 supertype HLA-B*07:02", ("HLA-A3",)),
        # A class-I sentinel reaches it through `imprecise`, likewise unseen.
        ("HLA-A*02:01 HLA class I", ("human class I",)),
        # A locus is `exact` but names no protein.
        ("HLA-A*02:01 HLA-A", ("HLA-A",)),
    ],
)
def test_imprecise_class_i_candidates_of_any_kind_block_prediction(field, expected_unscorable):
    """#574: precision must be judged on every class-I designation.

    Each of these has a resolvable class-I candidate beside something that
    names no protein.  Scoring the resolvable one would pick a winner from a
    candidate space we know is incomplete.
    """
    scope = class_i_prediction_scope(field)
    assert scope.scorable, f"{field!r} should still expose its resolved candidate"
    assert scope.unscorable == expected_unscorable, f"{field!r} unscorable"
    assert not scope.is_eligible, f"{field!r} must not be eligible"


@pytest.mark.parametrize(
    "field",
    [
        # A class-II allele, pair, serotype or locus is a different candidate
        # space; none of them says anything is missing from the class-I one.
        "HLA-A*02:01 HLA-DRB1*15:01",
        "HLA-A*02:01 HLA-DRA1*01:01-DRB1*15:01",
        "HLA-A*02:01 HLA-DR15",
        "HLA-A*02:01 BoLA-DR",
        "HLA-A*02:01 SLA-DR",
        "HLA-A*02:01 HLA class II",
    ],
)
def test_other_class_designations_do_not_block_a_class_i_prediction(field):
    """The other side of the rule: only class-I gaps count against class I.

    ``BoLA-DR`` and ``SLA-DR`` parse to a class-II locus, so despite being
    imprecise they are not an unresolved *class-I* candidate.  Treating them as
    one would abstain on every mixed-class genotype.
    """
    scope = class_i_prediction_scope(field)
    assert list(scope.scorable) == ["HLA-A*02:01"], field
    assert scope.unscorable == (), f"{field!r} must not poison the class-I space"
    assert scope.is_eligible, field


def test_gene_only_candidates_are_never_predictor_input():
    """#574: ``_class_i_alleles("HLA-A")`` used to return ``["HLA-A"]``.

    A ``Gene`` is in ``SampleMhcCandidates.exact`` because the source named it
    outright, not because it identifies a protein.  Neither backend can score a
    locus.
    """
    for gene_only in ("HLA-A", "HLA-A HLA-B", "HLA-A HLA-B HLA-C"):
        scope = class_i_prediction_scope(gene_only)
        assert scope.scorable == (), gene_only
        assert list(scope.unscorable) == sorted(gene_only.split()), gene_only
        assert not scope.is_eligible, gene_only


def test_one_field_allele_groups_are_never_predictor_input():
    """#574: ``_class_i_alleles("HLA-A*02")`` used to return ``["HLA-A*02"]``.

    A one-field designation is an allele *group* -- ``HLA-B*27`` covers more
    than 100 proteins with materially different motifs.  This is the real
    curated pattern from PMID 24616531, which types six loci to one field.
    """
    scope = class_i_prediction_scope("HLA-A*01 HLA-A*03 HLA-B*07 HLA-B*27 HLA-C*02 HLA-C*07")
    assert scope.scorable == ()
    assert len(scope.unscorable) == 6
    assert not scope.is_eligible


def test_mixed_precise_and_unresolved_candidates_abstain_without_dropping_either():
    """#574: the mixed case is the dangerous one.

    The real curated pattern from PMID 32938616: five two-field alleles and
    ``HLA-B*47``.  Scoring only the five would name a best allele among a
    candidate space we know is incomplete -- if the peptide is really a
    ``B*47:xx`` ligand the winner is wrong.  Both halves stay visible so a
    caller can say *why* the context abstained.
    """
    scope = class_i_prediction_scope(
        "HLA-A*02:01 HLA-A*03:01 HLA-B*40:02 HLA-B*47 HLA-C*03:04 HLA-C*06:02"
    )
    assert list(scope.scorable) == [
        "HLA-A*02:01",
        "HLA-A*03:01",
        "HLA-B*40:02",
        "HLA-C*03:04",
        "HLA-C*06:02",
    ]
    assert list(scope.unscorable) == ["HLA-B*47"]
    assert not scope.is_eligible


@pytest.mark.parametrize(
    "fine,why",
    [
        ("HLA-A*02:01:01", "third field is a synonymous DNA substitution"),
        ("HLA-A*02:01:01:02L", "L annotates low surface expression, not absence"),
        ("HLA-A*24:02:01:02Q", "Q flags questionable expression, still expressed"),
    ],
)
def test_finer_than_two_field_alleles_stay_scorable(fine, why):
    """Extra fields refine the DNA sequence or annotate expression level.

    Neither changes the binding groove, and ``L``/``Q`` molecules still reach
    the cell surface, so all three remain candidates that could present the
    peptide.  Rejecting them would abstain on better data.
    """
    scope = class_i_prediction_scope(fine)
    assert list(scope.scorable) == [fine], why
    assert scope.unscorable == (), why
    assert scope.is_eligible, why


@pytest.mark.parametrize(
    "absent,annotation",
    [
        ("HLA-A*24:02:01:02N", "N -- null, no product"),
        ("HLA-B*44:02:01:02S", "S -- secreted, never membrane-bound"),
        ("HLA-A*01:01:01:02C", "C -- cytoplasm only, never presented on"),
    ],
)
def test_non_surface_alleles_are_dropped_not_counted_as_unresolved(absent, annotation):
    """#574: a molecule that reaches no cell surface cannot be the answer.

    Unlike an imprecise designation, dropping one of these does not make the
    remaining set incomplete -- it could never have presented the peptide -- so
    the context stays eligible on its expressed candidates rather than
    abstaining.
    """
    scope = class_i_prediction_scope(f"HLA-A*02:01 {absent}")
    assert list(scope.scorable) == ["HLA-A*02:01"], annotation
    assert scope.unscorable == (), annotation
    assert scope.is_eligible, annotation

    alone = class_i_prediction_scope(absent)
    assert alone.scorable == (), annotation
    assert alone.unscorable == (), annotation
    assert not alone.is_eligible, annotation


def test_mutation_selector_is_refused_rather_than_read_as_wild_type():
    """A mutant designation must not silently become its wild-type candidate.

    ``sample_mhc_candidates`` already refuses this, and the refusal reaches the
    prediction boundary unchanged -- that is the check #574 asks for on
    mutation selectors.  Loud is correct here: an engineered molecule scored as
    wild type would be a fabricated restriction.
    """
    with pytest.raises(ValueError, match="mutation"):
        class_i_prediction_scope("HLA-A*02:01 K66A")


def test_reassign_class_ii_not_implemented():
    import pytest

    from hitlist.predict import reassign_class_only_alleles

    with pytest.raises(NotImplementedError):
        reassign_class_only_alleles(mhc_class="II")


def test_reassign_empty_when_no_class_only_rows(monkeypatch):
    """If generate_observations_table returns no class-only rows,
    reassign returns an empty DataFrame with the documented schema.
    """
    from hitlist import predict

    # Stub generate_observations_table to return a dataframe with no
    # class-only rows.
    fake = pd.DataFrame(
        {
            "peptide": ["AAAAAAAAA"],
            "mhc_restriction": ["HLA-A*02:01"],
            "is_monoallelic": [False],
            "sample_mhc": ["HLA-A*02:01 HLA-B*07:02"],
            "sample_label": ["x"],
            "sample_match_type": ["allele_match"],
            "sample_mhc_origin": ["sample"],
            "pmid": [1],
        }
    )
    monkeypatch.setattr("hitlist.export.generate_observations_table", lambda *a, **kw: fake)
    result = predict.reassign_class_only_alleles(method="mhcflurry")
    assert result.empty
    assert set(result.columns) >= {
        "peptide",
        "best_allele",
        "best_presentation_percentile",
        "is_strong_binder",
    }


def test_predict_imports_no_private_names_from_export_or_curation():
    """Drift guard (#564/#574).

    ``predict`` reached into ``export._sample_alleles``, so the contract it
    depended on had no public name and no documented guarantees -- and the
    docstring it was not bound by turned out to be wrong about loci, which is
    how #574 survived. Anything ``predict`` needs from ``export`` or
    ``curation`` gets a public name there.

    Parsed rather than grepped so ``import public_name as _alias`` -- which is
    a local naming choice, not a private dependency -- is not a false positive.
    """
    import ast
    import pathlib

    source = pathlib.Path(__file__).resolve().parents[1] / "hitlist" / "predict.py"
    tree = ast.parse(source.read_text())
    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or not node.module:
            continue
        if node.module.split(".")[-1] not in {"export", "curation"}:
            continue
        for alias in node.names:
            if alias.name.startswith("_"):
                offenders.append(f"predict.py:{node.lineno}: {node.module}.{alias.name}")
    assert not offenders, (
        "predict must depend on public, documented names from export/curation:\n"
        + "\n".join(offenders)
    )


def test_abstained_contexts_are_reported_rather_than_silently_missing(monkeypatch):
    """#574: an abstention must not look like an absence.

    A skipped context simply does not appear in the result, which is
    indistinguishable from "this study had no class-only peptides". The tally
    names the designations responsible so a caller can tell the two apart.
    """
    from hitlist import predict
    from hitlist.curation import MHC_TYPING_COLUMNS

    rows = pd.DataFrame(
        [
            {
                "peptide": "AAAAAAAAA",
                "pmid": 1,
                "sample_label": "cell",
                "sample_mhc": "HLA-A*02:01 HLA-B*27",
                "mhc_restriction": "HLA class I",
                "is_monoallelic": False,
                "sample_match_type": "pmid_class_pool",
                "sample_mhc_origin": "sample",
                **dict.fromkeys(MHC_TYPING_COLUMNS, ""),
            }
        ]
    )
    monkeypatch.setattr("hitlist.export.generate_observations_table", lambda **kw: rows)

    with pytest.warns(UserWarning, match=r"HLA-B\*27") as recorded:
        result = predict.reassign_class_only_alleles()
    assert result.empty
    assert "1 class-only observations" in str(recorded[0].message)


def test_no_warning_when_every_context_is_resolved(monkeypatch):
    """The tally must stay quiet on clean input, or it is noise."""
    import warnings as _warnings

    from hitlist import predict
    from hitlist.curation import MHC_TYPING_COLUMNS

    rows = pd.DataFrame(
        [
            {
                "peptide": "AAAAAAAAA",
                "pmid": 1,
                "sample_label": "cell",
                "sample_mhc": "HLA-A*02:01 HLA-B*07:02",
                "mhc_restriction": "HLA class I",
                "is_monoallelic": False,
                "sample_match_type": "pmid_class_pool",
                "sample_mhc_origin": "sample",
                **dict.fromkeys(MHC_TYPING_COLUMNS, ""),
            }
        ]
    )
    monkeypatch.setattr("hitlist.export.generate_observations_table", lambda **kw: rows)
    monkeypatch.setattr(
        predict,
        "_predict_mhcflurry",
        lambda pairs: pairs.assign(affinity_nM=10.0, presentation_percentile=0.1),
    )
    with _warnings.catch_warnings():
        _warnings.simplefilter("error")
        assert len(predict.reassign_class_only_alleles()) == 1


def test_class_i_prediction_scope_cache_is_cleared_between_builds():
    """Registered in ``_clear_curation_caches`` like every other curation cache.

    The scope resolves retired allele names through curated identity data, so a
    rebuild in the same process must not answer from the previous YAML's
    mapping.
    """
    from hitlist import curation

    curation.class_i_prediction_scope("HLA-A*02:01")
    assert curation.class_i_prediction_scope.cache_info().currsize > 0
    curation._clear_curation_caches()
    assert curation.class_i_prediction_scope.cache_info().currsize == 0
