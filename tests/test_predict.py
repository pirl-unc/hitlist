import pandas as pd
import pytest

from hitlist.predict import class_i_prediction_scope


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


def test_class_i_alleles_empty_on_sentinels():
    for s in ("", "unknown", "HLA class I", "HLA class II", None):
        scope = class_i_prediction_scope(s)
        assert scope.scorable == ()
        assert not scope.is_eligible, s


def test_class_i_candidates_keep_precision_and_accept_curated_separators():
    assert _scorable("A*02:01; HLA-B*07:02 HLA-DRB1*15:01") == ["HLA-A*02:01", "HLA-B*07:02"]
    # A serotype is not an allele-level candidate, and a class token names
    # nothing: neither yields predictor input.
    assert _scorable("HLA-A2 HLA class I") == []


def test_class_i_candidates_exclude_nonhuman_and_nonclassical_molecules():
    """#574: the two exclusions are not the same kind of exclusion.

    ``HLA-E``/``HLA-G`` are class Ib, so they are not class-I ("Ia") candidates
    at all and never reach this rule.  ``H2-K*b`` and ``Mamu-A*01`` *are*
    class-I candidates that this module's wiring cannot score -- it hands
    MHCflurry's human models the string, and ``_netmhcpan_allele_arg`` would
    spell the mouse allele ``H2-Kb`` rather than netMHCpan's ``H-2-Kb``.  A
    class-I candidate we cannot score leaves the candidate space unresolved, so
    it belongs in ``unscorable`` and blocks the prediction.
    """
    scope = class_i_prediction_scope(
        "HLA-A*02:01 H2-K*b Mamu-A*01 HLA-E*01:01 HLA-G*01:01 HLA-DRB1*15:01"
    )
    assert list(scope.scorable) == ["HLA-A*02:01"]
    # ``Mamu-A*01`` is a retired name; the parser resolves it to the current
    # ``Mamu-A1*001``, which is the designation that would reach a backend.
    assert list(scope.unscorable) == ["H2-K*b", "Mamu-A1*001"]
    assert not scope.is_eligible


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


def test_finer_than_two_field_alleles_stay_scorable():
    """Three- and four-field designations are *more* precise, not less.

    The third field is a synonymous substitution, so the protein is the one the
    two-field name denotes; rejecting these would abstain on better data.
    """
    for fine in ("HLA-A*02:01:01", "HLA-A*02:01:01:02L"):
        scope = class_i_prediction_scope(fine)
        assert list(scope.scorable) == [fine], fine
        assert scope.is_eligible, fine


def test_null_allele_is_not_a_presentation_candidate():
    """A ``N``-suffixed allele is not expressed, so it presents no peptide.

    Absent from curation today; pinned because a best-allele call naming a null
    allele would be a biological falsehood, and because the backends have no
    model for one.
    """
    scope = class_i_prediction_scope("HLA-A*02:01 HLA-A*24:02:01:02N")
    assert list(scope.scorable) == ["HLA-A*02:01"]
    assert list(scope.unscorable) == ["HLA-A*24:02:01:02N"]
    assert not scope.is_eligible


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
