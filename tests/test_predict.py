import pandas as pd
import pytest

from hitlist.predict import _class_i_alleles


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
    assert _class_i_alleles("HLA-A*02:01 HLA-B*07:02 HLA-C*07:01") == [
        "HLA-A*02:01",
        "HLA-B*07:02",
        "HLA-C*07:01",
    ]


def test_class_i_alleles_skips_class_ii():
    assert _class_i_alleles("HLA-A*02:01 HLA-DRB1*15:01 HLA-DPB1*04:01") == ["HLA-A*02:01"]


def test_class_i_alleles_empty_on_sentinels():
    for s in ("", "unknown", "HLA class I", "HLA class II", None):
        assert _class_i_alleles(s) == []


def test_class_i_candidates_keep_precision_and_accept_curated_separators():
    assert _class_i_alleles("A*02:01; HLA-B*07:02 HLA-DRB1*15:01") == ["HLA-A*02:01", "HLA-B*07:02"]
    assert _class_i_alleles("HLA-A2 HLA class I") == []


def test_class_i_candidates_exclude_nonhuman_and_nonclassical_molecules():
    assert _class_i_alleles(
        "HLA-A*02:01 H2-K*b Mamu-A*01 HLA-E*01:01 HLA-G*01:01 HLA-DRB1*15:01"
    ) == ["HLA-A*02:01"]


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
