import pytest

from hitlist.observations import (
    binding_path,
    is_binding_built,
    is_built,
    load_binding,
    load_ms_observations,
    load_observations,
    observations_path,
)


def test_observations_path():
    p = observations_path()
    assert p.name == "observations.parquet"


def test_binding_path():
    p = binding_path()
    assert p.name == "binding.parquet"
    assert p.parent == observations_path().parent


def test_is_binding_built_bool():
    assert isinstance(is_binding_built(), bool)


def test_is_built_false_initially():
    # May or may not be built depending on test environment
    # Just verify the function returns a bool
    assert isinstance(is_built(), bool)


def test_load_observations_not_built():
    if not is_built():
        with pytest.raises(FileNotFoundError, match="not built"):
            load_observations()


def test_load_observations_if_built():
    if not is_built():
        pytest.skip("Observations table not built")
    df = load_observations()
    assert len(df) > 0
    assert "peptide" in df.columns
    assert "mhc_restriction" in df.columns
    assert "mhc_class" in df.columns
    assert "src_cancer" in df.columns
    assert "source" in df.columns


def test_load_observations_class_filter():
    if not is_built():
        pytest.skip("Observations table not built")
    df = load_observations(mhc_class="I")
    assert len(df) > 0
    assert (df["mhc_class"] == "I").all()


def test_load_observations_species_filter():
    if not is_built():
        pytest.skip("Observations table not built")
    df = load_observations(species="Homo sapiens")
    assert len(df) > 0
    assert (df["mhc_species"] == "Homo sapiens").all()


def test_load_observations_column_select():
    if not is_built():
        pytest.skip("Observations table not built")
    df = load_observations(columns=["peptide", "mhc_restriction"])
    assert set(df.columns) == {"peptide", "mhc_restriction"}


def test_load_ms_observations_alias(tmp_path, monkeypatch):
    """Alias should load the MS parquet with the same filter semantics."""
    import pandas as pd

    ms = pd.DataFrame(
        {
            "peptide": ["AAA"],
            "mhc_restriction": ["HLA-A*02:01"],
            "mhc_class": ["I"],
            "reference_iri": ["ms-1"],
            "pmid": pd.array([1], dtype="Int64"),
            "source": ["iedb"],
            "mhc_species": ["Homo sapiens"],
            "is_binding_assay": [False],
        }
    )
    obs_p = tmp_path / "observations.parquet"
    ms.to_parquet(obs_p, index=False)
    monkeypatch.setattr("hitlist.observations.observations_path", lambda: obs_p)

    df = load_ms_observations(mhc_class="I")
    assert list(df["peptide"]) == ["AAA"]


def test_load_binding_not_built(tmp_path, monkeypatch):
    """load_binding must raise FileNotFoundError when the parquet is missing,
    regardless of whether a sibling binding.parquet exists elsewhere on disk.
    """
    missing = tmp_path / "binding.parquet"
    monkeypatch.setattr("hitlist.observations.binding_path", lambda: missing)
    with pytest.raises(FileNotFoundError, match="not built"):
        load_binding()


def test_load_observations_never_contains_binding_rows(tmp_path, monkeypatch):
    """Smoke test: the MS parquet and binding parquet never overlap by IRI.

    Builds tiny fake fixtures, points paths at tmp_path, and confirms the
    two loaders return disjoint row sets.
    """
    import pandas as pd

    ms = pd.DataFrame(
        {
            "peptide": ["AAA"],
            "mhc_restriction": ["HLA-A*02:01"],
            "mhc_class": ["I"],
            "reference_iri": ["ms-1"],
            "pmid": pd.array([1], dtype="Int64"),
            "source": ["iedb"],
            "mhc_species": ["Homo sapiens"],
            "is_binding_assay": [False],
        }
    )
    bd = pd.DataFrame(
        {
            "peptide": ["BBB"],
            "mhc_restriction": ["HLA-A*02:01"],
            "mhc_class": ["I"],
            "reference_iri": ["bd-1"],
            "pmid": pd.array([1], dtype="Int64"),
            "source": ["iedb"],
            "mhc_species": ["Homo sapiens"],
            "is_binding_assay": [True],
        }
    )
    obs_p = tmp_path / "observations.parquet"
    bd_p = tmp_path / "binding.parquet"
    ms.to_parquet(obs_p, index=False)
    bd.to_parquet(bd_p, index=False)
    monkeypatch.setattr("hitlist.observations.observations_path", lambda: obs_p)
    monkeypatch.setattr("hitlist.observations.binding_path", lambda: bd_p)

    ms_df = load_observations()
    bd_df = load_binding()
    assert set(ms_df["peptide"]) == {"AAA"}
    assert set(bd_df["peptide"]) == {"BBB"}
    assert not ms_df["is_binding_assay"].any()
    assert bd_df["is_binding_assay"].all()


def test_load_binding_filters(tmp_path, monkeypatch):
    """Binding loader respects the same filters as the observations loader."""
    import pandas as pd

    bd = pd.DataFrame(
        {
            "peptide": ["AAA", "BBB", "CCC"],
            "mhc_restriction": ["HLA-A*02:01", "HLA-B*07:02", "HLA-A*02:01"],
            "mhc_class": ["I", "I", "II"],
            "reference_iri": ["b1", "b2", "b3"],
            "pmid": pd.array([1, 2, 3], dtype="Int64"),
            "source": ["iedb", "cedar", "iedb"],
            "mhc_species": ["Homo sapiens"] * 3,
            "is_binding_assay": [True] * 3,
            "serotypes": ["HLA-A2", "HLA-B7", "HLA-A2"],
        }
    )
    bd_p = tmp_path / "binding.parquet"
    bd.to_parquet(bd_p, index=False)
    monkeypatch.setattr("hitlist.observations.binding_path", lambda: bd_p)

    assert len(load_binding()) == 3
    assert len(load_binding(mhc_class="I")) == 2
    assert len(load_binding(source="cedar")) == 1
    assert list(load_binding(mhc_restriction="HLA-A*02:01")["peptide"]) == ["AAA", "CCC"]
    assert set(load_binding(serotype="A2")["peptide"]) == {"AAA", "CCC"}


# ── load_all_evidence (hitlist#47 union helper) ────────────────────────────


def test_load_all_evidence_unions_ms_and_binding(tmp_path, monkeypatch):
    """Union must tag rows with evidence_kind, apply filters symmetrically,
    and concatenate both indexes.  Closes the gap that forced downstream
    tools (tsarina --include-binding-assays) to silently drop binding rows."""
    import pandas as pd

    from hitlist.observations import load_all_evidence

    ms = pd.DataFrame(
        {
            "peptide": ["MS1", "MS2"],
            "mhc_restriction": ["HLA-A*02:01", "HLA-B*07:02"],
            "mhc_class": ["I", "I"],
            "reference_iri": ["m1", "m2"],
            "pmid": pd.array([10, 11], dtype="Int64"),
            "source": ["iedb", "iedb"],
            "mhc_species": ["Homo sapiens"] * 2,
            "is_binding_assay": [False, False],
        }
    )
    bd = pd.DataFrame(
        {
            "peptide": ["BD1", "BD2"],
            "mhc_restriction": ["HLA-A*02:01", "HLA-A*02:01"],
            "mhc_class": ["I", "II"],
            "reference_iri": ["b1", "b2"],
            "pmid": pd.array([20, 21], dtype="Int64"),
            "source": ["iedb", "cedar"],
            "mhc_species": ["Homo sapiens"] * 2,
            "is_binding_assay": [True, True],
        }
    )
    ms_p = tmp_path / "observations.parquet"
    bd_p = tmp_path / "binding.parquet"
    ms.to_parquet(ms_p, index=False)
    bd.to_parquet(bd_p, index=False)
    monkeypatch.setattr("hitlist.observations.observations_path", lambda: ms_p)
    monkeypatch.setattr("hitlist.observations.binding_path", lambda: bd_p)

    df = load_all_evidence()
    assert set(df["peptide"]) == {"MS1", "MS2", "BD1", "BD2"}
    assert set(df["evidence_kind"]) == {"ms", "binding"}
    assert df[df["peptide"] == "MS1"]["evidence_kind"].iloc[0] == "ms"
    assert df[df["peptide"] == "BD1"]["evidence_kind"].iloc[0] == "binding"

    # Filters apply symmetrically.
    f = load_all_evidence(mhc_class="I")
    assert set(f["peptide"]) == {"MS1", "MS2", "BD1"}  # BD2 is class II — drop
    assert set(f[f["evidence_kind"] == "ms"]["peptide"]) == {"MS1", "MS2"}
    assert set(f[f["evidence_kind"] == "binding"]["peptide"]) == {"BD1"}


def test_load_all_evidence_missing_indexes_is_empty(tmp_path, monkeypatch):
    """If neither index has been built, return an empty frame with
    evidence_kind column — don't raise FileNotFoundError (unlike
    load_observations / load_binding, which do)."""
    from hitlist.observations import load_all_evidence

    monkeypatch.setattr(
        "hitlist.observations.observations_path", lambda: tmp_path / "missing1.parquet"
    )
    monkeypatch.setattr("hitlist.observations.binding_path", lambda: tmp_path / "missing2.parquet")

    df = load_all_evidence()
    assert df.empty
    assert "evidence_kind" in df.columns


def test_load_all_evidence_ms_only_when_binding_missing(tmp_path, monkeypatch):
    """If only the MS index is built, return its rows without raising."""
    import pandas as pd

    from hitlist.observations import load_all_evidence

    ms = pd.DataFrame(
        {
            "peptide": ["X"],
            "mhc_restriction": ["HLA-A*02:01"],
            "mhc_class": ["I"],
            "reference_iri": ["ms-1"],
            "pmid": pd.array([1], dtype="Int64"),
            "source": ["iedb"],
            "mhc_species": ["Homo sapiens"],
            "is_binding_assay": [False],
        }
    )
    ms_p = tmp_path / "observations.parquet"
    ms.to_parquet(ms_p, index=False)
    monkeypatch.setattr("hitlist.observations.observations_path", lambda: ms_p)
    monkeypatch.setattr("hitlist.observations.binding_path", lambda: tmp_path / "missing.parquet")

    df = load_all_evidence()
    assert list(df["peptide"]) == ["X"]
    assert list(df["evidence_kind"]) == ["ms"]


# ---------------------------------------------------------------------------
# length_min / length_max filters (#118, v1.15.1+).
# ---------------------------------------------------------------------------


def test_load_observations_length_bounds(tmp_path, monkeypatch):
    """length_min / length_max filter peptide rows inclusively."""
    import pandas as pd

    from hitlist.observations import load_observations

    ms = pd.DataFrame(
        {
            "peptide": ["AAAAAAAA", "AAAAAAAAA", "AAAAAAAAAA", "AAAAAAAAAAAA"],
            "mhc_restriction": ["HLA-A*02:01"] * 4,
            "mhc_class": ["I"] * 4,
            "reference_iri": [f"iri-{i}" for i in range(4)],
            "pmid": pd.array([1, 2, 3, 4], dtype="Int64"),
            "source": ["iedb"] * 4,
            "mhc_species": ["Homo sapiens"] * 4,
            "is_binding_assay": [False] * 4,
        }
    )
    path = tmp_path / "observations.parquet"
    ms.to_parquet(path, index=False)
    monkeypatch.setattr("hitlist.observations.observations_path", lambda: path)

    # Only the 8/9/10 mers (8-10 inclusive).
    df = load_observations(length_min=8, length_max=10)
    assert set(df["peptide"].str.len()) == {8, 9, 10}
    # Only 9-mers.
    df9 = load_observations(length_min=9, length_max=9)
    assert (df9["peptide"].str.len() == 9).all()
    assert len(df9) == 1
    # Min-only excludes shorter.
    df_min = load_observations(length_min=10)
    assert (df_min["peptide"].str.len() >= 10).all()


def test_load_binding_length_bounds(tmp_path, monkeypatch):
    """length_min / length_max compose with other filters on load_binding."""
    import pandas as pd

    from hitlist.observations import load_binding

    binding = pd.DataFrame(
        {
            "peptide": ["AAAAAAAA", "AAAAAAAAA", "AAAAAAAAAAAA"],
            "mhc_restriction": ["HLA-A*02:01"] * 3,
            "mhc_class": ["I"] * 3,
            "reference_iri": [f"b-{i}" for i in range(3)],
            "pmid": pd.array([1, 2, 3], dtype="Int64"),
            "source": ["iedb"] * 3,
            "mhc_species": ["Homo sapiens"] * 3,
            "is_binding_assay": [True] * 3,
        }
    )
    path = tmp_path / "binding.parquet"
    binding.to_parquet(path, index=False)
    monkeypatch.setattr("hitlist.observations.binding_path", lambda: path)

    df = load_binding(length_min=9, length_max=11)
    assert set(df["peptide"].str.len()) == {9}


def test_load_observations_normalizes_unprefixed_alleles_at_load_time(tmp_path, monkeypatch):
    """v1.30.0 / #181: stale parquets carry unprefixed allele forms
    (``A*02:01``) from supplements that predate the
    ``normalize_allele`` call in ``supplement.py``. ``load_observations``
    canonicalizes them at load time so downstream groupbys / exact-match
    filters see one canonical string per allele."""
    import pandas as pd

    from hitlist.observations import load_observations

    df = pd.DataFrame(
        {
            "peptide": ["AAAAAAAAA", "BBBBBBBBB", "CCCCCCCCC"],
            "mhc_restriction": ["A*02:01", "HLA-A*02:01", "B*15:03"],
            "mhc_class": ["I", "I", "I"],
            "reference_iri": [f"r-{i}" for i in range(3)],
            "pmid": pd.array([1, 2, 3], dtype="Int64"),
            "source": ["supplement", "iedb", "supplement"],
            "mhc_species": ["Homo sapiens"] * 3,
        }
    )
    path = tmp_path / "observations.parquet"
    df.to_parquet(path, index=False)
    monkeypatch.setattr("hitlist.observations.observations_path", lambda: path)

    out = load_observations()
    # All three rows should report canonical, prefixed allele strings.
    assert set(out["mhc_restriction"]) == {"HLA-A*02:01", "HLA-B*15:03"}


def test_load_observations_flags_short_class_ii_as_suspect(tmp_path, monkeypatch):
    """v1.30.0 / #182: short peptides (≤10aa) labeled class II are
    flagged ``mhc_class_label_suspect=True`` so consumers can opt out
    of biologically-improbable rows. Class I peptides ≥18aa get the
    same flag in the symmetric direction."""
    import pandas as pd

    from hitlist.observations import load_observations

    df = pd.DataFrame(
        {
            "peptide": [
                "SLLQHLIGL",  # 9aa class II — suspect
                "AAAAAAAAAAAAAAA",  # 15aa class II — plausible
                "AAAAAAAAA",  # 9aa class I — plausible
                "AAAAAAAAAAAAAAAAAA",  # 18aa class I — suspect
            ],
            "mhc_restriction": ["HLA class II", "HLA-DRB1*15:01", "HLA-A*02:01", "HLA-A*02:01"],
            "mhc_class": ["II", "II", "I", "I"],
            "reference_iri": [f"r-{i}" for i in range(4)],
            "pmid": pd.array([1, 2, 3, 4], dtype="Int64"),
            "source": ["iedb"] * 4,
            "mhc_species": ["Homo sapiens"] * 4,
        }
    )
    path = tmp_path / "observations.parquet"
    df.to_parquet(path, index=False)
    monkeypatch.setattr("hitlist.observations.observations_path", lambda: path)

    out = load_observations()
    assert "mhc_class_label_suspect" in out.columns
    suspect = dict(zip(out["peptide"], out["mhc_class_label_suspect"]))
    assert suspect["SLLQHLIGL"]  # 9aa class II
    assert not suspect["AAAAAAAAAAAAAAA"]  # 15aa class II
    assert not suspect["AAAAAAAAA"]  # 9aa class I
    assert suspect["AAAAAAAAAAAAAAAAAA"]  # 18aa class I


def test_load_observations_projects_derived_column_without_pyarrow_failure(tmp_path, monkeypatch):
    """v1.30.4: ``mhc_class_label_suspect`` is computed at load time, not
    stored in the parquet. Projecting it via ``columns=[...]`` used to
    fail because the request was pushed straight to pyarrow which
    raised ``No match for FieldRef.Name(mhc_class_label_suspect)``. The
    loader now strips derived columns from the pushdown, reads their
    deps instead, and trims back to the requested projection."""
    import pandas as pd

    from hitlist.observations import load_observations

    df = pd.DataFrame(
        {
            "peptide": ["SLLQHLIGL", "AAAAAAAAAAAAAAA"],
            "mhc_restriction": ["HLA class II", "HLA-DRB1*15:01"],
            "mhc_class": ["II", "II"],
            "reference_iri": ["r-0", "r-1"],
            "pmid": pd.array([1, 2], dtype="Int64"),
            "source": ["iedb"] * 2,
            "mhc_species": ["Homo sapiens"] * 2,
        }
    )
    path = tmp_path / "observations.parquet"
    df.to_parquet(path, index=False)
    monkeypatch.setattr("hitlist.observations.observations_path", lambda: path)

    # 1. Project the derived column alone — used to error out.
    out1 = load_observations(columns=["mhc_class_label_suspect"])
    assert list(out1.columns) == ["mhc_class_label_suspect"]
    assert len(out1) == 2
    assert out1["mhc_class_label_suspect"].tolist() == [True, False]

    # 2. Project derived + a regular stored column. Order preserved.
    out2 = load_observations(columns=["peptide", "mhc_class_label_suspect"])
    assert list(out2.columns) == ["peptide", "mhc_class_label_suspect"]

    # 3. Projecting a stored column without the derived one keeps working
    #    (no regression on the common case).
    out3 = load_observations(columns=["peptide"])
    assert list(out3.columns) == ["peptide"]
