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


@pytest.mark.integration
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


@pytest.mark.integration
def test_load_observations_class_filter():
    if not is_built():
        pytest.skip("Observations table not built")
    df = load_observations(mhc_class="I")
    assert len(df) > 0
    assert (df["mhc_class"] == "I").all()


@pytest.mark.integration
def test_load_observations_species_filter():
    if not is_built():
        pytest.skip("Observations table not built")
    df = load_observations(species="Homo sapiens")
    assert len(df) > 0
    assert (df["mhc_species"] == "Homo sapiens").all()


@pytest.mark.integration
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
    """v1.30.0 / #182 + v1.30.17 / #201: rows whose class label
    disagrees with the bimodal length distribution flag
    ``mhc_class_label_suspect=True``. v1.30.17 narrowed the
    definition: the 8-10 aa class II range and 13-14 aa class I
    range are now ``borderline`` (uncommon but real), so the
    binary suspect flag fires only when severity is ``suspect`` or
    ``implausible``."""
    import pandas as pd

    from hitlist.observations import load_observations

    df = pd.DataFrame(
        {
            "peptide": [
                "SLLQHLI",  # 7aa class II — suspect (post-v1.30.17)
                "AAAAAAAAAAAAAAA",  # 15aa class II — plausible
                "AAAAAAAAA",  # 9aa class I — plausible
                "AAAAAAAAAAAAAAAAAA",  # 18aa class I — implausible
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
    assert suspect["SLLQHLI"]  # 7aa class II — clearly suspect
    assert not suspect["AAAAAAAAAAAAAAA"]  # 15aa class II — canonical
    assert not suspect["AAAAAAAAA"]  # 9aa class I — canonical
    assert suspect["AAAAAAAAAAAAAAAAAA"]  # 18aa class I — implausible


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
            # 7aa class II → suspect under v1.30.17 tiering;
            # 15aa class II → ok.
            "peptide": ["SLLQHLI", "AAAAAAAAAAAAAAA"],
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


def test_exclude_class_label_suspect_drops_short_class_ii_and_long_class_i(tmp_path, monkeypatch):
    """v1.30.11 / #182: ``exclude_class_label_suspect=True`` drops the
    rows the suspect flag would mark — model training pipelines can
    opt out of class-label drift in one line."""
    import pandas as pd

    from hitlist.observations import load_observations

    df = pd.DataFrame(
        {
            "peptide": [
                "SLLQHLI",  # 7aa class II — suspect (post-v1.30.17), dropped
                "AAAAAAAAAAAAAAA",  # 15aa class II — kept
                "AAAAAAAAA",  # 9aa class I — kept
                "AAAAAAAAAAAAAAAAAA",  # 18aa class I — implausible, dropped
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

    # Default: all 4 rows.
    out_all = load_observations()
    assert len(out_all) == 4

    # Filter on: 2 rows survive.
    out_filtered = load_observations(exclude_class_label_suspect=True)
    assert len(out_filtered) == 2
    survivors = set(out_filtered["peptide"])
    assert survivors == {"AAAAAAAAAAAAAAA", "AAAAAAAAA"}


def test_exclude_class_label_suspect_works_with_explicit_projection(tmp_path, monkeypatch):
    """The filter must also work when the caller projects with
    ``columns=[...]`` that doesn't include ``mhc_class`` or ``peptide``
    — those are the deps the suspect flag is computed from, so the
    loader must read them anyway and trim afterward."""
    import pandas as pd

    from hitlist.observations import load_observations

    df = pd.DataFrame(
        {
            # PMID 1: 7aa class II → suspect tier (post-v1.30.17), dropped.
            # PMID 2: 15aa class II → ok, kept.
            "peptide": ["SLLQHLI", "AAAAAAAAAAAAAAA"],
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

    # Caller projects only `pmid` — no mhc_class, no peptide. The filter
    # still has to drop the 7-aa class-II row (PMID 1) and keep PMID 2.
    out = load_observations(columns=["pmid"], exclude_class_label_suspect=True)
    assert list(out.columns) == ["pmid"]
    assert out["pmid"].tolist() == [2]


def test_exclude_non_peptide_ligand_default_drops_cd1_mr1_mic_rows(tmp_path, monkeypatch):
    """#228: by default, ``load_observations`` drops rows whose MHC
    molecule presents lipids/metabolites/stress-ligands rather than
    peptides. Opting in via ``exclude_non_peptide_ligand=False`` keeps
    them. The flag is also materialized as a column so consumers can
    inspect it."""
    import pandas as pd

    from hitlist.observations import load_observations

    df = pd.DataFrame(
        {
            # 4 non-peptide-MHC rows — all must be dropped by default.
            # 1 H2-M3 row — class Ib peptide presenter, MUST survive.
            # 1 plain HLA-A*02:01 row — must survive.
            "peptide": [
                "lipid-A",
                "5-OP-RU",
                "stress-ligand-X",
                "phosphoantigen-Y",
                "FAPGNYPAL",  # real N-formyl peptide on H2-M3
                "SIINFEKL",
            ],
            # MICA is left in canonical "HLA-MICA" form because the
            # loader's #181 backstop normalization (mhcgnomes) expands a
            # bare "MICA" to "HLA-MICA"; either form must flag the same way.
            "mhc_restriction": [
                "human-CD1d",
                "human-MR1",
                "HLA-MICA",
                "cattle-CD1b3",
                "H2-M3",
                "HLA-A*02:01",
            ],
            "mhc_class": ["non classical"] * 5 + ["I"],
            "reference_iri": [f"r-{i}" for i in range(6)],
            "pmid": pd.array(range(6), dtype="Int64"),
            "source": ["iedb"] * 6,
            "mhc_species": ["Homo sapiens"] * 6,
        }
    )
    path = tmp_path / "observations.parquet"
    df.to_parquet(path, index=False)
    monkeypatch.setattr("hitlist.observations.observations_path", lambda: path)

    # Default: 4 non-peptide rows dropped, H2-M3 + classical kept.
    default = load_observations()
    assert "is_non_peptide_ligand" in default.columns
    assert default["is_non_peptide_ligand"].dtype == bool
    assert not default["is_non_peptide_ligand"].any()
    assert set(default["mhc_restriction"]) == {"H2-M3", "HLA-A*02:01"}

    # Opt-in: all 6 rows surface; 4 carry the flag, 2 don't.
    optin = load_observations(exclude_non_peptide_ligand=False)
    assert len(optin) == 6
    flagged = dict(zip(optin["mhc_restriction"], optin["is_non_peptide_ligand"]))
    assert flagged == {
        "human-CD1d": True,
        "human-MR1": True,
        "HLA-MICA": True,
        "cattle-CD1b3": True,
        "H2-M3": False,
        "HLA-A*02:01": False,
    }


def test_exclude_non_peptide_ligand_works_with_projection(tmp_path, monkeypatch):
    """The filter must also work when the caller projects with
    ``columns=[...]`` that omits ``mhc_restriction`` (the dep). The
    loader pulls the dep in for the filter, then trims back."""
    import pandas as pd

    from hitlist.observations import load_observations

    df = pd.DataFrame(
        {
            "peptide": ["GPI-mannoside", "SIINFEKL"],
            "mhc_restriction": ["mouse-CD1d", "HLA-A*02:01"],
            "mhc_class": ["non classical", "I"],
            "reference_iri": ["r-0", "r-1"],
            "pmid": pd.array([1, 2], dtype="Int64"),
            "source": ["iedb"] * 2,
            "mhc_species": ["Mus musculus", "Homo sapiens"],
        }
    )
    path = tmp_path / "observations.parquet"
    df.to_parquet(path, index=False)
    monkeypatch.setattr("hitlist.observations.observations_path", lambda: path)

    out = load_observations(columns=["peptide"])
    assert list(out.columns) == ["peptide"]
    assert out["peptide"].tolist() == ["SIINFEKL"]


# ── #45 multi-allele restriction / set-membership filters ──────────────


def test_mhc_restriction_filter_matches_donor_set_member(tmp_path, monkeypatch):
    """#45: ``mhc_restriction`` post-#45 may be a single 4-digit allele
    OR a semicolon-joined donor set.  A query for a single allele must
    match BOTH single-allele rows and donor-set rows that contain that
    allele as a token.  Pre-#45 the filter was strict equality which
    silently excluded multi-allelic donor cohorts."""
    import pandas as pd

    from hitlist.observations import load_observations

    df = pd.DataFrame(
        {
            "peptide": ["SIINFEKL", "GLCTLVAML", "DONOR_SET_PEP", "OTHER_DONOR"],
            "mhc_restriction": [
                "HLA-A*02:01",  # exact A*02:01 — matches
                "HLA-A*11:01",  # exact A*11:01 — does not match
                # MEL3 donor set: contains A*02:01 → matches
                "HLA-A*02:01;HLA-A*03:01;HLA-B*27:05;HLA-C*06:02",
                # MEL2 donor set: no A*02:01 → does not match
                "HLA-A*01:01;HLA-B*38:01;HLA-C*01:02",
            ],
            "mhc_class": ["I"] * 4,
            "reference_iri": [f"r-{i}" for i in range(4)],
            "pmid": pd.array([1, 2, 3, 4], dtype="Int64"),
            "source": ["iedb"] * 4,
            "mhc_species": ["Homo sapiens"] * 4,
        }
    )
    path = tmp_path / "observations.parquet"
    df.to_parquet(path, index=False)
    monkeypatch.setattr("hitlist.observations.observations_path", lambda: path)

    out = load_observations(mhc_restriction="HLA-A*02:01")
    assert set(out["peptide"]) == {"SIINFEKL", "DONOR_SET_PEP"}


def test_mhc_allele_in_set_filter_post_load(tmp_path, monkeypatch):
    """``mhc_allele_in_set`` filters on ``mhc_allele_set`` membership
    (semicolon-joined).  Equivalent semantics to ``mhc_restriction``
    filter post-#45 but operates on the dedicated set column — kept
    available for callers who want the explicit knob."""
    import pandas as pd

    from hitlist.observations import load_observations

    df = pd.DataFrame(
        {
            "peptide": ["P1", "P2", "P3"],
            "mhc_restriction": ["HLA class I"] * 3,
            "mhc_class": ["I"] * 3,
            "mhc_allele_set": [
                "HLA-A*02:01;HLA-A*03:01;HLA-B*15:01",  # has A*02:01
                "HLA-A*11:01;HLA-B*44:02",  # no A*02:01
                "HLA-A*02:01;HLA-B*07:02",  # has A*02:01
            ],
            "reference_iri": [f"r-{i}" for i in range(3)],
            "pmid": pd.array([1, 2, 3], dtype="Int64"),
            "source": ["iedb"] * 3,
            "mhc_species": ["Homo sapiens"] * 3,
        }
    )
    path = tmp_path / "observations.parquet"
    df.to_parquet(path, index=False)
    monkeypatch.setattr("hitlist.observations.observations_path", lambda: path)

    out = load_observations(mhc_allele_in_set="HLA-A*02:01")
    assert set(out["peptide"]) == {"P1", "P3"}


def test_mhc_allele_provenance_filter(tmp_path, monkeypatch):
    """``mhc_allele_provenance`` selects rows by how their set was
    obtained.  Useful for strict-allele training (``"exact"``) vs.
    sample-narrowed multi-allele (``"peptide_attribution"``) vs. donor
    set (``"sample_allele_match"``)."""
    import pandas as pd

    from hitlist.observations import load_observations

    df = pd.DataFrame(
        {
            "peptide": ["P_EXACT", "P_ATTR", "P_SAMPLE", "P_POOL"],
            "mhc_restriction": [
                "HLA-A*02:01",
                "HLA-A*02:01;HLA-B*38:01",
                "HLA-A*01:01;HLA-A*02:01;HLA-B*38:01",
                "HLA-A*02:01;HLA-A*03:01",
            ],
            "mhc_class": ["I"] * 4,
            "mhc_allele_provenance": [
                "exact",
                "peptide_attribution",
                "sample_allele_match",
                "pmid_class_pool",
            ],
            "reference_iri": [f"r-{i}" for i in range(4)],
            "pmid": pd.array([1, 2, 3, 4], dtype="Int64"),
            "source": ["iedb"] * 4,
            "mhc_species": ["Homo sapiens"] * 4,
        }
    )
    path = tmp_path / "observations.parquet"
    df.to_parquet(path, index=False)
    monkeypatch.setattr("hitlist.observations.observations_path", lambda: path)

    # Strict allele-resolved only.
    out = load_observations(mhc_allele_provenance="exact")
    assert set(out["peptide"]) == {"P_EXACT"}

    # Strict + per-peptide attribution: drop the loose pool case.
    out = load_observations(mhc_allele_provenance=["exact", "peptide_attribution"])
    assert set(out["peptide"]) == {"P_EXACT", "P_ATTR"}


def test_load_observations_emits_severity_tiers(tmp_path, monkeypatch):
    """v1.30.17 / #201: ``mhc_class_label_severity`` returns one of
    {ok, borderline, suspect, implausible} per row.

    Class I tiers:
      8-12 → ok          (canonical)
      13-14 → borderline (bulged class-I)
      15-17 → suspect    (very unusual but documented)
      ≥18 → implausible  (curation drift)
      ≤7 → implausible

    Class II tiers:
      11-30 → ok         (canonical)
      8-10 → borderline  (genuinely short class-II ligands)
      5-7 → suspect
      ≤4 or ≥45 → implausible (v1.30.22: was ≥31; v1.30.28: was ≥40)
    """
    import pandas as pd

    from hitlist.observations import load_observations

    df = pd.DataFrame(
        {
            "peptide": [
                "AAAAAAAAA",  # 9aa class I — ok
                "AAAAAAAAAAAAA",  # 13aa class I — borderline
                "AAAAAAAAAAAAAAA",  # 15aa class I — suspect
                "AAAAAAAAAAAAAAAAAA",  # 18aa class I — implausible
                "EEEEEEEEEEEEE",  # 13aa class II — ok
                "EEEEEEEEE",  # 9aa class II — borderline
                "EEEEEEE",  # 7aa class II — suspect
                "EEEE",  # 4aa class II — implausible
                # v1.30.22 (#209): a 35aa class-II peptide is now "ok"
                # (was "implausible" before). Stražar 2023 contains
                # ~13K legitimate class-II ligands in the 31-39aa range.
                "E" * 35,
                # v1.30.28: 40aa class-II is now "ok" (was implausible
                # in v1.30.22-v1.30.27). Stražar's published tail
                # extends to ~51aa, so the cutoff moved up to 45.
                "E" * 40,
                # v1.30.28: 45aa class-II is the new implausibility
                # boundary.
                "E" * 45,
            ],
            "mhc_restriction": (["HLA-A*02:01"] * 4) + (["HLA-DRB1*15:01"] * 7),
            "mhc_class": (["I"] * 4) + (["II"] * 7),
            "reference_iri": [f"r-{i}" for i in range(11)],
            "pmid": pd.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], dtype="Int64"),
            "source": ["iedb"] * 11,
            "mhc_species": ["Homo sapiens"] * 11,
        }
    )
    path = tmp_path / "observations.parquet"
    df.to_parquet(path, index=False)
    monkeypatch.setattr("hitlist.observations.observations_path", lambda: path)

    out = load_observations()
    sev = dict(zip(out["peptide"], out["mhc_class_label_severity"]))
    # Class I.
    assert sev["AAAAAAAAA"] == "ok"
    assert sev["AAAAAAAAAAAAA"] == "borderline"
    assert sev["AAAAAAAAAAAAAAA"] == "suspect"
    assert sev["AAAAAAAAAAAAAAAAAA"] == "implausible"
    # Class II.
    assert sev["EEEEEEEEEEEEE"] == "ok"
    assert sev["EEEEEEEEE"] == "borderline"
    assert sev["EEEEEEE"] == "suspect"
    assert sev["EEEE"] == "implausible"
    # v1.30.22 / v1.30.28: extended class-II ligands.
    assert sev["E" * 35] == "ok"  # 35aa — was implausible pre-v1.30.22
    assert sev["E" * 40] == "ok"  # 40aa — was implausible v1.30.22-v1.30.27
    assert sev["E" * 45] == "implausible"  # ≥45aa is the v1.30.28 cutoff

    # Backwards-compat: suspect+implausible → mhc_class_label_suspect=True;
    # ok+borderline → False.
    susp = dict(zip(out["peptide"], out["mhc_class_label_suspect"]))
    assert susp["AAAAAAAAA"] is False  # ok
    assert susp["AAAAAAAAAAAAA"] is False  # borderline (NOT suspect)
    assert susp["AAAAAAAAAAAAAAA"] is True  # suspect
    assert susp["AAAAAAAAAAAAAAAAAA"] is True  # implausible
    assert susp["EEEEEEEEE"] is False  # borderline (NOT suspect)
    assert susp["EEEE"] is True  # implausible
    assert susp["E" * 35] is False  # ok per v1.30.22
    assert susp["E" * 40] is False  # ok per v1.30.28 (was True in v1.30.22-27)
    assert susp["E" * 45] is True  # implausible per v1.30.28


def test_exclude_class_label_implausible_keeps_borderline_and_suspect(tmp_path, monkeypatch):
    """v1.30.17: stricter variant of exclude_class_label_suspect that
    keeps borderline and suspect rows but drops implausible. Useful
    for analyses that want to retain bulged class-I peptides
    (15-17 aa) while still filtering out clear curation drift."""
    import pandas as pd

    from hitlist.observations import load_observations

    df = pd.DataFrame(
        {
            "peptide": [
                "AAAAAAAAA",  # 9aa class I — ok, kept
                "AAAAAAAAAAAAA",  # 13aa class I — borderline, kept
                "AAAAAAAAAAAAAAA",  # 15aa class I — suspect, kept
                "AAAAAAAAAAAAAAAAAA",  # 18aa class I — implausible, dropped
            ],
            "mhc_restriction": ["HLA-A*02:01"] * 4,
            "mhc_class": ["I"] * 4,
            "reference_iri": [f"r-{i}" for i in range(4)],
            "pmid": pd.array([1, 2, 3, 4], dtype="Int64"),
            "source": ["iedb"] * 4,
            "mhc_species": ["Homo sapiens"] * 4,
        }
    )
    path = tmp_path / "observations.parquet"
    df.to_parquet(path, index=False)
    monkeypatch.setattr("hitlist.observations.observations_path", lambda: path)

    # Strict (legacy): drops both suspect and implausible.
    strict = load_observations(exclude_class_label_suspect=True)
    assert set(strict["peptide"]) == {"AAAAAAAAA", "AAAAAAAAAAAAA"}

    # Implausible-only: keeps suspect, drops only implausible.
    relaxed = load_observations(exclude_class_label_implausible=True)
    assert set(relaxed["peptide"]) == {
        "AAAAAAAAA",
        "AAAAAAAAAAAAA",
        "AAAAAAAAAAAAAAA",
    }


def test_severity_tier_strips_ptm_annotation_when_computing_length(tmp_path, monkeypatch):
    """v1.30.20: pre-v1.30.10 parquets may carry IEDB's inline PTM
    annotation in the peptide column (e.g. ``"CGPSGLVREL + METH(C1)"``
    instead of the bare ``"CGPSGLVREL"``). The severity tier
    classifier must compute length on the bare AA sequence — otherwise
    the 6-char ``" + METH(C1)"`` suffix pushes a 10-aa class-I
    peptide past the 18-aa implausibility cutoff. Misclassified ~36k
    Sarkizova 2020 rows in production parquets."""
    import pandas as pd

    from hitlist.observations import load_observations

    df = pd.DataFrame(
        {
            "peptide": [
                # bare 10 aa class-I — should be ok, NOT implausible
                "CGPSGLVREL + METH(C1)",
                # bare 11 aa class-I + deamidation — should be ok
                "AIDHNQMFQYK + DEAM(N4)",
                # bare 9 aa, no PTM — should be ok
                "AAAAAAAAA",
                # bare 18 aa, no PTM — genuinely implausible class I
                "AAAAAAAAAAAAAAAAAA",
                # malformed PTM annotation: " + " present but parse fails;
                # bare-len fallback should still split on " + " and
                # measure the 9 aa prefix as ok.
                "AAAAAAAAA + ???",
            ],
            "mhc_restriction": ["HLA-A*02:01"] * 5,
            "mhc_class": ["I"] * 5,
            "reference_iri": [f"r-{i}" for i in range(5)],
            "pmid": pd.array([1, 2, 3, 4, 5], dtype="Int64"),
            "source": ["iedb"] * 5,
            "mhc_species": ["Homo sapiens"] * 5,
        }
    )
    path = tmp_path / "observations.parquet"
    df.to_parquet(path, index=False)
    monkeypatch.setattr("hitlist.observations.observations_path", lambda: path)

    out = load_observations()
    sev = dict(zip(out["peptide"], out["mhc_class_label_severity"]))
    assert sev["CGPSGLVREL + METH(C1)"] == "ok"
    assert sev["AIDHNQMFQYK + DEAM(N4)"] == "ok"
    assert sev["AAAAAAAAA"] == "ok"
    assert sev["AAAAAAAAAAAAAAAAAA"] == "implausible"
    assert sev["AAAAAAAAA + ???"] == "ok"


def _write_minimal_observations_parquet(path):
    import pandas as pd

    pd.DataFrame(
        {
            "peptide": ["P1"],
            "mhc_restriction": ["HLA-A*02:01"],
            "mhc_class": ["I"],
            "mhc_allele_set": ["HLA-A*02:01"],
            "reference_iri": ["r-1"],
            "pmid": pd.array([1], dtype="Int64"),
            "source": ["iedb"],
            "mhc_species": ["Homo sapiens"],
        }
    ).to_parquet(path, index=False)


def test_mhc_restriction_filter_empty_string_raises(tmp_path, monkeypatch):
    """Empty ``mhc_restriction`` filter values surface a ``ValueError``
    instead of silently returning an empty frame — guards against
    callers passing an unset variable by accident."""
    from hitlist.observations import load_observations

    path = tmp_path / "observations.parquet"
    _write_minimal_observations_parquet(path)
    monkeypatch.setattr("hitlist.observations.observations_path", lambda: path)

    with pytest.raises(ValueError, match="mhc_restriction filter received no usable"):
        load_observations(mhc_restriction="")
    with pytest.raises(ValueError, match="mhc_restriction filter received no usable"):
        load_observations(mhc_restriction=[""])


def test_mhc_allele_in_set_filter_empty_string_raises(tmp_path, monkeypatch):
    """Empty ``mhc_allele_in_set`` filter values surface a ``ValueError``
    same as ``mhc_restriction``."""
    from hitlist.observations import load_observations

    path = tmp_path / "observations.parquet"
    _write_minimal_observations_parquet(path)
    monkeypatch.setattr("hitlist.observations.observations_path", lambda: path)

    with pytest.raises(ValueError, match="mhc_allele_in_set filter received no usable"):
        load_observations(mhc_allele_in_set="")
    with pytest.raises(ValueError, match="mhc_allele_in_set filter received no usable"):
        load_observations(mhc_allele_in_set=[""])
