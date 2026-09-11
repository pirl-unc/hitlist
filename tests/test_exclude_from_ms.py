"""The curated ``exclude_from_ms`` flag, and the corpus it keeps honest.

``exclude_from_ms`` marks a study a curator read and concluded was not a
mass-spectrometry elution experiment — yeast display, peptide microarray,
computational prediction, refolding crystallography.  It was documented
and set on 11 studies while nothing read it, so 40,355 rows / 33,101
peptides sat in a corpus whose entire premise is MS-observed eluted
ligands, and reached every public consumer (#444).

The exclusion is scoped to MS evidence on purpose.  These papers measure
real peptide-MHC binding; their ``binding.parquet`` rows stay.  The flag
says "this is not an elution experiment", not "distrust this paper".
"""

from __future__ import annotations

import pandas as pd
import pytest

from hitlist import curation
from hitlist.builder import _OBSERVATIONS_ARTIFACT_VERSION, _drop_excluded_from_ms
from hitlist.curation import PMID_ENTRY_FIELDS, load_pmid_overrides, ms_excluded_pmids

# Wendorff 2020 — HLA-DP peptide microarray, 69,815 random 13-mers against
# recombinant HLA-DP.  Curated exclude_from_ms; 418,890 binding rows.
A_CURATED_NON_MS_PMID = 32903714


def test_exclude_from_ms_has_a_reader():
    """The regression this issue is really about.

    The flag was curated on 11 studies and consumed by nothing.  A
    declaration in PMID_ENTRY_FIELDS is not a reader, so assert the
    reader itself resolves the flag out of the packaged YAML.
    """
    excluded = ms_excluded_pmids()
    assert excluded, "exclude_from_ms resolves to no studies — the reader is dead"
    from_yaml = {
        pmid
        for pmid, entry in load_pmid_overrides().items()
        if entry.get("exclude_from_ms") is True
    }
    assert excluded == from_yaml
    assert A_CURATED_NON_MS_PMID in excluded


def test_exclude_from_ms_is_no_longer_declared_unread():
    """#444's other half: the description must not outlive the fix.

    ``donors`` is still genuinely unread and is still asserted as such
    in test_sample_attribution_audit.py — this pins only the key that
    gained a reader.
    """
    described = PMID_ENTRY_FIELDS["exclude_from_ms"]
    assert "UNREAD" not in described
    assert "ms_excluded_pmids" in described
    assert "#444" in described


def test_ms_excluded_pmids_rejects_a_non_boolean_value(tmp_path, monkeypatch):
    """A typo'd ``exclude_from_ms: 1`` must fail loudly, not silently leave the
    study in the MS corpus (#471). ``1 is True`` is ``False`` in Python, so an
    unvalidated ``is True`` check would treat this as "not excluded" with no
    error anywhere.
    """
    bad_yaml = tmp_path / "pmid_overrides.yaml"
    bad_yaml.write_text("- pmid: 12345678\n  study_label: bad flag\n  exclude_from_ms: 1\n")
    real_data_path = curation._data_path
    monkeypatch.setattr(
        curation,
        "_data_path",
        lambda fn: str(bad_yaml) if fn == "pmid_overrides.yaml" else real_data_path(fn),
    )
    curation.load_pmid_overrides.cache_clear()
    curation.ms_excluded_pmids.cache_clear()
    try:
        with pytest.raises(ValueError, match="12345678"):
            curation.ms_excluded_pmids()
    finally:
        curation.load_pmid_overrides.cache_clear()
        curation.ms_excluded_pmids.cache_clear()


def test_drop_excluded_from_ms_drops_only_the_curated_studies():
    df = pd.DataFrame(
        {
            "pmid": pd.array([A_CURATED_NON_MS_PMID, 25576301, None], dtype="Int64"),
            "peptide": ["MICROARRAY", "REALELUTED", "NOPMIDROWX"],
        }
    )
    out = _drop_excluded_from_ms(df, "MS observations")
    assert list(out["peptide"]) == ["REALELUTED", "NOPMIDROWX"]
    # A row with no PMID cannot be attributed to an excluded study and
    # must not be swept up by the mask.
    assert out["pmid"].isna().sum() == 1


def test_drop_excluded_from_ms_tolerates_an_empty_or_pmidless_frame():
    empty = pd.DataFrame({"pmid": pd.array([], dtype="Int64"), "peptide": []})
    assert _drop_excluded_from_ms(empty, "MS observations").empty
    pmidless = pd.DataFrame({"peptide": ["NOPMIDCOLS"]})
    assert len(_drop_excluded_from_ms(pmidless, "MS observations")) == 1


def _corpus_predates_the_fix() -> bool:
    """True when the built corpus was written before #444 landed.

    A build-time filter is a property of the artifact, so asserting it
    against a corpus built by the previous builder tests nothing.  The
    CI corpus is a published release asset that lags the code until it
    is republished; skip rather than fail against it.
    """
    from hitlist.builder import _cache_meta

    return _cache_meta().get("artifact_version") != _OBSERVATIONS_ARTIFACT_VERSION


@pytest.mark.integration
def test_no_excluded_study_reaches_the_enriched_export():
    """The regression #444 asks for, on the real corpus."""
    from hitlist.observations import is_built

    if not is_built():
        pytest.skip("Observations table not built")
    if _corpus_predates_the_fix():
        pytest.skip(
            f"corpus predates #444 (artifact_version "
            f"!= {_OBSERVATIONS_ARTIFACT_VERSION}); rebuild to check"
        )
    from hitlist.export import generate_observations_table

    df = generate_observations_table(columns=["pmid", "peptide"])
    leaked = df[df["pmid"].isin(ms_excluded_pmids())]
    assert leaked.empty, (
        f"{len(leaked):,} rows from exclude_from_ms studies "
        f"{sorted(leaked['pmid'].dropna().unique())} reached the enriched export"
    )


@pytest.mark.integration
def test_the_exclusion_does_not_reach_the_binding_index():
    """The scope constraint, pinned against the real binding corpus.

    These studies measure real peptide-MHC binding — Wendorff 2020 alone
    contributes 418,890 microarray rows.  Dropping them would delete
    evidence the exclusion never claimed was wrong.
    """
    from hitlist.observations import is_binding_built, load_binding

    if not is_binding_built():
        pytest.skip("Binding table not built")
    binding = load_binding(columns=["pmid", "peptide"])
    kept = binding[binding["pmid"].isin(ms_excluded_pmids())]
    assert not kept.empty, (
        "binding.parquet lost every exclude_from_ms row — the exclusion "
        "must be scoped to MS evidence (#444)"
    )


def test_the_excluded_studies_curate_no_ms_samples():
    """A study excluded from the MS index should claim no MS arms.

    ``ms_samples`` is a claim that an experimental arm was run on an
    instrument.  If one ever appears on an excluded study, the two
    curations contradict each other and a curator should reconcile them
    rather than let the exporter quietly emit an arm for a study whose
    observations are gone.
    """
    overrides = load_pmid_overrides()
    contradictory = {
        pmid: overrides[pmid].get("study_label")
        for pmid in ms_excluded_pmids()
        if overrides[pmid].get("ms_samples")
    }
    assert not contradictory, (
        f"studies are both exclude_from_ms and curated with ms_samples: {contradictory}"
    )


def test_ms_excluded_pmids_is_cleared_between_builds():
    """The builder reloads curation before a build; this cache must follow.

    Without registration in ``_clear_curation_caches`` a rebuild in the
    same process would honor the previous YAML's exclusions.
    """
    ms_excluded_pmids()
    curation._clear_curation_caches()
    assert ms_excluded_pmids.cache_info().currsize == 0
