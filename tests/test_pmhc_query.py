# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for hitlist.pmhc_query — protein x allele MS evidence lookup."""

from __future__ import annotations

import pandas as pd


def _write_obs_fixture(tmp_path):
    """Tiny observations.parquet + peptide_mappings.parquet fixture covering:
    - one peptide that multi-maps to NRAS+KRAS (semicolon-joined gene_names)
    - one peptide unique to NRAS
    - one peptide on a different allele
    - one peptide on a different gene (BRAF)

    Post-#238, gene_names/gene_ids are no longer stored on the obs parquet
    by build_observations — pmhc_query resolves gene→peptide via the
    mappings sidecar, so the fixture must write both files.
    """
    df = pd.DataFrame(
        {
            "peptide": ["KLVVVGAGGV", "KLVVVGAGGV", "ILDTAGREEY", "ALAVLGFFV", "FLPNKQRTV"],
            "pmid": [9263005, 36099883, 33858848, 38480730, 33298915],
            "mhc_class": ["I"] * 5,
            "mhc_restriction": [
                "HLA-A*02:01",
                "HLA-A*02:01",
                "HLA-A*02:01",
                "HLA-B*07:02",
                "HLA-A*02:01",
            ],
            "species": ["Homo sapiens"] * 5,
            "mhc_species": ["Homo sapiens"] * 5,
            "attributed_sample_label": [
                "sample_a",
                "sample_b",  # second KLVVVGAGGV row → different sample
                "sample_a",
                "sample_c",
                "sample_a",
            ],
            "cell_line_name": [""] * 5,
            "monoallelic_host": [""] * 5,
            "source": ["iedb"] * 5,
        }
    )
    obs_path = tmp_path / "observations.parquet"
    df.to_parquet(obs_path, index=False)

    # Sidecar peptide_mappings.parquet — long-form (one row per peptide x
    # protein).  Schema must include the columns load_peptide_mappings
    # uses for filtering and the agg in annotate_observations_with_genes.
    mappings = pd.DataFrame(
        {
            "peptide": [
                "KLVVVGAGGV",
                "KLVVVGAGGV",
                "ILDTAGREEY",
                "ALAVLGFFV",
                "FLPNKQRTV",
            ],
            "gene_name": ["NRAS;KRAS", "NRAS;KRAS", "NRAS", "NRAS", "BRAF"],
            "gene_id": [
                "ENSG00000213281;ENSG00000133703",
                "ENSG00000213281;ENSG00000133703",
                "ENSG00000213281",
                "ENSG00000213281",
                "ENSG00000157764",
            ],
            "protein_id": [
                "ENSP00000358548;ENSP00000308495",
                "ENSP00000358548;ENSP00000308495",
                "ENSP00000358548",
                "ENSP00000358548",
                "ENSP00000288602",
            ],
        }
    )
    # Explode the multi-mapping rows so the sidecar is one-row-per-protein
    # (matches the shape produced by build_peptide_mappings).
    expanded = []
    for _, row in mappings.iterrows():
        for gene_name, gene_id, protein_id in zip(
            row["gene_name"].split(";"),
            row["gene_id"].split(";"),
            row["protein_id"].split(";"),
        ):
            expanded.append(
                {
                    "peptide": row["peptide"],
                    "gene_name": gene_name,
                    "gene_id": gene_id,
                    "protein_id": protein_id,
                }
            )
    mappings_long = pd.DataFrame(expanded)
    mappings_path = tmp_path / "peptide_mappings.parquet"
    mappings_long.to_parquet(mappings_path, index=False)
    return obs_path, mappings_path


def _patch_paths(monkeypatch, obs_path, mappings_path):
    """Helper: monkeypatch both observations_path and mappings_path so
    pmhc_query's gene→peptide resolution finds the test fixture's
    sidecar instead of ~/.hitlist/peptide_mappings.parquet."""
    monkeypatch.setattr("hitlist.observations.observations_path", lambda: obs_path)
    monkeypatch.setattr("hitlist.mappings.mappings_path", lambda: mappings_path)


def test_pmhc_query_groups_by_gene_and_allele(tmp_path, monkeypatch):
    """Single protein, two alleles: rows group by (gene, allele) and
    aggregate observation counts + PMID lists per peptide."""
    from hitlist import pmhc_query

    obs_path, mappings_path = _write_obs_fixture(tmp_path)
    _patch_paths(monkeypatch, obs_path, mappings_path)

    df = pmhc_query.query(proteins=["NRAS"], alleles=["HLA-A*02:01", "HLA-B*07:02"], use_hgnc=False)
    # NRAS rows only: KLVVVGAGGV (x2 obs), ILDTAGREEY, and ALAVLGFFV (the
    # one on B*07:02).  KLVVVGAGGV's two obs come from different PMIDs.
    assert set(df["gene_name"]) == {"NRAS"}
    klvv = df[df["peptide"] == "KLVVVGAGGV"].iloc[0]
    assert klvv["n_observations"] == 2
    assert "9263005" in klvv["pmids"] and "36099883" in klvv["pmids"]
    # Allele grouping preserved.
    assert set(df["mhc_allele"]) == {"HLA-A*02:01", "HLA-B*07:02"}
    # Per-allele evidence sort: most-attested peptide first within an allele.
    a02 = df[df["mhc_allele"] == "HLA-A*02:01"].reset_index(drop=True)
    assert a02.iloc[0]["peptide"] == "KLVVVGAGGV"  # n=2 wins over n=1


def test_pmhc_query_multi_mapping_peptide_appears_under_each_gene(tmp_path, monkeypatch):
    """KLVVVGAGGV multi-maps to NRAS+KRAS.  Asking for both genes returns
    the peptide once per gene, not just once."""
    from hitlist import pmhc_query

    obs_path, mappings_path = _write_obs_fixture(tmp_path)
    _patch_paths(monkeypatch, obs_path, mappings_path)

    df = pmhc_query.query(proteins=["NRAS", "KRAS"], alleles=["HLA-A*02:01"], use_hgnc=False)
    # KLVVVGAGGV should appear under both NRAS and KRAS rows.
    klvv = df[df["peptide"] == "KLVVVGAGGV"]
    assert set(klvv["gene_name"]) == {"NRAS", "KRAS"}


def test_pmhc_query_filters_to_requested_proteins(tmp_path, monkeypatch):
    """Asking for only NRAS must NOT return KRAS rows even though
    KLVVVGAGGV multi-maps to both."""
    from hitlist import pmhc_query

    obs_path, mappings_path = _write_obs_fixture(tmp_path)
    _patch_paths(monkeypatch, obs_path, mappings_path)

    df = pmhc_query.query(proteins=["NRAS"], alleles=["HLA-A*02:01"], use_hgnc=False)
    assert set(df["gene_name"]) == {"NRAS"}
    assert "KRAS" not in df["gene_name"].values


def test_pmhc_query_empty_when_no_match(tmp_path, monkeypatch):
    """A protein with no MS evidence returns an empty (but well-shaped)
    DataFrame."""
    from hitlist import pmhc_query

    obs_path, mappings_path = _write_obs_fixture(tmp_path)
    _patch_paths(monkeypatch, obs_path, mappings_path)

    df = pmhc_query.query(proteins=["TP53"], alleles=["HLA-A*02:01"], use_hgnc=False)
    assert df.empty
    # Schema is preserved so consumers can still read columns.
    for col in ("gene_name", "gene_id", "mhc_allele", "peptide", "n_observations", "pmids"):
        assert col in df.columns


def test_pmhc_query_no_proteins_returns_all_genes(tmp_path, monkeypatch):
    """v1.29.6: ``proteins=None`` (or empty) is no longer an error — it
    means "all genes". Filtering by allele alone returns every gene that
    presents on that allele."""
    from hitlist import pmhc_query

    obs_path, mappings_path = _write_obs_fixture(tmp_path)
    _patch_paths(monkeypatch, obs_path, mappings_path)

    df = pmhc_query.query(alleles=["HLA-A*02:01"], use_hgnc=False)
    # The fixture has KLVVVGAGGV (NRAS+KRAS), ILDTAGREEY (NRAS), FLPNKQRTV (BRAF) on A*02:01.
    assert set(df["gene_name"]) == {"NRAS", "KRAS", "BRAF"}
    assert set(df["mhc_allele"]) == {"HLA-A*02:01"}


def test_pmhc_query_no_alleles_returns_all_alleles(tmp_path, monkeypatch):
    """v1.29.6: ``alleles=None`` (or empty) is no longer an error — it
    means "all alleles". Filtering by protein alone returns every allele
    that presents the requested gene's peptides."""
    from hitlist import pmhc_query

    obs_path, mappings_path = _write_obs_fixture(tmp_path)
    _patch_paths(monkeypatch, obs_path, mappings_path)

    df = pmhc_query.query(proteins=["NRAS"], use_hgnc=False)
    assert set(df["gene_name"]) == {"NRAS"}
    # NRAS peptides span both A*02:01 and B*07:02 in the fixture.
    assert set(df["mhc_allele"]) == {"HLA-A*02:01", "HLA-B*07:02"}


def test_pmhc_query_no_filters_returns_everything(tmp_path, monkeypatch):
    """v1.29.6: ``query()`` with no filters returns the full corpus
    aggregated to (gene, allele, peptide). Useful as a "show me all
    pMHC evidence" command."""
    from hitlist import pmhc_query

    obs_path, mappings_path = _write_obs_fixture(tmp_path)
    _patch_paths(monkeypatch, obs_path, mappings_path)

    df = pmhc_query.query(use_hgnc=False)
    # All 3 genes from the fixture, both alleles.
    assert set(df["gene_name"]) == {"NRAS", "KRAS", "BRAF"}
    assert set(df["mhc_allele"]) == {"HLA-A*02:01", "HLA-B*07:02"}


def test_classify_by_affinity_tiers():
    """v1.30.1: affinity tiers are strong ≤100nM / medium ≤500nM /
    weak ≤2000nM / non > 2000nM."""
    from hitlist.pmhc_query import _classify_by_affinity

    assert _classify_by_affinity(50) == "strong"
    assert _classify_by_affinity(100) == "strong"
    assert _classify_by_affinity(101) == "medium"
    assert _classify_by_affinity(500) == "medium"
    assert _classify_by_affinity(501) == "weak"
    assert _classify_by_affinity(2000) == "weak"
    assert _classify_by_affinity(2001) == "non-binder"
    assert _classify_by_affinity(None) is None


def test_classify_by_percentile_tiers():
    """v1.30.1: percentile tiers are strong ≤0.5% / medium ≤1% /
    weak ≤2% / non > 2%."""
    from hitlist.pmhc_query import _classify_by_percentile

    assert _classify_by_percentile(0.5) == "strong"
    assert _classify_by_percentile(0.6) == "medium"
    assert _classify_by_percentile(1.0) == "medium"
    assert _classify_by_percentile(1.01) == "weak"
    assert _classify_by_percentile(2.0) == "weak"
    assert _classify_by_percentile(2.01) == "non-binder"
    assert _classify_by_percentile(None) is None


def test_classify_binder_takes_strongest_tier():
    """v1.30.1: ``_classify_binder`` returns the strongest call across
    affinity and percentile signals — predictors disagree more about
    absolute IC50 than about per-allele rank, so a "strong by percentile"
    peptide should not be downgraded by a weak-IC50 prediction."""
    from hitlist.pmhc_query import _classify_binder

    # Both signals agree
    assert _classify_binder(50, 0.4) == "strong"
    assert _classify_binder(3000, 5.0) == "non-binder"
    # Disagreement → take the stronger
    assert _classify_binder(800, 0.4) == "strong"  # weak aff, strong pct
    assert _classify_binder(50, 5.0) == "strong"  # strong aff, non pct
    assert _classify_binder(800, 1.5) == "weak"
    # Missing signal is ignored
    assert _classify_binder(50, None) == "strong"
    assert _classify_binder(None, 0.4) == "strong"
    # Both missing → empty
    assert _classify_binder(None, None) == ""


def test_pmhc_query_normalizes_unprefixed_alleles(tmp_path, monkeypatch):
    """v1.29.8: ``A*02:01`` and ``HLA-A*02:01`` are the same allele in
    different sources — the parquet stores both forms because curators
    aren't consistent. Normalize before grouping so peptides don't get
    split across two unrelated buckets."""
    from hitlist import pmhc_query

    df = pd.DataFrame(
        {
            "peptide": ["SLLQHLIGL", "SLLQHLIGL"],
            "pmid": [38480730, 22424782],
            "mhc_class": ["I", "I"],
            "mhc_restriction": ["A*02:01", "HLA-A*02:01"],
            "species": ["Homo sapiens"] * 2,
            "mhc_species": ["Homo sapiens"] * 2,
            "attributed_sample_label": ["sample_a", "sample_b"],
            "cell_line_name": ["", ""],
            "monoallelic_host": ["", ""],
            "source": ["iedb"] * 2,
        }
    )
    obs_path = tmp_path / "observations.parquet"
    df.to_parquet(obs_path, index=False)
    # Sidecar peptide_mappings.parquet — needed post-#238 for pmhc_query
    # to resolve the gene → peptide filter.
    mappings = pd.DataFrame(
        {
            "peptide": ["SLLQHLIGL"],
            "gene_name": ["PRAME"],
            "gene_id": ["ENSG00000185686"],
            "protein_id": ["ENSP00000312790"],
        }
    )
    mappings_path = tmp_path / "peptide_mappings.parquet"
    mappings.to_parquet(mappings_path, index=False)
    _patch_paths(monkeypatch, obs_path, mappings_path)

    result = pmhc_query.query(proteins=["PRAME"], use_hgnc=False)
    # Both rows should collapse to a single canonical HLA-A*02:01 bucket.
    assert set(result["mhc_allele"]) == {"HLA-A*02:01"}
    row = result.iloc[0]
    assert row["n_observations"] == 2
    assert "38480730" in row["pmids"] and "22424782" in row["pmids"]


def test_pmhc_query_format_table_sorts_alleles_by_evidence_count(tmp_path, monkeypatch):
    """v1.29.8: alleles within a gene are ordered by total evidence count
    descending — the most-attested allele appears first, not the
    alphabetically-first one."""
    from hitlist import pmhc_query

    obs_path, mappings_path = _write_obs_fixture(tmp_path)
    _patch_paths(monkeypatch, obs_path, mappings_path)

    df = pmhc_query.query(proteins=["NRAS"], use_hgnc=False)
    text = pmhc_query.format_table(df)
    # NRAS has 3 obs on A*02:01 and 1 obs on B*07:02 — A*02:01 should
    # appear first.
    a02_idx = text.find("HLA-A*02:01")
    b07_idx = text.find("HLA-B*07:02")
    assert 0 < a02_idx < b07_idx


def test_pmhc_query_format_table_prints_headers_once_per_gene(tmp_path, monkeypatch):
    """v1.29.8: column headers are printed once per gene, not per allele.
    A two-allele gene should have exactly one ``peptide  n_obs  pmids``
    header line, not two."""
    from hitlist import pmhc_query

    obs_path, mappings_path = _write_obs_fixture(tmp_path)
    _patch_paths(monkeypatch, obs_path, mappings_path)

    df = pmhc_query.query(proteins=["NRAS"], use_hgnc=False)
    text = pmhc_query.format_table(df)
    # The header column word "n_obs" should appear exactly once per gene.
    assert text.count("n_obs") == 1


def test_pmhc_query_format_table_appends_predictor_tip_when_unscored(tmp_path, monkeypatch):
    """v1.29.8: when no ``--predictor`` was passed, the table footer
    points users at ``--predictor netmhcpan`` so they don't have to
    rediscover the option from --help."""
    from hitlist import pmhc_query

    obs_path, mappings_path = _write_obs_fixture(tmp_path)
    _patch_paths(monkeypatch, obs_path, mappings_path)

    df = pmhc_query.query(proteins=["NRAS"], use_hgnc=False)
    text = pmhc_query.format_table(df)
    assert "--predictor netmhcpan" in text


def test_pmhc_query_format_table_renders_grouped_table(tmp_path, monkeypatch):
    """The table formatter renders protein > allele as section headers with
    peptide rows in a column-aligned table beneath each allele.  The shape
    is "table grouped under section headers", not a flat CSV nor a
    tree-drawing-character tree.
    """
    from hitlist import pmhc_query

    obs_path, mappings_path = _write_obs_fixture(tmp_path)
    _patch_paths(monkeypatch, obs_path, mappings_path)

    df = pmhc_query.query(proteins=["NRAS"], alleles=["HLA-A*02:01", "HLA-B*07:02"], use_hgnc=False)
    text = pmhc_query.format_table(df)

    # Section headers present (gene with id, then nested allele).
    assert "NRAS (ENSG00000213281)" in text
    assert "HLA-A*02:01" in text
    assert "HLA-B*07:02" in text
    # Column header row + rule + data rows under each allele.
    assert "peptide" in text
    assert "n_obs" in text
    assert "pmids" in text
    assert "----" in text  # rule line under each header
    assert "KLVVVGAGGV" in text
    assert "9263005" in text

    # Order: gene → allele → peptide (top-down).
    nras_idx = text.find("NRAS")
    a02_idx = text.find("HLA-A*02:01")
    pep_idx = text.find("KLVVVGAGGV")
    assert nras_idx < a02_idx < pep_idx


# ── Species inference + multi-species output sectioning (#256) ──────


def test_pmhc_query_attaches_mhc_species_column(tmp_path, monkeypatch):
    """The result frame carries an mhc_species column derived from each
    row's mhc_allele.  HLA rows resolve to ``Homo sapiens``."""
    from hitlist import pmhc_query

    obs_path, mappings_path = _write_obs_fixture(tmp_path)
    _patch_paths(monkeypatch, obs_path, mappings_path)

    df = pmhc_query.query(proteins=["NRAS"], use_hgnc=False)
    assert "mhc_species" in df.columns
    assert set(df["mhc_species"].unique()) == {"Homo sapiens"}


def test_pmhc_query_uses_obs_mhc_species_column_not_reparses_allele(tmp_path, monkeypatch):
    """The mhc_species column comes from the authoritative obs.parquet
    column (populated by the scanner at build time), NOT from
    re-parsing mhc_restriction at query time.  This catches the
    failure mode where someone re-introduces an _infer_species_column
    helper that would drift from the canonical scanner classification.
    """
    from hitlist import pmhc_query

    obs_path, mappings_path = _write_obs_fixture(tmp_path)
    _patch_paths(monkeypatch, obs_path, mappings_path)

    # Reach into curation to verify we DON'T call it during query() —
    # the obs column is the source of truth.
    n_classify_calls = {"n": 0}
    import hitlist.curation as curation

    real_classify = curation.classify_mhc_species

    def counting_classify(*a, **kw):
        n_classify_calls["n"] += 1
        return real_classify(*a, **kw)

    monkeypatch.setattr("hitlist.curation.classify_mhc_species", counting_classify)

    df = pmhc_query.query(proteins=["NRAS"], use_hgnc=False)
    assert "mhc_species" in df.columns
    assert set(df["mhc_species"].unique()) == {"Homo sapiens"}
    # Zero re-classification calls during query — the obs column already
    # has the answer.  Non-zero means a regression to per-row reparsing.
    assert n_classify_calls["n"] == 0


def test_pmhc_query_warns_on_unresolved_source_organism(tmp_path, monkeypatch, capsys):
    """Verbose mode emits a WARNING line when rows have unresolved
    source organism (``species`` is empty or literal "unidentified" in
    IEDB metadata).  Both upstream sentinels fold into the same
    "unknown" bucket in the normalized result.  Goal is curation —
    these rows should ideally go to zero.
    """
    import pandas as pd

    from hitlist import pmhc_query

    obs = pd.DataFrame(
        {
            "peptide": ["KLVVVGAGGV", "KLVVVGAGGV", "KLVVVGAGGV"],
            "pmid": [1, 2, 3],
            "mhc_class": ["I"] * 3,
            "mhc_restriction": ["HLA-A*02:01"] * 3,
            "species": ["Homo sapiens", "", "unidentified"],
            "mhc_species": ["Homo sapiens"] * 3,
            "attributed_sample_label": ["sample_a", "sample_b", "sample_c"],
            "cell_line_name": [""] * 3,
            "monoallelic_host": [""] * 3,
            "source": ["iedb"] * 3,
        }
    )
    mappings = pd.DataFrame(
        {
            "peptide": ["KLVVVGAGGV"],
            "gene_name": ["NRAS"],
            "gene_id": ["ENSG00000213281"],
            "protein_id": ["ENSP00000358548"],
        }
    )
    obs_path = tmp_path / "observations.parquet"
    obs.to_parquet(obs_path, index=False)
    mappings_path = tmp_path / "peptide_mappings.parquet"
    mappings.to_parquet(mappings_path, index=False)
    _patch_paths(monkeypatch, obs_path, mappings_path)

    pmhc_query.query(proteins=["NRAS"], use_hgnc=False, verbose=True)
    err = capsys.readouterr().err
    assert "WARNING" in err
    # Both the "" and "unidentified" rows fold into "unknown" → 2.
    assert "2 row(s) have unresolved source organism" in err


def test_normalize_species_column_folds_empty_and_unidentified_to_unknown():
    """Direct contract test on the normalization helper — pins the
    exact mapping ``"" / NaN / "unidentified" → "unknown"`` without
    going through the query path (where the column gets dropped at
    aggregation and we can only verify normalization indirectly).
    """
    import numpy as np
    import pandas as pd

    from hitlist.pmhc_query import _normalize_species_column

    s = pd.Series(["Homo sapiens", "", np.nan, "unidentified", "Mus musculus"])
    out = _normalize_species_column(s)
    assert list(out) == [
        "Homo sapiens",
        "unknown",
        "unknown",
        "unknown",
        "Mus musculus",
    ]


def test_normalize_species_column_handles_categorical_dtype():
    """Regression: obs.parquet columns come back as ``Categorical`` from
    pyarrow's dictionary-encoded read.  ``Series.replace({...})`` on a
    Categorical without all target values pre-declared as categories
    can raise / warn on newer pandas.  The helper's ``.astype(str)``
    cast must move us off Categorical before the replace.
    """
    import pandas as pd

    from hitlist.pmhc_query import _normalize_species_column

    cat = pd.Series(
        ["Homo sapiens", "unidentified", "", "Mus musculus"],
        dtype="category",
    )
    out = _normalize_species_column(cat)
    # All upstream sentinels folded to "unknown".
    assert sorted(out.unique()) == ["Homo sapiens", "Mus musculus", "unknown"]
    # Result is NOT Categorical — downstream consumers won't run into
    # category-constraint failures on subsequent replace / merge calls.
    # (Could be object or StringDtype depending on pandas version.)
    assert not isinstance(out.dtype, pd.CategoricalDtype)


def test_species_sort_key_order_human_first():
    """Output ordering: human leads, mouse / rat as standard models, then
    everything else alphabetical, ``unknown`` sinks to the bottom."""
    from hitlist.pmhc_query import _species_sort_key

    species = ["Macaca mulatta", "Mus musculus", "unknown", "Homo sapiens", "Canis lupus"]
    sorted_species = sorted(species, key=_species_sort_key)
    assert sorted_species[0] == "Homo sapiens"
    assert sorted_species[1] == "Mus musculus"
    assert sorted_species[-1] == "unknown"
    # Canis (C) sorts before Macaca (M) in the alphabetical tail.
    assert sorted_species.index("Canis lupus") < sorted_species.index("Macaca mulatta")


# ── Output formatting cleanup (#256 review) ──────────────────────────


def test_format_table_truncates_long_pmid_lists():
    """PMID lists with >3 references render as ``a;b;c; +N more`` so the
    table doesn't get visually dominated by a 50-char-wide pmids column.
    Full list is still on the underlying frame for CSV/JSON consumers."""
    from hitlist import pmhc_query

    df = pd.DataFrame(
        {
            "gene_name": ["NRAS"],
            "gene_id": ["ENSG00000213281"],
            "mhc_allele": ["HLA-A*02:01"],
            "best_guess_allele": [None],
            "peptide": ["KLVVVGAGGV"],
            "n_observations": [6],
            "n_references": [6],
            "n_samples": [6],
            "pmids": ["111;222;333;444;555;666"],
            "mhc_class": ["I"],
            "mhc_species": ["Homo sapiens"],
        }
    )
    text = pmhc_query.format_table(df)
    assert "111;222;333; +3 more" in text
    # First three PMIDs are shown; the 4th is not in the rendered table.
    assert "444" not in text


def test_format_table_short_pmid_lists_not_truncated():
    """PMID lists with <= 3 entries pass through verbatim."""
    from hitlist import pmhc_query

    df = pd.DataFrame(
        {
            "gene_name": ["NRAS"],
            "gene_id": ["ENSG00000213281"],
            "mhc_allele": ["HLA-A*02:01"],
            "best_guess_allele": [None],
            "peptide": ["KLVVVGAGGV"],
            "n_observations": [3],
            "n_references": [3],
            "n_samples": [3],
            "pmids": ["111;222;333"],
            "mhc_class": ["I"],
            "mhc_species": ["Homo sapiens"],
        }
    )
    text = pmhc_query.format_table(df)
    assert "111;222;333" in text
    assert "+0 more" not in text  # don't tag short lists


def test_format_table_renders_empty_allele_as_synthetic_header():
    """A row with empty ``mhc_allele`` (IEDB didn't record a specific
    allele) used to render under a phantom blank-line header, looking
    like a layout bug.  Now it gets a clear "(allele not specified)"
    label."""
    from hitlist import pmhc_query

    df = pd.DataFrame(
        {
            "gene_name": ["NRAS", "NRAS"],
            "gene_id": ["ENSG00000213281", "ENSG00000213281"],
            "mhc_allele": ["HLA-A*02:01", ""],
            "best_guess_allele": [None, None],
            "peptide": ["KLVVVGAGGV", "PLACEHOLDER"],
            "n_observations": [3, 1],
            "n_references": [3, 1],
            "n_samples": [3, 1],
            "pmids": ["111", "222"],
            "mhc_class": ["I", "I"],
            "mhc_species": ["Homo sapiens", "Homo sapiens"],
        }
    )
    text = pmhc_query.format_table(df)
    assert "(allele not specified)" in text
    # The placeholder peptide still renders — we labeled the section,
    # didn't drop the row.
    assert "PLACEHOLDER" in text


def test_format_table_orders_specific_alleles_before_class_only():
    """4-digit alleles (containing ``*``) appear before class-only
    placeholders like ``HLA class I``, so the high-evidence specific
    sections lead and the catch-all lands at the bottom."""
    from hitlist import pmhc_query

    df = pd.DataFrame(
        {
            "gene_name": ["NRAS"] * 3,
            "gene_id": ["ENSG00000213281"] * 3,
            "mhc_allele": ["HLA class I", "HLA-A*02:01", ""],
            "best_guess_allele": [None] * 3,
            "peptide": ["PEPTIDE_A", "PEPTIDE_B", "PEPTIDE_C"],
            # Class-only and empty rows have higher n_observations so we
            # know specifc-allele ordering isn't accidentally happening
            # via the existing evidence-count sort.
            "n_observations": [99, 1, 50],
            "n_references": [1, 1, 1],
            "n_samples": [1, 1, 1],
            "pmids": ["111"] * 3,
            "mhc_class": ["I"] * 3,
            "mhc_species": ["Homo sapiens"] * 3,
        }
    )
    text = pmhc_query.format_table(df)
    a02_idx = text.find("HLA-A*02:01")
    class_i_idx = text.find("HLA class I")
    unspec_idx = text.find("(allele not specified)")
    assert 0 < a02_idx < class_i_idx
    assert 0 < a02_idx < unspec_idx


def test_format_table_strips_trailing_whitespace_per_row():
    """No rendered line ends with trailing spaces — keeps the right edge
    clean without affecting inter-column alignment."""
    from hitlist import pmhc_query

    df = pd.DataFrame(
        {
            "gene_name": ["NRAS"],
            "gene_id": ["ENSG00000213281"],
            "mhc_allele": ["HLA-A*02:01"],
            "best_guess_allele": [None],
            "peptide": ["KLVVVGAGGV"],
            "n_observations": [3],
            "n_references": [2],
            "n_samples": [3],
            "pmids": ["111;222"],
            "mhc_class": ["I"],
            "mhc_species": ["Homo sapiens"],
        }
    )
    text = pmhc_query.format_table(df)
    for line in text.splitlines():
        assert line == line.rstrip(), f"trailing whitespace in line: {line!r}"


def _multispecies_table_input() -> pd.DataFrame:
    """Hand-rolled multi-species result frame for format_table tests.
    Avoids the parquet fixture so the test pins the formatter behavior
    in isolation from the query path.
    """
    return pd.DataFrame(
        {
            "gene_name": ["PRAME", "PRAME", "PRAME"],
            "gene_id": ["ENSG00000185686", "ENSG00000185686", "ENSG00000185686"],
            "mhc_allele": ["HLA-A*02:01", "DLA-88*501:01", "HLA-A*02:01"],
            "best_guess_allele": [None, None, None],
            "peptide": ["SLLQHLIGL", "YIAQFTSQFL", "ALYVDSLFFL"],
            "n_observations": [42, 1, 7],
            "n_references": [1, 1, 1],
            "n_samples": [42, 1, 7],
            "pmids": ["12345", "27893789", "34567"],
            "mhc_class": ["I", "I", "I"],
            "mhc_species": ["Homo sapiens", "Canis lupus", "Homo sapiens"],
        }
    )


def test_format_table_inserts_species_section_when_multi_species():
    """Multi-species result gets ``=== species: X ===`` outer headers,
    human-first by sort order."""
    from hitlist import pmhc_query

    text = pmhc_query.format_table(_multispecies_table_input())
    assert "=== species: Homo sapiens ===" in text
    assert "=== species: Canis lupus ===" in text
    # Human section comes before canine.
    assert text.find("Homo sapiens") < text.find("Canis lupus")
    # The dog peptide still appears, just under its own section.
    assert "YIAQFTSQFL" in text
    assert "DLA-88*501:01" in text


def test_format_table_omits_species_section_when_single_species(tmp_path, monkeypatch):
    """Single-species result (the typical human-only case) keeps the
    pre-#256 layout — no ``=== species:`` header line."""
    from hitlist import pmhc_query

    obs_path, mappings_path = _write_obs_fixture(tmp_path)
    _patch_paths(monkeypatch, obs_path, mappings_path)

    df = pmhc_query.query(proteins=["NRAS"], use_hgnc=False)
    text = pmhc_query.format_table(df)
    assert "=== species:" not in text
    # The existing gene > allele structure is unchanged.
    assert "NRAS" in text
    assert "HLA-A*02:01" in text


def test_mhc_species_survives_predictor_path(tmp_path, monkeypatch):
    """Regression: mhc_species was previously dropped by
    ``_consolidate_after_narrowing`` (whose group_cols don't include it)
    when the user passed ``--predictor``.  The fix re-derives the column
    AFTER ``_attach_predictions`` so it survives consolidation AND
    reflects the post-narrowing single-allele string.
    """
    from hitlist import pmhc_query

    # Stub the actual predictor so we don't shell out to NetMHCpan in tests.
    def fake_predict(pairs):
        return pairs.assign(affinity_nM=100.0, presentation_percentile=0.5)

    monkeypatch.setattr("hitlist.predict._predict_netmhcpan", fake_predict)

    obs_path, mappings_path = _write_obs_fixture(tmp_path)
    _patch_paths(monkeypatch, obs_path, mappings_path)

    df = pmhc_query.query(proteins=["NRAS"], predictor="netmhcpan", use_hgnc=False)
    assert "mhc_species" in df.columns
    assert df["mhc_species"].notna().all()
    assert set(df["mhc_species"].unique()) == {"Homo sapiens"}
    # Predictor columns also present — confirms _attach_predictions
    # actually ran (vs. a stub fall-through).
    assert "affinity_nM" in df.columns
    assert "presentation_percentile" in df.columns


# ── Serotype handling (v1.30.2) ───────────────────────────────────────


def _write_serotype_obs_fixture(tmp_path):
    """Fixture mixing 4-digit and serotype-resolution rows for the same gene.

    Writes both observations.parquet and the peptide_mappings.parquet
    sidecar (post-#238 the obs parquet no longer carries gene columns;
    pmhc_query resolves gene→peptide via the mappings sidecar).
    """
    df = pd.DataFrame(
        {
            "peptide": ["KLVVVGAGGV", "ILDTAGREEY", "FLPNKQRTV"],
            "pmid": [9263005, 33858848, 33298915],
            "mhc_class": ["I", "I", "I"],
            "mhc_restriction": ["HLA-A*02:01", "HLA-A*02:02", "HLA-A2"],
            "species": ["Homo sapiens"] * 3,
            "mhc_species": ["Homo sapiens"] * 3,
            "attributed_sample_label": ["sample_a", "sample_b", "sample_c"],
            "cell_line_name": [""] * 3,
            "monoallelic_host": [""] * 3,
            "source": ["iedb"] * 3,
        }
    )
    obs_path = tmp_path / "observations.parquet"
    df.to_parquet(obs_path, index=False)

    mappings = pd.DataFrame(
        {
            "peptide": ["KLVVVGAGGV", "ILDTAGREEY", "FLPNKQRTV"],
            "gene_name": ["NRAS"] * 3,
            "gene_id": ["ENSG00000213281"] * 3,
            "protein_id": ["ENSP00000358548"] * 3,
        }
    )
    mappings_path = tmp_path / "peptide_mappings.parquet"
    mappings.to_parquet(mappings_path, index=False)
    return obs_path, mappings_path


def test_pmhc_query_input_serotype_expands_to_4digit_members(tmp_path, monkeypatch):
    """v1.30.2: querying ``--allele HLA-A2`` returns evidence on A*02:01,
    A*02:02, AND literal HLA-A2 rows. Without expansion the parquet
    pushdown would only match the literal "HLA-A2" string."""
    from hitlist import pmhc_query

    obs_path, mappings_path = _write_serotype_obs_fixture(tmp_path)
    _patch_paths(monkeypatch, obs_path, mappings_path)

    df = pmhc_query.query(proteins=["NRAS"], alleles=["HLA-A2"], use_hgnc=False)
    # All three rows from the fixture are A2-related and must come back.
    assert set(df["mhc_allele"]) == {"HLA-A*02:01", "HLA-A*02:02", "HLA-A2"}


def test_pmhc_query_best_guess_allele_for_serotype_rows(tmp_path, monkeypatch):
    """v1.30.2: rows whose stored allele is a serotype get a
    ``best_guess_allele`` column filled with the most-likely 4-digit
    member. Already-4-digit rows preserve their value."""
    from hitlist import pmhc_query

    obs_path, mappings_path = _write_serotype_obs_fixture(tmp_path)
    _patch_paths(monkeypatch, obs_path, mappings_path)

    df = pmhc_query.query(proteins=["NRAS"], alleles=["HLA-A2"], use_hgnc=False)
    by_allele = df.set_index("mhc_allele")["best_guess_allele"].to_dict()
    # 4-digit rows: best_guess_allele equals the stored allele.
    assert by_allele["HLA-A*02:01"] == "HLA-A*02:01"
    assert by_allele["HLA-A*02:02"] == "HLA-A*02:02"
    # Serotype row gets the most-likely 4-digit guess.
    assert by_allele["HLA-A2"] == "HLA-A*02:01"


def test_pmhc_query_format_table_shows_best_guess_for_serotype(tmp_path, monkeypatch):
    """v1.30.2: the grouped-table allele header annotates serotype
    rows with the best 4-digit guess so the user sees what the predictor
    will (or would) score under."""
    from hitlist import pmhc_query

    obs_path, mappings_path = _write_serotype_obs_fixture(tmp_path)
    _patch_paths(monkeypatch, obs_path, mappings_path)

    df = pmhc_query.query(proteins=["NRAS"], alleles=["HLA-A2"], use_hgnc=False)
    text = pmhc_query.format_table(df)
    # The HLA-A2 section gets a "(best guess: HLA-A*02:01)" annotation;
    # the 4-digit sections do not.
    assert "HLA-A2  (best guess: HLA-A*02:01)" in text
    # 4-digit allele headers are unannotated.
    assert "HLA-A*02:01\n" in text or "HLA-A*02:01 " in text  # header line bare


# ── Per-sample paired query (v1.30.5) ─────────────────────────────────


def test_query_by_samples_returns_per_sample_rows(tmp_path, monkeypatch):
    """v1.30.5: ``query_by_samples`` calls ``query`` once per sample with
    that sample's allele list and tags every output row with
    ``sample_name``. Each sample sees only its own alleles, not the union
    (which is the cross-product behavior of plain ``--mhc-allele``)."""
    from hitlist import pmhc_query

    obs_path, mappings_path = _write_obs_fixture(tmp_path)
    _patch_paths(monkeypatch, obs_path, mappings_path)

    df = pmhc_query.query_by_samples(
        samples_to_alleles={
            "patient1": ["HLA-A*02:01"],
            "patient2": ["HLA-B*07:02"],
        },
        proteins=["NRAS"],
        use_hgnc=False,
    )
    assert "sample_name" in df.columns
    # Each sample only sees the allele it asked for.
    p1 = df[df["sample_name"] == "patient1"]
    p2 = df[df["sample_name"] == "patient2"]
    assert set(p1["mhc_allele"]) == {"HLA-A*02:01"}
    assert set(p2["mhc_allele"]) == {"HLA-B*07:02"}
    # Both samples are present in the output.
    assert set(df["sample_name"]) == {"patient1", "patient2"}


def test_query_by_samples_empty_input_returns_empty_with_schema():
    """v1.30.5: empty ``samples_to_alleles`` returns an empty DataFrame
    that still carries the expected columns (incl. ``sample_name``) so
    downstream consumers can iterate the groupby without exploding."""
    from hitlist import pmhc_query

    df = pmhc_query.query_by_samples(samples_to_alleles={})
    assert df.empty
    assert "sample_name" in df.columns
    assert "mhc_allele" in df.columns


def test_query_by_samples_empty_per_sample_allele_list_raises(tmp_path, monkeypatch):
    """v1.30.5 / #188: a per-sample empty allele list would silently
    fan out to "all alleles" inside ``query``, which is not the
    paired-API contract. Raise ``ValueError`` so the caller fixes it."""
    import pytest

    from hitlist import pmhc_query

    obs_path, mappings_path = _write_obs_fixture(tmp_path)
    _patch_paths(monkeypatch, obs_path, mappings_path)

    with pytest.raises(ValueError, match="empty allele lists"):
        pmhc_query.query_by_samples(
            samples_to_alleles={"p1": ["HLA-A*02:01"], "p2": []},
            proteins=["NRAS"],
            use_hgnc=False,
        )


def test_format_table_groups_by_sample_when_sample_name_present(tmp_path, monkeypatch):
    """v1.30.5: when ``sample_name`` is present in the input, the grouped
    text output emits one outer ``=== sample: NAME ===`` section per
    sample, with the existing gene/allele/peptide structure nested
    inside."""
    from hitlist import pmhc_query

    obs_path, mappings_path = _write_obs_fixture(tmp_path)
    _patch_paths(monkeypatch, obs_path, mappings_path)

    df = pmhc_query.query_by_samples(
        samples_to_alleles={
            "patient1": ["HLA-A*02:01"],
            "patient2": ["HLA-B*07:02"],
        },
        proteins=["NRAS"],
        use_hgnc=False,
    )
    text = pmhc_query.format_table(df)
    # Outer per-sample sections in stable (alphabetical) order.
    p1_idx = text.find("=== sample: patient1 ===")
    p2_idx = text.find("=== sample: patient2 ===")
    assert 0 <= p1_idx < p2_idx
    # The gene block under each sample is indented (nested).
    # Each sample's allele appears inside its own section.
    p1_block = text[p1_idx:p2_idx]
    p2_block = text[p2_idx:]
    assert "HLA-A*02:01" in p1_block
    assert "HLA-A*02:01" not in p2_block  # patient2 didn't ask for A*02:01


# ── Predictor-driven multi-allele narrowing (issue #239) ──────────────────


def _make_grouped(rows: list[dict]) -> pd.DataFrame:
    """Build the post-aggregation frame _attach_predictions consumes."""
    base_cols = {
        "gene_name": "PRAME",
        "gene_id": "ENSG00000185686",
        "mhc_class": "I",
        "n_observations": 1,
        "pmids": "31844290",
    }
    return pd.DataFrame([{**base_cols, **r} for r in rows])


def test_attach_predictions_narrows_multi_allele_to_best_binder(monkeypatch):
    """Issue #239: a multi-allele row gets expanded to per-allele
    predictions; the lowest-percentile allele wins; mhc_allele +
    best_guess_allele are narrowed; best_predicted_allele records the
    choice.
    """
    from hitlist import pmhc_query

    df = _make_grouped(
        [
            {
                "peptide": "SLLQHLIGL",
                "mhc_allele": "HLA-A*02:01;HLA-A*03:01;HLA-B*27:05",
                "best_guess_allele": "HLA-A*02:01;HLA-A*03:01;HLA-B*27:05",
            }
        ]
    )

    def fake_predict(pairs: pd.DataFrame) -> pd.DataFrame:
        # A*02:01 wins (lowest percentile); A*03:01 weak; B*27:05 non-binder.
        scores = {
            "HLA-A*02:01": (12.0, 0.05),
            "HLA-A*03:01": (1500.0, 1.8),
            "HLA-B*27:05": (8000.0, 5.5),
        }
        out = pairs.copy()
        out["affinity_nM"] = [scores[a][0] for a in out["allele"]]
        out["presentation_percentile"] = [scores[a][1] for a in out["allele"]]
        return out

    monkeypatch.setattr("hitlist.predict._predict_mhcflurry", fake_predict)

    out = pmhc_query._attach_predictions(df, "mhcflurry")
    assert len(out) == 1
    r = out.iloc[0]
    assert r["mhc_allele"] == "HLA-A*02:01"
    assert r["best_guess_allele"] == "HLA-A*02:01"
    assert r["best_predicted_allele"] == "HLA-A*02:01"
    assert r["affinity_nM"] == 12.0
    assert r["presentation_percentile"] == 0.05
    assert r["binder_class"] == "strong"


def test_attach_predictions_consolidates_per_donor_rows_after_narrowing(monkeypatch):
    """Issue #239 + #236 interaction: three per-donor rows for SLLQHLIGL
    (MEL3 / MEL15 / OV1, each with their own 6-allele typing) all
    contain A*02:01.  After predictor narrowing, all three collapse to
    a single ``HLA-A*02:01`` row with summed n_observations and the
    union of pmids — the user sees one consolidated allele-resolved row
    instead of three redundant per-donor rows pointing at the same
    allele.
    """
    from hitlist import pmhc_query

    rows = [
        {
            "peptide": "SLLQHLIGL",
            "mhc_allele": "HLA-A*02:01;HLA-A*03:01;HLA-B*27:05;HLA-B*47:01;HLA-C*01:02;HLA-C*06:02",
            "best_guess_allele": "HLA-A*02:01;HLA-A*03:01;HLA-B*27:05;HLA-B*47:01;HLA-C*01:02;HLA-C*06:02",
            "n_observations": 2,
            "pmids": "31844290",
        },
        {
            "peptide": "SLLQHLIGL",
            "mhc_allele": "HLA-A*02:01;HLA-A*02:02;HLA-B*13:02;HLA-B*40:02;HLA-C*02:02;HLA-C*06:02",
            "best_guess_allele": "HLA-A*02:01;HLA-A*02:02;HLA-B*13:02;HLA-B*40:02;HLA-C*02:02;HLA-C*06:02",
            "n_observations": 2,
            "pmids": "31844290",
        },
        {
            "peptide": "SLLQHLIGL",
            "mhc_allele": "HLA-A*02:01;HLA-A*24:02;HLA-B*35:03;HLA-B*44:02;HLA-C*05:01;HLA-C*12:03",
            "best_guess_allele": "HLA-A*02:01;HLA-A*24:02;HLA-B*35:03;HLA-B*44:02;HLA-C*05:01;HLA-C*12:03",
            "n_observations": 2,
            "pmids": "31844290",
        },
    ]
    df = _make_grouped(rows)

    def fake_predict(pairs: pd.DataFrame) -> pd.DataFrame:
        # Only A*02:01 binds well; everything else is a non-binder.
        out = pairs.copy()
        out["affinity_nM"] = [12.0 if a == "HLA-A*02:01" else 8000.0 for a in out["allele"]]
        out["presentation_percentile"] = [
            0.05 if a == "HLA-A*02:01" else 5.5 for a in out["allele"]
        ]
        return out

    monkeypatch.setattr("hitlist.predict._predict_mhcflurry", fake_predict)
    out = pmhc_query._attach_predictions(df, "mhcflurry")

    assert len(out) == 1
    r = out.iloc[0]
    assert r["mhc_allele"] == "HLA-A*02:01"
    assert r["best_predicted_allele"] == "HLA-A*02:01"
    assert r["n_observations"] == 6  # 2 + 2 + 2
    assert r["pmids"] == "31844290"  # union of identical PMIDs


def test_attach_predictions_keeps_single_allele_rows_unchanged(monkeypatch):
    """Single-allele rows pass through with score columns added but no
    structural change — mhc_allele isn't rewritten, no row collapse."""
    from hitlist import pmhc_query

    df = _make_grouped(
        [
            {
                "peptide": "GILGFVFTL",
                "mhc_allele": "HLA-A*02:01",
                "best_guess_allele": "HLA-A*02:01",
            }
        ]
    )

    def fake_predict(pairs: pd.DataFrame) -> pd.DataFrame:
        out = pairs.copy()
        out["affinity_nM"] = [10.0]
        out["presentation_percentile"] = [0.02]
        return out

    monkeypatch.setattr("hitlist.predict._predict_mhcflurry", fake_predict)
    out = pmhc_query._attach_predictions(df, "mhcflurry")

    assert len(out) == 1
    r = out.iloc[0]
    assert r["mhc_allele"] == "HLA-A*02:01"
    assert r["best_predicted_allele"] == "HLA-A*02:01"
    assert r["affinity_nM"] == 10.0
    assert r["binder_class"] == "strong"


def test_attach_predictions_keeps_multi_allele_when_no_predictions(monkeypatch):
    """If every allele in the set returns NaN (predictor failure / length
    mismatch / unknown allele), the multi-allele mhc_allele is preserved
    and best_predicted_allele is empty.  Avoids wrongly committing to a
    non-binding allele just because it sorted first."""
    from hitlist import pmhc_query

    multi = "HLA-A*02:01;HLA-A*03:01;HLA-B*27:05"
    df = _make_grouped(
        [{"peptide": "WEIRDPEPTIDE", "mhc_allele": multi, "best_guess_allele": multi}]
    )

    def fake_predict(pairs: pd.DataFrame) -> pd.DataFrame:
        out = pairs.copy()
        out["affinity_nM"] = pd.NA
        out["presentation_percentile"] = pd.NA
        return out

    monkeypatch.setattr("hitlist.predict._predict_mhcflurry", fake_predict)
    out = pmhc_query._attach_predictions(df, "mhcflurry")

    assert len(out) == 1
    r = out.iloc[0]
    assert r["mhc_allele"] == multi  # unchanged
    assert r["best_predicted_allele"] == ""
    assert pd.isna(r["affinity_nM"])
    assert pd.isna(r["presentation_percentile"])


def test_query_by_samples_empty_sample_section_has_placeholder(tmp_path, monkeypatch):
    """v1.30.5: a sample whose alleles match nothing in the corpus still
    appears in the output, with a one-line ``(no pMHC evidence ...)``
    placeholder so the user can see which samples returned nothing."""
    from hitlist import pmhc_query

    obs_path, mappings_path = _write_obs_fixture(tmp_path)
    _patch_paths(monkeypatch, obs_path, mappings_path)

    df = pmhc_query.query_by_samples(
        samples_to_alleles={
            "real": ["HLA-A*02:01"],  # matches NRAS rows
            "miss": ["HLA-C*07:02"],  # no rows in fixture
        },
        proteins=["NRAS"],
        use_hgnc=False,
    )
    assert set(df["sample_name"]) == {"real", "miss"}  # both present
    text = pmhc_query.format_table(df)
    # The empty sample gets a placeholder line.
    miss_idx = text.find("=== sample: miss ===")
    assert miss_idx >= 0
    miss_block = text[miss_idx:]
    assert "(no pMHC evidence on this sample's alleles)" in miss_block


# ── #259: filter flags + n_samples / n_references columns ──────────────


def test_query_attaches_n_references_and_n_samples_columns(tmp_path, monkeypatch):
    """Result frame always has n_references (distinct PMIDs) and
    n_samples (distinct attributed_sample_label) columns."""
    from hitlist import pmhc_query

    obs_path, mappings_path = _write_obs_fixture(tmp_path)
    _patch_paths(monkeypatch, obs_path, mappings_path)

    df = pmhc_query.query(proteins=["NRAS"], use_hgnc=False)
    assert "n_references" in df.columns
    assert "n_samples" in df.columns
    klvv = df[df["peptide"] == "KLVVVGAGGV"].iloc[0]
    # KLVVVGAGGV has 2 obs from 2 PMIDs in 2 samples (sample_a, sample_b).
    assert int(klvv["n_observations"]) == 2
    assert int(klvv["n_references"]) == 2
    assert int(klvv["n_samples"]) == 2


def test_query_n_samples_distinct_from_n_references(tmp_path, monkeypatch):
    """A single PMID with multiple samples → n_references=1 but n_samples=N.
    This is the whole reason n_samples exists separately."""
    from hitlist import pmhc_query

    obs = pd.DataFrame(
        {
            "peptide": ["KLVVVGAGGV"] * 3,
            "pmid": [100, 100, 100],  # same PMID
            "mhc_class": ["I"] * 3,
            "mhc_restriction": ["HLA-A*02:01"] * 3,
            "species": ["Homo sapiens"] * 3,
            "mhc_species": ["Homo sapiens"] * 3,
            # Three distinct donors in one cohort paper.
            "attributed_sample_label": ["donor1", "donor2", "donor3"],
            "cell_line_name": [""] * 3,
            "monoallelic_host": [""] * 3,
            "source": ["iedb"] * 3,
        }
    )
    mappings = pd.DataFrame(
        {
            "peptide": ["KLVVVGAGGV"],
            "gene_name": ["NRAS"],
            "gene_id": ["ENSG00000213281"],
            "protein_id": ["ENSP00000358548"],
        }
    )
    obs.to_parquet(tmp_path / "observations.parquet", index=False)
    mappings.to_parquet(tmp_path / "peptide_mappings.parquet", index=False)
    _patch_paths(
        monkeypatch, tmp_path / "observations.parquet", tmp_path / "peptide_mappings.parquet"
    )

    df = pmhc_query.query(proteins=["NRAS"], use_hgnc=False)
    row = df.iloc[0]
    assert int(row["n_observations"]) == 3
    assert int(row["n_references"]) == 1  # single PMID
    assert int(row["n_samples"]) == 3  # three distinct samples


def test_query_min_references_filter_drops_singletons(tmp_path, monkeypatch):
    """min_references=2 drops singleton-PMID rows."""
    from hitlist import pmhc_query

    obs_path, mappings_path = _write_obs_fixture(tmp_path)
    _patch_paths(monkeypatch, obs_path, mappings_path)

    df_all = pmhc_query.query(proteins=["NRAS"], use_hgnc=False)
    df_filtered = pmhc_query.query(proteins=["NRAS"], min_references=2, use_hgnc=False)
    # KLVVVGAGGV has 2 PMIDs and survives; others have 1 and are dropped.
    assert set(df_filtered["peptide"]) == {"KLVVVGAGGV"}
    assert len(df_filtered) < len(df_all)


def test_query_min_samples_filter_drops_singletons(tmp_path, monkeypatch):
    """min_samples=2 drops rows backed by only one sample."""
    from hitlist import pmhc_query

    obs_path, mappings_path = _write_obs_fixture(tmp_path)
    _patch_paths(monkeypatch, obs_path, mappings_path)

    df = pmhc_query.query(proteins=["NRAS"], min_samples=2, use_hgnc=False)
    # Only KLVVVGAGGV has 2 distinct samples in the fixture.
    assert set(df["peptide"]) == {"KLVVVGAGGV"}


def test_query_n_samples_uses_composite_of_cell_line_and_host(tmp_path, monkeypatch):
    """Regression for the #260 audit: a single PMID with multiple
    distinct ``cell_line_name`` values must count as MULTIPLE samples — not 1.
    attributed_sample_label is empty on 98.8% of IEDB rows; the
    sample-id synthesis composes pmid + cell_line_name +
    monoallelic_host + attributed_sample_label so n_samples reflects
    the per-cell-line granularity that 58% of the corpus carries.

    Note: uses cell_line_name (not cell_name) because cell_name mixes
    real lines with coarse cell-type categories like "B cell" /
    "Other" — including those would over-split.  cell_line_name is
    gated on src_cell_line=True at build time so it's clean."""
    from hitlist import pmhc_query

    # 4 rows from ONE pmid, four DIFFERENT cell lines, NO attributed_sample_label.
    # Pre-composite logic would call this 1 sample (the PMID).
    # Composite synthesis correctly identifies 4 distinct (pmid, cell_line_name) pairs.
    obs = pd.DataFrame(
        {
            "peptide": ["KLVVVGAGGV"] * 4,
            "pmid": [42] * 4,
            "mhc_class": ["I"] * 4,
            "mhc_restriction": ["HLA-A*02:01"] * 4,
            "species": ["Homo sapiens"] * 4,
            "mhc_species": ["Homo sapiens"] * 4,
            "attributed_sample_label": [""] * 4,
            "cell_line_name": ["MEL2", "MEL3", "MEL5", "OV1"],
            "monoallelic_host": [""] * 4,
            "source": ["iedb"] * 4,
        }
    )
    mappings = pd.DataFrame(
        {
            "peptide": ["KLVVVGAGGV"],
            "gene_name": ["NRAS"],
            "gene_id": ["ENSG00000213281"],
            "protein_id": ["ENSP00000358548"],
        }
    )
    obs.to_parquet(tmp_path / "observations.parquet", index=False)
    mappings.to_parquet(tmp_path / "peptide_mappings.parquet", index=False)
    _patch_paths(
        monkeypatch,
        tmp_path / "observations.parquet",
        tmp_path / "peptide_mappings.parquet",
    )

    df = pmhc_query.query(proteins=["NRAS"], use_hgnc=False)
    row = df.iloc[0]
    assert int(row["n_references"]) == 1  # one PMID
    assert int(row["n_samples"]) == 4  # four distinct cell lines


def test_query_n_samples_ignores_cell_type_only_signal(tmp_path, monkeypatch):
    """``cell_name`` mixes real cell-line names with coarse cell-type
    categories ("B cell", "Glial cell").  The composite uses
    ``cell_line_name`` (gated on src_cell_line=True) instead so two
    rows differing ONLY in cell_name's cell-type field don't falsely
    inflate the sample count.

    Fixture has 3 rows on one PMID with empty cell_line_name (i.e.
    src_cell_line was False at build time, so these are primary-cell
    rows where cell_name would have been "B cell" / "Other" etc.).
    Without an attributed_sample_label they collapse to 1 sample,
    which is the right answer given the metadata can't actually
    distinguish them."""
    from hitlist import pmhc_query

    obs = pd.DataFrame(
        {
            "peptide": ["KLVVVGAGGV"] * 3,
            "pmid": [42] * 3,
            "mhc_class": ["I"] * 3,
            "mhc_restriction": ["HLA-A*02:01"] * 3,
            "species": ["Homo sapiens"] * 3,
            "mhc_species": ["Homo sapiens"] * 3,
            "attributed_sample_label": [""] * 3,
            "cell_line_name": [""] * 3,  # primary-cell, no real line
            "monoallelic_host": [""] * 3,
            "source": ["iedb"] * 3,
        }
    )
    mappings = pd.DataFrame(
        {
            "peptide": ["KLVVVGAGGV"],
            "gene_name": ["NRAS"],
            "gene_id": ["ENSG00000213281"],
            "protein_id": ["ENSP00000358548"],
        }
    )
    obs.to_parquet(tmp_path / "observations.parquet", index=False)
    mappings.to_parquet(tmp_path / "peptide_mappings.parquet", index=False)
    _patch_paths(
        monkeypatch,
        tmp_path / "observations.parquet",
        tmp_path / "peptide_mappings.parquet",
    )

    df = pmhc_query.query(proteins=["NRAS"], use_hgnc=False)
    row = df.iloc[0]
    # All three rows share the same (pmid, "", "", "") composite → 1 sample.
    # Conservative under-count: we can't tell them apart from metadata.
    assert int(row["n_samples"]) == 1


def test_query_n_samples_uses_monoallelic_host_when_cell_name_empty(tmp_path, monkeypatch):
    """monoallelic_host is the only sample-distinguishing field on
    ~9K corpus rows.  Verify it's pulled into the composite."""
    from hitlist import pmhc_query

    obs = pd.DataFrame(
        {
            "peptide": ["KLVVVGAGGV"] * 3,
            "pmid": [42, 42, 42],
            "mhc_class": ["I"] * 3,
            "mhc_restriction": ["HLA-A*02:01"] * 3,
            "species": ["Homo sapiens"] * 3,
            "mhc_species": ["Homo sapiens"] * 3,
            "attributed_sample_label": ["", "", ""],
            "cell_line_name": ["", "", ""],
            "monoallelic_host": ["host_A", "host_B", "host_C"],
            "source": ["iedb"] * 3,
        }
    )
    mappings = pd.DataFrame(
        {
            "peptide": ["KLVVVGAGGV"],
            "gene_name": ["NRAS"],
            "gene_id": ["ENSG00000213281"],
            "protein_id": ["ENSP00000358548"],
        }
    )
    obs.to_parquet(tmp_path / "observations.parquet", index=False)
    mappings.to_parquet(tmp_path / "peptide_mappings.parquet", index=False)
    _patch_paths(
        monkeypatch,
        tmp_path / "observations.parquet",
        tmp_path / "peptide_mappings.parquet",
    )

    df = pmhc_query.query(proteins=["NRAS"], use_hgnc=False)
    row = df.iloc[0]
    assert int(row["n_samples"]) == 3


def test_query_min_binder_class_filter_requires_predictor():
    """--min-binder-class without --predictor raises a clear error."""
    import pytest

    from hitlist import pmhc_query

    with pytest.raises(ValueError, match="requires --predictor"):
        pmhc_query.query(min_binder_class="strong")


def test_query_min_binder_class_rejects_unknown_tier():
    """Validation: only strong/medium/weak are accepted."""
    import pytest

    from hitlist import pmhc_query

    with pytest.raises(ValueError, match="must be one of"):
        pmhc_query.query(min_binder_class="bogus", predictor="netmhcpan")


def test_query_min_binder_class_filters_predictor_path(tmp_path, monkeypatch):
    """When predictor is set, min_binder_class drops rows below the tier.
    Stubs the predictor so we don't shell out."""
    from hitlist import pmhc_query

    obs_path, mappings_path = _write_obs_fixture(tmp_path)
    _patch_paths(monkeypatch, obs_path, mappings_path)

    # Stub predictor: alternate strong / non-binder by peptide so we
    # can verify the threshold filter actually drops rows.
    def fake_predict(pairs):
        out = pairs.copy()
        out["affinity_nM"] = [10.0 if i % 2 == 0 else 50000.0 for i in range(len(pairs))]
        out["presentation_percentile"] = [0.01 if i % 2 == 0 else 99.9 for i in range(len(pairs))]
        return out

    monkeypatch.setattr("hitlist.predict._predict_netmhcpan", fake_predict)

    df_all = pmhc_query.query(proteins=["NRAS"], predictor="netmhcpan", use_hgnc=False)
    df_strong = pmhc_query.query(
        proteins=["NRAS"], predictor="netmhcpan", min_binder_class="strong", use_hgnc=False
    )
    assert len(df_strong) < len(df_all)
    assert set(df_strong["binder_class"]) == {"strong"}
