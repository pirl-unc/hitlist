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

"""Per-protein x allele pMHC evidence lookup.

The ``hitlist pmhc query`` CLI command (and the underlying :func:`query`
function) answers the most common downstream question: *for these
proteins and these MHC alleles, what peptides has mass-spec
immunopeptidomics actually surfaced, and how strongly does the
predictor think each one binds?*

Returns one flat row per (gene, allele, peptide) with PMIDs and
affinity prediction.  The ``--format grouped`` text output renders the
same rows visually grouped: gene → allele → peptides sorted by
evidence count.
"""

from __future__ import annotations

import pandas as pd

from .genes import resolve_gene_query


def query(
    proteins: list[str],
    alleles: list[str],
    *,
    predictor: str | None = None,
    use_hgnc: bool = True,
) -> pd.DataFrame:
    """Find pMHC MS evidence for the cross-product of proteins x alleles.

    Parameters
    ----------
    proteins
        List of gene symbols, Ensembl gene IDs, or HGNC aliases. Each is
        resolved via :func:`hitlist.genes.resolve_gene_query`.
    alleles
        List of 4-digit MHC allele strings (``"HLA-A*02:01"``).  Filter
        is exact-match against ``mhc_restriction``.
    predictor
        ``"mhcflurry"``, ``"netmhcpan"``, or ``None`` (skip prediction).
        If set, attaches ``affinity_nM`` / ``presentation_percentile`` /
        ``binder_class`` columns per (peptide, allele) row.
    use_hgnc
        Pass through to ``resolve_gene_query`` — set False to disable
        the HGNC alias REST lookup (offline use).

    Returns
    -------
    pd.DataFrame
        Columns: ``gene_name``, ``gene_id``, ``mhc_allele``,
        ``peptide``, ``n_observations``, ``pmids``, ``mhc_class``.
        Plus the affinity columns when ``predictor`` is set.
        Sorted by (gene_name, mhc_allele, -n_observations).
        Empty DataFrame with these columns if nothing matched.
    """
    from .observations import is_built, load_observations

    if not is_built():
        raise FileNotFoundError(
            "observations.parquet has not been built. Run `hitlist build observations` first."
        )

    if not proteins or not alleles:
        return _empty_result(predictor is not None)

    # 1. Resolve every protein query to gene_name / gene_id sets.
    names: set[str] = set()
    ids: set[str] = set()
    for q in proteins:
        spec = resolve_gene_query(q, use_hgnc=use_hgnc)
        names |= spec["names"]
        ids |= spec["ids"]
    if not names and not ids:
        return _empty_result(predictor is not None)

    # 2. Load observations filtered by gene + allele list.  ``load_observations``
    #    accepts singular ``gene_name``/``gene_id`` filters for ergonomics, but
    #    the parquet itself stores semicolon-joined ``gene_names`` / ``gene_ids``
    #    columns (one peptide can map to multiple genes).
    # We post-filter on the parquet's ``gene_names`` / ``gene_ids`` columns
    # ourselves rather than passing ``gene_name=`` to ``load_observations``,
    # which would route through ``peptide_mappings.parquet`` and require it
    # to be built.  Allele-pushdown alone already cuts the corpus 10-50x for
    # typical (HLA-A*02:01, HLA-B*07:02) queries.
    df = load_observations(
        mhc_restriction=alleles,
        columns=[
            "peptide",
            "pmid",
            "mhc_class",
            "mhc_restriction",
            "gene_names",
            "gene_ids",
        ],
    )
    if df.empty:
        return _empty_result(predictor is not None)

    # 3. Explode the parallel ``gene_names`` / ``gene_ids`` lists into one
    #    row per (gene_name, gene_id) so we can group cleanly.  Pad the
    #    shorter list with empties so the pairs stay aligned.
    for col in ("gene_names", "gene_ids"):
        df[col] = df[col].fillna("").astype(str)
    df["_gene_name"] = df["gene_names"].str.split(";")
    df["_gene_id"] = df["gene_ids"].str.split(";")
    pad_lens = [max(len(a), len(b)) for a, b in zip(df["_gene_name"], df["_gene_id"])]
    df["_gene_name"] = [
        (lst + [""] * (n - len(lst)))[:n] for lst, n in zip(df["_gene_name"], pad_lens)
    ]
    df["_gene_id"] = [(lst + [""] * (n - len(lst)))[:n] for lst, n in zip(df["_gene_id"], pad_lens)]
    df = df.explode(["_gene_name", "_gene_id"]).reset_index(drop=True)
    df["gene_name"] = df["_gene_name"].astype(str).str.strip()
    df["gene_id"] = df["_gene_id"].astype(str).str.strip()
    df = df.drop(columns=["_gene_name", "_gene_id", "gene_names", "gene_ids"])
    # Filter to the genes the user actually asked about (multi-mapping rows
    # surface sibling genes during the explode).
    keep_mask = pd.Series(False, index=df.index)
    if names:
        keep_mask = keep_mask | df["gene_name"].isin(names)
    if ids:
        keep_mask = keep_mask | df["gene_id"].isin(ids)
    df = df[keep_mask].reset_index(drop=True)
    if df.empty:
        return _empty_result(predictor is not None)

    # 4. Aggregate to (gene_name, gene_id, mhc_restriction, peptide):
    #    n_observations = row count, pmids = sorted unique semicolon-joined.
    grouped = (
        df.groupby(
            ["gene_name", "gene_id", "mhc_restriction", "peptide", "mhc_class"],
            dropna=False,
        )
        .agg(
            n_observations=("pmid", "size"),
            pmids=(
                "pmid",
                lambda s: ";".join(str(int(p)) for p in sorted(set(s.dropna()))),
            ),
        )
        .reset_index()
        .rename(columns={"mhc_restriction": "mhc_allele"})
    )

    # 5. Optional binding-affinity prediction.
    if predictor is not None:
        grouped = _attach_predictions(grouped, predictor)

    # 6. Order: by gene then allele then evidence count desc.
    grouped = grouped.sort_values(
        ["gene_name", "mhc_allele", "n_observations"],
        ascending=[True, True, False],
        kind="stable",
    ).reset_index(drop=True)
    return grouped


def _attach_predictions(df: pd.DataFrame, predictor: str) -> pd.DataFrame:
    """Score each (peptide, mhc_allele) row with MHCflurry or NetMHCpan and
    append ``affinity_nM`` / ``presentation_percentile`` / ``binder_class``."""
    pairs = df[["peptide", "mhc_allele"]].rename(columns={"mhc_allele": "allele"})
    if predictor == "mhcflurry":
        from .predict import _predict_mhcflurry

        scored = _predict_mhcflurry(pairs.copy())
    elif predictor == "netmhcpan":
        from .predict import _predict_netmhcpan

        scored = _predict_netmhcpan(pairs.copy())
    else:
        raise ValueError(f"Unknown predictor {predictor!r}; use 'mhcflurry' or 'netmhcpan'")

    # NetMHCpan path returns its own DataFrame keyed on (peptide, allele);
    # MHCflurry preserves caller index.  Merge defensively.
    scored = scored.rename(columns={"allele": "mhc_allele"})
    df = df.merge(
        scored[["peptide", "mhc_allele", "affinity_nM", "presentation_percentile"]],
        on=["peptide", "mhc_allele"],
        how="left",
    )
    df["binder_class"] = df["affinity_nM"].apply(_classify_binder)
    return df


def _classify_binder(affinity_nM: float | None) -> str:
    """Standard community thresholds.

    - strong: <= 50 nM
    - weak:   <= 500 nM
    - non:    > 500 nM
    """
    if affinity_nM is None or pd.isna(affinity_nM):
        return ""
    if affinity_nM <= 50:
        return "strong"
    if affinity_nM <= 500:
        return "weak"
    return "non"


def _empty_result(with_predictions: bool) -> pd.DataFrame:
    cols = [
        "gene_name",
        "gene_id",
        "mhc_allele",
        "peptide",
        "n_observations",
        "pmids",
        "mhc_class",
    ]
    if with_predictions:
        cols += ["affinity_nM", "presentation_percentile", "binder_class"]
    return pd.DataFrame(columns=cols)


def format_table(df: pd.DataFrame) -> str:
    """Render a query result with protein > allele as section headers and
    peptide rows as an aligned table beneath each allele.

    Layout::

        GENE_NAME (GENE_ID)
          MHC_ALLELE
            peptide        n_obs  pmids               [affinity_nM  binder]
            -------------  -----  -----------------   -------------------
            PEPTIDE_SEQ        N  pmid1;pmid2          ...
            ...
          MHC_ALLELE
            ...

    Each peptide row is column-aligned so values line up vertically.
    Empty result yields a one-line "(no evidence)" message.
    """
    if df.empty:
        return "(no pMHC evidence for the requested proteins x alleles)"

    has_pred = "affinity_nM" in df.columns

    pep_columns: list[tuple[str, str]] = [
        ("peptide", "peptide"),
        ("n_obs", "n_observations"),
        ("pmids", "pmids"),
    ]
    if has_pred:
        pep_columns += [
            ("affinity_nM", "affinity_nM"),
            ("binder", "binder_class"),
        ]

    def _fmt(header: str, value) -> str:
        if pd.isna(value):
            return ""
        if header == "n_obs":
            return f"{int(value)}"
        if header == "affinity_nM":
            return f"{float(value):.1f}"
        return str(value)

    pep_headers = [h for h, _ in pep_columns]
    pep_keys = [k for _, k in pep_columns]

    # First pass: figure out per-column widths across the whole result so the
    # table columns align uniformly under every (gene, allele) section.
    widths = [len(h) for h in pep_headers]
    for _, row in df.iterrows():
        for i, key in enumerate(pep_keys):
            cell = _fmt(pep_headers[i], row[key])
            if len(cell) > widths[i]:
                widths[i] = len(cell)

    sep = "  "
    indent = "    "
    header_line = indent + sep.join(h.ljust(widths[i]) for i, h in enumerate(pep_headers))
    rule_line = indent + sep.join("-" * widths[i] for i in range(len(pep_headers)))

    out: list[str] = []
    for gene_name, gene_df in df.groupby("gene_name", sort=True):
        gene_id = gene_df["gene_id"].dropna().astype(str)
        gene_id = gene_id.iloc[0] if len(gene_id) else ""
        out.append(f"{gene_name} ({gene_id})" if gene_id else gene_name)
        for allele in sorted(gene_df["mhc_allele"].unique()):
            allele_df = gene_df[gene_df["mhc_allele"] == allele]
            out.append(f"  {allele}")
            out.append(header_line)
            out.append(rule_line)
            for _, row in allele_df.iterrows():
                cells = [_fmt(pep_headers[i], row[k]) for i, k in enumerate(pep_keys)]
                out.append(indent + sep.join(cells[i].ljust(widths[i]) for i in range(len(cells))))
        out.append("")
    return "\n".join(out).rstrip() + "\n"


# Backwards-compatible alias for the original name.  ``format_grouped``
# never shipped on PyPI; kept only for any in-tree imports.
format_grouped = format_table
