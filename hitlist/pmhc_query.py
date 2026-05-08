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

The ``hitlist pmhc`` CLI command (and the underlying :func:`query`
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

import re
import sys
import time

import pandas as pd

from .genes import resolve_gene_query


def _progress(msg: str, verbose: bool) -> None:
    """Print a stderr progress hint when running interactively.

    The query can take 5-30s when no allele filter is given (full parquet
    load), so users get easily worried it's hung — see user reports
    against v1.29.6. Stderr lines are unobtrusive (don't pollute stdout
    pipes) but answer the "is it doing anything?" question.
    """
    if verbose:
        print(f"[pmhc] {msg}", file=sys.stderr, flush=True)


def query(
    proteins: list[str] | None = None,
    alleles: list[str] | None = None,
    *,
    predictor: str | None = None,
    use_hgnc: bool = True,
    verbose: bool = False,
) -> pd.DataFrame:
    """Find pMHC MS evidence, optionally filtered by proteins and/or alleles.

    Both filters are independent: pass neither to scan the whole corpus,
    just one to fix that axis, or both for the original cross-product
    behavior.

    Parameters
    ----------
    proteins
        List of gene symbols, Ensembl gene IDs, or HGNC aliases. Each is
        resolved via :func:`hitlist.genes.resolve_gene_query`. Pass
        ``None`` (or empty list) to scan all genes.
    alleles
        List of 4-digit MHC allele strings (``"HLA-A*02:01"``).  Filter
        is exact-match against ``mhc_restriction``. Pass ``None`` (or
        empty list) to scan all alleles.
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

    t_start = time.perf_counter()

    # 1. Resolve every protein query to gene_name / gene_id sets — only if
    #    the user asked for one. Empty / None means "all genes".
    names: set[str] = set()
    ids: set[str] = set()
    if proteins:
        _progress(
            f"resolving {len(proteins)} protein quer{'y' if len(proteins) == 1 else 'ies'}...",
            verbose,
        )
        for q in proteins:
            spec = resolve_gene_query(q, use_hgnc=use_hgnc)
            names |= spec["names"]
            ids |= spec["ids"]
        _progress(f"resolved to {len(names)} gene names + {len(ids)} gene IDs", verbose)
        if not names and not ids:
            return _empty_result(predictor is not None)

    # 2. Load observations. The allele filter pushes down to parquet only
    #    when given; otherwise we read the whole corpus. ``load_observations``
    #    accepts singular ``gene_name``/``gene_id`` filters for ergonomics,
    #    but the parquet stores semicolon-joined ``gene_names`` /
    #    ``gene_ids`` (one peptide can map to multiple genes), so we
    #    post-filter on those columns instead — that also avoids routing
    #    through ``peptide_mappings.parquet`` (which may not be built).
    load_kwargs: dict = {
        "columns": [
            "peptide",
            "pmid",
            "mhc_class",
            "mhc_restriction",
            "gene_names",
            "gene_ids",
        ],
    }
    if alleles:
        # Serotype inputs (e.g. "HLA-A2") are expanded to their 4-digit
        # members before pushdown so HLA-A*02:01 / A*02:02 / ... rows
        # show up alongside any literal "HLA-A2" rows.  Keep the original
        # serotype string in the filter — some sources store at serotype
        # resolution and we want both kinds of evidence.
        from .curation import serotype_to_alleles

        expanded: list[str] = []
        n_expanded_serotypes = 0
        for a in alleles:
            expanded.append(a)
            members = serotype_to_alleles(a)
            if members:
                expanded.extend(members)
                n_expanded_serotypes += 1
        # Dedup while preserving order — order isn't load-correctness, but
        # tidier in verbose progress.
        seen: set[str] = set()
        load_kwargs["mhc_restriction"] = [x for x in expanded if not (x in seen or seen.add(x))]
        if n_expanded_serotypes:
            _progress(
                f"loading observations.parquet (allele pushdown: "
                f"{len(load_kwargs['mhc_restriction'])} alleles after expanding "
                f"{n_expanded_serotypes} serotype{'s' if n_expanded_serotypes != 1 else ''})...",
                verbose,
            )
        else:
            _progress(
                f"loading observations.parquet (allele pushdown: "
                f"{len(load_kwargs['mhc_restriction'])} alleles)...",
                verbose,
            )
    else:
        _progress("loading observations.parquet (no allele filter, ~3-5s)...", verbose)
    df = load_observations(**load_kwargs)
    _progress(f"loaded {len(df):,} rows in {time.perf_counter() - t_start:.1f}s", verbose)
    if df.empty:
        return _empty_result(predictor is not None)

    # 3. Pre-filter to candidate rows BEFORE the explode so we don't
    #    pointlessly explode the entire 4.4M-row corpus when the user
    #    asked for a single gene. ``gene_names`` / ``gene_ids`` are
    #    semicolon-joined strings, so a substring match finds candidate
    #    rows; the explode + post-filter step still filters precisely.
    for col in ("gene_names", "gene_ids"):
        df[col] = df[col].fillna("").astype(str)
    if names or ids:
        _progress("narrowing to candidate rows for requested protein(s)...", verbose)
        candidate_mask = pd.Series(False, index=df.index)
        if names:
            name_pat = "|".join(re.escape(n) for n in names if n)
            if name_pat:
                candidate_mask = candidate_mask | df["gene_names"].str.contains(
                    name_pat, regex=True, na=False
                )
        if ids:
            id_pat = "|".join(re.escape(i) for i in ids if i)
            if id_pat:
                candidate_mask = candidate_mask | df["gene_ids"].str.contains(
                    id_pat, regex=True, na=False
                )
        df = df[candidate_mask]
        _progress(f"  {len(df):,} candidate rows", verbose)
        if df.empty:
            return _empty_result(predictor is not None)

    # 3b. Normalize MHC restriction strings before grouping. The parquet
    #     stores both ``A*02:01`` and ``HLA-A*02:01`` for the same allele
    #     because different sources used different conventions; passing
    #     the raw strings through to groupby would split the peptides
    #     across two unrelated buckets. ``normalize_allele`` is mhcgnomes-
    #     backed and idempotent on canonical inputs; the LRU cache keeps
    #     the per-row cost negligible (~hundreds of unique values).
    from .curation import best_4digit_for_serotype, normalize_allele

    df["mhc_restriction"] = (
        df["mhc_restriction"].fillna("").map(lambda s: normalize_allele(s) if s else s)
    )

    # 3c. For rows whose stored allele is a serotype (HLA-A2, HLA-DR4, ...),
    #     fill ``best_guess_allele`` with the most likely 4-digit member.
    #     Binding predictors can't operate on serotypes, and downstream
    #     consumers want a usable 4-digit handle. The guess is a heuristic
    #     (lowest-numbered member ≈ population-dominant); see
    #     ``best_4digit_for_serotype``.
    def _best_guess(s: str) -> str:
        guess = best_4digit_for_serotype(s)
        return guess or s

    df["best_guess_allele"] = df["mhc_restriction"].map(_best_guess)

    # 4. Split the parallel ``gene_names`` / ``gene_ids`` semicolon-joined
    #    strings into one row per (gene_name, gene_id) so we can group
    #    cleanly. Pad the shorter list with empties so the pairs stay
    #    aligned. (This is what pandas calls ``DataFrame.explode``.)
    _progress("splitting multi-gene rows (one row per gene)...", verbose)
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
    _progress(f"  {len(df):,} rows after split", verbose)
    # Final precise gene filter — the substring pre-filter above can
    # surface sibling genes from the same multi-mapping cell.
    if names or ids:
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
    #    ``best_guess_allele`` is functionally dependent on ``mhc_restriction``
    #    (one-to-one map), so include it in the group key — that lets us
    #    keep the column without an extra merge.
    grouped = (
        df.groupby(
            [
                "gene_name",
                "gene_id",
                "mhc_restriction",
                "best_guess_allele",
                "peptide",
                "mhc_class",
            ],
            dropna=False,
            observed=True,
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


def query_by_samples(
    samples_to_alleles: dict[str, list[str]],
    proteins: list[str] | None = None,
    *,
    predictor: str | None = None,
    use_hgnc: bool = True,
    verbose: bool = False,
) -> pd.DataFrame:
    """Per-sample pMHC evidence — call ``query`` once per sample with that
    sample's allele list, and return the union with a leading ``sample_name``
    column.

    Replaces the cross-product behavior of ``--mhc-allele`` when the user
    has paired (sample, allele-set) data — e.g. a cohort of patients each
    with their own HLA typing. Each sample is queried independently
    against the same protein list (or the whole corpus if ``proteins`` is
    empty); rows are tagged with ``sample_name`` so the output can be
    grouped per sample.

    Parameters
    ----------
    samples_to_alleles
        ``{sample_name: [allele1, allele2, ...]}``. Allele lists are
        passed through verbatim to ``query``, so serotype expansion
        (#185) still applies — ``"HLA-A2"`` pulls in the 4-digit members.
    proteins, predictor, use_hgnc, verbose
        Same as :func:`query`; passed through unchanged.

    Returns
    -------
    pd.DataFrame
        One row per (sample, gene, allele, peptide). Columns: same as
        ``query`` plus a leading ``sample_name``. Empty DataFrame with
        the expected schema if no sample matched anything.
    """
    if not samples_to_alleles:
        empty = _empty_result(predictor is not None)
        empty.insert(0, "sample_name", pd.Series(dtype="string"))
        return empty

    # #188: validate up-front. ``query(alleles=[])`` silently scans every
    # allele in the corpus, which is not what a paired API should do —
    # an empty per-sample allele list is a caller bug, surface it loudly
    # rather than blowing up the result with the cross-product fallback.
    bad = [name for name, alleles in samples_to_alleles.items() if not alleles]
    if bad:
        raise ValueError(
            f"sample(s) {bad!r} have empty allele lists; pass at least one "
            "allele per sample (an empty list would otherwise silently expand "
            "to the whole corpus)"
        )

    pieces: list[pd.DataFrame] = []
    for sample_name, alleles in samples_to_alleles.items():
        if verbose:
            _progress(f"sample {sample_name!r}: querying {len(alleles)} allele(s)...", verbose)
        sub = query(
            proteins=proteins,
            alleles=alleles,
            predictor=predictor,
            use_hgnc=use_hgnc,
            verbose=verbose,
        )
        if sub.empty:
            # #188: ``concat`` of an empty sub-frame yields zero rows for the
            # sample, dropping it from any downstream groupby. Emit one
            # placeholder row tagged with the sample name (other columns
            # NaN) so the sample stays visible. ``format_table`` detects
            # the all-NaN row and prints a "(no pMHC evidence on this
            # sample's alleles)" line.
            sub = _empty_result(predictor is not None)
            sub.loc[0] = pd.NA
        sub.insert(0, "sample_name", sample_name)
        pieces.append(sub)

    return pd.concat(pieces, ignore_index=True)


def _attach_predictions(df: pd.DataFrame, predictor: str) -> pd.DataFrame:
    """Score each (peptide, allele) row with MHCflurry or NetMHCpan and
    append ``affinity_nM`` / ``presentation_percentile`` / ``binder_class``.

    Uses ``best_guess_allele`` as the prediction input — predictors only
    accept 4-digit alleles, so serotype rows (HLA-A2, HLA-DR4) need the
    heuristic 4-digit substitute. Where ``best_guess_allele`` falls back
    to the original serotype string (no expansion available), the
    predictor will return NaN and the row stays unscored.
    """
    pairs = df[["peptide", "best_guess_allele"]].rename(columns={"best_guess_allele": "allele"})
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
    scored = scored.rename(columns={"allele": "best_guess_allele"})
    df = df.merge(
        scored[["peptide", "best_guess_allele", "affinity_nM", "presentation_percentile"]],
        on=["peptide", "best_guess_allele"],
        how="left",
    )
    df["binder_class"] = [
        _classify_binder(a, p) for a, p in zip(df["affinity_nM"], df["presentation_percentile"])
    ]
    return df


# Tier ordering for taking the strongest call across affinity / percentile.
_BINDER_RANK = {"non-binder": 0, "weak": 1, "medium": 2, "strong": 3}


def _classify_by_affinity(affinity_nM: float | None) -> str | None:
    """Classify by predicted IC50 (nM); ``None`` when affinity is missing."""
    if affinity_nM is None or pd.isna(affinity_nM):
        return None
    if affinity_nM <= 100:
        return "strong"
    if affinity_nM <= 500:
        return "medium"
    if affinity_nM <= 2000:
        return "weak"
    return "non-binder"


def _classify_by_percentile(percentile: float | None) -> str | None:
    """Classify by predicted-rank percentile; ``None`` when missing."""
    if percentile is None or pd.isna(percentile):
        return None
    if percentile <= 0.5:
        return "strong"
    if percentile <= 1.0:
        return "medium"
    if percentile <= 2.0:
        return "weak"
    return "non-binder"


def _classify_binder(affinity_nM: float | None, percentile: float | None = None) -> str:
    """Combine affinity and percentile classifications, taking the stronger.

    Affinity tiers (IC50, nM):
        strong:     ≤ 100
        medium:     ≤ 500
        weak:       ≤ 2000
        non-binder: > 2000

    Percentile tiers (predicted rank, %):
        strong:     ≤ 0.5
        medium:     ≤ 1.0
        weak:       ≤ 2.0
        non-binder: > 2.0

    Returns the strongest tier from either signal — a peptide that scores
    "strong" by percentile but only "weak" by affinity is reported as
    strong, since predictors disagree more about absolute IC50 than about
    the rank against the allele's per-length background.
    Empty string if both inputs are missing.
    """
    by_aff = _classify_by_affinity(affinity_nM)
    by_pct = _classify_by_percentile(percentile)
    if by_aff is None and by_pct is None:
        return ""
    candidates = [c for c in (by_aff, by_pct) if c is not None]
    return max(candidates, key=lambda c: _BINDER_RANK[c])


def _empty_result(with_predictions: bool) -> pd.DataFrame:
    cols = [
        "gene_name",
        "gene_id",
        "mhc_allele",
        "best_guess_allele",
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
            peptide        n_obs  pmids               [affinity_nM  binder]
            -------------  -----  -----------------   -------------------
          MHC_ALLELE
            PEPTIDE_SEQ        N  pmid1;pmid2          ...
            ...
          MHC_ALLELE
            ...

    Column headers are printed **once per gene** (not per allele) so the
    output stays scannable; alleles within a gene are ordered by total
    observation count, descending. If ``--predictor`` was not used, a
    one-line tip mentioning ``--predictor netmhcpan`` is appended.
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
            ("pct_rank", "presentation_percentile"),
            ("binder", "binder_class"),
        ]

    def _fmt(header: str, value) -> str:
        if pd.isna(value):
            return ""
        if header == "n_obs":
            return f"{int(value)}"
        if header == "affinity_nM":
            return f"{float(value):.1f}"
        if header == "pct_rank":
            return f"{float(value):.2f}"
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

    def _render_gene_block(gene_df: pd.DataFrame, gene_indent: str = "") -> list[str]:
        """Render one gene's section (gene header + per-allele peptide rows).

        ``gene_indent`` lets callers nest the block under an outer (e.g.
        per-sample) section without re-flowing the column widths.
        """
        block: list[str] = []
        gene_name = gene_df["gene_name"].iloc[0]
        gene_id = gene_df["gene_id"].dropna().astype(str)
        gene_id = gene_id.iloc[0] if len(gene_id) else ""
        block.append(
            f"{gene_indent}{gene_name} ({gene_id})" if gene_id else f"{gene_indent}{gene_name}"
        )
        block.append(gene_indent + header_line)
        block.append(gene_indent + rule_line)
        allele_totals = (
            gene_df.groupby("mhc_allele", observed=True)["n_observations"]
            .sum()
            .sort_values(ascending=False, kind="stable")
        )
        for allele in allele_totals.index:
            allele_df = gene_df[gene_df["mhc_allele"] == allele]
            # Serotype rows get the best-guess 4-digit member annotated in
            # the header (heuristic: lowest-numbered ≈ population-dominant).
            best_guess = ""
            if "best_guess_allele" in allele_df.columns:
                guesses = allele_df["best_guess_allele"].dropna().astype(str).unique()
                if len(guesses) == 1 and guesses[0] and guesses[0] != allele:
                    best_guess = guesses[0]
            header = f"{gene_indent}  {allele}"
            if best_guess:
                header += f"  (best guess: {best_guess})"
            block.append(header)
            for _, row in allele_df.iterrows():
                cells = [_fmt(pep_headers[i], row[k]) for i, k in enumerate(pep_keys)]
                block.append(
                    gene_indent
                    + indent
                    + sep.join(cells[i].ljust(widths[i]) for i in range(len(cells)))
                )
        return block

    out: list[str] = []
    has_sample = "sample_name" in df.columns
    if has_sample:
        # Per-sample sections: one outer block per sample, nested gene
        # blocks beneath. Empty samples (no evidence on their alleles) get a
        # placeholder line so the user can see which samples returned nothing.
        for sample_name, sample_df in df.groupby("sample_name", sort=True, observed=True):
            out.append(f"=== sample: {sample_name} ===")
            non_empty = sample_df.dropna(subset=["gene_name"])
            if non_empty.empty:
                out.append("  (no pMHC evidence on this sample's alleles)")
                out.append("")
                continue
            for _, gene_df in non_empty.groupby("gene_name", sort=True, observed=True):
                out.extend(_render_gene_block(gene_df, gene_indent="  "))
            out.append("")
    else:
        for _, gene_df in df.groupby("gene_name", sort=True, observed=True):
            out.extend(_render_gene_block(gene_df))
            out.append("")

    if not has_pred:
        out.append(
            "Tip: pass `--predictor netmhcpan` (or mhcflurry) to add binding-affinity columns."
        )
    return "\n".join(out).rstrip() + "\n"


# Backwards-compatible alias for the original name.  ``format_grouped``
# never shipped on PyPI; kept only for any in-tree imports.
format_grouped = format_table
