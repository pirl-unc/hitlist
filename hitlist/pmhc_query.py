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

import sys
import time

import pandas as pd

from .genes import resolve_gene_query

#: Named source-context filters for ``--source-context`` / ``query(source_context=)``.
#: Each maps a context name to the ``src_*`` classification columns on
#: observations.parquet; a row matches the context if ANY of its columns is True.
#: The classification is the curated source provenance of the eluting material
#: (see :func:`hitlist.curation.classify_ms_row`), independent of the MHC species.
SOURCE_CONTEXTS: dict[str, tuple[str, ...]] = {
    "healthy": ("src_healthy_tissue", "src_healthy_thymus", "src_healthy_reproductive"),
    "cancer": ("src_cancer",),
    "cell_line": ("src_cell_line",),
    "ebv_lcl": ("src_ebv_lcl",),
    "adjacent": ("src_adjacent_to_tumor",),
    "activated_apc": ("src_activated_apc",),
    "ex_vivo": ("src_ex_vivo",),
}

#: One-line descriptions for ``--list-source-contexts``.
SOURCE_CONTEXT_DESCRIPTIONS: dict[str, str] = {
    "healthy": "Direct-ex-vivo healthy tissue (somatic, thymus, or reproductive) — the safety signal",
    "cancer": "Tumor / malignant-derived material (patient tumor or cancer cell line)",
    "cell_line": "Any cell line / clone (malignant OR non-malignant immortalized)",
    "ebv_lcl": "EBV-transformed B-lymphoblastoid lines",
    "adjacent": "Tumor-adjacent normal tissue (resection margins)",
    "activated_apc": "Activated antigen-presenting cells (dendritic cells, macrophages)",
    "ex_vivo": "Any direct-ex-vivo material (modifier; not mutually exclusive with the above)",
}


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
    species: str | None = None,
    source_context: str | None = None,
    predictor: str | None = None,
    min_binder_class: str | None = None,
    min_references: int = 1,
    min_samples: int = 1,
    cell_type: str | list[str] | None = None,
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
    species
        Filter to one **MHC** species (e.g. ``"human"`` / ``"Homo sapiens"``;
        normalized via ``normalize_species``).  Without it the query reports
        every species the gene's peptides were observed on, sectioned with
        human first.  ``None`` (default) applies no filter.
    source_context
        Keep only observations whose curated **source classification** matches
        one of :data:`SOURCE_CONTEXTS` (``"healthy"``, ``"cancer"``,
        ``"cell_line"``, ``"ebv_lcl"``, ``"adjacent"``, ``"activated_apc"``,
        ``"ex_vivo"``).  A row matches if any of the context's ``src_*`` flags
        is True.  ``None`` (default) applies no filter.
    predictor
        ``"mhcflurry"``, ``"netmhcpan"``, or ``None`` (skip prediction).
        If set, attaches ``affinity_nM`` / ``presentation_percentile`` /
        ``binder_class`` columns per (peptide, allele) row.
    min_binder_class
        Drop rows whose ``binder_class`` tier is below this threshold.
        One of ``"strong" | "medium" | "weak"``.  Requires ``predictor``.
        ``None`` (default) skips this filter.  ``"weak"`` drops only
        ``"non-binder"`` rows; ``"medium"`` also drops ``"weak"``; etc.
    min_references
        Drop rows with fewer than this many distinct PMIDs.  Defaults to
        1 (no filter).  Use ``2`` to drop singleton-PMID rows when
        looking for independently re-observed peptides.
    min_samples
        Drop rows with fewer than this many distinct sample labels
        (``attributed_sample_label``).  Defaults to 1 (no filter).  A
        single PMID often contributes many samples (cohort papers,
        mono-allelic cell-line panels), so ``min_samples`` is usually
        a stronger evidence signal than ``min_references``.
    cell_type
        Keep only observations whose build-time ``cell_type`` (the tissue /
        cell-type half of IEDB's Cell Name, e.g. ``"Melanocyte"``,
        ``"B cell"``; see #261) matches.  A single string or a list;
        matching is case-insensitive.  ``None`` (default) applies no
        filter.  Requires an ``observations.parquet`` built with hitlist
        ≥ 1.30.57 (raises a clear error otherwise).
    use_hgnc
        Pass through to ``resolve_gene_query`` — set False to disable
        the HGNC alias REST lookup (offline use).

    Returns
    -------
    pd.DataFrame
        Columns: ``gene_name``, ``gene_id``, ``mhc_allele``,
        ``peptide``, ``n_observations``, ``n_references``,
        ``n_samples``, ``pmids``, ``mhc_class``.  Plus the affinity
        columns when ``predictor`` is set.
        Sorted by (mhc_species, gene_name, mhc_allele, -n_observations).
        Empty DataFrame with these columns if nothing matched.
    """
    # Argument validation runs BEFORE is_built() so callers passing bad
    # flags get a clear ValueError regardless of whether observations
    # have been built yet (tests + fresh-install error paths).
    if min_binder_class is not None:
        if predictor is None:
            raise ValueError(
                "--min-binder-class requires --predictor; binder_class is only "
                "computed when a predictor is attached."
            )
        if min_binder_class not in _BINDER_RANK:
            raise ValueError(
                f"min_binder_class must be one of {sorted(_BINDER_RANK)}, got {min_binder_class!r}"
            )

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

    # 2. Load observations.  Post-#238 ``gene_names`` / ``gene_ids`` are
    #    no longer stored on observations.parquet — ``load_observations``
    #    auto-attaches them from peptide_mappings.parquet when requested.
    #    For the gene filter we resolve the gene → peptide mapping
    #    manually (OR semantics across names and ids) and push the
    #    peptide list down to obs as a parquet filter.  This is
    #    dramatically cheaper than the pre-#238 approach of loading the
    #    full 4.4M-row corpus and substring-matching gene_names.
    # ``mhc_species`` and ``species`` are first-class columns on
    # observations.parquet (scanner populates them at build time via
    # ``classify_mhc_species`` + ``normalize_species``).  Loading them
    # here is cheaper and more authoritative than re-deriving from
    # ``mhc_restriction`` at query time, and ``species`` lets us flag
    # chimeric rows (HLA-transgenic mouse etc., where the source
    # organism differs from the MHC system).
    # #259: n_samples and --min-samples need a per-row distinct-sample
    # identifier.  We load three columns to compose it (#260 audit):
    #
    #   attributed_sample_label   1.2%, 11 distinct  per-donor patient ID
    #   cell_name                98.5%, 192 distinct IEDB's catch-all
    #                                                "Cell Name" field
    #                                                (covers both cell
    #                                                lines AND cell types
    #                                                — strict superset of
    #                                                cell_line_name, with
    #                                                100% agreement when
    #                                                both populated.  Two
    #                                                different cell types
    #                                                in one study ARE
    #                                                different samples
    #                                                — different MS runs,
    #                                                different sample
    #                                                preps — so cell_name
    #                                                belongs in the
    #                                                composite even when
    #                                                the values are
    #                                                cell-type categories
    #                                                like "B cell".)
    #   monoallelic_host         21.4%,  7 distinct  engineering host platform
    #                                                (C1R, 721.221, K562, ...)
    load_kwargs: dict = {
        "columns": [
            "peptide",
            "pmid",
            "mhc_class",
            "mhc_restriction",
            "mhc_species",
            "species",
            "attributed_sample_label",
            "cell_name",
            "cell_line_name",
            "cell_type",
            "src_cell_line",
            "monoallelic_host",
            "gene_names",
            "gene_ids",
        ],
    }
    if species is not None:
        # MHC-species pushdown (load_observations normalizes "human" etc.).
        load_kwargs["species"] = species
    if source_context is not None:
        if source_context not in SOURCE_CONTEXTS:
            raise ValueError(
                f"unknown source_context {source_context!r}; choose from {sorted(SOURCE_CONTEXTS)}"
            )
        # Load the src_* flag columns this context needs (idempotent — some,
        # e.g. src_cell_line, are already requested above).
        for col in SOURCE_CONTEXTS[source_context]:
            if col not in load_kwargs["columns"]:
                load_kwargs["columns"].append(col)
    if names or ids:
        from .mappings import load_peptide_mappings

        _progress("resolving gene → peptide mapping (peptide_mappings.parquet)...", verbose)
        pep_sets: list = []
        if names:
            pep_sets.append(
                load_peptide_mappings(gene_name=sorted(names), columns=["peptide"])["peptide"]
            )
        if ids:
            pep_sets.append(
                load_peptide_mappings(gene_id=sorted(ids), columns=["peptide"])["peptide"]
            )
        matching_peptides = sorted({p for s in pep_sets for p in s.dropna().unique()})
        _progress(f"  {len(matching_peptides):,} candidate peptides", verbose)
        if not matching_peptides:
            return _empty_result(predictor is not None)
        load_kwargs["peptide"] = matching_peptides
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

    # 2b. Cell-type filter (#261 stage 3).  ``cell_type`` is build-time-
    #     derived (the ``<line>-<type>`` Cell Name split); parquets built
    #     before v1.30.57 don't carry it, so fail loudly rather than
    #     silently returning everything.  Row-level filter applied before
    #     the gene explode so it's cheap.
    if cell_type is not None:
        if "cell_type" not in df.columns:
            raise ValueError(
                "--cell-type filtering requires an observations.parquet built with "
                "hitlist >= 1.30.57 (the build that adds the cell_type column). "
                "Rebuild with `hitlist build`."
            )
        wanted = [cell_type] if isinstance(cell_type, str) else list(cell_type)
        wanted_lower = {w.strip().lower() for w in wanted if w and w.strip()}
        ct = df["cell_type"].astype(str).str.strip().str.lower()
        df = df[ct.isin(wanted_lower)]
        _progress(f"  {len(df):,} rows after cell_type filter", verbose)
        if df.empty:
            return _empty_result(predictor is not None)

    # 2c. Source-context filter (cancer / healthy / cell_line / ...).  Keep
    #     rows where ANY of the context's curated src_* flags is True.  Row-
    #     level, applied before the gene explode so it's cheap.
    if source_context is not None:
        flag_cols = [c for c in SOURCE_CONTEXTS[source_context] if c in df.columns]
        missing = [c for c in SOURCE_CONTEXTS[source_context] if c not in df.columns]
        if not flag_cols:
            raise ValueError(
                f"--source-context {source_context!r} requires the {missing} column(s), "
                "absent from this observations.parquet. Rebuild with `hitlist build`."
            )
        mask = df[flag_cols].fillna(False).astype(bool).any(axis=1)
        df = df[mask]
        _progress(f"  {len(df):,} rows after source-context filter ({source_context})", verbose)
        if df.empty:
            return _empty_result(predictor is not None)

    # 3. Normalize the auto-attached gene columns to strings (the merge
    #    in load_observations leaves them as object dtype).  The
    #    candidate-row pre-filter that pre-#238 lived here is no longer
    #    needed — the parquet-side peptide pushdown above already
    #    narrowed obs to the matched peptides.
    for col in ("gene_names", "gene_ids"):
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)

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
    # Final precise gene filter — the parquet-side peptide pushdown
    # above can surface sibling genes when a peptide multi-maps
    # (e.g. KRAS-attributed peptide that also matches NRAS).  Drop
    # those sibling-gene rows so the user sees only the genes they asked for.
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
    # 3c. Normalize mhc_species / species sentinels.  See
    #     _normalize_species_column for the rules — extracted so tests
    #     can pin the contract directly instead of inferring it from
    #     downstream behavior.
    for col in ("mhc_species", "species"):
        if col in df.columns:
            df[col] = _normalize_species_column(df[col])

    # 3d. Surface unresolved source-organism rows (#256 review).
    #     mhc_species is always derivable from the allele prefix
    #     (HLA, H-2, DLA, ...) so it never lands in "unknown" today.
    #     The source-proteome axis is stored in TWO IEDB columns at
    #     different granularity — ``source_organism`` (strain-level) and
    #     ``species`` (species-rank).  They describe the same axis (#306),
    #     so a row is only genuinely *unresolved* when BOTH are missing.
    #     Warn on the canonical coalesce so we don't spuriously flag rows
    #     where one field is curated but the other happens to be blank
    #     (the inconsistency that surfaced the Gomez-Zepeda warning).
    #
    #     (A previous draft also warned on species != mhc_species, but
    #     that warning would fire 100K+ times on a broad query dominated
    #     by legitimate viral / bacterial peptides presented on host MHC.
    #     Low signal, removed.)
    src_cols = [c for c in ("species", "source_organism") if c in df.columns]
    if src_cols:
        unknown_mask = pd.Series(True, index=df.index)
        for c in src_cols:
            unknown_mask &= _normalize_species_column(df[c]) == "unknown"
        n_unknown = int(unknown_mask.sum())
        if n_unknown and verbose:
            _progress(
                f"WARNING: {n_unknown:,} row(s) have unresolved source organism "
                '(empty source_organism field or literal "unidentified" in IEDB '
                "metadata).  These should ideally be curated — file a follow-up "
                "if you see this in a query you care about.",
                verbose,
            )

    # 4. Aggregate to (gene_name, gene_id, mhc_restriction, peptide, mhc_class):
    #    mhc_species is included in the group key — it's functionally
    #    dependent on mhc_restriction (one species per allele string)
    #    so it doesn't change cardinality but flows through aggregation
    #    without an extra merge.
    # ── Sample-identity decomposition (#260 review) ──────────────────
    #
    # n_samples splits into two orthogonal sample categories that
    # shouldn't be conflated:
    #
    #   n_cell_lines        = distinct cell lines profiled
    #                         (src_cell_line=True rows; identified by
    #                          cell_line_name + monoallelic_host)
    #
    #   n_donor_cell_types  = distinct (donor, cell-type) combos for
    #                         primary-cell / tissue rows
    #                         (src_cell_line=False rows; identified by
    #                          donor_id + cell_name where donor_id =
    #                          attributed_sample_label or fallback to
    #                          pmid when curation didn't record the donor)
    #
    #   n_donors            = distinct donors (always ≤ n_donor_cell_types
    #                         because one donor can yield multiple
    #                         tissue / cell-type profiles)
    #
    #   n_samples           = n_cell_lines + n_donor_cell_types
    #
    # We carry three internal semicolon-joined ID columns through the
    # groupby so _collapse_rows_sharing_narrowed_allele can union them; the final
    # counts are derived from them after consolidation.
    def _str_col(name: str) -> pd.Series:
        if name in df.columns:
            return df[name].astype(object).fillna("").astype(str)
        return pd.Series([""] * len(df), index=df.index)

    src_cell_line = (
        df["src_cell_line"].astype("boolean").fillna(False)
        if "src_cell_line" in df.columns
        else pd.Series([False] * len(df), index=df.index)
    )
    cell_line_name = _str_col("cell_line_name")
    cell_name = _str_col("cell_name")
    monoallelic_host = _str_col("monoallelic_host")
    asl = _str_col("attributed_sample_label")
    pmid_str = df["pmid"].astype("Int64").astype(str)

    # Cell-line ID per row (empty for non-cell-line rows).  Set when
    # src_cell_line=True AND at least one of cell_line_name /
    # monoallelic_host is populated — the latter catches the ~9K
    # mono-allelic rows where the host platform is the only line ID.
    df["_line_id"] = (cell_line_name + "|" + monoallelic_host).where(
        src_cell_line & ((cell_line_name != "") | (monoallelic_host != "")),
        "",
    )
    # Donor ID per row (empty for cell-line rows).
    donor_id = asl.where(asl != "", "pmid:" + pmid_str)
    df["_donor_id"] = donor_id.where(~src_cell_line, "")
    # (donor, cell-type) per row (empty for cell-line rows).
    df["_donor_type_id"] = (donor_id + "|" + cell_name).where(~src_cell_line, "")

    def _join_distinct_nonempty(s: pd.Series) -> str:
        return ";".join(sorted({str(x) for x in s.dropna() if str(x)}))

    grouped = (
        df.groupby(
            [
                "gene_name",
                "gene_id",
                "mhc_restriction",
                "mhc_species",
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
            _line_ids=("_line_id", _join_distinct_nonempty),
            _donor_ids=("_donor_id", _join_distinct_nonempty),
            _donor_type_ids=("_donor_type_id", _join_distinct_nonempty),
        )
        .reset_index()
        .rename(columns={"mhc_restriction": "mhc_allele"})
    )

    # 5. Optional binding-affinity prediction.  _collapse_rows_sharing_narrowed_allele
    #    (inside _score_and_narrow_to_best_allele) preserves mhc_species AND the
    #    three internal ID columns via its group_cols / agg_spec.
    if predictor is not None:
        grouped = _score_and_narrow_to_best_allele(grouped, predictor)

    # 5b. Derive the user-facing count columns from the joined-ID columns,
    #     then drop the internals.
    def _count_distinct(s: str) -> int:
        if not s:
            return 0
        return len([x for x in str(s).split(";") if x])

    grouped["n_references"] = grouped["pmids"].apply(lambda s: len(str(s).split(";")) if s else 0)
    grouped["n_cell_lines"] = grouped["_line_ids"].apply(_count_distinct)
    grouped["n_donors"] = grouped["_donor_ids"].apply(_count_distinct)
    grouped["n_donor_cell_types"] = grouped["_donor_type_ids"].apply(_count_distinct)
    # n_samples is the headline total: one count per cell-line profiled +
    # one count per (donor, cell-type) combo profiled.  A donor with N
    # cell types contributes N samples; a donor with 1 cell type → 1.
    grouped["n_samples"] = grouped["n_cell_lines"] + grouped["n_donor_cell_types"]
    grouped = grouped.drop(columns=["_line_ids", "_donor_ids", "_donor_type_ids"])

    # 5c. Apply user-supplied filters (#259).
    if min_binder_class is not None:
        threshold = _BINDER_RANK[min_binder_class]
        binder_rank = grouped["binder_class"].map(_BINDER_RANK).fillna(-1)
        grouped = grouped[binder_rank >= threshold].reset_index(drop=True)
    if min_references > 1:
        grouped = grouped[grouped["n_references"] >= min_references].reset_index(drop=True)
    if min_samples > 1:
        grouped = grouped[grouped["n_samples"] >= min_samples].reset_index(drop=True)

    # 6. Order: species first (humans top of the page when present, then
    #    alphabetical), then by gene → allele → evidence count desc.
    grouped = grouped.sort_values(
        ["mhc_species", "gene_name", "mhc_allele", "n_observations"],
        ascending=[True, True, True, False],
        kind="stable",
        key=lambda col: col.map(_species_sort_key) if col.name == "mhc_species" else col,
    ).reset_index(drop=True)
    return grouped


#: Display abbreviations for verbose IEDB source-tissue names (--by-tissue).
_TISSUE_ABBREV: dict[str, str] = {
    "Central nervous system (CNS)": "CNS",
    "Peripheral Nervous System": "PNS",
    "Bronchial Aveolar Lavage (BAL)": "BAL",
    "Gastrointestinal Tract": "GI tract",
}

#: source_tissue values too vague to use as a cell-line fallback label (lower-cased;
#: ``""`` already became ``"(unspecified)"`` upstream).
_UNINFORMATIVE_TISSUES: frozenset[str] = frozenset(
    {"", "(unspecified)", "other", "unknown", "n/a", "na", "not determined", "not applicable"}
)


#: --by-tissue sections, in display order, with the column each groups by.
#: Healthy / cancer primary material groups by anatomical ``source_tissue``;
#: cell lines group by the actual cell-line name (a THP-1 leukemia line is
#: "THP-1", not "Blood").
_SOURCE_SECTIONS: list[tuple[str, str]] = [
    ("healthy tissues", "source_tissue"),
    ("cancer tissues", "source_tissue"),
    ("cancer cell lines", "cell_group"),
    ("non-cancer cell lines", "cell_group"),
]


def tissue_distribution(
    proteins: list[str] | None = None,
    *,
    species: str | None = None,
    source_context: str | None = None,
    show_empty: bool = False,
    expand_lines: bool = False,
    use_hgnc: bool = True,
    verbose: bool = False,
) -> pd.DataFrame:
    """Sectioned distribution of a gene's MS-attested peptides by source.

    Splits the evidence into four sections — **healthy tissues** and **cancer
    tissues** (grouped by anatomical ``source_tissue``) and **cancer cell
    lines** / **non-cancer cell lines**.  By default the cell-line sections
    group by the **cell TYPE** of origin (``Melanocyte``, ``Monocyte``, ``B
    cell``, ``Epithelial cell``, ...) with EBV-transformed B-LCLs as their own
    ``EBV-LCL`` category; pass ``expand_lines=True`` to group by the individual
    line name (``THP-1``, ``HeLa``, ...).  Answers *"which healthy tissues /
    tumors / cell types is <gene> presented in?"* — an antigen's safety profile.

    Returns a long frame with columns ``section``, ``group`` (tissue / cell type
    / line), ``n_observations`` (rows), ``n_unique_peptides``, ``n_references``;
    each section sorted by ``n_observations`` descending.  Empty sections are
    elided unless ``show_empty``.  Parameters otherwise mirror :func:`query`.
    """
    from .observations import load_observations

    out_cols = ["section", "group", "n_observations", "n_unique_peptides", "n_references"]
    names: set[str] = set()
    ids: set[str] = set()
    if proteins:
        for q in proteins:
            spec = resolve_gene_query(q, use_hgnc=use_hgnc)
            names |= spec["names"]
            ids |= spec["ids"]
        if not names and not ids:
            return pd.DataFrame(columns=out_cols)

    healthy_flags = list(SOURCE_CONTEXTS["healthy"])
    cols = [
        "peptide",
        "pmid",
        "source_tissue",
        "cell_name",
        "cell_line_name",
        "src_cancer",
        "src_cell_line",
        "src_ebv_lcl",
        *healthy_flags,
    ]
    if source_context is not None:
        if source_context not in SOURCE_CONTEXTS:
            raise ValueError(
                f"unknown source_context {source_context!r}; choose from {sorted(SOURCE_CONTEXTS)}"
            )
        cols += [c for c in SOURCE_CONTEXTS[source_context] if c not in cols]

    _progress("loading observations for source distribution...", verbose)
    df = load_observations(
        gene_name=sorted(names) or None,
        gene_id=sorted(ids) or None,
        species=species,
        columns=cols,
    )
    if source_context is not None and not df.empty:
        flag_cols = [c for c in SOURCE_CONTEXTS[source_context] if c in df.columns]
        if flag_cols:
            df = df[df[flag_cols].fillna(False).astype(bool).any(axis=1)]
    if df.empty:
        return pd.DataFrame(columns=out_cols)

    df = df.copy()
    df["source_tissue"] = (
        df["source_tissue"].astype(str).replace("", "(unspecified)").replace(_TISSUE_ABBREV)
    )

    def _flag(name: str) -> pd.Series:
        return (
            df[name].fillna(False).astype(bool)
            if name in df.columns
            else pd.Series(False, index=df.index)
        )

    # Cell-line grouping: by default the CELL TYPE of origin (Melanocyte,
    # Monocyte, B cell, ...) with EBV-LCLs as their own category; ``expand_lines``
    # switches to the individual line name.  Parse once per unique cell_name.
    from .cell_name_parser import parse_cell_name

    cn = df["cell_name"].astype(str) if "cell_name" in df else pd.Series("", index=df.index)
    type_map: dict[str, str] = {}
    line_map: dict[str, str] = {}
    for raw in cn.unique():
        parsed = parse_cell_name(raw)
        type_map[raw] = parsed.cell_type or ""
        line_map[raw] = parsed.cell_line_name or ""
    ebv = _flag("src_ebv_lcl")
    # When the cell TYPE is unknown (IEDB logged "Other"/blank) but the row still
    # records a real source TISSUE, fall back to the tissue name (e.g. "Skin")
    # rather than collapsing every such line into one opaque "(unspecified type)"
    # bucket — this keeps heterogeneous patient-derived cohorts (e.g. one study's
    # skin + ovary tumor lines) split by what we do know.  The section banner
    # ("▌ CANCER CELL LINES") already says these are lines, so no suffix is
    # needed.  Only fires for genuinely informative tissues.
    tissue_str = df["source_tissue"].astype(str)
    informative_tissue = ~tissue_str.str.strip().str.lower().isin(_UNINFORMATIVE_TISSUES)
    tissue_fallback = tissue_str
    if expand_lines:
        cell_group = cn.map(line_map)
        empty = cell_group.str.strip() == ""
        cell_group = cell_group.mask(empty & ebv, "EBV-LCL (unnamed)")
        empty = cell_group.str.strip() == ""
        cell_group = cell_group.mask(empty & informative_tissue, tissue_fallback)
        cell_group = cell_group.replace("", "(unnamed line)")
    else:
        cell_group = cn.map(type_map)
        empty = cell_group.str.strip() == ""
        cell_group = cell_group.mask(empty & ~ebv & informative_tissue, tissue_fallback)
        cell_group = cell_group.where(cell_group.str.strip().ne(""), "(unspecified type)")
        cell_group = cell_group.mask(ebv, "EBV-LCL")
    df["cell_group"] = cell_group

    present_healthy = [c for c in healthy_flags if c in df.columns]
    healthy = (
        df[present_healthy].fillna(False).astype(bool).any(axis=1)
        if present_healthy
        else pd.Series(False, index=df.index)
    )
    cancer = _flag("src_cancer")
    line = _flag("src_cell_line")
    section_masks = {
        "healthy tissues": healthy & ~line,
        "cancer tissues": cancer & ~line,
        "cancer cell lines": cancer & line,
        "non-cancer cell lines": line & ~cancer,
    }

    frames: list[pd.DataFrame] = []
    for order, (section, key) in enumerate(_SOURCE_SECTIONS):
        sub = df[section_masks[section]]
        if sub.empty:
            if show_empty:
                frames.append(
                    pd.DataFrame(
                        [
                            {
                                "section": section,
                                "group": "(none)",
                                "n_observations": 0,
                                "n_unique_peptides": 0,
                                "n_references": 0,
                                "_o": order,
                            }
                        ]
                    )
                )
            continue
        g = (
            sub.groupby(key, observed=True)
            .agg(
                n_observations=("peptide", "size"),
                n_unique_peptides=("peptide", "nunique"),
                n_references=("pmid", "nunique"),
            )
            .reset_index()
            .rename(columns={key: "group"})
        )
        g["section"] = section
        g["_o"] = order
        frames.append(g)

    if not frames:
        return pd.DataFrame(columns=out_cols)
    res = (
        pd.concat(frames, ignore_index=True)
        .sort_values(["_o", "n_observations", "group"], ascending=[True, False, True])
        .reset_index(drop=True)
    )
    res = res[out_cols]
    # Grand total ACROSS all four sub-categories: observations summed, but
    # peptides / references the UNION over every displayed row (a peptide or PMID
    # shared by, say, a healthy tissue and a cancer line counts once).  Computed
    # from the union of the section masks, so rows in no displayed section (e.g.
    # tumor-adjacent only) are excluded.
    any_mask = pd.Series(False, index=df.index)
    for m in section_masks.values():
        any_mask = any_mask | m
    shown = df[any_mask]
    res.attrs["grand_total"] = {
        "n_observations": len(shown),
        "n_unique_peptides": int(shown["peptide"].nunique()),
        "n_references": int(shown["pmid"].nunique()),
    }
    return res


def query_by_samples(
    samples_to_alleles: dict[str, list[str]],
    proteins: list[str] | None = None,
    *,
    species: str | None = None,
    source_context: str | None = None,
    predictor: str | None = None,
    min_binder_class: str | None = None,
    min_references: int = 1,
    min_samples: int = 1,
    cell_type: str | list[str] | None = None,
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
            species=species,
            source_context=source_context,
            predictor=predictor,
            min_binder_class=min_binder_class,
            min_references=min_references,
            min_samples=min_samples,
            cell_type=cell_type,
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


def _score_and_narrow_to_best_allele(df: pd.DataFrame, predictor: str) -> pd.DataFrame:
    """Score each row's (peptide, allele-set) and narrow multi-allele rows
    to the single best-binding allele within their candidate set (#239).

    For rows where ``mhc_allele`` is a single 4-digit allele, the
    predictor scores that one pair and the row is unchanged apart from
    the new score columns.  For multi-allele rows (semicolon-joined
    typings — every per-donor row from #236, every IEDB
    ``sample_allele_match`` row carrying the donor's full HLA typing),
    the function:

    1. Expands the row to one ``(peptide, allele)`` prediction call per
       individual allele in the set.
    2. Scores each pair via MHCflurry or NetMHCpan.
    3. Picks the best binder by ``presentation_percentile`` (with
       ``affinity_nM`` as the tiebreaker).
    4. Narrows ``mhc_allele`` and ``best_guess_allele`` to that single
       allele, records the choice in ``best_predicted_allele``.
    5. Re-aggregates rows that now share the same narrowed allele
       (e.g. SLLQHLIGL was attributed to MEL3 / MEL15 / OV1, all narrow
       to A\\*02:01 — the three rows collapse into one with summed
       ``n_observations`` and unioned ``pmids``).

    Rows where every allele in the set returns NaN (predictor failure
    or peptide-length mismatch) keep their original multi-allele
    ``mhc_allele`` and have empty ``best_predicted_allele``.
    """
    if df.empty:
        df = df.copy()
        df["affinity_nM"] = pd.Series(dtype="float64")
        df["presentation_percentile"] = pd.Series(dtype="float64")
        df["binder_class"] = pd.Series(dtype="string")
        df["best_predicted_allele"] = pd.Series(dtype="string")
        return df

    # Build a long frame: one (peptide, allele) candidate per individual
    # allele in each row's best_guess_allele set, tagged with the
    # source row's positional index so we can map results back.
    candidates: list[dict] = []
    for pos, (_, row) in enumerate(df.iterrows()):
        peptide = str(row["peptide"])
        allele_str = str(row.get("best_guess_allele") or "")
        for allele in allele_str.split(";"):
            allele = allele.strip()
            if allele:
                candidates.append({"_row_pos": pos, "peptide": peptide, "allele": allele})

    if not candidates:
        df = df.copy()
        df["affinity_nM"] = pd.NA
        df["presentation_percentile"] = pd.NA
        df["binder_class"] = "non-binder"
        df["best_predicted_allele"] = ""
        return df

    cand_df = pd.DataFrame(candidates)

    # Score the unique (peptide, allele) pairs only — many rows share
    # the same pair after the per-donor split (every Sarkizova patient
    # carries A\*02:01, every B*07:02-only row asks the same question).
    unique_pairs = cand_df[["peptide", "allele"]].drop_duplicates().reset_index(drop=True)
    if predictor == "mhcflurry":
        from .predict import _predict_mhcflurry

        scored = _predict_mhcflurry(unique_pairs.copy())
    elif predictor == "netmhcpan":
        from .predict import _predict_netmhcpan

        scored = _predict_netmhcpan(unique_pairs.copy())
    else:
        raise ValueError(f"Unknown predictor {predictor!r}; use 'mhcflurry' or 'netmhcpan'")

    cand_df = cand_df.merge(
        scored[["peptide", "allele", "affinity_nM", "presentation_percentile"]],
        on=["peptide", "allele"],
        how="left",
    )

    # Best per row: lowest presentation_percentile, then lowest affinity_nM.
    # ``na_position="last"`` keeps unscored alleles out of the way so they
    # only "win" when nothing in the set scored.
    best = (
        cand_df.sort_values(
            ["_row_pos", "presentation_percentile", "affinity_nM"], na_position="last"
        )
        .drop_duplicates("_row_pos", keep="first")
        .set_index("_row_pos")
    )

    df = df.reset_index(drop=True).copy()
    df["affinity_nM"] = best["affinity_nM"].reindex(df.index)
    df["presentation_percentile"] = best["presentation_percentile"].reindex(df.index)

    # Narrow mhc_allele / best_guess_allele only when at least one allele
    # in the set produced a real prediction.  ``best_predicted_allele``
    # records the choice (empty string when no allele scored).
    has_score = df["presentation_percentile"].notna()
    df["best_predicted_allele"] = ""
    df.loc[has_score, "best_predicted_allele"] = best.loc[has_score, "allele"].values
    df.loc[has_score, "mhc_allele"] = best.loc[has_score, "allele"].values
    df.loc[has_score, "best_guess_allele"] = best.loc[has_score, "allele"].values

    df["binder_class"] = [
        _classify_binder(a, p) for a, p in zip(df["affinity_nM"], df["presentation_percentile"])
    ]

    # After narrowing, the per-donor rows from #236 (3 rows for SLLQHLIGL,
    # all of whose donor typings contain A\*02:01) collapse to a single
    # mhc_allele.  Re-aggregate so the user sees one consolidated row
    # per (gene, narrowed allele, peptide, class) instead of N redundant
    # per-donor rows that all point at the same allele.
    return _collapse_rows_sharing_narrowed_allele(df)


def _collapse_rows_sharing_narrowed_allele(df: pd.DataFrame) -> pd.DataFrame:
    """Sum ``n_observations`` and union ``pmids`` for rows that share
    ``(gene_name, gene_id, mhc_allele, peptide, mhc_class)`` post-#239
    narrowing.  Score columns are taken from the first row (all rows in
    a group have the same peptide-allele pair → same prediction)."""
    if df.empty:
        return df

    def _union_pmids(values: pd.Series) -> str:
        seen: set[str] = set()
        for v in values.dropna():
            for p in str(v).split(";"):
                if p:
                    seen.add(p)
        return ";".join(sorted(seen, key=int))

    def _union_semicolon_set(values: pd.Series) -> str:
        seen: set[str] = set()
        for v in values.dropna():
            for label in str(v).split(";"):
                if label:
                    seen.add(label)
        return ";".join(sorted(seen))

    score_cols = ["affinity_nM", "presentation_percentile", "binder_class", "best_predicted_allele"]
    # mhc_species is FD on mhc_allele (one species per allele string) so
    # including it doesn't change cardinality — but it MUST be in the
    # group_cols or .agg() silently drops it from the result frame.
    group_cols = [
        "gene_name",
        "gene_id",
        "mhc_allele",
        "best_guess_allele",
        "peptide",
        "mhc_class",
    ]
    if "mhc_species" in df.columns:
        group_cols.append("mhc_species")
    agg_spec: dict = {
        "n_observations": "sum",
        "pmids": _union_pmids,
    }
    # _line_ids / _donor_ids / _donor_type_ids carry the per-row
    # distinct-sample lists as semicolon-joined strings; union them on
    # consolidation so the post-narrowing counts reflect the true
    # distinct-sample sets.  Same silently-dropped-on-groupby gotcha as
    # mhc_species — must be in agg_spec.
    for col in ("_line_ids", "_donor_ids", "_donor_type_ids"):
        if col in df.columns:
            agg_spec[col] = _union_semicolon_set
    for col in score_cols:
        if col in df.columns:
            agg_spec[col] = "first"

    return df.groupby(group_cols, dropna=False, observed=True).agg(agg_spec).reset_index()


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
        "n_references",
        "n_cell_lines",
        "n_donors",
        "n_donor_cell_types",
        "n_samples",
        "pmids",
        "mhc_class",
        "mhc_species",
    ]
    if with_predictions:
        cols += ["affinity_nM", "presentation_percentile", "binder_class"]
    return pd.DataFrame(columns=cols)


# ── Species normalization + ordering for output grouping (#256) ────────
#
# The ``mhc_species`` column is loaded from observations.parquet (the
# scanner populates it at build time via classify_mhc_species), so
# pmhc_query doesn't need to re-derive it from the allele string.


def _normalize_species_column(s: pd.Series) -> pd.Series:
    """Fold empty / NaN / literal "unidentified" upstream sentinels into
    the single ``"unknown"`` bucket.

    The scanner writes ``""`` for rows where classify_mhc_species or
    normalize_species couldn't resolve their input.  IEDB also has a
    literal ``"unidentified"`` source_organism value with the same
    semantics.  Downstream consumers (formatter, sort key, warning
    logic) should only need to handle one sentinel.

    The ``.astype(str)`` step is load-bearing: obs.parquet columns come
    back as ``Categorical`` after pyarrow's dictionary-encoded read,
    and ``Series.replace({...})`` on a Categorical without all the
    target values pre-declared as categories raises (or warns) on
    newer pandas.  Casting to object first makes the replace safe.
    """
    return s.fillna("").astype(str).replace({"": "unknown", "unidentified": "unknown"})


# Order species sections so the most common case (human) leads, mouse/rat
# follow as the standard model organisms, then everything else
# alphabetical.  "unknown" sinks to the bottom — those rows have missing
# upstream metadata that should be curated separately.
_SPECIES_SORT_ORDER = {
    "Homo sapiens": "0",
    "Mus musculus": "1",
    "Rattus norvegicus": "2",
}


def _species_sort_key(species: str) -> str:
    if species in _SPECIES_SORT_ORDER:
        return _SPECIES_SORT_ORDER[species]
    if species == "unknown":
        return "z"
    return f"5:{species}"


def unrecognized_genes(proteins: list[str], *, use_hgnc: bool = True) -> list[str]:
    """Of the requested gene queries, those that resolve to NO identifier present
    in the corpus (``peptide_mappings.parquet``).

    A query lands here when neither its symbol/aliases nor an Ensembl ID match
    any gene the corpus has peptide evidence for — usually a typo (e.g. ``XAGE12``
    vs the real ``XAGE1A``), occasionally a real gene with no MS evidence.
    Ensembl IDs are accepted; matching is case-insensitive.  Returns ``[]`` when
    the mappings aren't built (can't validate) so callers never warn spuriously.
    """
    from .mappings import known_gene_identifiers

    universe = known_gene_identifiers()
    if not universe:
        return []
    unknown: list[str] = []
    for q in proteins:
        spec = resolve_gene_query(q, use_hgnc=use_hgnc)
        candidates = {s.upper() for s in spec["names"] | spec["ids"]}
        if not (candidates & universe):
            unknown.append(q)
    return unknown


def format_tissue_table(df: pd.DataFrame) -> str:
    """Render a sectioned :func:`tissue_distribution` frame as text.

    One block per section (healthy tissues / cancer tissues / cancer cell lines
    / non-cancer cell lines), each an aligned table with the group column
    left-justified and a vertical rule before the ``n_unique_peptides`` /
    ``n_references`` summaries.  A final ``▌ TOTAL`` block sums observations
    across every sub-category but reports unique-peptides / references as the
    UNION across all of them (a peptide or PMID shared by, say, a healthy tissue
    and a cancer line counts once), read from ``df.attrs["grand_total"]``.  Empty
    sections are omitted.  The group column carries no header word — the section
    banner (``▌ CANCER CELL LINES`` …) already names what each row is.
    """
    if df is None or df.empty:
        return "(no matching observations)"

    num_cols = ["n_observations", "n_unique_peptides", "n_references"]
    grand: dict[str, int] | None = df.attrs.get("grand_total")

    # Column widths are computed ACROSS ALL sections so every block lines up: the
    # group column (and each numeric column) is the same width in every table,
    # regardless of which section's values are widest.  The grand-total values
    # are folded in too (a summed total can be wider than any single group).
    group_strs = [str(g) for g in df["group"]]
    first_w = max(len("total"), *(len(s) for s in group_strs))
    grand_vals = {c: [grand[c]] if grand else [] for c in num_cols}
    num_w = {
        c: max(len(c), *(len(str(v)) for v in df[c]), *(len(str(v)) for v in grand_vals[c]))
        for c in num_cols
    }
    widths = [first_w, *(num_w[c] for c in num_cols)]

    def _line(vals: list[str]) -> str:
        out: list[str] = []
        for i, v in enumerate(vals):
            cell = v.ljust(widths[i]) if i == 0 else v.rjust(widths[i])
            if i > 0:
                out.append("  │  " if num_cols[i - 1] == "n_unique_peptides" else "  ")
            out.append(cell)
        return "    " + "".join(out)

    def _render(sub: pd.DataFrame) -> str:
        lines = [_line(["", *num_cols])]
        lines += [
            _line([str(g), *(str(v) for v in r)])
            for g, *r in sub[["group", *num_cols]].itertuples(index=False)
        ]
        return "\n".join(lines)

    blocks: list[str] = []
    for section, _key in _SOURCE_SECTIONS:
        sub = df[df["section"] == section]
        if sub.empty:
            continue
        blocks.append(f"▌ {section.upper()}\n{_render(sub)}")
    if not blocks:
        return "(no matching observations)"
    if grand is not None:
        # Grand total across every sub-category above (union of peptides / refs).
        total_block = "\n".join(
            [
                _line(["", *num_cols]),
                _line(["all sources", *(str(grand[c]) for c in num_cols)]),
            ]
        )
        blocks.append(f"▌ TOTAL (across all sub-categories)\n{total_block}")
    return "\n\n".join(blocks)


def gene_distribution(
    proteins: list[str] | None = None,
    *,
    species: str | None = None,
    source_context: str | None = None,
    use_hgnc: bool = True,
    verbose: bool = False,
) -> pd.DataFrame:
    """Per-GENE rollup of MS evidence — one row per gene with totals.

    Answers *"across this set of genes (e.g. a CTA panel), how much evidence does
    each have?"*  Returns columns ``gene_name``, ``gene_id``, ``n_observations``
    (rows), ``n_unique_peptides``, ``n_references`` (distinct PMIDs),
    ``n_samples`` (distinct sample labels), sorted by ``n_observations`` desc.
    A peptide that multi-maps to several requested genes counts for each (it is
    real evidence for each).  ``species`` / ``source_context`` filter exactly as
    in :func:`tissue_distribution`.
    """
    from .mappings import load_peptide_mappings
    from .observations import load_observations

    out_cols = [
        "gene_name",
        "gene_id",
        "n_observations",
        "n_unique_peptides",
        "n_references",
        "n_samples",
    ]
    names: set[str] = set()
    ids: set[str] = set()
    if proteins:
        for q in proteins:
            spec = resolve_gene_query(q, use_hgnc=use_hgnc)
            names |= spec["names"]
            ids |= spec["ids"]
        if not names and not ids:
            return pd.DataFrame(columns=out_cols)

    _progress("loading peptide→gene mapping...", verbose)
    mp = load_peptide_mappings(
        gene_name=sorted(names) or None,
        gene_id=sorted(ids) or None,
        columns=["peptide", "gene_name", "gene_id"],
    ).drop_duplicates()
    if mp.empty:
        return pd.DataFrame(columns=out_cols)

    cols = ["peptide", "pmid", "attributed_sample_label"]
    if source_context is not None:
        if source_context not in SOURCE_CONTEXTS:
            raise ValueError(
                f"unknown source_context {source_context!r}; choose from {sorted(SOURCE_CONTEXTS)}"
            )
        cols += [c for c in SOURCE_CONTEXTS[source_context] if c not in cols]

    _progress("loading observations for gene rollup...", verbose)
    obs = load_observations(
        gene_name=sorted(names) or None,
        gene_id=sorted(ids) or None,
        species=species,
        columns=cols,
    )
    if source_context is not None and not obs.empty:
        flag_cols = [c for c in SOURCE_CONTEXTS[source_context] if c in obs.columns]
        if flag_cols:
            obs = obs[obs[flag_cols].fillna(False).astype(bool).any(axis=1)]
    if obs.empty:
        return pd.DataFrame(columns=out_cols)

    merged = obs.merge(mp, on="peptide", how="inner")
    if merged.empty:
        return pd.DataFrame(columns=out_cols)
    g = (
        merged.groupby(["gene_name", "gene_id"], observed=True)
        .agg(
            n_observations=("peptide", "size"),
            n_unique_peptides=("peptide", "nunique"),
            n_references=("pmid", "nunique"),
            n_samples=("attributed_sample_label", "nunique"),
        )
        .reset_index()
        .sort_values(["n_observations", "gene_name"], ascending=[False, True])
        .reset_index(drop=True)
    )
    g = g[out_cols]
    # Panel total: observations SUMMED; unique-peptides / references / samples
    # the UNION across the whole panel (a peptide multi-mapping to two genes,
    # or a PMID shared by several, counts once).
    g.attrs["panel_total"] = {
        "n_observations": len(merged),
        "n_unique_peptides": int(merged["peptide"].nunique()),
        "n_references": int(merged["pmid"].nunique()),
        "n_samples": int(merged["attributed_sample_label"].nunique()),
    }
    return g


def format_gene_table(df: pd.DataFrame) -> str:
    """Render a :func:`gene_distribution` frame as an aligned per-gene table,
    ending in a ``total`` row (observations summed, the rest unioned across the
    panel)."""
    if df is None or df.empty:
        return "(no matching observations)"
    num_cols = ["n_observations", "n_unique_peptides", "n_references", "n_samples"]
    total = df.attrs.get("panel_total")
    head = ["gene", *num_cols]
    rows = [
        [str(r.gene_name), *(str(getattr(r, c)) for c in num_cols)]
        for r in df.itertuples(index=False)
    ]
    total_row = ["total", *(str(total[c]) for c in num_cols)] if total else None
    pool = [head, *rows] + ([total_row] if total_row else [])
    widths = [max(len(r[i]) for r in pool) for i in range(len(head))]

    def _line(vals: list[str]) -> str:
        cells = [vals[0].ljust(widths[0])]
        cells += [vals[i].rjust(widths[i]) for i in range(1, len(vals))]
        return "    " + "  ".join(cells)

    out = [_line(head), *(_line(r) for r in rows)]
    if total_row:
        out.append("    " + "  ".join("-" * widths[i] for i in range(len(head))))
        out.append(_line(total_row))
    return "\n".join(out)


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
        ("n_refs", "n_references"),
        ("n_lines", "n_cell_lines"),
        ("n_donors", "n_donors"),
        ("n_samples", "n_samples"),
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
        if header in ("n_obs", "n_refs", "n_lines", "n_donors", "n_samples"):
            return f"{int(value)}"
        if header == "affinity_nM":
            return f"{float(value):.1f}"
        if header == "pct_rank":
            return f"{float(value):.2f}"
        if header == "pmids":
            # Truncate long PMID lists.  Full list is still in the CSV/JSON
            # output via the ``pmids`` column; the table view just shows
            # the first 3 + a count so the column doesn't dominate the page.
            parts = str(value).split(";")
            if len(parts) > 3:
                return f"{';'.join(parts[:3])}; +{len(parts) - 3} more"
            return str(value)
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

        Allele ordering: specific 4-digit alleles (with ``*`` in the name)
        first, by total evidence count; then class-only / empty alleles
        (``"HLA class I"`` / ``""`` — rows where IEDB didn't record an
        allele) at the bottom under a clear synthetic header so they don't
        masquerade as a missing-data blank line.
        """
        block: list[str] = []
        gene_name = gene_df["gene_name"].iloc[0]
        gene_id = gene_df["gene_id"].dropna().astype(str)
        gene_id = gene_id.iloc[0] if len(gene_id) else ""
        block.append(
            f"{gene_indent}{gene_name} ({gene_id})" if gene_id else f"{gene_indent}{gene_name}"
        )
        block.append((gene_indent + header_line).rstrip())
        block.append((gene_indent + rule_line).rstrip())
        allele_totals = (
            gene_df.groupby("mhc_allele", observed=True)["n_observations"]
            .sum()
            .sort_values(ascending=False, kind="stable")
        )

        def _is_specific(allele: str) -> bool:
            # 4-digit / fully-specified allele has a "*" (HLA-A*02:01,
            # DLA-88*501:01, ...).  H-2-* and Mamu rows also use "*".
            # Class-only ("HLA class I") and empty strings → not specific.
            return bool(allele) and "*" in str(allele)

        specific = [a for a in allele_totals.index if _is_specific(a)]
        unspecific = [a for a in allele_totals.index if not _is_specific(a)]
        ordered_alleles = specific + unspecific

        for allele in ordered_alleles:
            allele_df = gene_df[gene_df["mhc_allele"] == allele]
            # Serotype rows get the best-guess 4-digit member annotated in
            # the header (heuristic: lowest-numbered ≈ population-dominant).
            best_guess = ""
            if "best_guess_allele" in allele_df.columns:
                guesses = allele_df["best_guess_allele"].dropna().astype(str).unique()
                if len(guesses) == 1 and guesses[0] and guesses[0] != allele:
                    best_guess = guesses[0]
            # Render class-only / empty alleles with a clear synthetic
            # header — otherwise an empty allele renders as a phantom
            # blank line that looks like a layout bug.
            allele_label = allele if allele else "(allele not specified)"
            header = f"{gene_indent}  {allele_label}"
            if best_guess:
                header += f"  (best guess: {best_guess})"
            block.append(header)
            for _, row in allele_df.iterrows():
                cells = [_fmt(pep_headers[i], row[k]) for i, k in enumerate(pep_keys)]
                line = (
                    gene_indent
                    + indent
                    + sep.join(cells[i].ljust(widths[i]) for i in range(len(cells)))
                )
                # Strip trailing pad — keeps the right edge tidy without
                # disturbing inter-column alignment (the rstrip only nukes
                # padding past the last column's content).
                block.append(line.rstrip())
        return block

    out: list[str] = []
    has_sample = "sample_name" in df.columns

    # Multi-species results get an outer "=== species: X ===" header.
    # Single-species results (the typical human-only case) skip the
    # header so output stays compact and unchanged from pre-#256.
    multi_species = (
        "mhc_species" in df.columns and df["mhc_species"].dropna().astype(str).nunique() > 1
    )

    def _render_genes(scope_df: pd.DataFrame, indent: str, blank_between: bool) -> None:
        first = True
        for _, gene_df in scope_df.groupby("gene_name", sort=True, observed=True):
            if blank_between and not first:
                out.append("")
            out.extend(_render_gene_block(gene_df, gene_indent=indent))
            first = False

    def _render_species_partition(scope_df: pd.DataFrame, base_indent: str) -> None:
        if not multi_species:
            _render_genes(scope_df, base_indent, blank_between=not has_sample)
            return
        species_iter = sorted(
            scope_df["mhc_species"].dropna().astype(str).unique(),
            key=_species_sort_key,
        )
        first = True
        for sp in species_iter:
            sp_df = scope_df[scope_df["mhc_species"].astype(str) == sp]
            if sp_df.empty:
                continue
            if not first:
                out.append("")
            out.append(f"{base_indent}=== species: {sp} ===")
            _render_genes(sp_df, base_indent + ("  " if base_indent else ""), blank_between=True)
            first = False

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
            _render_species_partition(non_empty, base_indent="  ")
            out.append("")
    else:
        _render_species_partition(df, base_indent="")
        out.append("")

    if not has_pred:
        out.append(
            "Tip: pass `--predictor netmhcpan` (or mhcflurry) to add binding-affinity columns."
        )
    return "\n".join(out).rstrip() + "\n"


# Backwards-compatible alias for the original name.  ``format_grouped``
# never shipped on PyPI; kept only for any in-tree imports.
format_grouped = format_table
