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

"""Load the built peptide indexes with optional filters.

Two parallel parquet indexes are built by
:func:`hitlist.builder.build_observations`:

- ``observations.parquet`` — MS-eluted immunopeptidome rows (IEDB +
  CEDAR + supplementary).  Load with :func:`load_ms_observations`
  (or the older alias :func:`load_observations`).
- ``binding.parquet`` — binding-assay rows (refolding, MEDi, peptide
  microarray, quantitative-tier measurements).  Load with
  :func:`load_binding`.

The two indexes share the same schema but are never mixed: MS and
binding data go to separate files so downstream consumers cannot
accidentally conflate them.  Only the MS index gets supplementary
data and sample-level metadata joins (see :mod:`hitlist.export`).

Usage::

    from hitlist.observations import load_ms_observations, load_binding

    ms = load_ms_observations(mhc_class="I")
    bd = load_binding(mhc_class="I", mhc_restriction="HLA-A*02:01")

For callers that explicitly want both — e.g. affinity-predictor training
pipelines, or CLI flags like tsarina's ``--include-binding-assays`` —
:func:`load_all_evidence` returns a UNION with an ``evidence_kind`` column
(``"ms"`` / ``"binding"``).  Filters apply symmetrically to both indexes.
"""

from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from .downloads import data_dir


@lru_cache(maxsize=8)
def _unique_restrictions(path_str: str, mtime_ns: int, size: int) -> tuple[str, ...]:
    """Cached unique values of the ``mhc_restriction`` column for a parquet
    file, keyed by ``(path, mtime_ns, size)`` so a rebuild invalidates.

    Nanosecond mtime + size guards against same-second rebuild collisions
    on filesystems with 1s mtime resolution (HFS+, some network mounts).
    The set of distinct restriction strings is small (~hundreds) and
    changes only when the parquet rebuilds, but we use it on every
    set-aware ``mhc_restriction`` filter call — caching avoids re-reading
    the full column off disk for each query.
    """
    table = pq.read_table(path_str, columns=["mhc_restriction"])
    return tuple(table.column("mhc_restriction").unique().to_pylist())


def _unique_restrictions_for(path: Path) -> tuple[str, ...]:
    """``_unique_restrictions`` keyed by the file's stat tuple."""
    st = os.stat(path)
    return _unique_restrictions(str(path), st.st_mtime_ns, st.st_size)


def observations_path() -> Path:
    """Path to the MS-eluted observations parquet file."""
    return data_dir() / "observations.parquet"


def binding_path() -> Path:
    """Path to the binding-assay parquet file."""
    return data_dir() / "binding.parquet"


def is_built() -> bool:
    """Check if the observations table has been built."""
    return observations_path().exists()


def is_binding_built() -> bool:
    """Check if the binding-assay table has been built."""
    return binding_path().exists()


def load_observations(
    mhc_class: str | None = None,
    species: str | None = None,
    source: str | None = None,
    mhc_restriction: str | list[str] | None = None,
    mhc_allele_in_set: str | list[str] | None = None,
    mhc_allele_provenance: str | list[str] | None = None,
    gene_name: str | list[str] | None = None,
    gene_id: str | list[str] | None = None,
    peptide: str | list[str] | None = None,
    serotype: str | list[str] | None = None,
    length_min: int | None = None,
    length_max: int | None = None,
    exclude_class_label_suspect: bool = False,
    exclude_class_label_implausible: bool = False,
    exclude_non_peptide_ligand: bool = True,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Load the built MS observations table with optional filters.

    The table contains only MS-eluted immunopeptidome observations.
    Binding-assay data is in a separate parquet — use
    :func:`load_binding` for that.

    Parameters
    ----------
    mhc_class
        Filter to ``"I"``, ``"II"``, or ``"non classical"``.
    species
        Filter by MHC species (e.g. ``"Homo sapiens"``).
    source
        Filter by data source (``"iedb"``, ``"cedar"``, ``"supplement"``).
    mhc_restriction
        **Exact** MHC allele filter on the row's recorded restriction
        (e.g. ``"HLA-A*02:01"``).  Misses class-only rows where the
        donor is multi-allelic — for those, use ``mhc_allele_in_set``.
    mhc_allele_in_set
        Set-membership filter: keep rows whose ``mhc_allele_set``
        (the candidate-allele set from ``expand_allele_set`` — see #137,
        #45) contains any of the listed alleles.  This is the right
        knob for queries like *"show me HLA-A*02:01 melanoma peptides"*
        that need to recover **multi-allelic patient tumor cohorts**
        where IEDB stored only the class label.  Strict subset of
        ``mhc_restriction``: a 4-digit row passes both filters; a
        class-only row with a curated set passes only ``mhc_allele_in_set``.
    mhc_allele_provenance
        Filter by how a row's allele set was obtained:

        - ``"exact"`` — restriction was already 4-digit, set = {restriction}
        - ``"peptide_attribution"`` — set narrowed via per-peptide
          attribution from the paper supplement (#45, e.g. Sarkizova 2020
          patient tumor cohort)
        - ``"sample_allele_match"`` — set = donor's typed alleles from
          IEDB ``Host | MHC Types Present``
        - ``"pmid_class_pool"`` — set = curated per-PMID pool when no
          per-row donor typing was recorded
        - ``"unmatched"`` — set empty (no donor typing or pool curation)

        Use ``"exact"`` for strict allele-resolved training data;
        ``"peptide_attribution"`` for sample-narrowed multi-allelic
        cohorts; the others depending on tolerance for set noise.
    gene_name, gene_id
        Gene filters — resolved through the peptide mappings sidecar.
    length_min, length_max
        Inclusive peptide length bounds. ``length_min=8, length_max=11``
        filters to MHC-I-compatible peptides; ``length_min=12,
        length_max=25`` to MHC-II. ``None`` (default) means no bound.
    exclude_class_label_suspect
        When True, drop rows where the peptide length disagrees with
        the curated MHC class (class II ≤ 10 aa, or class I ≥ 18 aa).
        See ``mhc_class_label_suspect`` flag (#182). Useful for model
        training pipelines that should not see IEDB class-label drift.
    exclude_non_peptide_ligand
        When True (default), drop rows whose MHC molecule presents
        lipids/glycolipids/metabolites rather than peptides — CD1
        family, MR1, MIC{A,B}, RAET1*, ULBP*, NKG2[A-C], HFE (#228).
        These rows carry chemical names or compound identifiers in the
        ``peptide`` column, not amino-acid sequences, and silently
        pollute peptide-prediction models, motif analyses, and length
        distributions. Pass ``False`` to retain them (e.g. for CD1 /
        MR1 lipid-antigen analyses).
    peptide, serotype, columns
        See module docstring.

    Raises
    ------
    FileNotFoundError
        If the observations table has not been built yet.
    """
    return _load_peptide_index(
        observations_path(),
        index_name="Observations",
        mhc_class=mhc_class,
        species=species,
        source=source,
        mhc_restriction=mhc_restriction,
        mhc_allele_in_set=mhc_allele_in_set,
        mhc_allele_provenance=mhc_allele_provenance,
        gene_name=gene_name,
        gene_id=gene_id,
        peptide=peptide,
        serotype=serotype,
        length_min=length_min,
        length_max=length_max,
        exclude_class_label_suspect=exclude_class_label_suspect,
        exclude_class_label_implausible=exclude_class_label_implausible,
        exclude_non_peptide_ligand=exclude_non_peptide_ligand,
        columns=columns,
    )


def load_ms_observations(
    mhc_class: str | None = None,
    species: str | None = None,
    source: str | None = None,
    mhc_restriction: str | list[str] | None = None,
    mhc_allele_in_set: str | list[str] | None = None,
    mhc_allele_provenance: str | list[str] | None = None,
    gene_name: str | list[str] | None = None,
    gene_id: str | list[str] | None = None,
    peptide: str | list[str] | None = None,
    serotype: str | list[str] | None = None,
    length_min: int | None = None,
    length_max: int | None = None,
    exclude_class_label_suspect: bool = False,
    exclude_class_label_implausible: bool = False,
    exclude_non_peptide_ligand: bool = True,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Alias for :func:`load_observations` with modality explicit in the name."""
    return load_observations(
        mhc_class=mhc_class,
        species=species,
        source=source,
        mhc_restriction=mhc_restriction,
        mhc_allele_in_set=mhc_allele_in_set,
        mhc_allele_provenance=mhc_allele_provenance,
        gene_name=gene_name,
        gene_id=gene_id,
        peptide=peptide,
        serotype=serotype,
        length_min=length_min,
        length_max=length_max,
        exclude_class_label_suspect=exclude_class_label_suspect,
        exclude_class_label_implausible=exclude_class_label_implausible,
        exclude_non_peptide_ligand=exclude_non_peptide_ligand,
        columns=columns,
    )


def load_binding(
    mhc_class: str | None = None,
    species: str | None = None,
    source: str | None = None,
    mhc_restriction: str | list[str] | None = None,
    mhc_allele_in_set: str | list[str] | None = None,
    mhc_allele_provenance: str | list[str] | None = None,
    gene_name: str | list[str] | None = None,
    gene_id: str | list[str] | None = None,
    peptide: str | list[str] | None = None,
    serotype: str | list[str] | None = None,
    length_min: int | None = None,
    length_max: int | None = None,
    exclude_class_label_suspect: bool = False,
    exclude_class_label_implausible: bool = False,
    exclude_non_peptide_ligand: bool = True,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Load the built binding-assay table with optional filters.

    The binding index contains rows flagged as binding assays (peptide
    microarray, refolding, MEDi, quantitative-tier measurements).
    Supplementary data never contributes here — all supplementary
    rows are manually curated as MS.

    Filters match :func:`load_observations`.  Raises FileNotFoundError
    if the binding index has not been built yet.
    """
    return _load_peptide_index(
        binding_path(),
        index_name="Binding",
        mhc_class=mhc_class,
        species=species,
        source=source,
        mhc_restriction=mhc_restriction,
        mhc_allele_in_set=mhc_allele_in_set,
        mhc_allele_provenance=mhc_allele_provenance,
        gene_name=gene_name,
        gene_id=gene_id,
        peptide=peptide,
        serotype=serotype,
        length_min=length_min,
        length_max=length_max,
        exclude_class_label_suspect=exclude_class_label_suspect,
        exclude_class_label_implausible=exclude_class_label_implausible,
        exclude_non_peptide_ligand=exclude_non_peptide_ligand,
        columns=columns,
    )


def load_all_evidence(
    mhc_class: str | None = None,
    species: str | None = None,
    source: str | None = None,
    mhc_restriction: str | list[str] | None = None,
    mhc_allele_in_set: str | list[str] | None = None,
    mhc_allele_provenance: str | list[str] | None = None,
    gene_name: str | list[str] | None = None,
    gene_id: str | list[str] | None = None,
    peptide: str | list[str] | None = None,
    serotype: str | list[str] | None = None,
    length_min: int | None = None,
    length_max: int | None = None,
    exclude_class_label_suspect: bool = False,
    exclude_class_label_implausible: bool = False,
    exclude_non_peptide_ligand: bool = True,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Union of MS observations + binding assays with an ``evidence_kind`` column.

    Applies the same filters to both indexes, tags each row with
    ``evidence_kind ∈ {"ms", "binding"}``, and concatenates.  Missing
    indexes are silently skipped — the result is whatever has been built
    (both, one, or empty).

    Filter semantics match :func:`load_observations`.  Column projection
    via ``columns=`` will always also include ``evidence_kind`` in the
    output, even if not listed, so downstream consumers can always tell
    the two row populations apart.

    Returns
    -------
    pd.DataFrame
        Concatenated frame.  Empty with an ``evidence_kind`` column when
        neither index has been built.
    """
    kwargs = {
        "mhc_class": mhc_class,
        "species": species,
        "source": source,
        "mhc_restriction": mhc_restriction,
        "mhc_allele_in_set": mhc_allele_in_set,
        "mhc_allele_provenance": mhc_allele_provenance,
        "gene_name": gene_name,
        "gene_id": gene_id,
        "peptide": peptide,
        "serotype": serotype,
        "length_min": length_min,
        "length_max": length_max,
        "exclude_class_label_suspect": exclude_class_label_suspect,
        "exclude_class_label_implausible": exclude_class_label_implausible,
        "exclude_non_peptide_ligand": exclude_non_peptide_ligand,
        "columns": columns,
    }

    parts: list[pd.DataFrame] = []
    if is_built():
        obs = load_observations(**kwargs)
        obs["evidence_kind"] = "ms"
        parts.append(obs)
    if is_binding_built():
        binding = load_binding(**kwargs)
        binding["evidence_kind"] = "binding"
        parts.append(binding)

    if not parts:
        return pd.DataFrame({"evidence_kind": pd.Series(dtype=str)})
    return pd.concat(parts, ignore_index=True, sort=False)


# Columns added at load time, not stored in the parquet. The map records
# which underlying parquet columns each derived column depends on so a
# caller-supplied ``columns=[...]`` projection can pull the deps in
# (otherwise pyarrow rejects the pushdown with "No match for FieldRef").
_DERIVED_COLUMN_DEPS: dict[str, tuple[str, ...]] = {
    "mhc_class_label_suspect": ("mhc_class", "peptide"),
    "mhc_class_label_severity": ("mhc_class", "peptide"),
    # Stored at scan time post-#228, but recomputable from
    # ``mhc_restriction`` so caller projections work on stale parquets.
    "is_non_peptide_ligand": ("mhc_restriction",),
}


def _load_peptide_index(
    path: Path,
    *,
    index_name: str,
    mhc_class: str | None,
    species: str | None,
    source: str | None,
    mhc_restriction: str | list[str] | None,
    mhc_allele_in_set: str | list[str] | None,
    mhc_allele_provenance: str | list[str] | None,
    gene_name: str | list[str] | None,
    gene_id: str | list[str] | None,
    peptide: str | list[str] | None,
    serotype: str | list[str] | None,
    length_min: int | None,
    length_max: int | None,
    exclude_class_label_suspect: bool,
    exclude_class_label_implausible: bool,
    exclude_non_peptide_ligand: bool,
    columns: list[str] | None,
) -> pd.DataFrame:
    """Shared loader for the observations and binding parquets.

    Both indexes share the same schema; this helper centralizes filter
    pushdown, gene resolution via the mappings sidecar, and the
    semicolon-joined ``serotypes`` post-filter.
    """
    if not path.exists():
        raise FileNotFoundError(f"{index_name} table not built. Run: hitlist data build")

    from .curation import normalize_allele, normalize_species

    def _as_list(v) -> list[str]:
        if isinstance(v, str):
            return [s.strip() for s in v.split(",") if s.strip()]
        return [s for s in v if s]

    filters: list = []
    if mhc_class is not None:
        filters.append(("mhc_class", "==", mhc_class))
    if species is not None:
        filters.append(("mhc_species", "==", normalize_species(species)))
    if source is not None:
        filters.append(("source", "==", source))
    if mhc_restriction is not None:
        # Set-membership match (#45): post-#45, ``mhc_restriction`` may
        # be a single 4-digit allele OR a semicolon-joined multi-allele
        # set (the donor's typed alleles when the per-peptide attribution
        # is multi-allelic).  A query for ``"HLA-A*02:01"`` should match
        # both.  We expand the wanted values to all stored restriction
        # strings whose ``;``-split tokens contain any wanted allele,
        # then push down the expanded list — keeps pyarrow's IN
        # predicate fast while honoring set-membership semantics.  The
        # unique-restriction set is cached per ``(path, mtime_ns, size)``
        # so we don't re-read the column on every call.
        wanted = {normalize_allele(v) for v in _as_list(mhc_restriction)} - {""}
        if not wanted:
            raise ValueError(
                "mhc_restriction filter received no usable allele values "
                "after normalization; pass at least one non-empty allele."
            )
        all_restrictions = _unique_restrictions_for(path)
        matching = [
            r
            for r in all_restrictions
            if r and (r in wanted or any(a in r.split(";") for a in wanted))
        ]
        if not matching:
            return pd.read_parquet(path, columns=columns, filters=[("peptide", "==", "__NONE__")])
        filters.append(("mhc_restriction", "in", matching))
    if peptide is not None:
        filters.append(("peptide", "in", _as_list(peptide)))
    if mhc_allele_provenance is not None:
        filters.append(("mhc_allele_provenance", "in", _as_list(mhc_allele_provenance)))

    if gene_name is not None or gene_id is not None:
        schema_names = set(pq.read_schema(path).names)
        if "gene_names" not in schema_names:
            raise ValueError(
                "Gene filtering requires a mappings-built index.\nRun: hitlist data build"
            )
        from .mappings import is_mappings_built, load_peptide_mappings

        if not is_mappings_built():
            raise ValueError("Peptide mappings not built.  Run: hitlist data build")
        mapping_filters: dict = {}
        if gene_name is not None:
            mapping_filters["gene_name"] = _as_list(gene_name)
        if gene_id is not None:
            mapping_filters["gene_id"] = _as_list(gene_id)
        hits = load_peptide_mappings(columns=["peptide"], **mapping_filters)
        matching_peptides = hits["peptide"].unique().tolist()
        if not matching_peptides:
            return pd.read_parquet(path, columns=columns, filters=[("peptide", "==", "__NONE__")])
        filters.append(("peptide", "in", matching_peptides))

    # Serotype filter runs after load — `serotypes` is a semicolon-joined
    # string column (an allele may belong to a locus-specific serotype AND
    # a public epitope like Bw4), so parquet pushdown can't express it.
    post_serotypes: list[str] | None = None
    if serotype is not None:
        post_serotypes = [_normalize_serotype_query(s) for s in _as_list(serotype)]
        if columns is not None and "serotypes" not in columns:
            read_columns = [*columns, "serotypes"]
        else:
            read_columns = columns

        schema_names = set(pq.read_schema(path).names)
        if "serotypes" not in schema_names:
            raise ValueError(
                "Serotype filtering requires an index built with\n"
                "hitlist >= 1.7.0.  Run: hitlist data build --force"
            )
    else:
        read_columns = columns

    # Derived columns (computed at load time, not stored in the parquet) need
    # special handling when the caller projects with ``columns=[...]``: they
    # must be stripped from the pushdown list (else pyarrow raises "No match
    # for FieldRef.Name(...)") and replaced with their underlying inputs so
    # the post-load step can compute them.
    requested_derived: list[str] = []
    if read_columns is not None:
        kept: list[str] = []
        for c in read_columns:
            if c in _DERIVED_COLUMN_DEPS:
                requested_derived.append(c)
                for dep in _DERIVED_COLUMN_DEPS[c]:
                    if dep not in kept:
                        kept.append(dep)
            elif c not in kept:
                kept.append(c)
        # The exclude_class_label_* filters need the same deps whether
        # or not the caller explicitly projected the derived flags.
        if exclude_class_label_suspect or exclude_class_label_implausible:
            for dep in _DERIVED_COLUMN_DEPS["mhc_class_label_suspect"]:
                if dep not in kept:
                    kept.append(dep)
        if exclude_non_peptide_ligand:
            for dep in _DERIVED_COLUMN_DEPS["is_non_peptide_ligand"]:
                if dep not in kept:
                    kept.append(dep)
        read_columns = kept

    df = pd.read_parquet(path, columns=read_columns, filters=filters if filters else None)

    if post_serotypes:
        wanted = set(post_serotypes)
        mask = df["serotypes"].map(
            lambda s: bool(wanted & set(s.split(";"))) if isinstance(s, str) and s else False
        )
        df = df[mask]
        if columns is not None and "serotypes" not in columns:
            df = df.drop(columns=["serotypes"])

    # Set-membership filter (#45 / #137).  ``mhc_allele_set`` is a
    # ``;``-joined string so parquet pushdown can't express the filter;
    # apply post-load.  Cheap on a filtered frame (the heavy filters
    # above already shrunk the row count).
    if mhc_allele_in_set is not None and "mhc_allele_set" in df.columns:
        # Vectorized set-membership: pad each cell with leading/trailing
        # ``;`` and substring-match ``;<allele>;``.  Anchors prevent
        # ``HLA-A*02`` from matching ``HLA-A*02:01``.  ``str.contains`` runs
        # in C; one pass per wanted allele beats a per-row Python apply
        # for low-selectivity queries on millions of rows.
        wanted_set = {normalize_allele(a.strip()) for a in _as_list(mhc_allele_in_set)} - {""}
        if not wanted_set:
            raise ValueError(
                "mhc_allele_in_set filter received no usable allele values "
                "after normalization; pass at least one non-empty allele."
            )
        padded = ";" + df["mhc_allele_set"].fillna("").astype(str) + ";"
        mask = pd.Series(False, index=df.index)
        for allele in wanted_set:
            mask |= padded.str.contains(f";{re.escape(allele)};", regex=True)
        df = df[mask]

    # Length bounds (#118). observations.parquet / binding.parquet don't
    # carry an explicit length column — we compute it from the peptide
    # string on read. Post-load filter because parquet pushdown doesn't
    # apply to derived expressions; for the full 4.4M-row observations
    # parquet this costs ~100 ms of str.len on the final frame, which is
    # small relative to the read.
    if length_min is not None or length_max is not None:
        if "peptide" not in df.columns:
            raise ValueError(
                "length_min/length_max require the 'peptide' column; "
                "include it in columns= if projecting."
            )
        lo = length_min if length_min is not None else -1
        hi = length_max if length_max is not None else 10**9
        df = df[df["peptide"].str.len().between(lo, hi)]

    # ── Backstop: normalize ``mhc_restriction`` strings (#181) ────────────
    # Stale parquets built before the supplement-side ``normalize_allele``
    # call (supplement.py:118) carry unprefixed forms like ``A*02:01``
    # alongside canonical ``HLA-A*02:01`` for the same allele. Normalizing
    # at load time guarantees downstream groupbys / filters / sample-allele
    # joins see one canonical string per allele without forcing every
    # consumer to rebuild the parquet. Unique-map over ~hundreds of unique
    # values keeps the per-call cost sub-second on the full 4.4M-row index.
    if "mhc_restriction" in df.columns and len(df) > 0:
        uniq = df["mhc_restriction"].dropna().unique()
        if len(uniq) > 0:
            norm_map = {str(a): normalize_allele(a) for a in uniq}
            # Cast to ``StringDtype`` before map/fillna — categorical
            # ``mhc_restriction`` (post-#137) rejects assignments outside
            # its category set, which would silently break this normalization
            # path on old-schema parquets.  Round-trip back to whatever
            # dtype pandas chooses for the assigned column.
            normalized = df["mhc_restriction"].astype("string")
            df["mhc_restriction"] = normalized.map(norm_map).fillna(normalized)

    # ── MHC class-label severity tiers (#182, #201) ──────────────────────
    # Flags rows whose curated ``mhc_class`` disagrees with the
    # peptide's length, since IEDB occasionally mislabels class. Four
    # tiers per row, computed off the bare peptide length:
    #
    #             ok            borderline    suspect     implausible
    #   class I   8-12          13-14         15-17       ≥18 or ≤7
    #   class II  11-44         8-10          5-7         ≥45 or ≤4
    #
    # Borderline = uncommon-but-real biology (bulged class-I, short
    # class-II). Implausible = almost certainly curation drift; cutoffs
    # set off the empirical break in Stražar 2023's HLA-II
    # immunopeptidome which extends to ~51 aa.
    #
    # ``mhc_class_label_suspect`` is the backwards-compatible binary
    # flag — equals ``severity in {"suspect", "implausible"}``.
    # Callers wanting only the strict drift filter use the
    # ``exclude_class_label_implausible`` loader parameter.
    if "mhc_class" in df.columns and "peptide" in df.columns and len(df) > 0:
        # Strip IEDB inline PTM annotation before measuring length.
        # Pre-v1.30.10 parquets may carry "LQPFPQPQLPY + DEAM(Q8)" in the
        # peptide column; the bare-length split keeps the severity tier
        # honest on those without affecting v1.30.10+ rows where
        # ``peptide`` is already the bare sequence. ``regex=False`` is
        # required — pandas otherwise reads ``+`` as a regex quantifier
        # and the split silently no-ops.
        plen = df["peptide"].astype(str).str.split(" + ", n=1, regex=False).str[0].str.len()
        # ``mhc_class`` is post-#137 categorical; fillna with ``""`` requires
        # the value to already be in the category set, which it generally
        # isn't (categories are usually ``{"I", "II", "non classical"}``).
        # Cast to plain ``StringDtype`` first — accepts any string fill.
        cls = df["mhc_class"].astype("string").fillna("")

        # Default everything to "ok"; refine downward.
        severity = pd.Series("ok", index=df.index, dtype="object")

        # Class I tiers (canonical 8-12).
        cls_i = cls == "I"
        severity[cls_i & (plen.between(13, 14))] = "borderline"
        severity[cls_i & (plen.between(15, 17))] = "suspect"
        severity[cls_i & (plen >= 18)] = "implausible"
        severity[cls_i & (plen <= 7)] = "implausible"

        # Class II tiers (canonical 11-30).
        cls_ii = cls == "II"
        severity[cls_ii & (plen.between(8, 10))] = "borderline"
        severity[cls_ii & (plen.between(5, 7))] = "suspect"
        severity[cls_ii & (plen <= 4)] = "implausible"
        severity[cls_ii & (plen >= 45)] = "implausible"

        df["mhc_class_label_severity"] = severity
        # Backwards-compatible binary flag — same semantics as v1.30.0:
        # any row that's worse than "borderline".
        df["mhc_class_label_suspect"] = severity.isin({"suspect", "implausible"})

    # Drop rows whose curated class disagrees with the bimodal length
    # distribution (#182). One-line opt-in for training pipelines that
    # want clean class-conditioned inputs without re-deriving the same
    # check.
    if exclude_class_label_suspect and "mhc_class_label_suspect" in df.columns:
        df = df[~df["mhc_class_label_suspect"]]
    if exclude_class_label_implausible and "mhc_class_label_severity" in df.columns:
        df = df[df["mhc_class_label_severity"] != "implausible"]

    # ── Non-peptide-presenting MHC molecules (#228) ───────────────────────
    # CD1 / MR1 / MIC / ULBP / RAET1 / NKG2[A-C] / HFE present lipids,
    # metabolites, or stress ligands rather than peptides; default-exclude
    # so peptide consumers don't ingest IEDB's chemical-name / compound-id
    # strings. Always materialize the column (cheap unique-allele map) so
    # ``columns=`` projections work and stale parquets stay correct.
    # Derived again at scan time and in :func:`_apply_training_defaults`
    # — same regex everywhere, redundancy is intentional.
    if "mhc_restriction" in df.columns and len(df) > 0:
        from .curation import is_non_peptide_ligand

        if "is_non_peptide_ligand" not in df.columns:
            uniq = df["mhc_restriction"].dropna().unique()
            flag_map = {str(a): is_non_peptide_ligand(a) for a in uniq}
            df["is_non_peptide_ligand"] = (
                df["mhc_restriction"].map(flag_map).fillna(False).astype(bool)
            )
        else:
            df["is_non_peptide_ligand"] = df["is_non_peptide_ligand"].astype(bool)
        if exclude_non_peptide_ligand:
            df = df[~df["is_non_peptide_ligand"]]

    # If the caller explicitly projected, trim back to that exact list now
    # — derived columns pulled extra dependency columns into the read above
    # and the caller doesn't want those leaking into the result.
    if columns is not None:
        df = df[[c for c in columns if c in df.columns]]

    return df


def _normalize_serotype_query(raw: str) -> str:
    """Normalize user serotype input to canonical ``HLA-X`` form.

    Accepts ``A24``, ``HLA-A24``, ``hla-a24``, ``Bw4``, etc.
    """
    s = raw.strip()
    if not s:
        return ""
    if s.upper().startswith("HLA-"):
        return "HLA-" + s[4:]
    return f"HLA-{s}"
