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
  CEDAR + supplementary).  Load with :func:`load_observations`, or the
  modality-explicit alias :func:`load_ms_observations`.
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

Species axes (#46 / #306)
-------------------------
A pMHC observation has THREE independent species axes — keep them straight:

================  =========================================================
``mhc_species``   Species of the MHC molecule, derived authoritatively from
                  the allele string (``HLA-*`` → Homo sapiens, ``H2-*`` →
                  Mus musculus, ``DLA-*`` → Canis, ...).  Filter: ``species=``.
``source_species``  **Canonical** species of the source proteome the peptide
                  was sequenced from.  Normalized; coalesces the two raw IEDB
                  inputs below.  Filter: ``source_species=``.  PREFER THIS.
``host_organism`` Species the cells/tissue lived in at MS sampling (normalized
                  from raw ``host``).  Filter: ``host_species=``.
================  =========================================================

The source-proteome axis has two **raw** IEDB inputs that ``source_species``
normalizes over: ``source_organism`` ("Source Organism", strain-level, e.g.
``Mus musculus C57BL/6``) and ``species`` ("Epitope | Species", species-rank,
e.g. ``Mus musculus``).  They describe the *same* axis at different
granularity.  **``species`` is a deprecated/legacy column name** — it collides
with the English "which species" and is kept only for backward compatibility
(the species-summary export still emits it).  New code should read
``source_species`` (or the raw ``source_organism``), never bare ``species``.
When the source proteome and the MHC come from different vertebrate genera the
row is *chimeric* — see ``is_chimeric`` / ``is_engineered_mhc`` / ``xenograft``
and ``exclude_chimeric=``.
"""

from __future__ import annotations

import os
import re
from functools import WRAPPER_ASSIGNMENTS, lru_cache, wraps
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
    source_species: str | list[str] | None = None,
    host_species: str | list[str] | None = None,
    exclude_chimeric: bool = False,
    source: str | None = None,
    mhc_restriction: str | list[str] | None = None,
    mhc_allele_in_set: str | list[str] | None = None,
    mhc_allele_provenance: str | list[str] | None = None,
    restriction_evidence: str | list[str] | None = None,
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
        Filter by **MHC** species (e.g. ``"Homo sapiens"``) — peptides on a
        human MHC molecule, engineered or native. One of the three species
        axes (#46) that the legacy single ``species`` flag conflated; pair
        with ``source_species`` / ``host_species`` to disambiguate.
    source_species
        Filter by **proteome** species — peptides sequenced from a given
        organism's proteins (``source_organism``), regardless of MHC. In a
        chimeric system (e.g. dog tumor expressing human HLA) this is the dog,
        while ``species`` is human. Accepts a string or list.
    host_species
        Filter by **host** species — the organism the cells/tissue lived in at
        MS sampling (``host``). Distinguishes a native human sample from a
        human-tumor xenograft grown in a mouse. Accepts a string or list.
    exclude_chimeric
        When True, drop rows where the source proteome and the MHC come from
        different vertebrate-host genera (engineered-MHC / xenograft systems —
        HLA-transgenic rats, allogeneic-HLA transfectants, NetH2pan training).
        Use for high-confidence fully-isogenic training data. Default False.
        See ``is_chimeric`` / ``is_engineered_mhc`` / ``xenograft`` (the
        load-time-derived axis columns, projectable via ``columns=``).
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
    restriction_evidence
        Filter independently by how the named peptide-to-MHC restriction was
        established: ``"experimental"``, ``"monoallelic"``, ``"predicted"``,
        or ``"unknown"``. Unlike ``mhc_allele_provenance``, this axis describes
        evidentiary strength rather than where the candidate allele set came
        from.
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
        source_species=source_species,
        host_species=host_species,
        exclude_chimeric=exclude_chimeric,
        source=source,
        mhc_restriction=mhc_restriction,
        mhc_allele_in_set=mhc_allele_in_set,
        mhc_allele_provenance=mhc_allele_provenance,
        restriction_evidence=restriction_evidence,
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


#: Everything ``wraps`` normally copies except the identity fields: the
#: alias must keep reporting its own name, or ``help()``, tracebacks,
#: profiler rows and Sphinx all label it ``load_observations``.
_ALIAS_ASSIGNMENTS = tuple(
    a for a in WRAPPER_ASSIGNMENTS if a not in ("__name__", "__qualname__", "__doc__")
)


@wraps(load_observations, assigned=_ALIAS_ASSIGNMENTS, updated=())
def load_ms_observations(*args, **kwargs) -> pd.DataFrame:
    return load_observations(*args, **kwargs)


load_ms_observations.__doc__ = (
    "Alias for :func:`load_observations` with the modality explicit in "
    "the name.\n\n    Delegates rather than restating the signature: it "
    "previously re-declared all 20 filter parameters and hand-forwarded "
    "each one, so adding a filter to ``load_observations`` and forgetting "
    "this copy would silently drop it. ``functools.wraps`` keeps the full "
    "signature visible to ``inspect.signature`` and to IDEs.\n    "
)


def load_binding(
    mhc_class: str | None = None,
    species: str | None = None,
    source_species: str | list[str] | None = None,
    host_species: str | list[str] | None = None,
    exclude_chimeric: bool = False,
    source: str | None = None,
    mhc_restriction: str | list[str] | None = None,
    mhc_allele_in_set: str | list[str] | None = None,
    mhc_allele_provenance: str | list[str] | None = None,
    restriction_evidence: str | list[str] | None = None,
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
        source_species=source_species,
        host_species=host_species,
        exclude_chimeric=exclude_chimeric,
        source=source,
        mhc_restriction=mhc_restriction,
        mhc_allele_in_set=mhc_allele_in_set,
        mhc_allele_provenance=mhc_allele_provenance,
        restriction_evidence=restriction_evidence,
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
    source_species: str | list[str] | None = None,
    host_species: str | list[str] | None = None,
    exclude_chimeric: bool = False,
    source: str | None = None,
    mhc_restriction: str | list[str] | None = None,
    mhc_allele_in_set: str | list[str] | None = None,
    mhc_allele_provenance: str | list[str] | None = None,
    restriction_evidence: str | list[str] | None = None,
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
        "source_species": source_species,
        "host_species": host_species,
        "exclude_chimeric": exclude_chimeric,
        "source": source,
        "mhc_restriction": mhc_restriction,
        "mhc_allele_in_set": mhc_allele_in_set,
        "mhc_allele_provenance": mhc_allele_provenance,
        "restriction_evidence": restriction_evidence,
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
    # Post-#238 these columns are NO LONGER stored on observations.parquet.
    # ``load_observations`` joins them on demand from peptide_mappings.parquet
    # using ``peptide`` as the join key.  Pre-#238 parquets that still carry
    # the columns pass through unchanged.
    "gene_names": ("peptide",),
    "gene_ids": ("peptide",),
    "protein_ids": ("peptide",),
    "n_source_proteins": ("peptide",),
    # Multi-axis species columns (#46). Derived at load time from the
    # already-stored host / source_organism / mhc_species so they work on
    # parquets built before this schema, mirroring is_non_peptide_ligand.
    "host_organism": ("host",),
    "source_species": ("source_organism", "species"),
    "is_chimeric": ("source_organism", "mhc_species"),
    "is_engineered_mhc": ("source_organism", "mhc_species", "host"),
    "xenograft": ("source_organism", "host", "mhc_species"),
}

#: The #46 species-axis columns derived together by ``_attach_species_axes``.
_SPECIES_AXIS_COLUMNS: tuple[str, ...] = (
    "host_organism",
    "source_species",
    "is_chimeric",
    "is_engineered_mhc",
    "xenograft",
)


def _attach_species_axes(df: pd.DataFrame) -> pd.DataFrame:
    """Add the #46 multi-axis species columns to ``df`` in place.

    Derives, when absent, the normalized ``host_organism`` / ``source_species``
    binomials and the ``is_chimeric`` / ``is_engineered_mhc`` / ``xenograft``
    booleans from the stored ``host`` / ``source_organism`` / ``mhc_species``
    columns. Uses unique-value maps so the per-call cost stays sub-second on
    the full index (only a few hundred distinct organism tuples). Columns that
    already exist (e.g. on a future parquet that materializes them) are left
    untouched. Missing input columns yield empty / ``False`` axis columns.
    """
    from .curation import (
        is_chimeric_system,
        is_engineered_mhc,
        is_xenograft,
        normalize_species,
    )

    # fillna("") BEFORE astype(str): a bare .astype(str) turns NaN into the
    # literal "nan"/"None", which then survives the .dropna()/.fillna("") below
    # and gets normalized into a phantom species value instead of blank.
    host = (
        df["host"].fillna("").astype(str) if "host" in df.columns else pd.Series("", index=df.index)
    )
    src = (
        df["source_organism"].fillna("").astype(str)
        if "source_organism" in df.columns
        else pd.Series("", index=df.index)
    )
    mhc = (
        df["mhc_species"].fillna("").astype(str)
        if "mhc_species" in df.columns
        else pd.Series("", index=df.index)
    )
    # The source-proteome axis lives in two IEDB columns (#306): source_organism
    # (strain-level) and species (species-rank).  Coalesce so source_species is
    # resolved when EITHER is populated.
    spc = (
        df["species"].fillna("").astype(str)
        if "species" in df.columns
        else pd.Series("", index=df.index)
    )

    if "host_organism" not in df.columns:
        uniq = host.dropna().unique()
        m = {h: normalize_species(h) for h in uniq}
        df["host_organism"] = host.map(m).fillna("")
    if "source_species" not in df.columns:
        uniq = set(src.dropna().unique()) | set(spc.dropna().unique())
        m = {s: normalize_species(s) for s in uniq}
        # Prefer source_organism's normalized form; fall back to species when
        # source_organism is blank/unresolved (#306 coalesce).
        from_src = src.map(m).fillna("")
        from_spc = spc.map(m).fillna("")
        df["source_species"] = from_src.where(from_src != "", from_spc)
    if "is_chimeric" not in df.columns:
        pairs = {(s, m) for s, m in zip(src, mhc)}
        flag = {p: is_chimeric_system(*p) for p in pairs}
        df["is_chimeric"] = [flag[(s, m)] for s, m in zip(src, mhc)]
    if "is_engineered_mhc" not in df.columns:
        triples = {(s, m, h) for s, m, h in zip(src, mhc, host)}
        flag = {t: is_engineered_mhc(*t) for t in triples}
        df["is_engineered_mhc"] = [flag[(s, m, h)] for s, m, h in zip(src, mhc, host)]
    if "xenograft" not in df.columns:
        triples = {(s, h, m) for s, h, m in zip(src, host, mhc)}
        flag = {t: is_xenograft(*t) for t in triples}
        df["xenograft"] = [flag[(s, h, m)] for s, h, m in zip(src, host, mhc)]

    for col in ("is_chimeric", "is_engineered_mhc", "xenograft"):
        df[col] = df[col].astype(bool)
    return df


def _load_peptide_index(
    path: Path,
    *,
    index_name: str,
    mhc_class: str | None,
    species: str | None,
    source_species: str | list[str] | None = None,
    host_species: str | list[str] | None = None,
    exclude_chimeric: bool = False,
    source: str | None,
    mhc_restriction: str | list[str] | None,
    mhc_allele_in_set: str | list[str] | None,
    mhc_allele_provenance: str | list[str] | None,
    restriction_evidence: str | list[str] | None,
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
        raise FileNotFoundError(f"{index_name} table not built. Run: hitlist build observations")

    from .curation import normalize_allele, normalize_species

    def _as_list(v) -> list[str]:
        if isinstance(v, str):
            return [s.strip() for s in v.split(",") if s.strip()]
        return [s for s in v if s]

    # Determine which of the caller's requested columns actually live in
    # the parquet vs are derived (computed at load time, e.g. post-#238
    # gene_names / gene_ids / protein_ids / n_source_proteins, or the
    # mhc_class_label_* family).  Use this to filter the read-side columns
    # list — passing a derived column to ``pd.read_parquet(columns=...)``
    # raises ``ArrowInvalid: No match for FieldRef.Name(...)``.
    parquet_columns = set(pq.read_schema(path).names)
    if columns is not None:
        columns_for_read = [c for c in columns if c in parquet_columns]
    else:
        columns_for_read = None

    filters: list = []
    if mhc_class is not None:
        # The parquet stores IEDB's spelling ("non classical") while the
        # canonical token is hyphenated, so an exact predicate on either
        # spelling returned nothing.  Match every stored spelling that
        # normalizes to the requested class (#363 follow-up).
        from .curation import mhc_class_spellings

        filters.append(("mhc_class", "in", mhc_class_spellings(mhc_class)))
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
            return pd.read_parquet(
                path, columns=columns_for_read, filters=[("peptide", "==", "__NONE__")]
            )
        filters.append(("mhc_restriction", "in", matching))
    if peptide is not None:
        filters.append(("peptide", "in", _as_list(peptide)))
    if mhc_allele_provenance is not None:
        filters.append(("mhc_allele_provenance", "in", _as_list(mhc_allele_provenance)))
    if restriction_evidence is not None:
        filters.append(("restriction_evidence", "in", _as_list(restriction_evidence)))

    if gene_name is not None or gene_id is not None:
        # Gene filters resolve to a peptide list via peptide_mappings.parquet,
        # then push down on the obs frame.  Post-#238, the obs parquet itself
        # no longer carries gene_names / gene_ids — peptide_mappings is the
        # authoritative source for both filtering AND for the auto-attached
        # gene_names / gene_ids columns at the bottom of this function.
        from .mappings import is_mappings_built, load_peptide_mappings

        if not is_mappings_built():
            raise FileNotFoundError(
                "Gene filtering requires peptide_mappings.parquet, which has not been built.\n"
                "Run: hitlist build observations"
            )
        mapping_filters: dict = {}
        if gene_name is not None:
            mapping_filters["gene_name"] = _as_list(gene_name)
        if gene_id is not None:
            mapping_filters["gene_id"] = _as_list(gene_id)
        hits = load_peptide_mappings(columns=["peptide"], **mapping_filters)
        matching_peptides = hits["peptide"].unique().tolist()
        if not matching_peptides:
            return pd.read_parquet(
                path, columns=columns_for_read, filters=[("peptide", "==", "__NONE__")]
            )
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
                "hitlist >= 1.7.0.  Run: hitlist build observations --force"
            )
    else:
        read_columns = columns

    # Derived columns (computed at load time, not stored in the parquet) need
    # special handling when the caller projects with ``columns=[...]``: they
    # must be stripped from the pushdown list (else pyarrow raises "No match
    # for FieldRef.Name(...)") and replaced with their underlying inputs so
    # the post-load step can compute them.  ``gene_names`` and friends are
    # in ``_DERIVED_COLUMN_DEPS`` post-#238 BUT pre-#238 parquets still
    # carry them on disk — read them directly when present, treat as
    # derived only when absent.  ``parquet_columns`` was computed at the
    # top of the function for the early-return paths.
    requested_derived: list[str] = []
    if read_columns is not None:
        kept: list[str] = []
        for c in read_columns:
            if c in _DERIVED_COLUMN_DEPS and c not in parquet_columns:
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
        # The #46 species-axis filters need host / source_organism /
        # mhc_species read even when the derived axis columns aren't projected.
        if exclude_chimeric or source_species is not None or host_species is not None:
            for col in ("host", "source_organism", "mhc_species"):
                if col not in kept:
                    kept.append(col)
        # The mhc_allele_in_set filter reads mhc_allele_set post-load; without
        # this the column may be unprojected and the filter SILENTLY skipped
        # (the guard below is `... in df.columns`), returning the full table.
        if mhc_allele_in_set is not None and "mhc_allele_set" not in kept:
            kept.append("mhc_allele_set")
        # Drop any requested column that isn't on this parquet (e.g.
        # ``cell_type`` / ``sample_match_type`` on a pre-v1.30.57 build).
        # Projecting a missing column into a *filtered* pyarrow scan raises
        # "No match for FieldRef.Name(...)"; absent columns simply won't
        # appear in the result, and callers that require one check
        # ``df.columns`` themselves (mirrors the no-filter path above).
        read_columns = [c for c in kept if c in parquet_columns]

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
        # Don't leak the helper column when the caller didn't project it
        # (mirrors the serotypes post-filter drop above).
        if columns is not None and "mhc_allele_set" not in columns:
            df = df.drop(columns=["mhc_allele_set"])

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

    # ── Multi-axis species columns + filters (#46) ───────────────────────
    # Disambiguate the three species axes that the legacy single ``species``
    # (== mhc_species) filter conflated:
    #   species=        peptides on a given MHC species  (existing)
    #   source_species= peptides from a given proteome species
    #   host_species=   peptides observed in a given host species
    #   exclude_chimeric=True  drop engineered-MHC / xenograft rows
    # Materialize the derived axis columns when filtering OR when the caller
    # projected any of them; compute from host / source_organism / mhc_species
    # so the filters work on parquets built before this schema.
    want_axis_cols = columns is not None and any(c in columns for c in _SPECIES_AXIS_COLUMNS)
    if (
        exclude_chimeric or source_species is not None or host_species is not None or want_axis_cols
    ) and len(df) > 0:
        _attach_species_axes(df)
        if source_species is not None:
            wanted = {normalize_species(s) for s in _as_list(source_species)} - {""}
            df = df[df["source_species"].isin(wanted)]
        if host_species is not None:
            wanted = {normalize_species(s) for s in _as_list(host_species)} - {""}
            df = df[df["host_organism"].isin(wanted)]
        if exclude_chimeric:
            df = df[~df["is_chimeric"]]
        # Drop axis columns the caller didn't explicitly request.
        if columns is not None:
            drop = [c for c in _SPECIES_AXIS_COLUMNS if c not in columns and c in df.columns]
            if drop:
                df = df.drop(columns=drop)

    # ── Auto-attach gene/protein columns from peptide_mappings (#238) ────
    # Post-v1.30.46 builds elide ``gene_names`` / ``gene_ids`` /
    # ``protein_ids`` / ``n_source_proteins`` from observations.parquet —
    # the same data lives in peptide_mappings.parquet (one row per
    # peptide x protein) and was previously denormalized at build time.
    # When a caller EXPLICITLY requests one of those four columns via
    # ``columns=[...]`` and the parquet doesn't carry it (post-v1.30.46),
    # join on demand against the matched-peptides slice of peptide_mappings.
    # Pre-v1.30.46 parquets that still carry the columns pass through
    # unchanged — no double-join.
    #
    # Auto-attach only fires when the caller named the column in
    # ``columns=[...]``.  No-projection loads (``columns=None``) get the
    # parquet columns as-is — avoids forcing a peptide_mappings dependency
    # on every full load (which would break test fixtures and any consumer
    # that doesn't actually need the gene columns).  Callers that DO need
    # the gene columns post-#238 must list them explicitly in ``columns=``.
    _GENE_DERIVED = {"gene_names", "gene_ids", "protein_ids", "n_source_proteins"}
    requested_gene_cols = _GENE_DERIVED & set(columns) if columns is not None else set()
    missing_gene_cols = requested_gene_cols - set(df.columns)
    if missing_gene_cols and "peptide" in df.columns and len(df) > 0:
        from .mappings import (
            annotate_observations_with_genes,
            is_mappings_built,
            load_peptide_mappings,
        )

        if not is_mappings_built():
            raise FileNotFoundError(
                f"Loading {sorted(missing_gene_cols)} requires peptide_mappings.parquet, "
                "which has not been built.  Run: hitlist build observations"
            )
        unique_peptides = df["peptide"].dropna().unique().tolist()
        if unique_peptides:
            mappings = load_peptide_mappings(
                peptide=unique_peptides,
                columns=["peptide", "gene_name", "gene_id", "protein_id"],
            )
            df = annotate_observations_with_genes(df, mappings)
        else:
            for col in ("gene_names", "gene_ids", "protein_ids"):
                df[col] = ""
            df["n_source_proteins"] = 0

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
