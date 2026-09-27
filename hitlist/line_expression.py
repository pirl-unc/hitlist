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

"""Per-line RNA/transcript expression anchors for sample-aware exports.

``ms_samples`` curation in ``pmid_overrides.yaml`` names many line-like
human systems (C1R, 721.221, T2, JY and sibling donor EBV-LCLs, HAP1,
HeLa, HEK293, THP-1, SaOS-2, A375, ...) that do not map cleanly onto the
tumor/normal-tissue references used for whole-organism CTA work.

This module adds a dedicated expression-anchor layer so a downstream
sample-aware export can attach per-sample RNA evidence to every peptide
observation and, crucially, preserve *how trustworthy* that evidence is
via explicit provenance columns.

Three building blocks live here:

1. **Registry** — ``line_expression_anchors.yaml`` (per-line backend +
   aliases + parent-line + family). Parsed by
   :func:`load_line_expression_anchors`.

2. **Resolver** — :func:`resolve_sample_expression_anchor` applies the
   6-tier fallback hierarchy from issue #140:

   ===  =================================================  =====================
   1    exact line RNA / transcript quant                   registry hit w/ data
   2    parent-line or engineered-derivative RNA            follow ``parent_line``
   3    line-family stand-in (EBV-LCL → GM12878,            ``line_family``
        mono-allelic host → K562)
   4    cancer-type surrogate (pirlygenes)                  caller-supplied backend
   5    broad tissue / lineage surrogate (HPA)              ``lineage_tissue``
   6    no_expression_anchor                                tier-6 sentinel
   ===  =================================================  =====================

   Returns a :class:`SampleExpressionAnchor` carrying
   ``expression_backend``, ``expression_key``, ``expression_match_tier``,
   and ``expression_parent_key`` — the provenance contract from #140.

3. **Data** — ``line_expression.parquet`` in ``~/.hitlist/`` (built by
   :func:`hitlist.builder.build_line_expression`).  Without an index, reads
   use the packaged CSVs under ``hitlist/data/line_expression/`` enriched by
   :func:`enrich_line_expression_rows` to the same row shape a build writes,
   so pure-registry workflows work out-of-the-box.

   :func:`write_line_expression_index` stamps the index with the packaged
   build inputs' fingerprint and :data:`LINE_EXPRESSION_BUILD_VERSION`.  An
   index stamped by another build version, or not at all, is not used; one
   that differs only in packaged inputs contributes its downloaded rows for
   (line, source) pairs the current registry lists, alongside this release's
   packaged rows.  Either way the first read warns once, saying what was
   left out; reads never rebuild (#577).

Typical usage::

    from hitlist.line_expression import resolve_sample_expression_anchor

    anchor = resolve_sample_expression_anchor("JY (EBV-LCL)", pmid=28099872)
    # → SampleExpressionAnchor(expression_backend='packaged_rnaseq',
    #                          expression_key='JY',
    #                          expression_match_tier=1,
    #                          expression_parent_key=None, ...)

    from hitlist.line_expression import load_line_expression

    tpm = load_line_expression(line_key=anchor.expression_key,
                               gene_name=['TP53', 'MYC'])
"""

from __future__ import annotations

import hashlib
import json
import warnings
from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
from pathlib import Path
from typing import Callable

import pandas as pd
from pyarrow.lib import ArrowInvalid

from .curation_yaml import load_curation_yaml
from .parquet_io import atomic_write_parquet

_DATA_MODULE = "hitlist.data.line_expression"
_ANCHORS_RESOURCE = "hitlist.data"
_ANCHORS_FILE = "line_expression_anchors.yaml"

#: Column order of ``line_expression.parquet``: the canonical row every reader
#: sees, whether it came from a built index or from the packaged sources.
LINE_EXPRESSION_COLUMNS = (
    "backend",
    "source_id",
    "pmid",
    "reference",
    "study_label",
    "line_key",
    "parent_line_key",
    "granularity",
    "gene_id",
    "gene_name",
    "transcript_id",
    "profile_id",
    "tpm",
    "log2_tpm",
    "normalization",
    "quantifier",
    "species",
    "license",
)

#: Version of the build logic and output schema of ``line_expression.parquet``.
#: Bump it whenever :func:`hitlist.builder.build_line_expression` or
#: :func:`enrich_line_expression_rows` changes what a row holds or how it is
#: derived: an index stamped with another version is not used at all, while
#: one that differs only in packaged inputs keeps its downloaded rows (#577).
LINE_EXPRESSION_BUILD_VERSION = 1

#: Parquet key-value metadata key holding :func:`line_expression_index_stamp`
#: as JSON.  It travels inside the file, so a copied or downloaded index (the
#: CI corpus, a colleague's build) carries its own provenance.
INDEX_STAMP_METADATA_KEY = b"hitlist_line_expression_index"

#: Packaged ENSG -> HGNC symbol map that fills blank ``gene_name`` values.
#: Pre-resolved offline (Ensembl 75 -> 112 -> 90, mygene.info for the tail), so
#: filling needs no pyensembl release data — which a read path must not need.
_GENE_NAME_MAP_FILE = "ensembl_gene_id_symbol.csv"

# Tier-3 class anchors — mapped by ``line_family``.  Keys MUST exist as a
# ``name`` entry in the registry so downstream resolution has a target.
_LINE_FAMILY_CLASS_ANCHOR: dict[str, str] = {
    "ebv_lcl": "GM12878",
    "mono_allelic_host": "K562",
}


# ── Paths ───────────────────────────────────────────────────────────────────


def line_expression_path() -> Path:
    """Path to the built ``line_expression.parquet`` in ``data_dir()``."""
    from .downloads import data_dir

    return data_dir() / "line_expression.parquet"


def is_line_expression_built() -> bool:
    """Check whether the line expression parquet has been built."""
    return line_expression_path().exists()


# ── YAML loaders ────────────────────────────────────────────────────────────


@lru_cache(maxsize=1)
def _load_anchors_yaml() -> list[dict]:
    data = load_curation_yaml(files(_ANCHORS_RESOURCE) / _ANCHORS_FILE)
    return list(data.get("lines", []) or [])


@lru_cache(maxsize=1)
def _load_sources_yaml() -> list[dict]:
    data = load_curation_yaml(files(_DATA_MODULE) / "sources.yaml")
    return list(data.get("sources", []) or [])


def load_line_expression_anchors() -> list[dict]:
    """Return the line-expression anchor registry (one dict per line)."""
    # Defensive copy so callers mutating the dicts never corrupt the cache.
    return [dict(entry) for entry in _load_anchors_yaml()]


def load_line_expression_sources() -> list[dict]:
    """Return per-source metadata from ``line_expression/sources.yaml``."""
    return [dict(entry) for entry in _load_sources_yaml()]


@lru_cache(maxsize=1)
def _alias_to_expression_key() -> dict[str, str]:
    """Casefolded alias / canonical-name → ``expression_key`` lookup.

    Used by the builder to harmonize external line identifiers (e.g. DepMap
    ``StrippedCellLineName``) onto the stable keys the registry expects.
    Entries with ``expression_backend == "none"`` are skipped — they carry no
    data and shouldn't claim a ``line_key``.
    """
    m: dict[str, str] = {}
    for entry in _load_anchors_yaml():
        key = entry.get("expression_key")
        backend = entry.get("expression_backend") or ""
        if not key or backend == "none":
            continue
        for alias in entry.get("aliases") or []:
            a = str(alias).casefold().strip()
            if a:
                m.setdefault(a, str(key))
        name = entry.get("name")
        if name:
            m.setdefault(str(name).casefold().strip(), str(key))
            # Also map with hyphens/dots stripped so DepMap's
            # ``StrippedCellLineName`` (e.g. "SAOS2", "THP1") matches
            # registry names like "SaOS-2", "THP-1".
            stripped = "".join(c for c in str(name) if c.isalnum()).casefold()
            if stripped:
                m.setdefault(stripped, str(key))
    return m


def resolve_line_key(label: str) -> str | None:
    """Map an external label (e.g. DepMap name) to the registry ``expression_key``.

    Returns the stable ``expression_key`` when ``label`` matches a registered
    alias, the canonical line name, or the punctuation-stripped canonical
    name. ``None`` on no match. Case-insensitive.
    """
    if not label:
        return None
    lookup = _alias_to_expression_key()
    low = str(label).casefold().strip()
    if low in lookup:
        return lookup[low]
    stripped = "".join(c for c in low if c.isalnum())
    return lookup.get(stripped)


# ── Resolver ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SampleExpressionAnchor:
    """Outcome of :func:`resolve_sample_expression_anchor`.

    Attributes
    ----------
    expression_backend
        Backend identifier matching the registry (``depmap_rna``,
        ``packaged_rnaseq``, ``encode_rnaseq``, or ``none``).
    expression_key
        Key joined against the line-expression parquet's ``line_key``
        column.  Empty string when ``expression_backend == "none"``.
    expression_match_tier
        1 = exact line, 2 = parent-line, 3 = line-family class anchor,
        4 = cancer-type surrogate (pirlygenes), 5 = broad tissue
        surrogate (HPA), 6 = no anchor available.
    expression_parent_key
        Canonical line that actually provided the data when the match
        was a fallback (tier ≥ 2).  ``None`` at tier 1 and tier 6.
    source_ids
        Source IDs contributing the resolved line's data (from the
        anchor registry).  Empty when resolution is tier ≥ 4 or tier 6.
    reason
        Short human-readable explanation of why this tier fired.
    matched_alias
        The alias (substring) that matched the input label, if any.
    """

    expression_backend: str
    expression_key: str
    expression_match_tier: int
    expression_parent_key: str | None = None
    source_ids: tuple[str, ...] = ()
    reason: str = ""
    matched_alias: str | None = None


def _anchor_by_name(name: str) -> dict | None:
    if not name:
        return None
    for entry in _load_anchors_yaml():
        if entry.get("name") == name:
            return entry
    return None


def _is_alias_boundary_ok(alias: str, label: str, pos: int) -> bool:
    """Word-boundary check for a substring hit.

    Required on the left only when the alias starts with an alphanumeric
    character; ditto on the right for the last character.  Aliases that
    already begin (or end) with punctuation (``.221``, ``hla-b*51:01``)
    provide their own boundary, so forcing an additional one would break
    matches like ``".221"`` inside ``"721.221-HLA-A*11:01"``.

    This rule lets us drop the trailing-space hack from the YAML while
    still rejecting false positives like ``"jy"`` inside ``"JYH"``.
    """
    first, last = alias[0], alias[-1]
    if first.isalnum() and pos > 0 and label[pos - 1].isalnum():
        return False
    end = pos + len(alias)
    return not (last.isalnum() and end < len(label) and label[end].isalnum())


def _find_anchor_by_label(label: str) -> tuple[dict, str] | None:
    """Return the (entry, matched_alias) for the longest alias that hits.

    Matches aliases with word-boundary awareness (see
    :func:`_is_alias_boundary_ok`): a substring hit only counts when it's
    flanked by punctuation / whitespace / string-boundary on the side(s)
    where the alias is alphanumeric.  Longest winning alias wins; ties
    are resolved deterministically by YAML entry order.
    """
    if not label:
        return None
    low = label.casefold()
    best: tuple[dict, str] | None = None
    best_len = -1
    for entry in _load_anchors_yaml():
        for alias in entry.get("aliases", []) or []:
            # Strip whitespace so the YAML doesn't need to encode word
            # boundaries via trailing spaces.
            a = str(alias).casefold().strip()
            if not a or len(a) <= best_len:
                continue
            idx = 0
            hit_pos = -1
            while True:
                i = low.find(a, idx)
                if i < 0:
                    break
                if _is_alias_boundary_ok(a, low, i):
                    hit_pos = i
                    break
                idx = i + 1
            if hit_pos >= 0:
                best = (entry, str(alias))
                best_len = len(a)
    return best


def _entry_has_exact_line_data(entry: dict) -> bool:
    """True iff the entry has a backend and a source with rows for its key."""
    backend = entry.get("expression_backend") or ""
    key = entry.get("expression_key") or ""
    if not backend or backend == "none" or not key:
        return False
    source_ids = [str(s) for s in (entry.get("source_ids") or [])]
    if not source_ids:
        return False
    available = _available_expression_sources()
    return any((str(key), sid) in available for sid in source_ids)


def _available_expression_sources() -> frozenset[tuple[str, str]]:
    path = line_expression_path()
    return _expression_sources_at(str(path), _artifact_signature(path))


@lru_cache(maxsize=4)
def _expression_sources_at(
    path: str, signature: tuple[int, int, int] | None
) -> frozenset[tuple[str, str]]:
    """Read source availability once per artifact identity, including rebuilds."""
    rows = _line_expression_rows(Path(path), signature, columns=["line_key", "source_id"])
    return frozenset(rows[["line_key", "source_id"]].itertuples(index=False, name=None))


def _curated_engineering(pmid: int | None, sample_label: str) -> bool:
    """Whether ``(pmid, sample_label)`` names a curated arm whose material is engineered.

    ``False`` when it names no curated arm: nothing then says the material was
    modified.  No curated study reuses a label across arms that disagree about
    engineering; if one did, any engineered arm makes the label engineered,
    since tier 1 would claim a measurement its curation cannot back.
    """
    if pmid is None or not sample_label:
        return False
    from .conditions import is_engineered_material
    from .curation import load_pmid_overrides

    entry = load_pmid_overrides().get(int(pmid))
    if not entry:
        return False
    label = sample_label.strip()
    return any(
        is_engineered_material(sample)
        for sample in entry.get("ms_samples") or []
        if str(sample.get("sample_label") or "").strip() == label
    )


def _resolve_via_parent(entry: dict) -> tuple[dict, str] | None:
    """Walk ``parent_line`` until a parent with exact-line data is found."""
    visited: set[str] = set()
    current = entry
    while True:
        parent_name = current.get("parent_line")
        if not parent_name or parent_name in visited:
            return None
        visited.add(parent_name)
        parent = _anchor_by_name(parent_name)
        if parent is None:
            return None
        if _entry_has_exact_line_data(parent):
            return parent, parent_name
        current = parent


def _resolve_via_class_anchor(entry: dict) -> tuple[dict, str] | None:
    """Resolve tier-3 via ``line_family``."""
    family = entry.get("line_family")
    anchor_name = _LINE_FAMILY_CLASS_ANCHOR.get(family or "")
    if not anchor_name:
        return None
    anchor = _anchor_by_name(anchor_name)
    if anchor is None or not _entry_has_exact_line_data(anchor):
        return None
    return anchor, anchor_name


def _tier_source_ids(entry: dict) -> tuple[str, ...]:
    available = _available_expression_sources()
    return tuple(
        str(source_id)
        for source_id in (entry.get("source_ids") or [])
        if (str(entry.get("expression_key") or ""), str(source_id)) in available
    )


def resolve_sample_expression_anchor(
    sample_label: str,
    *,
    cell_name: str | None = None,
    pmid: int | None = None,
    study_label: str | None = None,
    engineered: bool | None = None,
    lineage_tissue: str | None = None,
    cancer_type: str | None = None,
    cancer_type_backend: Callable[[str], dict] | None = None,
) -> SampleExpressionAnchor:
    """Resolve a sample to its best available expression anchor.

    Applies the 6-tier fallback hierarchy from issue pirl-unc/hitlist#140
    and returns a :class:`SampleExpressionAnchor` whose fields are written
    as provenance columns on downstream export rows.

    Parameters
    ----------
    sample_label
        Free-text sample label (from ``ms_samples[].sample_label``).  The
        primary key matched against the registry's ``aliases``.
    cell_name
        Optional additional matchable label (e.g. IEDB ``cell_name``).
        Merged with ``sample_label`` for alias lookup.
    pmid, study_label
        Currently unused for resolution, accepted for forward
        compatibility with study-level overrides.
    engineered
        Whether the sample's own material is engineered — a genetic
        intervention or introduced MHC, per
        :data:`hitlist.conditions.ENGINEERING_CONDITION_COLUMNS`.  ``None``
        (the default) derives it from curation when ``pmid`` and
        ``sample_label`` name a curated ``ms_samples`` arm, and reads as not
        engineered otherwise; ``True`` / ``False`` override.  An engineered
        sample never resolves at tier 1: a label can match the reference
        line's alias — ``HeLa-CIITA (Mock)`` hits ``hela``, ``K562
        transfectant DPB1*01:01`` hits ``k562`` — and registry derivative
        entries are catch-alls (``HAP1-KO`` covers eleven knockouts), so RNA
        registered under any entry is at best the reference material's.  Such
        a hit resolves at tier 2 against the same data with
        ``expression_parent_key`` naming that material (#576).
    lineage_tissue
        Fallback-5 tissue bucket when the sample doesn't hit the line
        registry at all and the caller has an explicit HPA tissue label.
    cancer_type
        Fallback-4 cancer-type surrogate label.  Passed into
        ``cancer_type_backend`` when provided.
    cancer_type_backend
        Optional callable ``cancer_type -> {expression_backend,
        expression_key, source_ids?}`` for tier-4 resolution (pirlygenes
        integration).  When ``None``, tier 4 is skipped.

    Returns
    -------
    SampleExpressionAnchor
        Never ``None``; tier-6 is the no-match sentinel.
    """
    if engineered is None:
        engineered = _curated_engineering(pmid, sample_label)
    label_parts = [p for p in (sample_label, cell_name) if p]
    joined_label = " ".join(label_parts)

    match = _find_anchor_by_label(joined_label)

    if match is not None:
        entry, matched_alias = match

        # Tier 1 — registry hit with exact-line data, unless the sample's
        # material is engineered: then the same data is the reference
        # material's, reported at tier 2 with that material as parent (#576).
        if _entry_has_exact_line_data(entry):
            name = str(entry.get("name") or entry["expression_key"])
            return SampleExpressionAnchor(
                expression_backend=str(entry["expression_backend"]),
                expression_key=str(entry["expression_key"]),
                expression_match_tier=2 if engineered else 1,
                expression_parent_key=name if engineered else None,
                source_ids=_tier_source_ids(entry),
                reason=(
                    f"engineered material matched '{name}' on alias "
                    f"'{matched_alias}'; its reference RNA stands in"
                    if engineered
                    else f"exact line match on alias '{matched_alias}'"
                ),
                matched_alias=matched_alias,
            )

        # Tier 2 — parent-line has data.
        parent_hit = _resolve_via_parent(entry)
        if parent_hit is not None:
            parent_entry, parent_name = parent_hit
            return SampleExpressionAnchor(
                expression_backend=str(parent_entry["expression_backend"]),
                expression_key=str(parent_entry["expression_key"]),
                expression_match_tier=2,
                expression_parent_key=parent_name,
                source_ids=_tier_source_ids(parent_entry),
                reason=(f"parent-line fallback: '{entry.get('name')}' → '{parent_name}'"),
                matched_alias=matched_alias,
            )

        # Tier 3 — line-family class anchor (EBV-LCL, mono-allelic host).
        class_hit = _resolve_via_class_anchor(entry)
        if class_hit is not None:
            anchor_entry, anchor_name = class_hit
            return SampleExpressionAnchor(
                expression_backend=str(anchor_entry["expression_backend"]),
                expression_key=str(anchor_entry["expression_key"]),
                expression_match_tier=3,
                expression_parent_key=anchor_name,
                source_ids=_tier_source_ids(anchor_entry),
                reason=(
                    f"line-family class anchor: '{entry.get('name')}' "
                    f"(family '{entry.get('line_family')}') → '{anchor_name}'"
                ),
                matched_alias=matched_alias,
            )

        # Inherit the registry entry's cancer_type / lineage_tissue when the
        # caller didn't supply explicit values — the registry is the
        # authoritative biology source.
        cancer_type = cancer_type or entry.get("cancer_type") or None
        lineage_tissue = lineage_tissue or entry.get("lineage_tissue") or None

    # Tier 4 — cancer-type surrogate (pirlygenes) via caller-supplied backend.
    if cancer_type and cancer_type_backend is not None:
        try:
            result = cancer_type_backend(cancer_type) or {}
        except Exception as exc:
            warnings.warn(
                f"cancer_type_backend raised {exc!r}; skipping tier 4",
                RuntimeWarning,
                stacklevel=2,
            )
            result = {}
        backend = result.get("expression_backend")
        key = result.get("expression_key")
        if backend and key:
            return SampleExpressionAnchor(
                expression_backend=str(backend),
                expression_key=str(key),
                expression_match_tier=4,
                expression_parent_key=None,
                source_ids=tuple(result.get("source_ids") or ()),
                reason=f"cancer-type surrogate for '{cancer_type}'",
                matched_alias=match[1] if match else None,
            )

    # Tier 5 — broad tissue / lineage surrogate.
    if lineage_tissue:
        return SampleExpressionAnchor(
            expression_backend="hpa_tissue",
            expression_key=str(lineage_tissue),
            expression_match_tier=5,
            expression_parent_key=None,
            source_ids=("hpa_rna",),
            reason=f"tissue surrogate via '{lineage_tissue}'",
            matched_alias=match[1] if match else None,
        )

    # Tier 6 — no anchor.
    return SampleExpressionAnchor(
        expression_backend="none",
        expression_key="",
        expression_match_tier=6,
        expression_parent_key=None,
        source_ids=(),
        reason="no expression anchor available",
        matched_alias=match[1] if match else None,
    )


# ── Packaged CSV loaders + parquet with fallback ────────────────────────────


def _iter_packaged_csvs() -> Iterable[tuple[str, Path]]:
    """Yield (source_id, path) for every packaged CSV referenced by sources.yaml."""
    for source in _load_sources_yaml():
        if source.get("build_status") != "packaged":
            continue
        fname = source.get("file")
        if not fname:
            continue
        path = files(_DATA_MODULE) / fname
        yield str(source.get("source_id", "")), Path(str(path))


@lru_cache(maxsize=8)
def _load_packaged_csv(path_str: str) -> pd.DataFrame:
    path = Path(path_str)
    df = pd.read_csv(
        path,
        compression="gzip" if path.name.endswith(".gz") else None,
    )
    # Harmonize dtypes — gene_name / transcript_id may be empty strings in
    # the CSV; pandas reads them as NaN which is fine downstream.
    return df


def _load_packaged_union() -> pd.DataFrame:
    """Concatenate every packaged CSV into one long-form frame."""
    frames: list[pd.DataFrame] = []
    for source_id, path in _iter_packaged_csvs():
        if not path.exists():
            continue
        df = _load_packaged_csv(str(path)).copy()
        # Guarantee the source_id column even if a CSV omits it.
        if "source_id" not in df.columns or df["source_id"].isna().all():
            df["source_id"] = source_id
        frames.append(df)
    if not frames:
        return pd.DataFrame(
            columns=[
                "line_key",
                "source_id",
                "granularity",
                "gene_id",
                "gene_name",
                "transcript_id",
                "tpm",
                "log2_tpm",
            ]
        )
    return pd.concat(frames, ignore_index=True, sort=False)


# ── Canonical rows ──────────────────────────────────────────────────────────


def _source_stamp(source: dict) -> dict:
    """Per-source metadata stamped onto every row of that source."""
    pmid_raw = source.get("pmid")
    return {
        "backend": source.get("backend") or "",
        "source_id": source.get("source_id") or "",
        "pmid": int(pmid_raw) if pmid_raw is not None else pd.NA,
        "reference": source.get("reference") or "",
        "study_label": source.get("study_label") or "",
        "normalization": source.get("normalization") or "",
        "quantifier": source.get("quantifier") or "",
        "species": source.get("species") or "",
        "license": source.get("license") or "",
    }


def _parent_line_lookup() -> dict[str, str]:
    """Map each ``expression_key`` to the registry's canonical line name."""
    return {
        str(entry.get("expression_key")): str(entry.get("name", ""))
        for entry in _load_anchors_yaml()
        if entry.get("expression_key")
    }


@lru_cache(maxsize=1)
def _gene_name_map() -> dict[str, str]:
    """The packaged ENSG (version-stripped) -> HGNC symbol map."""
    m = pd.read_csv(files(_DATA_MODULE) / _GENE_NAME_MAP_FILE, dtype=str).fillna("")
    return {gid.split(".")[0]: nm for gid, nm in zip(m["gene_id"], m["gene_name"]) if gid and nm}


def _fill_gene_names_from_packaged_map(df: pd.DataFrame) -> pd.DataFrame:
    """Fill blank ``gene_name`` from :data:`_GENE_NAME_MAP_FILE`, in place.

    IDs absent from the map stay blank; the builder's optional pyensembl pass
    retries those.
    """
    gn = df["gene_name"].astype("string").fillna("")
    gid = df["gene_id"].astype("string").fillna("")
    need = (gn.str.strip() == "") & gid.str.startswith("ENSG")
    if need.any():
        id2name = _gene_name_map()
        df.loc[need, "gene_name"] = (
            gid[need].map(lambda g: id2name.get(g.split(".")[0], "")).to_numpy()
        )
    return df


def enrich_line_expression_rows(rows: pd.DataFrame) -> pd.DataFrame:
    """Canonical index rows from value rows: the one definition of a row.

    Stamps each row's source metadata from ``sources.yaml`` (by ``source_id``),
    ``parent_line_key`` from the anchor registry and blank ``gene_name`` from
    the packaged ENSG -> HGNC map, then normalizes dtypes and orders
    :data:`LINE_EXPRESSION_COLUMNS`.  The builder applies it to everything it
    writes, and every read that cannot use a built index applies it to the
    packaged sources — so a reader without an index sees the rows a build
    would write, less only the builder's optional pyensembl backfill, which a
    read path must not need (#577).
    """
    out = rows.copy()
    for column in LINE_EXPRESSION_COLUMNS:
        if column not in out.columns:
            out[column] = pd.NA if column in {"pmid", "tpm", "log2_tpm"} else ""
    stamps = {str(s.get("source_id") or ""): _source_stamp(s) for s in _load_sources_yaml()}
    source_ids = out["source_id"].astype(str)
    for column, default in _source_stamp({}).items():
        if column == "source_id":
            continue
        values = {sid: stamp[column] for sid, stamp in stamps.items()}
        out[column] = source_ids.map(values).where(source_ids.isin(values), default)
    out["parent_line_key"] = out["line_key"].astype(str).map(_parent_line_lookup()).fillna("")
    out = _fill_gene_names_from_packaged_map(out)
    out["pmid"] = pd.to_numeric(out["pmid"], errors="coerce").astype("Int64")
    out["tpm"] = pd.to_numeric(out["tpm"], errors="coerce").astype("float64")
    out["log2_tpm"] = pd.to_numeric(out["log2_tpm"], errors="coerce").astype("float64")
    out["profile_id"] = out["profile_id"].fillna("")
    return out[list(LINE_EXPRESSION_COLUMNS)].reset_index(drop=True)


@lru_cache(maxsize=1)
def _packaged_rows() -> pd.DataFrame:
    return enrich_line_expression_rows(_load_packaged_union())


def packaged_line_expression_rows() -> pd.DataFrame:
    """Every packaged source as canonical rows — what readers use without an index."""
    return _packaged_rows().copy()


# ── Index stamp and staleness (#577) ────────────────────────────────────────


def _packaged_input_paths() -> dict[str, Path]:
    """Every packaged file a built index is derived from, keyed by file name.

    The anchor registry, ``sources.yaml``, each packaged source CSV and the
    ENSG -> HGNC map: every packaged *build input*, not only the sources that
    contribute rows, because each of them changes the bytes a build writes
    (#582).  Optional DepMap matrices are deliberately absent: they are
    user-local, so an index that carries them is richer than the packaged
    sources, not inconsistent with them.
    """
    data = files(_DATA_MODULE)
    paths = {
        _ANCHORS_FILE: Path(str(files(_ANCHORS_RESOURCE) / _ANCHORS_FILE)),
        "sources.yaml": Path(str(data / "sources.yaml")),
        _GENE_NAME_MAP_FILE: Path(str(data / _GENE_NAME_MAP_FILE)),
    }
    for _source_id, path in _iter_packaged_csvs():
        paths[path.name] = path
    return paths


def _fingerprint_of(paths: dict[str, Path]) -> str:
    """Name-aware digest of files' contents, streamed rather than held in memory."""
    digest = hashlib.sha256()
    for name, path in sorted(paths.items()):
        content = hashlib.sha256()
        with open(path, "rb") as stream:
            for chunk in iter(lambda: stream.read(1 << 20), b""):
                content.update(chunk)
        digest.update(f"{name}\0".encode())
        digest.update(content.digest())
    return digest.hexdigest()


@lru_cache(maxsize=1)
def packaged_line_expression_fingerprint() -> str:
    """Content fingerprint of this release's packaged line-expression build inputs.

    A hash, not a version or an mtime: an index is built on one machine and
    read on another (a downloaded CI corpus, a shared cache), so only the
    bytes the build consumed can say whether it is the index this release
    would produce.
    """
    return _fingerprint_of(_packaged_input_paths())


def line_expression_index_stamp() -> dict:
    """What an index built by this release is stamped with.

    ``packaged_inputs`` is :func:`packaged_line_expression_fingerprint` and
    ``build_version`` is :data:`LINE_EXPRESSION_BUILD_VERSION`.  The builder's
    observations cache records the same mapping, so the rebuild a stale-index
    warning asks for is one ``hitlist build observations`` agrees is needed.
    """
    return {
        "packaged_inputs": packaged_line_expression_fingerprint(),
        "build_version": LINE_EXPRESSION_BUILD_VERSION,
    }


def write_line_expression_index(df: pd.DataFrame, path: Path | None = None) -> Path:
    """Write ``df`` as a line-expression index stamped for this release.

    The one writer of ``line_expression.parquet``: the builder uses it, and so
    should any test or tool that installs an index the resolver is meant to
    trust — an index without this release's stamp is treated as stale.
    Atomic: concurrent readers never see a half-written index.
    """
    out = Path(path) if path is not None else line_expression_path()
    stamp = json.dumps(line_expression_index_stamp(), sort_keys=True).encode()
    return atomic_write_parquet(df, out, metadata={INDEX_STAMP_METADATA_KEY: stamp})


def _artifact_signature(path: Path) -> tuple[int, int, int] | None:
    """Identity of the file on disk, so a rebuild invalidates cached reads."""
    if not path.exists():
        return None
    stat = path.stat()
    return (stat.st_ino, stat.st_mtime_ns, stat.st_size)


@lru_cache(maxsize=8)
def _index_status(path: str, signature: tuple[int, int, int] | None) -> tuple[str, str]:
    """Classify a built index: ``(state, detail)``.  Pure — never warns.

    ``absent``; ``unreadable``; ``current``; ``inputs_changed`` (this build
    version, other packaged inputs); ``outdated`` (unstamped, or another build
    version).  Cached per artifact identity, which changes on every rebuild.
    """
    if signature is None:
        return ("absent", "")
    import pyarrow.parquet as pq

    try:
        raw = (pq.read_schema(path).metadata or {}).get(INDEX_STAMP_METADATA_KEY)
        stamp = json.loads(raw) if raw else None
    except (ArrowInvalid, OSError, ValueError) as exc:
        return ("unreadable", str(exc))
    if stamp is None:
        return ("outdated", "it has no build stamp")
    if stamp.get("build_version") != LINE_EXPRESSION_BUILD_VERSION:
        return (
            "outdated",
            f"index format {stamp.get('build_version')}; this release writes "
            f"{LINE_EXPRESSION_BUILD_VERSION}",
        )
    if stamp.get("packaged_inputs") != packaged_line_expression_fingerprint():
        return ("inputs_changed", "")
    return ("current", "")


def line_expression_index_is_current(path: Path | None = None) -> bool:
    """Whether a built index carries this release's stamp.

    ``False`` when it is missing, unreadable, unstamped, or stamped with other
    packaged inputs or another :data:`LINE_EXPRESSION_BUILD_VERSION`.  A pure
    check: it never warns, so asking does not consume the one warning a read
    of a stale index emits.
    """
    p = Path(path) if path is not None else line_expression_path()
    return _index_status(str(p), _artifact_signature(p))[0] == "current"


def _registered_pairs() -> frozenset[tuple[str, str]]:
    """Every (line_key, source_id) pair the anchor registry lists."""
    return frozenset(
        (str(entry["expression_key"]), str(source_id))
        for entry in _load_anchors_yaml()
        if entry.get("expression_key")
        for source_id in entry.get("source_ids") or []
    )


def _packaged_source_ids() -> frozenset[str]:
    return frozenset(
        str(s.get("source_id") or "")
        for s in _load_sources_yaml()
        if s.get("build_status") == "packaged"
    )


def _read_parquet_or_warn(path: Path, columns=None) -> pd.DataFrame | None:
    try:
        return pd.read_parquet(path, columns=columns)
    except (ArrowInvalid, OSError, ValueError) as exc:
        warnings.warn(
            f"Failed to read built line expression parquet at {path}; "
            f"falling back to packaged sources. {exc}",
            RuntimeWarning,
            stacklevel=3,
        )
        return None


def _index_download_pairs(path: Path) -> frozenset[tuple[str, str]]:
    """(line_key, source_id) pairs of the index's downloaded (non-packaged) rows."""
    rows = _read_parquet_or_warn(path, columns=["line_key", "source_id"])
    if rows is None:
        return frozenset()
    pairs = rows.drop_duplicates().astype(str).itertuples(index=False, name=None)
    packaged = _packaged_source_ids()
    return frozenset(pair for pair in pairs if pair[1] not in packaged)


def _carried_download_rows(path: Path, columns=None) -> pd.DataFrame | None:
    """The index's downloaded rows for pairs the current registry still lists.

    Re-enriched against this release's registry and ``sources.yaml``, so a
    carried row is the row a rebuild would write from the same download.
    """
    wanted = None if columns is None else sorted({*columns, "line_key", "source_id"})
    rows = _read_parquet_or_warn(path, columns=wanted)
    if rows is None:
        return None
    packaged = _packaged_source_ids()
    allowed = {
        f"{line}\t{source}" for line, source in _registered_pairs() if source not in packaged
    }
    key = rows["line_key"].astype(str) + "\t" + rows["source_id"].astype(str)
    return enrich_line_expression_rows(rows[key.isin(allowed)])


def _describe_pairs(pairs: frozenset[tuple[str, str]]) -> str:
    lines = sorted({line for line, _ in pairs})
    sources = sorted({source for _, source in pairs})
    shown = ", ".join(lines[:6]) + (f" and {len(lines) - 6} more" if len(lines) > 6 else "")
    return f"{shown} ({', '.join(sources)})"


def _rebuild_hint(has_downloads: bool) -> str:
    if not has_downloads:
        return (
            "Rebuild it with `hitlist build observations` or "
            "`hitlist.builder.build_line_expression()` to silence this."
        )
    from .downloads import depmap_bundle_is_registered

    if depmap_bundle_is_registered():
        return (
            "`hitlist data fetch depmap` rebuilds it from the registered DepMap "
            "files without downloading them again."
        )
    return (
        "`hitlist data fetch depmap` rebuilds it, but its DepMap files are no "
        "longer registered on disk, so it will download them again (about 4.7 GB)."
    )


@lru_cache(maxsize=8)
def _warn_about_index(path: str, signature: tuple[int, int, int] | None) -> None:
    """The one warning a read of a non-current index emits, per artifact identity.

    Said in terms of what the reader actually loses: DepMap is only mentioned
    to someone whose index held downloaded rows.
    """
    state, detail = _index_status(path, signature)
    if state == "unreadable":
        message = (
            f"Failed to read built line expression parquet at {path}; "
            f"falling back to packaged sources. {detail}"
        )
    else:
        held = _index_download_pairs(Path(path))
        if state == "inputs_changed":
            kept = held & _registered_pairs()
            dropped = held - kept
            message = (
                f"Line expression index at {path} was built from different packaged "
                f"line-expression inputs than this release ships; reads use this "
                f"release's packaged sources"
            )
            if kept:
                message += f" plus the index's downloaded rows for {_describe_pairs(kept)}"
            if dropped:
                message += (
                    f", leaving out its rows for {_describe_pairs(dropped)}, which "
                    f"the registry no longer lists"
                )
            message += "."
        else:
            message = (
                f"Line expression index at {path} predates this release's index "
                f"format ({detail}), so reads use the packaged sources instead"
            )
            message += (
                f"; its downloaded rows for {_describe_pairs(held)} are not used "
                f"until it is rebuilt."
                if held
                else ", which hold everything it contained."
            )
        message += " " + _rebuild_hint(bool(held))
    warnings.warn(message, RuntimeWarning, stacklevel=4)


def _line_expression_rows(
    path: Path, signature: tuple[int, int, int] | None, columns=None
) -> pd.DataFrame:
    """The rows every reader sees — the one decision about the built index (#577).

    ``current``: the index itself.  ``inputs_changed``: this release's packaged
    rows plus the index's downloaded rows for (line, source) pairs the
    registry still lists — a changed anchor file must not cost a DepMap user
    HeLa or K562.  ``outdated``, ``unreadable`` or ``absent``: packaged rows.
    Warns once per index per process, only here on the read path, and never
    writes: a read that rebuilt the index would rebuild under whichever
    process happened to read first.
    """
    state, _detail = _index_status(str(path), signature)
    if state == "current":
        rows = _read_parquet_or_warn(path, columns=columns)
        if rows is not None:
            return rows
    elif state != "absent":
        _warn_about_index(str(path), signature)
    rows = _packaged_rows()
    if state == "inputs_changed":
        carried = _carried_download_rows(path, columns)
        if carried is not None and not carried.empty:
            rows = pd.concat([rows, carried], ignore_index=True)
    return rows[list(columns)] if columns else rows.copy()


def _apply_series_filter(df: pd.DataFrame, col: str, values) -> pd.DataFrame:
    if values is None or col not in df.columns:
        return df
    if isinstance(values, (str, int)):
        values = [values]
    wanted = list(values)
    return df[df[col].isin(wanted)]


def load_line_expression(
    line_key: str | Iterable[str] | None = None,
    gene_name: str | Iterable[str] | None = None,
    gene_id: str | Iterable[str] | None = None,
    transcript_id: str | Iterable[str] | None = None,
    granularity: str | None = None,
    source_id: str | Iterable[str] | None = None,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Load per-line RNA/transcript TPM with optional filters.

    Reads ``line_expression.parquet`` when it matches this release, and the
    packaged sources under ``hitlist/data/line_expression/`` — enriched to the
    same row shape — when it is absent or stale (see
    :func:`line_expression_index_is_current`).

    Parameters
    ----------
    line_key
        One or more ``line_key`` values from the anchor registry
        (e.g. ``"HeLa"``, ``"GM12878"``, ``"C1R"``).
    gene_name
        Filter to HGNC gene symbols.
    gene_id
        Filter to Ensembl gene IDs (``ENSG...``), unversioned.
    transcript_id
        Filter to Ensembl transcript IDs (``ENST...``), unversioned.
    granularity
        ``"gene"`` or ``"transcript"``.
    source_id
        Restrict to one or more sources from ``sources.yaml``.
    columns
        Project to a subset of columns.
    """
    path = line_expression_path()
    df = _line_expression_rows(path, _artifact_signature(path))

    df = _apply_series_filter(df, "line_key", line_key)
    df = _apply_series_filter(df, "gene_name", gene_name)
    df = _apply_series_filter(df, "gene_id", gene_id)
    df = _apply_series_filter(df, "transcript_id", transcript_id)
    df = _apply_series_filter(df, "source_id", source_id)
    if granularity is not None and "granularity" in df.columns:
        df = df[df["granularity"] == granularity]

    if columns:
        present = [c for c in columns if c in df.columns]
        df = df[present]

    return df.reset_index(drop=True)


# ── Peptide-origin assignment ──────────────────────────────────────────────


_EMPTY_PEPTIDE_ORIGIN = {
    "peptide_origin_gene": "",
    "peptide_origin_gene_id": "",
    "peptide_origin_tpm": float("nan"),
    "peptide_origin_log2_tpm": float("nan"),
    "peptide_origin_dominant_transcript": "",
    "peptide_origin_n_supporting_transcripts": 0,
    "peptide_origin_resolution": "no_anchor",
}


def _gene_tpm_from_line_expression(
    line_expression_df: pd.DataFrame,
) -> pd.Series:
    """Return a Series gene_name -> tpm from gene-granularity rows."""
    if line_expression_df.empty or "granularity" not in line_expression_df.columns:
        return pd.Series(dtype=float)
    gene_rows = line_expression_df[line_expression_df["granularity"] == "gene"]
    if gene_rows.empty:
        return pd.Series(dtype=float)
    # If multiple sources cover the same gene (e.g. GM12878 ENCODE + DepMap
    # GM12878 via different release), keep the max.  The resolver picks a
    # single backend upstream so this is rarely exercised.
    return gene_rows.groupby("gene_name")["tpm"].max()


def _transcript_tpm_for_gene(
    gene_name: str,
    line_expression_df: pd.DataFrame,
) -> pd.Series:
    """Return a Series transcript_id -> tpm for one gene."""
    if line_expression_df.empty or "granularity" not in line_expression_df.columns:
        return pd.Series(dtype=float)
    tx = line_expression_df[
        (line_expression_df["granularity"] == "transcript")
        & (line_expression_df["gene_name"] == gene_name)
    ]
    if tx.empty:
        return pd.Series(dtype=float)
    return tx.groupby("transcript_id")["tpm"].max()


def compute_peptide_origin(
    peptide: str,
    candidate_genes: list[str] | list[dict],
    line_expression_df: pd.DataFrame,
    *,
    transcript_lookup: Callable[[str], list[tuple[str, str]]] | None = None,
) -> dict:
    """Pick the most-likely source gene for a peptide in a given sample.

    The caller passes the line-expression rows already filtered to the
    sample's resolved ``line_key``; this function does no anchor
    resolution of its own.

    Scoring:

    - When transcript-granularity rows are present AND a
      ``transcript_lookup`` is provided, per-gene score = sum of TPM of
      transcripts whose protein translation contains the peptide at
      least once.  A transcript encoding the peptide at multiple
      positions contributes its TPM *once* (the transcript is either
      present at that abundance or not — copy number within a transcript
      is already folded into TPM).  Transcripts that splice out the
      peptide's exon(s) contribute zero.  Resolution tag:
      ``transcript_isoform_sum``.

    - Otherwise, per-gene score = gene-level TPM from
      ``line_expression_df``; resolution tag ``gene_only``.

    Ties broken deterministically by gene_name (alphabetical).

    Parameters
    ----------
    peptide
        The peptide sequence (used only on the transcript path to
        verify which isoforms encode it).
    candidate_genes
        Either a flat list of ``gene_name`` strings or a list of dicts
        with at least ``gene_name`` and optionally ``gene_id``.
    line_expression_df
        Long-form TPM rows filtered to one sample's ``line_key``.
    transcript_lookup
        Optional ``gene_name -> [(transcript_id, protein_seq), ...]``
        callable.  Injected so tests can exercise the transcript path
        without needing pyensembl; production callers pass a
        pyensembl-backed closure.

    Returns
    -------
    dict
        Keys: ``peptide_origin_gene``, ``peptide_origin_gene_id``,
        ``peptide_origin_tpm``, ``peptide_origin_log2_tpm``,
        ``peptide_origin_dominant_transcript``,
        ``peptide_origin_n_supporting_transcripts``,
        ``peptide_origin_resolution``.
    """
    import math

    if not candidate_genes:
        return dict(_EMPTY_PEPTIDE_ORIGIN)

    if isinstance(candidate_genes[0], str):
        gene_entries = [{"gene_name": g, "gene_id": ""} for g in candidate_genes]
    else:
        gene_entries = [
            {
                "gene_name": str(g.get("gene_name") or ""),
                "gene_id": str(g.get("gene_id") or ""),
            }
            for g in candidate_genes
        ]
    # Deduplicate while preserving gene_id when available.
    seen: dict[str, dict] = {}
    for entry in gene_entries:
        gname = entry["gene_name"]
        if not gname:
            continue
        if gname not in seen:
            seen[gname] = entry
        elif entry["gene_id"] and not seen[gname]["gene_id"]:
            seen[gname]["gene_id"] = entry["gene_id"]
    if not seen:
        return dict(_EMPTY_PEPTIDE_ORIGIN)

    has_transcript_rows = (
        not line_expression_df.empty
        and "granularity" in line_expression_df.columns
        and (line_expression_df["granularity"] == "transcript").any()
    )

    best_gene: str = ""
    best_gene_id: str = ""
    best_tpm: float = -1.0
    best_dominant_tx: str = ""
    best_n_supporting: int = 0
    resolution = "gene_only"

    if has_transcript_rows and transcript_lookup is not None:
        resolution = "transcript_isoform_sum"
        for gname in sorted(seen.keys()):
            tx_tpm = _transcript_tpm_for_gene(gname, line_expression_df)
            if tx_tpm.empty:
                continue
            isoforms = transcript_lookup(gname) or []
            encoding_tx: list[tuple[str, float]] = []
            for tid, seq in isoforms:
                if not seq or peptide not in seq:
                    continue
                # Strip version from parquet transcript IDs if present.
                lookup_id = str(tid).split(".")[0]
                if lookup_id in tx_tpm.index:
                    encoding_tx.append((lookup_id, float(tx_tpm[lookup_id])))
                elif str(tid) in tx_tpm.index:
                    encoding_tx.append((str(tid), float(tx_tpm[str(tid)])))
            if not encoding_tx:
                continue
            total = sum(tpm for _, tpm in encoding_tx)
            dom_tid = max(encoding_tx, key=lambda t: t[1])[0]
            if total > best_tpm or (
                math.isclose(total, best_tpm) and (best_gene == "" or gname < best_gene)
            ):
                best_tpm = total
                best_gene = gname
                best_gene_id = seen[gname]["gene_id"]
                best_dominant_tx = dom_tid
                best_n_supporting = len(encoding_tx)

    if best_gene == "":
        # Gene-only path (either no transcript backend or no lookup function).
        resolution = "gene_only"
        gene_series = _gene_tpm_from_line_expression(line_expression_df)
        if gene_series.empty:
            return dict(_EMPTY_PEPTIDE_ORIGIN)
        for gname in sorted(seen.keys()):
            if gname not in gene_series.index:
                continue
            tpm = float(gene_series[gname])
            if tpm > best_tpm or (
                math.isclose(tpm, best_tpm) and (best_gene == "" or gname < best_gene)
            ):
                best_tpm = tpm
                best_gene = gname
                best_gene_id = seen[gname]["gene_id"]
                best_dominant_tx = ""
                best_n_supporting = 0

    if best_gene == "":
        return dict(_EMPTY_PEPTIDE_ORIGIN)

    log2 = math.log2(best_tpm + 1.0) if best_tpm >= 0 else float("nan")
    return {
        "peptide_origin_gene": best_gene,
        "peptide_origin_gene_id": best_gene_id,
        "peptide_origin_tpm": best_tpm,
        "peptide_origin_log2_tpm": log2,
        "peptide_origin_dominant_transcript": best_dominant_tx,
        "peptide_origin_n_supporting_transcripts": best_n_supporting,
        "peptide_origin_resolution": resolution,
    }


# Convenience re-exports for downstream callers.

__all__ = [
    "INDEX_STAMP_METADATA_KEY",
    "LINE_EXPRESSION_BUILD_VERSION",
    "LINE_EXPRESSION_COLUMNS",
    "SampleExpressionAnchor",
    "compute_peptide_origin",
    "enrich_line_expression_rows",
    "is_line_expression_built",
    "line_expression_index_is_current",
    "line_expression_index_stamp",
    "line_expression_path",
    "load_line_expression",
    "load_line_expression_anchors",
    "load_line_expression_sources",
    "packaged_line_expression_fingerprint",
    "packaged_line_expression_rows",
    "resolve_line_key",
    "resolve_sample_expression_anchor",
    "write_line_expression_index",
]
