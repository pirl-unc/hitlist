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

"""Parse IEDB's ``Cell Name`` strings into structured fields (#261).

IEDB records the cell source of an MS observation in a single
``Cell Name`` column.  In practice the field carries three different
shapes, often hybridized:

1. Pure cell-line names: ``"Expi293F"``, ``"MAPTAC"``, ``"Raji"``,
   ``"HAP1 wildtype"``, ``"HAP1 CALR KO"``.
2. Pure cell-type names: ``"B cell"``, ``"Splenocyte"``, ``"PBMC"``,
   ``"Glial cell"``, ``"Cell found in tissue"``.
3. Hybrid ``<line>-<type>``: ``"K562-Myeloid cell"``,
   ``"C1R cells-B cell"``, ``"HeLa cells-Epithelial cell"``,
   ``"MDA-MB-231-Epithelial cell"``.

:func:`parse_cell_name` decomposes the string into:

- ``is_cell_line``: bool — is this a cell-line MS run or a primary-cell
  / tissue / donor sample?
- ``cell_line_name``: canonical name (e.g. ``"HEK293T"``) or ``""``.
- ``cell_line_input``: the synonym actually present in the input
  (e.g. ``"293-T"``) for traceability.
- ``cell_type``: tissue / cell type (e.g. ``"B cell"``) or ``""``.
- ``donor_id``: patient / donor identifier, if present, else ``""``.
- ``genetic_modification``: knock-out / wildtype / engineering tag,
  e.g. ``"CALR KO"``, ``"wildtype"``, ``"MAPTAC"``.
- ``raw_cell_name``: the original input string verbatim.

Auxiliary inputs from the same observation row (``attributed_sample_label``,
``monoallelic_host``, ``src_cell_line``) sharpen the parse — they're
optional but recommended.

The cell-line registry is :file:`hitlist/data/cell_lines.yaml`; new
lines should be added there rather than hard-coded in this module.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

# ── Cell-type vocabulary (the right-hand side of hybrid strings) ─────────
#
# Sorted longest-first so the greedy matcher in ``parse_cell_name``
# picks "Epithelial cell" over a partial "cell" match.  Source: IEDB
# Cell Name strings observed in the corpus + Cellosaurus-style
# convention.

CELL_TYPES: tuple[str, ...] = (
    "Epithelial cell",
    "Dendritic cell",
    "Endothelial cell",
    "Stromal cell",
    "Glial cell",
    "Myeloid cell",
    "Mesenchymal cell",
    "Lymphoblast",
    "Lymphocyte",
    "Macrophage",
    "Melanocyte",
    "Hepatocyte",
    "Fibroblast",
    "Thymocyte",
    "Splenocyte",
    "Monocyte",
    "B cell",
    "T cell",
    "NK cell",
    "PBMC",
)

# Words that genericize a name without changing identity — strip them
# before dictionary lookup so "JY cells" matches "JY".
_LINE_NOISE_SUFFIXES: tuple[str, ...] = (" cells", " cell")

# Genetic-modification keywords we extract from line strings.  Order
# matters for the regex below: list longer / more specific first.
_GENETIC_MOD_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\bwildtype\b", "wildtype"),
    (r"\bwild[- ]type\b", "wildtype"),
    (r"\bWT\b", "wildtype"),
    (r"\b([A-Z0-9]{2,10})\s+(KO|knockout)\b", r"\1 KO"),  # "CALR KO"
    (r"\b(CRISPR|knockout|knock-out)\b", "CRISPR"),
)


# ── Public dataclass ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class CellNameInfo:
    """Structured decomposition of one ``Cell Name`` string.

    All string fields default to ``""`` (empty) when not extractable —
    callers should branch on ``is_cell_line`` and on per-field
    emptiness, NOT on ``None``.

    Attributes
    ----------
    is_cell_line
        ``True`` if the parse identified a cell-line sample.
        ``False`` if the parse identified a primary-cell / tissue /
        donor sample (or the input was empty / "Other" / unparseable).
    cell_line_name
        Canonical line name from the registry, e.g. ``"HEK293T"``.
        Empty when the parser couldn't match a known line.
    cell_line_input
        The exact synonym present in the input, e.g. ``"293-T"``.
        Useful for traceability when the canonical was applied.
        Empty when ``cell_line_name`` is empty.
    cell_type
        Tissue / cell type label, e.g. ``"B cell"``.  Inherits from
        the registry for known lines; extracted from the hybrid
        right-hand side otherwise.
    donor_id
        Patient / donor identifier, e.g. ``"13240-005"`` from a
        ``"MEL2 (13240-005)"`` :attr:`attributed_sample_label`.
        Empty when not extractable.
    genetic_modification
        Knock-out / wildtype / engineering construct, e.g.
        ``"CALR KO"``, ``"wildtype"``, ``"MAPTAC"``.
        Empty when the input had no modification info.
    raw_cell_name
        The original :attr:`cell_name` input verbatim, preserved for
        traceability and any downstream re-parse.
    """

    is_cell_line: bool
    cell_line_name: str
    cell_line_input: str
    cell_type: str
    donor_id: str
    genetic_modification: str
    raw_cell_name: str


# ── Registry loader ──────────────────────────────────────────────────────


def _registry_path() -> Path:
    return Path(__file__).resolve().parent / "data" / "cell_lines.yaml"


@lru_cache(maxsize=1)
def _load_registry() -> tuple[dict[str, dict], dict[str, str], list[str], list[str]]:
    """Load and index :file:`cell_lines.yaml`.

    Returns
    -------
    canonical_to_entry
        ``{canonical_name: {cell_type, tissue, notes, synonyms}}``.
    synonym_to_canonical
        ``{lowercase_synonym: canonical_name}`` for all known lines
        plus engineering systems.
    line_synonyms_by_length
        Synonyms sorted by length descending — for greedy
        longest-prefix matching against hybrid strings.
    engineering_synonyms_by_length
        Same for engineering-system entries.  Separated so the parser
        can tag them as constructs distinct from real cell lines.
    """
    data = yaml.safe_load(_registry_path().read_text())
    canonical_to_entry: dict[str, dict] = {}
    synonym_to_canonical: dict[str, str] = {}
    line_synonyms: list[str] = []
    engineering_synonyms: list[str] = []

    for canonical, entry in (data.get("cell_lines") or {}).items():
        canonical_to_entry[canonical] = entry
        for syn in entry.get("synonyms", []):
            # Defensive: numeric-looking synonyms in YAML (e.g. 721.221)
            # parse as float unless quoted.  Coerce to str so the
            # dictionary lookup works either way.
            syn_str = str(syn)
            synonym_to_canonical[syn_str.lower()] = canonical
            line_synonyms.append(syn_str)

    for canonical, entry in (data.get("engineering_systems") or {}).items():
        # Engineering systems are tagged with their canonical name as
        # the genetic_modification.  Reuse synonym_to_canonical so
        # _resolve_line can find them, but we'll branch on the
        # source ("engineering" vs "cell_lines") to decide how to
        # populate the output.
        for syn in entry.get("synonyms", []):
            syn_str = str(syn)
            synonym_to_canonical[syn_str.lower()] = f"ENG:{canonical}"
            engineering_synonyms.append(syn_str)

    line_synonyms.sort(key=len, reverse=True)
    engineering_synonyms.sort(key=len, reverse=True)
    return canonical_to_entry, synonym_to_canonical, line_synonyms, engineering_synonyms


# ── Helpers ──────────────────────────────────────────────────────────────


def _strip_noise(s: str) -> str:
    """Lowercase + trim + drop trailing "cells"/"cell" tokens used to
    genericize names ("JY cells" → "jy")."""
    lower = s.strip().lower()
    for suffix in _LINE_NOISE_SUFFIXES:
        if lower.endswith(suffix):
            lower = lower[: -len(suffix)].rstrip()
    return lower


def _resolve_line(token: str) -> tuple[str, str, bool]:
    """Look ``token`` up in the registry.

    Returns ``(canonical_name, cell_type_from_registry, is_engineering)``.
    ``canonical_name`` is empty when no match is found; the caller
    decides whether to treat the raw input as an unknown line or as
    not-a-line at all.
    """
    _, synonym_to_canonical, _, _ = _load_registry()
    canonical = synonym_to_canonical.get(_strip_noise(token), "")
    if not canonical:
        return "", "", False
    is_engineering = canonical.startswith("ENG:")
    if is_engineering:
        return canonical[4:], "", True
    canonical_to_entry, _, _, _ = _load_registry()
    cell_type = canonical_to_entry.get(canonical, {}).get("cell_type", "")
    return canonical, cell_type, False


def _extract_genetic_modification(s: str) -> tuple[str, str]:
    """Return ``(modification, remainder)`` — strip the modification
    keyword from ``s`` and return what's left for downstream parsing.
    Empty modification means none was found."""
    for pat, replacement in _GENETIC_MOD_PATTERNS:
        m = re.search(pat, s, flags=re.IGNORECASE)
        if m:
            if r"\1" in replacement:
                mod = re.sub(pat, replacement, m.group(0), flags=re.IGNORECASE)
            else:
                mod = replacement
            remainder = (s[: m.start()] + s[m.end() :]).strip()
            return mod, remainder
    return "", s


def _split_hybrid(s: str) -> tuple[str, str]:
    """Split a hybrid ``<line>-<cell_type>`` string into its two parts.

    Greedy from the right: scan the known cell-type vocabulary
    longest-first, return ``(line_part, cell_type)`` on the first
    suffix match (after stripping leading ``-`` / whitespace from the
    line side).  Returns ``("", "")`` when no cell-type suffix is
    detected.
    """
    s = s.strip()
    for cell_type in CELL_TYPES:
        if s.lower().endswith(cell_type.lower()):
            line_part = s[: -len(cell_type)].rstrip(" -")
            if line_part:
                return line_part, cell_type
            # Pure cell-type string with no line prefix — caller will
            # treat that as a primary-cell row.
            return "", cell_type
    return "", ""


def _extract_donor_id(attributed_sample_label: str) -> str:
    """Pull the donor identifier out of an ``attributed_sample_label``
    string like ``"MEL2 (13240-005)"`` → ``"13240-005"``.

    Falls back to returning the full label when no parens-enclosed
    identifier is present so downstream consumers still get the
    distinguishing string."""
    if not attributed_sample_label:
        return ""
    m = re.search(r"\(([^)]+)\)", attributed_sample_label)
    if m:
        return m.group(1).strip()
    return attributed_sample_label.strip()


# ── Public parser ────────────────────────────────────────────────────────


# Cell-name values that explicitly mean "no per-sample info recorded".
_UNINFORMATIVE_CELL_NAMES: frozenset[str] = frozenset(
    {"", "other", "cell found in tissue", "n/a", "na", "unknown"}
)


def parse_cell_name(
    cell_name: str,
    *,
    attributed_sample_label: str = "",
    monoallelic_host: str = "",
    src_cell_line: bool | None = None,
) -> CellNameInfo:
    """Decompose an IEDB ``Cell Name`` string into structured fields.

    Parameters
    ----------
    cell_name
        Raw IEDB ``Cell Name`` field (or empty string).
    attributed_sample_label
        Per-donor patient label, e.g. ``"MEL2 (13240-005)"``.  Used to
        extract ``donor_id``.  Pass empty when not available.
    monoallelic_host
        Mono-allelic engineering host string, e.g. ``"C1R"``,
        ``"MAPTAC"``, ``"Strep-tag II"``.  Used as a secondary input
        when ``cell_name`` is uninformative.  Pass empty when not
        available.
    src_cell_line
        Build-time boolean (from observations.parquet) indicating
        whether the source was a cell line.  When ``None`` the parser
        infers it from the content.

    Returns
    -------
    CellNameInfo
        Structured decomposition; see :class:`CellNameInfo`.

    Notes
    -----
    The parser is conservative — it returns ``cell_line_name=""``
    when the input doesn't match a known synonym in the registry,
    even if the string "looks like" a cell line.  Unknown lines
    should be added to :file:`hitlist/data/cell_lines.yaml`.
    """
    raw = cell_name or ""
    raw_stripped = raw.strip()
    donor_id = _extract_donor_id(attributed_sample_label)

    # First: try the monoallelic_host channel.  Engineering constructs
    # like "MAPTAC" / "Strep-tag II" show up there and tag the row as a
    # cell-line sample even when cell_name is uninformative.
    eng_canonical, _, eng_is_engineering = _resolve_line(monoallelic_host)
    if eng_is_engineering and not raw_stripped:
        return CellNameInfo(
            is_cell_line=True,
            cell_line_name="",
            cell_line_input=monoallelic_host.strip(),
            cell_type="",
            donor_id=donor_id,
            genetic_modification=eng_canonical,
            raw_cell_name=raw,
        )

    # Empty / explicitly-uninformative cell_name → fall back to
    # whatever the auxiliary inputs tell us.  If monoallelic_host
    # names a real cell line, use that.  Otherwise mark as unknown.
    if raw_stripped.lower() in _UNINFORMATIVE_CELL_NAMES:
        if monoallelic_host:
            canon, ct, is_eng = _resolve_line(monoallelic_host)
            if canon and not is_eng:
                return CellNameInfo(
                    is_cell_line=True,
                    cell_line_name=canon,
                    cell_line_input=monoallelic_host.strip(),
                    cell_type=ct,
                    donor_id=donor_id,
                    genetic_modification="",
                    raw_cell_name=raw,
                )
            if is_eng:
                return CellNameInfo(
                    is_cell_line=True,
                    cell_line_name="",
                    cell_line_input=monoallelic_host.strip(),
                    cell_type="",
                    donor_id=donor_id,
                    genetic_modification=canon,
                    raw_cell_name=raw,
                )
        return CellNameInfo(
            is_cell_line=bool(src_cell_line),
            cell_line_name="",
            cell_line_input="",
            cell_type="",
            donor_id=donor_id,
            genetic_modification="",
            raw_cell_name=raw,
        )

    # Extract any leading genetic modification BEFORE splitting on the
    # cell-type suffix — "HAP1 wildtype" / "HAP1 CALR KO" / etc.
    genetic_mod, remainder = _extract_genetic_modification(raw_stripped)

    # Try the hybrid split: "<line>-<cell_type>".
    line_part, hybrid_type = _split_hybrid(remainder)

    if line_part:
        # Hybrid case: try to resolve the line part to a canonical name.
        canon, ct_from_registry, is_eng = _resolve_line(line_part)
        # Registry-derived cell_type wins when available (it's more
        # consistent than IEDB's hybrid right-hand side, which varies
        # in capitalization: "Glial cell" vs "Glial Cell").
        final_type = ct_from_registry or hybrid_type
        if is_eng:
            return CellNameInfo(
                is_cell_line=True,
                cell_line_name="",
                cell_line_input=line_part,
                cell_type=final_type,
                donor_id=donor_id,
                genetic_modification=genetic_mod or canon,
                raw_cell_name=raw,
            )
        return CellNameInfo(
            is_cell_line=True,
            cell_line_name=canon,
            cell_line_input=line_part if canon else "",
            cell_type=final_type,
            donor_id=donor_id,
            genetic_modification=genetic_mod,
            raw_cell_name=raw,
        )

    if hybrid_type and not line_part:
        # Pure cell-type string — primary-cell / tissue sample.
        return CellNameInfo(
            is_cell_line=False,
            cell_line_name="",
            cell_line_input="",
            cell_type=hybrid_type,
            donor_id=donor_id,
            genetic_modification=genetic_mod,
            raw_cell_name=raw,
        )

    # Not a hybrid — try resolving the whole (modification-stripped)
    # string as a cell line.
    canon, ct_from_registry, is_eng = _resolve_line(remainder)
    if canon and not is_eng:
        return CellNameInfo(
            is_cell_line=True,
            cell_line_name=canon,
            cell_line_input=remainder.strip(),
            cell_type=ct_from_registry,
            donor_id=donor_id,
            genetic_modification=genetic_mod,
            raw_cell_name=raw,
        )
    if canon and is_eng:
        return CellNameInfo(
            is_cell_line=True,
            cell_line_name="",
            cell_line_input=remainder.strip(),
            cell_type="",
            donor_id=donor_id,
            genetic_modification=genetic_mod or canon,
            raw_cell_name=raw,
        )

    # Couldn't resolve.  If src_cell_line tells us this WAS a line,
    # preserve the raw string as cell_line_input so downstream
    # consumers can still distinguish it from other unknown lines.
    if src_cell_line:
        return CellNameInfo(
            is_cell_line=True,
            cell_line_name="",
            cell_line_input=remainder.strip(),
            cell_type="",
            donor_id=donor_id,
            genetic_modification=genetic_mod,
            raw_cell_name=raw,
        )

    # Otherwise treat as a primary-cell / tissue sample with the raw
    # string as the cell_type (matches IEDB's primary-cell category
    # strings like "Lymphoblast" / "Splenocyte" / "Other" — though
    # the explicit-uninformative branch above caught "Other"
    # already).
    return CellNameInfo(
        is_cell_line=False,
        cell_line_name="",
        cell_line_input="",
        cell_type=remainder.strip(),
        donor_id=donor_id,
        genetic_modification=genetic_mod,
        raw_cell_name=raw,
    )


def known_cell_lines() -> list[str]:
    """Return the sorted list of canonical cell-line names in the
    registry — useful for ``--cell-line`` CLI completion or for
    surfacing the vocabulary in error messages."""
    canonical_to_entry, _, _, _ = _load_registry()
    return sorted(canonical_to_entry)


def known_cell_types() -> tuple[str, ...]:
    """Return the cell-type vocabulary used for hybrid-string parsing."""
    return CELL_TYPES
