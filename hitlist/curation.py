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

"""Data-driven MS evidence curation for IEDB/CEDAR MHC ligand data.

Classification rules and PMID overrides are loaded from YAML data files,
not hardcoded in Python. This makes the curation logic transparent,
auditable, and easy to extend without code changes.

Source categories (mutually exclusive priority order)::

    cancer              Cancer tissue, cancer patient biofluids, non-EBV cell lines
    adjacent            Tumor-adjacent normal tissue (resection margins)
    activated_apc       Monocyte-derived DCs, macrophages with activation
    healthy_tissue      Direct ex vivo healthy somatic tissue (THE SAFETY SIGNAL)
    healthy_thymus      Direct ex vivo thymus (expected for CTAs)
    healthy_reproductive Direct ex vivo reproductive tissue (expected for CTAs)
    ebv_lcl             EBV-transformed B-cell lines
    cell_line           Other cell lines
"""

from __future__ import annotations

import contextlib
import re
from functools import cache, lru_cache
from os.path import dirname, join

import pandas as pd
import yaml


def _data_path(filename: str) -> str:
    return join(dirname(__file__), "data", filename)


@lru_cache(maxsize=1)
def load_pmid_overrides() -> dict[int, dict]:
    """Load PMID curation overrides from YAML.

    Validates that every ``mono_allelic_host`` name resolves to an entry
    in ``monoallelic_lines.yaml`` (typos would otherwise silently
    produce rows with a non-existent ``monoallelic_host`` string).
    Warns on legacy YAML keys (``type:``, ``label:``) that were renamed
    to ``sample_label:`` / ``study_label:`` in v1.7.0.

    Returns
    -------
    dict[int, dict]
        Mapping from PMID to override dict with keys: study_label,
        override, note, and optionally tissue_overrides, donors,
        ms_samples, tissues.
    """
    import warnings

    with open(_data_path("pmid_overrides.yaml")) as f:
        entries = yaml.safe_load(f)

    known_hosts = {e["name"] for e in load_monoallelic_lines()}
    for e in entries:
        host = e.get("mono_allelic_host")
        if host and host not in known_hosts:
            raise ValueError(
                f"PMID {e.get('pmid')} has mono_allelic_host={host!r} but that "
                f"name is not in monoallelic_lines.yaml (known hosts: "
                f"{sorted(known_hosts)}).  Add the host to monoallelic_lines.yaml "
                f"or fix the typo."
            )
        # Legacy key detection (v1.7.0 rename)
        if "label" in e and "study_label" not in e:
            warnings.warn(
                f"PMID {e.get('pmid')}: YAML key 'label:' is deprecated, "
                f"use 'study_label:' (v1.7.0).  Value ignored.",
                DeprecationWarning,
                stacklevel=2,
            )
        for sample in e.get("ms_samples") or []:
            if "type" in sample and "sample_label" not in sample:
                warnings.warn(
                    f"PMID {e.get('pmid')}: ms_samples entry uses deprecated "
                    f"'type:' key, use 'sample_label:' (v1.7.0).  Value ignored.",
                    DeprecationWarning,
                    stacklevel=2,
                )
                break  # one warning per PMID is enough

    return {int(e["pmid"]): e for e in entries}


@lru_cache(maxsize=1)
def load_tissue_categories() -> dict[str, frozenset[str]]:
    """Load tissue category definitions from YAML.

    Returns
    -------
    dict[str, frozenset[str]]
        Keys: ``reproductive``, ``thymus``, ``activated_apc_cell_names``,
        ``activated_apc_tissues``.
    """
    with open(_data_path("tissue_categories.yaml")) as f:
        data = yaml.safe_load(f)
    return {
        "reproductive": frozenset(data.get("reproductive", [])),
        "thymus": frozenset(data.get("thymus", [])),
        "activated_apc_cell_names": frozenset(data.get("activated_apc_cell_names", [])),
        "activated_apc_tissues": frozenset(data.get("activated_apc_tissues", [])),
    }


# ── Cached mhcgnomes parse ─────────────────────────────────────────────────


@cache
def _cached_parse(mhc_restriction: str):
    """Cache mhcgnomes parse results.

    ~100k unique allele strings across millions of IEDB rows.  Uses
    an unbounded cache: mhcgnomes parse results are small, the total
    memory footprint for the vocabulary is well under 100 MB, and
    eviction at 1024 was causing re-parse churn for alleles that
    reappear later in the scan (e.g. species-specific alleles that
    cluster by cell line).
    """
    try:
        from mhcgnomes import parse

        return parse(mhc_restriction)
    except (ImportError, Exception):
        return None


# ── MHC species ────────────────────────────────────────────────────────────


@cache
def classify_mhc_species(mhc_restriction: str) -> str:
    """Determine the species of an MHC restriction annotation.

    Uses mhcgnomes when available, falls back to prefix matching.

    Cached by input string: ~100k unique mhc_restriction values across
    millions of IEDB rows, and the mhcgnomes parse + species lookup is
    one of the hottest per-row calls in the scanner.

    Parameters
    ----------
    mhc_restriction
        IEDB "MHC Restriction" field value.

    Returns
    -------
    str
        Species name (e.g. ``"Homo sapiens"``, ``"Mus musculus"``),
        or empty string if undetermined.
    """
    if not mhc_restriction:
        return ""

    result = _cached_parse(mhc_restriction)
    if result is not None and hasattr(result, "species"):
        return result.species.name

    # Regex fallback
    if mhc_restriction.startswith("HLA"):
        return "Homo sapiens"
    if mhc_restriction.startswith(("H-2", "H2")):
        return "Mus musculus"
    return ""


# ── Species normalization ─────────────────────────────────────────────────

# Hardcoded fallback when mhcgnomes is not installed.
_SPECIES_ALIASES: dict[str, str] = {
    "human": "Homo sapiens",
    "homo sapiens": "Homo sapiens",
    "mouse": "Mus musculus",
    "mus musculus": "Mus musculus",
    "rat": "Rattus norvegicus",
    "rattus norvegicus": "Rattus norvegicus",
    "pig": "Sus scrofa",
    "sus scrofa": "Sus scrofa",
    "cattle": "Bos taurus",
    "cow": "Bos taurus",
    "bos taurus": "Bos taurus",
    "dog": "Canis lupus",
    "canis lupus": "Canis lupus",
    "canis lupus familiaris": "Canis lupus",
    "chicken": "Gallus gallus",
    "gallus gallus": "Gallus gallus",
    "rhesus macaque": "Macaca mulatta",
    "macaca mulatta": "Macaca mulatta",
}


@lru_cache(maxsize=256)
def normalize_species(raw: str) -> str:
    """Normalize a species string to its canonical scientific name.

    Accepts any common format — scientific name, common name,
    underscore-separated, or IEDB parenthetical style — and returns
    a consistent canonical form using ``mhcgnomes`` when available.

    Examples::

        normalize_species("human")                    # "Homo sapiens"
        normalize_species("Homo sapiens (human)")     # "Homo sapiens"
        normalize_species("homo_sapiens")             # "Homo sapiens"
        normalize_species("Mus musculus (mouse)")     # "Mus musculus"

    Parameters
    ----------
    raw
        Species string in any supported format.

    Returns
    -------
    str
        Canonical species name, or the stripped input if unrecognized.
        Empty string for empty input.
    """
    if not raw:
        return ""
    cleaned = raw.strip()
    if not cleaned:
        return ""

    # Try mhcgnomes first — it handles scientific names, common names,
    # underscores, and even parenthetical IEDB format.
    try:
        from mhcgnomes import Species

        result = Species.get(cleaned)
        if result is not None:
            return result.name
    except ImportError:
        pass

    # Fallback: normalize to lowercase with spaces for alias lookup
    key = cleaned.lower().replace("_", " ")

    # Strip parenthetical suffix: "Homo sapiens (human)" → "homo sapiens"
    paren = key.find("(")
    if paren > 0:
        key = key[:paren].strip()

    if key in _SPECIES_ALIASES:
        return _SPECIES_ALIASES[key]

    # Return stripped form (without parenthetical) for unrecognized species
    paren = cleaned.find("(")
    return cleaned[:paren].strip() if paren > 0 else cleaned


# Vertebrate host genera that bear classical MHC molecules. The chimeric
# flag is only meaningful when BOTH the source proteome and the MHC come
# from this set — otherwise we have either ordinary pathogen-on-host
# biology (bacterium/virus/parasite source) or an unrecognized species,
# neither of which constitutes an engineered cross-species system.
#
# This whitelist is small and stable by design: the closed set of
# vertebrate species seen as ``mhc_species`` in the curated index is
# enumerated explicitly so the rule is robust against the long tail of
# bacterial / viral source names that no pattern-based pathogen filter
# can fully cover.
_MHC_BEARING_HOST_GENERA = frozenset(
    {
        "homo",
        "mus",
        "rattus",
        "canis",
        "felis",
        "sus",
        "bos",
        "equus",
        "gallus",
        "anas",
        "macaca",
        "pan",
        "xenopus",
        "sarcophilus",
        "trichosurus",
        "danio",
        "ovis",
        "capra",
        "oryctolagus",
        "mesocricetus",
        "cricetulus",
        "cavia",
    }
)

_UNKNOWN_SOURCE_TOKENS = frozenset({"unidentified", "unknown", "n/a", "na", "-"})


@cache
def is_chimeric_system(source_organism: str, mhc_species: str) -> bool:
    """True iff source proteome and MHC come from different vertebrate-host genera.

    Engineered chimeric systems deliberately decouple the proteome species
    from the MHC species: HLA-transgenic rats / mice, mono-allelic mouse-MHC
    transfectants in human cells, NetH2pan binding-prediction training data
    (human peptides on mouse MHC), AAV-transduced allogeneic HLA expression,
    etc. Downstream consumers want to filter these out (or keep only these,
    for chimeric-aware training).

    Returns ``False`` when:

    - either side is empty / ``"unidentified"`` — chimerism cannot be
      asserted without both halves resolved.
    - either side resolves to a genus outside the MHC-bearing vertebrate
      host whitelist — viral / bacterial / parasite source on a host MHC
      is normal infection biology, not an engineered cross-species system.
    - normalized genus tokens match — substrains and subspecies
      (``"Mus musculus C57BL/6"`` vs ``"Mus musculus"``;
      ``"Canis lupus familiaris"`` vs ``"Canis sp."``) are not chimeric.

    Cached because the (source_organism, mhc_species) tuple repeats heavily
    across an observations index of millions of rows but resolves to only a
    couple hundred unique pairs.
    """
    if not source_organism or not mhc_species:
        return False
    src = str(source_organism).strip()
    mhc = str(mhc_species).strip()
    if not src or not mhc:
        return False
    if src.lower() in _UNKNOWN_SOURCE_TOKENS or mhc.lower() in _UNKNOWN_SOURCE_TOKENS:
        return False
    src_norm = normalize_species(src)
    mhc_norm = normalize_species(mhc)
    if not src_norm or not mhc_norm:
        return False
    src_genus = src_norm.split()[0].lower()
    mhc_genus = mhc_norm.split()[0].lower()
    if src_genus not in _MHC_BEARING_HOST_GENERA:
        return False
    if mhc_genus not in _MHC_BEARING_HOST_GENERA:
        return False
    return src_genus != mhc_genus


# ── Allele normalization ──────────────────────────────────────────────────


@lru_cache(maxsize=4096)
def normalize_allele(raw: str) -> str:
    """Normalize an MHC allele string to canonical Species-Gene[*allele] form.

    Uses mhcgnomes to parse and re-serialize.  Handles HLA, H-2 (mouse),
    Saha (Tasmanian devil), Mamu (rhesus), SLA (pig), BoLA (cow), DLA
    (dog), Patr (chimp), and any other species mhcgnomes supports.

    Examples::

        normalize_allele("HLA-A*02:01")        # "HLA-A*02:01"
        normalize_allele("H-2Kb")              # "H2-K*b"
        normalize_allele("SLA-1*0201")         # "SLA-1*02:01"
        normalize_allele("Saha-UA")            # "Saha-UA"

    Returns the input stripped for unparseable strings (e.g. "HLA class I").
    """
    if not raw:
        return ""
    cleaned = raw.strip()
    if not cleaned:
        return ""

    try:
        from mhcgnomes import parse

        result = parse(cleaned)
        # Only return normalized form for actual alleles/genes/pairs
        # (not generic Species or Class-only designations like "HLA class I")
        if result is not None and type(result).__name__ in ("Allele", "Gene", "Pair"):
            return result.to_string()
    except (ImportError, Exception):
        pass

    return cleaned


# ── Allele resolution ──────────────────────────────────────────────────────

#: Resolution tiers, ordered from most to least specific.
ALLELE_RESOLUTION_ORDER: list[str] = [
    "four_digit",
    "two_digit",
    "serological",
    "class_only",
    "unresolved",
]

_RESOLUTION_RANK: dict[str, int] = {v: i for i, v in enumerate(ALLELE_RESOLUTION_ORDER)}


@cache
def classify_allele_resolution(mhc_restriction: str) -> str:
    """Classify the resolution level of an MHC restriction annotation.

    Uses mhcgnomes if available for authoritative parsing, otherwise
    falls back to regex patterns.

    Cached by input string: same vocabulary as ``classify_mhc_species``,
    same argument for caching at the outer layer.

    Parameters
    ----------
    mhc_restriction
        IEDB "MHC Restriction" field value.

    Returns
    -------
    str
        One of: ``"four_digit"``, ``"two_digit"``, ``"serological"``,
        ``"class_only"``, ``"unresolved"``.
    """
    if not mhc_restriction:
        return "unresolved"

    result = _cached_parse(mhc_restriction)
    if result is not None:
        try:
            from mhcgnomes.allele import Allele
            from mhcgnomes.mhc_class import MhcClass
            from mhcgnomes.pair import Pair
            from mhcgnomes.serotype import Serotype

            if isinstance(result, Allele):
                if len(result.allele_fields) >= 2:
                    return "four_digit"
                return "two_digit"
            if isinstance(result, Pair):
                # Either side can be a Gene (e.g. "HLA-DRA/DRB1",
                # "HLA-DPA1*01:03/DPB1") — Gene has no allele_fields, so
                # guard the attribute access. Pair resolution is the *min*
                # of the two sides; a gene-only side means the pair is not
                # even two-digit resolved and falls through to "unresolved".
                alpha_fields = (
                    len(result.alpha.allele_fields) if isinstance(result.alpha, Allele) else 0
                )
                beta_fields = (
                    len(result.beta.allele_fields) if isinstance(result.beta, Allele) else 0
                )
                if alpha_fields >= 2 and beta_fields >= 2:
                    return "four_digit"
                if alpha_fields >= 1 and beta_fields >= 1:
                    return "two_digit"
                return "unresolved"
            if isinstance(result, Serotype):
                return "serological"
            if isinstance(result, MhcClass):
                return "class_only"
            return "unresolved"
        except ImportError:
            pass

    # Regex fallback when mhcgnomes is not installed
    if not mhc_restriction.startswith("HLA"):
        return "unresolved"
    if "class" in mhc_restriction.lower():
        return "class_only"
    if ("/" in mhc_restriction or "," in mhc_restriction) and "*" in mhc_restriction:
        return "four_digit" if ":" in mhc_restriction else "two_digit"
    if "*" in mhc_restriction and ":" in mhc_restriction:
        return "four_digit"
    if "*" in mhc_restriction:
        return "two_digit"
    # HLA-A2, HLA-B7 etc.
    if mhc_restriction.startswith("HLA-"):
        return "serological"
    return "unresolved"


def allele_resolution_rank(resolution: str) -> int:
    """Integer rank for resolution (lower = more specific)."""
    return _RESOLUTION_RANK.get(resolution, len(ALLELE_RESOLUTION_ORDER))


_LOCUS_SEROTYPE_RE = re.compile(r"^(A|B|C|DR|DQ|DP|DM|DO)\d")


def _serotype_specificity_rank(name: str) -> int:
    """Lower = more specific / preferred as the canonical serotype.

    0: locus-specific (A24, B57, DR15, ...) — what a clinician usually means
    1: public epitopes (Bw4, Bw6, C1, C2, ...) — orthogonal axis, less useful
       as the canonical answer to "what serotype is this allele?"
    """
    return 0 if _LOCUS_SEROTYPE_RE.match(name) else 1


@lru_cache(maxsize=1)
def _build_allele_to_serotypes_map() -> dict[str, tuple[str, ...]]:
    """Build a reverse map from allele compact key to ALL its serotypes.

    Returns a dict of ``{allele_key: (serotype1, serotype2, ...)}`` where
    the tuple is ordered by specificity:
    1. Locus-specific serotypes first (A24, B57, DR15)
    2. Public epitopes after (Bw4, Bw6)
    3. Within a class, broader (shorter) names first

    Returns empty dict if mhcgnomes is unavailable.
    """
    try:
        from mhcgnomes.data import serotypes
    except ImportError:
        return {}

    reverse: dict[str, list[str]] = {}
    hla = serotypes["HLA"]
    for sero_name, allele_list in hla.items():
        for allele_str in allele_list:
            reverse.setdefault(allele_str, []).append(sero_name)

    return {
        allele: tuple(
            f"HLA-{s}"
            for s in sorted(names, key=lambda n: (_serotype_specificity_rank(n), len(n), n))
        )
        for allele, names in reverse.items()
    }


@lru_cache(maxsize=1)
def _build_allele_to_serotype_map() -> dict[str, str]:
    """Build a reverse map from allele compact key to its canonical serotype.

    Ranks serotypes by specificity (locus-specific beats public epitopes),
    then by broader-first (A2 over A2.1), so A\\*24:02 → HLA-A24 rather
    than HLA-Bw4.
    """
    return {a: names[0] for a, names in _build_allele_to_serotypes_map().items() if names}


@lru_cache(maxsize=8192)
@cache
def allele_to_all_serotypes(mhc_restriction: str) -> tuple[str, ...]:
    """All serotypes an allele belongs to, most-specific first.

    Unlike :func:`allele_to_serotype`, this returns every serotype the
    allele is a member of.  Many alleles legitimately belong to both a
    locus-specific serotype (A24, B57) and a public epitope shared
    across loci (Bw4 is carried by subsets of A- and B-locus alleles —
    the axis KIR3DL1 recognizes).

    Cached by input string — same ~100k-vocab argument as
    ``classify_mhc_species`` / ``classify_allele_resolution``. Returns
    a tuple (immutable) so cache aliasing is safe.

    Examples::

        allele_to_all_serotypes("HLA-A*24:02")  # ("HLA-A24", "HLA-Bw4")
        allele_to_all_serotypes("HLA-B*57:01")  # ("HLA-B57", "HLA-B17", "HLA-Bw4")
        allele_to_all_serotypes("HLA-A*02:01")  # ("HLA-A2",)

    Returns an empty tuple when the allele cannot be mapped or input is empty.
    """
    if not mhc_restriction:
        return ()

    result = _cached_parse(mhc_restriction)
    if result is not None:
        try:
            from mhcgnomes.allele import Allele
            from mhcgnomes.serotype import Serotype

            if isinstance(result, Serotype):
                return (f"HLA-{result.name}",)
            if isinstance(result, Allele):
                key = f"{result.gene.name}*{''.join(result.allele_fields)}"
                return _build_allele_to_serotypes_map().get(key, ())
        except ImportError:
            pass

    return ()


def allele_to_serotype(mhc_restriction: str) -> str:
    """Map an HLA allele or serotype annotation to its canonical serotype.

    Uses mhcgnomes when available. Returns the most-specific serotype
    (e.g. ``"HLA-A24"`` rather than ``"HLA-Bw4"`` for HLA-A*24:02).  Use
    :func:`allele_to_all_serotypes` for the full list when an allele
    belongs to both a locus-specific serotype and a public epitope.

    Parameters
    ----------
    mhc_restriction
        IEDB "MHC Restriction" field value.

    Returns
    -------
    str
        Serotype name (e.g. ``"HLA-A2"``), or empty string if the
        allele cannot be mapped.
    """
    all_sero = allele_to_all_serotypes(mhc_restriction)
    return all_sero[0] if all_sero else ""


@lru_cache(maxsize=1)
def _build_serotype_to_alleles_map() -> dict[str, tuple[str, ...]]:
    """Forward map from canonical serotype name to its 4-digit members.

    Reads ``mhcgnomes.data.serotypes["HLA"]`` directly so the values come
    from the same IPD-IMGT/HLA table mhcgnomes ships with. The shipped
    table stores alleles in compact form (``A*0201``) — we canonicalize
    each via ``normalize_allele`` to the colon-separated 4-digit form
    (``HLA-A*02:01``) so callers can match against the parquet's
    normalized strings. Members are sorted lexicographically; for typical
    locus-specific serotypes that puts the lowest-numbered allele first,
    which is also (by IPD's discovery-order numbering) usually the
    population-dominant allele (A2 → A*02:01, B7 → B*07:02, DR4 →
    DRB1*04:01).

    Keys are HLA-prefixed (``"HLA-A2"``). Returns empty dict if
    mhcgnomes is unavailable.
    """
    try:
        from mhcgnomes.data import serotypes
    except ImportError:
        return {}

    out: dict[str, tuple[str, ...]] = {}
    for sero_name, allele_list in serotypes["HLA"].items():
        canon: list[str] = []
        for compact in allele_list:
            # ``compact`` is the IPD-style "A*0201" — prefix with HLA-
            # and let normalize_allele insert the colon and validate.
            normalized = normalize_allele(f"HLA-{compact}")
            if normalized:
                canon.append(normalized)
        out[f"HLA-{sero_name}"] = tuple(sorted(set(canon)))
    return out


@lru_cache(maxsize=8192)
@cache
def serotype_to_alleles(serotype: str) -> tuple[str, ...]:
    """Enumerate the 4-digit alleles that belong to a serotype.

    Inverse of :func:`allele_to_serotype`. Used to expand a user-supplied
    serotype query (``HLA-A2``) into the candidate 4-digit alleles
    (``HLA-A*02:01``, ``HLA-A*02:02``, ...) so the parquet pushdown finds
    the matching rows.

    Returns an empty tuple when the input is empty, not a serotype, or
    not in the IPD-IMGT/HLA serotype catalog.

    Examples::

        serotype_to_alleles("HLA-A2")   # ("HLA-A*02:01", "HLA-A*02:02", ...)
        serotype_to_alleles("HLA-A*02:01")  # () — already 4-digit
    """
    if not serotype:
        return ()
    norm = normalize_allele(serotype)
    return _build_serotype_to_alleles_map().get(norm, ())


@lru_cache(maxsize=8192)
@cache
def best_4digit_for_serotype(serotype: str) -> str:
    """Pick the most-likely 4-digit allele for a serotype.

    Heuristic: lowest-numbered member. By IPD-IMGT/HLA convention, the
    earliest-discovered alleles get the lowest numbers, and these are
    almost always the population-dominant ones (A2 → A*02:01 is ~50%
    of A2 carriers; B7 → B*07:02 is ~95% of B7). The guess is wrong for
    a handful of serotypes (notably A24, where A*24:02 dominates over
    A*24:01), so callers should treat the answer as a best-effort
    default, not ground truth.

    Returns ``""`` when the input is empty or not a known serotype.
    """
    members = serotype_to_alleles(serotype)
    return members[0] if members else ""


# ── Mono-allelic cell line detection ──────────────────────────────────────


@lru_cache(maxsize=1)
def load_monoallelic_lines() -> list[dict]:
    """Load known mono-allelic cell line systems from YAML.

    Returns
    -------
    list[dict]
        Each entry has keys: name, aliases (list[str]), hla_status,
        endogenous_alleles (list[str]).
    """
    with open(_data_path("monoallelic_lines.yaml")) as f:
        return yaml.safe_load(f)


def detect_monoallelic(cell_name: str, mhc_restriction: str = "") -> tuple[bool, str]:
    """Detect if a row comes from a known mono-allelic cell line system.

    Parameters
    ----------
    cell_name
        IEDB "Cell Name" field value.
    mhc_restriction
        IEDB "MHC Restriction" field value (the reported allele).

    Returns
    -------
    tuple[bool, str]
        ``(is_monoallelic, host_name)``. ``is_monoallelic`` is True when
        the cell_name matches a known HLA-null/low host AND the reported
        allele is not one of the host's endogenous alleles.
    """
    if not cell_name:
        return False, ""

    cell_name_lower = cell_name.lower()
    for entry in load_monoallelic_lines():
        for alias in entry["aliases"]:
            if alias in cell_name_lower:
                endogenous = entry.get("endogenous_alleles", [])
                if mhc_restriction and mhc_restriction in endogenous:
                    return False, ""
                return True, entry["name"]
    return False, ""


def _is_resolved_allele(mhc_restriction: str) -> bool:
    """True when the allele is specific enough to claim mono-allelic status.

    Rows with empty, ``HLA class I`` / ``class_only``, or ``unresolved``
    MHC restriction cannot be flagged mono-allelic — we do not know
    which allele (if any) produced the peptide.  This is the sole gate
    on the PMID-level override: cell_name is not a reliable
    discriminator because IEDB frequently mis-annotates the host
    (e.g., 721.221 recorded as ``"HeLa cells-Epithelial cell"`` in
    Trolle 2016) — the PMID override exists to correct exactly that.
    """
    return classify_allele_resolution(mhc_restriction) in ("four_digit", "two_digit", "serological")


# ── Exact-allele bag expansion (issue #137) ────────────────────────────────


_BAG_PROVENANCE_VALUES = (
    "exact",
    "sample_allele_match",
    "pmid_class_pool",
    "unmatched",
)


def _looks_like_four_digit_allele(s: str) -> bool:
    """Quick syntactic check that a string is a 4-digit-ish HLA allele.

    Used to filter free-text descriptions out of YAML ``hla_alleles`` blocks
    (e.g. PMID 33858848 has ``class_i: "51 HLA-I allotypes (...)"`` that we
    never want to treat as an allele).  We only require the obvious
    structural markers ``HLA-`` + ``*`` + ``:``; mhcgnomes parses the
    string for downstream allele logic.
    """
    if not isinstance(s, str):
        return False
    s = s.strip()
    return s.startswith("HLA-") and "*" in s and ":" in s


def _flatten_hla_alleles(value) -> set[str]:
    """Recursively collect 4-digit allele strings from a curated ``hla_alleles`` value.

    Tolerates the four shapes seen in pmid_overrides.yaml:
    flat list, dict-of-lists keyed by donor / cell line, dict-of-strings
    (free-text descriptions are filtered out by the syntactic check),
    and **space-separated multi-allele genotype strings** like
    ``"HLA-A*01:01 HLA-B*07:02 HLA-C*12:03"`` — used in ~32% of
    ms_samples to encode a donor's genotype as one field. Without
    this split, downstream qc reports report the whole genotype as a
    single phantom "allele".
    """
    out: set[str] = set()
    if value is None:
        return out
    if isinstance(value, str):
        s = value.strip()
        tokens = s.split()
        if len(tokens) > 1:
            # Multi-allele genotype string: keep only the tokens that
            # look like real alleles (drops noise like 'or' / commas
            # in free-text fields that happened to have HLA in them).
            for tok in tokens:
                if _looks_like_four_digit_allele(tok):
                    out.add(tok)
        elif _looks_like_four_digit_allele(s):
            out.add(s)
    elif isinstance(value, list):
        for v in value:
            out |= _flatten_hla_alleles(v)
    elif isinstance(value, dict):
        for v in value.values():
            out |= _flatten_hla_alleles(v)
    return out


@lru_cache(maxsize=512)
def _pmid_allele_pool(pmid_int: int) -> frozenset[str]:
    """All curated 4-digit alleles for a PMID, flattened across the
    ``hla_alleles`` block regardless of nested shape.

    Returns an empty frozenset if no override exists or the override has
    no ``hla_alleles`` curation.
    """
    overrides = load_pmid_overrides()
    entry = overrides.get(pmid_int)
    if entry is None:
        return frozenset()
    return frozenset(_flatten_hla_alleles(entry.get("hla_alleles")))


_HOST_MHC_SPLIT_RE = re.compile(r"[;,]")


def _parse_host_mhc_types(host_mhc_types: str) -> frozenset[str]:
    """Parse IEDB ``Host | MHC Types Present`` into a set of 4-digit alleles.

    IEDB uses ``;``-separated ``HLA-A*01:01;HLA-B*13:02;...`` strings.
    Free-text or non-allele tokens are dropped.
    """
    if not host_mhc_types:
        return frozenset()
    parts = _HOST_MHC_SPLIT_RE.split(host_mhc_types)
    return frozenset(p.strip() for p in parts if _looks_like_four_digit_allele(p))


def _filter_alleles_by_class(alleles: frozenset[str], mhc_class: str) -> set[str]:
    """Filter a candidate allele set to those matching the row's MHC class.

    ``mhc_class`` is the IEDB ``Class`` field (``"I"``, ``"II"``, ``"non
    classical"``, or ``""``).  Classical class I = HLA-A/B/C; class II =
    any HLA-D*.  Non-classical class I (E/F/G) is treated as class I for
    bag expansion since restrictions like ``"HLA class I"`` could legitimately
    map to those.  Empty ``mhc_class`` disables filtering.
    """
    if not mhc_class or mhc_class == "non classical":
        return set(alleles)
    if mhc_class == "I":
        return {
            a for a in alleles if a[:5] in ("HLA-A", "HLA-B", "HLA-C", "HLA-E", "HLA-F", "HLA-G")
        }
    if mhc_class == "II":
        return {a for a in alleles if a.startswith("HLA-D")}
    return set(alleles)


@lru_cache(maxsize=16384)
def expand_allele_bag(
    mhc_restriction: str,
    host_mhc_types: str = "",
    pmid: int | str = "",
    mhc_class: str = "",
) -> tuple[str, str, int]:
    """Expand a (possibly coarse) MHC restriction to a candidate exact-allele
    bag with provenance.

    Issue #137: downstream training pipelines need to know which exact
    4-digit alleles a row's restriction could plausibly map to, plus how
    that mapping was obtained, so they can do MIL / noisy-OR training over
    allele bags instead of silently collapsing or dropping coarse
    restrictions.

    Logic:

    - ``four_digit`` rows are returned as-is with provenance ``exact``.
    - ``class_only`` rows (e.g. ``"HLA class I"``) are expanded against
      first the row's ``Host | MHC Types Present`` (the donor's typed
      alleles, when IEDB carries them) and otherwise the per-PMID
      ``hla_alleles`` block; provenance is ``sample_allele_match`` or
      ``pmid_class_pool`` accordingly.  In both cases the candidate set
      is filtered to the row's MHC class.
    - All other resolutions (``two_digit``, ``serological``,
      ``unresolved``) are returned as ``unmatched``.  Two-digit and
      serotype expansion against an external IPD-IMGT/HLA catalog is a
      planned follow-up.

    Returns
    -------
    tuple[str, str, int]
        ``(mhc_allele_set, mhc_allele_provenance, mhc_allele_bag_size)``
        where the set is ``;``-joined (parquet-friendly, consistent with
        existing ``serotypes`` / ``gene_names`` columns).
    """
    resolution = classify_allele_resolution(mhc_restriction)
    if resolution == "four_digit":
        return mhc_restriction.strip(), "exact", 1

    if resolution != "class_only":
        return "", "unmatched", 0

    sample_alleles = _parse_host_mhc_types(host_mhc_types)
    pmid_int: int | None = None
    if pmid:
        with contextlib.suppress(ValueError, TypeError):
            pmid_int = int(pmid)
    pool = _pmid_allele_pool(pmid_int) if pmid_int is not None else frozenset()

    candidates = sample_alleles or pool
    if not candidates:
        return "", "unmatched", 0

    candidates = _filter_alleles_by_class(candidates, mhc_class)
    if not candidates:
        return "", "unmatched", 0

    provenance = "sample_allele_match" if sample_alleles else "pmid_class_pool"
    return ";".join(sorted(candidates)), provenance, len(candidates)


def _matches_condition(row_fields: dict[str, str], condition: dict) -> bool:
    """Check if a row's fields match a condition dict.

    Each condition key is an IEDB field name, value is either a string
    or a list of strings. All conditions must match (AND logic).
    String matching is case-insensitive for Source Tissue.

    Field-specific matching:
      - "Source Tissue" — case-insensitive equality.
      - "Assay Comments" — case-insensitive substring match. IEDB sometimes
        concatenates per-arm provenance into a single Assay Comments cell
        (e.g. "eluted from CRC tissue. eluted from NMC tissue") so substring
        match is the only way to recognize an arm in a combined row.
      - All other fields — exact equality (case-sensitive).
    """
    for field, expected in condition.items():
        actual = row_fields.get(field, "")
        if field == "Assay Comments":
            actual_lower = actual.lower()
            if isinstance(expected, list):
                if not any(str(v).lower() in actual_lower for v in expected):
                    return False
            elif str(expected).lower() not in actual_lower:
                return False
        elif isinstance(expected, list):
            # Any of the listed values matches
            if field == "Source Tissue":
                if actual.lower() not in {v.lower() for v in expected}:
                    return False
            elif actual not in expected:
                return False
        else:
            if field == "Source Tissue":
                if actual.lower() != str(expected).lower():
                    return False
            elif actual != expected:
                return False
    return True


@lru_cache(maxsize=16384)
@cache
def classify_ms_row(
    process_type: str,
    disease: str,
    culture_condition: str,
    source_tissue: str = "",
    cell_name: str = "",
    pmid: int | str = "",
    mhc_restriction: str = "",
    submission_id: str = "",
    assay_comments: str = "",
) -> dict[str, bool | str]:
    """Classify a public-MS row into curated source-context flags.

    Uses data-driven PMID overrides and tissue categories from YAML files.

    Cached by the full argument tuple. In the IEDB scanner the same
    (process_type, disease, culture_condition, source_tissue, cell_name)
    tuple repeats across all rows of a study with at most a handful of
    unique mhc_restriction values per sample, so the cache keeps a
    couple tens of thousands of entries at most vs millions of row
    classifications. The returned dict is not mutated by any known
    caller (both scanner.py and supplement.py splat it with ``**`` or
    ``record.update`` — no in-place edits), so sharing the cached
    instance across rows is safe.

    Parameters
    ----------
    process_type
        IEDB "Process Type" field.
    disease
        IEDB "Disease" field.
    culture_condition
        IEDB "Culture Condition" field.
    source_tissue
        IEDB "Source Tissue" field.
    cell_name
        IEDB "Cell Name" field.
    pmid
        PubMed ID for per-study override lookup.
    mhc_restriction
        IEDB "MHC Restriction" field value. Used to check whether
        the reported allele is endogenous to a mono-allelic host.
    assay_comments
        IEDB "Assay Comments" field. Some studies (e.g. PMID 29789417,
        Löffler 2018 CRC) tag per-row arm provenance only in the
        free-text Assay Comments column, so PMID rules can match on it
        via substring search.

    Returns
    -------
    dict[str, bool | str]
        Source flags plus ``cell_line_name``.
    """
    process_type = str(process_type).strip() if pd.notna(process_type) else ""
    disease = str(disease).strip() if pd.notna(disease) else ""
    culture_condition = str(culture_condition).strip() if pd.notna(culture_condition) else ""
    source_tissue_str = str(source_tissue).strip() if pd.notna(source_tissue) else ""
    source_tissue_lower = source_tissue_str.lower()
    cell_name_str = str(cell_name).strip() if pd.notna(cell_name) else ""
    assay_comments_str = str(assay_comments).strip() if pd.notna(assay_comments) else ""

    categories = load_tissue_categories()
    overrides = load_pmid_overrides()

    # Parse PMID and submission_id
    pmid_int = None
    if pmid:
        with contextlib.suppress(ValueError, TypeError):
            pmid_int = int(pmid)
    submission_id_str = str(submission_id).strip() if pd.notna(submission_id) else ""

    # Base signals
    is_ex_vivo = culture_condition == "Direct Ex Vivo"
    is_cell_line = culture_condition in (
        "Cell Line / Clone",
        "Cell Line / Clone (EBV transformed, B-LCL)",
    )
    is_ebv_lcl = culture_condition == "Cell Line / Clone (EBV transformed, B-LCL)"
    is_reproductive = source_tissue_lower in categories["reproductive"]
    is_thymus = source_tissue_lower in categories["thymus"]

    # Auto-detect activated APCs: DCs/macrophages from blood
    is_activated_apc = (
        cell_name_str.lower() in categories["activated_apc_cell_names"]
        and source_tissue_lower in categories["activated_apc_tissues"]
    )

    # ── Apply PMID override (three-level specificity) ─────────────────
    # Level 1: conditional rules (checked first, in order)
    # Level 2: PMID-level default override
    # Level 3: no match → fall through to structured-field classification
    effective_override = None
    # Look up override by PMID (int) or submission_id (str)
    entry = None
    if pmid_int and pmid_int in overrides:
        entry = overrides[pmid_int]
    elif submission_id_str and submission_id_str in overrides:
        entry = overrides[submission_id_str]

    if entry is not None:
        # Build row fields for condition matching
        row_fields = {
            "Source Tissue": source_tissue_str,
            "Culture Condition": culture_condition,
            "Cell Name": cell_name_str,
            "Disease": disease,
            "Process Type": process_type,
            "Assay Comments": assay_comments_str,
        }

        # Level 1: check conditional rules
        for rule in entry.get("rules", []):
            condition = rule.get("condition", {})
            if _matches_condition(row_fields, condition):
                effective_override = rule.get("override")
                break

        # Level 2: PMID-level default (only if no rule matched)
        if effective_override is None:
            effective_override = entry.get("override")

    # ── Classification ──────────────────────────────────────────────────
    if effective_override == "cancer_patient":
        is_cancer = True
        is_adjacent = False
        is_activated_apc = False
    elif effective_override == "adjacent":
        is_cancer = False
        is_adjacent = True
        is_activated_apc = False
    elif effective_override == "activated_apc":
        is_cancer = False
        is_adjacent = False
        is_activated_apc = True
    elif effective_override == "cell_line":
        is_cancer = not is_ebv_lcl  # EBV-LCLs are not cancer
        is_adjacent = False
        is_activated_apc = False
    elif effective_override == "ebv_lcl":
        is_cancer = False
        is_adjacent = False
        is_activated_apc = False
        is_ebv_lcl = True  # force even if IEDB culture_condition is wrong
        is_cell_line = True
    elif effective_override == "healthy":
        is_cancer = False
        is_adjacent = False
        is_activated_apc = False
    else:
        # Default: non-EBV cell lines are cancer-derived
        is_cancer = process_type == "Occurrence of cancer" or (is_cell_line and not is_ebv_lcl)
        is_adjacent = False

    # Healthy requires: ex vivo, no cancer/adjacent/apc, no disease.
    # When override is "healthy", force the healthy path regardless of
    # process_type / disease fields (the override corrects bad IEDB metadata).
    is_healthy_donor = effective_override == "healthy" or (
        is_ex_vivo
        and not is_cancer
        and not is_adjacent
        and not is_activated_apc
        and process_type == "No immunization"
        and disease in ("healthy", "")
    )

    cl_name = cell_name_str if (is_cell_line or is_ebv_lcl) else ""

    # Mono-allelic detection: cell_name alias matching
    is_monoallelic = False
    mono_host = ""
    if is_cell_line or is_ebv_lcl:
        is_monoallelic, mono_host = detect_monoallelic(cell_name_str, mhc_restriction)

    # PMID-level mono-allelic override — is_monoallelic is a SAMPLE-level
    # claim, so the override applies per-row only when the row's allele
    # is resolved (four/two-digit or serological).  Class-only /
    # unresolved rows cannot be mono-allelic because we don't know which
    # allele produced the peptide — this single gate correctly de-flags
    # validation rows in mixed papers (Sarkizova 2020's 12 patient
    # tumors all have mhc_restriction == "HLA class I" in IEDB).  We
    # intentionally do NOT gate on cell_name: IEDB frequently records
    # the host under a wrong specific label (Trolle 2016's 721.221
    # transfectants appear as "HeLa cells-Epithelial cell"), and the
    # whole purpose of the PMID override is to correct that annotation.
    # ``entry`` is the PMID-level override dict looked up above (not
    # the rule-specific override), so it remains bound even when no
    # rules matched.
    if not is_monoallelic and entry is not None:
        host = entry.get("mono_allelic_host")
        if host and _is_resolved_allele(mhc_restriction):
            is_monoallelic = True
            mono_host = host
        # Method-based mono-allelic (e.g., MAPTAC tagged pulldown) —
        # not a cell line, so not in monoallelic_lines.yaml.
        method = entry.get("mono_allelic_method")
        if not is_monoallelic and method and _is_resolved_allele(mhc_restriction):
            is_monoallelic = True
            mono_host = method

    return {
        "src_cancer": is_cancer,
        "src_adjacent_to_tumor": is_adjacent,
        "src_activated_apc": is_activated_apc,
        "src_healthy_tissue": is_healthy_donor and not is_reproductive and not is_thymus,
        "src_healthy_thymus": is_healthy_donor and is_thymus,
        "src_healthy_reproductive": is_healthy_donor and is_reproductive,
        "src_cell_line": is_cell_line,
        "src_ebv_lcl": is_ebv_lcl,
        "src_ex_vivo": is_ex_vivo,
        "cell_line_name": cl_name,
        "is_monoallelic": is_monoallelic,
        "monoallelic_host": mono_host,
        "allele_resolution": classify_allele_resolution(mhc_restriction),
        "serotype": allele_to_serotype(mhc_restriction),
        "serotypes": ";".join(allele_to_all_serotypes(mhc_restriction)),
        "mhc_species": classify_mhc_species(mhc_restriction),
    }


_BINDING_ASSAY_KEYWORDS = re.compile(
    r"microarray|binding|refolding|MEDi|iTopia|yeast display|mammalian epitope display",
    re.IGNORECASE,
)

_COMPETITIVE_BINDING_ASSAY_KEYWORDS = re.compile(
    r"acid strip(?:ped)?|reference peptide|IC50|(?:human\s+)?(?:beta2m|β2m)",
    re.IGNORECASE,
)


@cache
def is_binding_assay(qualitative_measurement: str, assay_comments: str) -> bool:
    """Classify whether an observation is from a binding assay vs MS elution.

    Returns True for binding assay data (peptide microarrays, refolding
    assays, MEDi display, etc.) which should be excluded from
    immunopeptidome-focused analyses.

    Cached by ``(qualitative_measurement, assay_comments)`` tuple —
    qualitative_measurement is drawn from a handful of values and
    assay_comments is highly repetitive across IEDB rows, so the cache
    quickly saturates at O(a few thousand) distinct keys vs millions of
    per-row calls in the scanner.
    """
    qm = qualitative_measurement.strip() if qualitative_measurement else ""
    # Negative results and quantitative tiers are binding assays
    if qm in ("Negative", "Positive-High", "Positive-Intermediate", "Positive-Low"):
        return True
    # "Positive" rows can still be binding assays when the comments
    # describe the assay format explicitly.
    return bool(
        qm == "Positive"
        and assay_comments
        and (
            _BINDING_ASSAY_KEYWORDS.search(assay_comments)
            or _COMPETITIVE_BINDING_ASSAY_KEYWORDS.search(assay_comments)
        )
    )


def is_cancer_specific(flags: dict[str, bool]) -> bool:
    """Test if a peptide's aggregated flags indicate cancer-specificity.

    Cancer-specific = found in cancer AND NOT found in healthy somatic tissue.
    """
    return bool(
        flags.get("found_in_cancer", False) and not flags.get("found_in_healthy_tissue", False)
    )
