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
    healthy_reproductive Direct ex vivo immune-privileged reproductive tissue
                        (testis, epididymis, placenta, embryo; CTA-safe in either sex)
    healthy_reproductive_female Direct ex vivo female reproductive tract / breast
                        (sex-stratified: a safety signal only in female patients)
    healthy_reproductive_male Direct ex vivo male reproductive tract
                        (sex-stratified: a safety signal only in male patients)
    ebv_lcl             EBV-transformed B-cell lines
    cell_line           Other cell lines (tumor / malignant-derived)
    noncancer_cell_line Non-malignant cell lines / clones derived from normal
                        cells — immortalized (hTERT / SV40-LT / engineered, e.g.
                        the ECN90 beta-cell line) or in-vitro-expanded primary
                        clones (e.g. activated CD4+ T-cell clones). IEDB tags
                        these "Cell Line / Clone"; they are a line/clone but NOT
                        cancer.
"""

from __future__ import annotations

import contextlib
import re
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cache, lru_cache
from os.path import basename, dirname, join
from types import MappingProxyType

import pandas as pd
import yaml
from mhcgnomes import Species

from .cell_name_parser import parse_cell_name


def _data_path(filename: str) -> str:
    return join(dirname(__file__), "data", filename)


def _asset_path(rel_path: str) -> str:
    """Resolve a *registry* data asset, fetching it if it is not installed.

    Mirrors :func:`hitlist.bulk_proteomics._bulk_data_path`.  ``_data_path``
    is a bare join with no fallback, so a registry-known file missing from
    the install raised ``FileNotFoundError`` out of ``pd.read_csv`` instead
    of being fetched from the ``data-assets-v1`` mirror (#348).

    Only for files listed in ``data_assets.yaml``.  The bundled YAML config
    (``pmid_overrides.yaml``, ``tissue_categories.yaml``,
    ``monoallelic_lines.yaml``) is never externalized and keeps using
    :func:`_data_path`.
    """
    from .downloads import packaged_or_fetched

    return str(packaged_or_fetched(_data_path(rel_path), basename(rel_path)))


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
        Keys: ``reproductive``, ``reproductive_female``, ``reproductive_male``,
        ``thymus``, ``activated_apc_cell_names``, ``activated_apc_tissues``.
    """
    with open(_data_path("tissue_categories.yaml")) as f:
        data = yaml.safe_load(f)
    return {
        "reproductive": frozenset(data.get("reproductive", [])),
        "reproductive_female": frozenset(data.get("reproductive_female", [])),
        "reproductive_male": frozenset(data.get("reproductive_male", [])),
        "thymus": frozenset(data.get("thymus", [])),
        "activated_apc_cell_names": frozenset(data.get("activated_apc_cell_names", [])),
        "activated_apc_tissues": frozenset(data.get("activated_apc_tissues", [])),
    }


@lru_cache(maxsize=1)
def load_tissue_groups() -> dict[str, tuple[str, ...]]:
    """Anatomical roll-up groups for the ``--by-tissue`` display: ``{umbrella:
    (member organs, ...)}`` (e.g. ``"GI tract" -> ("Esophagus", "Colon", ...)``).

    Display-only; not used for classification.  Member organs (and the umbrella
    term itself) collapse to the umbrella by default.  See ``tissue_categories``.
    """
    with open(_data_path("tissue_categories.yaml")) as f:
        data = yaml.safe_load(f)
    return {umb: tuple(members or []) for umb, members in (data.get("tissue_groups") or {}).items()}


#: Provenance fields a PMID override may curate; the scanner fills each ONLY
#: where the IEDB/CEDAR row left it blank/"unidentified" (#307, #314).
_PROVENANCE_FIELDS = (
    "source_organism",
    "species",
    "source_tissue",
    "cell_name",
    "disease",
    "culture_condition",
)


@cache
def pmid_provenance(pmid) -> dict:
    """Per-PMID curated provenance fills for fields IEDB/CEDAR left blank.

    Some datasets omit the source proteome, source tissue, cell line, or disease
    even when the paper states them (self-peptidomes with blank source organism;
    epitope-identification papers with no recorded tissue/cell line). A PMID
    entry in ``pmid_overrides.yaml`` may carry any of ``source_organism`` /
    ``species`` / ``source_tissue`` / ``cell_name`` / ``disease``; the scanner
    fills each PER PAPER, ONLY where the row's value is unresolved (#307, #314).

    This is explicit per-paper curation, never a heuristic — e.g. random-library
    refolding assays (pig SLA-I, possum) deliberately get no ``source_organism``
    so their synthetic source stays blank rather than being mislabeled.

    Returns a dict of the curated fields present (empty if no entry / no fields).
    """
    if not pmid:
        return {}
    try:
        key = int(pmid)
    except (TypeError, ValueError):
        return {}
    entry = load_pmid_overrides().get(key)
    if not entry:
        return {}
    out = {f: str(entry[f]) for f in _PROVENANCE_FIELDS if entry.get(f)}
    if "source_organism" in out and "species" not in out:
        out["species"] = out["source_organism"]
    return out


def pmid_source_organism(pmid) -> tuple[str, str]:
    """Per-PMID curated ``(source_organism, species)`` (#307). Thin wrapper over
    :func:`pmid_provenance`; the scanner fills these where the row is blank."""
    p = pmid_provenance(pmid)
    return p.get("source_organism", ""), p.get("species", "") or p.get("source_organism", "")


@cache
def pmid_mhc_species_context(pmid) -> str:
    """Canonical curated species context for parsing an observation's MHC.

    The context is deliberately limited to explicit PMID curation. Raw source
    organism and host strings are separate biological axes and must not be
    guessed into an MHC species. Explicitly prefixed restrictions still win
    when they describe a legitimate engineered-MHC system.
    """
    provenance = pmid_provenance(pmid)
    return normalize_species(provenance.get("species", ""))


# ── Cached mhcgnomes parse ─────────────────────────────────────────────────


@cache
def _cached_parse(mhc_restriction: str, species_context: str = ""):
    """Cache mhcgnomes parse results.

    The cache key includes ``species_context``: a bare or ambiguous MHC
    designation can resolve differently in cattle, chicken, or the default
    human namespace. Uses an unbounded cache because the distinct
    ``(designation, context)`` vocabulary is small relative to the corpus.
    """
    try:
        from mhcgnomes import parse

        if species_context:
            return parse(mhc_restriction, species=species_context)
        return parse(mhc_restriction)
    except Exception:
        return None


@cache
def species_compatible(a: str, b: str) -> bool:
    """Whether two mhcgnomes species are equal or ancestor-compatible.

    Compatibility is directional only in the ontology implementation, so ask
    both ways. Sibling species remain incompatible even when they share a
    genus-level ancestor; this prevents a specific cattle context from
    accepting a specific buffalo result.
    """
    if not a or not b:
        return False
    left = Species.get(a)
    right = Species.get(b)
    if left is None or right is None:
        return False
    return bool(left.compatible_with(right) or right.compatible_with(left))


@cache
def _parse_with_context(mhc_restriction: str, species_context: str = ""):
    """Return ``(result, used_context, incompatible_guess_replaced)``.

    Parse without context first so an explicit HLA/BoLA/etc. designation can
    survive a legitimate cross-species experiment. If a context is available
    and differs from that result, try the constrained parse. A successful
    constrained result refines generic or inferred nomenclature; a failed one
    falls back to the explicit unconstrained result.
    """
    text = (mhc_restriction or "").strip()
    context = normalize_species(species_context)
    unconstrained = _cached_parse(text)
    if not context:
        return unconstrained, False, False

    unconstrained_species = str(getattr(getattr(unconstrained, "species", None), "name", ""))
    if unconstrained_species == context:
        return unconstrained, False, False

    contextual = _cached_parse(text, context)
    if contextual is None:
        return unconstrained, False, False

    incompatible = bool(
        unconstrained_species and not species_compatible(unconstrained_species, context)
    )
    return contextual, True, incompatible


@dataclass(frozen=True)
class MhcAnnotation:
    """Resolved identity and provenance for one MHC restriction.

    ``mhc_class_source`` is ``derived`` for one molecule, ``donor_set``
    for a consistent semicolon-separated molecule set, or ``export`` when
    the source-reported class must be retained. ``mhc_species_source`` uses
    mhcgnomes' ``explicit``/``default``/``inferred`` vocabulary plus
    ``context``, ``mixed``, and ``unresolved``. Resolution and serotype
    fields come from the same final restriction, so callers can persist
    :meth:`as_record_fields` atomically without metadata drift.
    """

    restriction: str
    mhc_species: str
    mhc_species_source: str
    mhc_species_context_disagrees: bool
    mhc_class: str
    mhc_class_reported: str
    mhc_class_source: str
    mhc_class_corrected: bool
    allele_resolution: str
    serotype: str
    serotypes: str

    def as_record_fields(self) -> dict[str, bool | str]:
        """Return the persisted scanner columns for this annotation."""
        return {
            "mhc_restriction": self.restriction,
            "mhc_species": self.mhc_species,
            "mhc_species_source": self.mhc_species_source,
            "mhc_species_context_disagrees": self.mhc_species_context_disagrees,
            "mhc_class": self.mhc_class,
            "mhc_class_reported": self.mhc_class_reported,
            "mhc_class_source": self.mhc_class_source,
            "mhc_class_corrected": self.mhc_class_corrected,
            "allele_resolution": self.allele_resolution,
            "serotype": self.serotype,
            "serotypes": self.serotypes,
        }


def _molecule_class(parsed) -> str:
    if type(parsed).__name__ not in _MHC_MOLECULE_TYPES:
        return ""
    return _FINE_MHC_CLASS_TO_TOKEN.get(str(getattr(parsed, "mhc_class", "")), "")


@cache
def resolve_mhc_annotation(
    mhc_restriction: str,
    reported_mhc_class: str = "",
    species_context: str = "",
) -> MhcAnnotation:
    """Resolve a restriction's normalized identity, class, and provenance.

    Parameters
    ----------
    mhc_restriction
        Source restriction or a semicolon-separated donor candidate set.
    reported_mhc_class
        Source-reported class. Preserved verbatim in ``mhc_class_reported``
        and used only when molecule-level derivation is unavailable.
    species_context
        Optional curated study species. It constrains ambiguous parsing but
        never erases an explicit cross-species MHC designation that cannot be
        parsed in that context.

    Returns
    -------
    MhcAnnotation
        Immutable normalized identity, provenance, allele resolution, and
        serotype projection. :meth:`MhcAnnotation.as_record_fields` returns
        the complete set of scanner columns owned by this resolver.
    """
    text = (mhc_restriction or "").strip()
    reported_raw = (reported_mhc_class or "").strip()
    reported = normalize_mhc_class_token(reported_raw)
    parts = [part.strip() for part in text.split(";") if part.strip()] if ";" in text else [text]

    normalized_parts: list[str] = []
    parsed_parts: list = []
    species: set[str] = set()
    species_sources: set[str] = set()
    context_disagrees = False
    for part in parts:
        parsed, used_context, incompatible = _parse_with_context(part, species_context)
        parsed_parts.append(parsed)
        if type(parsed).__name__ in _MHC_MOLECULE_TYPES:
            normalized_parts.append(parsed.to_string())
        else:
            normalized_parts.append(part)
        parsed_species = str(getattr(getattr(parsed, "species", None), "name", ""))
        if parsed_species:
            species.add(parsed_species)
        source = "context" if used_context else str(getattr(parsed, "species_source", "") or "")
        if source:
            species_sources.add(source)
        context_disagrees = context_disagrees or incompatible

    normalized = ";".join(normalized_parts) if len(parts) > 1 else normalized_parts[0]
    resolved_species = ";".join(sorted(species))
    if not species_sources:
        species_source = "unresolved"
    elif len(species_sources) == 1:
        species_source = next(iter(species_sources))
    else:
        species_source = "mixed"

    derived_class = ""
    class_source = "export"
    if len(parts) > 1:
        component_classes = [_molecule_class(parsed) for parsed in parsed_parts]
        if all(component_classes) and len(set(component_classes)) == 1:
            derived_class = component_classes[0]
            class_source = "donor_set"
    elif parsed_parts:
        derived_class = _molecule_class(parsed_parts[0])
        if derived_class:
            class_source = "derived"

    resolved_class = derived_class or reported
    corrected = bool(derived_class and reported and derived_class != reported)
    component_serotypes = tuple(
        dict.fromkeys(
            serotype for part in normalized_parts for serotype in allele_to_all_serotypes(part)
        )
    )
    return MhcAnnotation(
        restriction=normalized,
        mhc_species=resolved_species,
        mhc_species_source=species_source,
        mhc_species_context_disagrees=context_disagrees,
        mhc_class=resolved_class,
        mhc_class_reported=reported_raw,
        mhc_class_source=class_source,
        mhc_class_corrected=corrected,
        allele_resolution=classify_allele_resolution(normalized),
        serotype=component_serotypes[0] if component_serotypes else "",
        serotypes=";".join(component_serotypes),
    )


# ── MHC species ────────────────────────────────────────────────────────────


@cache
def classify_mhc_species(mhc_restriction: str, species_context: str = "") -> str:
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

    result, _, _ = _parse_with_context(mhc_restriction, species_context)
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

    - either side normalizes to empty (``normalize_species`` already
      handles None / empty / whitespace).
    - either side resolves to a genus outside the MHC-bearing vertebrate
      host whitelist — viral / bacterial / parasite source on a host MHC
      is normal infection biology, not an engineered cross-species system,
      and IEDB sentinels like ``"unidentified"`` / ``"unknown"`` fall
      through to the same rejection without an explicit allowlist.
    - normalized genus tokens match — substrains and subspecies
      (``"Mus musculus C57BL/6"`` vs ``"Mus musculus"``;
      ``"Canis lupus familiaris"`` vs ``"Canis sp."``) are not chimeric.

    Cached because the (source_organism, mhc_species) tuple repeats heavily
    across an observations index of millions of rows but resolves to only a
    couple hundred unique pairs.
    """
    src_norm = normalize_species(source_organism)
    mhc_norm = normalize_species(mhc_species)
    if not src_norm or not mhc_norm:
        return False
    src_genus = src_norm.split()[0].lower()
    mhc_genus = mhc_norm.split()[0].lower()
    return (
        src_genus in _MHC_BEARING_HOST_GENERA
        and mhc_genus in _MHC_BEARING_HOST_GENERA
        and src_genus != mhc_genus
    )


@cache
def is_engineered_mhc(source_organism: str, mhc_species: str, host: str) -> bool:
    """True iff the MHC molecule is heterologous to the host cells / tissue.

    Distinguishes engineered-MHC systems (HLA-transgenic rats, NetH2pan
    training, allogeneic HLA transfectants — where the cells live in one
    species but display the MHC of another) from heterologous-antigen
    studies (Lewis-rat EAE with guinea-pig MBP, bovine-allergen on HLA —
    where the cells use their *native* MHC but present a foreign protein).

    Both scenarios trigger :func:`is_chimeric_system` because in both
    ``source_organism != mhc_species``. The discriminator is the IEDB
    ``host`` field: in an engineered-MHC system the host genus matches
    the source proteome (the cells are native, the MHC is transgenic);
    in a heterologous-antigen system the host genus matches the MHC
    species (the cells are native, the antigen is foreign).

    Returns ``False`` when:

    - the row is not chimeric (same source / MHC genus → no question of
      engineered MHC).
    - ``host`` is empty / unrecognized (chimerism cannot be classified
      without the host axis; conservative default).
    - host genus matches the MHC genus (heterologous antigen, not
      engineered MHC).

    Cached on the (source, mhc, host) triple — same call shape and
    cardinality as :func:`is_chimeric_system`.
    """
    if not is_chimeric_system(source_organism, mhc_species):
        return False
    host_norm = normalize_species(host)
    if not host_norm:
        return False
    host_genus = host_norm.split()[0].lower()
    if host_genus not in _MHC_BEARING_HOST_GENERA:
        return False
    src_genus = normalize_species(source_organism).split()[0].lower()
    mhc_genus = normalize_species(mhc_species).split()[0].lower()
    # Engineered MHC: cells are from the source species, MHC differs.
    return host_genus == src_genus and host_genus != mhc_genus


@cache
def is_xenograft(source_organism: str, host: str, mhc_species: str) -> bool:
    """True iff the eluted cells are a different vertebrate genus than the host.

    A xenograft decouples the **proteome species** (the cells the peptide was
    sequenced from) from the **host species** the cells lived in at MS sampling
    — e.g. a human or dog tumor grown in an NSG mouse. This is the host-axis
    counterpart of :func:`is_engineered_mhc`.

    The discriminator that separates a true xenograft from a *heterologous
    antigen* study (a foreign protein presented by the host's own native
    cells — e.g. bovine allergen on a human-MHC human cell) is the same as in
    :func:`is_engineered_mhc`: the IEDB ``host`` field. In a xenograft the
    grafted cells are NOT the host's own, so the displayed MHC is not the
    host's native MHC (``host`` genus differs from both the proteome and the
    MHC). In a heterologous-antigen study the cells are native (``host`` genus
    matches the ``mhc_species`` genus) and only the antigen is foreign.

    Returns ``False`` when any side normalizes to empty, any genus is outside
    the MHC-bearing vertebrate whitelist (a viral / bacterial *antigen* source
    on an animal host is ordinary infection biology, not a xenograft), the
    proteome and host genera match (native cells / substrains), or the host
    genus matches the MHC genus (heterologous antigen on native cells).

    Cached on the (source, host, mhc) triple — same call shape and cardinality
    as :func:`is_engineered_mhc`.
    """
    src_norm = normalize_species(source_organism)
    host_norm = normalize_species(host)
    mhc_norm = normalize_species(mhc_species)
    if not src_norm or not host_norm or not mhc_norm:
        return False
    src_genus = src_norm.split()[0].lower()
    host_genus = host_norm.split()[0].lower()
    mhc_genus = mhc_norm.split()[0].lower()
    return (
        src_genus in _MHC_BEARING_HOST_GENERA
        and host_genus in _MHC_BEARING_HOST_GENERA
        and mhc_genus in _MHC_BEARING_HOST_GENERA
        and src_genus != host_genus
        and host_genus != mhc_genus
    )


# ── Non-peptide-presenting MHC molecules (#228) ──────────────────────────
#
# CD1, MR1, MIC, ULBP, RAET1, NKG2, and HFE are not peptide presenters.
# CD1 (a/b/c/d/e) presents lipids and glycolipids to NKT and CD1-restricted
# T cells; MR1 presents riboflavin-derived small-molecule metabolites to MAIT
# cells; MICA/MICB/RAET1/ULBP are stress-induced ligands for the NKG2D
# activating receptor on NK cells; HFE is involved in iron regulation.
#
# IEDB curates these alongside peptide-MHC rows but populates the
# ``peptide`` column with chemical names ("L-idopyranose 6-monomycolate")
# or compound identifiers ("HS44") rather than amino-acid sequences.
# Downstream consumers iterating ``observations.parquet`` for peptide-level
# work (motif analyses, length distributions, source-protein mapping,
# peptide-prediction models) silently produce nonsense on these rows.
#
# H2-M3 is intentionally excluded — it IS a class Ib peptide presenter
# (N-formyl peptides), so its single row in the corpus carries a real
# peptide sequence.
_NON_PEPTIDE_MHC_RE = re.compile(
    r"\bCD1(?:[a-eA-E]\d?|-\d)?\b"  # CD1, CD1a..e, CD1d2, CD1-2
    r"|\bMR1\b"
    r"|\bMICA\b|\bMICB\b"
    r"|\bRAET1[A-Z]?\b"
    r"|\bULBP\d?\b"
    # NKG2 is a NK-cell *receptor* family, not an MHC molecule — it
    # should never appear in IEDB's MHC-restriction column. Kept
    # defensive in case future curation drift puts it there; matches
    # the issue's stated whitelist.
    r"|\bNKG2[A-C]\b"
    r"|\bHFE\b",
    re.IGNORECASE,
)


@cache
def is_non_peptide_ligand(mhc_restriction: str) -> bool:
    """True iff ``mhc_restriction`` names a non-peptide-presenting MHC molecule.

    Detects CD1 family, MR1, MIC{A,B}, RAET1*, ULBP*, NKG2[A-C], and HFE
    in any normalized restriction string ("mouse-CD1d", "human-MR1",
    "cattle-CD1b3", "chicken-CD1-2", "human-MR1 K43A mutant", etc.).
    Classical and class-Ib peptide presenters (HLA-A/B/C/E/F/G, H2-K/D/L,
    H2-Q*, H2-T*, H2-M3, Patr-AL, BoLA-*) are not flagged.

    Rows flagged here carry lipid, glycolipid, or small-molecule
    identifiers in the ``peptide`` column, not amino-acid sequences.
    Default ``load_observations`` behavior excludes them; opt-in via
    ``exclude_non_peptide_ligand=False``.

    Cached on the unique mhc_restriction vocabulary (~hundreds of values
    across millions of rows).
    """
    if not mhc_restriction:
        return False
    return bool(_NON_PEPTIDE_MHC_RE.search(mhc_restriction))


# ── Allele normalization ──────────────────────────────────────────────────


@lru_cache(maxsize=4096)
def normalize_allele(raw: str, species_context: str = "") -> str:
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

    result, _, _ = _parse_with_context(cleaned, species_context)
    # Only return normalized form for actual alleles/genes/pairs
    # (not generic Species or Class-only designations like "HLA class I")
    if result is not None and type(result).__name__ in _MHC_MOLECULE_TYPES:
        return result.to_string()

    return cleaned


# ── MHC class (mhcgnomes-derived) ──────────────────────────────────────────

#: mhcgnomes reports a fine-grained class taxonomy.  Map it onto the three
#: tokens this corpus uses.  ``Ib`` / ``Ic`` / ``Id`` are the non-classical
#: class-I families (HLA-E/F/G, H2-Q/T/M, CD1, MR1); folding them into ``I``
#: would make ``--mhc-class I`` silently sweep up HLA-E rows, so they keep
#: their own token.
_FINE_MHC_CLASS_TO_TOKEN: dict[str, str] = {
    "Ia": "I",
    "Ib": "non-classical",
    "Ic": "non-classical",
    "Id": "non-classical",
    "IIa": "II",
    "IIb": "II",
    "I": "I",
    "II": "II",
}

#: mhcgnomes result types that name an actual MHC molecule.  Everything else
#: (``Species``, ``MhcClass``, ``Haplotype``) is either too coarse or a false
#: positive — notably ``parse("n/a")`` returns a *rat haplotype* ``RT1-n/A``,
#: so free-text fields must never be accepted on "it parsed" alone.
_MHC_MOLECULE_TYPES = frozenset({"Allele", "Gene", "Pair"})

#: Curators write ``non-classical``; IEDB exports write ``non classical``.
_MHC_CLASS_TOKEN_ALIASES: dict[str, str] = {
    "i": "I",
    "ii": "II",
    "non classical": "non-classical",
    "non-classical": "non-classical",
    "nonclassical": "non-classical",
}


@cache
def normalize_mhc_class_token(value: str) -> str:
    """Canonicalize a curated / IEDB ``mhc_class`` spelling.

    The YAML and the IEDB export disagree on the non-classical token —
    ``non-classical`` vs ``non classical`` — which meant the two could
    never compare equal and non-classical samples were unreachable from
    the observation join (issue #363).  Compound ``I+II`` values are
    normalized component-wise and rejoined.
    """
    text = (value or "").strip()
    if not text:
        return ""
    parts = [p.strip() for p in text.split("+") if p.strip()]
    out = [_MHC_CLASS_TOKEN_ALIASES.get(p.lower(), p) for p in parts]
    return "+".join(out)


@cache
def mhc_class_spellings(mhc_class: str) -> tuple[str, ...]:
    """Every stored spelling of a class token, for a parquet predicate.

    The corpus and the YAML disagree on the non-classical token, so an
    exact predicate on either one silently returns nothing (#363).
    Derived from :data:`_MHC_CLASS_TOKEN_ALIASES` rather than patched up
    per-case, so a spelling the normalizer already understands cannot be
    missed by the filter.
    """
    wanted = normalize_mhc_class_token(mhc_class)
    out = {mhc_class, wanted}
    out.update(
        alias for alias, canonical in _MHC_CLASS_TOKEN_ALIASES.items() if canonical == wanted
    )
    return tuple(sorted(out))


@cache
def mhc_class_of(mhc_restriction: str, species_context: str = "") -> str:
    """Canonical MHC class for an allele / gene / pair, via mhcgnomes.

    Returns ``"I"``, ``"II"``, ``"non-classical"`` or ``""`` (unknown).

    Derived rather than string-matched, so it works for every species
    mhcgnomes knows without a per-species table.  Validated against the
    built corpus: it reproduces the curated class on 2,922,227 observation
    rows and differs on 72, all of which are curation errors in the other
    direction (``Caja-E`` and ``Mamu-E*02:11`` are the marmoset / rhesus
    MHC-E genes and are non-classical, not classical).

    Semicolon-joined donor sets (the ``donor_set`` resolution emitted by
    #45, e.g. ``"HLA-DQB1*03:01;HLA-DRB1*15:01"``) are resolved from their
    components and return a class only when the components agree.
    """
    text = (mhc_restriction or "").strip()
    if not text:
        return ""
    if ";" in text:
        classes = [mhc_class_of(part, species_context) for part in text.split(";") if part.strip()]
        return classes[0] if classes and all(classes) and len(set(classes)) == 1 else ""
    parsed, _, _ = _parse_with_context(text, species_context)
    return _molecule_class(parsed)


@cache
def is_class_only_token(value: str) -> bool:
    """True when a string names a *class* rather than a specific molecule.

    Replaces prefix matching on ``"hla class"`` / ``"mhc class"``: mhcgnomes
    returns an ``MhcClass`` for exactly these, in any species' notation.
    The legacy ``"unknown"`` sentinel is still recognised explicitly since
    it carries the same meaning but does not parse.
    """
    text = (value or "").strip()
    if not text:
        return False
    if text.lower() in ("unknown", "not typed", "n/a", "na"):
        return True
    return type(_cached_parse(text)).__name__ == "MhcClass"


@cache
def mhc_species_of(mhc_field: str) -> str:
    """Species named by a curated ``mhc`` value, semicolon-joined.

    This is the ``mhc_species`` axis from docs/source-classification.md —
    the MHC molecule's species, as distinct from ``source_species`` (the
    proteome the peptide came from) and ``host_organism``.

    Resolves every token through :func:`classify_mhc_species`, not just
    the ones that are alleles.  ``HLA-DR15`` is a Serotype and
    ``BoLA-DR`` a Class2Locus; both name a species perfectly well, and
    filtering to Allele/Gene/Pair would report ``""`` for 19 curated
    rows whose species is not in doubt (#380 is about them not reaching
    the *allele* join, which is a different question).

    A genotype spanning several species — an engineered chimera — is
    joined rather than collapsed, so the signal survives on exactly the
    rows it matters for.  Returns ``""`` only when nothing resolves.
    """
    text = (mhc_field or "").strip()
    if not text:
        return ""
    if is_class_only_token(text):
        return classify_mhc_species(text)
    species = {classify_mhc_species(tok) for tok in re.split(r"[\s;,]+", text) if tok}
    species.discard("")
    return ";".join(sorted(species))


@cache
def species_axes_agreement(source_species: str, mhc_species: str) -> str:
    """Do a sample's source and MHC species axes describe one system?

    Returns ``"true"`` / ``"false"`` / ``""`` (undeterminable), matching
    the tri-state string convention used by ``profiled`` and
    ``is_control_arm``.  Deliberately *not* named as a predicate: it
    returns a string, and ``bool("false")`` is ``True``, so a
    question-shaped name would invert the first caller's branch.

    ``mhc_species`` may name several species (semicolon-joined); the
    axes agree when *any* of them is compatible with the source, which
    is what an engineered chimera looks like from the source side.

    ``"false"`` is not automatically an error — a human HLA transgene in
    a mouse legitimately disagrees (#46).  It means the two axes differ
    and the row deserves a look, which is how PMID 41459947 shipped a
    Prussian carp sample carrying human MHC.

    Compatibility is :meth:`mhcgnomes.Species.compatible_with`: same
    species, or one a direct ancestor of the other.  Both sides are
    resolved first — that method returns ``False`` both for "these
    differ" and for "the other name is not a species at all", so
    without the guard an unresolvable name would be reported as a
    contradiction rather than as unknown.
    """
    if not source_species or not mhc_species:
        return ""
    source = Species.get(source_species)
    if source is None:
        return ""
    derived = [Species.get(part) for part in mhc_species.split(";") if part]
    derived = [d for d in derived if d is not None]
    if not derived:
        return ""
    return "true" if any(source.compatible_with(d) for d in derived) else "false"


def allele_locus(allele_token: str) -> str:
    """The species-qualified locus an allele belongs to (``HLA-A``, ``BoLA-6``).

    Species-qualified on purpose: ``HLA-DRB3`` and ``BoLA-DRB3`` share a
    gene *name* but are different loci, so keying on the bare name would
    merge two species' alleles into one bucket.

    Returns ``""`` for anything with no single locus — an unparseable
    token, a class or serotype designation, or a class-II heterodimer
    pair.  A pair spans two loci by construction; decompose it with
    :func:`expand_allele_components` first and ask about each chain.

    Examples::

        allele_locus("HLA-A*02:01")       # "HLA-A"
        allele_locus("BoLA-6*013:01")     # "BoLA-6"
        allele_locus("HLA-DRB1*15:01")    # "HLA-DRB1"
        allele_locus("HLA-DR15")          # "" — a serotype, not a molecule
    """
    token = (allele_token or "").strip()
    if not token:
        return ""
    parsed = _cached_parse(token)
    if type(parsed).__name__ not in _MHC_MOLECULE_TYPES:
        return ""
    gene = getattr(parsed, "gene", None)
    if gene is None:
        return ""
    return gene.to_string()


def expand_allele_components(allele_token: str) -> list[str]:
    """Return an allele plus, for a class-II pair, its alpha/beta chains.

    ``ms_samples`` curate DP/DQ heterodimers as a paired string
    (``"HLA-DPB1*06:01/DPA1*01:03"``) while many IEDB rows report only one
    chain (``"HLA-DPB1*06:01"``).  Emitting the pair *plus* each chain lets
    a single-chain observation match a heterodimer sample (#151).

    mhcgnomes splits the pair, so the chain strings come back canonical for
    any species rather than depending on where the ``HLA-`` prefix happened
    to sit in the curated text.  Non-pairs pass through unchanged.
    """
    token = (allele_token or "").strip()
    if not token:
        return []
    out = [token]
    parsed = _cached_parse(token)
    if type(parsed).__name__ == "Pair":
        for chain in (parsed.alpha, parsed.beta):
            text = chain.to_string()
            if text and text not in out:
                out.append(text)
    return out


def extract_allele_tokens(text: str) -> list[str]:
    """Pull MHC molecule tokens out of a free-text ``mhc`` field.

    Replaces an ``(?:HLA-)?[A-Z]+\\d*\\*\\d{2,4}:\\d{2,4}`` regex that
    encoded HLA's digit syntax and therefore silently dropped every
    non-human allele — ``H-2Kb``, ``H2-K*b``, ``H-2Q1`` and ``Patr-AL`` all
    returned nothing.  Splitting on separators and asking mhcgnomes what
    each token is works for every species it knows.

    Only ``Allele`` / ``Gene`` / ``Pair`` results are accepted.  That
    matters: mhcgnomes resolves ``"n/a"`` to the rat haplotype ``RT1-n/A``,
    so accepting anything that merely parses would inject junk from
    free-text curation fields.
    """
    if not text:
        return []
    out: list[str] = []
    for raw in re.split(r"[\s;,]+", str(text)):
        token = raw.strip()
        if not token:
            continue
        parsed = _cached_parse(token)
        if type(parsed).__name__ not in _MHC_MOLECULE_TYPES:
            continue
        canonical = parsed.to_string()
        if canonical and canonical not in out:
            out.append(canonical)
    return out


# ── Allele resolution ──────────────────────────────────────────────────────

#: Resolution tiers, ordered from most to least specific.
#: ``donor_set`` is the multi-allele restriction emitted post-#45 when
#: a row's actual presenting MHC narrows to the donor's typed alleles
#: (or a specific subset via per-peptide attribution) — strictly more
#: specific than ``class_only`` (which is "any allele in this class")
#: but less specific than ``four_digit`` (which is "this exact allele").
ALLELE_RESOLUTION_ORDER: list[str] = [
    "four_digit",
    "donor_set",
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
        IEDB "MHC Restriction" field value, or a semicolon-joined
        multi-allele set emitted post-#45 (``"HLA-A*02:01;HLA-A*03:01;..."``).

    Returns
    -------
    str
        One of ``"four_digit"``, ``"donor_set"``, ``"two_digit"``,
        ``"serological"``, ``"class_only"``, ``"unresolved"``.
    """
    if not mhc_restriction:
        return "unresolved"

    # Multi-allele set (#45): semicolon-joined 4-digit alleles. Each
    # token must parse as a 4-digit allele; otherwise we fall through.
    if ";" in mhc_restriction:
        tokens = [t.strip() for t in mhc_restriction.split(";") if t.strip()]
        if len(tokens) > 1 and all(_looks_like_four_digit_allele(t) for t in tokens):
            return "donor_set"

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
_LOCUS_SEROTYPE_NAME_RE = re.compile(r"^(A|B|C|DR|DQ|DP|DM|DO)(\d+)$")


def _serotype_specificity_rank(name: str) -> int:
    """Lower = more specific / preferred as the canonical serotype.

    0: locus-specific (A24, B57, DR15, ...) — what a clinician usually means
    1: public epitopes (Bw4, Bw6, C1, C2, ...) — orthogonal axis, less useful
       as the canonical answer to "what serotype is this allele?"
    """
    return 0 if _LOCUS_SEROTYPE_RE.match(name) else 1


def _broader_locus_serotype_name(name: str, known_names: set[str]) -> str:
    """Return the nearest broader locus-specific serotype, if one exists.

    Some mhcgnomes entries are split serotypes like ``A2403`` or ``DR1404``.
    A query for the broad family (``A24`` / ``DR14``) should also match those
    split members — so when a broader parent serotype is itself present in the
    reference table, an allele carrying the split serotype is tagged with the
    parent too.  Returns ``""`` when there is no broader parent (``A2`` has no
    shorter form; ``A24`` is already two digits)."""
    match = _LOCUS_SEROTYPE_NAME_RE.match(name)
    if not match:
        return ""
    locus, digits = match.groups()
    if len(digits) <= 2:
        return ""
    for cut in range(len(digits) - 1, 0, -1):
        prefix = str(int(digits[:cut]))
        candidate = f"{locus}{prefix}"
        if candidate != name and candidate in known_names:
            return candidate
    return ""


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
    known_names = set(hla)
    for sero_name, allele_list in hla.items():
        # A split serotype (A2403) also implies its broad parent (A24) when
        # that parent serotype exists in the table, so broad queries match
        # split members.
        broader_name = _broader_locus_serotype_name(sero_name, known_names)
        names_for_sero = [sero_name] if not broader_name else [sero_name, broader_name]
        for allele_str in allele_list:
            reverse.setdefault(allele_str, []).extend(names_for_sero)

    return {
        allele: tuple(
            f"HLA-{s}"
            for s in sorted(set(names), key=lambda n: (_serotype_specificity_rank(n), len(n), n))
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


@lru_cache(maxsize=1)
def _ebv_lcl_mono_hosts() -> frozenset:
    """Canonical names of mono-allelic hosts that are EBV-transformed B-LCLs.

    Peptidomes produced on these hosts (721.221, C1R) are EBV-immortalized
    B-cell material, not tumors, so they classify as ``ebv_lcl`` (never
    ``src_cancer``) regardless of IEDB's inconsistent culture-condition
    tagging.  Flagged via ``ebv_lcl: true`` in ``monoallelic_lines.yaml``.
    """
    return frozenset(e["name"] for e in load_monoallelic_lines() if e.get("ebv_lcl"))


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


# ── Exact-allele set expansion (issue #137) ────────────────────────────────


_SET_PROVENANCE_VALUES = (
    "exact",
    "peptide_attribution",
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


#: mhcgnomes result types that name a *set* of molecules rather than one.
#: ``Serotype`` carries a member list and can be expanded; the rest name a
#: locus or a whole class and expand to nothing by design.
_MHC_SEROTYPE_TYPES = frozenset({"Serotype"})
_MHC_IMPRECISE_TYPES = frozenset({"Class2Locus", "MhcClass", "Species"})


@dataclass(frozen=True)
class SampleMhcCandidates:
    """What a curated ``ms_samples[].mhc`` value licenses as an attribution claim.

    A sample's ``mhc`` field is written at whatever precision the study
    reported, and three precisions occur.  Conflating them is how #380
    happened, so they are kept apart here:

    ``exact``
        Molecules named outright — ``HLA-A*02:01``, ``H2-K*b``,
        ``BoLA-6*013:01``.  Safe to report to a reader as an observed
        allele.
    ``serotypes`` / ``serotype_alleles``
        A serological designation — ``HLA-DR15``, ``HLA-DQ8``.  The study
        typed to serotype, not to allele.  ``serotype_alleles`` is the
        member set from the IPD-IMGT/HLA catalog, which is a legitimate
        *candidate* set for the allele-level join; ``serotypes`` is what
        should be shown to a reader.  Merging the two would turn "this
        donor is DR15" into the false claim "this peptide was observed on
        DRB1*15:01".
    ``imprecise``
        A locus or class designation — ``SLA-DR``, ``BoLA-DR``,
        ``Bos taurus class I``.  These name no allele, and expanding one
        would fabricate a restriction the study never reported.  Carried
        rather than dropped so a caller can say *why* a sample has no
        candidates instead of treating it as an unparseable typo.

    Notes
    -----
    Before #380 everything except ``exact`` yielded the empty set, so
    seven samples across five PMIDs fell out of the allele-level join
    silently — indistinguishable from a curation error.
    """

    exact: frozenset[str] = frozenset()
    serotypes: tuple[str, ...] = ()
    serotype_alleles: frozenset[str] = frozenset()
    imprecise: tuple[str, ...] = ()

    @property
    def join_alleles(self) -> frozenset[str]:
        """Candidate alleles for the allele-level join.

        Includes serotype members: for matching purposes "the donor is one
        of these seven DRB1*15 alleles" is a real narrowing, and strictly
        more specific than the study-wide class pool the sample otherwise
        falls back to.  Use :attr:`exact` anywhere the answer is presented
        as an allele that was actually observed.
        """
        return self.exact | self.serotype_alleles

    @property
    def is_empty(self) -> bool:
        """True when the field named nothing at all.

        Distinct from "named something imprecise": an empty result means
        the string parsed to no MHC entity, which is a curation defect
        worth surfacing, whereas ``imprecise`` is a faithful record of a
        study that only typed to locus or class.
        """
        return not (self.exact or self.serotypes or self.imprecise)


def sample_mhc_candidates(mhc_field) -> SampleMhcCandidates:
    """Classify a curated ``ms_samples[].mhc`` value by precision.

    The single place a curated sample genotype is turned into attribution
    candidates, so the allele-level join and the peptide summary cannot
    drift apart about what a given designation means (#380).

    Parameters
    ----------
    mhc_field
        A ``ms_samples[].mhc`` value: a string (``"HLA-A*02:01"``, the
        bare space-joined ``"A*02:01 A*24:02 B*15:01"``, a serotype, or a
        locus/class designation), a list of any of those, or ``None``.

    Returns
    -------
    SampleMhcCandidates
        Empty when the field is absent or names no MHC entity.

    Examples
    --------
    >>> sorted(sample_mhc_candidates("HLA-A*02:01 HLA-B*07:02").exact)
    ['HLA-A*02:01', 'HLA-B*07:02']
    >>> c = sample_mhc_candidates("HLA-DQ8")
    >>> c.exact, c.serotypes
    (frozenset(), ('HLA-DQ8',))
    >>> sorted(c.serotype_alleles)[:2]
    ['HLA-DQB1*03:02', 'HLA-DQB1*03:05']
    >>> sample_mhc_candidates("BoLA-DR").imprecise
    ('BoLA-DR',)

    See Also
    --------
    serotype_to_alleles : the serotype expansion this uses.
    """
    if mhc_field is None:
        return SampleMhcCandidates()
    if isinstance(mhc_field, (list, tuple)):
        parts = [sample_mhc_candidates(item) for item in mhc_field]
        return SampleMhcCandidates(
            exact=frozenset().union(*(p.exact for p in parts)) if parts else frozenset(),
            serotypes=tuple(dict.fromkeys(s for p in parts for s in p.serotypes)),
            serotype_alleles=(
                frozenset().union(*(p.serotype_alleles for p in parts)) if parts else frozenset()
            ),
            imprecise=tuple(dict.fromkeys(s for p in parts for s in p.imprecise)),
        )
    if not isinstance(mhc_field, str):
        return SampleMhcCandidates()

    # Whole-string check first.  Class and locus designations are often
    # multi-word ("Bos taurus class I"), and the token split below would
    # shred one into "Bos" / "taurus" / "class" / "I" — none of which
    # parses — reporting a faithfully curated sentinel as empty.
    whole = mhc_field.strip()
    whole_parsed = _cached_parse(whole) if whole else None
    whole_kind = type(whole_parsed).__name__
    if whole_kind in _MHC_SEROTYPE_TYPES:
        name = whole_parsed.to_string()
        return SampleMhcCandidates(
            serotypes=(name,), serotype_alleles=frozenset(serotype_to_alleles(name))
        )
    if whole_kind in _MHC_IMPRECISE_TYPES:
        return SampleMhcCandidates(imprecise=(normalize_allele(whole),))

    exact: list[str] = []
    serotypes: list[str] = []
    serotype_alleles: set[str] = set()
    imprecise: list[str] = []
    for raw in re.split(r"[\s;,]+", mhc_field):
        token = raw.strip()
        if not token:
            continue
        parsed = _cached_parse(token)
        kind = type(parsed).__name__
        if kind in _MHC_MOLECULE_TYPES:
            # Canonical form via ``to_string`` exactly as
            # :func:`extract_allele_tokens` does, so ``exact`` stays
            # byte-identical to what the pre-#380 path produced.
            name = parsed.to_string()
            if name and name not in exact:
                exact.append(name)
        elif kind in _MHC_SEROTYPE_TYPES:
            name = parsed.to_string()
            if name and name not in serotypes:
                serotypes.append(name)
                serotype_alleles.update(serotype_to_alleles(name))
        elif kind in _MHC_IMPRECISE_TYPES:
            name = normalize_allele(token)
            if name and name not in imprecise:
                imprecise.append(name)
    return SampleMhcCandidates(
        exact=frozenset(exact),
        serotypes=tuple(serotypes),
        serotype_alleles=frozenset(serotype_alleles),
        imprecise=tuple(imprecise),
    )


def _parse_sample_mhc_field(mhc_field) -> frozenset[str]:
    """Parse a ``ms_samples[].mhc`` value into a normalized allele set.

    Thin wrapper over :func:`sample_mhc_candidates`, kept because the
    per-peptide attribution path wants one flat candidate set rather than
    the precision breakdown.  Returns
    :attr:`SampleMhcCandidates.join_alleles`, so a serotype-typed sample
    contributes its member alleles as candidates instead of dropping out
    of the join entirely (#380).

    ms_samples curators use mixed formats — some entries are
    ``"HLA-A*01:01"`` (HLA-prefixed) and others are ``"A*02:01 A*24:02
    B*15:01 ..."`` (bare, space-joined).  Both shapes carry valid donor
    genotypes and normalize through mhcgnomes to the canonical form used
    elsewhere (``HLA-A*02:01``).

    Non-human genotypes are deliberately supported — ``"H-2Kb H-2Db"``,
    ``"H-2Q1 H-2Q2"`` and ``"Patr-AL"`` all resolve.  An HLA-shaped regex
    used to do the extraction, which silently returned nothing for every
    one of them, so per-peptide attribution could never narrow a mouse
    sample's candidate alleles.

    Callers presenting the result to a reader as an *observed* allele
    should use :func:`sample_mhc_candidates` directly and read
    :attr:`SampleMhcCandidates.exact`, which excludes serotype members.
    """
    return sample_mhc_candidates(mhc_field).join_alleles


@lru_cache(maxsize=512)
def _pmid_sample_alleles(pmid_int: int) -> dict[str, frozenset[str]]:
    """Map ``sample_label → frozenset(4-digit alleles)`` for a PMID's ms_samples.

    Many studies curate the donor / patient genotype on each
    ``ms_samples`` entry as the ``mhc:`` value — either as
    ``"HLA-A*01:01"`` or as a bare space-joined string like
    ``"A*02:01 A*24:02 B*15:01 ..."``.  Both shapes are accepted (see
    :func:`_parse_sample_mhc_field`) and normalized to canonical
    ``HLA-A*02:01`` form.  Per-peptide attribution overrides (#45) use
    this map to narrow a row's candidate-allele set from the
    disease-wide union down to the specific donor(s) the peptide was
    observed in.

    Empty mapping if no override or no ms_samples entries are present.
    """
    overrides = load_pmid_overrides()
    entry = overrides.get(pmid_int)
    if entry is None:
        return {}
    out: dict[str, frozenset[str]] = {}
    for sample in entry.get("ms_samples") or []:
        label = sample.get("sample_label", "")
        if not label:
            continue
        alleles = _parse_sample_mhc_field(sample.get("mhc"))
        if alleles:
            out[label] = alleles
    return out


@lru_cache(maxsize=512)
def _pmid_peptide_attributions(pmid_int: int) -> dict[str, frozenset[str]]:
    """Map ``peptide → frozenset(sample_label)`` for a PMID's per-peptide
    attribution CSV (#45).

    Some studies (Sarkizova 2020, etc.) deposit per-peptide → patient
    sample mappings in their supplementary tables, but IEDB ingests the
    rows with a class-only ``mhc_restriction`` and the union of donor
    HLAs in ``Host | MHC Types Present``.  Curators register the
    paper's per-peptide attribution via a CSV referenced from the
    PMID's ``peptide_attributions:`` key in pmid_overrides.yaml.

    The CSV must have columns ``peptide`` and ``sample_label`` (semicolon-
    joined when a peptide was observed in multiple samples).

    Returns an empty mapping when no attribution CSV is registered.
    """
    overrides = load_pmid_overrides()
    entry = overrides.get(pmid_int)
    if entry is None:
        return {}
    rel_path = entry.get("peptide_attributions")
    if not rel_path:
        return {}
    csv_path = _asset_path(rel_path)
    table = pd.read_csv(csv_path, usecols=["peptide", "sample_label"])
    out: dict[str, frozenset[str]] = {}
    for pep, labels in zip(table["peptide"].astype(str), table["sample_label"].astype(str)):
        sample_set = frozenset(s for s in labels.split(";") if s)
        if pep and sample_set:
            out[pep] = sample_set
    return out


def _coerce_pmid(pmid: int | str) -> int | None:
    """PMID as an int, or None if it is not one.

    PMIDs arrive as ints from YAML and as strings from dataframe columns.
    ``load_pmid_overrides`` is keyed by int, so a string lookup silently
    misses — and for these functions a miss is indistinguishable from
    "this study deposited no attributions", which is the one confusion
    their docstrings warn against.  Every public entry point coerces here.
    """
    with contextlib.suppress(ValueError, TypeError):
        return int(pmid)
    return None


@lru_cache(maxsize=512)
def _peptide_typings_by_pmid(
    pmid: int,
) -> Mapping[str, tuple[tuple[str, frozenset[str]], ...]]:
    """Build (and cache) the per-donor typing map for one study.

    Cached on the coerced int so ``31844290`` and ``"31844290"`` share one
    entry instead of building the 28k-peptide map twice.  Returned as a
    read-only view: the value is shared with every other caller and with
    the scanner, so a caller pruning it in place would silently change
    which observation rows the next build emits.
    """
    attributions = _pmid_peptide_attributions(pmid)
    if not attributions:
        return MappingProxyType({})
    sample_alleles = _pmid_sample_alleles(pmid)
    out: dict[str, tuple[tuple[str, frozenset[str]], ...]] = {}
    for pep, samples in attributions.items():
        per_sample = []
        for label in sorted(samples):
            alleles = sample_alleles.get(label, frozenset())
            if alleles:
                per_sample.append((label, alleles))
        if per_sample:
            out[pep] = tuple(per_sample)
    return MappingProxyType(out)


@lru_cache(maxsize=512)
def _peptide_alleles_by_pmid(pmid: int) -> Mapping[str, frozenset[str]]:
    """Merged view of :func:`_peptide_typings_by_pmid`, cached separately.

    Derived rather than rebuilt from the CSV, so the two public maps
    cannot disagree about which peptides survive the drop-empty filter.
    """
    return MappingProxyType(
        {
            peptide: frozenset().union(*(alleles for _, alleles in per_donor))
            for peptide, per_donor in _peptide_typings_by_pmid(pmid).items()
        }
    )


def sample_alleles_for_pmid(pmid: int | str) -> Mapping[str, frozenset[str]]:
    """Curated ``sample_label -> candidate alleles`` for one study's ms_samples.

    The sample-level counterpart to :func:`peptide_alleles_for_pmid`: what
    each curated sample of this study was typed to, before any per-peptide
    narrowing.  Public because it is the direct answer to "what did this
    study type its samples to?", and because every other way of asking had
    to reach through the per-peptide functions, which return nothing for
    the ~99% of studies with no ``peptide_attributions`` CSV.

    Values are :attr:`SampleMhcCandidates.join_alleles` — exact molecules
    plus, for a serotype-typed sample, that serotype's member alleles.
    Samples typed only to a locus or class (``BoLA-DR``, ``HLA class II``)
    name no allele and are **omitted**, as are samples with no ``mhc``
    field.  Use :func:`sample_mhc_candidates` on the raw field when you
    need to tell "typed imprecisely" from "not typed at all", or when you
    need the exact molecules without serotype expansion.

    Parameters
    ----------
    pmid
        PubMed ID of the study, as an int or a string of digits.

    Returns
    -------
    Mapping[str, frozenset[str]]
        Empty when the PMID has no override, no ``ms_samples``, or no
        sample that names an allele.

    Examples
    --------
    >>> sorted(sample_alleles_for_pmid(26768311))
    ['HeLa + vaccinia (VACV) infection', 'HeLa uninfected']

    See Also
    --------
    sample_mhc_candidates : the per-field parse, keeping precision apart.
    peptide_alleles_for_pmid : per-peptide narrowing, where a study
        deposited peptide-to-donor attributions.
    """
    pmid_int = _coerce_pmid(pmid)
    if pmid_int is None:
        return MappingProxyType({})
    return MappingProxyType(dict(_pmid_sample_alleles(pmid_int)))


def peptide_alleles_for_pmid(pmid: int | str) -> Mapping[str, frozenset[str]]:
    """Per-peptide candidate alleles for one study, from curated attributions.

    Some studies deposit which donor each peptide was observed in.  Where
    that exists, a peptide's presenting allele narrows from "any allele in
    this study" to "the alleles of the donors it was actually seen in" —
    the per-peptide attribution overrides of #45.  This folds the two-step
    lookup (``peptide → samples``, then ``sample → alleles``) into one
    cached map.

    Parameters
    ----------
    pmid
        PubMed ID of the study, as an int or a string of digits.

    Returns
    -------
    Mapping[str, frozenset[str]]
        ``peptide → candidate alleles``.  Peptides whose donors have no
        curated genotype are omitted rather than mapped to an empty set.

    Notes
    -----
    **Empty for most studies, and that is not an error.**  It requires a
    ``peptide_attributions`` CSV, which only a handful of PMIDs have — as
    of writing, PMID 31844290 is the only one, which
    ``test_only_one_pmid_has_peptide_attributions`` pins so this sentence
    fails loudly rather than going quietly stale.  Every other study
    returns an empty map, so a caller must treat "no entry" as "not
    narrowed", never as "no alleles".  This is worth stating because it is
    easy to reach for this function to explain an attribution result and
    conclude the wrong thing from an empty answer.

    Allele names are whatever the curated genotype uses, which is not
    always HLA: :func:`_parse_sample_mhc_field` deliberately handles
    non-human genotypes (``H2-K*b``, ``Patr-AL``), because restricting it
    to HLA-shaped names is the bug that stopped mouse studies from ever
    narrowing.  Do not assume an ``HLA-`` prefix.

    The value is a read-only view of a shared cache, so it cannot be
    mutated by accident; copy it if you need to modify it.

    Examples
    --------
    >>> alleles = peptide_alleles_for_pmid(31844290)
    >>> len(alleles)
    28031
    >>> sorted(alleles["AAAAAAAAAAAAAAPAP"])
    ['HLA-A*01:01', 'HLA-B*38:01', 'HLA-B*56:01', 'HLA-C*01:02', 'HLA-C*06:02']

    Those five are one donor's whole class-I genotype, not one presenting
    allele: the value is the union over every donor the peptide was seen
    in, so it narrows the candidates rather than identifying the restriction.
    Use :func:`peptide_typings_for_pmid` to see which donor supplied what.

    See Also
    --------
    peptide_typings_for_pmid : the same evidence kept per-donor rather
        than merged, for callers that need to know which donor
        contributed which alleles.
    attribute_peptide_to_sample_alleles : the same answer for a single
        peptide, without materializing the map.
    """
    pmid_int = _coerce_pmid(pmid)
    if pmid_int is None:
        return MappingProxyType({})
    return _peptide_alleles_by_pmid(pmid_int)


def peptide_typings_for_pmid(
    pmid: int | str,
) -> Mapping[str, tuple[tuple[str, frozenset[str]], ...]]:
    """Per-peptide donor typings for one study, kept per-donor.

    The same curated evidence as :func:`peptide_alleles_for_pmid`, but
    preserving which donor contributed which alleles instead of merging
    them into one union.  The scanner uses this to emit one observation
    row per matched donor (#236), so each row carries that donor's
    ``sample_label`` and its own typing.

    Parameters
    ----------
    pmid
        PubMed ID of the study, as an int or a string of digits.

    Returns
    -------
    Mapping[str, tuple[tuple[str, frozenset[str]], ...]]
        ``peptide → ((sample_label, alleles), ...)``, sorted by
        ``sample_label`` so emission order is deterministic.

    Notes
    -----
    Carries the same caveats as :func:`peptide_alleles_for_pmid`: it needs
    a ``peptide_attributions`` CSV, which only a handful of studies have,
    so an empty map means "not narrowed", not "no donors"; allele names
    are not necessarily HLA; and the result is a read-only shared view.

    Samples whose curated typing is empty are dropped — emitting a row
    with no allele set would just become an ``unmatched`` row downstream.

    Examples
    --------
    >>> typings = peptide_typings_for_pmid(31844290)
    >>> [(label, len(alleles)) for label, alleles in typings["AAAAAAAAAAAAAAPAP"]]
    [('MEL2 (13240-005)', 5)]

    See Also
    --------
    peptide_alleles_for_pmid : the same evidence merged into one candidate
        set per peptide.
    attribute_peptide_to_per_sample_typings : the same answer for a single
        peptide, without materializing the map.
    """
    pmid_int = _coerce_pmid(pmid)
    if pmid_int is None:
        return MappingProxyType({})
    return _peptide_typings_by_pmid(pmid_int)


def _clear_peptide_attribution_caches() -> None:
    """Drop both cached maps.

    The merged map is derived from the per-donor one, so clearing either
    alone leaves a stale layer behind.  Tests that monkeypatch
    ``_pmid_peptide_attributions`` or ``_pmid_sample_alleles`` need the
    whole chain gone, and previously had to know to clear two caches --
    which is the kind of thing a test forgets exactly once.
    """
    _peptide_typings_by_pmid.cache_clear()
    _peptide_alleles_by_pmid.cache_clear()


# `cache_clear` was reachable on these names while they were themselves
# `lru_cache`d.  Keep it working now that the cache sits one layer down, so
# existing callers do not silently lose the ability to invalidate.
peptide_alleles_for_pmid.cache_clear = _clear_peptide_attribution_caches
peptide_typings_for_pmid.cache_clear = _clear_peptide_attribution_caches


def attribute_peptide_to_per_sample_typings(
    pmid: int | str, peptide: str
) -> tuple[tuple[str, frozenset[str]], ...]:
    """Per-donor view of :func:`attribute_peptide_to_sample_alleles`.

    Returns a tuple of ``(sample_label, frozenset(alleles))`` pairs for
    each matched donor that observed the peptide, instead of merging
    them into one allele union.  Empty tuple when no attribution applies.

    The scanner uses this to emit one observation row per matched donor
    (issue #236), so polyspecific cohort rows decompose cleanly into
    per-sample rows with that donor's specific typing — instead of one
    row carrying the union of (e.g.) 15 alleles across 3 donors.
    """
    if not peptide:
        return ()
    return peptide_typings_for_pmid(pmid).get(peptide, ())


def attribute_peptide_to_sample_alleles(pmid: int | str, peptide: str) -> frozenset[str]:
    """Return the union of typed alleles across the samples a peptide was
    observed in within this PMID's curated cohort (#45), or an empty
    frozenset when no attribution is registered or the peptide is not
    in the map.

    Used at scan time to narrow ``host_mhc_types`` for class-only rows
    from the disease-wide union (e.g. all 14 alleles across GBM7+9+11)
    down to the matched donor's specific 6-allele genotype — turning
    14-18-allele candidate sets into 6-12-allele sets for
    set-membership / mhc_allele_in_set queries.

    Implemented as a single dict lookup against
    :func:`peptide_alleles_for_pmid` (the peptide-to-alleles map is
    pre-merged once per PMID).
    """
    if not peptide:
        return frozenset()
    return peptide_alleles_for_pmid(pmid).get(peptide, frozenset())


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
    set expansion since restrictions like ``"HLA class I"`` could legitimately
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
def expand_allele_set(
    mhc_restriction: str,
    host_mhc_types: str = "",
    pmid: int | str = "",
    mhc_class: str = "",
    attributed_alleles: frozenset[str] = frozenset(),
) -> tuple[str, str, int]:
    """Expand a (possibly coarse) MHC restriction to a candidate exact-allele
    set with provenance.

    Issue #137: downstream training pipelines need to know which exact
    4-digit alleles a row's restriction could plausibly map to, plus how
    that mapping was obtained, so they can do MIL / noisy-OR training over
    allele sets instead of silently collapsing or dropping coarse
    restrictions.

    Logic:

    - ``four_digit`` rows are returned as-is with provenance ``exact``.
    - ``class_only`` rows (e.g. ``"HLA class I"``) are expanded against,
      in priority order:

      1. ``attributed_alleles`` — a per-peptide attribution from the
         paper supplement (#45), passed in by the caller after looking
         up the peptide in :func:`attribute_peptide_to_sample_alleles`.
         Provenance: ``peptide_attribution``.
      2. The row's ``Host | MHC Types Present`` (the donor's typed
         alleles, when IEDB carries them). Provenance: ``sample_allele_match``.
      3. The per-PMID ``hla_alleles`` block. Provenance: ``pmid_class_pool``.

      In all cases the candidate set is filtered to the row's MHC class.
    - All other resolutions (``two_digit``, ``serological``,
      ``unresolved``) are returned as ``unmatched``.  Two-digit and
      serotype expansion against an external IPD-IMGT/HLA catalog is a
      planned follow-up.

    Note: ``attributed_alleles`` is taken as a hashable ``frozenset`` so the
    lru_cache key stays cheap. Pass ``frozenset()`` (the default) when no
    per-peptide attribution applies; otherwise the caller looks the
    peptide up against :func:`attribute_peptide_to_sample_alleles` once
    per row.  The ``frozenset`` shape gives identical sets from different
    peptides the same cache key, so peptides that map to the same
    sample union still share a cache slot.

    Returns
    -------
    tuple[str, str, int]
        ``(mhc_allele_set, mhc_allele_provenance, mhc_allele_set_size)``
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

    candidates = attributed_alleles or sample_alleles or pool
    if not candidates:
        return "", "unmatched", 0

    candidates = _filter_alleles_by_class(candidates, mhc_class)
    if not candidates:
        return "", "unmatched", 0

    if attributed_alleles:
        provenance = "peptide_attribution"
    elif sample_alleles:
        provenance = "sample_allele_match"
    else:
        provenance = "pmid_class_pool"
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
    mhc_species_context: str = "",
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
    mhc_species_context
        Optional curated study species used to resolve ambiguous MHC
        nomenclature. Explicit cross-species designations still win.
    assay_comments
        IEDB "Assay Comments" field. Some studies (e.g. PMID 29789417,
        Löffler 2018 CRC) tag per-row arm provenance only in the
        free-text Assay Comments column, so PMID rules can match on it
        via substring search.

    Returns
    -------
    dict[str, bool | str]
        Source flags plus ``cell_line_name`` (the line, with any
        ``<line>-<cell_type>`` hybrid suffix stripped) and ``cell_type``
        (the tissue / cell-type part, ``""`` when unknown) — see #261.
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
    is_reproductive_female = source_tissue_lower in categories["reproductive_female"]
    is_reproductive_male = source_tissue_lower in categories["reproductive_male"]
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
    elif effective_override == "noncancer_cell_line":
        # Non-malignant immortalized line (hTERT / SV40-LT / engineered from
        # normal cells). A cell line, but not a tumor — so src_cell_line stays
        # True while src_cancer is forced False. Force is_cell_line even when
        # IEDB's culture_condition is missing/wrong (same as ebv_lcl).
        is_cancer = False
        is_adjacent = False
        is_activated_apc = False
        is_cell_line = True
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

    # A mono-allelic peptidome produced on a known EBV-LCL transfection host
    # (721.221, C1R) is EBV-immortalized B-cell material, not a tumor — so it
    # classifies as ebv_lcl and never src_cancer, regardless of how IEDB tagged
    # the culture condition (which is inconsistent for these hosts).  This is
    # the single source of truth: it auto-corrects ~313K observations across
    # all 721.221/C1R studies (Sarkizova 2020, Abelin 2017, ...) that were
    # previously mislabeled src_cancer because IEDB tagged them plain
    # "Cell Line / Clone".  Genuinely malignant hosts (K562 = CML) are not
    # flagged ebv_lcl in monoallelic_lines.yaml and keep src_cancer.
    if is_monoallelic and mono_host in _ebv_lcl_mono_hosts():
        is_ebv_lcl = True
        is_cancer = False

    # Decompose IEDB's catch-all ``Cell Name`` into a clean line name plus a
    # separate ``cell_type`` (#261 stage 2).  ``parse_cell_name`` strips the
    # ``<line>-<cell_type>`` hybrid suffix ("K562-Myeloid cell" → line
    # "K562", type "Myeloid cell") and canonicalizes known synonyms
    # ("HeLa cells" → "HeLa").  We only overwrite ``cl_name`` when this row
    # already had a line name, and fall back to the parser's raw line input
    # (then the original string) so a cell-line row whose line isn't in the
    # registry never loses its identifier.
    cell_info = parse_cell_name(
        cell_name_str,
        monoallelic_host=mono_host,
        src_cell_line=(is_cell_line or is_ebv_lcl),
    )
    cell_type = cell_info.cell_type
    if cl_name:
        cl_name = cell_info.cell_line_name or cell_info.cell_line_input or cl_name

    return {
        "src_cancer": is_cancer,
        "src_adjacent_to_tumor": is_adjacent,
        "src_activated_apc": is_activated_apc,
        "src_healthy_tissue": (
            is_healthy_donor
            and not is_reproductive
            and not is_reproductive_female
            and not is_reproductive_male
            and not is_thymus
        ),
        "src_healthy_thymus": is_healthy_donor and is_thymus,
        "src_healthy_reproductive": is_healthy_donor and is_reproductive,
        "src_healthy_reproductive_female": is_healthy_donor and is_reproductive_female,
        "src_healthy_reproductive_male": is_healthy_donor and is_reproductive_male,
        "src_cell_line": is_cell_line,
        "src_ebv_lcl": is_ebv_lcl,
        "src_ex_vivo": is_ex_vivo,
        "cell_line_name": cl_name,
        "cell_type": cell_type,
        "is_monoallelic": is_monoallelic,
        "monoallelic_host": mono_host,
        "allele_resolution": classify_allele_resolution(mhc_restriction),
        "serotype": allele_to_serotype(mhc_restriction),
        "serotypes": ";".join(allele_to_all_serotypes(mhc_restriction)),
        "mhc_species": classify_mhc_species(mhc_restriction, mhc_species_context),
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
