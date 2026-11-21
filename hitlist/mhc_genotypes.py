"""Curated reference MHC genotype registry and condition-level resolver (#943, #944, #947).

Resolves mono- and multi-allele condition-level ``mhc_allele_set`` bags from:
1. Reported exact single-allele restriction (respecting experimental vs predicted evidence),
2. ``host_mhc_types`` / ``Host | MHC Present``,
3. Curated cell-line genotypes (:file:`hitlist/data/reference_mhc_genotypes.yaml`),
4. Curated inbred and HLA-transgenic animal strain haplotypes.

Experimental single-allele restriction is always kept separate from predicted
restriction (e.g. ``MHC binding prediction``) and from condition-level multi-allele
host genotypes.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import mhcgnomes
import pandas as pd
from mhcgnomes import Allele, MhcClass, Pair

from .cell_name_parser import parse_cell_name
from .curation_yaml import load_curation_yaml

REGISTRY_PATH = Path(__file__).resolve().parent / "data" / "reference_mhc_genotypes.yaml"

EXPERIMENTAL_RESTRICTION_EVIDENCE_CODES = frozenset(
    {
        "Single allele present",
        "T cell assay -Mismatched MHC molecules",
        "T cell assay -Single MHC type present",
        "MHC binding assay",
        "X-ray crystallography",
        "3D structure",
        "T cell assay -MHC mutant",
        "T cell assay -Blocking antibody",
    }
)

PREDICTED_RESTRICTION_EVIDENCE_CODES = frozenset(
    {
        "MHC binding prediction",
        "MHC binding prediction -NetMHC",
        "MHC binding prediction -NetMHCpan",
        "MHC binding prediction -IEDB",
    }
)

_HOST_SPECIES_PREFIXES: tuple[tuple[str, str], ...] = (
    ("homo sapiens", "Homo sapiens"),
    ("human", "Homo sapiens"),
    ("mus musculus", "Mus musculus"),
    ("mouse", "Mus musculus"),
    ("rattus norvegicus", "Rattus norvegicus"),
    ("rattus", "Rattus sp."),
    ("rat", "Rattus norvegicus"),
    ("macaca mulatta", "Macaca mulatta"),
    ("rhesus", "Macaca mulatta"),
    ("macaca fascicularis", "Macaca fascicularis"),
    ("pan troglodytes", "Pan troglodytes"),
    ("chimpanzee", "Pan troglodytes"),
    ("bos taurus", "Bos taurus"),
    ("bos ", "Bos sp."),
    ("bovine", "Bos taurus"),
    ("cattle", "Bos taurus"),
    ("sus scrofa", "Sus scrofa"),
    ("sus ", "Sus sp."),
    ("pig", "Sus scrofa"),
    ("canis lupus", "Canis lupus familiaris"),
    ("canis ", "Canis sp."),
    ("dog", "Canis lupus familiaris"),
    ("equus caballus", "Equus caballus"),
    ("horse", "Equus caballus"),
    ("gallus gallus", "Gallus gallus"),
    ("chicken", "Gallus gallus"),
    ("sarcophilus harrisii", "Sarcophilus harrisii"),
    ("tasmanian devil", "Sarcophilus harrisii"),
    ("oryctolagus cuniculus", "Oryctolagus cuniculus"),
    ("rabbit", "Oryctolagus cuniculus"),
    ("cavia porcellus", "Cavia porcellus"),
    ("guinea pig", "Cavia porcellus"),
)

_CLASS_SPLIT_RE = re.compile(r"[;,|\n]+")
_WHITESPACE_RE = re.compile(r"\s+")
_HLA_TG_RE = re.compile(r"(HLA-[A-Z0-9]+\*\d{2,3}:\d{2,3}|HLA-[A-Z0-9]+\d+)", re.IGNORECASE)


@dataclass(frozen=True)
class NormalizedMhcToken:
    """Canonical ``mhcgnomes`` normalization of a single MHC token."""

    raw: str
    canonical: str | None
    is_exact_molecule: bool
    mhc_species: str | None
    mhc_class: str | None


@dataclass(frozen=True)
class GenotypeMatch:
    """Curated genotype match for a canonical cell line or animal strain."""

    name: str
    kind: str  # "cell_line" | "inbred_strain" | "transgenic_strain"
    alleles: tuple[str, ...]
    mhc_species: str
    host_species: str
    evidence_reference: str
    mhc_null_class_i: bool = False
    low_endogenous_class_i: bool = False
    endogenous_mhc_ko: bool = False


@dataclass(frozen=True)
class ResolvedMhcGenotype:
    """Unified condition-level MHC genotype and restriction provenance."""

    mhc_allele_set: str
    mhc_allele_set_size: int
    mhc_allele_cardinality: str  # "monoallelic" | "multi_allelic_bag" | "class_only_unknown"
    mhc_allele_set_provenance: str
    mhc_species: str
    host_species: str
    is_chimeric_mhc_host: bool
    experimental_mhc_allele: str | None
    is_predicted_restriction: bool


def _normalize_class_hint(mhc_class: str | None) -> str | None:
    if not mhc_class:
        return None
    text = str(mhc_class).strip().upper()
    if text in {"I", "CLASS I", "MHC I", "MHC CLASS I", "HLA CLASS I", "1"}:
        return "I"
    if text in {"II", "CLASS II", "MHC II", "MHC CLASS II", "HLA CLASS II", "2"}:
        return "II"
    if text in {"I+II", "I/II", "BOTH"}:
        return "I+II"
    if "NON-CLASS" in text or "NONCLASS" in text:
        return "non-classical"
    return None


def _mhc_class_from_parsed(parsed) -> str | None:
    mhc_class = getattr(parsed, "mhc_class", None)
    if not mhc_class:
        return None
    text = str(mhc_class).strip()
    if text in {"I", "Ia"}:
        return "I"
    if text in {"II", "IIa"}:
        return "II"
    if text in {"Ib", "Ic", "Id", "non-classical"}:
        return "non-classical"
    return text


def _allele_is_exact_molecule(allele: Allele) -> bool:
    """Return True when ``allele`` names a single sequence-resolved MHC molecule.

    Uses ``mhcgnomes``' species haplotype roster and ``get_known_allele`` table
    (matching :func:`hitlist.curation._names_one_molecule`), so typos such as
    ``H2-K*nonsense`` are rejected while canonical two-field alleles
    (``HLA-A*02:01``, ``Mamu-A1*001:01``, ``DLA-88*501:01``, ``BoLA-2*026:01``)
    and single-field haplotype-designated rodent alleles (``H2-Kb``, ``H2-Db``,
    ``H2-IAb``, ``RT1-Al``) resolve as exact molecules.
    """
    fields = getattr(allele, "allele_fields", ()) or ()
    if len(fields) != 1:
        return len(fields) >= 2
    field = str(fields[0])
    species = getattr(allele, "species", None)
    if species is None:
        return False
    if field in getattr(species, "haplotypes", ()):
        return True
    gene = getattr(allele, "gene", None)
    gene_name = getattr(gene, "name", "") if gene is not None else ""
    return bool(
        not field.isdigit() and gene_name and species.get_known_allele(gene_name, field) is not None
    )


def _canonical_allele_string(parsed) -> str:
    """Return canonical ``mhcgnomes`` representation (preferring ``H2-Kb`` over ``H2-K*b``)."""
    if isinstance(parsed, Allele):
        species_prefix = getattr(getattr(parsed, "species", None), "mhc_prefix", "") or ""
        fields = getattr(parsed, "allele_fields", ()) or ()
        if (
            species_prefix in {"H2", "RT1"}
            and len(fields) == 1
            and any(c.isalpha() for c in fields[0])
        ):
            return f"{species_prefix}-{parsed.gene_name}{fields[0]}"
        return parsed.to_string()
    if isinstance(parsed, Pair):
        species_prefix = getattr(getattr(parsed, "species", None), "mhc_prefix", "") or ""
        if species_prefix == "H2":
            alpha_fields = getattr(parsed.alpha, "allele_fields", ()) or ()
            beta_fields = getattr(parsed.beta, "allele_fields", ()) or ()
            if len(alpha_fields) == 1 and len(beta_fields) == 1:
                if (
                    parsed.alpha.gene_name == "AA"
                    and parsed.beta.gene_name == "AB"
                    and alpha_fields[0] == "d"
                    and beta_fields[0] == "g7"
                ):
                    return "H2-IAg7"
                if alpha_fields[0] == beta_fields[0]:
                    if parsed.alpha.gene_name == "AA" and parsed.beta.gene_name == "AB":
                        return f"H2-IA{alpha_fields[0]}"
                    if parsed.alpha.gene_name == "EA" and parsed.beta.gene_name == "EB":
                        return f"H2-IE{alpha_fields[0]}"
        return parsed.to_string()
    return parsed.to_string()


@cache
def normalize_mhc_token(raw: str) -> NormalizedMhcToken:
    """Parse and canonicalize a single MHC string via ``mhcgnomes``."""
    text = (raw or "").strip()
    if not text:
        return NormalizedMhcToken(
            raw=text, canonical=None, is_exact_molecule=False, mhc_species=None, mhc_class=None
        )
    parsed = mhcgnomes.parse(text, raise_on_error=False)
    if parsed is None:
        return NormalizedMhcToken(
            raw=text, canonical=None, is_exact_molecule=False, mhc_species=None, mhc_class=None
        )
    species_obj = getattr(parsed, "species", None)
    species_name = getattr(species_obj, "name", None) if species_obj is not None else None
    mhc_class = _mhc_class_from_parsed(parsed)
    if isinstance(parsed, Allele):
        exact = _allele_is_exact_molecule(parsed)
        return NormalizedMhcToken(
            raw=text,
            canonical=_canonical_allele_string(parsed) if exact else parsed.to_string(),
            is_exact_molecule=exact,
            mhc_species=species_name,
            mhc_class=mhc_class,
        )
    if isinstance(parsed, Pair):
        alpha_ok = isinstance(parsed.alpha, Allele) and _allele_is_exact_molecule(parsed.alpha)
        beta_ok = isinstance(parsed.beta, Allele) and _allele_is_exact_molecule(parsed.beta)
        exact = alpha_ok and beta_ok
        return NormalizedMhcToken(
            raw=text,
            canonical=_canonical_allele_string(parsed) if exact else parsed.to_string(),
            is_exact_molecule=exact,
            mhc_species=species_name,
            mhc_class=mhc_class,
        )
    if isinstance(parsed, MhcClass):
        return NormalizedMhcToken(
            raw=text,
            canonical=None,
            is_exact_molecule=False,
            mhc_species=species_name,
            mhc_class=mhc_class,
        )
    return NormalizedMhcToken(
        raw=text,
        canonical=None,
        is_exact_molecule=False,
        mhc_species=species_name,
        mhc_class=mhc_class,
    )


def infer_host_species(host_name: str | None) -> str:
    """Normalize a free-text IEDB/CEDAR host string into a canonical species name."""
    if not host_name:
        return ""
    lowered = str(host_name).strip().casefold()
    if not lowered:
        return ""
    for prefix, canonical in _HOST_SPECIES_PREFIXES:
        if prefix in lowered:
            return canonical
    return ""


def parse_mhc_allele_list(
    raw: str | None,
    *,
    mhc_class_hint: str | None = None,
) -> tuple[str, ...]:
    """Parse a delimited MHC list (e.g. ``host_mhc_types`` or ``Host | MHC Present``).

    Returns sorted, deduplicated canonical exact alleles, optionally filtered to
    ``mhc_class_hint`` (``"I"`` or ``"II"``).
    """
    if not raw:
        return ()
    text = str(raw).strip()
    if not text:
        return ()
    hint = _normalize_class_hint(mhc_class_hint)
    # Split on delimiters first; if a chunk has multiple space-separated alleles
    # like "HLA-A*02:01 HLA-B*07:02", split those too while keeping pairs intact.
    raw_tokens: list[str] = []
    for chunk in _CLASS_SPLIT_RE.split(text):
        chunk = chunk.strip()
        if not chunk:
            continue
        if normalize_mhc_token(chunk).is_exact_molecule:
            raw_tokens.append(chunk)
            continue
        for sub in _WHITESPACE_RE.split(chunk):
            sub = sub.strip()
            if sub:
                raw_tokens.append(sub)

    alleles: set[str] = set()
    for token in raw_tokens:
        norm = normalize_mhc_token(token)
        if not norm.is_exact_molecule or not norm.canonical:
            continue
        if hint in {"I", "II"} and norm.mhc_class in {"I", "II"} and norm.mhc_class != hint:
            continue
        alleles.add(norm.canonical)
    return tuple(sorted(alleles))


@cache
def load_reference_mhc_genotypes(registry_path: Path | None = None) -> dict:
    """Load and validate :file:`hitlist/data/reference_mhc_genotypes.yaml`."""
    data = load_curation_yaml(registry_path or REGISTRY_PATH)
    if not isinstance(data, dict) or "cell_lines" not in data or "strains" not in data:
        raise ValueError("reference_mhc_genotypes.yaml must define 'cell_lines' and 'strains'")
    return data


@cache
def _cell_line_index(registry_path: Path | None = None) -> dict[str, tuple[str, dict]]:
    data = load_reference_mhc_genotypes(registry_path)
    index: dict[str, tuple[str, dict]] = {}
    for canonical, entry in data["cell_lines"].items():
        keys = [canonical, *(entry.get("synonyms") or ())]
        for alias in keys:
            index[str(alias).strip().casefold()] = (canonical, entry)
    return index


@cache
def _strain_index(registry_path: Path | None = None) -> tuple[tuple[str, str, dict], ...]:
    data = load_reference_mhc_genotypes(registry_path)
    pairs: list[tuple[str, str, dict]] = []
    for canonical, entry in data["strains"].items():
        for alias in (canonical, *(entry.get("synonyms") or ())):
            key = str(alias).strip().casefold()
            if key:
                pairs.append((key, canonical, entry))
    # Match longest/most-specific strain synonym first (e.g. HHD before C57BL/6).
    pairs.sort(key=lambda item: len(item[0]), reverse=True)
    return tuple(pairs)


def _select_alleles(entry: dict, mhc_class: str | None) -> tuple[str, ...]:
    hint = _normalize_class_hint(mhc_class)
    class_i = [normalize_mhc_token(a).canonical or a for a in (entry.get("class_i_alleles") or ())]
    class_ii = [
        normalize_mhc_token(a).canonical or a for a in (entry.get("class_ii_alleles") or ())
    ]
    if hint == "I":
        selected = class_i
    elif hint == "II":
        selected = class_ii
    else:
        selected = [*class_i, *class_ii]
    return tuple(sorted(dict.fromkeys(selected)))


def lookup_cell_line_genotype(
    cell_name: str | None,
    *,
    mhc_class: str | None = None,
    allow_low_endogenous: bool = False,
    registry_path: Path | None = None,
) -> GenotypeMatch | None:
    """Look up a curated cell-line MHC genotype by raw or canonical cell name.

    Returns ``None`` for Class-I null host lines (``K562``, ``721.221``) when
    Class I is requested, and skips ``low_endogenous_class_i`` hosts (``C1R``)
    unless ``allow_low_endogenous=True``, so engineered mono-allelic hosts are
    never mis-attributed to endogenous trace alleles.
    """
    if not cell_name:
        return None
    index = _cell_line_index(registry_path)
    raw = str(cell_name).strip()
    candidates = [raw]
    parsed = parse_cell_name(raw)
    if parsed.cell_line_name:
        candidates.insert(0, parsed.cell_line_name)
    # Strip trailing " cells" / " cell" and "-<lineage>" suffixes for lines not in cell_lines.yaml
    cleaned = re.sub(r"\s*-\s*[A-Za-z ]+$", "", raw).strip()
    cleaned = re.sub(r"\s+cells?$", "", cleaned, flags=re.IGNORECASE).strip()
    if cleaned and cleaned not in candidates:
        candidates.append(cleaned)

    hit: tuple[str, dict] | None = None
    for candidate in candidates:
        found = index.get(candidate.casefold())
        if found is not None:
            hit = found
            break
    if hit is None:
        return None

    canonical, entry = hit
    hint = _normalize_class_hint(mhc_class)
    if (
        entry.get("mhc_null_class_i")
        and hint in {None, "I"}
        and (hint == "I" or not entry.get("class_ii_alleles"))
    ):
        # K562 / 721.221: never impute endogenous Class I; allow Class II if explicitly requested.
        return None
    if entry.get("low_endogenous_class_i") and hint in {None, "I"} and not allow_low_endogenous:
        return None

    alleles = _select_alleles(entry, mhc_class)
    if not alleles:
        return None
    species = str(entry.get("species", ""))
    return GenotypeMatch(
        name=canonical,
        kind="cell_line",
        alleles=alleles,
        mhc_species=species,
        host_species=species,
        evidence_reference=str(entry.get("evidence_reference", "")),
        mhc_null_class_i=bool(entry.get("mhc_null_class_i", False)),
        low_endogenous_class_i=bool(entry.get("low_endogenous_class_i", False)),
    )


def lookup_strain_genotype(
    host_name: str | None,
    *,
    mhc_class: str | None = None,
    registry_path: Path | None = None,
) -> GenotypeMatch | None:
    """Look up a curated inbred or HLA-transgenic animal strain genotype."""
    if not host_name:
        return None
    text = str(host_name).strip()
    if not text:
        return None
    lowered = text.casefold()
    hint = _normalize_class_hint(mhc_class)

    for alias, canonical, entry in _strain_index(registry_path):
        # Require token boundaries so short aliases like "b6" or "nod" don't match
        # unrelated substrings inside longer words.
        pattern = r"(?<![a-z0-9])" + re.escape(alias) + r"(?![a-z0-9])"
        if re.search(pattern, lowered):
            alleles = _select_alleles(entry, hint)
            if not alleles:
                return None
            tg = tuple(entry.get("transgenic_alleles") or ())
            host_species = str(entry.get("host_species", ""))
            # Determine MHC groove species from selected alleles (e.g. HLA-A*02:01 -> Homo sapiens).
            allele_species = {
                normalize_mhc_token(a).mhc_species
                for a in alleles
                if normalize_mhc_token(a).mhc_species
            }
            if len(allele_species) == 1:
                mhc_species = next(iter(allele_species))
            elif tg and "Homo sapiens" in allele_species and hint in {None, "I", "II"}:
                mhc_species = "Homo sapiens" if entry.get("endogenous_mhc_ko") else host_species
            else:
                mhc_species = host_species
            return GenotypeMatch(
                name=canonical,
                kind="transgenic_strain" if tg else "inbred_strain",
                alleles=alleles,
                mhc_species=mhc_species,
                host_species=host_species,
                evidence_reference=str(entry.get("evidence_reference", "")),
                endogenous_mhc_ko=bool(entry.get("endogenous_mhc_ko", False)),
            )

    # Dynamic fallback for HLA-transgenic mouse descriptions naming an explicit
    # 4-digit HLA allele + Tg/transgenic (e.g. "Mus musculus HLA-A*11:01 Tg").
    if ("tg" in lowered or "transgenic" in lowered) and ("mus" in lowered or "mouse" in lowered):
        match = _HLA_TG_RE.search(text)
        if match:
            norm = normalize_mhc_token(match.group(1))
            if (
                norm.is_exact_molecule
                and norm.canonical
                and (hint in {None, "I+II"} or norm.mhc_class == hint)
            ):
                return GenotypeMatch(
                    name=f"{norm.canonical}_Tg",
                    kind="transgenic_strain",
                    alleles=(norm.canonical,),
                    mhc_species=norm.mhc_species or "Homo sapiens",
                    host_species="Mus musculus",
                    evidence_reference="Parsed HLA-transgenic mouse construct from host_name.",
                    endogenous_mhc_ko=False,
                )
    return None


def resolve_condition_mhc_genotype(
    *,
    reported_mhc_name: str | None = None,
    restriction_evidence_code: str | None = None,
    reported_mhc_class: str | None = None,
    host_mhc_types: str | None = None,
    cell_name: str | None = None,
    apc_name: str | None = None,
    host_name: str | None = None,
    registry_path: Path | None = None,
) -> ResolvedMhcGenotype:
    """Resolve unified condition-level ``mhc_allele_set`` and provenance.

    Priority order:
    1. Exact single-allele ``reported_mhc_name`` (when NOT prediction-only).
    2. Explicit multi-allele list in ``reported_mhc_name`` (e.g. semicolon-joined).
    3. ``host_mhc_types`` / ``Host | MHC Present`` (filtered by ``reported_mhc_class``).
    4. Curated cell-line genotype from ``apc_name`` or ``cell_name``.
    5. Curated inbred or HLA-transgenic strain genotype from ``host_name``.
    6. Fallback: ``class_only_unknown``.

    ``experimental_mhc_allele`` is ONLY populated when ``reported_mhc_name`` is
    a single exact molecule AND ``restriction_evidence_code`` is in
    :data:`EXPERIMENTAL_RESTRICTION_EVIDENCE_CODES`. Prediction-derived
    restrictions (``MHC binding prediction``) set ``is_predicted_restriction=True``
    and NEVER populate ``experimental_mhc_allele`` or override donor/host typing.
    """
    evidence = (restriction_evidence_code or "").strip()
    is_predicted = evidence in PREDICTED_RESTRICTION_EVIDENCE_CODES or (
        "prediction" in evidence.casefold()
    )
    class_hint = _normalize_class_hint(reported_mhc_class)
    host_species = infer_host_species(host_name)

    reported = (reported_mhc_name or "").strip()
    reported_norm = normalize_mhc_token(reported) if reported else None
    if class_hint is None and reported_norm is not None:
        class_hint = _normalize_class_hint(reported_norm.mhc_class)

    experimental_allele: str | None = None
    if (
        reported_norm is not None
        and reported_norm.is_exact_molecule
        and reported_norm.canonical
        and not is_predicted
        and evidence in EXPERIMENTAL_RESTRICTION_EVIDENCE_CODES
    ):
        experimental_allele = reported_norm.canonical

    # 1. Exact single-allele reported restriction (non-predicted).
    if (
        reported_norm is not None
        and reported_norm.is_exact_molecule
        and reported_norm.canonical
        and not is_predicted
    ):
        mhc_species = reported_norm.mhc_species or host_species
        if not host_species:
            host_species = mhc_species
        return ResolvedMhcGenotype(
            mhc_allele_set=reported_norm.canonical,
            mhc_allele_set_size=1,
            mhc_allele_cardinality="monoallelic",
            mhc_allele_set_provenance="reported_single_allele",
            mhc_species=mhc_species or "",
            host_species=host_species or "",
            is_chimeric_mhc_host=bool(mhc_species and host_species and mhc_species != host_species),
            experimental_mhc_allele=experimental_allele,
            is_predicted_restriction=is_predicted,
        )

    # 2. Multi-allele reported string (e.g. semicolon-joined donor set in MS exports).
    if reported and (";" in reported or "," in reported) and not is_predicted:
        parsed_reported = parse_mhc_allele_list(reported, mhc_class_hint=class_hint)
        if parsed_reported:
            species_set = {
                normalize_mhc_token(a).mhc_species
                for a in parsed_reported
                if normalize_mhc_token(a).mhc_species
            }
            mhc_species = next(iter(species_set)) if len(species_set) == 1 else host_species
            if not host_species:
                host_species = mhc_species or ""
            return ResolvedMhcGenotype(
                mhc_allele_set=";".join(parsed_reported),
                mhc_allele_set_size=len(parsed_reported),
                mhc_allele_cardinality=(
                    "monoallelic" if len(parsed_reported) == 1 else "multi_allelic_bag"
                ),
                mhc_allele_set_provenance=(
                    "reported_single_allele" if len(parsed_reported) == 1 else "paper_donor_table"
                ),
                mhc_species=mhc_species or "",
                host_species=host_species or "",
                is_chimeric_mhc_host=bool(
                    mhc_species and host_species and mhc_species != host_species
                ),
                experimental_mhc_allele=None,
                is_predicted_restriction=is_predicted,
            )

    # 3. Host | MHC Present / host_mhc_types.
    host_alleles = parse_mhc_allele_list(host_mhc_types, mhc_class_hint=class_hint)
    if host_alleles:
        species_set = {
            normalize_mhc_token(a).mhc_species
            for a in host_alleles
            if normalize_mhc_token(a).mhc_species
        }
        mhc_species = next(iter(species_set)) if len(species_set) == 1 else host_species
        if not host_species:
            host_species = mhc_species or ""
        return ResolvedMhcGenotype(
            mhc_allele_set=";".join(host_alleles),
            mhc_allele_set_size=len(host_alleles),
            mhc_allele_cardinality=(
                "monoallelic" if len(host_alleles) == 1 else "multi_allelic_bag"
            ),
            mhc_allele_set_provenance="host_mhc_types",
            mhc_species=mhc_species or "",
            host_species=host_species or "",
            is_chimeric_mhc_host=bool(mhc_species and host_species and mhc_species != host_species),
            experimental_mhc_allele=experimental_allele,
            is_predicted_restriction=is_predicted,
        )

    # 4. Curated cell-line genotype (APC preferred over generic cell_name).
    for candidate_cell in (apc_name, cell_name):
        match = lookup_cell_line_genotype(
            candidate_cell, mhc_class=class_hint, registry_path=registry_path
        )
        if match is not None:
            resolved_host = host_species or match.host_species
            return ResolvedMhcGenotype(
                mhc_allele_set=";".join(match.alleles),
                mhc_allele_set_size=len(match.alleles),
                mhc_allele_cardinality=(
                    "monoallelic" if len(match.alleles) == 1 else "multi_allelic_bag"
                ),
                mhc_allele_set_provenance="curated_cell_line_hla",
                mhc_species=match.mhc_species,
                host_species=resolved_host,
                is_chimeric_mhc_host=bool(
                    match.mhc_species and resolved_host and match.mhc_species != resolved_host
                ),
                experimental_mhc_allele=experimental_allele,
                is_predicted_restriction=is_predicted,
            )

    # 5. Curated inbred or HLA-transgenic strain genotype from host_name.
    strain_match = lookup_strain_genotype(
        host_name, mhc_class=class_hint, registry_path=registry_path
    )
    if strain_match is not None:
        provenance = (
            "transgenic_mhc_construct"
            if strain_match.kind == "transgenic_strain"
            else "inbred_strain_haplotype"
        )
        return ResolvedMhcGenotype(
            mhc_allele_set=";".join(strain_match.alleles),
            mhc_allele_set_size=len(strain_match.alleles),
            mhc_allele_cardinality=(
                "monoallelic" if len(strain_match.alleles) == 1 else "multi_allelic_bag"
            ),
            mhc_allele_set_provenance=provenance,
            mhc_species=strain_match.mhc_species,
            host_species=strain_match.host_species,
            is_chimeric_mhc_host=bool(
                strain_match.mhc_species
                and strain_match.host_species
                and strain_match.mhc_species != strain_match.host_species
            ),
            experimental_mhc_allele=experimental_allele,
            is_predicted_restriction=is_predicted,
        )

    # 6. Fallback: unresolved / class_only_unknown.
    mhc_species = (reported_norm.mhc_species if reported_norm else None) or host_species or ""
    if not host_species:
        host_species = mhc_species
    return ResolvedMhcGenotype(
        mhc_allele_set="",
        mhc_allele_set_size=0,
        mhc_allele_cardinality="class_only_unknown",
        mhc_allele_set_provenance="unknown",
        mhc_species=mhc_species,
        host_species=host_species,
        is_chimeric_mhc_host=bool(mhc_species and host_species and mhc_species != host_species),
        experimental_mhc_allele=experimental_allele,
        is_predicted_restriction=is_predicted,
    )


_PRESERVED_PROVENANCE_MAP: dict[str, str] = {
    "exact": "reported_single_allele",
    "reported_single_allele": "reported_single_allele",
    "sample_allele_match": "host_mhc_types",
    "sample_locus_match": "host_mhc_types",
    "sample_serotype_match": "host_mhc_types",
    "host_mhc_types": "host_mhc_types",
    "peptide_attribution": "paper_donor_table",
    "peptide_locus_match": "paper_donor_table",
    "peptide_serotype_match": "paper_donor_table",
    "paper_donor_table": "paper_donor_table",
    "haplotype_roster": "inbred_strain_haplotype",
    "inbred_strain_haplotype": "inbred_strain_haplotype",
    "curated_cell_line_hla": "curated_cell_line_hla",
    "transgenic_mhc_construct": "transgenic_mhc_construct",
}


def _first_present_series(df: pd.DataFrame, candidates: tuple[str, ...]) -> pd.Series:
    for col in candidates:
        if col in df.columns:
            return df[col].astype("string").fillna("")
    return pd.Series("", index=df.index, dtype="string")


def recover_mhc_genotypes_for_frame(
    df: pd.DataFrame,
    *,
    overwrite_resolved: bool = False,
    registry_path: Path | None = None,
) -> pd.DataFrame:
    """Enrich an MS, perturbed MS, binding, or T-cell frame with condition-level MHC bags (#943).

    Deduplicates unique metadata tuples before resolving so multi-million-row
    DataFrames process quickly. Emits canonical columns:
    - ``mhc_allele_set`` (semicolon-joined canonical alleles)
    - ``mhc_allele_set_size`` (``int64``)
    - ``mhc_allele_cardinality`` (``monoallelic`` / ``multi_allelic_bag`` / ``class_only_unknown``)
    - ``mhc_allele_set_provenance`` (``reported_single_allele`` / ``host_mhc_types`` /
      ``curated_cell_line_hla`` / ``inbred_strain_haplotype`` / ``transgenic_mhc_construct`` /
      ``paper_donor_table`` / ``unknown``)
    - ``mhc_species`` (MHC groove species)
    - ``host_species`` (host/APC processing species)
    - ``is_chimeric_mhc_host`` (``bool``)

    Already-resolved primary rows (non-empty ``mhc_allele_set`` with non-pool provenance
    such as ``exact``, ``sample_allele_match``, ``peptide_attribution``) are preserved
    verbatim unless ``overwrite_resolved=True``.
    """
    out = df.copy()
    if out.empty:
        out["mhc_allele_set"] = pd.Series(dtype="string")
        out["mhc_allele_set_size"] = pd.Series(dtype="int64")
        out["mhc_allele_cardinality"] = pd.Series(dtype="string")
        out["mhc_allele_set_provenance"] = pd.Series(dtype="string")
        out["mhc_species"] = pd.Series(dtype="string")
        out["host_species"] = pd.Series(dtype="string")
        out["is_chimeric_mhc_host"] = pd.Series(dtype="bool")
        return out

    reported_s = _first_present_series(out, ("mhc_restriction", "reported_mhc_name"))
    ev_s = _first_present_series(out, ("restriction_evidence", "restriction_evidence_code"))
    cls_s = _first_present_series(out, ("mhc_class", "reported_mhc_class"))
    existing_set_s = _first_present_series(out, ("mhc_allele_set",))
    existing_prov_s = _first_present_series(
        out, ("mhc_allele_set_provenance", "mhc_allele_provenance")
    )
    host_mhc_s = _first_present_series(out, ("host_mhc_types", "Host | MHC Present"))
    cell_line_s = _first_present_series(out, ("cell_line_name",))
    cell_name_s = _first_present_series(out, ("cell_name",))
    apc_s = _first_present_series(out, ("apc_name",))
    host_s = _first_present_series(out, ("host", "host_name", "species"))
    existing_mhc_sp_s = _first_present_series(out, ("mhc_species",))

    keys = list(
        zip(
            reported_s.tolist(),
            ev_s.tolist(),
            cls_s.tolist(),
            existing_set_s.tolist(),
            existing_prov_s.tolist(),
            host_mhc_s.tolist(),
            cell_line_s.tolist(),
            cell_name_s.tolist(),
            apc_s.tolist(),
            host_s.tolist(),
            existing_mhc_sp_s.tolist(),
        )
    )
    unique_cache: dict[tuple, tuple[str, int, str, str, str, str, bool]] = {}
    for key in set(keys):
        (
            reported,
            ev,
            mhc_cls,
            existing_set,
            existing_prov,
            host_mhc,
            cell_line,
            cell_name,
            apc_name,
            host_raw,
            existing_mhc_sp,
        ) = key
        mapped_prov = _PRESERVED_PROVENANCE_MAP.get(existing_prov.strip(), "")
        if not overwrite_resolved and existing_set.strip() and mapped_prov:
            tokens = [t.strip() for t in existing_set.split(";") if t.strip()]
            norm_tokens = sorted({normalize_mhc_token(t).canonical or t for t in tokens if t})
            allele_set_str = ";".join(norm_tokens)
            size = len(norm_tokens)
            cardinality = (
                "monoallelic"
                if size == 1
                else "multi_allelic_bag"
                if size > 1
                else "class_only_unknown"
            )
            species_set = {
                normalize_mhc_token(t).mhc_species
                for t in norm_tokens
                if normalize_mhc_token(t).mhc_species
            }
            mhc_sp = (
                next(iter(species_set))
                if len(species_set) == 1
                else existing_mhc_sp.strip() or infer_host_species(host_raw)
            )
            host_sp = infer_host_species(host_raw) or mhc_sp
            unique_cache[key] = (
                allele_set_str,
                size,
                cardinality,
                mapped_prov,
                mhc_sp,
                host_sp,
                bool(mhc_sp and host_sp and mhc_sp != host_sp),
            )
            continue

        resolved = resolve_condition_mhc_genotype(
            reported_mhc_name=reported,
            restriction_evidence_code=ev,
            reported_mhc_class=mhc_cls,
            host_mhc_types=host_mhc,
            cell_name=cell_line or cell_name,
            apc_name=apc_name or cell_line,
            host_name=host_raw,
            registry_path=registry_path,
        )
        mhc_sp = resolved.mhc_species or existing_mhc_sp.strip()
        host_sp = resolved.host_species or infer_host_species(host_raw) or mhc_sp
        unique_cache[key] = (
            resolved.mhc_allele_set,
            resolved.mhc_allele_set_size,
            resolved.mhc_allele_cardinality,
            resolved.mhc_allele_set_provenance,
            mhc_sp,
            host_sp,
            bool(mhc_sp and host_sp and mhc_sp != host_sp),
        )

    rows = [unique_cache[k] for k in keys]
    out["mhc_allele_set"] = pd.Series([r[0] for r in rows], index=out.index, dtype="string")
    out["mhc_allele_set_size"] = pd.Series([r[1] for r in rows], index=out.index, dtype="int64")
    out["mhc_allele_cardinality"] = pd.Series([r[2] for r in rows], index=out.index, dtype="string")
    out["mhc_allele_set_provenance"] = pd.Series(
        [r[3] for r in rows], index=out.index, dtype="string"
    )
    out["mhc_species"] = pd.Series([r[4] for r in rows], index=out.index, dtype="string")
    out["host_species"] = pd.Series([r[5] for r in rows], index=out.index, dtype="string")
    out["is_chimeric_mhc_host"] = pd.Series([r[6] for r in rows], index=out.index, dtype="bool")
    return out


def summarize_mhc_genotype_recovery(df: pd.DataFrame) -> dict:
    """Summarize condition-level MHC genotype recovery coverage on an enriched frame."""
    if df.empty:
        return {
            "n_rows": 0,
            "n_resolved_rows": 0,
            "n_monoallelic_rows": 0,
            "n_multi_allelic_bag_rows": 0,
            "n_class_only_unknown_rows": 0,
            "n_chimeric_mhc_host_rows": 0,
            "by_provenance": {},
            "by_cardinality": {},
            "by_mhc_species": {},
            "by_host_species": {},
        }
    cardinality = df["mhc_allele_cardinality"].astype("string").fillna("class_only_unknown")
    provenance = df["mhc_allele_set_provenance"].astype("string").fillna("unknown")
    mhc_sp = df["mhc_species"].astype("string").fillna("")
    host_sp = df["host_species"].astype("string").fillna("")
    chimeric = df["is_chimeric_mhc_host"].astype("bool")
    card_counts = Counter(cardinality.tolist())
    return {
        "n_rows": len(df),
        "n_resolved_rows": int(
            card_counts.get("monoallelic", 0) + card_counts.get("multi_allelic_bag", 0)
        ),
        "n_monoallelic_rows": int(card_counts.get("monoallelic", 0)),
        "n_multi_allelic_bag_rows": int(card_counts.get("multi_allelic_bag", 0)),
        "n_class_only_unknown_rows": int(card_counts.get("class_only_unknown", 0)),
        "n_chimeric_mhc_host_rows": int(chimeric.sum()),
        "by_provenance": dict(sorted(Counter(provenance.tolist()).items())),
        "by_cardinality": dict(sorted(card_counts.items())),
        "by_mhc_species": dict(sorted(Counter(s or "unknown" for s in mhc_sp.tolist()).items())),
        "by_host_species": dict(sorted(Counter(s or "unknown" for s in host_sp.tolist()).items())),
    }
