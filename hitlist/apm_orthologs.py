"""Cross-species ortholog mapping for antigen processing & presentation genes (#944)."""

from __future__ import annotations

from functools import cache
from pathlib import Path

from .curation_yaml import load_curation_yaml

ORTHOLOGS_PATH = Path(__file__).resolve().parent / "data" / "apm_orthologs.yaml"

_SPECIES_ALIASES = {
    "human": "Homo sapiens",
    "homo sapiens": "Homo sapiens",
    "mouse": "Mus musculus",
    "mus musculus": "Mus musculus",
    "rat": "Rattus norvegicus",
    "rattus": "Rattus norvegicus",
    "rattus norvegicus": "Rattus norvegicus",
    "macaque": "Macaca mulatta",
    "rhesus": "Macaca mulatta",
    "macaca mulatta": "Macaca mulatta",
    "dog": "Canis lupus familiaris",
    "canis": "Canis lupus familiaris",
    "canis lupus": "Canis lupus familiaris",
    "canis lupus familiaris": "Canis lupus familiaris",
    "cow": "Bos taurus",
    "cattle": "Bos taurus",
    "bovine": "Bos taurus",
    "bos taurus": "Bos taurus",
    "pig": "Sus scrofa",
    "sus scrofa": "Sus scrofa",
    "chicken": "Gallus gallus",
    "gallus gallus": "Gallus gallus",
}


def _normalize_species_name(species: str | None) -> str | None:
    if not species:
        return None
    key = str(species).strip().casefold()
    return _SPECIES_ALIASES.get(key, str(species).strip())


@cache
def load_apm_orthologs(path: Path | None = None) -> dict:
    """Load :file:`hitlist/data/apm_orthologs.yaml`."""
    data = load_curation_yaml(path or ORTHOLOGS_PATH)
    if not isinstance(data, dict) or "ortholog_groups" not in data:
        raise ValueError("apm_orthologs.yaml must define 'ortholog_groups'")
    return data


@cache
def _symbol_index(path: Path | None = None) -> dict[str, str]:
    groups = load_apm_orthologs(path)["ortholog_groups"]
    index: dict[str, str] = {}
    for canonical, entry in groups.items():
        index[canonical.casefold()] = canonical
        for alias in entry.get("aliases") or ():
            index[str(alias).strip().casefold()] = canonical
        for symbols in (entry.get("species_symbols") or {}).values():
            for sym in symbols:
                index[str(sym).strip().casefold()] = canonical
    return index


def canonical_apm_ortholog_group(
    gene_symbol: str | None, *, path: Path | None = None
) -> str | None:
    """Map any species-specific APM gene symbol or alias to its canonical HGNC ortholog group.

    Examples: ``"H2-DMa" -> "HLA-DMA"``, ``"H2-DMb1" -> "HLA-DMB"``,
    ``"Ifi30" -> "IFI30"``, ``"GILT" -> "IFI30"``, ``"Erap1" -> "ERAP1"``.
    """
    if not gene_symbol:
        return None
    text = str(gene_symbol).strip()
    if not text:
        return None
    # Strip trailing KO / -/- suffixes if passed directly from perturbation notes.
    cleaned = text.removesuffix("-/-").removesuffix("_KO").removesuffix(" KO").strip()
    return _symbol_index(path).get(cleaned.casefold())


def species_ortholog_symbols(
    canonical_group: str, species: str, *, path: Path | None = None
) -> tuple[str, ...]:
    """Return the species-specific gene symbol(s) for a canonical APM ortholog group."""
    groups = load_apm_orthologs(path)["ortholog_groups"]
    group = canonical_apm_ortholog_group(canonical_group, path=path) or canonical_group
    entry = groups.get(group)
    if entry is None:
        return ()
    norm_species = _normalize_species_name(species)
    if not norm_species:
        return ()
    return tuple((entry.get("species_symbols") or {}).get(norm_species) or ())


def is_apm_gene_absent_in_species(
    canonical_group: str, species: str, *, path: Path | None = None
) -> bool:
    """Return True when the APM gene is biologically absent in ``species`` (e.g. ERAP2 in mouse)."""
    groups = load_apm_orthologs(path)["ortholog_groups"]
    group = canonical_apm_ortholog_group(canonical_group, path=path) or canonical_group
    entry = groups.get(group)
    if entry is None:
        return False
    norm_species = _normalize_species_name(species)
    return bool(norm_species and norm_species in (entry.get("absent_in_species") or ()))


def map_apm_genes_to_canonical(
    genes: str | list[str] | tuple[str, ...], *, path: Path | None = None
) -> tuple[str, ...]:
    """Normalize a semicolon/comma-separated string or iterable of APM gene symbols."""
    if isinstance(genes, str):
        raw_tokens = [tok.strip() for chunk in genes.split(";") for tok in chunk.split(",")]
    else:
        raw_tokens = [str(tok).strip() for tok in genes]
    resolved: list[str] = []
    for token in raw_tokens:
        if not token or token.casefold() == "none":
            continue
        canonical = canonical_apm_ortholog_group(token, path=path)
        resolved.append(canonical or token)
    return tuple(dict.fromkeys(resolved))
