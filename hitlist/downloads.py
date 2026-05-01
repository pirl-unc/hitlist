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

"""Data management for IEDB, CEDAR, HPA, viral proteomes, and other external datasets.

Tracks downloaded data files across sessions with metadata (source URL,
download date, file size, row count where applicable). Supports both
auto-fetchable datasets (UniProt proteomes, HPA downloads) and manually
downloaded datasets (IEDB/CEDAR behind terms-of-use).

Storage location: ``~/.hitlist/`` (override with ``HITLIST_DATA_DIR`` env var).

Python API::

    from hitlist.downloads import register, get_path, fetch, info, list_datasets

    register("iedb", "/data/mhc_ligand_full.csv")
    path = get_path("iedb")
    fetch("hpv16")
    info("iedb")  # detailed metadata
    list_datasets()

CLI::

    hitlist data register iedb /data/mhc_ligand_full.csv
    hitlist data fetch hpv16
    hitlist data list
    hitlist data info iedb
    hitlist data path iedb
    hitlist data refresh hpv16
    hitlist data remove iedb
"""

from __future__ import annotations

import json
import os
import shutil
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

# ── Data directory ──────────────────────────────────────────────────────────

_DEFAULT_DATA_DIR = Path.home() / ".hitlist"
_override_data_dir: Path | None = None


def set_data_dir(path: str | Path) -> None:
    """Override the data directory for this session.

    Parameters
    ----------
    path
        Directory to use for all data storage. Created if it doesn't exist.

    Example
    -------
    >>> from hitlist.downloads import set_data_dir
    >>> set_data_dir("/data/shared/hitlist")
    """
    global _override_data_dir
    _override_data_dir = Path(path)


def data_dir() -> Path:
    """Return the hitlist data directory, creating it if needed.

    Priority: ``set_data_dir()`` > ``HITLIST_DATA_DIR`` env var > ``~/.hitlist/``.
    """
    if _override_data_dir is not None:
        d = _override_data_dir
    else:
        d = Path(os.environ.get("HITLIST_DATA_DIR", str(_DEFAULT_DATA_DIR)))
    d.mkdir(parents=True, exist_ok=True)
    return d


def _manifest_path() -> Path:
    return data_dir() / "manifest.json"


def _load_manifest() -> dict:
    p = _manifest_path()
    if p.exists():
        return json.loads(p.read_text())
    return {"datasets": {}}


def _save_manifest(manifest: dict) -> None:
    p = _manifest_path()
    p.write_text(json.dumps(manifest, indent=2, default=str) + "\n")


# ── Known datasets ──────────────────────────────────────────────────────────

FETCHABLE_DATASETS: dict[str, dict[str, str]] = {
    # Viral proteomes (auto-downloadable from UniProt REST API)
    "hpv16": {
        "url": "https://rest.uniprot.org/uniprotkb/stream?query=proteome:UP000006729&format=fasta",
        "filename": "hpv16.fasta",
        "description": "HPV-16 proteome (UniProt UP000006729)",
        "usage": "Peptide generation for cervical/oropharyngeal cancer viral targets",
    },
    "hpv18": {
        "url": "https://rest.uniprot.org/uniprotkb/stream?query=proteome:UP000006728&format=fasta",
        "filename": "hpv18.fasta",
        "description": "HPV-18 proteome (UniProt UP000006728)",
        "usage": "Peptide generation for cervical cancer viral targets",
    },
    "ebv": {
        "url": "https://rest.uniprot.org/uniprotkb/stream?query=proteome:UP000153037&format=fasta",
        "filename": "ebv.fasta",
        "description": "EBV/HHV-4 proteome (UniProt UP000153037)",
        "usage": "Peptide generation for Burkitt lymphoma, NPC, Hodgkin lymphoma",
    },
    "htlv1": {
        "url": "https://rest.uniprot.org/uniprotkb/stream?query=proteome:UP000002063&format=fasta",
        "filename": "htlv1.fasta",
        "description": "HTLV-1 proteome (UniProt UP000002063)",
        "usage": "Peptide generation for adult T-cell leukemia/lymphoma",
    },
    "hbv": {
        "url": "https://rest.uniprot.org/uniprotkb/stream?query=proteome:UP000126453&format=fasta",
        "filename": "hbv.fasta",
        "description": "HBV proteome (UniProt UP000126453)",
        "usage": "Peptide generation for hepatocellular carcinoma",
    },
    "hcv": {
        "url": "https://rest.uniprot.org/uniprotkb/stream?query=proteome:UP000000518&format=fasta",
        "filename": "hcv.fasta",
        "description": "HCV proteome (UniProt UP000000518)",
        "usage": "Peptide generation for hepatocellular carcinoma, B-cell lymphoma",
    },
    "kshv": {
        "url": "https://rest.uniprot.org/uniprotkb/stream?query=proteome:UP000009113&format=fasta",
        "filename": "kshv.fasta",
        "description": "KSHV/HHV-8 proteome (UniProt UP000009113)",
        "usage": "Peptide generation for Kaposi sarcoma",
    },
    "mcpyv": {
        "url": "https://rest.uniprot.org/uniprotkb/stream?query=proteome:UP000116695&format=fasta",
        "filename": "mcpyv.fasta",
        "description": "MCPyV proteome (UniProt UP000116695)",
        "usage": "Peptide generation for Merkel cell carcinoma",
    },
    "hiv1": {
        "url": "https://rest.uniprot.org/uniprotkb/stream?query=proteome:UP000002241&format=fasta",
        "filename": "hiv1.fasta",
        "description": "HIV-1 proteome (UniProt UP000002241)",
        "usage": "Peptide generation for Kaposi sarcoma, lymphoma (indirect)",
    },
}

MANUAL_DATASETS: dict[str, dict[str, str]] = {
    "iedb": {
        "download_url": "https://www.iedb.org/downloader.php?file_name=doc/mhc_ligand_full_single_file.zip",
        "description": "IEDB MHC ligand full export",
        "expected_filename": "mhc_ligand_full.csv",
        "usage": "Mass spec evidence for peptide-MHC presentation. Requires IEDB terms acceptance.",
    },
    "cedar": {
        "download_url": "https://cedar.iedb.org/downloader.php?file_name=doc/cedar_mhc_ligand_full.zip",
        "description": "CEDAR MHC ligand full export",
        "expected_filename": "cedar-mhc-ligand-full.csv",
        "usage": "Additional mass spec evidence (companion to IEDB).",
    },
    "hpa_bulk": {
        "download_url": "https://www.proteinatlas.org/download/proteinatlas.tsv.zip",
        "description": "HPA proteinatlas.tsv bulk summary",
        "expected_filename": "proteinatlas.tsv",
        "usage": "RNA tissue specificity, distribution, nTPM per gene for CTA restriction analysis.",
    },
    "hpa_rna": {
        "download_url": "https://www.proteinatlas.org/download/rna_tissue_consensus.tsv.zip",
        "description": "HPA RNA tissue consensus (50 tissues)",
        "expected_filename": "rna_tissue_consensus.tsv",
        "usage": "Per-tissue nTPM values for deflated reproductive fraction computation.",
    },
    "hpa_protein": {
        "download_url": "https://www.proteinatlas.org/download/normal_tissue.tsv.zip",
        "description": "HPA normal tissue IHC (63 tissues)",
        "expected_filename": "normal_tissue.tsv",
        "usage": "Protein-level tissue expression for CTA restriction analysis.",
    },
    "depmap_rna": {
        "download_url": "https://depmap.org/portal/data_page/?tab=allData",
        "description": "DepMap 24Q4 protein-coding gene TPM (log2(TPM+1))",
        "expected_filename": "OmicsExpressionProteinCodingGenesTPMLogp1.csv",
        "usage": (
            "Per-cell-line RNA expression anchor for tier-1 exact-line "
            "resolution of HeLa / A375 / SaOS-2 / THP-1 / K562 / HEK293 in "
            "hitlist.line_expression. Gene-level only. Ships as log2(TPM+1). "
            "See the DepMap portal for the Figshare-hosted CSV; ~160MB."
        ),
    },
    "depmap_rna_transcript": {
        "download_url": "https://depmap.org/portal/data_page/?tab=allData",
        "description": "DepMap 24Q4 transcript-level TPM (log2(TPM+1))",
        "expected_filename": "OmicsExpressionTranscriptsTPMLogp1.csv",
        "usage": (
            "Optional companion to depmap_rna. When registered, enables "
            "transcript-isoform-aware peptide-origin summation in "
            "generate_training_table(with_peptide_origin=True). ~600MB."
        ),
    },
    "depmap_models": {
        "download_url": "https://depmap.org/portal/data_page/?tab=allData",
        "description": "DepMap Model.csv (ModelID → StrippedCellLineName metadata)",
        "expected_filename": "Model.csv",
        "usage": (
            "Required companion to depmap_rna / depmap_rna_transcript: lets "
            "the line-expression builder map DepMap ModelIDs (ACH-xxxxxx) "
            "onto the registry's expression_key values. Without it, most "
            "DepMap rows are dropped with a warning. ~5MB."
        ),
    },
}


# ── Species / viral proteome registry ───────────────────────────────────────
#
# Maps canonical species names (from normalize_species()) to reference
# proteomes.  For "ensembl" species, callers use pyensembl directly
# (ProteomeIndex.from_ensembl).  For "uniprot" species, we download the
# reference proteome FASTA from UniProt's REST API and cache it locally.
#
# Proteome IDs: https://www.uniprot.org/proteomes/
#
# Keys must match the output of ``curation.normalize_species()``.

_UNIPROT_PROTEOME_URL = (
    "https://rest.uniprot.org/uniprotkb/stream"
    "?query=proteome:{proteome_id}&format=fasta&compressed=false"
)


SPECIES_PROTEOMES: dict[str, dict[str, str | int]] = {
    # Ensembl-supported (pyensembl)
    "Homo sapiens": {"kind": "ensembl", "release": 112, "species": "human"},
    "Mus musculus": {"kind": "ensembl", "release": 112, "species": "mouse"},
    "Rattus norvegicus": {"kind": "ensembl", "release": 112, "species": "rat"},
    # UniProt reference proteomes (auto-downloaded)
    "Sarcophilus harrisii": {"kind": "uniprot", "proteome_id": "UP000007648"},
    "Canis lupus": {"kind": "uniprot", "proteome_id": "UP000002254"},
    "Bos taurus": {"kind": "uniprot", "proteome_id": "UP000009136"},
    "Gallus gallus": {"kind": "uniprot", "proteome_id": "UP000000539"},
    "Sus scrofa": {"kind": "uniprot", "proteome_id": "UP000008227"},
    "Macaca mulatta": {"kind": "uniprot", "proteome_id": "UP000006718"},
    "Equus caballus": {"kind": "uniprot", "proteome_id": "UP000002281"},
    "Pan troglodytes": {"kind": "uniprot", "proteome_id": "UP000002277"},
    "Trichosurus vulpecula": {"kind": "uniprot", "proteome_id": "UP000504604"},
    # IEDB genus-abbreviated names — treat as the type species
    "Sus sp.": {"kind": "uniprot", "proteome_id": "UP000008227"},  # → Sus scrofa
    "Canis sp.": {"kind": "uniprot", "proteome_id": "UP000002254"},  # → Canis lupus
    "Bos sp.": {"kind": "uniprot", "proteome_id": "UP000009136"},  # → Bos taurus
    "Rattus sp.": {"kind": "ensembl", "release": 112, "species": "rat"},
    # Parasites / plants / other with curated UPIDs
    "Theileria parva": {"kind": "uniprot", "proteome_id": "UP000001949"},
    "Ascaris suum": {"kind": "uniprot", "proteome_id": "UP000036681"},
    "Ascaris lumbricoides": {"kind": "uniprot", "proteome_id": "UP000036681"},
    # Bacteria — canonical reference strains. (#39, surfaced via
    # qc.proteome_coverage in v1.30.23: Mtb is the largest non-host
    # bacterial source organism in IEDB at ~1.3K rows / 56 PMIDs across
    # the unstrained and H37Rv entries combined.)
    "Mycobacterium tuberculosis": {"kind": "uniprot", "proteome_id": "UP000001584"},  # H37Rv
    "Mycobacterium tuberculosis H37Rv": {"kind": "uniprot", "proteome_id": "UP000001584"},
    # Apicomplexan parasites — 3D7 is the canonical P. falciparum
    # reference used by virtually all immunology and antimalarial
    # epitope work.
    "Plasmodium falciparum": {"kind": "uniprot", "proteome_id": "UP000001450"},  # 3D7
}


# Viral proteomes keyed by the IEDB ``source_organism`` string we observe.
# For matching, we lowercase both sides and check substring inclusion of the
# registry key.  This tolerates IEDB variations (e.g. "Epstein-Barr virus
# (strain B95-8)" matches "epstein-barr virus").
VIRAL_PROTEOMES: dict[str, dict[str, str]] = {
    # SARS coronaviruses — "sars-cov" must come before "sars-cov-2" check below
    # but we sort by key length at lookup time to match most-specific first
    "severe acute respiratory syndrome coronavirus 2": {
        "proteome_id": "UP000464024",
        "key": "sars-cov-2",
    },
    "sars-cov2": {"proteome_id": "UP000464024", "key": "sars-cov-2"},
    "sars-cov-2": {"proteome_id": "UP000464024", "key": "sars-cov-2"},
    "severe acute respiratory syndrome coronavirus": {
        "proteome_id": "UP000000354",
        "key": "sars-cov-1",
    },
    "sars-cov1": {"proteome_id": "UP000000354", "key": "sars-cov-1"},
    "sars-cov-1": {"proteome_id": "UP000000354", "key": "sars-cov-1"},
    "sars coronavirus": {"proteome_id": "UP000000354", "key": "sars-cov-1"},
    # Herpes viruses
    "human immunodeficiency virus 1": {"proteome_id": "UP000002241", "key": "hiv1"},
    "hiv-1": {"proteome_id": "UP000002241", "key": "hiv1"},
    "epstein-barr virus": {"proteome_id": "UP000153037", "key": "ebv"},
    "human gammaherpesvirus 4": {"proteome_id": "UP000153037", "key": "ebv"},
    "human herpesvirus 4": {"proteome_id": "UP000153037", "key": "ebv"},
    "human betaherpesvirus 5": {"proteome_id": "UP000000938", "key": "hcmv"},
    "human cytomegalovirus": {"proteome_id": "UP000000938", "key": "hcmv"},
    "human herpesvirus 5": {"proteome_id": "UP000000938", "key": "hcmv"},
    "human gammaherpesvirus 8": {"proteome_id": "UP000009113", "key": "kshv"},
    "human herpesvirus 8": {"proteome_id": "UP000009113", "key": "kshv"},
    "kaposi": {"proteome_id": "UP000009113", "key": "kshv"},
    "human alphaherpesvirus 1": {"proteome_id": "UP000009294", "key": "hsv-1"},
    "herpes simplex virus type 1": {"proteome_id": "UP000009294", "key": "hsv-1"},
    "human herpesvirus 1": {"proteome_id": "UP000009294", "key": "hsv-1"},
    "human alphaherpesvirus 2": {"proteome_id": "UP000001874", "key": "hsv-2"},
    "herpes simplex virus type 2": {"proteome_id": "UP000001874", "key": "hsv-2"},
    "human herpesvirus 2": {"proteome_id": "UP000001874", "key": "hsv-2"},
    "human betaherpesvirus 6b": {"proteome_id": "UP000006930", "key": "hhv-6b"},
    "human herpesvirus 6b": {"proteome_id": "UP000006930", "key": "hhv-6b"},
    "murid betaherpesvirus 1": {"proteome_id": "UP000008774", "key": "mcmv"},
    "murid herpesvirus 1": {"proteome_id": "UP000008774", "key": "mcmv"},
    "murine cytomegalovirus": {"proteome_id": "UP000008774", "key": "mcmv"},
    # Hepatitis
    "hepatitis b virus": {"proteome_id": "UP000126453", "key": "hbv"},
    "hepatitis c virus": {"proteome_id": "UP000000518", "key": "hcv"},
    "hepacivirus hominis": {"proteome_id": "UP000000518", "key": "hcv"},
    # Papillomaviruses
    "human papillomavirus type 16": {"proteome_id": "UP000006729", "key": "hpv16"},
    "human papillomavirus 16": {"proteome_id": "UP000006729", "key": "hpv16"},
    "human papillomavirus type 18": {"proteome_id": "UP000006728", "key": "hpv18"},
    "human papillomavirus 18": {"proteome_id": "UP000006728", "key": "hpv18"},
    # Influenza
    "influenza a virus": {"proteome_id": "UP000009255", "key": "influenza-a"},
    "influenza b virus": {"proteome_id": "UP000008158", "key": "influenza-b"},
    # Poxviruses
    "vaccinia virus": {"proteome_id": "UP000000344", "key": "vaccinia"},
    "orf virus": {"proteome_id": "UP000000870", "key": "orf"},
    # Animal pathogens
    "canine distemper virus": {"proteome_id": "UP000117312", "key": "cdv"},
    "african swine fever virus": {"proteome_id": "UP000000624", "key": "asfv"},
    "porcine reproductive and respiratory syndrome virus": {
        "proteome_id": "UP000006706",
        "key": "prrsv",
    },
    "wobbly possum disease virus": {"proteome_id": "UP000147130", "key": "wpdv"},
    "peste-des-petits-ruminants virus": {"proteome_id": "UP000100083", "key": "pprv"},
    # Respiratory / other human viruses
    "human respiratory syncytial virus": {"proteome_id": "UP000002472", "key": "rsv"},
    "human orthopneumovirus": {"proteome_id": "UP000002472", "key": "rsv"},
    "human metapneumovirus": {"proteome_id": "UP000001398", "key": "hmpv"},
    "zika virus": {"proteome_id": "UP000054557", "key": "zika"},
    "rotavirus a": {"proteome_id": "UP000001119", "key": "rotavirus-a"},
    "lymphocytic choriomeningitis virus": {"proteome_id": "UP000002474", "key": "lcmv"},
    # Polyomaviruses
    "betapolyomavirus hominis": {"proteome_id": "UP000008475", "key": "bkv"},
    "bk polyomavirus": {"proteome_id": "UP000008475", "key": "bkv"},
    "alphapolyomavirus muris": {"proteome_id": "UP000007212", "key": "mpyv"},
}


def _proteomes_dir() -> Path:
    d = data_dir() / "proteomes"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _safe_filename(species: str) -> str:
    """Convert a species name to a filesystem-safe filename."""
    safe = species.lower().replace("/", "_").replace("\\", "_")
    safe = "".join(c if c.isalnum() or c in "_-." else "_" for c in safe)
    # Collapse runs of underscores
    while "__" in safe:
        safe = safe.replace("__", "_")
    return safe.strip("_") + ".fasta"


_UNIPROT_PROTEOME_SEARCH_URL = "https://rest.uniprot.org/proteomes/search"
_PROTEOME_TYPE_RANK = {
    "reference and representative proteome": 0,
    "reference proteome": 1,
    "representative proteome": 2,
    "other proteome": 3,
    "redundant proteome": 4,
}


# Organism strings that are known placeholders/noise — never send to UniProt.
_ORGANISM_DENYLIST: set[str] = {
    "",
    "unidentified",
    "unknown",
    "unclassified",
    "mixed",
    "various",
    "not available",
    "n/a",
    "na",
}


def resolve_proteome_via_uniprot(
    organism: str,
    timeout: int = 15,
) -> dict | None:
    """Query UniProt REST to find the best reference proteome for an organism.

    Returns a dict with ``proteome_id``, ``scientific_name``, ``taxon_id``,
    ``proteome_type``, and ``protein_count`` fields.  Prefers "Reference
    and representative" over plain "Representative" proteomes.  Returns
    ``None`` if no match is found or if the organism is on the denylist
    (e.g. ``"unidentified"``).

    The raw organism string is used as a free-text query, so strain
    suffixes like ``"(strain B95-8)"`` are tolerated.
    """
    import json
    import urllib.parse

    if not organism:
        return None
    cleaned = organism.strip()
    if cleaned.lower() in _ORGANISM_DENYLIST:
        return None
    query = urllib.parse.quote(cleaned)
    url = f"{_UNIPROT_PROTEOME_SEARCH_URL}?query={query}&format=json&size=10"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            payload = json.load(r)
    except Exception:
        return None

    results = payload.get("results", [])
    if not results:
        return None

    def sort_key(p: dict) -> tuple:
        ptype = (p.get("proteomeType") or "").lower()
        rank = _PROTEOME_TYPE_RANK.get(ptype, 99)
        # Within same rank, prefer lower taxon id (parent species often
        # has a smaller ID than strain-specific entries) and higher count
        tax_id = int(p.get("taxonomy", {}).get("taxonId") or 1_000_000_000)
        count = int(p.get("proteinCount") or 0)
        return (rank, tax_id, -count)

    best = min(results, key=sort_key)
    tax = best.get("taxonomy", {})
    return {
        "proteome_id": best.get("id"),
        "scientific_name": tax.get("scientificName") or organism,
        "taxon_id": tax.get("taxonId"),
        "proteome_type": best.get("proteomeType"),
        "protein_count": best.get("proteinCount"),
    }


def _find_existing_proteome_by_upid(proteomes: dict, proteome_id: str) -> dict | None:
    """Find a previously-downloaded proteome with the same UniProt ID."""
    for entry in proteomes.values():
        if (
            entry.get("kind") == "uniprot"
            and entry.get("proteome_id") == proteome_id
            and entry.get("path")
        ):
            return entry
    return None


def _uniprot_cache() -> dict:
    """Load the manifest's uniprot_resolutions section."""
    manifest = _load_manifest()
    return manifest.get("uniprot_resolutions", {})


def _save_uniprot_cache_entry(organism: str, entry: dict | None) -> None:
    manifest = _load_manifest()
    cache = manifest.setdefault("uniprot_resolutions", {})
    cache[organism] = {
        "resolved_at": datetime.now(timezone.utc).isoformat(),
        **(entry or {"not_found": True}),
    }
    _save_manifest(manifest)


def lookup_proteome(
    species_or_organism: str,
    use_uniprot: bool = False,
) -> dict | None:
    """Resolve a species or IEDB source_organism string to a proteome registry entry.

    Applies in order:
    1. Curated ``SPECIES_PROTEOMES`` registry (normalized via mhcgnomes)
    2. Curated ``VIRAL_PROTEOMES`` substring match
    3. (Optional) UniProt REST lookup — only if ``use_uniprot=True``
       and the resolution hasn't been cached yet.  Negative results
       (``{"not_found": True}``) are also cached to avoid re-querying.

    Returns ``None`` if no proteome is registered/discoverable.
    """
    if not species_or_organism:
        return None

    from .curation import normalize_species

    canonical = normalize_species(species_or_organism)
    if canonical in SPECIES_PROTEOMES:
        entry = dict(SPECIES_PROTEOMES[canonical])
        entry["canonical_species"] = canonical
        return entry

    # Viral fallback: substring match on raw organism string.
    # Longer keys are checked first so "sars-cov-2" wins over "sars-cov".
    lowered = species_or_organism.lower()
    for viral_key in sorted(VIRAL_PROTEOMES, key=len, reverse=True):
        if viral_key in lowered:
            viral_entry = VIRAL_PROTEOMES[viral_key]
            return {
                "kind": "uniprot",
                "proteome_id": viral_entry["proteome_id"],
                "canonical_species": species_or_organism,
                "key": viral_entry["key"],
            }

    # Species-name substring fallback (handles strain suffixes like
    # "Theileria parva strain Muguga" → Theileria parva).  Skip
    # generic genus-only entries ("Sus sp.", "Canis sp.") to avoid
    # spurious matches.
    for species_key in sorted(SPECIES_PROTEOMES, key=len, reverse=True):
        if species_key.endswith(" sp."):
            continue
        if species_key.lower() in lowered:
            entry = dict(SPECIES_PROTEOMES[species_key])
            entry["canonical_species"] = species_key
            return entry

    if not use_uniprot:
        return None

    # UniProt REST fallback — with manifest-cached results
    cache = _uniprot_cache()
    cached = cache.get(species_or_organism)
    if cached is not None:
        if cached.get("not_found"):
            return None
        return {
            "kind": "uniprot",
            "proteome_id": cached["proteome_id"],
            "canonical_species": cached.get("scientific_name", species_or_organism),
            "source": "uniprot_search",
            "taxon_id": cached.get("taxon_id"),
            "proteome_type": cached.get("proteome_type"),
        }

    resolved = resolve_proteome_via_uniprot(species_or_organism)
    _save_uniprot_cache_entry(species_or_organism, resolved)
    if resolved is None:
        return None
    return {
        "kind": "uniprot",
        "proteome_id": resolved["proteome_id"],
        "canonical_species": resolved["scientific_name"],
        "source": "uniprot_search",
        "taxon_id": resolved.get("taxon_id"),
        "proteome_type": resolved.get("proteome_type"),
    }


def fetch_species_proteome(
    species: str,
    force: bool = False,
    verbose: bool = True,
    use_uniprot: bool = False,
) -> Path | None:
    """Fetch (or return cached) reference proteome FASTA for a species.

    Returns the local FASTA path.  For Ensembl-supported species this is
    a sentinel (marker file) indicating the caller should use pyensembl
    instead.  Returns ``None`` if no proteome is registered.

    Parameters
    ----------
    species
        Any species or source_organism string.
    force
        Re-download even if already cached.
    verbose
        Print progress messages.
    use_uniprot
        Fall back to UniProt REST search for organisms not in the curated
        registry.  Resolved mappings are cached in the manifest to avoid
        re-querying.
    """
    entry = lookup_proteome(species, use_uniprot=use_uniprot)
    if entry is None:
        return None

    canonical = entry.get("canonical_species", species)
    manifest = _load_manifest()
    proteomes = manifest.setdefault("proteomes", {})

    # Ensembl species: pyensembl manages its own cache — no FASTA to download
    if entry["kind"] == "ensembl":
        proteomes[canonical] = {
            "kind": "ensembl",
            "species": entry.get("species", canonical),
            "release": entry.get("release", 112),
            "registered": datetime.now(timezone.utc).isoformat(),
        }
        _save_manifest(manifest)
        return None  # caller should use ProteomeIndex.from_ensembl()

    # UniProt species: download the FASTA (dedup by UPID — multiple strain
    # variants often resolve to the same UniProt reference proteome)
    proteome_id = entry["proteome_id"]
    url = _UNIPROT_PROTEOME_URL.format(proteome_id=proteome_id)

    existing = _find_existing_proteome_by_upid(proteomes, proteome_id)
    if existing is not None and Path(existing["path"]).exists() and not force:
        dest = Path(existing["path"])
        if verbose:
            print(
                f"  [{canonical}] reusing cached FASTA from "
                f"{existing.get('canonical_species', '?')} ({dest.stat().st_size:,} bytes)"
            )
        proteomes[canonical] = {
            "kind": "uniprot",
            "proteome_id": proteome_id,
            "path": str(dest),
            "size_bytes": dest.stat().st_size,
            "source_url": url,
            "canonical_species": canonical,
            "registered": datetime.now(timezone.utc).isoformat(),
        }
        _save_manifest(manifest)
        return dest

    fname = _safe_filename(canonical)
    dest = _proteomes_dir() / fname

    if dest.exists() and not force:
        if verbose:
            print(f"  [{canonical}] already cached ({dest.stat().st_size:,} bytes)")
        proteomes[canonical] = {
            "kind": "uniprot",
            "proteome_id": proteome_id,
            "path": str(dest),
            "size_bytes": dest.stat().st_size,
            "source_url": url,
            "canonical_species": canonical,
            "registered": proteomes.get(canonical, {}).get(
                "registered", datetime.now(timezone.utc).isoformat()
            ),
        }
        _save_manifest(manifest)
        return dest

    if verbose:
        print(f"  [{canonical}] fetching UniProt {proteome_id} ...")
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    try:
        urllib.request.urlretrieve(url, str(tmp))
        shutil.move(str(tmp), str(dest))
    finally:
        if tmp.exists():
            tmp.unlink()

    size = dest.stat().st_size
    if verbose:
        print(f"  [{canonical}] downloaded {size:,} bytes → {dest}")

    proteomes[canonical] = {
        "kind": "uniprot",
        "proteome_id": proteome_id,
        "path": str(dest),
        "size_bytes": size,
        "source_url": url,
        "registered": datetime.now(timezone.utc).isoformat(),
    }
    _save_manifest(manifest)
    return dest


def list_proteomes() -> dict:
    """Return the proteomes section of the manifest."""
    return _load_manifest().get("proteomes", {})


def fetch_proteome_by_upid(
    upid: str,
    label: str | None = None,
    force: bool = False,
    verbose: bool = True,
) -> Path | None:
    """Fetch (or return cached) a UniProt reference proteome by UPID.

    Unlike ``fetch_species_proteome`` which requires the organism to be
    in the curated registry, this fetches any UPID directly.  Used by
    the ``reference_proteomes`` override on ``ms_samples`` for per-sample
    viral/custom proteomes.

    Parameters
    ----------
    upid
        UniProt proteome ID (e.g. ``"UP000153037"``).
    label
        Optional human-readable name for logging and the filename.
        Defaults to the UPID.
    force
        Re-download even if already cached.
    verbose
        Print progress messages.
    """
    if not upid:
        return None
    manifest = _load_manifest()
    proteomes = manifest.setdefault("proteomes", {})

    # Dedup by UPID if we already have this proteome under another key
    existing = _find_existing_proteome_by_upid(proteomes, upid)
    if existing is not None and Path(existing["path"]).exists() and not force:
        path = Path(existing["path"])
        if verbose:
            print(f"  [{label or upid}] reusing cached {path.name} ({path.stat().st_size:,} bytes)")
        return path

    name = label or upid
    fname = _safe_filename(name)
    dest = _proteomes_dir() / fname
    url = _UNIPROT_PROTEOME_URL.format(proteome_id=upid)

    if dest.exists() and not force:
        if verbose:
            print(f"  [{name}] already cached ({dest.stat().st_size:,} bytes)")
    else:
        if verbose:
            print(f"  [{name}] fetching UniProt {upid} ...")
        tmp = dest.with_suffix(dest.suffix + ".tmp")
        try:
            urllib.request.urlretrieve(url, str(tmp))
            shutil.move(str(tmp), str(dest))
        finally:
            if tmp.exists():
                tmp.unlink()
        if verbose:
            print(f"  [{name}] downloaded {dest.stat().st_size:,} bytes → {dest}")

    proteomes[name] = {
        "kind": "uniprot",
        "proteome_id": upid,
        "path": str(dest),
        "size_bytes": dest.stat().st_size,
        "source_url": url,
        "canonical_species": name,
        "registered": datetime.now(timezone.utc).isoformat(),
    }
    _save_manifest(manifest)
    return dest


# ── Core API ────────────────────────────────────────────────────────────────


def register(name: str, path: str | Path, description: str | None = None) -> Path:
    """Register a local file path for a named dataset."""
    p = Path(path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")

    manifest = _load_manifest()
    desc = description
    if desc is None and name in MANUAL_DATASETS:
        desc = MANUAL_DATASETS[name]["description"]
    if desc is None and name in FETCHABLE_DATASETS:
        desc = FETCHABLE_DATASETS[name]["description"]

    manifest["datasets"][name] = {
        "path": str(p),
        "registered": datetime.now(timezone.utc).isoformat(),
        "size_bytes": p.stat().st_size,
        "description": desc or "",
        "source": "registered",
    }
    _save_manifest(manifest)
    return p


def fetch(name: str, force: bool = False) -> Path:
    """Download a fetchable dataset."""
    if name not in FETCHABLE_DATASETS:
        if name in MANUAL_DATASETS:
            info = MANUAL_DATASETS[name]
            raise ValueError(
                f"'{name}' requires manual download from:\n"
                f"  {info['download_url']}\n"
                f"Then register: hitlist data register {name} /path/to/{info['expected_filename']}"
            )
        available = sorted(set(FETCHABLE_DATASETS) | set(MANUAL_DATASETS))
        raise ValueError(f"Unknown dataset '{name}'. Available: {available}")

    ds = FETCHABLE_DATASETS[name]
    dest = data_dir() / ds["filename"]

    if dest.exists() and not force:
        manifest = _load_manifest()
        if name not in manifest.get("datasets", {}):
            register(name, dest, ds["description"])
        return dest

    print(f"Downloading {ds['description']}...")
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    try:
        urllib.request.urlretrieve(ds["url"], str(tmp))
        shutil.move(str(tmp), str(dest))
    finally:
        if tmp.exists():
            tmp.unlink()

    manifest = _load_manifest()
    manifest["datasets"][name] = {
        "path": str(dest),
        "registered": datetime.now(timezone.utc).isoformat(),
        "size_bytes": dest.stat().st_size,
        "description": ds["description"],
        "source": ds["url"],
    }
    _save_manifest(manifest)
    print(f"  Saved to {dest} ({dest.stat().st_size:,} bytes)")
    return dest


def get_path(name: str) -> Path:
    """Resolve a dataset name to its local file path."""
    manifest = _load_manifest()
    entry = manifest.get("datasets", {}).get(name)
    if entry is None:
        hint = ""
        if name in FETCHABLE_DATASETS:
            hint = f"\n  Fetch with: hitlist data fetch {name}"
        elif name in MANUAL_DATASETS:
            info = MANUAL_DATASETS[name]
            hint = (
                f"\n  Download from: {info['download_url']}"
                f"\n  Then register: hitlist data register {name} /path/to/file"
            )
        raise KeyError(f"Dataset '{name}' not registered.{hint}")

    p = Path(entry["path"])
    if not p.exists():
        raise FileNotFoundError(
            f"Registered path for '{name}' no longer exists: {p}\nRe-register or re-fetch."
        )
    return p


def info(name: str) -> dict:
    """Get detailed metadata for a registered dataset."""
    manifest = _load_manifest()
    entry = manifest.get("datasets", {}).get(name)
    if entry is None:
        # Return known info even if not registered
        if name in FETCHABLE_DATASETS:
            return {**FETCHABLE_DATASETS[name], "status": "not installed", "type": "auto-fetch"}
        if name in MANUAL_DATASETS:
            return {**MANUAL_DATASETS[name], "status": "not installed", "type": "manual download"}
        raise KeyError(f"Unknown dataset '{name}'")
    result = dict(entry)
    result["status"] = "installed"
    if name in FETCHABLE_DATASETS:
        result["type"] = "auto-fetch"
        result["usage"] = FETCHABLE_DATASETS[name].get("usage", "")
    elif name in MANUAL_DATASETS:
        result["type"] = "manual download"
        result["usage"] = MANUAL_DATASETS[name].get("usage", "")
    return result


def list_datasets() -> dict[str, dict]:
    """Return all registered/fetched datasets."""
    return dict(_load_manifest().get("datasets", {}))


def available_datasets() -> dict[str, str]:
    """Return all known dataset names with descriptions."""
    result = {}
    for name, ds in FETCHABLE_DATASETS.items():
        result[name] = ds["description"] + " [auto-fetch]"
    for name, ds in MANUAL_DATASETS.items():
        result[name] = ds["description"] + " [manual download]"
    return result


def refresh(name: str) -> Path:
    """Re-download a fetchable dataset."""
    return fetch(name, force=True)


def remove(name: str, delete_file: bool = False) -> None:
    """Remove a dataset from the registry. Optionally delete the file.

    Parameters
    ----------
    name
        Dataset name to unregister.
    delete_file
        If True, also delete the file on disk.
    """
    manifest = _load_manifest()
    entry = manifest.get("datasets", {}).pop(name, None)
    _save_manifest(manifest)
    if delete_file and entry:
        p = Path(entry["path"])
        if p.exists():
            p.unlink()
            print(f"Deleted {p}")
