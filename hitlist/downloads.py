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


def data_dir() -> Path:
    """Return the hitlist data directory, creating it if needed."""
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
}


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
