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

"""Build the unified observations table from IEDB + CEDAR.

Runs the full scanner (with source classification from YAML overrides)
on each registered data source, deduplicates by assay IRI, optionally
maps peptides to source proteins with flanking context, and writes
a single ``observations.parquet`` to ``~/.hitlist/``.

Usage::

    from hitlist.builder import build_observations

    path = build_observations()                     # IEDB + CEDAR
    path = build_observations(with_flanking=True)   # + proteome mapping

CLI::

    hitlist data build
    hitlist data build --with-flanking
    hitlist data build --force
"""

from __future__ import annotations

import contextlib
import json
from pathlib import Path

import pandas as pd

from .downloads import data_dir


def _source_paths() -> dict[str, Path]:
    """Resolve registered IEDB/CEDAR paths."""
    from .downloads import get_path

    sources: dict[str, Path] = {}
    for name in ("iedb", "cedar"):
        with contextlib.suppress(KeyError, FileNotFoundError):
            p = get_path(name)
            if p.exists():
                sources[name] = p
    return sources


def _source_fingerprints(paths: dict[str, Path]) -> dict:
    """File identity for cache invalidation."""
    fp = {
        name: {"path": str(p), "size": p.stat().st_size, "mtime": p.stat().st_mtime}
        for name, p in paths.items()
    }
    # Include supplementary manifest so adding a new supplement invalidates cache
    from .supplement import load_supplementary_manifest, manifest_path

    mp = manifest_path()
    if mp.exists():
        fp["supplementary_manifest"] = {
            "path": str(mp),
            "size": mp.stat().st_size,
            "mtime": mp.stat().st_mtime,
        }
        # Also fingerprint each referenced CSV so edits invalidate cache
        supp_dir = mp.parent / "supplementary"
        for entry in load_supplementary_manifest():
            csv_path = supp_dir / entry["file"]
            if csv_path.exists():
                key = f"supplementary_csv:{entry['file']}"
                fp[key] = {
                    "path": str(csv_path),
                    "size": csv_path.stat().st_size,
                    "mtime": csv_path.stat().st_mtime,
                }
    return fp


def _observations_path() -> Path:
    return data_dir() / "observations.parquet"


def _meta_path() -> Path:
    return data_dir() / "observations_meta.json"


def _cache_is_valid(paths: dict[str, Path]) -> bool:
    meta = _meta_path()
    if not meta.exists():
        return False
    if not _observations_path().exists():
        return False
    stored = json.loads(meta.read_text())
    current = _source_fingerprints(paths)
    return stored.get("sources") == current


def build_observations(
    with_flanking: bool = False,
    proteome_release: int = 112,
    force: bool = False,
) -> Path:
    """Build the unified observations table from IEDB + CEDAR.

    Scans each registered source CSV through the full classification
    pipeline (YAML overrides, tissue categories, mono-allelic detection),
    deduplicates by assay IRI, and writes ``observations.parquet``.

    Parameters
    ----------
    with_flanking
        Map all unique peptides to source proteins with 10aa flanking
        context via :class:`~hitlist.proteome.ProteomeIndex`.
    proteome_release
        Ensembl release for proteome mapping (default 112).
    force
        Rebuild even if cache is valid.

    Returns
    -------
    Path
        Path to ``observations.parquet``.
    """
    paths = _source_paths()
    if not paths:
        raise FileNotFoundError(
            "No IEDB/CEDAR data registered. Use: hitlist data register iedb /path/to/file.csv"
        )

    out_path = _observations_path()
    if not force and _cache_is_valid(paths):
        return out_path

    from .scanner import scan

    dfs: list[pd.DataFrame] = []
    seen_iris: set[str] = set()

    for name in ("iedb", "cedar"):
        if name not in paths:
            continue
        print(f"Scanning {name} ({paths[name].name})...")
        df = scan(
            peptides=None,
            iedb_path=paths[name] if name == "iedb" else None,
            cedar_path=paths[name] if name == "cedar" else None,
            human_only=False,
            classify_source=True,
        )
        df["source"] = name

        # Deduplicate across sources by assay IRI
        if seen_iris:
            before = len(df)
            df = df[~df["reference_iri"].isin(seen_iris)]
            dupes = before - len(df)
            if dupes:
                print(f"  Deduplicated {dupes:,} rows (shared IRIs with prior source)")

        seen_iris.update(df["reference_iri"].values)
        dfs.append(df)
        print(f"  {len(df):,} rows from {name}")

    if not dfs:
        raise RuntimeError("No data scanned.")

    obs = pd.concat(dfs, ignore_index=True)

    # --- Supplementary data (peptides not in IEDB/CEDAR) ---
    from .supplement import scan_supplementary

    supp = scan_supplementary(classify_source=True)
    if not supp.empty:
        supp["source"] = "supplement"
        # Deduplicate: IEDB/CEDAR rows win over supplementary
        existing_keys = set(zip(obs["peptide"], obs["mhc_restriction"], obs["pmid"]))
        before = len(supp)
        supp = supp[
            ~pd.Series(
                [
                    (p, a, m) in existing_keys
                    for p, a, m in zip(supp["peptide"], supp["mhc_restriction"], supp["pmid"])
                ],
                index=supp.index,
            )
        ]
        dupes = before - len(supp)
        if dupes:
            print(f"  Deduplicated {dupes:,} supplementary rows (already in IEDB/CEDAR)")
        obs = pd.concat([obs, supp], ignore_index=True)
        print(f"  {len(supp):,} rows from supplementary data")

    print(f"\nTotal: {len(obs):,} observations")
    print(f"  Unique peptides: {obs['peptide'].nunique():,}")
    print(f"  Unique alleles:  {obs['mhc_restriction'].nunique():,}")
    print(f"  Species:         {obs['mhc_species'].nunique()}")

    # Fix mixed types for parquet compatibility
    if "pmid" in obs.columns:
        obs["pmid"] = pd.to_numeric(obs["pmid"], errors="coerce").astype("Int64")

    if with_flanking:
        obs = _add_flanking(obs, proteome_release)

    # Write parquet
    obs.to_parquet(out_path, index=False)
    print(f"\nWrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")

    # Save metadata
    meta = {
        "sources": _source_fingerprints(paths),
        "n_rows": len(obs),
        "n_peptides": int(obs["peptide"].nunique()),
        "n_alleles": int(obs["mhc_restriction"].nunique()),
        "n_species": int(obs["mhc_species"].nunique()),
        "with_flanking": with_flanking,
    }
    _meta_path().write_text(json.dumps(meta, indent=2, default=str) + "\n")

    return out_path


def _add_flanking(obs: pd.DataFrame, release: int) -> pd.DataFrame:
    """Map peptides to source proteins with flanking context."""
    from .proteome import ProteomeIndex

    peptides = obs["peptide"].unique().tolist()
    print(f"\nMapping {len(peptides):,} unique peptides to proteome (Ensembl {release})...")

    idx = ProteomeIndex.from_ensembl(release=release)
    flanking = idx.map_peptides(peptides, flank=10, verbose=True)

    if flanking.empty:
        print("  No proteome matches found.")
        for col in (
            "gene_name",
            "gene_id",
            "protein_id",
            "position",
            "n_flank",
            "c_flank",
            "n_sources",
        ):
            obs[col] = None
        return obs

    # Keep one representative mapping per peptide (prefer unique flank)
    best = flanking.sort_values("n_sources").drop_duplicates("peptide", keep="first")
    best = best[
        [
            "peptide",
            "gene_name",
            "gene_id",
            "protein_id",
            "position",
            "n_flank",
            "c_flank",
            "n_sources",
        ]
    ]
    best = best.rename(columns={"n_sources": "n_source_proteins"})

    obs = obs.merge(best, on="peptide", how="left")
    mapped = obs["gene_name"].notna().sum()
    print(f"  Mapped {mapped:,} / {len(obs):,} observations to source proteins")
    return obs
