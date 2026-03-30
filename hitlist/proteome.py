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

"""Peptide-to-source-protein mapping with flanking sequence extraction.

Builds an efficient k-mer index from a protein sequence database (pyensembl
or FASTA) and maps observed peptides to their source protein(s) with
N-terminal and C-terminal flanking context.

The index is built once per proteome and cached. Lookups are O(1) per
peptide via a dict-based inverted index mapping each unique k-mer to
the set of (protein_id, position) occurrences.

Typical usage::

    from hitlist.proteome import ProteomeIndex

    idx = ProteomeIndex.from_ensembl(release=112)
    hits = idx.map_peptides(["SLYNTVATL", "GILGFVFTL"], flank=5)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

try:
    from tqdm.auto import tqdm as _tqdm
except ImportError:
    _tqdm = None


@dataclass
class ProteinSource:
    """A source protein occurrence for a peptide."""

    protein_id: str
    gene_name: str
    gene_id: str
    position: int  # 0-based start in protein
    n_flank: str
    c_flank: str


@dataclass
class ProteomeIndex:
    """Inverted index mapping peptide sequences to source proteins.

    Attributes
    ----------
    proteins : dict[str, str]
        Mapping from protein_id to amino acid sequence.
    protein_meta : dict[str, dict]
        Mapping from protein_id to metadata (gene_name, gene_id).
    index : dict[str, list[tuple[str, int]]]
        Inverted index: peptide -> [(protein_id, position), ...].
    lengths : tuple[int, ...]
        Peptide lengths that were indexed.
    """

    proteins: dict[str, str] = field(repr=False)
    protein_meta: dict[str, dict] = field(repr=False)
    index: dict[str, list[tuple[str, int]]] = field(repr=False)
    lengths: tuple[int, ...]

    @classmethod
    def from_ensembl(
        cls,
        release: int = 112,
        lengths: tuple[int, ...] = (8, 9, 10, 11),
        biotype: str = "protein_coding",
        verbose: bool = True,
    ) -> ProteomeIndex:
        """Build index from Ensembl protein sequences.

        Parameters
        ----------
        release
            Ensembl release number.
        lengths
            Peptide lengths to index.
        biotype
            Gene biotype filter (default ``"protein_coding"``).
        verbose
            Print progress messages.

        Returns
        -------
        ProteomeIndex
        """
        from pyensembl import EnsemblRelease

        ensembl = EnsemblRelease(release)
        proteins: dict[str, str] = {}
        meta: dict[str, dict] = {}

        genes = list(ensembl.genes())
        gene_iter = (
            _tqdm(genes, desc="Loading proteins", leave=False) if (_tqdm and verbose) else genes
        )

        for gene in gene_iter:
            if gene.biotype != biotype:
                continue
            # Pick canonical (longest) transcript
            best_t = None
            best_len = 0
            for t in gene.transcripts:
                if t.biotype != "protein_coding":
                    continue
                try:
                    seq = t.protein_sequence
                except Exception:
                    continue
                if seq and len(seq) > best_len:
                    best_t = t
                    best_len = len(seq)
            if best_t is None:
                continue
            proteins[best_t.id] = best_t.protein_sequence
            meta[best_t.id] = {"gene_name": gene.name, "gene_id": gene.id}

        return cls._build(proteins, meta, lengths, verbose)

    @classmethod
    def from_fasta(
        cls,
        path: str | Path,
        lengths: tuple[int, ...] = (8, 9, 10, 11),
        gene_name: str = "",
        gene_id: str = "",
        verbose: bool = True,
    ) -> ProteomeIndex:
        """Build index from a FASTA file.

        Parameters
        ----------
        path
            Path to FASTA file.
        lengths
            Peptide lengths to index.
        gene_name, gene_id
            Default gene name/ID for all proteins (overridden by FASTA headers).
        verbose
            Print progress.

        Returns
        -------
        ProteomeIndex
        """
        proteins: dict[str, str] = {}
        meta: dict[str, dict] = {}
        current_id = ""
        current_seq: list[str] = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line.startswith(">"):
                    if current_id:
                        proteins[current_id] = "".join(current_seq)
                    header = line[1:].strip()
                    current_id = header.split()[0] if header else header
                    # Try to parse gene name from header (UniProt format: GN=GENENAME)
                    gn = gene_name
                    if "GN=" in header:
                        gn = header.split("GN=")[1].split()[0]
                    meta[current_id] = {"gene_name": gn, "gene_id": gene_id}
                    current_seq = []
                else:
                    current_seq.append(line)
        if current_id:
            proteins[current_id] = "".join(current_seq)
        return cls._build(proteins, meta, lengths, verbose)

    @classmethod
    def _build(
        cls,
        proteins: dict[str, str],
        meta: dict[str, dict],
        lengths: tuple[int, ...],
        verbose: bool,
    ) -> ProteomeIndex:
        """Build the inverted k-mer index."""
        index: dict[str, list[tuple[str, int]]] = {}
        items = list(proteins.items())
        prot_iter = (
            _tqdm(items, desc="Building index", leave=False) if (_tqdm and verbose) else items
        )

        for prot_id, seq in prot_iter:
            for k in lengths:
                for i in range(len(seq) - k + 1):
                    kmer = seq[i : i + k]
                    if kmer not in index:
                        index[kmer] = []
                    index[kmer].append((prot_id, i))

        if verbose:
            print(
                f"ProteomeIndex: {len(proteins):,} proteins, "
                f"{len(index):,} unique k-mers ({'/'.join(str(k) for k in lengths)}-mers)"
            )
        return cls(proteins=proteins, protein_meta=meta, index=index, lengths=lengths)

    def lookup(self, peptide: str) -> list[tuple[str, int]]:
        """Look up a peptide in the index.

        Returns list of (protein_id, position) tuples.
        """
        return self.index.get(peptide, [])

    def map_peptides(
        self,
        peptides: list[str] | set[str],
        flank: int = 5,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """Map peptides to source proteins with flanking context.

        Parameters
        ----------
        peptides
            Peptide sequences to map.
        flank
            Number of flanking residues to extract on each side (default 5).
        verbose
            Show progress bar.

        Returns
        -------
        pd.DataFrame
            One row per (peptide, source protein) occurrence. Columns:
            ``peptide``, ``protein_id``, ``gene_name``, ``gene_id``,
            ``position`` (0-based), ``n_flank``, ``c_flank``,
            ``n_sources`` (total source count for this peptide),
            ``unique_nflank`` (set if all sources have same n_flank, else ""),
            ``unique_cflank`` (same for c_flank).
        """
        peptide_list = list(set(peptides))
        pep_iter = (
            _tqdm(peptide_list, desc="Mapping peptides", leave=False)
            if (_tqdm and verbose)
            else peptide_list
        )

        rows: list[dict] = []
        for pep in pep_iter:
            hits = self.lookup(pep)
            for prot_id, pos in hits:
                seq = self.proteins[prot_id]
                m = self.protein_meta.get(prot_id, {})
                n_flank = seq[max(0, pos - flank) : pos]
                c_flank = seq[pos + len(pep) : pos + len(pep) + flank]
                rows.append(
                    {
                        "peptide": pep,
                        "protein_id": prot_id,
                        "gene_name": m.get("gene_name", ""),
                        "gene_id": m.get("gene_id", ""),
                        "position": pos,
                        "n_flank": n_flank,
                        "c_flank": c_flank,
                    }
                )

        if not rows:
            return pd.DataFrame(
                columns=[
                    "peptide",
                    "protein_id",
                    "gene_name",
                    "gene_id",
                    "position",
                    "n_flank",
                    "c_flank",
                    "n_sources",
                    "unique_nflank",
                    "unique_cflank",
                ]
            )

        df = pd.DataFrame(rows)

        # Compute per-peptide source counts and unique flank indicators
        source_counts = df.groupby("peptide")["protein_id"].nunique().rename("n_sources")
        df = df.merge(source_counts, on="peptide", how="left")

        # Unique flank: if all sources have the same flank, store it; else ""
        for side in ["n_flank", "c_flank"]:
            unique_col = f"unique_{side}"
            unique_map: dict[str, str] = {}
            for pep, group in df.groupby("peptide"):
                flanks = set(group[side])
                unique_map[str(pep)] = next(iter(flanks)) if len(flanks) == 1 else ""
            df[unique_col] = df["peptide"].map(unique_map)

        return df
