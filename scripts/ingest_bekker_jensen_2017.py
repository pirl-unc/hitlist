#!/usr/bin/env python3
"""Rebuild the Bekker-Jensen 2017 (PMID 28591648, PXD004452) bulk
proteomics CSVs to be *comprehensive* over the paper's Figure 1b
design matrix — every experiment in ``peptides.txt`` that corresponds
to a HeLa (or 5-line panel) arm on the three axes:

  - fractionation depth (14 / 39 / 46 / 70)
  - digestion enzyme (Trypsin/P, Chymotrypsin, GluC, LysC)
  - enrichment (none vs. TiO2 phospho)

Emits two CSVs under ``hitlist/data/bulk_proteomics/``:

  - ``bekker_jensen_2017_peptides.csv.gz`` — one row per
    ``(peptide, cell_line, digestion_enzyme, n_fractions_in_run, enrichment, modifications)``,
    union across biological replicates (``n_replicates_detected`` records
    reproducibility inside the arm cheaply).
  - ``bekker_jensen_2017_protein_abundance.csv.gz`` — one row per
    ``(cell_line, gene_symbol, uniprot_acc, digestion_enzyme, n_fractions_in_run, enrichment)``,
    with ``log2_intensity`` derived by summing peptide intensities per
    protein per arm. proteinGroups.txt is available in the archive but
    the peptide-sum fallback is adequate and keeps the ingest self-
    contained to peptides.txt; ``abundance_percentile`` is recomputed
    PER arm because absolute log2 intensities are only comparable
    within a single ``(cell_line, enzyme, fracs, enrichment)`` run —
    different digests and fractionation depths ionize and sample the
    proteome differently so cross-arm percentile joins would be wrong.

Experiments we SKIP (present in the deposit, NOT part of Fig 1b):
  - ``*-Pandey``          reanalysis of Kim et al. 2014 draft human
                          proteome (PMID 24870542) — different source
                          study, not Bekker-Jensen's new data.
  - ``Colon-human-46fracs`` / ``Liver-human-46fracs`` / ``Prostate-Pt2/3-human-46fracs``
                          human tissue specimens used for Fig 5/6
                          coverage comparisons, not cell lines.
  - ``SY5Y-46fracs-E1/E2`` SH-SY5Y neuroblastoma — not in Fig 1b's
                          5-cell-line panel; would widen scope.

Range-request strategy: unchanged from the prior tryptic-only
version — one HTTP range read parses the ZIP64 central directory,
one more grabs the full compressed ``peptides.txt`` (~172 MB), which
we decompress locally. No full-archive download (21.7 GB).

Modification handling: ``peptides.txt`` does NOT carry a single
``Modifications`` column (that lives in evidence.txt /
modificationSpecificPeptides.txt). Instead it has three per-mod site-ID
columns — ``Phospho (STY) site IDs``, ``Oxidation (M) site IDs``,
``Deamidation (NQ) site IDs``. Any one of them being non-empty/non-zero
for a row means that modification was observed on that peptide
somewhere in the deposit. We build a semicolon-joined set — e.g.
``"Phospho (STY);Oxidation (M)"`` or ``"Unmodified"``. This is a
peptide-level attribute (one row per unique Sequence) so it cannot be
sliced per-experiment from peptides.txt alone.

Usage
-----
    python scripts/ingest_bekker_jensen_2017.py

Outputs (overwrites existing files):
    hitlist/data/bulk_proteomics/bekker_jensen_2017_peptides.csv.gz
    hitlist/data/bulk_proteomics/bekker_jensen_2017_protein_abundance.csv.gz
"""

from __future__ import annotations

import csv
import gzip
import io
import math
import re
import struct
import sys
import urllib.request
import zipfile
import zlib
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

SEARCH_RESULTS_URL = (
    "https://ftp.pride.ebi.ac.uk/pride/data/archive/2017/06/PXD004452/SearchResults.zip"
)

UA = "hitlist-bekker-jensen-ingest/1.0 (+https://github.com/pirl-unc/hitlist)"

# Canonical enzyme strings — MUST match sources.yaml so joins stay consistent.
TRYPTIC_ENZYME = "Trypsin/P (cleaves K/R except before P)"
CHYMO_ENZYME = "Chymotrypsin"
GLUC_ENZYME = "GluC"
LYSC_ENZYME = "LysC"

# Known 5-cell-line panel explicitly called out in Fig 1b.
FIG1B_CELL_LINES = {"HeLa", "A549", "HCT116", "HEK293", "MCF7"}

# Experiments we intentionally SKIP (documented in the module docstring).
# Matched by prefix because MaxQuant experiment names have reproducible
# prefixes for each sample class.
SKIP_PREFIXES = (
    "Colon-",
    "Liver-",
    "Prostate-",
    "SY5Y-",
)

BASE_COLS = [
    "Sequence",
    "Proteins",
    "Leading razor protein",
    "Start position",
    "End position",
    "Length",
    "Gene names",
    "Reverse",
    "Potential contaminant",
    # Modification signal columns — any non-empty/non-zero value here
    # means the named modification was observed on this peptide at
    # least once in the deposit.
    "Phospho (STY) site IDs",
    "Oxidation (M) site IDs",
    "Deamidation (NQ) site IDs",
]


# ---------------------------------------------------------------------------
# Experiment name parser
# ---------------------------------------------------------------------------
#
# Observed experiment names in PXD004452/peptides.txt (Fig 1b-relevant
# subset):
#
#   HeLa-46fracs-IT-E1/E2   tryptic, HeLa, 46-frac, reps (IT = ion trap
#                           detection? — kept only for naming fidelity)
#   A549/HCT116/HEK293/MCF7-46fracs-E1/E2   tryptic, 46-frac, reps
#   Tryp-14Frac-A/B/C       tryptic, HeLa, 14-frac, three replicates
#   Tryp-39fracs            tryptic, HeLa, 39-frac, single run
#   Tryp-46fracs            tryptic, HeLa, 46-frac, single run
#   Tryp-70fracs            tryptic, HeLa, 70-frac, single run
#   Tryp-Phos-pH10          tryptic, HeLa, 12-frac, TiO2 phospho, pH10
#   Tryp-Phos-pH8           tryptic, HeLa, 12-frac, TiO2 phospho, pH8
#   Phospho-50fracs         tryptic, HeLa, 50-frac, TiO2 phospho
#   Chymo-39fracs           HeLa, Chymotrypsin, 39-frac
#   GluC-39fracs            HeLa, GluC, 39-frac
#   LysC-39fracs            HeLa, LysC, 39-frac
#
# n_fractions_in_run and replicate come from summary.txt but we also
# encode the fraction count directly in the MaxQuant experiment name
# (that's how the authors laid it out), so we can parse it off the
# experiment string without re-reading summary.txt at ingest time.
#
# This dict is checked first; everything else falls through to the
# regex parser. If we land in neither branch we FAIL LOUDLY.

_ENZYME_TOKENS = {
    "Tryp": TRYPTIC_ENZYME,
    "Chymo": CHYMO_ENZYME,
    "GluC": GLUC_ENZYME,
    "LysC": LYSC_ENZYME,
    "Phospho": TRYPTIC_ENZYME,  # Phospho-50fracs is tryptic + TiO2
}


# High-pH reverse-phase SPE default (pH 10.0). Bekker-Jensen 2017 Methods
# use pH 10 for every high-pH SPE fractionation except the two explicit
# Tryp-Phos pH-comparison experiments. See sources.yaml for rationale and
# the per-source default discussion.
DEFAULT_FRACTIONATION_PH = 10.0


def _parse_experiment(name: str) -> dict | None:
    """Return axis dict for an experiment name or None if it should be skipped.

    Raises ValueError on unparseable names (explicit failure, not silent drop).

    Output keys: ``cell_line``, ``digestion_enzyme``, ``n_fractions_in_run``,
    ``enrichment``, ``fractionation_ph``, ``replicate``.
    """
    for pref in SKIP_PREFIXES:
        if name.startswith(pref):
            return None
    if "-Pandey" in name:
        return None

    # Case 1: HeLa-46fracs-IT-E1/E2 (explicit HeLa prefix, the "IT" is
    # an instrument/method tag from MaxQuant setup; kept in the raw
    # experiment id for traceability but does NOT affect axes).
    m = re.fullmatch(r"HeLa-(\d+)fracs-IT-E(\d+)", name)
    if m:
        return {
            "cell_line": "HeLa",
            "digestion_enzyme": TRYPTIC_ENZYME,
            "n_fractions_in_run": int(m.group(1)),
            "enrichment": "none",
            "fractionation_ph": DEFAULT_FRACTIONATION_PH,
            "replicate": f"E{m.group(2)}",
        }

    # Case 2: <CellLine>-46fracs-E1/E2 (A549/HCT116/HEK293/MCF7).
    m = re.fullmatch(r"(A549|HCT116|HEK293|MCF7)-(\d+)fracs-E(\d+)", name)
    if m:
        cell = m.group(1)
        if cell not in FIG1B_CELL_LINES:
            raise ValueError(f"Unexpected cell-line prefix in experiment {name!r}")
        return {
            "cell_line": cell,
            "digestion_enzyme": TRYPTIC_ENZYME,
            "n_fractions_in_run": int(m.group(2)),
            "enrichment": "none",
            "fractionation_ph": DEFAULT_FRACTIONATION_PH,
            "replicate": f"E{m.group(3)}",
        }

    # Case 3: Tryp-14Frac-A/B/C — three technical replicates at 14 frac.
    # Note: ``Frac`` (capitalized, no 's') is a separate naming pattern
    # from ``fracs`` used elsewhere. Treat letter replicates as E1/E2/E3
    # by alphabetical order for consistency with the rest of the dataset.
    m = re.fullmatch(r"Tryp-(\d+)Frac-([A-Z])", name)
    if m:
        letter = m.group(2)
        rep = f"E{ord(letter) - ord('A') + 1}"  # A->E1, B->E2, C->E3
        return {
            "cell_line": "HeLa",
            "digestion_enzyme": TRYPTIC_ENZYME,
            "n_fractions_in_run": int(m.group(1)),
            "enrichment": "none",
            "fractionation_ph": DEFAULT_FRACTIONATION_PH,
            "replicate": rep,
        }

    # Case 4: Tryp-Phos-pH8 / Tryp-Phos-pH10 — HeLa TiO2 phospho-enrichment
    # runs at two different high-pH SPE fractionation buffers. These are
    # **two complementary separations**, NOT biological replicates of the
    # same condition — peptides elute differently at pH 8 vs pH 10 because
    # of side-chain charge shifts. As of v1.14.1 (#98) we treat them as
    # distinct arms (separate rows in the output) via the
    # ``fractionation_ph`` axis, each with replicate=E1.
    m = re.fullmatch(r"Tryp-Phos-pH(\d+)", name)
    if m:
        return {
            "cell_line": "HeLa",
            "digestion_enzyme": TRYPTIC_ENZYME,
            "n_fractions_in_run": 12,
            "enrichment": "TiO2",
            "fractionation_ph": float(m.group(1)),
            "replicate": "E1",
        }

    # Case 5: Phospho-50fracs — HeLa 50-fraction tryptic TiO2 phospho.
    if name == "Phospho-50fracs":
        return {
            "cell_line": "HeLa",
            "digestion_enzyme": TRYPTIC_ENZYME,
            "n_fractions_in_run": 50,
            "enrichment": "TiO2",
            "fractionation_ph": DEFAULT_FRACTIONATION_PH,
            "replicate": "E1",
        }

    # Case 6: <Enzyme>-<N>fracs  (Tryp/Chymo/GluC/LysC on HeLa).
    m = re.fullmatch(r"(Tryp|Chymo|GluC|LysC)-(\d+)fracs", name)
    if m:
        enz_token = m.group(1)
        enzyme = _ENZYME_TOKENS.get(enz_token)
        if enzyme is None:
            raise ValueError(f"Unknown enzyme token {enz_token!r} in {name!r}")
        return {
            "cell_line": "HeLa",
            "digestion_enzyme": enzyme,
            "n_fractions_in_run": int(m.group(2)),
            "enrichment": "none",
            "fractionation_ph": DEFAULT_FRACTIONATION_PH,
            "replicate": "E1",
        }

    raise ValueError(
        f"Unrecognised experiment name {name!r} -- add a case to _parse_experiment "
        f"or extend SKIP_PREFIXES."
    )


# ---------------------------------------------------------------------------
# HTTP seekable file -- unchanged from the prior script.
# ---------------------------------------------------------------------------
class HttpSeekableFile:
    """File-like wrapper that reads an HTTP URL via Range headers."""

    def __init__(self, url: str, user_agent: str = UA) -> None:
        self.url = url
        self.user_agent = user_agent
        req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": user_agent})
        with urllib.request.urlopen(req) as resp:
            size = resp.headers.get("Content-Length")
            if not size:
                raise RuntimeError(f"{url} did not return Content-Length header")
            accept_ranges = resp.headers.get("Accept-Ranges", "")
            if "bytes" not in accept_ranges.lower():
                raise RuntimeError(
                    f"{url} advertises Accept-Ranges={accept_ranges!r}; "
                    "range-request strategy won't work."
                )
        self.size = int(size)
        self.pos = 0

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = 0) -> int:
        if whence == 0:
            self.pos = offset
        elif whence == 1:
            self.pos += offset
        elif whence == 2:
            self.pos = self.size + offset
        else:
            raise ValueError(f"bad whence {whence}")
        return self.pos

    def read(self, n: int = -1) -> bytes:
        if n == -1 or n is None:
            end = self.size - 1
        else:
            end = min(self.pos + n - 1, self.size - 1)
        if end < self.pos:
            return b""
        req = urllib.request.Request(
            self.url,
            headers={
                "User-Agent": self.user_agent,
                "Range": f"bytes={self.pos}-{end}",
            },
        )
        with urllib.request.urlopen(req) as resp:
            data = resp.read()
        self.pos += len(data)
        return data


def _resolve_uniprot_acc(leading_razor: str, proteins: str) -> str:
    return leading_razor or (proteins.split(";", 1)[0] if proteins else "")


def _fetch_peptides_txt_stream(verbose: bool = True) -> io.TextIOWrapper:
    """Fetch peptides.txt bytes in one range read and return a text stream."""
    if verbose:
        print(f"Opening {SEARCH_RESULTS_URL} via HTTP range reads...", flush=True)
    remote = HttpSeekableFile(SEARCH_RESULTS_URL)
    if verbose:
        print(f"  archive size: {remote.size:,} bytes ({remote.size / 1e9:.2f} GB)", flush=True)

    zf = zipfile.ZipFile(remote)
    if "peptides.txt" not in zf.namelist():
        raise RuntimeError("peptides.txt not found in SearchResults.zip")
    info = zf.getinfo("peptides.txt")
    if verbose:
        print(
            f"  peptides.txt: uncompressed={info.file_size:,} "
            f"compressed={info.compress_size:,}",
            flush=True,
        )

    local_hdr_len = 30
    remote.seek(info.header_offset)
    lfh = remote.read(local_hdr_len)
    if lfh[:4] != b"PK\x03\x04":
        raise RuntimeError(
            f"expected local file header signature PK\\x03\\x04 at offset "
            f"{info.header_offset}, got {lfh[:4]!r}"
        )
    (name_len, extra_len) = struct.unpack("<HH", lfh[26:30])
    compressed_start = info.header_offset + local_hdr_len + name_len + extra_len
    compressed_end = compressed_start + info.compress_size - 1

    if verbose:
        print(
            f"  fetching compressed bytes [{compressed_start:,}..{compressed_end:,}] "
            f"({info.compress_size / 1e6:.1f} MB) in one request...",
            flush=True,
        )
    req = urllib.request.Request(
        SEARCH_RESULTS_URL,
        headers={"User-Agent": UA, "Range": f"bytes={compressed_start}-{compressed_end}"},
    )
    with urllib.request.urlopen(req) as resp:
        compressed = resp.read()
    if len(compressed) != info.compress_size:
        raise RuntimeError(
            f"expected {info.compress_size} compressed bytes, got {len(compressed)}"
        )
    if verbose:
        print(f"  decompressing {len(compressed):,} bytes...", flush=True)
    raw = zlib.decompress(compressed, -zlib.MAX_WBITS)
    if len(raw) != info.file_size:
        raise RuntimeError(f"expected {info.file_size} uncompressed bytes, got {len(raw)}")
    if verbose:
        print(f"  decompressed: {len(raw):,} bytes", flush=True)
    return io.TextIOWrapper(io.BytesIO(raw), encoding="latin-1", newline="")


# ---------------------------------------------------------------------------
# Modification string construction
# ---------------------------------------------------------------------------
# Peptides.txt has three independent modification-tracking columns. Each
# stores semicolon-joined site IDs pointing into the three per-mod site
# tables (Phospho (STY)Sites.txt etc.). An empty string or "0" means no
# site of that mod was observed on this peptide across any experiment
# in the deposit. We collapse the site IDs to a binary signal per mod.

_MOD_COL_TO_LABEL = {
    "Phospho (STY) site IDs": "Phospho (STY)",
    "Oxidation (M) site IDs": "Oxidation (M)",
    "Deamidation (NQ) site IDs": "Deamidation (NQ)",
}


def _mod_string(row: list[str], col_idx: dict[str, int]) -> str:
    labels = []
    for col, label in _MOD_COL_TO_LABEL.items():
        v = row[col_idx[col]].strip()
        if v and v != "0":
            labels.append(label)
    return ";".join(labels) if labels else "Unmodified"


# ---------------------------------------------------------------------------
# Main extraction
# ---------------------------------------------------------------------------
def extract(verbose: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stream peptides.txt and return (peptides_df, protein_abundance_df).

    peptides_df columns:
        peptide, cell_line, uniprot_acc, gene_symbol, length,
        start_position, end_position, digestion_enzyme,
        n_fractions_in_run, enrichment, modifications,
        n_replicates_detected, source, reference

    protein_abundance_df columns:
        cell_line, gene_symbol, uniprot_acc, digestion_enzyme,
        n_fractions_in_run, enrichment, n_peptides, log2_intensity,
        abundance_percentile, source, reference
    """
    csv.field_size_limit(sys.maxsize)

    text = _fetch_peptides_txt_stream(verbose=verbose)
    reader = csv.reader(text, delimiter="\t")
    header = [c.rstrip("\r") for c in next(reader)]
    col_idx = {name: i for i, name in enumerate(header)}

    missing_base = [c for c in BASE_COLS if c not in col_idx]
    if missing_base:
        raise RuntimeError(f"peptides.txt missing required columns: {missing_base}")

    # Authoritative list of experiments from the header.
    all_experiments = [c[len("Intensity ") :] for c in header if c.startswith("Intensity ")]
    # 'Intensity' on its own (the grand-total column) is not an experiment.
    all_experiments = [e for e in all_experiments if e]

    if verbose:
        print(f"\n=== {len(all_experiments)} experiments discovered in peptides.txt ===")
        for e in sorted(all_experiments):
            print(f"  {e}")

    # Parse each experiment into axes (or mark as skipped).
    experiment_axes: dict[str, dict] = {}
    skipped_experiments: list[str] = []
    for exp in all_experiments:
        axes = _parse_experiment(exp)
        if axes is None:
            skipped_experiments.append(exp)
        else:
            experiment_axes[exp] = axes

    if verbose:
        print(f"\n=== Parsed {len(experiment_axes)} experiments into Fig 1b axes ===")
        for exp, axes in sorted(experiment_axes.items()):
            print(
                f"  {exp:30s} -> cell={axes['cell_line']:6s} "
                f"enz={axes['digestion_enzyme'][:12]:12s} "
                f"fracs={axes['n_fractions_in_run']:3d} "
                f"enrich={axes['enrichment']:5s} "
                f"pH={axes['fractionation_ph']:4.1f} rep={axes['replicate']}"
            )
        print(f"\n=== Skipped {len(skipped_experiments)} out-of-scope experiments ===")
        for exp in sorted(skipped_experiments):
            print(f"  {exp}  (documented skip: Pandey reanalysis / tissue / SY5Y)")

    # Intensity column indices for experiments we keep.
    intensity_cols: dict[str, int] = {}
    for exp in experiment_axes:
        key = f"Intensity {exp}"
        if key not in col_idx:
            raise RuntimeError(f"peptides.txt has no column {key!r}")
        intensity_cols[exp] = col_idx[key]

    i_seq = col_idx["Sequence"]
    i_proteins = col_idx["Proteins"]
    i_razor = col_idx["Leading razor protein"]
    i_start = col_idx["Start position"]
    i_end = col_idx["End position"]
    i_len = col_idx["Length"]
    i_gene = col_idx["Gene names"]
    i_reverse = col_idx["Reverse"]
    i_contam = col_idx["Potential contaminant"]

    # --- aggregation state ----------------------------------------------------
    # peptide-level: key = (sequence, cell_line, enzyme, n_fracs, enrichment)
    # Values carry the row dict + a set of replicates seen + cumulative intensity.
    pep_rows: dict[tuple, dict] = {}

    # protein-level intensity aggregator. We sum raw intensities across all
    # peptides and replicates within an arm, then log2 at the end.
    prot_intensity: dict[tuple, float] = defaultdict(float)
    # set of distinct peptide sequences per protein-arm (for n_peptides)
    prot_peptides: dict[tuple, set[str]] = defaultdict(set)
    # remember gene_symbol for each uniprot_acc (first non-empty observed)
    uniprot_to_gene: dict[str, str] = {}

    scanned = 0
    kept_rows = 0
    skipped_reverse = 0
    skipped_contam = 0
    skipped_no_razor = 0
    skipped_bad_pos = 0

    for row in reader:
        scanned += 1
        if verbose and scanned % 250_000 == 0:
            print(
                f"  scanned {scanned:,} peptides.txt rows... "
                f"(peptide-arm keys so far: {len(pep_rows):,})",
                flush=True,
            )

        if len(row) < len(header):
            row = row + [""] * (len(header) - len(row))

        if row[i_reverse] == "+":
            skipped_reverse += 1
            continue
        if row[i_contam] == "+":
            skipped_contam += 1
            continue

        razor = row[i_razor]
        if not razor or razor.startswith("REV__") or razor.startswith("CON__"):
            skipped_no_razor += 1
            continue

        try:
            start = int(row[i_start])
            end = int(row[i_end])
        except (ValueError, IndexError):
            skipped_bad_pos += 1
            continue

        seq = row[i_seq]
        length = int(row[i_len]) if row[i_len] else len(seq)
        gene = row[i_gene] or ""
        # Gene names column is semicolon-joined; take first for consistency
        # with the prior packaged CSV.
        gene_primary = gene.split(";", 1)[0] if gene else ""
        uniprot = _resolve_uniprot_acc(razor, row[i_proteins])
        if uniprot and gene_primary and uniprot not in uniprot_to_gene:
            uniprot_to_gene[uniprot] = gene_primary

        mod_str = _mod_string(row, col_idx)

        # Per-row iterate experiments with non-zero intensity.
        for exp, idx in intensity_cols.items():
            val = row[idx]
            if not val:
                continue
            try:
                intensity = float(val)
            except ValueError:
                continue
            if intensity <= 0:
                continue

            axes = experiment_axes[exp]
            key = (
                seq,
                axes["cell_line"],
                axes["digestion_enzyme"],
                axes["n_fractions_in_run"],
                axes["enrichment"],
                axes["fractionation_ph"],
            )

            if key not in pep_rows:
                pep_rows[key] = {
                    "peptide": seq,
                    "cell_line": axes["cell_line"],
                    "uniprot_acc": uniprot,
                    "gene_symbol": gene_primary if gene_primary else None,
                    "length": length,
                    "start_position": start,
                    "end_position": end,
                    "digestion_enzyme": axes["digestion_enzyme"],
                    "n_fractions_in_run": axes["n_fractions_in_run"],
                    "enrichment": axes["enrichment"],
                    "fractionation_ph": axes["fractionation_ph"],
                    "modifications": mod_str,
                    "_replicates": set(),
                    "source": "Bekker-Jensen_2017",
                    "reference": "PMID:28591648",
                }
                kept_rows += 1
            pep_rows[key]["_replicates"].add(axes["replicate"])

            # Protein-level aggregation — same arm granularity as peptide
            # rows, which now includes fractionation_ph so the two
            # Tryp-Phos arms (pH 8 vs pH 10) stay distinct.
            prot_key = (
                axes["cell_line"],
                axes["digestion_enzyme"],
                axes["n_fractions_in_run"],
                axes["enrichment"],
                axes["fractionation_ph"],
                uniprot,
            )
            prot_intensity[prot_key] += intensity
            prot_peptides[prot_key].add(seq)

    if verbose:
        print(
            f"\nScan complete: {scanned:,} rows scanned, "
            f"{kept_rows:,} output peptide-arm rows\n"
            f"  skipped reverse: {skipped_reverse:,}\n"
            f"  skipped contaminant: {skipped_contam:,}\n"
            f"  skipped no-razor/decoy: {skipped_no_razor:,}\n"
            f"  skipped missing position: {skipped_bad_pos:,}",
            flush=True,
        )

    # Materialize peptide frame (convert replicate set -> count).
    out_rows = []
    for v in pep_rows.values():
        v["n_replicates_detected"] = len(v.pop("_replicates"))
        out_rows.append(v)
    pep_df = pd.DataFrame.from_records(out_rows)

    pep_df = pep_df[
        [
            "peptide",
            "cell_line",
            "uniprot_acc",
            "gene_symbol",
            "length",
            "start_position",
            "end_position",
            "digestion_enzyme",
            "n_fractions_in_run",
            "enrichment",
            "fractionation_ph",
            "modifications",
            "n_replicates_detected",
            "source",
            "reference",
        ]
    ]

    # ---- protein abundance frame --------------------------------------------
    prot_rows = []
    for (cell, enzyme, fracs, enrich, ph, uniprot), intensity in prot_intensity.items():
        if intensity <= 0 or uniprot == "":
            continue
        prot_rows.append(
            {
                "cell_line": cell,
                "gene_symbol": uniprot_to_gene.get(uniprot, "") or None,
                "uniprot_acc": uniprot,
                "digestion_enzyme": enzyme,
                "n_fractions_in_run": fracs,
                "enrichment": enrich,
                "fractionation_ph": ph,
                "n_peptides": len(
                    prot_peptides[(cell, enzyme, fracs, enrich, ph, uniprot)]
                ),
                "log2_intensity": math.log2(intensity),
                "source": "Bekker-Jensen_2017",
                "reference": "PMID:28591648",
            }
        )

    prot_df = pd.DataFrame.from_records(prot_rows)
    # Recompute abundance_percentile PER arm — absolute log2 intensities
    # are only comparable within one
    # (cell_line, enzyme, fracs, enrichment, fractionation_ph) group.
    # Different digests ionize different sets of peptides, different
    # fractionation depths have different dynamic range, and different
    # pH values fractionate to a different peptide population. Pooling
    # across any of these axes for the percentile would be wrong.
    grp_cols = [
        "cell_line",
        "digestion_enzyme",
        "n_fractions_in_run",
        "enrichment",
        "fractionation_ph",
    ]
    prot_df["abundance_percentile"] = prot_df.groupby(grp_cols)["log2_intensity"].rank(pct=True)
    prot_df = prot_df[
        [
            "cell_line",
            "gene_symbol",
            "uniprot_acc",
            "digestion_enzyme",
            "n_fractions_in_run",
            "enrichment",
            "fractionation_ph",
            "n_peptides",
            "log2_intensity",
            "abundance_percentile",
            "source",
            "reference",
        ]
    ]

    return pep_df, prot_df


def print_sanity_checks(pep_df: pd.DataFrame, prot_df: pd.DataFrame) -> None:
    print("\n=== Peptide sanity checks ===")
    print(f"  total peptide-arm rows: {len(pep_df):,}")
    print(f"  unique peptides: {pep_df['peptide'].nunique():,}")

    print("\n-- Row counts by (enzyme, fracs, enrichment, pH) --")
    grp = (
        pep_df.groupby(
            ["digestion_enzyme", "n_fractions_in_run", "enrichment", "fractionation_ph"],
            as_index=False,
        )
        .size()
        .sort_values(
            ["digestion_enzyme", "n_fractions_in_run", "enrichment", "fractionation_ph"]
        )
    )
    for _, r in grp.iterrows():
        print(
            f"  {r['digestion_enzyme'][:25]:25s} fracs={r['n_fractions_in_run']:3d} "
            f"enrich={r['enrichment']:5s} pH={r['fractionation_ph']:4.1f}  "
            f"rows={r['size']:,}"
        )

    print("\n-- Row counts by cell_line --")
    print(pep_df.groupby("cell_line").size().to_string())

    print("\n-- Enrichment counts --")
    print(pep_df["enrichment"].value_counts().to_string())

    print("\n-- Phospho modification by enrichment --")
    for enrich in sorted(pep_df["enrichment"].unique()):
        sub = pep_df[pep_df["enrichment"] == enrich]
        pct = sub["modifications"].str.contains("Phospho").mean() * 100
        print(f"  enrichment={enrich}: {pct:.1f}% rows carry phospho modification")

    print("\n-- C-term distribution per enzyme --")
    for enzyme in sorted(pep_df["digestion_enzyme"].unique()):
        sub = pep_df[pep_df["digestion_enzyme"] == enzyme]
        cterm = Counter(sub["peptide"].str[-1])
        total = sum(cterm.values())
        top = cterm.most_common(6)
        top_str = ", ".join(f"{aa}={c / total:.1%}" for aa, c in top)
        print(f"  {enzyme[:30]:30s}  C-term top-6: {top_str}")

    print("\n-- n_replicates_detected distribution --")
    print(pep_df["n_replicates_detected"].value_counts().sort_index().to_string())

    print("\n=== Protein abundance sanity checks ===")
    print(f"  total protein-arm rows: {len(prot_df):,}")
    print(f"  unique uniprots: {prot_df['uniprot_acc'].nunique():,}")
    print("\n-- Row counts by (enzyme, fracs, enrichment, pH) --")
    grp = (
        prot_df.groupby(
            [
                "cell_line",
                "digestion_enzyme",
                "n_fractions_in_run",
                "enrichment",
                "fractionation_ph",
            ],
            as_index=False,
        )
        .size()
        .sort_values(
            [
                "cell_line",
                "digestion_enzyme",
                "n_fractions_in_run",
                "enrichment",
                "fractionation_ph",
            ]
        )
    )
    for _, r in grp.iterrows():
        print(
            f"  {r['cell_line']:6s} {r['digestion_enzyme'][:25]:25s} "
            f"fracs={r['n_fractions_in_run']:3d} enrich={r['enrichment']:5s} "
            f"pH={r['fractionation_ph']:4.1f}  proteins={r['size']:,}"
        )


def main() -> int:
    base = Path(__file__).resolve().parent.parent / "hitlist" / "data" / "bulk_proteomics"
    pep_out = base / "bekker_jensen_2017_peptides.csv.gz"
    prot_out = base / "bekker_jensen_2017_protein_abundance.csv.gz"
    print(f"Peptide output:  {pep_out}")
    print(f"Protein output:  {prot_out}")

    pep_df, prot_df = extract(verbose=True)
    print(f"\nPeptide frame:  {pep_df.shape[0]:,} rows x {pep_df.shape[1]} cols")
    print(f"Protein frame:  {prot_df.shape[0]:,} rows x {prot_df.shape[1]} cols")

    # Deterministic ordering.
    pep_df = pep_df.sort_values(
        by=[
            "cell_line",
            "digestion_enzyme",
            "n_fractions_in_run",
            "enrichment",
            "fractionation_ph",
            "uniprot_acc",
            "start_position",
            "peptide",
        ],
        kind="stable",
    ).reset_index(drop=True)
    prot_df = prot_df.sort_values(
        by=[
            "cell_line",
            "digestion_enzyme",
            "n_fractions_in_run",
            "enrichment",
            "fractionation_ph",
            "uniprot_acc",
        ],
        kind="stable",
    ).reset_index(drop=True)

    print_sanity_checks(pep_df, prot_df)

    base.mkdir(parents=True, exist_ok=True)
    print(f"\nWriting {pep_out} ...", flush=True)
    with gzip.open(pep_out, "wt", encoding="utf-8", newline="") as f:
        pep_df.to_csv(f, index=False)
    print(f"  wrote {len(pep_df):,} rows, {pep_out.stat().st_size / 1e6:.1f} MB compressed")

    print(f"\nWriting {prot_out} ...", flush=True)
    with gzip.open(prot_out, "wt", encoding="utf-8", newline="") as f:
        prot_df.to_csv(f, index=False)
    print(f"  wrote {len(prot_df):,} rows, {prot_out.stat().st_size / 1e6:.1f} MB compressed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
