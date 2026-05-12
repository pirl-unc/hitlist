#!/usr/bin/env python3
"""Quick cold-vs-warm timing demo for the proteome-index disk cache (#246).

Builds a ProteomeIndex twice for the same FASTA — once cold (no cache),
once warm (in-memory cache cleared between calls so the disk cache is
forced to serve the second build).  Reports the speedup.

Usage::

    python scripts/demo_proteome_cache.py [FASTA_PATH] [LENGTH]

Defaults to ``~/.hitlist/proteomes/macaca_mulatta.fasta`` at length 9 —
medium-size proteome (~32 MB) that builds in ~30-40s cold and loads
in ~1-3s warm on SSD, so the speedup is unambiguous within ~1 min of
total runtime.

Single length keeps peak memory under a few GB so it runs cleanly on
laptop-class hardware (no OOM risk).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path


def main() -> int:
    fasta = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        Path.home() / ".hitlist/proteomes/macaca_mulatta.fasta"
    )
    length = int(sys.argv[2]) if len(sys.argv) > 2 else 9

    if not fasta.is_file():
        print(f"ERROR: FASTA not found: {fasta}", file=sys.stderr)
        print(
            "       Pass an existing path as the first arg, e.g.:\n"
            "       python scripts/demo_proteome_cache.py "
            "~/.hitlist/proteomes/plasmodium_falciparum.fasta 9",
            file=sys.stderr,
        )
        return 1

    from hitlist.proteome import (
        ProteomeIndex,
        clear_disk_cache,
        clear_fasta_index_cache,
    )

    print(f"FASTA:  {fasta}  ({fasta.stat().st_size / 1024**2:.1f} MB)")
    print(f"Length: {length}")
    print()

    # Clean slate so "cold" really is cold.
    clear_disk_cache()
    clear_fasta_index_cache()

    print("Cold build (no cache)...")
    t0 = time.perf_counter()
    idx_cold = ProteomeIndex.from_fasta(fasta, lengths=(length,), verbose=False)
    cold = time.perf_counter() - t0
    print(
        f"  {cold:6.2f}s  "
        f"({len(idx_cold.proteins):,} proteins, {len(idx_cold.index):,} k-mers)"
    )

    # Drop in-memory cache so the next call must use the disk cache.
    clear_fasta_index_cache()

    print("Warm load (disk cache hit)...")
    t0 = time.perf_counter()
    idx_warm = ProteomeIndex.from_fasta(fasta, lengths=(length,), verbose=False)
    warm = time.perf_counter() - t0
    print(f"  {warm:6.2f}s")
    print()

    speedup = cold / warm if warm > 0 else float("inf")
    print(f"Speedup: {speedup:.1f}x  ({cold:.1f}s → {warm:.1f}s)")
    print()

    # Round-trip integrity.
    assert idx_cold.proteins == idx_warm.proteins, "proteins mismatch"
    assert len(idx_cold.index) == len(idx_warm.index), "index size mismatch"
    print("Round-trip integrity: OK")

    if warm >= cold:
        print(
            f"\nWARNING: warm ({warm:.1f}s) was not faster than cold "
            f"({cold:.1f}s).  Cache may not have been hit.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
