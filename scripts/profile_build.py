#!/usr/bin/env python3
"""Profile the hitlist parquet build pipeline (issue #176).

Runs ``build_observations(force=True)`` end-to-end under cProfile and
captures:

* wall time + peak resident-set size
* stage-by-stage wall-time breakdown (intercepts the builder's
  ``print()`` checkpoints)
* top-N cumulative-time functions

Usage::

    python scripts/profile_build.py [--no-mappings] [--profile-out PATH]

The cProfile instrumentation adds ~10-30% overhead, so absolute wall
times are slightly inflated.  Per-stage *relative* timing is accurate.
Open the binary with snakeviz for an interactive flame graph::

    snakeviz /tmp/hitlist_build.prof
"""

from __future__ import annotations

import argparse
import builtins
import cProfile
import pstats
import resource
import sys
import time


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--no-mappings",
        action="store_true",
        help="Skip peptide_mappings.parquet build",
    )
    ap.add_argument(
        "--profile-out",
        default="/tmp/hitlist_build.prof",
        help="Where to write the cProfile binary (default: /tmp/hitlist_build.prof)",
    )
    ap.add_argument(
        "--top",
        type=int,
        default=30,
        help="How many hot functions to print (default: 30)",
    )
    args = ap.parse_args()

    # Intercept print() to record stage timestamps.  The builder emits
    # human-readable progress lines at each major stage boundary; we use
    # those as our stage markers without modifying the builder itself.
    stage_log: list[tuple[float, str]] = []
    real_print = builtins.print
    t0 = time.perf_counter()

    def timed_print(*a, **kw):
        ts = time.perf_counter() - t0
        msg = " ".join(str(x) for x in a)
        stage_log.append((ts, msg))
        # Also echo to terminal with a timestamp prefix for live monitoring.
        real_print(f"[{ts:7.1f}s]", *a, **kw)

    builtins.print = timed_print  # noqa: A001

    # Defer hitlist import until after wrap so any module-init prints get caught.
    from hitlist.builder import build_observations

    profiler = cProfile.Profile()
    real_print("=" * 72)
    real_print(f"Profiling build_observations(force=True, build_mappings={not args.no_mappings})")
    real_print("=" * 72)

    wall_start = time.perf_counter()
    profiler.enable()
    try:
        build_observations(force=True, build_mappings=not args.no_mappings)
    finally:
        profiler.disable()
        builtins.print = real_print  # noqa: A001
    wall = time.perf_counter() - wall_start

    # Peak resident set size.  macOS reports bytes; Linux reports KiB.
    ru = resource.getrusage(resource.RUSAGE_SELF)
    if sys.platform == "darwin":
        peak_gb = ru.ru_maxrss / (1024**3)
    else:
        peak_gb = ru.ru_maxrss / (1024**2)

    real_print("\n" + "=" * 72)
    real_print("BUILD PROFILE SUMMARY")
    real_print("=" * 72)
    real_print(f"Wall time:     {wall:7.1f}s  ({wall / 60:.1f} min, includes ~15-30% cProfile overhead)")
    real_print(f"Peak RSS:      {peak_gb:7.2f} GB")

    # Stage breakdown: diff successive stage timestamps.
    # Filter to "interesting" stages — gaps >= 1s OR header lines ending in "...".
    real_print("\n── Stage Wall-Time Breakdown (>=1s gaps) ──")
    for i, (ts, msg) in enumerate(stage_log):
        next_ts = stage_log[i + 1][0] if i + 1 < len(stage_log) else wall
        delta = next_ts - ts
        if delta >= 1.0 or msg.rstrip().endswith("..."):
            short = msg.strip()[:80]
            real_print(f"  [{ts:7.1f}s]  +{delta:7.1f}s   {short}")

    # Top hot functions by cumulative time.
    profiler.dump_stats(args.profile_out)
    real_print(f"\n(binary profile written to {args.profile_out})")
    real_print(f"\n── Top {args.top} Functions by Cumulative Time ──")
    stats = pstats.Stats(profiler).sort_stats("cumulative")
    stats.print_stats(args.top)

    real_print(f"\n── Top {args.top} Functions by Internal (Self) Time ──")
    stats = pstats.Stats(profiler).sort_stats("tottime")
    stats.print_stats(args.top)


if __name__ == "__main__":
    main()
