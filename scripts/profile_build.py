#!/usr/bin/env python3
"""Measure a forced build; optionally collect a separate cProfile diagnostic.

Use unprofiled repetitions for speed comparisons. Stage timings are inclusive;
CPU includes this process and children that have exited and been waited for.
RSS is the process high-water mark, not summed concurrent worker memory.
"""

from __future__ import annotations

import argparse
import contextlib
import cProfile
import functools
import importlib
import importlib.metadata
import json
import platform
import pstats
import resource
import sys
import time
from pathlib import Path
from unittest.mock import patch

STAGES = (
    ("hitlist.scanner", "scan"),
    ("hitlist.supplement", "scan_supplementary"),
    ("hitlist.mappings", "build_peptide_mappings"),
    ("hitlist.builder", "build_bulk_proteomics"),
    ("hitlist.builder", "build_line_expression"),
    ("hitlist.provenance", "ContributorCollector.write"),
    ("hitlist.builder", "atomic_write_parquet"),
)


def usage():
    own = resource.getrusage(resource.RUSAGE_SELF)
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    return {
        "wall_seconds": time.perf_counter(),
        "process_cpu_seconds": own.ru_utime + own.ru_stime,
        "waited_children_cpu_seconds": children.ru_utime + children.ru_stime,
        "process_peak_rss_bytes": own.ru_maxrss * (1 if sys.platform == "darwin" else 1024),
        "waited_child_peak_rss_bytes": children.ru_maxrss
        * (1 if sys.platform == "darwin" else 1024),
    }


def elapsed(before):
    after = usage()
    return {
        key: value - before[key] if key.endswith("seconds") else value
        for key, value in after.items()
    }


@contextlib.contextmanager
def measured_stages(events):
    def instrument(function, name):
        @functools.wraps(function)
        def measured(*args, **kwargs):
            before = usage()
            event = {"stage": name, "status": "failed"}
            events.append(event)
            try:
                result = function(*args, **kwargs)
                event["status"] = "success"
                return result
            finally:
                event.update(elapsed(before))
                print("BUILD_STAGE " + json.dumps(event, sort_keys=True), flush=True)

        return measured

    with contextlib.ExitStack() as stack:
        for module_name, attribute in STAGES:
            owner = importlib.import_module(module_name)
            parts = attribute.split(".")
            for part in parts[:-1]:
                owner = getattr(owner, part)
            name = f"{module_name}.{attribute}"
            stack.enter_context(
                patch.object(owner, parts[-1], instrument(getattr(owner, parts[-1]), name))
            )
        yield


def measure_build(report_path, *, options, profile_path=None, sampling_path=None, context=None):
    from hitlist.builder import build_observations

    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    profiler = cProfile.Profile() if profile_path else None
    sampler = None
    if sampling_path:
        from pyinstrument import Profiler

        sampler = Profiler(interval=0.01, async_mode="disabled")
    report = {
        "schema_version": 1,
        "status": "failed",
        "profiled": profiler is not None or sampler is not None,
        "sampling_interval_seconds": 0.01 if sampler else None,
        "stage_timings": "inclusive; sequential calls are separate entries",
        "rss_scope": "per-process lifetime high-water mark; not process-tree sum",
        "cpu_scope": "parent plus exited/waited children reported separately",
        "cprofile_scope": "parent process only; worker costs remain in stage wall time",
        "platform": platform.platform(),
        "python": sys.version,
        "versions": {
            dist.metadata["Name"]: dist.version for dist in importlib.metadata.distributions()
        },
        "options": {**options, "force": True},
        "context": context or {},
        "stages": [],
    }
    before = usage()
    try:
        with measured_stages(report["stages"]):
            if profiler:
                profiler.enable()
            if sampler:
                sampler.start()
            output = build_observations(**report["options"])
        report.update(status="success", output=str(output))
        return output
    except BaseException as error:
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        if sampler:
            sampler.stop()
            sampling_path = Path(sampling_path)
            sampling_path.parent.mkdir(parents=True, exist_ok=True)
            sampler.last_session.save(sampling_path)
            sampling_path.with_suffix(".txt").write_text(sampler.output_text(show_all=True))
        if profiler:
            profiler.disable()
            profile_path = Path(profile_path)
            profile_path.parent.mkdir(parents=True, exist_ok=True)
            profiler.dump_stats(str(profile_path))
            with profile_path.with_suffix(".txt").open("w") as stream:
                stats = pstats.Stats(profiler, stream=stream)
                stats.sort_stats("cumulative").print_stats(50)
                stats.sort_stats("tottime").print_stats(50)
        report["total"] = elapsed(before)
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print("BUILD_TOTAL " + json.dumps(report["total"], sort_keys=True), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-mappings", action="store_true")
    parser.add_argument("--cached-references-only", action="store_true")
    parser.add_argument("--report", type=Path, default=Path("build-profile.json"))
    parser.add_argument(
        "--profile-out", type=Path, help="Enable cProfile (diagnostic, not a speed measurement)"
    )
    parser.add_argument(
        "--sampling-out",
        type=Path,
        help="Save a pyinstrument sampling session (install pyinstrument)",
    )
    parser.add_argument("--context", type=Path, help="JSON input/resource receipt to attach")
    args = parser.parse_args()
    measure_build(
        args.report,
        options={
            "build_mappings": not args.no_mappings,
            "fetch_missing_proteomes": not args.cached_references_only,
            "use_uniprot_search": False,
        },
        profile_path=args.profile_out,
        sampling_path=args.sampling_out,
        context=json.loads(args.context.read_text()) if args.context else None,
    )


if __name__ == "__main__":
    main()
