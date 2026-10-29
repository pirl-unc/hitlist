"""Picklable diagnostic wrapper; production workers and results stay unchanged."""

import json
import multiprocessing
import os
import resource
import sys
from pathlib import Path

from hitlist.mappings import _per_canonical_mapping_worker as original_worker

_sampler = None
_n_tasks = 0


def save_profile():
    if _sampler is None or _sampler.last_session is None:
        return
    directory = Path(os.environ["HITLIST_WORKER_PROFILE_DIR"])
    directory.mkdir(parents=True, exist_ok=True)
    stem = directory / f"mapping-{os.getpid()}"
    _sampler.last_session.save(stem.with_suffix(".pyisession"))
    stem.with_suffix(".txt").write_text(_sampler.output_text(show_all=True))
    usage = resource.getrusage(resource.RUSAGE_SELF)
    stem.with_suffix(".json").write_text(
        json.dumps(
            {
                "n_tasks": _n_tasks,
                "cpu_seconds": usage.ru_utime + usage.ru_stime,
                "process_peak_rss_bytes": usage.ru_maxrss
                * (1 if sys.platform == "darwin" else 1024),
                "scope": "worker process lifetime; diagnostic sampling enabled",
            },
            indent=2,
        )
        + "\n"
    )


def profile_worker(task):
    global _sampler, _n_tasks
    if multiprocessing.current_process().name == "MainProcess":
        return original_worker(task)  # A serial pass is already sampled by its parent.
    if _sampler is None:
        from multiprocessing.util import Finalize

        from pyinstrument import Profiler

        _sampler = Profiler(interval=0.01, async_mode="disabled")
        Finalize(None, save_profile, exitpriority=1)
    _sampler.start()
    try:
        return original_worker(task)
    finally:
        _sampler.stop()
        _n_tasks += 1
