"""Compare collector revisions on a bounded, identical sample of real CSV rows.

Run each revision in a fresh process under /usr/bin/time. This measures the
collector, not scientific classification or whole-build resource usage.
"""

import argparse
import importlib.util
import itertools
import json
import resource
import sys
import time
from pathlib import Path

import pandas as pd

from hitlist.provenance import ContributorCollector
from hitlist.scanner import _open_csv, _safe_col


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--n-rows", type=int, default=30000)
    parser.add_argument("--collector-module", type=Path)
    args = parser.parse_args()
    collector_type = ContributorCollector
    if args.collector_module:
        spec = importlib.util.spec_from_file_location("collector_revision", args.collector_module)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        collector_type = module.ContributorCollector
    started = time.perf_counter()
    reader, columns, _, handle = _open_csv(args.source)
    roots = []
    previous = ""
    with collector_type() as collector:
        try:
            for row_number, values in enumerate(itertools.islice(reader, args.n_rows), 1):
                record = collector.record(
                    "iedb",
                    row_number,
                    {name: _safe_col(values, index) for name, index in columns.items()},
                    values,
                )
                if row_number % 5 == 0:
                    collector.redirect(record, previous, "assay_copy")
                else:
                    roots.append(
                        collector.observe(record, "donor A" if row_number % 7 == 0 else "")
                    )
                previous = record
        finally:
            handle.close()
        collector.db.commit()
        captured = time.perf_counter()
        page_size = collector.db.execute("PRAGMA page_size").fetchone()[0]
        capture_bytes = collector.db.execute("PRAGMA page_count").fetchone()[0] * page_size
        metadata = collector.write([pd.DataFrame({"provenance_id": roots})], args.output)
        written = time.perf_counter()
        print(
            json.dumps(
                {
                    "collector": str(
                        args.collector_module
                        or Path(__file__).resolve().parents[1] / "hitlist/provenance.py"
                    ),
                    "n_roots": len(roots),
                    "n_links": metadata["n_contributor_links"],
                    "capture_seconds": captured - started,
                    "write_seconds": written - captured,
                    "capture_database_bytes": capture_bytes,
                    "final_database_bytes": collector.db.execute("PRAGMA page_count").fetchone()[0]
                    * page_size,
                    "parquet_bytes": args.output.stat().st_size,
                    "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                    * (1 if sys.platform == "darwin" else 1024),
                },
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
