"""Temporary, reproducible #611 Linux export comparison (removed before merge)."""

import hashlib
import json
import resource
import subprocess
import sys
import time
from pathlib import Path

root = Path(sys.argv[1]).resolve()
output = Path(sys.argv[2])
sys.path.insert(0, str(root))

import pandas as pd

from hitlist import export

assert Path(export.__file__).resolve() == root / "hitlist/export.py"


def value_digest(values):
    hashes = pd.util.hash_pandas_object(values, index=True, categorize=True)
    return hashlib.sha256(hashes.to_numpy().tobytes()).hexdigest()


def fingerprint(frame):
    columns = []
    for name in frame.columns:
        series = frame[name]
        entry = {
            "name": name,
            "dtype": str(series.dtype),
            "dtype_repr": repr(series.dtype),
            "values_sha256": value_digest(series),
            "nulls_sha256": value_digest(series.isna()),
        }
        if isinstance(series.dtype, pd.CategoricalDtype):
            entry["categories_sha256"] = value_digest(series.cat.categories)
            entry["categories_dtype"] = str(series.cat.categories.dtype)
            entry["n_categories"] = len(series.cat.categories)
            entry["ordered"] = series.cat.ordered
        columns.append(entry)
    return {
        "n_rows": len(frame),
        "n_columns": len(frame.columns),
        "columns": columns,
        "column_index_type": type(frame.columns).__name__,
        "column_index_names": frame.columns.names,
        "index_type": type(frame.index).__name__,
        "index_dtype": str(frame.index.dtype),
        "index_names": frame.index.names,
        "index_sha256": value_digest(frame.index),
    }


# Confirm the comparator catches changed cells and preserves equal copies.
control = pd.DataFrame({"value": [1.0, float("nan")], "label": pd.Categorical(["a", "b"])})
assert fingerprint(control) == fingerprint(control.copy())
changed = control.copy()
changed.loc[0, "value"] = 2.0
assert fingerprint(control) != fingerprint(changed)
changed = control.copy()
changed["label"] = changed["label"].cat.reorder_categories(["b", "a"])
assert fingerprint(control) != fingerprint(changed)
del control, changed

start = time.monotonic()
frame = export.generate_observations_table()
# Capture before hashing: this is the export peak, not verification overhead.
elapsed_seconds = time.monotonic() - start
peak_rss_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
result = {
    "commit": subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip(),
    "export_sha256": hashlib.sha256((root / "hitlist/export.py").read_bytes()).hexdigest(),
    "peak_rss_kib": peak_rss_kib,
    "elapsed_seconds": elapsed_seconds,
    "fingerprint": fingerprint(frame),
}
output.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({key: value for key, value in result.items() if key != "fingerprint"}), flush=True)
print(f"Exported {len(frame)} rows by {len(frame.columns)} columns", flush=True)
