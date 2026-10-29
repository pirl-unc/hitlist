"""Compare exact CSV contributor serialization on a bounded real-source prefix."""

import argparse
import itertools
import json
import time
from functools import lru_cache
from operator import itemgetter
from pathlib import Path


def measure(sample):
    from hitlist.scanner import _open_csv, _safe_col

    reader, columns, _, handle = _open_csv(sample)
    try:
        rows = list(itertools.islice(reader, 10000))
    finally:
        handle.close()
    encoder = json.JSONEncoder(sort_keys=True)
    names, indices = zip(*columns.items())
    project = itemgetter(*indices)
    max_index = max(indices)
    prefixes = [(json.dumps(name) + ": ", index) for name, index in sorted(columns.items())]

    def baseline(row):
        fields = {name: _safe_col(row, index) for name, index in columns.items()}
        return (encoder.encode(fields) + "\n" + json.dumps(row)).encode()

    def projected(row):
        fields = (
            dict(zip(names, project(row)))
            if len(row) > max_index
            else {name: _safe_col(row, index) for name, index in columns.items()}
        )
        return (encoder.encode(fields) + "\n" + json.dumps(row)).encode()

    def tokens(row, escape=json.encoder.encode_basestring_ascii):
        encoded = [escape(value) for value in row]
        fields = ", ".join(
            prefix + (encoded[index] if index < len(encoded) else '""')
            for prefix, index in prefixes
        )
        return ("{" + fields + "}\n[" + ", ".join(encoded) + "]").encode()

    cached_escape = lru_cache(maxsize=2048)(json.encoder.encode_basestring_ascii)

    def escape_short(value):
        return (
            cached_escape(value)
            if len(value) <= 128
            else json.encoder.encode_basestring_ascii(value)
        )

    def cached_tokens(row):
        return tokens(row, escape_short)

    expected = [baseline(row) for row in rows]
    results = []
    for repeat in range(5):
        variants = [
            ("baseline", baseline),
            ("projected", projected),
            ("tokens", tokens),
            ("cached_tokens", cached_tokens),
        ]
        if repeat % 2:
            variants.reverse()
        for name, function in variants:
            cached_escape.cache_clear()
            wall, cpu = time.perf_counter(), time.process_time()
            actual = [function(row) for row in rows]
            elapsed = {
                "wall_seconds": time.perf_counter() - wall,
                "cpu_seconds": time.process_time() - cpu,
            }
            if actual != expected:
                raise ValueError(f"Serialization differs: {name}")
            results.append(
                {
                    "variant": name,
                    "repeat": repeat + 1,
                    "n_rows": len(rows),
                    "identical_bytes": True,
                    **elapsed,
                }
            )
    return results


def main(label):
    import csv
    import tempfile

    from hitlist.provenance import file_digest

    if not label or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for c in label):
        raise ValueError("Invalid label")
    destination = Path("/data/hitlist-reliability-20261028") / label
    destination.mkdir(parents=True, exist_ok=False)
    pin = json.loads(Path("/opt/benchmark/sources.json").read_text())["iedb"]
    with tempfile.TemporaryDirectory(prefix="hitlist-json-sample-") as directory:
        sample = Path(directory) / "sample.csv"
        with (
            (Path("/data") / pin["remote_path"]).open(newline="") as source,
            sample.open("w", newline="") as output,
        ):
            csv.writer(output).writerows(itertools.islice(csv.reader(source), 10002))
        receipt = {
            "scope": "10,000-row serialization diagnostic, not full-build timing",
            "source": pin,
            "sample": file_digest(sample),
            "producer_archive": file_digest(Path("/opt/benchmark/producer.tar.gz")),
            "recipe": file_digest(Path(__file__)),
            "measurements": measure(sample),
        }
        (destination / "serialization.json").write_text(json.dumps(receipt, indent=2) + "\n")
        print(json.dumps(receipt["measurements"], indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("label")
    main(parser.parse_args().label)
