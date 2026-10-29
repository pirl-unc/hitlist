"""Profile 100,000 real source rows in a standalone interpreter.

This diagnostic identifies row-level hotspots; it is not a whole-build benchmark.
"""

import argparse


def measure_codecs(sample):
    """Compare lossless scratch codecs on 10,000 identical real row payloads."""
    import itertools
    import json
    import time
    import zlib

    import pyarrow as pa

    from hitlist.scanner import _open_csv, _safe_col

    reader, columns, _, handle = _open_csv(sample)
    try:
        payloads = [
            (
                json.dumps(
                    {name: _safe_col(row, col) for name, col in columns.items()}, sort_keys=True
                )
                + "\n"
                + json.dumps(row)
            ).encode()
            for row in itertools.islice(reader, 10000)
        ]
    finally:
        handle.close()
    results = []
    for name in ("zlib-1", "zstd-1", "zstd-3", "lz4"):
        codec = (
            None
            if name == "zlib-1"
            else pa.Codec(
                "zstd" if name.startswith("zstd") else "lz4",
                compression_level=int(name[-1]) if name.startswith("zstd") else None,
            )
        )
        for repeat in range(3):
            started, cpu = time.perf_counter(), time.process_time()
            compressed = [
                zlib.compress(payload, level=1)
                if codec is None
                else len(payload).to_bytes(8, "little") + codec.compress(payload, asbytes=True)
                for payload in payloads
            ]
            encoded, encoded_cpu = time.perf_counter(), time.process_time()
            restored = [
                zlib.decompress(payload)
                if codec is None
                else codec.decompress(
                    payload[8:],
                    decompressed_size=int.from_bytes(payload[:8], "little"),
                    asbytes=True,
                )
                for payload in compressed
            ]
            decoded, decoded_cpu = time.perf_counter(), time.process_time()
            if restored != payloads:
                raise ValueError(f"Non-lossless scratch codec: {name}")
            results.append(
                {
                    "codec": name,
                    "repeat": repeat + 1,
                    "n_rows": len(payloads),
                    "raw_bytes": sum(map(len, payloads)),
                    "compressed_bytes": sum(map(len, compressed)),
                    "encode_wall_seconds": encoded - started,
                    "encode_cpu_seconds": encoded_cpu - cpu,
                    "decode_wall_seconds": decoded - encoded,
                    "decode_cpu_seconds": decoded_cpu - encoded_cpu,
                }
            )
    codec = pa.Codec("zstd", compression_level=1)
    groups = [payloads[index : index + 128] for index in range(0, len(payloads), 128)]
    for repeat in range(3):
        started, cpu = time.perf_counter(), time.process_time()
        blocks = [b"".join(group) for group in groups]
        compressed = [codec.compress(block, asbytes=True) for block in blocks]
        encoded, encoded_cpu = time.perf_counter(), time.process_time()
        restored = [
            codec.decompress(block, decompressed_size=len(raw), asbytes=True)
            for block, raw in zip(compressed, blocks)
        ]
        decoded, decoded_cpu = time.perf_counter(), time.process_time()
        if restored != blocks:
            raise ValueError("Non-lossless block codec")
        results.append(
            {
                "codec": "zstd-1-block-128",
                "repeat": repeat + 1,
                "n_rows": len(payloads),
                "raw_bytes": sum(map(len, payloads)),
                "compressed_bytes": sum(map(len, compressed)) + 24 * len(payloads),
                "size_note": "includes 24 bytes per row for block/offset/length references",
                "encode_wall_seconds": encoded - started,
                "encode_cpu_seconds": encoded_cpu - cpu,
                "decode_wall_seconds": decoded - encoded,
                "decode_cpu_seconds": decoded_cpu - encoded_cpu,
            }
        )
    return results


def main(label, codecs_only=False):
    import csv
    import itertools
    import json
    import os
    import tempfile
    from pathlib import Path

    from pyinstrument import Profiler

    if not label or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for c in label):
        raise ValueError("Invalid label")
    destination = Path("/data/hitlist-reliability-20261028") / label
    destination.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory(prefix="hitlist-scan-profile-") as directory:
        os.environ.update(HITLIST_DATA_DIR=directory, XDG_CACHE_HOME=directory)
        from hitlist.provenance import ContributorCollector, file_digest
        from hitlist.scanner import scan

        pin = json.loads(Path("/opt/benchmark/sources.json").read_text())["iedb"]
        sample = Path(directory) / "iedb-prefix.csv"
        with (
            (Path("/data") / pin["remote_path"]).open(newline="") as source,
            sample.open("w", newline="") as target,
        ):
            writer = csv.writer(target)
            writer.writerows(itertools.islice(csv.reader(source), 10002 if codecs_only else 100002))
        receipt = {
            "scope": f"first {10000 if codecs_only else 100000} logical IEDB data rows; diagnostic only",
            "source": pin,
            "sample": file_digest(sample),
            "producer_archive": file_digest(Path("/opt/benchmark/producer.tar.gz")),
            "recipe": file_digest(Path(__file__)),
        }
        if codecs_only:
            (destination / "codecs.json").write_text(
                json.dumps(measure_codecs(sample), indent=2) + "\n"
            )
            (destination / "inputs.json").write_text(json.dumps(receipt, indent=2) + "\n")
            return
        profiler = Profiler(interval=0.01, async_mode="disabled")
        profiler.start()
        try:
            with ContributorCollector() as collector:
                frame = scan(
                    iedb_path=sample,
                    cedar_path=None,
                    mhc_species=None,
                    classify_source=True,
                    provenance=collector,
                )
                receipt["n_rows"] = len(frame)
                receipt["provenance"] = collector.write(
                    [frame], Path(directory) / "contributors.parquet"
                )
        finally:
            profiler.stop()
            profiler.last_session.save(destination / "scan.pyisession")
            (destination / "scan.txt").write_text(profiler.output_text(show_all=True))
            receipt["profiler"] = {"tool": "pyinstrument", "interval_seconds": 0.01}
            (destination / "inputs.json").write_text(json.dumps(receipt, indent=2) + "\n")
        (destination / "codecs.json").write_text(
            json.dumps(measure_codecs(sample), indent=2) + "\n"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("label")
    parser.add_argument("--codecs-only", action="store_true")
    args = parser.parse_args()
    main(args.label, args.codecs_only)
