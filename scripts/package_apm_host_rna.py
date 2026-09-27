#!/usr/bin/env python3
"""Package verified host RNA measurements; never reinterpret CPM/FPKM as TPM."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

DATA = Path(__file__).resolve().parents[1] / "hitlist/data/line_expression"


def checksum(path, algorithm="sha256"):
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hap1_genes(path):
    expected_md5 = "71794802b750ce77c422dad0720a40af"
    if path.stat().st_size != 506628654 or checksum(path, "md5") != expected_md5:
        raise ValueError("Expected the original DepMap 24Q4 gene matrix")
    selected = []
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        for row in reader:
            if row and row[0] == "ACH-002475":
                selected.append(row)
    if len(selected) != 1 or len(header) != 19194:
        raise ValueError("Expected exactly one complete 19,193-gene HAP1 row")
    log2_tpm = np.array(selected[0][1:], dtype=float)
    return pd.DataFrame(
        {
            "gene_id": "",
            "gene_name": [name.rsplit(" (", 1)[0] for name in header[1:]],
            "tpm": 2.0**log2_tpm - 1.0,
            "log2_tpm": log2_tpm,
            "profile_id": "PR-QtHaIL",
        }
    )


def write_source(frame, source_id, line_key, sources, output_dir):
    source = sources[source_id]
    frame = frame.copy()
    frame["line_key"] = line_key
    frame["source_id"] = source_id
    frame["granularity"] = "gene"
    frame["transcript_id"] = ""
    if "profile_id" not in frame:
        frame["profile_id"] = ""
    if "log2_tpm" not in frame:
        frame["log2_tpm"] = np.log2(frame.tpm + 1)
    for column in [
        "backend",
        "pmid",
        "reference",
        "study_label",
        "normalization",
        "quantifier",
        "species",
        "license",
    ]:
        frame[column] = source.get(column)
    if not np.isfinite(frame.tpm).all() or frame.tpm.lt(0).any():
        raise ValueError(f"Invalid gene TPM for {line_key}")
    frame = frame.sort_values(["gene_id", "gene_name"]).reset_index(drop=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / source["file"]
    with (
        path.open("wb") as raw,
        gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as compressed,
        io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text,
    ):
        frame.to_csv(text, index=False, float_format="%.17g")
    return {
        "source_id": source_id,
        "file": path.name,
        "sha256": checksum(path),
        "n_genes": len(frame),
        "n_positive_genes": int(frame.tpm.gt(0).sum()),
        "sum_tpm": float(frame.tpm.sum()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--depmap-gene-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DATA)
    args = parser.parse_args()
    sources = {
        source["source_id"]: source
        for source in yaml.safe_load((DATA / "sources.yaml").read_text())["sources"]
    }
    report = {
        "sources": [
            write_source(
                hap1_genes(args.depmap_gene_csv),
                "DepMap_24Q4_HAP1_gene",
                "HAP1",
                sources,
                args.output_dir,
            )
        ]
    }
    report_path = args.output_dir / "apm_rna_build_report.yaml"
    report_path.write_text(yaml.safe_dump(report, sort_keys=False))
    print(json.dumps(report["sources"], indent=2))


if __name__ == "__main__":
    main()
