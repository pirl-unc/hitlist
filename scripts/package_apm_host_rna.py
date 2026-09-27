#!/usr/bin/env python3
"""Package DepMap 24Q4's HAP1 gene RNA as a hitlist line-expression source (#358).

Writes only the long-form value columns, as the other packaged CSVs do;
``build_line_expression`` stamps source metadata from ``sources.yaml``.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

DATA = Path(__file__).resolve().parents[1] / "hitlist/data/line_expression"
SOURCE_ID = "DepMap_24Q4_HAP1_gene"
FILENAME = "hap1_depmap24q4_gene_tpm.csv.gz"
#: The original 24Q4 OmicsExpressionProteinCodingGenesTPMLogp1.csv.
DEPMAP_GENE_SIZE = 506628654
DEPMAP_GENE_MD5 = "71794802b750ce77c422dad0720a40af"
HAP1_MODEL_ID = "ACH-002475"
HAP1_PROFILE_ID = "PR-QtHaIL"


def checksum(path, algorithm="sha256"):
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hap1_genes(path):
    if path.stat().st_size != DEPMAP_GENE_SIZE or checksum(path, "md5") != DEPMAP_GENE_MD5:
        raise ValueError("Expected the original DepMap 24Q4 gene matrix")
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        selected = [row for row in reader if row and row[0] == HAP1_MODEL_ID]
    if len(selected) != 1:
        raise ValueError(f"Expected exactly one {HAP1_MODEL_ID} row")
    log2_tpm = np.array(selected[0][1:], dtype=float)
    frame = pd.DataFrame(
        {
            "line_key": "HAP1",
            "source_id": SOURCE_ID,
            "granularity": "gene",
            "gene_id": "",
            # Labels are "SYMBOL (entrez)"; the builder's DepMap reader keeps
            # only the symbol too.
            "gene_name": [name.rsplit(" (", 1)[0] for name in header[1:]],
            "transcript_id": "",
            "tpm": 2.0**log2_tpm - 1.0,
            "log2_tpm": log2_tpm,
            "profile_id": HAP1_PROFILE_ID,
        }
    )
    if not np.isfinite(frame.tpm).all() or frame.tpm.lt(0).any():
        raise ValueError("Invalid HAP1 gene TPM")
    return frame.sort_values("gene_name", kind="stable").reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--depmap-gene-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DATA)
    args = parser.parse_args()
    frame = hap1_genes(args.depmap_gene_csv)
    content = frame.to_csv(index=False, float_format="%.17g").encode("utf-8")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / FILENAME
    # mtime=0 and no embedded filename keep the gzip deterministic.
    with path.open("wb") as raw, gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as gz:
        gz.write(content)
    report = {
        "sources": [
            {
                "source_id": SOURCE_ID,
                "file": FILENAME,
                "sha256": checksum(path),
                # Independent of the zlib build that compressed the file.
                "content_sha256": hashlib.sha256(content).hexdigest(),
                "n_genes": len(frame),
                "n_positive_genes": int(frame.tpm.gt(0).sum()),
                "sum_tpm": float(frame.tpm.sum()),
            }
        ]
    }
    (args.output_dir / "apm_rna_build_report.yaml").write_text(
        yaml.safe_dump(report, sort_keys=False)
    )
    print(json.dumps(report["sources"], indent=2))


if __name__ == "__main__":
    main()
