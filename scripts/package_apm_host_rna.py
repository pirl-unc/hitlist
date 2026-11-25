#!/usr/bin/env python3
"""Package APM host bulk RNA-seq references as hitlist line-expression sources (#358).

Supports:
- DepMap 24Q4 parental HAP1 gene RNA (`DepMap_24Q4_HAP1_gene`)
- Ensembl 110 / kallisto 0.51.1 (`--rf-stranded`) parental C1R (`SRP142649_C1R_parental_gene`)
- Ensembl 110 / kallisto 0.51.1 (`--rf-stranded`) 721.221 Cas9/sgNT controls
  (`GSE228188_721_221_Cas9_sgNT_gene`, mean across 3 biological replicates)

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

C1R_SOURCE_ID = "SRP142649_C1R_parental_gene"
C1R_FILENAME = "c1r_srp142649_gene_tpm.csv.gz"
C1R_RUNS = ("SRR7065975",)

B721_SOURCE_ID = "GSE228188_721_221_Cas9_sgNT_gene"
B721_FILENAME = "b721_221_gse228188_sgnt_gene_tpm.csv.gz"
B721_RUNS = ("SRR23958328", "SRR23958329", "SRR23958330")


def checksum(path, algorithm="sha256"):
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_deterministic_gz(path: Path, frame: pd.DataFrame) -> tuple[str, str]:
    content = frame.to_csv(index=False, float_format="%.17g").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    # mtime=0 and no embedded filename keep the gzip deterministic.
    with path.open("wb") as raw, gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as gz:
        gz.write(content)
    return checksum(path), hashlib.sha256(content).hexdigest()


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


def kallisto_protein_coding_genes(
    kallisto_dir: Path,
    *,
    runs: tuple[str, ...],
    line_key: str,
    source_id: str,
    profile_id: str,
) -> pd.DataFrame:
    """Aggregate kallisto transcript TPMs to Ensembl 110 protein-coding gene TPMs.

    Preserves the full Ensembl 110 cDNA+ncRNA TPM denominator (protein-coding gene
    TPMs sum to ~920k-940k rather than renormalizing to 1e6), retains zero values,
    and averages TPM across biological replicates when multiple runs are supplied.
    """
    tx2gene = pd.read_csv(
        kallisto_dir / "ensembl110_tx2gene.tsv",
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )
    per_run: list[pd.DataFrame] = []
    for run in runs:
        ab_path = kallisto_dir / f"quant_{run}" / "abundance.tsv"
        ab = pd.read_csv(ab_path, sep="\t")
        merged = ab.merge(tx2gene, left_on="target_id", right_on="transcript_id", how="inner")
        if len(merged) != len(ab):
            raise ValueError(f"Transcript-to-gene join mismatch for {run}: {len(merged)} vs {len(ab)}")
        pc = merged[merged["gene_biotype"] == "protein_coding"]
        gene_df = pc.groupby(["gene_id", "gene_symbol"], as_index=False)["tpm"].sum()
        gene_df = gene_df.rename(columns={"tpm": f"tpm_{run}"})
        per_run.append(gene_df)

    combined = per_run[0]
    for nxt in per_run[1:]:
        combined = combined.merge(nxt, on=["gene_id", "gene_symbol"], how="inner")
        if len(combined) != len(per_run[0]):
            raise ValueError(f"Replicate gene set mismatch for {source_id}")

    tpm_cols = [f"tpm_{run}" for run in runs]
    tpm = combined[tpm_cols].mean(axis=1).to_numpy(dtype=float)
    if not np.isfinite(tpm).all() or (tpm < 0).any():
        raise ValueError(f"Invalid kallisto gene TPM for {source_id}")

    frame = pd.DataFrame(
        {
            "line_key": line_key,
            "source_id": source_id,
            "granularity": "gene",
            "gene_id": combined["gene_id"].astype(str),
            "gene_name": combined["gene_symbol"].astype(str),
            "transcript_id": "",
            "tpm": tpm,
            "log2_tpm": np.log2(tpm + 1.0),
            "profile_id": profile_id,
        }
    )
    return frame.sort_values(["gene_id", "gene_name"], kind="stable").reset_index(drop=True)


def _kallisto_report_entry(
    *,
    source_id: str,
    filename: str,
    path: Path,
    frame: pd.DataFrame,
    sha256: str,
    content_sha256: str,
    receipt: dict,
    runs: tuple[str, ...],
) -> dict:
    run_map = {r["run"]: r for r in receipt["runs"]}
    return {
        "source_id": source_id,
        "file": filename,
        "sha256": sha256,
        "content_sha256": content_sha256,
        "n_genes": len(frame),
        "n_positive_genes": int(frame.tpm.gt(0).sum()),
        "sum_tpm": float(frame.tpm.sum()),
        "quantifier": "kallisto",
        "kallisto_version": receipt["kallisto_version"],
        "strandedness": "rf-stranded",
        "kallisto_index_sha256": receipt["kallisto_index_sha256"],
        "ensembl_110_reference": receipt["ensembl_110"],
        "n_replicates": len(runs),
        "runs": [
            {
                "run": run,
                "gsm": run_map[run]["gsm"],
                "biosample": run_map[run]["biosample"],
                "n_processed_reads": run_map[run]["run_info"]["n_processed"],
                "n_pseudoaligned_reads": run_map[run]["run_info"]["n_pseudoaligned"],
                "p_pseudoaligned": run_map[run]["run_info"]["p_pseudoaligned"],
                "fastqs": run_map[run]["fastqs"],
            }
            for run in runs
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--depmap-gene-csv", type=Path, default=None)
    parser.add_argument("--kallisto-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DATA)
    args = parser.parse_args()
    if args.depmap_gene_csv is None and args.kallisto_dir is None:
        parser.error("Supply at least one of --depmap-gene-csv or --kallisto-dir")

    report_path = args.output_dir / "apm_rna_build_report.yaml"
    existing_by_id: dict[str, dict] = {}
    if report_path.exists():
        existing = yaml.safe_load(report_path.read_text()) or {}
        for entry in existing.get("sources", []):
            existing_by_id[entry["source_id"]] = entry

    if args.depmap_gene_csv is not None:
        frame = hap1_genes(args.depmap_gene_csv)
        path = args.output_dir / FILENAME
        sha256, content_sha256 = _write_deterministic_gz(path, frame)
        existing_by_id[SOURCE_ID] = {
            "source_id": SOURCE_ID,
            "file": FILENAME,
            "sha256": sha256,
            # Independent of the zlib build that compressed the file.
            "content_sha256": content_sha256,
            "n_genes": len(frame),
            "n_positive_genes": int(frame.tpm.gt(0).sum()),
            "sum_tpm": float(frame.tpm.sum()),
        }

    if args.kallisto_dir is not None:
        receipt = json.loads((args.kallisto_dir / "quant_receipt.json").read_text())

        c1r_frame = kallisto_protein_coding_genes(
            args.kallisto_dir,
            runs=C1R_RUNS,
            line_key="C1R",
            source_id=C1R_SOURCE_ID,
            profile_id="SRR7065975",
        )
        c1r_path = args.output_dir / C1R_FILENAME
        c1r_sha, c1r_csha = _write_deterministic_gz(c1r_path, c1r_frame)
        existing_by_id[C1R_SOURCE_ID] = _kallisto_report_entry(
            source_id=C1R_SOURCE_ID,
            filename=C1R_FILENAME,
            path=c1r_path,
            frame=c1r_frame,
            sha256=c1r_sha,
            content_sha256=c1r_csha,
            receipt=receipt,
            runs=C1R_RUNS,
        )

        b721_frame = kallisto_protein_coding_genes(
            args.kallisto_dir,
            runs=B721_RUNS,
            line_key="721.221-Cas9-sgNT",
            source_id=B721_SOURCE_ID,
            profile_id="GSE228188_sgNT_mean3",
        )
        b721_path = args.output_dir / B721_FILENAME
        b721_sha, b721_csha = _write_deterministic_gz(b721_path, b721_frame)
        existing_by_id[B721_SOURCE_ID] = _kallisto_report_entry(
            source_id=B721_SOURCE_ID,
            filename=B721_FILENAME,
            path=b721_path,
            frame=b721_frame,
            sha256=b721_sha,
            content_sha256=b721_csha,
            receipt=receipt,
            runs=B721_RUNS,
        )

    ordered_ids = [SOURCE_ID, C1R_SOURCE_ID, B721_SOURCE_ID]
    sources = [existing_by_id[sid] for sid in ordered_ids if sid in existing_by_id]
    for sid, entry in existing_by_id.items():
        if sid not in ordered_ids:
            sources.append(entry)

    report = {"sources": sources}
    report_path.write_text(yaml.safe_dump(report, sort_keys=False))
    print(json.dumps(sources, indent=2))


if __name__ == "__main__":
    main()
