"""Materialize source-linked, row-preserving bulk pH corrections through datacache."""

import argparse
import gzip
import hashlib
import io
import json
import platform
import resource
import subprocess
import time
from collections import Counter
from pathlib import Path

import pandas as pd
from datacache import install_bundle, materialize

from hitlist.bulk_proteomics import _curate_bj_metadata
from hitlist.curation_yaml import load_curation_yaml
from hitlist.provenance import file_digest
from hitlist.version import __version__

ROOT = Path(__file__).resolve().parents[1]
TABLES = {
    "peptides": "bekker_jensen_2017_peptides.csv.gz",
    "proteins": "bekker_jensen_2017_protein_abundance.csv.gz",
}


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache-root", type=Path, default=Path.home() / ".hitlist/bulk-fractionation"
    )
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True):
        raise ValueError("Commit the exact audit recipe before execution")
    registry_path = ROOT / "hitlist/data/bulk_proteomics/detectability.yaml"
    registry = load_curation_yaml(registry_path)
    assets = load_curation_yaml(ROOT / "hitlist/data/data_assets.yaml")
    sources = {
        item["filename"]: {
            "url": assets["base_url"] + "/" + item["filename"],
            "sha256": item["sha256"],
            "size": item["size_bytes"],
        }
        for item in assets["assets"]
        if item["filename"] in TABLES.values()
    }
    pxd = load_curation_yaml(ROOT / "hitlist/data/pxd013455.yaml")
    paper = pxd["supporting_files"]["bekker-jensen-2017-paper.xml"]
    sources["paper.xml"] = {
        "url": paper["url"],
        "sha256": paper["sha256"],
        "size": paper["size_bytes"],
    }
    paths = install_bundle(
        args.cache_root / "sources", sources, download_options={"show_progress": False}
    )
    inputs = {
        name: {"path": str(path), "sha256": sources[name]["sha256"]} for name, path in paths.items()
    }
    for name, path in {
        "registry.yaml": registry_path,
        "audit.py": Path(__file__),
        "projection.py": ROOT / "hitlist/bulk_fractionation.py",
        "loader.py": ROOT / "hitlist/bulk_proteomics.py",
    }.items():
        inputs[name] = {"path": str(path), "sha256": file_digest(path)["sha256"]}
    output_names = [
        "peptides.csv.gz",
        "proteins.csv.gz",
        "scope-counts.tsv",
        "scope-evidence.tsv",
        "audit.json",
        "source-registry.json",
        "registry.yaml",
        "audit.py",
        "projection.py",
        "loader.py",
    ]

    def builder(acquired, outputs):
        import shutil

        started = time.monotonic()
        counts, tables = Counter(), {}
        for kind, filename in TABLES.items():
            n_rows = n_changed_ph = n_unknown_ph = 0
            identity = hashlib.sha256()
            with (
                Path(outputs[kind + ".csv.gz"]).open("wb") as compressed,
                gzip.GzipFile(filename="", fileobj=compressed, mode="wb", mtime=0) as zipped,
                io.TextIOWrapper(zipped, encoding="utf-8", newline="") as stream,
            ):
                for before in pd.read_csv(
                    acquired[filename], chunksize=25000, dtype=str, keep_default_na=False
                ):
                    after = _curate_bj_metadata(before)
                    stable = before.columns.drop(["fractionation_ph", "reference"])
                    pd.testing.assert_frame_equal(before[stable], after[stable])
                    pd.testing.assert_series_equal(
                        before.reference,
                        after.reported_reference,
                        check_names=False,
                        check_dtype=False,
                    )
                    pd.testing.assert_frame_equal(after, _curate_bj_metadata(after))
                    assert after.fractionation_protocol_id.ne("").all()
                    assert after.reference.eq("PMID:28601559").all()
                    old_ph = pd.to_numeric(before.fractionation_ph)
                    assert old_ph.eq(after.legacy_fractionation_ph).all()
                    n_changed_ph += int(
                        (after.fractionation_ph.isna() | after.fractionation_ph.ne(old_ph)).sum()
                    )
                    n_unknown_ph += int(after.fractionation_ph.isna().sum())
                    identity.update(before[stable].to_csv(index=False, header=False).encode())
                    keys = [
                        "fractionation_protocol_id",
                        "cell_line",
                        "digestion_enzyme",
                        "n_fractions_in_run",
                        "enrichment",
                        "legacy_fractionation_ph",
                        "fractionation_ph",
                        "fractionation_ph_basis",
                        "fractionation_ph_reference",
                    ]
                    grouped = (
                        after[keys].astype("string").fillna("").groupby(keys, dropna=False).size()
                    )
                    for key, n_observations in grouped.items():
                        counts[(kind, *key)] += int(n_observations)
                    after.to_csv(stream, index=False, header=n_rows == 0)
                    n_rows += len(before)
            tables[kind] = {
                "n_rows": n_rows,
                "n_changed_ph_rows": n_changed_ph,
                "n_unknown_ph_rows": n_unknown_ph,
                "n_other_changed_cells": 0,
                "n_corrected_reference_rows": n_rows,
                "all_original_references_preserved": True,
                "stable_columns": stable.tolist(),
                "stable_row_order_sha256": identity.hexdigest(),
                "original": sources[filename],
                "corrected": file_digest(outputs[kind + ".csv.gz"]),
            }
        count_columns = ["granularity", *keys, "n_rows"]
        pd.DataFrame(
            [(*key, value) for key, value in sorted(counts.items())], columns=count_columns
        ).to_csv(outputs["scope-counts.tsv"], sep="\t", index=False)
        evidence = []
        for scope in registry["scopes"]:
            attribution = registry["fractionation_ph_evidence"][
                scope["fractionation_ph_evidence_id"]
            ]
            evidence.append(
                {
                    "protocol_id": scope["protocol_id"],
                    "experiment_ids": ";".join(scope["experiment_ids"]),
                    "cell_line_name": scope["cell_line_name"],
                    "legacy_fractionation_ph": scope["legacy_fractionation_ph"],
                    "fractionation_ph": scope["fractionation_ph"],
                    "evidence_id": scope["fractionation_ph_evidence_id"],
                    **attribution,
                }
            )
        pd.DataFrame(evidence).to_csv(outputs["scope-evidence.tsv"], sep="\t", index=False)
        write(outputs["source-registry.json"], sources)
        for name in ("registry.yaml", "audit.py", "projection.py", "loader.py"):
            shutil.copyfile(acquired[name], outputs[name])
        write(
            outputs["audit.json"],
            {
                "commit": commit,
                "dirty": False,
                "hitlist_version": __version__,
                "python": platform.python_version(),
                "pandas": pd.__version__,
                "runtime_seconds": time.monotonic() - started,
                "process_peak_rss_bytes": int(
                    resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                    * (1 if platform.system() == "Darwin" else 1024)
                ),
                "tables": tables,
                "n_reviewed_scopes": len(evidence),
                "preservation": "Every original row/order and all fields other than reference/pH are unchanged; both prior annotations are retained separately.",
                "scope": "Original-analysis bulk proteomics summaries; not MHC ligands. No MHC allele set applies. Cell lines retained; donor/specimen/biological-replicate identities remain unresolved.",
                "limitations": "Source-level protocol annotation only. No measured eluate pH, comparable depth ladder, original FASTA recovery or new detectability qualification.",
            },
        )

    outputs = materialize(
        args.cache_root / "audit" / commit,
        inputs,
        transform={
            "version": "bulk-fractionation-review-v1",
            "options": {"commit": commit, "hitlist_version": __version__, "pandas": pd.__version__},
        },
        outputs={name: {} for name in output_names},
        builder=builder,
    )
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    write(
        args.receipt,
        {
            "commit": commit,
            "paths": {n: str(p) for n, p in outputs.items()},
            "files": {n: file_digest(p) for n, p in outputs.items()},
        },
    )
    print(Path(outputs["audit.json"]).read_text())


if __name__ == "__main__":
    main()
