"""Freeze a complete source-backed experiment-count audit through datacache."""

import argparse
import gzip
import hashlib
import io
import json
import platform
import resource
import shutil
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from datacache import install_bundle, install_zip_member, materialize, validate_file

from hitlist.bulk_proteomics import _curate_bj_metadata
from hitlist.bulk_replicates import curate_replicate_counts
from hitlist.curation_yaml import load_curation_yaml
from hitlist.provenance import file_digest
from hitlist.version import __version__

ROOT = Path(__file__).resolve().parents[1]
SOURCE_NAME = "bekker_jensen_2017_peptides.csv.gz"


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, default=Path.home() / ".hitlist/bulk-replicates")
    parser.add_argument("--receipt", required=True, type=Path)
    args = parser.parse_args()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    assert not subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)
    registry_path = ROOT / "hitlist/data/bulk_proteomics/detectability.yaml"
    registry = load_curation_yaml(registry_path)
    assets = load_curation_yaml(ROOT / "hitlist/data/data_assets.yaml")
    asset = next(item for item in assets["assets"] if item["filename"] == SOURCE_NAME)
    sources = {
        SOURCE_NAME: {
            "url": assets["base_url"] + "/" + SOURCE_NAME,
            "sha256": asset["sha256"],
            "size": asset["size_bytes"],
        }
    }
    paths = install_bundle(
        args.cache_root / "sources", sources, download_options={"show_progress": False}
    )
    spec = registry["provenance"]["members"]["summary.txt"]
    member = install_zip_member(
        args.cache_root / "original-summary",
        registry["provenance"]["archive"],
        "summary.txt",
        expected_sha256=spec["sha256"],
        expected_size=spec["size_bytes"],
        max_compressed_bytes=spec["size_bytes"],
        max_bytes=spec["size_bytes"],
    )
    validate_file(member["payload"], spec["sha256"], spec["size_bytes"])
    inputs = {
        name: {
            "path": str(path),
            "sha256": file_digest(path)["sha256"],
            "size": Path(path).stat().st_size,
        }
        for name, path in {
            **paths,
            "summary.txt": member["payload"],
            "summary-receipt.json": member["receipt.json"],
            "registry.yaml": registry_path,
            "audit.py": Path(__file__),
            "projection.py": ROOT / "hitlist/bulk_replicates.py",
            "fractionation.py": ROOT / "hitlist/bulk_fractionation.py",
        }.items()
    }
    output_names = [
        "peptides.csv.gz",
        "scope-counts.tsv",
        "scope-evidence.tsv",
        "audit.json",
        "source-registry.json",
        "summary.txt",
        "summary-receipt.json",
        "registry.yaml",
        "audit.py",
        "projection.py",
        "fractionation.py",
    ]

    def build(acquired, outputs):
        started = time.monotonic()
        summary = pd.read_csv(acquired["summary.txt"], sep="\t", dtype=str, keep_default_na=False)
        evidence = []
        for scope in registry["scopes"]:
            rows = summary[summary.Experiment.isin(scope["experiment_ids"])]
            assert set(rows.Experiment) == set(scope["experiment_ids"])
            raw_files = sorted(set(rows["Raw file"]))
            assert len(raw_files) == len(rows) == scope["n_acquisition_runs"]
            assert (
                hashlib.sha256("\n".join(raw_files).encode()).hexdigest()
                == scope["raw_file_ids_sha256"]
            )
            assert len(scope["experiment_ids"]) == scope["n_replicates_possible"]
            evidence.append(
                {
                    **scope,
                    "experiment_ids": ";".join(scope["experiment_ids"]),
                    "n_experiments_possible": len(scope["experiment_ids"]),
                    "summary_sha256": spec["sha256"],
                }
            )
        # Reconstruct the historical observed-maximum denominator only as an
        # explicitly labelled diagnostic, never as the source of the new total.
        maxima = defaultdict(int)
        for raw in pd.read_csv(
            acquired[SOURCE_NAME], chunksize=25000, dtype=str, keep_default_na=False
        ):
            before = _curate_bj_metadata(raw)
            by_scope = (
                pd.to_numeric(before.n_replicates_detected)
                .groupby(before.fractionation_protocol_id)
                .max()
            )
            for scope, value in by_scope.items():
                maxima[scope] = max(maxima[scope], int(value))
        counts, total, stable_hash = Counter(), Counter(), hashlib.sha256()
        with (
            Path(outputs["peptides.csv.gz"]).open("wb") as target,
            gzip.GzipFile(filename="", fileobj=target, mode="wb", mtime=0) as zipped,
            io.TextIOWrapper(zipped, encoding="utf-8", newline="") as stream,
        ):
            for raw in pd.read_csv(
                acquired[SOURCE_NAME], chunksize=25000, dtype=str, keep_default_na=False
            ):
                before = _curate_bj_metadata(raw)
                after = curate_replicate_counts(before)
                stable = before.columns.drop("n_replicates_detected")
                pd.testing.assert_frame_equal(before[stable], after[stable])
                pd.testing.assert_series_equal(
                    before.n_replicates_detected,
                    after.reported_n_replicates_detected,
                    check_names=False,
                )
                pd.testing.assert_frame_equal(after, curate_replicate_counts(after))
                assert after.replicate_protocol_id.ne("").all()
                stable_hash.update(before[stable].to_csv(index=False, header=False).encode())
                after["legacy_inferred_n_replicates_possible"] = after.replicate_protocol_id.map(
                    maxima
                )
                after["legacy_inferred_reproducibility_fraction"] = (
                    pd.to_numeric(after.reported_n_replicates_detected)
                    / after.legacy_inferred_n_replicates_possible
                )
                after["denominator_changed"] = after.n_replicates_possible.ne(
                    after.legacy_inferred_n_replicates_possible
                )
                for protocol, group in after.groupby("replicate_protocol_id", sort=False):
                    metrics = {
                        "n_rows": len(group),
                        "n_unknown_numerators": int(group.n_experiments_detected.isna().sum()),
                        "n_changed_denominators": int(group.denominator_changed.sum()),
                    }
                    for cutoff, label in [(0, "zero"), (0.5, "half"), (1, "all")]:
                        metrics["n_legacy_pass_" + label] = int(
                            group.legacy_inferred_reproducibility_fraction.ge(cutoff).sum()
                        )
                        metrics["n_qualified_pass_" + label] = int(
                            group.reproducibility_fraction.ge(cutoff).sum()
                        )
                    for name, value in metrics.items():
                        counts[(protocol, name)] += value
                        total[name] += value
                after.to_csv(stream, index=False, header=total["n_rows"] == len(after))
        pd.DataFrame(
            [
                {
                    "protocol_id": scope["protocol_id"],
                    **{name: counts[(scope["protocol_id"], name)] for name in total},
                }
                for scope in registry["scopes"]
            ]
        ).to_csv(outputs["scope-counts.tsv"], sep="\t", index=False)
        pd.DataFrame(evidence).to_csv(outputs["scope-evidence.tsv"], sep="\t", index=False)
        write(outputs["source-registry.json"], sources)
        for name in output_names[5:]:
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
                "counts": dict(total),
                "n_reviewed_scopes": len(evidence),
                "stable_columns": stable.tolist(),
                "stable_row_order_sha256": stable_hash.hexdigest(),
                "historical_maxima_by_scope": dict(maxima),
                "source": sources[SOURCE_NAME],
                "corrected": file_digest(outputs["peptides.csv.gz"]),
                "summary": {"archive": registry["provenance"]["archive"], **spec},
                "preservation": "All rows/order, non-count quantities and reviewed pH/reference evidence unchanged; original detected counts retained separately.",
                "scope": "Original-analysis bulk MS, not MHC ligands; MHC allele set not applicable. Experiment labels and acquisition runs are not biological replicates.",
                "limitations": "HeLa 46 pooled per-peptide counts unresolved; biological replicate/donor/specimen independence unknown. No new fitting or PXD013455 artifact changes.",
            },
        )

    outputs = materialize(
        args.cache_root / "audit" / commit,
        inputs,
        transform={
            "version": "bulk-experiment-counts-v1",
            "options": {"commit": commit, "hitlist_version": __version__, "pandas": pd.__version__},
        },
        outputs={name: {} for name in output_names},
        builder=build,
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
