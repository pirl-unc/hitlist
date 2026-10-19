"""Publish a reproducible, donor-disjoint primary Atlas source cohort."""

import argparse
import gzip
import hashlib
import io
import itertools
import json
import platform
import resource
import shutil
import subprocess
import time
from pathlib import Path

import pandas as pd
from datacache import materialize

from hitlist.atlas import CATALOG_PATH, atlas_sources, load_atlas_observations
from hitlist.curation_yaml import load_curation_yaml
from hitlist.provenance import file_digest
from hitlist.version import __version__

ROOT = Path(__file__).resolve().parents[1]
SEED = "atlas-donor-holdout-v1"


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_tsv(path, frame):
    with (
        Path(path).open("wb") as raw,
        gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped,
        io.TextIOWrapper(zipped, encoding="utf-8", newline="") as text,
    ):
        frame.to_csv(text, sep="\t", index=False, chunksize=10000)


def donor_summary(rows):
    grouped = rows.groupby(["donor_id", "partition"], observed=True)
    if grouped.sample_alleles.nunique().ne(1).any():
        raise ValueError("A donor group has inconsistent reported typing")
    return (
        grouped.agg(
            n_observations=("evidence_row_id", "size"),
            n_peptides=("peptide", "nunique"),
            n_tissues=("source_tissue", "nunique"),
            n_reported_sample_groups=("reported_sample_group_id", "nunique"),
            sample_alleles=("sample_alleles", "first"),
        )
        .reset_index()
        .sort_values("donor_id")
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", required=True, type=Path)
    args = parser.parse_args()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    assert not subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)
    paths = atlas_sources()
    source_specs = load_curation_yaml(CATALOG_PATH)["files"]
    code_paths = {
        "audit.py": Path(__file__),
        "atlas.py": ROOT / "hitlist/atlas.py",
        "atlas_reader.py": ROOT / "hitlist/tissue_blacklist.py",
        "source_catalog.yaml": CATALOG_PATH,
    }
    inputs = {
        name: {
            "path": str(path),
            "sha256": file_digest(path)["sha256"],
            "size": Path(path).stat().st_size,
        }
        for name, path in {**paths, **code_paths}.items()
    }
    outputs = [
        *inputs,
        "observations.tsv.gz",
        "donors.tsv",
        "sample-groups.tsv",
        "split-policy.json",
        "label-policy.json",
        "coverage.json",
        "overlap.json",
        "shared-peptides.tsv.gz",
        "source-registry.json",
        "audit.json",
    ]

    def build(acquired, target):
        started = time.monotonic()
        rows, sources = load_atlas_observations(Path(acquired["peptides.tsv.gz"]).parent)
        donors = sorted(
            rows.donor_id.unique(),
            key=lambda d: hashlib.sha256((SEED + "\n" + d).encode()).hexdigest(),
        )
        assert len(donors) == 21
        assignments = {
            donor: "test" if i < 3 else "val" if i < 6 else "train"
            for i, donor in enumerate(donors)
        }
        rows["partition"] = rows.donor_id.astype(str).map(assignments)
        assert rows.partition.notna().all()
        write_tsv(target["observations.tsv.gz"], rows)
        donor_table = donor_summary(rows)
        donor_table.to_csv(target["donors.tsv"], sep="\t", index=False)
        groups = (
            rows.groupby(
                ["reported_sample_group_id", "donor_id", "source_tissue", "mhc_class", "partition"],
                observed=True,
            )
            .agg(n_observations=("evidence_row_id", "size"), n_peptides=("peptide", "nunique"))
            .reset_index()
        )
        groups.to_csv(target["sample-groups.tsv"], sep="\t", index=False)
        membership = {p: rows.loc[rows.partition.eq(p)] for p in ("train", "val", "test")}
        coverage = {}
        for partition, frame in membership.items():
            coverage[partition] = {
                "n_observations": len(frame),
                "n_peptides": frame.peptide.nunique(),
                "n_donors": frame.donor_id.nunique(),
                "n_reported_sample_groups": frame.reported_sample_group_id.nunique(),
                "n_tissues": frame.source_tissue.nunique(),
                "n_unknown_donor_observations": 0,
                "n_unknown_specimen_observations": len(frame),
                "n_unknown_acquisition_observations": len(frame),
                "n_observations_by_mhc_class": frame.mhc_class.value_counts().to_dict(),
            }
        overlaps, shared = [], []
        for left, right in itertools.combinations(sorted(membership), 2):
            a, b = membership[left], membership[right]
            metrics = {"left": left, "right": right}
            for axis in ["donor_id", "reported_sample_group_id", "evidence_row_id", "peptide"]:
                common = sorted(set(a[axis]) & set(b[axis]))
                metrics[
                    "n_shared_"
                    + {
                        "donor_id": "donors",
                        "reported_sample_group_id": "sample_groups",
                        "evidence_row_id": "observations",
                        "peptide": "peptides",
                    }[axis]
                ] = len(common)
                if axis == "peptide":
                    shared.extend(
                        {"left": left, "right": right, "peptide": value} for value in common
                    )
                else:
                    assert not common
            overlaps.append(metrics)
        write_tsv(
            target["shared-peptides.tsv.gz"],
            pd.DataFrame(shared, columns=["left", "right", "peptide"]),
        )
        write(target["coverage.json"], coverage)
        write(
            target["overlap.json"],
            {
                "donor_disjoint_verdict": "pass",
                "reported_sample_group_disjoint_verdict": "pass",
                "specimen_identity_verdict": "inconclusive",
                "acquisition_identity_verdict": "inconclusive",
                "peptide_disjoint_verdict": "fail" if shared else "pass",
                "pairs": overlaps,
                "scope": "This primary Atlas release only; no independence from IEDB mirrors, other studies or historical fitting cohorts is established.",
            },
        )
        write(
            target["split-policy.json"],
            {
                "algorithm": "sha256_seed_newline_donor_id_lexical_ascending_v1",
                "seed": SEED,
                "allocation": {
                    "test": "first 3 donors",
                    "val": "next 3 donors",
                    "train": "remaining 15 donors",
                },
                "assignments": assignments,
                "label_blind": True,
                "claim": "donor-disjoint positive-only primary source cohort; peptide overlap is allowed and reported",
                "historical_partitions_changed": False,
                "model_fitting_performed": False,
            },
        )
        write(
            target["label-policy.json"],
            {
                "assay": "cellular MHC/mass spectrometry",
                "endpoint": "observed eluted ligand",
                "positive_only": True,
                "absence_is_negative": False,
                "generated_negatives": False,
                "donor_typing_is_peptide_restriction": False,
                "processing_context_applicability": "applicable_to_cellular_presentation",
                "physical_specimen_and_acquisition_identity": "unknown",
                "condition": "benign primary tissue; direct ex vivo; cell type unspecified",
                "upstream_qc": "Source used binding predictions during sample quality control; predictions are not measured restriction labels.",
                "cautions": sources["limits"],
            },
        )
        for value in sources["files"].values():
            value.pop("path", None)
        sources["pinned_acquisition"] = source_specs
        write(target["source-registry.json"], sources)
        for name in inputs:
            shutil.copyfile(acquired[name], target[name])
        write(
            target["audit.json"],
            {
                "commit": commit,
                "dirty": False,
                "agent": "codex",
                "model": "GPT-6",
                "hitlist_version": __version__,
                "python": platform.python_version(),
                "pandas": pd.__version__,
                "n_observations": len(rows),
                "n_donors": len(donors),
                "n_peptides": rows.peptide.nunique(),
                "n_reported_sample_groups": len(groups),
                "n_tissues": rows.source_tissue.nunique(),
                "runtime_seconds": time.monotonic() - started,
                "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                * (1 if platform.system() == "Darwin" else 1024),
                "runtime_scope": "builder, imports/acquisition excluded",
                "execution": "CPU source curation and partition audit; no model, optimization or predictions",
                "code": {name: file_digest(path) for name, path in code_paths.items()},
            },
        )

    result = materialize(
        Path.home() / ".hitlist/atlas-lineage/audit" / commit,
        inputs,
        transform={
            "version": "primary-atlas-donor-cohort-v2",
            "options": {"commit": commit, "seed": SEED},
        },
        outputs={name: {} for name in outputs},
        builder=build,
    )
    receipt = {
        "commit": commit,
        "paths": {name: str(path) for name, path in result.items()},
        "files": {name: file_digest(path) for name, path in result.items()},
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    write(args.receipt, receipt)
    print(Path(result["audit.json"]).read_text())


if __name__ == "__main__":
    main()
