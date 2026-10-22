"""Frozen assay-level training bundles for human-origin peptide T-cell assays."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from .curation_yaml import load_curation_yaml
from .tcell_assays import CATALOG_PATH
from .tcell_functional import FUNCTIONAL_SCHEMA, write_human_tcell_functional_assays
from .tcell_multimer import MULTIMER_SCHEMA, write_human_tcell_multimer_assays
from .version import __version__

TASK = "assay_specific_tcell_qualitative_response"
FORMAT = "hitlist-human-tcell-training-v1"
MULTIMER_TASK = "assay_specific_pmhc_multimer_cell_binding"
MULTIMER_FORMAT = "hitlist-human-tcell-multimer-training-v1"
TRAINING_COLUMNS = (
    "example_id",
    "example_unit",
    "task",
    "target_value",
    "source_qualitative_result",
    "label_consistency_status",
    "peptide",
    "antigen_format",
    "mhc_allele",
    "reported_mhc_name",
    "mhc_assignment_status",
    "restriction_evidence_code",
    "assay_family",
    "assay_method",
    "response_measured",
    "quantitative_measurement",
    "measurement_inequality",
    "measurement_units",
    "n_subjects_tested",
    "n_subjects_positive",
    "host_name",
    "in_vitro_process_type",
    "in_vitro_responder_cell",
    "in_vitro_stimulator_cell",
    "effector_cell_name",
    "effector_culture_condition",
    "apc_name",
    "apc_culture_condition",
    "source_assay_id",
    "source_dataset",
    "source_reference_id",
    "reference_pmid",
    "study_group_id",
    "cedar_present",
    "specimen_identity_status",
    "receptor_identity_status",
)
TRAINING_SCHEMA = pa.schema(
    [
        (
            name,
            pa.bool_()
            if name in {"target_value", "cedar_present"}
            else pa.int64()
            if name in {"n_subjects_tested", "n_subjects_positive"}
            else pa.string(),
        )
        for name in TRAINING_COLUMNS
    ]
)
EXCLUDED_SCHEMA = pa.schema(
    [
        ("source_assay_id", pa.string()),
        ("source_reference_id", pa.string()),
        ("reason", pa.string()),
    ]
)


def _subject_count(value: str) -> int | None:
    if not value:
        return None
    if not value.isascii() or not value.isdecimal():
        raise ValueError("Invalid reported subject count")
    count = int(value)
    if count > 2**63 - 1:
        raise ValueError("Reported subject count exceeds int64")
    return count


def _route(
    row: dict, *, task: str = TASK, observed_field: str = "response_observed"
) -> tuple[dict | None, str | None]:
    """Separate ambiguous binary targets without changing source evidence."""
    if row[observed_field] != (row["qualitative_measurement"] != "Negative"):
        raise ValueError("Source qualitative result and derived response disagree")
    try:
        n_tested = _subject_count(row["reported_n_subjects_tested"])
        n_positive = _subject_count(row["reported_n_subjects_positive"])
    except ValueError:
        return None, "invalid_reported_subject_count"
    if n_tested is not None and n_positive is not None and n_positive > n_tested:
        return None, "positive_subjects_exceed_tested"
    target = row[observed_field]
    if n_positive is not None and target != (n_positive > 0):
        return None, "qualitative_subject_count_disagreement"
    mhc_allele = (
        row["reported_mhc_name"]
        if row["mhc_assignment_status"] == "single_hla_experimental_restriction_evidence"
        else None
    )
    return (
        {
            "example_id": f"iedb:tcell:{row['source_assay_id']}",
            "example_unit": "assay",
            "task": task,
            "target_value": target,
            "source_qualitative_result": row["qualitative_measurement"],
            "label_consistency_status": (
                "agrees_with_reported_subject_count"
                if n_positive is not None
                else "subject_count_not_reported"
            ),
            "peptide": row["peptide_sequence"],
            "antigen_format": "direct_linear_peptide",
            "mhc_allele": mhc_allele,
            "reported_mhc_name": row["reported_mhc_name"],
            "mhc_assignment_status": row["mhc_assignment_status"],
            "restriction_evidence_code": row["restriction_evidence_code"],
            "assay_family": row["assay_family"],
            "assay_method": row["assay_method"],
            "response_measured": row["response_measured"],
            "quantitative_measurement": row["quantitative_measurement"],
            "measurement_inequality": row["measurement_inequality"],
            "measurement_units": row["measurement_units"],
            "n_subjects_tested": n_tested,
            "n_subjects_positive": n_positive,
            "host_name": row["host_name"],
            "in_vitro_process_type": row["in_vitro_process_type"],
            "in_vitro_responder_cell": row["in_vitro_responder_cell"],
            "in_vitro_stimulator_cell": row["in_vitro_stimulator_cell"],
            "effector_cell_name": row["effector_cell_name"],
            "effector_culture_condition": row["effector_culture_condition"],
            "apc_name": row["apc_name"],
            "apc_culture_condition": row["apc_culture_condition"],
            "source_assay_id": row["source_assay_id"],
            "source_dataset": "IEDB",
            "source_reference_id": row["source_reference_id"],
            "reference_pmid": row["reference_pmid"],
            "study_group_id": row["source_reference_id"],
            "cedar_present": row["cedar_present"],
            "specimen_identity_status": row["specimen_identity_status"],
            "receptor_identity_status": row["receptor_identity_status"],
        },
        None,
    )


def _file_digest(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return {"sha256": digest.hexdigest(), "size_bytes": path.stat().st_size}


def _flush(writer, path: Path, rows: list[dict], schema: pa.Schema):
    if writer is None:
        writer = pq.ParquetWriter(path, schema, compression="zstd")
    writer.write_table(pa.Table.from_pylist(rows, schema=schema))
    rows.clear()
    return writer


def _write_training_bundle(
    output_dir,
    *,
    evidence_writer,
    format_name,
    task,
    observed_field,
    iedb_path=None,
    cedar_path=None,
    data_root=None,
    catalog_path=None,
    batch_size=25000,
) -> dict:
    """Write source evidence, typed training rows, exclusions and manifest."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{destination.name}-", dir=destination.parent) as tmp:
        staging = Path(tmp)
        evidence_manifest = evidence_writer(
            staging / "evidence",
            iedb_path=iedb_path,
            cedar_path=cedar_path,
            data_root=data_root,
            catalog_path=catalog_path,
            batch_size=batch_size,
        )
        training_path = staging / "training.parquet"
        excluded_path = staging / "excluded.parquet"
        training_writer = excluded_writer = None
        training_rows, excluded_rows = [], []
        counts = Counter()
        families = defaultdict(Counter)
        studies = set()
        try:
            source = pq.ParquetFile(staging / "evidence" / "assays.parquet")
            for batch in source.iter_batches(batch_size=batch_size):
                for row in batch.to_pylist():
                    example, reason = _route(row, task=task, observed_field=observed_field)
                    if reason is not None:
                        excluded_rows.append(
                            {
                                "source_assay_id": row["source_assay_id"],
                                "source_reference_id": row["source_reference_id"],
                                "reason": reason,
                            }
                        )
                        counts[f"excluded:{reason}"] += 1
                    else:
                        training_rows.append(example)
                        counts["n_training_examples"] += 1
                        counts["n_positive_examples"] += int(example["target_value"])
                        counts["n_examples_with_mhc_allele"] += int(
                            example["mhc_allele"] is not None
                        )
                        family = families[example["assay_family"]]
                        family["n_examples"] += 1
                        family["n_positive_examples"] += int(example["target_value"])
                        family["n_examples_with_mhc_allele"] += int(
                            example["mhc_allele"] is not None
                        )
                        studies.add(example["study_group_id"])
                if training_rows:
                    training_writer = _flush(
                        training_writer, training_path, training_rows, TRAINING_SCHEMA
                    )
                if excluded_rows:
                    excluded_writer = _flush(
                        excluded_writer, excluded_path, excluded_rows, EXCLUDED_SCHEMA
                    )
            if training_writer is None:
                training_writer = _flush(
                    training_writer, training_path, training_rows, TRAINING_SCHEMA
                )
            if excluded_writer is None:
                excluded_writer = _flush(
                    excluded_writer, excluded_path, excluded_rows, EXCLUDED_SCHEMA
                )
        finally:
            if training_writer is not None:
                training_writer.close()
            if excluded_writer is not None:
                excluded_writer.close()
        exclusion_counts = {
            key.removeprefix("excluded:"): value
            for key, value in sorted(counts.items())
            if key.startswith("excluded:")
        }
        n_excluded = sum(exclusion_counts.values())
        if counts["n_training_examples"] + n_excluded != evidence_manifest["n_assays"]:
            raise ValueError("Training routing omitted source assays")
        files = {
            name: _file_digest(staging / name)
            for name in (
                "evidence/assays.parquet",
                "evidence/manifest.json",
                "training.parquet",
                "excluded.parquet",
            )
        }
        family_coverage = {
            name: {
                "n_examples": value["n_examples"],
                "n_positive_examples": value["n_positive_examples"],
                "n_negative_examples": value["n_examples"] - value["n_positive_examples"],
                "n_examples_with_mhc_allele": value["n_examples_with_mhc_allele"],
            }
            for name, value in sorted(families.items())
        }
        manifest = {
            "format": format_name,
            "schema_version": 1,
            "hitlist_version": __version__,
            "parquet_batch_size": batch_size,
            "evidence_manifest": evidence_manifest,
            "files": files,
            "n_assays": evidence_manifest["n_assays"],
            "n_training_examples": counts["n_training_examples"],
            "n_positive_examples": counts["n_positive_examples"],
            "n_negative_examples": counts["n_training_examples"] - counts["n_positive_examples"],
            "n_excluded_assays": n_excluded,
            "exclusion_counts": exclusion_counts,
            "family_coverage": family_coverage,
            "n_examples_with_mhc_allele": counts["n_examples_with_mhc_allele"],
            "n_study_groups": len(studies),
            "target_policy": task,
            "mhc_policy": "single_hla_experimental_evidence_candidate_only",
            "split_policy": "unsplit_study_group_available_specimen_unknown",
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        os.replace(staging, destination)
    return manifest


def write_human_tcell_training_bundle(
    output_dir,
    *,
    task="functional",
    iedb_path=None,
    cedar_path=None,
    data_root=None,
    catalog_path=None,
    batch_size=25000,
) -> dict:
    """Write a human peptide assay bundle for ``functional`` or ``multimer``."""
    if task not in {"functional", "multimer"}:
        raise ValueError("task must be 'functional' or 'multimer'")
    is_multimer = task == "multimer"
    return _write_training_bundle(
        output_dir,
        evidence_writer=(
            write_human_tcell_multimer_assays
            if is_multimer
            else write_human_tcell_functional_assays
        ),
        format_name=MULTIMER_FORMAT if is_multimer else FORMAT,
        task=MULTIMER_TASK if is_multimer else TASK,
        observed_field="multimer_binding_observed" if is_multimer else "response_observed",
        iedb_path=iedb_path,
        cedar_path=cedar_path,
        data_root=data_root,
        catalog_path=catalog_path,
        batch_size=batch_size,
    )


def write_human_tcell_multimer_training_bundle(
    output_dir,
    *,
    iedb_path=None,
    cedar_path=None,
    data_root=None,
    catalog_path=None,
    batch_size=25000,
) -> dict:
    """Write a distinct qualitative multimer cell-binding training bundle."""
    return write_human_tcell_training_bundle(
        output_dir,
        task="multimer",
        iedb_path=iedb_path,
        cedar_path=cedar_path,
        data_root=data_root,
        catalog_path=catalog_path,
        batch_size=batch_size,
    )


def _verify_training_bundle(
    directory, *, format_name, task, observed_field, evidence_schema, catalog_path=None
) -> dict:
    """Verify a bundle's source pins, hashes, schema and row routing."""
    root = Path(directory)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("format") != format_name or manifest.get("schema_version") != 1:
        raise ValueError("Unknown T-cell training bundle format")
    catalog = load_curation_yaml(catalog_path or CATALOG_PATH)
    evidence = manifest["evidence_manifest"]
    if evidence["sources"] != catalog["files"]:
        raise ValueError("T-cell source pins differ from the catalog")
    if json.loads((root / "evidence" / "manifest.json").read_text()) != evidence:
        raise ValueError("Evidence manifest differs from bundle receipt")
    expected_files = {
        "evidence/assays.parquet",
        "evidence/manifest.json",
        "training.parquet",
        "excluded.parquet",
    }
    if set(manifest["files"]) != expected_files:
        raise ValueError("T-cell bundle file inventory differs")
    for name, expected in manifest["files"].items():
        if _file_digest(root / name) != expected:
            raise ValueError(f"T-cell bundle file hash differs: {name}")
    if evidence["parquet_sha256"] != manifest["files"]["evidence/assays.parquet"]["sha256"]:
        raise ValueError("Evidence Parquet hash differs from its source receipt")
    assays = pq.read_table(root / "evidence" / "assays.parquet")
    training = pq.read_table(root / "training.parquet")
    excluded = pq.read_table(root / "excluded.parquet")
    if (
        assays.schema != evidence_schema
        or training.schema != TRAINING_SCHEMA
        or excluded.schema != EXCLUDED_SCHEMA
    ):
        raise ValueError("T-cell training bundle schema differs")
    if assays.num_rows != manifest["n_assays"] or assays.num_rows != evidence["n_assays"]:
        raise ValueError("T-cell assay count differs")
    if training.num_rows != manifest["n_training_examples"]:
        raise ValueError("T-cell training count differs")
    if excluded.num_rows != manifest["n_excluded_assays"]:
        raise ValueError("T-cell excluded count differs")
    source_ids = assays.column("source_assay_id").to_pylist()
    train_ids = training.column("source_assay_id").to_pylist()
    excluded_ids = excluded.column("source_assay_id").to_pylist()
    if (
        len(source_ids) != len(set(source_ids))
        or len(train_ids) != len(set(train_ids))
        or len(excluded_ids) != len(set(excluded_ids))
        or set(source_ids) != set(train_ids) | set(excluded_ids)
        or set(train_ids) & set(excluded_ids)
    ):
        raise ValueError("T-cell assays are duplicated or missing from routing")
    expected_training, expected_excluded = [], []
    for row in assays.to_pylist():
        example, reason = _route(row, task=task, observed_field=observed_field)
        if reason is None:
            expected_training.append(example)
        else:
            expected_excluded.append(
                {
                    "source_assay_id": row["source_assay_id"],
                    "source_reference_id": row["source_reference_id"],
                    "reason": reason,
                }
            )
    if training.to_pylist() != expected_training or excluded.to_pylist() != expected_excluded:
        raise ValueError("T-cell training routing differs from source evidence")
    if manifest["exclusion_counts"] != dict(Counter(row["reason"] for row in expected_excluded)):
        raise ValueError("T-cell exclusion counts differ")
    if manifest["n_examples_with_mhc_allele"] != sum(
        row["mhc_allele"] is not None for row in expected_training
    ):
        raise ValueError("T-cell MHC coverage differs")
    positives = sum(row["target_value"] for row in expected_training)
    if (
        manifest["n_positive_examples"] != positives
        or manifest["n_negative_examples"] != len(expected_training) - positives
    ):
        raise ValueError("T-cell target coverage differs")
    expected_families = defaultdict(Counter)
    for row in expected_training:
        family = expected_families[row["assay_family"]]
        family["n_examples"] += 1
        family["n_positive_examples"] += int(row["target_value"])
        family["n_examples_with_mhc_allele"] += int(row["mhc_allele"] is not None)
    coverage = {
        name: {
            "n_examples": value["n_examples"],
            "n_positive_examples": value["n_positive_examples"],
            "n_negative_examples": value["n_examples"] - value["n_positive_examples"],
            "n_examples_with_mhc_allele": value["n_examples_with_mhc_allele"],
        }
        for name, value in sorted(expected_families.items())
    }
    if manifest["family_coverage"] != coverage:
        raise ValueError("T-cell family coverage differs")
    if manifest["n_study_groups"] != len({row["study_group_id"] for row in expected_training}):
        raise ValueError("T-cell study group count differs")
    if manifest["target_policy"] != task:
        raise ValueError("T-cell target policy differs")
    return manifest


def verify_human_tcell_training_bundle(directory, *, task="functional", catalog_path=None) -> dict:
    """Verify a ``functional`` or ``multimer`` bundle against source evidence."""
    if task not in {"functional", "multimer"}:
        raise ValueError("task must be 'functional' or 'multimer'")
    is_multimer = task == "multimer"
    return _verify_training_bundle(
        directory,
        format_name=MULTIMER_FORMAT if is_multimer else FORMAT,
        task=MULTIMER_TASK if is_multimer else TASK,
        observed_field="multimer_binding_observed" if is_multimer else "response_observed",
        evidence_schema=MULTIMER_SCHEMA if is_multimer else FUNCTIONAL_SCHEMA,
        catalog_path=catalog_path,
    )


def verify_human_tcell_multimer_training_bundle(directory, *, catalog_path=None) -> dict:
    """Verify the multimer cell-binding bundle against source evidence and pins."""
    return verify_human_tcell_training_bundle(directory, task="multimer", catalog_path=catalog_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--task", choices=("functional", "multimer"), default="functional")
    parser.add_argument("--iedb", type=Path)
    parser.add_argument("--cedar", type=Path)
    args = parser.parse_args()
    manifest = write_human_tcell_training_bundle(
        args.output_dir, task=args.task, iedb_path=args.iedb, cedar_path=args.cedar
    )
    print(
        json.dumps(
            {
                key: manifest[key]
                for key in (
                    "n_training_examples",
                    "n_excluded_assays",
                    "n_examples_with_mhc_allele",
                )
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
