"""Export and inspect a bounded human-source cohort from the pinned current generation."""

from __future__ import annotations

import contextlib
import hashlib
import importlib.metadata
import json
import os
import resource
import shutil
import signal
import sys
import tempfile
import time
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

RUN_ID = "2d794b88453e75e4ae548326c4ffaaea05748d7df81140fcd24dbf0c8053ff71"
MAPPING_RECEIPT_SHA256 = "bb3177fea10370a13ff101f8babec125252896e450c84e5300968994ef7f02c0"
MAPPING_RECEIPT = "receipts/mappings/bundles/2026-10-16T02-46-56Z/receipt.json"
OPTIONS = {
    "include_evidence": "both",
    "species": "Homo sapiens",
    "source_species": "Homo sapiens",
    "length_min": 8,
    "length_max": 25,
    "map_source_proteins": False,
    "with_peptide_origin": False,
    "exclude_non_peptide_ligand": True,
}
NATIVE_FILES = (
    "training.parquet",
    "identities.parquet",
    "contributors.parquet",
    "lineage.json",
    "manifest.json",
)
REPORT_FILES = (
    "coverage.json",
    "source_contexts.tsv",
    "field_counts.tsv",
    "inspection.tsv",
    "README.md",
)
FIELDS = (
    "evidence_kind",
    "pmid",
    "reference_iri",
    "reference_title",
    "source",
    "supplementary_file",
    "mhc_class",
    "mhc_species",
    "source_organism",
    "host",
    "mhc_restriction",
    "mhc_allele_set",
    "mhc_allele_provenance",
    "restriction_evidence",
    "allele_resolution",
    "condition_id",
    "sample_attribution",
    "attributed_sample_label",
    "cell_type",
    "cell_line_name",
    "cell_name",
    "source_tissue",
    "culture_condition",
    "assay_method",
    "response_measured",
    "measurement_units",
    "measurement_inequality",
    "qualitative_measurement",
    "experimental_origin_status",
    "specimen_status",
    "donor_status",
    "acquisition_status",
)
CONTEXT_FIELDS = (
    "evidence_kind",
    "pmid",
    "reference_iri",
    "reference_title",
    "source",
    "mhc_class",
    "condition_id",
    "cell_type",
    "cell_line_name",
    "culture_condition",
    "mhc_allele_set",
    "mhc_restriction",
    "restriction_evidence",
    "assay_method",
    "response_measured",
    "measurement_units",
    "measurement_inequality",
)


def encode(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def digest(path):
    path = Path(path)
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            value.update(block)
    return {"sha256": value.hexdigest(), "size_bytes": path.stat().st_size}


def inspection_reports(directory):
    import pandas as pd
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(directory / "training.parquet")
    available = set(parquet.schema_arrow.names)
    fields = [name for name in FIELDS if name in available]
    contexts = [name for name in CONTEXT_FIELDS if name in available]
    preview_fields = [
        name
        for name in (
            "evidence_row_id",
            "provenance_id",
            "peptide",
            *FIELDS,
            "quantitative_measurement",
            "quantitative_value",
            "assay_iri",
            "experimental_origin_ids",
            "specimen_ids",
            "donor_ids",
        )
        if name in available
    ]
    counts = {name: Counter() for name in fields}
    context_counts, seen_ids, preview_counts, previews = Counter(), set(), Counter(), []
    for batch in parquet.iter_batches(batch_size=50000, columns=preview_fields):
        frame = batch.to_pandas()
        ids = frame.evidence_row_id.tolist()
        if len(ids) != len(set(ids)) or seen_ids.intersection(ids):
            raise ValueError(
                "Compact cohort has duplicate observation IDs; do not count them as independent"
            )
        seen_ids.update(ids)
        display = frame[fields].astype("string").fillna("<missing>").replace("", "<missing>")
        for name in fields:
            counts[name].update(
                {str(key): int(value) for key, value in display[name].value_counts().items()}
            )
        context_counts.update(
            {
                key: int(value)
                for key, value in display.groupby(contexts, observed=True, dropna=False)
                .size()
                .items()
            }
        )
        # A declared preview, not a representative sample: first 250 rows per evidence/class.
        for key, group in frame.groupby(
            ["evidence_kind", "mhc_class"], observed=True, dropna=False
        ):
            take = max(0, 250 - preview_counts[key])
            if take:
                part = group.head(take)
                previews.append(part)
                preview_counts[key] += len(part)
    n_observations = len(seen_ids)
    rows = [
        {"field": field, "value": value, "n_observations": count}
        for field, values in counts.items()
        for value, count in sorted(values.items())
    ]
    pd.DataFrame(rows).to_csv(directory / "field_counts.tsv", sep="\t", index=False)
    context_rows = [
        {**dict(zip(contexts, key)), "n_observations": count}
        for key, count in sorted(context_counts.items())
    ]
    pd.DataFrame(context_rows).to_csv(directory / "source_contexts.tsv", sep="\t", index=False)
    pd.concat(previews, ignore_index=True).to_csv(
        directory / "inspection.tsv", sep="\t", index=False
    )
    coverage = {
        "n_observations": n_observations,
        "n_training_rows": parquet.metadata.num_rows,
        "n_source_contexts": len(context_counts),
        "n_preview_rows": sum(preview_counts.values()),
        "preview_policy": "first 250 export-ordered observations per evidence kind and MHC class; not representative",
        "missing_display_value": "<missing>; original values remain in training.parquet",
        "fields": {
            field: {
                "n_nonempty_observations": n_observations - values.get("<missing>", 0),
                "n_missing_observations": values.get("<missing>", 0),
                "n_distinct_nonempty_values": len(values) - int("<missing>" in values),
            }
            for field, values in counts.items()
        },
        "unavailable_fields": sorted(set(FIELDS) - available),
        "lineage_statuses": {
            field: dict(counts[field]) for field in counts if field.endswith("_status")
        },
        "evidence_kinds": dict(counts["evidence_kind"]),
        "mhc_classes": dict(counts["mhc_class"]),
        "limitations": [
            "Coverage is observation-weighted, not evidence of donor/specimen independence.",
            "Compact observations do not supply protein/flank context; the parent corpus pins mappings.",
            "Only human MHC and reported human peptide sources, lengths 8-25, are included.",
            "No partition or model-quality claim is made by this export.",
        ],
    }
    (directory / "coverage.json").write_bytes(encode(coverage))
    (directory / "README.md").write_text(
        "# Frozen human-source HLA cohort\n\n"
        "The native manifest pins the exact producer, filter contract, input generation and all four "
        "training/identity/contributor/lineage sidecars. It records current-index admission at export time. "
        "Original assay endpoints, units and inequalities remain distinct; cellular MHC sets are not "
        "interchangeable with assay restrictions. No mapping or contributor expansion creates additional "
        "independent observations.\n\n"
        "| File | Human inspection purpose |\n| --- | --- |\n"
        "| source_contexts.tsv | Citation, cell/condition, MHC and assay combinations with observation counts |\n"
        "| field_counts.tsv | Complete value counts, including explicit missing values |\n"
        "| inspection.tsv | First 250 rows per evidence kind/class; a readable, nonrepresentative preview |\n"
        "| coverage.json | Observation-weighted metadata and biological-lineage coverage |\n"
        "| lineage.json | Reviewed identity registry and its primary-source evidence |\n\n"
        "Partitions, leakage audits and Presto performance are separate qualification stages. Unknown "
        "donors, specimens or experiments remain unknown. Raw-export dates are unverified. The source "
        "build and all metadata limits are recorded in Hitlist #692/#680/#678.\n"
    )
    return coverage


def main():
    root = Path("/data/hitlist-qualification/runs") / RUN_ID
    os.environ.update(
        HITLIST_DATA_DIR=str(root / "hitlist"),
        XDG_CACHE_HOME=str(root / "cache"),
        PYENSEMBL_CACHE_DIR=str(root / "references"),
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
    )
    from datacache import materialize, validate_file
    from qualify_corpus import Tee, validate

    import hitlist
    from hitlist import verify_training_bundle, write_training_bundle

    if hitlist.__version__ != "1.76.0":
        raise ValueError("Qualification requires the pinned published Hitlist 1.76.0 producer")
    receipt_path = root / MAPPING_RECEIPT
    recipe = digest(__file__)
    options = {
        "parent_run_id": RUN_ID,
        "mapping_receipt_sha256": MAPPING_RECEIPT_SHA256,
        "recipe_sha256": recipe["sha256"],
        "training_options": OPTIONS,
    }
    identity = hashlib.sha256(encode(options)).hexdigest()
    destination = root / "human-cohort" / identity
    result = {
        "options": options,
        "status": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "versions": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
        "cpu_limit": 8,
        "memory_limit_mib": 65536,
        "gpu": "none",
        "producer_import": hitlist.__file__,
    }
    started = time.monotonic()
    error = None

    def deadline(*_):
        raise TimeoutError("Human cohort qualification exceeded its 85-minute work limit")

    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(5100)
    with tempfile.TemporaryDirectory(prefix="human-cohort-") as temporary:
        log = Path(temporary) / "export.log"
        with (
            log.open("w") as stream,
            contextlib.redirect_stdout(Tee(sys.stdout, stream)),
            contextlib.redirect_stderr(Tee(sys.stderr, stream)),
        ):
            try:
                validate_file(receipt_path, MAPPING_RECEIPT_SHA256)
                receipt = json.loads(receipt_path.read_text())
                if receipt["status"] != "success":
                    raise ValueError("Parent mapping stage did not succeed")
                validate(receipt["artifacts"])

                def builder(inputs, outputs):
                    native = Path(temporary) / "native"
                    write_training_bundle(native, require_current_indexes=True, **OPTIONS)
                    manifest = verify_training_bundle(native)
                    result["native_coverage"] = manifest["coverage"]
                    result["coverage"] = inspection_reports(native)
                    for name in (*NATIVE_FILES, *REPORT_FILES):
                        shutil.copyfile(native / name, outputs[name])

                paths = materialize(
                    destination / "artifacts",
                    {
                        "parent-receipt.json": {
                            "path": str(receipt_path),
                            "sha256": MAPPING_RECEIPT_SHA256,
                        },
                        "recipe.py": {"path": __file__, "sha256": recipe["sha256"]},
                    },
                    transform={"version": "human-cohort-qualification-v1", "options": options},
                    outputs={name: {} for name in (*NATIVE_FILES, *REPORT_FILES)},
                    builder=builder,
                )
                result["coverage"] = json.loads(Path(paths["coverage.json"]).read_text())
                result["native_coverage"] = verify_training_bundle(
                    Path(paths["manifest.json"]).parent
                )["coverage"]
                result["artifacts"] = {
                    name: {"path": str(path), **digest(path)} for name, path in paths.items()
                }
                result["status"] = "success"
            except Exception as caught:
                error = caught
                result["status"] = "failed"
                result["error"] = f"{type(caught).__name__}: {caught}"
                traceback.print_exc()
        signal.alarm(0)
        result.update(
            runtime_seconds=time.monotonic() - started,
            process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            rss_scope="runner process only; child processes excluded",
            finished_utc=datetime.now(timezone.utc).isoformat(),
        )

        def save_receipt(inputs, outputs):
            Path(outputs["receipt.json"]).write_bytes(encode(result))
            shutil.copyfile(log, outputs["export.log"])

        saved = materialize(
            destination / "receipts",
            {},
            transform={"version": "human-cohort-receipt-v1", "options": options},
            outputs={"receipt.json": {}, "export.log": {}},
            builder=save_receipt,
            force=True,
        )
    print(
        encode(
            {
                "receipt": result,
                "receipt_files": {
                    name: {"path": str(path), **digest(path)} for name, path in saved.items()
                },
            }
        ).decode()
    )
    if error:
        raise RuntimeError(
            f"Human cohort failed; preserved receipt at {saved['receipt.json']}"
        ) from error


if __name__ == "__main__":
    main()
