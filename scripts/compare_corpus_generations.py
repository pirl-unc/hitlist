"""Publish a bounded, citable field audit of aligned frozen observation generations."""

import argparse
import csv
import hashlib
import itertools
import json
import time
from collections import Counter
from pathlib import Path

KEYS = ("peptide", "pmid", "assay_iri", "provenance_id", "source", "supplementary_file")
CONTEXT = (*KEYS, "reference_iri", "reference_title")


def pin(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            digest.update(block)
    return {"sha256": digest.hexdigest(), "size_bytes": path.stat().st_size}


def compare(old_path, new_path, outputs):
    import pandas as pd
    import pyarrow.parquet as pq

    old, new = pq.ParquetFile(old_path), pq.ParquetFile(new_path)
    if not old.schema_arrow.equals(new.schema_arrow, check_metadata=False):
        raise ValueError("Observation schemas differ; an explicit schema audit is required")
    if old.metadata.num_rows != new.metadata.num_rows:
        raise ValueError("Observation counts differ; an explicit identity join is required")
    fields = old.schema_arrow.names
    if not set(CONTEXT) <= set(fields):
        raise ValueError("Missing source identity/citation fields")
    keys = (*KEYS, "evidence_row_id") if "evidence_row_id" in fields else KEYS
    context_fields = (*CONTEXT, "evidence_row_id") if "evidence_row_id" in fields else CONTEXT
    field_counts, study_counts = Counter(), Counter()
    n_rows = n_changed = 0
    started = time.monotonic()
    with Path(outputs["changed-cells.tsv"]).open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["row_index", *context_fields, "field", "original_value", "corrected_value"],
            delimiter="\t",
        )
        writer.writeheader()
        batches = itertools.zip_longest(
            old.iter_batches(batch_size=8192), new.iter_batches(batch_size=8192)
        )
        for old_batch, new_batch in batches:
            if old_batch is None or new_batch is None:
                raise ValueError("Observation batch counts differ")
            left, right = old_batch.to_pandas(), new_batch.to_pandas()
            for field in fields:
                if isinstance(left[field].dtype, pd.CategoricalDtype):
                    left[field] = left[field].astype(object)
                    right[field] = right[field].astype(object)
            equal = (left.eq(right) | (left.isna() & right.isna())).fillna(False)
            if not equal[list(keys)].all().all():
                raise ValueError("Observation identity/order changed; explicit join required")
            changed = ~equal
            n_changed += int(changed.any(axis=1).sum())
            for field in fields:
                indexes = changed.index[changed[field]]
                field_counts[field] += len(indexes)
                for index in indexes:
                    context = {
                        key: "" if pd.isna(left.at[index, key]) else left.at[index, key]
                        for key in context_fields
                    }
                    study_counts[str(context["pmid"]), field] += 1
                    writer.writerow(
                        {
                            "row_index": n_rows + int(index),
                            **context,
                            "field": field,
                            "original_value": json.dumps(
                                None if pd.isna(left.at[index, field]) else left.at[index, field],
                                ensure_ascii=False,
                                default=str,
                            ),
                            "corrected_value": json.dumps(
                                None if pd.isna(right.at[index, field]) else right.at[index, field],
                                ensure_ascii=False,
                                default=str,
                            ),
                        }
                    )
            n_rows += len(left)
    if n_rows != old.metadata.num_rows:
        raise ValueError("Comparison did not cover every observation")
    pd.DataFrame(
        [{"field": field, "n_changed_observations": field_counts[field]} for field in fields]
    ).to_csv(outputs["field-counts.tsv"], sep="\t", index=False)
    pd.DataFrame(
        [
            {"pmid": pmid, "field": field, "n_changed_observations": count}
            for (pmid, field), count in sorted(study_counts.items())
        ],
        columns=["pmid", "field", "n_changed_observations"],
    ).to_csv(outputs["study-counts.tsv"], sep="\t", index=False)
    return {
        "n_observations": n_rows,
        "n_changed_observations": n_changed,
        "n_changed_cells": sum(field_counts.values()),
        "n_fields": len(fields),
        "changed_fields": {key: value for key, value in field_counts.items() if value},
        "identity_fields": list(keys),
        "identity_and_order_unchanged": True,
        "runtime_seconds": time.monotonic() - started,
        "n_rows_per_batch": 8192,
    }


def main():
    from datacache import materialize

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    inputs = json.loads(args.inputs.read_text())
    for value in inputs.values():
        if pin(value["path"]) != {key: value[key] for key in ("sha256", "size_bytes")}:
            raise ValueError("Observation input does not match its frozen pin")
    recipe = pin(__file__)

    def build(source, outputs):
        summary = compare(source["original"], source["corrected"], outputs)
        summary.update(inputs=inputs, recipe=recipe)
        Path(outputs["summary.json"]).write_text(json.dumps(summary, indent=2) + "\n")

    paths = materialize(
        args.cache_dir,
        {
            name: {"path": value["path"], "sha256": value["sha256"], "size": value["size_bytes"]}
            for name, value in inputs.items()
        },
        transform={"version": "aligned-corpus-field-audit-v1", "options": {"recipe": recipe}},
        outputs={
            name: {}
            for name in (
                "changed-cells.tsv",
                "field-counts.tsv",
                "study-counts.tsv",
                "summary.json",
            )
        },
        builder=build,
    )
    receipt = {name: {"path": str(path), **pin(path)} for name, path in paths.items()}
    args.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
