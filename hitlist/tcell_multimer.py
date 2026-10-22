"""Pinned human-origin peptide-MHC multimer binding evidence.

The source outcome is binding to tested cells, not a cellular response,
peptide-MHC affinity, or a paired-receptor measurement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter
from collections.abc import Iterator
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from .curation_yaml import load_curation_yaml
from .tcell_assays import CATALOG_PATH, EXPORT_COLUMNS
from .tcell_peptides import iter_human_direct_peptide_tcell_assays

MULTIMER_COLUMNS = (
    *EXPORT_COLUMNS,
    "peptide_sequence",
    "reported_mhc_name",
    "mhc_assignment_status",
    "multimer_binding_observed",
)
MULTIMER_SCHEMA = pa.schema(
    [
        (
            name,
            pa.bool_() if name in {"cedar_present", "multimer_binding_observed"} else pa.string(),
        )
        for name in MULTIMER_COLUMNS
    ]
)


def iter_human_tcell_multimer_assays(
    *, iedb_path=None, cedar_path=None, data_root=None, catalog_path=None
) -> Iterator[dict]:
    """Yield direct human peptide assays with measured multimer cell binding."""
    for row in iter_human_direct_peptide_tcell_assays(
        assay_families={"pmhc_multimer"},
        response_measured="qualitative binding",
        iedb_path=iedb_path,
        cedar_path=cedar_path,
        data_root=data_root,
        catalog_path=catalog_path,
    ):
        assay = row.copy()
        assay["multimer_binding_observed"] = assay.pop("qualitative_signal_observed")
        yield assay


def _write_batch(writer, path: Path, rows: list[dict]):
    if writer is None:
        writer = pq.ParquetWriter(path, MULTIMER_SCHEMA, compression="zstd")
    writer.write_table(pa.Table.from_pylist(rows, schema=MULTIMER_SCHEMA))
    return writer


def write_human_tcell_multimer_assays(
    output_dir,
    *,
    iedb_path=None,
    cedar_path=None,
    data_root=None,
    catalog_path=None,
    batch_size=25000,
) -> dict:
    """Write source-faithful multimer assay rows and a pinned manifest."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    catalog = load_curation_yaml(catalog_path or CATALOG_PATH)
    with tempfile.TemporaryDirectory(prefix=f".{destination.name}-", dir=destination.parent) as tmp:
        staging = Path(tmp)
        table_path = staging / "assays.parquet"
        counts = Counter()
        writer = None
        batch = []
        try:
            for row in iter_human_tcell_multimer_assays(
                iedb_path=iedb_path,
                cedar_path=cedar_path,
                data_root=data_root,
                catalog_path=catalog_path,
            ):
                batch.append(row)
                counts["n_assays"] += 1
                counts["n_cedar_assays"] += int(row["cedar_present"])
                counts["n_assays_with_experimental_restriction_evidence"] += int(
                    row["mhc_assignment_status"] == "single_hla_experimental_restriction_evidence"
                )
                if len(batch) >= batch_size:
                    writer = _write_batch(writer, table_path, batch)
                    batch.clear()
            if batch or writer is None:
                writer = _write_batch(writer, table_path, batch)
        finally:
            if writer is not None:
                writer.close()
        manifest = {
            "schema_version": 1,
            "source_observed": catalog["observed"],
            "sources": catalog["files"],
            "n_assays": counts["n_assays"],
            "n_cedar_assays": counts["n_cedar_assays"],
            "n_assays_with_experimental_restriction_evidence": counts[
                "n_assays_with_experimental_restriction_evidence"
            ],
            "parquet_sha256": hashlib.sha256(table_path.read_bytes()).hexdigest(),
            "target_policy": "reported_human_peptide_multimer_cell_binding",
            "restriction_policy": "reported_name_with_separate_experimental_evidence_status",
            "independence_policy": "study_id_available_specimen_id_unknown",
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        os.replace(staging, destination)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--iedb", type=Path)
    parser.add_argument("--cedar", type=Path)
    args = parser.parse_args()
    manifest = write_human_tcell_multimer_assays(
        args.output_dir, iedb_path=args.iedb, cedar_path=args.cedar
    )
    print(json.dumps({"n_assays": manifest["n_assays"]}, sort_keys=True))


if __name__ == "__main__":
    main()
