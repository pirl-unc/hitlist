"""Replay #668 consumer acceptance against an already installed released artifact."""

import argparse
import hashlib
import json
import socket
import time
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from orfdata import DatasetRegistry, ScopedTarget

from hitlist import curate_canine_bulk, load_bulk_evidence, open_canine_peptide_evidence


def offline(*args, **kwargs):
    raise AssertionError("Offline consumer attempted network access")


def relation_digest(dataset):
    digest = hashlib.sha256()
    n_relations = 0
    for value in dataset.relations():
        digest.update(json.dumps(value.to_dict(), sort_keys=True, separators=(",", ":")).encode())
        digest.update(b"\n")
        n_relations += 1
    return {"n_relations": n_relations, "sha256": digest.hexdigest()}


def verify(cache_root, output):
    """Create a fresh frozen bundle, then compare its public views with upstream."""
    started = time.monotonic()
    with patch.object(socket, "socket", offline), patch.object(socket, "getaddrinfo", offline):
        curate_canine_bulk(
            "pfizer_2026_03",
            output,
            provider="orfdata_peptides",
            orfdata_cache_root=cache_root,
        )
        reader = open_canine_peptide_evidence(output, verify=False)
        registry = DatasetRegistry(cache_root=cache_root)
        upstream = registry.open(
            "pfizer-atlas-dog-peptide-evidence", "2026-10-27.1", verify_files=False
        )
        assert upstream.path.stat().st_ino != reader.dataset.path.stat().st_ino
        sequence = "TLGPQDMLWMLDPQEDGGK"
        target = ScopedTarget(
            upstream.manifest.references[0], "peptide", sequence, "amino-acid-sequence"
        )
        values = list(upstream.measurements(target=target))
        frame = load_bulk_evidence(output, peptide=sequence)
        assert frame.measurement_id.is_unique
        assert set(frame.measurement_id) == {value.measurement_id for value in values}
        rows = frame.set_index("measurement_id", drop=False)
        samples = {sample.sample_id: sample.to_dict() for sample in upstream.samples()}
        for value in values:
            row = rows.loc[value.measurement_id]
            for field, expected in value.to_dict().items():
                observed = row[field]
                assert pd.isna(observed) if expected is None else observed == expected, field
            assert row["sample_context"] == samples[value.sample_id]
        assert not frame.is_ms_observation.any()
        assert frame.mhc_restriction.eq("").all()
        assert frame.donor_id.isna().all()
        assert {
            sample.sample_id: sample.to_dict() for sample in reader.dataset.samples()
        } == samples
        expected = [value.to_dict() for value in upstream.mappings(source_identifier=sequence)]
        observed = load_bulk_evidence(output, table="mappings", peptide=sequence)
        assert expected
        assert [
            {key: row[key] for key in expected[0]} for row in observed.to_dict("records")
        ] == expected
        roles = Counter(sample.attributes["role"] for sample in reader.dataset.samples())
        relations = relation_digest(upstream)
        assert relation_digest(reader.dataset) == relations
        assert roles == {"tissue_pool": 38, "bridge": 4, "empty": 2}
        assert reader.dataset.counts == upstream.counts
        return {
            "state": "passed",
            "provider": "orfdata_peptides",
            "dataset_id": upstream.manifest.dataset_id,
            "dataset_version": upstream.manifest.version,
            "artifact": reader.manifest["artifacts"]["dataset.sqlite"],
            "artifact_verification": (
                "complete released artifact hash verified in the installed cache and on the "
                "independent snapshot; every retained byte is unchanged"
            ),
            "n_native_rows": sum(reader.native.metadata["reconciliation"]["counts"].values()),
            "n_native_tables": len(reader.native.tables),
            "logical_counts": reader.dataset.counts,
            "sample_roles": dict(roles),
            "all_sample_contexts_match": True,
            "n_searched_sequences": reader.native.search_reference.expected_sequences,
            "search_reference": reader.native.search_reference.to_dict(),
            "same_study_summary": reader.native.metadata["protein_summary"],
            "selected_query": {
                "peptide": sequence,
                "n_measurements": len(values),
                "n_mappings": len(expected),
                "every_measurement_and_mapping_field_matches": True,
            },
            "all_group_relations": relations,
            "network": "disabled during curation and queries",
            "source_validation_scope": (
                "Consumes upstream validated source output; no new vendor extraction or "
                "independent full vendor audit was performed. Full installed artifact integrity, "
                "all common count metadata and sample contexts, every group relation and the "
                "stated selected query were checked."
            ),
            "elapsed_seconds": time.monotonic() - started,
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New frozen-bundle directory")
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    receipt = verify(args.cache_root, args.output)
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2))
