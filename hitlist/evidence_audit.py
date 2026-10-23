"""Reconcile a verified package by source, endpoint, grain and field readiness."""

from __future__ import annotations

from collections import Counter

import pyarrow.dataset as ds
import pyarrow.parquet as pq

from .evidence_catalog import BASELINES

AGGREGATE_POPULATIONS = {
    "tcell_2026_10_20": ("tcell_availability",),
    "bulk_v4": ("bulk_ms_peptide", "bulk_ms_protein"),
    "line_expression_v4": ("line_expression_gene",),
    "pxd057588": ("external_peptide_source_row",),
}


def _counts(store, evidence_type, fields):
    item = store.bindings[evidence_type][0]
    expression = store._expression(evidence_type, {})
    scanner = ds.dataset(store.directory / item.materialized_artifact, format="parquet").scanner(
        columns=fields, filter=expression, batch_size=65_536
    )
    counts = Counter()
    for batch in scanner.to_batches():
        columns = batch.to_pydict()
        for values in zip(*(columns[field] for field in fields)):
            counts[values] += 1
    return counts


def audit_evidence_store(store):
    """Return a machine-readable, no-implicit-sum reconciliation report."""
    catalog = store.catalog()
    counts = {entry["name"]: entry["n_rows"] for entry in catalog["evidence"] if "n_rows" in entry}
    baselines = []
    for baseline in BASELINES:
        names = AGGREGATE_POPULATIONS.get(baseline.population, (baseline.population,))
        if any(name not in counts for name in names):
            baselines.append(
                {
                    "population": baseline.population,
                    "generation": baseline.generation,
                    "count_unit": baseline.count_unit,
                    "n_expected_rows": baseline.n_rows,
                    "n_actual_rows": None,
                    "n_revisions": None,
                    "status": "not_in_generation",
                }
            )
            continue
        actual = sum(counts[name] for name in names)
        n_revisions = 1
        if baseline.count_unit.endswith("_per_revision"):
            item = store.bindings[baseline.population][0]
            revision = pq.read_table(
                store.directory / item.materialized_artifact, columns=["curation_revision"]
            )["curation_revision"]
            n_revisions = len(set(revision.to_pylist()))
        expected = baseline.n_rows * n_revisions
        baselines.append(
            {
                "population": baseline.population,
                "generation": baseline.generation,
                "count_unit": baseline.count_unit,
                "n_expected_rows": expected,
                "n_actual_rows": actual,
                "n_revisions": n_revisions,
                "status": "matched" if actual == expected else "mismatch",
            }
        )
    dispositions = {}
    for entry in catalog["evidence"]:
        native_disposition = {}
        if entry["name"] in store.bindings:
            item, _, aliases = store.bindings[entry["name"]]
            for physical in store.schemas[item.name].names:
                logical = [field for field in item.fields if aliases.get(field, field) == physical]
                native_disposition[physical] = {
                    "logical_fields": logical,
                    "query_field": f"native.{physical}",
                    "status": "logical_and_native" if logical else "native_only",
                }
        dispositions[entry["name"]] = {
            "status": entry["status"],
            "grain": entry["grain"],
            "fields": entry["fields"],
            "n_native_fields": len(entry["native_fields"]),
            "native_field_disposition": native_disposition,
            "n_rows": entry.get("n_rows"),
        }
    by_source_endpoint = {}
    for name, (item, _, _) in store.bindings.items():
        schema = store.schemas[name]
        columns = [
            field
            for field in ("source_dataset", "endpoint", "representation")
            if field in schema.names
        ]
        if not columns:
            continue
        joint = _counts(store, name, columns)
        by_source_endpoint[name] = [
            {
                **dict(zip(columns, values)),
                "n_rows": count,
                "source_population": item.source_population,
                "grain": item.grain,
            }
            for values, count in sorted(joint.items(), key=lambda pair: repr(pair[0]))
        ]
    eligibility = _counts(
        store,
        "tcell_eligibility",
        ["assay_family", "task", "training_status", "first_failure_reason"],
    )
    tcell_funnel = [
        {
            "assay_family": family,
            "task": task,
            "training_status": status,
            "first_failure_reason": reason,
            "n_assays": count,
        }
        for (family, task, status, reason), count in sorted(
            eligibility.items(), key=lambda pair: repr(pair[0])
        )
    ]
    reference_funnel = []
    reference_components = None
    reference_support = []
    if "external_reference_source_mapping" in store.bindings:
        statuses = _counts(
            store,
            "external_reference_source_mapping",
            ["table_number", "measured_pool", "source_parent_status"],
        )
        reference_funnel = [
            {
                "table_number": table,
                "measured_pool": pool,
                "source_parent_status": status,
                "n_source_rows": count,
            }
            for (table, pool, status), count in sorted(statuses.items())
        ]
        sequence_item = store.bindings["external_reference_sequence"][0]
        values = pq.read_table(
            store.directory / sequence_item.materialized_artifact, columns=["component"]
        )["component"].to_pylist()
        groups = Counter(value for value in values if value)
        reference_components = {
            "n_components": len(groups),
            "n_sequences_largest_component": max(groups.values()),
            "n_distinct_sequences": len(values),
            "n_unmatched_sequences": sum(not value for value in values),
            "interpretation": "candidate split constraint, not donor or specimen identity",
        }
        reference_support = [
            {"definition": definition, "n_parent_support_sets": count}
            for (definition,), count in sorted(
                _counts(store, "external_reference_parent_support", ["definition"]).items()
            )
        ]
    return {
        "package_manifest_sha256": catalog["package_manifest_sha256"],
        "catalog_schema_version": catalog["schema_version"],
        "package_schema_version": catalog["package_schema_version"],
        "n_materialized_types": len(counts),
        "n_contract_only_types": sum(
            entry["status"] == "contract_only" for entry in catalog["evidence"]
        ),
        "n_not_in_generation_types": sum(
            entry["status"] == "not_in_generation" for entry in catalog["evidence"]
        ),
        "baselines": baselines,
        "field_disposition": dispositions,
        "source_endpoint_counts": by_source_endpoint,
        "tcell_eligibility_funnel": tcell_funnel,
        "reference_mapping_funnel": reference_funnel,
        "reference_component_summary": reference_components,
        "reference_support_definitions": reference_support,
        "source_registry": catalog["source_registry"],
    }
