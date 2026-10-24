"""Audited baseline candidate cells; not full processing-model eligibility."""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import tempfile
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

from .evidence_catalog import SCHEMA_VERSION
from .evidence_snapshot import snapshot_file
from .goldberg_mapp import ROOT, _id, _json
from .mapp_evidence import _schema, _Sink
from .provenance import file_digest

RECIPE_ID = "goldberg2025_baseline_candidates_v1"
TABLES = (
    "peptides",
    "forms",
    "identifications",
    "groups",
    "references",
    "quantities",
    "acquisitions",
    "assertions",
    "material_contexts",
    "material_links",
    "publisher_links",
    "source_records",
)
CONTRACT = {
    "recipe_id": RECIPE_ID,
    "schema_version": 1,
    "endpoint": "original_baseline_MAPP_search_intensity_candidate_cells",
    "population": "original_deposition_control_labels_in_proteasome_associated_peptide_acquisitions",
    "decision_universe": "all_original_peptide_intensity_cells_including_aggregate_and_other_arms",
    "candidate_conditions": [
        "positive_finite",
        "canonical_sequence",
        "source_flags_explicitly_blank",
        "all_parent_accessions_have_only_source_human_header_annotations",
    ],
    "confidence_policy": "retain_peptide_PEP_and_per_run_identification_type_for_later_declared_policy",
    "publisher_selection": "retained_as_an_axis_never_used_as_an_eligibility_condition",
    "reference_support": "full_original_search_FASTA_absent",
    "precursor_pairing": "no_exact_specimen_or_aliquot_pairing_established",
    "normalization": "none_original_linear_intensity_and_log10_of_positive_cells",
    "training_status": "candidate_review_only_not_fully_qualified_processing_supervision",
    "split": "none_no_model_or_outcome_selection",
    "limitations": [
        "source_human_annotation_not_verified_origin",
        "MAPP_capture_not_complete_free_peptide_pool",
        "nonprofessional_APC_cell_line",
        "not_molarity",
        "zeros_not_biological_negatives",
        "confidence_matching_and_source_selection_require_downstream_policy",
    ],
}
DECISION_SCHEMA = _schema(
    (
        "candidate_id",
        "measurement_id",
        "source_record_id",
        "peptide_id",
        "acquisition_id",
        "context_id",
        "peptide_sequence",
        "parent_label_raw",
        "source_group_ids_raw",
        "source_evidence_ids_raw",
        "source_form_ids_raw",
        "sample_label_raw",
        "raw_file",
        "source_field",
        "quantitative_raw",
        "value_status",
        "source_organism_status",
        "reverse_raw",
        "contaminant_raw",
        "confidence_raw",
        "confidence_status",
        "identification_type",
        "publisher_selection_status",
        "status",
        "reasons_json",
        "full_reference_status",
        "precursor_pairing_status",
        "confidence_policy_status",
        "training_eligibility_status",
    ),
    numbers=("quantitative_value", "log10_intensity", "posterior_error_probability"),
)


def _rows(directory, name):
    for batch in pq.ParquetFile(directory / f"{name}.parquet").iter_batches():
        yield from batch.to_pylist()


def _compile(source, output):
    output.mkdir(parents=True)
    peptides = {r["peptide_id"]: r for r in _rows(source, "peptides")}
    acquisitions = {r["acquisition_id"]: r for r in _rows(source, "acquisitions")}
    published = {
        r["peptide_id"]
        for r in _rows(source, "publisher_links")
        if r["assertion_predicate"] == "publisher_row_original_identity"
        and r["link_status"] == "exact_native_id_and_sequence"
    }
    sinks = {
        name: _Sink(output, name, schema=DECISION_SCHEMA) for name in ("decisions", "candidates")
    }
    counts, reasons, axes = Counter(), Counter(), Counter()
    try:
        for quantity in _rows(source, "quantities"):
            if quantity["quantity_kind"] != "peptide_intensity":
                continue
            peptide = peptides[quantity["peptide_id"]]
            acquisition = acquisitions.get(quantity["acquisition_id"])
            baseline = (
                acquisition is not None
                and acquisition["condition_status"] == "source_control_label"
                and acquisition["measured_pool"] == "proteasome_associated_peptides"
            )
            excluded = []
            if not baseline:
                excluded.append("outside_baseline_acquisition_scope")
            if quantity["value_status"] != "positive":
                excluded.append("not_positive_finite_source_intensity")
            if not re.fullmatch(r"[ACDEFGHIKLMNPQRSTVWY]+", peptide["peptide_sequence"] or ""):
                excluded.append("noncanonical_or_missing_sequence")
            if peptide["reverse_raw"] != "" or peptide["contaminant_raw"] != "":
                excluded.append("source_flagged_or_flag_unresolved")
            if peptide["source_organism_status"] != "source_human_only":
                excluded.append("source_human_only_annotation_not_established")
            status = "outside_scope" if not baseline else "excluded" if excluded else "candidate"
            row = {
                "candidate_id": _id("baseline_candidate", quantity["measurement_id"]),
                **{
                    k: quantity[k]
                    for k in (
                        "measurement_id",
                        "source_record_id",
                        "peptide_id",
                        "acquisition_id",
                        "source_field",
                        "quantitative_raw",
                        "quantitative_value",
                        "value_status",
                        "identification_type",
                    )
                },
                **{
                    k: peptide[k]
                    for k in (
                        "peptide_sequence",
                        "parent_label_raw",
                        "source_group_ids_raw",
                        "source_evidence_ids_raw",
                        "source_form_ids_raw",
                        "source_organism_status",
                        "reverse_raw",
                        "contaminant_raw",
                        "confidence_raw",
                        "confidence_status",
                        "posterior_error_probability",
                    )
                },
                "context_id": acquisition["context_id"] if acquisition else None,
                "sample_label_raw": acquisition["sample_label_raw"] if acquisition else None,
                "raw_file": acquisition["raw_file"] if acquisition else None,
                "log10_intensity": math.log10(quantity["quantitative_value"])
                if quantity["value_status"] == "positive"
                else None,
                "publisher_selection_status": "published_exact_identity"
                if quantity["peptide_id"] in published
                else "omitted_or_unresolved_publisher_identity",
                "status": status,
                "reasons_json": _json(excluded),
                "full_reference_status": "full_original_search_FASTA_absent",
                "precursor_pairing_status": "exact_physical_pairing_unestablished",
                "confidence_policy_status": "retained_not_yet_selected",
                "training_eligibility_status": "not_fully_qualified_processing_supervision",
            }
            sinks["decisions"].add(row)
            counts[status] += 1
            reasons.update(excluded)
            if baseline:
                counts["baseline_cells"] += 1
            if status == "candidate":
                sinks["candidates"].add(row)
                axes[
                    (
                        row["publisher_selection_status"],
                        row["identification_type"],
                        row["confidence_status"],
                    )
                ] += 1
        return {
            "n_source_peptide_cells": sinks["decisions"].n_rows,
            "n_baseline_cells": counts["baseline_cells"],
            "n_candidate_cells": counts["candidate"],
            "n_excluded_baseline_cells": counts["excluded"],
            "n_outside_scope_cells": counts["outside_scope"],
            "reason_counts": dict(sorted(reasons.items())),
            "candidate_axes": [
                {
                    "publisher_selection_status": key[0],
                    "identification_type": key[1],
                    "confidence_status": key[2],
                    "n_cells": value,
                }
                for key, value in sorted(axes.items())
            ],
        }
    finally:
        for sink in sinks.values():
            sink.close()


def _validate(recipe):
    from .evidence_recipe import RecipeError

    if (
        set(recipe) != {"schema_version", "name", "registered_recipe"}
        or recipe["schema_version"] != 1
        or not isinstance(recipe["name"], str)
        or not recipe["name"].strip()
        or recipe["registered_recipe"] != RECIPE_ID
    ):
        raise RecipeError("Invalid Goldberg baseline candidate recipe")


def describe(store):
    return {
        "contract": json.loads(json.dumps(CONTRACT)),
        "status": "materialized"
        if {"goldberg_" + n for n in TABLES} <= store.bindings.keys()
        else "not_in_generation",
    }


def materialize_candidates(store, recipe, destination):
    from .evidence_recipe import RecipeError, verify_recipe_package

    _validate(recipe)
    if describe(store)["status"] != "materialized":
        raise RecipeError("Goldberg source is not materialized")
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    try:
        snapshot_file(store.directory / "manifest.json", staging / "source_manifest.json")
        for name in TABLES:
            snapshot_file(
                store.directory / ROOT / "typed" / f"{name}.parquet",
                staging / "source_tables" / f"{name}.parquet",
            )
        summary = _compile(staging / "source_tables", staging / "typed")
        (staging / "contract.json").write_text(_json(CONTRACT) + "\n")
        (staging / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        manifest = {
            "schema_version": 1,
            "output_kind": "registered_goldberg_candidates",
            "catalog_schema_version": SCHEMA_VERSION,
            "hitlist_version": store._software_version(),
            "recipe": recipe,
            "recipe_sha256": hashlib.sha256(_json(recipe).encode()).hexdigest(),
            "source_package_manifest_sha256": store._manifest_hash(),
            "source_package_schema_version": store.manifest["schema_version"],
            "coverage": summary,
            "files": {
                str(p.relative_to(staging)): file_digest(p)
                for p in sorted(staging.rglob("*"))
                if p.is_file()
            },
        }
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        verify_recipe_package(staging, source_package=store.directory)
        staging.rename(destination)
        return destination / "manifest.json"
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def verify_candidate_contents(directory, manifest, source_package=None):
    from .evidence_recipe import RecipeError

    _validate(manifest["recipe"])
    source = json.loads((directory / "source_manifest.json").read_text())
    if (
        file_digest(directory / "source_manifest.json")["sha256"]
        != manifest["source_package_manifest_sha256"]
        or source["schema_version"] != manifest["source_package_schema_version"]
    ):
        raise RecipeError("Goldberg candidate source binding differs")
    if json.loads((directory / "contract.json").read_text()) != CONTRACT:
        raise RecipeError("Goldberg candidate contract differs")
    expected = {
        "source_manifest.json",
        "contract.json",
        "summary.json",
        "typed/decisions.parquet",
        "typed/candidates.parquet",
    } | {f"source_tables/{name}.parquet" for name in TABLES}
    if set(manifest["files"]) != expected:
        raise RecipeError("Goldberg candidate file inventory differs")
    for name in TABLES:
        relative = str(ROOT / "typed" / f"{name}.parquet")
        if (
            file_digest(directory / "source_tables" / f"{name}.parquet")
            != source["files"][relative]
        ):
            raise RecipeError("Goldberg candidate source table differs")
        if (
            source_package is not None
            and file_digest(Path(source_package) / relative) != source["files"][relative]
        ):
            raise RecipeError("Goldberg candidate source table changed")
    with tempfile.TemporaryDirectory(prefix="hitlist-goldberg-candidates-") as temp:
        replay = Path(temp) / "typed"
        summary = _compile(directory / "source_tables", replay)
        if summary != manifest["coverage"] or summary != json.loads(
            (directory / "summary.json").read_text()
        ):
            raise RecipeError("Goldberg candidate coverage differs")
        for name in ("decisions", "candidates"):
            left = pq.ParquetFile(replay / f"{name}.parquet")
            right = pq.ParquetFile(directory / "typed" / f"{name}.parquet")
            if not right.schema_arrow.equals(DECISION_SCHEMA):
                raise RecipeError("Goldberg candidate schema differs")
            for expected_batch, actual in zip(
                left.iter_batches(), right.iter_batches(), strict=True
            ):
                if not expected_batch.equals(actual):
                    raise RecipeError("Goldberg candidate rows differ from source decisions")
    return manifest
