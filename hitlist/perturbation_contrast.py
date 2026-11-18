"""Source-qualified unpaired reporter contrasts, without automatic TRAIN admission."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import statistics
import tempfile
from collections import defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from .evidence_snapshot import snapshot_file
from .filter_values import normalize_filter_values
from .perturbation_quantities import (
    PACKAGE_SCHEMA_VERSION,
    REVISION,
    ROOT,
    _inventory,
    _space,
    verify_perturbation_quantity_package,
)
from .perturbation_quantity_compiler import encoded
from .perturbation_quantity_schema import _schema
from .provenance import file_digest

RECIPE_ID = "stopfer2020_unpaired_reporter_contrast_v1"
CONTRACT = {
    "recipe_id": RECIPE_ID,
    "source_revision": REVISION,
    "source_package_schema_version": PACKAGE_SCHEMA_VERSION,
    "endpoint": "same_plex_relative_reporter_log2_fold_change",
    "quantity_kind": "hipmhc_normalized_reporter",
    "target": "log2(arithmetic_mean_treated/arithmetic_mean_DMSO)",
    "admission": "exactly_three_positive_finite_distinct_channels_per_arm_for_one_reported_form",
    "pairing": "unpaired_arm_means_no_numbered_slot_pairing",
    "dispersion": "sample_standard_deviation_ddof_1_in_reporter_intensity_units",
    "p_values": "original_reported_values_not_recomputed_or_multiplicity_corrected",
    "split_group": "entire_PXD017407_source_project",
    "training_admission": "pending_external_reserved_cohort_affinity_and_source_overlap_review",
    "limitations": [
        "identification_q_threshold_0.05",
        "conditional_on_reported_positive_quantities_not_absence_labels",
        "condition_dependent_extraction_and_MS_response_not_cancelled",
        "separate_within_condition_normalization_for_large_shifts",
        "no_absolute_abundance_or_copies_per_cell",
        "sample_MHC_genotype_not_peptide_restriction",
        "physical_specimen_identity_and_cross_plex_reuse_unresolved",
        "modified_forms_require_explicit_consumer_support",
    ],
}
SCHEMAS = {
    "decisions": _schema(
        "source_record_id form_id plex_id status reasons_json",
        "n_control_samples n_treated_samples",
    ),
    "contrasts": _schema(
        "contrast_id source_record_id form_id plex_id peptide_sequence modified_sequence_raw "
        "modifications_json cell_line_name control_condition_id treated_condition_id "
        "cellular_mhc_genotype measurement_ids_json source_dataset source_reference_id "
        "split_group_id training_admission measurement_unit",
        "n_control_samples n_treated_samples",
        "mean_control_intensity mean_treated_intensity sd_control_intensity sd_treated_intensity "
        "target_log2_fold_change reported_log2_fold_change reported_p_value recomputation_delta",
    ),
}


def describe(store):
    return {
        "contract": json.loads(encoded(CONTRACT)),
        "status": "available"
        if store.manifest.get("revision") == REVISION
        else "not_in_generation",
    }


def _validate(recipe):
    from .evidence_recipe import RecipeError

    if (
        not isinstance(recipe, dict)
        or set(recipe) - {"schema_version", "name", "registered_recipe", "plex_ids"}
        or recipe.get("schema_version") != 1
        or recipe.get("registered_recipe") != RECIPE_ID
        or not isinstance(recipe.get("name"), str)
        or not recipe["name"].strip()
    ):
        raise RecipeError("Invalid perturbation contrast recipe")
    return normalize_filter_values(recipe.get("plex_ids"), "plex_ids")


def compile_contrasts(source_directory, destination, recipe):
    """Recompute effects while retaining every candidate's disposition."""
    from .evidence_recipe import RecipeError

    selected = _validate(recipe)
    source = Path(source_directory) / ROOT / "typed"
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)

    def rows(name):
        for batch in pq.ParquetFile(source / f"{name}.parquet").iter_batches(batch_size=1024):
            yield from batch.to_pylist()

    plexes = {r["plex_id"]: r for r in rows("plexes")}
    if selected is not None and set(selected) - plexes.keys():
        raise RecipeError("Unknown selected plex identity")
    selected = set(plexes) if selected is None else set(selected)
    samples = {r["sample_id"]: r for r in rows("samples")}
    conditions = {r["condition_id"]: r for r in rows("conditions")}
    quantities = defaultdict(list)
    for row in rows("measurements"):
        if row["quantity_kind"] == CONTRACT["quantity_kind"]:
            quantities[row["form_id"]].append(row)
    summaries = defaultdict(dict)
    for row in rows("reported_summaries"):
        summaries[row["form_id"]][row["statistic_name"]] = row["quantitative_value"]
    decisions, contrasts = [], []
    for form in rows("forms"):
        key, plex_id = form["form_id"], form["plex_id"]
        plex = plexes[plex_id]
        values = {"control": [], "treated": []}
        reasons, seen_samples, measurements = [], set(), []
        if plex_id not in selected:
            reasons.append("not_selected_plex")
        for quantity in quantities[key]:
            sample = samples[quantity["sample_id"]]
            if quantity["sample_id"] in seen_samples or sample["plex_id"] != plex_id:
                raise RecipeError("Ambiguous reporter channel identity")
            seen_samples.add(quantity["sample_id"])
            value = quantity["quantitative_value"]
            if value is not None and math.isfinite(value) and value > 0:
                values[sample["arm_role"]].append(value)
                measurements.append(quantity["measurement_id"])
        counts = {
            "n_control_samples": len(values["control"]),
            "n_treated_samples": len(values["treated"]),
        }
        if counts != {"n_control_samples": 3, "n_treated_samples": 3}:
            reasons.append("requires_three_positive_finite_channels_per_arm")
        decisions.append(
            {
                "source_record_id": form["source_record_id"],
                "form_id": key,
                "plex_id": plex_id,
                "status": "excluded" if reasons else "source_qualified",
                "reasons_json": encoded(reasons),
                **counts,
            }
        )
        if reasons:
            continue
        control, treated = statistics.mean(values["control"]), statistics.mean(values["treated"])
        effect = math.log2(treated) - math.log2(control)
        reported = summaries[key].get("log2_fold_change")
        contrasts.append(
            {
                "contrast_id": key + ":unpaired_contrast",
                "source_record_id": form["source_record_id"],
                "form_id": key,
                "plex_id": plex_id,
                "peptide_sequence": form["peptide_sequence"],
                "modified_sequence_raw": form["modified_sequence_raw"],
                "modifications_json": form["modifications_json"],
                "cell_line_name": plex["cell_line_name"],
                "control_condition_id": plex["control_condition_id"],
                "treated_condition_id": plex["treated_condition_id"],
                "cellular_mhc_genotype": conditions[plex["treated_condition_id"]][
                    "cellular_mhc_genotype"
                ],
                "measurement_ids_json": encoded(sorted(measurements)),
                **counts,
                "mean_control_intensity": control,
                "mean_treated_intensity": treated,
                "sd_control_intensity": statistics.stdev(values["control"]),
                "sd_treated_intensity": statistics.stdev(values["treated"]),
                "target_log2_fold_change": effect,
                "reported_log2_fold_change": reported,
                "reported_p_value": summaries[key].get("p_value"),
                "recomputation_delta": None if reported is None else effect - reported,
                "source_dataset": plex["source_dataset"],
                "source_reference_id": plex["source_reference_id"],
                "split_group_id": plex["source_dataset"],
                "training_admission": CONTRACT["training_admission"],
                "measurement_unit": "log2_relative_reporter_ratio",
            }
        )
    for row in rows("source_records"):
        if row["row_disposition"] in {"excluded_aggregate", "pooled_auc_not_condition_specific"}:
            decisions.append(
                {
                    "source_record_id": row["source_record_id"],
                    "status": "excluded",
                    "reasons_json": encoded([row["row_disposition"]]),
                    "n_control_samples": 0,
                    "n_treated_samples": 0,
                }
            )
    for name, data in (("decisions", decisions), ("contrasts", contrasts)):
        pq.write_table(
            pa.Table.from_pylist(data, schema=SCHEMAS[name]), destination / f"{name}.parquet"
        )
    return {
        "n_source_candidates": len(decisions),
        "n_contrasts": len(contrasts),
        "n_excluded_candidates": len(decisions) - len(contrasts),
        "n_selected_plexes": len(selected),
        "n_automatic_training_admissions": 0,
    }


def materialize_contrasts(store, recipe, destination):
    from .evidence_catalog import SCHEMA_VERSION
    from .evidence_recipe import RecipeError, verify_recipe_package

    _validate(recipe)
    if (
        store.manifest.get("revision") != REVISION
        or store.manifest.get("schema_version") != PACKAGE_SCHEMA_VERSION
    ):
        raise RecipeError("Perturbation contrasts require the qualified original source package")
    destination = Path(destination)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _space(destination.parent)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    try:
        # Snapshot the complete admitted source, including original workbooks.
        # Offline verification therefore replays the actual source rather than
        # trusting editable manifest hashes of derived quantity tables alone.
        for relative in ["manifest.json", *store.manifest["files"]]:
            _space(staging)
            snapshot_file(store.directory / relative, staging / "source_package" / relative)
        coverage = compile_contrasts(staging / "source_package", staging / "typed", recipe)
        (staging / "contract.json").write_text(encoded(CONTRACT) + "\n")
        manifest = {
            "schema_version": 1,
            "output_kind": "registered_perturbation_contrast",
            "catalog_schema_version": SCHEMA_VERSION,
            "hitlist_version": store._software_version(),
            "recipe": recipe,
            "recipe_sha256": hashlib.sha256(encoded(recipe).encode()).hexdigest(),
            "source_package_manifest_sha256": store._manifest_hash(),
            "source_package_schema_version": PACKAGE_SCHEMA_VERSION,
            "coverage": coverage,
            "files": _inventory(staging),
        }
        (staging / "manifest.json").write_text(encoded(manifest) + "\n")
        verify_recipe_package(staging, source_package=store.directory)
        staging.rename(destination)
        return destination / "manifest.json"
    except Exception:
        (staging / "failure-context.json").write_text(
            encoded({"source": str(store.directory), "destination": str(destination)}) + "\n"
        )
        raise


def verify_contrast_contents(directory, manifest, source_package=None):
    from .evidence_recipe import RecipeError

    directory = Path(directory)
    _validate(manifest["recipe"])
    if json.loads((directory / "contract.json").read_text()) != CONTRACT:
        raise RecipeError("Perturbation contrast contract changed")
    source = directory / "source_package"
    source_manifest = verify_perturbation_quantity_package(source)
    if (
        file_digest(source / "manifest.json")["sha256"]
        != manifest["source_package_manifest_sha256"]
        or manifest.get("source_package_schema_version") != PACKAGE_SCHEMA_VERSION
    ):
        raise RecipeError("Perturbation contrast source binding changed")
    expected = (
        {"contract.json", "source_package/manifest.json"}
        | {"source_package/" + relative for relative in source_manifest["files"]}
        | {f"typed/{name}.parquet" for name in SCHEMAS}
    )
    if set(manifest["files"]) != expected or _inventory(directory) != manifest["files"]:
        raise RecipeError("Perturbation contrast inventory changed")
    if source_package is not None:
        verify_perturbation_quantity_package(source_package)
    _space(directory.parent)
    scratch = Path(tempfile.mkdtemp(prefix="perturbation-contrast-verify-", dir=directory.parent))
    try:
        coverage = compile_contrasts(source, scratch / "typed", manifest["recipe"])
        if coverage != manifest["coverage"]:
            raise RecipeError("Perturbation contrast coverage differs from source replay")
        for name, schema in SCHEMAS.items():
            actual = pq.read_table(directory / "typed" / f"{name}.parquet")
            replay = pq.read_table(scratch / "typed" / f"{name}.parquet")
            if not actual.schema.equals(schema) or not actual.equals(replay):
                raise RecipeError(f"Perturbation contrast source replay differs: {name}")
    except Exception:
        (scratch / "failure-context.json").write_text(encoded({"package": str(directory)}) + "\n")
        raise
    else:
        shutil.rmtree(scratch)
    return manifest
