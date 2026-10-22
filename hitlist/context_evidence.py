"""Source-faithful biological context, curated arms and protocol claims.

The corrected human cohort adds context to an immutable MHC assay keyspace.
The package keeps every native source and both curation revisions. Typed
relations describe claims at their original scope; they do not invent donors,
specimens, runs or peptide-level restricting alleles.
"""

from __future__ import annotations

import json
import shutil
import sqlite3
import tempfile
from collections import Counter
from itertools import zip_longest
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .conditions import (
    MULTI_VALUE_CONDITION_COLUMNS,
    split_condition_tokens,
    validate_study_conditions,
)
from .curation import MS_SAMPLE_FIELDS, PMID_ENTRY_FIELDS
from .curation_yaml import load_curation_yaml
from .lineage import lineage_path, validate_lineage
from .mhc_evidence import (
    NATIVE_FIELDS,
    _link_or_copy,
    verify_mhc_evidence_package,
)
from .provenance import file_digest

PACKAGE_SCHEMA_VERSION = 2
CONTEXT_CATALOG_SCHEMA_VERSION = 3
COHORT_MANIFEST_SHA256 = "b180eba804532f633a6d61bb350584dbe24f8df4b4b96f02c6ec492c60144372"
HISTORICAL_CURATION_SHA256 = "5223fc63d0a0a563f469641b2abb58578a359b9bfac32218ce2611aa9cc8ef27"
PINNED_COHORT_COUNTS = {
    "n_assay_context_rows": 4_071_463,
    "n_contributor_links": 7_646_178,
    "n_condition_link_rows": 2_042_689,
    "n_curated_arms_per_revision": 807,
    "n_curated_studies_per_revision": 216,
    "n_curation_amendments": 35,
}

# Frozen to the corrected cohort artifact. The living curation registry can
# grow without changing how this historical 161-column source is verified.
COHORT_CONDITION_FIELDS = (
    "condition_id",
    "condition_status",
    "condition_evidence",
    "condition_reference",
    "condition_control",
    "condition_control_for",
    "condition_combination",
    "condition_knockout_genes",
    "condition_knockdown_genes",
    "condition_overexpression_genes",
    "condition_genetic_variants",
    "condition_transfection",
    "condition_transduction",
    "condition_cytokines",
    "condition_drugs",
    "condition_infection",
    "condition_stimulation",
    "condition_antigen_exposure",
    "condition_background",
    "condition_mhc_context",
    "condition_culture",
    "condition_material",
    "condition_labeling",
)

# Exact 90-column extension in the published cohort training table. A new
# upstream column requires a reviewed disposition before it can be ingested.
COHORT_EXTRA_FIELDS = (
    "mhc_class_label_severity",
    "mhc_class_label_suspect",
    "host_organism",
    "source_species",
    "is_chimeric",
    "is_engineered_mhc",
    "xenograft",
    "quantification_method",
    "study_apm_perturbed",
    "study_apm_genes",
    "sample_label",
    "perturbation",
    "condition_category",
    "sample_mhc",
    "mhc_basis",
    "mhc_genotype",
    "mhc_genotype_cell",
    "mhc_genotype_complete_loci",
    "mhc_genotype_source",
    "mhc_genotype_reported_loci",
    "instrument",
    "instrument_type",
    "acquisition_mode",
    "fragmentation",
    "labeling",
    "ip_antibody",
    "search_engine",
    "fdr",
    "apm_perturbed",
    "apm_genes_perturbed",
    "is_control_arm",
    "sample_attribution",
    *COHORT_CONDITION_FIELDS,
    "species_context_id",
    "species_context_status",
    "species_context_scope",
    "species_context_kind",
    "species_context_reference",
    "presenting_species",
    "in_vivo_host_species",
    "lineage_host_species",
    "introduced_mhc_species",
    "supported_foreign_species",
    "reviewed_candidate_species",
    "species_context_note",
    "effective_override",
    "effective_override_origin",
    "sample_note",
    "arm_resolution",
    "sample_group",
    "sample_mhc_origin",
    "matched_sample_count",
    "sample_match_type",
    "has_peptide_level_allele",
    "evidence_kind",
    "provenance_status",
    "evidence_source_id",
    "evidence_row_id",
    "lineage_context_id",
    "lineage_status",
    "experimental_origin_ids",
    "experimental_origin_status",
    "specimen_ids",
    "specimen_status",
    "donor_ids",
    "donor_status",
    "acquisition_ids",
    "acquisition_status",
)
IDENTITY_FIELDS = (
    "evidence_kind",
    "evidence_row_id",
    "evidence_source_id",
    "provenance_id",
    "provenance_status",
    "peptide",
    "mhc_restriction",
    "has_peptide_level_allele",
    "pmid",
    "condition_id",
    "sample_attribution",
    "attributed_sample_label",
    "assay_iri",
    "reference_iri",
    "lineage_context_id",
    "lineage_status",
    "experimental_origin_ids",
    "experimental_origin_status",
    "specimen_ids",
    "specimen_status",
    "donor_ids",
    "donor_status",
    "acquisition_ids",
    "acquisition_status",
    "contributor_resolution",
    "contributor_pmids",
    "contributor_study_status",
)
IDENTITY_ONLY_FIELDS = (
    "contributor_resolution",
    "contributor_pmids",
    "contributor_study_status",
)
COHORT_ARTIFACTS = {
    "training": "training.parquet",
    "identities": "identities.parquet",
    "contributors": "contributors.parquet",
    "lineage": "lineage.json",
}
PROTOCOL_FIELDS = (
    "ip_antibody",
    "acquisition_mode",
    "instrument",
    "fragmentation",
    "labeling",
    "search_engine",
    "fdr",
    "quantification_method",
)
FACTOR_FIELDS = (
    "condition_control",
    "condition_knockout_genes",
    "condition_knockdown_genes",
    "condition_overexpression_genes",
    "condition_genetic_variants",
    "condition_transfection",
    "condition_transduction",
    "condition_cytokines",
    "condition_drugs",
    "condition_infection",
    "condition_stimulation",
    "condition_antigen_exposure",
    "condition_background",
    "condition_culture",
    "condition_material",
    "condition_labeling",
)
MHC_CLAIM_FIELDS = (
    "mhc",
    "mhc_genotype",
    "condition_mhc_context",
)

_ARM_SCHEMA = pa.schema(
    [
        ("arm_id", pa.string()),
        ("curation_revision", pa.string()),
        ("study_pmid", pa.int64()),
        ("study_index", pa.int32()),
        ("sample_index", pa.int32()),
        ("condition_id", pa.string()),
        ("sample_label", pa.string()),
        ("condition_text", pa.string()),
        ("condition_status", pa.string()),
        ("condition_evidence", pa.string()),
        ("condition_reference", pa.string()),
        ("condition_control", pa.string()),
        ("condition_combination", pa.string()),
        ("profiled", pa.bool_()),
        ("source_description", pa.string()),
        ("mhc_class_raw", pa.string()),
        ("species_raw", pa.string()),
        ("control_for_raw", pa.string()),
    ]
)
_STUDY_SCHEMA = pa.schema(
    [
        ("study_id", pa.string()),
        ("curation_revision", pa.string()),
        ("study_pmid", pa.int64()),
        ("study_index", pa.int32()),
        ("study_label", pa.string()),
        ("title", pa.string()),
        ("note", pa.string()),
        ("arm_resolution", pa.string()),
        ("arm_resolution_note", pa.string()),
        ("source_organism", pa.string()),
        ("species", pa.string()),
        ("source_tissue", pa.string()),
        ("cell_name", pa.string()),
        ("disease", pa.string()),
        ("culture_condition", pa.string()),
    ]
)
_CONTROL_LINK_SCHEMA = pa.schema(
    [
        ("claim_id", pa.string()),
        ("curation_revision", pa.string()),
        ("control_arm_id", pa.string()),
        ("comparison_arm_id", pa.string()),
        ("citation", pa.string()),
    ]
)
_FACTOR_SCHEMA = pa.schema(
    [
        ("claim_id", pa.string()),
        ("arm_id", pa.string()),
        ("curation_revision", pa.string()),
        ("source_field", pa.string()),
        ("factor_type", pa.string()),
        ("token", pa.string()),
        ("claim_status", pa.string()),
        ("evidence_basis", pa.string()),
        ("raw_value", pa.string()),
        ("combination", pa.string()),
        ("citation", pa.string()),
    ]
)
_MHC_CLAIM_SCHEMA = pa.schema(
    [
        ("claim_id", pa.string()),
        ("arm_id", pa.string()),
        ("curation_revision", pa.string()),
        ("role", pa.string()),
        ("raw_value", pa.string()),
        ("basis", pa.string()),
        ("evidence_basis", pa.string()),
        ("genotype_cell", pa.string()),
        ("complete_loci_raw", pa.string()),
        ("genotype_source", pa.string()),
        ("citation", pa.string()),
    ]
)
_PROTOCOL_SCHEMA = pa.schema(
    [
        ("claim_id", pa.string()),
        ("curation_revision", pa.string()),
        ("study_pmid", pa.int64()),
        ("arm_id", pa.string()),
        ("source_scope", pa.string()),
        ("field_name", pa.string()),
        ("raw_value", pa.string()),
        ("claim_status", pa.string()),
        ("evidence_basis", pa.string()),
        ("citation", pa.string()),
    ]
)
_AMENDMENT_SCHEMA = pa.schema(
    [
        ("arm_id", pa.string()),
        ("field_name", pa.string()),
        ("old_value_json", pa.string()),
        ("new_value_json", pa.string()),
        ("old_revision", pa.string()),
        ("new_revision", pa.string()),
        ("citation", pa.string()),
    ]
)
_LINEAGE_SCHEMA = pa.schema(
    [
        ("claim_id", pa.string()),
        ("arm_id", pa.string()),
        ("axis", pa.string()),
        ("status", pa.string()),
        ("identity_ids_json", pa.string()),
        ("evidence_reference", pa.string()),
    ]
)


def _historical_path():
    return Path(__file__).parent / "data" / "evidence_revisions" / "pmid_overrides_1_78.yaml"


def _cohort_manifest(cohort_dir, mhc_manifest, historical_path=None, lineage_file=None):
    historical_path = Path(historical_path or _historical_path())
    lineage_file = Path(lineage_file or lineage_path())
    path = cohort_dir / "manifest.json"
    manifest = json.loads(path.read_text())
    digest = file_digest(path)
    if manifest.get("schema_version") != 1 or set(manifest.get("artifacts", {})) != set(
        COHORT_ARTIFACTS
    ):
        raise ValueError("Unsupported or incomplete cohort bundle")
    for name, filename in COHORT_ARTIFACTS.items():
        declared = manifest["artifacts"][name]
        if declared.get("file") != filename or file_digest(cohort_dir / filename) != {
            "sha256": declared.get("sha256"),
            "size_bytes": declared.get("size_bytes"),
        }:
            raise ValueError(f"Cohort artifact mismatch: {filename}")
    source_inputs = manifest.get("inputs", {}).get("files", {})
    for name in (
        "observations.parquet",
        "binding.parquet",
        "other_assays.parquet",
        "observation_contributors.parquet",
    ):
        expected = mhc_manifest["files"][f"native/{name}"]["sha256"]
        if source_inputs.get(name, {}).get("sha256") != expected:
            raise ValueError(f"Cohort and MHC packages have different source: {name}")
    old_hash = source_inputs.get("curation:pmid_overrides.yaml", {}).get("sha256")
    if old_hash != file_digest(historical_path)["sha256"]:
        raise ValueError("Historical cohort curation revision is unavailable")
    if (
        source_inputs.get("specimen_lineage.yaml", {}).get("sha256")
        != file_digest(lineage_file)["sha256"]
    ):
        raise ValueError("Cohort lineage revision differs from packaged lineage")
    n_rows = pq.ParquetFile(cohort_dir / "training.parquet").metadata.num_rows
    if (
        n_rows != manifest.get("n_rows")
        or n_rows != pq.ParquetFile(cohort_dir / "identities.parquet").metadata.num_rows
    ):
        raise ValueError("Cohort observation count mismatch")
    if tuple(pq.ParquetFile(cohort_dir / "training.parquet").schema_arrow.names) != (
        *NATIVE_FIELDS,
        *COHORT_EXTRA_FIELDS,
    ):
        raise ValueError("Unreviewed cohort training schema")
    if tuple(pq.ParquetFile(cohort_dir / "identities.parquet").schema_arrow.names) != (
        IDENTITY_FIELDS
    ):
        raise ValueError("Unreviewed cohort identity schema")
    if pq.ParquetFile(cohort_dir / "contributors.parquet").metadata.num_rows != manifest.get(
        "coverage", {}
    ).get("n_contributor_links"):
        raise ValueError("Cohort contributor count mismatch")
    if digest["sha256"] == COHORT_MANIFEST_SHA256 and (
        n_rows != PINNED_COHORT_COUNTS["n_assay_context_rows"]
        or manifest["coverage"]["n_contributor_links"]
        != PINNED_COHORT_COUNTS["n_contributor_links"]
    ):
        raise ValueError("Pinned cohort counts changed")
    return manifest, digest


def _arm_id(pmid, condition_id):
    return f"pmid:{pmid}:arm:{condition_id}"


def _revision_rows(path, revision):
    entries = load_curation_yaml(path)
    studies = []
    arms = []
    control_links = []
    factors = []
    mhc_claims = []
    protocols = []
    arm_map = {}
    seen_pmids = set()
    study_fields = set()
    sample_fields = set()
    for study_index, study in enumerate(entries):
        validate_study_conditions(study)
        pmid = study.get("pmid")
        if not isinstance(pmid, int) or pmid in seen_pmids:
            raise ValueError("Curated studies need distinct numeric PMIDs")
        seen_pmids.add(pmid)
        study_fields.update(study)
        studies.append(
            {
                "study_id": f"pmid:{pmid}",
                "curation_revision": revision,
                "study_pmid": pmid,
                "study_index": study_index,
                "study_label": str(study.get("study_label") or ""),
                "title": str(study.get("title") or ""),
                "note": str(study.get("note") or ""),
                "arm_resolution": str(study.get("arm_resolution") or ""),
                "arm_resolution_note": str(study.get("arm_resolution_note") or ""),
                "source_organism": str(study.get("source_organism") or ""),
                "species": str(study.get("species") or ""),
                "source_tissue": str(study.get("source_tissue") or ""),
                "cell_name": str(study.get("cell_name") or ""),
                "disease": str(study.get("disease") or ""),
                "culture_condition": str(study.get("culture_condition") or ""),
            }
        )
        for field in PROTOCOL_FIELDS:
            if field in study:
                raw = study[field]
                protocols.append(
                    {
                        "claim_id": f"{revision}:study:{pmid}:{field}",
                        "curation_revision": revision,
                        "study_pmid": pmid,
                        "arm_id": None,
                        "source_scope": "study_default",
                        "field_name": field,
                        "raw_value": None if raw is None else str(raw),
                        "claim_status": "explicit_null" if raw is None else "reported",
                        "evidence_basis": "curated_study",
                        "citation": f"PMID:{pmid}",
                    }
                )
        for sample_index, sample in enumerate(study.get("ms_samples") or []):
            sample_fields.update(sample)
            condition_id = sample.get("condition_id")
            if not condition_id:
                raise ValueError(f"Curated sample lacks condition ID in PMID {pmid}")
            key = (pmid, str(condition_id))
            if key in arm_map:
                raise ValueError(f"Duplicate curated arm: {key}")
            arm_map[key] = sample
            arm_id = _arm_id(*key)
            citation = str(sample.get("condition_reference") or "")
            arms.append(
                {
                    "arm_id": arm_id,
                    "curation_revision": revision,
                    "study_pmid": pmid,
                    "study_index": study_index,
                    "sample_index": sample_index,
                    "condition_id": str(condition_id),
                    "sample_label": str(sample.get("sample_label") or ""),
                    "condition_text": str(sample.get("condition") or ""),
                    "condition_status": str(sample.get("condition_status") or "unreported"),
                    "condition_evidence": str(sample.get("condition_evidence") or ""),
                    "condition_reference": citation,
                    "condition_control": str(sample.get("condition_control") or ""),
                    "condition_combination": str(sample.get("condition_combination") or ""),
                    "profiled": bool(sample.get("profiled", True)),
                    "source_description": str(sample.get("source") or ""),
                    "mhc_class_raw": str(sample.get("mhc_class") or ""),
                    "species_raw": str(sample.get("species") or ""),
                    "control_for_raw": str(sample.get("condition_control_for") or ""),
                }
            )
            for target in split_condition_tokens(sample.get("condition_control_for")):
                control_links.append(
                    {
                        "claim_id": f"{revision}:{arm_id}:control_for:{target}",
                        "curation_revision": revision,
                        "control_arm_id": arm_id,
                        "comparison_arm_id": _arm_id(pmid, target),
                        "citation": citation or f"PMID:{pmid}",
                    }
                )
            for field in FACTOR_FIELDS:
                raw = sample.get(field)
                if raw is None or raw == "":
                    continue
                tokens = (
                    split_condition_tokens(str(raw))
                    if field in MULTI_VALUE_CONDITION_COLUMNS
                    else [str(raw)]
                )
                for index, token in enumerate(tokens):
                    status = (
                        "explicit_absence"
                        if token == "none"
                        else "applied_target_unreported"
                        if token == "unspecified"
                        else "reported"
                    )
                    factors.append(
                        {
                            "claim_id": f"{revision}:{arm_id}:{field}:{index}",
                            "arm_id": arm_id,
                            "curation_revision": revision,
                            "source_field": field,
                            "factor_type": field.removeprefix("condition_"),
                            "token": token,
                            "claim_status": status,
                            "evidence_basis": str(sample.get("condition_evidence") or "unreported"),
                            "raw_value": str(raw),
                            "combination": str(sample.get("condition_combination") or ""),
                            "citation": citation or f"PMID:{pmid}",
                        }
                    )
            for field in MHC_CLAIM_FIELDS:
                raw = sample.get(field)
                if raw is None or raw == "":
                    continue
                if field == "mhc":
                    basis = str(sample.get("mhc_basis") or "unreviewed")
                    role = {
                        "selected_restriction": "selected_restriction",
                        "sample_typing": "presenting_sample_candidates",
                    }.get(basis, "reported_mhc_unreviewed")
                elif field == "mhc_genotype":
                    basis = "independent_cellular_typing"
                    role = "cellular_mhc_genotype"
                else:
                    basis = "curated_context"
                    role = "mhc_expression_or_capture_context"
                mhc_claims.append(
                    {
                        "claim_id": f"{revision}:{arm_id}:{field}",
                        "arm_id": arm_id,
                        "curation_revision": revision,
                        "role": role,
                        "raw_value": str(raw),
                        "basis": basis,
                        "evidence_basis": (
                            "independent_cellular_typing_source"
                            if field == "mhc_genotype" and sample.get("mhc_genotype_source")
                            else str(sample.get("condition_evidence") or "curated_mhc")
                        ),
                        "genotype_cell": str(sample.get("mhc_genotype_cell") or ""),
                        "complete_loci_raw": str(sample.get("mhc_genotype_complete_loci") or ""),
                        "genotype_source": str(sample.get("mhc_genotype_source") or ""),
                        "citation": (
                            str(sample.get("mhc_genotype_source") or "")
                            if field == "mhc_genotype"
                            else citation or f"PMID:{pmid}"
                        ),
                    }
                )
            for field in PROTOCOL_FIELDS:
                if field in sample:
                    raw = sample[field]
                    protocols.append(
                        {
                            "claim_id": f"{revision}:{arm_id}:{field}",
                            "curation_revision": revision,
                            "study_pmid": pmid,
                            "arm_id": arm_id,
                            "source_scope": "arm_override",
                            "field_name": field,
                            "raw_value": None if raw is None else str(raw),
                            "claim_status": "explicit_null" if raw is None else "reported",
                            "evidence_basis": "curated_arm",
                            "citation": citation or f"PMID:{pmid}",
                        }
                    )
    if not study_fields <= set(PMID_ENTRY_FIELDS):
        raise ValueError(
            f"Unreviewed curated study fields: {sorted(study_fields - set(PMID_ENTRY_FIELDS))}"
        )
    if not sample_fields <= set(MS_SAMPLE_FIELDS):
        raise ValueError(
            f"Unreviewed curated arm fields: {sorted(sample_fields - set(MS_SAMPLE_FIELDS))}"
        )
    known_arm_ids = {_arm_id(*key) for key in arm_map}
    if any(row["comparison_arm_id"] not in known_arm_ids for row in control_links):
        raise ValueError("Curated control points to an unregistered comparison arm")
    return (
        arm_map,
        arms,
        factors,
        mhc_claims,
        protocols,
        study_fields,
        sample_fields,
        studies,
        control_links,
    )


def _amendments(old_arms, current_arms):
    rows = []
    for pmid, condition_id in sorted(old_arms.keys() | current_arms.keys()):
        old = old_arms.get((pmid, condition_id), {})
        current = current_arms.get((pmid, condition_id), {})
        for field in sorted(old.keys() | current.keys()):
            if old.get(field) == current.get(field):
                continue
            rows.append(
                {
                    "arm_id": _arm_id(pmid, condition_id),
                    "field_name": field,
                    "old_value_json": json.dumps(old.get(field), sort_keys=True),
                    "new_value_json": json.dumps(current.get(field), sort_keys=True),
                    "old_revision": "cohort_1_78",
                    "new_revision": "current",
                    "citation": str(current.get("condition_reference") or ""),
                }
            )
    return rows


def _field_disposition(inventory):
    study_fields = sorted(
        set(inventory["historical_study_fields"]) | set(inventory["current_study_fields"])
    )
    sample_fields = sorted(
        set(inventory["historical_sample_fields"]) | set(inventory["current_sample_fields"])
    )
    return {
        "cohort_training": {
            field: (
                {
                    "status": "mhc_native_preserved",
                    "relations": [
                        "native/observations.parquet",
                        "native/binding.parquet",
                        "native/other_assays.parquet",
                    ],
                }
                if field in NATIVE_FIELDS
                else {"status": "typed", "relation": "typed/assay_context.parquet"}
            )
            for field in (*NATIVE_FIELDS, *COHORT_EXTRA_FIELDS)
        },
        "cohort_identities": {
            field: (
                {"status": "typed", "relation": "typed/assay_context.parquet"}
                if field in IDENTITY_ONLY_FIELDS or field in COHORT_EXTRA_FIELDS
                else {
                    "status": "native_source_preserved",
                    "relation": "cohort_native/identities.parquet",
                }
            )
            for field in IDENTITY_FIELDS
        },
        "curated_study": {
            field: (
                {"status": "typed", "relation": "typed/protocol_claims.parquet"}
                if field in PROTOCOL_FIELDS
                else {"status": "typed", "relation": "typed/curated_arms.parquet"}
                if field == "ms_samples"
                else {"status": "typed", "relation": "typed/curated_studies.parquet"}
                if field
                in (
                    "pmid",
                    "study_label",
                    "title",
                    "note",
                    "arm_resolution",
                    "arm_resolution_note",
                    "source_organism",
                    "species",
                    "source_tissue",
                    "cell_name",
                    "disease",
                    "culture_condition",
                )
                else {"status": "source_only", "relation": "curation/*.yaml"}
            )
            for field in study_fields
        },
        "curated_sample": {
            field: (
                {"status": "typed", "relation": "typed/arm_factors.parquet"}
                if field in FACTOR_FIELDS
                else {"status": "typed", "relation": "typed/arm_mhc_claims.parquet"}
                if field in MHC_CLAIM_FIELDS
                or field.startswith("mhc_genotype")
                or field == "mhc_basis"
                else {"status": "typed", "relation": "typed/protocol_claims.parquet"}
                if field in PROTOCOL_FIELDS
                else {"status": "typed", "relation": "typed/arm_control_links.parquet"}
                if field == "condition_control_for"
                else {"status": "typed", "relation": "typed/curated_arms.parquet"}
                if field
                in (
                    "sample_label",
                    "condition",
                    "condition_id",
                    "condition_status",
                    "condition_evidence",
                    "condition_reference",
                    "condition_combination",
                    "profiled",
                    "source",
                    "mhc_class",
                    "species",
                )
                else {"status": "source_only", "relation": "curation/*.yaml"}
            )
            for field in sample_fields
        },
    }


def _context_source_registry(files, cohort_manifest_sha256):
    cohort_access = (
        "public_pinned_release"
        if cohort_manifest_sha256 == COHORT_MANIFEST_SHA256
        else "provided_snapshot"
    )
    return {
        "corrected_human_cohort": {
            "manifest_sha256": cohort_manifest_sha256,
            "native_artifacts": [f"cohort_native/{name}" for name in COHORT_ARTIFACTS.values()],
            "access": cohort_access,
            "license_status": "unverified",
        },
        "cohort_time_pmid_curation": {
            "sha256": files["curation/pmid_overrides_1_78.yaml"]["sha256"],
            "access": "bundled_source_snapshot",
            "license_status": "unverified",
        },
        "current_pmid_curation": {
            "sha256": files["curation/pmid_overrides_current.yaml"]["sha256"],
            "access": "bundled_source_snapshot",
            "license_status": "unverified",
        },
        "reviewed_lineage": {
            "sha256": files["curation/specimen_lineage.yaml"]["sha256"],
            "access": "bundled_source_snapshot",
            "license_status": "unverified",
        },
    }


def _write_small_tables(typed_dir, historical_path, current_path):
    old = _revision_rows(historical_path, "cohort_1_78")
    current = _revision_rows(current_path, "current")
    amendments = _amendments(old[0], current[0])
    for name, rows, schema in (
        ("curated_studies.parquet", old[7] + current[7], _STUDY_SCHEMA),
        ("curated_arms.parquet", old[1] + current[1], _ARM_SCHEMA),
        ("arm_control_links.parquet", old[8] + current[8], _CONTROL_LINK_SCHEMA),
        ("arm_factors.parquet", old[2] + current[2], _FACTOR_SCHEMA),
        ("arm_mhc_claims.parquet", old[3] + current[3], _MHC_CLAIM_SCHEMA),
        ("protocol_claims.parquet", old[4] + current[4], _PROTOCOL_SCHEMA),
        ("curation_amendments.parquet", amendments, _AMENDMENT_SCHEMA),
    ):
        pq.write_table(
            pa.Table.from_pylist(rows, schema=schema), typed_dir / name, compression="zstd"
        )
    return {
        "historical_arms": old[0],
        "current_arms": current[0],
        "n_curated_arms_per_revision": len(old[1]),
        "n_curated_studies_per_revision": len(old[7]),
        "n_arm_control_links": len(old[8]) + len(current[8]),
        "n_arm_factors": len(old[2]) + len(current[2]),
        "n_arm_mhc_claims": len(old[3]) + len(current[3]),
        "n_protocol_claims": len(old[4]) + len(current[4]),
        "n_curation_amendments": len(amendments),
        "historical_study_fields": sorted(old[5]),
        "current_study_fields": sorted(current[5]),
        "historical_sample_fields": sorted(old[6]),
        "current_sample_fields": sorted(current[6]),
    }


def _write_lineage_claims(lineage_path_in_package, typed_dir):
    registry = validate_lineage(json.loads(lineage_path_in_package.read_text()))
    rows = []
    seen = set()
    for context in registry.get("contexts", []):
        pmid = context.get("pmid")
        condition_id = context.get("condition_id")
        if not isinstance(pmid, int) or not condition_id:
            raise ValueError("Lineage context lacks a reviewed arm key")
        for axis in ("experimental_origin", "specimen", "donor", "acquisition"):
            key = (pmid, condition_id, axis)
            if key in seen:
                raise ValueError("Duplicate lineage axis claim")
            seen.add(key)
            claim = context.get(axis, {"status": "unknown", "ids": []})
            rows.append(
                {
                    "claim_id": f"pmid:{pmid}:arm:{condition_id}:lineage:{axis}",
                    "arm_id": _arm_id(pmid, condition_id),
                    "axis": axis,
                    "status": claim["status"],
                    "identity_ids_json": json.dumps(claim.get("ids", []), sort_keys=True),
                    "evidence_reference": str(context.get("evidence_reference") or ""),
                }
            )
    pq.write_table(
        pa.Table.from_pylist(rows, schema=_LINEAGE_SCHEMA),
        typed_dir / "lineage_claims.parquet",
        compression="zstd",
    )
    return len(rows)


def _batch_column(batch, name):
    return batch.column(batch.schema.get_field_index(name))


def _insert_ids(connection, table, values):
    try:
        connection.executemany(f"INSERT INTO {table} VALUES (?)", ((value,) for value in values))
    except sqlite3.IntegrityError as error:
        raise ValueError(f"Duplicate {table} observation ID") from error


def _write_assay_context(mhc_dir, cohort_dir, typed_dir, historical_arms, audit_path):
    audit = sqlite3.connect(audit_path)
    audit.execute("PRAGMA journal_mode=OFF")
    audit.execute("PRAGMA synchronous=OFF")
    for table in ("mhc_ids", "cohort_ids", "cohort_link_ids"):
        audit.execute(f"CREATE TABLE {table} (id TEXT PRIMARY KEY) WITHOUT ROWID")
    counts = Counter()
    attribution = Counter()
    lineage = Counter()
    try:
        mhc = pq.ParquetFile(mhc_dir / "typed" / "mhc_observations.parquet")
        for batch in mhc.iter_batches(batch_size=65536, columns=["observation_id"]):
            _insert_ids(audit, "mhc_ids", batch.column(0).to_pylist())
            counts["n_mhc_assays"] += batch.num_rows
        training = pq.ParquetFile(cohort_dir / "training.parquet")
        identities = pq.ParquetFile(cohort_dir / "identities.parquet")
        context_schema = None
        writer = None
        try:
            training_fields = ["provenance_id", "pmid", *COHORT_EXTRA_FIELDS]
            identity_fields = ["provenance_id", *IDENTITY_ONLY_FIELDS]
            for training_batch, identity_batch in zip_longest(
                training.iter_batches(batch_size=16384, columns=training_fields),
                identities.iter_batches(batch_size=16384, columns=identity_fields),
            ):
                if training_batch is None or identity_batch is None:
                    raise ValueError("Cohort identity/training row count mismatch")
                if (
                    training_batch.num_rows != identity_batch.num_rows
                    or not pc.all(
                        pc.equal(
                            _batch_column(training_batch, "provenance_id"),
                            _batch_column(identity_batch, "provenance_id"),
                        )
                    ).as_py()
                ):
                    raise ValueError("Cohort identity/training row order mismatch")
                ids = _batch_column(training_batch, "provenance_id").to_pylist()
                if any(not value for value in ids):
                    raise ValueError("Cohort row lacks observation ID")
                _insert_ids(audit, "cohort_ids", ids)
                pmids = _batch_column(training_batch, "pmid").to_pylist()
                conditions = _batch_column(training_batch, "condition_id").to_pylist()
                attributions = _batch_column(training_batch, "sample_attribution").to_pylist()
                arm_ids = []
                link_statuses = []
                for pmid, condition_id, basis in zip(pmids, conditions, attributions):
                    if condition_id:
                        if pmid is None or (int(pmid), str(condition_id)) not in historical_arms:
                            raise ValueError(
                                "Cohort condition points to an unregistered historical arm"
                            )
                        if not basis:
                            raise ValueError("Cohort arm assignment lacks attribution basis")
                        arm_ids.append(_arm_id(int(pmid), str(condition_id)))
                        link_statuses.append(
                            "class_pool_context" if basis == "class_pool" else "attributed_arm"
                        )
                        counts["n_condition_link_rows"] += 1
                    else:
                        arm_ids.append(None)
                        link_statuses.append("unresolved_or_unreported")
                    attribution[basis or "unreported"] += 1
                for axis in ("experimental_origin", "specimen", "donor", "acquisition"):
                    values = _batch_column(training_batch, f"{axis}_status").to_pylist()
                    lineage.update((axis, value or "unreported") for value in values)
                table = pa.Table.from_batches([training_batch]).select(COHORT_EXTRA_FIELDS)
                for field in IDENTITY_ONLY_FIELDS:
                    table = table.append_column(field, _batch_column(identity_batch, field))
                table = table.add_column(0, "observation_id", pa.array(ids, type=pa.string()))
                table = table.add_column(
                    1,
                    "cohort_row_index",
                    pa.array(
                        range(
                            counts["n_assay_context_rows"],
                            counts["n_assay_context_rows"] + len(ids),
                        ),
                        type=pa.int64(),
                    ),
                )
                table = table.add_column(2, "arm_id", pa.array(arm_ids, type=pa.string()))
                table = table.add_column(
                    3, "arm_link_status", pa.array(link_statuses, type=pa.string())
                )
                if writer is None:
                    context_schema = table.schema
                    writer = pq.ParquetWriter(
                        typed_dir / "assay_context.parquet", context_schema, compression="zstd"
                    )
                writer.write_table(table.cast(context_schema))
                counts["n_assay_context_rows"] += len(ids)
            if writer is None:
                raise ValueError("Cohort source has no assay rows")
        finally:
            if writer is not None:
                writer.close()
        contributors = pq.ParquetFile(cohort_dir / "contributors.parquet")
        for batch in contributors.iter_batches(batch_size=65536, columns=["provenance_id"]):
            audit.executemany(
                "INSERT OR IGNORE INTO cohort_link_ids VALUES (?)",
                ((value,) for value in batch.column(0).to_pylist()),
            )
            counts["n_contributor_links"] += batch.num_rows
        counts["n_cohort_ids_not_in_mhc"] = audit.execute(
            "SELECT count(*) FROM cohort_ids LEFT JOIN mhc_ids USING (id) WHERE mhc_ids.id IS NULL"
        ).fetchone()[0]
        counts["n_orphan_contributor_ids"] = audit.execute(
            "SELECT count(*) FROM cohort_link_ids LEFT JOIN cohort_ids USING (id) "
            "WHERE cohort_ids.id IS NULL"
        ).fetchone()[0]
        counts["n_cohort_rows_without_contributors"] = audit.execute(
            "SELECT count(*) FROM cohort_ids LEFT JOIN cohort_link_ids USING (id) "
            "WHERE cohort_link_ids.id IS NULL"
        ).fetchone()[0]
        if any(
            counts[name]
            for name in (
                "n_cohort_ids_not_in_mhc",
                "n_orphan_contributor_ids",
                "n_cohort_rows_without_contributors",
            )
        ):
            raise ValueError("Cohort observation or contributor link mismatch")
        return {
            **dict(counts),
            "by_sample_attribution": dict(sorted(attribution.items())),
            "by_lineage_status": [
                {"axis": axis, "status": status, "n_observations": n_rows}
                for (axis, status), n_rows in sorted(lineage.items())
            ],
            "assay_context_schema": context_schema.to_string(),
        }
    finally:
        audit.close()


def verify_context_evidence_package(directory):
    """Recheck all source hashes, parent relations and context row inventories."""
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("schema_version") != PACKAGE_SCHEMA_VERSION:
        raise ValueError("Unsupported context evidence package schema")
    if manifest.get("catalog_schema_version") != CONTEXT_CATALOG_SCHEMA_VERSION:
        raise ValueError("Unsupported context evidence catalog schema")
    if manifest.get("field_disposition") != _field_disposition(manifest["reconciliation"]):
        raise ValueError("Context field disposition mismatch")
    files = manifest.get("files", {})
    actual_files = {
        str(path.relative_to(directory))
        for path in directory.rglob("*")
        if path.is_file() and path != directory / "manifest.json"
    }
    if set(files) != actual_files:
        raise ValueError("Context package file inventory mismatch")
    for relative, expected in files.items():
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts or file_digest(directory / path) != expected:
            raise ValueError(f"Context package artifact mismatch: {relative}")
    if manifest.get("source_registry") != _context_source_registry(
        files, manifest.get("cohort_manifest_sha256")
    ):
        raise ValueError("Context source registry mismatch")
    if files["manifests/mhc.json"]["sha256"] != manifest.get("parent_manifest_sha256"):
        raise ValueError("Context parent MHC manifest mismatch")
    parent_manifest = verify_mhc_evidence_package(directory, manifest_path="manifests/mhc.json")
    cohort_dir = directory / "cohort_native"
    _, cohort_digest = _cohort_manifest(
        cohort_dir,
        parent_manifest,
        directory / "curation" / "pmid_overrides_1_78.yaml",
        directory / "curation" / "specimen_lineage.yaml",
    )
    if cohort_digest["sha256"] != manifest.get("cohort_manifest_sha256"):
        raise ValueError("Context cohort manifest mismatch")
    if (
        file_digest(directory / "curation" / "pmid_overrides_current.yaml")["sha256"]
        != manifest["current_curation_sha256"]
    ):
        raise ValueError("Context current curation mismatch")
    inventory = manifest["reconciliation"]
    required_tables = {
        "assay_context.parquet": inventory["n_assay_context_rows"],
        "curated_studies.parquet": 2 * inventory["n_curated_studies_per_revision"],
        "curated_arms.parquet": 2 * inventory["n_curated_arms_per_revision"],
        "arm_control_links.parquet": inventory["n_arm_control_links"],
        "arm_factors.parquet": inventory["n_arm_factors"],
        "arm_mhc_claims.parquet": inventory["n_arm_mhc_claims"],
        "protocol_claims.parquet": inventory["n_protocol_claims"],
        "curation_amendments.parquet": inventory["n_curation_amendments"],
        "lineage_claims.parquet": inventory["n_lineage_claims"],
    }
    for name, expected in required_tables.items():
        if pq.ParquetFile(directory / "typed" / name).metadata.num_rows != expected:
            raise ValueError(f"Context typed row count mismatch: {name}")
    if (
        pq.ParquetFile(directory / "typed" / "assay_context.parquet").schema_arrow.to_string()
        != inventory["assay_context_schema"]
    ):
        raise ValueError("Context typed assay schema mismatch")
    for name, schema in (
        ("curated_studies.parquet", _STUDY_SCHEMA),
        ("curated_arms.parquet", _ARM_SCHEMA),
        ("arm_control_links.parquet", _CONTROL_LINK_SCHEMA),
        ("arm_factors.parquet", _FACTOR_SCHEMA),
        ("arm_mhc_claims.parquet", _MHC_CLAIM_SCHEMA),
        ("protocol_claims.parquet", _PROTOCOL_SCHEMA),
        ("curation_amendments.parquet", _AMENDMENT_SCHEMA),
        ("lineage_claims.parquet", _LINEAGE_SCHEMA),
    ):
        if pq.ParquetFile(directory / "typed" / name).schema_arrow != schema:
            raise ValueError(f"Context typed schema mismatch: {name}")
    if (
        inventory["n_cohort_ids_not_in_mhc"]
        or inventory["n_orphan_contributor_ids"]
        or inventory["n_cohort_rows_without_contributors"]
    ):
        raise ValueError("Context reconciliation contains unresolved source links")
    return manifest


def write_context_evidence_package(
    mhc_package,
    cohort_dir,
    destination,
    *,
    historical_curation_path=None,
    current_curation_path=None,
    lineage_file=None,
):
    """Publish one atomic package with all MHC and cohort sources and typed context."""
    mhc_package = Path(mhc_package)
    cohort_dir = Path(cohort_dir)
    destination = Path(destination)
    historical_curation_path = Path(historical_curation_path or _historical_path())
    current_curation_path = Path(
        current_curation_path or Path(__file__).parent / "data" / "pmid_overrides.yaml"
    )
    lineage_file = Path(lineage_file or lineage_path())
    if destination.exists():
        raise FileExistsError(f"Context evidence package already exists: {destination}")
    parent_manifest = verify_mhc_evidence_package(mhc_package)
    _, cohort_digest = _cohort_manifest(
        cohort_dir, parent_manifest, historical_curation_path, lineage_file
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    try:
        for relative in parent_manifest["files"]:
            target = staging / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            _link_or_copy(mhc_package / relative, target)
        (staging / "manifests").mkdir()
        _link_or_copy(mhc_package / "manifest.json", staging / "manifests" / "mhc.json")
        native = staging / "cohort_native"
        native.mkdir()
        for filename in (*COHORT_ARTIFACTS.values(), "manifest.json"):
            _link_or_copy(cohort_dir / filename, native / filename)
        curation = staging / "curation"
        curation.mkdir()
        _link_or_copy(historical_curation_path, curation / "pmid_overrides_1_78.yaml")
        _link_or_copy(current_curation_path, curation / "pmid_overrides_current.yaml")
        _link_or_copy(lineage_file, curation / "specimen_lineage.yaml")
        typed = staging / "typed"
        small = _write_small_tables(
            typed,
            curation / "pmid_overrides_1_78.yaml",
            curation / "pmid_overrides_current.yaml",
        )
        n_lineage_claims = _write_lineage_claims(native / "lineage.json", typed)
        audit_path = staging / "context_audit.sqlite"
        reconciliation = _write_assay_context(
            staging,
            native,
            typed,
            small["historical_arms"],
            audit_path,
        )
        audit_path.unlink()
        if len(small["historical_arms"]) != len(small["current_arms"]):
            raise ValueError("Curated arm revision changed key count")
        reconciliation.update(
            {
                key: value
                for key, value in small.items()
                if key not in ("historical_arms", "current_arms")
            }
        )
        reconciliation["n_lineage_claims"] = n_lineage_claims
        if cohort_digest["sha256"] == COHORT_MANIFEST_SHA256:
            for key in (
                "n_assay_context_rows",
                "n_contributor_links",
                "n_condition_link_rows",
                "n_curated_arms_per_revision",
                "n_curated_studies_per_revision",
            ):
                if reconciliation[key] != PINNED_COHORT_COUNTS[key]:
                    raise ValueError(f"Pinned cohort reconciliation changed: {key}")
            if file_digest(current_curation_path)["sha256"] == (
                "f563a8f1a6224d668bf06aed2f034b439ec9a1b607f78f883ace21e053850b09"
            ) and (
                reconciliation["n_curation_amendments"]
                != PINNED_COHORT_COUNTS["n_curation_amendments"]
            ):
                raise ValueError("Pinned curation amendment count changed")
        # A source file can be replaced during a long package build.
        verify_mhc_evidence_package(mhc_package)
        _cohort_manifest(cohort_dir, parent_manifest, historical_curation_path, lineage_file)
        files = {
            str(path.relative_to(staging)): file_digest(path)
            for path in sorted(staging.rglob("*"))
            if path.is_file()
        }
        from .version import __version__

        manifest = {
            "schema_version": PACKAGE_SCHEMA_VERSION,
            "catalog_schema_version": CONTEXT_CATALOG_SCHEMA_VERSION,
            "hitlist_version": __version__,
            "parent_manifest_sha256": files["manifests/mhc.json"]["sha256"],
            "cohort_manifest_sha256": cohort_digest["sha256"],
            "cohort_source_access": (
                "public_pinned_release"
                if cohort_digest["sha256"] == COHORT_MANIFEST_SHA256
                else "provided_snapshot"
            ),
            "cohort_source_license_status": "unverified",
            "historical_curation_sha256": files["curation/pmid_overrides_1_78.yaml"]["sha256"],
            "current_curation_sha256": files["curation/pmid_overrides_current.yaml"]["sha256"],
            "field_disposition": _field_disposition(reconciliation),
            "source_registry": _context_source_registry(files, cohort_digest["sha256"]),
            "relation_keys": {
                "mhc_observation_id": "observation_id",
                "cohort_observation_id": "provenance_id",
                "typed_assay_context_id": "observation_id",
                "curated_arm": ["study_pmid", "condition_id"],
                "cohort_contributor_observation_id": "provenance_id",
            },
            "reconciliation": reconciliation,
            "files": files,
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        verify_context_evidence_package(staging)
        staging.rename(destination)
        return destination / "manifest.json"
    finally:
        if staging.exists():
            shutil.rmtree(staging)
