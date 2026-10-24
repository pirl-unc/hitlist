"""Identify historical evidence families without conflating their version numbers."""

from __future__ import annotations


def package_family(manifest):
    """Classify the envelope; the family's verifier still validates all content."""
    schema = manifest.get("schema_version")
    if (
        schema == 20
        and manifest.get("package_family") == "hap1_quantitative"
        and manifest.get("revision") == "shapiro2025_hap1_quantitative_v1"
    ):
        return "hap1_quantitative"
    if schema in (6, 7):
        return {6: "supporting", 7: "reference"}[schema]
    if schema in (8, 9):
        external = "appended_sources" in manifest
        reviews = "review_revision" in manifest
        if external and not reviews:
            return "external"
        if schema == 8 and reviews and not external:
            return "adjudication"
    if (
        schema == 10
        and manifest.get("parent_kind") == "external"
        and manifest.get("review_revision")
        and manifest.get("appended_sources")
    ):
        return "binding_review"
    if (
        schema == 11
        and manifest.get("parent_kind") == "binding_review"
        and manifest.get("revision") == "gomez_zepeda_2024_original_results_v1"
    ):
        return "gomez_source"
    if (
        schema == 12
        and manifest.get("parent_kind") == "gomez_source"
        and manifest.get("revision") == "gomez_zepeda_2024_arm_projection_v1"
    ):
        return "gomez_arm_projection"
    if (
        schema == 13
        and manifest.get("parent_kind") == "gomez_arm_projection"
        and manifest.get("revision") == "gomez_zepeda_2024_prediction_panels_v1"
    ):
        return "gomez_prediction_panel"
    if (
        schema == 14
        and manifest.get("parent_kind") == "binding_review"
        and manifest.get("revision") == "binding_assay_context_availability_v1"
    ):
        return "binding_context_availability"
    if (
        schema == 17
        and manifest.get("package_family") == "tmab_peptidomics"
        and manifest.get("revision") == "tmab_native_peptidomics_v3"
    ):
        return "tmab_peptidomics"
    if (
        schema == 18
        and manifest.get("package_family") == "tcell_processing"
        and manifest.get("revision") == "tcell_processing_evidence_v1"
    ):
        return "tcell_processing"
    if (
        schema == 19
        and manifest.get("package_family") == "digestion_processing"
        and manifest.get("revision") == "soh2024_digestion_processing_v1"
    ):
        return "digestion_processing"
    if (
        schema == 22
        and manifest.get("package_family") == "precursor_context"
        and manifest.get("revision") == "pxd009360_precursor_context_v3"
    ):
        return "precursor_context"
    raise ValueError(f"Unknown or ambiguous evidence package family: schema {schema}")


def adjudication_manifest_path(manifest):
    """Locate the reviewed predecessor only when explicitly present."""
    family = package_family(manifest)
    if family == "adjudication":
        return "manifest.json"
    if family == "external" and manifest.get("parent_kind") == "adjudication":
        return "manifests/adjudication.json"
    if family == "binding_review":
        return manifest.get("adjudication_manifest_path")
    if family == "gomez_source":
        return manifest.get("adjudication_manifest_path")
    if family == "gomez_arm_projection":
        return manifest.get("adjudication_manifest_path")
    if family == "gomez_prediction_panel":
        return manifest.get("adjudication_manifest_path")
    if family == "binding_context_availability":
        return manifest.get("adjudication_manifest_path")
    return None
