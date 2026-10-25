"""Method-supported search assignments, with raw modification missingness preserved."""

from __future__ import annotations

import hashlib
import json
import sys
import xml.etree.ElementTree as ET

import pyarrow as pa

from . import intact_kinetics as kinetics
from . import intact_organ_transfer as strict
from .intact_degradomics import ROOT, _json
from .intact_degradomics_schema import SCHEMAS
from .mapp_evidence import _schema

RECIPE_ID = "pxd058121_organ_method_baseline_change_v1"
OUTPUT_KIND = "registered_intact_organ_method"
TABLES = strict.TABLES
SOURCE_ASSETS = ("article.xml",)
MODIFICATION_REJECTION = "unmodified_search_assignment_not_supported"
CONTRACT = {
    **strict.CONTRACT,
    "recipe_id": RECIPE_ID,
    "modification_policy": "explicit_row_or_reported_method_no_additional_modifications",
    "admission": [
        "explicit_row_or_reported_method_unmodified_search_assignment"
        if value == "explicit_unmodified"
        else value
        for value in strict.CONTRACT["admission"]
    ],
    "source_assets": list(SOURCE_ASSETS),
    "raw_modification_values": "preserved_in_immutable_source_tables_never_filled",
    "limitations": [
        *strict.CONTRACT["limitations"],
        "search_assignment_not_biochemical_confirmation_of_no_modifications",
        "exact_study_software_version_and_blank_export_roundtrip_unknown",
    ],
}
DECISIONS = pa.schema(
    list(strict.DECISIONS)
    + list(_schema(("modification_policy", "modification_assertion_ids_json")))
)
LABEL_FIELDS = strict.LABEL_FIELDS
COVARIATES = pa.schema([field for field in DECISIONS if field.name not in LABEL_FIELDS])
POLICIES = pa.schema(
    list(SCHEMAS["assertions"])
    + list(_schema(("source_asset_sha256", "source_paragraph_sha256", "declaration_excerpt")))
)
OUTPUT_SCHEMAS = {
    **strict.OUTPUT_SCHEMAS,
    "decisions": DECISIONS,
    "examples": DECISIONS,
    "baselines": DECISIONS,
    "covariates": COVARIATES,
    "search_policies": POLICIES,
}


def search_policies(source):
    """Interpret only the pinned original study's specific Methods paragraph."""
    raw = (source.parent / "source_assets/article.xml").read_bytes()
    article = ET.fromstring(raw)
    if not any(n.text == "10.1073/pnas.2419607122" for n in article.iter("article-id")):
        return []
    section = article.find(".//sec[@id='s6']/sec[@id='s15']")
    if section is None or section.find("title") is None:
        return []
    if "".join(section.find("title").itertext()) != "ID-MS Data Analysis and Statistics.":
        return []
    phrase = "No additional modifications were included"
    texts = ["".join(p.itertext()) for p in section.iter("p")]
    matches = [text for text in texts if phrase in text]
    if len(matches) != 1 or not all(
        term in matches[0]
        for term in ("ClipsMS", "10 ppm", "P37840", "4 amino acids", "b-, y-, or by-")
    ):
        return []
    asset_hash = hashlib.sha256(raw).hexdigest()
    paragraph_hash = hashlib.sha256(matches[0].encode()).hexdigest()
    claim = hashlib.sha256((asset_hash + paragraph_hash).encode()).hexdigest()
    return [
        {
            "claim_id": "pxd058121:search_policy:" + claim,
            "source_record_id": None,
            "subject_id": "PXD058121:original_ClipsMS_search_assignments",
            "source_scope": "study_method_for_original_ClipsMS_search_workbooks",
            "assertion_predicate": "additional_search_modifications",
            "assertion_value": "none",
            "assertion_status": "reported_method",
            "source_locator": "article.xml#s15:ID-MS Data Analysis and Statistics",
            "source_asset_sha256": asset_hash,
            "source_paragraph_sha256": paragraph_hash,
            "declaration_excerpt": phrase,
        }
    ]


def modification_eligibility(assignment, policies):
    explicit, _ = strict.modification_eligibility(assignment, [])
    if explicit:
        return True, {
            "modification_policy": "explicit_row",
            "modification_assertion_ids_json": "[]",
        }
    values = json.loads(assignment["modifications_json"])
    supported = (
        values == {"Localized Mod": "0", "Term Mod": "0", "Unlocalized Mod": None}
        and len(policies) == 1
    )
    return supported, {
        "modification_policy": "reported_method_no_additional_modifications"
        if supported
        else "unsupported",
        "modification_assertion_ids_json": _json([p["claim_id"] for p in policies])
        if supported
        else "[]",
    }


def compile_kinetics(source, destination):
    return strict._compile_organ_changes(source, destination, sys.modules[__name__])


def _validate(recipe):
    from .evidence_recipe import RecipeError

    if recipe.get("registered_recipe") != RECIPE_ID:
        raise RecipeError("Invalid method-supported intact organ recipe")
    strict._validate({**recipe, "registered_recipe": strict.RECIPE_ID})


def describe(store):
    supported = strict.describe(store)["status"] == "materialized" and all(
        (store.directory / ROOT / "source_assets" / name).is_file() for name in SOURCE_ASSETS
    )
    return {
        "contract": json.loads(_json(CONTRACT)),
        "status": "materialized" if supported else "not_in_generation",
    }


def materialize_organ_method(store, recipe, destination):
    return kinetics._materialize(store, recipe, destination, sys.modules[__name__])


def verify_organ_method_contents(directory, manifest, source_package=None):
    return kinetics._verify_contents(directory, manifest, source_package, sys.modules[__name__])
