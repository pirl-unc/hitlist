# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Data-driven MS evidence curation for IEDB/CEDAR MHC ligand data.

Classification rules and PMID overrides are loaded from YAML data files,
not hardcoded in Python. This makes the curation logic transparent,
auditable, and easy to extend without code changes.

Source categories (mutually exclusive priority order)::

    cancer              Cancer tissue, cancer patient biofluids, non-EBV cell lines
    adjacent            Tumor-adjacent normal tissue (resection margins)
    activated_apc       Monocyte-derived DCs, macrophages with activation
    healthy_tissue      Direct ex vivo healthy somatic tissue (THE SAFETY SIGNAL)
    healthy_thymus      Direct ex vivo thymus (expected for CTAs)
    healthy_reproductive Direct ex vivo reproductive tissue (expected for CTAs)
    ebv_lcl             EBV-transformed B-cell lines
    cell_line           Other cell lines
"""

from __future__ import annotations

import contextlib
from functools import lru_cache
from os.path import dirname, join

import pandas as pd
import yaml


def _data_path(filename: str) -> str:
    return join(dirname(__file__), "data", filename)


@lru_cache(maxsize=1)
def load_pmid_overrides() -> dict[int, dict]:
    """Load PMID curation overrides from YAML.

    Returns
    -------
    dict[int, dict]
        Mapping from PMID to override dict with keys: label, override,
        note, and optionally tissue_overrides, donors, samples, tissues.
    """
    with open(_data_path("pmid_overrides.yaml")) as f:
        entries = yaml.safe_load(f)
    return {int(e["pmid"]): e for e in entries}


@lru_cache(maxsize=1)
def load_tissue_categories() -> dict[str, frozenset[str]]:
    """Load tissue category definitions from YAML.

    Returns
    -------
    dict[str, frozenset[str]]
        Keys: ``reproductive``, ``thymus``, ``activated_apc_cell_names``,
        ``activated_apc_tissues``.
    """
    with open(_data_path("tissue_categories.yaml")) as f:
        data = yaml.safe_load(f)
    return {
        "reproductive": frozenset(data.get("reproductive", [])),
        "thymus": frozenset(data.get("thymus", [])),
        "activated_apc_cell_names": frozenset(data.get("activated_apc_cell_names", [])),
        "activated_apc_tissues": frozenset(data.get("activated_apc_tissues", [])),
    }


# ── Allele resolution ──────────────────────────────────────────────────────

#: Resolution tiers, ordered from most to least specific.
ALLELE_RESOLUTION_ORDER: list[str] = [
    "four_digit",
    "two_digit",
    "serological",
    "class_only",
    "unresolved",
]

_RESOLUTION_RANK: dict[str, int] = {v: i for i, v in enumerate(ALLELE_RESOLUTION_ORDER)}


def classify_allele_resolution(mhc_restriction: str) -> str:
    """Classify the resolution level of an MHC restriction annotation.

    Uses mhcgnomes if available for authoritative parsing, otherwise
    falls back to regex patterns.

    Parameters
    ----------
    mhc_restriction
        IEDB "MHC Restriction" field value.

    Returns
    -------
    str
        One of: ``"four_digit"``, ``"two_digit"``, ``"serological"``,
        ``"class_only"``, ``"unresolved"``.
    """
    if not mhc_restriction:
        return "unresolved"

    try:
        from mhcgnomes import parse
        from mhcgnomes.allele import Allele
        from mhcgnomes.mhc_class import MhcClass
        from mhcgnomes.serotype import Serotype

        result = parse(mhc_restriction)
        if isinstance(result, Allele):
            if len(result.allele_fields) >= 2:
                return "four_digit"
            return "two_digit"
        if isinstance(result, Serotype):
            return "serological"
        if isinstance(result, MhcClass):
            return "class_only"
        return "unresolved"
    except ImportError:
        pass

    # Regex fallback when mhcgnomes is not installed
    if not mhc_restriction.startswith("HLA"):
        return "unresolved"
    if "class" in mhc_restriction.lower():
        return "class_only"
    if "*" in mhc_restriction and ":" in mhc_restriction:
        return "four_digit"
    if "*" in mhc_restriction:
        return "two_digit"
    # HLA-A2, HLA-B7 etc.
    if mhc_restriction.startswith("HLA-"):
        return "serological"
    return "unresolved"


def allele_resolution_rank(resolution: str) -> int:
    """Integer rank for resolution (lower = more specific)."""
    return _RESOLUTION_RANK.get(resolution, len(ALLELE_RESOLUTION_ORDER))


# ── Mono-allelic cell line detection ──────────────────────────────────────


@lru_cache(maxsize=1)
def load_monoallelic_lines() -> list[dict]:
    """Load known mono-allelic cell line systems from YAML.

    Returns
    -------
    list[dict]
        Each entry has keys: name, aliases (list[str]), hla_status,
        endogenous_alleles (list[str]).
    """
    with open(_data_path("monoallelic_lines.yaml")) as f:
        return yaml.safe_load(f)


def detect_monoallelic(cell_name: str, mhc_restriction: str = "") -> tuple[bool, str]:
    """Detect if a row comes from a known mono-allelic cell line system.

    Parameters
    ----------
    cell_name
        IEDB "Cell Name" field value.
    mhc_restriction
        IEDB "MHC Restriction" field value (the reported allele).

    Returns
    -------
    tuple[bool, str]
        ``(is_monoallelic, host_name)``. ``is_monoallelic`` is True when
        the cell_name matches a known HLA-null/low host AND the reported
        allele is not one of the host's endogenous alleles.
    """
    if not cell_name:
        return False, ""

    cell_name_lower = cell_name.lower()
    for entry in load_monoallelic_lines():
        for alias in entry["aliases"]:
            if alias in cell_name_lower:
                endogenous = entry.get("endogenous_alleles", [])
                if mhc_restriction and mhc_restriction in endogenous:
                    return False, ""
                return True, entry["name"]
    return False, ""


def _matches_condition(row_fields: dict[str, str], condition: dict) -> bool:
    """Check if a row's fields match a condition dict.

    Each condition key is an IEDB field name, value is either a string
    or a list of strings. All conditions must match (AND logic).
    String matching is case-insensitive for Source Tissue.
    """
    for field, expected in condition.items():
        actual = row_fields.get(field, "")
        if isinstance(expected, list):
            # Any of the listed values matches
            if field == "Source Tissue":
                if actual.lower() not in {v.lower() for v in expected}:
                    return False
            elif actual not in expected:
                return False
        else:
            if field == "Source Tissue":
                if actual.lower() != str(expected).lower():
                    return False
            elif actual != expected:
                return False
    return True


def classify_ms_row(
    process_type: str,
    disease: str,
    culture_condition: str,
    source_tissue: str = "",
    cell_name: str = "",
    pmid: int | str = "",
    mhc_restriction: str = "",
    submission_id: str = "",
) -> dict[str, bool | str]:
    """Classify a public-MS row into curated source-context flags.

    Uses data-driven PMID overrides and tissue categories from YAML files.

    Parameters
    ----------
    process_type
        IEDB "Process Type" field.
    disease
        IEDB "Disease" field.
    culture_condition
        IEDB "Culture Condition" field.
    source_tissue
        IEDB "Source Tissue" field.
    cell_name
        IEDB "Cell Name" field.
    pmid
        PubMed ID for per-study override lookup.
    mhc_restriction
        IEDB "MHC Restriction" field value. Used to check whether
        the reported allele is endogenous to a mono-allelic host.

    Returns
    -------
    dict[str, bool | str]
        Source flags plus ``cell_line_name``.
    """
    process_type = str(process_type).strip() if pd.notna(process_type) else ""
    disease = str(disease).strip() if pd.notna(disease) else ""
    culture_condition = str(culture_condition).strip() if pd.notna(culture_condition) else ""
    source_tissue_str = str(source_tissue).strip() if pd.notna(source_tissue) else ""
    source_tissue_lower = source_tissue_str.lower()
    cell_name_str = str(cell_name).strip() if pd.notna(cell_name) else ""

    categories = load_tissue_categories()
    overrides = load_pmid_overrides()

    # Parse PMID and submission_id
    pmid_int = None
    if pmid:
        with contextlib.suppress(ValueError, TypeError):
            pmid_int = int(pmid)
    submission_id_str = str(submission_id).strip() if pd.notna(submission_id) else ""

    # Base signals
    is_ex_vivo = culture_condition == "Direct Ex Vivo"
    is_cell_line = culture_condition in (
        "Cell Line / Clone",
        "Cell Line / Clone (EBV transformed, B-LCL)",
    )
    is_ebv_lcl = culture_condition == "Cell Line / Clone (EBV transformed, B-LCL)"
    is_reproductive = source_tissue_lower in categories["reproductive"]
    is_thymus = source_tissue_lower in categories["thymus"]

    # Auto-detect activated APCs: DCs/macrophages from blood
    is_activated_apc = (
        cell_name_str.lower() in categories["activated_apc_cell_names"]
        and source_tissue_lower in categories["activated_apc_tissues"]
    )

    # ── Apply PMID override (three-level specificity) ─────────────────
    # Level 1: conditional rules (checked first, in order)
    # Level 2: PMID-level default override
    # Level 3: no match → fall through to structured-field classification
    effective_override = None
    # Look up override by PMID (int) or submission_id (str)
    entry = None
    if pmid_int and pmid_int in overrides:
        entry = overrides[pmid_int]
    elif submission_id_str and submission_id_str in overrides:
        entry = overrides[submission_id_str]

    if entry is not None:
        # Build row fields for condition matching
        row_fields = {
            "Source Tissue": source_tissue_str,
            "Culture Condition": culture_condition,
            "Cell Name": cell_name_str,
            "Disease": disease,
            "Process Type": process_type,
        }

        # Level 1: check conditional rules
        for rule in entry.get("rules", []):
            condition = rule.get("condition", {})
            if _matches_condition(row_fields, condition):
                effective_override = rule.get("override")
                break

        # Level 2: PMID-level default (only if no rule matched)
        if effective_override is None:
            effective_override = entry.get("override")

    # ── Classification ──────────────────────────────────────────────────
    if effective_override == "cancer_patient":
        is_cancer = True
        is_adjacent = False
        is_activated_apc = False
    elif effective_override == "adjacent":
        is_cancer = False
        is_adjacent = True
        is_activated_apc = False
    elif effective_override == "activated_apc":
        is_cancer = False
        is_adjacent = False
        is_activated_apc = True
    elif effective_override == "cell_line":
        is_cancer = True
        is_adjacent = False
        is_activated_apc = False
    elif effective_override == "healthy":
        is_cancer = False
        is_adjacent = False
        is_activated_apc = False
    else:
        # Default: non-EBV cell lines are cancer-derived
        is_cancer = process_type == "Occurrence of cancer" or (is_cell_line and not is_ebv_lcl)
        is_adjacent = False

    # Healthy requires: ex vivo, no cancer/adjacent/apc, no disease.
    # When override is "healthy", force the healthy path regardless of
    # process_type / disease fields (the override corrects bad IEDB metadata).
    is_healthy_donor = effective_override == "healthy" or (
        is_ex_vivo
        and not is_cancer
        and not is_adjacent
        and not is_activated_apc
        and process_type == "No immunization"
        and disease in ("healthy", "")
    )

    cl_name = cell_name_str if (is_cell_line or is_ebv_lcl) else ""

    # Mono-allelic detection: only for cell line rows
    is_monoallelic = False
    mono_host = ""
    if is_cell_line or is_ebv_lcl:
        is_monoallelic, mono_host = detect_monoallelic(cell_name_str, mhc_restriction)

    return {
        "src_cancer": is_cancer,
        "src_adjacent_to_tumor": is_adjacent,
        "src_activated_apc": is_activated_apc,
        "src_healthy_tissue": is_healthy_donor and not is_reproductive and not is_thymus,
        "src_healthy_thymus": is_healthy_donor and is_thymus,
        "src_healthy_reproductive": is_healthy_donor and is_reproductive,
        "src_cell_line": is_cell_line,
        "src_ebv_lcl": is_ebv_lcl,
        "src_ex_vivo": is_ex_vivo,
        "cell_line_name": cl_name,
        "is_monoallelic": is_monoallelic,
        "monoallelic_host": mono_host,
        "allele_resolution": classify_allele_resolution(mhc_restriction),
    }


def is_cancer_specific(flags: dict[str, bool]) -> bool:
    """Test if a peptide's aggregated flags indicate cancer-specificity.

    Cancer-specific = found in cancer AND NOT found in healthy somatic tissue.
    """
    return bool(
        flags.get("found_in_cancer", False) and not flags.get("found_in_healthy_tissue", False)
    )
