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


def classify_ms_row(
    process_type: str,
    disease: str,
    culture_condition: str,
    source_tissue: str = "",
    cell_name: str = "",
    pmid: int | str = "",
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

    # Parse PMID
    pmid_int = None
    if pmid:
        with contextlib.suppress(ValueError, TypeError):
            pmid_int = int(pmid)

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

    # ── Apply PMID override ─────────────────────────────────────────────
    override = None
    tissue_override = None
    if pmid_int and pmid_int in overrides:
        entry = overrides[pmid_int]
        override = entry.get("override")
        # Check per-tissue overrides within the study
        tissue_overrides = entry.get("tissue_overrides")
        if tissue_overrides:
            # Try matching Source Tissue against tissue override keys (case-insensitive)
            for tissue_key, tissue_val in tissue_overrides.items():
                if source_tissue_str.lower() == tissue_key.lower():
                    tissue_override = tissue_val
                    break

    # Resolve: tissue-level override takes priority over study-level
    effective_override = tissue_override or override

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
    else:
        # Default: non-EBV cell lines are cancer-derived
        is_cancer = process_type == "Occurrence of cancer" or (is_cell_line and not is_ebv_lcl)
        is_adjacent = False

    # Healthy requires: ex vivo, no cancer/adjacent/apc, no disease
    is_healthy_donor = (
        is_ex_vivo
        and not is_cancer
        and not is_adjacent
        and not is_activated_apc
        and process_type == "No immunization"
        and disease in ("healthy", "")
    )

    cl_name = cell_name_str if (is_cell_line or is_ebv_lcl) else ""

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
    }


def is_cancer_specific(flags: dict[str, bool]) -> bool:
    """Test if a peptide's aggregated flags indicate cancer-specificity.

    Cancer-specific = found in cancer AND NOT found in healthy somatic tissue.
    """
    return bool(
        flags.get("found_in_cancer", False) and not flags.get("found_in_healthy_tissue", False)
    )
