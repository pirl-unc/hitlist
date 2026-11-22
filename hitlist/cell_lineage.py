"""Orthogonal APC vs tumor / healthy tissue lineage ontology (#945).

Decomposes noisy free-text ``cell_type`` / ``cell_name`` / ``disease`` /
``culture_condition`` fields into four orthogonal controlled columns:
- ``cell_lineage``
- ``malignancy_status``
- ``tumor_type``
- ``activation_state``
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import pandas as pd

from .cell_name_parser import parse_cell_name, registry_verdict
from .curation_yaml import load_curation_yaml

ONTOLOGY_PATH = Path(__file__).resolve().parent / "data" / "cell_lineage_ontology.yaml"

ORTHOGONAL_LINEAGE_COLUMNS: tuple[str, ...] = (
    "cell_lineage",
    "malignancy_status",
    "tumor_type",
    "activation_state",
)

_TUMOR_TYPE_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\bmelanoma\b", "melanoma"),
    (r"\b(nsclc|non-small cell lung|lung adenocarcinoma|lung squamous)\b", "nsclc"),
    (r"\b(colorectal|colon carcinoma|colon adenocarcinoma|rectal carcinoma|\bcrc\b)\b", "crc"),
    (
        r"\b(breast carcinoma|breast adenocarcinoma|breast cancer|mammary carcinoma)\b",
        "breast_carcinoma",
    ),
    (r"\b(renal cell carcinoma|\brcc\b|clear cell renal|kidney carcinoma)\b", "rcc"),
    (r"\b(ovarian carcinoma|ovarian cancer|high-grade serous ovarian)\b", "ovarian"),
    (r"\b(glioblastoma|\bgbm\b|glioma|astrocytoma)\b", "gbm"),
    (r"\b(acute myeloid leukemia|acute monocytic leukemia|\baml\b)\b", "aml"),
    (r"\b(chronic myeloid leukemia|chronic myelogenous leukemia|\bcml\b)\b", "cml"),
    (r"\b(chronic lymphocytic leukemia|\bcll\b)\b", "cll"),
    (r"\b(diffuse large b[- ]cell lymphoma|\bdlbcl\b)\b", "dlbcl"),
    (r"\b(mantle cell lymphoma|\bmcl\b)\b", "mcl"),
    (r"\b(burkitt lymphoma)\b", "burkitt_lymphoma"),
    (r"\b(multiple myeloma|plasma cell myeloma)\b", "mm"),
    (r"\b(pancreatic ductal adenocarcinoma|\bpdac\b|pancreatic adenocarcinoma)\b", "pdac"),
    (r"\b(cervical carcinoma|cervical adenocarcinoma)\b", "cervical_carcinoma"),
    (r"\b(osteosarcoma)\b", "osteosarcoma"),
    (r"\b(t[- ]cell acute lymphoblastic|\bt[- ]all\b|t[- ]cell leukemia)\b", "t_all"),
    (r"\b(mastocytoma)\b", "mastocytoma"),
    (r"\b(devil facial tumor|\bdftd\b|dft1|dft2)\b", "dftd"),
    (r"\blymphoma\b", "lymphoma_unspecified"),
    (r"\b(carcinoma|adenocarcinoma)\b", "carcinoma_unspecified"),
)


@dataclass(frozen=True)
class CellLineageClassification:
    """Orthogonal four-axis cell lineage classification."""

    cell_lineage: str
    malignancy_status: str
    tumor_type: str
    activation_state: str


@cache
def load_cell_lineage_ontology(path: Path | None = None) -> dict:
    """Load :file:`hitlist/data/cell_lineage_ontology.yaml`."""
    data = load_curation_yaml(path or ONTOLOGY_PATH)
    if not isinstance(data, dict) or "vocabularies" not in data:
        raise ValueError("cell_lineage_ontology.yaml must define 'vocabularies'")
    return data


def _infer_tumor_type(text: str) -> str:
    lowered = text.casefold()
    for pattern, label in _TUMOR_TYPE_PATTERNS:
        if re.search(pattern, lowered):
            return label
    return ""


def _infer_activation_state(
    *,
    culture_condition: str,
    condition: str,
    condition_cytokines: str,
    condition_material: str,
    cell_name: str,
    src_ex_vivo: bool | None,
) -> str:
    combined = " ".join(
        s for s in (culture_condition, condition, condition_cytokines, cell_name) if s
    ).casefold()
    if re.search(r"\b(ifng|ifn-g|ifn-gamma|interferon[- ]gamma)\b", combined):
        return "ifng_stimulated"
    if re.search(r"\b(lps|lipopolysaccharide|poly\(?i:?c\)?|tlr[0-9]?)\b", combined):
        return "activated_lps"
    if re.search(r"\b(mature\s+(mo)?dc|matured\s+dendritic|mature\s+dendritic)\b", combined):
        return "mature_dc"
    if re.search(r"\b(immature\s+(mo)?dc|immature\s+dendritic|idc)\b", combined):
        return "immature_dc"
    if condition_cytokines.strip() and condition_cytokines.strip().casefold() != "none":
        return "cytokine_stimulated"
    if re.search(r"\b(ifna|ifnb|ifn-a|ifn-b|tnf|il-4|gm-csf|cytokine)\b", combined):
        return "cytokine_stimulated"
    mat = (condition_material or "").strip().casefold()
    if mat in {"ex_vivo", "primary_tissue", "fresh_tissue"} or src_ex_vivo is True:
        return "ex_vivo_primary"
    if "resting" in combined or "unperturbed" in combined or "untreated" in combined:
        return "resting"
    if mat == "cultured" or "cultured" in combined or "in vitro" in combined:
        return "cultured"
    return "unknown"


def _infer_lineage_from_text(text: str, *, is_ebv_lcl: bool, is_malignant: bool) -> str:
    lowered = text.casefold()
    if is_ebv_lcl or re.search(r"\b(ebv[- ]?lcl|b[- ]?lcl|lymphoblastoid)\b", lowered):
        return "ebv_lcl"
    if re.search(r"\b(modc|mo-dc|monocyte[- ]derived dendritic)\b", lowered):
        return "moDC"
    if re.search(r"\bcdc1\b", lowered):
        return "cDC1"
    if re.search(r"\bcdc2\b", lowered):
        return "cDC2"
    if re.search(r"\b(pdc|plasmacytoid dendritic)\b", lowered):
        return "pDC"
    if re.search(r"\b(dendritic cell|\bdc\b)\b", lowered):
        return "dc_unspecified"
    if "macrophage" in lowered:
        return "macrophage"
    if "monocyte" in lowered:
        return "monocyte"
    if "pbmc" in lowered or "peripheral blood mononuclear" in lowered:
        return "pbmc"
    if "splenocyte" in lowered:
        return "splenocyte"
    if re.search(r"\b(b[- ]?cell|b lymphocyte)\b", lowered):
        return "b_cell_malignant" if is_malignant else "b_cell_primary"
    if re.search(r"\b(t[- ]?cell|t lymphocyte|thymocyte)\b", lowered):
        return "t_cell"
    if re.search(r"\b(nk[- ]?cell|natural killer)\b", lowered):
        return "nk_cell"
    if "thymic epithelial" in lowered or "mtec" in lowered or "ctec" in lowered:
        return "thymic_epithelial"
    if "pancreatic beta" in lowered or "beta-cell" in lowered or "beta cell" in lowered:
        return "pancreatic_beta"
    if "melanocyte" in lowered or "melanoma" in lowered:
        return "melanocyte"
    if "fibroblast" in lowered:
        return "fibroblast"
    if "glial" in lowered or "astrocyte" in lowered or "glioma" in lowered or "schwann" in lowered:
        return "glial"
    if "hepatocyte" in lowered or "liver" in lowered:
        return "hepatocyte"
    if "endothelial" in lowered:
        return "endothelial"
    if "myeloblast" in lowered or "myeloid" in lowered:
        return "myeloid_leukemia" if is_malignant else "monocyte"
    if "mast cell" in lowered:
        return "mast_cell"
    if "osteoblast" in lowered or "osteosarcoma" in lowered:
        return "osteoblast"
    if "epithelial" in lowered or "carcinoma" in lowered or "adenocarcinoma" in lowered:
        return "epithelial"
    return "unknown"


def classify_cell_lineage(
    *,
    cell_line_name: str = "",
    cell_name: str = "",
    cell_type: str = "",
    source_tissue: str = "",
    disease: str = "",
    culture_condition: str = "",
    condition: str = "",
    condition_cytokines: str = "",
    condition_material: str = "",
    is_monoallelic: bool = False,
    src_cancer: bool | None = None,
    src_healthy_tissue: bool | None = None,
    src_ebv_lcl: bool | None = None,
    src_cell_line: bool | None = None,
    src_ex_vivo: bool | None = None,
    ontology_path: Path | None = None,
) -> CellLineageClassification:
    """Classify a sample/observation along four orthogonal lineage axes (#945)."""
    ontology = load_cell_lineage_ontology(ontology_path)
    known_lines = ontology.get("known_cell_lines") or {}
    engineered_hosts = set(ontology.get("engineered_hosts") or ())

    activation_state = _infer_activation_state(
        culture_condition=culture_condition,
        condition=condition,
        condition_cytokines=condition_cytokines,
        condition_material=condition_material,
        cell_name=cell_name,
        src_ex_vivo=src_ex_vivo,
    )

    canonical_line = (cell_line_name or "").strip()
    if not canonical_line and cell_name:
        parsed = parse_cell_name(cell_name)
        canonical_line = parsed.cell_line_name or ""

    # 1. Known curated cell line lookup (authoritative over noisy IEDB cell_type suffixes).
    if canonical_line and canonical_line in known_lines:
        entry = known_lines[canonical_line]
        malignancy = entry["malignancy_status"]
        if is_monoallelic and canonical_line in engineered_hosts:
            malignancy = "engineered_monoallelic_host"
        return CellLineageClassification(
            cell_lineage=entry["cell_lineage"],
            malignancy_status=malignancy,
            tumor_type=entry["tumor_type"],
            activation_state=activation_state,
        )

    # 2. Fallback to cell_lines.yaml Cellosaurus catalog entry.
    cat_entry = registry_verdict(canonical_line) if canonical_line else None
    is_ebv = bool(src_ebv_lcl) or bool(cat_entry and cat_entry.get("is_ebv_lcl"))
    is_line = bool(src_cell_line) or bool(canonical_line) or (cat_entry is not None)
    cat_disease = str(cat_entry.get("disease", "")) if cat_entry else ""
    cat_type = str(cat_entry.get("cell_type", "")) if cat_entry else ""
    disease_text = " ".join(s for s in (disease, cat_disease) if s)
    inferred_tumor = _infer_tumor_type(f"{disease_text} {cell_name} {source_tissue}")
    is_malignant = (
        bool(src_cancer) or bool(cat_entry and cat_entry.get("is_cancer")) or bool(inferred_tumor)
    )
    if is_ebv and not (cat_entry and cat_entry.get("is_cancer")):
        is_malignant = False

    lineage = _infer_lineage_from_text(
        " ".join(s for s in (cell_name, cell_type, cat_type, source_tissue, disease_text) if s),
        is_ebv_lcl=is_ebv,
        is_malignant=is_malignant,
    )
    primary_immune_lineages = {
        "moDC",
        "cDC1",
        "cDC2",
        "pDC",
        "dc_unspecified",
        "macrophage",
        "monocyte",
        "b_cell_primary",
        "pbmc",
        "splenocyte",
        "thymic_epithelial",
    }
    if is_monoallelic and (canonical_line in engineered_hosts or is_line):
        malignancy_status = "engineered_monoallelic_host"
    elif is_malignant:
        malignancy_status = "tumor_cell_line" if is_line else "tumor_primary"
    elif is_ebv or is_line:
        malignancy_status = "immortalized_nonmalignant"
    elif (
        src_healthy_tissue is True
        or src_ex_vivo is True
        or source_tissue
        or cell_type
        or lineage in primary_immune_lineages
    ):
        malignancy_status = "healthy_primary"
    else:
        malignancy_status = "unknown"

    if is_malignant:
        tumor_type = inferred_tumor or "other_tumor"
    elif malignancy_status in {
        "healthy_primary",
        "immortalized_nonmalignant",
        "engineered_monoallelic_host",
    }:
        tumor_type = "none"
    else:
        tumor_type = "unknown"

    return CellLineageClassification(
        cell_lineage=lineage,
        malignancy_status=malignancy_status,
        tumor_type=tumor_type,
        activation_state=activation_state,
    )


def _str_col(df: pd.DataFrame, name: str) -> list[str]:
    if name not in df.columns:
        return [""] * len(df)
    return df[name].astype("string").fillna("").tolist()


def _opt_bool_col(df: pd.DataFrame, name: str) -> list[bool | None]:
    if name not in df.columns:
        return [None] * len(df)
    return [None if pd.isna(v) else bool(v) for v in df[name].tolist()]


def attach_orthogonal_cell_lineage(
    df: pd.DataFrame, *, ontology_path: Path | None = None
) -> pd.DataFrame:
    """Attach orthogonal ``cell_lineage``, ``malignancy_status``, ``tumor_type``, ``activation_state``."""
    out = df.copy()
    if out.empty:
        for col in ORTHOGONAL_LINEAGE_COLUMNS:
            out[col] = pd.Series(dtype="string")
        return out

    keys = list(
        zip(
            _str_col(out, "cell_line_name"),
            _str_col(out, "cell_name"),
            _str_col(out, "cell_type"),
            _str_col(out, "source_tissue"),
            _str_col(out, "disease"),
            _str_col(out, "culture_condition"),
            _str_col(out, "condition"),
            _str_col(out, "condition_cytokines"),
            _str_col(out, "condition_material"),
            _opt_bool_col(out, "is_monoallelic"),
            _opt_bool_col(out, "src_cancer"),
            _opt_bool_col(out, "src_healthy_tissue"),
            _opt_bool_col(out, "src_ebv_lcl"),
            _opt_bool_col(out, "src_cell_line"),
            _opt_bool_col(out, "src_ex_vivo"),
        )
    )
    cache_map: dict[tuple, CellLineageClassification] = {}
    for key in set(keys):
        cache_map[key] = classify_cell_lineage(
            cell_line_name=key[0],
            cell_name=key[1],
            cell_type=key[2],
            source_tissue=key[3],
            disease=key[4],
            culture_condition=key[5],
            condition=key[6],
            condition_cytokines=key[7],
            condition_material=key[8],
            is_monoallelic=bool(key[9]),
            src_cancer=key[10],
            src_healthy_tissue=key[11],
            src_ebv_lcl=key[12],
            src_cell_line=key[13],
            src_ex_vivo=key[14],
            ontology_path=ontology_path,
        )
    results = [cache_map[k] for k in keys]
    out["cell_lineage"] = pd.Series(
        [r.cell_lineage for r in results], index=out.index, dtype="string"
    )
    out["malignancy_status"] = pd.Series(
        [r.malignancy_status for r in results], index=out.index, dtype="string"
    )
    out["tumor_type"] = pd.Series([r.tumor_type for r in results], index=out.index, dtype="string")
    out["activation_state"] = pd.Series(
        [r.activation_state for r in results], index=out.index, dtype="string"
    )
    return out
