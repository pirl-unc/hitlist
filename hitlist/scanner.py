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

"""General-purpose IEDB/CEDAR MHC ligand CSV scanner.

Scans IEDB and CEDAR MHC ligand full exports, resolves column indices
dynamically from CSV headers, and applies source classification via
the curation module.

Supports both targeted peptide scanning and full dataset profiling,
with class I/II separation and tqdm progress bars.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

import pandas as pd

try:
    from tqdm.auto import tqdm as _tqdm
except ImportError:
    _tqdm = None

from .curation import classify_ms_row

# ── Column name → index resolution ─────────────────────────────────────────

_COLUMN_NAMES: dict[str, list[str]] = {
    "assay_iri": ["Assay IRI", "Assay - Assay IRI"],
    "ref_iri": ["Reference IRI", "Reference - Reference IRI", "Reference | IEDB IRI"],
    "pmid": ["PMID", "Reference | PMID"],
    "submission_id": ["Submission ID", "Reference | Submission ID"],
    "ref_title": ["Title", "Reference | Title"],
    "epitope_name": ["Epitope | Name", "Epitope Name"],
    "source_organism": ["Source Organism", "Epitope | Source Organism"],
    "species": ["Epitope | Species", "Species"],
    "host": ["Host", "Host | Name"],
    "host_age": ["Host Age", "Host | Age"],
    "process_type": ["Process Type", "Host | Process Type"],
    "disease": ["Disease", "Host | Disease"],
    "disease_stage": ["Disease Stage", "Host | Disease Stage"],
    "antigen_processing_comments": ["Antigen Processing Comments"],
    "qualitative_measurement": ["Qualitative Measurement"],
    "assay_comments": ["Assay Comments"],
    "source_tissue": ["Source Tissue", "Effector Cells | Source Tissue"],
    "cell_name": ["Cell Name", "Effector Cells | Cell Name"],
    "culture_condition": ["Culture Condition", "Assay | Culture Condition"],
    "mhc_restriction": ["MHC Restriction | Name", "MHC Restriction Name"],
    "mhc_class": ["MHC Allele Class", "Class"],
}

_FALLBACK_INDICES: dict[str, int] = {
    "assay_iri": 0,
    "ref_iri": 1,
    "pmid": 3,
    "submission_id": 4,
    "ref_title": 8,
    "epitope_name": 11,
    "source_organism": 23,
    "species": 25,
    "host": 43,
    "host_age": 48,
    "process_type": 50,
    "disease": 51,
    "disease_stage": 53,
    "antigen_processing_comments": 88,
    "qualitative_measurement": 94,
    "assay_comments": 101,
    "source_tissue": 102,
    "cell_name": 104,
    "culture_condition": 106,
    "mhc_restriction": 107,
    "mhc_class": 111,
}


def _resolve_columns(cat_header: list[str], field_header: list[str]) -> dict[str, int]:
    """Resolve column indices from IEDB header rows, with fallbacks."""
    indices: dict[str, int] = {}
    combined_lower = []
    for i in range(max(len(cat_header), len(field_header))):
        cat = cat_header[i].strip() if i < len(cat_header) else ""
        fld = field_header[i].strip() if i < len(field_header) else ""
        combined_lower.append(f"{cat} | {fld}".lower() if cat and fld else (fld or cat).lower())
    field_lower = [f.strip().lower() for f in field_header]

    for key, candidates in _COLUMN_NAMES.items():
        for candidate in candidates:
            cl = candidate.lower()
            for i, hv in enumerate(combined_lower):
                if cl in hv or hv.endswith(cl):
                    indices[key] = i
                    break
            if key in indices:
                break
            for i, hv in enumerate(field_lower):
                if cl in hv or hv == cl:
                    indices[key] = i
                    break
            if key in indices:
                break
    for key, fallback in _FALLBACK_INDICES.items():
        if key not in indices:
            indices[key] = fallback
    return indices


def _safe_col(row: list[str], idx: int) -> str:
    return row[idx] if len(row) > idx else ""


def _open_csv(path: Path) -> tuple[csv.reader, dict[str, int], Path]:
    fh = open(path, newline="")  # noqa: SIM115
    reader = csv.reader(fh)
    cat_header = next(reader, [])
    field_header = next(reader, [])
    cols = _resolve_columns(cat_header, field_header)
    return reader, cols, path


def _progress(reader, path: Path, desc: str):
    if _tqdm is None:
        yield from reader
        return
    total = os.path.getsize(path)
    with _tqdm(total=total, unit="B", unit_scale=True, desc=desc, leave=False) as pbar:
        for row in reader:
            pbar.update(sum(len(f) for f in row) + len(row))
            yield row


# ── Public API ──────────────────────────────────────────────────────────────


def scan(
    peptides: set[str] | None = None,
    iedb_path: str | Path | None = None,
    cedar_path: str | Path | None = None,
    human_only: bool = True,
    hla_only: bool = True,
    mhc_class: str | None = None,
    classify_source: bool = True,
) -> pd.DataFrame:
    """Scan IEDB/CEDAR for matching peptides, or profile entire dataset.

    Parameters
    ----------
    peptides
        Peptide sequences to match. If None, scans ALL rows (profile mode).
    iedb_path, cedar_path
        Paths to IEDB/CEDAR exports.
    human_only
        Keep only human-source rows (default True).
    hla_only
        Keep only HLA-restricted rows (default True).
    mhc_class
        Filter to ``"I"`` or ``"II"``. None = both.
    classify_source
        Run source classification (default True).

    Returns
    -------
    pd.DataFrame
        Deduplicated by assay IRI. Includes source classification columns
        when ``classify_source=True``.
    """
    source_paths: list[Path] = []
    if iedb_path is not None:
        source_paths.append(Path(iedb_path))
    if cedar_path is not None:
        source_paths.append(Path(cedar_path))

    rows: list[dict] = []
    seen: set[str] = set()

    for source_path in source_paths:
        if not source_path.exists():
            continue
        reader, c, p = _open_csv(source_path)
        for row in _progress(reader, p, f"Scanning {p.name}"):
            if peptides is not None:
                pep = _safe_col(row, c["epitope_name"])
                if pep not in peptides:
                    continue
            iri = row[c["assay_iri"]] if row else ""
            if iri in seen:
                continue
            seen.add(iri)

            src_org = _safe_col(row, c["source_organism"])
            species = _safe_col(row, c["species"])
            mhc_res = _safe_col(row, c["mhc_restriction"])

            if human_only and "Homo sapiens" not in (src_org, species):
                continue
            if hla_only and not mhc_res.startswith("HLA-"):
                continue
            if mhc_class is not None and _safe_col(row, c["mhc_class"]) != mhc_class:
                continue

            raw_pmid = _safe_col(row, c["pmid"]).strip()
            pmid: str | int = ""
            if raw_pmid:
                try:
                    pmid = int(raw_pmid)
                except ValueError:
                    pmid = raw_pmid

            process_type = _safe_col(row, c["process_type"])
            disease = _safe_col(row, c["disease"])
            culture_condition = _safe_col(row, c["culture_condition"])
            source_tissue = _safe_col(row, c["source_tissue"])
            cell_name = _safe_col(row, c["cell_name"])

            record: dict = {
                "peptide": _safe_col(row, c["epitope_name"]),
                "mhc_restriction": mhc_res,
                "mhc_class": _safe_col(row, c["mhc_class"]),
                "reference_iri": _safe_col(row, c["ref_iri"]),
                "pmid": pmid,
                "submission_id": _safe_col(row, c["submission_id"]),
                "reference_title": _safe_col(row, c["ref_title"]),
                "source_organism": src_org,
                "species": species,
                "host": _safe_col(row, c["host"]),
                "host_age": _safe_col(row, c["host_age"]),
                "process_type": process_type,
                "disease": disease,
                "disease_stage": _safe_col(row, c["disease_stage"]),
                "source_tissue": source_tissue,
                "cell_name": cell_name,
                "culture_condition": culture_condition,
                "antigen_processing_comments": _safe_col(row, c["antigen_processing_comments"]),
                "assay_comments": _safe_col(row, c["assay_comments"]),
                "qualitative_measurement": _safe_col(row, c["qualitative_measurement"]),
            }

            if classify_source:
                record.update(
                    classify_ms_row(
                        process_type, disease, culture_condition, source_tissue, cell_name, pmid
                    )
                )

            rows.append(record)

    return pd.DataFrame(rows)
