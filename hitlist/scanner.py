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
import warnings
from pathlib import Path

import pandas as pd

try:
    from tqdm.auto import tqdm as _tqdm
except ImportError:
    _tqdm = None

from .curation import classify_ms_row

# Sentinel so we can tell "user didn't pass mhc_species" apart from
# "user explicitly passed mhc_species=None (disable filter)".
_UNSET: object = object()

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
    "assay_method": ["Assay | Method", "Method"],
    "response_measured": ["Assay | Response measured", "Response measured"],
    "measurement_units": ["Assay | Units", "Units"],
    "measurement_inequality": ["Assay | Measurement Inequality", "Measurement Inequality"],
    "quantitative_measurement": [
        "Assay | Quantitative measurement",
        "Quantitative measurement",
    ],
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
    "assay_method": 90,
    "response_measured": 91,
    "measurement_units": 92,
    "measurement_inequality": 95,
    "quantitative_measurement": 96,
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


def _parse_float(value: str) -> float:
    """Parse a cell to ``float``; return NaN for empty or non-numeric values.

    IEDB's quantitative-measurement column is mostly numeric but contains
    stray empty strings, whitespace, and occasional free-text notes.
    Callers downstream filter with ``value.notna()`` + ``value.between(...)``,
    so NaN is the right sentinel.
    """
    if not value:
        return float("nan")
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return float("nan")


class _ByteCountingFile:
    """Wrap a text file so we can track how many bytes have been read.

    ``csv.reader`` disables ``tell()`` on the underlying file via its
    buffered-read optimization, so we count bytes ourselves in
    ``readline``.  Supports the subset of file methods ``csv.reader``
    actually uses.
    """

    __slots__ = ("_fh", "bytes_read")

    def __init__(self, fh):
        self._fh = fh
        self.bytes_read = 0

    def readline(self, *args, **kwargs) -> str:
        line = self._fh.readline(*args, **kwargs)
        # bytes_read tracks encoded size; line is already str, so use
        # its length (close enough — ASCII dominates IEDB exports).
        self.bytes_read += len(line)
        return line

    def __iter__(self):
        return self

    def __next__(self):
        line = self._fh.readline()
        if not line:
            raise StopIteration
        self.bytes_read += len(line)
        return line

    def close(self) -> None:
        self._fh.close()


def _open_csv(path: Path) -> tuple[csv.reader, dict[str, int], Path, _ByteCountingFile]:
    fh = _ByteCountingFile(open(path, newline=""))  # noqa: SIM115
    reader = csv.reader(fh)
    cat_header = next(reader, [])
    field_header = next(reader, [])
    cols = _resolve_columns(cat_header, field_header)
    return reader, cols, path, fh


def _progress(reader, path: Path, desc: str, fh: _ByteCountingFile | None = None):
    """Yield rows with a tqdm progress bar.

    Updates progress in bytes every ``update_every`` rows instead of
    per-row — avoids per-row ``sum(len(f) for f in row)`` which was
    >50% of scan time for large files.
    """
    if _tqdm is None:
        yield from reader
        return
    total = os.path.getsize(path)
    update_every = 1000
    with _tqdm(total=total, unit="B", unit_scale=True, desc=desc, leave=False) as pbar:
        last_bytes = 0
        for i, row in enumerate(reader):
            yield row
            if fh is not None and (i + 1) % update_every == 0:
                now = fh.bytes_read
                pbar.update(now - last_bytes)
                last_bytes = now
        if fh is not None:
            pbar.update(max(total - last_bytes, 0))


# ── Public API ──────────────────────────────────────────────────────────────


def scan(
    peptides: set[str] | None = None,
    iedb_path: str | Path | None = None,
    cedar_path: str | Path | None = None,
    mhc_species: str | None | object = _UNSET,
    species_fallback: bool = True,
    mhc_class: str | None = None,
    classify_source: bool = True,
    min_allele_resolution: str | None = None,
    human_only: bool | None = None,
) -> pd.DataFrame:
    """Scan IEDB/CEDAR for matching peptides, or profile entire dataset.

    Parameters
    ----------
    peptides
        Peptide sequences to match. If None, scans ALL rows (profile mode).
    iedb_path, cedar_path
        Paths to IEDB/CEDAR exports.
    mhc_species
        Keep only rows where the MHC molecule belongs to this species
        (e.g. ``"Homo sapiens"``, ``"Mus musculus"``).  Uses mhcgnomes on
        the MHC restriction annotation as the authoritative species source.
        Default ``"Homo sapiens"``.  Pass ``None`` to disable species
        filtering entirely.
    species_fallback
        When mhcgnomes cannot resolve the species from the MHC restriction
        string (e.g. bare ``"HLA class I"``), fall back to the ``host`` /
        ``species`` text columns for the species decision.  Default ``True``
        — matches historical behavior.  Pass ``False`` for strict
        mhcgnomes-only filtering; rows with unparseable restrictions are
        dropped.
    mhc_class
        Filter to ``"I"`` or ``"II"``. None = both.
    classify_source
        Run source classification (default True).
    min_allele_resolution
        Minimum allele resolution to keep. One of ``"four_digit"``,
        ``"two_digit"``, ``"serological"``, ``"class_only"``. Rows
        with coarser resolution are dropped. None = no filtering.
    human_only
        .. deprecated:: 1.9.0
            Use ``mhc_species`` instead.  ``human_only=True`` maps to
            ``mhc_species="Homo sapiens"``; ``human_only=False`` maps to
            ``mhc_species=None``.  If both are passed, ``mhc_species`` wins.
            Slated for removal in hitlist 2.0.

    Returns
    -------
    pd.DataFrame
        Deduplicated by assay IRI. Includes source classification columns
        when ``classify_source=True``.
    """
    # ── Deprecation / migration for human_only ────────────────────────────
    if human_only is not None:
        warnings.warn(
            "human_only is deprecated and will be removed in hitlist 2.0; "
            "use mhc_species='Homo sapiens' (or mhc_species=None to disable "
            "species filtering).",
            DeprecationWarning,
            stacklevel=2,
        )
        if mhc_species is _UNSET:
            mhc_species = "Homo sapiens" if human_only else None
        # else: both passed — mhc_species wins silently (explicit new kwarg).
    if mhc_species is _UNSET:
        mhc_species = "Homo sapiens"
    source_paths: list[Path] = []
    if iedb_path is not None:
        source_paths.append(Path(iedb_path))
    if cedar_path is not None:
        source_paths.append(Path(cedar_path))

    from .curation import (
        allele_resolution_rank,
        classify_allele_resolution,
        classify_mhc_species,
        is_binding_assay,
        normalize_allele,
        normalize_species,
    )

    if mhc_species is not None:
        mhc_species = normalize_species(mhc_species)
    min_res_rank = allele_resolution_rank(min_allele_resolution) if min_allele_resolution else None

    rows: list[dict] = []
    seen: set[str] = set()

    for source_path in source_paths:
        if not source_path.exists():
            continue
        reader, c, p, fh = _open_csv(source_path)
        for row in _progress(reader, p, f"Scanning {p.name}", fh=fh):
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
            host = _safe_col(row, c["host"])
            mhc_res_raw = _safe_col(row, c["mhc_restriction"])
            # Canonicalize the allele string at ingest so downstream exact-
            # match filters see a consistent representation. mhcgnomes /
            # normalize_allele handles formats like "A*02:01" (missing
            # HLA- prefix), different separator styles, and multi-allele
            # pairs.  normalize_allele is already @cache'd on the unique
            # vocabulary so the per-row cost is ~100ns.  See #121.
            mhc_res = normalize_allele(mhc_res_raw) if mhc_res_raw else mhc_res_raw

            # Species filtering: use MHC allele name (via mhcgnomes) as the
            # authoritative species source; when species_fallback=True and
            # mhcgnomes can't parse, fall back to host/species text columns.
            mhc_sp = classify_mhc_species(mhc_res)

            if mhc_species is not None:
                if mhc_sp:
                    if mhc_sp != mhc_species:
                        continue
                elif species_fallback:
                    if mhc_species not in (
                        normalize_species(host),
                        normalize_species(species),
                    ):
                        continue
                else:
                    # Strict mode: unparseable MHC restriction → drop.
                    continue
            if mhc_class is not None and _safe_col(row, c["mhc_class"]) != mhc_class:
                continue
            if min_res_rank is not None:
                res = classify_allele_resolution(mhc_res)
                if allele_resolution_rank(res) > min_res_rank:
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
                "assay_iri": iri,
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
                # Structured quantitative binding-assay fields from IEDB /
                # CEDAR (issue #148, issue #135).  The raw string is preserved
                # so consumers can audit parsing; ``quantitative_value`` is the
                # float cast (NaN when the cell is empty / non-numeric) for
                # downstream filtering.  ``measurement_inequality`` ("", "<",
                # ">", "=", "<=", ">=") paired with ``quantitative_value``
                # distinguishes exact IC50 measurements from bounded ones.
                # ``response_measured`` (e.g. "qualitative binding",
                # "half life", "ligand presentation", "dissociation constant
                # KD") is the readout type — combined with ``assay_method``
                # and ``measurement_units`` it tells consumers whether a
                # numeric value is IC50 vs Kd vs t_half vs Tm.
                "assay_method": _safe_col(row, c["assay_method"]),
                "response_measured": _safe_col(row, c["response_measured"]),
                "measurement_units": _safe_col(row, c["measurement_units"]),
                "measurement_inequality": _safe_col(row, c["measurement_inequality"]),
                "quantitative_measurement": _safe_col(row, c["quantitative_measurement"]),
                "quantitative_value": _parse_float(_safe_col(row, c["quantitative_measurement"])),
                "is_binding_assay": is_binding_assay(
                    _safe_col(row, c["qualitative_measurement"]),
                    _safe_col(row, c["assay_comments"]),
                ),
            }

            if classify_source:
                record.update(
                    classify_ms_row(
                        process_type,
                        disease,
                        culture_condition,
                        source_tissue,
                        cell_name,
                        pmid,
                        mhc_restriction=mhc_res,
                        submission_id=record.get("submission_id", ""),
                        assay_comments=record.get("assay_comments", ""),
                    )
                )
            else:
                from .curation import allele_to_all_serotypes

                all_sero = allele_to_all_serotypes(mhc_res)
                record["allele_resolution"] = classify_allele_resolution(mhc_res)
                record["serotype"] = all_sero[0] if all_sero else ""
                record["serotypes"] = ";".join(all_sero)
                record["mhc_species"] = mhc_sp

            rows.append(record)

    return pd.DataFrame(rows)
