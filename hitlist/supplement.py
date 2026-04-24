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

"""Load supplementary peptide data from curated CSVs.

Some immunopeptidome studies deposit peptide-level data in PRIDE or
paper supplements that is not (or only partially) in IEDB.  This
module reads a YAML manifest that links each supplementary CSV to
its PMID (already curated in ``pmid_overrides.yaml``) and provides
IEDB-equivalent column defaults for source classification.

The output DataFrame matches the schema of :func:`hitlist.scanner.scan`
so it can be concatenated directly into the observations table.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from .curation import classify_ms_row, normalize_species

_DATA_DIR = Path(__file__).parent / "data"
_MANIFEST_PATH = _DATA_DIR / "supplementary.yaml"
_SUPP_DIR = _DATA_DIR / "supplementary"


def manifest_path() -> Path:
    """Return the path to the supplementary manifest YAML."""
    return _MANIFEST_PATH


def load_supplementary_manifest() -> list[dict]:
    """Load the supplementary data manifest.

    Returns
    -------
    list[dict]
        Each entry has ``pmid``, ``file``, ``label``, and ``defaults``.
        Returns an empty list if the manifest does not exist.
    """
    if not _MANIFEST_PATH.exists():
        return []
    with open(_MANIFEST_PATH) as f:
        entries = yaml.safe_load(f)
    return entries if entries else []


def scan_supplementary(classify_source: bool = True) -> pd.DataFrame:
    """Load all supplementary CSVs and return a scanner-compatible DataFrame.

    For each entry in the manifest, reads the CSV, fills missing columns
    from the entry's ``defaults`` dict, and runs :func:`classify_ms_row`
    for source classification.

    Parameters
    ----------
    classify_source
        Run source classification (default True).  When False, only
        allele resolution and species are computed.

    Returns
    -------
    pd.DataFrame
        Same column schema as :func:`hitlist.scanner.scan` output.
        Empty DataFrame if no supplementary data is configured.
    """
    entries = load_supplementary_manifest()
    if not entries:
        return pd.DataFrame()

    from .curation import (
        allele_to_serotype,
        classify_allele_resolution,
        classify_mhc_species,
    )

    per_entry_frames: list[pd.DataFrame] = []

    for entry in entries:
        pmid = entry["pmid"]
        csv_path = _SUPP_DIR / entry["file"]
        defaults = entry.get("defaults", {})

        if not csv_path.exists():
            continue

        df = pd.read_csv(csv_path, dtype=str).fillna("")
        if "peptide" not in df.columns:
            continue

        df["peptide"] = df["peptide"].str.strip()
        df = df[df["peptide"] != ""]
        if df.empty:
            continue

        if "mhc_restriction" in df.columns:
            # Normalize allele strings at ingest so bare forms like "A*02:01"
            # (missing the "HLA-" prefix) canonicalize to "HLA-A*02:01" and
            # downstream exact-match filters (``load_observations(
            # mhc_restriction="HLA-A*02:01")``) see the full row population.
            # ``normalize_allele`` is already ``@cache``d in curation.py;
            # the cost here is dominated by the one-time mhcgnomes parse
            # per unique string, not per row.  See pirl-unc/hitlist#121.
            from .curation import normalize_allele

            df["mhc_restriction"] = df["mhc_restriction"].str.strip().map(normalize_allele)
        else:
            df["mhc_restriction"] = ""
        if "mhc_class" in df.columns:
            df["mhc_class"] = df["mhc_class"].str.strip()
        else:
            df["mhc_class"] = ""

        if "is_potential_contaminant" in df.columns:
            df["is_potential_contaminant"] = (
                df["is_potential_contaminant"]
                .astype(str)
                .str.strip()
                .str.lower()
                .isin({"true", "1", "yes"})
            )
        else:
            df["is_potential_contaminant"] = False

        process_type = defaults.get("process_type", "")
        disease = defaults.get("disease", "")
        culture_condition = defaults.get("culture_condition", "")
        source_tissue = defaults.get("source_tissue", "")
        cell_name = defaults.get("cell_name", "")

        synth_iri = (
            "supplement:" + str(pmid) + ":" + df["peptide"] + ":" + df["mhc_restriction"]
        ).to_numpy()
        record = pd.DataFrame(
            {
                "peptide": df["peptide"].to_numpy(),
                "mhc_restriction": df["mhc_restriction"].to_numpy(),
                "mhc_class": df["mhc_class"].to_numpy(),
                # Supplementary rows don't carry an IEDB assay IRI, but the
                # synthesized string is already row-unique within a PMID, so
                # reuse it as ``assay_iri`` too.  Downstream exports can then
                # treat ``assay_iri`` as the stable evidence-row identifier
                # (issue #146) without branching on source.
                "assay_iri": synth_iri,
                "reference_iri": synth_iri,
                "pmid": pmid,
                "submission_id": "",
                "reference_title": entry.get("study_label", ""),
                "source_organism": defaults.get("source_organism", ""),
                "species": defaults.get("species", ""),
                "host": defaults.get("host", ""),
                "host_age": "",
                "process_type": process_type,
                "disease": disease,
                "disease_stage": "",
                "source_tissue": source_tissue,
                "cell_name": cell_name,
                "culture_condition": culture_condition,
                "antigen_processing_comments": "",
                "assay_comments": "",
                "qualitative_measurement": "",
                "is_binding_assay": False,
                "is_potential_contaminant": df["is_potential_contaminant"].to_numpy(),
            }
        )

        # Classify per-unique-allele, then map back onto every row.
        # Within a single supplementary entry the non-allele inputs are
        # constant (from manifest defaults), so classify_ms_row varies
        # only by mhc_restriction.
        unique_alleles = record["mhc_restriction"].unique()
        if classify_source:
            flag_rows = [
                {
                    "mhc_restriction": a,
                    **classify_ms_row(
                        process_type,
                        disease,
                        culture_condition,
                        source_tissue,
                        cell_name,
                        pmid,
                        mhc_restriction=a,
                    ),
                }
                for a in unique_alleles
            ]
        else:
            flag_rows = [
                {
                    "mhc_restriction": a,
                    "allele_resolution": classify_allele_resolution(a),
                    "serotype": allele_to_serotype(a),
                    "mhc_species": classify_mhc_species(a),
                }
                for a in unique_alleles
            ]
        flags = pd.DataFrame(flag_rows)
        record = record.merge(flags, on="mhc_restriction", how="left")

        # Fallback: derive mhc_species from host when allele-based
        # classification is empty (e.g. supplementary peptides without
        # per-peptide allele assignments).
        host_species = normalize_species(defaults.get("host", ""))
        if "mhc_species" in record.columns:
            record["mhc_species"] = record["mhc_species"].fillna("").replace("", host_species)
        else:
            record["mhc_species"] = host_species

        per_entry_frames.append(record)

    if not per_entry_frames:
        return pd.DataFrame()

    result = pd.concat(per_entry_frames, ignore_index=True)

    # Deduplicate within supplementary data: one row per (peptide, mhc_restriction, pmid)
    result = result.drop_duplicates(subset=["peptide", "mhc_restriction", "pmid"])

    return result
