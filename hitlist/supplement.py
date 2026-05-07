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

from .curation import classify_ms_row, expand_allele_set, is_non_peptide_ligand, normalize_species

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

        # Include the supplementary filename in the synthesized IRI so the
        # same peptide / allele seen in, e.g. ``gomez_zepeda_2024_jy.csv`` and
        # ``gomez_zepeda_2024_raji.csv`` yields distinct row identifiers
        # and doesn't get collapsed by the ``drop_duplicates`` below (issue
        # #147).  The filename doubles as a per-sample provenance column.
        supp_file = entry["file"]
        synth_iri = (
            "supplement:"
            + str(pmid)
            + ":"
            + supp_file
            + ":"
            + df["peptide"]
            + ":"
            + df["mhc_restriction"]
        ).to_numpy()
        record = pd.DataFrame(
            {
                "peptide": df["peptide"].to_numpy(),
                "mhc_restriction": df["mhc_restriction"].to_numpy(),
                "mhc_class": df["mhc_class"].to_numpy(),
                # Supplementary rows don't carry an IEDB assay IRI, but the
                # synthesized string is already row-unique within a PMID + file,
                # so reuse it as ``assay_iri`` too.  Downstream exports can then
                # treat ``assay_iri`` as the stable evidence-row identifier
                # (issue #146) without branching on source.
                "assay_iri": synth_iri,
                "reference_iri": synth_iri,
                "supplementary_file": supp_file,
                "pmid": pmid,
                "submission_id": "",
                "reference_title": entry.get("study_label", ""),
                "source_organism": defaults.get("source_organism", ""),
                "species": defaults.get("species", ""),
                "host": defaults.get("host", ""),
                "host_age": "",
                # Supplementary rows don't carry IEDB's per-row "Host | MHC
                # Types Present"; set expansion will fall back to the
                # per-PMID curated allele pool when the row's
                # mhc_restriction is class-only (issue #137).
                "host_mhc_types": "",
                "process_type": process_type,
                "disease": disease,
                "disease_stage": "",
                "source_tissue": source_tissue,
                "cell_name": cell_name,
                "culture_condition": culture_condition,
                "antigen_processing_comments": "",
                "assay_comments": "",
                "qualitative_measurement": "",
                # Supplementary rows are MS-only (is_binding_assay=False)
                # and don't carry quantitative-binding-assay values, but
                # the columns are kept so observations.parquet stays
                # schema-compatible with scanner output (issue #148, #135).
                "assay_method": "",
                "response_measured": "",
                "measurement_units": "",
                "measurement_inequality": "",
                "quantitative_measurement": "",
                "quantitative_value": float("nan"),
                "is_binding_assay": False,
                "is_non_peptide_ligand": [
                    is_non_peptide_ligand(a) for a in df["mhc_restriction"].to_numpy()
                ],
                "is_potential_contaminant": df["is_potential_contaminant"].to_numpy(),
            }
        )

        # Classify per-unique-allele, then map back onto every row.
        # Within a single supplementary entry the non-allele inputs are
        # constant (from manifest defaults), so classify_ms_row varies
        # only by mhc_restriction.  Set expansion (issue #137) is also
        # per-allele since host_mhc_types is empty for supplements and
        # pmid + mhc_class don't vary within an entry.
        unique_alleles = record["mhc_restriction"].unique()

        def _set_for(allele: str, _record=record, _pmid=pmid) -> dict:
            mhc_class_filter = ""
            # Pull per-allele class from the just-built ``record`` if the
            # supplement CSV carries one; otherwise let set expansion
            # skip the class filter (returns the unfiltered candidate
            # set, which is fine because supplements rarely mix class I
            # and class II in one entry).
            if "mhc_class" in _record.columns:
                cls_match = _record.loc[_record["mhc_restriction"] == allele, "mhc_class"]
                if len(cls_match):
                    mhc_class_filter = str(cls_match.iloc[0])
            allele_set, prov, size = expand_allele_set(allele, "", _pmid, mhc_class_filter)
            return {
                "mhc_allele_set": allele_set,
                "mhc_allele_provenance": prov,
                "mhc_allele_set_size": size,
            }

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
                    **_set_for(a),
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
                    **_set_for(a),
                }
                for a in unique_alleles
            ]
        flags = pd.DataFrame(flag_rows)
        record = record.merge(flags, on="mhc_restriction", how="left")

        # Promote set → mhc_restriction (#45) for class-only supplement
        # rows where PMID-pool expansion produced a tightened set.  When
        # provenance is "exact" the row was already 4-digit and the
        # restriction is unchanged; for "pmid_class_pool" / "sample_*"
        # the multi-allele set IS the actual presenting MHC.  See the
        # parallel block in scanner.py for the IEDB path.
        if "mhc_allele_set" in record.columns and "mhc_allele_provenance" in record.columns:
            promote_mask = (
                (record["mhc_allele_set_size"].fillna(0) > 0)
                & (record["mhc_allele_provenance"] != "exact")
                & record["mhc_allele_set"].notna()
                & (record["mhc_allele_set"] != "")
            )
            if promote_mask.any():
                record.loc[promote_mask, "mhc_restriction"] = record.loc[
                    promote_mask, "mhc_allele_set"
                ]

        # Fallback: derive mhc_species from host when allele-based
        # classification is empty (e.g. supplementary peptides without
        # per-peptide allele assignments).
        host_species = normalize_species(defaults.get("host", ""))
        if "mhc_species" in record.columns:
            # Cast to ``StringDtype`` before fillna/replace — ``mhc_species``
            # may be a post-#137 categorical whose category set excludes
            # ``""`` and ``host_species``, in which case the in-place
            # assignment would raise ``TypeError``.  StringDtype accepts
            # any string fill.
            record["mhc_species"] = (
                record["mhc_species"].astype("string").fillna("").replace("", host_species)
            )
        else:
            record["mhc_species"] = host_species

        per_entry_frames.append(record)

    if not per_entry_frames:
        return pd.DataFrame()

    result = pd.concat(per_entry_frames, ignore_index=True)

    # Deduplicate within supplementary data.  Before #147 the key was
    # ``(peptide, mhc_restriction, pmid)`` which collapsed the same
    # peptide / allele seen in multiple sample CSVs from a single paper
    # (e.g. a peptide presented by both JY and Raji in Gomez-Zepeda 2024)
    # onto one arbitrary sample context.  Including ``supplementary_file``
    # preserves per-sample evidence while still de-duplicating within one
    # CSV — file-level duplicates still collapse correctly.
    dedupe_cols = ["peptide", "mhc_restriction", "pmid"]
    if "supplementary_file" in result.columns:
        dedupe_cols.append("supplementary_file")
    result = result.drop_duplicates(subset=dedupe_cols)

    return result
