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

"""Export curated study metadata as structured tables.

Reads ``pmid_overrides.yaml`` and generates per-sample, per-species,
and allele-validation reports from the ``ms_samples`` and ``hla_alleles``
metadata fields.
"""

from __future__ import annotations

import pandas as pd

from .curation import load_pmid_overrides


def generate_ms_samples_table(mhc_class: str | None = None) -> pd.DataFrame:
    """Export all ms_samples entries as a flat DataFrame.

    Parameters
    ----------
    mhc_class
        Filter to ``"I"`` or ``"II"``.  Entries with ``"I+II"`` match
        either filter.  ``None`` returns all rows.

    Returns
    -------
    pd.DataFrame
        Columns: species, sample, perturbation, pmid, study, mhc_class,
        n_samples, notes.
    """
    overrides = load_pmid_overrides()
    rows: list[dict] = []

    for pmid_int, entry in sorted(overrides.items()):
        label = entry.get("label", "")
        species = entry.get("species", "Homo sapiens (human)")
        ms_samples = entry.get("ms_samples", [])

        for sample in ms_samples:
            cls = sample.get("mhc_class", "")
            if mhc_class and not _mhc_class_matches(cls, mhc_class):
                continue

            condition = sample.get("condition", "")
            if (
                not condition
                or condition.startswith("unperturbed")
                or condition == "—"
                or condition.startswith("NOT ")
            ):
                perturbation = ""
            else:
                perturbation = condition

            n = sample.get("n", "")
            if n == 0:
                continue  # skip "NOT profiled" placeholder rows

            rows.append(
                {
                    "species": species,
                    "sample": sample.get("type", ""),
                    "perturbation": perturbation,
                    "pmid": pmid_int,
                    "study": label,
                    "mhc_class": cls,
                    "n_samples": n if n != "" else None,
                    "notes": sample.get("classification", sample.get("reason", "")),
                }
            )

    return pd.DataFrame(rows)


def generate_species_summary(mhc_class: str | None = None) -> pd.DataFrame:
    """Summarize sample counts by species and MHC class.

    Parameters
    ----------
    mhc_class
        Optional filter (``"I"`` or ``"II"``).

    Returns
    -------
    pd.DataFrame
        Columns: species, mhc_class, n_studies, n_sample_types, n_samples.
    """
    df = generate_ms_samples_table(mhc_class=mhc_class)
    if df.empty:
        return pd.DataFrame(
            columns=["species", "mhc_class", "n_studies", "n_sample_types", "n_samples"]
        )

    # Expand I+II into separate rows for grouping
    expanded_rows: list[dict] = []
    for _, row in df.iterrows():
        cls = row["mhc_class"]
        classes = []
        parts = {p.strip() for p in cls.split("+")} if cls else {"unknown"}
        classes = sorted(parts)

        for c in classes:
            expanded_rows.append({**row.to_dict(), "mhc_class": c})

    expanded = pd.DataFrame(expanded_rows)
    if mhc_class:
        expanded = expanded[expanded["mhc_class"] == mhc_class]

    summary = (
        expanded.groupby(["species", "mhc_class"])
        .agg(
            n_studies=("pmid", "nunique"),
            n_sample_types=("sample", "count"),
            n_samples=("n_samples", lambda x: x.dropna().sum()),
        )
        .reset_index()
    )
    summary["n_samples"] = summary["n_samples"].astype(int)
    return summary


def validate_mhc_alleles() -> pd.DataFrame:
    """Parse all MHC alleles in pmid_overrides with mhcgnomes.

    Returns
    -------
    pd.DataFrame
        Columns: pmid, study, allele, parsed_name, parsed_type,
        species, valid.
    """
    try:
        from mhcgnomes import parse
    except ImportError:
        return pd.DataFrame(
            columns=["pmid", "study", "allele", "parsed_name", "parsed_type", "species", "valid"]
        )

    overrides = load_pmid_overrides()
    rows: list[dict] = []

    for pmid_int, entry in sorted(overrides.items()):
        label = entry.get("label", "")
        hla_alleles = entry.get("hla_alleles", {})
        if not hla_alleles:
            continue

        allele_strings = _extract_allele_strings(hla_alleles)
        for allele_str in sorted(set(allele_strings)):
            result = parse(allele_str)
            parsed_name = str(result)
            parsed_type = type(result).__name__
            species_name = ""
            if hasattr(result, "species"):
                species_name = result.species.name

            valid = parsed_type not in ("ParseError", "str")

            rows.append(
                {
                    "pmid": pmid_int,
                    "study": label,
                    "allele": allele_str,
                    "parsed_name": parsed_name,
                    "parsed_type": parsed_type,
                    "species": species_name,
                    "valid": valid,
                }
            )

    return pd.DataFrame(rows)


def _mhc_class_matches(sample_class: str, filter_class: str) -> bool:
    """Check if a sample's mhc_class matches a filter.

    ``"I"`` matches ``"I"`` and ``"I+II"`` but NOT ``"II"``.
    ``"II"`` matches ``"II"`` and ``"I+II"`` but NOT ``"I"``.
    """
    if not sample_class:
        return False
    parts = {p.strip() for p in sample_class.split("+")}
    return filter_class in parts


def _extract_allele_strings(hla_alleles: dict | list | str) -> list[str]:
    """Recursively extract allele strings from the hla_alleles field."""
    results: list[str] = []
    if isinstance(hla_alleles, str):
        # Could be a description like "51 HLA-I allotypes ..." — skip non-allele text
        if ("HLA-" in hla_alleles and "*" in hla_alleles) or hla_alleles.startswith("HLA-"):
            results.append(hla_alleles)
    elif isinstance(hla_alleles, list):
        for item in hla_alleles:
            results.extend(_extract_allele_strings(item))
    elif isinstance(hla_alleles, dict):
        for value in hla_alleles.values():
            results.extend(_extract_allele_strings(value))
    return results
