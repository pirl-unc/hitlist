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

"""Load the built observations table with optional filters.

The observations table is a single parquet file containing MS-eluted
immunopeptidome observations (IEDB + CEDAR + supplementary) with
source classification.  Binding assay data is excluded at build time.

Built by :func:`hitlist.builder.build_observations`.

Usage::

    from hitlist.observations import load_observations

    df = load_observations()                        # everything
    df = load_observations(mhc_class="I")           # class I only
    df = load_observations(species="Homo sapiens")  # human only
    df = load_observations(columns=["peptide", "mhc_restriction", "src_cancer"])
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .downloads import data_dir


def observations_path() -> Path:
    """Path to the built observations parquet file."""
    return data_dir() / "observations.parquet"


def is_built() -> bool:
    """Check if the observations table has been built."""
    return observations_path().exists()


def load_observations(
    mhc_class: str | None = None,
    species: str | None = None,
    source: str | None = None,
    mhc_restriction: str | list[str] | None = None,
    gene_name: str | list[str] | None = None,
    gene_id: str | list[str] | None = None,
    peptide: str | list[str] | None = None,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Load the built observations table with optional filters.

    The table contains only MS-eluted immunopeptidome observations.
    Binding assay data (peptide microarrays, refolding assays, etc.)
    is excluded at build time.

    Parameters
    ----------
    mhc_class
        Filter to ``"I"``, ``"II"``, or ``"non classical"``.
    species
        Filter by MHC species (e.g. ``"Homo sapiens"``).
    source
        Filter by data source (``"iedb"``, ``"cedar"``, ``"supplement"``).
    columns
        Load only these columns (pushed down to parquet reader).

    Returns
    -------
    pd.DataFrame
        The observations table, filtered as requested.

    Raises
    ------
    FileNotFoundError
        If the observations table has not been built yet.
    """
    path = observations_path()
    if not path.exists():
        raise FileNotFoundError("Observations table not built. Run: hitlist data build")

    from .curation import normalize_allele, normalize_species

    def _as_list(v) -> list[str]:
        if isinstance(v, str):
            return [s.strip() for s in v.split(",") if s.strip()]
        return [s for s in v if s]

    # Build parquet filters for pushdown (fast — pyarrow skips row groups
    # without matching values)
    filters = []
    if mhc_class is not None:
        filters.append(("mhc_class", "==", mhc_class))
    if species is not None:
        filters.append(("mhc_species", "==", normalize_species(species)))
    if source is not None:
        filters.append(("source", "==", source))
    if mhc_restriction is not None:
        values = [normalize_allele(v) for v in _as_list(mhc_restriction)]
        filters.append(("mhc_restriction", "in", values))

    if peptide is not None:
        filters.append(("peptide", "in", _as_list(peptide)))

    # Gene filters use the central semicolon-joined gene_names / gene_ids
    # columns (always populated by build when mappings are built).  These
    # columns are multi-valued to preserve paralog attribution; pushdown
    # exact-equality won't work, so we push down a peptide-set derived from
    # the peptide_mappings sidecar for efficiency.
    if gene_name is not None or gene_id is not None:
        import pyarrow.parquet as pq

        schema_names = set(pq.read_schema(path).names)
        if "gene_names" not in schema_names:
            raise ValueError(
                "Gene filtering requires a mappings-built observations table.\n"
                "Run: hitlist data build"
            )
        # Resolve gene → peptides via the mappings sidecar (fast parquet
        # pushdown on that table), then filter observations by peptide set.
        from .mappings import is_mappings_built, load_peptide_mappings

        if not is_mappings_built():
            raise ValueError("Peptide mappings not built.  Run: hitlist data build")
        mapping_filters: dict = {}
        if gene_name is not None:
            mapping_filters["gene_name"] = _as_list(gene_name)
        if gene_id is not None:
            mapping_filters["gene_id"] = _as_list(gene_id)
        hits = load_peptide_mappings(columns=["peptide"], **mapping_filters)
        matching_peptides = hits["peptide"].unique().tolist()
        if not matching_peptides:
            # No peptides match — return empty df with correct schema
            df = pd.read_parquet(path, columns=columns, filters=[("peptide", "==", "__NONE__")])
            return df
        filters.append(("peptide", "in", matching_peptides))

    df = pd.read_parquet(
        path,
        columns=columns,
        filters=filters if filters else None,
    )
    return df
