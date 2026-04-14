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

"""Load the built peptide indexes with optional filters.

Two parallel parquet indexes are built by
:func:`hitlist.builder.build_observations`:

- ``observations.parquet`` — MS-eluted immunopeptidome rows (IEDB +
  CEDAR + supplementary).  Load with :func:`load_observations`.
- ``binding.parquet`` — binding-assay rows (refolding, MEDi, peptide
  microarray, quantitative-tier measurements).  Load with
  :func:`load_binding`.

The two indexes share the same schema but are never mixed: MS and
binding data go to separate files so downstream consumers cannot
accidentally conflate them.  Only the MS index gets supplementary
data and sample-level metadata joins (see :mod:`hitlist.export`).

Usage::

    from hitlist.observations import load_observations, load_binding

    ms = load_observations(mhc_class="I")
    bd = load_binding(mhc_class="I", mhc_restriction="HLA-A*02:01")
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .downloads import data_dir


def observations_path() -> Path:
    """Path to the MS-eluted observations parquet file."""
    return data_dir() / "observations.parquet"


def binding_path() -> Path:
    """Path to the binding-assay parquet file."""
    return data_dir() / "binding.parquet"


def is_built() -> bool:
    """Check if the observations table has been built."""
    return observations_path().exists()


def is_binding_built() -> bool:
    """Check if the binding-assay table has been built."""
    return binding_path().exists()


def load_observations(
    mhc_class: str | None = None,
    species: str | None = None,
    source: str | None = None,
    mhc_restriction: str | list[str] | None = None,
    gene_name: str | list[str] | None = None,
    gene_id: str | list[str] | None = None,
    peptide: str | list[str] | None = None,
    serotype: str | list[str] | None = None,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Load the built MS observations table with optional filters.

    The table contains only MS-eluted immunopeptidome observations.
    Binding-assay data is in a separate parquet — use
    :func:`load_binding` for that.

    Parameters
    ----------
    mhc_class
        Filter to ``"I"``, ``"II"``, or ``"non classical"``.
    species
        Filter by MHC species (e.g. ``"Homo sapiens"``).
    source
        Filter by data source (``"iedb"``, ``"cedar"``, ``"supplement"``).
    mhc_restriction
        Exact MHC allele filter.  Repeatable or comma-separated.
    gene_name, gene_id
        Gene filters — resolved through the peptide mappings sidecar.
    peptide, serotype, columns
        See module docstring.

    Raises
    ------
    FileNotFoundError
        If the observations table has not been built yet.
    """
    return _load_peptide_index(
        observations_path(),
        index_name="Observations",
        mhc_class=mhc_class,
        species=species,
        source=source,
        mhc_restriction=mhc_restriction,
        gene_name=gene_name,
        gene_id=gene_id,
        peptide=peptide,
        serotype=serotype,
        columns=columns,
    )


def load_binding(
    mhc_class: str | None = None,
    species: str | None = None,
    source: str | None = None,
    mhc_restriction: str | list[str] | None = None,
    gene_name: str | list[str] | None = None,
    gene_id: str | list[str] | None = None,
    peptide: str | list[str] | None = None,
    serotype: str | list[str] | None = None,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Load the built binding-assay table with optional filters.

    The binding index contains rows flagged as binding assays (peptide
    microarray, refolding, MEDi, quantitative-tier measurements).
    Supplementary data never contributes here — all supplementary
    rows are manually curated as MS.

    Filters match :func:`load_observations`.  Raises FileNotFoundError
    if the binding index has not been built yet.
    """
    return _load_peptide_index(
        binding_path(),
        index_name="Binding",
        mhc_class=mhc_class,
        species=species,
        source=source,
        mhc_restriction=mhc_restriction,
        gene_name=gene_name,
        gene_id=gene_id,
        peptide=peptide,
        serotype=serotype,
        columns=columns,
    )


def _load_peptide_index(
    path: Path,
    *,
    index_name: str,
    mhc_class: str | None,
    species: str | None,
    source: str | None,
    mhc_restriction: str | list[str] | None,
    gene_name: str | list[str] | None,
    gene_id: str | list[str] | None,
    peptide: str | list[str] | None,
    serotype: str | list[str] | None,
    columns: list[str] | None,
) -> pd.DataFrame:
    """Shared loader for the observations and binding parquets.

    Both indexes share the same schema; this helper centralizes filter
    pushdown, gene resolution via the mappings sidecar, and the
    semicolon-joined ``serotypes`` post-filter.
    """
    if not path.exists():
        raise FileNotFoundError(f"{index_name} table not built. Run: hitlist data build")

    from .curation import normalize_allele, normalize_species

    def _as_list(v) -> list[str]:
        if isinstance(v, str):
            return [s.strip() for s in v.split(",") if s.strip()]
        return [s for s in v if s]

    filters: list = []
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

    if gene_name is not None or gene_id is not None:
        import pyarrow.parquet as pq

        schema_names = set(pq.read_schema(path).names)
        if "gene_names" not in schema_names:
            raise ValueError(
                "Gene filtering requires a mappings-built index.\nRun: hitlist data build"
            )
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
            return pd.read_parquet(path, columns=columns, filters=[("peptide", "==", "__NONE__")])
        filters.append(("peptide", "in", matching_peptides))

    # Serotype filter runs after load — `serotypes` is a semicolon-joined
    # string column (an allele may belong to a locus-specific serotype AND
    # a public epitope like Bw4), so parquet pushdown can't express it.
    post_serotypes: list[str] | None = None
    if serotype is not None:
        post_serotypes = [_normalize_serotype_query(s) for s in _as_list(serotype)]
        if columns is not None and "serotypes" not in columns:
            read_columns = [*columns, "serotypes"]
        else:
            read_columns = columns

        import pyarrow.parquet as pq

        schema_names = set(pq.read_schema(path).names)
        if "serotypes" not in schema_names:
            raise ValueError(
                "Serotype filtering requires an index built with\n"
                "hitlist >= 1.7.0.  Run: hitlist data build --force"
            )
    else:
        read_columns = columns

    df = pd.read_parquet(path, columns=read_columns, filters=filters if filters else None)

    if post_serotypes:
        wanted = set(post_serotypes)
        mask = df["serotypes"].map(
            lambda s: bool(wanted & set(s.split(";"))) if isinstance(s, str) and s else False
        )
        df = df[mask]
        if columns is not None and "serotypes" not in columns:
            df = df.drop(columns=["serotypes"])

    return df


def _normalize_serotype_query(raw: str) -> str:
    """Normalize user serotype input to canonical ``HLA-X`` form.

    Accepts ``A24``, ``HLA-A24``, ``hla-a24``, ``Bw4``, etc.
    """
    s = raw.strip()
    if not s:
        return ""
    if s.upper().startswith("HLA-"):
        return "HLA-" + s[4:]
    return f"HLA-{s}"
