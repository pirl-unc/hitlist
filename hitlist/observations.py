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

The observations table is a single parquet file containing all
IEDB + CEDAR MHC ligand observations with source classification.
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
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Load the built observations table with optional filters.

    Parameters
    ----------
    mhc_class
        Filter to ``"I"``, ``"II"``, or ``"non classical"``.
    species
        Filter by MHC species (e.g. ``"Homo sapiens"``).
    source
        Filter by data source (``"iedb"`` or ``"cedar"``).
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

    # Build parquet filters for pushdown
    filters = []
    if mhc_class is not None:
        filters.append(("mhc_class", "==", mhc_class))
    if species is not None:
        filters.append(("mhc_species", "==", species))
    if source is not None:
        filters.append(("source", "==", source))

    df = pd.read_parquet(
        path,
        columns=columns,
        filters=filters if filters else None,
    )
    return df
