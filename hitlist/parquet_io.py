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

"""Atomic parquet writes shared by every index writer.

A leaf module — it imports nothing from ``hitlist`` — so the builder and the
per-index modules (``line_expression``) can both depend on it without an
import cycle.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pandas as pd


def atomic_write_parquet(
    df: pd.DataFrame,
    path: Path,
    *,
    metadata: Mapping[bytes, bytes] | None = None,
) -> Path:
    """Write ``df`` to a sibling ``.partial`` file, then rename over ``path``.

    Readers of ``path`` keep seeing whatever was there before this call until
    the rename swaps the new file in, which closes the mid-rebuild window
    (#105) where a canonical parquet was briefly incomplete.

    ``metadata`` is merged into the parquet schema's key-value metadata
    (alongside pandas' own), so a stamp that describes the file travels
    inside it: a copied or downloaded index carries its own provenance.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = Path(path)
    table = pa.Table.from_pandas(df, preserve_index=False)
    if metadata:
        table = table.replace_schema_metadata({**(table.schema.metadata or {}), **metadata})
    partial = path.with_suffix(path.suffix + ".partial")
    pq.write_table(table, partial)
    partial.replace(path)
    return path
