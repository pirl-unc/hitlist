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

"""Index-writing helpers shared by the builder and the per-index modules.

A leaf module — it imports nothing from ``hitlist`` — so the builder and the
per-index modules (``line_expression``) can both depend on it without an
import cycle.
"""

from __future__ import annotations

import contextlib
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


def concat_non_empty(frames, columns, *, sort: bool = True) -> pd.DataFrame:
    """Concatenate frames without letting empty / all-NA entries pick dtypes.

    pandas deprecated inferring result dtypes from empty or all-NA
    entries: when the default flips, a concat where one frame contributes
    only NA for a column will infer ``object`` where a typed column is
    expected today.  Every caller feeds a parquet write or a reader of
    one, and an
    ``object`` column either changes the on-disk schema or makes pyarrow
    reject the write outright.

    Dropping whole frames is not enough — the deprecation is about
    all-NA *columns*, and a frame can be perfectly good apart from one.
    So instead of excluding data, this removes the ambiguity: for every
    column, the dtype is taken from the first frame that actually has
    values for it, and any frame contributing only NA is cast to that
    dtype before the concat. The result is identical to today's
    behaviour and stays identical when pandas changes.

    ``columns`` shapes the empty frame returned when every input is
    empty, so callers keep their column contract.
    """
    usable = [f for f in frames if f is not None and not f.empty]
    if not usable:
        return pd.DataFrame(columns=list(columns))

    # Authoritative dtype per column: the first frame with real values.
    authoritative: dict = {}
    for frame in usable:
        for col in frame.columns:
            if col not in authoritative and not frame[col].isna().all():
                authoritative[col] = frame[col].dtype

    aligned = []
    for frame in usable:
        frame = frame.copy()
        for col, dtype in authoritative.items():
            if col in frame.columns and frame[col].isna().all():
                # A dtype that cannot represent NA (e.g. plain int) is
                # left alone; pandas' own promotion is correct there.
                with contextlib.suppress(TypeError, ValueError):
                    frame[col] = frame[col].astype(dtype)
        aligned.append(frame)
    return pd.concat(aligned, ignore_index=True, sort=sort)
