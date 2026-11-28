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
from collections.abc import Mapping, Sequence
from pathlib import Path

import pandas as pd

_MAX_SMALL_STRING_BYTES = 1_500_000_000


def _infer_concrete_type_for_null_column(name: str, pa):
    """Pick a deterministic non-null Arrow type for an all-NA column (#990)."""
    if name.startswith(("is_", "has_", "src_")):
        return pa.bool_()
    if (
        name in {"pmid", "position", "start_position", "end_position", "length", "taxon_id"}
        or name.startswith("n_")
        or name.endswith(
            ("_size", "_count", "_replicates_possible", "_replicates_detected", "_position")
        )
    ):
        return pa.int64()
    if name.endswith(("_value", "_fraction", "_percentile", "_coverage", "_tpm")) or name in {
        "tpm",
        "log2_tpm",
        "log2_intensity",
    }:
        return pa.float64()
    return pa.string()


def _column_buffer_bytes(column) -> int:
    total = 0
    for chunk in column.chunks:
        for buf in chunk.buffers():
            if buf is not None:
                total += buf.size
    return total


def standardize_arrow_table(
    table,
    *,
    column_types: Mapping[str, object] | None = None,
):
    """Standardize Arrow dtypes before writing Parquet (#990).

    - Replaces Arrow ``null``-typed columns with explicit concrete types.
    - Normalizes ``large_string`` (and ``dictionary<*, large_string>``) to
      standard ``string`` whenever the column fits safely below the 2 GiB
      32-bit offset limit, avoiding cross-table ``string`` vs ``large_string``
      coercion copies on joins.
    """
    import pyarrow as pa

    overrides = column_types or {}
    new_fields = []
    new_columns = []
    changed = False

    for i, field in enumerate(table.schema):
        col = table.column(i)
        ftype = field.type
        target_type = overrides.get(field.name)

        if target_type is None:
            if pa.types.is_null(ftype):
                target_type = _infer_concrete_type_for_null_column(field.name, pa)
            elif pa.types.is_large_string(ftype):
                if _column_buffer_bytes(col) < _MAX_SMALL_STRING_BYTES:
                    target_type = pa.string()
            elif pa.types.is_dictionary(ftype) and pa.types.is_large_string(ftype.value_type):
                target_type = pa.dictionary(
                    ftype.index_type, pa.string(), ordered=bool(ftype.ordered)
                )

        if target_type is not None and target_type != ftype:
            col = col.cast(target_type)
            field = pa.field(field.name, target_type, nullable=field.nullable)
            changed = True

        new_fields.append(field)
        new_columns.append(col)

    if not changed:
        return table
    return pa.Table.from_arrays(
        new_columns, schema=pa.schema(new_fields, metadata=table.schema.metadata)
    )


def _sort_frame_for_parquet(df: pd.DataFrame, sort_by: Sequence[str]) -> pd.DataFrame:
    """Sort ``df`` lexicographically on ``sort_by`` without changing column dtypes (#990)."""
    present = [c for c in sort_by if c in df.columns]
    if not present or len(df) <= 1:
        return df
    keys: dict[str, pd.Series] = {}
    for col in present:
        s = df[col]
        if (
            isinstance(s.dtype, pd.CategoricalDtype)
            or pd.api.types.is_string_dtype(s)
            or s.dtype == object
        ):
            keys[col] = s.astype("string").fillna("")
        else:
            keys[col] = s
    order = (
        pd.DataFrame(keys, index=df.index)
        .sort_values(present, kind="mergesort", na_position="last")
        .index
    )
    return df.loc[order]


def atomic_write_parquet(
    df: pd.DataFrame,
    path: Path,
    *,
    metadata: Mapping[bytes, bytes] | None = None,
    sort_by: Sequence[str] | None = None,
    row_group_size: int | None = None,
    column_types: Mapping[str, object] | None = None,
    standardize_types: bool = True,
) -> Path:
    """Write ``df`` to a sibling ``.partial`` file, then rename over ``path``.

    Readers of ``path`` keep seeing whatever was there before this call until
    the rename swaps the new file in, which closes the mid-rebuild window
    (#105) where a canonical parquet was briefly incomplete.

    ``metadata`` is merged into the parquet schema's key-value metadata
    (alongside pandas' own), so a stamp that describes the file travels
    inside it: a copied or downloaded index carries its own provenance.

    ``sort_by`` orders rows lexicographically before writing so Parquet
    row-group min/max statistics support fast predicate pushdown (#990).
    ``standardize_types=True`` eliminates Arrow ``null``-typed columns and
    normalizes ``large_string`` to ``string`` across bundle tables (#990).
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = Path(path)
    # This function is the single owner of data-directory creation on the
    # index write path: resolving ``data_dir()`` is deliberately side-effect
    # free (#291/#579), so the first build into a never-used cache dir arrives
    # here with no directory yet.  Every canonical parquet — observations,
    # binding, bulk_proteomics, line_expression, peptide_mappings — is written
    # through here, so there is one place to get this right.
    path.parent.mkdir(parents=True, exist_ok=True)
    if sort_by:
        df = _sort_frame_for_parquet(df, sort_by)
    table = pa.Table.from_pandas(df, preserve_index=False)
    if standardize_types or column_types:
        table = standardize_arrow_table(table, column_types=column_types)
    if metadata:
        table = table.replace_schema_metadata({**(table.schema.metadata or {}), **metadata})
    partial = path.with_suffix(path.suffix + ".partial")
    write_kwargs = {"row_group_size": row_group_size} if row_group_size else {}
    pq.write_table(table, partial, **write_kwargs)
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
