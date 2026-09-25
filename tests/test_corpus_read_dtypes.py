"""The corpus read must not spend memory it does not need (#566).

``bool`` columns that carry nulls come back from ``read_parquet`` as
``object`` -- one Python pointer per row for two distinct values. Narrowing
them to pandas' nullable ``boolean`` is the part of this that measurably pays;
see ``_narrow_nullable_bools`` for the paths that did not.
"""

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from hitlist import observations


@pytest.fixture
def corpus(tmp_path):
    """A parquet shaped like the built corpus: a bool column carrying nulls,
    and a bool column that does not."""
    path = tmp_path / "observations.parquet"
    rows = 300
    table = pa.table(
        {
            "peptide": [f"PEPTIDE{i % 97}" for i in range(rows)],
            "has_ptm": pa.array([True, False, None] * (rows // 3), type=pa.bool_()),
            "is_binding_assay": pa.array([True, False] * (rows // 2), type=pa.bool_()),
        }
    )
    pq.write_table(table, path)
    return path


def test_null_carrying_bools_narrow_to_boolean(corpus):
    frame = observations._narrow_nullable_bools(pd.read_parquet(corpus), corpus)
    assert frame["has_ptm"].dtype == "boolean"
    assert frame["has_ptm"].memory_usage(deep=True) < pd.read_parquet(corpus)[
        "has_ptm"
    ].memory_usage(deep=True)


def test_null_free_bools_are_left_alone(corpus):
    """Nullable ``boolean`` costs a mask byte per row, so widening the columns
    that have no nulls would spend memory rather than save it."""
    frame = observations._narrow_nullable_bools(pd.read_parquet(corpus), corpus)
    assert frame["is_binding_assay"].dtype == bool


def test_values_and_nulls_survive_the_narrowing(corpus):
    plain = pd.read_parquet(corpus)
    frame = observations._narrow_nullable_bools(plain.copy(), corpus)
    assert frame["has_ptm"].isna().tolist() == plain["has_ptm"].isna().tolist()
    both = frame["has_ptm"].notna()
    assert frame.loc[both, "has_ptm"].astype(bool).tolist() == [
        bool(v) for v in plain.loc[both, "has_ptm"]
    ]


def test_columns_stay_writable(corpus):
    """A frame handed to callers must still accept in-place assignment: the
    zero-copy Arrow paths that saved memory returned read-only buffers and
    broke every ``df.loc[mask, col] = ...`` in the load path (#566)."""
    frame = observations._narrow_nullable_bools(pd.read_parquet(corpus), corpus)
    frame.loc[frame.index[:5], "peptide"] = "REWRITTEN"
    assert set(frame["peptide"].head(5)) == {"REWRITTEN"}
