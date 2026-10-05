"""Dtype-preserving pandas operations shared by readers and exports."""

import pandas as pd


def fillna_scalar_safe(series: pd.Series, value) -> pd.Series:
    """``series.fillna(value)`` that tolerates Categorical dtype.

    Filling a Categorical with a value not already in its category set
    raises ``TypeError``; widening the categories first keeps the fill
    working without dropping the memory-saving categorical encoding (#263).
    No-op widening for non-categorical columns.
    """
    if isinstance(series.dtype, pd.CategoricalDtype) and value not in series.cat.categories:
        series = series.cat.add_categories([value])
    return series.fillna(value)
