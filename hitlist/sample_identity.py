# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

"""The single definition of *what counts as one sample* (#260, #502).

Every consumer that reports a sample count MUST go through this module.
This is a public, load-bearing contract, not a helper: the numbers it
produces are how a reader judges whether a candidate target is
well-attested, so two callers disagreeing about it is a correctness bug,
not a style problem.

Two orthogonal categories, never conflated:

``line_id``
    One per cell line profiled.  Keyed on ``cell_line_name`` +
    ``monoallelic_host``; the host catches the ~9K mono-allelic rows where
    the engineering platform is the only line identifier.

``donor_id`` / ``donor_type_id``
    Donors, for everything that isn't a cell line.  Keyed on
    ``attributed_sample_label``, falling back to the PMID.
    ``donor_type_id`` additionally splits a donor by ``cell_name``, because
    one donor yielding two cell types is two profiles.

The headline total is ``n_cell_lines + n_donor_cell_types`` — see
:func:`count_samples`.  Donors are counted by (donor, cell type) rather
than by donor so that a cohort paper profiling one donor's blood and
tumour counts as two samples, which is what it is.

**Why both tiers end in a PMID fallback.**  ``attributed_sample_label`` is
blank on 4,295,716 of 4,440,124 observation rows — 96.7%.  A count keyed
directly on it collapses every uncurated row into a single bucket: 13
distinct samples corpus-wide, against 2,244 with the fallback.  That was a
live bug (#502) in ``pmhc --by-gene``, which reported 1 sample for CTAG2
across 17 references.  The cell-line tier had the same hole with no
fallback at all, so a cell-line row carrying neither a name nor a host
silently vanished from the count.  A row we cannot identify precisely
still represents *some* sample; dropping it under-reports the evidence
behind a peptide, and a reader deciding whether a target is well-attested
should never be shown fewer samples than exist.

**Blank means "not in this category", not "unidentifiable".**  A cell-line
row has no donor ID and vice versa.  That is why callers use
:func:`count_distinct_ids` rather than ``Series.nunique()``, which would
count the blank as a bucket of its own.

**Not to be confused with** :func:`hitlist.samples.sample_peptidomes`,
which groups raw *scanner* output by ``(pmid, antigen_processing_comments)``.
That runs a pipeline stage earlier, before curation attaches
``src_cell_line`` / ``cell_line_name`` / ``monoallelic_host``, so it
cannot use this definition and deliberately does not try to.
"""

from __future__ import annotations

import pandas as pd

#: Observation columns :func:`sample_identity_ids` reads.  Callers must
#: project these explicitly — a missing column degrades to blank, which
#: silently undercounts rather than failing loudly (#502).
SAMPLE_IDENTITY_COLUMNS: tuple[str, ...] = (
    "pmid",
    "src_cell_line",
    "cell_line_name",
    "cell_name",
    "monoallelic_host",
    "attributed_sample_label",
)

#: Output column names.  Public so callers name them rather than
#: hard-coding strings that drift apart.
LINE_ID_COLUMN = "line_id"
DONOR_ID_COLUMN = "donor_id"
DONOR_TYPE_ID_COLUMN = "donor_type_id"

#: The three identity columns, in the order :func:`sample_identity_ids`
#: returns them.
SAMPLE_IDENTITY_ID_COLUMNS: tuple[str, str, str] = (
    LINE_ID_COLUMN,
    DONOR_ID_COLUMN,
    DONOR_TYPE_ID_COLUMN,
)


def _str_col(df: pd.DataFrame, name: str) -> pd.Series:
    if name in df.columns:
        return df[name].astype(object).fillna("").astype(str)
    return pd.Series([""] * len(df), index=df.index)


def sample_identity_ids(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Per-row ``(line_id, donor_id, donor_type_id)`` for an observations frame.

    Returns three ``Series`` aligned to ``df.index``.  Empty string means
    the row is not in that category (a cell-line row has no donor), so
    count with :func:`count_distinct_ids`, never ``nunique()``.

    See the module docstring for the definition and why each tier ends in
    a PMID fallback.
    """
    src_cell_line = (
        df["src_cell_line"].astype("boolean").fillna(False)
        if "src_cell_line" in df.columns
        else pd.Series([False] * len(df), index=df.index)
    )
    cell_line_name = _str_col(df, "cell_line_name")
    cell_name = _str_col(df, "cell_name")
    monoallelic_host = _str_col(df, "monoallelic_host")
    attributed_label = _str_col(df, "attributed_sample_label")
    pmid_str = (
        df["pmid"].astype("Int64").astype(str)
        if "pmid" in df.columns
        else pd.Series([""] * len(df), index=df.index)
    )
    pmid_key = "pmid:" + pmid_str

    named_line = cell_line_name + "|" + monoallelic_host
    line_id = named_line.where(
        (cell_line_name != "") | (monoallelic_host != ""),
        pmid_key,
    ).where(src_cell_line, "")

    donor_id = attributed_label.where(attributed_label != "", pmid_key)
    donor_type_id = (donor_id + "|" + cell_name).where(~src_cell_line, "")
    return line_id, donor_id.where(~src_cell_line, ""), donor_type_id


def add_sample_identity_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with the three identity columns attached
    under their public names (:data:`SAMPLE_IDENTITY_ID_COLUMNS`)."""
    line_id, donor_id, donor_type_id = sample_identity_ids(df)
    return df.assign(
        **{
            LINE_ID_COLUMN: line_id,
            DONOR_ID_COLUMN: donor_id,
            DONOR_TYPE_ID_COLUMN: donor_type_id,
        }
    )


def count_distinct_ids(values: pd.Series) -> int:
    """Distinct non-blank sample IDs.

    Blank means "not in this category", so a bare ``nunique()`` would
    count it as a bucket of its own and inflate every count by one.
    """
    return int(values[values.astype(str) != ""].nunique())


def count_samples(df: pd.DataFrame) -> int:
    """Headline sample count for a frame: distinct cell lines plus
    distinct (donor, cell-type) profiles.

    The union over the whole frame — for per-group counts, attach the
    columns with :func:`add_sample_identity_columns` and aggregate each
    with :func:`count_distinct_ids`.
    """
    line_id, _, donor_type_id = sample_identity_ids(df)
    return count_distinct_ids(line_id) + count_distinct_ids(donor_type_id)
