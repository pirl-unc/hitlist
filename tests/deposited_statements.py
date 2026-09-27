"""Deposited elution-statement wording, in one place.

IEDB's phrasing is a join key: the exporter matches a curated
``elution_condition_ids`` entry against ``assay_comments`` verbatim, typos
included. Several suites build synthetic rows from it, so it lives here rather
than being transcribed into each of them or imported across test modules.
"""

#: PMID 33592498. ``HRGO02`` is IEDB's spelling of the HROG02 line and must be
#: reproduced exactly; the curated map keys on it.
GBM_STATEMENT = "The epitope was eluted from the following conditions: {}."
