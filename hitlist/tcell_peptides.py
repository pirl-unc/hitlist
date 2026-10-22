"""Public source projection for directly tested human-origin T-cell peptides.

This iterator retains assay outcomes and context without assigning a training
target. Consumers must keep cellular response and multimer binding separate.
"""

from __future__ import annotations

import re
from collections.abc import Collection, Iterator

from .tcell_assays import _iter_assays_with_cells

HUMAN_TAXON_IRI = "http://purl.obolibrary.org/obo/NCBITaxon_9606"
CANONICAL_PEPTIDE = re.compile(r"^[ACDEFGHIKLMNPQRSTVWY]{5,40}$")
EXPERIMENTAL_RESTRICTION_CODES = frozenset(
    {"Single allele present", "T cell assay -Mismatched MHC molecules"}
)
QUALITATIVE_RESULTS = frozenset(
    {"Negative", "Positive", "Positive-Low", "Positive-Intermediate", "Positive-High"}
)
ASSAY_FAMILIES = frozenset(
    {
        "soluble_factor_release",
        "activation_or_proliferation",
        "cytotoxicity",
        "pmhc_multimer",
        "other_tcell_assay",
    }
)
EXTRA_FIELDS = (
    "Epitope | Source Organism IRI",
    "Epitope | Object Type",
    "Epitope | Name",
    "Epitope | Modified residues",
    "Epitope | Modifications",
    "Assay Antigen | Epitope Relation",
    "Assay Antigen | Name",
    "Assay Antigen | Source Organism IRI",
)


def _direct_human_peptide(row: dict, cells: dict[str, str]) -> bool:
    peptide = cells["Epitope | Name"]
    return (
        cells["Epitope | Source Organism IRI"] == HUMAN_TAXON_IRI
        and cells["Epitope | Object Type"] == "Linear peptide"
        and CANONICAL_PEPTIDE.fullmatch(peptide) is not None
        and not cells["Epitope | Modified residues"]
        and not cells["Epitope | Modifications"]
        and row["antigen_object_type"] == "Linear peptide"
        and cells["Assay Antigen | Epitope Relation"] == "Epitope"
        and cells["Assay Antigen | Name"] == peptide
        and cells["Assay Antigen | Source Organism IRI"] == HUMAN_TAXON_IRI
    )


def _restriction_status(metadata: dict) -> str:
    if metadata["restriction_name_resolution"] != "single_hla_two_field_name":
        return "no_single_hla_assignment"
    if metadata["restriction_evidence_code"] in EXPERIMENTAL_RESTRICTION_CODES:
        return "single_hla_experimental_restriction_evidence"
    return "single_hla_reported_only"


def iter_human_direct_peptide_tcell_assays(
    *,
    assay_families: Collection[str] | None = None,
    response_measured: str | None = None,
    iedb_path=None,
    cedar_path=None,
    data_root=None,
    catalog_path=None,
) -> Iterator[dict]:
    """Yield source-faithful assay rows for directly tested human peptides.

    ``assay_families`` accepts one family or a collection of source
    method/readout groups and defaults to all groups. ``response_measured``
    optionally requires one exact source
    readout. The boolean ``qualitative_signal_observed`` only maps the source's
    positive tiers versus ``Negative``; it is not a target shared across assay
    families. The pinned IEDB/CEDAR source files are hash-checked and shared
    assay IDs must agree before any row is returned.
    """
    if assay_families is None:
        selected = ASSAY_FAMILIES
    elif isinstance(assay_families, str):
        selected = frozenset({assay_families})
    else:
        selected = frozenset(assay_families)
    if not selected or not selected <= ASSAY_FAMILIES:
        raise ValueError(f"assay_families must be a nonempty subset of {sorted(ASSAY_FAMILIES)}")
    for metadata, cells in _iter_assays_with_cells(
        iedb_path=iedb_path,
        cedar_path=cedar_path,
        data_root=data_root,
        catalog_path=catalog_path,
        extra_fields=EXTRA_FIELDS,
    ):
        if not (
            _direct_human_peptide(metadata, cells)
            and metadata["assay_family"] in selected
            and metadata["qualitative_measurement"] in QUALITATIVE_RESULTS
            and (response_measured is None or metadata["response_measured"] == response_measured)
        ):
            continue
        yield {
            **metadata,
            "peptide_sequence": cells["Epitope | Name"],
            "reported_mhc_name": cells["MHC Restriction | Name"],
            "mhc_assignment_status": _restriction_status(metadata),
            "qualitative_signal_observed": metadata["qualitative_measurement"] != "Negative",
        }
