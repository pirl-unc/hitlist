"""Direct donor-linked HLA Ligand Atlas observations, separate from IEDB rows."""

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import quote

from datacache import install_bundle

from .curation_yaml import load_curation_yaml
from .downloads import data_dir
from .tissue_blacklist import ATLAS_RELEASE, _load_atlas_evidence

CATALOG_PATH = Path(__file__).parent / "data" / "hla_ligand_atlas.yaml"


def atlas_sources(*, data_root=None):
    """Install the pinned primary tables as one shared datacache generation."""
    catalog = load_curation_yaml(CATALOG_PATH)
    return install_bundle(
        Path(data_root or data_dir()) / "hla-ligand-atlas" / ATLAS_RELEASE / "sources",
        catalog["files"],
        download_options={"timeout": 60, "show_progress": False},
    )


def load_atlas_observations(directory=None, *, data_root=None, batch_size=25000):
    """Read every primary sample hit with its directly reported donor identity.

    Omit ``directory`` to acquire the pinned public source tables through datacache.
    A supplied directory is read without network access and its exact hashes are
    returned. Donor typing is not a measured peptide restriction. A reported
    donor/tissue/HLA-class group is not a uniquely resolved physical specimen or
    acquisition. This source must not be merged with its IEDB mirror as independent
    observations or used to infer donor attribution for historical IEDB assays.
    """
    if directory is None:
        paths = atlas_sources(data_root=data_root)
        directory = Path(paths["peptides.tsv.gz"]).parent
    observations, sources = _load_atlas_evidence(directory, batch_size=batch_size)
    keys = ["peptide_sequence_id", "donor", "tissue", "hla_class"]
    if observations[keys].eq("").any().any() or observations.duplicated(keys).any():
        raise ValueError("Atlas sample-hit keys must be nonempty and unique")
    # Preserve raw table cells and physical row pointers. Semantic identities
    # include escaped components so delimiters cannot collapse distinct records.
    prefix = f"hla-ligand-atlas:{ATLAS_RELEASE}"

    def encode(values):
        return values.map(lambda value: quote(value, safe=""))

    group = encode(observations.donor) + ":" + encode(observations.tissue)
    observations["reported_sample_group_id"] = (
        prefix + ":sample-group:" + group + ":" + encode(observations.hla_class)
    )
    observations["evidence_row_id"] = (
        observations.reported_sample_group_id
        + ":peptide:"
        + encode(observations.peptide_sequence_id)
    )
    observations["donor_evidence_basis"] = "primary_sample_hit_donor_reference"
    observations["reported_sample_group_basis"] = "source_donor_tissue_hla_class_triplet"
    observations["specimen_ids"] = "[]"
    observations["specimen_status"] = "unknown"
    observations["acquisition_ids"] = "[]"
    observations["acquisition_status"] = "unknown"
    observations["assay_method"] = "cellular MHC/mass spectrometry"
    observations["label_kind"] = "observed_eluted_ligand"
    observations["label_policy"] = "positive_observation_only_no_unobserved_negative"
    observations["source_reference"] = "https://doi.org/10.1136/jitc-2020-002071"
    observations["source_url"] = "https://hla-ligand-atlas.org/data"
    observations["source_accession"] = "PXD019643"
    observations["cell_type"] = ""
    observations["cell_condition"] = "direct_ex_vivo"
    observations["cell_condition_basis"] = "primary_publication_tissue_ligand_isolation"
    # Low-cardinality factual fields are categorical without changing their values.
    for column in (
        "donor",
        "tissue",
        "hla_class",
        "donor_id",
        "donor_status",
        "source_tissue",
        "tissue_status",
        "assay_modality",
        "mhc_class",
        "sample_alleles",
        "source_dataset",
        "reported_sample_group_id",
        "donor_evidence_basis",
        "reported_sample_group_basis",
        "specimen_ids",
        "specimen_status",
        "acquisition_ids",
        "acquisition_status",
        "assay_method",
        "label_kind",
        "label_policy",
        "source_reference",
        "source_url",
        "source_accession",
        "cell_type",
        "cell_condition",
        "cell_condition_basis",
    ):
        observations[column] = observations[column].astype("category")
    sources.update(
        {
            "n_observations": len(observations),
            "n_reported_sample_groups": observations.reported_sample_group_id.nunique(),
            "n_observed_donors": observations.donor_id.nunique(),
            "n_tissues": observations.source_tissue.nunique(),
            "identity_policy": "direct_source_donor_and_sample_group_v1",
            "limits": [
                "Physical specimen, aliquot and acquisition identities are unresolved.",
                "Positive-only ligand observations; unobserved does not mean negative.",
                "Complete donor typing is not a measured peptide-specific MHC restriction.",
                "Source QC included binding predictions; no prediction scores become labels.",
                "IEDB mirrors and reanalyses must not be treated as independent evidence.",
            ],
        }
    )
    # Keep the returned provenance strictly serializable, including NumPy counts.
    sources = json.loads(json.dumps(sources, default=int))
    return observations, sources
