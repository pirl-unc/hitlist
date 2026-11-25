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
CROSSWALK_CATALOG_PATH = Path(__file__).parent / "data" / "atlas_community_crosswalk.yaml"


def atlas_sources(*, data_root=None):
    """Install the pinned primary tables as one shared datacache generation."""
    catalog = load_curation_yaml(CATALOG_PATH)
    return install_bundle(
        Path(data_root or data_dir()) / "hla-ligand-atlas" / ATLAS_RELEASE / "sources",
        catalog["files"],
        download_options={"timeout": 60, "show_progress": False},
    )


def atlas_crosswalk_sources(*, data_root=None):
    """Install the pinned PXD019643 community SDRF crosswalk tables via datacache."""
    catalog = load_curation_yaml(CROSSWALK_CATALOG_PATH)
    return install_bundle(
        Path(data_root or data_dir()) / "hla-ligand-atlas" / ATLAS_RELEASE / "community-crosswalk",
        catalog["files"],
        download_options={"timeout": 120, "show_progress": False},
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


def atlas_specimen_lineage_contract() -> dict:
    """Return pinned PXD019643 SDRF/inventory crosswalk and QC-inclusion limits (#705)."""
    return {
        "tracked_issue": 705,
        "atlas_release": ATLAS_RELEASE,
        "source_accession": "PXD019643",
        "crosswalk_snapshot_date": "2026-10-20",
        "pinned_sources": {
            "sample_hits.tsv.gz": {
                "sha256": "0e8771113cae27c620b2b9619f035675ddcc87b8d7dfac50750e71ab60eb0167",
                "n_observations": 967449,
                "n_hla_i_observations": 409503,
                "n_hla_ii_observations": 557946,
                "n_primary_donor_tissue_class_groups": 418,
                "has_run_or_result_key": False,
            },
            "PXD019643_community_annotated.sdrf.tsv": {
                "sha256": "baa1907cc847abc382683492970eef9da6fbbb24ef4448dab6dbe76357c67fb2",
                "size_bytes": 1177127,
                "n_rows": 1468,
                "n_hla_i_rows": 752,
                "n_hla_ii_rows": 716,
                "n_exact_peak_inventory_matches": 1468,
                "n_exact_raw_stem_matches": 1466,
                "n_unmatched_raw_stems": 2,
            },
            "pride_files_inventory.json": {
                "sha256": "3bac909670e737f9f71d8bb5aad1f5e26b33f68f2c3b8bace1a350f8d6b987b0",
                "n_records": 3426,
                "n_result_records_unlinked_to_observations": 483,
            },
        },
        "n_multi_source_name_primary_groups": 5,
        "n_multi_source_name_observations": 28350,
        "qc_inclusion_status": "unknown",
        "observation_specimen_status": "unknown",
        "observation_acquisition_status": "unknown",
        "acquisition_independence_verdict": "inconclusive",
        "limits": [
            "PRIDE PXD019643 deposits QC-excluded runs alongside accepted runs; inventory membership does not prove QC inclusion.",
            "Primary 2020.12 sample_hits.tsv.gz lacks a run or result-file key linking peptides to individual acquisitions or physical specimens.",
            "Five primary donor/tissue/class groups carry multiple SDRF source names (28,350 observations); do not manufacture specimen IDs from filenames or display labels.",
            "Frozen donor-only cohort partitions remain unchanged until a source-backed included-run/result mapping is recovered.",
        ],
    }


def audit_atlas_sdrf_specimen_lineage(sdrf_df, inventory_df, primary_groups_df) -> dict:
    """Audit exact SDRF-to-inventory peak matching and observation-level QC/specimen status (#705).

    Enforces:
    - Exact 1:1 matching of SDRF ``comment[data file]`` to PRIDE ``PEAK`` filenames
      (no silent regex/punctuation alias guessing).
    - Explicit flagging of primary donor/tissue/class groups spanning multiple SDRF
      ``source name`` entries.
    - Explicit ``n_observations_qc_resolved == 0`` and ``specimen_status == "unknown"``
      whenever ``primary_groups_df`` lacks a source-backed included run/result key.
    """
    import pandas as pd

    for name, frame in (
        ("sdrf_df", sdrf_df),
        ("inventory_df", inventory_df),
        ("primary_groups_df", primary_groups_df),
    ):
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(f"{name} must be a pandas DataFrame")

    for col in ("source name", "comment[data file]", "donor", "tissue", "hla_class"):
        if col not in sdrf_df.columns:
            raise ValueError(f"sdrf_df missing required column {col!r}")
    for col in ("fileName", "fileCategory"):
        if col not in inventory_df.columns:
            raise ValueError(f"inventory_df missing required column {col!r}")
    for col in ("donor", "tissue", "hla_class", "n_observations"):
        if col not in primary_groups_df.columns:
            raise ValueError(f"primary_groups_df missing required column {col!r}")

    peak_files = inventory_df.loc[
        inventory_df["fileCategory"].astype(str).str.upper() == "PEAK", "fileName"
    ].astype(str)
    peak_counts = peak_files.value_counts().to_dict()

    n_exact_unique_peak = 0
    n_unmatched_peak = 0
    n_ambiguous_peak = 0
    for fname in sdrf_df["comment[data file]"].astype(str):
        cnt = peak_counts.get(fname, 0)
        if cnt == 1:
            n_exact_unique_peak += 1
        elif cnt == 0:
            n_unmatched_peak += 1
        else:
            n_ambiguous_peak += 1

    # Group SDRF source names by normalized (donor, lower(strip(tissue)), hla_class)
    sdrf_keyed = sdrf_df.copy()
    sdrf_keyed["_tissue_norm"] = sdrf_keyed["tissue"].astype(str).str.strip().str.casefold()
    sdrf_sources_by_group = (
        sdrf_keyed.groupby(["donor", "_tissue_norm", "hla_class"])["source name"]
        .apply(lambda s: sorted({str(v).strip() for v in s if str(v).strip()}))
        .to_dict()
    )

    has_run_key = "included_run_id" in primary_groups_df.columns
    n_total_obs = 0
    n_candidate_context_obs = 0
    n_multi_source_groups = 0
    n_multi_source_obs = 0
    n_qc_resolved_obs = 0
    multi_source_group_keys: list[str] = []

    for _, row in primary_groups_df.iterrows():
        donor = str(row["donor"]).strip()
        tissue_norm = str(row["tissue"]).strip().casefold()
        hla_class = str(row["hla_class"]).strip()
        n_obs = int(row["n_observations"])
        n_total_obs += n_obs
        sources = sdrf_sources_by_group.get((donor, tissue_norm, hla_class), [])
        if sources:
            n_candidate_context_obs += n_obs
        if len(sources) > 1:
            n_multi_source_groups += 1
            n_multi_source_obs += n_obs
            multi_source_group_keys.append(f"{donor}|{tissue_norm}|{hla_class}")
        if has_run_key and str(row.get("included_run_id") or "").strip():
            n_qc_resolved_obs += n_obs

    return {
        "tracked_issue": 705,
        "n_sdrf_rows": len(sdrf_df),
        "n_exact_unique_peak_matches": n_exact_unique_peak,
        "n_unmatched_peak_files": n_unmatched_peak,
        "n_ambiguous_peak_files": n_ambiguous_peak,
        "n_primary_groups": len(primary_groups_df),
        "n_total_observations": n_total_obs,
        "n_candidate_context_observations": n_candidate_context_obs,
        "n_multi_source_name_groups": n_multi_source_groups,
        "n_multi_source_name_observations": n_multi_source_obs,
        "multi_source_name_groups": sorted(multi_source_group_keys),
        "n_observations_qc_resolved": n_qc_resolved_obs,
        "qc_inclusion_status": "resolved"
        if n_qc_resolved_obs == n_total_obs and n_total_obs > 0
        else "unknown",
        "observation_specimen_status": "unknown",
        "acquisition_independence_verdict": "inconclusive",
    }
