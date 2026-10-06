"""Sequence exclusions from positive MS evidence in distinct tissue donors.

Donor identities must be evidenced, never the PMID/sample-count fallback. The
primary HLA Ligand Atlas retains the donor axis absent from its IEDB deposit.
Donor HLA typing remains a sample genotype, not a measured peptide restriction.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .curation_yaml import load_curation_yaml
from .provenance import file_digest

ATLAS_RELEASE = "2020.12"
ATLAS_URL = f"https://hla-ligand-atlas.org/rel/{ATLAS_RELEASE}/"
POLICY_PATH = Path(__file__).parent / "data" / "tissue_blacklist.yaml"
_REQUIRED = (
    "peptide",
    "donor_id",
    "donor_status",
    "source_tissue",
    "tissue_status",
    "assay_modality",
    "is_cell_line",
    "source_record_id",
)


def tissue_blacklist_policy():
    """Return the recorded default rule and exact tissue aliases."""
    return load_curation_yaml(POLICY_PATH)


def _tissue_groups(values, policy=None):
    aliases = {
        name.casefold(): group
        for group, names in (policy or tissue_blacklist_policy())["tissue_groups"].items()
        for name in names
    }
    return values.astype("string").fillna("").str.strip().str.casefold().map(aliases).fillna("")


def build_tissue_blacklist(observations, *, min_donors=2, donor_aliases=None, policy=None):
    """Return peptide counts and audited rows, without filtering by HLA.

    Counts are distinct resolved people across the union of the configured
    tissues. Nonmalignant primary tissue with positive MS modality qualifies.
    Every input row survives in the audit. Unknown/pooled identities do not
    establish independent people, and zero qualifying donors is not safety.
    ``donor_aliases`` must contain reviewed, direct aliases to canonical people;
    chains and cycles are rejected rather than guessed.
    """
    if type(min_donors) is not int or min_donors < 1:
        raise ValueError("min_donors must be a positive integer")
    missing = set(_REQUIRED) - set(observations.columns)
    if missing:
        raise ValueError(f"Tissue observations lack required evidence fields: {sorted(missing)}")
    aliases = donor_aliases or {}
    if any(not k or not v or k == v or v in aliases for k, v in aliases.items()):
        raise ValueError("Donor aliases must directly identify distinct canonical people")
    audit = observations.copy()
    for col in _REQUIRED:
        if col != "is_cell_line":
            audit[col] = audit[col].astype("string").fillna("")
    if not audit.is_cell_line.isin([True, False]).all():
        raise ValueError("is_cell_line must be an explicit boolean")
    policy = policy or tissue_blacklist_policy()
    if policy.get("tissue_status") != "nonmalignant" or set(policy["tissue_groups"]) != {
        "heart",
        "brain",
        "lung",
    }:
        raise ValueError("Tissue policy must describe nonmalignant heart, brain and lung")
    audit["tissue_group"] = _tissue_groups(audit.source_tissue, policy)
    audit["canonical_donor_id"] = audit.donor_id.map(lambda d: aliases.get(d, d))
    audit["exclusion_reason"] = ""
    # First failing condition is the displayed reason; raw axes remain intact.
    gates = (
        (audit.peptide.str.fullmatch("[ACDEFGHIKLMNPQRSTVWYU]+"), "invalid_peptide_sequence"),
        (audit.tissue_group.ne(""), "outside_tissue_scope"),
        (audit.tissue_status.eq("nonmalignant"), "not_nonmalignant_tissue"),
        (~audit.is_cell_line.astype(bool), "cell_line"),
        (audit.assay_modality.eq("mass_spectrometry"), "not_positive_ms"),
        (
            audit.donor_status.eq("resolved") & audit.canonical_donor_id.str.strip().ne(""),
            "unresolved_donor",
        ),
        (audit.source_record_id.str.strip().ne(""), "missing_source_record"),
    )
    for valid, reason in gates:
        audit.loc[audit.exclusion_reason.eq("") & ~valid, "exclusion_reason"] = reason
    audit["qualified"] = audit.exclusion_reason.eq("")
    groups = ("heart", "brain", "lung")
    columns = [
        "peptide",
        "n_donors",
        *[f"n_donors_{group}" for group in groups],
        "n_qualified_observations",
        "n_unresolved_donor_observations",
        "donor_ids",
        "blacklisted",
    ]
    qualified = audit[audit.qualified]
    result = pd.DataFrame(index=pd.Index(sorted(audit.peptide.unique()), name="peptide"))
    grouped = qualified.groupby("peptide", sort=True)
    result["n_donors"] = grouped.canonical_donor_id.nunique()
    for group in groups:
        result[f"n_donors_{group}"] = (
            qualified.loc[qualified.tissue_group.eq(group)]
            .groupby("peptide")
            .canonical_donor_id.nunique()
        )
    result["n_qualified_observations"] = grouped.size()
    result["n_unresolved_donor_observations"] = (
        audit.loc[audit.exclusion_reason.eq("unresolved_donor")].groupby("peptide").size()
    )
    for col in result.columns:
        result[col] = result[col].fillna(0).astype("int64")
    result["donor_ids"] = grouped.canonical_donor_id.agg(
        lambda values: json.dumps(sorted(set(values)))
    )
    result["donor_ids"] = result.donor_ids.fillna("[]")
    result["blacklisted"] = result.n_donors >= min_donors
    return result.reset_index()[columns], audit


def _atlas_path(directory, name):
    matches = [
        directory / filename
        for filename in (f"HLA_{name}.tsv", f"{name}.tsv", f"{name}.tsv.gz", f"HLA_{name}.tsv.gz")
        if (directory / filename).is_file()
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one Atlas {name} table in {directory}; found {len(matches)}"
        )
    return matches[0]


def _table(path, required):
    frame = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    if not set(required) <= set(frame.columns):
        raise ValueError(f"Atlas {path.name} lacks columns {sorted(set(required) - set(frame))}")
    frame.insert(0, "source_row", range(1, len(frame) + 1))
    return frame


def load_atlas_tissue_evidence(directory, *, batch_size=25000):
    """Read a supplied 2020.12 Atlas snapshot in bounded sample-hit batches.

    Accept the release ZIP's HLA_*.tsv names or the public *.tsv.gz names. No
    network or implicit index rebuild occurs. Original sample-hit fields and
    logical row numbers survive; peptide and donor tables plus hashes bind the
    relationships. Only the configured essential tissues are materialized.
    The manifest records snapshot coverage, not a claim of exhaustive MS data.
    """
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    directory = Path(directory)
    paths = {name: _atlas_path(directory, name) for name in ("peptides", "donors", "sample_hits")}
    fingerprints = {name: file_digest(path) for name, path in paths.items()}
    peptides = _table(paths["peptides"], ("peptide_sequence_id", "peptide_sequence"))
    if peptides.peptide_sequence_id.duplicated().any() or peptides.peptide_sequence_id.eq("").any():
        raise ValueError("Atlas peptide IDs must be nonempty and unique")
    if not peptides.peptide_sequence.str.fullmatch("[ACDEFGHIKLMNPQRSTVWYU]+").all():
        raise ValueError(
            "Atlas peptide sequences must be unmodified amino acids (including selenocysteine U)"
        )
    donors = _table(paths["donors"], ("donor", "hla_allele"))
    if donors.donor.eq("").any() or donors.hla_allele.eq("").any():
        raise ValueError("Atlas donor typing requires nonempty donor and allele values")
    typing = donors.groupby("donor").hla_allele.agg(lambda x: ";".join(sorted(set(x))))
    lookup = peptides.set_index("peptide_sequence_id")
    frames, n_rows = [], 0
    for frame in pd.read_csv(
        paths["sample_hits"], sep="\t", dtype=str, keep_default_na=False, chunksize=batch_size
    ):
        required = {"peptide_sequence_id", "donor", "tissue", "hla_class"}
        if not required <= set(frame):
            raise ValueError("Atlas sample hits lack peptide, donor, tissue or HLA class")
        frame.insert(0, "sample_hits_row", range(n_rows + 1, n_rows + len(frame) + 1))
        n_rows += len(frame)
        if not frame.peptide_sequence_id.isin(lookup.index).all():
            raise ValueError("Atlas sample hit has no peptide reference")
        if not frame.donor.isin(typing.index).all():
            raise ValueError("Atlas sample hit has no donor reference")
        if not frame.hla_class.isin(["HLA-I", "HLA-II"]).all():
            raise ValueError("Atlas sample hit has an unknown HLA class")
        selected = frame[_tissue_groups(frame.tissue).ne("")].copy()
        if len(selected):
            frames.append(selected)
    hits = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(
            columns=["sample_hits_row", "peptide_sequence_id", "donor", "tissue", "hla_class"]
        )
    )
    hits["peptide"] = hits.peptide_sequence_id.map(lookup.peptide_sequence)
    hits["peptide_row"] = hits.peptide_sequence_id.map(lookup.source_row)
    # Preserve all original peptide-reference fields, including future extra
    # source columns, without replacing the sample-hit table's original cells.
    for column in lookup.columns:
        if column != "source_row":
            hits[f"peptide_source:{column}"] = hits.peptide_sequence_id.map(lookup[column])
    hits["source_tissue"] = hits.tissue
    hits["source_dataset"] = f"hla-ligand-atlas:{ATLAS_RELEASE}"
    hits["source_record_id"] = (
        hits.source_dataset + ":sample_hits:" + hits.sample_hits_row.astype(str)
    )
    hits["donor_id"] = "hla-ligand-atlas:" + hits.donor
    hits["donor_status"] = "resolved"
    hits["tissue_status"] = "nonmalignant"
    hits["assay_modality"] = "mass_spectrometry"
    hits["is_cell_line"] = False
    hits["mhc_class"] = hits.hla_class.map({"HLA-I": "I", "HLA-II": "II"})
    hits["mhc_restriction"] = ""
    hits["sample_alleles"] = hits.donor.map(typing)
    hits["pmid"] = 33858848
    if fingerprints != {name: file_digest(path) for name, path in paths.items()}:
        raise ValueError("Atlas inputs changed while reading; retry a stable snapshot")
    sources = {
        "release": ATLAS_RELEASE,
        "source_url": "https://hla-ligand-atlas.org/data",
        "publication": "https://doi.org/10.1136/jitc-2020-002071",
        "license": "CC-BY-4.0",
        "coverage": "Supplied Atlas snapshot; not an exhaustive survey of all tissue MS studies",
        "n_source_sample_hits": n_rows,
        "n_source_peptides": len(peptides),
        "n_source_donors": len(typing),
        "files": {
            name: {
                "path": str(path.resolve()),
                "url": f"{ATLAS_URL}{name}.tsv.gz",
                **fingerprints[name],
            }
            for name, path in paths.items()
        },
        "donor_typing_rows": json.loads(donors.to_json(orient="records")),
    }
    return hits, sources
