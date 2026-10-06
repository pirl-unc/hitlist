"""Portable CTA-expression evidence and allele-independent tissue exclusions."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import pandas as pd

from .cta_expression import resolve_expression_table, summarize_cta_mappings
from .provenance import file_digest
from .tissue_blacklist import (
    _tissue_groups,
    build_tissue_blacklist,
    load_atlas_tissue_evidence,
    tissue_blacklist_policy,
)
from .training_bundle import (
    _contributor_identities,
    _coverage,
    _identity_table,
    _inputs,
    _stream_contributors,
    _write_json,
)

_TISSUE_FILES = {
    "tissue_evidence.parquet",
    "tissue_risk.parquet",
    "blacklist.parquet",
    "atlas_donors.parquet",
    "forbidden_sequences.txt",
}
_CTA_FILES = {
    "expression.parquet",
    "cta_reference.parquet",
    "peptides.parquet",
    "mappings.parquet",
    "expression_mappings.parquet",
    "presentation.parquet",
    "excluded_observations.parquet",
    "identities.parquet",
    "contributors.parquet",
    "lineage.json",
}


def _parquet(frame, path):
    frame.to_parquet(path, index=False, compression="zstd")


def _write_tissues(staging, atlas_dir):
    observations, sources = load_atlas_tissue_evidence(atlas_dir)
    summary, audit = build_tissue_blacklist(observations)
    blacklist = summary[summary.blacklisted].reset_index(drop=True)
    _parquet(audit, staging / "tissue_evidence.parquet")
    _parquet(summary, staging / "tissue_risk.parquet")
    _parquet(blacklist, staging / "blacklist.parquet")
    _parquet(pd.DataFrame(sources.pop("donor_typing_rows")), staging / "atlas_donors.parquet")
    (staging / "forbidden_sequences.txt").write_text("".join(f"{p}\n" for p in blacklist.peptide))
    return summary, sources


def _stable_source_files(metadata):
    for entry in metadata.values():
        if file_digest(Path(entry["path"])) != {k: entry[k] for k in ("sha256", "size_bytes")}:
            raise ValueError("Evidence source inputs changed during export")


def _write_bundle(directory, kind, build, final_check=None):
    from .version import __version__

    destination = Path(directory)
    if destination.exists():
        raise FileExistsError(f"Evidence bundle already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    try:
        metadata = build(staging)
        files = _TISSUE_FILES | (_CTA_FILES if kind == "cta_expression" else set())
        policy = {
            **tissue_blacklist_policy(),
            "hla_filter": None,
            "donor_count_scope": "distinct people across the union of the selected tissues",
            "sequence_match": "exact contiguous sequence; no inferred nested epitopes",
            "downstream_rule": "Exclude blacklisted sequences wherever they occur, including inside longer antigens",
        }
        manifest = {
            "schema_version": 1,
            "kind": kind,
            "hitlist_version": __version__,
            "policy": policy,
            **metadata,
            "artifacts": {name: file_digest(staging / name) for name in sorted(files)},
        }
        _write_json(staging / "manifest.json", manifest)
        verify_evidence_bundle(staging)
        _stable_source_files(metadata["atlas"]["files"])
        if final_check is not None:
            final_check(metadata)
        # Reserve without replacing another producer's destination; manifest is
        # published last so consumers never accept an unfinished generation.
        destination.mkdir()
        try:
            for path in staging.iterdir():
                if path.name != "manifest.json":
                    path.replace(destination / path.name)
            (staging / "manifest.json").replace(destination / "manifest.json")
        except BaseException:
            shutil.rmtree(destination)
            raise
    finally:
        shutil.rmtree(staging)
    return destination / "manifest.json"


def write_tissue_blacklist_bundle(directory, *, atlas_dir):
    """Export a standalone, donor-resolved nonmalignant-tissue blacklist.

    No expression or HLA filter can narrow this reusable sequence exclusion
    set. A supplied primary Atlas snapshot is required, with source scope and
    hashes recorded. No index rebuild or network download is implicit.
    """

    def build(staging):
        summary, sources = _write_tissues(staging, atlas_dir)
        return {"atlas": sources, "n_blacklisted_peptides": int(summary.blacklisted.sum())}

    return _write_bundle(directory, "tissue_blacklist", build)


def _check_indexes(release):
    from .mappings import mappings_cache_is_current, mappings_meta_path
    from .observations import observations_cache_is_current

    if observations_cache_is_current() is not True:
        raise ValueError(
            "A current, source-verifiable observations index is required; run hitlist build observations explicitly"
        )
    path = mappings_meta_path()
    if not path.exists():
        raise ValueError("Current complete human-reference mappings are required")
    metadata = json.loads(path.read_text())
    contract = metadata.get("contract", {})
    if contract.get("use_uniprot") is not False or not mappings_cache_is_current(
        release=release,
        use_uniprot=False,
        fetch_missing=contract.get("fetch_missing", True),
        flank=contract.get("flank", 15),
    ):
        raise ValueError(
            "Stale or incomplete Ensembl mappings cannot establish CTA specificity; rebuild mappings"
        )
    return metadata


def _selected_mapping_rows(expression):
    from oncoref.gene_ids import ensembl_id_aliases

    from .cta_expression import _canonical_mapping_gene
    from .mappings import load_peptide_mappings

    selected = expression[expression.selected]
    genes = set(selected.gene_id)
    aliases = {alias for alias, primary in ensembl_id_aliases().items() if primary in genes}
    mappings = load_peptide_mappings(gene_id=sorted(genes | aliases), proteome="Homo sapiens")
    # Do not use gene expression to select every isoform for transcript inputs.
    transcripts = set(selected.transcript_id) - {""}
    if transcripts:
        mappings = mappings[mappings.transcript_id.isin(transcripts)]
    canonical = mappings.gene_id.map(_canonical_mapping_gene)
    return mappings[canonical.isin(genes)]


def _positive_ms(frame):
    """Positive assay evidence, not the legacy nonbinding partition (#644)."""
    method = (
        frame.get("assay_method", pd.Series("", index=frame.index))
        .astype("string")
        .fillna("")
        .str.strip()
        .str.lower()
    )
    source = frame.get("source", pd.Series("", index=frame.index)).astype("string").fillna("")
    accepted = method.str.contains("mass spectrometry", regex=False) | (
        method.eq("") & source.eq("supplement")
    )
    if "is_binding_assay" in frame:
        accepted &= ~frame.is_binding_assay.astype("boolean").fillna(False)
    if "qualitative_measurement" in frame:
        accepted &= ~frame.qualitative_measurement.astype("string").fillna(
            ""
        ).str.strip().str.lower().str.startswith("negative")
    return accepted


def _expression_links(expression, mappings, level):
    selected = expression.loc[
        expression.selected, ["input_row", "gene_id", "transcript_id", "tpm"]
    ].rename(
        columns={
            "gene_id": "expression_gene_id",
            "transcript_id": "expression_transcript_id",
            "tpm": "expression_tpm",
        }
    )
    links = mappings[mappings.human_reference].merge(
        selected, left_on="canonical_gene_id", right_on="expression_gene_id", how="inner"
    )
    if level == "transcript":
        links = links[links.transcript_id.eq(links.expression_transcript_id)]
    return links


def _peptide_summary(mappings, cta_ids, evidence, tissue_risk, *, use_captured_identities=False):
    summary, annotated = summarize_cta_mappings(
        mappings, cta_ids, use_captured_identities=use_captured_identities
    )
    counts = evidence.groupby("peptide").size()
    summary["n_ms_observations"] = summary.peptide.map(counts).fillna(0).astype(int)
    risk = tissue_risk.set_index("peptide")
    summary["blacklisted"] = (
        summary.peptide.map(risk.blacklisted).astype("boolean").fillna(False).astype(bool)
    )
    summary["n_essential_tissue_donors"] = summary.peptide.map(risk.n_donors).fillna(0).astype(int)
    forbidden = set(tissue_risk.loc[tissue_risk.blacklisted, "peptide"])
    lengths = sorted({len(sequence) for sequence in forbidden})
    matches = [
        sorted(
            {
                peptide[start : start + length]
                for length in lengths
                if length <= len(peptide)
                for start in range(len(peptide) - length + 1)
            }
            & forbidden
        )
        for peptide in summary.peptide
    ]
    summary["blacklist_matches"] = [json.dumps(values) for values in matches]
    summary["contains_blacklisted_sequence"] = [bool(values) for values in matches]
    # This flags an observed shorter sequence inside a longer candidate. It
    # does not claim that a shorter, unobserved epitope was measured in MS.
    summary["avoid_sequence"] = (
        ~summary.cta_specific.astype(bool) | summary.contains_blacklisted_sequence
    )
    empty = pd.Series(False, index=evidence.index)
    benign = evidence.get("src_healthy_tissue", empty).astype("boolean").fillna(
        False
    ) | evidence.get("src_adjacent_to_tumor", empty).astype("boolean").fillna(False)
    primary = ~evidence.get("src_cell_line", empty).astype("boolean").fillna(False)
    tissue = evidence.get("source_tissue", pd.Series("", index=evidence.index))
    donor_status = evidence.get("donor_status", pd.Series("unknown", index=evidence.index))
    unresolved = evidence[
        benign & primary & _tissue_groups(tissue).ne("") & donor_status.ne("resolved")
    ]
    summary["n_unresolved_essential_tissue_observations"] = (
        summary.peptide.map(unresolved.groupby("peptide").size()).fillna(0).astype(int)
    )
    summary["tissue_review_required"] = summary.n_unresolved_essential_tissue_observations.gt(0)
    summary["avoid_reasons"] = [
        ";".join(
            reason
            for condition, reason in (
                (row.has_non_cta_match, "non_cta_reference_match"),
                (not row.cta_specific and not row.has_non_cta_match, "unresolved_reference"),
                (row.blacklisted, "essential_tissue_blacklist"),
                (
                    row.contains_blacklisted_sequence and not row.blacklisted,
                    "contains_blacklisted_sequence",
                ),
            )
            if condition
        )
        for row in summary.itertuples(index=False)
    ]
    return summary, annotated


def write_cta_evidence_bundle(
    directory, expression_path, *, atlas_dir, id_column, tpm_column, **expression_options
):
    """Export observed CTA peptide evidence, reference sharing and tissue risk.

    Input is one gene/transcript TPM column, not VCF/BAM. This exports evidence
    for indexed peptides; it neither predicts presentation nor assembles or
    ranks vaccines. CTA eligibility never filters the self-reference background
    or the standalone tissue blacklist. Complete raw provenance is required.
    """
    from .curation import _clear_curation_caches
    from .export import generate_training_table
    from .lineage import load_lineage
    from .mappings import load_peptide_mappings

    def build(staging):
        expression, reference, options = resolve_expression_table(
            expression_path, id_column=id_column, tpm_column=tpm_column, **expression_options
        )
        mapping_metadata = _check_indexes(options["ensembl_release"])
        before = _inputs()
        _clear_curation_caches()
        selected = _selected_mapping_rows(expression)
        peptides = sorted(set(selected.peptide))
        mappings = load_peptide_mappings(peptide=peptides)
        complete = generate_training_table(
            include_evidence="ms", species="Homo sapiens", peptide=peptides
        )
        if complete.provenance_id.isna().any() or complete.provenance_id.eq("").any():
            raise ValueError(
                "Full source contributors are required; legacy observations cannot form this bundle"
            )
        accepted = _positive_ms(complete)
        presentation = complete[accepted].copy()
        excluded = complete[~accepted].copy()
        excluded["exclusion_reason"] = "not_positive_ms"
        identities = _identity_table(complete)
        contributors = _stream_contributors(
            identities.provenance_id, staging / "contributors.parquet"
        )
        identities = _contributor_identities(identities, contributors)
        risk, atlas = _write_tissues(staging, atlas_dir)
        summary, mappings = _peptide_summary(
            mappings, set(options["cta_gene_ids"]), presentation, risk
        )
        links = _expression_links(expression, mappings, options["level"])
        tables = {
            "expression": expression,
            "cta_reference": reference,
            "peptides": summary,
            "mappings": mappings,
            "expression_mappings": links,
            "presentation": presentation,
            "excluded_observations": excluded,
            "identities": identities,
        }
        for name, frame in tables.items():
            _parquet(frame, staging / f"{name}.parquet")
        _write_json(staging / "lineage.json", load_lineage())
        _stable_source_files({"expression": options["input"]})
        if before != _inputs() or mapping_metadata != _check_indexes(options["ensembl_release"]):
            raise ValueError("Evidence index inputs changed during export")
        return {
            "atlas": atlas,
            "expression": options,
            "index_inputs": before,
            "mapping_reference": mapping_metadata,
            "coverage": _coverage(identities, contributors),
            "n_blacklisted_peptides": int(risk.blacklisted.sum()),
            "scope": "Indexed peptide sequences mapping to eligible expressed CTAs; full human multi-mapping; Atlas blacklist independent of expression and HLA",
            "ms_policy": "Positive mass-spectrometry method or curated MS-only supplement; explicit non-MS excluded",
            "interpretation": "CTA-specific means all resolved human mappings are CTAs, including sharing between CTAs. Missing observations do not establish safety. Assembly and treatment ranking belong downstream.",
        }

    def final_check(metadata):
        options = metadata["expression"]
        _stable_source_files({"expression": options["input"]})
        if "transcript_reference" in options:
            _stable_source_files({"transcripts": options["transcript_reference"]})
        if metadata["index_inputs"] != _inputs() or metadata["mapping_reference"] != _check_indexes(
            options["ensembl_release"]
        ):
            raise ValueError("Evidence index inputs changed during export")

    return _write_bundle(directory, "cta_expression", build, final_check)


def _equal_frame(actual, expected, name):
    try:
        pd.testing.assert_frame_equal(
            actual.reset_index(drop=True),
            expected.reset_index(drop=True),
            check_dtype=False,
            check_categorical=False,
        )
    except AssertionError as error:
        raise ValueError(f"Evidence bundle relationship mismatch: {name}") from error


def _verify_atlas_links(evidence, donors, source):
    """Check normalized claims against captured primary-table cells, offline."""
    typing = donors.groupby("donor").hla_allele.agg(lambda x: ";".join(sorted(set(x))))
    dataset = f"hla-ligand-atlas:{source['release']}"
    valid = (
        evidence.source_record_id.eq(
            dataset + ":sample_hits:" + evidence.sample_hits_row.astype(str)
        )
        & evidence.source_dataset.eq(dataset)
        & evidence.sample_hits_row.between(1, source["n_source_sample_hits"])
        & evidence.peptide_row.between(1, source["n_source_peptides"])
        & evidence.donor_id.eq("hla-ligand-atlas:" + evidence.donor)
        & evidence.source_tissue.eq(evidence.tissue)
        & evidence.peptide.eq(evidence["peptide_source:peptide_sequence"])
        & evidence.sample_alleles.eq(evidence.donor.map(typing))
        & evidence.mhc_class.eq(evidence.hla_class.map({"HLA-I": "I", "HLA-II": "II"}))
        & evidence.mhc_restriction.eq("")
        & evidence.donor_status.eq("resolved")
        & evidence.tissue_status.eq("nonmalignant")
        & evidence.assay_modality.eq("mass_spectrometry")
        & evidence.is_cell_line.eq(False)
        & evidence.pmid.eq(33858848)
    )
    references = evidence[["peptide_sequence_id", "peptide_row", "peptide"]].drop_duplicates()
    if (
        not valid.all()
        or evidence.sample_hits_row.duplicated().any()
        or references.peptide_sequence_id.duplicated().any()
        or references.peptide_row.duplicated().any()
        or len(typing) != source["n_source_donors"]
    ):
        raise ValueError("Evidence bundle Atlas source relationships mismatch")


def verify_evidence_bundle(directory):
    """Verify artifact hashes and recompute blacklist/mapping relationships."""
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    kind = manifest.get("kind")
    if manifest.get("schema_version") != 1 or kind not in {"tissue_blacklist", "cta_expression"}:
        raise ValueError("Unsupported evidence bundle schema")
    files = _TISSUE_FILES | (_CTA_FILES if kind == "cta_expression" else set())
    if set(manifest.get("artifacts", {})) != files:
        raise ValueError("Incomplete evidence bundle")
    for filename in files:
        if file_digest(directory / filename) != manifest["artifacts"][filename]:
            raise ValueError(f"Evidence bundle artifact mismatch: {filename}")

    def read(name):
        return pd.read_parquet(directory / f"{name}.parquet")

    if (
        manifest["policy"]["tissue_status"] != "nonmalignant"
        or manifest["policy"]["hla_filter"] is not None
    ):
        raise ValueError("Blacklist must use nonmalignant tissue without an HLA filter")
    tissue_evidence = read("tissue_evidence")
    _verify_atlas_links(tissue_evidence, read("atlas_donors"), manifest["atlas"])
    summary, audit = build_tissue_blacklist(
        tissue_evidence,
        min_donors=manifest["policy"]["min_donors"],
        policy=manifest["policy"],
    )
    _equal_frame(tissue_evidence, audit, "tissue eligibility")
    _equal_frame(read("tissue_risk"), summary, "tissue donor counts")
    blacklist = summary[summary.blacklisted].reset_index(drop=True)
    _equal_frame(read("blacklist"), blacklist, "blacklist")
    if (
        directory / "forbidden_sequences.txt"
    ).read_text().splitlines() != blacklist.peptide.tolist():
        raise ValueError("Evidence bundle forbidden sequence list mismatch")
    if len(blacklist) != manifest["n_blacklisted_peptides"]:
        raise ValueError("Evidence bundle blacklist count mismatch")
    if kind == "cta_expression":
        from .training_bundle import _CONTRIBUTOR_SUMMARY_COLUMNS

        reference = read("cta_reference")
        if set(reference.loc[reference.included_in_definition, "Ensembl_Gene_ID"]) != set(
            manifest["expression"]["cta_gene_ids"]
        ):
            raise ValueError("Evidence bundle CTA reference membership mismatch")
        identities = read("identities")
        contributors = pd.read_parquet(
            directory / "contributors.parquet", columns=_CONTRIBUTOR_SUMMARY_COLUMNS
        )
        presentation, excluded = read("presentation"), read("excluded_observations")
        nonempty = [frame for frame in (presentation, excluded) if len(frame)]
        observations = pd.concat(nonempty, ignore_index=True) if nonempty else presentation.copy()
        if identities.evidence_row_id.duplicated().any() or set(
            observations.evidence_row_id
        ) != set(identities.evidence_row_id):
            raise ValueError("Evidence bundle observation identities mismatch")
        expected_identities = _contributor_identities(_identity_table(observations), contributors)
        _equal_frame(
            identities.sort_values("evidence_row_id"),
            expected_identities.sort_values("evidence_row_id"),
            "observation identities and contributors",
        )
        if set(identities.provenance_id) != set(contributors.provenance_id):
            raise ValueError("Evidence bundle contributor coverage mismatch")
        if _coverage(identities, contributors) != manifest["coverage"]:
            raise ValueError("Evidence bundle provenance counts mismatch")
        if not _positive_ms(read("presentation")).all():
            raise ValueError("Non-MS evidence appears in presentation table")
        if _positive_ms(read("excluded_observations")).any():
            raise ValueError("Positive MS evidence appears in excluded observations")
        expected, mappings = _peptide_summary(
            read("mappings"),
            set(manifest["expression"]["cta_gene_ids"]),
            read("presentation"),
            summary,
            use_captured_identities=True,
        )
        _equal_frame(read("peptides"), expected, "peptide specificity and tissue risk")
        _equal_frame(read("mappings"), mappings, "mapping CTA annotations")
        links = _expression_links(read("expression"), mappings, manifest["expression"]["level"])
        _equal_frame(read("expression_mappings"), links, "expression mapping links")
        if set(mappings.peptide) != set(links.peptide) or not set(observations.peptide) <= set(
            mappings.peptide
        ):
            raise ValueError("Evidence bundle expression peptide coverage mismatch")
    return manifest
