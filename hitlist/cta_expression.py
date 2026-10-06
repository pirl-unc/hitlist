"""Resolve expression inputs and annotate complete CTA peptide multi-mapping."""

from __future__ import annotations

import csv
import gzip
import json
import math
from fnmatch import fnmatchcase
from functools import lru_cache
from importlib.metadata import version
from pathlib import Path

import pandas as pd

from .provenance import file_digest


def _cta_reference(definition):
    from oncoref import cta

    if definition == "strict":
        return cta.cta_df(), set(cta.cta_gene_ids())
    if definition == "extended":
        return cta.cta_extended_df(), set(cta.cta_extended_gene_ids())
    raise ValueError("CTA definition must be strict or extended")


def _gene_identity(identifier):
    from oncoref.gene_ids import canonical_gene_id, canonical_gene_symbol

    gene_id = canonical_gene_id(identifier)
    return (gene_id or "", canonical_gene_symbol(gene_id) or "") if gene_id else ("", "")


@lru_cache(maxsize=2)
def _reference_genome(release):
    from pyensembl import EnsemblRelease

    genome = EnsemblRelease(release)
    if not Path(genome.db.local_db_path).is_file():
        raise FileNotFoundError(f"Install/index Ensembl {release} before resolving transcripts")
    return genome


def _transcript_reference(release):
    database = Path(_reference_genome(release).db.local_db_path)
    return {"path": str(database), **file_digest(database)}


def _transcript_identity(identifier, release):
    genome = _reference_genome(release)
    try:
        transcript = genome.transcript_by_id(identifier)
    except (KeyError, ValueError):
        return "", ""
    return _gene_identity(transcript.gene_id)


def _canonical_mapping_gene(identifier):
    from oncoref.gene_ids import resolve_ensembl_id

    return resolve_ensembl_id(identifier) if identifier else ""


def resolve_expression_table(
    path,
    *,
    id_column,
    tpm_column,
    level="gene",
    definition="strict",
    min_tpm=2.0,
    ensembl_release=112,
    exclude_gene_patterns=("MAGE*",),
    allow_genes=("MAGEA4",),
):
    """Resolve one sample's explicit gene/transcript TPM column, without VCF/BAM.

    CSV and TSV (including gzip) are accepted. Original columns and logical
    input-row numbers survive in the audit. Missing TPM is unmeasured, never an
    inclusion request. Duplicate resolved genes/transcripts fail rather than
    silently sum alias copies; distinct transcripts keep their own TPMs.
    Exclusions affect eligibility only, not the canonical CTA background.
    """
    if level not in {"gene", "transcript"}:
        raise ValueError("Expression level must be gene or transcript")
    if not math.isfinite(min_tpm) or min_tpm < 0:
        raise ValueError("min_tpm must be finite and nonnegative")
    for values in (exclude_gene_patterns, allow_genes):
        if isinstance(values, str) or any(not isinstance(x, str) or not x.strip() for x in values):
            raise ValueError("Gene exclusions and exceptions must be sequences of nonempty strings")
    path = Path(path)
    before = file_digest(path)
    suffix = path.with_suffix("").suffix if path.suffix == ".gz" else path.suffix
    if suffix.lower() not in {".csv", ".tsv", ".tab", ".sf"}:
        raise ValueError(
            "Expression input must be CSV or tab-separated TSV/TAB/SF (optionally .gz)"
        )
    separator = "," if suffix.lower() == ".csv" else "\t"
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8-sig", newline="") as stream:
        header = next(csv.reader(stream, delimiter=separator), [])
    if not header or len(header) != len(set(header)) or any(not name.strip() for name in header):
        raise ValueError("Expression columns must have unique, nonempty names")
    frame = pd.read_csv(path, sep=separator, dtype=str, keep_default_na=False)
    if id_column == tpm_column or not {id_column, tpm_column} <= set(frame):
        raise ValueError(
            "Specify distinct identifier and TPM columns present in the expression table"
        )
    frame = frame.rename(columns={name: f"input:{name}" for name in frame.columns})
    frame.insert(0, "input_row", range(1, len(frame) + 1))
    frame["original_identifier"] = frame[f"input:{id_column}"]
    frame["original_tpm"] = frame[f"input:{tpm_column}"]
    transcript_reference = _transcript_reference(ensembl_release) if level == "transcript" else None
    raw_tpm = frame.original_tpm.str.strip()
    measured = ~raw_tpm.str.casefold().isin(["", "na", "nan", "n/a"])
    values = pd.to_numeric(raw_tpm.where(measured), errors="coerce")
    if (
        measured
        & (values.isna() | ~values.map(lambda x: pd.isna(x) or math.isfinite(x)) | values.lt(0))
    ).any():
        raise ValueError("Measured TPM values must be finite, numeric and nonnegative")
    frame["tpm"] = values
    identifiers = frame.original_identifier.str.strip()
    frame["transcript_id"] = identifiers.str.split(".").str[0] if level == "transcript" else ""
    identities = [
        _transcript_identity(name, ensembl_release)
        if level == "transcript"
        else _gene_identity(name)
        for name in (frame.transcript_id if level == "transcript" else identifiers)
    ]
    frame["gene_id"] = [identity[0] for identity in identities]
    frame["gene_name"] = [identity[1] for identity in identities]
    key = frame.transcript_id if level == "transcript" else frame.gene_id
    resolved = frame.gene_id.ne("")
    if key[resolved].duplicated().any():
        raise ValueError(
            "Duplicate resolved expression identifiers; disambiguate rather than sum aliases"
        )
    reference, cta_ids = _cta_reference(definition)
    reference = reference.copy()
    reference["included_in_definition"] = reference.Ensembl_Gene_ID.isin(cta_ids)
    exceptions = {name.strip().upper() for name in allow_genes}
    patterns = [name.strip().upper() for name in exclude_gene_patterns]
    excluded = frame.gene_name.map(
        lambda name: (
            name.upper() not in exceptions
            and any(fnmatchcase(name.upper(), pattern) for pattern in patterns)
        )
    )
    frame["selection_reason"] = "selected"
    for valid, reason in (
        (resolved, "unresolved_identifier"),
        (frame.gene_id.isin(cta_ids), "not_cta"),
        (measured, "missing_tpm"),
        (frame.tpm.ge(min_tpm), "below_tpm"),
        (~excluded, "excluded_gene_pattern"),
    ):
        frame.loc[frame.selection_reason.eq("selected") & ~valid, "selection_reason"] = reason
    frame["selected"] = frame.selection_reason.eq("selected")
    if file_digest(path) != before:
        raise ValueError("Expression input changed during resolution")
    metadata = {
        "input": {"path": str(path.resolve()), **before},
        "level": level,
        "id_column": id_column,
        "tpm_column": tpm_column,
        "units": "TPM",
        "min_tpm": min_tpm,
        "definition": definition,
        "oncoref_version": version("oncoref"),
        "ensembl_release": ensembl_release,
        "exclude_gene_patterns": list(exclude_gene_patterns),
        "allow_genes": list(allow_genes),
        "cta_gene_ids": sorted(cta_ids),
        "isoform_policy": "Transcript TPM applies only to the named transcript; gene TPM does not establish an expressed isoform",
    }
    if transcript_reference is not None:
        if _transcript_reference(ensembl_release) != transcript_reference:
            raise ValueError("Transcript reference changed during expression resolution")
        metadata["transcript_reference"] = transcript_reference
    return frame, reference, metadata


def summarize_cta_mappings(mappings, cta_gene_ids, *, use_captured_identities=False):
    """Annotate every mapping and exact peptide's human-reference specificity.

    Pass the COMPLETE mapping slice for each peptide. Callers must verify the
    mapping artifact's reference contract before using these statements.
    Multiple CTA genes retain source ambiguity while remaining CTA-specific;
    independent non-CTA proteins make a peptide unsuitable for CTA exclusivity.
    """
    required = {"peptide", "gene_id", "protein_id", "transcript_id", "proteome"}
    if not required <= set(mappings):
        raise ValueError(f"Incomplete mapping schema: {sorted(required - set(mappings))}")
    result = mappings.copy()
    genes = result.gene_id.astype("string").fillna("")
    if use_captured_identities:
        if "canonical_gene_id" not in result:
            raise ValueError("Captured canonical gene identities are missing")
        if result.groupby("gene_id", dropna=False).canonical_gene_id.nunique().gt(1).any():
            raise ValueError("Conflicting captured canonical gene identities")
    else:
        canonical = {gene: _canonical_mapping_gene(gene) for gene in genes.unique()}
        result["canonical_gene_id"] = genes.map(canonical).astype("string")
    result["human_reference"] = result.proteome.eq("Homo sapiens")
    result["is_cta"] = result.human_reference & result.canonical_gene_id.isin(cta_gene_ids)
    rows = []
    columns = [
        "peptide",
        "cta_specific",
        "shared_between_ctas",
        "shared_with_other_proteins",
        "has_non_cta_match",
        "n_cta_genes",
        "n_non_cta_genes",
        "n_unresolved_human_mappings",
        "n_human_proteins",
        "human_gene_ids",
    ]
    for peptide, frame in result.groupby("peptide", sort=True):
        human = frame[frame.human_reference]
        known = human[human.canonical_gene_id.ne("")]
        ctas = set(known.loc[known.is_cta, "canonical_gene_id"])
        others = set(known.loc[~known.is_cta, "canonical_gene_id"])
        unresolved = int(
            (
                human.canonical_gene_id.astype("string").fillna("").eq("")
                | human.protein_id.astype("string").fillna("").eq("")
                | human.transcript_id.astype("string").fillna("").eq("")
            ).sum()
        )
        n_proteins = human.loc[
            human.protein_id.astype("string").fillna("").ne(""), "protein_id"
        ].nunique()
        rows.append(
            {
                "peptide": peptide,
                "cta_specific": bool(ctas) and not others and not unresolved,
                "shared_between_ctas": len(ctas) > 1,
                "shared_with_other_proteins": n_proteins > 1,
                "has_non_cta_match": bool(others),
                "n_cta_genes": len(ctas),
                "n_non_cta_genes": len(others),
                "n_unresolved_human_mappings": unresolved,
                "n_human_proteins": n_proteins,
                "human_gene_ids": json.dumps(sorted(ctas | others)),
            }
        )
    summary = pd.DataFrame(rows, columns=columns)
    for column in columns:
        dtype = (
            "int64"
            if column.startswith("n_")
            else "string"
            if column in {"peptide", "human_gene_ids"}
            else "bool"
        )
        summary[column] = summary[column].astype(dtype)
    return summary, result
