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

from .expression_inputs import expression_input_path, infer_expression_columns
from .provenance import file_digest

CTA_POLICY_SELECTORS = ("legacy", "target_sets", "cta_selection", "target_selection")

# Exact-protein/gene precedence when multiple records resolve to one gene ID (#852):
# Unsafe > Other / Neither / Unassessed / partial-set > CTA.
_CATEGORY_PRECEDENCE = {
    "Unsafe": 0,
    "Unassessed": 1,
    "Unassigned": 1,
    "Neither": 2,
    "Other": 2,
    "Cancer Antigen only": 3,
    "Reproductive Antigen only": 3,
    "CTA": 4,
}


def _cta_reference(definition):
    from oncoref import cta

    if definition == "strict":
        return cta.cta_df(), set(cta.cta_gene_ids())
    if definition == "extended":
        return cta.cta_extended_df(), set(cta.cta_extended_gene_ids())
    raise ValueError("CTA definition must be strict or extended")


def _serialize_parquet_cell(value):
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value)
    return value


def _serialize_parquet_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for column in out.columns:
        if (
            out[column].dtype == object
            and out[column].map(lambda value: isinstance(value, (list, tuple, dict))).any()
        ):
            out[column] = out[column].map(_serialize_parquet_cell).astype("string")
    return out


def _parse_gene_id_list(value) -> list[str]:
    if isinstance(value, (list, tuple)):
        items = value
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text.startswith("["):
            items = json.loads(text)
        else:
            items = [text]
    else:
        return []
    return [str(item).strip() for item in items if str(item).strip()]


def _deduplicate_policy_genes(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.reset_index(drop=True)
    ranked = frame.copy()
    ranked["_precedence"] = ranked["policy_category"].map(
        lambda value: _CATEGORY_PRECEDENCE.get(str(value), 2)
    )
    ranked = ranked.sort_values(
        ["_precedence", "Ensembl_Gene_ID"], kind="mergesort"
    ).drop_duplicates(subset=["Ensembl_Gene_ID"], keep="first")
    return (
        ranked.drop(columns=["_precedence"])
        .sort_values("Ensembl_Gene_ID", kind="mergesort")
        .reset_index(drop=True)
    )


def _load_target_sets_policy():
    try:
        from oncoref import cta_policy, target_sets
    except ImportError as error:
        raise ImportError(
            "cta_policy='target_sets' requires oncoref>=1.20.0 (oncoref.target_sets and "
            "oncoref.cta_policy); upgrade oncoref or use cta_policy='legacy'"
        ) from error

    frame = target_sets.results().copy()
    manifest = cta_policy.manifest() if hasattr(cta_policy, "manifest") else {}
    policy_id = str(
        manifest.get("policy_id")
        or getattr(cta_policy, "POLICY_ID", "")
        or getattr(target_sets, "POLICY_ID", "oncoref.target-sets")
    )
    reference_id = str(manifest.get("reference") or manifest.get("reference_id") or "")
    frame["Ensembl_Gene_ID"] = frame["Gene ID"].astype("string").fillna("").str.strip()
    frame = frame[frame["Ensembl_Gene_ID"].ne("")].copy()
    frame["Symbol"] = (
        frame.get("Gene", pd.Series("", index=frame.index)).astype("string").fillna("")
    )
    frame["policy_category"] = frame["Category"].astype("string").fillna("Neither")
    detail = frame.get("Detail", pd.Series("", index=frame.index)).astype("string").fillna("")
    frame["policy_classification"] = frame["policy_category"]
    has_detail = detail.ne("") & frame["policy_category"].eq("Neither")
    frame.loc[has_detail, "policy_classification"] = (
        frame.loc[has_detail, "policy_category"] + ":" + detail.loc[has_detail]
    )
    frame["policy_id"] = policy_id
    frame["reference_id"] = reference_id
    frame = _deduplicate_policy_genes(_serialize_parquet_frame(frame))
    cta_ids = set(frame.loc[frame["policy_category"].eq("CTA"), "Ensembl_Gene_ID"])
    unsafe_ids = set(frame.loc[frame["policy_category"].eq("Unsafe"), "Ensembl_Gene_ID"])
    counts = frame["policy_category"].value_counts().to_dict()
    receipt = {
        "selector": "target_sets",
        "provider": "oncoref.target_sets",
        "policy_id": policy_id,
        "reference_id": reference_id,
        "source_snapshot_id": str(manifest.get("source_snapshot_id", "")),
        "oncoref_version": version("oncoref"),
        "counts": {str(k): int(v) for k, v in counts.items()},
        "n_cta_genes": len(cta_ids),
        "n_unsafe_genes": len(unsafe_ids),
        "unsafe_gene_ids": sorted(unsafe_ids),
        "unsafe_only_forbidden": True,
        "other_is_not_forbidden": True,
        "observed_peptide_blacklist_independent": True,
        "implicit_upgrade": False,
        "owner_manifest": {
            k: v
            for k, v in manifest.items()
            if k in {"schema", "policy_id", "base_policy_id", "reference", "reference_id", "counts"}
        },
    }
    return frame, cta_ids, receipt


def _load_cta_selection_policy():
    try:
        from oncoref import cta_selection
    except ImportError as error:
        raise ImportError(
            "cta_policy='cta_selection' requires oncoref>=1.22.0 (oncoref.cta_selection); "
            "upgrade oncoref or use cta_policy='legacy'"
        ) from error

    base = (
        cta_selection.profiles(species="human").copy()
        if hasattr(cta_selection, "profiles")
        else cta_selection.results(species="human").copy()
    )
    if hasattr(cta_selection, "sequence_dispositions"):
        dispositions = cta_selection.sequence_dispositions(species="human")
        extra_cols = [
            col for col in dispositions.columns if col == "entity_id" or col not in base.columns
        ]
        if "entity_id" in extra_cols and len(extra_cols) > 1:
            base = base.merge(dispositions[extra_cols], on="entity_id", how="left")

    contract = cta_selection.contract(species="human") if hasattr(cta_selection, "contract") else {}
    coverage = cta_selection.coverage(species="human") if hasattr(cta_selection, "coverage") else {}
    non_null_policies = sorted(
        {str(v) for v in base.get("policy_id", pd.Series(dtype=str)).dropna() if str(v).strip()}
    )
    policy_id = str(
        contract.get("selection_policy_id")
        or coverage.get("policy_id")
        or (non_null_policies[0] if non_null_policies else "oncoref.cta-selection")
    )
    non_null_refs = sorted(
        {str(v) for v in base.get("reference_id", pd.Series(dtype=str)).dropna() if str(v).strip()}
    )
    reference_id = str(
        contract.get("reference_id")
        or (coverage.get("reference", {}) or {}).get("reference_id", "")
        or (non_null_refs[0] if non_null_refs else "")
    )
    snapshots = sorted(
        {
            str(v)
            for v in base.get("source_snapshot_id", pd.Series(dtype=str)).dropna()
            if str(v).strip()
        }
    )

    exploded_rows = []
    for row in base.to_dict(orient="records"):
        gene_ids = _parse_gene_id_list(row.get("gene_ids"))
        raw_cat = row.get("category")
        policy_category = (
            "Unassessed" if pd.isna(raw_cat) or not str(raw_cat).strip() else str(raw_cat)
        )
        raw_cls = row.get("classification")
        policy_classification = (
            policy_category if pd.isna(raw_cls) or not str(raw_cls).strip() else str(raw_cls)
        )
        for gene_id in gene_ids:
            record = dict(row)
            record["Ensembl_Gene_ID"] = gene_id
            record["Symbol"] = str(row.get("label") or "")
            record["policy_category"] = policy_category
            record["policy_classification"] = policy_classification
            record["policy_id"] = policy_id
            record["reference_id"] = reference_id
            exploded_rows.append(record)

    frame = _deduplicate_policy_genes(_serialize_parquet_frame(pd.DataFrame(exploded_rows)))
    cta_ids = set(frame.loc[frame["policy_category"].eq("CTA"), "Ensembl_Gene_ID"])
    unsafe_ids = set(frame.loc[frame["policy_category"].eq("Unsafe"), "Ensembl_Gene_ID"])
    counts = frame["policy_category"].value_counts().to_dict()
    receipt = {
        "selector": "cta_selection",
        "provider": "oncoref.cta_selection",
        "policy_id": policy_id,
        "reference_id": reference_id,
        "source_snapshot_id": ";".join(snapshots),
        "oncoref_version": version("oncoref"),
        "counts": {str(k): int(v) for k, v in counts.items()},
        "entity_counts": {str(k): int(v) for k, v in (coverage.get("counts") or {}).items()},
        "n_cta_genes": len(cta_ids),
        "n_unsafe_genes": len(unsafe_ids),
        "unsafe_gene_ids": sorted(unsafe_ids),
        "unsafe_only_forbidden": True,
        "other_is_not_forbidden": True,
        "observed_peptide_blacklist_independent": True,
        "implicit_upgrade": False,
        "has_independent_profiles": bool(hasattr(cta_selection, "profiles")),
    }
    return frame, cta_ids, receipt


def _load_target_selection_policy():
    try:
        from oncoref import target_selection
    except ImportError as error:
        raise ImportError(
            "cta_policy='target_selection' requires oncoref>=1.37.0 (oncoref.target_selection); "
            "upgrade oncoref or use cta_policy='cta_selection' / 'legacy'"
        ) from error

    base = target_selection.results().copy()
    defn = target_selection.definition() if hasattr(target_selection, "definition") else {}
    meta = target_selection.metadata() if hasattr(target_selection, "metadata") else {}
    policy_id = str(
        defn.get("policy_id")
        or meta.get("policy_id")
        or getattr(target_selection, "POLICY_ID", "oncoref.target-selection")
    )
    reference_id = str(meta.get("reference_id") or meta.get("reference") or "")

    exploded_rows = []
    for row in base.to_dict(orient="records"):
        gene_ids = _parse_gene_id_list(row.get("member_gene_ids") or row.get("gene_ids"))
        raw_cat = row.get("category")
        policy_category = (
            "Unassessed" if pd.isna(raw_cat) or not str(raw_cat).strip() else str(raw_cat)
        )
        raw_sub = row.get("subtype")
        subtype = "" if pd.isna(raw_sub) else str(raw_sub).strip()
        policy_classification = (
            f"{policy_category}:{subtype}"
            if policy_category == "Other" and subtype
            else policy_category
        )
        for gene_id in gene_ids:
            record = dict(row)
            record["Ensembl_Gene_ID"] = gene_id
            record["Symbol"] = str(row.get("protein_group_label") or row.get("label") or "")
            record["policy_category"] = policy_category
            record["policy_classification"] = policy_classification
            record["policy_id"] = policy_id
            record["reference_id"] = reference_id
            exploded_rows.append(record)

    frame = _deduplicate_policy_genes(_serialize_parquet_frame(pd.DataFrame(exploded_rows)))
    cta_ids = set(frame.loc[frame["policy_category"].eq("CTA"), "Ensembl_Gene_ID"])
    unsafe_ids = set(frame.loc[frame["policy_category"].eq("Unsafe"), "Ensembl_Gene_ID"])
    counts = frame["policy_category"].value_counts().to_dict()
    receipt = {
        "selector": "target_selection",
        "provider": "oncoref.target_selection",
        "policy_id": policy_id,
        "reference_id": reference_id,
        "source_snapshot_id": str(meta.get("source_snapshot_id", "")),
        "oncoref_version": version("oncoref"),
        "counts": {str(k): int(v) for k, v in counts.items()},
        "n_cta_genes": len(cta_ids),
        "n_unsafe_genes": len(unsafe_ids),
        "unsafe_gene_ids": sorted(unsafe_ids),
        "unsafe_only_forbidden": True,
        "other_is_not_forbidden": True,
        "observed_peptide_blacklist_independent": True,
        "implicit_upgrade": False,
    }
    return frame, cta_ids, receipt


def _resolve_cta_policy(cta_policy: str, definition: str):
    if cta_policy not in CTA_POLICY_SELECTORS:
        raise ValueError(
            f"Unknown cta_policy {cta_policy!r}; expected one of {CTA_POLICY_SELECTORS}"
        )
    if cta_policy == "legacy":
        reference, cta_ids = _cta_reference(definition)
        receipt = {
            "selector": "legacy",
            "definition": definition,
            "provider": "oncoref.cta",
            "policy_id": f"oncoref.cta.legacy.{definition}",
            "reference_id": "",
            "source_snapshot_id": "",
            "oncoref_version": version("oncoref"),
            "n_cta_genes": len(cta_ids),
            "n_unsafe_genes": None,
            "unsafe_gene_ids": None,
            "unsafe_only_forbidden": False,
            "other_is_not_forbidden": True,
            "observed_peptide_blacklist_independent": True,
            "implicit_upgrade": False,
        }
        return reference, cta_ids, receipt
    if definition != "strict":
        raise ValueError(f"CTA definition={definition!r} only applies to cta_policy='legacy'")
    if cta_policy == "target_sets":
        return _load_target_sets_policy()
    if cta_policy == "cta_selection":
        return _load_cta_selection_policy()
    return _load_target_selection_policy()


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
    path=None,
    *,
    id_column=None,
    tpm_column=None,
    level="auto",
    definition="strict",
    cta_policy="legacy",
    min_tpm=2.0,
    ensembl_release=112,
    exclude_gene_patterns=("MAGE*",),
    allow_genes=("MAGEA4",),
):
    """Resolve one sample's gene/transcript TPM column, without VCF/BAM.

    CSV and TSV (including gzip) are accepted. Original columns and logical
    input-row numbers survive in the audit. Missing TPM is unmeasured, never an
    inclusion request. Duplicate resolved genes/transcripts fail rather than
    silently sum alias copies; distinct transcripts keep their own TPMs.
    Conventional files, identifier headers and TPM columns are inferred when
    unambiguous; explicit arguments override inference. Exclusions affect
    eligibility only, not the canonical CTA background.

    ``cta_policy`` defaults to ``"legacy"`` (the historical 624-gene
    ``oncoref.cta`` provider) so upgrading ``oncoref`` never implicitly switches
    a consumer's CTA target or exclusion policy (#852). Pass ``"cta_selection"``,
    ``"target_sets"``, or ``"target_selection"`` to adopt an explicit versioned
    OncoRef owner policy and record its policy/reference receipt in the bundle.
    """
    if cta_policy not in CTA_POLICY_SELECTORS:
        raise ValueError(
            f"Unknown cta_policy {cta_policy!r}; expected one of {CTA_POLICY_SELECTORS}"
        )
    if cta_policy != "legacy" and definition != "strict":
        raise ValueError(f"CTA definition={definition!r} only applies to cta_policy='legacy'")
    if not math.isfinite(min_tpm) or min_tpm < 0:
        raise ValueError("min_tpm must be finite and nonnegative")
    for values in (exclude_gene_patterns, allow_genes):
        if isinstance(values, str) or any(not isinstance(x, str) or not x.strip() for x in values):
            raise ValueError("Gene exclusions and exceptions must be sequences of nonempty strings")
    path, inferred_path = expression_input_path(path)
    before = file_digest(path)
    suffix = path.with_suffix("").suffix if path.suffix == ".gz" else path.suffix
    if suffix.lower() not in {".csv", ".tsv", ".tab", ".sf", ".results"}:
        raise ValueError(
            "Expression input must be CSV or tab-separated TSV/TAB/SF/RESULTS (optionally .gz)"
        )
    separator = "," if suffix.lower() == ".csv" else "\t"
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8-sig", newline="") as stream:
        header = next(csv.reader(stream, delimiter=separator), [])
    if not header or len(header) != len(set(header)) or any(not name.strip() for name in header):
        raise ValueError("Expression columns must have unique, nonempty names")
    frame = pd.read_csv(path, sep=separator, dtype=str, keep_default_na=False)
    id_column, tpm_column, level, inferred = infer_expression_columns(
        frame, id_column=id_column, tpm_column=tpm_column, level=level
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
    reference, cta_ids, policy_receipt = _resolve_cta_policy(cta_policy, definition)
    reference = reference.copy()
    reference["included_in_definition"] = reference.Ensembl_Gene_ID.isin(cta_ids)
    if cta_policy != "legacy":
        cat_map = dict(zip(reference.Ensembl_Gene_ID, reference.policy_category))
        frame["policy_category"] = frame.gene_id.map(cat_map).fillna("")
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
        "inferred_inputs": {"path": inferred_path, **inferred},
        "level": level,
        "id_column": id_column,
        "tpm_column": tpm_column,
        "units": "TPM",
        "min_tpm": min_tpm,
        "definition": definition,
        "cta_policy": policy_receipt,
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
