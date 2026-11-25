# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Access-aware TCRvdb paired-receptor functional evidence adapter (#722).

TCRvdb (Messemaker et al., Immunity 2026, DOI 10.1016/j.immuni.2026.09.007;
PMC12132471) is access-controlled by request (https://tcrvdb.nkiai.nl) and
prohibits redistribution without author permission. This module ships ONLY the
schema contract, local-file adapter, leakage/near-duplicate auditor, and
reproducible local-bundle exporter — zero restricted TCRvdb records are bundled
in the package, CI, or public releases.

Scientific invariants enforced by this adapter:
- Paired alpha+beta chains must belong to the same receptor row; unrelated
  chains sharing an epitope are never paired.
- Supplied full-length receptor chains are kept strictly separate from
  V/J+CDR3-only annotations and in-silico reconstructions; reconstruction is
  never labeled ``experimentally_observed_full_length``.
- Five validation/QC states (``validated``, ``tested_not_validating``,
  ``indeterminate``, ``failed_qc``, ``unassayed``) are preserved explicitly;
  non-validation is assay-scoped and never universal molecular non-binding.
- No shuffled pair or unobserved database entry is emitted as a measured
  negative; VDJdb confidence scores are retained for leakage auditing only and
  are never treated as functional ground truth.
- Kept strictly separate from MHC binding affinity and positive-only MS elution.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from collections import Counter
from importlib.resources import files
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .provenance import file_digest

TCRVDB_CONTRACT_REVISION = "tcrvdb_paired_receptor_adapter_v1"
TCRVDB_CONTRACT_SOURCE = files("hitlist").joinpath(
    "data/evidence_revisions/tcrvdb_adapter_contract.json"
)
TCRVDB_VALIDATION_STATUSES: frozenset[str] = frozenset(
    {
        "validated",
        "tested_not_validating",
        "indeterminate",
        "failed_qc",
        "unassayed",
    }
)
TCRVDB_SEQUENCE_PROVENANCE_STATUSES: frozenset[str] = frozenset(
    {
        "experimentally_observed_full_length",
        "reconstructed_from_vj_cdr3",
        "cdr3_vj_only",
        "missing",
    }
)

TCRVDB_ADAPTER_COLUMNS: tuple[str, ...] = (
    "source_row_index",
    "source_row_id",
    "source_file_sha256",
    "source_release_version",
    "receptor_id",
    "receptor_pair_id",
    "assay_record_id",
    "pairing_status",
    "peptide",
    "mhc_restriction",
    "mhc_class",
    "trav",
    "traj",
    "cdr3_alpha",
    "trbv",
    "trbj",
    "cdr3_beta",
    "trav_allele_collapsed",
    "trbv_allele_collapsed",
    "alpha_full_aa",
    "beta_full_aa",
    "alpha_sequence_provenance",
    "beta_sequence_provenance",
    "reconstruction_method",
    "assay_type",
    "assay_context",
    "validation_status",
    "raw_validation_label",
    "non_validation_scope",
    "qc_status",
    "expression_status",
    "author_validation_policy",
    "n_replicates",
    "base_mean",
    "log2_fold_change",
    "lfc_se",
    "stat",
    "pvalue",
    "padj",
    "cd69_positive_pct",
    "vdjdb_reference_id",
    "vdjdb_score",
    "source_pmid",
    "source_study",
    "evidence_modality",
    "near_duplicate_group_id",
)

NEAR_DUPLICATE_GROUP_COLUMNS: tuple[str, ...] = (
    "near_duplicate_group_id",
    "peptide",
    "mhc_restriction",
    "n_receptors",
    "n_rows",
    "n_validated",
    "n_tested_not_validating",
    "n_indeterminate",
    "n_failed_qc",
    "n_unassayed",
    "has_conflicting_validation",
    "n_distinct_studies",
    "source_studies",
    "representative_cdr3_alpha",
    "representative_cdr3_beta",
)

_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "source_row_id": ("source_row_id", "row_id", "entry_id", "id"),
    "receptor_id": ("receptor_id", "tcr_id", "tcr_name", "name", "TCR"),
    "peptide": ("peptide", "epitope", "antigen.epitope", "epitope_seq", "Epitope"),
    "mhc_restriction": ("mhc_restriction", "mhc.a", "mhc_allele", "mhc", "HLA", "MHC"),
    "mhc_class": ("mhc_class", "mhc.class"),
    "trav": ("trav", "TRAV", "TRAV_IMGT", "v.alpha", "v_alpha"),
    "traj": ("traj", "TRAJ", "TRAJ_IMGT", "j.alpha", "j_alpha"),
    "cdr3_alpha": ("cdr3_alpha", "CDR3a", "cdr3a", "cdr3.alpha", "CDR3_alpha", "CDR3A"),
    "trbv": ("trbv", "TRBV", "TRBV_IMGT", "v.beta", "v_beta"),
    "trbj": ("trbj", "TRBJ", "TRBJ_IMGT", "j.beta", "j_beta"),
    "cdr3_beta": ("cdr3_beta", "CDR3b", "cdr3b", "cdr3.beta", "CDR3_beta", "CDR3B"),
    "trav_allele_collapsed": ("trav_allele_collapsed", "TRAV_IMGT_allele_collapsed"),
    "trbv_allele_collapsed": ("trbv_allele_collapsed", "TRBV_IMGT_allele_collapsed"),
    "alpha_full_aa": ("alpha_full_aa", "full_alpha_aa", "tcr_alpha_aa", "alpha_chain_aa"),
    "beta_full_aa": ("beta_full_aa", "full_beta_aa", "tcr_beta_aa", "beta_chain_aa"),
    "alpha_sequence_provenance": ("alpha_sequence_provenance", "alpha_seq_source"),
    "beta_sequence_provenance": ("beta_sequence_provenance", "beta_seq_source"),
    "reconstruction_method": ("reconstruction_method", "sequence_reconstruction_method"),
    "assay_type": ("assay_type", "screen_type", "experiment_type"),
    "assay_context": ("assay_context", "effector_context", "target_context"),
    "raw_validation_label": (
        "validation_status",
        "validation_call",
        "functional_status",
        "screen_result",
        "status",
        "validating",
    ),
    "qc_status": ("qc_status", "library_qc", "qc_pass"),
    "expression_status": ("expression_status", "tcr_expression", "surface_expression"),
    "author_validation_policy": ("author_validation_policy", "validation_policy", "calling_policy"),
    "n_replicates": ("n_replicates", "replicate_count", "replicates"),
    "base_mean": ("base_mean", "baseMean"),
    "log2_fold_change": ("log2_fold_change", "log2FoldChange", "lfc"),
    "lfc_se": ("lfc_se", "lfcSE"),
    "stat": ("stat", "wald_stat"),
    "pvalue": ("pvalue", "p_value"),
    "padj": ("padj", "p_adj", "fdr"),
    "cd69_positive_pct": ("cd69_positive_pct", "cd69_pct", "pct_cd69_pos"),
    "vdjdb_reference_id": ("vdjdb_reference_id", "vdjdb_id", "reference.id"),
    "vdjdb_score": ("vdjdb_score", "vdjdb.score"),
    "source_pmid": ("source_pmid", "pmid", "PMID"),
    "source_study": ("source_study", "study", "reference_study"),
}

_VALIDATION_LABEL_MAP: dict[str, str] = {
    "validated": "validated",
    "validating": "validated",
    "true": "validated",
    "positive": "validated",
    "reactive": "validated",
    "hit": "validated",
    "tested_not_validating": "tested_not_validating",
    "non-validating": "tested_not_validating",
    "non_validating": "tested_not_validating",
    "not_validating": "tested_not_validating",
    "non-reactive": "tested_not_validating",
    "non_reactive": "tested_not_validating",
    "false": "tested_not_validating",
    "negative": "tested_not_validating",
    "indeterminate": "indeterminate",
    "ambiguous": "indeterminate",
    "inconclusive": "indeterminate",
    "borderline": "indeterminate",
    "failed_qc": "failed_qc",
    "qc_fail": "failed_qc",
    "failed": "failed_qc",
    "low_expression": "failed_qc",
    "dropped_qc": "failed_qc",
    "unassayed": "unassayed",
    "not_tested": "unassayed",
    "not_assayed": "unassayed",
    "untested": "unassayed",
    "": "unassayed",
}


def tcrvdb_source_contract(path=TCRVDB_CONTRACT_SOURCE) -> dict[str, Any]:
    """Load and validate the access-aware TCRvdb paired-receptor adapter contract (#722)."""
    contract = json.loads(Path(path).read_text(encoding="utf-8"))
    if contract.get("schema_version") != 1 or contract.get("revision") != TCRVDB_CONTRACT_REVISION:
        raise ValueError("Unsupported TCRvdb adapter contract revision")
    terms = contract.get("access_terms") or {}
    if (
        terms.get("redistribution_allowed") is not False
        or terms.get("public_package_embedding_prohibited") is not True
        or terms.get("vdjdb_score_is_ground_truth") is not False
    ):
        raise ValueError("TCRvdb access terms invariant violated")
    policies = contract.get("policies") or {}
    if (
        policies.get("require_paired_alpha_beta_same_receptor") is not True
        or policies.get("allow_pairing_unrelated_chains_by_epitope") is not False
        or policies.get("label_reconstructed_sequences_as_observed_full_length") is not False
        or policies.get("treat_non_validation_as_universal_nonbinding") is not False
        or policies.get("treat_shuffled_or_unobserved_pairs_as_measured_negatives") is not False
    ):
        raise ValueError("TCRvdb scientific policy invariant violated")
    return contract


def _resolve_col(df: pd.DataFrame, field: str, default: str = "") -> pd.Series:
    for col in _COLUMN_ALIASES.get(field, (field,)):
        if col in df.columns:
            return df[col].astype("string").fillna(default).str.strip()
    return pd.Series(default, index=df.index, dtype="string")


def _resolve_numeric(df: pd.DataFrame, field: str) -> pd.Series:
    for col in _COLUMN_ALIASES.get(field, (field,)):
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce").astype("float64")
    return pd.Series(np.nan, index=df.index, dtype="float64")


def _digest(prefix: str, parts: tuple[str, ...]) -> str:
    return f"{prefix}:{hashlib.sha256('|'.join(parts).encode('utf-8')).hexdigest()[:16]}"


def _normalize_validation_status(raw_label: str, qc_status: str) -> str:
    qc_norm = (qc_status or "").strip().casefold()
    if qc_norm in {"fail", "failed", "failed_qc", "qc_fail", "low_expression", "dropout"}:
        return "failed_qc"
    norm = (raw_label or "").strip().casefold()
    if norm in _VALIDATION_LABEL_MAP:
        return _VALIDATION_LABEL_MAP[norm]
    return "indeterminate"


def _resolve_chain_provenance(
    explicit: str, full_seq: str, cdr3: str, v_gene: str, j_gene: str, recon_method: str
) -> str:
    exp_norm = (explicit or "").strip().casefold()
    if exp_norm in TCRVDB_SEQUENCE_PROVENANCE_STATUSES:
        # Never allow a non-empty reconstruction method to claim experimentally observed full length.
        if exp_norm == "experimentally_observed_full_length" and (recon_method or "").strip():
            return "reconstructed_from_vj_cdr3"
        if (
            exp_norm in {"experimentally_observed_full_length", "reconstructed_from_vj_cdr3"}
            and not (full_seq or "").strip()
        ):
            return "cdr3_vj_only" if (cdr3 or v_gene or j_gene) else "missing"
        return exp_norm
    if (full_seq or "").strip():
        return (
            "reconstructed_from_vj_cdr3"
            if (recon_method or "").strip()
            else "experimentally_observed_full_length"
        )
    if (cdr3 or "").strip() or (v_gene or "").strip() or (j_gene or "").strip():
        return "cdr3_vj_only"
    return "missing"


def _hamming_le_1(a: str, b: str) -> bool:
    if len(a) != len(b):
        return False
    diffs = 0
    for ca, cb in zip(a, b, strict=True):
        if ca != cb:
            diffs += 1
            if diffs > 1:
                return False
    return True


def _uf_find(parent: dict[int, int], x: int) -> int:
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def _uf_union(parent: dict[int, int], x: int, y: int) -> None:
    rx, ry = _uf_find(parent, x), _uf_find(parent, y)
    if rx != ry:
        if rx < ry:
            parent[ry] = rx
        else:
            parent[rx] = ry


def _assign_near_duplicate_groups(df: pd.DataFrame) -> pd.Series:
    """Cluster receptors sharing (peptide, mhc_restriction) with exact or <=1-aa CDR3a/b match."""
    if len(df) == 0:
        return pd.Series(dtype="string")

    group_ids = pd.Series("", index=df.index, dtype="string")
    for (pep, mhc), grp in df.groupby(["peptide", "mhc_restriction"], sort=False):
        unique_pairs = sorted(
            {(str(a), str(b)) for a, b in zip(grp["cdr3_alpha"], grp["cdr3_beta"], strict=True)}
        )
        parent = {i: i for i in range(len(unique_pairs))}

        # Bucket by (len_a, len_b) to keep pairwise comparisons tiny and exact.
        buckets: dict[tuple[int, int], list[int]] = {}
        for idx, (ca, cb) in enumerate(unique_pairs):
            if not ca or not cb:
                continue
            buckets.setdefault((len(ca), len(cb)), []).append(idx)

        for indices in buckets.values():
            n_b = len(indices)
            for i_pos in range(n_b):
                i = indices[i_pos]
                ai, bi = unique_pairs[i]
                for j_pos in range(i_pos + 1, n_b):
                    j = indices[j_pos]
                    aj, bj = unique_pairs[j]
                    if (ai == aj and _hamming_le_1(bi, bj)) or (bi == bj and _hamming_le_1(ai, aj)):
                        _uf_union(parent, i, j)

        pair_to_gid: dict[tuple[str, str], str] = {}
        for idx, pair in enumerate(unique_pairs):
            root = _uf_find(parent, idx)
            rep_a, rep_b = unique_pairs[root]
            pair_to_gid[pair] = _digest("tcrgrp", (str(pep), str(mhc), rep_a, rep_b))

        row_gids = [
            pair_to_gid[(str(a), str(b))]
            for a, b in zip(grp["cdr3_alpha"], grp["cdr3_beta"], strict=True)
        ]
        group_ids.loc[grp.index] = pd.Series(row_gids, index=grp.index, dtype="string")
    return group_ids


def load_tcrvdb_functional_evidence(
    source_path: str | Path | None = None,
    *,
    release_version: str = "",
    expected_sha256: str | None = None,
    require_paired_chains: bool = True,
) -> pd.DataFrame:
    """Load an authorized local TCRvdb CSV into the canonical paired-receptor schema (#722).

    Parameters
    ----------
    source_path
        Path to an authorized local TCRvdb CSV (e.g.
        ``2026-10-01_TCRvdb_ttd_gil_ylq_nlv_glc.csv``). When ``None``, resolves
        the locally registered ``tcrvdb`` dataset via :func:`hitlist.downloads.get_path`.
    release_version
        Optional explicit release identifier (defaults to the source filename stem).
    expected_sha256
        Optional SHA-256 hex digest pinned by the caller; raises ``ValueError``
        if the local file bytes differ.
    require_paired_chains
        When ``True`` (default), rejects any row missing either alpha or beta
        chain CDR3/V/J pairing (never pairs unrelated chains across rows).
    """
    if source_path is None:
        from .downloads import get_path

        try:
            resolved_path = get_path("tcrvdb")
        except (KeyError, FileNotFoundError) as exc:
            raise FileNotFoundError(
                "TCRvdb is access-controlled by request (https://tcrvdb.nkiai.nl; "
                "DOI 10.1016/j.immuni.2026.09.007) and is not bundled with Hitlist. "
                "After obtaining authorized access, register your local CSV via "
                "`hitlist data register tcrvdb /path/to/2026-10-01_TCRvdb_ttd_gil_ylq_nlv_glc.csv` "
                "or pass source_path explicitly."
            ) from exc
    else:
        resolved_path = Path(source_path)

    if not resolved_path.exists():
        raise FileNotFoundError(f"TCRvdb source file not found: {resolved_path}")

    digest_info = file_digest(resolved_path)
    src_sha256 = digest_info["sha256"]
    if expected_sha256 is not None and src_sha256 != expected_sha256.strip().lower():
        raise ValueError(
            f"TCRvdb source SHA-256 mismatch: expected {expected_sha256}, got {src_sha256}"
        )

    sep = "\t" if resolved_path.suffix.lower() in {".tsv", ".txt"} else ","
    raw = pd.read_csv(resolved_path, sep=sep, dtype=str, keep_default_na=False)
    rel_ver = release_version.strip() or resolved_path.stem
    if len(raw) == 0:
        return pd.DataFrame(columns=list(TCRVDB_ADAPTER_COLUMNS))

    row_indices = pd.Series(range(len(raw)), index=raw.index, dtype="Int64")
    source_row_id = _resolve_col(raw, "source_row_id")
    missing_row_id = source_row_id.eq("")
    if missing_row_id.any():
        source_row_id.loc[missing_row_id] = [
            f"{rel_ver}:row:{i}" for i in row_indices.loc[missing_row_id]
        ]

    peptide = _resolve_col(raw, "peptide").str.upper()
    mhc_restriction = _resolve_col(raw, "mhc_restriction")
    mhc_class = _resolve_col(raw, "mhc_class", default="I").replace("", "I")

    trav = _resolve_col(raw, "trav")
    traj = _resolve_col(raw, "traj")
    cdr3_alpha = _resolve_col(raw, "cdr3_alpha").str.upper()
    trbv = _resolve_col(raw, "trbv")
    trbj = _resolve_col(raw, "trbj")
    cdr3_beta = _resolve_col(raw, "cdr3_beta").str.upper()
    trav_collapsed = _resolve_col(raw, "trav_allele_collapsed")
    trbv_collapsed = _resolve_col(raw, "trbv_allele_collapsed")

    has_alpha = cdr3_alpha.ne("") & (trav.ne("") | traj.ne(""))
    has_beta = cdr3_beta.ne("") & (trbv.ne("") | trbj.ne(""))
    paired_mask = has_alpha & has_beta
    if require_paired_chains and not paired_mask.all():
        bad_rows = row_indices.loc[~paired_mask].tolist()[:5]
        raise ValueError(
            f"TCRvdb rows {bad_rows} lack paired alpha+beta chain identity; "
            "unrelated chains sharing an epitope are never paired (#722)."
        )

    pairing_status = pd.Series("unpaired_incomplete", index=raw.index, dtype="string")
    pairing_status.loc[paired_mask] = "paired_alpha_beta_same_receptor"

    receptor_id = _resolve_col(raw, "receptor_id")
    receptor_pair_ids = pd.Series(
        [
            _digest("tcrpair", (a_v, a_j, a_c, b_v, b_j, b_c))
            for a_v, a_j, a_c, b_v, b_j, b_c in zip(
                trav, traj, cdr3_alpha, trbv, trbj, cdr3_beta, strict=True
            )
        ],
        index=raw.index,
        dtype="string",
    )
    missing_rec_id = receptor_id.eq("")
    if missing_rec_id.any():
        receptor_id.loc[missing_rec_id] = receptor_pair_ids.loc[missing_rec_id]

    alpha_full = _resolve_col(raw, "alpha_full_aa").str.upper()
    beta_full = _resolve_col(raw, "beta_full_aa").str.upper()
    recon_method = _resolve_col(raw, "reconstruction_method")
    alpha_prov_raw = _resolve_col(raw, "alpha_sequence_provenance")
    beta_prov_raw = _resolve_col(raw, "beta_sequence_provenance")
    alpha_prov = pd.Series(
        [
            _resolve_chain_provenance(ex, fl, c3, vg, jg, rm)
            for ex, fl, c3, vg, jg, rm in zip(
                alpha_prov_raw, alpha_full, cdr3_alpha, trav, traj, recon_method, strict=True
            )
        ],
        index=raw.index,
        dtype="string",
    )
    beta_prov = pd.Series(
        [
            _resolve_chain_provenance(ex, fl, c3, vg, jg, rm)
            for ex, fl, c3, vg, jg, rm in zip(
                beta_prov_raw, beta_full, cdr3_beta, trbv, trbj, recon_method, strict=True
            )
        ],
        index=raw.index,
        dtype="string",
    )

    assay_type = _resolve_col(raw, "assay_type", default="pooled_cd69_ont_screen").replace(
        "", "pooled_cd69_ont_screen"
    )
    assay_context = _resolve_col(raw, "assay_context", default="jurkat_cd8_coculture").replace(
        "", "jurkat_cd8_coculture"
    )
    raw_val_label = _resolve_col(raw, "raw_validation_label")
    qc_status = _resolve_col(raw, "qc_status", default="unreported").replace("", "unreported")
    expression_status = _resolve_col(raw, "expression_status", default="unreported").replace(
        "", "unreported"
    )
    author_policy = _resolve_col(raw, "author_validation_policy", default="tcrvdb_default").replace(
        "", "tcrvdb_default"
    )

    val_status = pd.Series(
        [
            _normalize_validation_status(lbl, qc)
            for lbl, qc in zip(raw_val_label, qc_status, strict=True)
        ],
        index=raw.index,
        dtype="string",
    )
    non_val_scope = pd.Series("not_applicable", index=raw.index, dtype="string")
    non_val_scope.loc[val_status.eq("tested_not_validating")] = (
        "assay_scoped_non_validation_not_universal_nonbinding"
    )

    assay_record_ids = pd.Series(
        [
            _digest("tcrassay", (src_sha256[:16], rid, rpid, pep, mhc, atype))
            for rid, rpid, pep, mhc, atype in zip(
                source_row_id, receptor_pair_ids, peptide, mhc_restriction, assay_type, strict=True
            )
        ],
        index=raw.index,
        dtype="string",
    )

    n_reps_num = _resolve_numeric(raw, "n_replicates")
    n_replicates = n_reps_num.round().astype("Int64")

    out = pd.DataFrame(
        {
            "source_row_index": row_indices,
            "source_row_id": source_row_id,
            "source_file_sha256": pd.Series(src_sha256, index=raw.index, dtype="string"),
            "source_release_version": pd.Series(rel_ver, index=raw.index, dtype="string"),
            "receptor_id": receptor_id,
            "receptor_pair_id": receptor_pair_ids,
            "assay_record_id": assay_record_ids,
            "pairing_status": pairing_status,
            "peptide": peptide,
            "mhc_restriction": mhc_restriction,
            "mhc_class": mhc_class,
            "trav": trav,
            "traj": traj,
            "cdr3_alpha": cdr3_alpha,
            "trbv": trbv,
            "trbj": trbj,
            "cdr3_beta": cdr3_beta,
            "trav_allele_collapsed": trav_collapsed,
            "trbv_allele_collapsed": trbv_collapsed,
            "alpha_full_aa": alpha_full,
            "beta_full_aa": beta_full,
            "alpha_sequence_provenance": alpha_prov,
            "beta_sequence_provenance": beta_prov,
            "reconstruction_method": recon_method,
            "assay_type": assay_type,
            "assay_context": assay_context,
            "validation_status": val_status,
            "raw_validation_label": raw_val_label,
            "non_validation_scope": non_val_scope,
            "qc_status": qc_status,
            "expression_status": expression_status,
            "author_validation_policy": author_policy,
            "n_replicates": n_replicates,
            "base_mean": _resolve_numeric(raw, "base_mean"),
            "log2_fold_change": _resolve_numeric(raw, "log2_fold_change"),
            "lfc_se": _resolve_numeric(raw, "lfc_se"),
            "stat": _resolve_numeric(raw, "stat"),
            "pvalue": _resolve_numeric(raw, "pvalue"),
            "padj": _resolve_numeric(raw, "padj"),
            "cd69_positive_pct": _resolve_numeric(raw, "cd69_positive_pct"),
            "vdjdb_reference_id": _resolve_col(raw, "vdjdb_reference_id"),
            "vdjdb_score": _resolve_col(raw, "vdjdb_score"),
            "source_pmid": _resolve_col(raw, "source_pmid"),
            "source_study": _resolve_col(raw, "source_study"),
            "evidence_modality": pd.Series(
                "tcr_paired_functional", index=raw.index, dtype="string"
            ),
        },
        index=raw.index,
    )
    out["near_duplicate_group_id"] = _assign_near_duplicate_groups(out)
    return out[list(TCRVDB_ADAPTER_COLUMNS)]


def build_tcrvdb_near_duplicate_groups(df: pd.DataFrame) -> pd.DataFrame:
    """Build separate-grain near-duplicate receptor clusters for leakage auditing (#722)."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    if len(df) == 0:
        return pd.DataFrame(columns=list(NEAR_DUPLICATE_GROUP_COLUMNS))

    rows: list[dict[str, Any]] = []
    for gid, grp in df.groupby("near_duplicate_group_id", sort=False):
        counts = Counter(grp["validation_status"].astype("string"))
        studies = sorted({s for s in grp["source_study"].astype("string") if s})
        rows.append(
            {
                "near_duplicate_group_id": str(gid),
                "peptide": str(grp["peptide"].iloc[0]),
                "mhc_restriction": str(grp["mhc_restriction"].iloc[0]),
                "n_receptors": int(grp["receptor_pair_id"].nunique()),
                "n_rows": len(grp),
                "n_validated": int(counts.get("validated", 0)),
                "n_tested_not_validating": int(counts.get("tested_not_validating", 0)),
                "n_indeterminate": int(counts.get("indeterminate", 0)),
                "n_failed_qc": int(counts.get("failed_qc", 0)),
                "n_unassayed": int(counts.get("unassayed", 0)),
                "has_conflicting_validation": bool(
                    counts.get("validated", 0) > 0 and counts.get("tested_not_validating", 0) > 0
                ),
                "n_distinct_studies": len(studies),
                "source_studies": ";".join(studies),
                "representative_cdr3_alpha": str(grp["cdr3_alpha"].iloc[0]),
                "representative_cdr3_beta": str(grp["cdr3_beta"].iloc[0]),
            }
        )
    return pd.DataFrame(rows, columns=list(NEAR_DUPLICATE_GROUP_COLUMNS))


def audit_tcrvdb_leakage_and_coverage(df: pd.DataFrame) -> dict[str, Any]:
    """Summarize epitope/MHC coverage, sequence provenance, and receptor/study overlap (#722)."""
    groups = build_tcrvdb_near_duplicate_groups(df)
    receptors_per_pmhc = (
        df.groupby(["peptide", "mhc_restriction"], sort=True)["receptor_pair_id"]
        .nunique()
        .to_dict()
        if len(df)
        else {}
    )
    pmhcs_per_receptor = (
        df.groupby("receptor_pair_id")["peptide"].nunique() if len(df) else pd.Series(dtype=int)
    )
    studies_per_receptor = (
        df[df["source_study"].astype("string").ne("")]
        .groupby("receptor_pair_id")["source_study"]
        .nunique()
        if len(df)
        else pd.Series(dtype=int)
    )
    return {
        "contract_revision": TCRVDB_CONTRACT_REVISION,
        "evidence_modality": "tcr_paired_functional",
        "n_rows": len(df),
        "n_unique_receptors": int(df["receptor_pair_id"].nunique()) if len(df) else 0,
        "n_unique_epitopes": int(df["peptide"].nunique()) if len(df) else 0,
        "n_unique_pmhc": len(receptors_per_pmhc),
        "n_near_duplicate_groups": len(groups),
        "n_multi_receptor_near_duplicate_groups": int((groups["n_receptors"] > 1).sum())
        if len(groups)
        else 0,
        "n_conflicting_validation_groups": int(groups["has_conflicting_validation"].sum())
        if len(groups)
        else 0,
        "n_receptors_across_multiple_epitopes": int((pmhcs_per_receptor > 1).sum())
        if len(pmhcs_per_receptor)
        else 0,
        "n_receptors_across_multiple_studies": int((studies_per_receptor > 1).sum())
        if len(studies_per_receptor)
        else 0,
        "by_validation_status": dict(
            sorted(Counter(df["validation_status"].astype("string")).items())
        ),
        "by_alpha_sequence_provenance": dict(
            sorted(Counter(df["alpha_sequence_provenance"].astype("string")).items())
        ),
        "by_beta_sequence_provenance": dict(
            sorted(Counter(df["beta_sequence_provenance"].astype("string")).items())
        ),
        "n_receptors_by_pmhc": {
            f"{pep}|{mhc}": int(n) for (pep, mhc), n in receptors_per_pmhc.items()
        },
        "arbitrary_epitope_generalization_claimed": False,
        "redistribution_allowed": False,
    }


def export_tcrvdb_local_bundle(
    source_path: str | Path,
    destination: str | Path,
    *,
    release_version: str = "",
    expected_sha256: str | None = None,
) -> Path:
    """Export a reproducible local TCRvdb bundle + receipt respecting source access terms (#722)."""
    source_path = Path(source_path)
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)

    contract = tcrvdb_source_contract()
    paired_df = load_tcrvdb_functional_evidence(
        source_path,
        release_version=release_version,
        expected_sha256=expected_sha256,
    )
    groups_df = build_tcrvdb_near_duplicate_groups(paired_df)
    coverage = audit_tcrvdb_leakage_and_coverage(paired_df)
    src_digest = file_digest(source_path)

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    try:
        receptors_path = staging / "paired_receptors.parquet"
        groups_path = staging / "near_duplicate_groups.parquet"
        pq.write_table(pa.Table.from_pandas(paired_df, preserve_index=False), receptors_path)
        pq.write_table(pa.Table.from_pandas(groups_df, preserve_index=False), groups_path)

        from .version import __version__

        receipt = {
            "schema_version": 1,
            "revision": TCRVDB_CONTRACT_REVISION,
            "hitlist_version": __version__,
            "evidence_modality": "tcr_paired_functional",
            "access_terms": contract["access_terms"],
            "policies": contract["policies"],
            "source_file": {
                "filename": source_path.name,
                "release_version": release_version.strip() or source_path.stem,
                **src_digest,
            },
            "coverage": coverage,
            "artifacts": {
                "paired_receptors.parquet": file_digest(receptors_path),
                "near_duplicate_groups.parquet": file_digest(groups_path),
            },
        }
        receipt_path = staging / "bundle_receipt.json"
        receipt_path.write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        verify_tcrvdb_local_bundle(staging)
        staging.rename(destination)
        return destination / "bundle_receipt.json"
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def verify_tcrvdb_local_bundle(directory: str | Path) -> dict[str, Any]:
    """Verify integrity and access-term compliance of a local TCRvdb bundle (#722)."""
    directory = Path(directory)
    receipt_path = directory / "bundle_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("revision") != TCRVDB_CONTRACT_REVISION:
        raise ValueError("Unsupported TCRvdb bundle revision")
    if receipt.get("access_terms", {}).get("redistribution_allowed") is not False:
        raise ValueError("TCRvdb bundle receipt violates redistribution policy")
    for rel_name, expected in (receipt.get("artifacts") or {}).items():
        actual = file_digest(directory / rel_name)
        if actual["sha256"] != expected["sha256"]:
            raise ValueError(f"TCRvdb bundle artifact digest mismatch: {rel_name}")
    receptors = pd.read_parquet(directory / "paired_receptors.parquet")
    groups = pd.read_parquet(directory / "near_duplicate_groups.parquet")
    if len(receptors) != receipt["coverage"]["n_rows"]:
        raise ValueError("TCRvdb bundle receptor row count mismatch")
    if len(groups) != receipt["coverage"]["n_near_duplicate_groups"]:
        raise ValueError("TCRvdb bundle near-duplicate group count mismatch")
    return receipt
