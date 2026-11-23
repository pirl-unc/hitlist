"""Source-qualified Bekker-Jensen 2017 (PMID 28601559, PXD004452) bulk proteomics contract.

Implements #654, #96, and #95:
- Pins verified original PXD004452/SearchResults.zip provenance (`parameters.txt` SHA-256
  `d1d760c8...`, `summary.txt` SHA-256 `5eb5c41a...`), exact MaxQuant 1.5.3.19 vs paper 1.5.3.6
  search settings, enzyme/missed-cleavage rules (`Trypsin/P` 3, `Chymotrypsin+` 4, `GluC;D.P` 3,
  `LysC/P` 2), and the `HUMAN.fasta` non-availability audit separate from `PXD013455`.
- Resolves #96 by parsing MaxQuant `modificationSpecificPeptides.txt` per-experiment rows so
  each `(peptide, arm)` gets TRUE per-arm modifications (`per_arm_modification_specific_peptides`)
  rather than deposit-wide aggregate modifications from `peptides.txt` (preventing TiO2
  `Phospho (STY)` bleed into baseline arms).
- Resolves #95 (including Presto review constraints) by preserving separate raw MS1 intensity
  (`log2_raw_intensity`, `log2_raw_intensity_std`) and LFQ (`log2_lfq_intensity`, always NaN on
  PXD004452 peptide tables because MaxQuant emits 0 peptide-level LFQ columns) — NEVER silently
  mixing scales — alongside exact per-experiment IDs (`experiment_ids_detected`) and direct
  MS/MS vs Match-Between-Runs counts (`n_direct_msms_experiments`, `n_mbr_only_experiments`).
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_DATA_PATH = (
    Path(__file__).parent / "data" / "evidence_revisions" / "bekker_jensen_2017_pxd004452.json"
)
_TRYPTIC_LABEL = "Trypsin/P (cleaves K/R except before P)"
_SKIP_PREFIXES = ("Colon-", "Liver-", "Prostate-", "SY5Y-")
_ENZYME_TOKENS = {
    "Tryp": _TRYPTIC_LABEL,
    "Chymo": "Chymotrypsin",
    "GluC": "GluC",
    "LysC": "LysC",
    "Phospho": _TRYPTIC_LABEL,
}


def bekker_jensen_source_contract() -> dict[str, Any]:
    """Return the pinned Bekker-Jensen 2017 (PMID 28601559, PXD004452) source contract."""
    return json.loads(_DATA_PATH.read_text(encoding="utf-8"))


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _parse_experiment_axes_pxd004452(name: str) -> dict[str, Any] | None:
    """Map one PXD004452 MaxQuant experiment name to its Fig 1b design axes."""
    from .bulk_fractionation import experiment_fractionation_ph

    if any(name.startswith(pref) for pref in _SKIP_PREFIXES) or "-Pandey" in name:
        return None
    m = re.fullmatch(r"HeLa-(\d+)fracs-IT-E(\d+)", name)
    if m:
        return {
            "experiment_id": name,
            "cell_line": "HeLa",
            "digestion_enzyme": _TRYPTIC_LABEL,
            "search_enzyme": "Trypsin/P",
            "n_fractions_in_run": int(m.group(1)),
            "enrichment": "none",
            "fractionation_ph": experiment_fractionation_ph(name),
            "replicate_label": f"E{m.group(2)}",
        }
    m = re.fullmatch(r"(A549|HCT116|HEK293|MCF7)-(\d+)fracs-E(\d+)", name)
    if m:
        return {
            "experiment_id": name,
            "cell_line": m.group(1),
            "digestion_enzyme": _TRYPTIC_LABEL,
            "search_enzyme": "Trypsin/P",
            "n_fractions_in_run": int(m.group(2)),
            "enrichment": "none",
            "fractionation_ph": experiment_fractionation_ph(name),
            "replicate_label": f"E{m.group(3)}",
        }
    m = re.fullmatch(r"Tryp-(\d+)Frac-([A-Z])", name)
    if m:
        return {
            "experiment_id": name,
            "cell_line": "HeLa",
            "digestion_enzyme": _TRYPTIC_LABEL,
            "search_enzyme": "Trypsin/P",
            "n_fractions_in_run": int(m.group(1)),
            "enrichment": "none",
            "fractionation_ph": experiment_fractionation_ph(name),
            "replicate_label": f"E{ord(m.group(2)) - ord('A') + 1}",
        }
    m = re.fullmatch(r"Tryp-Phos-pH(\d+)", name)
    if m:
        return {
            "experiment_id": name,
            "cell_line": "HeLa",
            "digestion_enzyme": _TRYPTIC_LABEL,
            "search_enzyme": "Trypsin/P",
            "n_fractions_in_run": 12,
            "enrichment": "TiO2",
            "fractionation_ph": experiment_fractionation_ph(name),
            "replicate_label": "E1",
        }
    if name == "Phospho-50fracs":
        return {
            "experiment_id": name,
            "cell_line": "HeLa",
            "digestion_enzyme": _TRYPTIC_LABEL,
            "search_enzyme": "Trypsin/P",
            "n_fractions_in_run": 50,
            "enrichment": "TiO2",
            "fractionation_ph": experiment_fractionation_ph(name),
            "replicate_label": "E1",
        }
    m = re.fullmatch(r"(Tryp|Chymo|GluC|LysC)-(\d+)fracs", name)
    if m:
        tok = m.group(1)
        search_map = {
            "Tryp": "Trypsin/P",
            "Chymo": "Chymotrypsin+",
            "GluC": "GluC;D.P",
            "LysC": "LysC/P",
        }
        return {
            "experiment_id": name,
            "cell_line": "HeLa",
            "digestion_enzyme": _ENZYME_TOKENS[tok],
            "search_enzyme": search_map[tok],
            "n_fractions_in_run": int(m.group(2)),
            "enrichment": "none",
            "fractionation_ph": experiment_fractionation_ph(name),
            "replicate_label": "E1",
        }
    return None


def fetch_bekker_jensen_provenance_files(
    dest_dir: str | Path | None = None,
    *,
    force: bool = False,
    offline: bool | None = None,
    range_getter: Any = None,
) -> Path:
    """Fetch and SHA-256-verify PXD004452 ``parameters.txt`` and ``summary.txt`` via HTTP Range.

    Extracts only the two small provenance files (~118 KB compressed / ~589 KB uncompressed)
    from the 21.7 GB ``SearchResults.zip`` archive using HTTP 206 Range reads.
    """
    import urllib.request

    from .downloads import _require_online, data_dir
    from .erap2_oe_qualification import _extract_remote_zip_members
    from .errors import IntegrityError

    contract = bekker_jensen_source_contract()
    archive = contract["archive"]
    members = {k: archive["verified_members"][k] for k in ("parameters.txt", "summary.txt")}
    target_dir = (
        Path(dest_dir) if dest_dir is not None else data_dir() / "bekker_jensen_2017_pxd004452"
    )
    target_dir.mkdir(parents=True, exist_ok=True)

    needed = {
        fname: meta["member_suffix"]
        for fname, meta in members.items()
        if force or not (target_dir / fname).is_file()
    }
    if needed:
        _require_online(f"PXD004452 provenance files from {archive['url']}", offline=offline)
        if range_getter is None:
            head_req = urllib.request.Request(archive["url"], method="HEAD")
            with urllib.request.urlopen(head_req, timeout=60.0) as resp:
                archive_size = int(resp.headers["Content-Length"])
        else:
            archive_size = int(archive.get("size_bytes", 0))
            if archive_size <= 0:
                raise ValueError("archive_size must be provided when using a custom range_getter")
        extracted = _extract_remote_zip_members(
            archive["url"],
            archive_size,
            needed,
            range_getter=range_getter,
        )
        for fname, payload in extracted.items():
            meta = members[fname]
            actual_sha = _sha256_bytes(payload)
            if actual_sha != meta["sha256"] or len(payload) != int(meta["size_bytes"]):
                raise IntegrityError(
                    f"Integrity mismatch for PXD004452 {fname}: expected {meta['sha256']}, "
                    f"got {actual_sha}"
                )
            tmp = target_dir / f".{fname}.part"
            tmp.write_bytes(payload)
            tmp.replace(target_dir / fname)

    for fname, meta in members.items():
        path = target_dir / fname
        actual_sha = _sha256_file(path)
        if actual_sha != meta["sha256"] or path.stat().st_size != int(meta["size_bytes"]):
            raise IntegrityError(
                f"Cached PXD004452 {fname} failed SHA-256 verification: expected {meta['sha256']}, "
                f"got {actual_sha}"
            )
    return target_dir


def _normalize_mod_tokens(mod_raw: str) -> list[str]:
    """Normalize MaxQuant modificationSpecificPeptides.txt ``Modifications`` tokens."""
    text = mod_raw.strip()
    if not text or text.lower() == "unmodified":
        return []
    tokens: list[str] = []
    for part in text.split(","):
        cleaned = re.sub(r"^\d+\s+", "", part.strip())
        if cleaned and cleaned.lower() != "unmodified":
            tokens.append(cleaned)
    return tokens


def parse_bekker_jensen_mod_specific_peptides(
    path: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Parse MaxQuant ``modificationSpecificPeptides.txt`` into per-experiment and per-arm frames.

    Resolves #96 (per-arm modifications without deposit-wide phospho bleed) and #95 (separate
    raw MS1 intensity vs LFQ scales, per-experiment lineage, and direct MS/MS vs MBR provenance).

    Returns
    -------
    (experiment_observations_df, arm_summary_df)
    """
    src_path = Path(path)
    src_sha = _sha256_file(src_path)

    exp_obs_rows: list[dict[str, Any]] = []
    # arm_key -> accumulator for arm-level summary
    arm_acc: dict[tuple[Any, ...], dict[str, Any]] = {}

    with src_path.open("r", encoding="latin-1", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        fieldnames = list(reader.fieldnames or ())
        if "Sequence" not in fieldnames or "Modifications" not in fieldnames:
            raise ValueError(f"{src_path.name} must contain 'Sequence' and 'Modifications' columns")

        exp_names = [
            c[len("Intensity ") :]
            for c in fieldnames
            if c.startswith("Intensity ") and c != "Intensity"
        ]
        parsed_exps = {
            exp: axes
            for exp in exp_names
            if (axes := _parse_experiment_axes_pxd004452(exp)) is not None
        }
        lfq_cols = {
            exp: f"LFQ intensity {exp}"
            for exp in parsed_exps
            if f"LFQ intensity {exp}" in fieldnames
        }
        id_type_cols = {
            exp: f"Identification type {exp}"
            for exp in parsed_exps
            if f"Identification type {exp}" in fieldnames
        }

        for row_idx, row in enumerate(reader, start=2):
            if (
                row.get("Reverse", "").strip() == "+"
                or row.get("Potential contaminant", "").strip() == "+"
            ):
                continue
            seq = row["Sequence"].strip()
            if not seq:
                continue
            mod_raw = row["Modifications"].strip() or "Unmodified"
            mod_tokens = _normalize_mod_tokens(mod_raw)
            proteins = row.get("Proteins", "").strip()
            uniprot = proteins.split(";", 1)[0] if proteins else ""
            gene_names = row.get("Gene Names", row.get("Gene names", "")).strip()
            gene_primary = gene_names.split(";", 1)[0] if gene_names else ""

            for exp, axes in parsed_exps.items():
                raw_str = row.get(f"Intensity {exp}", "").strip()
                if not raw_str:
                    continue
                try:
                    raw_val = float(raw_str)
                except ValueError:
                    continue
                if not math.isfinite(raw_val) or raw_val <= 0.0:
                    continue

                lfq_val = float("nan")
                if exp in lfq_cols:
                    lfq_str = row.get(lfq_cols[exp], "").strip()
                    if lfq_str:
                        try:
                            cand = float(lfq_str)
                            if math.isfinite(cand) and cand > 0.0:
                                lfq_val = cand
                        except ValueError:
                            pass

                id_type = row.get(id_type_cols[exp], "").strip() if exp in id_type_cols else ""
                is_direct_msms = id_type == "By MS/MS"
                is_mbr = id_type == "By matching"
                log2_raw = math.log2(raw_val)
                log2_lfq = (
                    math.log2(lfq_val) if math.isfinite(lfq_val) and lfq_val > 0 else float("nan")
                )

                exp_obs_rows.append(
                    {
                        "source_asset": src_path.name,
                        "source_sha256": src_sha,
                        "source_row": row_idx,
                        "peptide": seq,
                        "modifications_raw": mod_raw,
                        "uniprot_acc": uniprot,
                        "gene_symbol": gene_primary or None,
                        "experiment_id": exp,
                        "cell_line_name": axes["cell_line"],
                        "digestion_enzyme": axes["digestion_enzyme"],
                        "search_enzyme": axes["search_enzyme"],
                        "n_fractions_in_run": axes["n_fractions_in_run"],
                        "enrichment": axes["enrichment"],
                        "fractionation_ph": axes["fractionation_ph"],
                        "raw_intensity": raw_val,
                        "log2_raw_intensity": log2_raw,
                        "lfq_intensity": lfq_val,
                        "log2_lfq_intensity": log2_lfq,
                        "intensity_scale": "raw_ms1_intensity",
                        "identification_type": id_type,
                        "is_direct_msms": is_direct_msms,
                        "is_mbr_transferred": is_mbr,
                        "modifications_scope": "per_arm_modification_specific_peptides",
                        "source": "Bekker-Jensen_2017",
                        "reference": "PMID:28601559",
                    }
                )

                arm_key = (
                    seq,
                    axes["cell_line"],
                    axes["digestion_enzyme"],
                    axes["search_enzyme"],
                    axes["n_fractions_in_run"],
                    axes["enrichment"],
                    axes["fractionation_ph"],
                )
                acc = arm_acc.get(arm_key)
                if acc is None:
                    acc = {
                        "peptide": seq,
                        "cell_line_name": axes["cell_line"],
                        "uniprot_acc": uniprot,
                        "gene_symbol": gene_primary or None,
                        "length": len(seq),
                        "digestion_enzyme": axes["digestion_enzyme"],
                        "search_enzyme": axes["search_enzyme"],
                        "n_fractions_in_run": axes["n_fractions_in_run"],
                        "enrichment": axes["enrichment"],
                        "fractionation_ph": axes["fractionation_ph"],
                        "_mods": set(),
                        "_exp_raw_sums": {},
                        "_exp_lfq_sums": {},
                        "_msms_exps": set(),
                        "_mbr_exps": set(),
                        "source": "Bekker-Jensen_2017",
                        "reference": "PMID:28601559",
                    }
                    arm_acc[arm_key] = acc

                acc["_mods"].update(mod_tokens)
                acc["_exp_raw_sums"][exp] = acc["_exp_raw_sums"].get(exp, 0.0) + raw_val
                if math.isfinite(lfq_val) and lfq_val > 0:
                    acc["_exp_lfq_sums"][exp] = acc["_exp_lfq_sums"].get(exp, 0.0) + lfq_val
                if is_direct_msms:
                    acc["_msms_exps"].add(exp)
                elif is_mbr:
                    acc["_mbr_exps"].add(exp)

    arm_rows: list[dict[str, Any]] = []
    for acc in arm_acc.values():
        mods = sorted(acc.pop("_mods"))
        exp_raw = acc.pop("_exp_raw_sums")
        exp_lfq = acc.pop("_exp_lfq_sums")
        msms_exps = acc.pop("_msms_exps")
        mbr_exps = acc.pop("_mbr_exps") - msms_exps

        raw_log2_vals = [math.log2(v) for v in exp_raw.values() if v > 0]
        lfq_log2_vals = [math.log2(v) for v in exp_lfq.values() if v > 0]

        mean_raw = float(np.mean(raw_log2_vals)) if raw_log2_vals else float("nan")
        std_raw = float(np.std(raw_log2_vals, ddof=1)) if len(raw_log2_vals) >= 2 else float("nan")
        mean_lfq = float(np.mean(lfq_log2_vals)) if lfq_log2_vals else float("nan")
        std_lfq = float(np.std(lfq_log2_vals, ddof=1)) if len(lfq_log2_vals) >= 2 else float("nan")

        acc["modifications"] = ";".join(mods) if mods else "Unmodified"
        acc["modifications_scope"] = "per_arm_modification_specific_peptides"
        acc["n_experiments_detected"] = len(exp_raw)
        acc["experiment_ids_detected"] = ";".join(sorted(exp_raw))
        acc["n_direct_msms_experiments"] = len(msms_exps)
        acc["n_mbr_only_experiments"] = len(mbr_exps)
        acc["log2_raw_intensity"] = mean_raw
        acc["log2_raw_intensity_std"] = std_raw
        acc["log2_lfq_intensity"] = mean_lfq
        acc["log2_lfq_intensity_std"] = std_lfq
        # Preserve raw-scale alias without EVER silently falling back across scales.
        acc["log2_intensity"] = mean_raw
        acc["log2_intensity_std"] = std_raw
        acc["intensity_scale"] = "raw_ms1_intensity"
        acc["experiment_scope_provenance"] = "per_experiment_preserved"
        acc["source_asset"] = src_path.name
        acc["source_sha256"] = src_sha
        arm_rows.append(acc)

    return pd.DataFrame(exp_obs_rows), pd.DataFrame(arm_rows)


def qualify_bekker_jensen_peptide_table(df: pd.DataFrame) -> pd.DataFrame:
    """Stamp explicit intensity-scale and modification-scope provenance onto a Bekker-Jensen frame.

    Ensures callers can distinguish legacy deposit-aggregate ``peptides.txt`` rows from
    per-arm ``modificationSpecificPeptides.txt`` rows and never confuse missing peptide-level
    LFQ with raw MS1 intensity or zero.
    """
    out = df.copy()
    if "modifications_scope" not in out.columns:
        out["modifications_scope"] = "deposit_aggregate_peptides_txt"
    if "log2_raw_intensity" not in out.columns:
        out["log2_raw_intensity"] = (
            out["log2_intensity"] if "log2_intensity" in out.columns else float("nan")
        )
    if "log2_raw_intensity_std" not in out.columns:
        out["log2_raw_intensity_std"] = (
            out["log2_intensity_std"] if "log2_intensity_std" in out.columns else float("nan")
        )
    if "log2_lfq_intensity" not in out.columns:
        out["log2_lfq_intensity"] = float("nan")
    if "log2_lfq_intensity_std" not in out.columns:
        out["log2_lfq_intensity_std"] = float("nan")
    if "intensity_scale" not in out.columns:
        has_raw = out["log2_raw_intensity"].notna().any()
        out["intensity_scale"] = "raw_ms1_intensity" if has_raw else "unquantified_presence_only"
    if "experiment_scope_provenance" not in out.columns:
        out["experiment_scope_provenance"] = (
            "per_experiment_preserved"
            if "experiment_ids_detected" in out.columns
            else "pooled_legacy_csv"
        )
    return out
