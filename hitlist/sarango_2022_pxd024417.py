"""Source-qualified Sarango 2022 (PMID 36215666, PXD024417) MHC-II measurement qualification (#858).

Implements #858:
- Pins the verified 81-file PRIDE PXD024417 inventory (40 .raw, 40 .msf, 1 community SDRF
  PXD024417_community_annotated.sdrf.tsv, SHA-256 793eeb55...).
- Separates biological material, condition (Mock, siCTRL, siT6BP), MHC class/capture (W6/32 vs
  Tue39), and 5 technical acquisitions per group, explicitly recording that 5 technical files are
  NOT 5 independent biological replicates and that the paper's second biological siRNA experiment
  mapping remains unresolved in the 40-file deposit.
- Preserves community SDRF MHC typing (where characteristics[mhc typing method] = 'not available')
  strictly as secondary unverified annotation — never promoting it to primary genotype, never
  restoring withdrawn HeLa Class-I alleles, and never manufacturing a DRA allele.
- Keeps peptide MS intensities, PLAtEAU core summaries, computational DRB1*01:02 affinity
  estimates, and shared IEDB presence/absence memberships strictly distinct.
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd

_DATA_PATH = Path(__file__).parent / "data" / "evidence_revisions" / "sarango_2022_pxd024417.json"


def sarango_2022_source_contract() -> dict[str, Any]:
    """Return the pinned Sarango 2022 (PMID 36215666, PXD024417) source qualification contract."""
    return json.loads(_DATA_PATH.read_text(encoding="utf-8"))


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_sarango_2022_sdrf(
    dest_dir: str | Path | None = None,
    *,
    force: bool = False,
    allow_unverified_checksum: bool = False,
) -> Path:
    """Download and SHA-256-verify the PXD024417 community-annotated SDRF file."""
    from .downloads import data_dir

    contract = sarango_2022_source_contract()
    asset = contract["pride_inventory"]["verified_sdrf_asset"]
    target_dir = Path(dest_dir) if dest_dir is not None else data_dir() / "sarango_2022_pxd024417"
    target_dir.mkdir(parents=True, exist_ok=True)
    out_path = target_dir / asset["filename"]
    if force or not out_path.exists():
        tmp = out_path.with_suffix(out_path.suffix + ".part")
        urllib.request.urlretrieve(asset["url"], tmp)
        tmp.replace(out_path)
    actual = _sha256_file(out_path)
    if not allow_unverified_checksum and actual != asset["sha256"]:
        raise ValueError(
            f"SHA-256 mismatch for {out_path.name}: expected {asset['sha256']}, got {actual}"
        )
    return out_path


def _arm_from_source_name(source_name: str) -> tuple[str, str]:
    text = source_name.strip()
    if "_Mock" in text:
        return "Mock", "hela_ciita_mock"
    if "_siCTRL" in text:
        return "siCTRL", "hela_ciita_control_sirna"
    if "_siT6BP" in text:
        return "siT6BP", "hela_ciita_t6bp_sirna"
    return "unknown", "unknown"


def parse_sarango_2022_sdrf(
    sdrf_path: str | Path,
    *,
    allow_unverified_checksum: bool = False,
) -> pd.DataFrame:
    """Parse PXD024417_community_annotated.sdrf.tsv while enforcing genotype/replicate boundaries."""
    contract = sarango_2022_source_contract()
    asset = contract["pride_inventory"]["verified_sdrf_asset"]
    path = Path(sdrf_path)
    actual_sha = _sha256_file(path)
    if not allow_unverified_checksum and actual_sha != asset["sha256"]:
        raise ValueError(
            f"SHA-256 mismatch for {path.name}: expected {asset['sha256']}, got {actual_sha}"
        )

    raw_df = pd.read_csv(path, sep="\t", dtype=str).fillna("")
    rows: list[dict[str, Any]] = []
    for idx, rec in raw_df.iterrows():
        source_name = rec.get("source name", "").strip()
        arm_label, condition_id = _arm_from_source_name(source_name)
        mhc_complex = rec.get("characteristics[mhc protein complex]", "").strip()
        mhc_class = (
            "II" if "class II" in mhc_complex else ("I" if "class I" in mhc_complex else "unknown")
        )
        antibody = rec.get("characteristics[antibody enrichment]", "").strip()
        bio_rep_raw = rec.get("characteristics[biological replicate]", "").strip()
        tech_rep_raw = rec.get("comment[technical replicate]", "").strip()

        # Primary genotype policy: Class I remains withdrawn/unspecified; Class II retains ONLY
        # paper-verified HLA-DRB1*01:02 (NEVER manufacture DRA or promote unverified SDRF typing!)
        primary_mhc = "HLA-DRB1*01:02" if mhc_class == "II" else None
        primary_genotype_status = (
            "paper_verified_drb1_only_no_manufactured_dra"
            if mhc_class == "II"
            else "withdrawn_unverified_for_hela_ciita_derivative"
        )
        sirna_bio_status = (
            "resolved_two_mock_biological_replicates"
            if arm_label == "Mock"
            else "unresolved_second_biological_sirna_experiment_in_pride"
        )

        rows.append(
            {
                "study_id": "sarango_2022_pxd024417",
                "pmid": 36215666,
                "pride_accession": "PXD024417",
                "source_asset": path.name,
                "source_sha256": actual_sha,
                "source_row": int(idx) + 2,
                "source_capture_group": source_name,
                "assay_name": rec.get("assay name", "").strip(),
                "raw_filename": rec.get("comment[data file]", "").strip(),
                "cell_line": rec.get("characteristics[cell line]", "").strip() or "HeLa-CIITA",
                "cellosaurus_id": rec.get("characteristics[cellosaurus accession]", "").strip(),
                "arm_label": arm_label,
                "condition_id": condition_id,
                "condition_knockdown_genes": "TAX1BP1" if arm_label == "siT6BP" else "",
                "condition_control": (
                    "mock"
                    if arm_label == "Mock"
                    else ("non_targeting" if arm_label == "siCTRL" else "treated")
                ),
                "mhc_class": mhc_class,
                "capture_antibody": antibody,
                "biological_replicate_index": int(bio_rep_raw) if bio_rep_raw.isdigit() else None,
                "technical_replicate_index": int(tech_rep_raw) if tech_rep_raw.isdigit() else None,
                "technical_replicate_is_independent_biological_sample": False,
                "biological_replicate_qualification": sirna_bio_status,
                "primary_verified_mhc_genotype": primary_mhc,
                "primary_genotype_status": primary_genotype_status,
                "secondary_community_sdrf_mhc_typing": rec.get(
                    "characteristics[mhc typing]", ""
                ).strip(),
                "secondary_community_sdrf_mhc_typing_method": rec.get(
                    "characteristics[mhc typing method]", ""
                ).strip(),
                "secondary_typing_promoted_to_primary": False,
                "sample_prep_batch": rec.get("comment[sample preparation batch]", "").strip(),
                "acquisition_date": rec.get("comment[acquisition date]", "").strip(),
                "modification_parameters": rec.get("comment[modification parameters]", "").strip(),
                "alkylation_reagent": rec.get("comment[alkylation reagent]", "").strip(),
            }
        )
    return pd.DataFrame(rows)


def qualify_sarango_2022_measurements(
    measurements_df: pd.DataFrame,
    sdrf_df: pd.DataFrame,
    *,
    evidence_family: str = "pan_mhc_peptide_ms_intensity",
) -> pd.DataFrame:
    """Attach PXD024417 run/replicate lineage and evidence-type guards to peptide measurements.

    Explicitly forbids mixing PLAtEAU core summaries or computational DRB1*01:02 affinity
    predictions into peptide-level MS intensity tables, and keeps missing detections as
    explicit missing (never measured negatives).
    """
    allowed = set(
        sarango_2022_source_contract()["evidence_type_boundaries"]["distinct_evidence_families"]
    )
    if evidence_family not in allowed:
        raise ValueError(
            f"Unknown Sarango 2022 evidence_family {evidence_family!r}; expected one of {sorted(allowed)}"
        )

    df = measurements_df.copy()
    if "raw_filename" not in df.columns:
        raise ValueError("measurements_df must include a 'raw_filename' column for PXD024417 join")

    merged = df.merge(
        sdrf_df[
            [
                "raw_filename",
                "source_capture_group",
                "arm_label",
                "condition_id",
                "condition_knockdown_genes",
                "condition_control",
                "mhc_class",
                "capture_antibody",
                "biological_replicate_index",
                "technical_replicate_index",
                "technical_replicate_is_independent_biological_sample",
                "biological_replicate_qualification",
                "primary_verified_mhc_genotype",
                "primary_genotype_status",
                "secondary_community_sdrf_mhc_typing",
                "secondary_typing_promoted_to_primary",
            ]
        ],
        on="raw_filename",
        how="left",
        validate="many_to_one",
    )
    merged["evidence_family"] = evidence_family
    merged["is_plateau_core_summary"] = evidence_family == "plateau_core_summary"
    merged["is_computational_affinity_estimate"] = (
        evidence_family == "computational_drb1_01_02_affinity_estimate"
    )
    merged["missing_detection_is_measured_negative"] = False
    merged["tax1bp1_pathway_activity_measured"] = False
    return merged


def summarize_sarango_2022_sdrf(sdrf_df: pd.DataFrame) -> dict[str, Any]:
    """Return a concise qualification receipt reconciling the 40 SDRF rows and 8 capture groups."""
    groups = (
        sdrf_df.groupby(["source_capture_group", "arm_label", "mhc_class", "capture_antibody"])
        .size()
        .reset_index(name="n_technical_replicates")
        .to_dict(orient="records")
    )
    return {
        "n_runs": len(sdrf_df),
        "n_source_capture_groups": int(sdrf_df["source_capture_group"].nunique()),
        "n_class_i_runs": int((sdrf_df["mhc_class"] == "I").sum()),
        "n_class_ii_runs": int((sdrf_df["mhc_class"] == "II").sum()),
        "any_secondary_typing_promoted": bool(
            sdrf_df["secondary_typing_promoted_to_primary"].any()
        ),
        "any_class_i_primary_genotype_restored": bool(
            sdrf_df.loc[sdrf_df["mhc_class"] == "I", "primary_verified_mhc_genotype"].notna().any()
        ),
        "groups": groups,
    }
