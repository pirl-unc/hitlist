"""Structured biophysical stability & kinetic assay covariates and normalization audit (#946).

Extracts explicit biophysical covariates on pMHC kinetic stability (``half_life``),
thermal stability (``thermal_stability`` / ``Tm`` / ``Td50``), and kinetic rate
(``off_rate`` / ``koff``, ``on_rate`` / ``kon``) rows so cellular decay assays,
purified 37C SPA assays, DSF thermal shift assays, SPR/BLI kinetics, and
HLA-DM/TAPBPR-catalyzed Class II/I editing assays are never conflated.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

import pandas as pd

ASSAY_PROTOCOLS: tuple[str, ...] = (
    "spa_37c",
    "dsf_tm",
    "cd_tm",
    "bfa_cell_surface_decay",
    "fluorescent_peptide_exchange",
    "spr_kinetics",
    "bli_kinetics",
    "t2_rmas_stabilization",
    "radiolabeled_dissociation",
    "other_stability_protocol",
)

CATALYST_CONTEXTS: tuple[str, ...] = (
    "none",
    "hla_dm",
    "hla_dm_do",
    "tapbpr",
    "tapasin",
    "unknown",
)

STABILITY_COVARIATE_COLUMNS: tuple[str, ...] = (
    "assay_protocol",
    "assay_temperature_c",
    "assay_ph",
    "catalyst_context",
    "assay_floor_value",
    "assay_ceiling_value",
)

_TEMP_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:°\s*C|deg\s*C|degrees?\s*C|\bC\b)", re.IGNORECASE)
_PH_RE = re.compile(r"\bpH\s*[=:]?\s*(\d+(?:\.\d+)?)\b", re.IGNORECASE)


@dataclass(frozen=True)
class BiophysicalCovariates:
    """Extracted biophysical assay covariates for a stability/kinetic row."""

    assay_protocol: str
    assay_temperature_c: float | None
    assay_ph: float | None
    catalyst_context: str
    assay_floor_value: float | None
    assay_ceiling_value: float | None


def _extract_temperature_c(text: str, protocol: str) -> float | None:
    match = _TEMP_RE.search(text)
    if match:
        val = float(match.group(1))
        if 0.0 <= val <= 100.0:
            return val
    if protocol == "spa_37c":
        return 37.0
    if protocol == "bfa_cell_surface_decay":
        return 37.0
    return None


def _extract_ph(text: str) -> float | None:
    match = _PH_RE.search(text)
    if match:
        val = float(match.group(1))
        if 2.0 <= val <= 11.0:
            return val
    return None


def _extract_catalyst_context(text: str, *, mhc_class: str = "") -> str:
    lowered = text.casefold()
    has_dm = bool(re.search(r"\b(hla-?dm|h2-?dm|\bdm\b)\b", lowered))
    has_do = bool(re.search(r"\b(hla-?do|h2-?do|\bdo\b)\b", lowered))
    without_dm = bool(
        re.search(
            r"(?:\b(?:without|no|minus)\s+|(?:^|[\s,(])-\s*)(?:hla-?dm|h2-?dm|\bdm\b)",
            lowered,
        )
    )
    if without_dm:
        return "none"
    if has_dm and has_do:
        return "hla_dm_do"
    if has_dm:
        return "hla_dm"
    if "tapbpr" in lowered or "tapbpl" in lowered:
        return "tapbpr"
    if "tapasin" in lowered or "tapbp" in lowered:
        return "tapasin"
    if mhc_class.strip().upper() == "I" and any(
        tok in lowered for tok in ("scintillation proximity", "spa", "dsf", "circular dichroism")
    ):
        return "none"
    return "unknown"


def _extract_protocol(
    *,
    assay_method: str,
    response_measured: str,
    assay_comments: str,
    source: str,
) -> str:
    combined = " ".join(s for s in (assay_method, response_measured, assay_comments, source) if s)
    lowered = combined.casefold()
    if (
        "scintillation proximity" in lowered
        or re.search(r"\bspa\b", lowered)
        or "netmhcstabpan" in lowered
    ):
        return "spa_37c"
    if any(
        tok in lowered
        for tok in (
            "differential scanning fluorimetry",
            "nanodsf",
            "dsf",
            "sypro orange",
            "thermal shift",
        )
    ):
        return "dsf_tm"
    if "circular dichroism" in lowered or re.search(r"\bcd\b.*(melt|tm|thermal)", lowered):
        return "cd_tm"
    if "brefeldin" in lowered or re.search(r"\bbfa\b", lowered) or "cell surface decay" in lowered:
        return "bfa_cell_surface_decay"
    if (
        "surface plasmon resonance" in lowered
        or re.search(r"\bspr\b", lowered)
        or "biacore" in lowered
    ):
        return "spr_kinetics"
    if (
        "biolayer interferometry" in lowered
        or "bio-layer interferometry" in lowered
        or re.search(r"\bbli\b", lowered)
    ):
        return "bli_kinetics"
    if any(
        tok in lowered
        for tok in ("rma-s", "rmas", "t2 stabilization", "t2 binding", "cellular mhc")
    ):
        return "t2_rmas_stabilization"
    if "fluorescence" in lowered or "fluorescent" in lowered or "peptide exchange" in lowered:
        return "fluorescent_peptide_exchange"
    if "radioactivity" in lowered or "radiolabeled" in lowered or "125i" in lowered:
        return "radiolabeled_dissociation"
    if "50% dissociation temperature" in lowered or "melting temperature" in lowered:
        return "dsf_tm"
    return "other_stability_protocol"


def extract_biophysical_covariates(
    *,
    assay_method: str = "",
    response_measured: str = "",
    assay_comments: str = "",
    source: str = "",
    mhc_class: str = "",
) -> BiophysicalCovariates:
    """Extract structured biophysical covariates for a single stability/kinetic row."""
    protocol = _extract_protocol(
        assay_method=assay_method,
        response_measured=response_measured,
        assay_comments=assay_comments,
        source=source,
    )
    combined = " ".join(s for s in (assay_method, response_measured, assay_comments, source) if s)
    temp_c = _extract_temperature_c(combined, protocol)
    ph = _extract_ph(combined)
    catalyst = _extract_catalyst_context(combined, mhc_class=mhc_class)
    # Standard NetMHCstabpan / 37C SPA assays have a 0.0h nondetection floor and
    # ~48h-100h practical assay window; only record explicit floor for SPA.
    floor_val = 0.0 if protocol == "spa_37c" else None
    return BiophysicalCovariates(
        assay_protocol=protocol,
        assay_temperature_c=temp_c,
        assay_ph=ph,
        catalyst_context=catalyst,
        assay_floor_value=floor_val,
        assay_ceiling_value=None,
    )


def qualify_stability_and_kinetics(df: pd.DataFrame) -> pd.DataFrame:
    """Qualify stability/kinetic rows with standardized units and biophysical covariates (#946).

    Calls :func:`hitlist.model_training.qualify_binding_measurements` for unit
    conversion (``half_life`` -> ``min``, ``thermal_stability`` -> ``celsius``,
    ``off_rate`` -> ``s^-1``, ``on_rate`` -> ``M^-1 s^-1``) and censoring
    classification, then attaches ``assay_protocol``, ``assay_temperature_c``,
    ``assay_ph``, ``catalyst_context``, ``assay_floor_value``, and
    ``assay_ceiling_value``.
    """
    from .model_training import qualify_binding_measurements

    out = qualify_binding_measurements(df)
    if out.empty:
        out["assay_protocol"] = pd.Series(dtype="string")
        out["assay_temperature_c"] = pd.Series(dtype="float64")
        out["assay_ph"] = pd.Series(dtype="float64")
        out["catalyst_context"] = pd.Series(dtype="string")
        out["assay_floor_value"] = pd.Series(dtype="float64")
        out["assay_ceiling_value"] = pd.Series(dtype="float64")
        return out

    def _col(name: str) -> list[str]:
        if name not in out.columns:
            return [""] * len(out)
        return out[name].astype("string").fillna("").tolist()

    keys = list(
        zip(
            _col("assay_method"),
            _col("response_measured"),
            _col("assay_comments"),
            _col("source"),
            _col("mhc_class"),
        )
    )
    cache_map: dict[tuple, BiophysicalCovariates] = {}
    for key in set(keys):
        cache_map[key] = extract_biophysical_covariates(
            assay_method=key[0],
            response_measured=key[1],
            assay_comments=key[2],
            source=key[3],
            mhc_class=key[4],
        )
    covs = [cache_map[k] for k in keys]
    out["assay_protocol"] = pd.Series(
        [c.assay_protocol for c in covs], index=out.index, dtype="string"
    )
    out["assay_temperature_c"] = pd.Series(
        [c.assay_temperature_c if c.assay_temperature_c is not None else math.nan for c in covs],
        index=out.index,
        dtype="float64",
    )
    out["assay_ph"] = pd.Series(
        [c.assay_ph if c.assay_ph is not None else math.nan for c in covs],
        index=out.index,
        dtype="float64",
    )
    out["catalyst_context"] = pd.Series(
        [c.catalyst_context for c in covs], index=out.index, dtype="string"
    )
    out["assay_floor_value"] = pd.Series(
        [c.assay_floor_value if c.assay_floor_value is not None else math.nan for c in covs],
        index=out.index,
        dtype="float64",
    )
    out["assay_ceiling_value"] = pd.Series(
        [c.assay_ceiling_value if c.assay_ceiling_value is not None else math.nan for c in covs],
        index=out.index,
        dtype="float64",
    )
    return out


def audit_stability_normalization(df: pd.DataFrame) -> dict:
    """Audit unit conversion, censoring, protocol, species, and covariate coverage (#946)."""
    qualified = (
        df
        if {"binding_endpoint_family", "assay_protocol", "catalyst_context"} <= set(df.columns)
        else qualify_stability_and_kinetics(df)
    )
    stab_mask = qualified["binding_endpoint_family"].isin(
        ["half_life", "thermal_stability", "off_rate", "on_rate"]
    )
    sub = qualified.loc[stab_mask]
    if sub.empty:
        return {
            "n_stability_kinetic_rows": 0,
            "n_zero_half_life_rows": 0,
            "n_rows_with_temperature": 0,
            "n_rows_with_ph": 0,
            "by_endpoint_family": {},
            "by_assay_protocol": {},
            "by_catalyst_context": {},
            "by_unit_conversion_status": {},
            "by_censoring_status": {},
            "by_mhc_class": {},
            "by_mhc_species": {},
        }

    zero_hl = (
        sub["binding_endpoint_family"].eq("half_life")
        & sub["standardized_value"].notna()
        & sub["standardized_value"].le(0.0)
    )
    return {
        "n_stability_kinetic_rows": len(sub),
        "n_zero_half_life_rows": int(zero_hl.sum()),
        "n_rows_with_temperature": int(sub["assay_temperature_c"].notna().sum()),
        "n_rows_with_ph": int(sub["assay_ph"].notna().sum()),
        "by_endpoint_family": dict(
            sorted(Counter(sub["binding_endpoint_family"].astype("string").tolist()).items())
        ),
        "by_assay_protocol": dict(
            sorted(Counter(sub["assay_protocol"].astype("string").tolist()).items())
        ),
        "by_catalyst_context": dict(
            sorted(Counter(sub["catalyst_context"].astype("string").tolist()).items())
        ),
        "by_unit_conversion_status": dict(
            sorted(Counter(sub["unit_conversion_status"].astype("string").tolist()).items())
        ),
        "by_censoring_status": dict(
            sorted(Counter(sub["censoring_status"].astype("string").tolist()).items())
        ),
        "by_mhc_class": dict(
            sorted(
                Counter(
                    sub["mhc_class"].astype("string").fillna("unknown").tolist()
                    if "mhc_class" in sub.columns
                    else ["unknown"] * len(sub)
                ).items()
            )
        ),
        "by_mhc_species": dict(
            sorted(
                Counter(
                    sub["mhc_species"].astype("string").fillna("unknown").tolist()
                    if "mhc_species" in sub.columns
                    else ["unknown"] * len(sub)
                ).items()
            )
        ),
    }
