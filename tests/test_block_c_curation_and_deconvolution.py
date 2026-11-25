"""Tests for #8 (non-classical HLA-E/G curation), #35/#36 (human PMID curation), and #37 (deconvolution)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

import hitlist
from hitlist.deconvolution import (
    audit_pmid_class_pools,
    build_monoallelic_motif_models,
    deconvolve_allele_pools,
    narrow_candidate_pools,
    parse_allele_pool,
)

_OVERRIDES_PATH = Path(hitlist.__file__).parent / "data" / "pmid_overrides.yaml"


def _load_overrides_by_pmid() -> dict[int, dict]:
    entries = yaml.safe_load(_OVERRIDES_PATH.read_text(encoding="utf-8"))
    return {int(e["pmid"]): e for e in entries if e.get("pmid") is not None}


def test_issue_35_all_19_large_human_pmids_curated_with_samples():
    by_pmid = _load_overrides_by_pmid()
    issue_35_pmids = [
        28649982,
        33537174,
        24366607,
        27726376,
        34405274,
        34078123,
        33123756,
        31020640,
        26929215,
        26360328,
        25505266,
        36581221,
        27397612,
        25810538,
        20112406,
        22424782,
        26768311,
        30626607,
        22544926,
    ]
    for pmid in issue_35_pmids:
        assert pmid in by_pmid, f"#35 PMID {pmid} missing from pmid_overrides.yaml"
        assert by_pmid[pmid].get("ms_samples"), f"#35 PMID {pmid} missing ms_samples"
    # Verify 22544926 author correction
    assert "Thommen 2012" in by_pmid[22544926]["study_label"]


def test_issue_8_remaining_non_classical_hla_eg_ms_pmids_curated():
    by_pmid = _load_overrides_by_pmid()
    expected_non_classical = [7584149, 29602958, 38691588, 23783831, 8805247, 10479050]
    for pmid in expected_non_classical:
        assert pmid in by_pmid, f"#8 PMID {pmid} missing from pmid_overrides.yaml"
        samples = by_pmid[pmid].get("ms_samples") or []
        assert any(str(s.get("mhc_class", "")).lower() == "non-classical" for s in samples), (
            f"#8 PMID {pmid} lacks non-classical ms_samples arm"
        )


def test_issue_36_medium_human_pmids_batch_curated():
    by_pmid = _load_overrides_by_pmid()
    batch_pmids = [
        25548167,
        31556206,
        22645359,
        32162841,
        27543614,
        26758079,
        25802875,
        24397554,
        38117652,
        25469497,
        28860189,
        31869419,
        21872626,
    ]
    for pmid in batch_pmids:
        assert pmid in by_pmid, f"#36 PMID {pmid} missing from pmid_overrides.yaml"
        assert by_pmid[pmid].get("ms_samples"), f"#36 PMID {pmid} missing ms_samples"


def test_deconvolution_public_exports_and_pool_parsing():
    for symbol in (
        "parse_allele_pool",
        "audit_pmid_class_pools",
        "narrow_candidate_pools",
        "build_monoallelic_motif_models",
        "deconvolve_allele_pools",
    ):
        assert hasattr(hitlist, symbol), f"Missing public export: {symbol}"
    assert parse_allele_pool("HLA-B*07:02; HLA-A*02:01, HLA-A*02:01") == (
        "HLA-A*02:01",
        "HLA-B*07:02",
    )
    assert parse_allele_pool("") == ()
    assert parse_allele_pool(None) == ()


def test_audit_and_narrow_pmid_class_pools_option_b():
    obs = pd.DataFrame(
        {
            "pmid": [1001, 1001, 1001, 2002, 2002],
            "peptide": ["SLYNTVATL", "SLYNTVATL", "GILGFVFTL", "RPHERNGFTVL", "NLVPMVATV"],
            "mhc_class": ["I", "I", "I", "I", "I"],
            "mhc_allele_set": [
                "HLA-A*02:01;HLA-A*03:01;HLA-B*07:02;HLA-B*44:02",
                "HLA-A*02:01;HLA-B*07:02",
                "HLA-A*02:01;HLA-A*03:01;HLA-B*07:02;HLA-B*44:02",
                "HLA-A*02:01",
                "HLA-A*01:01;HLA-B*08:01",
            ],
            "mhc_allele_provenance": [
                "pmid_class_pool",
                "sample_allele_match",
                "pmid_class_pool",
                "pmid_class_pool",
                "pmid_class_pool",
            ],
        }
    )

    audit = audit_pmid_class_pools(obs)
    assert audit["n_total_observations"] == 5
    assert audit["n_pool_observations"] == 4
    assert audit["n_pool_pmids"] == 2
    assert audit["n_singleton_pool_observations"] == 1

    narrowed, summary = narrow_candidate_pools(obs)
    # Native mhc_allele_set must never be mutated
    assert narrowed["mhc_allele_set"].tolist() == obs["mhc_allele_set"].tolist()
    # Row 0 (pmid_class_pool) narrows via row 1 (sample_allele_match) in same PMID 1001
    assert narrowed.loc[0, "narrowed_allele_pool"] == "HLA-A*02:01;HLA-B*07:02"
    assert narrowed.loc[0, "n_narrowed_alleles"] == 2
    assert narrowed.loc[0, "pool_narrowing_basis"] == "within_pmid_sample_intersection"
    # Row 2 has no corroborating sample row -> stays unchanged
    assert narrowed.loc[2, "pool_narrowing_basis"] == "unchanged_pool"
    assert narrowed.loc[2, "n_narrowed_alleles"] == 4
    # Row 3 is singleton pool
    assert narrowed.loc[3, "pool_narrowing_basis"] == "singleton_pool"
    assert narrowed.loc[3, "n_narrowed_alleles"] == 1
    assert summary["n_narrowed_rows"] == 1
    assert summary["n_singleton_pool_rows"] == 1


def test_license_free_monoallelic_motif_deconvolution_options_a_and_c():
    # Synthetic mono-allelic 9-mers with canonical HLA-A*02:01 (P2=L/M, P9=V/L)
    # vs HLA-B*07:02 (P2=P, P9=L/F) anchors.
    aa_mid = "ACDEFGHIKLMNPQRSTVWY"
    a02_unique = [
        f"A{'L' if i % 3 else 'M'}{aa_mid[i % 20]}{aa_mid[(i * 3) % 20]}FGHI{'V' if i % 2 else 'L'}"
        for i in range(30)
    ]
    b07_unique = [
        f"R{'P'}{aa_mid[(i + 5) % 20]}{aa_mid[(i * 7) % 20]}KMNQ{'L' if i % 2 else 'F'}"
        for i in range(30)
    ]
    mono = pd.DataFrame(
        {
            "peptide": a02_unique + b07_unique,
            "mhc_restriction": ["HLA-A*02:01"] * len(a02_unique)
            + ["HLA-B*07:02"] * len(b07_unique),
        }
    )
    models = build_monoallelic_motif_models(mono, lengths=(9,), min_peptides=20)
    assert models["n_models"] == 2
    assert models["n_alleles"] == 2

    pool_df = pd.DataFrame(
        {
            "peptide": [
                "ALACFGHLV",  # strong A*02:01 anchor (P2=L, P9=V)
                "RPKDFMNQL",  # strong B*07:02 anchor (P2=P, P9=L)
                "ALACFGHLV",  # incomplete pool (contains unmodeled HLA-C*07:01 -> must abstain #574)
                "ALACFGHL",  # length 8 not in motif models -> unsupported_length
                "ALACFGHLV",  # singleton pool
            ],
            "mhc_allele_set": [
                "HLA-A*02:01;HLA-B*07:02",
                "HLA-A*02:01;HLA-B*07:02",
                "HLA-A*02:01;HLA-B*07:02;HLA-C*07:01",
                "HLA-A*02:01;HLA-B*07:02",
                "HLA-A*02:01",
            ],
        }
    )
    out, summary = deconvolve_allele_pools(
        pool_df,
        models,
        min_log_odds_margin=1.5,
        require_complete_pool=True,
    )
    # Native pool column stays untouched
    assert out["mhc_allele_set"].tolist() == pool_df["mhc_allele_set"].tolist()
    assert out.loc[0, "deconvolution_status"] == "assigned"
    assert out.loc[0, "predicted_allele_rank1"] == "HLA-A*02:01"
    assert out.loc[0, "motif_score_margin"] > 1.5

    assert out.loc[1, "deconvolution_status"] == "assigned"
    assert out.loc[1, "predicted_allele_rank1"] == "HLA-B*07:02"
    assert out.loc[1, "motif_score_margin"] > 1.5

    # #574 complete candidate space guard
    assert out.loc[2, "deconvolution_status"] == "incomplete_motif_coverage"
    assert out.loc[2, "predicted_allele_rank1"] == ""

    assert out.loc[3, "deconvolution_status"] == "unsupported_length"
    assert out.loc[4, "deconvolution_status"] == "singleton_pool"
    assert out.loc[4, "predicted_allele_rank1"] == "HLA-A*02:01"
    assert summary["n_assigned_rows"] == 3
    assert summary["n_abstained_rows"] == 2
