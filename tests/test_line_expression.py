"""Tests for hitlist.line_expression — registry, resolver, peptide origin.

Covers:
- YAML registry parses and contains the issue-#140 starter list.
- The 6-tier fallback resolver fires the right tier for each class of
  sample label (exact / parent / family / tissue / unknown), and honours
  the placeholder-source guard (entries whose only source_ids are
  placeholders must NOT claim tier 1).
- Packaged-CSV loader returns rows before any build has run.
- Peptide-origin scoring — gene-only argmax and transcript-isoform-aware
  sum with hand-crafted fixtures (no pyensembl required).
"""

from __future__ import annotations

import pandas as pd
import pytest

from hitlist.line_expression import (
    SampleExpressionAnchor,
    compute_peptide_origin,
    load_line_expression,
    load_line_expression_anchors,
    load_line_expression_sources,
    resolve_sample_expression_anchor,
)

# ── Registry ───────────────────────────────────────────────────────────────


_STARTER_LINES = {
    "C1R",
    "721.221",
    "T2",
    "JY",
    "HHC",
    "GR",
    "CD165",
    "RA957",
    "HAP1",
    "HeLa",
    "HEK293",
    "THP-1",
    "SaOS-2",
    "A375",
    "K562",
    "GM12878",
}


def test_registry_covers_starter_lines():
    names = {str(e.get("name")) for e in load_line_expression_anchors()}
    missing = _STARTER_LINES - names
    assert not missing, f"registry missing lines: {missing}"


def test_registry_entries_have_required_fields():
    for entry in load_line_expression_anchors():
        assert entry.get("name")
        assert "aliases" in entry
        assert "line_family" in entry
        assert "expression_backend" in entry


def test_sources_yaml_parses():
    sources = load_line_expression_sources()
    ids = {s.get("source_id") for s in sources}
    assert "ENCODE_GM12878_polyA_rnaseq" in ids
    assert "DepMap_24Q4_gene" in ids
    # Every source has a build_status the builder understands.
    for s in sources:
        assert s.get("build_status") in {"packaged", "downloadable", "placeholder"}


# ── Resolver: tier 1 ───────────────────────────────────────────────────────


def test_tier1_hela():
    a = resolve_sample_expression_anchor("HeLa cells")
    assert a.expression_match_tier == 1
    assert a.expression_key == "HeLa"
    assert a.expression_backend == "depmap_rna"
    assert a.expression_parent_key is None


def test_tier1_gm12878_direct():
    a = resolve_sample_expression_anchor("GM12878 cell line")
    assert a.expression_match_tier == 1
    assert a.expression_key == "GM12878"
    assert a.expression_backend == "encode_rnaseq"


def test_tier1_saos2_typo_alias_raos():
    # RaOS is curated as a SaOS-2 alias per user note on issue #140.
    a = resolve_sample_expression_anchor("RaOS cells")
    assert a.expression_match_tier == 1
    assert a.expression_key == "SAOS2"


# ── Resolver: tier 2 (parent-line fallback) ────────────────────────────────


def test_tier2_hela_abc_ko_resolves_to_parent_hela():
    a = resolve_sample_expression_anchor("HeLa.ABC-KO-HLA-B*51:01 (ERAP1 shRNA)")
    assert a.expression_match_tier == 2
    assert a.expression_key == "HeLa"
    assert a.expression_parent_key == "HeLa"


def test_tier2_hek293t_ace2_resolves_to_parent_hek293():
    a = resolve_sample_expression_anchor("HEK293T-ACE2-TMPRSS2 (SARS-CoV-2-infected)")
    assert a.expression_match_tier == 2
    assert a.expression_key == "HEK293"
    assert a.expression_parent_key == "HEK293"


# ── Resolver: tier 3 (class anchor) ────────────────────────────────────────


def test_tier3_ebv_lcl_jy_falls_through_to_gm12878():
    # JY references a placeholder Pearson source — tier 1 must NOT fire
    # because no CSV ships yet.  Class-anchor = GM12878.
    a = resolve_sample_expression_anchor("JY (EBV-LCL)")
    assert a.expression_match_tier == 3
    assert a.expression_key == "GM12878"
    assert a.expression_backend == "encode_rnaseq"
    assert a.expression_parent_key == "GM12878"


def test_tier3_ebv_lcl_gr_goes_to_gm12878():
    a = resolve_sample_expression_anchor("GR (EBV-LCL)")
    assert a.expression_match_tier == 3
    assert a.expression_key == "GM12878"


def test_tier3_mono_allelic_host_t2_falls_to_k562():
    a = resolve_sample_expression_anchor("T2 cells")
    assert a.expression_match_tier == 3
    assert a.expression_key == "K562"


def test_tier3_c1r_transfectant_falls_to_k562_via_family():
    # C1R has a placeholder source, so C1R tier-1 doesn't fire; its parent
    # is itself (no parent line defined that has data); family =
    # mono_allelic_host → K562 class anchor.
    a = resolve_sample_expression_anchor("C1R-HLA-B*27:02")
    assert a.expression_match_tier == 3
    assert a.expression_key == "K562"


# ── Resolver: tier 5 (tissue surrogate) + tier 6 (no anchor) ───────────────


def test_tier5_caller_supplied_tissue_for_unknown_label():
    a = resolve_sample_expression_anchor("Some unmapped widget sample", lineage_tissue="liver")
    assert a.expression_match_tier == 5
    assert a.expression_backend == "hpa_tissue"
    assert a.expression_key == "liver"


def test_tier5_hap1_inherits_tissue_from_registry():
    # HAP1 sources are all placeholders; line_family == normal_immortalized
    # has no class anchor; registry lineage_tissue is 'blood, myeloid'.
    a = resolve_sample_expression_anchor("HAP1 wildtype")
    assert a.expression_match_tier == 5
    assert a.expression_backend == "hpa_tissue"
    assert a.expression_key == "blood, myeloid"


def test_tier6_truly_unknown():
    a = resolve_sample_expression_anchor("completely unknown widget line")
    assert a.expression_match_tier == 6
    assert a.expression_backend == "none"
    assert a.expression_key == ""


# ── Resolver: tier 4 (cancer-type surrogate, caller-supplied backend) ──────


def test_tier4_cancer_type_backend_invoked_for_known_tumor_line():
    # SaOS-2 has tier-1 data via DepMap → tier 4 won't fire for it.  Use a
    # sample whose exact-line and tier-3 don't apply: an unmapped skin
    # melanoma label with an explicit cancer_type and a tier-4 backend.
    def fake_backend(cancer_type):
        assert cancer_type == "cutaneous melanoma"
        return {
            "expression_backend": "pirlygenes_cohort",
            "expression_key": "SKCM_TCGA",
            "source_ids": ["pirlygenes:SKCM"],
        }

    a = resolve_sample_expression_anchor(
        "unmapped melanoma patient sample 42",
        cancer_type="cutaneous melanoma",
        cancer_type_backend=fake_backend,
    )
    assert a.expression_match_tier == 4
    assert a.expression_backend == "pirlygenes_cohort"
    assert a.expression_key == "SKCM_TCGA"


def test_tier4_backend_exception_is_swallowed():
    def bad_backend(cancer_type):
        raise RuntimeError("boom")

    a = resolve_sample_expression_anchor(
        "unmapped widget",
        cancer_type="lymphoma",
        cancer_type_backend=bad_backend,
        lineage_tissue="lymph node",
    )
    assert a.expression_match_tier == 5  # falls through to tier 5


# ── Packaged-CSV loader ────────────────────────────────────────────────────


def test_packaged_gm12878_csv_loads_without_build(tmp_path, monkeypatch):
    # Point the hitlist data dir at an empty tmp dir so parquet load fails
    # and the loader MUST use the packaged CSV path.
    from hitlist import downloads

    monkeypatch.setattr(downloads, "_override_data_dir", tmp_path, raising=False)
    df = load_line_expression(line_key="GM12878")
    assert not df.empty
    assert set(df["line_key"].unique()) == {"GM12878"}
    # Sanity: ~20k protein-coding genes with TPM > 0 for GM12878.
    assert len(df) > 10_000
    # Every row carries the expected provenance columns.
    for col in ("line_key", "source_id", "granularity", "tpm", "log2_tpm"):
        assert col in df.columns


# ── Peptide-origin ─────────────────────────────────────────────────────────


def _gene_only_fixture() -> pd.DataFrame:
    """Fixture: three genes expressed at distinct TPM in one sample."""
    return pd.DataFrame(
        {
            "line_key": ["TEST"] * 3,
            "source_id": ["test"] * 3,
            "granularity": ["gene"] * 3,
            "gene_id": ["ENSG1", "ENSG2", "ENSG3"],
            "gene_name": ["AAA", "BBB", "CCC"],
            "transcript_id": ["", "", ""],
            "tpm": [50.0, 5.0, 500.0],
            "log2_tpm": [5.67, 2.58, 8.97],
        }
    )


def test_peptide_origin_gene_only_picks_highest_tpm():
    df = _gene_only_fixture()
    result = compute_peptide_origin(
        peptide="ABCDEFGHIJK",
        candidate_genes=["AAA", "BBB", "CCC"],
        line_expression_df=df,
    )
    assert result["peptide_origin_gene"] == "CCC"
    assert result["peptide_origin_tpm"] == pytest.approx(500.0)
    assert result["peptide_origin_resolution"] == "gene_only"
    assert result["peptide_origin_n_supporting_transcripts"] == 0


def test_peptide_origin_gene_only_ignores_absent_genes():
    df = _gene_only_fixture()
    result = compute_peptide_origin(
        peptide="ABCDEFGHIJK",
        candidate_genes=["AAA", "ZZZ_NOT_IN_BACKEND"],
        line_expression_df=df,
    )
    assert result["peptide_origin_gene"] == "AAA"
    assert result["peptide_origin_tpm"] == pytest.approx(50.0)


def test_peptide_origin_empty_inputs_return_no_anchor():
    result = compute_peptide_origin(
        peptide="ABCDEFGHIJK",
        candidate_genes=[],
        line_expression_df=_gene_only_fixture(),
    )
    assert result["peptide_origin_resolution"] == "no_anchor"
    assert result["peptide_origin_gene"] == ""


def _transcript_fixture() -> pd.DataFrame:
    """Two genes with per-transcript TPM:

    - GENE_X: ENST-x1 (tpm 100), ENST-x2 (tpm 20).  Peptide PEPPY is encoded
      by ENST-x1 only.  Per-gene TPM = 100 (sum of encoding transcripts).
    - GENE_Y: ENST-y1 (tpm 300).  Peptide PEPPY NOT in ENST-y1's protein.
      Per-gene TPM = 0 (no encoding transcript).  Should lose to GENE_X
      despite raw transcript TPM being higher, because the peptide is
      spliced out.
    """
    rows = [
        ("GENE_X", "ENST-x1", 100.0),
        ("GENE_X", "ENST-x2", 20.0),
        ("GENE_Y", "ENST-y1", 300.0),
    ]
    return pd.DataFrame(
        [
            {
                "line_key": "TEST",
                "source_id": "test",
                "granularity": "transcript",
                "gene_id": "",
                "gene_name": g,
                "transcript_id": t,
                "tpm": tpm,
                "log2_tpm": 0.0,
            }
            for g, t, tpm in rows
        ]
    )


def test_peptide_origin_transcript_isoform_sum_filters_non_encoding_isoforms():
    """GENE_Y has higher raw TPM but splices out the peptide → GENE_X wins."""
    df = _transcript_fixture()

    transcript_protein = {
        "GENE_X": [
            ("ENST-x1", "MSTARTPEPPYENDPROTEIN"),  # contains PEPPY
            ("ENST-x2", "MSTARTNOPEPHEREENDPROTEIN"),  # no PEPPY
        ],
        "GENE_Y": [
            ("ENST-y1", "MSTARTNOPEPHEREENDPROTEIN"),  # no PEPPY
        ],
    }

    def lookup(gene_name):
        return transcript_protein.get(gene_name, [])

    result = compute_peptide_origin(
        peptide="PEPPY",
        candidate_genes=["GENE_X", "GENE_Y"],
        line_expression_df=df,
        transcript_lookup=lookup,
    )
    assert result["peptide_origin_gene"] == "GENE_X"
    assert result["peptide_origin_tpm"] == pytest.approx(100.0)
    assert result["peptide_origin_resolution"] == "transcript_isoform_sum"
    assert result["peptide_origin_dominant_transcript"] == "ENST-x1"
    assert result["peptide_origin_n_supporting_transcripts"] == 1


def test_peptide_origin_transcript_isoform_sum_handles_multiple_encoding_transcripts():
    df = _transcript_fixture()

    # Now both GENE_X transcripts encode the peptide — summed TPM = 120.
    transcript_protein = {
        "GENE_X": [
            ("ENST-x1", "MPEPPYX"),
            ("ENST-x2", "MSTARTPEPPYEND"),
        ],
        "GENE_Y": [
            ("ENST-y1", "MSTARTPEPPY"),  # single encoding transcript, tpm 300
        ],
    }

    def lookup(gene_name):
        return transcript_protein.get(gene_name, [])

    result = compute_peptide_origin(
        peptide="PEPPY",
        candidate_genes=["GENE_X", "GENE_Y"],
        line_expression_df=df,
        transcript_lookup=lookup,
    )
    # GENE_Y wins at 300 (single encoding transcript), GENE_X sum is 120.
    assert result["peptide_origin_gene"] == "GENE_Y"
    assert result["peptide_origin_tpm"] == pytest.approx(300.0)
    assert result["peptide_origin_dominant_transcript"] == "ENST-y1"
    assert result["peptide_origin_n_supporting_transcripts"] == 1


def test_peptide_origin_transcript_fallback_to_gene_only_when_no_lookup():
    """No transcript_lookup → gene-only path even if transcript rows are present."""
    df = pd.concat([_gene_only_fixture(), _transcript_fixture()], ignore_index=True)
    result = compute_peptide_origin(
        peptide="PEPPY",
        candidate_genes=["AAA", "BBB", "CCC"],
        line_expression_df=df,
        transcript_lookup=None,
    )
    assert result["peptide_origin_resolution"] == "gene_only"
    assert result["peptide_origin_gene"] == "CCC"


# ── SampleExpressionAnchor dataclass ───────────────────────────────────────


def test_sample_expression_anchor_is_immutable_dataclass():
    from dataclasses import FrozenInstanceError

    a = SampleExpressionAnchor(
        expression_backend="depmap_rna",
        expression_key="HeLa",
        expression_match_tier=1,
    )
    with pytest.raises(FrozenInstanceError):
        a.expression_backend = "mutated"  # type: ignore[misc]
