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

from pathlib import Path

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


def test_packaged_gm12878_gene_symbol_query_returns_tpm(tmp_path, monkeypatch):
    """Gene-symbol queries against GM12878 must return rows.

    Regression guard against the missing-``gene_name``-column bug
    (issue #150): if the packaged CSV stored only ``gene_id`` with
    an empty ``gene_name`` column, ``load_line_expression(gene_name="TP53")``
    would silently return zero rows and peptide-origin scoring via the
    EBV-LCL fallback anchor would be useless.
    """
    from hitlist import downloads

    monkeypatch.setattr(downloads, "_override_data_dir", tmp_path, raising=False)
    # These three house-keeping genes must be highly expressed in any
    # healthy B-LCL; the probability of all three being zero is negligible.
    for gene in ("TP53", "GAPDH", "HLA-A"):
        df = load_line_expression(line_key="GM12878", gene_name=gene)
        assert not df.empty, f"gene_name={gene!r} returned no rows — CSV gene_name column empty?"
        assert df["gene_name"].iloc[0] == gene
        assert df["tpm"].iloc[0] > 0


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


# ── Word-boundary alias matcher ────────────────────────────────────────────


@pytest.mark.parametrize(
    "label",
    [
        "JYH",  # JY prefix, extra char — must NOT match
        "MC1R",  # C1R prefix on melanocortin receptor gene name
        "FOOHeLa",  # HeLa at end of longer token
        "AHelaX",
        "xhap1",
        "THP11",  # THP-1 / THP1 false positive
    ],
)
def test_word_boundary_blocks_false_positives(label):
    a = resolve_sample_expression_anchor(label)
    assert a.expression_match_tier == 6, f"{label!r} should not match any anchor"


@pytest.mark.parametrize(
    "label,expected_tier,expected_key",
    [
        ("JY", 3, "GM12878"),  # tier-3 via EBV-LCL class anchor
        ("JY cells", 3, "GM12878"),
        ("HAP1", 5, "blood, myeloid"),  # tier-5 (placeholder source, no parent)
        ("HAP1 cells", 5, "blood, myeloid"),
        ("HeLa", 1, "HeLa"),
        ("hela cells", 1, "HeLa"),  # case-insensitive
        ("C1R", 3, "K562"),  # tier-3 via mono-allelic-host family
        ("HHC (EBV-LCL)", 3, "GM12878"),
        ("T2 cells", 3, "K562"),
    ],
)
def test_word_boundary_allows_true_positives(label, expected_tier, expected_key):
    a = resolve_sample_expression_anchor(label)
    assert a.expression_match_tier == expected_tier
    assert a.expression_key == expected_key


def test_alias_starting_with_punctuation_matches_mid_string():
    # ``.221`` alias must match inside ``721.221-...`` even though the
    # character before ``.`` is alphanumeric.  The left-boundary check is
    # skipped for punctuation-initial aliases.
    a = resolve_sample_expression_anchor("721.221-B*51:01 ERAP1 KO")
    assert a.expression_match_tier == 3  # .221 → 721.221 (placeholder) → K562 class anchor
    assert a.expression_key == "K562"
    assert a.expression_parent_key == "K562"


def test_longest_alias_wins_over_shorter_substring():
    # "HAP1 TAP1 KO" contains both the HAP1 alias (4 chars) and
    # HAP1-KO's "hap1 tap1 ko" alias (12 chars).  The longer one wins.
    # HAP1-KO has parent HAP1 (placeholder) and normal_immortalized family,
    # so it falls through to tier 5 via HAP1's lineage_tissue.
    a = resolve_sample_expression_anchor("HAP1 TAP1 KO")
    assert a.expression_match_tier == 5
    assert a.expression_key == "blood, myeloid"
    assert a.matched_alias == "hap1 tap1 ko"


def test_hek293t_derivative_routes_through_catchall_to_hek293():
    a = resolve_sample_expression_anchor("HEK293T-ACE2-TMPRSS2 (SARS-CoV-2-infected)")
    assert a.expression_match_tier == 2
    assert a.expression_parent_key == "HEK293"


# ── resolve_line_key (builder harmonization) ───────────────────────────────


def test_resolve_line_key_exact_alias():
    from hitlist.line_expression import resolve_line_key

    assert resolve_line_key("hela") == "HeLa"
    assert resolve_line_key("HELA") == "HeLa"
    assert resolve_line_key("a375") == "A375"


def test_resolve_line_key_canonical_name():
    from hitlist.line_expression import resolve_line_key

    # Canonical registry names (even though they're not in the aliases list).
    assert resolve_line_key("HeLa") == "HeLa"
    assert resolve_line_key("GM12878") == "GM12878"


def test_resolve_line_key_punctuation_stripped():
    from hitlist.line_expression import resolve_line_key

    # DepMap's StrippedCellLineName drops punctuation: "SaOS-2" → "SAOS2",
    # "THP-1" → "THP1".  The resolver matches both forms.
    assert resolve_line_key("SAOS2") == "SAOS2"
    assert resolve_line_key("saos2") == "SAOS2"
    assert resolve_line_key("THP1") == "THP1"
    assert resolve_line_key("thp1") == "THP1"


def test_resolve_line_key_missing_returns_none():
    from hitlist.line_expression import resolve_line_key

    assert resolve_line_key("completely-unknown-line") is None
    assert resolve_line_key("") is None


def test_resolve_line_key_skips_none_backend_entries():
    from hitlist.line_expression import resolve_line_key

    # "HeLa.ABC-KO" has expression_backend == "none" — it shouldn't be a
    # harmonization target (the DepMap builder would end up stamping
    # line_key="HeLa.ABC-KO" on rows that should be routed to HeLa).
    assert resolve_line_key("HeLa.ABC-KO") != "HeLa.ABC-KO"


# ── SampleExpressionAnchor — provenance fidelity ───────────────────────────


def test_tier1_carries_entry_source_ids():
    a = resolve_sample_expression_anchor("HeLa cells")
    assert "DepMap_24Q4_gene" in a.source_ids
    assert a.matched_alias == "hela"


def test_tier2_carries_parent_source_ids():
    a = resolve_sample_expression_anchor("HeLa.ABC-KO-HLA-B*51:01 (ERAP1 shRNA)")
    assert a.expression_match_tier == 2
    assert a.expression_parent_key == "HeLa"
    assert "DepMap_24Q4_gene" in a.source_ids


def test_tier3_carries_class_anchor_source_ids():
    a = resolve_sample_expression_anchor("JY (EBV-LCL)")
    assert a.expression_match_tier == 3
    assert a.source_ids == ("ENCODE_GM12878_polyA_rnaseq",)


def test_tier5_carries_hpa_source_id():
    a = resolve_sample_expression_anchor("unknown line", lineage_tissue="liver")
    assert a.expression_match_tier == 5
    assert a.source_ids == ("hpa_rna",)


def test_tier6_has_empty_source_ids():
    a = resolve_sample_expression_anchor("completely unknown widget line")
    assert a.source_ids == ()
    assert a.expression_parent_key is None


# ── compute_peptide_origin — more edge cases ───────────────────────────────


def test_peptide_origin_gene_only_no_genes_in_backend_returns_no_anchor():
    # All candidate genes absent from the expression table → no_anchor
    # (distinct from "empty candidate list").
    df = _gene_only_fixture()
    result = compute_peptide_origin(
        peptide="ABCDEFGHIJK",
        candidate_genes=["NOT_X", "NOT_Y"],
        line_expression_df=df,
    )
    assert result["peptide_origin_resolution"] == "no_anchor"


def test_peptide_origin_deterministic_tie_break():
    # Two candidate genes with identical TPM → alphabetical winner.
    df = pd.DataFrame(
        {
            "line_key": ["TEST"] * 2,
            "source_id": ["test"] * 2,
            "granularity": ["gene"] * 2,
            "gene_id": ["ENSG1", "ENSG2"],
            "gene_name": ["BETA", "ALPHA"],
            "transcript_id": ["", ""],
            "tpm": [100.0, 100.0],
            "log2_tpm": [6.66, 6.66],
        }
    )
    result = compute_peptide_origin(
        peptide="ABCDEFGHIJK",
        candidate_genes=["BETA", "ALPHA"],
        line_expression_df=df,
    )
    assert result["peptide_origin_gene"] == "ALPHA"


def test_peptide_origin_dict_input_preserves_gene_id():
    df = _gene_only_fixture()
    result = compute_peptide_origin(
        peptide="ABCDEFGHIJK",
        candidate_genes=[
            {"gene_name": "CCC", "gene_id": "ENSG_CCC"},
            {"gene_name": "AAA", "gene_id": ""},
        ],
        line_expression_df=df,
    )
    assert result["peptide_origin_gene"] == "CCC"
    assert result["peptide_origin_gene_id"] == "ENSG_CCC"


def test_peptide_origin_transcript_no_encoding_isoforms_returns_no_anchor():
    """When transcript rows exist but no isoform encodes the peptide,
    the transcript path yields nothing — fall through to gene-only which
    also has no gene rows (this fixture is transcript-only) → no_anchor.
    """
    df = _transcript_fixture()

    def lookup(gene_name):
        # No transcript's translation contains "NOTPEP".
        if gene_name == "GENE_X":
            return [("ENST-x1", "MONLY"), ("ENST-x2", "MELSE")]
        if gene_name == "GENE_Y":
            return [("ENST-y1", "MSOMETHING")]
        return []

    result = compute_peptide_origin(
        peptide="NOTPEP",
        candidate_genes=["GENE_X", "GENE_Y"],
        line_expression_df=df,
        transcript_lookup=lookup,
    )
    assert result["peptide_origin_resolution"] == "no_anchor"


def test_peptide_origin_transcript_counts_each_transcript_once():
    """A transcript that encodes the peptide at multiple positions is
    counted *once* — the transcript's TPM is the abundance of that
    molecule, not copies of the peptide per molecule.
    """
    df = _transcript_fixture()

    def lookup(gene_name):
        # GENE_X transcript x1 encodes PEPPY at TWO positions.
        if gene_name == "GENE_X":
            return [
                ("ENST-x1", "MPEPPYPEPPYEND"),  # two PEPPY sites
                ("ENST-x2", "MEMPTY"),
            ]
        return []

    result = compute_peptide_origin(
        peptide="PEPPY",
        candidate_genes=["GENE_X"],
        line_expression_df=df,
        transcript_lookup=lookup,
    )
    # Despite two sites in ENST-x1 (tpm 100), contribute only 100, not 200.
    assert result["peptide_origin_tpm"] == pytest.approx(100.0)
    assert result["peptide_origin_n_supporting_transcripts"] == 1


def test_peptide_origin_transcript_strips_version_before_lookup():
    """Proteome transcript IDs are usually versioned (ENST0000...2); parquet
    IDs are unversioned.  The compute kernel strips version before matching.
    """
    df = pd.DataFrame(
        [
            {
                "line_key": "TEST",
                "source_id": "test",
                "granularity": "transcript",
                "gene_id": "",
                "gene_name": "GENE_X",
                "transcript_id": "ENST1",
                "tpm": 100.0,
                "log2_tpm": 6.66,
            },
        ]
    )

    def lookup(gene_name):
        # Simulate pyensembl returning a versioned ID.
        return [("ENST1.12", "MPEPPY")] if gene_name == "GENE_X" else []

    result = compute_peptide_origin(
        peptide="PEPPY",
        candidate_genes=["GENE_X"],
        line_expression_df=df,
        transcript_lookup=lookup,
    )
    assert result["peptide_origin_gene"] == "GENE_X"
    assert result["peptide_origin_dominant_transcript"] == "ENST1"
    assert result["peptide_origin_tpm"] == pytest.approx(100.0)


# ── Corrupted-parquet fallback ─────────────────────────────────────────────


def test_corrupted_parquet_falls_back_to_packaged_csv(tmp_path, monkeypatch):
    """If line_expression.parquet is unreadable, the loader warns and
    returns the packaged CSV union.
    """
    from hitlist import downloads
    from hitlist import line_expression as le

    monkeypatch.setattr(downloads, "_override_data_dir", tmp_path, raising=False)
    # Write a garbage file where the parquet is expected.
    p = tmp_path / "line_expression.parquet"
    p.write_bytes(b"NOT A PARQUET FILE")

    # The loader cache from earlier tests may remember a good parquet —
    # invalidate it.
    le._load_packaged_csv.cache_clear()

    with pytest.warns(RuntimeWarning, match="Failed to read"):
        df = le.load_line_expression(line_key="GM12878")

    assert not df.empty
    assert set(df["line_key"].unique()) == {"GM12878"}


# ── generate_sample_expression_table ───────────────────────────────────────


def test_generate_sample_expression_table_shape():
    from hitlist.export import generate_sample_expression_table

    df = generate_sample_expression_table(mhc_class="I")
    assert len(df) > 0
    for col in (
        "sample_label",
        "pmid",
        "study_label",
        "mhc_class",
        "expression_backend",
        "expression_key",
        "expression_match_tier",
        "expression_parent_key",
        "expression_source_ids",
        "expression_reason",
        "expression_matched_alias",
    ):
        assert col in df.columns, f"missing column: {col}"
    # Every row must carry a tier in 1..6.
    tiers = set(df["expression_match_tier"].dropna().astype(int).unique())
    assert tiers.issubset({1, 2, 3, 4, 5, 6})
    # The packaged GM12878 entry means at least one tier-1 or tier-3 row
    # fires for EBV-LCL samples in the curated overrides.
    assert (df["expression_backend"] != "none").any()


def test_generate_sample_expression_table_semicolon_joined_source_ids():
    from hitlist.export import generate_sample_expression_table

    df = generate_sample_expression_table(mhc_class="I")
    non_empty = df[df["expression_source_ids"] != ""]
    assert not non_empty.empty
    # Source IDs are semicolon-joined strings, never Python tuples/lists.
    for val in non_empty["expression_source_ids"]:
        assert isinstance(val, str)
        assert "," not in val  # separator should be ";"


# ── _attach_peptide_origin end-to-end ──────────────────────────────────────


def _mini_observations_df() -> pd.DataFrame:
    """Three MS rows: two JY samples (same peptide, one with a candidate gene
    that has transcript rows) and one HeLa sample.
    """
    return pd.DataFrame(
        {
            "peptide": ["ABCDEFGHIJK", "ABCDEFGHIJK", "LMNOPQRSTUV"],
            "mhc_restriction": ["HLA-A*02:01", "HLA-A*02:01", "HLA-A*02:01"],
            "pmid": [12345, 12345, 67890],
            "study_label": ["JY-study", "JY-study", "HeLa-study"],
            "sample_label": ["JY (EBV-LCL)", "JY (EBV-LCL)", "HeLa cells"],
            "evidence_kind": ["ms", "ms", "ms"],
        }
    )


def test_attach_peptide_origin_populates_provenance_columns(monkeypatch):
    from hitlist import line_expression as le
    from hitlist.export import _attach_peptide_origin

    # Stub load_peptide_mappings so no real parquet is required.
    def fake_mappings(peptide=None, columns=None, **_):
        peps = list(peptide or [])
        rows = []
        for p in peps:
            if p == "ABCDEFGHIJK":
                rows.append(
                    {"peptide": p, "gene_name": "TP53", "gene_id": "ENSG_TP53", "protein_id": ""}
                )
            elif p == "LMNOPQRSTUV":
                rows.append(
                    {"peptide": p, "gene_name": "MYC", "gene_id": "ENSG_MYC", "protein_id": ""}
                )
        return pd.DataFrame(
            rows,
            columns=["peptide", "gene_name", "gene_id", "protein_id"],
        )

    monkeypatch.setattr("hitlist.mappings.load_peptide_mappings", fake_mappings)

    # Capture the un-patched loader before monkeypatching, otherwise the
    # fallback branch would recurse into the stub.
    original_load = le.load_line_expression

    def fake_load(line_key=None, **_):
        if line_key == "HeLa":
            return pd.DataFrame(
                {
                    "line_key": ["HeLa"],
                    "source_id": ["mock"],
                    "granularity": ["gene"],
                    "gene_id": ["ENSG_MYC"],
                    "gene_name": ["MYC"],
                    "transcript_id": [""],
                    "tpm": [123.4],
                    "log2_tpm": [7.0],
                }
            )
        # JY resolves to GM12878 at tier 3; load packaged union for it.
        return original_load(line_key=line_key)

    monkeypatch.setattr("hitlist.line_expression.load_line_expression", fake_load)

    df = _attach_peptide_origin(_mini_observations_df())

    # Every row has the four provenance columns populated.
    for col in (
        "expression_backend",
        "expression_key",
        "expression_match_tier",
        "expression_parent_key",
        "peptide_origin_gene",
        "peptide_origin_tpm",
        "peptide_origin_resolution",
    ):
        assert col in df.columns

    # JY samples get tier-3 GM12878.
    jy_rows = df[df["sample_label"] == "JY (EBV-LCL)"]
    assert set(jy_rows["expression_match_tier"]) == {3}
    assert set(jy_rows["expression_key"]) == {"GM12878"}
    assert set(jy_rows["expression_parent_key"]) == {"GM12878"}

    # HeLa row gets tier-1 HeLa with the stubbed MYC TPM.
    hela_rows = df[df["sample_label"] == "HeLa cells"]
    assert set(hela_rows["expression_match_tier"]) == {1}
    assert set(hela_rows["expression_key"]) == {"HeLa"}
    assert set(hela_rows["peptide_origin_gene"]) == {"MYC"}
    assert hela_rows["peptide_origin_tpm"].iloc[0] == pytest.approx(123.4)
    assert set(hela_rows["peptide_origin_resolution"]) == {"gene_only"}


def test_attach_peptide_origin_empty_frame_is_a_noop():
    from hitlist.export import _attach_peptide_origin

    empty = pd.DataFrame(columns=["peptide", "sample_label", "pmid"])
    out = _attach_peptide_origin(empty)
    # All provenance columns present, even when df is empty.
    for col in (
        "expression_backend",
        "expression_key",
        "expression_match_tier",
        "expression_parent_key",
        "peptide_origin_gene",
        "peptide_origin_tpm",
    ):
        assert col in out.columns
    assert len(out) == 0


def test_attach_peptide_origin_deduplicates_on_unique_pairs(monkeypatch):
    """Same (peptide, sample) pair repeated N times must be scored once."""
    from hitlist.export import _attach_peptide_origin

    call_count = {"n": 0}

    def fake_mappings(peptide=None, columns=None, **_):
        peps = list(peptide or [])
        return pd.DataFrame(
            [{"peptide": p, "gene_name": "MYC", "gene_id": "", "protein_id": ""} for p in peps]
        )

    monkeypatch.setattr("hitlist.mappings.load_peptide_mappings", fake_mappings)

    def fake_compute(peptide, **_):
        call_count["n"] += 1
        return {
            "peptide_origin_gene": "MYC",
            "peptide_origin_gene_id": "",
            "peptide_origin_tpm": 1.0,
            "peptide_origin_log2_tpm": 1.0,
            "peptide_origin_dominant_transcript": "",
            "peptide_origin_n_supporting_transcripts": 0,
            "peptide_origin_resolution": "gene_only",
        }

    monkeypatch.setattr("hitlist.line_expression.compute_peptide_origin", fake_compute)

    def fake_load(line_key=None, **_):
        return pd.DataFrame(
            {
                "line_key": [line_key or "HeLa"],
                "source_id": ["mock"],
                "granularity": ["gene"],
                "gene_id": [""],
                "gene_name": ["MYC"],
                "transcript_id": [""],
                "tpm": [1.0],
                "log2_tpm": [1.0],
            }
        )

    monkeypatch.setattr("hitlist.line_expression.load_line_expression", fake_load)

    df = pd.DataFrame(
        {
            "peptide": ["ABCDE"] * 100,
            "sample_label": ["HeLa cells"] * 100,
            "pmid": [1] * 100,
            "study_label": ["S"] * 100,
        }
    )
    _attach_peptide_origin(df)
    assert call_count["n"] == 1, f"scorer should run once, ran {call_count['n']}"


# ── Builder: _read_depmap_csv + harmonization ──────────────────────────────


def test_read_depmap_csv_gene_granularity(tmp_path):
    from hitlist.builder import _read_depmap_csv

    src = tmp_path / "depmap_gene.csv"
    # Wide DepMap-shape: rows = ModelID, columns = "GENE (entrez)"
    pd.DataFrame(
        {
            "ModelID": ["ACH-001", "ACH-002"],
            "TP53 (7157)": [6.0, 3.0],
            "MYC (4609)": [4.5, 5.5],
        }
    ).to_csv(src, index=False)

    long = _read_depmap_csv(src, granularity="gene")
    assert set(long["line_key"]) == {"ACH-001", "ACH-002"}
    assert set(long["gene_name"]) == {"TP53", "MYC"}
    assert set(long["granularity"]) == {"gene"}
    # log2(TPM+1) → TPM round-trips.
    tp53_ach001 = long[(long["gene_name"] == "TP53") & (long["line_key"] == "ACH-001")]
    assert tp53_ach001["log2_tpm"].iloc[0] == pytest.approx(6.0)
    assert tp53_ach001["tpm"].iloc[0] == pytest.approx(2.0**6.0 - 1.0)


def test_read_depmap_csv_transcript_granularity(tmp_path):
    from hitlist.builder import _read_depmap_csv

    src = tmp_path / "depmap_tx.csv"
    pd.DataFrame(
        {
            "ModelID": ["ACH-001"],
            "ENST00000269305.9 (TP53)": [5.0],
            "ENST00000377970.9 (MYC)": [3.0],
        }
    ).to_csv(src, index=False)

    long = _read_depmap_csv(src, granularity="transcript")
    assert set(long["transcript_id"]) == {"ENST00000269305", "ENST00000377970"}
    assert set(long["gene_name"]) == {"TP53", "MYC"}
    assert set(long["granularity"]) == {"transcript"}


def test_read_depmap_csv_empty_file_returns_empty_frame(tmp_path):
    from hitlist.builder import _read_depmap_csv

    src = tmp_path / "empty.csv"
    pd.DataFrame(columns=["ModelID"]).to_csv(src, index=False)
    out = _read_depmap_csv(src, granularity="gene")
    assert out.empty


def test_read_depmap_csv_drops_nan_values(tmp_path):
    from hitlist.builder import _read_depmap_csv

    src = tmp_path / "depmap_gene.csv"
    pd.DataFrame(
        {
            "ModelID": ["ACH-001", "ACH-002"],
            "TP53 (7157)": [6.0, None],
        }
    ).to_csv(src, index=False)

    long = _read_depmap_csv(src, granularity="gene")
    # The NaN row is dropped.
    assert len(long) == 1
    assert long["line_key"].iloc[0] == "ACH-001"


def test_load_depmap_model_lookup_reads_stripped_cell_line_name(tmp_path):
    from hitlist.builder import _load_depmap_model_lookup

    model = tmp_path / "Model.csv"
    pd.DataFrame(
        {
            "ModelID": ["ACH-001", "ACH-002", "ACH-003"],
            "StrippedCellLineName": ["HELA", "A375", ""],
            "CellLineName": ["HeLa", "A-375", "Some Line"],
        }
    ).to_csv(model, index=False)

    lookup = _load_depmap_model_lookup(model)
    assert lookup["ACH-001"] == "HELA"
    assert lookup["ACH-002"] == "A375"
    # Falls back to CellLineName when StrippedCellLineName is empty.
    assert lookup["ACH-003"] == "Some Line"


def test_load_depmap_model_lookup_missing_file_returns_empty():
    from hitlist.builder import _load_depmap_model_lookup

    assert _load_depmap_model_lookup(None) == {}
    assert _load_depmap_model_lookup(Path("/nonexistent/Model.csv")) == {}


def test_harmonize_depmap_line_keys_maps_modelids_via_registry():
    from hitlist.builder import _harmonize_depmap_line_keys

    df = pd.DataFrame(
        {
            "line_key": ["ACH-001", "ACH-002", "ACH-UNKNOWN"],
            "granularity": ["gene"] * 3,
            "gene_name": ["TP53"] * 3,
            "tpm": [1.0] * 3,
            "log2_tpm": [1.0] * 3,
        }
    )
    model_lookup = {
        "ACH-001": "HELA",
        "ACH-002": "SAOS2",
        "ACH-UNKNOWN": "TOTALLY_UNKNOWN_LINE",
    }

    out = _harmonize_depmap_line_keys(df, model_lookup=model_lookup)
    # ACH-001 → HELA → HeLa ; ACH-002 → SAOS2 → SAOS2 ; ACH-UNKNOWN drops.
    assert set(out["line_key"]) == {"HeLa", "SAOS2"}
    assert len(out) == 2


def test_harmonize_depmap_line_keys_empty_lookup_tries_alias_passthrough():
    """Without a Model.csv lookup, the harmonizer still resolves if the
    raw row IDs happen to be display names (e.g. some older DepMap releases).
    """
    from hitlist.builder import _harmonize_depmap_line_keys

    df = pd.DataFrame(
        {
            "line_key": ["HELA", "A375", "bogus-name"],
            "granularity": ["gene"] * 3,
            "gene_name": ["TP53"] * 3,
            "tpm": [1.0] * 3,
            "log2_tpm": [1.0] * 3,
        }
    )
    out = _harmonize_depmap_line_keys(df, model_lookup={})
    assert set(out["line_key"]) == {"HeLa", "A375"}


def test_harmonize_depmap_empty_input_returns_empty():
    from hitlist.builder import _harmonize_depmap_line_keys

    out = _harmonize_depmap_line_keys(pd.DataFrame())
    assert out.empty


# ── Builder: build_line_expression end-to-end ──────────────────────────────


def test_build_line_expression_with_stubbed_depmap(tmp_path, monkeypatch):
    """End-to-end: register synthetic DepMap files + Model.csv, run the
    builder, check the parquet holds registry-harmonized line_keys.
    """
    from hitlist import downloads
    from hitlist.builder import build_line_expression

    monkeypatch.setattr(downloads, "_override_data_dir", tmp_path, raising=False)

    # Synthetic DepMap gene TPM + Model.csv.
    gene_csv = tmp_path / "OmicsExpressionProteinCodingGenesTPMLogp1.csv"
    pd.DataFrame(
        {
            "ModelID": ["ACH-001", "ACH-002", "ACH-099"],
            "TP53 (7157)": [6.0, 3.0, 2.0],
            "MYC (4609)": [4.5, 5.5, 1.0],
        }
    ).to_csv(gene_csv, index=False)

    model_csv = tmp_path / "Model.csv"
    pd.DataFrame(
        {
            "ModelID": ["ACH-001", "ACH-002", "ACH-099"],
            "StrippedCellLineName": ["HELA", "SAOS2", "UNRELATED_LINE"],
            "CellLineName": ["HeLa", "SaOS-2", "Unrelated"],
        }
    ).to_csv(model_csv, index=False)

    downloads.register("depmap_rna", gene_csv)
    downloads.register("depmap_models", model_csv)

    df = build_line_expression(verbose=False)

    # DepMap rows harmonized: ACH-001→HeLa, ACH-002→SAOS2, ACH-099 dropped.
    depmap_rows = df[df["source_id"] == "DepMap_24Q4_gene"]
    assert set(depmap_rows["line_key"]) == {"HeLa", "SAOS2"}
    assert len(depmap_rows) == 4  # 2 lines x 2 genes

    # Packaged GM12878 rows still present.
    assert (df["source_id"] == "ENCODE_GM12878_polyA_rnaseq").any()
    assert "GM12878" in df["line_key"].unique()

    # parent_line_key stamped from the registry.
    hela_rows = df[df["line_key"] == "HeLa"]
    assert set(hela_rows["parent_line_key"]) == {"HeLa"}


def test_build_line_expression_no_sources_writes_empty_parquet(tmp_path, monkeypatch):
    from hitlist import downloads
    from hitlist.builder import build_line_expression

    monkeypatch.setattr(downloads, "_override_data_dir", tmp_path, raising=False)

    # Swap the packaged-union loader to return empty — simulates an
    # environment where neither packaged CSVs nor DepMap are available.
    import hitlist.line_expression as le

    monkeypatch.setattr(le, "_load_packaged_union", lambda: pd.DataFrame())

    df = build_line_expression(verbose=False)
    assert df.empty
    # Empty parquet is still written (so downstream loaders see a built state).
    p = tmp_path / "line_expression.parquet"
    assert p.exists()
    # Schema is preserved.
    read = pd.read_parquet(p)
    assert "line_key" in read.columns
    assert "backend" in read.columns


# ── Resolver: additional edge cases ────────────────────────────────────────


def test_resolver_prefers_cell_name_when_sample_label_empty():
    a = resolve_sample_expression_anchor("", cell_name="HeLa cells")
    assert a.expression_match_tier == 1
    assert a.expression_key == "HeLa"


def test_resolver_joins_sample_label_and_cell_name():
    # Both contribute to the search string.
    a = resolve_sample_expression_anchor("donor 42", cell_name="HeLa")
    assert a.expression_match_tier == 1
    assert a.expression_key == "HeLa"


def test_resolver_empty_label_falls_to_tier6():
    a = resolve_sample_expression_anchor("")
    assert a.expression_match_tier == 6


def test_resolver_reason_is_informative():
    a = resolve_sample_expression_anchor("HeLa cells")
    assert "exact line match" in a.reason
    assert a.matched_alias is not None


def test_resolver_tier2_reason_mentions_parent():
    a = resolve_sample_expression_anchor("HeLa.ABC-KO-HLA-B*51:01")
    assert "parent-line fallback" in a.reason
    assert "HeLa" in a.reason


def test_resolver_tier3_reason_mentions_class_anchor():
    a = resolve_sample_expression_anchor("JY (EBV-LCL)")
    assert "class anchor" in a.reason
    assert "ebv_lcl" in a.reason


# ── Resolver: cancer-type tier 4 additional cases ──────────────────────────


def test_tier4_backend_returning_empty_result_falls_through():
    """A backend that returns {} must not be counted as a tier-4 hit."""

    def empty_backend(cancer_type):
        return {}

    a = resolve_sample_expression_anchor(
        "random patient sample",
        cancer_type="melanoma",
        cancer_type_backend=empty_backend,
        lineage_tissue="skin",
    )
    assert a.expression_match_tier == 5  # falls through to tier 5


def test_tier4_preserves_backend_source_ids():
    def fake(cancer_type):
        return {
            "expression_backend": "pirlygenes",
            "expression_key": "BRCA",
            "source_ids": ["pirlygenes:BRCA", "tcga:BRCA"],
        }

    a = resolve_sample_expression_anchor(
        "unmapped tumor",
        cancer_type="breast",
        cancer_type_backend=fake,
    )
    assert a.expression_match_tier == 4
    assert a.source_ids == ("pirlygenes:BRCA", "tcga:BRCA")


# ── _source_stamp pmid handling ────────────────────────────────────────────


def test_source_stamp_preserves_pmid_zero():
    """pmid==0 is not a real case, but ``int(x) if x else pd.NA`` would lose
    it — this test pins down the ``is not None`` semantics in
    :func:`hitlist.builder._source_stamp`.
    """
    from hitlist.builder import _source_stamp

    s = _source_stamp({"source_id": "x", "pmid": 0})
    assert s["pmid"] == 0


def test_source_stamp_missing_pmid_is_na():
    from hitlist.builder import _source_stamp

    s = _source_stamp({"source_id": "x"})
    assert pd.isna(s["pmid"])


# ── CLI dispatch smoke tests ───────────────────────────────────────────────


def test_cli_export_samples_with_expression_anchors_calls_new_function(
    monkeypatch, tmp_path, capsys
):
    """`hitlist export samples --with-expression-anchors` must route to
    ``generate_sample_expression_table`` (not the default acquisition-metadata
    table).  Stub both candidate functions and assert only the new one fires.
    """
    import argparse

    import hitlist.cli as cli
    import hitlist.export as export

    called: dict[str, bool] = {"anchor": False, "plain": False}

    def fake_anchor(mhc_class=None, cancer_type_backend=None):
        called["anchor"] = True
        return pd.DataFrame({"sample_label": ["X"], "expression_backend": ["depmap_rna"]})

    def fake_plain(mhc_class=None, apm_only=False):
        called["plain"] = True
        return pd.DataFrame({"sample_label": ["Y"]})

    # Both symbols are re-imported into ``_export``'s local scope from
    # ``hitlist.export`` at call time, so patching the source module is what
    # the dispatcher actually sees.
    monkeypatch.setattr(export, "generate_sample_expression_table", fake_anchor)
    monkeypatch.setattr(export, "generate_ms_samples_table", fake_plain)

    args = argparse.Namespace(
        command="export",
        export_command="samples",
        mhc_class="I",
        with_expression_anchors=True,
        output=str(tmp_path / "out.csv"),
    )
    cli._export(args)
    assert called == {"anchor": True, "plain": False}
    assert (tmp_path / "out.csv").exists()


def test_cli_export_samples_default_routes_to_ms_samples_table(monkeypatch, tmp_path):
    """Without --with-expression-anchors, the dispatcher keeps the original
    ``generate_ms_samples_table`` behavior.
    """
    import argparse

    import hitlist.cli as cli
    import hitlist.export as export

    called: dict[str, bool] = {"anchor": False, "plain": False}

    def fake_anchor(mhc_class=None, cancer_type_backend=None):
        called["anchor"] = True
        return pd.DataFrame({"sample_label": ["X"]})

    def fake_plain(mhc_class=None, apm_only=False):
        called["plain"] = True
        return pd.DataFrame({"sample_label": ["Y"]})

    monkeypatch.setattr(export, "generate_sample_expression_table", fake_anchor)
    monkeypatch.setattr(export, "generate_ms_samples_table", fake_plain)

    args = argparse.Namespace(
        command="export",
        export_command="samples",
        mhc_class=None,
        with_expression_anchors=False,
        output=str(tmp_path / "out.csv"),
    )
    cli._export(args)
    assert called == {"anchor": False, "plain": True}


def test_cli_export_line_expression_passes_filters_through(monkeypatch, tmp_path):
    """`hitlist export line-expression` must forward --line-key / --gene-name /
    --granularity / --source-id to :func:`load_line_expression`.
    """
    import argparse

    import hitlist.cli as cli
    import hitlist.line_expression as le

    captured: dict = {}

    def fake_load(
        line_key=None,
        gene_name=None,
        gene_id=None,
        granularity=None,
        source_id=None,
        columns=None,
        transcript_id=None,
    ):
        captured.update(
            line_key=line_key,
            gene_name=gene_name,
            gene_id=gene_id,
            granularity=granularity,
            source_id=source_id,
        )
        return pd.DataFrame({"line_key": ["HeLa"], "gene_name": ["TP53"], "tpm": [10.0]})

    monkeypatch.setattr(le, "load_line_expression", fake_load)

    args = argparse.Namespace(
        command="export",
        export_command="line-expression",
        line_key=["HeLa"],
        gene_name=["TP53"],
        gene_id=None,
        granularity="gene",
        source_id=None,
        output=str(tmp_path / "out.csv"),
    )
    cli._export(args)
    assert captured == {
        "line_key": ["HeLa"],
        "gene_name": ["TP53"],
        "gene_id": None,
        "granularity": "gene",
        "source_id": None,
    }
    assert (tmp_path / "out.csv").exists()


def test_cli_export_training_with_peptide_origin_flag_threads_through(monkeypatch):
    """`hitlist export training --with-peptide-origin --proteome-release N`
    must forward both flags to :func:`generate_training_table`.
    """
    import argparse

    from hitlist.cli import _export_training

    captured: dict = {}

    def fake(**kwargs):
        captured.update(kwargs)
        return pd.DataFrame({"evidence_kind": ["ms"], "peptide": ["AAAAAAAAA"]})

    monkeypatch.setattr("hitlist.export.generate_training_table", fake)

    args = argparse.Namespace(
        include_evidence="ms",
        mhc_class="I",
        species=None,
        source=None,
        instrument_type=None,
        acquisition_mode=None,
        mono_allelic=None,
        min_allele_resolution=None,
        mhc_allele=None,
        gene=None,
        gene_name=None,
        gene_id=None,
        peptide=None,
        serotype=None,
        length_min=None,
        length_max=None,
        map_source_proteins=False,
        with_peptide_origin=True,
        proteome_release=114,
    )
    _export_training(args)
    assert captured["with_peptide_origin"] is True
    assert captured["proteome_release"] == 114
