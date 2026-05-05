import pytest

from hitlist.curation import (
    ALLELE_RESOLUTION_ORDER,
    allele_resolution_rank,
    allele_to_all_serotypes,
    allele_to_serotype,
    best_4digit_for_serotype,
    classify_allele_resolution,
    classify_mhc_species,
    classify_ms_row,
    detect_monoallelic,
    expand_allele_bag,
    is_binding_assay,
    is_cancer_specific,
    load_monoallelic_lines,
    load_pmid_overrides,
    load_tissue_categories,
    normalize_allele,
    normalize_species,
    serotype_to_alleles,
)


def test_load_pmid_overrides():
    overrides = load_pmid_overrides()
    assert len(overrides) >= 10
    assert 33858848 in overrides
    assert overrides[33858848]["override"] == "healthy"


def test_load_tissue_categories():
    cats = load_tissue_categories()
    assert "testis" in cats["reproductive"]
    assert "thymus" in cats["thymus"]


def test_cell_line_is_cancer():
    flags = classify_ms_row("No immunization", "healthy", "Cell Line / Clone", "Blood", "HeLa")
    assert flags["src_cancer"] is True
    assert flags["src_healthy_tissue"] is False
    assert flags["cell_line_name"] == "HeLa"


def test_ebv_lcl_not_cancer():
    flags = classify_ms_row(
        "No immunization", "healthy", "Cell Line / Clone (EBV transformed, B-LCL)", cell_name="LCL1"
    )
    assert flags["src_cancer"] is False
    assert flags["src_ebv_lcl"] is True


def test_healthy_somatic():
    flags = classify_ms_row("No immunization", "healthy", "Direct Ex Vivo", "Liver")
    assert flags["src_healthy_tissue"] is True
    assert flags["src_cancer"] is False


def test_healthy_thymus():
    flags = classify_ms_row("No immunization", "healthy", "Direct Ex Vivo", "Thymus")
    assert flags["src_healthy_thymus"] is True
    assert flags["src_healthy_tissue"] is False


def test_healthy_reproductive():
    flags = classify_ms_row("No immunization", "healthy", "Direct Ex Vivo", "Testis")
    assert flags["src_healthy_reproductive"] is True


def test_pmid_override_adjacent():
    flags = classify_ms_row("No immunization", "healthy", "Direct Ex Vivo", "Lung", pmid=35051231)
    assert flags["src_adjacent_to_tumor"] is True
    assert flags["src_healthy_tissue"] is False


def test_pmid_override_activated_apc():
    flags = classify_ms_row(
        "No immunization", "healthy", "Direct Ex Vivo", "Blood", "DC", pmid=32983136
    )
    assert flags["src_activated_apc"] is True


def test_ebv_lcl_override():
    """override: ebv_lcl → not cancer, not healthy, is EBV-LCL + cell line."""
    flags = classify_ms_row(
        "No immunization",
        "healthy",
        "Cell Line / Clone (EBV transformed, B-LCL)",
        "Blood",
        "GR",
        pmid=24616531,
    )
    assert flags["src_cancer"] is False
    assert flags["src_ebv_lcl"] is True
    assert flags["src_cell_line"] is True
    assert flags["src_healthy_tissue"] is False


def test_ebv_lcl_override_forces_flags():
    """override: ebv_lcl should force EBV-LCL even if culture_condition is wrong."""
    flags = classify_ms_row(
        "No immunization",
        "healthy",
        "Cell Line / Clone",  # NOT the EBV string
        "Blood",
        "GR",
        pmid=24616531,
    )
    assert flags["src_ebv_lcl"] is True
    assert flags["src_cell_line"] is True
    assert flags["src_cancer"] is False


def test_cell_line_override_ebv_lcl_not_cancer():
    """cell_line override with EBV culture_condition → not cancer."""
    # Ritz 2017 is override: cell_line (mixed study), JY is EBV-LCL
    flags = classify_ms_row(
        "No immunization",
        "healthy",
        "Cell Line / Clone (EBV transformed, B-LCL)",
        "Blood",
        "JY",
        pmid=28834231,
    )
    assert flags["src_cancer"] is False
    assert flags["src_ebv_lcl"] is True


def test_cell_line_override_non_ebv_still_cancer():
    """cell_line override with non-EBV cell line → still cancer."""
    # Ritz 2017 also has HEK293 (non-EBV)
    flags = classify_ms_row(
        "No immunization",
        "healthy",
        "Cell Line / Clone",
        "Blood",
        "HEK293",
        pmid=28834231,
    )
    assert flags["src_cancer"] is True
    assert flags["src_ebv_lcl"] is False


def test_patient_b_all_is_cancer():
    """Patient B-ALL (not EBV-LCL) should still be cancer."""
    flags = classify_ms_row(
        "Occurrence of cancer",
        "acute lymphoblastic leukemia",
        "Direct Ex Vivo",
        "Blood",
        "B cell",
    )
    assert flags["src_cancer"] is True
    assert flags["src_ebv_lcl"] is False


def test_neidert_tissue_override_blood_healthy():
    flags = classify_ms_row("No immunization", "healthy", "Direct Ex Vivo", "Blood", pmid=29557506)
    assert flags["src_healthy_tissue"] is True


def test_neidert_tissue_override_colon_adjacent():
    flags = classify_ms_row("No immunization", "healthy", "Direct Ex Vivo", "Colon", pmid=29557506)
    assert flags["src_adjacent_to_tumor"] is True
    assert flags["src_healthy_tissue"] is False


def test_neidert_tissue_override_kidney_adjacent():
    flags = classify_ms_row("No immunization", "healthy", "Direct Ex Vivo", "Kidney", pmid=29557506)
    assert flags["src_adjacent_to_tumor"] is True


def test_auto_detect_activated_apc():
    flags = classify_ms_row(
        "No immunization", "healthy", "Direct Ex Vivo", "Blood", "Dendritic cell"
    )
    assert flags["src_activated_apc"] is True
    assert flags["src_healthy_tissue"] is False


def test_cancer_specific_true():
    assert is_cancer_specific({"found_in_cancer": True, "found_in_healthy_tissue": False})


def test_cancer_specific_false():
    assert not is_cancer_specific({"found_in_cancer": True, "found_in_healthy_tissue": True})


def test_cancer_specific_thymus_ok():
    assert is_cancer_specific(
        {"found_in_cancer": True, "found_in_healthy_tissue": False, "found_in_healthy_thymus": True}
    )


# ── Mono-allelic detection ─────────────────────────────────────────────


def test_load_monoallelic_lines():
    lines = load_monoallelic_lines()
    assert len(lines) >= 3
    names = {entry["name"] for entry in lines}
    assert "721.221" in names
    assert "C1R" in names
    assert "K562" in names


def test_detect_monoallelic_721_221():
    is_mono, host = detect_monoallelic("B721.221", "HLA-A*02:01")
    assert is_mono is True
    assert host == "721.221"


def test_detect_monoallelic_c1r_transfected():
    is_mono, host = detect_monoallelic("C1R cells-B cell", "HLA-A*02:01")
    assert is_mono is True
    assert host == "C1R"


def test_detect_monoallelic_c1r_endogenous():
    is_mono, _host = detect_monoallelic("C1R cells-B cell", "HLA-B*35:03")
    assert is_mono is False


def test_detect_monoallelic_k562():
    is_mono, host = detect_monoallelic("K562 cells", "HLA-A*24:02")
    assert is_mono is True
    assert host == "K562"


def test_detect_monoallelic_unrelated():
    is_mono, host = detect_monoallelic("HeLa", "HLA-A*02:01")
    assert is_mono is False
    assert host == ""


def test_detect_monoallelic_empty():
    is_mono, _host = detect_monoallelic("", "HLA-A*02:01")
    assert is_mono is False


def test_classify_ms_row_monoallelic():
    flags = classify_ms_row(
        "No immunization",
        "healthy",
        "Cell Line / Clone",
        "Blood",
        "B721.221",
        pmid="",
        mhc_restriction="HLA-A*02:01",
    )
    assert flags["is_monoallelic"] is True
    assert flags["monoallelic_host"] == "721.221"
    assert flags["src_cell_line"] is True


def test_classify_ms_row_not_monoallelic_ex_vivo():
    flags = classify_ms_row(
        "No immunization",
        "healthy",
        "Direct Ex Vivo",
        "Blood",
        "B721.221",
        mhc_restriction="HLA-A*02:01",
    )
    assert flags["is_monoallelic"] is False


def test_classify_ms_row_backward_compat():
    flags = classify_ms_row("No immunization", "healthy", "Cell Line / Clone", "Blood", "HeLa")
    assert "is_monoallelic" in flags
    assert flags["is_monoallelic"] is False


def test_healthy_override_forces_healthy_path():
    """PMID 33858848 (Marcu 2021 HLA Ligand Atlas) declares override: healthy.

    Even if IEDB fields say "Occurrence of cancer", the override should
    force the row into the healthy-donor classification.
    """
    flags = classify_ms_row(
        "Occurrence of cancer",
        "melanoma",
        "Direct Ex Vivo",
        "Liver",
        "",
        pmid=33858848,
    )
    assert flags["src_cancer"] is False
    assert flags["src_healthy_tissue"] is True


def test_healthy_override_conditional_rule():
    """PMID 36589698 has conditional rules: ex vivo -> healthy."""
    flags = classify_ms_row(
        "No immunization",
        "",
        "Direct Ex Vivo",
        "Blood",
        "",
        pmid=36589698,
    )
    assert flags["src_healthy_tissue"] is True
    assert flags["src_cancer"] is False


# ── PMID curation audits (issues #128 / #129 / #130 / #131) ───────────


def test_pmid_28832583_blood_b_cell_classifies_as_ebv_lcl():
    """Issue #131: IEDB tags Bassani-Sternberg 2017 EBV-LCL rows as
    Culture Condition="Cell Line / Clone" (not the EBV-LCL variant), so a
    PMID rule on Source Tissue=Blood + Cell Name=B cell is required to
    classify them as ebv_lcl rather than cancer/cell_line."""
    flags = classify_ms_row(
        "No immunization",
        "",
        "Cell Line / Clone",
        source_tissue="Blood",
        cell_name="B cell",
        pmid=28832583,
    )
    assert flags["src_ebv_lcl"] is True
    assert flags["src_cancer"] is False
    assert flags["src_cell_line"] is True


def test_pmid_28832583_skin_lymphocyte_classifies_as_cancer():
    """Issue #131: melanoma TIL rows (Source Tissue=Skin + Cell Name=Lymphocyte)
    in Bassani-Sternberg 2017 should remain cancer-derived under the
    cancer_patient override."""
    flags = classify_ms_row(
        "Occurrence of cancer",
        "skin melanoma",
        "Cell Line / Clone",
        source_tissue="Skin",
        cell_name="Lymphocyte",
        pmid=28832583,
    )
    assert flags["src_cancer"] is True
    assert flags["src_ebv_lcl"] is False


def test_pmid_29789417_crc_arm_assay_comments():
    """Issue #128: Löffler 2018 CRC paper — IEDB has no per-row structured
    arm provenance; the only signal is Assay Comments substring. A row tagged
    "eluted from colorectal carcinoma (CRC) tissue." must classify as cancer."""
    flags = classify_ms_row(
        "Occurrence of cancer",
        "colonic benign neoplasm",
        "Direct Ex Vivo",
        source_tissue="Gastrointestinal Tract",
        cell_name="Unknown/Unspecified",
        pmid=29789417,
        assay_comments="The epitope was eluted from colorectal carcinoma (CRC) tissue.",
    )
    assert flags["src_cancer"] is True
    assert flags["src_adjacent_to_tumor"] is False


def test_pmid_29789417_nmc_arm_assay_comments():
    """Issue #128: Löffler 2018 — NMC-only rows must classify as adjacent."""
    flags = classify_ms_row(
        "Occurrence of cancer",
        "colonic benign neoplasm",
        "Direct Ex Vivo",
        source_tissue="Gastrointestinal Tract",
        cell_name="Unknown/Unspecified",
        pmid=29789417,
        assay_comments="The epitope was eluted from nonmalignant colon (NMC) tissue.",
    )
    assert flags["src_cancer"] is False
    assert flags["src_adjacent_to_tumor"] is True


def test_pmid_29789417_combined_arm_keeps_cancer_signal():
    """Issue #128: combined "CRC and NMC" rows preserve the cancer signal —
    NMC-only evidence is captured separately by NMC-only rows for the same
    peptide."""
    flags = classify_ms_row(
        "Occurrence of cancer",
        "colonic benign neoplasm",
        "Direct Ex Vivo",
        source_tissue="Gastrointestinal Tract",
        cell_name="Unknown/Unspecified",
        pmid=29789417,
        assay_comments=(
            "The epitope was eluted from colorectal carcinoma (CRC) and "
            "corresponding nonmalignant colon (NMC) tissue."
        ),
    )
    assert flags["src_cancer"] is True
    assert flags["src_adjacent_to_tumor"] is False


def test_pmid_29789417_combined_then_crc_restatement_classifies_as_cancer():
    """Issue #128 rule-order: when IEDB concatenates the combined-arm
    sentence with an explicit CRC-only restatement (3 rows in shipped data),
    the trailing "(CRC) tissue." substring trips the CRC rule and the row
    classifies as cancer.  Locks first-match-wins ordering: the CRC rule must
    win over the combined-arm rule for these compound comments."""
    flags = classify_ms_row(
        "Occurrence of cancer",
        "colonic benign neoplasm",
        "Direct Ex Vivo",
        source_tissue="Gastrointestinal Tract",
        cell_name="Unknown/Unspecified",
        pmid=29789417,
        assay_comments=(
            "The epitope was eluted from colorectal carcinoma (CRC) and "
            "corresponding nonmalignant colon (NMC) tissue. "
            "The epitope was eluted from colorectal carcinoma (CRC) tissue."
        ),
    )
    assert flags["src_cancer"] is True
    assert flags["src_adjacent_to_tumor"] is False


def test_pmid_29789417_combined_then_nmc_restatement_classifies_as_cancer():
    """Issue #128 rule-order + caveat: when the combined-arm sentence is
    followed by an explicit NMC-only restatement (5 rows in shipped data),
    the combined-arm rule fires first and the row classifies as cancer
    even though the peptide also has explicit NMC-only evidence in its row.
    The trade-off is documented in the PMID 29789417 note: peptides that
    only appear in NMC will be captured separately by NMC-only rows when
    such rows exist."""
    flags = classify_ms_row(
        "Occurrence of cancer",
        "colonic benign neoplasm",
        "Direct Ex Vivo",
        source_tissue="Gastrointestinal Tract",
        cell_name="Unknown/Unspecified",
        pmid=29789417,
        assay_comments=(
            "The epitope was eluted from colorectal carcinoma (CRC) and "
            "corresponding nonmalignant colon (NMC) tissue. "
            "The epitope was eluted from nonmalignant colon (NMC) tissue."
        ),
    )
    assert flags["src_cancer"] is True
    assert flags["src_adjacent_to_tumor"] is False


def test_pmid_29093164_ovarian_carcinoma_arm_classifies_as_cancer():
    """Issue #129: Schuster 2017 ovarian carcinoma rows."""
    flags = classify_ms_row(
        "Occurrence of cancer",
        "ovarian cancer",
        "Direct Ex Vivo",
        source_tissue="Ovary",
        cell_name="Other",
        pmid=29093164,
    )
    assert flags["src_cancer"] is True
    assert flags["src_adjacent_to_tumor"] is False


def test_pmid_29093164_benign_comparator_classifies_as_adjacent():
    """Issue #129: matched benign ovarian comparator from cancer patients
    must classify as adjacent, not healthy_tissue (default would make
    No-immunization + empty disease + ex vivo into healthy_donor)."""
    flags = classify_ms_row(
        "No immunization",
        "",
        "Direct Ex Vivo",
        source_tissue="Other",
        cell_name="Other",
        pmid=29093164,
    )
    assert flags["src_adjacent_to_tumor"] is True
    assert flags["src_cancer"] is False
    assert flags["src_healthy_tissue"] is False


def test_pmid_29331515_breast_cancer_panel_remains_cell_line():
    """Issue #130: Rozanov 2018 — every IEDB row for this PMID should
    classify as cell_line / cancer regardless of which Cell Name string IEDB
    has stamped on it (some rows carry non-breast cell names from
    cross-reference annotations rather than actual MS-elution evidence)."""
    flags = classify_ms_row(
        "No immunization",
        "",
        "Cell Line / Clone",
        source_tissue="Breast",
        cell_name="HCC1187-Epithelial cell",
        pmid=29331515,
    )
    assert flags["src_cell_line"] is True
    assert flags["src_cancer"] is True
    assert flags["src_ebv_lcl"] is False
    # Same classification even when IEDB has tagged a non-breast cell name.
    flags = classify_ms_row(
        "No immunization",
        "",
        "Cell Line / Clone",
        source_tissue="Skin",
        cell_name="MeWo-Fibroblast",
        pmid=29331515,
    )
    assert flags["src_cell_line"] is True
    assert flags["src_cancer"] is True


def test_pmid_36796642_sherpa_pyke_entry_present():
    """Issue #132: SHERPA / Pyke study (HLApollo "personalis_pyke_mhcI_2021")
    citation history. The withdrawn 2021 PMID 34126241 is recorded in the
    aliases of the corrected 2023 PMID 36796642 entry."""
    overrides = load_pmid_overrides()
    assert 36796642 in overrides
    entry = overrides[36796642]
    assert "Pyke" in entry["study_label"] or "SHERPA" in entry["study_label"]
    aliases = entry.get("aliases") or {}
    assert aliases.get("withdrawn_pmid") == 34126241
    assert aliases.get("hlapollo_dataset_name") == "personalis_pyke_mhcI_2021"


# ── Allele resolution ──────────────────────────────────────────────────


def test_classify_allele_resolution_four_digit():
    assert classify_allele_resolution("HLA-A*02:01") == "four_digit"
    assert classify_allele_resolution("HLA-B*49:01") == "four_digit"
    assert classify_allele_resolution("HLA-C*07:02") == "four_digit"
    assert classify_allele_resolution("HLA-DQA1*01:03/DQB1*06:03") == "four_digit"
    assert classify_allele_resolution("HLA-DPA1*02:01/DPB1*05:01") == "four_digit"


def test_classify_allele_resolution_serological():
    assert classify_allele_resolution("HLA-A2") == "serological"
    assert classify_allele_resolution("HLA-B7") == "serological"


def test_classify_allele_resolution_class_only():
    assert classify_allele_resolution("HLA class I") == "class_only"
    assert classify_allele_resolution("HLA class II") == "class_only"


def test_classify_allele_resolution_unresolved():
    assert classify_allele_resolution("") == "unresolved"


def test_classify_allele_resolution_pair_gene_gene_does_not_crash():
    """Regression for #87: mhcgnomes parses locus-only pair strings
    (e.g. "HLA-DRA/DRB1" or "HLA-DPA1*01:03/DPB1") into a Pair whose legs
    are Gene instances, not Allele.  Gene has no allele_fields; the prior
    isinstance dispatch crashed with AttributeError. A Pair[Gene, Gene]
    is below serological/two_digit resolution and should be "unresolved".
    Mixed Pair[Allele, Gene] falls through to "unresolved" as well since
    both sides need to clear the one-digit threshold."""
    assert classify_allele_resolution("HLA-DRA/DRB1") == "unresolved"
    # Mixed Pair — one side has allele fields, the other doesn't.
    assert classify_allele_resolution("HLA-DPA1*01:03/DPB1") == "unresolved"


def test_classify_allele_resolution_mouse():
    # H-2Kb is a valid mouse allele. mhcgnomes parses it as two_digit;
    # regex fallback returns unresolved (not HLA). Either is acceptable
    # since hla_only filters these out before they reach output.
    assert classify_allele_resolution("H-2Kb") in ("two_digit", "unresolved")


def test_allele_resolution_rank_ordering():
    ranks = [allele_resolution_rank(r) for r in ALLELE_RESOLUTION_ORDER]
    assert ranks == sorted(ranks)
    assert allele_resolution_rank("four_digit") < allele_resolution_rank("serological")
    assert allele_resolution_rank("serological") < allele_resolution_rank("class_only")


def test_classify_ms_row_includes_allele_resolution():
    flags = classify_ms_row(
        "No immunization",
        "healthy",
        "Direct Ex Vivo",
        "Liver",
        mhc_restriction="HLA-A*02:01",
    )
    assert flags["allele_resolution"] == "four_digit"

    flags = classify_ms_row(
        "No immunization",
        "healthy",
        "Direct Ex Vivo",
        "Liver",
        mhc_restriction="HLA class I",
    )
    assert flags["allele_resolution"] == "class_only"


# ── Serotype mapping ──────────────────────────────────────────────────


try:
    import mhcgnomes  # noqa: F401

    _HAS_MHCGNOMES = True
except ImportError:
    _HAS_MHCGNOMES = False


def test_allele_to_serotype_four_digit():
    # Requires mhcgnomes; returns "" without it
    result = allele_to_serotype("HLA-A*02:01")
    if _HAS_MHCGNOMES:
        assert result == "HLA-A2"
    else:
        assert result == ""


def test_allele_to_serotype_already_serotype():
    result = allele_to_serotype("HLA-A2")
    if _HAS_MHCGNOMES:
        assert result == "HLA-A2"
    else:
        assert result == ""


def test_allele_to_serotype_class_only():
    assert allele_to_serotype("HLA class I") == ""


def test_allele_to_serotype_empty():
    assert allele_to_serotype("") == ""


def test_classify_ms_row_includes_serotype():
    flags = classify_ms_row(
        "No immunization",
        "healthy",
        "Direct Ex Vivo",
        "Liver",
        mhc_restriction="HLA-A*02:01",
    )
    assert "serotype" in flags
    assert "serotypes" in flags
    if _HAS_MHCGNOMES:
        # Most-specific serotype is the broader locus-specific name (A2),
        # not the IEF sub-serotype (A2.1)
        assert flags["serotype"] == "HLA-A2"
        assert "HLA-A2" in flags["serotypes"].split(";")


def test_allele_to_serotype_prefers_locus_specific_over_bw4():
    """hitlist #44: A*24:02 must resolve to HLA-A24, not HLA-Bw4."""
    if not _HAS_MHCGNOMES:
        return
    assert allele_to_serotype("HLA-A*24:02") == "HLA-A24"
    # And the full list should include Bw4 as secondary
    all_sero = allele_to_all_serotypes("HLA-A*24:02")
    assert "HLA-A24" in all_sero
    assert "HLA-Bw4" in all_sero
    assert all_sero[0] == "HLA-A24"  # locus-specific wins


def test_allele_to_serotype_b57_prefers_b57():
    """B*57:01 belongs to B57, B17, AND Bw4 — B57 must win."""
    if not _HAS_MHCGNOMES:
        return
    assert allele_to_serotype("HLA-B*57:01") == "HLA-B57"
    all_sero = allele_to_all_serotypes("HLA-B*57:01")
    assert all_sero[0] == "HLA-B57"
    assert "HLA-Bw4" in all_sero


def test_allele_to_all_serotypes_a02_broader_first():
    """A*02:01 has A2 (broader) and A2.1 (IEF sub-serotype).

    Our ranking prefers the broader name as canonical — clinicians say "A2",
    not "A2.1".
    """
    if not _HAS_MHCGNOMES:
        return
    all_sero = allele_to_all_serotypes("HLA-A*02:01")
    assert all_sero[0] == "HLA-A2"
    assert "HLA-A2" in all_sero


def test_allele_to_all_serotypes_empty():
    assert allele_to_all_serotypes("") == ()
    assert allele_to_all_serotypes("HLA class I") == ()


# ── Forward serotype expansion (v1.30.2) ──────────────────────────────


def test_serotype_to_alleles_a2_includes_canonical_member():
    """v1.30.2: HLA-A2 expands to its 4-digit members; A*02:01 must be one
    of them and must be first (lowest-numbered = best-guess heuristic)."""
    if not _HAS_MHCGNOMES:
        return
    members = serotype_to_alleles("HLA-A2")
    assert len(members) >= 10  # A2 has dozens of members in IPD-IMGT/HLA
    assert "HLA-A*02:01" in members
    assert members[0] == "HLA-A*02:01"  # sorted-ascending → lowest-numbered first


def test_serotype_to_alleles_b7_dominated_by_b07_02():
    """v1.30.2: HLA-B7 → B*07:02 as the lowest-numbered member."""
    if not _HAS_MHCGNOMES:
        return
    members = serotype_to_alleles("HLA-B7")
    assert "HLA-B*07:02" in members
    # B*07:02 is the most common B7 in nearly all populations.
    assert members[0] == "HLA-B*07:02"


def test_serotype_to_alleles_no_op_on_4digit_input():
    """v1.30.2: passing a 4-digit allele in returns ``()`` — only true
    serotypes expand."""
    if not _HAS_MHCGNOMES:
        return
    assert serotype_to_alleles("HLA-A*02:01") == ()


def test_serotype_to_alleles_empty_and_unknown():
    """v1.30.2: empty / unknown / class-only inputs return ``()``."""
    assert serotype_to_alleles("") == ()
    if not _HAS_MHCGNOMES:
        return
    assert serotype_to_alleles("HLA class I") == ()
    assert serotype_to_alleles("HLA-A99") == ()  # not a real serotype


def test_best_4digit_for_serotype_a2_a0201():
    """v1.30.2: HLA-A2's best 4-digit guess is HLA-A*02:01."""
    if not _HAS_MHCGNOMES:
        return
    assert best_4digit_for_serotype("HLA-A2") == "HLA-A*02:01"


def test_best_4digit_for_serotype_returns_empty_for_non_serotype():
    """v1.30.2: 4-digit input or unknown serotype → empty string (caller
    decides the fallback)."""
    if not _HAS_MHCGNOMES:
        return
    assert best_4digit_for_serotype("HLA-A*02:01") == ""
    assert best_4digit_for_serotype("") == ""


def test_classify_ms_row_serotypes_plural_populated():
    """serotypes column must be semicolon-joined when multiple serotypes exist."""
    if not _HAS_MHCGNOMES:
        return
    flags = classify_ms_row(
        "No immunization",
        "healthy",
        "Direct Ex Vivo",
        "Liver",
        mhc_restriction="HLA-A*24:02",
    )
    assert flags["serotype"] == "HLA-A24"
    assert "HLA-A24" in flags["serotypes"].split(";")
    assert "HLA-Bw4" in flags["serotypes"].split(";")


def test_is_binding_assay_competitive_ic50_comment():
    """Marcilla-style acid-strip / flow-cytometry IC50 rows are binding assays."""
    comments = (
        "C1R-B*40:02 cells were acid stripped to dissociate surface HLA class I complexes. "
        "Then, a reference peptide that bound specifically to HLA-B*40:02 was added to the "
        "cells together with human β2m and different concentrations of the test peptides. "
        "Fluorescence was measured by flow cytometry. Experimental data were fitted to "
        "sigmoid curves to allow the estimation of the IC50 values."
    )
    assert is_binding_assay("Positive", comments) is True


# ── MHC species classification ─────────────────────────────────────────


def test_classify_mhc_species_human():
    assert classify_mhc_species("HLA-A*02:01") == "Homo sapiens"
    assert classify_mhc_species("HLA class I") == "Homo sapiens"


def test_classify_mhc_species_mouse():
    result = classify_mhc_species("H-2Kb")
    if _HAS_MHCGNOMES:
        assert result == "Mus musculus"
    else:
        assert result == "Mus musculus"  # regex fallback handles H-2


def test_classify_mhc_species_empty():
    assert classify_mhc_species("") == ""


def test_classify_ms_row_includes_mhc_species():
    flags = classify_ms_row(
        "No immunization",
        "healthy",
        "Direct Ex Vivo",
        "Liver",
        mhc_restriction="HLA-A*02:01",
    )
    assert flags["mhc_species"] == "Homo sapiens"


# ── PMID-level mono-allelic override ───────────────────────────────────


def test_pmid_mono_allelic_override():
    """PMID 28228285 (Sarkizova 2020) is a 721.221 study."""
    flags = classify_ms_row(
        "No immunization",
        "",
        "Cell Line / Clone",
        "Blood",
        "B cell",
        pmid=28228285,
        mhc_restriction="HLA-A*01:01",
    )
    assert flags["is_monoallelic"] is True
    assert flags["monoallelic_host"] == "721.221"


def test_pmid_mono_allelic_override_28514659():
    """PMID 28514659 (HLA-B*46:01) is also a 721.221 study."""
    flags = classify_ms_row(
        "No immunization",
        "",
        "Cell Line / Clone",
        "Spleen",
        "Other",
        pmid=28514659,
        mhc_restriction="HLA-B*46:01",
    )
    assert flags["is_monoallelic"] is True
    assert flags["monoallelic_host"] == "721.221"


def test_pmid_mono_override_skipped_for_unresolved_allele():
    """is_monoallelic is a sample-level claim — a row with no resolved
    allele cannot be flagged mono-allelic even if the paper uses a
    mono-allelic host.  Faridi 2018 (PMID 30315122) has samples with
    mhc: unknown; those rows must not be claimed as mono-allelic.
    """
    flags_empty = classify_ms_row(
        "No immunization",
        "",
        "Cell Line / Clone",
        "Blood",
        "B cell",
        pmid=30315122,
        mhc_restriction="",
    )
    assert flags_empty["is_monoallelic"] is False

    flags_class_only = classify_ms_row(
        "No immunization",
        "",
        "Cell Line / Clone",
        "Blood",
        "B cell",
        pmid=30315122,
        mhc_restriction="HLA class I",
    )
    assert flags_class_only["is_monoallelic"] is False


def test_pmid_mono_override_skipped_for_validation_class_only():
    """Sarkizova 2020 mixes 95 721.221 transfectants (mhc_restriction is a
    specific allele) with 12 patient-derived validation samples whose
    IEDB mhc_restriction is the class-only string ``"HLA class I"``.  The
    allele-resolution gate catches the validation rows — they cannot
    claim mono-allelic status without a resolved allele.
    """
    flags_validation = classify_ms_row(
        "No immunization",
        "",
        "Cell Line / Clone",
        "Blood",
        "Glial cell",
        pmid=31844290,
        mhc_restriction="HLA class I",
    )
    assert flags_validation["is_monoallelic"] is False


def test_pmid_mono_override_applies_across_cell_name_variants():
    """The PMID override is NOT gated on cell_name — IEDB frequently
    mis-labels the host (e.g. Trolle 2016 records 721.221 transfectants
    as ``"HeLa cells-Epithelial cell"``).  As long as the row has a
    resolved allele and the PMID has ``mono_allelic_host``, the row
    flags mono-allelic.
    """
    # Sarkizova transfectants — IEDB label is "B cell", ambiguous
    # Trolle   transfectants — IEDB label is "HeLa cells-Epithelial cell", specific but WRONG
    # Splenocyte etc. — anything goes as long as allele is resolved
    for pmid, cn in (
        (31844290, "B cell"),
        (26783342, "HeLa cells-Epithelial cell"),
        (28514659, "Splenocyte"),
        (28228285, ""),
    ):
        flags = classify_ms_row(
            "No immunization",
            "",
            "Cell Line / Clone",
            "Blood",
            cn,
            pmid=pmid,
            mhc_restriction="HLA-A*02:01",
        )
        assert flags["is_monoallelic"] is True, f"override should apply for PMID={pmid} cn={cn!r}"


def test_pmid_mono_allelic_method_override():
    """mono_allelic_method (e.g. MAPTAC) sets is_monoallelic for resolved
    alleles without requiring a cell-line entry in monoallelic_lines.yaml.
    """
    # PMID 31495665 has mono_allelic_method: "MAPTAC"
    flags = classify_ms_row(
        "No immunization",
        "",
        "Cell Line / Clone",
        "",
        "MAPTAC",
        pmid=31495665,
        mhc_restriction="HLA-A*02:01",
    )
    assert flags["is_monoallelic"] is True
    assert flags["monoallelic_host"] == "MAPTAC"

    # Class-only allele should NOT be flagged mono-allelic
    flags_class = classify_ms_row(
        "No immunization",
        "",
        "Cell Line / Clone",
        "",
        "MAPTAC",
        pmid=31495665,
        mhc_restriction="HLA class II",
    )
    assert flags_class["is_monoallelic"] is False


def test_pmid_mono_allelic_method_override_strazar():
    """Stražar 2023 uses a Strep-tag II tagged-allele pulldown rather than
    a known monoallelic host cell line, so the PMID-level method override
    should mark resolved HLA-II pairs as mono-allelic.
    """
    flags = classify_ms_row(
        "No immunization",
        "healthy",
        "Cell Line / Clone",
        "Kidney",
        "Expi293F",
        pmid=37301199,
        mhc_restriction="HLA-DQA1*01:03/DQB1*06:03",
    )
    assert flags["is_monoallelic"] is True
    assert flags["monoallelic_host"] == "Strep-tag II"

    flags_class = classify_ms_row(
        "No immunization",
        "healthy",
        "Cell Line / Clone",
        "Kidney",
        "Expi293F",
        pmid=37301199,
        mhc_restriction="HLA class II",
    )
    assert flags_class["is_monoallelic"] is False


def test_illing_2018_monoallelic_host_applies_without_cell_name():
    """Illing 2018 rows that lack the C1R cell_name in IEDB should still be
    flagged mono-allelic via the PMID-level ``mono_allelic_host: C1R``
    override, as long as the allele is resolved."""
    flags = classify_ms_row(
        "No immunization",
        "healthy",
        "Cell Line / Clone",
        "",
        "",  # empty cell_name (the ~33 IEDB rows that lack the C1R label)
        pmid=30410026,
        mhc_restriction="HLA-B*57:01",
    )
    assert flags["is_monoallelic"] is True
    assert flags["monoallelic_host"] == "C1R"


def test_load_pmid_overrides_rejects_unknown_mono_host(tmp_path, monkeypatch):
    """A PMID override with a mono_allelic_host that isn't in
    monoallelic_lines.yaml must raise at load time — silently producing
    rows with a bogus monoallelic_host string would leak typos into the
    published index.
    """
    import yaml as _yaml

    from hitlist import curation

    bad_yaml = tmp_path / "pmid_overrides.yaml"
    bad_yaml.write_text(
        _yaml.safe_dump(
            [
                {
                    "pmid": 99999999,
                    "study_label": "Bogus",
                    "mono_allelic_host": "NOT_A_REAL_HOST",
                }
            ]
        )
    )
    monkeypatch.setattr(
        curation, "_data_path", lambda fn: str(bad_yaml) if fn == "pmid_overrides.yaml" else fn
    )
    curation.load_pmid_overrides.cache_clear()
    try:
        with pytest.raises(ValueError, match="NOT_A_REAL_HOST"):
            curation.load_pmid_overrides()
    finally:
        curation.load_pmid_overrides.cache_clear()


def test_load_pmid_overrides_warns_on_legacy_keys(tmp_path, monkeypatch):
    """Legacy keys ``label:`` / ``type:`` were renamed in v1.7.0.  Loading
    a file with the old names must emit DeprecationWarning so users
    catch schema drift.
    """
    import warnings

    import yaml as _yaml

    from hitlist import curation

    legacy_yaml = tmp_path / "pmid_overrides.yaml"
    legacy_yaml.write_text(
        _yaml.safe_dump(
            [
                {
                    "pmid": 42,
                    "label": "legacy label",
                    "ms_samples": [{"type": "legacy sample"}],
                }
            ]
        )
    )
    # Route only the pmid_overrides.yaml lookup to the temp fixture; the
    # other data files (e.g. monoallelic_lines.yaml that
    # load_pmid_overrides validates against) must continue to resolve to
    # the real package data dir. The previous monkeypatch returned a
    # bare filename for non-pmid-overrides paths, which silently worked
    # only when the lru_cache was already warmed by an earlier test
    # (test-order-dependent flake).
    real_data_path = curation._data_path
    monkeypatch.setattr(
        curation,
        "_data_path",
        lambda fn: str(legacy_yaml) if fn == "pmid_overrides.yaml" else real_data_path(fn),
    )
    curation.load_pmid_overrides.cache_clear()
    curation.load_monoallelic_lines.cache_clear()
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", DeprecationWarning)
            curation.load_pmid_overrides()
        messages = [str(x.message) for x in w if issubclass(x.category, DeprecationWarning)]
        assert any("label:" in m for m in messages), messages
        assert any("type:" in m for m in messages), messages
    finally:
        curation.load_pmid_overrides.cache_clear()
        curation.load_monoallelic_lines.cache_clear()


# ── Species normalization ─────────────────────────────────────────────


def test_normalize_species_canonical():
    assert normalize_species("Homo sapiens") == "Homo sapiens"
    assert normalize_species("Mus musculus") == "Mus musculus"


def test_normalize_species_common_names():
    assert normalize_species("human") == "Homo sapiens"
    assert normalize_species("mouse") == "Mus musculus"


def test_normalize_species_underscore():
    assert normalize_species("homo_sapiens") == "Homo sapiens"
    assert normalize_species("Homo_sapiens") == "Homo sapiens"
    assert normalize_species("mus_musculus") == "Mus musculus"


def test_normalize_species_parenthetical():
    assert normalize_species("Homo sapiens (human)") == "Homo sapiens"
    assert normalize_species("Mus musculus (mouse)") == "Mus musculus"
    assert normalize_species("Sus scrofa (pig)") == "Sus scrofa"


def test_normalize_species_empty():
    assert normalize_species("") == ""


def test_normalize_species_idempotent():
    assert normalize_species(normalize_species("human")) == "Homo sapiens"
    assert normalize_species(normalize_species("Homo sapiens (human)")) == "Homo sapiens"


# ── Allele normalization ─────────────────────────────────────────────


def test_normalize_allele_hla():
    assert normalize_allele("HLA-A*02:01") == "HLA-A*02:01"
    assert normalize_allele("HLA-DRB1*04:01") == "HLA-DRB1*04:01"


def test_normalize_allele_mouse():
    # H-2Kb → H2-K*b (canonical mhcgnomes form)
    assert normalize_allele("H-2Kb") == "H2-K*b"


def test_normalize_allele_non_human_species():
    assert normalize_allele("Saha-UA") == "Saha-UA"
    assert normalize_allele("SLA-1*0201") == "SLA-1*02:01"


def test_normalize_allele_preserves_class_only():
    # "HLA class I" should pass through unchanged, not normalize to species
    assert normalize_allele("HLA class I") == "HLA class I"
    assert normalize_allele("HLA class II") == "HLA class II"


def test_normalize_allele_empty_and_unparseable():
    assert normalize_allele("") == ""
    assert normalize_allele("unknown") == "unknown"
    # A truly non-allele string — behaviour should be pass-through across
    # mhcgnomes versions (older mhcgnomes raised ParseError on the prior
    # test string "SahaI*35"; mhcgnomes >= 3.32 parses that successfully
    # as Saha-I*35, which invalidated the assumption).
    assert normalize_allele("definitely not an allele") == "definitely not an allele"


# ── Exact-allele bag expansion (issue #137) ─────────────────────────────


def test_expand_allele_bag_exact_four_digit():
    """A row already at four-digit resolution returns itself with provenance ``exact``."""
    bag, prov, n = expand_allele_bag("HLA-A*02:01")
    assert bag == "HLA-A*02:01"
    assert prov == "exact"
    assert n == 1


def test_expand_allele_bag_class_only_with_sample_mhc():
    """class_only + Host | MHC Types Present yields ``sample_allele_match``
    with the donor's typed alleles, filtered to the row's class.
    """
    host = "HLA-A*01:01;HLA-B*13:02;HLA-B*39:06;HLA-C*12:03"
    bag, prov, n = expand_allele_bag("HLA class I", host, mhc_class="I")
    assert prov == "sample_allele_match"
    assert n == 4
    assert bag == "HLA-A*01:01;HLA-B*13:02;HLA-B*39:06;HLA-C*12:03"


def test_expand_allele_bag_class_only_with_sample_mhc_class_filter():
    """Sample's typed alleles include both class I and class II; the row's
    class filter must drop the wrong-class entries."""
    host = "HLA-A*02:01;HLA-DRB1*15:01"
    bag_i, _, n_i = expand_allele_bag("HLA class I", host, mhc_class="I")
    assert bag_i == "HLA-A*02:01"
    assert n_i == 1
    bag_ii, _, n_ii = expand_allele_bag("HLA class II", host, mhc_class="II")
    assert bag_ii == "HLA-DRB1*15:01"
    assert n_ii == 1


def test_expand_allele_bag_class_only_falls_back_to_pmid_pool():
    """No sample mhc → PMID pool from curated hla_alleles.  Uses PMID
    28832583 (Bassani-Sternberg 2017) which has dict-of-lists hla_alleles."""
    bag, prov, n = expand_allele_bag("HLA class I", host_mhc_types="", pmid=28832583, mhc_class="I")
    assert prov == "pmid_class_pool"
    assert n > 0
    # All alleles in the bag are class I (no HLA-D*).
    assert all(not a.startswith("HLA-D") for a in bag.split(";"))
    # Spot-check: CD165's alleles should be in the pool.
    assert "HLA-A*02:05" in bag
    assert "HLA-B*15:01" in bag


def test_expand_allele_bag_class_only_no_curation_is_unmatched():
    """class_only with neither sample MHC nor PMID curation → unmatched."""
    bag, prov, n = expand_allele_bag("HLA class I", host_mhc_types="", pmid=99999999, mhc_class="I")
    assert bag == ""
    assert prov == "unmatched"
    assert n == 0


def test_expand_allele_bag_two_digit_is_unmatched():
    """Two-digit / serological / unresolved restrictions are emitted as
    unmatched until catalog-based expansion lands (planned follow-up)."""
    for restriction in ("HLA-A*02", "HLA-A2", ""):
        bag, prov, n = expand_allele_bag(restriction)
        assert bag == ""
        assert prov == "unmatched"
        assert n == 0


def test_expand_allele_bag_pmid_with_only_free_text_alleles():
    """PMID 33858848 (HLA Ligand Atlas) has hla_alleles as dict-of-strings
    free-text descriptions ('51 HLA-I allotypes') with no parseable allele
    strings.  The flatten helper must return an empty pool, and a class_only
    row falls through to unmatched rather than crashing."""
    bag, prov, n = expand_allele_bag("HLA class I", host_mhc_types="", pmid=33858848, mhc_class="I")
    assert bag == ""
    assert prov == "unmatched"
    assert n == 0


def test_expand_allele_bag_sample_match_wins_over_pmid_pool():
    """When both sample MHC and PMID pool are available, sample wins
    (more specific provenance)."""
    bag, prov, n = expand_allele_bag(
        "HLA class I",
        host_mhc_types="HLA-A*02:01",
        pmid=28832583,  # has a 32-allele PMID pool
        mhc_class="I",
    )
    assert prov == "sample_allele_match"
    assert bag == "HLA-A*02:01"
    assert n == 1


# ── v1.30.14: _flatten_hla_alleles tokenizes space-separated genotypes ──


def test_flatten_hla_alleles_splits_space_separated_genotype_string():
    """v1.30.14: ~32% of ms_samples in pmid_overrides.yaml encode a
    donor's full genotype as one space-separated string in the
    ``mhc:`` field. ``_flatten_hla_alleles`` must split these into
    individual alleles so qc cross_reference / curation_plan don't
    report the whole genotype as a single phantom 'allele'."""
    from hitlist.curation import _flatten_hla_alleles

    out = _flatten_hla_alleles(
        "HLA-A*01:01 HLA-A*23:01 HLA-B*07:02 HLA-B*15:01 HLA-C*12:03 HLA-C*14:02"
    )
    assert out == {
        "HLA-A*01:01",
        "HLA-A*23:01",
        "HLA-B*07:02",
        "HLA-B*15:01",
        "HLA-C*12:03",
        "HLA-C*14:02",
    }


def test_flatten_hla_alleles_single_allele_string_still_works():
    """Don't regress the single-allele string case while fixing the
    multi-allele case."""
    from hitlist.curation import _flatten_hla_alleles

    assert _flatten_hla_alleles("HLA-A*02:01") == {"HLA-A*02:01"}


def test_flatten_hla_alleles_multi_allele_drops_noise_tokens():
    """Free-text noise tokens (e.g. ``or``, commas) inside a
    multi-allele string fail the syntactic check and are dropped
    rather than mistaken for alleles."""
    from hitlist.curation import _flatten_hla_alleles

    out = _flatten_hla_alleles("HLA-A*02:01 or HLA-B*07:02")
    assert out == {"HLA-A*02:01", "HLA-B*07:02"}


# ── is_chimeric_system ───────────────────────────────────────────────────


def test_is_chimeric_hla_transgenic_rat():
    """HLA-B27-transgenic Lewis rat (PMID 28188227) is the canonical
    engineered chimeric system: rat proteome, human MHC."""
    from hitlist.curation import is_chimeric_system

    assert is_chimeric_system("Rattus norvegicus", "Homo sapiens") is True


def test_is_chimeric_human_peptides_on_mouse_mhc():
    """NetH2pan-style training data (PMID 29615400) — synthetic human
    peptides tested against mouse MHC. Different genus → chimeric."""
    from hitlist.curation import is_chimeric_system

    assert is_chimeric_system("Homo sapiens", "Mus musculus") is True


def test_is_chimeric_substrain_collapses_to_species():
    """``Mus musculus C57BL/6`` is the same species as ``Mus musculus``.
    Substrain labels must NOT trigger the flag."""
    from hitlist.curation import is_chimeric_system

    assert is_chimeric_system("Mus musculus C57BL/6", "Mus musculus") is False
    assert is_chimeric_system("Canis lupus familiaris", "Canis sp.") is False
    assert is_chimeric_system("Sus scrofa", "Sus sp.") is False


def test_is_chimeric_pathogen_source_is_not_chimeric():
    """Viral / bacterial / parasite peptides on a host MHC are
    ordinary infection biology, not engineered cross-species systems.
    Returning ``True`` here would silently misroute every infection
    study into the chimeric bucket."""
    from hitlist.curation import is_chimeric_system

    for src in (
        "Severe acute respiratory syndrome coronavirus 2",
        "Hepatitis B virus",
        "Mycobacterium tuberculosis",
        "Mycobacterium tuberculosis H37Rv",
        "Influenza A virus",
        "Human betaherpesvirus 5",
        "human gammaherpesvirus 4",
        "Fusobacterium nucleatum",
        "Hepacivirus hominis",
        "Eikenella corrodens",
        "Plasmodium falciparum",
        "adeno-associated virus 2",
        "Vaccinia virus WR",
        "SARS-CoV1",
    ):
        assert is_chimeric_system(src, "Homo sapiens") is False, (
            f"pathogen source {src!r} flagged as chimeric"
        )


def test_is_chimeric_unknown_or_empty_is_not_chimeric():
    """Chimerism is a positive claim — if we don't have both halves
    of the proteome/MHC pair, the flag stays False rather than
    guessing."""
    from hitlist.curation import is_chimeric_system

    assert is_chimeric_system("", "Homo sapiens") is False
    assert is_chimeric_system("Homo sapiens", "") is False
    assert is_chimeric_system("unidentified", "Homo sapiens") is False
    assert is_chimeric_system("   ", "Homo sapiens") is False


def test_is_chimeric_same_species_is_not_chimeric():
    """The vast majority of rows: human peptides on human MHC, mouse
    on mouse, etc. Flag must stay False."""
    from hitlist.curation import is_chimeric_system

    assert is_chimeric_system("Homo sapiens", "Homo sapiens") is False
    assert is_chimeric_system("Mus musculus", "Mus musculus") is False


def test_is_chimeric_xenogeneic_animal_systems():
    """Comparative-immunopeptidomics studies that put human peptides
    onto a non-human MHC (canine, primate, equine) — engineered
    cross-species, must flag True."""
    from hitlist.curation import is_chimeric_system

    assert is_chimeric_system("Homo sapiens", "Macaca mulatta") is True
    assert is_chimeric_system("Homo sapiens", "Canis sp.") is True
    assert is_chimeric_system("Mus musculus", "Equus caballus") is True


# ── is_engineered_mhc ────────────────────────────────────────────────────


def test_is_engineered_mhc_hla_transgenic_rat():
    """HLA-B27-Tg rat: rat tissue, rat host, human HLA transgene.
    Host genus matches source genus, MHC is heterologous → engineered."""
    from hitlist.curation import is_engineered_mhc

    assert (
        is_engineered_mhc("Rattus norvegicus", "Homo sapiens", "Rattus norvegicus (brown rat)")
        is True
    )
    # Lewis-strain host string must collapse to rat genus.
    assert is_engineered_mhc("Rattus norvegicus", "Homo sapiens", "Rattus norvegicus Lewis") is True


def test_is_engineered_mhc_neth2pan_training():
    """NetH2pan (PMID 29615400): synthetic human peptides bound by
    mouse MHC, recorded with human host. The MHC is non-native to
    the host → engineered."""
    from hitlist.curation import is_engineered_mhc

    assert is_engineered_mhc("Homo sapiens", "Mus musculus", "Homo sapiens (human)") is True


def test_is_engineered_mhc_lewis_rat_eae_with_gp_mbp_is_not_engineered():
    """The 4 GP-MBP / Lewis-rat EAE rows are heterologous-antigen, NOT
    engineered-MHC. Lewis rats present guinea-pig MBP on their own
    native RT1.B^L MHC. Host genus matches the MHC, not the source —
    the entire point of the engineered-vs-heterologous split (#226)."""
    from hitlist.curation import is_engineered_mhc

    assert (
        is_engineered_mhc("Cavia porcellus", "Rattus sp.", "Rattus norvegicus (brown rat)") is False
    )


def test_is_engineered_mhc_heterologous_allergens_are_not_engineered():
    """Allergen / heterologous-antigen studies (horse Equ c 1 on HLA,
    bovine antigens on human / mouse MHC) present a foreign protein
    on the host's native MHC — must NOT collide with engineered_mhc."""
    from hitlist.curation import is_engineered_mhc

    # PMID 17517108: horse allergen on HLA, human host
    assert is_engineered_mhc("Equus caballus", "Homo sapiens", "Homo sapiens (human)") is False
    # PMID 30573663: bovine antigen on HLA-DO, human host
    assert is_engineered_mhc("Bos taurus", "Homo sapiens", "Homo sapiens (human)") is False
    # PMID 26495903: bovine antigen on mouse MHC-II, mouse host
    assert is_engineered_mhc("Bos taurus", "Mus musculus", "Mus musculus BALB/cAnN") is False


def test_is_engineered_mhc_requires_chimerism():
    """Same-species rows (Hsap/Hsap/Hsap, Mmus/Mmus/Mmus) are not
    chimeric and therefore never engineered_mhc. The function must
    short-circuit cheaply on the common case."""
    from hitlist.curation import is_engineered_mhc

    assert is_engineered_mhc("Homo sapiens", "Homo sapiens", "Homo sapiens (human)") is False
    assert is_engineered_mhc("Mus musculus", "Mus musculus", "Mus musculus C57BL/6") is False


def test_is_engineered_mhc_missing_host_is_conservative_false():
    """Without a host signal the engineered-vs-heterologous distinction
    cannot be made. Default to False rather than guess — the broader
    is_chimeric flag still surfaces these rows for downstream consumers
    that want the union."""
    from hitlist.curation import is_engineered_mhc

    assert is_engineered_mhc("Rattus norvegicus", "Homo sapiens", "") is False
    assert is_engineered_mhc("Rattus norvegicus", "Homo sapiens", "   ") is False
    assert is_engineered_mhc("Rattus norvegicus", "Homo sapiens", "unidentified") is False


def test_is_engineered_mhc_pathogen_source_is_not_engineered():
    """Pathogen sources are never chimeric (gated by the host-genus
    whitelist) and therefore never engineered_mhc, regardless of what
    host is recorded."""
    from hitlist.curation import is_engineered_mhc

    assert (
        is_engineered_mhc(
            "Severe acute respiratory syndrome coronavirus 2",
            "Homo sapiens",
            "Homo sapiens (human)",
        )
        is False
    )


# ── is_non_peptide_ligand (#228) ─────────────────────────────────────────


def test_is_non_peptide_ligand_cd1_family():
    """CD1a/b/c/d/e present lipids and glycolipids to NKT and CD1-restricted
    T cells. IEDB curates these rows with chemical names in the peptide
    column (e.g. sulfatide, mycolic acid) — must flag True."""
    from hitlist.curation import is_non_peptide_ligand

    for s in (
        "human-CD1a",
        "human-CD1b",
        "human-CD1c",
        "human-CD1d",
        "mouse-CD1d",
        "cattle-CD1d",
    ):
        assert is_non_peptide_ligand(s) is True, f"{s!r} should flag"


def test_is_non_peptide_ligand_cd1_subtype_digits_and_dash():
    """IEDB curates a few CD1 paralogues with numeric subtype suffixes
    (mouse-CD1d1, mouse-CD1d2, cattle-CD1b3) and chicken-CD1-2 in
    dash-numeric form. The naive ``\\bCD1[abcd]?\\b`` regex from the
    issue would miss these — the implemented regex must catch them all."""
    from hitlist.curation import is_non_peptide_ligand

    for s in ("mouse-CD1d1", "mouse-CD1d2", "cattle-CD1b3", "chicken-CD1-2"):
        assert is_non_peptide_ligand(s) is True, f"{s!r} should flag"


def test_is_non_peptide_ligand_mr1_and_mutants():
    """MR1 presents riboflavin-derived metabolites to MAIT cells.
    Mutant strings (``human-MR1 K43A mutant``) must still flag —
    regex anchors on the gene token, not the trailing annotation."""
    from hitlist.curation import is_non_peptide_ligand

    assert is_non_peptide_ligand("human-MR1") is True
    assert is_non_peptide_ligand("cattle-MR1") is True
    assert is_non_peptide_ligand("human-MR1 K43A mutant") is True
    assert is_non_peptide_ligand("human-MR1 R9H mutant") is True
    assert is_non_peptide_ligand("mouse-CD1d Y73H mutant") is True


def test_is_non_peptide_ligand_mic_ulbp_raet1_nkg2_hfe():
    """MIC{A,B}, RAET1*, ULBP* are stress ligands for the NKG2D NK
    receptor — not peptide presenters. NKG2[A-C] and HFE round out the
    non-peptide-presenting whitelist from the issue."""
    from hitlist.curation import is_non_peptide_ligand

    for s in ("MICA", "MICB", "RAET1L", "ULBP1", "ULBP3", "NKG2A", "NKG2C", "HFE"):
        assert is_non_peptide_ligand(s) is True, f"{s!r} should flag"


def test_is_non_peptide_ligand_classical_mhc_does_not_flag():
    """Classical class I/II and class-Ib peptide presenters (HLA-A/B/C/E/F/G,
    H2-K/D/L/Q/T, Patr-AL, BoLA-*, SLA-*, DLA-*) all present peptides —
    none must be flagged. H2-M3 is class Ib but presents N-formyl peptides
    and is intentionally excluded from the non-peptide regex (see #228)."""
    from hitlist.curation import is_non_peptide_ligand

    for s in (
        "HLA-A*02:01",
        "HLA-B*07:02",
        "HLA-C*07:01",
        "HLA-DRB1*04:01",
        "HLA-DPB1*04:01",
        "HLA-E*01:03",
        "HLA-F*01:01",
        "HLA-G*01:01",
        "H2-Kb",
        "H2-Db",
        "H2-Q1",
        "H2-T23*a",
        "H2-M3",  # class Ib, N-formyl peptides — NOT a non-peptide presenter
        "Patr-AL",
        "BoLA-1*02301",
        "SLA-1*02:01",
    ):
        assert is_non_peptide_ligand(s) is False, f"{s!r} must not flag"


def test_is_non_peptide_ligand_empty_input():
    """Empty / missing restriction strings cannot be classified — return
    False rather than guessing (matches is_chimeric_system convention)."""
    from hitlist.curation import is_non_peptide_ligand

    assert is_non_peptide_ligand("") is False
    assert is_non_peptide_ligand(None) is False  # type: ignore[arg-type]


# ── Per-sample attribution / donor-bag promotion (#45) ───────────────────


def test_parse_sample_mhc_field_bare_format():
    """Patient ms_samples entries store the donor genotype as a bare
    space-joined string (``"A*02:01 A*24:02 ..."``).  The parser must
    canonicalize each token to ``"HLA-A*02:01"`` form via mhcgnomes
    so the bag matches what bag-expansion expects."""
    from hitlist.curation import _parse_sample_mhc_field

    out = _parse_sample_mhc_field("A*02:01 A*24:02 B*15:01 B*44:02 C*05:01 C*07:02")
    assert out == frozenset(
        {
            "HLA-A*02:01",
            "HLA-A*24:02",
            "HLA-B*15:01",
            "HLA-B*44:02",
            "HLA-C*05:01",
            "HLA-C*07:02",
        }
    )


def test_parse_sample_mhc_field_prefixed_format():
    """ms_samples on the 721.221 mono-allelic entries store HLA-prefixed
    single alleles. Same parser must accept both shapes interchangeably."""
    from hitlist.curation import _parse_sample_mhc_field

    assert _parse_sample_mhc_field("HLA-A*02:01") == frozenset({"HLA-A*02:01"})
    assert _parse_sample_mhc_field("") == frozenset()
    assert _parse_sample_mhc_field(None) == frozenset()


def test_parse_sample_mhc_field_homozygous():
    """Homozygous donor (e.g. MEL2: A*01:01 / A*01:01) collapses to a
    single allele in the bag — ``frozenset`` deduplicates by design."""
    from hitlist.curation import _parse_sample_mhc_field

    out = _parse_sample_mhc_field("A*01:01 A*01:01 B*38:01 B*56:01 C*01:02 C*06:02")
    # A*01:01 collapses; bag size 5 (not 6).
    assert "HLA-A*01:01" in out
    assert len(out) == 5


def test_classify_allele_resolution_donor_bag():
    """Multi-allele bag strings (semicolon-joined 4-digit alleles) classify
    as ``donor_bag`` — strictly more specific than ``class_only``, less
    specific than ``four_digit``.  Order in ALLELE_RESOLUTION_ORDER puts
    ``donor_bag`` between ``four_digit`` and ``two_digit``."""
    from hitlist.curation import (
        ALLELE_RESOLUTION_ORDER,
        allele_resolution_rank,
        classify_allele_resolution,
    )

    assert classify_allele_resolution("HLA-A*02:01;HLA-A*03:01") == "donor_bag"
    assert (
        classify_allele_resolution("HLA-A*01:01;HLA-A*32:01;HLA-B*15:01;HLA-C*03:03;HLA-C*03:04")
        == "donor_bag"
    )
    # Single allele still resolves to four_digit.
    assert classify_allele_resolution("HLA-A*02:01") == "four_digit"
    # Class label still resolves to class_only.
    assert classify_allele_resolution("HLA class I") == "class_only"
    # Trailing semicolon with single allele isn't a bag.
    assert classify_allele_resolution("HLA-A*02:01;") != "donor_bag"
    # Rank ordering: donor_bag is between four_digit and two_digit.
    assert (
        allele_resolution_rank("four_digit")
        < allele_resolution_rank("donor_bag")
        < allele_resolution_rank("two_digit")
        < allele_resolution_rank("class_only")
    )
    assert "donor_bag" in ALLELE_RESOLUTION_ORDER


def test_expand_allele_bag_attributed_alleles_narrows():
    """``attributed_alleles`` (per-peptide attribution from #45) takes
    priority over ``host_mhc_types`` and the PMID pool, narrowing the
    bag to that specific subset of donor alleles.  Provenance becomes
    ``peptide_attribution``."""
    from hitlist.curation import expand_allele_bag

    # Disease-wide host_mhc_types (18-allele MEL union)
    disease_union = (
        "HLA-A*01:01;HLA-A*02:01;HLA-A*02:02;HLA-A*03:01;HLA-A*24:02;"
        "HLA-B*13:02;HLA-B*15:01;HLA-B*27:05;HLA-B*38:01;HLA-B*40:02;"
        "HLA-B*44:02;HLA-B*47:01;HLA-B*56:01;HLA-C*01:02;HLA-C*02:02;"
        "HLA-C*05:01;HLA-C*06:02;HLA-C*07:02"
    )
    # Per-peptide attribution narrows to MEL2's 5-allele genotype.
    mel2 = frozenset({"HLA-A*01:01", "HLA-B*38:01", "HLA-B*56:01", "HLA-C*01:02", "HLA-C*06:02"})

    out_set, prov, size = expand_allele_bag("HLA class I", disease_union, 31844290, "I", mel2)
    assert prov == "peptide_attribution"
    assert size == 5
    assert set(out_set.split(";")) == set(mel2)


def test_expand_allele_bag_no_attribution_falls_back_to_sample_match():
    """Without attribution, a class-only row falls back to the
    ``sample_allele_match`` path (donor's typed alleles from
    ``host_mhc_types``).  Same row, different provenance and broader bag."""
    from hitlist.curation import expand_allele_bag

    disease_union = "HLA-A*01:01;HLA-A*02:01;HLA-B*38:01;HLA-B*56:01;HLA-C*01:02;HLA-C*06:02"
    out_set, prov, size = expand_allele_bag(
        "HLA class I", disease_union, 31844290, "I", frozenset()
    )
    assert prov == "sample_allele_match"
    assert size == 6


def test_attribute_peptide_to_sample_alleles_known_sarkizova_peptide():
    """End-to-end: a known Sarkizova MEL2 peptide pulls back MEL2's
    typed-allele bag (from the ms_samples mhc field via
    _pmid_sample_alleles, after the bare-format normalization fix)."""
    from hitlist.curation import attribute_peptide_to_sample_alleles

    # AAAAAAAAAAAAAAPAP is uniquely attributed to MEL2 in the Sup_Data2 CSV
    out = attribute_peptide_to_sample_alleles(31844290, "AAAAAAAAAAAAAAPAP")
    # MEL2 genotype is A*01:01 / A*01:01 / B*38:01 / B*56:01 / C*01:02 / C*06:02
    # — 5 alleles after homozygous-A collapse.
    assert out == frozenset(
        {"HLA-A*01:01", "HLA-B*38:01", "HLA-B*56:01", "HLA-C*01:02", "HLA-C*06:02"}
    )


def test_attribute_peptide_to_sample_alleles_unknown_peptide_returns_empty():
    """Peptides not in the attribution CSV (or PMIDs without an
    attribution registered) return the empty frozenset — caller falls
    back to ``host_mhc_types`` / pmid pool via expand_allele_bag."""
    from hitlist.curation import attribute_peptide_to_sample_alleles

    assert attribute_peptide_to_sample_alleles(31844290, "ZZZZZZZZZZ") == frozenset()
    assert attribute_peptide_to_sample_alleles(0, "ANYTHING") == frozenset()
    assert attribute_peptide_to_sample_alleles(31844290, "") == frozenset()
