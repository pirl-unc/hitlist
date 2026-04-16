import pytest

from hitlist.curation import (
    ALLELE_RESOLUTION_ORDER,
    allele_resolution_rank,
    allele_to_all_serotypes,
    allele_to_serotype,
    classify_allele_resolution,
    classify_mhc_species,
    classify_ms_row,
    detect_monoallelic,
    is_cancer_specific,
    load_monoallelic_lines,
    load_pmid_overrides,
    load_tissue_categories,
    normalize_allele,
    normalize_species,
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


# ── Allele resolution ──────────────────────────────────────────────────


def test_classify_allele_resolution_four_digit():
    assert classify_allele_resolution("HLA-A*02:01") == "four_digit"
    assert classify_allele_resolution("HLA-B*49:01") == "four_digit"
    assert classify_allele_resolution("HLA-C*07:02") == "four_digit"


def test_classify_allele_resolution_serological():
    assert classify_allele_resolution("HLA-A2") == "serological"
    assert classify_allele_resolution("HLA-B7") == "serological"


def test_classify_allele_resolution_class_only():
    assert classify_allele_resolution("HLA class I") == "class_only"
    assert classify_allele_resolution("HLA class II") == "class_only"


def test_classify_allele_resolution_unresolved():
    assert classify_allele_resolution("") == "unresolved"


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
    monkeypatch.setattr(
        curation, "_data_path", lambda fn: str(legacy_yaml) if fn == "pmid_overrides.yaml" else fn
    )
    curation.load_pmid_overrides.cache_clear()
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", DeprecationWarning)
            curation.load_pmid_overrides()
        messages = [str(x.message) for x in w if issubclass(x.category, DeprecationWarning)]
        assert any("label:" in m for m in messages), messages
        assert any("type:" in m for m in messages), messages
    finally:
        curation.load_pmid_overrides.cache_clear()


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
