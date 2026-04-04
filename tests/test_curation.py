from hitlist.curation import (
    classify_ms_row,
    detect_monoallelic,
    is_cancer_specific,
    load_monoallelic_lines,
    load_pmid_overrides,
    load_tissue_categories,
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
