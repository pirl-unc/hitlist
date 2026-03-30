from hitlist.curation import (
    classify_ms_row,
    is_cancer_specific,
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
