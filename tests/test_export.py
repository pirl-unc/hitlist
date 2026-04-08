from hitlist.export import (
    _extract_allele_strings,
    generate_ms_samples_table,
    generate_species_summary,
    validate_mhc_alleles,
)


def test_ms_samples_table_columns():
    df = generate_ms_samples_table()
    expected = {
        "species",
        "sample",
        "perturbation",
        "pmid",
        "study",
        "mhc_class",
        "n_samples",
        "notes",
        "ip_antibody",
        "acquisition_mode",
        "instrument",
        "fragmentation",
        "labeling",
        "search_engine",
        "fdr",
    }
    assert expected == set(df.columns)


def test_ms_samples_table_not_empty():
    df = generate_ms_samples_table()
    assert len(df) > 0


def test_ms_samples_filter_class_i():
    df_all = generate_ms_samples_table()
    df_i = generate_ms_samples_table(mhc_class="I")
    assert len(df_i) > 0
    assert len(df_i) < len(df_all)
    for cls in df_i["mhc_class"]:
        assert "I" in cls


def test_ms_samples_filter_class_ii():
    df_ii = generate_ms_samples_table(mhc_class="II")
    assert len(df_ii) > 0
    for cls in df_ii["mhc_class"]:
        assert "II" in cls


def test_ms_samples_no_zero_n():
    """Placeholder rows with n=0 should be excluded."""
    df = generate_ms_samples_table()
    for n in df["n_samples"].dropna():
        assert n > 0


def test_ms_samples_acquisition_metadata():
    """Spot-check that acquisition metadata is populated for curated studies."""
    df = generate_ms_samples_table()
    # Mommen 2014 — EThcD fragmentation
    mommen = df[df["pmid"] == 24616531]
    assert len(mommen) > 0
    assert mommen.iloc[0]["fragmentation"] == "EThcD"
    assert mommen.iloc[0]["ip_antibody"] == "W6/32"
    # Ritz 2017b — DIA acquisition
    ritz = df[df["pmid"] == 28834231]
    assert len(ritz) > 0
    assert ritz.iloc[0]["acquisition_mode"] == "DIA"
    # Pfammatter 2020 — TMT labeling
    pfammatter = df[df["pmid"] == 32502341]
    assert len(pfammatter) > 0
    assert pfammatter.iloc[0]["labeling"] == "TMT"


def test_ms_samples_two_level_inheritance():
    """Per-sample ip_antibody should override PMID-level default."""
    df = generate_ms_samples_table()
    # Abelin 2019 has MAPTAC (streptavidin) and tissue (L243) samples
    abelin = df[df["pmid"] == 31495665]
    if len(abelin) > 1:
        antibodies = set(abelin["ip_antibody"].dropna())
        # Should have at least 2 distinct ip_antibody values
        assert len(antibodies) > 1


def test_species_summary_columns():
    df = generate_species_summary()
    expected = {
        "species",
        "mhc_class",
        "n_studies",
        "n_sample_types",
        "n_samples",
    }
    assert expected == set(df.columns)


def test_species_summary_has_multiple_species():
    df = generate_species_summary()
    assert df["species"].nunique() > 1


def test_species_summary_class_filter():
    df_i = generate_species_summary(mhc_class="I")
    assert len(df_i) > 0
    assert set(df_i["mhc_class"]) == {"I"}


def test_validate_alleles_columns():
    df = validate_mhc_alleles()
    expected = {"pmid", "study", "allele", "parsed_name", "parsed_type", "species", "valid"}
    assert expected == set(df.columns)


def test_validate_alleles_most_valid():
    df = validate_mhc_alleles()
    if len(df) > 0:
        valid_pct = df["valid"].mean()
        assert valid_pct > 0.5, f"Only {valid_pct:.0%} of alleles parsed successfully"


def test_extract_allele_strings_list():
    result = _extract_allele_strings(["HLA-A*02:01", "HLA-B*07:02"])
    assert result == ["HLA-A*02:01", "HLA-B*07:02"]


def test_extract_allele_strings_dict():
    result = _extract_allele_strings({"donor_1": ["HLA-A*02:01"], "donor_2": ["HLA-B*07:02"]})
    assert "HLA-A*02:01" in result
    assert "HLA-B*07:02" in result


def test_extract_allele_strings_description():
    result = _extract_allele_strings("51 HLA-I allotypes (95% coverage)")
    assert result == []
