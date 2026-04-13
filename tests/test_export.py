from hitlist.export import (
    _classify_instrument,
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
        "mhc",
        "ip_antibody",
        "acquisition_mode",
        "instrument",
        "instrument_type",
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


def test_classify_instrument():
    assert _classify_instrument("Q Exactive HF-X") == "Orbitrap"
    assert _classify_instrument("Orbitrap Fusion Lumos") == "Orbitrap"
    assert _classify_instrument("LTQ Orbitrap Elite") == "Orbitrap"
    assert _classify_instrument("timsTOF Pro") == "timsTOF"
    assert _classify_instrument("TripleTOF 5600") == "TOF"
    assert _classify_instrument("TSQ Altis Plus") == "QqQ"
    assert _classify_instrument("") == ""
    assert _classify_instrument("Some Novel Instrument") == "Some Novel Instrument"


def test_ms_samples_instrument_type():
    """instrument_type should be derived from instrument."""
    df = generate_ms_samples_table()
    qe = df[df["instrument"] == "Q Exactive"]
    assert len(qe) > 0
    assert qe.iloc[0]["instrument_type"] == "Orbitrap"


def test_ms_samples_two_level_inheritance():
    """Per-sample ip_antibody should override PMID-level default."""
    df = generate_ms_samples_table()
    # Abelin 2019 has MAPTAC (streptavidin) and tissue (L243) samples
    abelin = df[df["pmid"] == 31495665]
    if len(abelin) > 1:
        antibodies = set(abelin["ip_antibody"].dropna())
        # Should have at least 2 distinct ip_antibody values
        assert len(antibodies) > 1


def test_ms_samples_mhc_per_sample():
    """Sarkizova validation samples should have per-sample MHC alleles."""
    df = generate_ms_samples_table()
    sarkizova = df[df["pmid"] == 31844290]
    cll_a = sarkizova[sarkizova["sample"].str.contains("CLL A")]
    assert len(cll_a) == 1
    mhc = cll_a.iloc[0]["mhc"]
    assert "A*03:01" in mhc
    assert "B*14:02" in mhc


def test_ms_samples_mhc_unknown():
    """Pat9 ccRCC has no MHC typing — mhc should be 'unknown'."""
    df = generate_ms_samples_table()
    sarkizova = df[df["pmid"] == 31844290]
    pat9 = sarkizova[sarkizova["sample"].str.contains("Pat9")]
    assert len(pat9) == 1
    assert pat9.iloc[0]["mhc"] == "unknown"


def test_generate_observations_table():
    """Observations table should join peptides with sample metadata."""
    from hitlist.observations import is_built

    if not is_built():
        import pytest

        pytest.skip("Observations table not built")
    from hitlist.export import generate_observations_table

    df = generate_observations_table()
    assert len(df) > 0
    # Original observation columns
    assert "peptide" in df.columns
    assert "mhc_restriction" in df.columns
    # Enriched sample metadata columns
    assert "instrument" in df.columns
    assert "instrument_type" in df.columns
    assert "sample_mhc" in df.columns
    assert "quantification_method" in df.columns


def test_generate_observations_monoallelic_filter():
    """--mono-allelic / --multi-allelic should change row counts."""
    from hitlist.observations import is_built

    if not is_built():
        import pytest

        pytest.skip("Observations table not built")
    from hitlist.export import generate_observations_table

    df_all = generate_observations_table()
    df_mono = generate_observations_table(is_mono_allelic=True)
    df_multi = generate_observations_table(is_mono_allelic=False)
    # Both subsets should be non-empty and smaller than the full table
    assert len(df_mono) > 0, "Mono-allelic filter returned no rows"
    assert len(df_multi) > 0, "Multi-allelic filter returned no rows"
    assert len(df_mono) < len(df_all), "Mono-allelic filter did not reduce row count"
    assert len(df_multi) < len(df_all), "Multi-allelic filter did not reduce row count"


def test_generate_observations_provenance_columns():
    """Provenance columns should be present and meaningful."""
    from hitlist.observations import is_built

    if not is_built():
        import pytest

        pytest.skip("Observations table not built")
    from hitlist.export import generate_observations_table

    df = generate_observations_table()
    assert "sample_match_type" in df.columns
    assert "matched_sample_count" in df.columns
    assert "has_peptide_level_allele" in df.columns
    # Match types should only be these values
    valid_types = {"allele_match", "single_sample_fallback", "pmid_class_pool", "unmatched"}
    assert set(df["sample_match_type"].unique()).issubset(valid_types)
    # Most rows with alleles should have allele_match
    has_allele = df[df["has_peptide_level_allele"]]
    if len(has_allele) > 0:
        allele_matched = (has_allele["sample_match_type"] == "allele_match").sum()
        assert allele_matched > 0, "No rows have allele_match despite having alleles"


def test_generate_observations_parquet_export(tmp_path):
    """Parquet export path should produce a readable file."""
    from hitlist.observations import is_built

    if not is_built():
        import pytest

        pytest.skip("Observations table not built")
    from hitlist.export import generate_observations_table

    df = generate_observations_table()
    out = tmp_path / "obs.parquet"
    df.to_parquet(out, index=False)
    import pandas as pd

    loaded = pd.read_parquet(out)
    assert len(loaded) == len(df)
    assert set(loaded.columns) == set(df.columns)


def test_generate_observations_table_not_built():
    """Should raise FileNotFoundError when observations not built."""
    from hitlist.observations import is_built

    if is_built():
        import pytest

        pytest.skip("Observations table is built — cannot test error path")
    import pytest

    from hitlist.export import generate_observations_table

    with pytest.raises(FileNotFoundError, match="not built"):
        generate_observations_table()


def test_ms_samples_species_normalized():
    """Species column should contain canonical names, not parenthetical IEDB format."""
    df = generate_ms_samples_table()
    for s in df["species"]:
        assert "(" not in s, f"Species not normalized: {s}"


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


def test_observations_join_with_synthetic_fixture(tmp_path, monkeypatch):
    """End-to-end join test using a small synthetic observations table."""
    import pandas as pd

    from hitlist.export import generate_observations_table

    # Build a tiny synthetic observations parquet
    obs_data = pd.DataFrame(
        {
            "peptide": ["AAAAAAAAA", "BBBBBBBBB", "CCCCCCCCC"],
            "mhc_restriction": ["HLA-A*02:01", "HLA-B*07:02", ""],
            "mhc_class": ["I", "I", "I"],
            "reference_iri": ["iri:1", "iri:2", "iri:3"],
            "pmid": pd.array([33858848, 33858848, 38480730], dtype="Int64"),
            "source": ["iedb", "iedb", "supplement"],
            "mhc_species": ["Homo sapiens", "Homo sapiens", "Homo sapiens"],
            "is_monoallelic": [False, False, False],
            "is_binding_assay": [False, False, False],
            "qualitative_measurement": ["Positive", "Positive", ""],
        }
    )
    obs_path = tmp_path / "observations.parquet"
    obs_data.to_parquet(obs_path, index=False)

    # Monkeypatch the observations path
    monkeypatch.setattr("hitlist.observations.observations_path", lambda: obs_path)

    df = generate_observations_table()
    assert len(df) == 3
    # Provenance columns should be present
    assert "sample_match_type" in df.columns
    assert "matched_sample_count" in df.columns
    assert "has_peptide_level_allele" in df.columns
    # Third row has no allele — flag should reflect that
    row3 = df[df["peptide"] == "CCCCCCCCC"].iloc[0]
    assert row3["has_peptide_level_allele"] is False or row3["has_peptide_level_allele"] == False  # noqa: E712


def test_generate_observations_gene_filter_requires_flanking(tmp_path, monkeypatch):
    """Using --gene on a non-flanking parquet should error with a clear message."""
    import pandas as pd
    import pytest

    from hitlist.export import generate_observations_table

    obs_data = pd.DataFrame(
        {
            "peptide": ["AAAAAAAAA"],
            "mhc_restriction": ["HLA-A*02:01"],
            "mhc_class": ["I"],
            "reference_iri": ["iri:1"],
            "pmid": pd.array([33858848], dtype="Int64"),
            "source": ["iedb"],
            "mhc_species": ["Homo sapiens"],
            "is_monoallelic": [False],
            "is_binding_assay": [False],
            "qualitative_measurement": ["Positive"],
        }
    )
    obs_path = tmp_path / "observations.parquet"
    obs_data.to_parquet(obs_path, index=False)
    monkeypatch.setattr("hitlist.observations.observations_path", lambda: obs_path)

    with pytest.raises(ValueError, match="flanking-built"):
        generate_observations_table(gene="PRAME")


def test_generate_observations_gene_filter_matches(tmp_path, monkeypatch):
    """--gene-name on a flanking-built table filters correctly."""
    import pandas as pd

    from hitlist.export import generate_observations_table

    obs_data = pd.DataFrame(
        {
            "peptide": ["AAAAAAAAA", "BBBBBBBBB", "CCCCCCCCC"],
            "mhc_restriction": ["HLA-A*02:01", "HLA-B*07:02", "HLA-A*02:01"],
            "mhc_class": ["I", "I", "I"],
            "reference_iri": ["iri:1", "iri:2", "iri:3"],
            "pmid": pd.array([33858848, 33858848, 33858848], dtype="Int64"),
            "source": ["iedb", "iedb", "iedb"],
            "mhc_species": ["Homo sapiens", "Homo sapiens", "Homo sapiens"],
            "is_monoallelic": [False, False, False],
            "is_binding_assay": [False, False, False],
            "qualitative_measurement": ["Positive", "Positive", "Positive"],
            "gene_name": ["PRAME", "MAGEA1", "PRAME"],
            "gene_id": ["ENSG00000185686", "ENSG00000198681", "ENSG00000185686"],
        }
    )
    obs_path = tmp_path / "observations.parquet"
    obs_data.to_parquet(obs_path, index=False)
    monkeypatch.setattr("hitlist.observations.observations_path", lambda: obs_path)

    # By gene_name
    df = generate_observations_table(gene_name="PRAME")
    assert len(df) == 2
    assert set(df["peptide"]) == {"AAAAAAAAA", "CCCCCCCCC"}

    # By gene_id
    df = generate_observations_table(gene_id="ENSG00000198681")
    assert len(df) == 1
    assert df.iloc[0]["peptide"] == "BBBBBBBBB"

    # Via --gene with HGNC disabled (direct symbol match still works)
    df = generate_observations_table(gene="MAGEA1")
    assert len(df) == 1

    # Comma-separated
    df = generate_observations_table(gene_name="PRAME,MAGEA1")
    assert len(df) == 3


def test_generate_observations_mhc_allele_filter(tmp_path, monkeypatch):
    """--mhc-allele should filter exact allele matches (after normalization)."""
    import pandas as pd

    from hitlist.export import generate_observations_table

    obs_data = pd.DataFrame(
        {
            "peptide": ["AAA", "BBB", "CCC"],
            "mhc_restriction": ["HLA-A*02:01", "HLA-B*07:02", "HLA-A*02:01"],
            "mhc_class": ["I", "I", "I"],
            "reference_iri": ["1", "2", "3"],
            "pmid": pd.array([33858848] * 3, dtype="Int64"),
            "source": ["iedb"] * 3,
            "mhc_species": ["Homo sapiens"] * 3,
            "is_monoallelic": [False] * 3,
            "is_binding_assay": [False] * 3,
            "qualitative_measurement": ["Positive"] * 3,
        }
    )
    obs_path = tmp_path / "observations.parquet"
    obs_data.to_parquet(obs_path, index=False)
    monkeypatch.setattr("hitlist.observations.observations_path", lambda: obs_path)

    df = generate_observations_table(mhc_allele="HLA-A*02:01")
    assert len(df) == 2
    assert set(df["peptide"]) == {"AAA", "CCC"}

    df = generate_observations_table(mhc_allele=["HLA-A*02:01", "HLA-B*07:02"])
    assert len(df) == 3
