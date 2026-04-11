from hitlist.supplement import (
    load_supplementary_manifest,
    scan_supplementary,
)


def test_load_manifest():
    """Manifest should load and have at least one entry."""
    entries = load_supplementary_manifest()
    assert len(entries) > 0
    entry = entries[0]
    assert "pmid" in entry
    assert "file" in entry
    assert "defaults" in entry


def test_load_manifest_structure():
    """Each manifest entry should have required fields."""
    entries = load_supplementary_manifest()
    for entry in entries:
        assert isinstance(entry["pmid"], int)
        assert isinstance(entry["file"], str)
        defaults = entry["defaults"]
        assert "culture_condition" in defaults
        assert "cell_name" in defaults


def test_scan_supplementary_not_empty():
    """Scanning supplementary data should produce rows."""
    df = scan_supplementary()
    assert len(df) > 0


def test_scan_supplementary_schema():
    """Supplementary scan output should match scanner output schema."""
    df = scan_supplementary()
    # Core peptide columns
    assert "peptide" in df.columns
    assert "mhc_class" in df.columns
    assert "pmid" in df.columns
    assert "reference_iri" in df.columns
    # Source classification columns (from classify_ms_row)
    assert "src_cancer" in df.columns
    assert "src_cell_line" in df.columns
    assert "src_ebv_lcl" in df.columns
    assert "is_monoallelic" in df.columns
    assert "allele_resolution" in df.columns
    assert "mhc_species" in df.columns
    # IEDB-equivalent metadata columns
    assert "process_type" in df.columns
    assert "culture_condition" in df.columns
    assert "cell_name" in df.columns


def test_scan_supplementary_gomez_zepeda():
    """Gomez-Zepeda JY peptides should be classified correctly."""
    df = scan_supplementary()
    gz = df[df["pmid"] == 38480730]
    assert len(gz) > 1000, f"Expected >1000 JY peptides, got {len(gz)}"

    # All should be class I
    assert set(gz["mhc_class"]) == {"I"}

    # JY is an EBV-LCL, not cancer
    row = gz.iloc[0]
    assert row["src_ebv_lcl"] is True or row["src_ebv_lcl"] == True  # noqa: E712
    assert row["src_cancer"] is False or row["src_cancer"] == False  # noqa: E712
    assert row["src_cell_line"] is True or row["src_cell_line"] == True  # noqa: E712
    assert row["cell_name"] == "JY" or row["cell_line_name"] == "JY"


def test_scan_supplementary_synthetic_iri():
    """Supplementary rows should have synthetic reference IRIs."""
    df = scan_supplementary()
    for iri in df["reference_iri"].head(10):
        assert iri.startswith("supplement:"), f"Expected supplement: prefix, got {iri}"


def test_scan_supplementary_dedup_within():
    """No duplicate (peptide, mhc_restriction, pmid) within supplementary data."""
    df = scan_supplementary()
    dupes = df.duplicated(subset=["peptide", "mhc_restriction", "pmid"])
    assert not dupes.any(), f"Found {dupes.sum()} duplicate rows"


def test_scan_supplementary_no_classify():
    """scan_supplementary(classify_source=False) should skip classification."""
    df = scan_supplementary(classify_source=False)
    assert len(df) > 0
    # Should have allele_resolution but not src_cancer
    assert "allele_resolution" in df.columns
    assert "src_cancer" not in df.columns


def test_scan_supplementary_mhc_species_propagation():
    """Supplementary rows should have mhc_species even without allele assignment."""
    df = scan_supplementary()
    gz = df[df["pmid"] == 38480730]
    # Rows without mhc_restriction should still have mhc_species from host
    no_allele = gz[gz["mhc_restriction"] == ""]
    assert len(no_allele) > 0, "Expected some peptides without allele assignment"
    assert (no_allele["mhc_species"] == "Homo sapiens").all(), (
        "Peptides without allele assignment should still have mhc_species='Homo sapiens'"
    )
