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
    # Issue #146: supplementary rows must carry assay_iri (synthesized
    # from PMID + peptide + restriction, row-unique within a PMID).
    assert "assay_iri" in df.columns
    assert (df["assay_iri"] != "").all()
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
    # Quantitative-binding-assay parity columns (issue #148, #135).  Supp
    # rows are MS-only so all of these must be present-but-empty so the
    # schema lines up with scanner output and the merged observations
    # parquet stays single-typed.
    for col in (
        "assay_method",
        "response_measured",
        "measurement_units",
        "measurement_inequality",
        "quantitative_measurement",
        "quantitative_value",
    ):
        assert col in df.columns, f"missing column: {col}"
    assert (df["response_measured"] == "").all()
    assert df["quantitative_value"].isna().all()

    # Allele-set columns (issue #137).  All three must be populated on
    # every supplementary row.  Most supplementary alleles are 4-digit so
    # provenance should be predominantly "exact".  Empty mhc_allele_set
    # is acceptable for "unmatched" rows but set_size must always be an
    # int so consumers can groupby on it without NaN handling.
    for col in ("mhc_allele_set", "mhc_allele_provenance", "mhc_allele_set_size"):
        assert col in df.columns, f"missing column: {col}"
    assert (
        df["mhc_allele_provenance"]
        .isin({"exact", "sample_allele_match", "pmid_class_pool", "unmatched"})
        .all()
    )
    assert df["mhc_allele_set_size"].notna().all()
    # On "exact" rows, the set is just the row's mhc_restriction.
    exact_mask = df["mhc_allele_provenance"] == "exact"
    if exact_mask.any():
        assert (df.loc[exact_mask, "mhc_allele_set"] == df.loc[exact_mask, "mhc_restriction"]).all()
        assert (df.loc[exact_mask, "mhc_allele_set_size"] == 1).all()


def test_scan_supplementary_gomez_zepeda():
    """Gomez-Zepeda data should include multiple cell lines."""
    df = scan_supplementary()
    gz = df[df["pmid"] == 38480730]
    assert len(gz) > 50000, f"Expected >50000 GZ peptides, got {len(gz)}"

    # All should be class I
    assert set(gz["mhc_class"]) == {"I"}

    # Should have multiple cell lines via different defaults
    cell_names = set(gz["cell_name"].unique())
    assert "JY" in cell_names
    assert "HeLa" in cell_names
    assert "Raji" in cell_names


def test_supplementary_file_column_is_populated():
    """Every supplementary row carries the originating CSV filename (issue #147).

    Needed so peptides seen in multiple sample CSVs from one paper don't
    collapse onto one arbitrary sample context at dedupe time.
    """
    df = scan_supplementary()
    assert "supplementary_file" in df.columns
    assert (df["supplementary_file"] != "").all()


def test_gomez_zepeda_shared_peptide_survives_in_all_sample_files():
    """Peptides presented by multiple Gomez-Zepeda cell lines must keep
    one row per CSV after dedupe (issue #147).  Pre-fix the key was
    (peptide, mhc_restriction, pmid) so shared (peptide, allele) pairs
    across JY / HeLa / Raji collapsed to a single arbitrary row.
    """
    df = scan_supplementary()
    gz = df[df["pmid"] == 38480730]
    assert not gz.empty

    # Group by peptide + restriction; any shared peptide that exists in
    # multiple supplementary files must have N rows, not 1.
    per_pep = gz.groupby(["peptide", "mhc_restriction"])["supplementary_file"].nunique()
    shared = per_pep[per_pep > 1]
    assert not shared.empty, (
        "expected at least one (peptide, allele) to appear in multiple "
        "Gomez-Zepeda supplementary files; if that is no longer true the "
        "dedupe fix cannot be regression-tested here"
    )

    # Pick the most-shared pair and confirm each file contributes a row.
    top_pep, top_allele = shared.idxmax()
    rows = gz[(gz["peptide"] == top_pep) & (gz["mhc_restriction"] == top_allele)]
    assert rows["supplementary_file"].nunique() == rows.shape[0], (
        "expected one row per (peptide, allele, file) — dedupe collapsed "
        "rows that belong to distinct samples"
    )
    # assay_iri must be row-unique now, even with same peptide+allele.
    assert rows["assay_iri"].nunique() == rows.shape[0]


def test_gomez_zepeda_raji_sample_has_exact_alleles():
    """PMID 38480730 Raji sample entry should carry the curated HLA genotype
    (A*03:01 / B*15:10 / C*03:04 / C*04:01) so Raji supplementary rows
    with those exact alleles join to a real sample instead of the class
    pool (issue #147).
    """
    from hitlist.curation import load_pmid_overrides

    gz = load_pmid_overrides()[38480730]
    raji_samples = [
        s
        for s in gz["ms_samples"]
        if "Raji" in s["sample_label"] and "spike" not in s["sample_label"].lower()
    ]
    assert len(raji_samples) == 1
    mhc = raji_samples[0]["mhc"]
    for allele in ("HLA-A*03:01", "HLA-B*15:10", "HLA-C*03:04", "HLA-C*04:01"):
        assert allele in mhc, f"missing {allele} in Raji sample mhc"


def test_gomez_zepeda_plasma_sample_exists():
    """PMID 38480730 must carry a plasma sample entry so the 973 predicted-
    allele rows in the plasma supplementary CSV can attach sample context
    (even if donor HLA is unknown → class-only restriction).
    """
    from hitlist.curation import load_pmid_overrides

    gz = load_pmid_overrides()[38480730]
    plasma = [s for s in gz["ms_samples"] if "plasma" in s["sample_label"].lower()]
    assert len(plasma) == 1
    assert plasma[0]["mhc_class"] == "I"


def test_gomez_zepeda_src_flags_by_cell_line():
    """Existing src_ebv_lcl / src_cancer flags must still be set correctly
    after the #147 dedupe + curation changes.
    """
    df = scan_supplementary()
    gz = df[df["pmid"] == 38480730]

    # JY rows should be EBV-LCL, not cancer.
    jy = gz[gz["cell_name"] == "JY"]
    assert len(jy) > 15000
    row = jy.iloc[0]
    assert row["src_ebv_lcl"] is True or row["src_ebv_lcl"] == True  # noqa: E712
    assert row["src_cancer"] is False or row["src_cancer"] == False  # noqa: E712

    # HeLa should be cancer.
    hela = gz[gz["cell_name"] == "HeLa"]
    assert len(hela) > 5000
    assert hela.iloc[0]["src_cancer"] is True or hela.iloc[0]["src_cancer"] == True  # noqa: E712


def test_scan_supplementary_strazar():
    """Stražar 2023 should load as class II mono-allelic Expi293F data."""
    df = scan_supplementary()
    st = df[df["pmid"] == 37301199]
    assert len(st) == 308418
    assert set(st["mhc_class"]) == {"II"}
    assert st["mhc_restriction"].nunique() == 42
    assert st["peptide"].nunique() == 176962
    assert (st["cell_name"] == "Expi293F").all()
    assert (st["is_monoallelic"]).all()
    assert set(st["monoallelic_host"].unique()) == {"Strep-tag II"}
    assert (st["mhc_species"] == "Homo sapiens").all()


def test_scan_supplementary_contaminant_flag():
    """is_potential_contaminant should be present and meaningful."""
    df = scan_supplementary()
    assert "is_potential_contaminant" in df.columns
    gz = df[df["pmid"] == 38480730]
    # Should have both True and False values
    assert gz["is_potential_contaminant"].any(), "No contaminants flagged"
    assert not gz["is_potential_contaminant"].all(), "All flagged as contaminants"
    # Contaminants should have empty mhc_restriction
    contams = gz[gz["is_potential_contaminant"]]
    assert (contams["mhc_restriction"] == "").all(), "Contaminants should have no allele"


def test_scan_supplementary_synthetic_iri():
    """Supplementary rows should have synthetic reference IRIs."""
    df = scan_supplementary()
    for iri in df["reference_iri"].head(10):
        assert iri.startswith("supplement:"), f"Expected supplement: prefix, got {iri}"


def test_scan_supplementary_dedup_within():
    """No duplicate rows within a single supplementary file.

    Issue #147: the dedupe key used to be ``(peptide, mhc_restriction, pmid)``,
    which collapsed the same peptide / allele seen in multiple sample CSVs
    under one PMID.  Now the key includes ``supplementary_file``, so
    within one CSV no duplicates remain but the *same* pair can still
    appear in distinct CSVs (one row per sample).
    """
    df = scan_supplementary()
    dupes = df.duplicated(subset=["peptide", "mhc_restriction", "pmid", "supplementary_file"])
    assert not dupes.any(), f"Found {dupes.sum()} within-file duplicate rows"


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


def test_supplement_mhc_species_fallback_pattern_handles_categorical():
    """Regression for the v1.30.41 categorical-fillna fix in
    ``hitlist/supplement.py``.  ``scan_supplementary`` runs::

        record["mhc_species"] = (
            record["mhc_species"].astype("string").fillna("").replace("", host_species)
        )

    on a column that may be categorical post-#137 if the input frame
    was compressed upstream.  Without the ``astype("string")`` cast,
    fillna with ``""`` fails when ``""`` isn't already in the
    category set::

        TypeError: Cannot setitem on a Categorical with a new category (),
        set the categories first

    The current production path doesn't compress before this fillna
    runs, but the fix is defensive — if categorical compression ever
    moves earlier in the pipeline (or a pre-compressed parquet is
    supplied), the cast keeps the fallback working.

    This is a unit test of the idiom used in supplement.py:291."""
    import pandas as pd

    record = pd.DataFrame(
        {
            "peptide": ["P1", "P2"],
            # Categorical with a NA cell — neither "" nor "Homo sapiens"
            # is in the existing category set.
            "mhc_species": pd.Categorical(["", None]),
        }
    )
    host_species = "Homo sapiens"
    # The exact idiom from supplement.py:
    record["mhc_species"] = (
        record["mhc_species"].astype("string").fillna("").replace("", host_species)
    )
    assert list(record["mhc_species"]) == ["Homo sapiens", "Homo sapiens"]
