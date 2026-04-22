from hitlist.bulk_proteomics import (
    available_cell_lines,
    available_peptide_cell_lines,
    available_protein_cell_lines,
    is_bulk_proteomics_built,
    load_bulk_peptides,
    load_bulk_proteomics,
    load_bulk_sources,
)

# Harmonized acquisition metadata columns — same names appear in the
# ms_samples schema emitted by hitlist.export for observations.parquet,
# so downstream MS-bias code can read both indexes with one column list.
_HARMONIZED_ACQUISITION_COLS = {
    "pmid",
    "reference",
    "study_label",
    "species",
    "cell_line_name",
    "sample_label",
    "instrument",
    "instrument_type",
    "fragmentation",
    "acquisition_mode",
    "labeling",
    "search_engine",
    "fdr",
}

_BULK_PREP_COLS = {
    "digestion",
    "digestion_enzyme",
    "fractionation",
    "n_fractions",
    "quantification",
}

# Per-row Fig 1b design-matrix axes added in v1.11.3 for Bekker-Jensen
# (CCLE rows carry sensible defaults: enrichment="none", n_fractions_in_run=NA).
# v1.14.1 (#98) added fractionation_ph.
_BJ_PER_ROW_AXES = {
    "digestion_enzyme",
    "n_fractions_in_run",
    "enrichment",
    "fractionation_ph",
    "modifications",
}


def test_available_cell_lines():
    cells = available_cell_lines()
    # Union across both indices
    assert "MDA-MB-231" in cells
    assert "HCT116" in cells
    assert "A549" in cells
    assert "K562" in cells
    assert "THP-1" in cells
    assert "MCF7" in cells
    assert "Jurkat" in cells
    # Peptide-only lines show up in the union
    assert "HeLa" in cells
    assert "HEK293" in cells


def test_available_protein_cell_lines():
    cells = available_protein_cell_lines()
    assert "MDA-MB-231" in cells
    assert "HCT116" in cells
    # HeLa and HEK293 are now covered via Bekker-Jensen-derived abundance
    assert "HeLa" in cells
    assert "HEK293" in cells


def test_available_peptide_cell_lines():
    cells = available_peptide_cell_lines()
    assert set(cells) == {"A549", "HCT116", "HEK293", "HeLa", "MCF7"}


def test_load_bulk_proteomics_full():
    df = load_bulk_proteomics()
    assert len(df) > 50_000, f"expected >50K rows across 7 cell lines, got {len(df)}"
    expected_cols = {
        "cell_line_name",
        "gene_symbol",
        "uniprot_acc",
        "source",
        "reference",
        "abundance_percentile",
    }
    assert expected_cols.issubset(df.columns)
    # Default load is the UNION of CCLE + Bekker-Jensen protein-level
    assert set(df["source"]) == {"CCLE_Nusinow_2020", "Bekker-Jensen_2017"}


def test_load_bulk_proteomics_filter_cell_line():
    df = load_bulk_proteomics(cell_line="MDA-MB-231")
    assert len(df) > 5_000
    assert set(df["cell_line_name"]) == {"MDA-MB-231"}
    # Case-insensitive match
    df2 = load_bulk_proteomics(cell_line="mda-mb-231")
    assert len(df2) == len(df)


def test_load_bulk_proteomics_filter_gene():
    df = load_bulk_proteomics(gene_name="TP53")
    assert len(df) >= 5, f"TP53 should be detected in most cell lines, got {len(df)}"
    assert set(df["gene_symbol"]) == {"TP53"}


def test_load_bulk_proteomics_combined_filter():
    df = load_bulk_proteomics(cell_line="HCT116", gene_name=["KRAS", "TP53"])
    assert set(df["cell_line_name"]) == {"HCT116"}
    assert set(df["gene_symbol"]).issubset({"KRAS", "TP53"})


def test_load_bulk_peptides_full():
    df = load_bulk_peptides()
    assert len(df) > 500_000, f"expected >500K peptide rows across 5 cell lines, got {len(df)}"
    expected_cols = {
        "peptide",
        "cell_line_name",
        "uniprot_acc",
        "gene_symbol",
        "length",
        "start_position",
        "end_position",
        "source",
        "reference",
    }
    assert expected_cols.issubset(df.columns)
    assert set(df["source"]) == {"Bekker-Jensen_2017"}


def test_load_bulk_peptides_mixed_digest():
    """Peptide index now includes HeLa non-tryptic arms (Chymo/GluC/LysC)."""
    df = load_bulk_peptides()
    # digestion_enzyme is per-row when the parquet is built; when the
    # loader falls back to the raw CSV the column is still present.
    assert "digestion_enzyme" in df.columns
    enzymes = set(df["digestion_enzyme"].dropna().unique())
    # All four digests should appear
    assert any(e.startswith("Trypsin") for e in enzymes)
    assert "Chymotrypsin" in enzymes
    assert "GluC" in enzymes
    assert "LysC" in enzymes
    # Non-tryptic arms are HeLa-only per the deposit design
    hela_only = df[df["digestion_enzyme"].isin({"Chymotrypsin", "GluC", "LysC"})]
    assert set(hela_only["cell_line_name"]) == {"HeLa"}
    # Other cell lines are strictly tryptic
    other_lines = df[df["cell_line_name"].isin({"A549", "HCT116", "HEK293", "MCF7"})]
    assert (other_lines["digestion_enzyme"].str.startswith("Trypsin")).all()
    # Enzyme-specific C-terminal residue sanity checks. Tighter bounds
    # than the original test — we expect ~99% specificity for LysC/GluC
    # (single-residue specificity, reflecting the MaxQuant cleavage
    # rule) and ~92% for Chymotrypsin (F/W/Y/L/M) and ~99% K/R for
    # Trypsin/P.
    gluc = df[df["digestion_enzyme"] == "GluC"]
    # GluC cleaves after E (and D, in bicarbonate buffer — both appear).
    gluc_pct = gluc["peptide"].str[-1].isin(["E", "D"]).mean()
    assert gluc_pct > 0.95, f"GluC C-term E/D specificity {gluc_pct:.3f} < 0.95"
    assert (gluc["peptide"].str[-1] == "E").mean() > 0.9
    lysc = df[df["digestion_enzyme"] == "LysC"]
    lysc_pct = (lysc["peptide"].str[-1] == "K").mean()
    assert lysc_pct > 0.97, f"LysC C-term K specificity {lysc_pct:.3f} < 0.97"
    chymo = df[df["digestion_enzyme"] == "Chymotrypsin"]
    # Chymotrypsin cleaves C-term of F/W/Y/L/M; this covers >90% of the peptides.
    chymo_pct = chymo["peptide"].str[-1].isin(list("FWYLM")).mean()
    assert chymo_pct > 0.9, f"Chymotrypsin C-term F/W/Y/L/M specificity {chymo_pct:.3f} < 0.9"
    tryp = df[df["digestion_enzyme"].str.startswith("Trypsin")]
    tryp_pct = tryp["peptide"].str[-1].isin(["K", "R"]).mean()
    assert tryp_pct > 0.97, f"Trypsin C-term K/R specificity {tryp_pct:.3f} < 0.97"


def test_load_bulk_peptides_fig1b_axes_complete():
    """The Fig 1b fractionation sweep (14/39/46/70) must all be present.

    Adds the 12-frac and 50-frac TiO2 arms (not part of the classical
    Fig 1b sweep axis but present in the same deposit) when enrichment=None.
    """
    # Default loader is non-enriched; that covers 14/39/46/70 on HeLa.
    df = load_bulk_peptides()
    fracs_none = set(df["n_fractions_in_run"].dropna().unique())
    assert {14, 39, 46, 70}.issubset(fracs_none), (
        f"missing Fig 1b fractionation depths in non-enriched arm: got {fracs_none}"
    )
    # Union loader exposes the TiO2 12-frac + 50-frac phospho arms.
    df_all = load_bulk_peptides(enrichment=None)
    fracs_all = set(df_all["n_fractions_in_run"].dropna().unique())
    assert fracs_all == {12, 14, 39, 46, 50, 70}, (
        f"authoritative fractionation set from peptides.txt header = "
        f"{{12,14,39,46,50,70}}; loader returned {fracs_all}"
    )
    # Both enrichment populations round-trip through the parquet.
    enrichments = set(df_all["enrichment"].dropna().unique())
    assert enrichments == {"none", "TiO2"}, enrichments


def test_load_bulk_peptides_default_filters_out_tio2():
    """Default load_bulk_peptides() excludes TiO2-enriched rows.

    This is the "baseline detectability" default: callers opt into
    TiO2 phospho rows by passing ``enrichment="TiO2"`` or ``enrichment=None``.
    """
    df = load_bulk_peptides()
    # Column must be present and must carry only "none"
    assert "enrichment" in df.columns
    assert (df["enrichment"] == "none").all(), (
        f"default load_bulk_peptides() must not include TiO2 rows: "
        f"got {df['enrichment'].value_counts().to_dict()}"
    )


def test_load_bulk_peptides_enrichment_opt_in():
    """Explicit enrichment="TiO2" returns only phospho rows, mostly phospho-modified."""
    df = load_bulk_peptides(enrichment="TiO2")
    assert len(df) > 10_000, f"expected >10K TiO2 rows, got {len(df)}"
    assert set(df["enrichment"]) == {"TiO2"}
    # The TiO2 enrichment step is >60% effective at retaining phospho
    # peptides (published spec is ~75-80%; 50% is a conservative
    # regression floor that still distinguishes the TiO2 rows from
    # baseline where <10% carry phospho).
    phospho_pct = df["modifications"].str.contains("Phospho", na=False).mean()
    assert phospho_pct > 0.5, f"TiO2 rows should be >50% phospho-modified; got {phospho_pct:.2%}"


def test_load_bulk_peptides_enrichment_union():
    """enrichment=None returns both populations (baseline + TiO2)."""
    default = load_bulk_peptides()
    tio2 = load_bulk_peptides(enrichment="TiO2")
    both = load_bulk_peptides(enrichment=None)
    assert len(both) == len(default) + len(tio2), (
        f"union should equal sum of partitions: "
        f"default={len(default):,} + tio2={len(tio2):,} != both={len(both):,}"
    )
    assert set(both["enrichment"]) == {"none", "TiO2"}


def test_load_bulk_peptides_filter_by_enzyme():
    """digestion_enzyme filter narrows rows to the requested digest."""
    lysc = load_bulk_peptides(digestion_enzyme="LysC")
    assert len(lysc) > 50_000
    assert set(lysc["digestion_enzyme"]) == {"LysC"}
    # LysC is HeLa-only
    assert set(lysc["cell_line_name"]) == {"HeLa"}


def test_load_bulk_peptides_filter_by_fractions():
    """n_fractions_in_run filter selects the requested fractionation depth."""
    df70 = load_bulk_peptides(n_fractions_in_run=70)
    assert len(df70) > 100_000
    assert set(df70["n_fractions_in_run"]) == {70}
    # 70-frac arm is HeLa tryptic only
    assert set(df70["cell_line_name"]) == {"HeLa"}
    assert set(df70["digestion_enzyme"]).issubset({"Trypsin/P (cleaves K/R except before P)"})


def test_load_bulk_peptides_fractionation_ph_axis():
    """fractionation_ph column populated; pH 10 default, pH 8 only on Tryp-Phos-pH8."""
    # Non-enriched rows are all pH 10.
    df = load_bulk_peptides()
    ph_values = set(df["fractionation_ph"].dropna().unique())
    assert ph_values == {10.0}, f"non-enriched rows should all be pH 10, got {ph_values}"

    # TiO2 arm splits pH 8 vs pH 10.
    tio2 = load_bulk_peptides(enrichment="TiO2")
    tio2_ph = set(tio2["fractionation_ph"].dropna().unique())
    assert tio2_ph == {8.0, 10.0}, f"TiO2 arm should have both pH 8 and pH 10 arms, got {tio2_ph}"

    # pH=8.0 filter returns only the Tryp-Phos-pH8 arm.
    ph8 = load_bulk_peptides(enrichment="TiO2", fractionation_ph=8.0)
    assert len(ph8) > 10_000
    assert set(ph8["fractionation_ph"]) == {8.0}
    # That arm is HeLa tryptic 12-frac TiO2 per the Fig 1b design.
    assert set(ph8["cell_line_name"]) == {"HeLa"}
    assert set(ph8["enrichment"]) == {"TiO2"}
    assert set(ph8["n_fractions_in_run"]) == {12}
    # Overwhelmingly phospho-modified since it's a phospho enrichment.
    assert ph8["modifications"].str.contains("Phospho", na=False).mean() > 0.6


def test_load_bulk_peptides_fractionation_ph_unpooled_from_v1_14_0():
    """Tryp-Phos-pH8 and Tryp-Phos-pH10 are now distinct arms, not pooled.

    Before v1.14.1 they were pooled into one (TiO2, 12-frac) arm with
    pseudo-replicate labels. Now they're keyed on the ``fractionation_ph``
    axis and sit as separate rows for the same peptide if it was
    detected in both pH conditions.
    """
    tio2_12 = load_bulk_peptides(enrichment="TiO2", n_fractions_in_run=12)
    # Must show both pH values.
    assert set(tio2_12["fractionation_ph"].dropna().unique()) == {8.0, 10.0}
    # A peptide that shows up in both pH arms will appear twice (once
    # per pH). Assert at least one such peptide exists.
    dup_by_pH = tio2_12.groupby("peptide")["fractionation_ph"].nunique()
    assert (dup_by_pH == 2).sum() > 100, (
        "Expect many peptides detected in BOTH Tryp-Phos-pH8 and pH10, "
        "but fewer than 100 peptides have rows at both pH values. "
        "This would indicate the unpooling broke."
    )


def test_load_bulk_proteomics_ccle_has_ph_10():
    """CCLE rows get ``fractionation_ph=10.0`` from the source-level default."""
    ccle = load_bulk_proteomics(source="CCLE_Nusinow_2020")
    assert len(ccle) > 5_000
    assert set(ccle["fractionation_ph"].dropna().unique()) == {10.0}


def test_load_bulk_peptides_tryptic_counts_preserved():
    """Per-cell-line tryptic counts from the prior ingest are preserved.

    Regression check: when the Fig 1b comprehensive ingest added the
    fractionation sweep and TiO2 arms it did NOT disturb the existing
    46-fraction tryptic panel row counts on A549/HCT116/HEK293/MCF7.
    These counts come straight from peptides.txt Intensity <Experiment>
    columns so any change here would indicate a parser regression.
    """
    df = load_bulk_peptides(
        digestion_enzyme="Trypsin/P (cleaves K/R except before P)",
        n_fractions_in_run=46,
    )
    by_cell = df.groupby("cell_line_name").size().to_dict()
    # These numbers were captured from the v1.11.2 tryptic-only CSV.
    expected = {
        "A549": 214925,
        "HCT116": 237738,
        "HEK293": 242564,
        "MCF7": 206130,
    }
    for cell, count in expected.items():
        assert by_cell.get(cell) == count, (
            f"{cell} 46-frac tryptic count changed: expected {count:,}, got {by_cell.get(cell)!r}"
        )


def test_load_bulk_peptides_filter_cell_line():
    df = load_bulk_peptides(cell_line="HeLa")
    assert len(df) > 100_000
    assert set(df["cell_line_name"]) == {"HeLa"}
    # Case-insensitive
    assert len(load_bulk_peptides(cell_line="hela")) == len(df)


def test_load_bulk_peptides_filter_gene():
    df = load_bulk_peptides(gene_name="TP53")
    assert len(df) > 10, f"TP53 should have peptides across ≥3 cell lines, got {len(df)}"
    assert set(df["gene_symbol"]) == {"TP53"}


def test_load_bulk_peptides_intra_protein_bias():
    """Core use case: within-protein peptide detectability differs by cell line."""
    df = load_bulk_peptides(gene_name="TP53")
    per_line = df.groupby("cell_line_name").size()
    # Different cell lines detect different numbers of TP53 peptides
    assert per_line.max() > per_line.min()
    # Positions are populated — needed for intra-protein coverage analysis
    assert df["start_position"].notna().all()
    assert df["end_position"].notna().all()
    assert (df["end_position"] >= df["start_position"]).all()


def test_load_bulk_peptides_filter_uniprot():
    # TP53 → P04637
    df = load_bulk_peptides(uniprot_acc="P04637")
    assert len(df) > 0
    assert set(df["uniprot_acc"]) == {"P04637"}


def test_load_bulk_proteomics_hela_via_bekker_jensen():
    """HeLa abundance is derived from Bekker-Jensen peptide intensities."""
    df = load_bulk_proteomics(cell_line="HeLa")
    assert len(df) > 10_000
    assert set(df["source"]) == {"Bekker-Jensen_2017"}
    # abundance_percentile is populated
    assert df["abundance_percentile"].between(0, 1).all()


def test_load_bulk_proteomics_source_filter():
    ccle = load_bulk_proteomics(cell_line="A549", source="CCLE_Nusinow_2020")
    bj = load_bulk_proteomics(cell_line="A549", source="Bekker-Jensen_2017")
    assert set(ccle["source"]) == {"CCLE_Nusinow_2020"}
    assert set(bj["source"]) == {"Bekker-Jensen_2017"}
    # Both sources independently cover A549
    assert len(ccle) > 5_000
    assert len(bj) > 10_000


def test_load_bulk_sources_shape():
    srcs = load_bulk_sources()
    assert len(srcs) == 2
    ids = {s["source_id"] for s in srcs}
    assert ids == {"CCLE_Nusinow_2020", "Bekker-Jensen_2017"}
    # Every source must declare digestion + instrument + granularity
    for s in srcs:
        assert s["digestion"]
        assert s["instrument"]
        assert s["granularity"] in {"peptide", "protein", "peptide_and_protein"}
        assert isinstance(s["cell_lines_covered"], list)
        assert len(s["cell_lines_covered"]) > 0


def test_load_bulk_sources_digest_is_tryptic():
    """Both sources use tryptic digest as the DOMINANT arm.

    Bekker-Jensen also carries three HeLa-only non-tryptic ancillary
    digests (Chymotrypsin, GluC, LysC) — those are declared via the
    ``ancillary_digests`` field, not the top-level ``digestion``.
    """
    srcs = load_bulk_sources()
    for s in srcs:
        assert s["digestion"] == "tryptic"
    bj = next(s for s in srcs if s["source_id"] == "Bekker-Jensen_2017")
    ancillary = bj.get("ancillary_digests", [])
    enzymes = {a["enzyme"] for a in ancillary}
    assert enzymes == {"Chymotrypsin", "GluC", "LysC"}
    assert all(a.get("included_in_index") is True for a in ancillary)


def test_load_bulk_sources_harmonized_fields():
    """Sources expose the harmonized acquisition fields (matching ms_samples schema)."""
    srcs = load_bulk_sources()
    for s in srcs:
        assert s.get("pmid") and isinstance(s["pmid"], int)
        assert s.get("study_label")
        assert s.get("species")
        assert s.get("fragmentation")
        assert s.get("acquisition_mode")
        assert s.get("labeling")
        assert isinstance(s.get("n_fractions"), int)


def test_build_bulk_proteomics_parquet():
    """build_bulk_proteomics writes a unified long-form parquet."""
    import pandas as pd

    from hitlist.builder import build_bulk_proteomics

    df = build_bulk_proteomics(verbose=False)
    assert is_bulk_proteomics_built()
    assert len(df) > 1_000_000
    # Both granularities present
    assert set(df["granularity"]) == {"protein", "peptide"}
    # Harmonized acquisition columns populated on every row
    missing = _HARMONIZED_ACQUISITION_COLS - set(df.columns)
    assert not missing, f"missing harmonized columns: {missing}"
    bulk_missing = _BULK_PREP_COLS - set(df.columns)
    assert not bulk_missing, f"missing bulk-prep columns: {bulk_missing}"
    # Per-row Fig 1b axes present on the parquet schema
    axes_missing = _BJ_PER_ROW_AXES - set(df.columns)
    assert not axes_missing, f"missing Fig 1b per-row axes: {axes_missing}"
    # Sanity: all rows have populated instrument + digestion.
    # "digestion" is tryptic for the vast majority of rows; the HeLa
    # non-tryptic ancillary arms (Chymotrypsin/GluC/LysC) contribute
    # <25% of rows and are labeled "non-tryptic" on a per-row basis.
    assert (df["instrument"] != "").all()
    digestion_values = set(df["digestion"].unique())
    assert digestion_values.issubset({"tryptic", "non-tryptic"})
    assert (df["digestion"] == "tryptic").mean() > 0.7
    # CCLE rows carry TMT label, BJ rows are label-free
    ccle_rows = df[df["source"] == "CCLE_Nusinow_2020"]
    bj_rows = df[df["source"] == "Bekker-Jensen_2017"]
    assert (ccle_rows["labeling"].str.contains("TMT")).all()
    assert (bj_rows["labeling"] == "label-free").all()
    # pmid is an integer with real PMIDs
    assert set(df["pmid"].dropna().unique()) == {31978347, 28591648}
    # Evidence kind stamped so downstream can filter a unified index
    assert set(df["evidence_kind"]) == {"bulk_proteomics"}
    # --- Fig 1b design matrix completeness assertions ---
    bj_pep = df[(df["source"] == "Bekker-Jensen_2017") & (df["granularity"] == "peptide")]
    bj_fracs = set(bj_pep["n_fractions_in_run"].dropna().astype(int).unique())
    # The four canonical Fig 1b depths must appear; 12 and 50 are the
    # two TiO2-enriched arms from the same deposit.
    assert {14, 39, 46, 70}.issubset(bj_fracs), bj_fracs
    assert bj_fracs == {12, 14, 39, 46, 50, 70}, bj_fracs
    # Enrichment set includes both "none" and "TiO2"
    assert set(bj_pep["enrichment"].dropna().unique()) == {"none", "TiO2"}
    # All four enzymes present
    bj_enzymes = set(bj_pep["digestion_enzyme"].dropna().unique())
    assert bj_enzymes == {
        "Trypsin/P (cleaves K/R except before P)",
        "Chymotrypsin",
        "GluC",
        "LysC",
    }, bj_enzymes
    # Parquet round-trips to the same shape
    from hitlist.bulk_proteomics import bulk_proteomics_path

    roundtrip = pd.read_parquet(bulk_proteomics_path())
    assert roundtrip.shape == df.shape


def test_loaders_read_from_parquet_when_built():
    """Once built, loaders return rows with the harmonized metadata columns."""
    from hitlist.builder import build_bulk_proteomics

    build_bulk_proteomics(verbose=False)
    proteins = load_bulk_proteomics()
    peptides = load_bulk_peptides()
    for df in (proteins, peptides):
        # Acquisition metadata present on every row
        missing = _HARMONIZED_ACQUISITION_COLS - set(df.columns)
        assert not missing, f"missing harmonized columns: {missing}"
        assert (df["instrument"] != "").all()
        assert (df["fragmentation"] != "").all()
