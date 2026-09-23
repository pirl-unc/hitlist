"""Primary-paper checks for source provenance, not inferred RNA coverage (#522)."""

from hitlist.line_expression import load_line_expression_anchors, load_line_expression_sources


def test_hap1_paper_reports_fpkm_not_kallisto_tpm():
    # Essletzbichler, Genome Res 2014, Methods and Data access;
    # https://pmc.ncbi.nlm.nih.gov/articles/PMC4248322/
    source = next(
        s for s in load_line_expression_sources() if s["source_id"] == "Essletzbichler_2014_HAP1"
    )
    assert source["pmid"] == 25373145
    assert source["normalization"] == "fpkm"
    assert source["quantifier"] != "kallisto"
    assert "SRP044391" in source["reference"]
    assert source["build_status"] == "placeholder"


def test_pearson_jy_validation_is_not_an_hhc_expression_measurement():
    # Pearson, JCI 2016, Methods: independent JY transcriptomic validation;
    # https://www.jci.org/articles/view/88590
    source = next(
        s for s in load_line_expression_sources() if s["source_id"] == "Pearson_BLCL_panel"
    )
    assert source["pmid"] == 27841757
    assert source["cell_lines_covered"] == ["JY"]
    assert source["quantifier"] == "kallisto"
    hhc = next(e for e in load_line_expression_anchors() if e["name"] == "HHC")
    assert source["source_id"] not in hhc["source_ids"]


def test_motif_deconvolution_paper_does_not_supply_host_rna():
    # Kaabinejadian, Front Immunol 2022, Materials and Methods;
    # https://doi.org/10.3389/fimmu.2022.835454
    sources = {s["source_id"]: s for s in load_line_expression_sources()}
    assert "Kaabinejadian_2022_HLA_mono" not in sources
    for entry in load_line_expression_anchors():
        for source_id in entry["source_ids"]:
            assert source_id in sources
