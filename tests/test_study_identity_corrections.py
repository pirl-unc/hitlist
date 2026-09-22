"""Primary-paper and deposited-identity regressions for #457."""

import pandas as pd

from hitlist.curation import load_pmid_overrides
from hitlist.export import generate_observations_table


def _row(pmid, cell, restriction, comment=""):
    return {
        "peptide": "AAAAAAAAA",
        "pmid": pmid,
        "cell_name": cell,
        "mhc_restriction": restriction,
        "mhc_class": "I",
        "mhc_species": "Homo sapiens",
        "source": "iedb",
        "source_tissue": "",
        "antigen_processing_comments": "",
        "assay_comments": comment,
        "is_monoallelic": False,
        "is_binding_assay": False,
        "qualitative_measurement": "Positive",
    }


def _export(monkeypatch, rows):
    monkeypatch.setattr(
        "hitlist.observations.load_observations", lambda **kwargs: pd.DataFrame(rows)
    )
    return generate_observations_table(exclude_non_peptide_ligand=False)


def test_ritz_deposited_genotypes_reach_the_paper_cell_lines(monkeypatch):
    # doi:10.1002/pmic.201700177, Results; both genotypes explicitly SSO/SSP typed.
    maver = "HLA-A*24:02;HLA-A*26:01;HLA-B*38:01;HLA-B*44:02;HLA-C*05:01;HLA-C*12:03"
    hek = "HLA-A*03:01;HLA-B*07:02;HLA-C*07:01"
    rows = [
        {**_row(28834231, "MAVER-1-Lymphoblast", maver), "source_tissue": "Blood"},
        {**_row(28834231, "293-T-Epithelial cell", hek), "source_tissue": "Embryo"},
    ]
    result = _export(monkeypatch, rows)
    assert result.sample_label.tolist() == [
        "MAVER-1 (mantle cell lymphoma)",
        "HEK293 (embryonic kidney)",
    ]
    entry = load_pmid_overrides()[28834231]
    assert entry["ms_samples"][0]["mhc"].split() == maver.split(";")
    assert not entry["ms_samples"][0].get("reference_proteomes")
    assert entry["arm_resolution"] == "resolved"


def test_schellens_measles_reference_and_merged_arm_evidence(monkeypatch):
    # doi:10.1371/journal.pone.0136417, Cell culture and infection; S1 Table.
    entry = load_pmid_overrides()[26375851]
    control, infected = entry["ms_samples"]
    assert infected["condition_infection"] == "Measles virus"
    assert infected["reference_proteomes"][-1]["uniprot"] == "UP000100252"
    assert control["n_samples"] == infected["n_samples"] == 4
    assert "condition_infection" not in control
    assert "vaccinia" not in str(entry).lower()
    result = _export(
        monkeypatch,
        [
            _row(
                26375851,
                "Lymphoblast",
                "HLA-B*44:02",
                "Eluted from four BLCL in both steady state and following measles infection.",
            )
        ],
    )
    assert result.iloc[0]["condition_id"] == ""
    assert result.iloc[0]["condition_infection"] == ""


def test_stopfer_source_roster_methods_and_mixed_treatment_evidence(monkeypatch):
    # doi:10.1038/s41467-020-16588-9, Figs 2-6 and Data 3/5 file maps.
    entry = load_pmid_overrides()[32488085]
    samples = entry["ms_samples"]
    assert len(samples) == 17
    groups = {s["sample_group"] for s in samples}
    assert groups == {"SK-MEL-2", "SK-MEL-5", "SK-MEL-28", "IPC-298", "MDA-MB-231"}
    assert entry["instrument"] == "Q Exactive HF-X"
    assert entry["search_engine"] == "Mascot 2.4 (Proteome Discoverer 2.2)"
    for group in groups - {"MDA-MB-231"}:
        arms = [s for s in samples if s["sample_group"] == group]
        assert len(arms) == 4
        assert sum(s.get("condition_control") == "vehicle" for s in arms) == 1
        assert sum(s.get("condition_drugs") == "palbociclib" for s in arms) == 2
        assert sum(s.get("condition_cytokines") == "IFNG" for s in arms) == 1
    mda = [s for s in samples if s["sample_group"] == "MDA-MB-231"]
    assert len(mda) == 1
    assert not mda[0].get("condition_drugs")
    # IEDB reports A*02:17/B*40:02 here, not the former A*02:50/B*40:01.
    assert {"HLA-A*02:17", "HLA-B*40:02"} <= set(mda[0]["mhc"].split())
    rows = [
        _row(
            32488085,
            line + "-Melanocyte",
            "HLA class I",
            "The epitope was eluted from cells treated with 1uM CDK4/6i, "
            "10uM CDK4/6i, and 10ng/mL IFNg.",
        )
        for line in ["SK-MEL-2", "SK-MEL-5", "SK-MEL-28", "IPC-298"]
    ]
    rows.append(_row(32488085, "MDA-MB-231-Epithelial cell", "HLA class I"))
    result = _export(monkeypatch, rows)
    assert result.sample_group.tolist() == [
        "SK-MEL-2",
        "SK-MEL-5",
        "SK-MEL-28",
        "IPC-298",
        "MDA-MB-231",
    ]
    assert result.condition_id.tolist() == ["", "", "", "", "mda_mb_231_quantification"]
