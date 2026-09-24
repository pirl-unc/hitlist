"""Primary-source and deposited-row regressions for the September 2026 audit (#554)."""

import pandas as pd
import pytest

from hitlist.curation import classify_ms_row, load_pmid_overrides
from hitlist.export import generate_ms_samples_table, generate_observations_table


def _export(monkeypatch, rows):
    frame = pd.DataFrame(rows)
    frame["peptide"] = "AAAAAAAAA"
    frame["source"] = "iedb"
    frame["mhc_species"] = "Homo sapiens"
    frame["is_binding_assay"] = False
    monkeypatch.setattr("hitlist.observations.load_observations", lambda **kwargs: frame.copy())
    return generate_observations_table(exclude_non_peptide_ligand=False)


def test_chen_host_knockout_is_distinct_from_erap1_knockdown(monkeypatch):
    # PMC7196583: CRISPR/lentiviral methods and Figure 3, two experiments.
    samples = load_pmid_overrides()[32161166]["ms_samples"]
    control, knockdown = samples
    assert control["condition_control"] == "non_targeting"
    assert control["condition_control_for"] == knockdown["condition_id"]
    assert not control.get("condition_knockdown_genes")
    assert knockdown["condition_knockdown_genes"] == "ERAP1"
    for sample in samples:
        assert sample["condition_knockout_genes"] == "HLA-A;HLA-B;HLA-C"
        assert sample["condition_transduction"] == "HLA-B"
        assert sample["mhc"] == "HLA-B*51:01"
        assert sample["n_samples"] == 2

    # IEDB says siRNA; the primary paper specifies shRNA in the HeLa MS arms.
    prefix = "The epitope was eluted from HLA-B*51:01 expressing HeLa cells"
    result = _export(
        monkeypatch,
        [
            {
                "pmid": 32161166,
                "cell_name": "HeLa cells-Epithelial cell",
                "mhc_restriction": "HLA-B*51:01",
                "mhc_class": "I",
                "assay_comments": prefix + ending,
            }
            for ending in [
                ".",
                " following ERAP1 silencing by siRNA.",
                " and from these cells following ERAP1 silencing by siRNA.",
            ]
        ],
    )
    assert result.condition_id.tolist() == [
        control["condition_id"],
        knockdown["condition_id"],
        "",
    ]
    assert result.condition_knockdown_genes.tolist() == ["", "ERAP1", ""]
    assert result.iloc[2].sample_label == ""


def test_venema_is_an_engineered_ebv_line_with_typed_wt_and_ko_arms(monkeypatch):
    # PMC7950316: Cell Lines, ERAP2 KO Generation, Figure 1 and first Results.
    entry = load_pmid_overrides()[33717175]
    assert entry["override"] == "ebv_lcl"
    control, knockout = entry["ms_samples"]
    assert control["condition_knockout_genes"] == "none"
    assert knockout["condition_knockout_genes"] == "ERAP2"
    genotype = {
        "HLA-A*03:01",
        "HLA-A*29:02",
        "HLA-B*40:01",
        "HLA-B*44:03",
        "HLA-C*03:04",
        "HLA-C*16:01",
    }
    for sample in entry["ms_samples"]:
        assert set(sample["mhc"].split()) == genotype
        assert sample["condition_transduction"] == "SAG"
        assert sample["condition_overexpression_genes"] == "SAG"
        assert sample["condition_material"] == "cultured"
        assert sample["n_samples"] == 2
    flags = classify_ms_row(
        process_type="",
        disease="autoimmune uveitis",
        culture_condition="",
        pmid=33717175,
        cell_name="B cell",
        source_tissue="Blood",
    )
    assert flags["src_ebv_lcl"]
    assert not flags["src_cancer"]
    assert not flags["src_healthy_tissue"]
    result = _export(
        monkeypatch,
        [
            {
                "pmid": 33717175,
                "cell_name": "B cell",
                "mhc_class": "I",
                "mhc_restriction": restriction,
                "assay_comments": "",
            }
            for restriction in [";".join(sorted(genotype)), "HLA-A29"]
        ],
    )
    assert result.condition_id.eq("").all()
    assert result.condition_knockout_genes.eq("").all()
    assert result.sample_label.eq("").all()
    assert result.mhc_restriction.tolist() == [";".join(sorted(genotype)), "HLA-A29"]


BOURNE_PREFIX = (
    "Cells were treated with decitabine (DAC), tazemetostat (TAZ), IFN-gamma (IFN), "
    "or DMSO individually or in combinations containing different subsets of these  "
    "treatments.  The epitope was eluted from cells receiving the following treatment(s):  "
)


@pytest.mark.parametrize(
    ("line", "prefix", "restriction"),
    [
        ("DB", "db", "HLA-A*02:01;HLA-B*18:01;HLA-C*05:01"),
        ("SU-DHL-4", "su_dhl_4", "HLA-A*02:01;HLA-B*15:01;HLA-C*03:04"),
        (
            "SU-DHL-6",
            "su_dhl_6",
            "HLA-A*02:01;HLA-A*23:01;HLA-B*15:01;HLA-B*49:01;HLA-C*03:03;HLA-C*07:01",
        ),
    ],
)
def test_bourne_single_treatments_resolve_but_shared_treatments_do_not(
    monkeypatch, line, prefix, restriction
):
    # PMC9327544: Cell culture and sources; Drug treatments; Figure 5.
    treatments = ["DMSO", "IFN", "IFN-DAC", "IFN-TAZ", "IFN-DAC-TAZ"]
    mixtures = ["IFN, IFN-DAC-TAZ", "DMSO IFN-DAC-TAZ", "IFN-DAC, IFN-TAZ"]
    # Include all systems: constant single-system subsets must not manufacture
    # a cell discriminator that the complete study cannot support.
    rows = []
    for observed_line in ["DB", "SU-DHL-4", "SU-DHL-6"]:
        for treatment in treatments + mixtures:
            rows.append(
                {
                    "pmid": 35561310,
                    "cell_name": observed_line + "-B cell",
                    "mhc_class": "I",
                    "mhc_restriction": restriction if observed_line == line else "HLA class I",
                    "assay_comments": BOURNE_PREFIX + treatment + ".",
                }
            )
    result = _export(monkeypatch, rows)
    target = result[result.cell_name == line + "-B cell"]
    assert target.condition_id.tolist() == [
        prefix + "_" + suffix for suffix in ["dmso", "ifng", "ifng_dac", "ifng_taz", "ifng_dac_taz"]
    ] + ["", "", ""]
    assert target.condition_cytokines.tolist() == ["", "IFNG", "IFNG", "IFNG", "IFNG", "", "", ""]
    assert target.iloc[0].condition_control == "vehicle"
    assert target.iloc[4].condition_drugs == "decitabine;tazemetostat"


def test_bourne_preserves_per_line_hla_typing_and_all_fifteen_arms():
    samples = load_pmid_overrides()[35561310]["ms_samples"]
    assert len(samples) == 15
    expected_dr = {
        "DB": {"HLA-DRB1*03:01"},
        "SU-DHL-4": {"HLA-DRB1*15:01"},
        "SU-DHL-6": {"HLA-DRB1*01:01", "HLA-DRB1*04:01"},
    }
    for line, dr in expected_dr.items():
        arms = [s for s in samples if s["sample_group"] == line]
        assert len(arms) == 5
        assert sum(s.get("condition_cytokines") == "IFNG" for s in arms) == 4
        for arm in arms:
            assert {a for a in arm["mhc"].split() if a.startswith("HLA-DR")} == dr


def test_ritz_comparison_lines_do_not_inherit_healthy_biofluid_identity():
    # PMC5557337, Figure 2: healthy serum/plasma versus THP1 and HEK cells.
    samples = generate_ms_samples_table()
    samples = samples[samples.pmid == 27862975].set_index("sample_label")
    assert samples.loc["healthy donor serum", "effective_override"] == "healthy"
    assert samples.loc["healthy donor plasma", "effective_override"] == "healthy"
    assert samples.loc["THP-1 cell line", "effective_override"] == "cell_line"
    assert samples.loc["HEK293 cell line", "effective_override"] == "noncancer_cell_line"
