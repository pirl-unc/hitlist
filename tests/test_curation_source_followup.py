"""Source/deposit regressions for the six-study follow-up (#555)."""

import pandas as pd
import pytest

from hitlist.curation import classify_ms_row, load_pmid_overrides, sample_mhc_candidates
from hitlist.export import generate_ms_samples_table
from tests.test_curation_sanity_pass import _export


def test_pfammatter_deposit_is_ebv_lcl_and_all_model_is_a_xenograft(monkeypatch):
    entry = load_pmid_overrides()[32502341]
    lcl, all_model = entry["ms_samples"]
    assert entry["override"] == "ebv_lcl"
    assert lcl["condition_material"] == "cultured"
    assert all_model["condition_material"] == "in_vivo"
    assert all_model["override"] == "cancer_patient"
    assert "10H080" in all_model["sample_label"]
    assert all(s["mhc"] == "HLA class I" for s in entry["ms_samples"])
    flags = classify_ms_row(
        pmid=32502341,
        process_type="No immunization",
        disease="healthy",
        culture_condition="Cell Line / Clone (EBV transformed, B-LCL)",
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
                "pmid": 32502341,
                "cell_name": "B cell",
                "mhc_class": "I",
                "mhc_restriction": "HLA class I",
                "assay_comments": "The epitope was eluted from the following conditions: TMT (20 millions).",
            }
        ],
    )
    assert result.condition_id.tolist() == ["b_lcl_tmt"]
    assert result.condition_labeling.tolist() == ["TMT"]


# The deposit's 21 distinct statements, including antibody-qualified variants.
SARANGO_PLAIN = [
    "Mock",
    "Mock, siCTRL, siT6BP",
    "Mock, siCTRL",
    "Mock, siT6BP",
    "siCTRL",
    "siCTRL, siT6BP",
    "siT6BP",
]
SARANGO_ANTIBODY = [
    *SARANGO_PLAIN,
    "Mock, siT6BP, siCTRL",
    "siCTRL, siT6BP, Mock",
    "siCTRL, Mock, siT6BP",
    "siCTRL, Mock",
    "siT6BP, siCTRL",
    "siT6BP, Mock",
    "siT6BP, Mock, siCTRL",
]


@pytest.mark.parametrize("mhc_class, restriction", [("I", "HLA class I"), ("II", "HLA-DRB1*01:02")])
def test_sarango_single_and_shared_arms_in_both_mhc_classes(monkeypatch, mhc_class, restriction):
    # PMC9724678 Figures 3/EV2: same three arms, two aliquots/biological samples,
    # five technical LC-MS runs each; DRB1*0102 is explicitly supported.
    samples = load_pmid_overrides()[36215666]["ms_samples"]
    assert len(samples) == 3
    assert all(s["n_samples"] == 2 for s in samples)
    assert all(s["mhc"] == "HLA class I; HLA-DRB1*01:02" for s in samples)
    typing = sample_mhc_candidates(samples[0]["mhc"])
    assert typing.join_alleles == frozenset({"HLA-DRB1*01:02"})
    assert typing.imprecise == ("HLA class I",)
    assert [s.get("condition_knockdown_genes", "") for s in samples] == ["", "", "TAX1BP1"]
    assert [s.get("condition_control", "") for s in samples] == ["mock", "non_targeting", ""]
    ids = {
        "Mock": "hela_ciita_mock",
        "siCTRL": "hela_ciita_control_sirna",
        "siT6BP": "hela_ciita_t6bp_sirna",
    }
    rows, expected = [], []
    for arms, suffix in [
        *[(a, "") for a in SARANGO_PLAIN],
        *[
            (a, " The epitope was eluted using the following mAbs: L243, Tu39.")
            for a in SARANGO_ANTIBODY
        ],
    ]:
        rows.append(
            {
                "pmid": 36215666,
                "cell_name": "HeLa cells-Epithelial cell",
                "mhc_class": mhc_class,
                "mhc_restriction": restriction,
                "antigen_processing_comments": "HeLa cells were used for ligand elution using the following experimental conditions: mock treated (Mock), transfected with control siRNA (siCTRL), or transfected with Autophagy Receptor TAX1BP1 siRNA (siT6BP).",
                "assay_comments": "The epitope was eluted under the following experimental conditions: "
                + arms
                + "."
                + suffix,
            }
        )
        expected.append(ids.get(arms, ""))
    assert len(rows) == 21
    result = _export(monkeypatch, rows)
    assert result.condition_id.tolist() == expected
    shared = result[pd.Series(expected).eq("")]
    assert shared.sample_label.eq("").all()
    assert shared.condition_knockdown_genes.eq("").all()
    assert shared.sample_attribution.eq("pmid_ambiguous").all()
    assert result.mhc_restriction.eq(restriction).all()


def test_martin_esteban_natural_erap_backgrounds_are_not_knockouts(monkeypatch):
    entry = load_pmid_overrides()[28063628]
    assert not entry.get("perturbations")
    assert entry["override"] == "noncancer_cell_line"
    samples = entry["ms_samples"]
    assert len(samples) == 4
    for sample in samples:
        assert sample["n_samples"] == 3
        assert sample["condition_knockout_genes"] == "none"
        assert sample["condition_material"] == "cultured"
    # Every deposited statement enumerates at least two natural lines.
    groups = [(0, 1, 2, 3), (1, 2), (1, 2, 3), (2, 3), (0, 3), (1, 3), (0, 1, 3), (0, 2, 3)]
    names = [
        "6370 cells (ERAP2+, ERAP1 Hap2)",
        "15510 cells (ERAP2-, ERAP1 Hap1)",
        "10151 cells (ERAP2-, ERAP1 Hap2)",
        "P50 cells (ERAP2+, ERAP1 Hap10)",
    ]
    result = _export(
        monkeypatch,
        [
            {
                "pmid": 28063628,
                "cell_name": "Lymphoblast",
                "mhc_class": "I",
                "mhc_restriction": "HLA-B*27:05",
                "assay_comments": "The epitope was eluted from "
                + ", ".join(names[i] for i in group)
                + ".",
            }
            for group in groups
        ],
    )
    assert result.condition_id.eq("").all()
    assert result.sample_label.eq("").all()
    assert result.sample_attribution.eq("pmid_ambiguous").all()
    assert not result.study_apm_perturbed.any()
    assert result.condition_knockout_genes.eq("none").all()


@pytest.mark.parametrize(
    "pmid, label, material",
    [
        (26154972, "primary PBMCs (healthy donors)", "direct_ex_vivo"),
        (27846572, "primary fibroblasts", "cultured"),
        (28228285, "validation PBMCs", "direct_ex_vivo"),
        (28228285, "validation primary fibroblasts", "cultured"),
    ],
)
def test_primary_material_does_not_inherit_cell_line_override(pmid, label, material):
    samples = generate_ms_samples_table()
    samples = samples[samples.pmid.eq(pmid)].set_index("sample_label")
    assert samples.loc[label, "effective_override"] == "healthy"
    assert samples.loc[label, "effective_override_origin"] == "sample"
    assert samples.loc[label, "condition_material"] == material
