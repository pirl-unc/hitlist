"""Regression tests for APC intervention curations (#899, #900, #856, #593)."""

from __future__ import annotations

import pandas as pd
import pytest
import yaml

from hitlist import curation
from hitlist.conditions import is_engineered_material
from hitlist.curation import classify_ms_row, load_pmid_overrides
from hitlist.export import (
    _arms_in_elution_conditions,
    expand_condition_memberships,
    generate_ms_samples_table,
    generate_observations_table,
    generate_training_table,
)


def _mock_observations(monkeypatch, rows: list[dict]) -> None:
    defaults = {
        "peptide": "AAAAAAAAA",
        "mhc_class": "I",
        "mhc_species": "Homo sapiens",
        "species": "Homo sapiens",
        "source_organism": "Homo sapiens",
        "host": "Homo sapiens (human)",
        "cell_name": "",
        "source_tissue": "",
        "culture_condition": "Cell Line / Clone",
        "antigen_processing_comments": "",
        "assay_comments": "",
        "is_binding_assay": False,
        "qualitative_measurement": "Positive",
        "assay_method": "mass spectrometry",
        "source": "iedb",
    }
    records = []
    for idx, item in enumerate(rows):
        rec = {
            **defaults,
            "provenance_id": f"prov:{idx}",
            "assay_iri": f"http://www.iedb.org/assay/{1000 + idx}",
            "reference_iri": f"http://www.iedb.org/reference/{item['pmid']}",
            **item,
        }
        records.append(rec)
    frame = pd.DataFrame(records)
    monkeypatch.setattr("hitlist.observations.load_observations", lambda **kw: frame.copy())


@pytest.mark.parametrize(
    "bad_spec",
    [
        {},
        {"not_detected": ["untreated"]},
        {"detected": []},
        {"detected": ["treated"], "unknown_key": ["untreated"]},
        {"detected": ["treated"], "not_detected": ["treated"]},
        {"detected": ["treated"], "not_detected": ["not_profiled"]},
        {"detected": ["treated"], "not_detected": ["untreated", "untreated"]},
    ],
)
def test_structured_elution_mapping_rejects_malformed_or_overlapping_arms(
    tmp_path, monkeypatch, bad_spec
):
    path = tmp_path / "pmid_overrides.yaml"
    path.write_text(
        yaml.safe_dump(
            [
                {
                    "pmid": 99999900,
                    "elution_condition_ids": {"The peptide was eluted.": bad_spec},
                    "ms_samples": [
                        {
                            "sample_label": "treated",
                            "condition_id": "treated",
                            "condition_status": "unreported",
                        },
                        {
                            "sample_label": "untreated",
                            "condition_id": "untreated",
                            "condition_status": "unreported",
                        },
                        {
                            "sample_label": "not profiled",
                            "condition_id": "not_profiled",
                            "condition_status": "unreported",
                            "profiled": False,
                        },
                    ],
                }
            ]
        )
    )
    real_path = curation._data_path
    monkeypatch.setattr(
        curation,
        "_data_path",
        lambda fn: str(path) if fn == "pmid_overrides.yaml" else real_path(fn),
    )
    curation.load_pmid_overrides.cache_clear()
    try:
        with pytest.raises(ValueError, match="elution"):
            curation.load_pmid_overrides()
    finally:
        curation.load_pmid_overrides.cache_clear()


def test_generic_elution_enumeration_ignores_negated_clauses():
    text = (
        "The peptide was eluted from the following experimental conditions: "
        "DG013A-treated A375 cells, but not from untreated cells."
    )
    assert _arms_in_elution_conditions(text) == {"perturbed"}


def test_ilca_tapbpr_rescue_variants_separate_background_from_constructs(monkeypatch):
    entry = load_pmid_overrides()[30484775]
    arms = {s["condition_id"]: s for s in entry["ms_samples"]}
    assert set(arms) == {
        "helam_tapbpr_ko_ifng_tapbpr_wt",
        "helam_tapbpr_ko_ifng_tapbpr_oloop",
        "helam_tapbpr_ko_ifng_tapbpr_og30l",
        "helam_tapbpr_ko_ifng_tapbpr_m29",
    }
    for cid, arm in arms.items():
        assert arm["condition_background"] == "TAPBPR_KO"
        assert arm["condition_cytokines"] == "IFNG"
        # KO background + rescue must never read as simple TAPBPR knockout or numeric OE (#899).
        assert not arm.get("condition_knockout_genes")
        assert not arm.get("condition_overexpression_genes")
        assert is_engineered_material(arm), cid
    assert arms["helam_tapbpr_ko_ifng_tapbpr_wt"]["condition_genetic_variants"] == "none"
    assert set(arms["helam_tapbpr_ko_ifng_tapbpr_wt"]["condition_control_for"].split(";")) == {
        "helam_tapbpr_ko_ifng_tapbpr_m29",
        "helam_tapbpr_ko_ifng_tapbpr_og30l",
        "helam_tapbpr_ko_ifng_tapbpr_oloop",
    }

    _mock_observations(
        monkeypatch,
        [
            {
                "peptide": "WTPEPTIDE",
                "pmid": 30484775,
                "mhc_restriction": "HLA-A*68:02",
                "cell_name": "HeLa cells-Epithelial cell",
                "assay_comments": (
                    "Peptides were eluted from W6/32-reactive MHC I complexes from IFNg "
                    "treated HeLaM-TAPBPR KO cells expressing TAPBPR WT."
                ),
            },
            {
                "peptide": "LOOPPEPTD",
                "pmid": 30484775,
                "mhc_restriction": "HLA-B*15:03",
                "cell_name": "HeLa cells-Epithelial cell",
                "assay_comments": (
                    "Peptides were eluted from W6/32-reactive MHC I complexes from IFNg "
                    "treated HeLaM-TAPBPR KO cells expressing TAPBPR \ufffd loop."
                ),
            },
            {
                "peptide": "G30LPEPTD",
                "pmid": 30484775,
                "mhc_restriction": "HLA-C*12:03",
                "cell_name": "HeLa cells-Epithelial cell",
                "assay_comments": (
                    "Peptides were eluted from W6/32-reactive MHC I complexes from IFNg "
                    "treated HeLaM-TAPBPR KO cells expressing TAPBPR \ufffd G30L."
                ),
            },
            {
                "peptide": "M29PEPTID",
                "pmid": 30484775,
                "mhc_restriction": "HLA class I",
                "cell_name": "HeLa cells-Epithelial cell",
                "assay_comments": (
                    "Peptides were eluted from W6/32-reactive MHC I complexes from IFNg "
                    "treated HeLaM-TAPBPR KO cells expressing TAPBPR M29."
                ),
            },
            {
                "peptide": "BLANKCMNT",
                "pmid": 30484775,
                "mhc_restriction": "HLA-A*68:02",
                "cell_name": "HeLa cells-Epithelial cell",
                "assay_comments": "",
            },
        ],
    )
    obs = generate_observations_table(exclude_non_peptide_ligand=False).set_index("peptide")
    assert obs.loc["WTPEPTIDE", "condition_id"] == "helam_tapbpr_ko_ifng_tapbpr_wt"
    assert obs.loc["LOOPPEPTD", "condition_id"] == "helam_tapbpr_ko_ifng_tapbpr_oloop"
    assert obs.loc["G30LPEPTD", "condition_id"] == "helam_tapbpr_ko_ifng_tapbpr_og30l"
    assert obs.loc["M29PEPTID", "condition_id"] == "helam_tapbpr_ko_ifng_tapbpr_m29"
    assert "\ufffd" in obs.loc["LOOPPEPTD", "assay_comments"]
    assert obs.loc["WTPEPTIDE", "species_context_id"] == "human_helam_tapbpr_immunopeptidome"
    # Blank/unmapped row stays unresolved rather than collapsing onto one variant.
    assert obs.loc["BLANKCMNT", "condition_id"] == ""
    assert obs.loc["BLANKCMNT", "supported_condition_ids"] == ""
    assert obs.loc["BLANKCMNT", "sample_attribution"] == "pmid_ambiguous"
    assert obs.loc["BLANKCMNT", "condition_background"] == "TAPBPR_KO"
    assert obs.loc["BLANKCMNT", "condition_cytokines"] == "IFNG"


def test_five_intervention_studies_preserve_shared_excluded_and_unresolved_memberships(
    monkeypatch,
):
    _mock_observations(
        monkeypatch,
        [
            # 1) PMID 31222486 (DG013A): untreated does NOT assert absence under treatment.
            {
                "peptide": "DG013UNT1",
                "pmid": 31222486,
                "mhc_restriction": "HLA-A*01:01;HLA-A*02:01;HLA-B*44:03;HLA-B*57:01;HLA-C*06:02;HLA-C*16:01",
                "assay_comments": "The peptide was eluted from untreated A375 melanoma cells.",
            },
            {
                "peptide": "DG013TRT1",
                "pmid": 31222486,
                "mhc_restriction": "HLA-A*01:01;HLA-A*02:01;HLA-B*44:03;HLA-B*57:01;HLA-C*06:02;HLA-C*16:01",
                "assay_comments": (
                    "The peptide was eluted from A375 melanoma cells treated with the "
                    "ERAP1 inhibitor DG013A, but not from untreated cells."
                ),
            },
            {
                "peptide": "DG013BTH1",
                "pmid": 31222486,
                "mhc_restriction": "HLA-A*01:01;HLA-A*02:01;HLA-B*44:03;HLA-B*57:01;HLA-C*06:02;HLA-C*16:01",
                "assay_comments": (
                    "The peptide was eluted from untreated A375 melanoma cells and A375 "
                    "cells treated with the ERAP1 inhibitor DG013A."
                ),
            },
            # 2) PMID 23328677 (entinostat): "or" stays unresolved, unlike "and".
            {
                "peptide": "ENTDMSONLY",
                "pmid": 23328677,
                "mhc_class": "II",
                "mhc_restriction": "HLA-DR",
                "assay_comments": (
                    "The epitope was eluted from DMSO-treated HLA-DR expressing Raji "
                    "B-cells but not from Entinostat -treated cells."
                ),
            },
            {
                "peptide": "ENTDRGONLY",
                "pmid": 23328677,
                "mhc_class": "II",
                "mhc_restriction": "HLA-DR",
                "assay_comments": (
                    "The epitope was eluted from Entinostat-treated HLA-DR expressing "
                    "Raji B-cells but not from DMSO-treated cells."
                ),
            },
            {
                "peptide": "ENTDISJOR1",
                "pmid": 23328677,
                "mhc_class": "II",
                "mhc_restriction": "HLA-DR",
                "assay_comments": (
                    "The epitope was eluted from DMSO or Entinostat-treated HLA-DR "
                    "expressing Raji B-cells."
                ),
            },
            # 3) PMID 32265295 (ERAP2 deletion): WT vs KO vs shared, no pooled KO label.
            {
                "peptide": "ERAP2WTON",
                "pmid": 32265295,
                "mhc_restriction": "HLA-B*27:05",
                "assay_comments": (
                    "The epitope was eluted from B27-C1R cells, but not from mutant C1R "
                    "cells missing ERAP2 exons 5 and 6."
                ),
            },
            {
                "peptide": "ERAP2KOON",
                "pmid": 32265295,
                "mhc_restriction": "HLA-B*27:05",
                "assay_comments": (
                    "The epitope was eluted only from mutant C1R cells with ERAP2 exons "
                    "5 and 6 deleted."
                ),
            },
            {
                "peptide": "ERAP2BOTH",
                "pmid": 32265295,
                "mhc_restriction": "HLA-B*27:05",
                "assay_comments": (
                    "The epitope was eluted from B27-C1R cells and mutant B27-C1R cells "
                    "missing ERAP2 exons 5 and 6."
                ),
            },
            # 4) PMID 29930882 (gemcitabine): exclusive vs both phrasings of shared.
            {
                "peptide": "GEMEXCLUS",
                "pmid": 29930882,
                "mhc_restriction": "HLA class I",
                "assay_comments": (
                    "The class I ligand was eluted exclusively in the presence of "
                    "gemcitabine (GEM) treatment, which is known to upregulate MHC class "
                    "I expression on the surface of tumor cells."
                ),
            },
            {
                "peptide": "GEMBOTH01",
                "pmid": 29930882,
                "mhc_restriction": "HLA class I",
                "assay_comments": (
                    "The class I ligand was eluted in the presence, as well as the "
                    "absence of gemcitabine (GEM) treatment, which is known to upregulate "
                    "MHC class I expression on the surface of tumor cells."
                ),
            },
            # 5) PMID 30779550 (doxorubicin): "terated" HCT 116 vs unrecorded systems.
            {
                "peptide": "DOXHCT116",
                "pmid": 30779550,
                "mhc_restriction": "HLA class I",
                "cell_name": "HCT 116-Epithelial cell",
                "source_tissue": "Gastrointestinal Tract",
                "assay_comments": (
                    "The epitope was eluted from HCT 116 cells that were terated with doxorubicin."
                ),
            },
            {
                "peptide": "DOXHCTBLK",
                "pmid": 30779550,
                "mhc_restriction": "HLA class I",
                "cell_name": "HCT 116-Epithelial cell",
                "source_tissue": "Gastrointestinal Tract",
                "assay_comments": "",
            },
            {
                "peptide": "DOXEL4CEL",
                "pmid": 30779550,
                "mhc_restriction": "H2-D*b",
                "mhc_species": "Mus musculus",
                "species": "Mus musculus",
                "source_organism": "Mus musculus",
                "host": "Mus musculus C57BL/6",
                "cell_name": "EL-4 cells-Lymphoblast",
                "source_tissue": "Lymphoid",
                "culture_condition": "Cell Line / Clone",
                "assay_comments": "The epitope was eluted from EL4 cells.",
            },
            {
                "peptide": "DOXSPLENO",
                "pmid": 30779550,
                "mhc_restriction": "H2-D*b",
                "mhc_species": "Mus musculus",
                "species": "Mus musculus",
                "source_organism": "Mus musculus",
                "host": "Mus musculus C57BL/6",
                "cell_name": "Splenocyte",
                "source_tissue": "Spleen",
                "culture_condition": "Direct Ex Vivo",
                "assay_comments": (
                    "The epitope was eluted from C57BL/6 mice that had received EL4 cells."
                ),
            },
        ],
    )
    monkeypatch.setattr(
        "hitlist.observations.load_binding",
        lambda **kw: pd.DataFrame(columns=["peptide", "mhc_restriction", "mhc_class"]),
    )
    monkeypatch.setattr("hitlist.provenance._contributor_contract", lambda: {"schema_version": 1})

    obs = generate_observations_table(exclude_non_peptide_ligand=False).set_index("peptide")
    assert len(obs) == 15

    # DG013A specificity includes >=4-fold enrichment, not only unique detection
    # (primary paper Fig. 3; #917). Deposited text cannot establish absence.
    assert obs.loc["DG013UNT1", "condition_id"] == "a375_melanoma_untreated"
    assert obs.loc["DG013UNT1", "excluded_condition_ids"] == ""
    assert obs.loc["DG013TRT1", "condition_id"] == "a375_erap1_inhibitor"
    assert obs.loc["DG013TRT1", "excluded_condition_ids"] == ""
    assert obs.loc["DG013TRT1", "condition_drugs"] == "DG013A"
    assert obs.loc["DG013BTH1", "condition_id"] == ""
    assert (
        obs.loc["DG013BTH1", "supported_condition_ids"]
        == "a375_erap1_inhibitor;a375_melanoma_untreated"
    )

    # Entinostat ("or" vs exclusionary)
    assert obs.loc["ENTDMSONLY", "condition_id"] == "raji_dmso_vehicle_control"
    assert obs.loc["ENTDMSONLY", "condition_control"] == "vehicle"
    assert obs.loc["ENTDMSONLY", "condition_drugs"] == "DMSO"
    assert obs.loc["ENTDMSONLY", "excluded_condition_ids"] == "raji_entinostat_hdac_inhibitor"
    assert obs.loc["ENTDRGONLY", "condition_id"] == "raji_entinostat_hdac_inhibitor"
    assert obs.loc["ENTDRGONLY", "excluded_condition_ids"] == "raji_dmso_vehicle_control"
    assert obs.loc["ENTDISJOR1", "condition_id"] == ""
    assert obs.loc["ENTDISJOR1", "supported_condition_ids"] == ""
    assert obs.loc["ENTDISJOR1", "excluded_condition_ids"] == ""

    # ERAP2 deletion
    assert obs.loc["ERAP2WTON", "condition_id"] == "b27_c1r_wt"
    assert obs.loc["ERAP2WTON", "condition_knockout_genes"] == "none"
    assert obs.loc["ERAP2WTON", "excluded_condition_ids"] == "b27_c1r_erap2_ko"
    assert obs.loc["ERAP2KOON", "condition_id"] == "b27_c1r_erap2_ko"
    assert obs.loc["ERAP2KOON", "condition_knockout_genes"] == "ERAP2"
    assert obs.loc["ERAP2KOON", "excluded_condition_ids"] == "b27_c1r_wt"
    assert obs.loc["ERAP2BOTH", "condition_id"] == ""
    assert obs.loc["ERAP2BOTH", "condition_knockout_genes"] == ""
    assert obs.loc["ERAP2BOTH", "supported_condition_ids"] == "b27_c1r_erap2_ko;b27_c1r_wt"

    # Gemcitabine
    assert obs.loc["GEMEXCLUS", "condition_id"] == "hct_116_gemcitabine_treated"
    assert obs.loc["GEMEXCLUS", "excluded_condition_ids"] == "hct_116_untreated_constitutive"
    assert obs.loc["GEMBOTH01", "condition_id"] == ""
    assert (
        obs.loc["GEMBOTH01", "supported_condition_ids"]
        == "hct_116_gemcitabine_treated;hct_116_untreated_constitutive"
    )

    # Doxorubicin: no fabricated HCT 116 untreated controls; ex vivo splenocytes not cell_line.
    assert obs.loc["DOXHCT116", "condition_id"] == "hct_116_doxorubicin"
    assert obs.loc["DOXHCTBLK", "condition_id"] == ""
    assert obs.loc["DOXHCTBLK", "supported_condition_ids"] == ""
    assert obs.loc["DOXHCTBLK", "sample_group"] == "HCT 116"
    assert obs.loc["DOXEL4CEL", "condition_id"] == ""
    assert obs.loc["DOXEL4CEL", "sample_group"] == "EL-4"
    assert obs.loc["DOXEL4CEL", "condition_material"] == ""
    dox_samples = generate_ms_samples_table()
    el4 = dox_samples[(dox_samples.pmid == 30779550) & (dox_samples.sample_group == "EL-4")]
    assert len(el4) == 4
    assert set(el4.condition_material) == {"cultured", "direct_ex_vivo"}
    assert obs.loc["DOXSPLENO", "condition_id"] == ""
    assert obs.loc["DOXSPLENO", "sample_group"] == "Splenocyte"
    cls_spleno = classify_ms_row(
        pmid="30779550",
        source_tissue="Spleen",
        cell_name="Splenocyte",
        culture_condition="Direct Ex Vivo",
        disease="",
        process_type="",
        mhc_restriction="H2-D*b",
    )
    assert not cls_spleno["src_cell_line"]

    # Training export & projected columns agree with full observations export.
    projected = generate_training_table(
        include_evidence="ms",
        exclude_non_peptide_ligand=False,
        columns=[
            "peptide",
            "pmid",
            "condition_id",
            "supported_condition_ids",
            "excluded_condition_ids",
        ],
    ).set_index("peptide")
    assert (
        projected["supported_condition_ids"].to_dict() == obs["supported_condition_ids"].to_dict()
    )
    assert projected["excluded_condition_ids"].to_dict() == obs["excluded_condition_ids"].to_dict()

    # Long-form condition-comparison view preserves provenance_id without inflating source table.
    edges = expand_condition_memberships(obs.reset_index())
    erap2_both_edges = edges[edges["peptide"] == "ERAP2BOTH"]
    assert set(erap2_both_edges["condition_id"]) == {"b27_c1r_wt", "b27_c1r_erap2_ko"}
    assert set(erap2_both_edges["membership_status"]) == {"detected"}
    assert erap2_both_edges["provenance_id"].nunique() == 1
    assert edges[edges["peptide"] == "ENTDISJOR1"].empty
    assert edges[edges["peptide"] == "DOXHCTBLK"].empty
    dg_unt_edges = edges[edges["peptide"] == "DG013UNT1"]
    assert list(dg_unt_edges["condition_id"]) == ["a375_melanoma_untreated"]
    assert list(dg_unt_edges["membership_status"]) == ["detected"]
    dg_treated_edges = edges[edges["peptide"] == "DG013TRT1"]
    assert list(dg_treated_edges["condition_id"]) == ["a375_erap1_inhibitor"]
    assert list(dg_treated_edges["membership_status"]) == ["detected"]


def test_laghmouchi_k562_hla_dp_heterodimers_and_hilton_shla_engineering(monkeypatch):
    dp_entry = load_pmid_overrides()[35251023]
    assert len(dp_entry["ms_samples"]) == 13
    assert all("19:01" not in sample["mhc"] for sample in dp_entry["ms_samples"])
    for sample in dp_entry["ms_samples"]:
        assert sample["condition_mhc_context"] == "mhc_transfectant;monoallelic"
        assert sample["condition_transduction"] == "HLA-DPA1;HLA-DPB1"
        assert is_engineered_material(sample)

    hilton_samples = generate_ms_samples_table()
    hilton_samples = hilton_samples[hilton_samples["pmid"] == 28514659]
    assert len(hilton_samples) == 3
    assert all(is_engineered_material(row) for _, row in hilton_samples.iterrows())
    assert hilton_samples.condition_reference.str.contains("PMC5510751", regex=False).all()

    _mock_observations(
        monkeypatch,
        [
            {
                "peptide": "DPEXACT01",
                "pmid": 35251023,
                "mhc_class": "II",
                "mhc_restriction": "HLA-DPA1*02:01/DPB1*13:01",
                "cell_name": "K562-Myeloid cell",
            },
            {
                "peptide": "DPUNVERIF",
                "pmid": 35251023,
                "mhc_class": "II",
                "mhc_restriction": "HLA-DPA1*02:02/DPB1*19:01",
                "cell_name": "K562-Myeloid cell",
            },
            {
                "peptide": "DPCOARSE1",
                "pmid": 35251023,
                "mhc_class": "II",
                "mhc_restriction": "HLA class II",
                "cell_name": "K562-Myeloid cell",
            },
        ],
    )
    obs = generate_observations_table(exclude_non_peptide_ligand=False).set_index("peptide")
    assert obs.loc["DPEXACT01", "condition_id"] == "k562_hla_dpa1_02_01_dpb1_13_01"
    assert obs.loc["DPEXACT01", "sample_attribution"] == "allele_exact"
    assert obs.loc["DPEXACT01", "condition_mhc_context"] == "mhc_transfectant;monoallelic"
    assert obs.loc["DPCOARSE1", "condition_id"] == ""
    assert obs.loc["DPCOARSE1", "supported_condition_ids"] == ""
    assert obs.loc["DPCOARSE1", "sample_attribution"] == "pmid_ambiguous"
    assert obs.loc["DPCOARSE1", "condition_mhc_context"] == "mhc_transfectant;monoallelic"
    assert obs.loc["DPUNVERIF", "condition_id"] == ""
    assert obs.loc["DPUNVERIF", "supported_condition_ids"] == ""
    assert obs.loc["DPUNVERIF", "mhc_restriction"] == "HLA-DPA1*02:02/DPB1*19:01"
