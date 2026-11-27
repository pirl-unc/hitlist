"""Tests for #945 (orthogonal APC vs tumor cell lineage ontology, structured perturbation
contrasts, and APM expression provenance) and #946 (biophysical stability & kinetic
assay covariates, koff/kon unit standardization, and stability normalization audit).
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from hitlist.cell_lineage import (
    attach_orthogonal_cell_lineage,
    classify_cell_lineage,
    load_cell_lineage_ontology,
)
from hitlist.model_training import (
    KINETIC_STABILITY_ENDPOINTS,
    prepare_stability_training_data,
    qualify_binding_measurements,
)
from hitlist.perturbation_schema import (
    extract_structured_perturbation_contrasts,
    tag_expression_provenance,
)
from hitlist.stability_context import (
    audit_stability_normalization,
    extract_biophysical_covariates,
    qualify_stability_and_kinetics,
)

# ── #945: Orthogonal APC vs Tumor Lineage Ontology ───────────────────────────


def test_cell_lineage_ontology_yaml_entries_use_controlled_vocabularies():
    data = load_cell_lineage_ontology()
    vocabs = data["vocabularies"]
    for line_name, entry in data["known_cell_lines"].items():
        assert entry["cell_lineage"] in vocabs["cell_lineage"], line_name
        assert entry["malignancy_status"] in vocabs["malignancy_status"], line_name
        assert entry["tumor_type"] in vocabs["tumor_type"], line_name


def test_classify_cell_lineage_separates_apc_subtypes_hosts_and_tumors():
    # Monoallelic host 721.221 vs immortalized non-malignant EBV-LCL JY vs melanoma A-375
    host_721 = classify_cell_lineage(cell_line_name="721.221", is_monoallelic=True)
    assert host_721.cell_lineage == "ebv_lcl"
    assert host_721.malignancy_status == "engineered_monoallelic_host"
    assert host_721.tumor_type == "none"

    jy = classify_cell_lineage(cell_line_name="JY", condition_cytokines="IFNG")
    assert jy.cell_lineage == "ebv_lcl"
    assert jy.malignancy_status == "immortalized_nonmalignant"
    assert jy.tumor_type == "none"
    assert jy.activation_state == "ifng_stimulated"

    a375 = classify_cell_lineage(cell_name="A-375-Melanocyte", condition="unperturbed")
    assert a375.cell_lineage == "melanocyte"
    assert a375.malignancy_status == "tumor_cell_line"
    assert a375.tumor_type == "melanoma"
    assert a375.activation_state == "resting"

    # Primary immature vs LPS-matured moDC
    imodc = classify_cell_lineage(
        cell_name="immature moDC",
        cell_type="Dendritic cell",
        src_healthy_tissue=True,
    )
    assert imodc.cell_lineage == "moDC"
    assert imodc.malignancy_status == "healthy_primary"
    assert imodc.tumor_type == "none"
    assert imodc.activation_state == "immature_dc"

    mmodc = classify_cell_lineage(
        cell_name="mature moDC",
        culture_condition="LPS stimulated 24h",
        src_healthy_tissue=True,
    )
    assert mmodc.cell_lineage == "moDC"
    assert mmodc.activation_state == "activated_lps"

    # Primary DLBCL tumor biopsy
    dlbcl = classify_cell_lineage(
        cell_type="B-cell",
        disease="Diffuse large B-cell lymphoma",
        src_cancer=True,
        src_ex_vivo=True,
    )
    assert dlbcl.cell_lineage == "b_cell_malignant"
    assert dlbcl.malignancy_status == "tumor_primary"
    assert dlbcl.tumor_type == "dlbcl"
    assert dlbcl.activation_state == "ex_vivo_primary"


def test_attach_orthogonal_cell_lineage_vectorized():
    df = pd.DataFrame(
        [
            {"cell_line_name": "HeLa", "condition_cytokines": "IFNG"},
            {"cell_name": "autologous moDC", "culture_condition": "LPS"},
            {"cell_line_name": "K562", "is_monoallelic": True},
        ]
    )
    enriched = attach_orthogonal_cell_lineage(df)
    assert enriched["cell_lineage"].tolist() == ["epithelial", "moDC", "myeloid_leukemia"]
    assert enriched["malignancy_status"].tolist() == [
        "tumor_cell_line",
        "healthy_primary",
        "engineered_monoallelic_host",
    ]
    assert enriched["tumor_type"].tolist() == ["cervical_carcinoma", "none", "cml"]
    assert enriched["activation_state"].tolist() == ["ifng_stimulated", "activated_lps", "unknown"]


# ── #945: Structured Perturbation Contrasts & Expression Provenance ──────────


def test_extract_structured_perturbation_contrasts_from_curated_overrides():
    # PMID 30484775 (Ilca 2018 TAPBPR rescue on TAPBPR_KO background) and
    # PMID 40113210 (Shapiro 2025 HAP1 APM KO panel)
    contrasts = extract_structured_perturbation_contrasts(pmids=[30484775, 40113210])
    assert not contrasts.empty
    ilca = contrasts[contrasts["pmid"] == 30484775]
    assert len(ilca) == 4
    # TAPBPR rescue arms on TAPBPR_KO baseline engineering are classified as 'rescue', not 'knockout'
    assert set(ilca["intervention_type"]) == {"rescue"}
    assert set(ilca["target_gene"]) == {"TAPBPL"}
    assert set(ilca["baseline_engineering"]) == {"TAPBPR_KO"}
    wt_arm = ilca[ilca["arm_id"] == "helam_tapbpr_ko_ifng_tapbpr_wt"]
    assert bool(wt_arm["is_control_arm"].iloc[0]) is True
    oloop = ilca[ilca["arm_id"] == "helam_tapbpr_ko_ifng_tapbpr_oloop"]
    assert oloop["control_arm_id"].iloc[0] == "helam_tapbpr_ko_ifng_tapbpr_wt"


def test_tag_expression_provenance_tiers():
    df = pd.DataFrame(
        [
            {
                "expression_backend": "depmap_rna",
                "expression_match_tier": 1,
                "source_id": "DepMap_24Q4_gene",
            },
            {
                "expression_backend": "depmap_rna",
                "expression_match_tier": 2,
                "source_id": "DepMap_24Q4_gene",
            },
            {
                "expression_backend": "encode_rnaseq",
                "expression_match_tier": 1,
                "source_id": "ENCODE_GM12878",
            },
            {
                "expression_backend": "study_matched_measured",
                "expression_match_tier": 1,
                "source_id": "study",
            },
            {"expression_backend": "dice", "expression_match_tier": 1, "source_id": "DICE_moDC"},
        ]
    )
    tagged = tag_expression_provenance(df)
    assert tagged["expression_provenance_tier"].tolist() == [
        "reference_cell_line_ccle_depmap",
        "parent_line_proxy",
        "reference_cell_line_encode",
        "study_matched_measured",
        "reference_primary_immune_dice_monaco",
    ]


# ── #946: Biophysical Stability & Kinetic Covariates + Unit Audit ────────────


def test_extract_biophysical_covariates_separates_protocols_temp_ph_and_catalysts():
    spa = extract_biophysical_covariates(
        assay_method="scintillation proximity assay",
        response_measured="half life",
        source="netmhcstabpan",
        mhc_class="I",
    )
    assert spa.assay_protocol == "spa_37c"
    assert spa.assay_temperature_c == 37.0
    assert spa.catalyst_context == "none"
    assert spa.assay_floor_value == 0.0

    dm_class_ii = extract_biophysical_covariates(
        assay_method="fluorescent peptide exchange",
        response_measured="half life",
        assay_comments="Dissociation at pH 5.2, 37 C in presence of HLA-DM",
        mhc_class="II",
    )
    assert dm_class_ii.assay_protocol == "fluorescent_peptide_exchange"
    assert dm_class_ii.assay_temperature_c == 37.0
    assert dm_class_ii.assay_ph == 5.2
    assert dm_class_ii.catalyst_context == "hla_dm"

    no_dm = extract_biophysical_covariates(
        assay_method="fluorescent peptide exchange",
        response_measured="half life",
        assay_comments="Measured at pH 7.4 without HLA-DM",
        mhc_class="II",
    )
    assert no_dm.assay_ph == 7.4
    assert no_dm.catalyst_context == "none"

    dsf = extract_biophysical_covariates(
        assay_method="differential scanning fluorimetry (nanoDSF)",
        response_measured="50% dissociation temperature",
        mhc_class="I",
    )
    assert dsf.assay_protocol == "dsf_tm"

    spr = extract_biophysical_covariates(
        assay_method="surface plasmon resonance",
        response_measured="off rate",
        assay_comments="Biacore at 25°C, pH 7.4",
        mhc_class="I",
    )
    assert spr.assay_protocol == "spr_kinetics"
    assert spr.assay_temperature_c == 25.0
    assert spr.assay_ph == 7.4


def test_koff_kon_unit_conversion_and_stability_training_covariates():
    df = pd.DataFrame(
        [
            {
                "evidence_kind": "binding",
                "evidence_row_id": "b1",
                "peptide": "GILGFVFTL",
                "mhc_restriction": "HLA-A*02:01",
                "mhc_class": "I",
                "mhc_species": "Homo sapiens",
                "is_binding_assay": True,
                "assay_method": "scintillation proximity assay",
                "response_measured": "half life",
                "quantitative_value": 4.0,
                "measurement_units": "hours",
                "measurement_inequality": "=",
                "source": "netmhcstabpan",
                "pmid": "24470595",
            },
            {
                "evidence_kind": "binding",
                "evidence_row_id": "b2",
                "peptide": "SIINFEKL",
                "mhc_restriction": "H2-Kb",
                "mhc_class": "I",
                "mhc_species": "Mus musculus",
                "is_binding_assay": True,
                "assay_method": "surface plasmon resonance",
                "response_measured": "off rate",
                "quantitative_value": 0.06,
                "measurement_units": "min^-1",
                "measurement_inequality": "=",
                "assay_comments": "25 C pH 7.4",
                "pmid": "12345678",
            },
            {
                "evidence_kind": "binding",
                "evidence_row_id": "b3",
                "peptide": "SIINFEKL",
                "mhc_restriction": "H2-Kb",
                "mhc_class": "I",
                "mhc_species": "Mus musculus",
                "is_binding_assay": True,
                "assay_method": "surface plasmon resonance",
                "response_measured": "on rate",
                "quantitative_value": 2.5,
                "measurement_units": "mM^-1 s^-1",
                "measurement_inequality": "=",
                "assay_comments": "25 C pH 7.4",
                "pmid": "12345678",
            },
            {
                "evidence_kind": "binding",
                "evidence_row_id": "b4",
                "peptide": "PKYVKQNTLKLAT",
                "mhc_restriction": "HLA-DRA*01:01/DRB1*01:01",
                "mhc_class": "II",
                "mhc_species": "Homo sapiens",
                "is_binding_assay": True,
                "assay_method": "differential scanning fluorimetry",
                "response_measured": "50% dissociation temperature",
                "quantitative_value": 64.5,
                "measurement_units": "°C",
                "measurement_inequality": "=",
                "assay_comments": "pH 5.2 with HLA-DM",
                "pmid": "22222222",
            },
            # Zero-hour SPA floor row for audit check
            {
                "evidence_kind": "binding",
                "evidence_row_id": "b5",
                "peptide": "AAAAAAAAA",
                "mhc_restriction": "HLA-A*02:01",
                "mhc_class": "I",
                "mhc_species": "Homo sapiens",
                "is_binding_assay": True,
                "assay_method": "scintillation proximity assay",
                "response_measured": "half life",
                "quantitative_value": 0.0,
                "measurement_units": "hours",
                "measurement_inequality": "=",
                "source": "netmhcstabpan",
                "pmid": "24470595",
            },
        ]
    )

    q = qualify_binding_measurements(df)
    assert q.loc[0, "standardized_value"] == pytest.approx(240.0)
    assert q.loc[0, "standardized_unit"] == "min"
    assert q.loc[1, "standardized_value"] == pytest.approx(0.001)
    assert q.loc[1, "standardized_unit"] == "s^-1"
    assert q.loc[2, "standardized_value"] == pytest.approx(2500.0)
    assert q.loc[2, "standardized_unit"] == "M^-1 s^-1"
    assert q.loc[3, "standardized_value"] == pytest.approx(64.5)
    assert q.loc[3, "standardized_unit"] == "celsius"

    enriched = qualify_stability_and_kinetics(df)
    audit = audit_stability_normalization(enriched)
    assert audit["n_stability_kinetic_rows"] == 5
    assert audit["n_zero_half_life_rows"] == 1
    assert audit["n_rows_with_temperature"] == 4
    assert audit["n_rows_with_ph"] == 3
    assert audit["by_endpoint_family"] == {
        "half_life": 2,
        "off_rate": 1,
        "on_rate": 1,
        "thermal_stability": 1,
    }

    # Multi-species prepare_stability_training_data with KINETIC_STABILITY_ENDPOINTS
    res = prepare_stability_training_data(
        df,
        endpoints=KINETIC_STABILITY_ENDPOINTS,
        species="all",
    )
    # b5 (0.0h) is excluded as non_positive_measurement_value; b1..b4 accepted!
    assert len(res.accepted) == 4
    assert len(res.excluded) == 1
    acc = res.accepted.set_index("evidence_row_id")
    assert acc.loc["b1", "assay_protocol"] == "spa_37c"
    assert acc.loc["b1", "assay_temperature_c"] == 37.0
    assert acc.loc["b2", "assay_protocol"] == "spr_kinetics"
    assert acc.loc["b2", "measurement_unit"] == "s^-1"
    assert acc.loc["b4", "assay_protocol"] == "dsf_tm"
    assert math.isclose(acc.loc["b4", "assay_ph"], 5.2)
    assert acc.loc["b4", "catalyst_context"] == "hla_dm"
