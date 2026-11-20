"""Unit and real-corpus regression tests for APC/cytokine contrasts (#904) and DLBCL dose conflicts (#930)."""

from __future__ import annotations

import pandas as pd
import pytest

from hitlist import (
    apc_stimulus_protocol_contract,
    build_stimulus_contrast_catalog,
    dlbcl_treatment_conflict_contract,
    prepare_stimulus_contrast_data,
)
from hitlist.curation import validate_elution_condition_ids
from hitlist.export import expand_condition_memberships, generate_observations_table


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
            "assay_iri": f"http://www.iedb.org/assay/{5000 + idx}",
            "reference_iri": f"http://www.iedb.org/reference/{item['pmid']}",
            **item,
        }
        records.append(rec)
    frame = pd.DataFrame(records)
    monkeypatch.setattr("hitlist.observations.load_observations", lambda **kw: frame.copy())


def test_dlbcl_35561310_treatment_conflict_contract_and_unit_invariants():
    contract = dlbcl_treatment_conflict_contract()
    assert contract["pmid"] == 35561310
    assert contract["pmcid"] == "PMC9327544"
    assert (
        contract["primary_sources_verified"]["supplemental_pdf"]["sha256"]
        == "bfae2561be9bf6e031cdc3d420d52ef00a30908435d3bf60d0e4a821cece06f1"
    )
    inv = contract["unit_conversion_invariants"]
    # 1 nmol/mL == 1 uM == 1000 nM (never normalize nmol/mL as nM)
    assert inv["nmol_per_ml_equals_uM"] is True
    assert inv["nmol_per_ml_equals_nM"] is False
    assert inv["ug_per_ml_to_ng_per_ml_factor"] == 1000.0

    by_id = {s["statement_id"]: s for s in contract["treatment_statements"]}
    assert by_id["pmc9327544:fig5:decitabine"]["literal_unit"] == "nM"
    assert by_id["pmc9327544:fig5:decitabine"]["equivalent_nM"] == 125.0
    assert by_id["pmc9327544:methods_drug_treatments:decitabine"]["literal_unit"] == "nmol/mL"
    assert by_id["pmc9327544:methods_drug_treatments:decitabine"]["equivalent_nM_range"] == [
        125_000.0,
        2_000_000.0,
    ]
    assert (
        by_id["pmc9327544:methods_drug_treatments:decitabine"]["conflict_status"]
        == "conflicting_unit_1000x_nmol_per_ml_vs_nM"
    )
    assert by_id["pmc9327544:fig5:ifng"]["literal_unit"] == "ng/mL"
    assert by_id["pmc9327544:suppl_fig10:ifng"]["literal_unit"] == "ug/ml"
    assert by_id["pmc9327544:suppl_fig10:ifng"]["equivalent_ng_per_ml"] == 100_000.0
    assert (
        by_id["pmc9327544:suppl_fig10:ifng"]["conflict_status"]
        == "conflicting_unit_1000x_ug_per_ml_vs_ng_per_ml"
    )
    # Transcription endpoint (Figures 2-3) remains separate from peptide detection (Figure 5)
    assert by_id["pmc9327544:figs2_3:decitabine"]["endpoint_kind"] == "transcript_expression"
    assert by_id["pmc9327544:figs2_3:ifng"]["endpoint_kind"] == "transcript_expression"
    assert by_id["pmc9327544:fig5:ifng"]["endpoint_kind"] == "peptide_detection"


def test_stimulus_contrast_catalog_separates_isolated_cocktails_and_non_apc():
    proto = apc_stimulus_protocol_contract()
    assert "IFNG" in proto["factor_classes"]["cytokines"]
    assert "LPS" in proto["factor_classes"]["non_cytokine_stimuli"]
    assert "PMA" in proto["factor_classes"]["non_cytokine_stimuli"]

    cat = build_stimulus_contrast_catalog()
    by_key = cat.set_index(["pmid", "condition_id"])

    # True isolated IFNG contrasts with matched comparator
    assert (
        by_key.loc[(35561310, "db_ifng"), "contrast_eligibility_status"]
        == "isolated_cytokine_contrast"
    )
    assert by_key.loc[(35561310, "db_ifng"), "comparator_condition_ids"] == "db_dmso"
    assert by_key.loc[(35561310, "db_ifng"), "delta_cytokines"] == "IFNG"
    assert (
        by_key.loc[(35561310, "db_ifng"), "dose_conflict_status"]
        == "source_dose_unit_conflict_documented"
    )
    assert (
        by_key.loc[(32488085, "sk_mel_5_ifng"), "contrast_eligibility_status"]
        == "isolated_cytokine_contrast"
    )
    assert (
        by_key.loc[(32938616, "lm_mel_44_ifn_gamma"), "contrast_eligibility_status"]
        == "isolated_cytokine_contrast"
    )
    assert (
        by_key.loc[(33968037, "mda_mb_231_ifn_gamma_class_i"), "contrast_eligibility_status"]
        == "isolated_cytokine_contrast"
    )
    # Class II on MDA-MB-231 has no untreated Class II baseline -> unpaired context-only
    assert (
        by_key.loc[(33968037, "mda_mb_231_ifn_gamma_class_ii"), "contrast_eligibility_status"]
        == "unpaired_cytokine_context_only"
    )

    # Multifactor & differentiation cocktails are never isolated cytokine contrasts
    assert (
        by_key.loc[(35561310, "db_ifng_dac_taz"), "contrast_eligibility_status"]
        == "multifactor_combination_contrast"
    )
    assert (
        by_key.loc[(30833945, "a549_lung_cancer_tnfa_ifng"), "contrast_eligibility_status"]
        == "multifactor_combination_contrast"
    )
    assert (
        by_key.loc[(32983136, "immature_dcs"), "contrast_eligibility_status"]
        == "differentiation_cocktail_contrast"
    )
    assert (
        by_key.loc[(32983136, "immature_dcs"), "comparator_condition_ids"]
        == "cd14_monocytes_resting"
    )
    assert (
        by_key.loc[(32983136, "mature_dcs"), "contrast_eligibility_status"]
        == "multifactor_maturation_contrast_on_differentiated_background"
    )
    assert by_key.loc[(32983136, "mature_dcs"), "comparator_condition_ids"] == "immature_dcs"
    assert by_key.loc[(32983136, "mature_dcs"), "shared_background_cytokines"] == "CSF2;IL4"
    assert by_key.loc[(32983136, "mature_dcs"), "delta_cytokines"] == "IFNG"
    assert by_key.loc[(32983136, "mature_dcs"), "delta_stimuli"] == "LPS"

    # Constant cocktails and unpaired cytokine arms are context-only
    assert (
        by_key.loc[
            (38920720, "mature_modcs_pulsed_with_synthetic_tsa_peptides"),
            "contrast_eligibility_status",
        ]
        == "constant_cocktail_context_only"
    )
    assert (
        by_key.loc[(32983136, "cd19_b_cells_activated"), "contrast_eligibility_status"]
        == "constant_cocktail_context_only"
    )
    assert (
        by_key.loc[(30484775, "helam_tapbpr_ko_ifng_tapbpr_oloop"), "contrast_eligibility_status"]
        == "constant_cytokine_background_other_contrast"
    )

    # Expanded melanoma TILs (PMID 28832583) are non-APC expanded T lymphocytes, not APCs
    til1 = by_key.loc[(28832583, "til1_melanoma_til_expanded")]
    assert til1["contrast_eligibility_status"] == "expansion_protocol_non_apc_context_only"
    assert bool(til1["is_measured_apc"]) is False
    assert til1["measured_population"] == "expanded_t_lymphocyte"
    assert til1["partner_cell_population"] == "irradiated_feeder_cells"


def test_elution_token_condition_ids_validation_and_marino_ghosh_attribution(monkeypatch):
    # Validation rejects unknown condition IDs or malformed token specs
    with pytest.raises(ValueError, match="unknown or unprofiled"):
        validate_elution_condition_ids(
            {
                "pmid": 99999999,
                "ms_samples": [{"condition_id": "arm_a"}],
                "elution_token_condition_ids": {
                    "prefix": "Cell types:",
                    "tokens": {"A": "missing_arm"},
                },
            }
        )

    _mock_observations(
        monkeypatch,
        [
            # Marino 2020 (32983136): single-subset MaDC, single-subset ImmDC spelling,
            # multi-subset shared CD8+MaDC (must NOT be misattributed to cd8_t_cells_activated!)
            {
                "peptide": "MARINOMADC1",
                "pmid": 32983136,
                "mhc_class": "I",
                "mhc_restriction": "HLA-A*02:01",
                "cell_name": "Dendritic cell",
                "assay_comments": "The epitope was eluted from the following cell types: MaDC.",
            },
            {
                "peptide": "MARINOIMMDC",
                "pmid": 32983136,
                "mhc_class": "II",
                "mhc_restriction": "HLA-DRB1*01:01",
                "cell_name": "Dendritic cell",
                "assay_comments": "The epitope was eluted from the following cell types: ImmDC.",
            },
            {
                "peptide": "MARINOSHARED",
                "pmid": 32983136,
                "mhc_class": "I",
                "mhc_restriction": "HLA-A*02:01",
                "cell_name": "T cell CD8+",
                "assay_comments": "The epitope was eluted from the following cell types: CD8, MaDC.",
            },
            # Ghosh 2020 (33392160): unattributed THP-1 rows must NOT receive a guessed LPS arm
            {
                "peptide": "GHOSHTHP1A",
                "pmid": 33392160,
                "mhc_class": "I",
                "mhc_restriction": "HLA-A*02:01",
                "cell_name": "THP-1-Monocyte",
                "assay_comments": "",
            },
            {
                "peptide": "GHOSHTHP1B",
                "pmid": 33392160,
                "mhc_class": "II",
                "mhc_restriction": "HLA-DRB1*01:01",
                "cell_name": "THP-1-Monocyte",
                "assay_comments": "HLA Class II peptides were also immunoprecipitated with the HLA-DR specific mAb, L243.",
            },
        ],
    )
    obs = generate_observations_table(exclude_non_peptide_ligand=False).set_index("peptide")
    assert obs.loc["MARINOMADC1", "condition_id"] == "mature_dcs"
    assert obs.loc["MARINOMADC1", "supported_condition_ids"] == "mature_dcs"
    assert obs.loc["MARINOIMMDC", "condition_id"] == "immature_dcs"
    assert obs.loc["MARINOSHARED", "condition_id"] == ""
    assert obs.loc["MARINOSHARED", "supported_condition_ids"] == "cd8_t_cells_activated;mature_dcs"
    assert obs.loc["GHOSHTHP1A", "condition_id"] == ""
    assert obs.loc["GHOSHTHP1B", "condition_id"] == ""

    edges = expand_condition_memberships(obs.reset_index())
    shared_edges = edges[edges["peptide"] == "MARINOSHARED"]
    assert set(shared_edges["condition_id"]) == {"cd8_t_cells_activated", "mature_dcs"}
    assert set(shared_edges["membership_status"]) == {"detected"}


def test_prepare_stimulus_contrast_data_conserves_rows_and_guards_class_pool_and_cocktails():
    df = pd.DataFrame(
        [
            # 1. Valid DLBCL DMSO comparator (Class I)
            {
                "provenance_id": "p1",
                "pmid": 35561310,
                "peptide": "SLYNTVATL",
                "mhc_class": "I",
                "condition_id": "db_dmso",
                "supported_condition_ids": "db_dmso",
                "sample_mhc_origin": "sample",
                "mhc_allele_provenance": "sample_allele_match",
            },
            # 2. Valid DLBCL isolated IFNG (Class II, carries #930 conflict metadata)
            {
                "provenance_id": "p2",
                "pmid": 35561310,
                "peptide": "PKYVKQNTLKLAT",
                "mhc_class": "II",
                "condition_id": "db_ifng",
                "supported_condition_ids": "db_ifng",
                "sample_mhc_origin": "sample",
                "mhc_allele_provenance": "sample_allele_match",
            },
            # 3. Class-pool IFNG row -> MUST be excluded by default (never guessed treatment)
            {
                "provenance_id": "p3",
                "pmid": 35561310,
                "peptide": "GILGFVFTL",
                "mhc_class": "I",
                "condition_id": "db_ifng",
                "supported_condition_ids": "db_ifng",
                "sample_mhc_origin": "class_pool",
                "mhc_allele_provenance": "pmid_class_pool",
            },
            # 4. Hesnard 2024 constant cocktail (38920720) -> excluded as context-only
            {
                "provenance_id": "p4",
                "pmid": 38920720,
                "peptide": "EVDPIGHLY",
                "mhc_class": "I",
                "condition_id": "mature_modcs_pulsed_with_synthetic_tsa_peptides",
                "supported_condition_ids": "mature_modcs_pulsed_with_synthetic_tsa_peptides",
                "sample_mhc_origin": "sample",
                "mhc_allele_provenance": "sample_allele_match",
            },
            # 5. Marino 2020 mature_dcs (multifactor IFNG+LPS on CSF2+IL4) -> excluded unless allow_multifactor_contrasts
            {
                "provenance_id": "p5",
                "pmid": 32983136,
                "peptide": "NLVPMVATV",
                "mhc_class": "I",
                "condition_id": "mature_dcs",
                "supported_condition_ids": "mature_dcs",
                "sample_mhc_origin": "sample",
                "mhc_allele_provenance": "sample_allele_match",
            },
            # 6. Marino 2020 shared multi-arm row -> excluded with edge-expansion guidance
            {
                "provenance_id": "p6",
                "pmid": 32983136,
                "peptide": "CLGGLLTMV",
                "mhc_class": "I",
                "condition_id": "",
                "supported_condition_ids": "immature_dcs;mature_dcs",
                "sample_mhc_origin": "sample",
                "mhc_allele_provenance": "sample_allele_match",
            },
            # 7. Bassani-Sternberg 2017 expanded TIL (non-APC) -> excluded
            {
                "provenance_id": "p7",
                "pmid": 28832583,
                "peptide": "YVLDHLIVV",
                "mhc_class": "I",
                "condition_id": "til1_melanoma_til_expanded",
                "supported_condition_ids": "til1_melanoma_til_expanded",
                "sample_mhc_origin": "sample",
                "mhc_allele_provenance": "sample_allele_match",
            },
            # 8. Ghosh 2020 unattributed THP-1 row -> excluded as unattributed
            {
                "provenance_id": "p8",
                "pmid": 33392160,
                "peptide": "GLCTLVAML",
                "mhc_class": "II",
                "condition_id": "",
                "supported_condition_ids": "",
                "sample_mhc_origin": "sample",
                "mhc_allele_provenance": "sample_allele_match",
            },
        ]
    )

    sel = prepare_stimulus_contrast_data(df, target_factor="IFNG")
    assert len(sel.accepted) + len(sel.excluded) == len(df)
    assert set(sel.accepted["provenance_id"]) == {"p1", "p2"}
    assert set(sel.accepted["mhc_class"]) == {"I", "II"}
    assert (
        sel.accepted.loc[sel.accepted["provenance_id"] == "p2", "dose_conflict_status"].iloc[0]
        == "source_dose_unit_conflict_documented"
    )
    excl_by_id = sel.excluded.set_index("provenance_id")["exclusion_reason"].to_dict()
    assert excl_by_id == {
        "p3": "class_pool_attribution_excluded",
        "p4": "constant_cocktail_context_only",
        "p5": "multifactor_contrast_excluded",
        "p6": "shared_multi_arm_requires_edge_expansion",
        "p7": "expansion_protocol_non_apc_context_only",
        "p8": "unattributed_or_mixed_population_excluded",
    }

    # Enabling allow_multifactor_contrasts admits p5 (mature_dcs IFNG+LPS) while still rejecting cocktails/TIL/class-pool
    sel_multi = prepare_stimulus_contrast_data(
        df, target_factor="IFNG", allow_multifactor_contrasts=True
    )
    assert set(sel_multi.accepted["provenance_id"]) == {"p1", "p2", "p5"}


@pytest.mark.integration
def test_real_corpus_marino_ghosh_hesnard_dlbcl_reconciliation():
    import pyarrow.dataset as ds

    from hitlist.curation import load_pmid_overrides
    from hitlist.export import _statement_maps, generate_ms_samples_table
    from hitlist.observations import is_built, observations_path

    if not is_built():
        pytest.skip("Observations corpus not built")

    dset = ds.dataset(observations_path(), format="parquet")
    smap = _statement_maps(load_pmid_overrides(), generate_ms_samples_table())[32983136]

    t32 = dset.to_table(columns=["assay_comments"], filter=ds.field("pmid") == 32983136).to_pandas()
    assert len(t32) == 34_606
    single_counts: dict[str, int] = {}
    n_multi = 0
    n_unmapped = 0
    for stmt, cnt in t32["assay_comments"].value_counts(dropna=False).items():
        allowed = smap.allowed("" if pd.isna(stmt) else str(stmt))
        if allowed is None:
            n_unmapped += int(cnt)
        elif len(allowed) == 1:
            single_counts[allowed[0]] = single_counts.get(allowed[0], 0) + int(cnt)
        else:
            n_multi += int(cnt)
    assert n_unmapped == 0
    assert sum(single_counts.values()) == 6_819
    assert n_multi == 27_787
    assert single_counts == {
        "mature_dcs": 4_730,
        "cd19_b_cells_activated": 1_046,
        "immature_dcs": 370,
        "cd14_monocytes_resting": 363,
        "cd4_t_cells_activated": 197,
        "cd8_t_cells_activated": 113,
    }
