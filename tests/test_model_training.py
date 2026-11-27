import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import hitlist
from hitlist import (
    prepare_affinity_training_data,
    prepare_ms_training_data,
    prepare_stability_training_data,
)
from hitlist.model_training import (
    BINDING_ENDPOINT_FAMILIES,
    MOLAR_AFFINITY_ENDPOINTS,
    STABILITY_ENDPOINTS,
    TrainingTableSelection,
    qualify_binding_measurements,
    to_mhcflurry_presentation_inputs,
)
from hitlist.observations import is_binding_built, load_binding


def test_qualify_binding_measurements_endpoints_units_censoring_and_non_binding():
    empty = qualify_binding_measurements(pd.DataFrame())
    assert {
        "binding_endpoint_family",
        "standardized_value",
        "standardized_unit",
        "unit_conversion_status",
        "censoring_status",
        "normalized_inequality",
        "assay_preparation",
    } <= set(empty.columns)

    df = pd.DataFrame(
        {
            "evidence_kind": [
                "binding",
                "binding",
                "binding",
                "binding",
                "binding",
                "binding",
                "binding",
                "binding",
                "binding",
                "binding",
                "binding",
                "ms",
            ],
            "assay_method": [
                "purified MHC/competitive/radioactivity",
                "cellular MHC/competitive/fluorescence",
                "purified MHC/direct/fluorescence",
                "purified MHC/direct/fluorescence",
                "purified MHC/direct/radioactivity",
                "purified MHC/direct/fluorescence",
                "purified MHC/competitive/radioactivity",
                "purified MHC/competitive/radioactivity",
                "purified MHC/competitive/radioactivity",
                "purified MHC/competitive/radioactivity",
                "purified MHC/direct/fluorescence",
                "mass spectrometry",
            ],
            "response_measured": [
                "half maximal inhibitory concentration (IC50)",
                "dissociation constant KD (~IC50)",
                "dissociation constant KD",
                "half life",
                "50% dissociation temperature",
                "qualitative binding",
                "half maximal inhibitory concentration (IC50)",
                "half maximal inhibitory concentration (IC50)",
                "half maximal inhibitory concentration (IC50)",
                "half maximal inhibitory concentration (IC50)",
                "association constant KA",
                "qualitative binding",
            ],
            "quantitative_value": [
                50.0,
                0.25,
                500.0,
                2.0,
                58.5,
                1.2,
                100.0,
                np.nan,
                2.3,
                10.0,
                1e7,
                np.nan,
            ],
            "measurement_units": [
                "nM",
                "uM",
                "pM",
                "h",
                "°C",
                "",
                "m",  # lowercase "m" must never be treated as Molar
                "nM",
                "log10(IC50)",
                "nM",
                "1/M",
                "",
            ],
            "measurement_inequality": [
                "=",
                "",
                "<=",
                ">=",
                "=",
                "",
                "=",
                "=",
                "=",
                "~",
                "=",
                "",
            ],
        }
    )
    out = qualify_binding_measurements(df)
    assert len(out) == len(df)
    assert set(out["binding_endpoint_family"]) <= set(BINDING_ENDPOINT_FAMILIES)

    assert out.loc[0, "binding_endpoint_family"] == "ic50"
    assert out.loc[0, "standardized_value"] == pytest.approx(50.0)
    assert out.loc[0, "standardized_unit"] == "nM"
    assert out.loc[0, "unit_conversion_status"] == "canonical_unit"
    assert out.loc[0, "censoring_status"] == "exact"
    assert out.loc[0, "normalized_inequality"] == "="
    assert out.loc[0, "donor_applicability"] == "not_applicable"

    # 0.25 uM -> 250 nM, blank operator preserved as unreported_operator_numeric.
    assert out.loc[1, "binding_endpoint_family"] == "kd_approx_ic50"
    assert out.loc[1, "standardized_value"] == pytest.approx(250.0)
    assert out.loc[1, "unit_conversion_status"] == "linear_converted"
    assert out.loc[1, "censoring_status"] == "unreported_operator_numeric"
    assert out.loc[1, "normalized_inequality"] == "="
    assert out.loc[1, "donor_applicability"] == "applicable_unresolved"

    # 500 pM -> 0.5 nM, <= -> <
    assert out.loc[2, "standardized_value"] == pytest.approx(0.5)
    assert out.loc[2, "censoring_status"] == "left_censored_lte"
    assert out.loc[2, "normalized_inequality"] == "<"

    # 2.0 h -> 120.0 min, >= -> >
    assert out.loc[3, "binding_endpoint_family"] == "half_life"
    assert out.loc[3, "standardized_value"] == pytest.approx(120.0)
    assert out.loc[3, "standardized_unit"] == "min"
    assert out.loc[3, "censoring_status"] == "right_censored_gte"
    assert out.loc[3, "normalized_inequality"] == ">"

    # 58.5 °C -> 58.5 celsius
    assert out.loc[4, "binding_endpoint_family"] == "thermal_stability"
    assert out.loc[4, "standardized_value"] == pytest.approx(58.5)
    assert out.loc[4, "standardized_unit"] == "celsius"

    # Unitless numeric qualitative binding is never converted to nM.
    assert out.loc[5, "binding_endpoint_family"] == "qualitative_binding"
    assert np.isnan(out.loc[5, "standardized_value"])
    assert out.loc[5, "unit_conversion_status"] == "not_applicable"

    # Lowercase "m" and nonlinear "log10(IC50)" are rejected.
    assert out.loc[6, "unit_conversion_status"] == "nonlinear_or_unsupported_unit"
    assert np.isnan(out.loc[6, "standardized_value"])
    assert out.loc[8, "unit_conversion_status"] == "nonlinear_or_unsupported_unit"
    assert np.isnan(out.loc[8, "standardized_value"])

    # Operator "=" with NaN value -> operator_without_value.
    assert out.loc[7, "censoring_status"] == "operator_without_value"
    assert out.loc[7, "normalized_inequality"] == ""

    # Unrecognized operator "~".
    assert out.loc[9, "censoring_status"] == "unrecognized_inequality"

    # Non-binding row untouched as not_binding.
    assert out.loc[11, "binding_endpoint_family"] == "not_binding"
    assert out.loc[11, "censoring_status"] == "not_applicable"


def test_prepare_affinity_and_stability_policies_and_row_conservation():
    rows = pd.DataFrame(
        {
            "evidence_row_id": [f"binding:http://iedb.org/assay/{i}" for i in range(1, 13)],
            "evidence_source_id": [f"http://iedb.org/assay/{i}" for i in range(1, 13)],
            "provenance_id": [f"observation:record:iedb:row:{i}" for i in range(1, 13)],
            "provenance_status": ["indexed"] * 12,
            "evidence_kind": ["binding"] * 11 + ["ms"],
            "assay_method": ["purified MHC/competitive/radioactivity"] * 11 + ["mass spectrometry"],
            "pmid": [1001] * 12,
            "peptide": [
                "SLYNTVATL",  # 0: valid human class-Ia four-field normalized to HLA-A*02:01
                "GILGFVFTL",  # 1: blank inequality numeric KD (~IC50) in uM
                "SIINFEKL",  # 2: mouse H2-K*b exact KD
                "PKYVKQNTLKLAT",  # 3: human single-chain HLA-DRB1*01:01 (monomorphic DRA)
                "PKYVKQNTLKLAT",  # 4: human single-chain HLA-DPB1*04:01 (unpaired polymorphic alpha)
                "PKYVKQNTLKLAT",  # 5: complete class-II pair HLA-DPA1*01:03/DPB1*04:01
                "NLVPMVATV",  # 6: non-classical HLA-E*01:01
                "SLYNTVATL",  # 7: censored > 5000 nM
                "SLYNTVATL",  # 8: zero nM (non-positive molar value)
                "SLYN+57TVATL",  # 9: PTM modified peptide
                "SLYNTVATL",  # 10: half life in minutes
                "SLYNTVATL",  # 11: MS row (not candidate binding row)
            ],
            "has_ptm": [False] * 9 + [True, False, False],
            "is_non_peptide_ligand": [False] * 12,
            "mhc_restriction": [
                "HLA-A*02:01:01:01",
                "HLA-A*02:01",
                "H2-K*b",
                "HLA-DRB1*01:01",
                "HLA-DPB1*04:01",
                "HLA-DPA1*01:03/DPB1*04:01",
                "HLA-E*01:01",
                "HLA-A*02:01",
                "HLA-A*02:01",
                "HLA-A*02:01",
                "HLA-A*02:01",
                "HLA-A*02:01",
            ],
            "response_measured": [
                "half maximal inhibitory concentration (IC50)",
                "dissociation constant KD (~IC50)",
                "dissociation constant KD",
                "half maximal inhibitory concentration (IC50)",
                "half maximal inhibitory concentration (IC50)",
                "half maximal inhibitory concentration (IC50)",
                "dissociation constant KD",
                "half maximal inhibitory concentration (IC50)",
                "half maximal inhibitory concentration (IC50)",
                "half maximal inhibitory concentration (IC50)",
                "half life",
                "qualitative binding",
            ],
            "quantitative_value": [
                12.0,
                0.05,
                4.0,
                25.0,
                30.0,
                18.0,
                90.0,
                5000.0,
                0.0,
                15.0,
                180.0,
                np.nan,
            ],
            "measurement_units": [
                "nM",
                "uM",
                "nM",
                "nM",
                "nM",
                "nM",
                "nM",
                "nM",
                "nM",
                "nM",
                "min",
                "",
            ],
            "measurement_inequality": ["=", "", "=", "=", "=", "=", "=", ">", "=", "=", "=", ""],
        }
    )

    mhcflurry_sel = prepare_affinity_training_data(
        rows, scorable_class_i_only=True, min_peptide_length=8, max_peptide_length=15
    )
    assert isinstance(mhcflurry_sel, TrainingTableSelection)
    assert len(mhcflurry_sel.accepted) + len(mhcflurry_sel.excluded) == 11
    assert list(mhcflurry_sel.accepted["allele"]) == ["HLA-A*02:01", "HLA-A*02:01", "HLA-A*02:01"]
    assert list(mhcflurry_sel.accepted["measurement_value"]) == pytest.approx([12.0, 50.0, 5000.0])
    assert list(mhcflurry_sel.accepted["measurement_inequality"]) == ["=", "=", ">"]
    assert list(mhcflurry_sel.accepted["censoring_status"]) == [
        "exact",
        "unreported_operator_numeric",
        "right_censored_gt",
    ]
    excl_reasons = dict(
        zip(mhcflurry_sel.excluded["mhc_restriction"], mhcflurry_sel.excluded["exclusion_reason"])
    )
    assert excl_reasons["H2-K*b"] == "not_scorable_class_i_allele"
    assert excl_reasons["HLA-E*01:01"] == "not_scorable_class_i_allele"

    # Strict blank inequality + uncensored only + literal IC50 only:
    strict_ic50 = prepare_affinity_training_data(
        rows,
        endpoints=("ic50",),
        blank_inequality_policy="exclude",
        allow_censored=False,
        species=None,
    )
    assert len(strict_ic50.accepted) + len(strict_ic50.excluded) == 11
    # Accepts row 0 (HLA-A*02:01:01:01), row 3 (HLA-DRB1*01:01), row 5 (HLA-DPA1*01:03/DPB1*04:01).
    # Excludes row 4 (unpaired HLA-DPB1*04:01), row 1 (kd_approx_ic50), row 7 (censored >).
    assert set(strict_ic50.accepted["allele_completeness"]) == {
        "complete_class_i",
        "single_chain_class_ii_dr",
        "complete_class_ii_pair",
    }
    assert "unpaired_class_ii_chain" in set(strict_ic50.excluded["exclusion_reason"])
    assert "censored_measurement_excluded" in set(strict_ic50.excluded["exclusion_reason"])
    assert "non_positive_measurement_value" in set(strict_ic50.excluded["exclusion_reason"])
    assert "modified_peptide_excluded" in set(strict_ic50.excluded["exclusion_reason"])

    exclude_blank = prepare_affinity_training_data(
        rows, blank_inequality_policy="exclude", species=None
    )
    assert "unreported_inequality_excluded" in set(exclude_blank.excluded["exclusion_reason"])

    # Enabling allow_unpaired_class_ii_chains admits row 4 explicitly as single_chain_class_ii_unpaired.
    with_unpaired = prepare_affinity_training_data(
        rows,
        endpoints=("ic50",),
        allow_censored=False,
        allow_unpaired_class_ii_chains=True,
        species=None,
    )
    assert "single_chain_class_ii_unpaired" in set(with_unpaired.accepted["allele_completeness"])

    # Stability selects only half_life / thermal_stability, excluding 0.0 half-life
    # nondetections and mutant constructs unless allow_mutant_mhc=True (#875).
    stab = prepare_stability_training_data(rows, species=None)
    assert len(stab.accepted) == 1
    assert len(stab.excluded) == 10
    assert stab.accepted.loc[0, "measurement_kind"] == "half_life"
    assert stab.accepted.loc[0, "measurement_unit"] == "min"
    assert stab.accepted.loc[0, "measurement_value"] == pytest.approx(180.0)

    stab_edge_rows = pd.DataFrame(
        {
            "evidence_row_id": ["s:zero", "s:mut", "s:tm"],
            "evidence_source_id": ["s:zero", "s:mut", "s:tm"],
            "provenance_id": ["p:zero", "p:mut", "p:tm"],
            "evidence_kind": ["binding", "binding", "binding"],
            "assay_method": ["purified MHC/direct/radioactivity"] * 3,
            "pmid": [27402703] * 3,
            "peptide": ["SLYNTVATL", "GILGFVFTL", "NLVPMVATV"],
            "mhc_restriction": ["HLA-A*02:01", "HLA-B*14:01 C67S mutant", "HLA-A*02:01"],
            "response_measured": ["half life", "half life", "50% dissociation temperature"],
            "quantitative_value": [0.0, 1.5, 56.0],
            "measurement_units": ["h", "h", "°C"],
            "measurement_inequality": ["", "", "="],
        }
    )
    stab_default = prepare_stability_training_data(stab_edge_rows, species=None)
    assert len(stab_default.accepted) == 1
    assert stab_default.accepted.loc[0, "measurement_kind"] == "thermal_stability"
    assert set(stab_default.excluded["exclusion_reason"]) == {
        "non_positive_measurement_value",
        "mutant_mhc_construct_excluded",
    }
    stab_with_mut = prepare_stability_training_data(
        stab_edge_rows, species=None, allow_mutant_mhc=True
    )
    assert len(stab_with_mut.accepted) == 2
    mut_row = stab_with_mut.accepted.loc[
        stab_with_mut.accepted["allele"] == "HLA-B*14:01 C67S mutant"
    ].iloc[0]
    assert mut_row["allele_completeness"] == "mutant_class_i"
    assert mut_row["measurement_value"] == pytest.approx(90.0)
    assert set(stab_with_mut.excluded["exclusion_reason"]) == {"non_positive_measurement_value"}

    with pytest.raises(ValueError, match="Unsupported molar affinity endpoints"):
        prepare_affinity_training_data(rows, endpoints=("half_life",))
    with pytest.raises(ValueError, match="Unsupported stability endpoints"):
        prepare_stability_training_data(rows, endpoints=("ic50",))
    with pytest.raises(ValueError, match="blank_inequality_policy"):
        prepare_affinity_training_data(rows, blank_inequality_policy="guess")
    assert set(MOLAR_AFFINITY_ENDPOINTS).isdisjoint(STABILITY_ENDPOINTS)


def test_prepare_ms_training_data_collapses_mappings_and_resolves_flanks():
    # 5 distinct observations expanded to 8 rows via protein mappings:
    ms = pd.DataFrame(
        {
            "evidence_row_id": [
                "ms:1",
                "ms:2",
                "ms:2",
                "ms:3",
                "ms:3",
                "ms:4",
                "ms:4",
                "ms:5",
            ],
            "evidence_source_id": ["s1", "s2", "s2", "s3", "s3", "s4", "s4", "s5"],
            "provenance_id": ["p1", "p2", "p2", "p3", "p3", "p4", "p4", "p5"],
            "provenance_status": ["indexed"] * 8,
            "evidence_kind": ["ms"] * 8,
            "peptide": [
                "SLYNTVATL",
                "GILGFVFTL",
                "GILGFVFTL",
                "NLVPMVATV",
                "NLVPMVATV",
                "GLCTLVAML",
                "GLCTLVAML",
                "FLPSDFFPSV",
            ],
            "mhc_restriction": ["HLA-A*02:01"] * 8,
            "restriction_evidence": ["monoallelic"] * 8,
            "is_monoallelic": [True] * 8,
            "mhc_allele_set": ["HLA-A*02:01"] * 8,
            "mhc_allele_provenance": ["peptide_attribution"] * 8,
            "mhc_class": ["I"] * 8,
            "mhc_species": ["Homo sapiens"] * 8,
            "pmid": [31844290] * 8,
            "condition_id": ["arm_a0201"] * 8,
            "protein_id": [
                "ENSP1",
                "ENSP2A",
                "ENSP2B",
                "ENSP3A",
                "ENSP3B",
                "ENSP4A",
                "ENSP4B",
                "",
            ],
            "gene_id": [
                "ENSG1",
                "ENSG2A",
                "ENSG2B",
                "ENSG3A",
                "ENSG3B",
                "ENSG4A",
                "ENSG4B",
                "",
            ],
            "gene_name": ["G1", "G2A", "G2B", "G3A", "G3B", "G4A", "G4B", ""],
            "transcript_id": [
                "ENST1",
                "ENST2A",
                "ENST2B",
                "ENST3A",
                "ENST3B",
                "ENST4A",
                "ENST4B",
                "",
            ],
            "n_flank": ["AAAA", "BBBB", "BBBB", "CCCC", "DDDD", "EEEE", "FFFF", ""],
            "c_flank": ["WWWW", "XXXX", "XXXX", "YYYY", "ZZZZ", "UUUU", "VVVV", ""],
            "origin_gene_id": ["", "", "", "ENSG3B", "ENSG3B", "", "", ""],
        }
    )

    sel = prepare_ms_training_data(ms, scorable_class_i_only=True)
    assert sel.summary["n_input_rows"] == 8
    assert sel.summary["n_input_candidate_rows"] == 5
    assert sel.summary["n_mapping_alternative_rows_collapsed"] == 3
    assert len(sel.accepted) == 5
    assert len(sel.excluded) == 0

    by_id = sel.accepted.set_index("evidence_row_id")
    assert by_id.loc["ms:1", "flank_status"] == "unique_mapping"
    assert by_id.loc["ms:1", "n_flank"] == "AAAA"

    # ms:2: two genes sharing identical flanks -> concordant_flanks, scalar gene_id blanked.
    assert by_id.loc["ms:2", "flank_status"] == "concordant_flanks"
    assert by_id.loc["ms:2", "n_flank"] == "BBBB"
    assert by_id.loc["ms:2", "gene_id"] == ""
    assert by_id.loc["ms:2", "candidate_gene_ids"] == "ENSG2A;ENSG2B"

    # ms:3: conflicting flanks disambiguated by origin_gene_id == ENSG3B.
    assert by_id.loc["ms:3", "flank_status"] == "expression_disambiguated"
    assert by_id.loc["ms:3", "n_flank"] == "DDDD"
    assert by_id.loc["ms:3", "c_flank"] == "ZZZZ"
    assert by_id.loc["ms:3", "gene_id"] == "ENSG3B"

    # ms:4: conflicting flanks without expression winner -> flanks blanked, candidates preserved.
    assert by_id.loc["ms:4", "flank_status"] == "ambiguous_flanks"
    assert by_id.loc["ms:4", "n_flank"] == ""
    assert by_id.loc["ms:4", "c_flank"] == ""
    assert len(json.loads(by_id.loc["ms:4", "candidate_flank_pairs"])) == 2

    # ms:5: unmapped peptide retained for allele deconvolution with empty flanks.
    assert by_id.loc["ms:5", "flank_status"] == "unmapped"
    assert by_id.loc["ms:5", "n_protein_mappings"] == 0

    # Without expression disambiguation, ms:3 also becomes ambiguous_flanks.
    no_expr = prepare_ms_training_data(
        ms, scorable_class_i_only=True, use_expression_disambiguation=False
    )
    assert (
        no_expr.accepted.set_index("evidence_row_id").loc["ms:3", "flank_status"]
        == "ambiguous_flanks"
    )


def test_prepare_ms_bag_statuses_and_mhcflurry_presentation_inputs():
    seven_alleles = (
        "HLA-A*01:01;HLA-A*02:01;HLA-B*07:02;HLA-B*08:01;HLA-C*07:01;HLA-C*07:02;HLA-A*03:01"
    )
    four_alleles = "HLA-A*01:01;HLA-A*02:01;HLA-B*07:02;HLA-C*07:02"
    ms = pd.DataFrame(
        {
            "evidence_row_id": [f"ms:{i}" for i in range(1, 7)],
            "evidence_source_id": [f"s:{i}" for i in range(1, 7)],
            "provenance_id": [f"p:{i}" for i in range(1, 7)],
            "evidence_kind": ["ms"] * 6,
            "peptide": [
                "SLYNTVATL",  # 0: monoallelic single
                "GILGFVFTL",  # 1: predicted restriction (excluded by default)
                "NLVPMVATV",  # 2: sample_allele_match 4-allele MIL bag
                "GLCTLVAML",  # 3: pmid_class_pool (excluded by default)
                "FLPSDFFPSV",  # 4: 7-allele pooled bag (> max_alleles_per_bag=6)
                "YVLDHLIVV",  # 5: potential contaminant
            ],
            "is_potential_contaminant": [False, False, False, False, False, True],
            "mhc_restriction": [
                "HLA-A*02:01",
                "HLA-A*02:01",
                "HLA class I",
                "HLA class I",
                "HLA class I",
                "HLA-A*02:01",
            ],
            "restriction_evidence": [
                "monoallelic",
                "predicted",
                "class_only",
                "class_only",
                "class_only",
                "monoallelic",
            ],
            "is_monoallelic": [True, False, False, False, False, True],
            "mhc_allele_set": [
                "HLA-A*02:01",
                "HLA-A*02:01",
                four_alleles,
                four_alleles,
                seven_alleles,
                "HLA-A*02:01",
            ],
            "mhc_allele_provenance": [
                "peptide_attribution",
                "peptide_attribution",
                "sample_allele_match",
                "pmid_class_pool",
                "sample_allele_match",
                "peptide_attribution",
            ],
            "mhc_class": ["I"] * 6,
            "mhc_species": ["Homo sapiens"] * 6,
            "pmid": [10, 11, 12, 13, 14, 15],
            "condition_id": ["c1", "c2", "c3", "c4", "c5", "c6"],
        }
    )

    sel = prepare_ms_training_data(ms, scorable_class_i_only=True, max_alleles_per_bag=6)
    assert len(sel.accepted) == 2
    assert len(sel.excluded) == 4
    assert set(sel.accepted["candidate_bag_status"]) == {
        "monoallelic_single",
        "sample_candidate_bag",
    }
    assert set(sel.excluded["exclusion_reason"]) == {
        "predicted_restriction_excluded",
        "study_pool_candidates_excluded",
        "exceeds_max_alleles_per_bag",
        "potential_contaminant_excluded",
    }

    hits, sample_alleles = to_mhcflurry_presentation_inputs(sel)
    assert len(hits) == 2
    assert set(hits["sample_id"]) == set(sample_alleles)
    assert all(1 <= len(alleles) <= 6 for alleles in sample_alleles.values())

    # Relaxing study pools & predicted restrictions admits rows 1, 3, and (without bag cap) 4.
    relaxed = prepare_ms_training_data(
        ms,
        scorable_class_i_only=True,
        allow_predicted_restrictions=True,
        allow_study_pools=True,
    )
    assert {"predicted_single_restriction", "study_pool_candidates"} <= set(
        relaxed.accepted["candidate_bag_status"]
    )
    with pytest.raises(ValueError, match="max_alleles_per_sample=6"):
        to_mhcflurry_presentation_inputs(relaxed, max_alleles_per_sample=6)


def test_docs_have_no_stale_1x_training_commands_or_imports():
    repo_root = Path(hitlist.__file__).resolve().parent.parent
    docs_to_check = [
        repo_root / "docs" / "index.md",
        repo_root / "docs" / "training-provenance.md",
        repo_root / "docs" / "model-training.md",
    ]
    import_re = re.compile(r"^from (hitlist(?:\.[a-zA-Z0-9_]+)?) import (.+)$")
    for doc_path in docs_to_check:
        text = doc_path.read_text()
        assert "hitlist data build" not in text, f"Stale 1.x CLI in {doc_path.name}"
        assert "from hitlist import generate_training_table" not in text, (
            f"Removed top-level import in {doc_path.name}"
        )
        if doc_path.name != "training-provenance.md":
            assert "hitlist export training" not in text, f"Removed CLI in {doc_path.name}"
        lines = text.splitlines()
        idx = 0
        while idx < len(lines):
            stripped = lines[idx].strip()
            match = import_re.match(stripped)
            if match:
                mod_name, imported = match.group(1), match.group(2)
                while "(" in imported and ")" not in imported and idx + 1 < len(lines):
                    idx += 1
                    imported += " " + lines[idx].strip()
                imported_clean = imported.strip().lstrip("(").rstrip(")")
                names = [tok.strip() for tok in imported_clean.split(",") if tok.strip()]
                mod = __import__(mod_name, fromlist=names)
                for name in names:
                    assert hasattr(mod, name), (
                        f"{mod_name}.{name} missing (referenced in {doc_path.name})"
                    )
            idx += 1


@pytest.mark.integration
def test_real_corpus_binding_qualification_conserves_all_rows():
    if not is_binding_built():
        pytest.skip("binding.parquet not built")
    bdf = load_binding()
    qualified = qualify_binding_measurements(bdf)
    assert len(qualified) == len(bdf)
    assert set(qualified["binding_endpoint_family"]) <= set(BINDING_ENDPOINT_FAMILIES)

    aff = prepare_affinity_training_data(
        bdf, scorable_class_i_only=True, min_peptide_length=5, max_peptide_length=15
    )
    assert len(aff.accepted) + len(aff.excluded) == len(bdf)
    assert len(aff.accepted) > 100_000
    assert (aff.accepted["measurement_value"] > 0).all()
    assert np.isfinite(aff.accepted["measurement_value"]).all()
    assert set(aff.accepted["measurement_unit"]) == {"nM"}
    assert "purified_mhc" in set(aff.accepted["assay_preparation"])
    assert "cellular_mhc" in set(aff.accepted["assay_preparation"])

    stab = prepare_stability_training_data(bdf, species=None)
    assert len(stab.accepted) + len(stab.excluded) == len(bdf)
    assert set(stab.accepted["measurement_kind"]) == {"half_life", "thermal_stability"}
    hl_accepted = stab.accepted[stab.accepted["measurement_kind"] == "half_life"]
    assert (hl_accepted["measurement_value"] > 0).all()
    assert not stab.accepted["allele_completeness"].str.startswith("mutant_").any()
