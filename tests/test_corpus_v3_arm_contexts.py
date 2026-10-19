"""Replay all actual deposited context patterns from the v3 curation re-audit."""

import json
from pathlib import Path

import pandas as pd
import pytest

from hitlist.curation import (
    load_pmid_overrides,
    reported_mhc_fields_overlap,
    restriction_evidence_for_row,
)
from hitlist.export import generate_observations_table


def test_all_v3_deposited_contexts_preserve_named_sources_and_treatment_uncertainty(monkeypatch):
    fixture = json.loads((Path(__file__).parent / "data/corpus_v3_arm_contexts.json").read_text())
    contexts = fixture["rows"]
    frame = pd.DataFrame(contexts).drop(columns=["named_lines", "n_observations"])
    # Only metadata is a fixture; this placeholder is not a measured peptide.
    frame["peptide"] = "AAAAAAAAA"
    frame["source"] = "iedb"
    frame["mhc_species"] = "Homo sapiens"
    frame["is_binding_assay"] = False
    monkeypatch.setattr("hitlist.observations.load_observations", lambda **kw: frame.copy())
    result = generate_observations_table(exclude_non_peptide_ligand=False)
    assert len(result) == len(contexts)
    resolved = ambiguous = becker = 0
    for context, (_, row) in zip(contexts, result.iterrows()):
        assert row.assay_comments == context["assay_comments"]
        assert row.mhc_restriction == context["mhc_restriction"]
        if context["pmid"] == 33981976:
            assert row.condition_id == ""
            assert row.sample_group == "HCT 116"
            assert row.sample_attribution == "group_ambiguous"
            assert row.arm_resolution == "arm_not_recorded"
            becker += context["n_observations"]
        elif len(context["named_lines"]) == 1:
            name = context["named_lines"][0]
            assert row.condition_id
            assert row.sample_attribution == "elution_conditions"
            if name.startswith("K562-"):
                assert row.sample_label.startswith(
                    "K562 transfectant " + name.removeprefix("K562-")
                )
            else:
                assert row.sample_label.startswith("EBV-LCL " + name + " (")
            resolved += context["n_observations"]
        else:
            assert row.condition_id == ""
            assert row.sample_attribution == "pmid_ambiguous"
            assert row.arm_resolution == "multi_arm_evidence"
            ambiguous += context["n_observations"]
    assert (resolved, ambiguous, becker) == (62512, 16894, 11049)


def test_reaudited_curated_evidence_distinguishes_prediction_and_vehicle_control():
    assert restriction_evidence_for_row("HLA-DPA1*02:01/DPB1*09:01", pmid=32350084) == "predicted"
    control, treatment = load_pmid_overrides()[33981976]["ms_samples"]
    assert control["condition_control"] == "vehicle"
    assert control["condition_control_for"] == treatment["condition_id"]
    assert "DMSO" in control["sample_label"]


@pytest.mark.parametrize(
    "second, expected",
    [
        ("HLA-DPB1*09:01/DPA1*02:01", True),
        ("HLA-DPB1*09:01/DPA1*01:03", False),
        ("HLA-DPB1*10:01/DPA1*02:01", False),
    ],
)
def test_pair_identity_checks_both_chains_independent_of_input_order(second, expected):
    first = "HLA-DPA1*02:01/DPB1*09:01"
    assert reported_mhc_fields_overlap(first, second) is expected
    assert reported_mhc_fields_overlap(second, first) is expected
