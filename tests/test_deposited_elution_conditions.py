"""Source-specific treatment statements must resolve exactly one arm (#512)."""

import pandas as pd
import pytest

from hitlist.curation import load_pmid_overrides
from hitlist.export import _select_by_elution_conditions, generate_observations_table

SINGLE_CONDITIONS = [
    ("1uM CDK4/6i", "palbociclib_1um"),
    ("10uM CDK4/6i", "palbociclib_10um"),
    ("10ng/mL IFNg", "ifng"),
]
MULTIPLE_CONDITIONS = [
    "1uM CDK4/6i and 10uM CDK4/6i",
    "1uM CDK4/6i and 10ng/mL IFNg",
    "10uM CDK4/6i and 10ng/mL IFNg",
    "1uM CDK4/6i, 10uM CDK4/6i, and 10ng/mL IFNg",
]


def _statement(treatments):
    return f"The epitope was eluted from cells treated with {treatments}."


@pytest.mark.parametrize("line", ["SK-MEL-2", "SK-MEL-5", "SK-MEL-28", "IPC-298"])
@pytest.mark.parametrize("treatment, suffix", SINGLE_CONDITIONS)
def test_stopfer_exact_deposited_condition_selects_one_arm(monkeypatch, line, treatment, suffix):
    # PMID 32488085, Figures 4/6 and Supplementary Data 3/5: separate arms.
    monkeypatch.setattr(
        "hitlist.observations.load_observations",
        lambda **kwargs: pd.DataFrame(
            [
                {
                    "peptide": "AAAAAAAAA",
                    "pmid": 32488085,
                    "mhc_restriction": "HLA class I",
                    "mhc_class": "I",
                    "mhc_species": "Homo sapiens",
                    "cell_name": observed_line + "-Melanocyte",
                    "assay_comments": _statement(treatment),
                    "is_binding_assay": False,
                    "source": "iedb",
                }
                for observed_line in ["SK-MEL-2", "SK-MEL-5", "SK-MEL-28", "IPC-298"]
            ]
        ),
    )
    result = generate_observations_table(exclude_non_peptide_ligand=False)
    row = result[result.cell_name == line + "-Melanocyte"].iloc[0]
    assert row.condition_id == line.lower().replace("-", "_") + "_" + suffix
    assert row.sample_attribution == "elution_conditions"


@pytest.mark.parametrize("treatment", [*MULTIPLE_CONDITIONS, "100uM CDK4/6i"])
def test_mixed_or_unregistered_dose_does_not_select_even_one_remaining_candidate(treatment):
    mapping = load_pmid_overrides()[32488085].get("elution_condition_ids", {})
    candidate = ("high", "palbociclib", {"condition_id": "sk_mel_5_palbociclib_10um"})
    assert (
        _select_by_elution_conditions(
            [candidate], _statement(treatment), curated_condition_ids=mapping
        )
        is None
    )


def test_curated_statement_does_not_apply_to_unregistered_studies():
    candidate = ("high", "palbociclib", {"condition_id": "sk_mel_5_palbociclib_10um"})
    assert _select_by_elution_conditions([candidate], _statement("10uM CDK4/6i")) is None


def test_curated_conditions_apply_after_an_ambiguous_exact_allele_join(monkeypatch):
    monkeypatch.setattr(
        "hitlist.observations.load_observations",
        lambda **kwargs: pd.DataFrame(
            [
                {
                    "peptide": "AAAAAAAAA",
                    "pmid": 32488085,
                    "mhc_restriction": "HLA-A*11:01",
                    "mhc_class": "I",
                    "mhc_species": "Homo sapiens",
                    "cell_name": line + "-Melanocyte",
                    "assay_comments": _statement("10uM CDK4/6i"),
                    "is_binding_assay": False,
                    "source": "iedb",
                }
                for line in ["SK-MEL-5", "SK-MEL-28"]
            ]
        ),
    )
    result = generate_observations_table(exclude_non_peptide_ligand=False)
    assert result.condition_id.tolist() == [
        "sk_mel_5_palbociclib_10um",
        "sk_mel_28_palbociclib_10um",
    ]
    assert set(result.sample_attribution) == {"elution_conditions"}
    assert set(result.sample_match_type) == {"allele_match"}


@pytest.mark.parametrize(
    "mapping",
    [
        [],
        {"": ["treated"]},
        {" statement ": ["treated"]},
        {"statement": "treated"},
        {"statement": []},
        {"statement": [None]},
        {"statement": ["treated", "treated"]},
        {"statement": ["misspelled"]},
        {"statement": ["not_profiled"]},
    ],
)
def test_elution_mapping_rejects_unsupported_or_ambiguous_ids(tmp_path, monkeypatch, mapping):
    import yaml

    from hitlist import curation

    path = tmp_path / "pmid_overrides.yaml"
    path.write_text(
        yaml.safe_dump(
            [
                {
                    "pmid": 99999512,
                    "elution_condition_ids": mapping,
                    "ms_samples": [
                        {
                            "sample_label": "treated",
                            "condition_id": "treated",
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
        lambda filename: str(path) if filename == "pmid_overrides.yaml" else real_path(filename),
    )
    curation.load_pmid_overrides.cache_clear()
    try:
        with pytest.raises(ValueError, match="elution"):
            curation.load_pmid_overrides()
    finally:
        curation.load_pmid_overrides.cache_clear()
