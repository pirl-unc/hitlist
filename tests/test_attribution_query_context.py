"""Output filters must not change a deposited row's sample identity (#532)."""

import pandas as pd
import pytest

from hitlist.export import generate_observations_table

ATTRIBUTION_FIELDS = [
    "sample_label",
    "sample_group",
    "sample_attribution",
    "sample_match_type",
    "condition_id",
    "sample_mhc",
    "arm_resolution",
]


def _write_observations(tmp_path, monkeypatch, rows):
    from hitlist import observations

    path = tmp_path / "observations.parquet"
    defaults = {
        "pmid": 32488085,
        "mhc_restriction": "HLA class I",
        "mhc_class": "I",
        "mhc_species": "Homo sapiens",
        "source": "iedb",
        "is_binding_assay": False,
        "source_tissue": "",
        "antigen_processing_comments": "",
        "assay_comments": "",
    }
    pd.DataFrame([{**defaults, **row} for row in rows]).to_parquet(path, index=False)
    monkeypatch.setattr(observations, "observations_path", lambda: path)
    return path


@pytest.mark.parametrize("restriction", ["HLA class I", "HLA-A*11:01"])
@pytest.mark.parametrize(
    "statement",
    [
        "The epitope was eluted from cells treated with 10uM CDK4/6i.",
        "The epitope was eluted from cells treated with 1uM CDK4/6i and 10uM CDK4/6i.",
        "",
    ],
)
def test_stopfer_identity_and_arm_survive_single_peptide_query(
    tmp_path, monkeypatch, restriction, statement
):
    _write_observations(
        tmp_path,
        monkeypatch,
        [
            {
                "peptide": peptide,
                "cell_name": line + "-Melanocyte",
                "mhc_restriction": restriction,
                "assay_comments": statement,
            }
            for peptide, line in [("AAAAAAAAA", "SK-MEL-5"), ("LLLLLLLLL", "SK-MEL-28")]
        ],
    )
    complete = generate_observations_table().set_index("peptide")
    selected = generate_observations_table(peptide="AAAAAAAAA").set_index("peptide")
    assert list(selected.index) == ["AAAAAAAAA"]
    assert complete.loc["AAAAAAAAA", "sample_group"] == "SK-MEL-5"
    pd.testing.assert_frame_equal(
        selected[ATTRIBUTION_FIELDS].astype(str),
        complete.loc[["AAAAAAAAA"], ATTRIBUTION_FIELDS].astype(str),
    )
    assert selected.loc["AAAAAAAAA", "mhc_restriction"] == restriction
    assert selected.loc["AAAAAAAAA", "assay_comments"] == statement
    if "and" in statement or not statement:
        assert selected.loc["AAAAAAAAA", "condition_id"] == ""
    else:
        assert selected.loc["AAAAAAAAA", "condition_id"] == "sk_mel_5_palbociclib_10um"


def _ungrouped_overrides():
    return {
        99999532: {
            "ms_samples": [
                {
                    "sample_label": name,
                    "mhc": "HLA-A*02:01",
                    "mhc_class": "I",
                    "condition": "unperturbed",
                }
                for name in ["Alpha cells", "Beta cells"]
            ]
        }
    }


@pytest.mark.parametrize(
    "query",
    [
        {"peptide": "LLLLLLLLLL"},
        {"source": "cedar"},
        {"length_min": 10},
        {"mhc_allele": "HLA-B*07:02"},
    ],
)
def test_ungrouped_factual_discriminator_survives_filter(tmp_path, monkeypatch, query):
    monkeypatch.setattr("hitlist.export.load_pmid_overrides", _ungrouped_overrides)
    _write_observations(
        tmp_path,
        monkeypatch,
        [
            {"pmid": 99999532, "peptide": "AAAAAAAAA", "cell_name": "Alpha cells"},
            {
                "pmid": 99999532,
                "peptide": "LLLLLLLLLL",
                "cell_name": "Beta cells",
                "source": "cedar",
                "mhc_restriction": "HLA-B*07:02",
            },
        ],
    )
    complete = generate_observations_table().set_index("peptide")
    selected = generate_observations_table(**query).set_index("peptide")
    assert list(selected.index) == ["LLLLLLLLLL"]
    assert complete.loc["LLLLLLLLLL", "sample_label"] == "Beta cells"
    pd.testing.assert_frame_equal(
        selected[ATTRIBUTION_FIELDS].astype(str),
        complete.loc[["LLLLLLLLLL"], ATTRIBUTION_FIELDS].astype(str),
    )


def test_constant_narrative_stays_blocked_after_filter(tmp_path, monkeypatch):
    monkeypatch.setattr("hitlist.export.load_pmid_overrides", _ungrouped_overrides)
    _write_observations(
        tmp_path,
        monkeypatch,
        [
            {
                "pmid": 99999532,
                "peptide": peptide,
                "cell_name": "",
                "assay_comments": "Alpha cells were used in this study.",
            }
            for peptide in ["AAAAAAAAA", "LLLLLLLLL"]
        ],
    )
    for query in ({}, {"peptide": "AAAAAAAAA"}):
        result = generate_observations_table(**query)
        assert set(result.sample_label.astype(str)) == {""}
        assert set(result.sample_attribution.astype(str)) == {"pmid_ambiguous"}


@pytest.mark.parametrize("mhc_class, restriction", [("I", "HLA class I"), ("II", "HLA class II")])
def test_class_filter_keeps_the_complete_study_sample_roster(
    tmp_path, monkeypatch, mhc_class, restriction
):
    overrides = _ungrouped_overrides()
    beta = overrides[99999532]["ms_samples"][1]
    beta.update(mhc_class="II", mhc="HLA-DRB1*01:01")
    monkeypatch.setattr("hitlist.export.load_pmid_overrides", lambda: overrides)
    _write_observations(
        tmp_path,
        monkeypatch,
        [
            {
                "pmid": 99999532,
                "peptide": "AAAAAAAAA",
                "cell_name": "",
                "mhc_class": mhc_class,
                "mhc_restriction": restriction,
            }
        ],
    )
    complete = generate_observations_table()
    selected = generate_observations_table(mhc_class=mhc_class)
    fields = [*ATTRIBUTION_FIELDS, "matched_sample_count"]
    pd.testing.assert_frame_equal(selected[fields].astype(str), complete[fields].astype(str))
    assert selected.matched_sample_count.tolist() == [2]


def test_exactly_matched_rows_remain_part_of_study_discriminator_context(tmp_path, monkeypatch):
    monkeypatch.setattr("hitlist.export.load_pmid_overrides", _ungrouped_overrides)
    _write_observations(
        tmp_path,
        monkeypatch,
        [
            {"pmid": 99999532, "peptide": "AAAAAAAAA", "cell_name": "Alpha cells"},
            {
                "pmid": 99999532,
                "peptide": "LLLLLLLLL",
                "cell_name": "Beta cells",
                "mhc_restriction": "HLA-A*02:01",
            },
        ],
    )
    complete = generate_observations_table().set_index("peptide")
    selected = generate_observations_table(peptide="AAAAAAAAA").set_index("peptide")
    assert complete.loc["LLLLLLLLL", "sample_label"] == "Beta cells"
    assert complete.loc["AAAAAAAAA", "sample_label"] == "Alpha cells"
    pd.testing.assert_frame_equal(
        selected[ATTRIBUTION_FIELDS].astype(str),
        complete.loc[["AAAAAAAAA"], ATTRIBUTION_FIELDS].astype(str),
    )


@pytest.mark.parametrize("treatment", ["IFN-DAC", "IFN-DAC, IFN-TAZ"])
def test_bourne_class_pool_uses_lines_already_resolved_by_exact_allele(
    tmp_path, monkeypatch, treatment
):
    from tests.test_curation_sanity_pass import BOURNE_PREFIX

    _write_observations(
        tmp_path,
        monkeypatch,
        [
            {
                "pmid": 35561310,
                "peptide": peptide,
                "cell_name": line + "-B cell",
                "mhc_class": "II",
                "mhc_restriction": restriction,
                "assay_comments": BOURNE_PREFIX + treatment + ".",
            }
            for peptide, line, restriction in [
                ("LLLLLLLLL", "DB", "HLA-DRB1*03:01"),
                ("VVVVVVVVV", "SU-DHL-4", "HLA-DRB1*15:01"),
                ("AAAAAAAAA", "SU-DHL-6", "HLA-DRB1*01:01;HLA-DRB1*04:01"),
            ]
        ],
    )
    complete = generate_observations_table().set_index("peptide")
    target = complete.loc[["AAAAAAAAA"]]
    assert target.iloc[0].sample_group == "SU-DHL-6"
    assert target.iloc[0].condition_id == ("su_dhl_6_ifng_dac" if treatment == "IFN-DAC" else "")
    assert target.iloc[0].sample_attribution == (
        "elution_conditions" if treatment == "IFN-DAC" else "group_ambiguous"
    )
    for projection in (None, ["peptide", *ATTRIBUTION_FIELDS]):
        selected = generate_observations_table(peptide="AAAAAAAAA", columns=projection).set_index(
            "peptide"
        )
        pd.testing.assert_frame_equal(
            selected[ATTRIBUTION_FIELDS].astype(str), target[ATTRIBUTION_FIELDS].astype(str)
        )


def test_shared_biofluids_cannot_be_first_picked_by_label_length(tmp_path, monkeypatch):
    # PMID 27862975: both materials are explicitly reported for shared rows.
    _write_observations(
        tmp_path,
        monkeypatch,
        [
            {
                "pmid": 27862975,
                "peptide": peptide,
                "cell_name": "",
                "mhc_restriction": "HLA class I",
                "assay_comments": f"The epitope was eluted from {material}.",
            }
            for peptide, material in [
                ("AAAAAAAAA", "serum and plasma"),
                ("LLLLLLLLL", "serum"),
                ("VVVVVVVVV", "plasma"),
            ]
        ],
    )
    complete = generate_observations_table().set_index("peptide")
    assert complete.loc["AAAAAAAAA", "condition_id"] == ""
    assert complete.loc["AAAAAAAAA", "sample_label"] == ""
    assert complete.loc["AAAAAAAAA", "sample_attribution"] == "pmid_ambiguous"
    assert complete.loc["AAAAAAAAA", "arm_resolution"] == "multi_arm_evidence"
    assert complete.loc["LLLLLLLLL", "sample_label"] == "healthy donor serum"
    assert complete.loc["VVVVVVVVV", "sample_label"] == "healthy donor plasma"
    selected = generate_observations_table(
        peptide="AAAAAAAAA", columns=["peptide", *ATTRIBUTION_FIELDS]
    ).set_index("peptide")
    pd.testing.assert_frame_equal(
        selected[ATTRIBUTION_FIELDS].astype(str),
        complete.loc[["AAAAAAAAA"], ATTRIBUTION_FIELDS].astype(str),
    )


def test_context_is_compact_isolated_and_invalidated_by_rebuild(tmp_path, monkeypatch):
    import os

    from hitlist.observations import _load_attribution_context

    rows = [
        {"peptide": peptide, "cell_name": "SK-MEL-5-Melanocyte"}
        for peptide in ["AAAAAAAAA", "LLLLLLLLL"]
    ]
    path = _write_observations(tmp_path, monkeypatch, rows)
    first = _load_attribution_context([32488085])
    assert len(first) == 1
    assert "peptide" not in first
    first.loc[:, "cell_name"] = "mutated by caller"
    assert _load_attribution_context([32488085]).cell_name.tolist() == ["SK-MEL-5-Melanocyte"]
    previous = path.stat()
    _write_observations(tmp_path, monkeypatch, [{**rows[0], "cell_name": "SK-MEL-2-Melanocyte"}])
    os.utime(path, ns=(previous.st_atime_ns, previous.st_mtime_ns + 1))
    assert _load_attribution_context([32488085]).cell_name.tolist() == ["SK-MEL-2-Melanocyte"]
    assert _load_attribution_context([99999532]).empty


@pytest.mark.parametrize(
    "query",
    [{"peptide": "AAAAAAAAA"}, {"mhc_class": "non-classical"}, {"mhc_allele": "HLA-G*01:01"}],
)
def test_query_context_preserves_legacy_cohort_repair(tmp_path, monkeypatch, query):
    mono = {
        "pmid": 31844290,
        "peptide": "AAAAAAAAA",
        "assay_iri": "http://www.iedb.org/assay/534",
        "mhc_restriction": "HLA-G*01:01",
        "mhc_class": "non-classical",
        "mhc_allele_provenance": "exact",
        "cell_name": "B cell",
        "is_monoallelic": True,
        "attributed_sample_label": "MEL2 (13240-005)",
    }
    _write_observations(
        tmp_path,
        monkeypatch,
        [
            mono,
            {**mono, "attributed_sample_label": "MEL3 (13240-006)"},
            {
                **mono,
                "peptide": "LLLLLLLLL",
                "assay_iri": "http://www.iedb.org/assay/535",
                "mhc_restriction": "HLA-A*01:01;HLA-A*02:01",
                "mhc_class": "I",
                "mhc_allele_provenance": "peptide_attribution",
                "cell_name": "melanoma",
                "is_monoallelic": False,
            },
        ],
    )
    complete = generate_observations_table().set_index("peptide")
    selected = generate_observations_table(**query).set_index("peptide")
    assert complete.loc["AAAAAAAAA", "sample_label"] == "721.221-HLA-G*01:01"
    assert list(selected.index) == ["AAAAAAAAA"]
    fields = [*ATTRIBUTION_FIELDS, "matched_sample_count"]
    pd.testing.assert_frame_equal(
        selected[fields].astype(str), complete.loc[["AAAAAAAAA"], fields].astype(str)
    )
