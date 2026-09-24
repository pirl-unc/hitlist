"""Reported molecule sets constrain heuristic sample attribution (#514)."""

import pandas as pd
import pytest

from hitlist.curation import reported_mhc_fields_overlap
from hitlist.export import generate_observations_table


def _row(restriction, cell_name="293-T-Epithelial cell", **extra):
    return {
        "peptide": "AAAAAAAAA",
        "pmid": 28834231,
        "cell_name": cell_name,
        "mhc_restriction": restriction,
        "mhc_class": "I",
        "mhc_species": "Homo sapiens",
        "source": "iedb",
        "source_tissue": "",
        "antigen_processing_comments": "",
        "assay_comments": "",
        "is_monoallelic": False,
        "is_binding_assay": False,
        "qualitative_measurement": "Positive",
        **extra,
    }


def _export(monkeypatch, rows):
    monkeypatch.setattr(
        "hitlist.observations.load_observations", lambda **kwargs: pd.DataFrame(rows)
    )
    return generate_observations_table(exclude_non_peptide_ligand=False)


MAVER = "HLA-A*24:02;HLA-A*26:01;HLA-B*38:01;HLA-B*44:02;HLA-C*05:01;HLA-C*12:03"
HEK = "HLA-A*03:01;HLA-B*07:02;HLA-C*07:01"


@pytest.mark.parametrize("same_text", [False, True])
def test_ritz_reported_genotypes_constrain_missing_tissue_attribution(monkeypatch, same_text):
    # PMID 28834231, Results: SSO/SSP typing of MAVER-1 and homozygous HEK293.
    # https://pmc.ncbi.nlm.nih.gov/articles/PMC5846733/
    result = _export(
        monkeypatch,
        [
            _row(MAVER, "293-T-Epithelial cell" if same_text else "MAVER-1-Lymphoblast"),
            _row(HEK),
        ],
    )
    assert result.sample_label.tolist() == [
        "MAVER-1 (mantle cell lymphoma)",
        "HEK293 (embryonic kidney)",
    ]
    assert result.mhc_restriction.tolist() == [MAVER, HEK]


def test_disjoint_genotype_cannot_be_assigned_by_generic_cell_text(monkeypatch):
    result = _export(
        monkeypatch,
        [_row(MAVER, "MAVER-1-Lymphoblast"), _row("HLA-A*01:01;HLA-B*08:01;HLA-C*07:02")],
    )
    assert result.iloc[1].sample_label == ""


def test_predicted_restriction_does_not_override_reported_sample_metadata(monkeypatch):
    result = _export(
        monkeypatch,
        [
            _row(MAVER, "MAVER-1-Lymphoblast", restriction_evidence="experimental"),
            _row(HEK, "MAVER-1-Lymphoblast", restriction_evidence="predicted"),
            _row(HEK, restriction_evidence="experimental"),
        ],
    )
    assert result.iloc[1].sample_label == "MAVER-1 (mantle cell lymphoma)"


@pytest.mark.parametrize(
    "first, second, overlap",
    [
        ("HLA-A*02:01", "HLA-A*03:01", False),
        ("HLA-A*02:01;HLA-B*08:01", "HLA-B*08:01", True),
        ("HLA-A*02:01:01", "HLA-A*02:01", True),
        ("HLA-A*02", "HLA-A*03:01", None),
        ("HLA-A2", "HLA-A*03:01", None),
        ("HLA-A*03:01 HLA-B7", "HLA-A*02:01", None),
        ("HLA class I", "HLA-A*03:01", None),
        ("HLA-A", "HLA-A*03:01", None),
        ("", "HLA-A*03:01", None),
        ("HLA-B*44:01", "HLA-B*44:02", True),
        ("HLA-B*08:01 E76C mutant", "HLA-B*08:01", False),
        ("HLA-DQA1*03:01/DQB1*03:02", "HLA-DQB1*03:02", True),
        ("HLA-DQA1*03:01/DQB1*03:02", "HLA-DQA1*03:01/DQB1*03:01", False),
        ("Calu-88*501:01", "DLA-88*501:01", True),
        ("HLA-DRB3*01:01", "HLA-DRB1*15:01", None),
        ("HLA-A*02:01;HLA-B*08:01", "HLA-A*03:01", None),
        (
            "HLA-DRA*01:01 F54C mutant/DRB1*01:01",
            "HLA-DRA*01:01/DRB1*01:01",
            False,
        ),
    ],
)
def test_overlap_respects_reported_precision_and_identity(first, second, overlap):
    assert reported_mhc_fields_overlap(first, second) is overlap
    assert reported_mhc_fields_overlap(second, first) is overlap


def test_unknown_typing_cannot_win_only_because_precise_typing_was_excluded(monkeypatch):
    monkeypatch.setattr(
        "hitlist.export.load_pmid_overrides",
        lambda: {
            28834231: {
                "ms_samples": [
                    {"sample_label": "first", "mhc": "HLA-A*02:01", "mhc_class": "I"},
                    {"sample_label": "second", "mhc": "HLA-A*03:01", "mhc_class": "I"},
                    {"sample_label": "untyped", "mhc": "HLA class I", "mhc_class": "I"},
                ]
            }
        },
    )
    result = _export(monkeypatch, [_row("HLA-A*01:01;HLA-B*08:01", "")])
    assert result.iloc[0].sample_label == ""


def test_rejected_text_guess_preserves_curated_consensus(monkeypatch):
    result = _export(
        monkeypatch,
        [
            _row(MAVER, "MAVER-1-Lymphoblast"),
            _row("HLA-A*01:01;HLA-B*08:01;HLA-C*07:02"),
        ],
    )
    rejected = result.iloc[1]
    assert rejected.sample_label == ""
    assert rejected.condition_id == ""
    assert rejected.sample_attribution == "pmid_ambiguous"
    assert rejected.matched_sample_count == 2
