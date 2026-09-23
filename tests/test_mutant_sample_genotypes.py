"""Mutation labels are parts of a molecule, never extra genotype alleles."""

import pytest

from hitlist.curation import extract_allele_tokens, mhc_species_of, sample_mhc_candidates


@pytest.mark.parametrize(
    "field, expected",
    [
        ("HLA-B*08:01 E76C mutant", ["HLA-B*08:01 E76C mutant"]),
        (
            "HLA-DRA*01:01 F54C mutant/DRB1*01:01",
            ["HLA-DRA*01:01 F54C mutant/DRB1*01:01"],
        ),
        (
            "H-2Kb E76C, R55T mutant; HLA-A*02:01",
            ["H2-K*b E76C R55T mutant", "HLA-A*02:01"],
        ),
        ("HLA-B*44:01 E76C mutant", ["HLA-B*44:01 E76C mutant"]),
    ],
)
def test_allele_extraction_preserves_reported_mutant_molecules(field, expected):
    assert extract_allele_tokens(field) == expected


@pytest.mark.parametrize(
    "field, expected",
    [
        ("H-2Kb E76C mutant", "Mus musculus"),
        ("H-2Kb E76C, R55T mutant; H-2Db", "Mus musculus"),
        ("H-2Kb E76C mutant HLA-A*02:01", "Homo sapiens;Mus musculus"),
        ("HLA-DRA*01:01 F54C mutant/DRB1*01:01", "Homo sapiens"),
        ("BoLA-DR; SLA class I", "Bos sp.;Sus sp."),
        ("RT1-a", "Rattus sp."),
        ("H-2b", "Mus musculus"),
    ],
)
def test_species_uses_complete_molecules(field, expected):
    assert mhc_species_of(field) == expected


@pytest.mark.parametrize("parse_field", [extract_allele_tokens, mhc_species_of])
@pytest.mark.parametrize("field", ["HLA-B*08:01 E76C", "H-2Kb nonsense mutant", "E76C"])
def test_all_field_consumers_reject_unassigned_mutations(parse_field, field):
    with pytest.raises(ValueError, match="mutation"):
        parse_field(field)


@pytest.mark.parametrize(
    "field, expected",
    [
        ("HLA-B*08:01 E76C mutant", {"HLA-B*08:01 E76C mutant"}),
        ("HLA-DRA*01:01 F54C mutant/DRB1*01:01", {"HLA-DRA*01:01 F54C mutant/DRB1*01:01"}),
        (
            "HLA-DRA*01:01/DRB1*01:01 alpha C30S beta G86Y mutant",
            {"HLA-DRA*01:01 C30S mutant/DRB1*01:01 G86Y mutant"},
        ),
        ("HLA-B*08:01 E76C mutant HLA-A*02:01", {"HLA-B*08:01 E76C mutant", "HLA-A*02:01"}),
        (
            "HLA-A*02:01; HLA-B*08:01 E76C, R55T mutant",
            {"HLA-A*02:01", "HLA-B*08:01 E76C R55T mutant"},
        ),
        (["HLA-A*02:01", "HLA-B*08:01 E76C mutant"], {"HLA-A*02:01", "HLA-B*08:01 E76C mutant"}),
        ("A*02:01 A*24:02 B*15:01", {"HLA-A*02:01", "HLA-A*24:02", "HLA-B*15:01"}),
    ],
)
def test_mutant_molecules_survive_sample_segmentation(field, expected):
    result = sample_mhc_candidates(field)
    assert result.exact == expected
    assert result.serotypes == ()
    assert result.imprecise == ()


def test_mixed_mutant_and_serotype_keep_different_typing_precision():
    result = sample_mhc_candidates("HLA-B*08:01 E76C mutant; HLA-A2")
    assert result.exact == {"HLA-B*08:01 E76C mutant"}
    assert result.serotypes == ("HLA-A2",)
    assert "HLA-A*02:01" in result.serotype_alleles


@pytest.mark.parametrize(
    "field",
    [
        "E76C",
        "HLA-B*08:01 E76C",
        "HLA-B*08:01 nonsense mutant",
        "HLA-DRB1*01:01 alpha C30S mutant",
        "HLA-A*02:01; E76C mutant",
    ],
)
def test_unconsumed_mutation_text_cannot_fall_back_to_wild_type(field):
    with pytest.raises(ValueError, match="mutation"):
        sample_mhc_candidates(field)


@pytest.mark.parametrize(
    "mhc_class, wild_type, mutant",
    [
        ("I", "HLA-B*08:01", "HLA-B*08:01 E76C mutant"),
        (
            "II",
            "HLA-DRA*01:01/DRB1*01:01",
            "HLA-DRA*01:01 F54C mutant/DRB1*01:01",
        ),
    ],
)
def test_observation_join_keeps_mutant_and_wild_type_arms_distinct(
    tmp_path, monkeypatch, mhc_class, wild_type, mutant
):
    import pandas as pd

    from hitlist.export import generate_ms_observations_table

    pmid = 99999528
    arms = {"wild type": wild_type, "engineered mutant": mutant}
    monkeypatch.setattr(
        "hitlist.export.load_pmid_overrides",
        lambda: {
            pmid: {
                "species": "Homo sapiens (human)",
                "ms_samples": [
                    {
                        "sample_label": label,
                        "n_samples": 1,
                        "mhc_class": mhc_class,
                        "mhc": allele,
                    }
                    for label, allele in arms.items()
                ],
            }
        },
    )
    path = tmp_path / "observations.parquet"
    pd.DataFrame(
        [
            {
                "peptide": "SYNTHETICPEPTIDE",
                "mhc_restriction": allele,
                "mhc_class": mhc_class,
                "reference_iri": f"iri:{label}",
                "pmid": pmid,
                "source": "iedb",
                "mhc_species": "Homo sapiens",
                "is_monoallelic": False,
                "is_binding_assay": False,
                "qualitative_measurement": "Positive",
            }
            for label, allele in arms.items()
        ]
    ).to_parquet(path, index=False)
    monkeypatch.setattr("hitlist.observations.observations_path", lambda: path)

    result = generate_ms_observations_table().set_index("mhc_restriction")

    for label, allele in arms.items():
        assert result.loc[allele, "sample_label"] == label
        assert result.loc[allele, "sample_mhc"] == allele
        assert result.loc[allele, "sample_match_type"] == "allele_match"
        assert result.loc[allele, "sample_attribution"] == "allele_exact"
