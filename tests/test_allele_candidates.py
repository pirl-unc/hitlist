"""Candidate inference must retain the precision of the paper's statement."""

import pytest

from hitlist import curation, supplement


@pytest.fixture(autouse=True)
def clear_candidate_caches():
    curation.expand_allele_set.cache_clear()
    yield
    curation.expand_allele_set.cache_clear()


TYPING = (
    "HLA-A*02:01;HLA-DRB1*12:01;HLA-DRB3*02:02;HLA-DQB1*03:01;"
    "HLA-DQA1*05:01/DQB1*03:01;HLA-DPA1*01:03/DPB1*04:01"
)


@pytest.mark.parametrize(
    "restriction,expected",
    [
        ("HLA-DR", {"HLA-DRB1*12:01", "HLA-DRB3*02:02"}),
        ("HLA-DRB1", {"HLA-DRB1*12:01"}),
        ("HLA-DQ", {"HLA-DQB1*03:01", "HLA-DQA1*05:01/DQB1*03:01"}),
        ("HLA-DP", {"HLA-DPA1*01:03/DPB1*04:01"}),
        ("HLA-DQA1", {"HLA-DQA1*05:01/DQB1*03:01"}),
    ],
)
def test_locus_candidates_keep_chains_and_supplied_pairs(restriction, expected):
    alleles, provenance, n_alleles = curation.expand_allele_set(restriction, TYPING)
    assert set(alleles.split(";")) == expected
    assert provenance == "sample_locus_match"
    assert n_alleles == len(expected)


@pytest.mark.parametrize("tier", ["donor", "peptide"])
def test_incompatible_strongest_tier_does_not_fall_back(monkeypatch, tier):
    monkeypatch.setattr(curation, "_pmid_allele_pool", lambda _: frozenset({"HLA-DRB1*12:01"}))
    args = (
        ("HLA-DQB1*03:01", frozenset())
        if tier == "donor"
        else ("HLA-DRB1*12:01", frozenset({"Mamu-DRB1*03:06"}))
    )
    assert curation.expand_allele_set("HLA-DR", args[0], 999, "II", args[1]) == ("", "unmatched", 0)


def test_locus_intersection_requires_reported_class_and_species():
    assert curation.expand_allele_set("HLA-DR", TYPING, mhc_class="I") == ("", "unmatched", 0)
    assert curation.expand_allele_set("HLA-DR", "Mamu-DRB1*03:06") == ("", "unmatched", 0)
    # Engineered MHC retains its own species, independent of a human cell host.
    assert curation.expand_allele_set(
        "Mamu-B", "Mamu-B*008:01;HLA-B*27:05", species_context="Homo sapiens"
    ) == ("Mamu-B*008:01", "sample_locus_match", 1)


def test_class_only_matching_uses_ontology_and_explicit_species():
    assert curation.expand_allele_set(
        "H2 class I", "H2-Kb;H2-Db;H2-IAb;HLA-A*02:01", mhc_class="I"
    ) == ("H2-D*b;H2-K*b", "sample_allele_match", 2)
    assert curation.classify_allele_resolution("H2-Kb;H2-Db") == "donor_set"
    assert curation.expand_allele_set("HLA class I", "H2-Kb", mhc_class="I") == ("", "unmatched", 0)


def test_generic_class_uses_curated_context_but_does_not_assume_human():
    assert curation.expand_allele_set(
        "MHC class I", "H2-Kb;HLA-A*02:01", mhc_class="I", species_context="Mus musculus"
    ) == ("H2-K*b", "sample_allele_match", 1)
    assert curation.expand_allele_set("MHC class I", "H2-Kb", mhc_class="I")[0] == "H2-K*b"


def test_generic_class_uses_pmid_context(monkeypatch):
    monkeypatch.setattr(curation, "pmid_mhc_species_context", lambda _: "Mus musculus")
    assert curation.expand_allele_set("MHC class I", "H2-Kb;HLA-A*02:01", pmid=999)[0] == "H2-K*b"


def test_genus_restriction_refines_to_evidenced_species():
    assert curation.expand_allele_set(
        "BoLA-DR", "Bota-DRB3*15:01;Bubu-DRB3*15:01", species_context="Bos taurus"
    ) == ("Bota-DRB3*15:01", "sample_locus_match", 1)


def test_locus_rejects_supplied_pair_with_a_chain_outside_locus():
    assert curation.expand_allele_set("HLA-DR", "HLA-DRA*01:01/DQB1*03:01") == ("", "unmatched", 0)


def test_explicit_nonclassical_class_filters_typing():
    assert curation.expand_allele_set(
        "HLA class I", "HLA-A*02:01;HLA-E*01:01;HLA-DQB1*03:01", mhc_class="non-classical"
    ) == ("HLA-E*01:01", "sample_allele_match", 1)


def test_peptide_typing_precedes_donor_and_pool(monkeypatch):
    monkeypatch.setattr(curation, "_pmid_allele_pool", lambda _: frozenset({"HLA-DRB1*12:02"}))
    assert curation.expand_allele_set(
        "HLA-DR", "HLA-DRB1*12:01", 999, "II", frozenset({"HLA-DRB1*15:01"})
    ) == ("HLA-DRB1*15:01", "peptide_locus_match", 1)


@pytest.mark.parametrize("typing", ["H2-Kx", "H2-Kbm999", "HLA-DRA*01:01/DRB1"])
def test_inexact_nonhuman_or_partial_pair_cannot_be_a_donor_set(typing):
    assert curation.classify_allele_resolution(f"H2-Kb;{typing}") != "donor_set"


@pytest.mark.parametrize(
    "typing",
    ["HLA-DRA*01:01/DRB1", "HLA-DRB1*12", "HLA-DRB1*12:01 class II", "garbage"],
)
def test_imprecise_or_misleading_typing_is_not_an_exact_candidate(typing):
    assert curation.expand_allele_set("HLA-DR", typing) == ("", "unmatched", 0)


@pytest.mark.parametrize("typing", ["HLA-DRA*01:01/DRB1", "HLA-DRB1*12", "HLA-DR3"])
def test_incomplete_donor_typing_does_not_fall_back_to_pmid_pool(monkeypatch, typing):
    monkeypatch.setattr(curation, "_pmid_allele_pool", lambda _: frozenset({"HLA-DRB1*12:01"}))
    assert curation.expand_allele_set("HLA-DR", typing, 999, "II") == ("", "unmatched", 0)


def test_typing_does_not_synthesize_class_ii_partners():
    assert curation.expand_allele_set("HLA-DQ", "HLA-DQA1*05:01;HLA-DQB1*03:01") == (
        "HLA-DQA1*05:01;HLA-DQB1*03:01",
        "sample_locus_match",
        2,
    )


def test_blank_restriction_remains_unmatched_with_typing_and_pool(monkeypatch):
    monkeypatch.setattr(curation, "_pmid_allele_pool", lambda _: frozenset({"HLA-DRB1*12:01"}))
    assert curation.expand_allele_set("", TYPING, 999, "II") == ("", "unmatched", 0)


def test_nonhuman_curated_pool_and_free_text(monkeypatch):
    monkeypatch.setattr(
        curation,
        "load_pmid_overrides",
        lambda: {999: {"hla_alleles": ["H2-Kb H2-Db", "HLA-A*02", "51 HLA-I allotypes (...)"]}},
    )
    curation._pmid_allele_pool.cache_clear()
    try:
        assert curation.expand_allele_set("H2-K", pmid=999, mhc_class="I") == (
            "H2-K*b",
            "pmid_locus_pool",
            1,
        )
    finally:
        curation._pmid_allele_pool.cache_clear()


@pytest.mark.parametrize("classify_source", [False, True])
@pytest.mark.parametrize("attributed", [False, True])
def test_scanner_preserves_locus_evidence(tmp_path, monkeypatch, classify_source, attributed):
    from hitlist.observations import load_observations
    from hitlist.scanner import scan
    from tests.test_scanner import _write_tiny_iedb_csv

    path = tmp_path / "iedb.csv"
    row = [""] * 23
    row[0], row[2], row[5] = "http://iedb.org/assay/locus", "99900003", "PEPTIDE"
    row[19], row[20], row[21] = "HLA-DRB1", "II", TYPING
    row[22] = "mass spectrometry"
    _write_tiny_iedb_csv(path, [row])
    if attributed:
        monkeypatch.setattr(curation, "peptide_attribution_applies_to_row", lambda *_: True)
        monkeypatch.setattr(
            curation,
            "attribute_peptide_to_per_sample_typings",
            lambda *_: (("donor", frozenset({"HLA-DRB1*15:01"})),),
        )
    frame = scan(
        peptides=None,
        iedb_path=path,
        cedar_path=None,
        mhc_species=None,
        classify_source=classify_source,
    )
    result = frame.iloc[0]
    assert result["mhc_restriction"] == "HLA-DRB1"
    assert result["allele_resolution"] == "unresolved"
    assert result["restriction_evidence"] == "unknown"
    candidate = "HLA-DRB1*15:01" if attributed else "HLA-DRB1*12:01"
    assert result["mhc_allele_set"] == candidate
    assert result["mhc_allele_provenance"] == (
        "peptide_locus_match" if attributed else "sample_locus_match"
    )
    parquet_path = tmp_path / "observations.parquet"
    frame.to_parquet(parquet_path, index=False)
    monkeypatch.setattr("hitlist.observations.observations_path", lambda: parquet_path)
    assert list(load_observations(columns=["peptide"], mhc_allele_in_set=candidate)["peptide"]) == [
        "PEPTIDE"
    ]
    assert load_observations(columns=["peptide"], mhc_allele_provenance="exact").empty
    assert load_observations(columns=["peptide"], mhc_restriction=candidate).empty
    from hitlist.export import generate_observations_table

    assert generate_observations_table(min_allele_resolution="four_digit").empty


@pytest.mark.parametrize("classify_source", [False, True])
def test_supplement_preserves_locus_evidence(tmp_path, monkeypatch, classify_source):
    path = tmp_path / "loci.csv"
    path.write_text("peptide,mhc_restriction,mhc_class\nPEPTIDE,HLA-DRB1,II\nBLANK,,II\n")
    monkeypatch.setattr(supplement, "_SUPP_DIR", tmp_path)
    monkeypatch.setattr(
        supplement,
        "load_supplementary_manifest",
        lambda: [{"pmid": 999, "file": path.name, "defaults": {}}],
    )
    monkeypatch.setattr(curation, "_pmid_allele_pool", lambda _: frozenset({"HLA-DRB1*12:01"}))
    result = supplement.scan_supplementary(classify_source=classify_source).set_index("peptide")
    assert result.loc["PEPTIDE", "mhc_restriction"] == "HLA-DRB1"
    assert result.loc["PEPTIDE", "allele_resolution"] == "unresolved"
    assert result.loc["PEPTIDE", "restriction_evidence"] == "unknown"
    assert result.loc["PEPTIDE", "mhc_allele_set"] == "HLA-DRB1*12:01"
    assert result.loc["PEPTIDE", "mhc_allele_provenance"] == "pmid_locus_pool"
    assert result.loc["BLANK", "mhc_allele_set"] == ""
