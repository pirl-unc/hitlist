import pandas as pd

from hitlist.samples import overlay_targets, sample_peptidomes


def _make_full_scan():
    """Simulated full scan with 3 samples."""
    return pd.DataFrame(
        {
            "peptide": [
                "AAA",
                "BBB",
                "CCC",
                "DDD",  # sample 1: 4 peptides
                "EEE",
                "FFF",  # sample 2: 2 peptides
                "GGG",
                "AAA",  # sample 3: 2 peptides (AAA shared)
            ],
            "pmid": [123, 123, 123, 123, 123, 123, 456, 456],
            "antigen_processing_comments": [
                "colon 1",
                "colon 1",
                "colon 1",
                "colon 1",
                "buffy coat 5",
                "buffy coat 5",
                "liver 3",
                "liver 3",
            ],
            "mhc_restriction": [
                "HLA-A*02:01",
                "HLA-A*02:01",
                "HLA-B*07:02",
                "HLA-A*02:01",
                "HLA-A*02:01",
                "HLA-A*02:01",
                "HLA-B*07:02",
                "HLA-B*07:02",
            ],
            "source_tissue": [
                "Colon",
                "Colon",
                "Colon",
                "Colon",
                "Blood",
                "Blood",
                "Liver",
                "Liver",
            ],
            "src_cancer": [False, False, False, False, False, False, False, False],
            "src_healthy_tissue": [True, True, True, True, True, True, True, True],
        }
    )


def test_sample_peptidomes():
    df = _make_full_scan()
    result = sample_peptidomes(df)
    assert len(result) == 3
    # colon 1 has 4 total, 4 unique
    colon = result[result["antigen_processing_comments"] == "colon 1"].iloc[0]
    assert colon["unique_peptides"] == 4
    # buffy coat 5 has 2 unique
    buffy = result[result["antigen_processing_comments"] == "buffy coat 5"].iloc[0]
    assert buffy["unique_peptides"] == 2


def test_overlay_targets():
    df = _make_full_scan()
    target_peps = {"AAA", "CCC"}  # 2 "CTA" peptides

    result = overlay_targets(df, target_peps, label="cta")
    assert "cta_peptides" in result.columns
    assert "cta_fraction" in result.columns

    # colon 1 has 2 target peptides out of 4 = 50%
    colon = result[result["antigen_processing_comments"] == "colon 1"].iloc[0]
    assert colon["cta_peptides"] == 2
    assert abs(colon["cta_fraction"] - 0.5) < 0.01

    # buffy coat 5 has 0 target peptides
    buffy = result[result["antigen_processing_comments"] == "buffy coat 5"].iloc[0]
    assert buffy["cta_peptides"] == 0

    # liver 3 has 1 target peptide (AAA) out of 2 = 50%
    liver = result[result["antigen_processing_comments"] == "liver 3"].iloc[0]
    assert liver["cta_peptides"] == 1


def test_overlay_no_matches():
    df = _make_full_scan()
    result = overlay_targets(df, {"ZZZ"}, label="viral")
    assert all(result["viral_peptides"] == 0)


def test_empty_scan():
    result = sample_peptidomes(pd.DataFrame())
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 0
