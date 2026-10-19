import json

import pandas as pd
import pytest

from hitlist.tissue_blacklist import build_tissue_blacklist, load_atlas_tissue_evidence


def observation(donor, tissue="Heart", peptide="SLYNTVATL", **changes):
    return {
        "peptide": peptide,
        "donor_id": donor,
        "donor_status": "resolved",
        "source_tissue": tissue,
        "tissue_status": "nonmalignant",
        "assay_modality": "mass_spectrometry",
        "is_cell_line": False,
        "source_record_id": f"source:{donor}:{tissue}",
        **changes,
    }


def test_counts_people_across_tissues_and_alleles_not_assays():
    rows = [observation("D1", mhc_class="I", mhc_restriction="HLA-A*02:01")] * 3
    rows += [observation("D1", "Lung", mhc_class="II")]
    rows += [observation("D2", "Cerebellum", mhc_class="II")]
    rows += [observation("D1", peptide="KLGGALQAK")]
    summary, evidence = build_tissue_blacklist(pd.DataFrame(rows))
    result = summary.set_index("peptide")
    assert result.loc["SLYNTVATL", "blacklisted"]
    assert result.loc["SLYNTVATL", "n_donors"] == 2
    assert result.loc["SLYNTVATL", "n_donors_brain"] == 1
    assert result.loc["SLYNTVATL", "n_donors_heart"] == 1
    assert not result.loc["KLGGALQAK", "blacklisted"]
    assert evidence.qualified.all()


def test_aliases_and_unresolved_or_non_ms_evidence_do_not_establish_two_people():
    rows = [observation("D1"), observation("copy:D1")]
    rows += [observation("D2", donor_status="pooled")]
    rows += [observation("D3", assay_modality="binding")]
    rows += [observation("D4", tissue_status="malignant")]
    rows += [observation("D5", is_cell_line=True)]
    rows += [observation("D6", "Liver")]
    rows += [observation("", donor_status="unknown")]
    summary, evidence = build_tissue_blacklist(pd.DataFrame(rows), donor_aliases={"copy:D1": "D1"})
    assert summary.iloc[0].n_donors == 1
    assert not summary.iloc[0].blacklisted
    assert json.loads(summary.iloc[0].donor_ids) == ["D1"]
    assert evidence.qualified.sum() == 2
    assert "unresolved_donor" in set(evidence.exclusion_reason)


def test_exact_sequences_do_not_infer_nested_epitopes():
    rows = [observation(d, peptide="ASLYNTVATLK") for d in ("D1", "D2")]
    rows += [observation("D1")]
    summary, _ = build_tissue_blacklist(pd.DataFrame(rows))
    assert set(summary.loc[summary.blacklisted, "peptide"]) == {"ASLYNTVATLK"}


def test_unknown_identity_is_visible_even_without_qualified_observations():
    summary, evidence = build_tissue_blacklist(
        pd.DataFrame([observation("", donor_status="unknown")])
    )
    assert summary.iloc[0].n_donors == 0
    assert summary.iloc[0].n_unresolved_donor_observations == 1
    assert not summary.iloc[0].blacklisted
    assert evidence.iloc[0].exclusion_reason == "unresolved_donor"


@pytest.mark.parametrize("minimum", [0, 1.5, True])
def test_invalid_threshold_fails(minimum):
    with pytest.raises(ValueError, match="positive integer"):
        build_tissue_blacklist(pd.DataFrame([observation("D1")]), min_donors=minimum)


@pytest.fixture
def atlas_dir(tmp_path):
    (tmp_path / "HLA_peptides.tsv").write_text(
        "peptide_sequence_id\tpeptide_sequence\n1\tSLYNTVATL\n2\tKLGGALQAK\n"
    )
    (tmp_path / "HLA_sample_hits.tsv").write_text(
        "peptide_sequence_id\tdonor\ttissue\thla_class\n"
        "1\tD1\tHeart\tHLA-I\n1\tD2\tLung\tHLA-II\n"
        "2\tD1\tCerebellum\tHLA-I\n2\tD2\tLiver\tHLA-I\n"
    )
    (tmp_path / "HLA_donors.tsv").write_text(
        "donor\thla_allele\nD1\tA*02:01\nD2\tB*07:02\nD2\tDRB1*01:01\n"
    )
    return tmp_path


def test_atlas_keeps_source_rows_and_sample_typing_without_assigning_presenter(atlas_dir):
    hits, sources = load_atlas_tissue_evidence(atlas_dir, batch_size=1)
    assert len(hits) == 3
    assert hits.sample_hits_row.tolist() == [1, 2, 3]
    assert hits.peptide_row.tolist() == [1, 1, 2]
    assert hits.donor_id.tolist() == [
        "hla-ligand-atlas:D1",
        "hla-ligand-atlas:D2",
        "hla-ligand-atlas:D1",
    ]
    assert hits.mhc_restriction.eq("").all()
    assert "DRB1*01:01" in hits.iloc[1].sample_alleles
    assert all(v["sha256"] for v in sources["files"].values())
    assert sources["n_source_sample_hits"] == 4
    summary, _ = build_tissue_blacklist(hits)
    assert set(summary.loc[summary.blacklisted, "peptide"]) == {"SLYNTVATL"}


def test_atlas_rejects_missing_peptide_or_donor_reference(atlas_dir):
    path = atlas_dir / "HLA_sample_hits.tsv"
    path.write_text(path.read_text().replace("1\tD1", "999\tD1"))
    with pytest.raises(ValueError, match="peptide"):
        load_atlas_tissue_evidence(atlas_dir)


def test_selenocysteine_sequence_is_preserved():
    peptide = "IRVTYCGLUS"
    summary, _ = build_tissue_blacklist(
        pd.DataFrame([observation("D1", peptide=peptide), observation("D2", peptide=peptide)])
    )
    assert summary.iloc[0].peptide == peptide
    assert summary.iloc[0].blacklisted
