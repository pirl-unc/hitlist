import json

import pandas as pd
import pytest

from hitlist.atlas import load_atlas_observations
from hitlist.tissue_blacklist import load_atlas_tissue_evidence
from tests import test_tissue_blacklist

atlas_dir = test_tissue_blacklist.atlas_dir


def test_all_tissues_retain_direct_donors_without_claiming_specimens(atlas_dir):
    frame, sources = load_atlas_observations(atlas_dir, batch_size=1)
    assert len(frame) == 4
    assert frame.sample_hits_row.tolist() == [1, 2, 3, 4]
    assert frame.donor.tolist() == ["D1", "D2", "D1", "D2"]
    assert frame.evidence_row_id.is_unique
    assert frame.reported_sample_group_id.nunique() == 4
    assert frame.specimen_status.eq("unknown").all()
    assert frame.specimen_ids.eq("[]").all()
    assert frame.acquisition_status.eq("unknown").all()
    assert frame.mhc_restriction.eq("").all()
    assert frame.donor_evidence_basis.eq("primary_sample_hit_donor_reference").all()
    assert "DRB1*01:01" in frame.iloc[1].sample_alleles
    assert sources["n_observed_donors"] == 2
    assert sources["n_observations"] == 4
    assert sources["n_reported_sample_groups"] == 4
    json.dumps(sources, allow_nan=False)
    old, _ = load_atlas_tissue_evidence(atlas_dir, batch_size=2)
    kept = frame[frame.source_tissue.ne("Liver")]
    pd.testing.assert_frame_equal(
        old, kept[old.columns].astype(old.dtypes), check_categorical=False
    )


def test_native_identity_survives_source_reordering_and_preserves_rows(atlas_dir):
    before, _ = load_atlas_observations(atlas_dir)
    path = atlas_dir / "HLA_sample_hits.tsv"
    raw = pd.read_csv(path, sep="\t")
    raw.iloc[::-1].to_csv(path, sep="\t", index=False)
    after, _ = load_atlas_observations(atlas_dir)
    assert before.evidence_row_id.tolist() == after.evidence_row_id.tolist()[::-1]
    assert before.sample_hits_row.tolist() == after.sample_hits_row.tolist()
    assert before.source_record_id.tolist() == after.source_record_id.tolist()


@pytest.mark.parametrize("change", ["duplicate", "blank_tissue", "unknown_donor", "unknown_class"])
def test_invalid_source_relations_are_rejected(atlas_dir, change):
    path = atlas_dir / "HLA_sample_hits.tsv"
    raw = pd.read_csv(path, sep="\t")
    if change == "duplicate":
        raw = pd.concat([raw, raw.iloc[:1]], ignore_index=True)
    elif change == "blank_tissue":
        raw.loc[0, "tissue"] = ""
    elif change == "unknown_donor":
        raw.loc[0, "donor"] = "unreported donor"
    else:
        raw.loc[0, "hla_class"] = "HLA-unknown"
    raw.to_csv(path, sep="\t", index=False)
    with pytest.raises(ValueError, match="Atlas"):
        load_atlas_observations(atlas_dir, batch_size=1)


def test_identifier_components_cannot_collide(atlas_dir):
    path = atlas_dir / "HLA_sample_hits.tsv"
    raw = pd.read_csv(path, sep="\t")
    raw.loc[0, "tissue"] = "Heart:HLA-I"
    raw.to_csv(path, sep="\t", index=False)
    frame, _ = load_atlas_observations(atlas_dir)
    assert "Heart%3AHLA-I" in frame.iloc[0].reported_sample_group_id


def test_default_source_route_uses_pinned_datacache_generation(atlas_dir, monkeypatch, tmp_path):
    from hitlist import atlas

    calls = []

    def install(destination, assets, **options):
        calls.append((destination, assets, options))
        return {"peptides.tsv.gz": atlas_dir / "HLA_peptides.tsv"}

    monkeypatch.setattr(atlas, "install_bundle", install)
    frame, _ = load_atlas_observations(data_root=tmp_path)
    assert len(frame) == 4
    destination, assets, _ = calls[0]
    assert destination == tmp_path / "hla-ligand-atlas/2020.12/sources"
    assert set(assets) == {"donors.tsv.gz", "peptides.tsv.gz", "sample_hits.tsv.gz"}
    assert all(len(asset["sha256"]) == 64 and asset["size"] > 0 for asset in assets.values())


def test_summary_typing_follows_donor_groups_with_reordered_categories(atlas_dir):
    from scripts.audit_atlas_lineage_v2 import donor_summary

    frame, _ = load_atlas_observations(atlas_dir)
    frame["partition"] = "train"
    frame["donor_id"] = frame.donor_id.cat.reorder_categories(
        list(reversed(frame.donor_id.cat.categories))
    )
    frame["sample_alleles"] = frame.sample_alleles.cat.reorder_categories(
        list(reversed(frame.sample_alleles.cat.categories))
    )
    summary = donor_summary(frame).set_index("donor_id")
    assert summary.loc["hla-ligand-atlas:D1", "sample_alleles"] == "A*02:01"
    assert summary.loc["hla-ligand-atlas:D2", "sample_alleles"] == "B*07:02;DRB1*01:01"
    assert summary.n_observations.sum() == len(frame)


def test_summary_refuses_inconsistent_typing_for_one_donor(atlas_dir):
    from scripts.audit_atlas_lineage_v2 import donor_summary

    frame, _ = load_atlas_observations(atlas_dir)
    frame["partition"] = "train"
    frame.loc[0, "sample_alleles"] = "B*07:02;DRB1*01:01"
    with pytest.raises(ValueError, match="inconsistent"):
        donor_summary(frame)
