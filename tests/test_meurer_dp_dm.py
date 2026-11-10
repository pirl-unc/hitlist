"""Source identity and shared-export regression checks for PXD017154 (#895)."""

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from hitlist.curation import load_pmid_overrides
from hitlist.export import generate_training_table
from hitlist.supplement import load_supplementary_manifest, scan_supplementary

DATA = Path(__file__).resolve().parents[1] / "hitlist/data/supplementary"


def entries():
    return [entry for entry in load_supplementary_manifest() if entry["pmid"] == 33025005]


@pytest.mark.integration
def test_native_assets_preserve_samples_confidence_and_controls():
    manifest = yaml.safe_load((DATA / "meurer_2021_dp_dm_source.yaml").read_text())
    frames = []
    for entry in entries():
        path = DATA / entry["file"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]
        assert path.stat().st_size == entry["size_bytes"]
        frame = pd.read_csv(path, keep_default_na=False)
        assert frame.attributed_sample_label.nunique() == 1
        assert not frame.has_ptm.any()
        assert frame.peptide_modifications.eq("").all()
        assert not frame.observed_in_no_dp_control.any()
        assert frame.native_confidence.eq("high").all()
        assert frame.native_target_fpr_high.eq(0.01).all()
        assert frame.source_qualified_candidate.all()
        assert frame.n_supporting_psms.ge(frame.n_acquisitions).all()
        frames.append(frame)
    positives = pd.concat(frames, ignore_index=True)
    assert len(positives) == 21547
    assert positives.peptide.nunique() == 10143
    assert not positives.duplicated(["attributed_sample_label", "peptide"]).any()
    assert (
        positives.groupby("attributed_sample_label").size().to_dict()
        == manifest["sample_candidate_counts"]
    )
    controls = pd.read_csv(DATA / manifest["control_companion"]["file"])
    assert len(controls) == 87 and controls.peptide.nunique() == 17
    assert not set(controls.peptide) & set(positives.peptide)
    assert manifest["control_companion"]["file"] not in {e["file"] for e in entries()}
    for support in json.loads(positives.iloc[0].native_psm_support):
        assert (
            support["source_sha256"]
            == manifest["source_original_msf_sha256"][support["source_file"]]
        )


def test_shared_export_keeps_dm_arms_separate_for_the_same_peptide(tmp_path, monkeypatch):
    selected = []
    for entry in entries():
        label = entry["defaults"]["attributed_sample_label"]
        if label not in {"DP10_A", "DP10_DM_A", "DP401_DM_A"}:
            continue
        fixture = {k: v for k, v in entry.items() if k not in {"sha256", "size_bytes"}}
        fixture["file"] = label + ".csv"
        pair = load_pmid_overrides()[33025005]["ms_samples"]
        mhc = next(s["mhc"] for s in pair if s["sample_label"] == label)
        pd.DataFrame(
            [
                {
                    "peptide": "ACDEFGHIKLMN",
                    "mhc_class": "II",
                    "mhc_restriction": mhc,
                    "has_ptm": "false",
                    "peptide_modifications": "",
                }
            ]
        ).to_csv(tmp_path / fixture["file"], index=False)
        selected.append(fixture)
    frame = scan_supplementary(entries=selected, directory=tmp_path, allow_download=False)
    assert len(frame) == 3 and frame.assay_iri.nunique() == 3
    path = tmp_path / "observations.parquet"
    frame.assign(source="supplement").to_parquet(path, index=False)
    monkeypatch.setattr("hitlist.observations.observations_path", lambda: path)
    result = generate_training_table(include_evidence="ms")
    assert len(result) == 3 and result.evidence_row_id.nunique() == 3
    indexed = result.set_index("sample_label")
    assert indexed.loc["DP10_A", "condition_transduction"] == "none"
    assert indexed.loc["DP10_DM_A", "condition_transduction"] == "HLA-DM"
    assert indexed.loc["DP401_DM_A", "mhc_allele_set"] == "HLA-DPA1*01:03/DPB1*04:01"
    assert result.sample_attribution.eq("curated_sample_label").all()
    assert result.condition_evidence.eq("primary_source").all()
    assert result.condition_background.eq("CD74_transduced;CD80_transduced;HLA-DP_transduced").all()
    assert result.experimental_origin_ids.eq('["origin:PXD017154"]').all()
    assert result.specimen_status.eq("unknown").all() and result.donor_status.eq("unknown").all()
    assert result.quantitative_value.isna().all()
    assert not result.is_binding_assay.any()


def test_pinned_native_asset_change_is_rejected(tmp_path):
    entry = entries()[0]
    (tmp_path / entry["file"]).write_bytes(b"wrong input")
    with pytest.raises(ValueError, match="checksum/size mismatch"):
        scan_supplementary(entries=[entry], directory=tmp_path, allow_download=False)
