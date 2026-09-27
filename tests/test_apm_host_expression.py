"""Measured HAP1 reference RNA and honest KO provenance (#358)."""

import gzip
import hashlib
from importlib.resources import files

import pandas as pd
import pytest
import yaml

from hitlist import downloads
from hitlist.builder import build_line_expression
from hitlist.curation import load_pmid_overrides
from hitlist.line_expression import load_line_expression, resolve_sample_expression_anchor


@pytest.fixture(autouse=True)
def empty_optional_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(downloads, "_override_data_dir", tmp_path)


def test_every_curated_hap1_ko_uses_parent_rna():
    study = load_pmid_overrides()[40113210]
    assert len(study["ms_samples"]) == 12
    for sample in study["ms_samples"]:
        anchor = resolve_sample_expression_anchor(sample["sample_label"])
        is_control = sample["condition_knockout_genes"] == "none"
        assert anchor.expression_match_tier == (1 if is_control else 2), sample["sample_label"]
        assert anchor.expression_key == "HAP1"
        assert anchor.expression_parent_key == (None if is_control else "HAP1")
        assert anchor.source_ids == ("DepMap_24Q4_HAP1_gene",)
    rows = load_line_expression(line_key="HAP1", granularity="gene")
    assert len(rows) == 19193
    assert rows.tpm.notna().all() and rows.tpm.ge(0).all()


def test_packaged_hap1_matches_its_build_report():
    # Catches an edited CSV whose recorded provenance was not regenerated.
    data = files("hitlist.data.line_expression")
    report = yaml.safe_load(data.joinpath("apm_rna_build_report.yaml").read_text())
    (entry,) = report["sources"]
    raw = data.joinpath(entry["file"]).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == entry["sha256"]
    assert hashlib.sha256(gzip.decompress(raw)).hexdigest() == entry["content_sha256"]
    rows = load_line_expression(source_id=entry["source_id"])
    assert len(rows) == entry["n_genes"]
    assert int(rows.tpm.gt(0).sum()) == entry["n_positive_genes"]
    assert rows.tpm.sum() == pytest.approx(entry["sum_tpm"])


def test_downloaded_bundle_adds_hap1_transcripts_without_duplicating_genes(tmp_path):
    frames = {
        "depmap_models": pd.DataFrame(
            {"ModelID": ["ACH-002475", "ACH-001"], "StrippedCellLineName": ["HAP1", "HELA"]}
        ),
        "depmap_profiles": pd.DataFrame(
            {
                "ProfileID": ["PR-QtHaIL", "PR-hela"],
                "ModelID": ["ACH-002475", "ACH-001"],
                "Datatype": ["rna", "rna"],
                "Stranded": [True, False],
            }
        ),
        "depmap_default_profiles": pd.DataFrame(
            {
                "ModelID": ["ACH-002475", "ACH-001"],
                "ProfileID": ["PR-QtHaIL", "PR-hela"],
                "ProfileType": ["rna", "rna"],
            }
        ),
        "depmap_rna": pd.DataFrame({"": ["ACH-002475", "ACH-001"], "TP53 (7157)": [20.0, 3.0]}),
        "depmap_rna_transcript": pd.DataFrame(
            {"": ["PR-QtHaIL", "PR-hela"], "TP53 (ENST00000269305)": [4.0, 2.0]}
        ),
    }
    for key, frame in frames.items():
        path = tmp_path / f"{key}.csv"
        frame.to_csv(path, index=False)
        downloads.register(key, path)
    rows = build_line_expression()

    hap1 = rows[rows.line_key.eq("HAP1")]
    genes = hap1[hap1.granularity.eq("gene")]
    assert len(genes) == 19193
    assert genes.source_id.eq("DepMap_24Q4_HAP1_gene").all()
    assert not genes.tpm.eq(2**20 - 1).any()
    # Metadata is stamped from sources.yaml, not carried in the packaged CSV.
    assert genes.backend.eq("depmap_rna").all()
    assert genes.license.eq("CC BY 4.0 (Broad DepMap)").all()
    transcripts = hap1[hap1.granularity.eq("transcript")]
    assert transcripts.source_id.tolist() == ["DepMap_24Q4_transcript"]
    assert transcripts.tpm.tolist() == [15.0]
    # Lines that list the downloaded gene source still receive it.
    hela = rows[rows.line_key.eq("HeLa") & rows.granularity.eq("gene")]
    assert hela.source_id.tolist() == ["DepMap_24Q4_gene"]

    anchor = resolve_sample_expression_anchor("HAP1 wildtype")
    assert anchor.expression_backend == "depmap_rna"
    assert anchor.source_ids == ("DepMap_24Q4_HAP1_gene", "DepMap_24Q4_transcript")
