"""DepMap 24Q4's published file shapes and optional-data lifecycle (#357)."""

import shutil

import pandas as pd
import pytest

from hitlist import downloads
from hitlist.builder import _read_depmap_csv, build_line_expression
from hitlist.line_expression import (
    compute_peptide_origin,
    load_line_expression,
    resolve_sample_expression_anchor,
)


def _files(tmp_path):
    # Headers/ID roles follow the 24Q4 README and downloaded CSV prefixes.
    frames = {
        "depmap_models": pd.DataFrame(
            {"ModelID": ["ACH-001", "ACH-002"], "StrippedCellLineName": ["HELA", "UNRELATED"]}
        ),
        "depmap_profiles": pd.DataFrame(
            {
                "ProfileID": ["PR-current", "PR-old", "PR-other"],
                "ModelID": ["ACH-001", "ACH-001", "ACH-002"],
                "Datatype": ["rna"] * 3,
                # Sequencing protocol does not imply absence from the matrix:
                # HAP1 has a stranded library and unstranded-mode gene values.
                "Stranded": [True, False, False],
            }
        ),
        "depmap_default_profiles": pd.DataFrame(
            {
                "ModelID": ["ACH-001", "ACH-002"],
                "ProfileID": ["PR-current", "PR-other"],
                "ProfileType": ["rna"] * 2,
            }
        ),
        "depmap_rna": pd.DataFrame({"": ["ACH-001", "ACH-002"], "TP53 (7157)": [3.0, 9.0]}),
        "depmap_rna_transcript": pd.DataFrame(
            {
                "": ["PR-current", "PR-old", "PR-other"],
                "TP53 (ENST00000269305)": [2.0, 9.0, 8.0],
                "TP53 (ENST00000604348.1)": [0.0, 10.0, 7.0],
            }
        ),
    }
    paths = {}
    for key, frame in frames.items():
        path = tmp_path / (key + ".csv")
        frame.to_csv(path, index=False)
        paths[key] = path
    return paths


def test_official_transcript_headers_keep_ids_symbols_and_zero_expression(tmp_path):
    path = _files(tmp_path)["depmap_rna_transcript"]
    rows = _read_depmap_csv(path, "transcript")
    current = rows[rows.line_key == "PR-current"]
    assert current.transcript_id.tolist() == ["ENST00000269305", "ENST00000604348"]
    assert current.gene_name.tolist() == ["TP53", "TP53"]
    assert current.tpm.tolist() == [3.0, 0.0]


def test_filtering_precedes_numeric_expansion_of_unregistered_rows(tmp_path):
    path = tmp_path / "selected.csv"
    path.write_text("ModelID,TP53 (7157)\nACH-unrelated,not-a-number\nHELA,3.0\n")
    rows = _read_depmap_csv(path, "gene", line_keys={"hela": "HeLa"})
    assert rows.line_key.tolist() == ["HeLa"]
    assert rows.tpm.tolist() == [7.0]


def test_profile_rows_require_their_mapping_companions(tmp_path):
    path = _files(tmp_path)["depmap_rna_transcript"]
    with pytest.raises(ValueError, match="depmap_profiles and depmap_default_profiles"):
        _read_depmap_csv(path, "transcript", line_keys={"ACH-001": "HeLa"})


def test_builder_uses_default_rna_profile_and_never_pools_old_profiles(tmp_path, monkeypatch):
    monkeypatch.setattr(downloads, "_override_data_dir", tmp_path)
    for key, path in _files(tmp_path).items():
        downloads.register(key, path)
    rows = build_line_expression()
    assert not rows.profile_id.isna().any()
    transcripts = rows[(rows.line_key == "HeLa") & (rows.granularity == "transcript")]
    assert transcripts.tpm.tolist() == [3.0, 0.0]
    assert transcripts.profile_id.tolist() == ["PR-current", "PR-current"]
    assert transcripts.source_id.eq("DepMap_24Q4_transcript").all()
    assert "ACH-002" not in set(rows.line_key)


def test_uninstalled_and_rebuilt_expression_anchors_follow_actual_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(downloads, "_override_data_dir", tmp_path)
    before = resolve_sample_expression_anchor("HeLa cells")
    assert before.expression_match_tier == 5
    path = tmp_path / "line_expression.parquet"
    pd.DataFrame({"line_key": ["HeLa"], "source_id": ["DepMap_24Q4_gene"]}).to_parquet(path)
    installed = resolve_sample_expression_anchor("HeLa cells")
    assert installed.expression_match_tier == 1
    assert installed.source_ids == ("DepMap_24Q4_gene",)
    parent = resolve_sample_expression_anchor("HeLa.ABC-KO-HLA-B*51:01")
    assert parent.expression_match_tier == 2
    pd.DataFrame(
        {"line_key": ["GM12878"], "source_id": ["ENCODE_GM12878_polyA_rnaseq"]}
    ).to_parquet(path)
    assert resolve_sample_expression_anchor("HeLa cells").expression_match_tier == 5
    assert resolve_sample_expression_anchor("C1R cells").expression_match_tier >= 4


def test_available_rows_must_match_both_line_and_source(tmp_path, monkeypatch):
    monkeypatch.setattr(downloads, "_override_data_dir", tmp_path)
    pd.DataFrame(
        {"line_key": ["HeLa", "K562"], "source_id": ["unrelated", "DepMap_24Q4_gene"]}
    ).to_parquet(tmp_path / "line_expression.parquet")
    assert resolve_sample_expression_anchor("HeLa cells").expression_match_tier == 5
    assert resolve_sample_expression_anchor("C1R cells").expression_key == "K562"


def test_depmap_fetch_bundle_reaches_peptide_origin(tmp_path, monkeypatch):
    downloads_dir = tmp_path / "source"
    downloads_dir.mkdir()
    fixtures = _files(downloads_dir)
    cache_dir = tmp_path / "cache"
    monkeypatch.setattr(downloads, "_override_data_dir", cache_dir)
    downloaded = []

    def fake_download(url, dest, *, label, **kwargs):
        downloaded.append(label)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(fixtures[label], dest)
        return dest

    monkeypatch.setattr(downloads, "download_to_file", fake_download)
    result = downloads.fetch("depmap")
    assert result == cache_dir / "line_expression.parquet"
    assert set(downloaded) == set(fixtures)
    anchor = resolve_sample_expression_anchor("HeLa cells")
    assert anchor.expression_match_tier == 1
    origin = compute_peptide_origin(
        "AAAAAAAAA",
        candidate_genes=["TP53"],
        line_expression_df=load_line_expression(
            line_key=anchor.expression_key, source_id=anchor.source_ids
        ),
    )
    assert origin["peptide_origin_gene"] == "TP53"
    assert origin["peptide_origin_tpm"] == pytest.approx(7.0)
    downloads.fetch("depmap")
    assert len(downloaded) == len(fixtures)
