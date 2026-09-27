"""Measured HAP1 reference RNA and honest KO provenance (#358)."""

import pandas as pd
import pytest

from hitlist import downloads
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
        assert "DepMap_24Q4_HAP1_gene" in anchor.source_ids
    rows = load_line_expression(line_key="HAP1", granularity="gene")
    assert len(rows) == 19193
    assert rows.tpm.notna().all() and rows.tpm.ge(0).all()


def test_old_index_does_not_hide_newly_packaged_hap1(tmp_path):
    pd.DataFrame(
        {"line_key": ["GM12878"], "source_id": ["ENCODE_GM12878_polyA_rnaseq"]}
    ).to_parquet(tmp_path / "line_expression.parquet")
    anchor = resolve_sample_expression_anchor("HAP1 wildtype")
    assert anchor.expression_match_tier == 1
    rows = load_line_expression(line_key=anchor.expression_key, source_id=anchor.source_ids)
    assert len(rows) == 19193


def test_downloaded_gene_copy_does_not_duplicate_packaged_hap1(tmp_path):
    from hitlist.builder import build_line_expression

    path = tmp_path / "gene.csv"
    pd.DataFrame({"ModelID": ["HAP1"], "TP53 (7157)": [20.0]}).to_csv(path, index=False)
    downloads.register("depmap_rna", path)
    rows = build_line_expression()
    hap1 = rows[rows.line_key.eq("HAP1") & rows.granularity.eq("gene")]
    assert len(hap1) == 19193
    assert hap1.source_id.eq("DepMap_24Q4_HAP1_gene").all()
    assert not hap1.tpm.eq(2**20 - 1).any()
