"""Tests for Parquet standardization, modality narrowing, samples.parquet, and alleles.parquet (#990, #992)."""

from __future__ import annotations

import json

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from hitlist import builder, bundle_tables, downloads, mappings
from hitlist.parquet_io import atomic_write_parquet
from tests.test_build_smoke import _write_synthetic_iedb


def test_atomic_write_parquet_standardizes_arrow_types_and_sorts(tmp_path):
    df = pd.DataFrame(
        {
            "mhc_class": pd.Categorical(["II", "I", "I", "II"]),
            "mhc_species": ["Homo sapiens", "Mus musculus", "Homo sapiens", "Bos taurus"],
            "mhc_allele_set": [
                "HLA-DRB1*01:01",
                "H2-Kb",
                "HLA-A*02:01",
                "BoLA-DRB3*01:01",
            ],
            "peptide": ["DDDDDDDDDDDD", "SIINFEKL", "GILGFVFTL", "AAAAAAAAAAAA"],
            "is_potential_contaminant": [None, None, None, None],
            "quantitative_value": [None, None, None, None],
            "n_replicates": [None, None, None, None],
            "notes": [None, None, None, None],
        }
    )
    out = tmp_path / "sorted.parquet"
    atomic_write_parquet(
        df,
        out,
        sort_by=bundle_tables.MODALITY_SORT_COLUMNS,
        row_group_size=2,
    )

    pf = pq.ParquetFile(out)
    assert pf.metadata.num_row_groups == 2
    for field in pf.schema_arrow:
        assert not pa.types.is_null(field.type), f"{field.name} should not be Arrow null"
        assert not pa.types.is_large_string(field.type), (
            f"{field.name} should be normalized from large_string to string"
        )
        if pa.types.is_dictionary(field.type):
            assert pa.types.is_string(field.type.value_type)

    assert pf.schema_arrow.field("is_potential_contaminant").type == pa.bool_()
    assert pf.schema_arrow.field("quantitative_value").type == pa.float64()
    assert pf.schema_arrow.field("n_replicates").type == pa.int64()
    assert pf.schema_arrow.field("notes").type == pa.string()

    reloaded = pd.read_parquet(out)
    assert reloaded["mhc_class"].astype(str).tolist() == ["I", "I", "II", "II"]
    assert reloaded["mhc_species"].tolist() == [
        "Homo sapiens",
        "Mus musculus",
        "Bos taurus",
        "Homo sapiens",
    ]


def test_narrow_modality_frame_drops_only_vacuous_cross_modality_columns():
    ms_df = pd.DataFrame(
        {
            "peptide": ["GILGFVFTL"],
            "quantitative_measurement": [""],
            "quantitative_value": [float("nan")],
            "measurement_units": [""],
            "measurement_inequality": [""],
            "is_binding_assay": [False],
            "is_ms_observation": [True],
        }
    )
    narrowed_ms = bundle_tables.narrow_modality_frame(ms_df, "ms")
    assert set(narrowed_ms.columns) == {"peptide", "is_ms_observation"}

    binding_df = pd.DataFrame(
        {
            "peptide": ["GILGFVFTL"],
            "quantitative_value": [12.5],
            "is_binding_assay": [True],
            "is_ms_observation": [False],
            "ms_outcome_status": [""],
            "is_potential_contaminant": [None],
            "src_healthy_thymus": [False],
            # Non-vacuous column must NOT be dropped even if listed in MS_ONLY_COLUMNS.
            "attributed_sample_label": ["keep_me"],
        }
    )
    narrowed_bd = bundle_tables.narrow_modality_frame(binding_df, "binding")
    assert "is_binding_assay" in narrowed_bd.columns
    assert "quantitative_value" in narrowed_bd.columns
    assert "attributed_sample_label" in narrowed_bd.columns
    assert "is_ms_observation" not in narrowed_bd.columns
    assert "is_potential_contaminant" not in narrowed_bd.columns
    assert "src_healthy_thymus" not in narrowed_bd.columns


def test_canonical_allele_set_roundtrip():
    assert bundle_tables.parse_allele_set("HLA-B*07:02; HLA-A*02:01; HLA-A*02:01") == (
        "HLA-A*02:01",
        "HLA-B*07:02",
    )
    assert (
        bundle_tables.format_allele_set(["HLA-B*07:02", "HLA-A*02:01", "HLA-A*02:01"])
        == "HLA-A*02:01;HLA-B*07:02"
    )
    assert bundle_tables.parse_allele_set("") == ()
    assert bundle_tables.parse_allele_set(None) == ()


def test_build_samples_table_counts_and_flank_coverage():
    obs = pd.DataFrame(
        {
            "pmid": [28228285, 28228285, 28228285, 36423003],
            "reference_title": ["Abelin 2017", "Abelin 2017", "Abelin 2017", "Cattle 2022"],
            "source": ["iedb", "iedb", "iedb", "iedb"],
            "mhc_class": ["I", "I", "I", "I"],
            "mhc_species": ["Homo sapiens", "Homo sapiens", "Homo sapiens", "Bos taurus"],
            "source_organism": ["Homo sapiens", "Homo sapiens", "Homo sapiens", "Bos taurus"],
            "mhc_allele_set": ["HLA-A*02:01", "HLA-A*02:01", "HLA-A*02:01", "BoLA-1*023:01"],
            "is_monoallelic": [True, True, True, True],
            "cell_line_name": ["721.221", "721.221", "721.221", ""],
            "cell_type": ["B-LCL", "B-LCL", "B-LCL", "PBMC"],
            "source_tissue": ["", "", "", "Blood"],
            "disease": ["Healthy", "Healthy", "Healthy", "Healthy"],
            "attributed_sample_label": ["a0201", "a0201", "a0201", ""],
            "peptide": ["GILGFVFTL", "NLVPMVATV", "GILGFVFTL", "KLQCVDLHV"],
            "has_ptm": [False, True, False, False],
            "is_non_peptide_ligand": [False, False, False, False],
            "is_potential_contaminant": [False, False, False, False],
            "src_cell_line": [True, True, True, False],
        }
    )
    flanks = pd.DataFrame(
        {
            "peptide": ["GILGFVFTL", "KLQCVDLHV"],
            "proteome_id": [
                "Homo sapiens",
                "Homo sapiens",
            ],  # KLQCVDLHV is human-only, NOT Bos taurus!
            "n_flank": ["AAAAA", "CCCCC"],
            "c_flank": ["TTTTT", "GGGGG"],
        }
    )
    samples = bundle_tables.build_samples_table(obs, primary_flanks=flanks)
    assert len(samples) == 2

    human = samples[samples["mhc_species"] == "Homo sapiens"].iloc[0]
    assert human["n_rows"] == 3
    assert human["n_unique_peptides"] == 2
    assert human["n_clean_peptides"] == 1  # NLVPMVATV has has_ptm=True
    assert human["n_mapped_peptides"] == 1
    assert human["n_flanked_peptides"] == 1
    assert human["flank_coverage"] == pytest.approx(0.5)
    assert human["sample_id"].startswith("pmid28228285:")

    # Cattle peptide must NOT inherit human flank!
    cattle = samples[samples["mhc_species"] == "Bos taurus"].iloc[0]
    assert cattle["n_unique_peptides"] == 1
    assert cattle["n_mapped_peptides"] == 0
    assert cattle["n_flanked_peptides"] == 0
    assert cattle["flank_coverage"] == pytest.approx(0.0)


def test_build_alleles_table_canonical_and_accessioned():
    obs = pd.DataFrame(
        {
            "peptide": ["GILGFVFTL", "NLVPMVATV", "SLYNTVATL", "KLQCVDLHV", "FMYSDFHFI"],
            "mhc_allele_set": [
                "HLA-A*02:01",
                "HLA-A*02:01;HLA-B*07:02",
                "DLA-88*501:01",
                "HLA-A*02",
                "",
            ],
            "mhc_restriction": [
                "HLA-A*02:01",
                "HLA class I",
                "DLA-88*501:01",
                "HLA-A*02",
                "HLA class I",
            ],
            "is_monoallelic": [True, False, True, False, False],
            "pmid": [1001, 1002, 1003, 1004, 1005],
        }
    )
    binding = pd.DataFrame(
        {
            "peptide": ["GILGFVFTL", "YLEPGPVTA"],
            "mhc_allele_set": ["HLA-A*02:01", "BoLA-1*023:01"],
            "mhc_restriction": ["HLA-A*02:01", "BoLA-1*023:01"],
            "pmid": [2001, 2002],
        }
    )
    other = pd.DataFrame(
        {
            "peptide": ["FMYSDFHFI"],
            "mhc_allele_set": ["HLA-A*02:01"],
            "mhc_restriction": ["HLA-A*02:01"],
            "pmid": [3001],
        }
    )
    alleles = bundle_tables.build_alleles_table(obs, binding=binding, other=other)
    by_allele = alleles.set_index("allele")

    a02 = by_allele.loc["HLA-A*02:01"]
    assert a02["mhc_species"] == "Homo sapiens"
    assert a02["mhc_class"] == "I"
    assert a02["gene"] == "A"
    assert a02["allele_resolution"] == "four_digit"
    assert bool(a02["is_accessioned"]) is True
    assert a02["n_ms_monoallelic_peptides"] == 1
    assert a02["n_ms_multiallelic_peptides"] == 1
    assert a02["n_binding_peptides"] == 1
    assert a02["n_other_assay_peptides"] == 1
    assert a02["n_studies"] == 4

    dla = by_allele.loc["DLA-88*501:01"]
    assert bool(dla["is_accessioned"]) is True
    assert dla["n_ms_monoallelic_peptides"] == 1

    bola = by_allele.loc["BoLA-1*023:01"]
    assert bool(bola["is_accessioned"]) is True
    assert bola["n_binding_peptides"] == 1

    low_res = by_allele.loc["HLA-A*02"]
    assert bool(low_res["is_accessioned"]) is False

    # Class-only restrictions must never be counted as monoallelic MS alleles.
    class_only = by_allele.loc["human class I"]
    assert class_only["allele_resolution"] == "class_only"
    assert bool(class_only["is_accessioned"]) is False
    assert class_only["n_ms_monoallelic_peptides"] == 0
    assert class_only["n_ms_multiallelic_peptides"] == 1


def test_build_observations_writes_summary_tables_and_clean_arrow_schemas(
    tmp_path, monkeypatch, _isolated_curation_root
):
    from hitlist import supplement

    (_isolated_curation_root / "pmid_overrides.yaml").write_text("[]\n")
    source = tmp_path / "iedb.csv"
    _write_synthetic_iedb(source)
    downloads.register("iedb", source)
    monkeypatch.setattr(supplement, "scan_supplementary", lambda **kw: pd.DataFrame())

    def empty_index(path):
        frame = pd.DataFrame()
        atomic_write_parquet(frame, path)
        return frame

    monkeypatch.setattr(
        builder, "build_bulk_proteomics", lambda **kw: empty_index(builder._bulk_proteomics_path())
    )
    monkeypatch.setattr(
        builder, "build_line_expression", lambda **kw: empty_index(builder._line_expression_path())
    )

    def tiny_mappings(**kwargs):
        path = mappings.mappings_path()
        frame = kwargs.get("obs_override")
        if frame is None:
            frame = pd.read_parquet(builder._observations_path())
        atomic_write_parquet(frame[["peptide"]], path)
        mappings.mappings_meta_path().write_text(
            json.dumps({"observations": mappings._obs_fingerprint()})
        )
        return path

    monkeypatch.setattr(mappings, "build_peptide_mappings", tiny_mappings)

    obs_path = builder.build_observations()
    bundle = obs_path.parent
    assert (bundle / "samples.parquet").is_file()
    assert (bundle / "alleles.parquet").is_file()

    for name in ("observations.parquet", "binding.parquet", "samples.parquet", "alleles.parquet"):
        pf = pq.ParquetFile(bundle / name)
        for field in pf.schema_arrow:
            assert not pa.types.is_null(field.type), f"{name}:{field.name} is Arrow null"
            assert not pa.types.is_large_string(field.type), (
                f"{name}:{field.name} is Arrow large_string"
            )

    samples = bundle_tables.load_samples_index()
    alleles = bundle_tables.load_alleles()
    assert not samples.empty
    assert not alleles.empty
    assert int(samples["n_rows"].sum()) == len(pd.read_parquet(obs_path))

    # Standalone backfill into an explicit output directory works too.
    backfill_dir = tmp_path / "backfill"
    out = bundle_tables.build_bundle_tables(bundle=bundle, output_dir=backfill_dir)
    assert out["samples"].is_file()
    assert out["alleles"].is_file()
