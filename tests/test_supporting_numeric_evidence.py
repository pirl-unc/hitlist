"""Offline native-value and immutable-snapshot checks for numeric evidence."""

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from hitlist.evidence_catalog import describe_evidence
from hitlist.supporting_numeric_evidence import _snapshot_file, _write_one
from hitlist.tcell_evidence import verify_tcell_evidence_package


def _write(path, rows):
    pq.write_table(pa.Table.from_pylist(rows), path)
    return path


def _typed_row(kind, index, source, grain, **extra):
    native = "bulk_proteomics.parquet" if kind == "bulk" else "line_expression.parquet"
    prefix = "bulk" if kind == "bulk" else "expression"
    return {
        "native_row_index": index,
        "source_record_id": f"{native}:{index}",
        "observation_id": f"{prefix}:v4:{index}",
        "source_dataset": source,
        "grain": grain,
        **extra,
    }


def test_bulk_numeric_cells_keep_scales_without_fabricating_peptide_quantity(tmp_path):
    native = _write(
        tmp_path / "native.parquet",
        [
            {
                "source": "Bekker-Jensen_2017",
                "granularity": "protein",
                "log2_intensity": 20.5,
                "abundance_log2_normalized": None,
                "abundance_percentile": 0.4,
            },
            {
                "source": "Bekker-Jensen_2017",
                "granularity": "peptide",
                "log2_intensity": None,
                "abundance_log2_normalized": None,
                "abundance_percentile": None,
            },
            {
                "source": "CCLE_Nusinow_2020",
                "granularity": "protein",
                "log2_intensity": None,
                "abundance_log2_normalized": -1.5,
                "abundance_percentile": 0.0,
            },
        ],
    )
    typed = _write(
        tmp_path / "typed.parquet",
        [
            _typed_row("bulk", 0, "Bekker-Jensen_2017", "protein"),
            _typed_row("bulk", 1, "Bekker-Jensen_2017", "peptide"),
            _typed_row("bulk", 2, "CCLE_Nusinow_2020", "protein"),
        ],
    )
    output = tmp_path / "measurements.parquet"
    counts = _write_one(native, typed, output, kind="bulk")
    rows = pq.read_table(output).to_pylist()
    assert counts["n_source_rows"] == 3
    assert counts["n_measurements"] == 4
    assert counts["n_peptide_rows_with_numeric_value"] == 0
    assert {row["source_value_field"] for row in rows} == {
        "log2_intensity",
        "abundance_log2_normalized",
        "abundance_percentile",
    }
    assert {row["observation_id"] for row in rows} == {"bulk:v4:0", "bulk:v4:2"}
    assert {row["quantitative_value"] for row in rows} == {20.5, 0.4, -1.5, 0.0}
    assert all(row["unit_status"] == "unreported" for row in rows)


def test_expression_keeps_measured_zero_and_native_gene_profile_link(tmp_path):
    native = _write(
        tmp_path / "native.parquet",
        [
            {
                "source_id": "ENCODE_GM12878_polyA_rnaseq",
                "granularity": "gene",
                "gene_id": "ENSG1",
                "profile_id": "",
                "tpm": 12.0,
                "log2_tpm": 3.7,
            },
            {
                "source_id": "DepMap_24Q4_HAP1_gene",
                "granularity": "gene",
                "gene_id": "ENSG2",
                "profile_id": "HAP1",
                "tpm": 0.0,
                "log2_tpm": 0.0,
            },
        ],
    )
    typed = _write(
        tmp_path / "typed.parquet",
        [
            _typed_row(
                "expression",
                0,
                "ENCODE_GM12878_polyA_rnaseq",
                "gene_profile",
                gene_id="ENSG1",
                profile_id_raw="",
            ),
            _typed_row(
                "expression",
                1,
                "DepMap_24Q4_HAP1_gene",
                "gene_profile",
                gene_id="ENSG2",
                profile_id_raw="HAP1",
            ),
        ],
    )
    output = tmp_path / "measurements.parquet"
    counts = _write_one(native, typed, output, kind="expression")
    rows = pq.read_table(output).to_pylist()
    assert counts["n_measurements"] == 4
    assert counts["n_zero_tpm_measurements"] == 1
    zeros = [row for row in rows if row["observation_id"] == "expression:v4:1"]
    assert len(zeros) == 2
    assert all(row["quantitative_value"] == 0.0 for row in zeros)
    assert {row["quantity_scale"] for row in zeros} == {"tpm", "log2_tpm"}
    assert next(row for row in zeros if row["quantity_scale"] == "tpm")["measurement_unit"] == "TPM"


def test_source_row_alignment_is_required(tmp_path):
    native = _write(
        tmp_path / "native.parquet",
        [
            {
                "source_id": "DepMap_24Q4_HAP1_gene",
                "granularity": "gene",
                "gene_id": "ENSG1",
                "profile_id": "HAP1",
                "tpm": 0.0,
                "log2_tpm": 0.0,
            }
        ],
    )
    typed = _write(
        tmp_path / "typed.parquet",
        [
            _typed_row(
                "expression",
                0,
                "DepMap_24Q4_HAP1_gene",
                "gene_profile",
                gene_id="DIFFERENT",
                profile_id_raw="HAP1",
            )
        ],
    )
    with pytest.raises(ValueError, match="gene/profile alignment mismatch"):
        _write_one(native, typed, tmp_path / "measurements.parquet", kind="expression")


def test_snapshot_is_independent_of_mutable_source(tmp_path):
    source = tmp_path / "source" / "asset.txt"
    source.parent.mkdir()
    source.write_text("original source bytes")
    target = tmp_path / "package" / "asset.txt"
    _snapshot_file(source, target)
    source.write_text("later changed bytes")
    assert target.read_text() == "original source bytes"
    assert source.stat().st_ino != target.stat().st_ino


def test_catalog_names_distinct_numeric_grains_and_rejects_parent_path_escape(tmp_path):
    bulk = describe_evidence("bulk_numeric_measurement")
    expression = describe_evidence("line_expression_measurement")
    assert bulk["materialized_artifact"] != expression["materialized_artifact"]
    assert bulk["kind"] == expression["kind"] == "measurement"
    assert bulk["grain"] == "protein_source_numeric_cell"
    assert expression["grain"] == "gene_profile_source_numeric_cell"
    for evidence in (bulk, expression):
        assert "quantitative_value" in {field["name"] for field in evidence["fields"]}
    with pytest.raises(ValueError, match="manifest path"):
        verify_tcell_evidence_package(Path(tmp_path), manifest_path="../outside.json")
