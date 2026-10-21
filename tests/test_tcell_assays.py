"""Row-level T-cell metadata stays source-faithful and independent of MS labels."""

import csv
import hashlib
import io
import json
import zipfile

import pyarrow.parquet as pq
import pytest
import yaml

from hitlist.tcell_assays import (
    CEDAR_NAME,
    IEDB_NAME,
    SOURCE_FIELDS,
    iter_tcell_assay_metadata,
    write_tcell_assay_metadata,
)


def _zip(path, records, *, cedar=False, omit=None):
    source = "CEDAR" if cedar else "IEDB"
    names = [
        f"Assay ID | {source} IRI",
        f"Reference | {source} IRI",
        *SOURCE_FIELDS.values(),
        "MHC Restriction | Name",
        "Epitope | Name",
        "Assay Antigen | Name",
    ]
    names = list(dict.fromkeys(name for name in names if name != omit))
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow([name.split(" | ")[0] for name in names])
    writer.writerow([name.split(" | ")[1] for name in names])
    for assay_id, overrides in records:
        row = {
            f"Assay ID | {source} IRI": f"https://example.org/assay/{assay_id}",
            f"Reference | {source} IRI": "https://example.org/reference/10",
            "Reference | PMID": "12345",
            "Assay | Method": "ELISPOT",
            "Assay | Response measured": "IFNg release",
            "Assay | Qualitative Measurement": "Positive-Low",
            "Assay | Quantitative measurement": "<2",
            "Assay | Measurement Inequality": "<",
            "Assay | Units": "spots/100000 cells",
            "Assay | Number of Subjects Tested": "2",
            "Assay | Number of Subjects Positive": "1",
            "MHC Restriction | Name": "HLA-A*02:01",
            "MHC Restriction | Evidence Code": "Single allele present",
            "Epitope | Name": "SYNTHETIC_SEQUENCE",
            "Assay Antigen | Name": "SYNTHETIC_SEQUENCE",
        }
        row.update(overrides)
        writer.writerow([row.get(name, "") for name in names])
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("tcell_full_v3.csv", output.getvalue())


def _sources(tmp_path, *, iedb=None, cedar=None, omit=None):
    iedb_path = tmp_path / IEDB_NAME
    cedar_path = tmp_path / CEDAR_NAME
    if iedb is None:
        iedb = [
            ("1", {}),
            ("2", {"Assay | Qualitative Measurement": "Negative", "MHC Restriction | Name": ""}),
            ("3", {"Assay | Method": "multimer/tetramer", "Assay | Response measured": "binding"}),
        ]
    if cedar is None:
        cedar = iedb[:2]
    _zip(iedb_path, iedb, omit=omit)
    _zip(cedar_path, cedar, cedar=True, omit=omit)
    catalog = tmp_path / "catalog.yaml"
    catalog.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "observed": "2026-10-20",
                "files": {
                    name: {
                        "url": f"https://example.org/{name}",
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "size": path.stat().st_size,
                    }
                    for name, path in ((IEDB_NAME, iedb_path), (CEDAR_NAME, cedar_path))
                },
            }
        )
    )
    return {"iedb_path": iedb_path, "cedar_path": cedar_path, "catalog_path": catalog}


def test_assays_are_deduplicated_and_keep_reported_values(tmp_path):
    paths = _sources(tmp_path)
    rows = list(iter_tcell_assay_metadata(**paths))
    assert [row["source_assay_id"] for row in rows] == ["1", "2", "3"]
    assert [row["cedar_present"] for row in rows] == [True, True, False]
    assert rows[0]["source_reference_id"] == "10"
    assert rows[0]["qualitative_measurement"] == "Positive-Low"
    assert rows[0]["quantitative_measurement"] == "<2"
    assert rows[0]["measurement_inequality"] == "<"
    assert rows[0]["reported_n_subjects_tested"] == "2"
    assert rows[0]["assay_family"] == "soluble_factor_release"
    assert rows[1]["restriction_name_resolution"] == "missing"
    assert rows[2]["assay_family"] == "pmhc_multimer"
    assert rows[0]["specimen_identity_status"] == "not_in_export"
    assert rows[0]["receptor_identity_status"] == "not_in_export"
    assert "SYNTHETIC_SEQUENCE" not in str(rows)
    assert "HLA-A*02:01" not in str(rows)


def test_hash_checked_before_any_row(tmp_path):
    paths = _sources(tmp_path)
    paths["cedar_path"].write_bytes(paths["cedar_path"].read_bytes() + b"changed")
    with pytest.raises(ValueError, match="Source size differs"):
        next(iter_tcell_assay_metadata(**paths))


def test_missing_source_schema_rejected(tmp_path):
    paths = _sources(tmp_path, omit="Assay | Qualitative Measurement")
    with pytest.raises(ValueError, match="Missing T-cell columns"):
        list(iter_tcell_assay_metadata(**paths))


def test_shared_id_must_match_source_content(tmp_path):
    paths = _sources(
        tmp_path,
        cedar=[("1", {"Assay | Qualitative Measurement": "Negative"})],
    )
    with pytest.raises(ValueError, match="metadata differs"):
        list(iter_tcell_assay_metadata(**paths))


def test_cedar_only_id_rejected_without_publishing(tmp_path):
    paths = _sources(tmp_path, cedar=[("4", {})])
    output = tmp_path / "export"
    with pytest.raises(ValueError, match="CEDAR assay IDs absent"):
        write_tcell_assay_metadata(output, **paths, batch_size=1)
    assert not output.exists()


def test_local_export_has_manifest_and_exact_rows(tmp_path):
    paths = _sources(tmp_path)
    output = tmp_path / "export"
    manifest = write_tcell_assay_metadata(output, **paths, batch_size=2)
    assert manifest["n_assays"] == 3
    assert manifest["n_cedar_assays"] == 2
    assert manifest["schema_version"] == 1
    assert json.loads((output / "manifest.json").read_text()) == manifest
    assert (
        hashlib.sha256((output / "assays.parquet").read_bytes()).hexdigest()
        == manifest["parquet_sha256"]
    )
    table = pq.read_table(output / "assays.parquet")
    assert table.num_rows == 3
    assert table.column("qualitative_measurement").to_pylist() == [
        "Positive-Low",
        "Negative",
        "Positive-Low",
    ]
    with pytest.raises(FileExistsError):
        write_tcell_assay_metadata(output, **paths)
