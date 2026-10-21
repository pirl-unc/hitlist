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
from hitlist.tcell_functional import (
    iter_human_tcell_functional_assays,
    write_human_tcell_functional_assays,
)
from hitlist.tcell_training import (
    TRAINING_SCHEMA,
    verify_human_tcell_training_bundle,
    write_human_tcell_training_bundle,
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
        "Epitope | Source Organism IRI",
        "Epitope | Object Type",
        "Epitope | Modified residues",
        "Epitope | Modifications",
        "Assay Antigen | Epitope Relation",
        "Assay Antigen | Source Organism IRI",
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
            "Assay Antigen | Object Type": "Linear peptide",
            "Epitope | Source Organism IRI": "http://purl.obolibrary.org/obo/NCBITaxon_9606",
            "Epitope | Object Type": "Linear peptide",
            "Assay Antigen | Epitope Relation": "Epitope",
            "Assay Antigen | Source Organism IRI": "http://purl.obolibrary.org/obo/NCBITaxon_9606",
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


def test_functional_export_links_direct_human_peptide_to_outcome(tmp_path):
    peptide = {"Epitope | Name": "AAAAAAAAA", "Assay Antigen | Name": "AAAAAAAAA"}
    records = [
        ("1", {**peptide}),
        (
            "2",
            {
                **peptide,
                "Assay | Qualitative Measurement": "Negative",
                "MHC Restriction | Evidence Code": "MHC binding prediction",
            },
        ),
        (
            "3",
            {
                **peptide,
                "Assay | Method": "multimer/tetramer",
                "Assay | Response measured": "qualitative binding",
            },
        ),
        ("4", {**peptide, "Epitope | Source Organism IRI": "other_taxon"}),
        ("5", {**peptide, "Assay Antigen | Name": "CCCCCCCCC"}),
        ("6", {**peptide, "Epitope | Modified residues": "modified"}),
        ("7", {**peptide, "MHC Restriction | Name": "HLA class I"}),
        ("8", {**peptide, "Assay Antigen | Source Organism IRI": "other_taxon"}),
    ]
    paths = _sources(tmp_path, iedb=records, cedar=records[:2])
    rows = list(iter_human_tcell_functional_assays(**paths))
    assert [row["source_assay_id"] for row in rows] == ["1", "2", "7"]
    assert [row["response_observed"] for row in rows] == [True, False, True]
    assert [row["mhc_assignment_status"] for row in rows] == [
        "single_hla_experimental_restriction_evidence",
        "single_hla_reported_only",
        "no_single_hla_assignment",
    ]
    assert rows[0]["peptide_sequence"] == "AAAAAAAAA"
    assert rows[0]["reported_mhc_name"] == "HLA-A*02:01"
    assert rows[0]["source_reference_id"] == "10"
    assert rows[0]["assay_method"] == "ELISPOT"
    assert rows[0]["specimen_identity_status"] == "not_in_export"

    output = tmp_path / "functional"
    manifest = write_human_tcell_functional_assays(output, **paths, batch_size=2)
    assert manifest["n_assays"] == 3
    assert manifest["n_cedar_assays"] == 2
    assert manifest["n_assays_with_experimental_restriction_evidence"] == 1
    table = pq.read_table(output / "assays.parquet")
    assert table.column("response_observed").to_pylist() == [True, False, True]
    assert json.loads((output / "manifest.json").read_text()) == manifest


def test_functional_export_checks_peptide_link_parity_and_is_atomic(tmp_path):
    peptide = {"Epitope | Name": "AAAAAAAAA", "Assay Antigen | Name": "AAAAAAAAA"}
    paths = _sources(
        tmp_path,
        iedb=[("1", peptide)],
        cedar=[("1", {**peptide, "Assay Antigen | Name": "CCCCCCCCC"})],
    )
    output = tmp_path / "functional"
    with pytest.raises(ValueError, match="metadata differs"):
        write_human_tcell_functional_assays(output, **paths)
    assert not output.exists()


def test_training_bundle_preserves_evidence_and_routes_ambiguous_labels(tmp_path):
    peptide = {"Epitope | Name": "AAAAAAAAA", "Assay Antigen | Name": "AAAAAAAAA"}
    records = [
        ("1", {**peptide}),
        ("2", {**peptide, "Assay | Qualitative Measurement": "Negative"}),
        (
            "3",
            {
                **peptide,
                "Assay | Qualitative Measurement": "Positive",
                "Assay | Number of Subjects Positive": "0",
            },
        ),
        (
            "4",
            {
                **peptide,
                "Assay | Qualitative Measurement": "Negative",
                "Assay | Number of Subjects Positive": "0",
                "MHC Restriction | Evidence Code": "MHC binding prediction",
            },
        ),
        (
            "5",
            {**peptide, "Assay | Number of Subjects Positive": ""},
        ),
        (
            "6",
            {**peptide, "Assay | Number of Subjects Tested": "many"},
        ),
        (
            "7",
            {
                **peptide,
                "Assay | Number of Subjects Tested": "1",
                "Assay | Number of Subjects Positive": "2",
            },
        ),
    ]
    paths = _sources(tmp_path, iedb=records, cedar=records[:2])
    output = tmp_path / "bundle"
    manifest = write_human_tcell_training_bundle(output, **paths, batch_size=2)
    assert manifest["n_assays"] == 7
    assert manifest["n_training_examples"] == 3
    assert manifest["n_positive_examples"] == 2
    assert manifest["n_negative_examples"] == 1
    assert manifest["n_excluded_assays"] == 4
    assert manifest["exclusion_counts"] == {
        "invalid_reported_subject_count": 1,
        "positive_subjects_exceed_tested": 1,
        "qualitative_subject_count_disagreement": 2,
    }
    assert manifest["n_examples_with_mhc_allele"] == 2
    assert manifest["family_coverage"]["soluble_factor_release"] == {
        "n_examples": 3,
        "n_positive_examples": 2,
        "n_negative_examples": 1,
        "n_examples_with_mhc_allele": 2,
    }
    assert (
        verify_human_tcell_training_bundle(output, catalog_path=paths["catalog_path"]) == manifest
    )
    assert pq.read_table(output / "evidence" / "assays.parquet").num_rows == 7
    training = pq.read_table(output / "training.parquet")
    assert training.schema == TRAINING_SCHEMA
    assert training.column("source_assay_id").to_pylist() == ["1", "4", "5"]
    assert training.column("example_unit").to_pylist() == ["assay"] * 3
    assert training.column("antigen_format").to_pylist() == ["direct_linear_peptide"] * 3
    assert training.column("target_value").to_pylist() == [True, False, True]
    assert training.column("mhc_allele").to_pylist() == ["HLA-A*02:01", None, "HLA-A*02:01"]
    assert training.column("n_subjects_tested").to_pylist() == [2, 2, 2]
    assert training.column("study_group_id").to_pylist() == ["10", "10", "10"]
    excluded = pq.read_table(output / "excluded.parquet")
    assert excluded.column("source_assay_id").to_pylist() == ["2", "3", "6", "7"]
    with pytest.raises(FileExistsError):
        write_human_tcell_training_bundle(output, **paths)
    (output / "training.parquet").write_bytes((output / "training.parquet").read_bytes() + b"x")
    with pytest.raises(ValueError, match="file hash differs"):
        verify_human_tcell_training_bundle(output, catalog_path=paths["catalog_path"])


def test_training_bundle_is_atomic_on_shared_source_mismatch(tmp_path):
    peptide = {"Epitope | Name": "AAAAAAAAA", "Assay Antigen | Name": "AAAAAAAAA"}
    paths = _sources(
        tmp_path,
        iedb=[("1", peptide)],
        cedar=[("1", {**peptide, "Assay Antigen | Name": "CCCCCCCCC"})],
    )
    output = tmp_path / "bundle"
    with pytest.raises(ValueError, match="metadata differs"):
        write_human_tcell_training_bundle(output, **paths)
    assert not output.exists()
