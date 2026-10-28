"""Source-faithful MHC migration and portable package failure modes."""

import json
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from hitlist.evidence_catalog import FIELDS
from hitlist.mhc_evidence import (
    MHC_POSITIVE_MS_CORE_CONTRACT,
    NATIVE_FIELDS,
    _native_disposition,
    verify_mhc_evidence_package,
    write_mhc_evidence_package,
)
from hitlist.provenance import CONTRIBUTOR_COLUMNS, file_digest


def _assay_row(row_id, modality, outcome, *, source="iedb", **updates):
    row = dict.fromkeys(NATIVE_FIELDS, "")
    row.update(
        {
            "provenance_id": row_id,
            "source": source,
            "assay_iri": f"http://www.iedb.org/assay/{row_id}",
            "pmid": 12345678,
            "peptide": "PEPTIDE",
            "has_ptm": False,
            "is_non_peptide_ligand": False,
            "is_potential_contaminant": False,
            "assay_modality": modality,
            "assay_modality_source": "assay_method",
            "ms_outcome_status": outcome,
            "assay_method": "reported assay",
            "response_measured": "ligand presentation",
            "qualitative_measurement": "Positive" if outcome == "positive_result" else "Negative",
            "mhc_restriction": "HLA-A*02:01",
            "mhc_allele_set": "HLA-A*02:01 HLA-B*07:02",
            "restriction_evidence": "unknown",
            "quantitative_value": None,
        }
    )
    row.update(updates)
    return row


def _contributor(row_id, source_record_id):
    row = dict.fromkeys(CONTRIBUTOR_COLUMNS, "")
    row.update(
        {
            "provenance_id": row_id,
            "source_record_id": source_record_id,
            "source_dataset": "iedb",
            "source_row": 1,
            "original_fields": '{"peptide": "PEPTIDE"}',
            "source_row_values": '["PEPTIDE"]',
            "relationships": '["retained"]',
            "relationship_status": "source_record",
        }
    )
    return row


def _source(
    tmp_path,
    *,
    duplicate_id=False,
    orphan_link=False,
    extra_field=False,
    blank_source_record=False,
    unregistered_source=False,
    missing_payload=False,
):
    source = tmp_path / "source"
    source.mkdir()
    ms = [
        _assay_row("ms-1", "ms", "positive_result"),
        _assay_row(
            "ms-2",
            "ms",
            "curated_ligand_table",
            source="supplement",
            supplementary_file="paper.csv",
            assay_method="",
            response_measured="",
            qualitative_measurement="",
            assay_modality_source="curated_ms_supplement",
            is_potential_contaminant=True,
        ),
    ]
    binding = [
        _assay_row(
            "binding-1",
            "binding",
            "negative",
            response_measured="half maximal inhibitory concentration (IC50)",
            quantitative_measurement="500",
            quantitative_value=500.0,
            measurement_units="nM",
            measurement_inequality=">",
            restriction_evidence="experimental",
        )
    ]
    other = [
        _assay_row("ms-1" if duplicate_id else "other-1", "ms", "negative"),
        _assay_row(
            "other-2",
            "structural",
            "positive_result",
            response_measured="3D structure",
            quantitative_measurement="2",
            quantitative_value=2.0,
            measurement_units="angstroms",
        ),
    ]
    for name, rows in (
        ("observations.parquet", ms),
        ("binding.parquet", binding),
        ("other_assays.parquet", other),
    ):
        if extra_field and name == "binding.parquet":
            rows[0]["unreviewed_source_column"] = "value"
        pq.write_table(pa.Table.from_pylist(rows), source / name)
    contributors = [
        _contributor(row["provenance_id"], f"record-{index}")
        for index, row in enumerate(ms + binding + other)
    ]
    contributors.append(_contributor("missing" if orphan_link else "ms-1", "cedar-copy"))
    if blank_source_record:
        contributors[0]["source_record_id"] = ""
    if unregistered_source:
        contributors[0]["source_dataset"] = "unregistered"
    if missing_payload:
        contributors[0]["source_row_values"] = ""
    pq.write_table(pa.Table.from_pylist(contributors), source / "observation_contributors.parquet")
    files = {
        name: file_digest(source / name)
        for name in (
            "observations.parquet",
            "binding.parquet",
            "other_assays.parquet",
            "observation_contributors.parquet",
        )
    }
    (source / "observations_meta.json").write_text(
        json.dumps(
            {
                "ci_corpus_manifest_version": 2,
                "artifact_version": 10,
                "files": files,
                "provenance": {
                    "sources": {
                        name: {"sha256": "source-snapshot", "locator_basis": "logical_row"}
                        for name in ("iedb", "cedar", "supplement:paper.csv")
                    }
                },
            }
        )
    )
    return source


def test_mhc_package_preserves_every_result_source_and_endpoint(tmp_path):
    source = _source(tmp_path)
    target = tmp_path / "package"
    write_mhc_evidence_package(source, target)
    manifest = verify_mhc_evidence_package(target)
    assert manifest["reconciliation"]["by_evidence_type"] == {
        "mhc_positive_ms": 2,
        "mhc_binding": 1,
        "mhc_other": 2,
    }
    assert manifest["reconciliation"]["n_contributor_links"] == 6
    assert len(manifest["native_field_disposition"]) == len(NATIVE_FIELDS) == 71
    assert manifest["source_license_status"] == "unverified"
    assert manifest["positive_ms_core_contract"] == MHC_POSITIVE_MS_CORE_CONTRACT
    assert set(manifest["source_registry"]) == {"iedb", "cedar", "supplement:paper.csv"}
    for name in manifest["files"]:
        if name.startswith("native/"):
            assert file_digest(target / name) == file_digest(source / name.removeprefix("native/"))
    observations = pq.read_table(target / "typed" / "mhc_observations.parquet").to_pandas()
    measurements = pq.read_table(target / "typed" / "mhc_measurements.parquet").to_pandas()
    assert set(observations.observation_id) == set(measurements.observation_id)
    assert set(observations.loc[observations.evidence_type == "mhc_other", "observation_id"]) == {
        "other-1",
        "other-2",
    }
    assert observations.set_index("observation_id").at["binding-1", "tested_mhc"] == "HLA-A*02:01"
    assert observations.set_index("observation_id").at["binding-1", "source_pmid"] == "12345678"
    assert observations.set_index("observation_id").at["ms-2", "is_potential_contaminant"]
    assert observations.peptide_mhc_restriction.isna().all()
    assert (
        measurements.set_index("observation_id").at["ms-2", "endpoint"] == "positive_mhc_ligand_ms"
    )
    assert measurements.set_index("observation_id").at["ms-2", "qualitative_status"] == "unreported"
    assert measurements.set_index("observation_id").at["binding-1", "measurement_inequality"] == ">"
    assert measurements.set_index("observation_id").at["other-2", "endpoint"] == "3D structure"


def test_typed_field_disposition_resolves_in_the_catalog():
    declared = {field.name for field in FIELDS}
    mapped = {
        name
        for disposition in _native_disposition().values()
        for name in disposition.get("logical_fields", ())
    }
    assert mapped <= declared


@pytest.mark.parametrize(
    "variation, message",
    [
        ({"duplicate_id": True}, "Duplicate MHC observation ID"),
        ({"orphan_link": True}, "contributor coverage mismatch"),
        ({"extra_field": True}, "Unreviewed MHC source schema"),
        ({"blank_source_record": True}, "contributor lacks source_record_id"),
        ({"unregistered_source": True}, "unregistered source dataset"),
        ({"missing_payload": True}, "contributor lacks source_row_values"),
    ],
)
def test_mhc_package_refuses_loss_or_unreviewed_columns(tmp_path, variation, message):
    source = _source(tmp_path, **variation)
    target = tmp_path / "package"
    with pytest.raises(ValueError, match=message):
        write_mhc_evidence_package(source, target)
    assert not target.exists()


def test_mhc_package_verifier_detects_artifact_change(tmp_path):
    source = _source(tmp_path)
    target = tmp_path / "package"
    write_mhc_evidence_package(source, target)
    with (target / "typed" / "mhc_measurements.parquet").open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ValueError, match="artifact mismatch"):
        verify_mhc_evidence_package(target)


def test_mhc_package_survives_mutated_ingestion_file(tmp_path):
    source = _source(tmp_path)
    target = tmp_path / "package"
    write_mhc_evidence_package(source, target)
    original = (target / "native" / "observations.parquet").read_bytes()

    with (source / "observations.parquet").open("ab") as stream:
        stream.write(b"later source edit")

    assert (target / "native" / "observations.parquet").read_bytes() == original
    assert (source / "observations.parquet").stat().st_ino != (
        target / "native" / "observations.parquet"
    ).stat().st_ino
    verify_mhc_evidence_package(target)


def test_mhc_package_manifest_remains_verifiable_inside_later_generation(tmp_path):
    source = _source(tmp_path)
    target = tmp_path / "package"
    write_mhc_evidence_package(source, target)
    (target / "manifests").mkdir()
    shutil.copy2(target / "manifest.json", target / "manifests" / "mhc.json")
    (target / "manifest.json").unlink()
    manifest = verify_mhc_evidence_package(target, manifest_path="manifests/mhc.json")
    assert manifest["catalog_schema_version"] == 3
    with pytest.raises(ValueError, match="Invalid MHC package manifest path"):
        verify_mhc_evidence_package(target, manifest_path="../outside.json")


@pytest.mark.parametrize(
    "updates, message",
    [
        ({"assay_modality_source": ""}, "core observation field: assay_modality_basis"),
        ({"source": ""}, "core observation field: source_dataset"),
        ({"assay_iri": ""}, "Missing MHC result identity"),
        ({"response_measured": ""}, "core measurement field: endpoint"),
        ({"assay_modality": "binding"}, "MHC family mismatch"),
        ({"ms_outcome_status": "negative"}, "MHC family mismatch"),
        ({"quantitative_value": 5.0}, "reported quantity lacks raw or parsed value"),
        ({"measurement_units": "nM"}, "unit_status without a quantity"),
        ({"measurement_inequality": ">"}, "inequality_status without a quantity"),
    ],
)
def test_positive_ms_core_contract_rejects_invalid_conditional_fields(tmp_path, updates, message):
    source = _source(tmp_path)
    table = pq.read_table(source / "observations.parquet").to_pylist()
    table[0].update(updates)
    pq.write_table(pa.Table.from_pylist(table), source / "observations.parquet")
    files = json.loads((source / "observations_meta.json").read_text())["files"]
    files["observations.parquet"] = file_digest(source / "observations.parquet")
    metadata = json.loads((source / "observations_meta.json").read_text())
    metadata["files"] = files
    (source / "observations_meta.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match=message):
        write_mhc_evidence_package(source, tmp_path / "package")
    assert not (tmp_path / "package").exists()
    assert not list(tmp_path.glob(".package-*"))


@pytest.mark.parametrize(
    "raw,value,unit,inequality,status,unit_status,inequality_status",
    [
        ("", None, "", "", "unreported", "not_applicable", "not_applicable"),
        ("0", 0.0, "", "", "reported", "unreported", "unreported"),
        ("5", 5.0, "nM", ">", "reported", "reported", "reported"),
        ("not quantified", None, "", "", "unresolved", "unreported", "unreported"),
        ("below detection", None, "nM", "<", "unresolved", "reported", "reported"),
    ],
)
@pytest.mark.parametrize("qualitative", ["", "Positive"])
def test_positive_ms_core_preserves_conditional_measurements(
    tmp_path, raw, value, unit, inequality, status, unit_status, inequality_status, qualitative
):
    source = _source(tmp_path)
    rows = pq.read_table(source / "observations.parquet").to_pylist()
    rows[0].update(
        quantitative_measurement=raw,
        quantitative_value=value,
        measurement_units=unit,
        measurement_inequality=inequality,
        qualitative_measurement=qualitative,
    )
    pq.write_table(pa.Table.from_pylist(rows), source / "observations.parquet")
    metadata_path = source / "observations_meta.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["files"]["observations.parquet"] = file_digest(source / "observations.parquet")
    metadata_path.write_text(json.dumps(metadata))
    target = tmp_path / "package"
    write_mhc_evidence_package(source, target)
    manifest = verify_mhc_evidence_package(target)
    assert manifest["reconciliation"]["n_observations"] == 5
    measurements = pq.read_table(target / "typed/mhc_measurements.parquet").to_pylist()
    result = next(row for row in measurements if row["observation_id"] == "ms-1")
    assert result["quantitative_raw"] == (raw or None)
    assert result["quantitative_value"] == value
    assert result["quantitative_status"] == status
    assert result["measurement_unit"] == (unit or None)
    assert result["unit_status"] == unit_status
    assert result["measurement_inequality"] == (inequality or None)
    assert result["inequality_status"] == inequality_status
    assert result["qualitative_result"] == (qualitative or None)
    assert result["qualitative_status"] == ("reported" if qualitative else "unreported")
    assert (
        file_digest(target / "native/observations.parquet")
        == metadata["files"]["observations.parquet"]
    )


@pytest.mark.parametrize("embedded", [False, True])
def test_historical_mhc_package_remains_verifiable(tmp_path, embedded):
    target = tmp_path / "package"
    shutil.copytree(Path(__file__).parent / "data/mhc_schema1", target)
    manifest_path = "manifest.json"
    if embedded:
        (target / "manifests").mkdir()
        (target / "manifest.json").rename(target / "manifests/mhc.json")
        manifest_path = "manifests/mhc.json"
    manifest = verify_mhc_evidence_package(target, manifest_path=manifest_path)
    assert manifest["schema_version"] == 1
    assert manifest["catalog_schema_version"] == 2
    assert "positive_ms_core_contract" not in manifest
    with (target / "typed/mhc_measurements.parquet").open("ab") as stream:
        stream.write(b"corrupt")
    with pytest.raises(ValueError, match="artifact mismatch"):
        verify_mhc_evidence_package(target, manifest_path=manifest_path)


@pytest.mark.parametrize(
    "updates,message",
    [
        ({"schema_version": 99}, "Unsupported MHC evidence package schema"),
        ({"catalog_schema_version": 2}, "catalog schema mismatch"),
        ({"positive_ms_core_contract": None}, "core contract mismatch"),
        ({"positive_ms_core_contract": {}}, "core contract mismatch"),
    ],
)
def test_new_mhc_package_requires_versioned_core_contract(tmp_path, updates, message):
    source = _source(tmp_path)
    target = tmp_path / "package"
    write_mhc_evidence_package(source, target)
    path = target / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest.update(updates)
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=message):
        verify_mhc_evidence_package(target)
