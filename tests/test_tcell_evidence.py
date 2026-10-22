"""Offline T-cell source-grain and target-separation fixtures."""

import csv
import io
import json
import zipfile

import pyarrow.parquet as pq
import pytest

from hitlist.tcell_assays import SOURCE_FIELDS
from hitlist.tcell_evidence import (
    _ANTIGEN_FIELDS,
    EXPECTED,
    _write_tcell_rows,
)
from hitlist.tcell_peptides import HUMAN_TAXON_IRI


def _source_fixture(tmp_path, *, alter_cedar=False):
    assets = tmp_path / "assets"
    typed = tmp_path / "typed"
    assets.mkdir()
    typed.mkdir()
    source_iris = {
        "IEDB": (
            "Assay ID | IEDB IRI",
            "Reference | IEDB IRI",
            "Epitope | IEDB IRI",
        ),
        "CEDAR": (
            "Assay ID | CEDAR IRI",
            "Reference | CEDAR IRI",
            "Epitope | CEDAR IRI",
        ),
    }
    common = set(SOURCE_FIELDS.values()) | (set(_ANTIGEN_FIELDS) - {"Epitope | IEDB IRI"})
    common.update(
        {
            "Assay | Method",
            "Assay | Response measured",
            "Assay | Qualitative Measurement",
            "Assay | Quantitative measurement",
            "Assay | Units",
            "Assay | Measurement Inequality",
            "Assay | Number of Subjects Tested",
            "Assay | Number of Subjects Positive",
        }
    )
    common.update({f"Unused {index} | Raw" for index in range(158 - len(common))})
    assert len(common) == 158
    shared = sorted(common)
    base = {
        "Reference | PMID": "1234",
        "Epitope | Object Type": "Linear peptide",
        "Epitope | Name": "PEPTIDE",
        "Epitope | Source Organism IRI": HUMAN_TAXON_IRI,
        "Assay Antigen | Object Type": "Linear peptide",
        "Assay Antigen | Epitope Relation": "Epitope",
        "Assay Antigen | Name": "PEPTIDE",
        "Assay Antigen | Source Organism IRI": HUMAN_TAXON_IRI,
        "Assay | Method": "ELISPOT",
        "Assay | Response measured": "IFNg release",
        "Assay | Qualitative Measurement": "Negative",
        "Assay | Number of Subjects Tested": "1",
        "Assay | Number of Subjects Positive": "0",
    }
    rows = []
    for assay_id in ("1", "2", "3", "4"):
        row = {**base}
        if assay_id == "2":
            row.update(
                {
                    "Assay | Method": "multimer/tetramer",
                    "Assay | Response measured": "qualitative binding",
                    "Assay | Qualitative Measurement": "Positive",
                    "Assay | Number of Subjects Positive": "1",
                }
            )
        if assay_id == "3":
            row["Assay Antigen | Object Type"] = "Peptide pool"
        if assay_id == "4":
            row["Assay | Qualitative Measurement"] = "Positive"
        rows.append((assay_id, row))

    for source, selected in (("IEDB", rows), ("CEDAR", rows[:1] + rows[3:])):
        headers = [*source_iris[source], *shared]
        if source == "CEDAR":
            headers.append("Epitope | Mutation")
        stream = io.StringIO(newline="")
        writer = csv.writer(stream)
        writer.writerow([field.split(" | ", 1)[0] for field in headers])
        writer.writerow([field.split(" | ", 1)[1] for field in headers])
        for assay_id, values in selected:
            if source == "CEDAR" and alter_cedar and assay_id == "1":
                values = {**values, "Assay | Qualitative Measurement": "Positive"}
            iri = (
                f"https://{source.lower()}.example/assay/{assay_id}",
                f"https://{source.lower()}.example/reference/1234",
                f"https://{source.lower()}.example/epitope/peptide",
            )
            cells = [*iri, *(values.get(header, "") for header in shared)]
            if source == "CEDAR":
                cells.append("P123L" if assay_id == "1" else "")
            writer.writerow(cells)
        path = assets / f"{source.lower()}_tcell_full_v3.zip"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("tcell_full_v3.csv", stream.getvalue())
    return assets, typed


def test_tcell_assays_keep_source_claims_and_separate_targets(tmp_path, monkeypatch):
    assets, typed = _source_fixture(tmp_path)
    monkeypatch.setitem(EXPECTED, "n_cedar_contributors", 2)
    counts = _write_tcell_rows(assets, typed)
    assert counts["n_iedb_assays"] == 4
    assert counts["n_cedar_contributors"] == 2
    assert counts["n_cedar_mutation_claims"] == 1
    assert counts["n_cedar_mutation_on_training_examples"] == 1
    assert counts["n_direct_peptide_candidates"] == 3
    assert counts["n_training_examples"] == 2
    assert counts["n_training_exclusions"] == 1
    training = pq.read_table(typed / "training_examples.parquet").to_pylist()
    assert {row["task"] for row in training} == {
        "assay_specific_tcell_qualitative_response",
        "assay_specific_pmhc_multimer_cell_binding",
    }
    assert {row["source_assay_id"]: row["target_value"] for row in training} == {
        "1": False,
        "2": True,
    }
    assert all(row["mhc_allele"] is None for row in training)
    eligibility = {
        row["source_assay_id"]: row
        for row in pq.read_table(typed / "eligibility.parquet").to_pylist()
    }
    assert eligibility["3"]["first_failure_reason"] == "antigen_not_linear_peptide"
    assert eligibility["3"]["training_status"] == "not_candidate"
    assert eligibility["4"]["training_exclusion_reason"] == (
        "qualitative_subject_count_disagreement"
    )
    mutation = pq.read_table(typed / "cedar_mutation_claims.parquet").to_pylist()
    assert mutation[0]["value_raw"] == "P123L"
    contributors = pq.read_table(typed / "source_contributors.parquet").to_pylist()
    assert {row["source_dataset"] for row in contributors} == {"IEDB", "CEDAR"}
    assert all(row["epitope_iri_raw"].endswith("/epitope/peptide") for row in contributors)
    field_inventory = json.loads((tmp_path / "source_field_disposition.json").read_text())
    assert {name: len(rows) for name, rows in field_inventory.items()} == {
        "IEDB": 161,
        "CEDAR": 162,
    }


def test_tcell_shared_source_disagreement_fails(tmp_path, monkeypatch):
    assets, typed = _source_fixture(tmp_path, alter_cedar=True)
    monkeypatch.setitem(EXPECTED, "n_cedar_contributors", 2)
    with pytest.raises(ValueError, match="full-field mismatch"):
        _write_tcell_rows(assets, typed)
