"""The availability audit counts source assays without manufacturing target labels."""

import csv
import hashlib
import io
import zipfile

import pytest
from scripts.audit_tcell_availability import audit_zip, family, run


def fixture_zip(path, *, cedar=False, duplicate=False):
    names = [
        ("Assay ID", "CEDAR IRI" if cedar else "IEDB IRI"),
        ("Reference", "CEDAR IRI" if cedar else "IEDB IRI"),
        ("Assay", "Method"),
        ("Assay", "Response measured"),
        ("Assay", "Qualitative Measurement"),
        ("MHC Restriction", "Name"),
        ("MHC Restriction", "Evidence Code"),
    ]
    from scripts.audit_tcell_availability import CONTEXT_FIELDS

    names.extend(
        tuple(field.split(" | "))
        for field in CONTEXT_FIELDS
        if field not in {"MHC Restriction | Name", "MHC Restriction | Evidence Code"}
    )
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer)
    writer.writerow([name[0] for name in names])
    writer.writerow([name[1] for name in names])
    keys = [" | ".join(name) for name in names]

    def add(assay_id, method, response, outcome, mhc, evidence):
        values = {
            keys[0]: f"https://example.org/assay/{assay_id}",
            keys[1]: "https://example.org/reference/10",
            "Assay | Method": method,
            "Assay | Response measured": response,
            "Assay | Qualitative Measurement": outcome,
            "MHC Restriction | Name": mhc,
            "MHC Restriction | Evidence Code": evidence,
        }
        writer.writerow([values.get(key, "") for key in keys])

    add(
        1,
        "multimer/tetramer",
        "qualitative binding",
        "Positive",
        "HLA-A*02:01",
        "Single allele present",
    )
    add(2, "ELISPOT", "IFNg release", "Negative", "", "")
    add(3, "in vitro assay", "proteasome cleavage", "", "HLA class I", "Not determined")
    if duplicate:
        add(3, "in vitro assay", "proteasome cleavage", "", "HLA class I", "Not determined")
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("tcell_full_v3.csv", buffer.getvalue())


def test_counts_assays_and_preserves_unknowns(tmp_path):
    source = tmp_path / "test.zip"
    fixture_zip(source, cedar=True)
    summary, keys = audit_zip(source)
    assert keys == {"1", "2", "3"}
    assert summary["counts"]["n_assays"] == 3
    assert summary["counts"]["n_studies"] == 1
    assert summary["counts"]["n_processing_keyword_candidate_assays"] == 1
    assert summary["outcomes"] == {"<missing>": 1, "Negative": 1, "Positive": 1}
    assert summary["families"] == {
        "other_tcell_assay": 1,
        "pmhc_multimer": 1,
        "soluble_factor_release": 1,
    }
    assert summary["restriction_name_resolution"] == {
        "missing": 1,
        "other_reported_name": 1,
        "single_hla_two_field_name": 1,
    }
    assert summary["n_assays_with_context_field"].get("Host | MHC Present", 0) == 0


def test_duplicate_assay_id_rejected(tmp_path):
    source = tmp_path / "test.zip"
    fixture_zip(source, duplicate=True)
    with pytest.raises(ValueError, match="Repeated assay ID"):
        audit_zip(source)


def test_target_routing_does_not_conflate_multimer_and_function():
    assert family("multimer/tetramer", "qualitative binding") == "pmhc_multimer"
    assert family("ELISPOT", "IFNg release") == "soluble_factor_release"
    assert family("51 chromium", "cytotoxicity") == "cytotoxicity"
    assert family("in vitro assay", "proteasome cleavage") == "other_tcell_assay"


def test_union_deduplicates_shared_iedb_and_cedar_assay_ids(tmp_path):
    iedb = tmp_path / "iedb.zip"
    cedar = tmp_path / "cedar.zip"
    fixture_zip(iedb)
    fixture_zip(cedar, cedar=True)
    catalog = tmp_path / "catalog.yaml"
    catalog.write_text(
        "schema_version: 1\nobserved: '2026-10-20'\nfiles:\n"
        + "".join(
            f"  {name}:\n    url: https://example.org/{name}\n"
            f"    sha256: {hashlib.sha256(path.read_bytes()).hexdigest()}\n"
            f"    size: {path.stat().st_size}\n"
            for name, path in (
                ("iedb_tcell_full_v3.zip", iedb),
                ("cedar_tcell_full_v3.zip", cedar),
            )
        )
    )
    result = run({"iedb_tcell_full_v3.zip": iedb, "cedar_tcell_full_v3.zip": cedar}, catalog)
    assert result["cross_source"] == {
        "n_cedar_only_assay_ids": 0,
        "n_iedb_only_assay_ids": 0,
        "n_shared_assay_ids": 3,
        "n_union_assay_ids": 3,
    }
