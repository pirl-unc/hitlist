"""Independently replay source-reviewed binding claims without importing Hitlist."""

from __future__ import annotations

import argparse
import hashlib
import json
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import pyarrow.dataset as ds
import pyarrow.parquet as pq

SOURCE_COLUMNS = (
    "provenance_id",
    "assay_iri",
    "pmid",
    "source",
    "response_measured",
    "assay_method",
    "quantitative_measurement",
    "measurement_units",
    "measurement_inequality",
)
CONTEXT_COLUMNS = (
    "observation_id",
    "source_assay_id",
    "source_pmid",
    "source_dataset",
    "endpoint_reported",
    "assay_method",
    "quantitative_raw",
    "measurement_unit",
    "measurement_inequality",
)
V4_SHA256 = "2318bc53cdae228285218991d58fbcc2bd7266327cec1dd3ed9a03b1c944a5b7"
COHORT_SHA256 = "b180eba804532f633a6d61bb350584dbe24f8df4b4b96f02c6ec492c60144372"


def digest(path):
    sha = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            sha.update(chunk)
    return {"sha256": sha.hexdigest(), "size_bytes": Path(path).stat().st_size}


def replay(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    parent_path = directory / "manifests/binding_review_parent.json"
    parent = json.loads(parent_path.read_text())
    assert (manifest["schema_version"], manifest["catalog_schema_version"]) == (10, 15)
    assert (parent["schema_version"], parent["catalog_schema_version"]) == (9, 14)
    assert manifest["parent_manifest_sha256"] == digest(parent_path)["sha256"]
    for name, recorded in parent["files"].items():
        assert manifest["files"][name] == recorded, name
    actual_files = {
        str(path.relative_to(directory))
        for path in directory.rglob("*")
        if path.is_file() and path != directory / "manifest.json"
    }
    assert actual_files == set(manifest["files"])
    for name, recorded in manifest["files"].items():
        assert digest(directory / name) == recorded, name

    mhc = json.loads((directory / "manifests/mhc.json").read_text())
    assert mhc["source_manifest"]["sha256"] == V4_SHA256
    assert digest(directory / "cohort_native/manifest.json")["sha256"] == COHORT_SHA256
    ledger = json.loads((directory / "binding_review/review.json").read_text())
    source_path = directory / "binding_review/source.xml"
    assert digest(source_path)["sha256"] == ledger["source"]["source_xml_sha256"]
    section = (
        ET.parse(source_path).getroot().find(f".//sec[@id='{ledger['source']['section_id']}']")
    )
    assert section is not None
    assert "".join(section.find("title").itertext()) == ledger["source"]["section_title"]
    source_text = " ".join("".join(section.itertext()).split())
    claims = pq.read_table(directory / "binding_review/binding_protocol_claims.parquet").to_pylist()
    assert len(claims) == len(ledger["claims"]) == 21
    for claim, source_claim in zip(claims, ledger["claims"]):
        assert claim["protocol_id"] == ledger["protocol_id"]
        assert claim["source_asset_sha256"] == ledger["source"]["source_xml_sha256"]
        assert claim["assertion_predicate"] == source_claim["predicate"]
        assert claim["assertion_value"] == source_claim["value"]
        assert claim["claim_status"] == source_claim["status"]
        assert claim["claim_applicability"] == source_claim["applicability"]
        assert claim["source_excerpt"] == source_claim["source_excerpt"]
        if claim["source_excerpt"]:
            assert claim["source_excerpt"] in source_text
    assert not any(claim["assertion_predicate"] == "KD" for claim in claims)

    binding = pq.read_table(
        directory / "native/binding.parquet", columns=list(SOURCE_COLUMNS)
    ).to_pylist()
    context = pq.read_table(directory / "binding_review/binding_assay_context.parquet").to_pylist()
    assert len(binding) == len(context) == 814_268
    source_by_id = {row["provenance_id"]: row for row in binding}
    assert len(source_by_id) == len(binding)
    context_by_id = {row["observation_id"]: row for row in context}
    assert len(context_by_id) == len(context)
    assert set(context_by_id) == set(source_by_id)
    expected_fields = (
        "provenance_id",
        "assay_iri",
        "pmid",
        "source",
        "response_measured",
        "assay_method",
        "quantitative_measurement",
        "measurement_units",
        "measurement_inequality",
    )
    reviewed_ids = set()
    endpoint_counts = Counter()
    method_counts = Counter()
    for observation_id, row in source_by_id.items():
        new = context_by_id[observation_id]
        values = tuple(row[name] for name in expected_fields)
        expected = (
            observation_id,
            row["assay_iri"],
            str(row["pmid"]) if row["pmid"] is not None else None,
            row["source"],
            row["response_measured"] or None,
            row["assay_method"] or None,
            row["quantitative_measurement"] or None,
            row["measurement_units"] or None,
            row["measurement_inequality"] or None,
        )
        assert values[0] == observation_id
        assert tuple(new[name] for name in CONTEXT_COLUMNS) == expected, observation_id
        endpoint_counts[row["response_measured"] or "<null>"] += 1
        method_counts[row["assay_method"] or "<null>"] += 1
        if row["pmid"] == ledger["source"]["pmid"]:
            reviewed_ids.add(observation_id)
            assert new["review_status"] == "source_reviewed_study_scope"
            assert new["protocol_id"] == ledger["protocol_id"]
            assert row["source"] == ledger["required_source"]
            assert row["response_measured"] == ledger["required_endpoint"]
            assert row["assay_method"] == ledger["required_method"]
            assert row["measurement_units"] == ledger["required_unit"]
        else:
            assert new["review_status"] == "not_source_reviewed"
            assert new["protocol_id"] is None
    assert len(reviewed_ids) == 938

    cohort = ds.dataset(directory / "cohort_native/training.parquet", format="parquet")
    cohort_rows = cohort.to_table(
        columns=[*SOURCE_COLUMNS, "is_binding_assay", "assay_comments"],
        filter=ds.field("is_binding_assay"),
    ).to_pylist()
    cohort_ids = {row["provenance_id"] for row in cohort_rows}
    assert len(cohort_ids) == len(cohort_rows) == 25_896
    assert cohort_ids <= set(source_by_id)
    for row in cohort_rows:
        assert all(row[name] == source_by_id[row["provenance_id"]][name] for name in SOURCE_COLUMNS)
    cohort_reviewed = reviewed_ids & cohort_ids
    ic50_rows = [
        row for row in cohort_rows if row["response_measured"] == ledger["required_endpoint"]
    ]
    n_ic50_rows = len(ic50_rows)
    ic50_by_pmid = Counter(
        str(row["pmid"]) if row["pmid"] is not None else "<null>" for row in ic50_rows
    )
    ic50_by_source = Counter(row["source"] for row in ic50_rows)
    assert len(cohort_reviewed) == 910
    assert n_ic50_rows == 8_084
    report = {
        "package_manifest_sha256": digest(directory / "manifest.json")["sha256"],
        "parent_manifest_sha256": digest(parent_path)["sha256"],
        "source_xml_sha256": digest(source_path)["sha256"],
        "n_inherited_files": len(parent["files"]),
        "n_package_files": len(manifest["files"]),
        "n_binding_assay_rows": len(binding),
        "n_reviewed_binding_rows": len(reviewed_ids),
        "n_reviewed_cohort_rows": len(cohort_reviewed),
        "n_reviewed_outside_cohort_rows": len(reviewed_ids - cohort_ids),
        "n_cohort_binding_rows": len(cohort_rows),
        "n_cohort_ic50_rows": n_ic50_rows,
        "n_cohort_ic50_assay_comment_rows": sum(bool(row["assay_comments"]) for row in ic50_rows),
        "n_cohort_ic50_distinct_pmids": len(ic50_by_pmid) - ("<null>" in ic50_by_pmid),
        "n_cohort_ic50_pmid_unknown_rows": ic50_by_pmid["<null>"],
        "n_protocol_claims": len(claims),
        "binding_endpoint_counts": dict(sorted(endpoint_counts.items())),
        "binding_method_counts": dict(sorted(method_counts.items())),
        "cohort_ic50_by_pmid": dict(sorted(ic50_by_pmid.items())),
        "cohort_ic50_by_source": dict(sorted(ic50_by_source.items())),
    }
    assert manifest["reconciliation"] == {
        "n_binding_assay_rows": len(binding),
        "n_reviewed_binding_rows": len(reviewed_ids),
        "n_protocol_claims": len(claims),
        "n_cohort_binding_rows": len(cohort_rows),
        "n_cohort_ic50_rows": n_ic50_rows,
        "n_reviewed_cohort_rows": len(cohort_reviewed),
    }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = replay(args.package)
    serialized = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized)
    else:
        print(serialized, end="")


if __name__ == "__main__":
    main()
