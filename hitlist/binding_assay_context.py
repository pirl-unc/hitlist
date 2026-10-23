"""Source-reviewed binding protocols appended to a verified evidence generation."""

from __future__ import annotations

import json
import shutil
import tempfile
import xml.etree.ElementTree as ET
from importlib.resources import files
from pathlib import Path

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from .evidence_snapshot import snapshot_file
from .external_evidence import _safe_relative
from .external_successor import verify_external_successor
from .provenance import file_digest

PACKAGE_SCHEMA_VERSION = 10
CATALOG_SCHEMA_VERSION = 15
REVIEW_REVISION = "sidney_2017_binding_protocol_v1"
LEDGER_SHA256 = "966c5e54c331cb5f7865a6f4acf0ad4719c847cb67c5cce72dda8c8f5328fb13"
V4_MANIFEST_SHA256 = "2318bc53cdae228285218991d58fbcc2bd7266327cec1dd3ed9a03b1c944a5b7"
COHORT_MANIFEST_SHA256 = "b180eba804532f633a6d61bb350584dbe24f8df4b4b96f02c6ec492c60144372"
LEDGER_SOURCE = files("hitlist").joinpath("data/evidence_revisions/binding_assay_protocols.json")
REVIEW_DIRECTORY = Path("binding_review")
PARENT_MANIFEST = Path("manifests/binding_review_parent.json")
CONTEXT_TABLE = REVIEW_DIRECTORY / "binding_assay_context.parquet"
CLAIM_TABLE = REVIEW_DIRECTORY / "binding_protocol_claims.parquet"
SOURCE_XML = REVIEW_DIRECTORY / "source.xml"
LEDGER_COPY = REVIEW_DIRECTORY / "review.json"

CONTEXT_SCHEMA = pa.schema(
    [
        ("observation_id", pa.string()),
        ("source_assay_id", pa.string()),
        ("source_pmid", pa.string()),
        ("source_dataset", pa.string()),
        ("endpoint_reported", pa.string()),
        ("assay_method", pa.string()),
        ("quantitative_raw", pa.string()),
        ("measurement_unit", pa.string()),
        ("measurement_inequality", pa.string()),
        ("review_status", pa.string()),
        ("protocol_id", pa.string()),
        ("review_revision", pa.string()),
    ]
)
CLAIM_SCHEMA = pa.schema(
    [
        ("claim_id", pa.string()),
        ("protocol_id", pa.string()),
        ("assertion_predicate", pa.string()),
        ("assertion_value", pa.string()),
        ("measurement_unit", pa.string()),
        ("claim_status", pa.string()),
        ("claim_applicability", pa.string()),
        ("source_scope", pa.string()),
        ("review_revision", pa.string()),
        ("source_pmid", pa.string()),
        ("publication_url", pa.string()),
        ("source_asset_sha256", pa.string()),
        ("source_locator", pa.string()),
        ("source_excerpt", pa.string()),
    ]
)
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
CLAIM_STATUSES = {
    "reported_study_scope",
    "reported_locus_scope",
    "reported_range_per_assay_unresolved",
    "reported_alternatives_per_assay_unresolved",
    "reported_interpretation_not_measurement",
    "unreported",
    "explicit_absence",
    "not_applicable",
}


def _load_ledger(path=LEDGER_SOURCE):
    if file_digest(path)["sha256"] != LEDGER_SHA256:
        raise ValueError("Unrecognized binding protocol review ledger")
    ledger = json.loads(Path(path).read_text())
    if ledger.get("schema_version") != 1 or ledger.get("revision") != REVIEW_REVISION:
        raise ValueError("Unsupported binding protocol review revision")
    if ledger.get("expected_binding_rows") != 938 or ledger.get("expected_cohort_rows") != 910:
        raise ValueError("Binding protocol source coverage changed")
    if ledger.get("source", {}).get("pmid") != 28481943:
        raise ValueError("Unexpected binding protocol PMID")
    claims = ledger.get("claims")
    if not isinstance(claims, list) or not claims:
        raise ValueError("Empty binding protocol claims")
    keys = set()
    for claim in claims:
        if set(claim) != {
            "predicate",
            "value",
            "unit",
            "status",
            "applicability",
            "source_excerpt",
        }:
            raise ValueError("Invalid binding protocol claim shape")
        if claim["status"] not in CLAIM_STATUSES:
            raise ValueError("Unknown binding protocol claim status")
        if not claim["predicate"] or not claim["applicability"]:
            raise ValueError("Incomplete binding protocol claim")
        if claim["status"] in {"unreported", "not_applicable"}:
            if claim["value"] is not None or claim["source_excerpt"] is not None:
                raise ValueError("An unreported binding protocol field has evidence or a value")
        elif claim["status"] == "explicit_absence":
            if claim["value"] is not None or not claim["source_excerpt"]:
                raise ValueError("Explicit absence requires a source excerpt and no value")
        elif claim["value"] is None or not claim["source_excerpt"]:
            raise ValueError("A reported binding protocol claim needs a value and source excerpt")
        key = claim["predicate"], claim["applicability"]
        if key in keys:
            raise ValueError("Duplicate binding protocol predicate/applicability")
        keys.add(key)
    return ledger


def _source_section(source_xml, ledger):
    if file_digest(source_xml)["sha256"] != ledger["source"]["source_xml_sha256"]:
        raise ValueError("Original binding protocol article bytes changed")
    root = ET.parse(source_xml).getroot()
    section = root.find(f".//sec[@id='{ledger['source']['section_id']}']")
    title = section.find("title") if section is not None else None
    if title is None or "".join(title.itertext()) != ledger["source"]["section_title"]:
        raise ValueError("Binding protocol source section changed")
    license_text = " ".join("".join(node.itertext()) for node in root.findall(".//license"))
    if "Creative Commons Attribution License" not in license_text:
        raise ValueError("Binding protocol source license is unverified")
    source_text = " ".join("".join(section.itertext()).split())
    for claim in ledger["claims"]:
        if claim["source_excerpt"] and claim["source_excerpt"] not in source_text:
            raise ValueError(f"Binding protocol source excerpt missing: {claim['predicate']}")
    return source_text


def _source_rows(directory):
    binding = pq.read_table(directory / "native/binding.parquet", columns=list(SOURCE_COLUMNS))
    if binding.num_rows != 814_268:
        raise ValueError("Pinned v4 binding assay count changed")
    source_ids = binding.column("provenance_id").to_pylist()
    if len(set(source_ids)) != len(source_ids):
        raise ValueError("Duplicate v4 binding observation ID")
    return binding.to_pylist()


def _cohort_anchors(directory, indexed, ledger):
    cohort_manifest = directory / "cohort_native/manifest.json"
    if file_digest(cohort_manifest)["sha256"] != COHORT_MANIFEST_SHA256:
        raise ValueError("Corrected human cohort manifest changed")
    columns = [*SOURCE_COLUMNS, "is_binding_assay"]
    cohort = ds.dataset(directory / "cohort_native/training.parquet", format="parquet")
    rows = cohort.to_table(columns=columns, filter=ds.field("is_binding_assay")).to_pylist()
    if len(rows) != 25_896:
        raise ValueError("Corrected human binding cohort count changed")
    cohort_ids = set()
    n_ic50_rows = 0
    n_reviewed_rows = 0
    for row in rows:
        observation_id = row["provenance_id"]
        if observation_id in cohort_ids or observation_id not in indexed:
            raise ValueError("Corrected human cohort binding identity mismatch")
        cohort_ids.add(observation_id)
        if any(row[name] != indexed[observation_id][name] for name in SOURCE_COLUMNS):
            raise ValueError(f"Corrected cohort changed source binding result: {observation_id}")
        if row["response_measured"] == ledger["required_endpoint"]:
            n_ic50_rows += 1
        if row["pmid"] == ledger["source"]["pmid"]:
            n_reviewed_rows += 1
    if n_ic50_rows != 8_084 or n_reviewed_rows != ledger["expected_cohort_rows"]:
        raise ValueError("Corrected human cohort IC50 review coverage changed")
    return {
        "n_cohort_binding_rows": len(rows),
        "n_cohort_ic50_rows": n_ic50_rows,
        "n_reviewed_cohort_rows": n_reviewed_rows,
    }


def _compile(directory, ledger):
    mhc_manifest = json.loads((directory / "manifests/mhc.json").read_text())
    if mhc_manifest.get("source_manifest", {}).get("sha256") != V4_MANIFEST_SHA256:
        raise ValueError("Unrecognized v4 source manifest")
    native_meta = json.loads((directory / "native/observations_meta.json").read_text())
    if native_meta.get("ci_corpus_manifest_version") != 2:
        raise ValueError("Unrecognized v4 source provenance")
    if native_meta.get("counts", {}).get("n_binding_rows") != 814_268:
        raise ValueError("Pinned v4 source count changed")
    source_rows = _source_rows(directory)
    indexed = {row["provenance_id"]: row for row in source_rows}
    cohort_counts = _cohort_anchors(directory, indexed, ledger)
    context_rows = []
    n_reviewed_rows = 0
    for source_row in source_rows:
        reviewed = source_row["pmid"] == ledger["source"]["pmid"]
        if reviewed:
            if any(
                source_row[name] != ledger[requirement]
                for name, requirement in (
                    ("source", "required_source"),
                    ("assay_method", "required_method"),
                    ("response_measured", "required_endpoint"),
                    ("measurement_units", "required_unit"),
                )
            ):
                raise ValueError("Reviewed PMID includes a different assay protocol or endpoint")
            n_reviewed_rows += 1
        context_rows.append(
            {
                "observation_id": source_row["provenance_id"],
                "source_assay_id": source_row["assay_iri"],
                "source_pmid": str(source_row["pmid"]) if source_row["pmid"] is not None else None,
                "source_dataset": source_row["source"],
                "endpoint_reported": source_row["response_measured"] or None,
                "assay_method": source_row["assay_method"] or None,
                "quantitative_raw": source_row["quantitative_measurement"] or None,
                "measurement_unit": source_row["measurement_units"] or None,
                "measurement_inequality": source_row["measurement_inequality"] or None,
                "review_status": "source_reviewed_study_scope"
                if reviewed
                else "not_source_reviewed",
                "protocol_id": ledger["protocol_id"] if reviewed else None,
                "review_revision": REVIEW_REVISION,
            }
        )
    if n_reviewed_rows != ledger["expected_binding_rows"]:
        raise ValueError("Reviewed source binding assay coverage changed")
    claim_rows = []
    for claim in ledger["claims"]:
        claim_rows.append(
            {
                "claim_id": f"{ledger['protocol_id']}:{claim['predicate']}:{claim['applicability']}",
                "protocol_id": ledger["protocol_id"],
                "assertion_predicate": claim["predicate"],
                "assertion_value": claim["value"],
                "measurement_unit": claim["unit"],
                "claim_status": claim["status"],
                "claim_applicability": claim["applicability"],
                "source_scope": "study_protocol",
                "review_revision": REVIEW_REVISION,
                "source_pmid": str(ledger["source"]["pmid"]),
                "publication_url": ledger["source"]["article_url"],
                "source_asset_sha256": ledger["source"]["source_xml_sha256"],
                "source_locator": f"PMC5421785/fullTextXML#{ledger['source']['section_id']}",
                "source_excerpt": claim["source_excerpt"],
            }
        )
    return (
        pa.Table.from_pylist(context_rows, schema=CONTEXT_SCHEMA),
        pa.Table.from_pylist(claim_rows, schema=CLAIM_SCHEMA),
        {
            "n_binding_assay_rows": len(context_rows),
            "n_reviewed_binding_rows": n_reviewed_rows,
            "n_protocol_claims": len(claim_rows),
            **cohort_counts,
        },
    )


def verify_binding_assay_context_package(directory):
    """Verify the inherited package and independently reconstruct every new row."""
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if (
        manifest.get("schema_version") != PACKAGE_SCHEMA_VERSION
        or manifest.get("catalog_schema_version") != CATALOG_SCHEMA_VERSION
        or manifest.get("parent_kind") != "external"
        or manifest.get("parent_manifest_path") != str(PARENT_MANIFEST)
        or manifest.get("review_revision") != REVIEW_REVISION
    ):
        raise ValueError("Unsupported binding assay review package")
    files = manifest.get("files", {})
    actual = {
        str(path.relative_to(directory))
        for path in directory.rglob("*")
        if path.is_file() and path != directory / "manifest.json"
    }
    if set(files) != actual:
        raise ValueError("Binding assay review inventory mismatch")
    for name, digest in files.items():
        if file_digest(directory / _safe_relative(name)) != digest:
            raise ValueError(f"Binding assay review artifact mismatch: {name}")
    if files[str(PARENT_MANIFEST)]["sha256"] != manifest.get("parent_manifest_sha256"):
        raise ValueError("Binding assay review predecessor hash mismatch")
    parent = verify_external_successor(
        directory, manifest_path=PARENT_MANIFEST, allow_additional_files=True
    )
    expected_files = set(parent["files"]) | {
        str(PARENT_MANIFEST),
        str(CONTEXT_TABLE),
        str(CLAIM_TABLE),
        str(SOURCE_XML),
        str(LEDGER_COPY),
    }
    if set(files) != expected_files:
        raise ValueError("Binding assay review contains unowned artifacts")
    if any(files.get(name) != digest for name, digest in parent["files"].items()):
        raise ValueError("Binding assay review changed predecessor bytes")
    if (
        manifest.get("reference") != parent["reference"]
        or manifest.get("appended_sources") != parent["appended_sources"]
        or manifest.get("adjudication_manifest_path")
        != ("manifests/adjudication.json" if parent["parent_kind"] == "adjudication" else None)
    ):
        raise ValueError("Binding assay review changed predecessor metadata")
    ledger = _load_ledger(directory / LEDGER_COPY)
    _source_section(directory / SOURCE_XML, ledger)
    expected_context, expected_claims, counts = _compile(directory, ledger)
    for relative_path, expected in (
        (CONTEXT_TABLE, expected_context),
        (CLAIM_TABLE, expected_claims),
    ):
        actual_table = pq.read_table(directory / relative_path)
        if not actual_table.equals(expected, check_metadata=False):
            raise ValueError(f"Binding assay review rows changed: {relative_path}")
    if manifest.get("reconciliation") != counts:
        raise ValueError("Binding assay review count mismatch")
    return manifest


def write_binding_assay_context_package(predecessor_package, source_xml, destination):
    """Append source-reviewed binding protocol evidence to a verified generation."""
    predecessor_package, source_xml, destination = (
        Path(predecessor_package),
        Path(source_xml),
        Path(destination),
    )
    if destination.exists():
        raise FileExistsError(destination)
    parent = verify_external_successor(predecessor_package)
    if (parent["schema_version"], parent["catalog_schema_version"]) != (9, 14):
        raise ValueError("Binding review requires the current combined external package")
    ledger = _load_ledger()
    _source_section(source_xml, ledger)
    added = {
        str(PARENT_MANIFEST),
        str(CONTEXT_TABLE),
        str(CLAIM_TABLE),
        str(SOURCE_XML),
        str(LEDGER_COPY),
    }
    if added & set(parent["files"]):
        raise ValueError("Binding review would replace predecessor artifacts")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    try:
        for name in parent["files"]:
            snapshot_file(predecessor_package / name, staging / name)
        snapshot_file(predecessor_package / "manifest.json", staging / PARENT_MANIFEST)
        snapshot_file(source_xml, staging / SOURCE_XML)
        snapshot_file(LEDGER_SOURCE, staging / LEDGER_COPY)
        context, claims, counts = _compile(staging, ledger)
        pq.write_table(context, staging / CONTEXT_TABLE, compression="zstd")
        pq.write_table(claims, staging / CLAIM_TABLE, compression="zstd")
        from .version import __version__

        files = {
            str(path.relative_to(staging)): file_digest(path)
            for path in sorted(staging.rglob("*"))
            if path.is_file()
        }
        manifest = {
            "schema_version": PACKAGE_SCHEMA_VERSION,
            "catalog_schema_version": CATALOG_SCHEMA_VERSION,
            "hitlist_version": __version__,
            "parent_kind": "external",
            "parent_manifest_path": str(PARENT_MANIFEST),
            "parent_manifest_sha256": files[str(PARENT_MANIFEST)]["sha256"],
            "adjudication_manifest_path": (
                "manifests/adjudication.json" if parent["parent_kind"] == "adjudication" else None
            ),
            "reference": parent["reference"],
            "appended_sources": parent["appended_sources"],
            "review_revision": REVIEW_REVISION,
            "reconciliation": counts,
            "files": files,
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        verify_binding_assay_context_package(staging)
        verify_external_successor(predecessor_package)
        if (
            file_digest(predecessor_package / "manifest.json")["sha256"]
            != manifest["parent_manifest_sha256"]
        ):
            raise ValueError("Binding review predecessor changed during generation")
        staging.rename(destination)
        return destination / "manifest.json"
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
