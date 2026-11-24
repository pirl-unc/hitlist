"""Source-reviewed binding protocols appended to a verified evidence generation."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import xml.etree.ElementTree as ET
from collections import Counter
from importlib.resources import files
from pathlib import Path

import numpy as np
import pandas as pd
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


def verify_binding_assay_context_package(
    directory, *, manifest_path="manifest.json", allow_additional_files=False
):
    """Verify the inherited package and independently reconstruct every new row."""
    directory = Path(directory)
    manifest = json.loads((directory / manifest_path).read_text())
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
    if (not allow_additional_files and set(files) != actual) or (
        allow_additional_files and not set(files) <= actual
    ):
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


METHOD_REVIEW_REVISION = "binding_assay_method_reviews_v1"
METHOD_REVIEW_LEDGER_SOURCE = files("hitlist").joinpath(
    "data/evidence_revisions/binding_assay_method_reviews.json"
)
METHOD_REVIEW_STATUSES: frozenset[str] = frozenset(
    {
        "not_source_reviewed",
        "source_method_conflict_submission_scope",
        "source_detector_conflict_submission_scope",
        "source_detector_conflict_study_scope",
        "upstream_tag_changed_requires_rereview",
        "uncataloged_citation_requires_review",
    }
)
METHOD_REVIEW_COLUMNS: tuple[str, ...] = (
    "method_review_status",
    "method_review_protocol_id",
    "method_review_revision",
    "method_review_evidence_scope",
    "reviewed_preparation",
    "reviewed_mechanism",
    "reviewed_detector",
    "method_review_cited_pmid",
    "method_review_source_locator",
    "upstream_inconsistency_status",
)


def binding_method_review_contract(path=METHOD_REVIEW_LEDGER_SOURCE) -> dict:
    """Load and validate the source-linked binding assay method review ledger (#871)."""
    contract = json.loads(Path(path).read_text(encoding="utf-8"))
    if contract.get("schema_version") != 1 or contract.get("revision") != METHOD_REVIEW_REVISION:
        raise ValueError("Unsupported binding assay method review revision")
    cited = contract.get("cited_protocols")
    if not isinstance(cited, dict) or not cited:
        raise ValueError("Binding assay method review ledger missing cited_protocols")
    reviews = contract.get("reviews")
    if not isinstance(reviews, list) or not reviews:
        raise ValueError("Binding assay method review ledger missing reviews")

    seen_protocols: set[str] = set()
    seen_refs: set[str] = set()
    for review in reviews:
        protocol_id = review.get("protocol_id")
        if not protocol_id or protocol_id in seen_protocols:
            raise ValueError(f"Duplicate or missing method review protocol_id: {protocol_id!r}")
        seen_protocols.add(protocol_id)
        if review.get("method_review_status") not in METHOD_REVIEW_STATUSES:
            raise ValueError(
                f"Unknown method_review_status: {review.get('method_review_status')!r}"
            )
        if str(review.get("cited_pmid", "")) not in cited:
            raise ValueError(f"Uncataloged cited_pmid in {protocol_id!r}")
        refs = review.get("references")
        if not isinstance(refs, list) or not refs:
            raise ValueError(f"Review {protocol_id!r} must list at least one source reference")
        row_total = 0
        for ref in refs:
            ref_iri = ref.get("reference_iri")
            if not ref_iri or ref_iri in seen_refs:
                raise ValueError(f"Duplicate or missing reference_iri: {ref_iri!r}")
            seen_refs.add(ref_iri)
            row_total += int(ref.get("n_ci_corpus_v4_rows", 0))
        expected_rows = review.get("expected_ci_corpus_v4_rows")
        if expected_rows is not None and row_total != int(expected_rows):
            raise ValueError(
                f"Review {protocol_id!r} per-reference count sum {row_total} "
                f"!= expected_ci_corpus_v4_rows {expected_rows}"
            )
    return contract


def review_binding_assay_methods(df: pd.DataFrame, *, contract: dict | None = None) -> pd.DataFrame:
    """Attach source-scoped binding method conflict annotations without altering raw fields.

    Preserves all original assay identifiers, reported ``assay_method`` and
    ``response_measured`` strings, raw quantitative values, units, inequalities,
    and comments verbatim. Only explicitly reviewed submission/reference panels
    receive conflict annotations, and if upstream IEDB tags later change away
    from the reviewed strings, rows are flagged
    ``upstream_tag_changed_requires_rereview`` rather than double-corrected.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    ledger = binding_method_review_contract() if contract is None else contract
    result = df.copy()
    n_rows = len(result)
    if n_rows == 0:
        for col in METHOD_REVIEW_COLUMNS:
            result[col] = pd.Series(dtype="string")
        return result

    revision = str(ledger["revision"])
    empty_str = pd.Series("", index=result.index, dtype="string")
    refs = (
        result["reference_iri"].astype("string").fillna("").str.strip()
        if "reference_iri" in result.columns
        else empty_str
    )
    subs = (
        result["submission_id"].astype("string").fillna("").str.strip()
        if "submission_id" in result.columns
        else empty_str
    )
    methods = (
        result["assay_method"].astype("string").fillna("").str.strip()
        if "assay_method" in result.columns
        else empty_str
    )
    endpoints = (
        result["response_measured"].astype("string").fillna("").str.strip()
        if "response_measured" in result.columns
        else empty_str
    )
    has_comments_col = "assay_comments" in result.columns
    comments = (
        result["assay_comments"].astype("string").fillna("") if has_comments_col else empty_str
    )

    status = pd.Series("not_source_reviewed", index=result.index, dtype="string")
    protocol_id = pd.Series("", index=result.index, dtype="string")
    rev_col = pd.Series(revision, index=result.index, dtype="string")
    evidence_scope = pd.Series("unreviewed", index=result.index, dtype="string")
    reviewed_prep = pd.Series("", index=result.index, dtype="string")
    reviewed_mech = pd.Series("", index=result.index, dtype="string")
    reviewed_det = pd.Series("", index=result.index, dtype="string")
    cited_pmid = pd.Series("", index=result.index, dtype="string")
    source_locator = pd.Series("", index=result.index, dtype="string")
    inconsistency = pd.Series("", index=result.index, dtype="string")

    matched_any = pd.Series(False, index=result.index, dtype=bool)
    for review in ledger["reviews"]:
        ref_set = {r["reference_iri"] for r in review["references"]}
        sub_set = {r["submission_id"] for r in review["references"] if r.get("submission_id")}
        in_panel = refs.isin(ref_set)
        if sub_set:
            in_panel = in_panel | (refs.eq("") & subs.isin(sub_set))
        if not in_panel.any():
            continue

        matched_any = matched_any | in_panel
        tags_match = methods.eq(review["expected_assay_method"]) & endpoints.eq(
            review["expected_response_measured"]
        )
        expected_substr = review.get("expected_comment_substring") or ""
        if has_comments_col and expected_substr:
            non_empty_comment = comments.str.strip().ne("")
            comment_mismatch = non_empty_comment & ~comments.str.contains(
                expected_substr, regex=False
            )
            tags_match = tags_match & ~comment_mismatch

        exact_mask = in_panel & tags_match
        drift_mask = in_panel & ~tags_match

        if exact_mask.any():
            status.loc[exact_mask] = review["method_review_status"]
            protocol_id.loc[exact_mask] = review["protocol_id"]
            evidence_scope.loc[exact_mask] = review["evidence_scope"]
            reviewed_prep.loc[exact_mask] = review["reviewed_preparation"]
            reviewed_mech.loc[exact_mask] = review["reviewed_mechanism"]
            reviewed_det.loc[exact_mask] = review["reviewed_detector"]
            cited_pmid.loc[exact_mask] = str(review["cited_pmid"])
            source_locator.loc[exact_mask] = review["source_locator"]
            inconsistency.loc[exact_mask] = review["upstream_inconsistency_status"]

        if drift_mask.any():
            status.loc[drift_mask] = "upstream_tag_changed_requires_rereview"
            protocol_id.loc[drift_mask] = review["protocol_id"]
            evidence_scope.loc[drift_mask] = "requires_rereview"
            cited_pmid.loc[drift_mask] = str(review["cited_pmid"])
            source_locator.loc[drift_mask] = review["source_locator"]
            inconsistency.loc[drift_mask] = "upstream_tag_changed_requires_rereview"

    # Detect newly appearing uncataloged fluorescence rows citing a reviewed ELISA protocol
    # (excluding explicit LOCI assays, which are a distinct luminescent readout).
    if has_comments_col:
        is_fluor = methods.str.contains("fluorescence", case=False, regex=False)
        is_loci = comments.str.contains("LOCI", case=False, regex=False)
        for pmid_key, proto in ledger["cited_protocols"].items():
            if proto.get("detector") != "elisa":
                continue
            uncataloged = (
                ~matched_any
                & is_fluor
                & ~is_loci
                & comments.str.contains(str(pmid_key), regex=False)
            )
            if uncataloged.any():
                status.loc[uncataloged] = "uncataloged_citation_requires_review"
                evidence_scope.loc[uncataloged] = "requires_review"
                cited_pmid.loc[uncataloged] = str(pmid_key)
                source_locator.loc[uncataloged] = str(proto.get("source_locator", ""))
                inconsistency.loc[uncataloged] = "uncataloged_fluorescence_elisa_citation"

    result["method_review_status"] = status
    result["method_review_protocol_id"] = protocol_id
    result["method_review_revision"] = rev_col
    result["method_review_evidence_scope"] = evidence_scope
    result["reviewed_preparation"] = reviewed_prep
    result["reviewed_mechanism"] = reviewed_mech
    result["reviewed_detector"] = reviewed_det
    result["method_review_cited_pmid"] = cited_pmid
    result["method_review_source_locator"] = source_locator
    result["upstream_inconsistency_status"] = inconsistency
    return result


def summarize_binding_method_reviews(df: pd.DataFrame) -> dict:
    """Summarize binding method review counts across a reviewed or raw binding frame."""
    reviewed = df if "method_review_status" in df.columns else review_binding_assay_methods(df)
    return {
        "review_revision": METHOD_REVIEW_REVISION,
        "n_rows": len(reviewed),
        "by_status": dict(
            sorted(Counter(reviewed["method_review_status"].astype("string")).items())
        ),
        "by_protocol_id": dict(
            sorted(
                Counter(
                    val for val in reviewed["method_review_protocol_id"].astype("string") if val
                ).items()
            )
        ),
        "by_upstream_inconsistency_status": dict(
            sorted(
                Counter(
                    val for val in reviewed["upstream_inconsistency_status"].astype("string") if val
                ).items()
            )
        ),
    }


PROTOCOL_REGISTRY_REVISION = "binding_assay_protocol_registry_v1"
PROTOCOL_REGISTRY_LEDGER_SOURCE = files("hitlist").joinpath(
    "data/evidence_revisions/binding_assay_protocol_registry.json"
)
CONTEXT_AVAILABILITY_FIELDS: tuple[str, ...] = (
    "assay_format_status",
    "temperature_status",
    "ph_status",
    "tracer_status",
    "competitor_status",
    "incubation_status",
    "range_status",
    "replicate_status",
    "uncertainty_status",
)
VALID_AVAILABILITY_STATUSES: frozenset[str] = frozenset(
    {
        "study_scope_only",
        "unresolved",
        "unreported",
        "explicit_absence",
        "not_applicable",
        "conflicting",
        "unreviewed",
    }
)


def binding_protocol_registry_contract(path=PROTOCOL_REGISTRY_LEDGER_SOURCE) -> dict:
    """Load and validate the multi-study source-reviewed binding protocol registry (#721, #785)."""
    contract = json.loads(Path(path).read_text(encoding="utf-8"))
    if (
        contract.get("schema_version") != 1
        or contract.get("revision") != PROTOCOL_REGISTRY_REVISION
    ):
        raise ValueError("Unsupported binding assay protocol registry revision")
    policy = contract.get("policy") or {}
    if (
        policy.get("project_study_conditions_onto_rows") is not False
        or policy.get("convert_ic50_to_kd_without_evidenced_derivation") is not False
        or policy.get("relabel_exact_values_to_censored") is not False
    ):
        raise ValueError("Binding protocol registry policy invariant violated")

    protocols = contract.get("protocols")
    if not isinstance(protocols, list) or not protocols:
        raise ValueError("Binding protocol registry missing protocols")

    seen_ids: set[str] = set()
    for proto in protocols:
        pid = proto.get("protocol_id")
        if not pid or pid in seen_ids:
            raise ValueError(f"Duplicate or missing protocol_id: {pid!r}")
        seen_ids.add(pid)
        src = proto.get("source") or {}
        if not isinstance(src.get("pmid"), int) or not src.get("source_xml_sha256"):
            raise ValueError(f"Protocol {pid!r} missing verified source pmid/sha256")
        avail = proto.get("field_availability") or {}
        if set(avail) != set(CONTEXT_AVAILABILITY_FIELDS):
            raise ValueError(f"Protocol {pid!r} field_availability keys mismatch")
        for field_name, val in avail.items():
            if val not in VALID_AVAILABILITY_STATUSES:
                raise ValueError(f"Invalid availability status {val!r} for {pid!r}.{field_name}")
        claims = proto.get("claims")
        if not isinstance(claims, list) or not claims:
            raise ValueError(f"Protocol {pid!r} missing source-linked claims")
    return contract


CENSOR_REVIEW_REVISION = "binding_censor_review_v1"
CENSOR_REVIEW_STATUSES: frozenset[str] = frozenset(
    {
        "explicit_right_censored",
        "explicit_left_censored",
        "explicit_exact_unflagged",
        "blank_inequality_unreviewed",
        "suspected_upper_limit_heap_same_value_bound",
        "suspected_upper_limit_heap_negative_max",
        "non_quantitative",
    }
)
LIMIT_APPLICABILITY_STATUSES: frozenset[str] = frozenset(
    {
        "not_applicable",
        "explicit_reported_bound",
        "suspected_limit_unconfirmed",
        "source_confirmed_study_limit",
        "suspected_limit_matching_study_ceiling",
    }
)
CENSOR_REVIEW_COLUMNS: tuple[str, ...] = (
    "censor_review_status",
    "censor_heap_cell_id",
    "censor_heap_n_rows",
    "censor_heap_negative_fraction",
    "limit_applicability_status",
)


def _pick_series(df: pd.DataFrame, candidates: tuple[str, ...], default: str = "") -> pd.Series:
    for col in candidates:
        if col in df.columns:
            return df[col].astype("string").fillna(default).str.strip()
    return pd.Series(default, index=df.index, dtype="string")


def _pick_numeric_series(df: pd.DataFrame, candidates: tuple[str, ...]) -> pd.Series:
    for col in candidates:
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce").astype("float64")
    return pd.Series(np.nan, index=df.index, dtype="float64")


def _build_study_key_series(df: pd.DataFrame) -> pd.Series:
    refs = _pick_series(df, ("reference_iri",))
    subs = _pick_series(df, ("submission_id",))
    pmids = _pick_series(df, ("pmid", "source_pmid"))
    sources = _pick_series(df, ("source", "source_dataset", "measurement_source"))
    key = refs.copy()
    blank = key.eq("")
    if blank.any():
        has_pmid = blank & pmids.ne("") & ~pmids.str.casefold().isin({"nan", "none", "<na>"})
        key.loc[has_pmid] = "pmid:" + pmids.loc[has_pmid]
    blank = key.eq("")
    if blank.any():
        has_sub = blank & subs.ne("")
        key.loc[has_sub] = "submission:" + subs.loc[has_sub]
    blank = key.eq("")
    if blank.any():
        key.loc[blank] = "source:" + sources.loc[blank].replace("", "unknown")
    return key


def _cell_digest(parts: tuple[str, ...]) -> str:
    joined = "|".join(parts).encode("utf-8")
    return "heap:" + hashlib.sha256(joined).hexdigest()[:16]


def review_binding_censoring(
    df: pd.DataFrame,
    *,
    min_heap_size: int = 3,
    min_negative_fraction: float = 0.5,
    protocol_registry: dict | None = None,
) -> pd.DataFrame:
    """Attach censoring and suspected-upper-limit heap review state without altering raw values.

    Implements the non-destructive censoring review requested in #721 across
    both binding affinity and stability assay frames (accepting either raw
    ``quantitative_value``/``measurement_units``/``measurement_inequality`` or
    canonical ``measurement_value``/``measurement_unit``/``raw_measurement_inequality``
    columns symmetrically).

    Invariants:
    - Never converts exact (``=``) or blank-inequality rows into right-censored
      bounds (``>`` / ``>=``).
    - Separates blank raw inequalities (``blank_inequality_unreviewed``) from
      explicit exact equalities (``explicit_exact_unflagged``).
    - Screens within-``(study, mhc, method, endpoint, unit)`` cells for
      suspected upper-limit heaps where exact/blank rows either share a value
      with an explicit right bound (``suspected_upper_limit_heap_same_value_bound``)
      or sit at the cell's maximum value with ``n_rows >= min_heap_size`` and
      qualitative negative fraction ``>= min_negative_fraction``
      (``suspected_upper_limit_heap_negative_max``).
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    if min_heap_size < 2:
        raise ValueError("min_heap_size must be >= 2")
    if not (0.0 <= min_negative_fraction <= 1.0):
        raise ValueError("min_negative_fraction must be in [0.0, 1.0]")

    registry = (
        binding_protocol_registry_contract() if protocol_registry is None else protocol_registry
    )
    result = df.copy()
    if len(result) == 0:
        result["censor_review_status"] = pd.Series(dtype="string")
        result["censor_heap_cell_id"] = pd.Series(dtype="string")
        result["censor_heap_n_rows"] = pd.Series(dtype="Int64")
        result["censor_heap_negative_fraction"] = pd.Series(dtype="float64")
        result["limit_applicability_status"] = pd.Series(dtype="string")
        return result

    raw_ineq = _pick_series(result, ("raw_measurement_inequality", "measurement_inequality"))
    vals = _pick_numeric_series(result, ("measurement_value", "quantitative_value"))
    units = _pick_series(result, ("measurement_unit", "measurement_units"))
    mhcs = _pick_series(result, ("mhc_restriction", "reported_mhc_restriction", "allele"))
    methods = _pick_series(result, ("assay_method",))
    endpoints = _pick_series(result, ("response_measured", "endpoint_reported", "measurement_kind"))
    quals = _pick_series(result, ("qualitative_measurement", "qualitative_result")).str.casefold()
    pmids = _pick_series(result, ("pmid", "source_pmid"))
    mhc_classes = _pick_series(result, ("mhc_class",))
    study_keys = _build_study_key_series(result)

    finite_mask = np.isfinite(vals.to_numpy(dtype="float64", na_value=np.nan))
    finite = pd.Series(finite_mask, index=result.index, dtype=bool)
    right_mask = finite & raw_ineq.isin({">", ">="})
    left_mask = finite & raw_ineq.isin({"<", "<="})
    explicit_eq_mask = finite & raw_ineq.eq("=")
    blank_mask = finite & raw_ineq.eq("")
    exact_or_blank = explicit_eq_mask | blank_mask
    is_neg = quals.eq("negative")

    status = pd.Series("non_quantitative", index=result.index, dtype="string")
    status.loc[right_mask] = "explicit_right_censored"
    status.loc[left_mask] = "explicit_left_censored"
    status.loc[explicit_eq_mask] = "explicit_exact_unflagged"
    status.loc[blank_mask] = "blank_inequality_unreviewed"

    heap_cell_id = pd.Series("", index=result.index, dtype="string")
    heap_n_rows = pd.Series(0, index=result.index, dtype="Int64")
    heap_neg_frac = pd.Series(np.nan, index=result.index, dtype="float64")
    limit_status = pd.Series("not_applicable", index=result.index, dtype="string")
    limit_status.loc[right_mask | left_mask] = "explicit_reported_bound"

    # Build study-level documented upper ceilings (nM) from source-reviewed protocols.
    study_ceilings: dict[tuple[str, str], float] = {}
    for proto in registry.get("protocols", []):
        ceiling = proto.get("upper_measurement_limit_nm")
        if ceiling is None:
            continue
        pmid_str = str(proto["source"]["pmid"])
        classes = proto.get("scope", {}).get("allowed_mhc_classes") or [""]
        for cls_tok in classes:
            study_ceilings[(pmid_str, str(cls_tok))] = float(ceiling)

    if exact_or_blank.any():
        work = pd.DataFrame(
            {
                "_study": study_keys,
                "_mhc": mhcs,
                "_method": methods,
                "_endpoint": endpoints,
                "_unit": units,
                "_val": vals,
                "_exact_or_blank": exact_or_blank,
                "_right": right_mask,
                "_neg": is_neg,
            },
            index=result.index,
        )
        fin = work[finite]
        cell_cols = ["_study", "_mhc", "_method", "_endpoint", "_unit"]
        val_cols = [*cell_cols, "_val"]

        cell_max = fin.groupby(cell_cols, sort=False, observed=True)["_val"].transform("max")
        has_right_same_val = fin.groupby(val_cols, sort=False, observed=True)["_right"].transform(
            "any"
        )
        eb = fin[fin["_exact_or_blank"]]
        if not eb.empty:
            eb_grp = eb.groupby(val_cols, sort=False, observed=True)
            eb_count = eb_grp["_val"].transform("size")
            eb_neg_frac = eb_grp["_neg"].transform("mean")
            same_val_mask = has_right_same_val.loc[eb.index]
            neg_max_mask = (
                ~same_val_mask
                & eb["_val"].eq(cell_max.loc[eb.index])
                & eb_count.ge(min_heap_size)
                & eb_neg_frac.ge(min_negative_fraction)
            )
            flagged_mask = same_val_mask | neg_max_mask
            if flagged_mask.any():
                flagged_idx = eb.index[flagged_mask]
                same_idx = eb.index[same_val_mask]
                neg_idx = eb.index[neg_max_mask]
                if len(same_idx):
                    status.loc[same_idx] = "suspected_upper_limit_heap_same_value_bound"
                if len(neg_idx):
                    status.loc[neg_idx] = "suspected_upper_limit_heap_negative_max"
                cell_tuples = eb.loc[flagged_idx, cell_cols].itertuples(index=False, name=None)
                unique_digests: dict[tuple[str, ...], str] = {}
                digests = []
                for tup in cell_tuples:
                    str_tup = tuple(str(part) for part in tup)
                    digest = unique_digests.get(str_tup)
                    if digest is None:
                        digest = _cell_digest(str_tup)
                        unique_digests[str_tup] = digest
                    digests.append(digest)
                heap_cell_id.loc[flagged_idx] = pd.Series(
                    digests, index=flagged_idx, dtype="string"
                )
                heap_n_rows.loc[flagged_idx] = eb_count.loc[flagged_idx].astype("Int64")
                heap_neg_frac.loc[flagged_idx] = eb_neg_frac.loc[flagged_idx].astype("float64")
                limit_status.loc[flagged_idx] = "suspected_limit_unconfirmed"

    # Annotate source-confirmed study ceilings where unit is nM (without mutating inequalities).
    if study_ceilings:
        is_nm = units.str.casefold().eq("nm")
        for (pmid_str, cls_tok), ceiling_nm in study_ceilings.items():
            in_study = finite & is_nm & pmids.eq(pmid_str)
            if cls_tok:
                in_study = in_study & mhc_classes.eq(cls_tok)
            if not in_study.any():
                continue
            confirmed_right = in_study & right_mask & vals.ge(ceiling_nm)
            if confirmed_right.any():
                limit_status.loc[confirmed_right] = "source_confirmed_study_limit"
            exact_at_ceiling = in_study & exact_or_blank & vals.ge(ceiling_nm)
            if exact_at_ceiling.any():
                limit_status.loc[exact_at_ceiling] = "suspected_limit_matching_study_ceiling"

    result["censor_review_status"] = status
    result["censor_heap_cell_id"] = heap_cell_id
    result["censor_heap_n_rows"] = heap_n_rows
    result["censor_heap_negative_fraction"] = heap_neg_frac
    result["limit_applicability_status"] = limit_status
    return result


def summarize_binding_censoring_reviews(df: pd.DataFrame) -> dict:
    """Summarize censoring and suspected-limit heap review counts across a binding/stability frame."""
    reviewed = df if "censor_review_status" in df.columns else review_binding_censoring(df)
    statuses = reviewed["censor_review_status"].astype("string")
    limits = reviewed["limit_applicability_status"].astype("string")
    suspected_mask = statuses.isin(
        {
            "suspected_upper_limit_heap_same_value_bound",
            "suspected_upper_limit_heap_negative_max",
        }
    )
    cells = {
        val for val in reviewed.loc[suspected_mask, "censor_heap_cell_id"].astype("string") if val
    }
    return {
        "review_revision": CENSOR_REVIEW_REVISION,
        "n_rows": len(reviewed),
        "n_explicit_right_censored_rows": int(statuses.eq("explicit_right_censored").sum()),
        "n_explicit_left_censored_rows": int(statuses.eq("explicit_left_censored").sum()),
        "n_explicit_exact_unflagged_rows": int(statuses.eq("explicit_exact_unflagged").sum()),
        "n_blank_inequality_unreviewed_rows": int(statuses.eq("blank_inequality_unreviewed").sum()),
        "n_suspected_heap_rows": int(suspected_mask.sum()),
        "n_suspected_heap_cells": len(cells),
        "by_censor_review_status": dict(sorted(Counter(statuses).items())),
        "by_limit_applicability_status": dict(sorted(Counter(limits).items())),
    }
