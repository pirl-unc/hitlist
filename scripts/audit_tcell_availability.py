"""Aggregate the pinned IEDB/CEDAR T-cell exports without exporting epitopes.

This is source availability, not a functional training table. Assay IDs shared
between the two databases are counted once in the union summary.
"""

import argparse
import csv
import hashlib
import io
import json
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

from datacache import install_bundle

from hitlist.curation_yaml import load_curation_yaml
from hitlist.downloads import data_dir

CATALOG = Path(__file__).resolve().parents[1] / "hitlist/data/tcell_audit_sources.yaml"
CONTEXT_FIELDS = (
    "Host | MHC Present",
    "In vitro Process | Process Type",
    "in vitro Responder Cell | Name",
    "in vitro Stimulator Cell | Name",
    "in vitro immunogen | Object Type",
    "Assay | Units",
    "Assay | Measurement Inequality",
    "Assay | Quantitative measurement",
    "Assay | Number of Subjects Tested",
    "Assay | Number of Subjects Positive",
    "Effector Cell | Name",
    "Effector Cell | Culture Condition",
    "Effector Cell | TCR Name",
    "Antigen Presenting Cell | Name",
    "Antigen Presenting Cell | Culture Condition",
    "MHC Restriction | Name",
    "MHC Restriction | Evidence Code",
    "Assay Antigen | Object Type",
)
REQUIRED = (
    "Assay | Method",
    "Assay | Response measured",
    "Assay | Qualitative Measurement",
    "MHC Restriction | Name",
    "MHC Restriction | Evidence Code",
    *CONTEXT_FIELDS,
)
HLA_TWO_FIELD = re.compile(r"^HLA-[A-Z0-9]+\*\d{2,3}:\d{2,3}$")
PROCESSING_TERMS = ("cleavage", "transport", "proteasom", "processing")


def pinned_sources(*, data_root=None):
    """Acquire exact source ZIP bytes through datacache."""
    catalog = load_curation_yaml(CATALOG)
    root = Path(data_root or data_dir())
    return install_bundle(
        root / "functional-evidence/tcell-availability-2026-10-20",
        catalog["files"],
        download_options={"timeout": 120, "show_progress": False},
    )


def check_pin(path, spec):
    path = Path(path)
    if path.stat().st_size != spec["size"]:
        raise ValueError(f"Source size differs from pinned bytes: {path.name}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    if digest.hexdigest() != spec["sha256"]:
        raise ValueError(f"Source hash differs from pinned bytes: {path.name}")


def family(method, response):
    """Conservative metadata routing; direct processing needs separate review."""
    method, response = method.casefold(), response.casefold()
    if method == "multimer/tetramer":
        return "pmhc_multimer"
    if response == "cytotoxicity":
        return "cytotoxicity"
    if response in {"activation", "degranulation", "proliferation"}:
        return "activation_or_proliferation"
    if response.endswith(" release"):
        return "soluble_factor_release"
    return "other_tcell_assay"


def _cell(row, positions, name):
    return row[positions[name]].strip()


def audit_zip(path):
    """Return counts by distinct assay IRI; reject malformed and repeated rows."""
    counts = Counter()
    methods = Counter()
    responses = Counter()
    outcomes = Counter()
    outcomes_by_family = defaultdict(Counter)
    restriction = Counter()
    evidence = Counter()
    evidence_by_resolution = defaultdict(Counter)
    families = Counter()
    context = Counter()
    studies = set()
    family_studies = defaultdict(set)
    assay_keys = set()
    with zipfile.ZipFile(path) as archive:
        if archive.namelist() != ["tcell_full_v3.csv"]:
            raise ValueError("Unexpected T-cell archive members")
        with archive.open("tcell_full_v3.csv") as raw:
            reader = csv.reader(io.TextIOWrapper(raw, encoding="utf-8-sig", newline=""))
            major, minor = next(reader), next(reader)
            headers = [f"{a} | {b}" for a, b in zip(major, minor)]
            if len(major) != len(minor) or len(headers) != len(set(headers)):
                raise ValueError("Invalid double-header schema")
            missing = set(REQUIRED) - set(headers)
            if missing:
                raise ValueError(f"Missing T-cell columns: {sorted(missing)}")
            iri_kind = "CEDAR IRI" if "Assay ID | CEDAR IRI" in headers else "IEDB IRI"
            positions = {key: headers.index(key) for key in set(REQUIRED)}
            assay_pos = headers.index(f"Assay ID | {iri_kind}")
            study_pos = headers.index(f"Reference | {iri_kind}")
            for row in reader:
                counts["n_rows"] += 1
                if len(row) != len(headers):
                    raise ValueError(f"Malformed T-cell row {counts['n_rows']}")
                assay_id = row[assay_pos].strip()
                study_id = row[study_pos].strip()
                if not assay_id or not study_id:
                    raise ValueError(f"Missing assay or study ID on row {counts['n_rows']}")
                key = assay_id.rsplit("/", 1)[-1]
                if key in assay_keys:
                    raise ValueError(f"Repeated assay ID: {key}")
                assay_keys.add(key)
                studies.add(study_id.rsplit("/", 1)[-1])
                method = _cell(row, positions, "Assay | Method")
                response = _cell(row, positions, "Assay | Response measured")
                methods[method or "<missing>"] += 1
                responses[response or "<missing>"] += 1
                outcome = _cell(row, positions, "Assay | Qualitative Measurement")
                outcomes[outcome or "<missing>"] += 1
                category = family(method, response)
                families[category] += 1
                outcomes_by_family[category][outcome or "<missing>"] += 1
                family_studies[category].add(study_id.rsplit("/", 1)[-1])
                mhc = _cell(row, positions, "MHC Restriction | Name")
                if not mhc:
                    resolution = "missing"
                elif HLA_TWO_FIELD.fullmatch(mhc):
                    resolution = "single_hla_two_field_name"
                else:
                    resolution = "other_reported_name"
                restriction[resolution] += 1
                evidence_code = (
                    _cell(row, positions, "MHC Restriction | Evidence Code") or "<missing>"
                )
                evidence[evidence_code] += 1
                evidence_by_resolution[resolution][evidence_code] += 1
                if any(term in f"{method} {response}".casefold() for term in PROCESSING_TERMS):
                    counts["n_processing_keyword_candidate_assays"] += 1
                for field in CONTEXT_FIELDS:
                    if _cell(row, positions, field):
                        context[field] += 1
    counts["n_assays"] = len(assay_keys)
    counts["n_studies"] = len(studies)
    counts.setdefault("n_processing_keyword_candidate_assays", 0)
    return {
        "counts": dict(sorted(counts.items())),
        "outcomes": dict(sorted(outcomes.items())),
        "outcomes_by_family": {
            k: dict(sorted(v.items())) for k, v in sorted(outcomes_by_family.items())
        },
        "families": dict(sorted(families.items())),
        "n_studies_by_family": {k: len(v) for k, v in sorted(family_studies.items())},
        "restriction_name_resolution": dict(sorted(restriction.items())),
        "restriction_evidence": dict(sorted(evidence.items())),
        "restriction_evidence_by_resolution": {
            k: dict(sorted(v.items())) for k, v in sorted(evidence_by_resolution.items())
        },
        "methods": dict(sorted(methods.items())),
        "responses": dict(sorted(responses.items())),
        "n_assays_with_context_field": dict(sorted(context.items())),
    }, assay_keys


def run(paths, catalog_path=CATALOG):
    catalog = load_curation_yaml(catalog_path)
    result = {"schema_version": 1, "source_observed": catalog["observed"], "sources": {}}
    key_sets = {}
    for name, spec in catalog["files"].items():
        path = Path(paths[name])
        check_pin(path, spec)
        summary, keys = audit_zip(path)
        result["sources"][name] = {"url": spec["url"], "sha256": spec["sha256"], **summary}
        key_sets[name] = keys
    iedb = key_sets["iedb_tcell_full_v3.zip"]
    cedar = key_sets["cedar_tcell_full_v3.zip"]
    result["cross_source"] = {
        "n_shared_assay_ids": len(iedb & cedar),
        "n_iedb_only_assay_ids": len(iedb - cedar),
        "n_cedar_only_assay_ids": len(cedar - iedb),
        "n_union_assay_ids": len(iedb | cedar),
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iedb", type=Path)
    parser.add_argument("--cedar", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if bool(args.iedb) != bool(args.cedar):
        parser.error("provide both --iedb and --cedar, or neither")
    paths = (
        {"iedb_tcell_full_v3.zip": args.iedb, "cedar_tcell_full_v3.zip": args.cedar}
        if args.iedb
        else pinned_sources()
    )
    result = run(paths)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
