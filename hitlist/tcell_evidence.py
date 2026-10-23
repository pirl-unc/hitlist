"""One source-faithful T-cell evidence generation appended to the v2 package.

The pinned IEDB and CEDAR ZIPs remain in the generation. Assay availability,
reported outcomes, antigens, source contributors and task eligibility have
different grains and identities. Only the existing bounded direct-peptide
policy produces training examples; all other assays remain source evidence.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import re
import shutil
import tempfile
import zipfile
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from .curation_yaml import load_curation_yaml
from .external_evidence import verify_external_evidence_package
from .mhc_evidence import _link_or_copy
from .provenance import file_digest
from .tcell_assays import (
    CATALOG_PATH,
    EXPORT_SCHEMA,
    HLA_TWO_FIELD,
    SOURCE_FIELDS,
    assay_family,
)
from .tcell_peptides import HUMAN_TAXON_IRI, QUALITATIVE_RESULTS, _restriction_status
from .tcell_training import MULTIMER_TASK, TASK, TRAINING_SCHEMA, _route

PACKAGE_SCHEMA_VERSION = 5
CATALOG_SCHEMA_VERSION = 6
SOURCE_NAMES = ("iedb_tcell_full_v3.zip", "cedar_tcell_full_v3.zip")
MEMBER = "tcell_full_v3.csv"
CANONICAL_PEPTIDE = re.compile(r"^[ACDEFGHIKLMNPQRSTVWY]{5,40}$")
FAMILY_COUNTS = {
    "soluble_factor_release": 396_243,
    "activation_or_proliferation": 103_103,
    "cytotoxicity": 24_962,
    "pmhc_multimer": 36_268,
    "other_tcell_assay": 17_953,
}
FIRST_FAILURE_COUNTS = {
    "nonhuman_epitope_origin": 511_628,
    "epitope_not_linear_peptide": 126,
    "noncanonical_or_unknown_sequence": 2_258,
    "antigen_not_linear_peptide": 4_723,
    "antigen_not_direct_epitope": 2_602,
    "unsupported_target_family": 1_167,
}
EXPECTED = {
    "n_iedb_assays": 578_529,
    "n_cedar_contributors": 157_263,
    "n_source_contributors": 735_792,
    "n_cedar_mutation_claims": 30_639,
    "n_direct_peptide_candidates": 56_025,
    "n_functional_candidates": 47_593,
    "n_multimer_candidates": 8_432,
    "n_training_examples": 55_844,
    "n_functional_examples": 47_421,
    "n_multimer_examples": 8_423,
    "n_training_exclusions": 181,
    "n_examples_with_candidate_single_hla": 4_054,
}


def _schema(*fields):
    return pa.schema([pa.field(name, type_) for name, type_ in fields])


SCHEMAS = {
    "assays": EXPORT_SCHEMA,
    "outcomes": _schema(
        ("measurement_id", pa.string()),
        ("source_assay_id", pa.string()),
        ("assay_method_raw", pa.string()),
        ("response_measured_raw", pa.string()),
        ("qualitative_result_raw", pa.string()),
        ("quantitative_raw", pa.string()),
        ("quantitative_value", pa.float64()),
        ("quantitative_status", pa.string()),
        ("measurement_unit_raw", pa.string()),
        ("measurement_inequality_raw", pa.string()),
        ("n_subjects_tested_raw", pa.string()),
        ("n_subjects_positive_raw", pa.string()),
        ("endpoint_status", pa.string()),
    ),
    "antigen_claims": _schema(
        ("source_assay_id", pa.string()),
        ("epitope_iri_raw", pa.string()),
        ("epitope_object_type_raw", pa.string()),
        ("epitope_name_raw", pa.string()),
        ("epitope_origin_iri_raw", pa.string()),
        ("epitope_modified_residues_raw", pa.string()),
        ("epitope_modifications_raw", pa.string()),
        ("assay_antigen_object_type_raw", pa.string()),
        ("assay_antigen_relation_raw", pa.string()),
        ("assay_antigen_name_raw", pa.string()),
        ("assay_antigen_origin_iri_raw", pa.string()),
        ("reported_mhc_name_raw", pa.string()),
        ("restriction_evidence_code_raw", pa.string()),
        ("reported_mhc_class_raw", pa.string()),
        ("host_mhc_present_raw", pa.string()),
    ),
    "source_contributors": _schema(
        ("source_record_id", pa.string()),
        ("source_assay_id", pa.string()),
        ("source_reference_id", pa.string()),
        ("source_dataset", pa.string()),
        ("source_asset", pa.string()),
        ("csv_data_row_1based", pa.int64()),
        ("assay_iri_raw", pa.string()),
        ("reference_iri_raw", pa.string()),
        ("epitope_iri_raw", pa.string()),
    ),
    "cedar_mutation_claims": _schema(
        ("claim_id", pa.string()),
        ("source_assay_id", pa.string()),
        ("source_record_id", pa.string()),
        ("predicate", pa.string()),
        ("value_raw", pa.string()),
        ("source_field", pa.string()),
        ("claim_status", pa.string()),
    ),
    "eligibility": _schema(
        ("source_assay_id", pa.string()),
        ("assay_family", pa.string()),
        ("direct_peptide_candidate", pa.bool_()),
        ("first_failure_reason", pa.string()),
        ("all_failure_reasons_json", pa.string()),
        ("task", pa.string()),
        ("training_status", pa.string()),
        ("training_exclusion_reason", pa.string()),
        ("example_id", pa.string()),
    ),
    "training_examples": TRAINING_SCHEMA,
}


class _Sink:
    def __init__(self, directory, name, *, batch_size=16_384):
        self.schema = SCHEMAS[name]
        self.writer = pq.ParquetWriter(
            directory / f"{name}.parquet", self.schema, compression="zstd"
        )
        self.batch_size = batch_size
        self.rows = []
        self.n_rows = 0

    def add(self, row):
        self.rows.append(row)
        if len(self.rows) >= self.batch_size:
            self.flush()

    def flush(self):
        if self.rows:
            self.writer.write_table(pa.Table.from_pylist(self.rows, schema=self.schema))
            self.n_rows += len(self.rows)
            self.rows.clear()

    def close(self):
        self.flush()
        self.writer.close()


@contextmanager
def _open_source(path):
    with zipfile.ZipFile(path) as archive:
        if archive.namelist() != [MEMBER]:
            raise ValueError(f"Unexpected T-cell ZIP contents: {path.name}")
        with archive.open(MEMBER) as raw:
            reader = csv.reader(io.TextIOWrapper(raw, encoding="utf-8-sig", newline=""))
            try:
                major, minor = next(reader), next(reader)
            except StopIteration as error:
                raise ValueError(f"Missing T-cell double header: {path.name}") from error
            headers = [f"{part} | {subpart}" for part, subpart in zip(major, minor)]
            if len(major) != len(minor) or len(headers) != len(set(headers)):
                raise ValueError(f"Invalid T-cell double header: {path.name}")
            yield headers, reader


def _source_id(iri, *, label):
    value = iri.rsplit("/", 1)[-1]
    if not iri or not value or value == iri:
        raise ValueError(f"Missing or invalid {label} IRI")
    return value


def _source_digest(row, positions):
    return hashlib.blake2b(
        json.dumps([row[index] for index in positions], ensure_ascii=False).encode(),
        digest_size=16,
    ).digest()


def _quantitative(raw):
    if not raw:
        return None, "unreported"
    try:
        value = float(raw)
    except ValueError:
        return None, "unparsed_source_text"
    if not math.isfinite(value):
        return None, "unparsed_source_text"
    return value, "source_reported"


def _row_values(row, positions):
    def raw(name):
        return row[positions[name]]

    def get(name):
        return raw(name).strip()

    return raw, get


def _source_checks(get, family):
    sequence = get("Epitope | Name")
    return (
        ("nonhuman_epitope_origin", get("Epitope | Source Organism IRI") == HUMAN_TAXON_IRI),
        ("epitope_not_linear_peptide", get("Epitope | Object Type") == "Linear peptide"),
        ("noncanonical_or_unknown_sequence", CANONICAL_PEPTIDE.fullmatch(sequence) is not None),
        (
            "modified_epitope",
            not get("Epitope | Modified residues") and not get("Epitope | Modifications"),
        ),
        ("antigen_not_linear_peptide", get("Assay Antigen | Object Type") == "Linear peptide"),
        ("antigen_not_direct_epitope", get("Assay Antigen | Epitope Relation") == "Epitope"),
        ("antigen_sequence_differs", get("Assay Antigen | Name") == sequence),
        (
            "antigen_not_human_origin",
            get("Assay Antigen | Source Organism IRI") == HUMAN_TAXON_IRI,
        ),
        (
            "qualitative_result_unavailable",
            get("Assay | Qualitative Measurement") in QUALITATIVE_RESULTS,
        ),
        (
            "unsupported_target_family",
            family
            in {
                "soluble_factor_release",
                "activation_or_proliferation",
                "cytotoxicity",
                "pmhc_multimer",
            },
        ),
        (
            "unsupported_multimer_readout",
            family != "pmhc_multimer" or get("Assay | Response measured") == "qualitative binding",
        ),
    )


_ANTIGEN_FIELDS = (
    "Epitope | IEDB IRI",
    "Epitope | Object Type",
    "Epitope | Name",
    "Epitope | Source Organism IRI",
    "Epitope | Modified residues",
    "Epitope | Modifications",
    "Assay Antigen | Object Type",
    "Assay Antigen | Epitope Relation",
    "Assay Antigen | Name",
    "Assay Antigen | Source Organism IRI",
    "MHC Restriction | Name",
    "MHC Restriction | Evidence Code",
    "MHC Restriction | Class",
    "Host | MHC Present",
)
_TYPED_SOURCE_FIELDS = frozenset(
    {
        *SOURCE_FIELDS.values(),
        *_ANTIGEN_FIELDS,
        "Assay | Method",
        "Assay | Response measured",
        "Assay | Qualitative Measurement",
        "Assay | Quantitative measurement",
        "Assay | Units",
        "Assay | Measurement Inequality",
        "Assay | Number of Subjects Tested",
        "Assay | Number of Subjects Positive",
        "Epitope | Mutation",
    }
)


def _field_disposition(iedb_headers, cedar_headers):
    result = {}
    for source, headers in (("IEDB", iedb_headers), ("CEDAR", cedar_headers)):
        rows = []
        for index, header in enumerate(headers, start=1):
            if header in _TYPED_SOURCE_FIELDS:
                disposition = (
                    "typed_cedar_claim" if header == "Epitope | Mutation" else "typed_projection"
                )
            elif header in {
                f"Assay ID | {source} IRI",
                f"Reference | {source} IRI",
                f"Epitope | {source} IRI",
            }:
                disposition = "typed_source_contributor"
            else:
                disposition = "source_only_in_embedded_csv"
            rows.append(
                {
                    "source_column_1based": index,
                    "source_header": header,
                    "disposition": disposition,
                    "source_asset": f"tcell/assets/{source.lower()}_tcell_full_v3.zip",
                }
            )
        result[source] = rows
    return result


def _write_tcell_rows(assets, typed):
    sinks = {name: _Sink(typed, name) for name in SCHEMAS}
    family_counts = Counter()
    first_failures = Counter()
    candidate_families = Counter()
    examples_by_task = Counter()
    exclusion_reasons = Counter()
    n_cedar_mutation_on_training = 0
    try:
        with (
            _open_source(assets / SOURCE_NAMES[0]) as (iedb_headers, iedb_reader),
            _open_source(assets / SOURCE_NAMES[1]) as (cedar_headers, cedar_reader),
        ):
            shared = sorted(set(iedb_headers) & set(cedar_headers))
            if len(iedb_headers) != 161 or len(cedar_headers) != 162 or len(shared) != 158:
                raise ValueError("Pinned T-cell source columns changed")
            if set(cedar_headers) - set(iedb_headers) != {
                "Assay ID | CEDAR IRI",
                "Reference | CEDAR IRI",
                "Epitope | CEDAR IRI",
                "Epitope | Mutation",
            }:
                raise ValueError("Pinned CEDAR-only T-cell fields changed")
            (typed.parent / "source_field_disposition.json").write_text(
                json.dumps(_field_disposition(iedb_headers, cedar_headers), indent=2) + "\n"
            )
            cpos = {name: index for index, name in enumerate(cedar_headers)}
            ipos = {name: index for index, name in enumerate(iedb_headers)}
            cedar_shared_positions = [cpos[name] for name in shared]
            iedb_shared_positions = [ipos[name] for name in shared]
            cedar = {}
            for row_number, row in enumerate(cedar_reader, start=1):
                if len(row) != len(cedar_headers):
                    raise ValueError(f"Malformed CEDAR T-cell row: {row_number}")
                assay_iri = row[cpos["Assay ID | CEDAR IRI"]]
                reference_iri = row[cpos["Reference | CEDAR IRI"]]
                assay_id = _source_id(assay_iri, label="CEDAR assay")
                reference_id = _source_id(reference_iri, label="CEDAR reference")
                if assay_id in cedar:
                    raise ValueError(f"Repeated CEDAR T-cell assay ID: {assay_id}")
                source_record_id = f"cedar:tcell:r{row_number}"
                mutation = row[cpos["Epitope | Mutation"]].strip()
                cedar[assay_id] = (
                    reference_id,
                    _source_digest(row, cedar_shared_positions),
                    mutation,
                )
                sinks["source_contributors"].add(
                    {
                        "source_record_id": source_record_id,
                        "source_assay_id": assay_id,
                        "source_reference_id": reference_id,
                        "source_dataset": "CEDAR",
                        "source_asset": f"tcell/assets/{SOURCE_NAMES[1]}",
                        "csv_data_row_1based": row_number,
                        "assay_iri_raw": assay_iri,
                        "reference_iri_raw": reference_iri,
                        "epitope_iri_raw": row[cpos["Epitope | CEDAR IRI"]],
                    }
                )
                if mutation:
                    sinks["cedar_mutation_claims"].add(
                        {
                            "claim_id": f"{source_record_id}:mutation",
                            "source_assay_id": assay_id,
                            "source_record_id": source_record_id,
                            "predicate": "epitope_mutation",
                            "value_raw": mutation,
                            "source_field": "Epitope | Mutation",
                            "claim_status": "cedar_source_reported",
                        }
                    )
            if len(cedar) != EXPECTED["n_cedar_contributors"]:
                raise ValueError("Pinned CEDAR assay count changed")

            seen_iedb = set()
            for row_number, row in enumerate(iedb_reader, start=1):
                if len(row) != len(iedb_headers):
                    raise ValueError(f"Malformed IEDB T-cell row: {row_number}")
                raw, get = _row_values(row, ipos)
                assay_iri = raw("Assay ID | IEDB IRI")
                reference_iri = raw("Reference | IEDB IRI")
                assay_id = _source_id(assay_iri, label="IEDB assay")
                reference_id = _source_id(reference_iri, label="IEDB reference")
                if assay_id in seen_iedb:
                    raise ValueError(f"Repeated IEDB T-cell assay ID: {assay_id}")
                seen_iedb.add(assay_id)
                matched = cedar.pop(assay_id, None)
                if matched is not None and matched[:2] != (
                    reference_id,
                    _source_digest(row, iedb_shared_positions),
                ):
                    raise ValueError(f"IEDB/CEDAR full-field mismatch: {assay_id}")
                family = assay_family(get("Assay | Method"), get("Assay | Response measured"))
                family_counts[family] += 1
                mhc_name = get("MHC Restriction | Name")
                resolution = (
                    "missing"
                    if not mhc_name
                    else "single_hla_two_field_name"
                    if HLA_TWO_FIELD.fullmatch(mhc_name)
                    else "other_reported_name"
                )
                metadata = {
                    "source_assay_id": assay_id,
                    "source_reference_id": reference_id,
                    **{name: get(source) for name, source in SOURCE_FIELDS.items()},
                    "assay_family": family,
                    "restriction_name_resolution": resolution,
                    "cedar_present": matched is not None,
                    "specimen_identity_status": "not_in_export",
                    "receptor_identity_status": "not_in_export",
                }
                sinks["assays"].add(metadata)
                sinks["source_contributors"].add(
                    {
                        "source_record_id": f"iedb:tcell:r{row_number}",
                        "source_assay_id": assay_id,
                        "source_reference_id": reference_id,
                        "source_dataset": "IEDB",
                        "source_asset": f"tcell/assets/{SOURCE_NAMES[0]}",
                        "csv_data_row_1based": row_number,
                        "assay_iri_raw": assay_iri,
                        "reference_iri_raw": reference_iri,
                        "epitope_iri_raw": raw("Epitope | IEDB IRI"),
                    }
                )
                number, number_status = _quantitative(get("Assay | Quantitative measurement"))
                sinks["outcomes"].add(
                    {
                        "measurement_id": f"iedb:tcell:{assay_id}:reported_outcome",
                        "source_assay_id": assay_id,
                        "assay_method_raw": raw("Assay | Method"),
                        "response_measured_raw": raw("Assay | Response measured"),
                        "qualitative_result_raw": raw("Assay | Qualitative Measurement"),
                        "quantitative_raw": raw("Assay | Quantitative measurement"),
                        "quantitative_value": number,
                        "quantitative_status": number_status,
                        "measurement_unit_raw": raw("Assay | Units"),
                        "measurement_inequality_raw": raw("Assay | Measurement Inequality"),
                        "n_subjects_tested_raw": raw("Assay | Number of Subjects Tested"),
                        "n_subjects_positive_raw": raw("Assay | Number of Subjects Positive"),
                        "endpoint_status": "source_readout_not_model_target",
                    }
                )
                sinks["antigen_claims"].add(
                    {
                        "source_assay_id": assay_id,
                        "epitope_iri_raw": raw("Epitope | IEDB IRI"),
                        "epitope_object_type_raw": raw("Epitope | Object Type"),
                        "epitope_name_raw": raw("Epitope | Name"),
                        "epitope_origin_iri_raw": raw("Epitope | Source Organism IRI"),
                        "epitope_modified_residues_raw": raw("Epitope | Modified residues"),
                        "epitope_modifications_raw": raw("Epitope | Modifications"),
                        "assay_antigen_object_type_raw": raw("Assay Antigen | Object Type"),
                        "assay_antigen_relation_raw": raw("Assay Antigen | Epitope Relation"),
                        "assay_antigen_name_raw": raw("Assay Antigen | Name"),
                        "assay_antigen_origin_iri_raw": raw("Assay Antigen | Source Organism IRI"),
                        "reported_mhc_name_raw": raw("MHC Restriction | Name"),
                        "restriction_evidence_code_raw": raw("MHC Restriction | Evidence Code"),
                        "reported_mhc_class_raw": raw("MHC Restriction | Class"),
                        "host_mhc_present_raw": raw("Host | MHC Present"),
                    }
                )
                failures = [reason for reason, passes in _source_checks(get, family) if not passes]
                first_failure = failures[0] if failures else None
                if first_failure:
                    first_failures[first_failure] += 1
                candidate = first_failure is None
                task = None
                example = None
                exclusion_reason = None
                if candidate:
                    candidate_families[family] += 1
                    is_multimer = family == "pmhc_multimer"
                    task = MULTIMER_TASK if is_multimer else TASK
                    evidence = {
                        **metadata,
                        "peptide_sequence": get("Epitope | Name"),
                        "reported_mhc_name": mhc_name,
                        "mhc_assignment_status": _restriction_status(metadata),
                        "response_observed" if not is_multimer else "multimer_binding_observed": (
                            get("Assay | Qualitative Measurement") != "Negative"
                        ),
                    }
                    example, exclusion_reason = _route(
                        evidence,
                        task=task,
                        observed_field=(
                            "multimer_binding_observed" if is_multimer else "response_observed"
                        ),
                    )
                    if example is not None:
                        sinks["training_examples"].add(example)
                        examples_by_task[task] += 1
                        if matched is not None and matched[2]:
                            n_cedar_mutation_on_training += 1
                    else:
                        exclusion_reasons[exclusion_reason] += 1
                sinks["eligibility"].add(
                    {
                        "source_assay_id": assay_id,
                        "assay_family": family,
                        "direct_peptide_candidate": candidate,
                        "first_failure_reason": first_failure,
                        "all_failure_reasons_json": json.dumps(failures),
                        "task": task,
                        "training_status": (
                            "accepted"
                            if example is not None
                            else "excluded"
                            if candidate
                            else "not_candidate"
                        ),
                        "training_exclusion_reason": exclusion_reason,
                        "example_id": example["example_id"] if example else None,
                    }
                )
            if cedar:
                raise ValueError(f"CEDAR T-cell assay IDs absent from IEDB: {len(cedar)}")
    finally:
        for sink in sinks.values():
            sink.close()

    reconciliation = {
        "n_iedb_assays": sinks["assays"].n_rows,
        "n_cedar_contributors": sinks["source_contributors"].n_rows - sinks["assays"].n_rows,
        "n_source_contributors": sinks["source_contributors"].n_rows,
        "n_cedar_mutation_claims": sinks["cedar_mutation_claims"].n_rows,
        "n_direct_peptide_candidates": sum(candidate_families.values()),
        "n_functional_candidates": sum(
            n for family, n in candidate_families.items() if family != "pmhc_multimer"
        ),
        "n_multimer_candidates": candidate_families["pmhc_multimer"],
        "n_training_examples": sinks["training_examples"].n_rows,
        "n_functional_examples": examples_by_task[TASK],
        "n_multimer_examples": examples_by_task[MULTIMER_TASK],
        "n_training_exclusions": sum(exclusion_reasons.values()),
        "n_examples_with_candidate_single_hla": 0,
        "n_cedar_mutation_on_training_examples": n_cedar_mutation_on_training,
        "family_counts": dict(sorted(family_counts.items())),
        "first_failure_counts": dict(sorted(first_failures.items())),
        "candidate_family_counts": dict(sorted(candidate_families.items())),
        "training_exclusion_counts": dict(sorted(exclusion_reasons.items())),
    }
    for batch in pq.ParquetFile(typed / "training_examples.parquet").iter_batches(
        columns=["mhc_allele"], batch_size=16_384
    ):
        reconciliation["n_examples_with_candidate_single_hla"] += sum(
            value is not None for value in batch.column(0).to_pylist()
        )
    return reconciliation


def _source_specs(catalog_path):
    catalog = load_curation_yaml(catalog_path)
    if catalog.get("observed") != "2026-10-20" or set(catalog["files"]) != set(SOURCE_NAMES):
        raise ValueError("Unsupported T-cell source revision")
    return catalog


def _check_sources(source_paths, catalog):
    if set(source_paths) != set(SOURCE_NAMES):
        raise ValueError("Both pinned T-cell source ZIPs are required")
    paths = {name: Path(source_paths[name]) for name in SOURCE_NAMES}
    for name, path in paths.items():
        spec = catalog["files"][name]
        if file_digest(path) != {"sha256": spec["sha256"], "size_bytes": spec["size"]}:
            raise ValueError(f"Pinned T-cell source asset mismatch: {name}")
    return paths


def _validate_reconciliation(reconciliation):
    for key, expected in EXPECTED.items():
        if reconciliation.get(key) != expected:
            raise ValueError(f"Pinned T-cell count mismatch: {key}")
    if reconciliation.get("family_counts") != FAMILY_COUNTS:
        raise ValueError("Pinned T-cell family counts changed")
    if reconciliation.get("first_failure_counts") != FIRST_FAILURE_COUNTS:
        raise ValueError("Pinned T-cell eligibility funnel changed")
    if reconciliation.get("training_exclusion_counts") != {
        "qualitative_subject_count_disagreement": 181
    }:
        raise ValueError("Pinned T-cell exclusion policy changed")
    if reconciliation.get("n_cedar_mutation_on_training_examples") != 17:
        raise ValueError("Pinned CEDAR-only mutation training overlap changed")


def verify_tcell_evidence_package(directory, *, manifest_path="manifest.json"):
    """Verify a T-cell generation, including one embedded in its successor."""
    directory = Path(directory)
    location = Path(manifest_path)
    if location.is_absolute() or ".." in location.parts or not location.parts:
        raise ValueError("Invalid T-cell package manifest path")
    manifest = json.loads((directory / location).read_text())
    if manifest.get("schema_version") != PACKAGE_SCHEMA_VERSION:
        raise ValueError("Unsupported T-cell package schema")
    if manifest.get("catalog_schema_version") != CATALOG_SCHEMA_VERSION:
        raise ValueError("Unsupported T-cell catalog schema")
    files = manifest.get("files", {})
    if location == Path("manifest.json"):
        actual = {
            str(path.relative_to(directory))
            for path in directory.rglob("*")
            if path.is_file() and path != directory / "manifest.json"
        }
        if set(files) != actual:
            raise ValueError("T-cell package file inventory mismatch")
    for relative, expected in files.items():
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts or file_digest(directory / path) != expected:
            raise ValueError(f"T-cell package artifact mismatch: {relative}")
    if files["manifests/external.json"]["sha256"] != manifest.get("parent_manifest_sha256"):
        raise ValueError("T-cell predecessor manifest mismatch")
    parent = verify_external_evidence_package(directory, manifest_path="manifests/external.json")
    if parent["catalog_schema_version"] != 5:
        raise ValueError("Unsupported predecessor external catalog")
    catalog = _source_specs(directory / "tcell" / "source_receipt.yaml")
    if manifest.get("source_asset_receipt_sha256") != files["tcell/source_receipt.yaml"]["sha256"]:
        raise ValueError("T-cell source receipt mismatch")
    _check_sources({name: directory / "tcell" / "assets" / name for name in SOURCE_NAMES}, catalog)
    inventory = json.loads((directory / "tcell" / "source_field_disposition.json").read_text())
    if {key: len(value) for key, value in inventory.items()} != {"IEDB": 161, "CEDAR": 162}:
        raise ValueError("T-cell source field disposition mismatch")
    reconciliation = manifest["reconciliation"]
    _validate_reconciliation(reconciliation)
    for name, schema in SCHEMAS.items():
        parquet = pq.ParquetFile(directory / "tcell" / "typed" / f"{name}.parquet")
        expected = {
            "assays": reconciliation["n_iedb_assays"],
            "outcomes": reconciliation["n_iedb_assays"],
            "antigen_claims": reconciliation["n_iedb_assays"],
            "source_contributors": reconciliation["n_source_contributors"],
            "cedar_mutation_claims": reconciliation["n_cedar_mutation_claims"],
            "eligibility": reconciliation["n_iedb_assays"],
            "training_examples": reconciliation["n_training_examples"],
        }[name]
        if parquet.schema_arrow != schema or parquet.metadata.num_rows != expected:
            raise ValueError(f"T-cell typed relation mismatch: {name}")
    if manifest.get("source_registry") != {
        "IEDB": {
            "source_asset": f"tcell/assets/{SOURCE_NAMES[0]}",
            "n_assays": reconciliation["n_iedb_assays"],
            "license_status": "unverified",
        },
        "CEDAR": {
            "source_asset": f"tcell/assets/{SOURCE_NAMES[1]}",
            "n_contributors": reconciliation["n_cedar_contributors"],
            "n_mutation_claims": reconciliation["n_cedar_mutation_claims"],
            "license_status": "unverified",
        },
    }:
        raise ValueError("T-cell source registry mismatch")
    return manifest


def write_tcell_evidence_package(external_package, destination, *, iedb_path, cedar_path):
    """Append pinned T-cell evidence and bounded targets atomically to v2."""
    external_package = Path(external_package)
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f"T-cell evidence package already exists: {destination}")
    catalog = _source_specs(CATALOG_PATH)
    sources = _check_sources({SOURCE_NAMES[0]: iedb_path, SOURCE_NAMES[1]: cedar_path}, catalog)
    parent = verify_external_evidence_package(external_package)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    try:
        for relative in parent["files"]:
            target = staging / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            _link_or_copy(external_package / relative, target)
        (staging / "manifests").mkdir(exist_ok=True)
        _link_or_copy(external_package / "manifest.json", staging / "manifests/external.json")
        tcell = staging / "tcell"
        assets = tcell / "assets"
        typed = tcell / "typed"
        assets.mkdir(parents=True)
        typed.mkdir()
        shutil.copy2(CATALOG_PATH, tcell / "source_receipt.yaml")
        for name, path in sources.items():
            shutil.copy2(path, assets / name)
        reconciliation = _write_tcell_rows(assets, typed)
        _validate_reconciliation(reconciliation)
        _check_sources({name: assets / name for name in SOURCE_NAMES}, catalog)
        verify_external_evidence_package(external_package)
        files = {
            str(path.relative_to(staging)): file_digest(path)
            for path in sorted(staging.rglob("*"))
            if path.is_file()
        }
        from .version import __version__

        manifest = {
            "schema_version": PACKAGE_SCHEMA_VERSION,
            "catalog_schema_version": CATALOG_SCHEMA_VERSION,
            "hitlist_version": __version__,
            "parent_manifest_sha256": files["manifests/external.json"]["sha256"],
            "source_asset_receipt_sha256": files["tcell/source_receipt.yaml"]["sha256"],
            "source_registry": {
                "IEDB": {
                    "source_asset": f"tcell/assets/{SOURCE_NAMES[0]}",
                    "n_assays": reconciliation["n_iedb_assays"],
                    "license_status": "unverified",
                },
                "CEDAR": {
                    "source_asset": f"tcell/assets/{SOURCE_NAMES[1]}",
                    "n_contributors": reconciliation["n_cedar_contributors"],
                    "n_mutation_claims": reconciliation["n_cedar_mutation_claims"],
                    "license_status": "unverified",
                },
            },
            "reconciliation": reconciliation,
            "files": files,
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        verify_tcell_evidence_package(staging)
        staging.rename(destination)
        return destination / "manifest.json"
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
