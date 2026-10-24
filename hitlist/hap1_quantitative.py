"""Original HAP1 elution/proteomics quantities and their source-scoped context.

This is an evidence package, not an endpoint-specific training recipe. Missing
supplement cells and deposited quantities remain distinct source claims.
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
from pathlib import Path

import openpyxl
import pyarrow.parquet as pq

from .evidence_snapshot import snapshot_file
from .mapp_evidence import _schema, _Sink
from .provenance import file_digest
from .sequence_reference_evidence import exact_occurrences

PACKAGE_SCHEMA_VERSION = 20
CATALOG_SCHEMA_VERSION = 25
PACKAGE_FAMILY = "hap1_quantitative"
REVISION = "shapiro2025_hap1_quantitative_v1"
ROOT = Path("hap1")
DATA = Path(__file__).parent / "data/evidence_revisions"
ASSETS = DATA / "hap1_assets.json"
CONTRACT = DATA / "hap1_contract.json"
COMMON = ("source_record_id", "source_file", "source_locator", "source_row_json")
SCHEMAS = {
    "assets": _schema(
        ("source_asset_id", "source_file", "asset_sha256", "source_asset_url", "source_role"),
        integers=("size_bytes",),
    ),
    "source_records": _schema((*COMMON, "source_role"), integers=("source_row_index",)),
    "contexts": _schema(
        (
            "context_id",
            "condition_id",
            "control_condition_id",
            "condition_text",
            "condition_knockout_genes",
            "component_state_status",
            "organism_name",
            "organism_taxon_id",
            "cell_line_name",
            "mhc_class",
            "mhc_genotype",
            "genotype_status",
            "measured_pool",
            "digestion_enzyme",
            "instrument",
            "normalization_policy",
            "protocol_evidence",
            "context_scope",
            "source_row_json",
        )
    ),
    "samples": _schema(
        (
            "sample_id",
            "context_id",
            "condition_id",
            "sample_label_raw",
            "modality",
            "biological_replicate_raw",
            "specimen_id",
            "aliquot_id",
            "clone_id",
            "material_link_precision",
            "control_link_precision",
        )
    ),
    "runs": _schema(
        (
            "run_id",
            "sample_id",
            "context_id",
            "source_record_id",
            "source_run_raw",
            "acquisition_mode",
            "technical_replicate_raw",
            "run_link_status",
            "deposit_aliases_json",
            "deposit_link_status",
        )
    ),
    "quantities": _schema(
        (
            "measurement_id",
            "source_record_id",
            "sample_id",
            "context_id",
            "source_field",
            "peptide_sequence",
            "parent_label_raw",
            "quantity_kind",
            "quantitative_raw",
            "value_status",
            "measurement_unit",
            "quantity_scale",
            "eligibility_status",
        ),
        numbers=("quantitative_value",),
    ),
    "precursors": _schema(
        (
            *COMMON,
            "run_id",
            "sample_id",
            "context_id",
            "peptide_sequence",
            "peptide_modifications_raw",
            "parent_label_raw",
            "precursor_id_raw",
            "precursor_charge_raw",
            "raw_quantity_text",
            "normalized_quantity_text",
            "raw_value_status",
            "normalized_value_status",
            "confidence_json",
        ),
        integers=("source_row_index",),
        numbers=("raw_quantity", "normalized_quantity"),
    ),
    "protein_quantities": _schema(
        (
            "measurement_id",
            "source_record_id",
            "sample_id",
            "context_id",
            "run_id",
            "source_field",
            "parent_label_raw",
            "quantity_kind",
            "quantitative_raw",
            "value_status",
            "measurement_unit",
            "quantity_scale",
        ),
        numbers=("quantitative_value",),
    ),
    "references": _schema(
        (
            "reference_record_id",
            "accession_raw",
            "header_raw",
            "reference_sequence",
            "sequence_sha256",
            "reference_role",
            "organism_taxon_id",
            "source_locator",
        )
    ),
    "occurrences": _schema(
        (
            "occurrence_id",
            "source_record_id",
            "peptide_sequence",
            "parent_label_raw",
            "reference_record_id",
            "mapping_status",
            "source_claim_status",
        ),
        integers=("start_1based", "end_1based"),
    ),
}


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _id(kind, *parts):
    return f"hap1:{kind}:" + hashlib.sha256(_json(parts).encode()).hexdigest()


def _number(raw):
    """Do not turn missing values or report zeros into censoring bounds."""
    if raw in (None, "", "NA"):
        return None, "source_missing_" + ("empty" if raw in (None, "") else "NA")
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None, "unparsed_source_text"
    if not math.isfinite(value):
        return None, "source_nonfinite"
    return (
        value,
        "source_positive" if value > 0 else "source_zero" if value == 0 else "source_negative",
    )


def _csv_rows(stream, *, delimiter=","):
    reader = csv.DictReader(stream, delimiter=delimiter)
    if not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames):
        raise ValueError("Missing or duplicate HAP1 source columns")
    for number, row in enumerate(reader, 2):
        if None in row or any(value is None for value in row.values()):
            raise ValueError("Malformed HAP1 source row")
        yield number, row


def _record(sinks, file, member, index, row, role):
    locator = f"{file}!{member}:row:{index}"
    common = {
        "source_record_id": _id("row", locator),
        "source_file": file,
        "source_locator": locator,
        "source_row_index": index,
        "source_row_json": _json(row),
    }
    sinks["source_records"].add({**common, "source_role": role})
    return common


def _candidate_aliases(name, deposited):
    if name in deposited:
        return [name], "literal_filename_match_not_specimen_pair"
    matches = sorted(
        value for value in deposited if value.replace("-", "_") == name.replace("-", "_")
    )
    status = (
        "unique_punctuation_candidate"
        if len(matches) == 1
        else "ambiguous_candidates"
        if matches
        else "not_in_inventory"
    )
    return matches, status


def _sample_id(modality, condition, replicate):
    return f"hap1:{modality}:{condition}_{replicate}"


def _reference_records(stream, locator):
    header, parts, index = None, [], 0

    def emit():
        sequence = "".join(parts)
        if not header or not sequence or not re.fullmatch("[A-Z]+", sequence):
            raise ValueError("Malformed original HAP1 FASTA record")
        token = header.split()[0]
        taxonomy = re.search(r"(?:^| )OX=(\d+)(?: |$)", header)
        role = (
            "reverse_decoy"
            if token.startswith("rev_")
            else "spiked_standard"
            if "Biognosys" in token or "iRT" in token
            else "human_target"
            if token.startswith("sp|") and taxonomy and taxonomy[1] == "9606"
            else "other_or_contaminant_reference"
        )
        return {
            "reference_record_id": _id("reference", locator, index),
            "accession_raw": token,
            "header_raw": header,
            "reference_sequence": sequence,
            "sequence_sha256": hashlib.sha256(sequence.encode()).hexdigest(),
            "reference_role": role,
            "organism_taxon_id": taxonomy[1] if taxonomy else None,
            "source_locator": f"{locator}:entry:{index}",
        }

    for line in stream:
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if header is not None:
                yield emit()
            index += 1
            header, parts = line[1:], []
        else:
            if header is None:
                raise ValueError("FASTA sequence precedes header")
            parts.append(line)
    if header is not None:
        yield emit()


def _compile(assets, typed, receipt, contract):
    """Stream complete source relations; return explicit reconciliation counts."""
    typed.mkdir(parents=True)
    sinks = {name: _Sink(typed, name, schema=schema) for name, schema in SCHEMAS.items()}
    counts = Counter()
    samples, runs, raw_runs, contexts = {}, {}, {}, {}
    peptide_rows = []
    try:
        for spec in receipt["assets"]:
            sinks["assets"].add(
                {
                    "source_asset_id": _id("asset", spec["filename"]),
                    "source_file": spec["filename"],
                    "asset_sha256": spec["sha256"],
                    "size_bytes": spec["size_bytes"],
                    "source_asset_url": spec["url"],
                    "source_role": "original_source_asset",
                }
            )
        for condition, reviewed in contract["contexts"].items():
            for modality, protocol in contract["protocols"].items():
                context_id = f"{reviewed['condition_id']}:{modality}"
                contexts[modality, condition] = context_id
                sinks["contexts"].add(
                    {
                        "context_id": context_id,
                        "condition_id": reviewed["condition_id"],
                        "control_condition_id": contract["contexts"]["wt"]["condition_id"],
                        "condition_text": reviewed["condition"],
                        "condition_knockout_genes": reviewed["condition_knockout_genes"],
                        "component_state_status": contract["context_status"],
                        "organism_name": contract["organism_name"],
                        "organism_taxon_id": contract["organism_taxon_id"],
                        "cell_line_name": contract["cell_line_name"],
                        "mhc_class": reviewed["mhc_class"],
                        "mhc_genotype": reviewed["mhc"],
                        "genotype_status": "supplied_cell_line_allele_set_not_peptide_restriction",
                        "instrument": contract["instrument"],
                        **protocol,
                        "context_scope": "study_protocol_and_condition_not_measured_activity",
                        "source_row_json": _json(reviewed),
                    }
                )
        deposited = set()
        for name in ("files-page0.json", "files-page1.json"):
            rows = json.loads((assets / name).read_text())
            for row in rows:
                if row["fileName"].lower().endswith(".raw"):
                    deposited.add(row["fileName"][:-4])

        def register_run(name, mode, common=None, row=None):
            modality = "proteomics" if mode == "proteomics" else "elution"
            key = modality, name
            if key in runs:
                raise ValueError(f"Duplicate declared HAP1 run: {key}")
            sample_id = context_id = None
            if row is not None:
                condition = str(row.get("knockout_condition", row.get("ko_cond")))
                replicate = str(row.get("biological_replicate", row.get("BR")))
                context_id = contexts[modality, condition]
                sample_id = _sample_id(modality, condition, replicate)
                samples[sample_id] = {
                    "sample_id": sample_id,
                    "context_id": context_id,
                    "condition_id": contract["contexts"][condition]["condition_id"],
                    "sample_label_raw": f"{condition}_{replicate}",
                    "modality": modality,
                    "biological_replicate_raw": replicate,
                    "specimen_id": None,
                    "aliquot_id": None,
                    "clone_id": None,
                    "material_link_precision": "declared_modality_condition_biological_replicate",
                    "control_link_precision": "WT_condition_only_no_exact_specimen_pair",
                }
            aliases, alias_status = _candidate_aliases(name, deposited)
            technical = row.get("technical_replicate", row.get("TR")) if row else None
            run = {
                "run_id": _id("run", modality, name),
                "sample_id": sample_id,
                "context_id": context_id,
                "source_record_id": common["source_record_id"] if common else None,
                "source_run_raw": name,
                "acquisition_mode": "DDA" if mode in ("proteomics", "DDA") else "DIA",
                "technical_replicate_raw": str(technical) if technical is not None else None,
                "run_link_status": "declared_source_workbook" if row else "unmapped_deposited_run",
                "deposit_aliases_json": _json(aliases),
                "deposit_link_status": alias_status,
            }
            runs[key] = run
            raw_runs[key] = run
            sinks["runs"].add(run)
            return run

        with (
            zipfile.ZipFile(assets / "supplements.zip") as supplements,
            zipfile.ZipFile(assets / "Search.zip") as search,
        ):
            for archive_name, archive in (("supplements.zip", supplements), ("Search.zip", search)):
                for entry in archive.infolist():
                    sinks["assets"].add(
                        {
                            "source_asset_id": _id("member", archive_name, entry.filename),
                            "source_file": f"{archive_name}!{entry.filename}",
                            "size_bytes": entry.file_size,
                            "source_role": "archive_member_preserved_in_original_bytes",
                        }
                    )
            for archive_name, archive, member in (
                ("supplements.zip", supplements, "mmc5.xlsx"),
                ("Search.zip", search, "Raw_file_list.xlsx"),
            ):
                book = openpyxl.load_workbook(
                    io.BytesIO(archive.read(member)), read_only=True, data_only=False
                )
                for sheet in book:
                    iterator = sheet.iter_rows(values_only=True)
                    headers = next(iterator)
                    if None in headers or len(set(headers)) != len(headers):
                        raise ValueError("Malformed HAP1 sample workbook header")
                    for index, values in enumerate(iterator, 2):
                        row = dict(zip(headers, values, strict=True))
                        common = _record(
                            sinks,
                            archive_name,
                            f"{member}/{sheet.title}",
                            index,
                            row,
                            "source_sample_workbook",
                        )
                        if archive_name == "supplements.zip":
                            mode = {
                                "immunopep_library_generation": "DDA",
                                "immunopep_library_search": "DIA",
                                "proteomics": "proteomics",
                            }[sheet.title]
                            register_run(row["raw_file"], mode, common, row)
                book.close()
            with (assets / "samples.sdrf.tsv").open() as stream:
                reader = csv.reader(stream, delimiter="\t")
                headers = next(reader)
                for index, values in enumerate(reader, 2):
                    if len(headers) != len(values):
                        raise ValueError("Malformed positional SDRF row")
                    _record(
                        sinks,
                        "samples.sdrf.tsv",
                        "",
                        index,
                        {"headers": headers, "values": values},
                        "community_sample_claim",
                    )
            for member in ("mmc6.csv", "mmc7.csv", "mmc8.csv"):
                with io.TextIOWrapper(supplements.open(member), encoding="utf-8-sig") as stream:
                    for index, row in _csv_rows(stream):
                        common = _record(
                            sinks, "supplements.zip", member, index, row, "original_supplement_row"
                        )
                        peptide = row.get("peptide")
                        parent = row.get("mapped_proteins", row.get("protein_id"))
                        if peptide:
                            peptide_rows.append((common["source_record_id"], peptide, parent))
                        for field, raw in row.items():
                            if field in ("peptide", "mapped_proteins", "protein_id"):
                                continue
                            match = re.fullmatch(
                                r"(.+)_([123])_?\.(sampling_density|intensity|protein_coverage_proteomics)",
                                field,
                            )
                            if match is None:
                                raise ValueError(f"Unrecognized source quantity column: {field}")
                            condition, replicate, suffix = match.groups()
                            modality = (
                                "elution"
                                if peptide or suffix == "sampling_density"
                                else "proteomics"
                            )
                            sample_id = _sample_id(modality, condition, replicate)
                            if sample_id not in samples:
                                raise ValueError(
                                    f"Quantity sample absent from workbook: {sample_id}"
                                )
                            kind = (
                                "normalized_peptide_intensity"
                                if member == "mmc6.csv"
                                else "unreplayed_noncanonical_quantity"
                                if member == "mmc7.csv"
                                else "protein_MaxLFQ_intensity"
                                if suffix == "intensity"
                                else "protein_" + suffix
                            )
                            value, status = _number(raw)
                            counts[f"{member}:{status}:n_cells"] += 1
                            sinks["quantities"].add(
                                {
                                    "measurement_id": _id(
                                        "cell", common["source_record_id"], field
                                    ),
                                    "source_record_id": common["source_record_id"],
                                    "sample_id": sample_id,
                                    "context_id": samples[sample_id]["context_id"],
                                    "source_field": field,
                                    "peptide_sequence": peptide,
                                    "parent_label_raw": parent,
                                    "quantity_kind": kind,
                                    "quantitative_raw": raw,
                                    "quantitative_value": value,
                                    "value_status": status,
                                    "measurement_unit": "source_arbitrary_units"
                                    if "intensity" in kind
                                    else "source_defined_unresolved_unit",
                                    "quantity_scale": kind,
                                    "eligibility_status": "deferred_endpoint_recipe"
                                    if member != "mmc7.csv"
                                    else "deferred_noncanonical_replay",
                                }
                            )
            with io.TextIOWrapper(
                search.open("Search_results_DIA.tsv"), encoding="utf-8-sig"
            ) as stream:
                for index, row in _csv_rows(stream, delimiter="\t"):
                    name = row["Run"]
                    run = raw_runs.get(("elution", name))
                    if run is None:
                        run = register_run(name, "DIA")
                    locator = f"Search.zip!Search_results_DIA.tsv:row:{index}"
                    raw, raw_status = _number(row["Precursor.Quantity"])
                    normalized, normalized_status = _number(row["Precursor.Normalised"])
                    if run["sample_id"] is None:
                        counts["n_unmapped_precursor_rows"] += 1
                    sinks["precursors"].add(
                        {
                            "source_record_id": _id("row", locator),
                            "source_file": "Search.zip",
                            "source_locator": locator,
                            "source_row_index": index,
                            "source_row_json": _json(row),
                            "run_id": run["run_id"],
                            "sample_id": run["sample_id"],
                            "context_id": run["context_id"],
                            "peptide_sequence": row["Stripped.Sequence"],
                            "peptide_modifications_raw": row["Modified.Sequence"],
                            "parent_label_raw": row["Protein.Ids"],
                            "precursor_id_raw": row["Precursor.Id"],
                            "precursor_charge_raw": row["Precursor.Charge"],
                            "raw_quantity": raw,
                            "raw_quantity_text": row["Precursor.Quantity"],
                            "normalized_quantity": normalized,
                            "normalized_quantity_text": row["Precursor.Normalised"],
                            "raw_value_status": raw_status,
                            "normalized_value_status": normalized_status,
                            "confidence_json": _json(
                                {
                                    key: value
                                    for key, value in row.items()
                                    if "Q.Value" in key or "PEP" in key
                                }
                            ),
                        }
                    )
            with io.TextIOWrapper(
                search.open("search_results_proteomics.tsv"), encoding="utf-8-sig"
            ) as stream:
                for index, row in _csv_rows(stream, delimiter="\t"):
                    common = _record(
                        sinks,
                        "Search.zip",
                        "search_results_proteomics.tsv",
                        index,
                        row,
                        "deposited_protein_report",
                    )
                    for field, raw in row.items():
                        suffix = next(
                            (
                                suffix
                                for suffix in (" MaxLFQ Intensity", " Intensity")
                                if field.endswith(suffix)
                            ),
                            None,
                        )
                        if suffix is None:
                            continue
                        name = field[: -len(suffix)]
                        run = raw_runs.get(("proteomics", name))
                        if run is None:
                            run = register_run(name, "proteomics")
                        value, status = _number(raw)
                        sinks["protein_quantities"].add(
                            {
                                "measurement_id": _id("cell", common["source_record_id"], field),
                                "source_record_id": common["source_record_id"],
                                "run_id": run["run_id"],
                                "sample_id": run["sample_id"],
                                "context_id": run["context_id"],
                                "source_field": field,
                                "parent_label_raw": row["Protein"],
                                "quantity_kind": suffix.strip(),
                                "quantitative_raw": raw,
                                "quantitative_value": value,
                                "value_status": status,
                                "measurement_unit": "source_arbitrary_units",
                                "quantity_scale": suffix.strip(),
                            }
                        )
            members = [name for name in search.namelist() if ".fasta" in name]
            if len(members) != 1:
                raise ValueError("Require one original search FASTA")
            references = {}
            with io.TextIOWrapper(search.open(members[0])) as stream:
                for record in _reference_records(stream, "Search.zip!" + members[0]):
                    sinks["references"].add(record)
                    references[record["reference_record_id"]] = record
            peptides = {
                sequence: sequence
                for _, sequence, _ in peptide_rows
                if re.fullmatch("[A-Z]+", sequence)
            }
            found = (
                exact_occurrences(
                    {
                        key: {"sequence": record["reference_sequence"]}
                        for key, record in references.items()
                    },
                    peptides,
                )
                if peptides
                else {}
            )
            for source_id, peptide, parent in peptide_rows:
                matches = found.get(peptide, [])
                for reference_id, start in matches or [(None, None)]:
                    literal = references[reference_id]["accession_raw"] if reference_id else None
                    sinks["occurrences"].add(
                        {
                            "occurrence_id": _id("occurrence", source_id, reference_id, start),
                            "source_record_id": source_id,
                            "peptide_sequence": peptide,
                            "parent_label_raw": parent,
                            "reference_record_id": reference_id,
                            "start_1based": start,
                            "end_1based": start + len(peptide) - 1 if start else None,
                            "mapping_status": "exact_reference_candidate_not_measured_origin"
                            if reference_id
                            else "no_exact_reference_occurrence",
                            "source_claim_status": "literal_source_parent_claim"
                            if literal and literal in parent.split(";")
                            else "not_literal_source_parent_claim",
                        }
                    )
        for sample in sorted(samples.values(), key=lambda item: item["sample_id"]):
            sinks["samples"].add(sample)
    finally:
        for sink in sinks.values():
            sink.close()
    return {
        "relation_rows": {name: sink.n_rows for name, sink in sinks.items()},
        "source_counts": dict(sorted(counts.items())),
    }


def _manifest_files(directory):
    return {
        str(path.relative_to(directory)): file_digest(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path != directory / "manifest.json"
    }


def _verify_sources(directory, receipt):
    for spec in receipt["assets"]:
        expected = {key: spec[key] for key in ("sha256", "size_bytes")}
        if file_digest(directory / ROOT / "source_assets" / spec["filename"]) != expected:
            raise ValueError(f"Changed pinned HAP1 source: {spec['filename']}")


def _registry(contract):
    return {
        "shapiro2025_hap1": {
            "source_pmid": contract["source_pmid"],
            "source_url": contract["source_url"],
            "source_role": "MHC_I_elution_and_global_proteomics",
            "training_status": "deferred_endpoint_recipe",
        }
    }


def verify_hap1_quantitative_package(directory):
    """Verify original bytes and replay every typed relation from those sources."""
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if (
        manifest.get("schema_version"),
        manifest.get("catalog_schema_version"),
        manifest.get("package_family"),
        manifest.get("revision"),
    ) != (PACKAGE_SCHEMA_VERSION, CATALOG_SCHEMA_VERSION, PACKAGE_FAMILY, REVISION):
        raise ValueError("Unsupported HAP1 quantitative package")
    if _manifest_files(directory) != manifest.get("files"):
        raise ValueError("HAP1 file inventory or hashes differ")
    for name, source in (("assets.json", ASSETS), ("contract.json", CONTRACT)):
        if file_digest(directory / ROOT / name) != file_digest(source):
            raise ValueError("HAP1 source contract differs from registered revision")
    receipt = json.loads(ASSETS.read_text())
    contract = json.loads(CONTRACT.read_text())
    if (
        manifest.get("source_registry") != _registry(contract)
        or manifest.get("limitations") != contract["limitations"]
    ):
        raise ValueError("HAP1 source metadata differs from registered contract")
    _verify_sources(directory, receipt)
    with tempfile.TemporaryDirectory(prefix="hitlist-hap1-verify-") as temporary:
        replay = Path(temporary) / "typed"
        counts = _compile(directory / ROOT / "source_assets", replay, receipt, contract)
        if counts != manifest.get("reconciliation"):
            raise ValueError("HAP1 reconciliation differs from original source")
        for name, schema in SCHEMAS.items():
            expected = pq.ParquetFile(replay / f"{name}.parquet")
            actual = pq.ParquetFile(directory / ROOT / "typed" / f"{name}.parquet")
            if not actual.schema_arrow.equals(schema):
                raise ValueError(f"HAP1 schema mismatch: {name}")
            for left, right in zip(
                expected.iter_batches(batch_size=8192),
                actual.iter_batches(batch_size=8192),
                strict=True,
            ):
                if not left.equals(right):
                    raise ValueError(f"HAP1 typed relation differs from source: {name}")
    return manifest


def write_hap1_quantitative_package(source_directory, destination):
    """Build a source-complete quantitative package without creating training labels."""
    source_directory, destination = Path(source_directory), Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    receipt, contract = json.loads(ASSETS.read_text()), json.loads(CONTRACT.read_text())
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    try:
        assets = staging / ROOT / "source_assets"
        assets.mkdir(parents=True)
        for spec in receipt["assets"]:
            source = source_directory / spec["filename"]
            if not source.exists():
                source = source_directory / spec["source_relative_path"]
            if file_digest(source) != {key: spec[key] for key in ("sha256", "size_bytes")}:
                raise ValueError(f"HAP1 source mismatch: {spec['filename']}")
            snapshot_file(source, assets / spec["filename"])
        snapshot_file(ASSETS, staging / ROOT / "assets.json")
        snapshot_file(CONTRACT, staging / ROOT / "contract.json")
        counts = _compile(assets, staging / ROOT / "typed", receipt, contract)
        from .version import __version__

        manifest = {
            "schema_version": PACKAGE_SCHEMA_VERSION,
            "catalog_schema_version": CATALOG_SCHEMA_VERSION,
            "package_family": PACKAGE_FAMILY,
            "revision": REVISION,
            "hitlist_version": __version__,
            "source_registry": _registry(contract),
            "limitations": contract["limitations"],
            "reconciliation": counts,
            "files": _manifest_files(staging),
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        verify_hap1_quantitative_package(staging)
        staging.rename(destination)
        return destination / "manifest.json"
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
