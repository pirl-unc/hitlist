"""Original purified-proteasome time courses, before training admission.

R is needed only to decode the deposited RData during construction. Portable
readers use typed relations and immutable original assets without invoking R.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import math
import shutil
import subprocess
import tempfile
import zipfile
from collections import Counter
from pathlib import Path
from typing import ClassVar

import pyarrow.parquet as pq

from .mapp_evidence import _schema, _Sink
from .provenance import file_digest

SOURCE_ID = "soh2024_digestion"
MS1_UNIT = "source_MS1_peak_area_arbitrary_units"
DATA = Path(__file__).parent / "data/evidence_revisions"
COMMON = ("source_record_id", "source_locator", "fields_json")
SCHEMAS = {
    "members": _schema(
        (*COMMON, "outer_archive", "inner_member", "materialization_status"),
        integers=("member_size_bytes", "member_crc32"),
    ),
    "metadata": _schema(
        (
            *COMMON,
            "sheet_name",
        ),
        integers=("row_1based",),
    ),
    "constructs": _schema(
        (*COMMON, "construct_id", "sequence", "sequence_sha256", "header_raw", "substrate_species"),
        integers=("n_residues",),
    ),
    "runs": _schema(
        (
            *COMMON,
            "run_id",
            "source_name",
            "raw_file",
            "protein_name_raw",
            "biological_replicate_raw",
            "instrument_raw",
            "control_role",
        ),
        numbers=("time_hours",),
    ),
    "contexts": _schema(
        (
            *COMMON,
            "condition_id",
            "construct_id",
            "enzyme_species",
            "enzyme_complex",
            "enzyme_preparation_cell_line",
            "substrate_species",
            "context_scope",
            "buffer_json",
            "measurement_population",
            "termination",
            "training_status",
        ),
        numbers=("temperature_c", "substrate_concentration_um", "enzyme_substrate_denominator"),
    ),
    "identifications": _schema(
        (
            *COMMON,
            "condition_id",
            "run_id",
            "source_name",
            "peptide_sequence",
            "modified_sequence",
            "assignment_id_raw",
            "product_type_raw",
            "positions_raw",
            "run_link_status",
        ),
        numbers=("q_value", "posterior_error_probability"),
    ),
    "ms1": _schema(
        (
            *COMMON,
            "condition_id",
            "run_id",
            "source_name",
            "peptide_sequence",
            "modified_sequence",
            "assignment_id_raw",
            "fragment_ion_raw",
            "precursor_charge_raw",
            "run_link_status",
            "value_status",
            "measurement_unit",
            "quantity_scale",
        ),
        numbers=("ms1_area", "ms1_background"),
    ),
    "raw_quantities": _schema(
        (
            *COMMON,
            "condition_id",
            "construct_id",
            "run_id",
            "source_name",
            "peptide_sequence",
            "biological_replicate_raw",
            "metadata_biological_replicate_raw",
            "control_role",
            "value_status",
            "time_status",
            "assignments_raw",
            "measurement_unit",
            "quantity_scale",
        ),
        numbers=(
            "quantity_time_hours",
            "metadata_time_hours",
            "intensity",
            "fraction_background",
            "noise",
            "mean_technical",
            "mean_biological",
        ),
    ),
    "filtered_vectors": _schema(
        (
            *COMMON,
            "condition_id",
            "construct_id",
            "peptide_sequence",
            "biological_replicate_raw",
            "technical_replicate_raw",
            "times_raw",
            "intensities_raw",
            "product_type_raw",
            "splice_type_raw",
            "positions_raw",
        ),
    ),
    "final_vectors": _schema(
        (
            *COMMON,
            "condition_id",
            "construct_id",
            "peptide_sequence",
            "biological_replicate_raw",
            "times_raw",
            "intensities_raw",
            "product_type_raw",
            "splice_type_raw",
            "positions_raw",
            "row_status",
        ),
    ),
    "final_values": _schema(
        (
            *COMMON,
            "condition_id",
            "construct_id",
            "vector_id",
            "peptide_sequence",
            "biological_replicate_raw",
            "value_raw",
            "value_status",
            "time_status",
            "measurement_unit",
            "quantity_scale",
        ),
        integers=("slot_0based",),
        numbers=("time_hours", "intensity"),
    ),
    "conflicts": _schema(
        (*COMMON, "condition_id", "run_id", "conflict_kind", "claim_a", "claim_b")
    ),
}

# Read/decode only; no author pipeline or object methods are executed.
R_EXPORT = r"""
args <- commandArgs(trailingOnly=TRUE)
stopifnot(length(args) == 2)
env <- new.env()
names <- load(args[[1]], envir=env)
stopifnot(length(names) == 1)
obj <- env[[names[[1]]]]
stopifnot(inherits(obj,"data.frame"))
class(obj) <- "data.frame"
stopifnot(all(vapply(obj,function(x) is.atomic(x) && is.null(dim(x)),logical(1))))
options(digits=17)
con <- gzfile(args[[2]],"wt")
write.table(obj,con,sep=",",row.names=FALSE,col.names=TRUE,na="NA",qmethod="double")
close(con)
"""


def numeric(value):
    """Keep source missingness separate from finite numeric values."""
    if value is None or value in ("", "NA", "#N/A", "NaN", "nan"):
        return None, "source_missing"
    number = float(value)
    if not math.isfinite(number):
        return None, "source_nonfinite"
    return (
        number,
        "source_zero" if number == 0 else "source_positive" if number > 0 else "source_negative",
    )


def run_context(row):
    name = row["protein_name"]
    time, _ = numeric(row["digestTime"])
    if time is None or time < 0:
        raise ValueError("Missing or invalid digestion metadata time")
    return {
        "run_id": f"{SOURCE_ID}:run:{row['raw_file']}",
        "source_name": row["raw_file"].removesuffix(".raw"),
        "raw_file": row["raw_file"],
        "protein_name_raw": name,
        "biological_replicate_raw": str(row["biological_replicate"]),
        "instrument_raw": row["Mass spectrometry"],
        "control_role": "substrate_only" if "substrate only" in name.lower() else "digestion",
        "time_hours": time,
    }


def vector_values(row):
    """Preserve duplicate time slots and forced-zero semantics without imputation."""
    if row["biological_replicate"] == "NA":
        if row["digestTimes"] != "NA" or row["intensities"] != "NA":
            raise ValueError("Annotation-only row unexpectedly has a quantitative vector")
        return []
    times, values = row["digestTimes"].split(";"), row["intensities"].split(";")
    if len(times) != len(values):
        raise ValueError("Digestion time/value vector lengths differ")
    repeated = len(times) != len(set(times))
    output = []
    for index, (time_raw, value_raw) in enumerate(zip(times, values)):
        time, _ = numeric(time_raw)
        value, status = numeric(value_raw)
        if time is None or time < 0:
            raise ValueError("Quantified final row has missing or invalid time")
        if time == 0 and value == 0:
            status = "postprocessing_forced_zero"
        elif value is None:
            status = "postprocessing_missing_unresolved_contributors"
        output.append(
            {
                "slot_0based": index,
                "time_hours": time,
                "intensity": value,
                "value_raw": value_raw,
                "value_status": status,
                "time_status": "repeated_source_time_slots" if repeated else "source_reported",
            }
        )
    return output


def raw_quantity(row, runs):
    run = runs.get(row["source"])
    if run is None:
        raise ValueError(f"Unknown digestion acquisition: {row['source']}")
    time, _ = numeric(row["digestTime"])
    if time is None or time < 0:
        raise ValueError("Raw quantity has missing or invalid time")
    intensity, status = numeric(row["intensity"])
    return {
        "run_id": run["run_id"],
        "source_name": row["source"],
        "peptide_sequence": row["pepSeq"],
        "biological_replicate_raw": row["biological_replicate"],
        "metadata_biological_replicate_raw": run["biological_replicate_raw"],
        "control_role": run["control_role"],
        "quantity_time_hours": time,
        "metadata_time_hours": run["time_hours"],
        "time_status": "source_metadata_conflict"
        if time != run["time_hours"]
        else "source_agreement",
        "intensity": intensity,
        "value_status": status,
        "measurement_unit": MS1_UNIT,
        "quantity_scale": "aSPIRE_raw_peptide_run_quantity",
        "assignments_raw": row["assignments"],
        "fraction_background": numeric(row["frac_background"])[0],
        "noise": numeric(row["noise"])[0],
        "mean_technical": numeric(row["mean_techRep"])[0],
        "mean_biological": numeric(row["mean_bioRep"])[0],
    }


def run_link_status(source_name, run):
    if not source_name:
        return "blank_original_source_name"
    return "matched_main_search" if run is not None else "outside_main_search_roster"


def member_materialization(name, is_directory):
    if is_directory:
        return "archive_directory"
    if name == "QUANTITIES_raw.RData":
        return "typed_raw_quantities"
    if name == "QUANTITIES_filtered.RData":
        return "typed_filtered_vectors"
    if name == "MS1_HPR.csv":
        return "typed_ms1"
    if name == "ASSIGNMENTS.csv":
        return "typed_identifications"
    if name.endswith("finalKinetics.csv"):
        return "typed_final_vectors"
    return "original_asset_only"


def _records(handle, locator):
    reader = csv.DictReader(handle)
    if not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames):
        raise ValueError(f"Invalid source header: {locator}")
    for number, row in enumerate(reader, 2):
        if None in row or any(value is None for value in row.values()):
            raise ValueError(f"Malformed source record: {locator}:{number}")
        yield (
            {
                "source_record_id": f"{SOURCE_ID}:{locator}:row:{number}",
                "source_locator": f"{locator}:row:{number}",
                "fields_json": json.dumps(row, sort_keys=True),
            },
            row,
        )


def _decode_rdata(rscript, exporter, original, exported):
    subprocess.run(
        [rscript, "--vanilla", str(exporter), str(original), str(exported)],
        check=True,
        capture_output=True,
        text=True,
    )


def write_tables(source_dir):
    import openpyxl

    source_dir = Path(source_dir)
    contract = json.loads((source_dir / "contracts/contract.json").read_text())
    assets = source_dir / "assets"
    rscript = shutil.which("Rscript")
    if rscript is None:
        raise RuntimeError(
            "Digestion source construction requires Rscript to read original RData; readers do not"
        )
    typed = source_dir / "typed"
    typed.mkdir()
    sinks = {name: _Sink(typed, name, schema=schema) for name, schema in SCHEMAS.items()}
    runs, constructs = {}, {}
    source_counts = Counter(dict.fromkeys(contract["expected"], 0))
    condition_conflicts = Counter()
    exports = []
    try:
        workbook = openpyxl.load_workbook(
            assets / "Identifier.xlsx", read_only=True, data_only=False
        )
        for sheet in workbook:
            rows = sheet.iter_rows(values_only=True)
            header = next(rows)
            for number, values in enumerate(rows, 2):
                row = dict(zip(header, values))
                common = {
                    "source_record_id": f"{SOURCE_ID}:Identifier.xlsx:{sheet.title}:{number}",
                    "source_locator": f"Identifier.xlsx:{sheet.title}:row:{number}",
                    "fields_json": json.dumps(
                        {"columns": list(header), "values": list(values)}, default=str
                    ),
                }
                sinks["metadata"].add({**common, "sheet_name": sheet.title, "row_1based": number})
                source_counts["n_metadata_rows"] += 1
                if sheet.title == "Main_searches":
                    run = run_context(row)
                    if run["source_name"] in runs:
                        raise ValueError("Duplicate digestion acquisition metadata")
                    runs[run["source_name"]] = run
                    sinks["runs"].add({**common, **run})
                elif sheet.title == "Contaminant_DB_generation":
                    filename = str(row.get("raw_file") or "").lower()
                    enzyme = str(row.get("Enzyme") or "").lower()
                    if ("_try_" in filename and enzyme == "chymotrypsin") or (
                        "_chymo_" in filename and enzyme == "trypsin"
                    ):
                        source_counts["n_metadata_enzyme_conflicts"] += 1
                        sinks["conflicts"].add(
                            {
                                **common,
                                "condition_id": None,
                                "run_id": None,
                                "conflict_kind": "contaminant_enzyme_vs_filename_label",
                                "claim_a": row["Enzyme"],
                                "claim_b": row["raw_file"],
                            }
                        )
        workbook.close()
        for filename in sorted(set(contract["construct_files"].values())):
            path = assets / "substrate-fastas" / filename
            lines = path.read_text().splitlines()
            if sum(line.startswith(">") for line in lines) != 1:
                raise ValueError("Expected one original substrate construct")
            sequence = "".join(line.strip() for line in lines if not line.startswith(">"))
            if not sequence or set(sequence) - set("ACDEFGHIKLMNPQRSTVWY"):
                raise ValueError("Unsupported literal substrate sequence")
            cid = f"{SOURCE_ID}:construct:{hashlib.sha256(sequence.encode()).hexdigest()}"
            species = "Homo sapiens" if "OX=9606" in lines[0] else None
            constructs[filename] = (cid, sequence, species)
            sinks["constructs"].add(
                {
                    "source_record_id": cid,
                    "construct_id": cid,
                    "source_locator": f"substrate-fastas/{filename}",
                    "fields_json": json.dumps(lines),
                    "sequence": sequence,
                    "sequence_sha256": hashlib.sha256(sequence.encode()).hexdigest(),
                    "header_raw": lines[0],
                    "substrate_species": species,
                    "n_residues": len(sequence),
                }
            )
        with (
            zipfile.ZipFile(assets / "aSPIRE_outputs_zip.zip") as outer,
            tempfile.TemporaryDirectory() as temporary,
        ):
            temporary = Path(temporary)
            exporter = temporary / "export.R"
            exporter.write_text(R_EXPORT)
            for member in outer.infolist():
                if member.is_dir():
                    continue
                archive_name = Path(member.filename).stem
                if archive_name not in contract["construct_files"]:
                    raise ValueError("Unexpected quantitative condition archive")
                source_counts["n_archives"] += 1
                cid, sequence, species = constructs[contract["construct_files"][archive_name]]
                condition = f"{SOURCE_ID}:condition:{archive_name}"
                local_zip = temporary / "condition.zip"
                with outer.open(member) as source, local_zip.open("wb") as target:
                    shutil.copyfileobj(source, target, 1024 * 1024)
                with zipfile.ZipFile(local_zip) as inner:
                    if inner.testzip() is not None:
                        raise ValueError("Corrupt nested digestion source")
                    for entry in inner.infolist():
                        source_counts["n_archive_members"] += 1
                        name = Path(entry.filename).name
                        locator = f"aSPIRE_outputs_zip.zip!{member.filename}!{entry.filename}"
                        sinks["members"].add(
                            {
                                "source_record_id": f"{SOURCE_ID}:member:{locator}",
                                "source_locator": locator,
                                "fields_json": json.dumps(
                                    {
                                        "filename": entry.filename,
                                        "file_size": entry.file_size,
                                        "compress_size": entry.compress_size,
                                        "CRC": entry.CRC,
                                        "date_time": entry.date_time,
                                        "external_attr": entry.external_attr,
                                    },
                                    sort_keys=True,
                                ),
                                "outer_archive": member.filename,
                                "inner_member": entry.filename,
                                "member_size_bytes": entry.file_size,
                                "member_crc32": entry.CRC,
                                "materialization_status": member_materialization(
                                    name, entry.is_dir()
                                ),
                            }
                        )
                        if name.endswith(".RData") and name in (
                            "QUANTITIES_raw.RData",
                            "QUANTITIES_filtered.RData",
                        ):
                            original = temporary / name
                            exported = temporary / f"{name}.csv.gz"
                            original.write_bytes(inner.read(entry))
                            _decode_rdata(rscript, exporter, original, exported)
                            exports.append(
                                {
                                    "source_locator": locator,
                                    "original": file_digest(original),
                                    "decoded_csv": file_digest(exported),
                                }
                            )
                            with gzip.open(exported, "rt", newline="") as handle:
                                for common, row in _records(handle, locator):
                                    if name == "QUANTITIES_raw.RData":
                                        if row["substrateSeq"] != sequence:
                                            raise ValueError(
                                                "Raw quantity substrate differs from original construct"
                                            )
                                        values = raw_quantity(row, runs)
                                        sinks["raw_quantities"].add(
                                            {
                                                **common,
                                                **values,
                                                "condition_id": condition,
                                                "construct_id": cid,
                                            }
                                        )
                                        source_counts["n_raw_quantities"] += 1
                                        if values["time_status"] == "source_metadata_conflict":
                                            condition_conflicts[condition] += 1
                                            source_counts["n_time_conflicts"] += 1
                                            sinks["conflicts"].add(
                                                {
                                                    **common,
                                                    "condition_id": condition,
                                                    "run_id": values["run_id"],
                                                    "conflict_kind": "quantity_time_vs_original_metadata",
                                                    "claim_a": row["digestTime"],
                                                    "claim_b": str(values["metadata_time_hours"]),
                                                }
                                            )
                                    else:
                                        if len(row["int_pasted"].split(";")) != len(
                                            row["times_pasted"].split(";")
                                        ):
                                            raise ValueError(
                                                "Filtered source vector lengths differ"
                                            )
                                        sinks["filtered_vectors"].add(
                                            {
                                                **common,
                                                "condition_id": condition,
                                                "construct_id": cid,
                                                "peptide_sequence": row["pepSeq"],
                                                "biological_replicate_raw": row[
                                                    "biological_replicate"
                                                ],
                                                "technical_replicate_raw": row[
                                                    "technical_replicate"
                                                ],
                                                "times_raw": row["times_pasted"],
                                                "intensities_raw": row["int_pasted"],
                                                "product_type_raw": row["productType"],
                                                "splice_type_raw": row["spliceType"],
                                                "positions_raw": row["positions"],
                                            }
                                        )
                                        source_counts["n_filtered_vectors"] += 1
                        elif name in ("MS1_HPR.csv", "ASSIGNMENTS.csv") or name.endswith(
                            "finalKinetics.csv"
                        ):
                            with inner.open(entry) as source:
                                for common, row in _records(io.TextIOWrapper(source), locator):
                                    base = {**common, "condition_id": condition}
                                    if name == "MS1_HPR.csv":
                                        source_name = row["FileName"].removesuffix(".raw")
                                        run = runs.get(source_name)
                                        # Other Skyline rows remain source data even if not in the main search roster.
                                        area, status = numeric(row["TotalAreaMs1"])
                                        sinks["ms1"].add(
                                            {
                                                **base,
                                                "run_id": run["run_id"] if run else None,
                                                "source_name": source_name,
                                                "run_link_status": run_link_status(
                                                    source_name, run
                                                ),
                                                "peptide_sequence": row["PeptideSequence"],
                                                "modified_sequence": row["PeptideModifiedSequence"],
                                                "assignment_id_raw": row["ProteinName"],
                                                "fragment_ion_raw": row["FragmentIon"],
                                                "precursor_charge_raw": row["PrecursorCharge"],
                                                "value_status": status,
                                                "measurement_unit": MS1_UNIT,
                                                "quantity_scale": "Skyline_MS1_peak_area",
                                                "ms1_area": area,
                                                "ms1_background": numeric(
                                                    row["TotalBackgroundMs1"]
                                                )[0],
                                            }
                                        )
                                        source_counts["n_ms1_rows"] += 1
                                        if run is None:
                                            source_counts["n_unlinked_ms1_rows"] += 1
                                    elif name == "ASSIGNMENTS.csv":
                                        run = runs.get(row["source"])
                                        sinks["identifications"].add(
                                            {
                                                **base,
                                                "run_id": run["run_id"] if run else None,
                                                "source_name": row["source"],
                                                "run_link_status": run_link_status(
                                                    row["source"], run
                                                ),
                                                "peptide_sequence": row["pepSeq"],
                                                "modified_sequence": row["modifiedSequence"],
                                                "assignment_id_raw": row["ID"],
                                                "product_type_raw": row["productType"],
                                                "positions_raw": row["positions"],
                                                "q_value": numeric(row["qValue"])[0],
                                                "posterior_error_probability": numeric(
                                                    row["postErrProb"]
                                                )[0],
                                            }
                                        )
                                        source_counts["n_identifications"] += 1
                                    else:
                                        if row["substrateSeq"] != sequence:
                                            raise ValueError(
                                                "Final substrate differs from original construct"
                                            )
                                        source_counts["n_final_rows"] += 1
                                        if row["biological_replicate"] == "NA":
                                            source_counts["n_annotation_only_rows"] += 1
                                        sinks["final_vectors"].add(
                                            {
                                                **base,
                                                "construct_id": cid,
                                                "peptide_sequence": row["pepSeq"],
                                                "biological_replicate_raw": row[
                                                    "biological_replicate"
                                                ],
                                                "times_raw": row["digestTimes"],
                                                "intensities_raw": row["intensities"],
                                                "product_type_raw": row["productType"],
                                                "splice_type_raw": row["spliceType"],
                                                "positions_raw": row["positions"],
                                                "row_status": "annotation_only"
                                                if row["biological_replicate"] == "NA"
                                                else "reported_vector",
                                            }
                                        )
                                        for value in vector_values(row):
                                            sinks["final_values"].add(
                                                {
                                                    **base,
                                                    **value,
                                                    "construct_id": cid,
                                                    "vector_id": common["source_record_id"],
                                                    "source_record_id": common["source_record_id"]
                                                    + f":slot:{value['slot_0based']}",
                                                    "peptide_sequence": row["pepSeq"],
                                                    "biological_replicate_raw": row[
                                                        "biological_replicate"
                                                    ],
                                                    "measurement_unit": MS1_UNIT,
                                                    "quantity_scale": "aSPIRE_background_adjusted_biological_mean",
                                                }
                                            )
                                            source_counts["n_final_values"] += 1
                sinks["contexts"].add(
                    {
                        "source_record_id": condition,
                        "condition_id": condition,
                        "source_locator": f"contracts/contract.json;{contract['protocol_citation']}",
                        "fields_json": json.dumps(contract, sort_keys=True),
                        "construct_id": cid,
                        "enzyme_species": contract["enzyme_species"],
                        "enzyme_complex": contract["enzyme_complex"],
                        "enzyme_preparation_cell_line": contract["enzyme_preparation_cell_line"],
                        "substrate_species": species,
                        "context_scope": contract["input_scope"],
                        "buffer_json": json.dumps(contract["buffer"], sort_keys=True),
                        "measurement_population": contract["endpoint"],
                        "termination": "acidification_for_MS",
                        "temperature_c": contract["temperature_c"],
                        "substrate_concentration_um": contract["substrate_concentration_um_aSyn"]
                        if archive_name == "aSPIRE_aSyn"
                        else contract["substrate_concentration_um_default"],
                        "enzyme_substrate_denominator": contract["enzyme_substrate_denominators"][
                            archive_name
                        ],
                        "training_status": "unresolved_source_time_conflict"
                        if condition_conflicts[condition]
                        else "pending_registered_recipe_and_independent_replay",
                    }
                )
                source_counts["n_contexts"] += 1
        source_counts["n_runs"] = len(runs)
        source_counts["n_constructs"] = len(constructs)
        if dict(source_counts) != contract["expected"]:
            raise ValueError(f"Digestion source census changed: {dict(source_counts)}")
    finally:
        for sink in sinks.values():
            sink.close()
    export_receipt = {
        "R_version": subprocess.check_output([rscript, "--version"], text=True).strip(),
        "exporter_sha256": hashlib.sha256(R_EXPORT.encode()).hexdigest(),
        "numeric_contract": "decimal CSV decoding; original binary values retained in assets",
        "exports": exports,
    }
    (source_dir / "rdata_decoding.json").write_text(
        json.dumps(export_receipt, indent=2, sort_keys=True) + "\n"
    )
    return {
        "source_counts": dict(source_counts),
        "relation_rows": {n: s.n_rows for n, s in sinks.items()},
    }


class DigestionAdapter:
    source_id = SOURCE_ID
    schema_version = 1
    asset_receipt_path = DATA / "digestion_assets.json"
    support_files: ClassVar[dict[str, Path]] = {"contract.json": DATA / "digestion_contract.json"}

    def write(self, source_dir):
        contract = json.loads((Path(source_dir) / "contracts/contract.json").read_text())
        if (
            file_digest(Path(source_dir) / "contracts/source_assets.json")["sha256"]
            != contract["source_receipt_sha256"]
        ):
            raise ValueError("Digestion original-asset contract mismatch")
        return write_tables(source_dir)

    def verify(self, source_dir, reconciliation):
        contract = json.loads((Path(source_dir) / "contracts/contract.json").read_text())
        if reconciliation["source_counts"] != contract["expected"]:
            raise ValueError("Digestion source count contract changed")
        for name, schema in SCHEMAS.items():
            parquet = pq.ParquetFile(Path(source_dir) / "typed" / f"{name}.parquet")
            if (
                parquet.schema_arrow != schema
                or parquet.metadata.num_rows != reconciliation["relation_rows"][name]
            ):
                raise ValueError(f"Digestion typed relation changed: {name}")

    def registry(self, reconciliation):
        return {
            "source_id": SOURCE_ID,
            "dataset_accessions": ["MSV000092631", "PXD044463"],
            "rights": "CC0-1.0",
            "source_counts": reconciliation["source_counts"],
            "endpoint": "purified_proteasome_fragment_MS_response",
            "training_admission": "pending_registered_recipe_and_all_admitted_arithmetic_replay",
            "limitations": [
                "MS_response_not_molar_concentration",
                "RF1_time_conflict",
                "forced_zero_and_background_missingness",
                "source_product_selection",
                "no_APC_transport_clearance_or_MHC_context",
            ],
        }
