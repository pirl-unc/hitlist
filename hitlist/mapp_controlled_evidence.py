"""Source-faithful unfiltered MAPP panels with distinct processing and capture axes."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from .mapp_evidence import SCHEMAS as MAPP_SCHEMAS
from .mapp_evidence import _integer, _number, _schema, _Sink

SOURCE_ID = "wolflevy2018_controlled"
PROJECTS = ("PXD009508", "PXD007263")
DATA = Path(__file__).parent / "data/evidence_revisions"
SUPPORT_FILES = {"contract.json": DATA / "mapp_controlled_contract.json"}
BASE_TABLES = tuple(name for name in MAPP_SCHEMAS if not name.startswith("publisher_"))
SCHEMAS = {
    name: MAPP_SCHEMAS[name].append(pa.field("source_dataset", pa.string())) for name in BASE_TABLES
}
SCHEMAS["channels"] = pa.schema(
    list(SCHEMAS["channels"])
    + list(
        _schema(
            (
                "capture_label_raw",
                "capture_role",
                "capture_target_gene",
                "treatment_label_raw",
                "treatment_kind",
                "treatment_name",
                "treatment_context_status",
                "treatment_dose_unit",
                "measured_cell_scope",
                "measured_species_scope",
                "specimen_id",
                "aliquot_id",
                "pair_id",
            ),
            numbers=("treatment_dose", "duration_hours", "time_hours"),
        )
    )
)
SCHEMAS["quantity_results"] = pa.schema(
    list(SCHEMAS["quantity_results"])
    + list(
        _schema(
            integers=(
                "n_source_evidence",
                "n_numeric_evidence",
                "n_direct_msms_evidence",
                "n_positive_direct_msms_evidence",
                "n_matched_evidence",
            )
        )
    )
)
SCHEMAS["label_aliases"] = _schema(
    (
        "source_dataset",
        "alias_id",
        "channel_id",
        "quantity_column_raw",
        "summary_experiment_raw",
        "match_rule",
        "source_locator",
    )
)


def _rows(directory, project, filename, sinks):
    relative = f"{project}/txt/{filename}.txt"
    with (directory / "assets" / relative).open(newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError("Missing or duplicated source columns")
        for number, row in enumerate(reader, 2):
            if None in row or any(value is None for value in row.values()):
                raise ValueError("Malformed controlled-panel source row")
            key = f"{project.lower()}:{filename}:row:{number}"
            sinks["source_records"].add(
                {
                    "source_dataset": project,
                    "source_record_id": key,
                    "asset_path": relative,
                    "row_1based": number,
                    "fields_json": json.dumps(row, sort_keys=True),
                }
            )
            yield key, row


def match_quantity_columns(columns, experiments):
    """Only exact or unique complete-identifier punctuation aliases are admitted."""
    candidates = defaultdict(list)
    for experiment in experiments:
        candidates[re.sub("[-_]", "", experiment)].append(experiment)
    matches = {}
    for column in columns:
        literal = column.removeprefix("Intensity ")
        targets = [literal] if literal in experiments else candidates[re.sub("[-_]", "", literal)]
        if len(targets) != 1 or targets[0] in {item[0] for item in matches.values()}:
            raise ValueError("Ambiguous or duplicate source run alias")
        matches[column] = (
            targets[0],
            "literal_equal"
            if targets[0] == literal
            else "unique_full_identifier_without_hyphen_underscore",
        )
    if {item[0] for item in matches.values()} != set(experiments):
        raise ValueError("Incomplete source quantity run coverage")
    return matches


def channel_context(project, experiment):
    """Retain acquisition labels and explicitly scoped paper-level context."""
    shared = {
        "measured_species": "Homo sapiens",
        "measured_species_scope": "project_organism",
        "measured_cell": "HEK293",
        "measured_cell_scope": "publication_experiment_context_not_run_specimen_identity",
        "material_link_precision": "acquisition_run_unresolved_specimen",
        "capture_target_gene": None,
        "specimen_id": None,
        "aliquot_id": None,
        "pair_id": None,
    }
    if project == "PXD009508":
        match = re.fullmatch(r"\d+_7181-300a_(alpha6|iso|gap)_(ut|epoxo)([123])", experiment)
        if not match:
            raise ValueError("Unknown controls acquisition label")
        capture, treatment, replicate = match.groups()
        return {
            **shared,
            "capture_label_raw": capture,
            "capture_role": "candidate_proteasome_capture"
            if capture == "alpha6"
            else "capture_control",
            "treatment_label_raw": treatment,
            "treatment_kind": "pharmacological" if treatment == "epoxo" else "untreated_control",
            "treatment_name": "epoxomicin" if treatment == "epoxo" else None,
            "treatment_context_status": "run_label_with_publication_level_dose_duration",
            "treatment_dose": 1.0 if treatment == "epoxo" else None,
            "treatment_dose_unit": "micromolar" if treatment == "epoxo" else None,
            "duration_hours": 4.0 if treatment == "epoxo" else None,
            "time_hours": None,
            "condition": treatment,
            "condition_role": "biological_condition",
            "replicate_label": int(replicate),
            "measured_pool": "capture_associated_fragment",
        }
    if project == "PXD007263":
        match = re.fullmatch(r"\d+_TNF-TP-0-6_([0126])_([123])", experiment)
        if not match:
            raise ValueError("Unknown time-course acquisition label")
        hour, replicate = match.groups()
        return {
            **shared,
            "capture_label_raw": None,
            "capture_role": "project_reported_proteasome_capture",
            "treatment_label_raw": "TNF",
            "treatment_kind": "cytokine_exposure_context",
            "treatment_name": None,
            "treatment_context_status": "literal_TNF_vs_publication_TNFalpha_plus_IFNgamma",
            "treatment_dose": None,
            "treatment_dose_unit": None,
            "duration_hours": float(hour),
            "time_hours": float(hour),
            "condition": f"TNF_label:{hour}h",
            "condition_role": "exposure_time",
            "replicate_label": int(replicate),
            "measured_pool": "proteasome_associated_fragment",
        }
    raise ValueError("Unregistered controlled MAPP project")


def _write_project(directory, project, sinks):
    sid = project.lower()

    def add(table, **row):
        sinks[table].add({"source_dataset": project, **row})

    def peptide_key(pid):
        return f"{sid}:peptide:{pid}"

    def channel_key(experiment):
        return f"{sid}:run:{experiment}"

    def result_key(pid, experiment):
        return f"{peptide_key(pid)}:run:{experiment}"

    runs, contexts = {}, {}
    for locator, row in _rows(directory, project, "summary", sinks):
        experiment = row["Experiment"]
        if not experiment:
            continue
        if experiment in runs or row["Raw file"] in {item["Raw file"] for item in runs.values()}:
            raise ValueError("Duplicate controlled-panel source run")
        runs[experiment] = row
        contexts[experiment] = channel_context(project, experiment)
        add(
            "channels",
            channel_id=channel_key(experiment),
            source_record_id=locator,
            experiment_raw=experiment,
            raw_file=row["Raw file"],
            search_enzyme_mode_raw=row["Enzyme mode"],
            **contexts[experiment],
        )
    if len(runs) != {"PXD009508": 18, "PXD007263": 12}[project]:
        raise ValueError("Controlled-panel acquisition count changed")
    if project == "PXD009508":
        observed = {
            (row["capture_label_raw"], row["treatment_label_raw"], row["replicate_label"])
            for row in contexts.values()
        }
        expected = {
            (capture, treatment, replicate)
            for capture in ("alpha6", "iso", "gap")
            for treatment in ("ut", "epoxo")
            for replicate in (1, 2, 3)
        }
    else:
        observed = {(row["time_hours"], row["replicate_label"]) for row in contexts.values()}
        expected = {(hour, replicate) for hour in (0, 1, 2, 6) for replicate in (1, 2, 3)}
    if observed != expected:
        raise ValueError("Controlled-panel condition/replicate design changed")
    peptides, locators, sequences = {}, {}, set()
    for locator, row in _rows(directory, project, "peptides", sinks):
        pid = row["id"]
        if not pid.isdigit() or pid in peptides or row["Sequence"] in sequences:
            raise ValueError("Invalid or repeated source peptide identity")
        peptides[pid], locators[pid] = row, locator
        sequences.add(row["Sequence"])
        add(
            "peptide_rows",
            peptide_row_id=peptide_key(pid),
            source_record_id=locator,
            peptide_id_raw=pid,
            sequence_raw=row["Sequence"],
            parent_labels_raw=row["Proteins"],
            leading_parent_raw=row["Leading razor protein"],
            group_ids_raw=row["Protein group IDs"],
            n_flank_raw=row["Amino acid before"],
            c_flank_raw=row["Amino acid after"],
            start_1based_reported=_integer(row["Start position"]),
            end_1based_reported=_integer(row["End position"]),
            pep=None if row["PEP"] in ("", "NaN") else _number(row["PEP"]),
            pep_raw=row["PEP"],
            pep_status="unreported_source_nan"
            if row["PEP"] == "NaN"
            else ("unreported" if not row["PEP"] else "reported"),
            reverse=row["Reverse"] == "+",
            potential_contaminant=row["Potential contaminant"] == "+",
            measured_pool="capture_associated_fragment",
            material_link_precision="acquisition_run_unresolved_specimen",
        )
    groups = {}
    for locator, row in _rows(directory, project, "proteinGroups", sinks):
        gid = row["id"]
        if not gid.isdigit() or gid in groups:
            raise ValueError("Invalid or repeated controlled-panel group identity")
        groups[gid] = row
        add(
            "groups",
            group_record_id=locator,
            source_record_id=locator,
            group_id_raw=gid,
            protein_ids_raw=row["Protein IDs"],
            peptide_ids_raw=row["Peptide IDs"],
            status="source_group",
            reverse=row["Reverse"] == "+",
            potential_contaminant=row["Potential contaminant"] == "+",
        )
    totals, counts, numeric, direct, positive_direct, matched = (
        defaultdict(float),
        Counter(),
        Counter(),
        Counter(),
        Counter(),
        Counter(),
    )
    reciprocal, seen = defaultdict(set), set()
    for locator, row in _rows(directory, project, "evidence", sinks):
        pid, experiment, eid = row["Peptide ID"], row["Experiment"], row["id"]
        if not eid.isdigit() or eid in seen or pid not in peptides or experiment not in runs:
            raise ValueError("Invalid controlled-panel evidence relationship")
        if (
            row["Sequence"] != peptides[pid]["Sequence"]
            or row["Raw file"] != runs[experiment]["Raw file"]
        ):
            raise ValueError("Evidence sequence/acquisition mismatch")
        if any(gid not in groups for gid in row["Protein group IDs"].split(";") if gid):
            raise ValueError("Evidence references missing protein group")
        seen.add(eid)
        reciprocal[pid].add(eid)
        value, key = _number(row["Intensity"]), (pid, experiment)
        totals[key] += value or 0.0
        counts[key] += 1
        numeric[key] += value is not None
        msms = _integer(row["MS/MS Count"])
        is_direct = bool(msms and row["Type"] != "MULTI-MATCH")
        direct[key] += is_direct
        positive_direct[key] += bool(is_direct and value is not None and value > 0)
        matched[key] += row["Type"] == "MULTI-MATCH"
        evidence_id = f"{sid}:evidence:{eid}"
        add(
            "evidence",
            evidence_id=evidence_id,
            source_record_id=locator,
            peptide_row_id=peptide_key(pid),
            channel_id=channel_key(experiment),
            sequence_raw=row["Sequence"],
            modified_sequence_raw=row["Modified sequence"],
            modifications_raw=row["Modifications"],
            type_raw=row["Type"],
            group_ids_raw=row["Protein group IDs"],
            intensity_raw=row["Intensity"],
            intensity=value,
            intensity_status="unreported" if value is None else "reported",
            msms_count=msms,
            pep=None if row["PEP"] in ("", "NaN") else _number(row["PEP"]),
            pep_raw=row["PEP"],
            pep_status="unreported_source_nan"
            if row["PEP"] == "NaN"
            else ("unreported" if not row["PEP"] else "reported"),
            reverse=row["Reverse"] == "+",
            potential_contaminant=row["Potential contaminant"] == "+",
        )
        add(
            "derivation_inputs",
            derived_measurement_id=result_key(pid, experiment) + ":evidence_sum",
            input_measurement_id=evidence_id,
            input_role="source_evidence_intensity",
        )
    columns = [name for name in next(iter(peptides.values())) if name.startswith("Intensity ")]
    aliases = match_quantity_columns(columns, runs)
    for column, (experiment, rule) in aliases.items():
        add(
            "label_aliases",
            alias_id=channel_key(experiment) + ":quantity_alias",
            channel_id=channel_key(experiment),
            quantity_column_raw=column,
            summary_experiment_raw=experiment,
            match_rule=rule,
            source_locator=f"{project}/txt/peptides.txt:header;{project}/txt/summary.txt",
        )
    max_error, n_positive, n_zero = 0.0, 0, 0
    for pid, row in peptides.items():
        if set(filter(None, row["Evidence IDs"].split(";"))) != reciprocal[pid]:
            raise ValueError("Peptide/evidence reciprocal relation changed")
        for gid in filter(None, row["Protein group IDs"].split(";")):
            if gid not in groups or pid not in groups[gid]["Peptide IDs"].split(";"):
                raise ValueError("Peptide/group reciprocal relation changed")
        for column, (experiment, _) in aliases.items():
            result, key = result_key(pid, experiment), (pid, experiment)
            reported, replay = _number(row[column]), totals[key]
            if reported is None or not math.isclose(reported, replay, rel_tol=1e-4, abs_tol=1.0):
                raise ValueError("Reported controlled-panel intensity does not replay")
            max_error = max(max_error, abs(reported - replay) / max(reported, replay, 1))
            n_positive += reported > 0
            n_zero += reported == 0
            add(
                "quantity_results",
                result_id=result,
                peptide_row_id=peptide_key(pid),
                channel_id=channel_key(experiment),
                measured_pool=contexts[experiment]["measured_pool"],
                endpoint="captured_fragment_MS_intensity",
                material_link_precision="acquisition_run_unresolved_specimen",
                n_source_evidence=counts[key],
                n_numeric_evidence=numeric[key],
                n_direct_msms_evidence=direct[key],
                n_positive_direct_msms_evidence=positive_direct[key],
                n_matched_evidence=matched[key],
            )
            sum_status = (
                "no_reported_evidence"
                if not counts[key]
                else "all_evidence_intensities_unreported"
                if not numeric[key]
                else "partial_evidence_intensity_sum"
                if numeric[key] < counts[key]
                else "reported_evidence_sum"
            )
            for representation, value, status, reconstructed, locator in (
                (
                    "evidence_sum",
                    replay if numeric[key] else None,
                    sum_status,
                    None,
                    f"{project}/txt/evidence.txt",
                ),
                (
                    "reported_run_intensity",
                    reported,
                    "reported_positive" if reported > 0 else "reported_zero_not_absence",
                    replay,
                    locators[pid] + ":" + column,
                ),
            ):
                add(
                    "quantity_measurements",
                    measurement_id=result + ":" + representation,
                    result_id=result,
                    peptide_row_id=peptide_key(pid),
                    representation=representation,
                    value_status=status,
                    unit="arbitrary_intensity",
                    quantity_scale="relative_label_free_signal",
                    source_locator=locator,
                    derivation_rule="sum_reported_evidence_intensities;missing_values_not_measured_zero",
                    value=value,
                    recomputed_value=reconstructed,
                    absolute_replay_error=abs(value - reconstructed)
                    if reconstructed is not None
                    else None,
                )
            add(
                "derivation_inputs",
                derived_measurement_id=result + ":reported_run_intensity",
                input_measurement_id=result + ":evidence_sum",
                input_role="arithmetic_replay",
            )
    for locator, row in _rows(directory, project, "parameters", sinks):
        if set(row) != {"Parameter", "Value"}:
            raise ValueError("Unexpected MaxQuant parameter columns")
        add(
            "source_assertions",
            assertion_id=locator + ":parameter",
            subject_id=sid,
            predicate=row["Parameter"],
            value_raw=row["Value"],
            normalized_value=None,
            assertion_status="source_reported",
            source_locator=locator,
            scope="deposited_search_parameters",
        )
    return {
        "n_peptides": len(peptides),
        "n_groups": len(groups),
        "n_evidence": len(seen),
        "n_runs": len(runs),
        "n_positive_run_quantities": n_positive,
        "n_zero_run_quantities": n_zero,
        "n_missing_evidence_intensities": sum(counts.values()) - sum(numeric.values()),
        "maximum_relative_replay_error": max_error,
    }


def write_controlled_tables(source_dir):
    source_dir = Path(source_dir)
    (source_dir / "typed").mkdir()
    sinks = {
        name: _Sink(source_dir / "typed", name, schema=schema) for name, schema in SCHEMAS.items()
    }
    try:
        projects = {project: _write_project(source_dir, project, sinks) for project in PROJECTS}
        contract = json.loads((source_dir / "contracts/contract.json").read_text())
        for index, row in enumerate(contract["assertions"]):
            sinks["source_assertions"].add({"assertion_id": f"{SOURCE_ID}:curated:{index}", **row})
    finally:
        for sink in sinks.values():
            sink.close()
    return {
        "projects": projects,
        "relation_rows": {name: sink.n_rows for name, sink in sinks.items()},
    }


class ControlledMAPPAdapter:
    source_id = SOURCE_ID
    schema_version = 1
    asset_receipt_path = DATA / "mapp_controlled_assets.json"
    support_files = SUPPORT_FILES

    def write(self, source_dir):
        from .sequence_reference_evidence import write_reference_relations

        source_dir = Path(source_dir)
        contract = json.loads((source_dir / "contracts/contract.json").read_text())
        if (
            hashlib.sha256((source_dir / "contracts/source_assets.json").read_bytes()).hexdigest()
            != contract["source_receipt_sha256"]
        ):
            raise ValueError("Controlled MAPP receipt changed")
        result = write_controlled_tables(source_dir)
        result.update(
            write_reference_relations(
                source_dir,
                contract["references"],
                pq.read_table(source_dir / "typed/peptide_rows.parquet").to_pylist(),
            )
        )
        return result

    def verify(self, source_dir, reconciliation):
        from .sequence_reference_evidence import verify_reference_relations

        verify_reference_relations(source_dir, reconciliation)
        for name, schema in SCHEMAS.items():
            parquet = pq.ParquetFile(Path(source_dir) / "typed" / f"{name}.parquet")
            if (
                parquet.schema_arrow != schema
                or parquet.metadata.num_rows != reconciliation["relation_rows"][name]
            ):
                raise ValueError(f"Controlled MAPP relation changed: {name}")
        for project, expected in {
            "PXD009508": (6060, 1058, 19581, 18, 15484, 93596, 400),
            "PXD007263": (3146, 733, 27196, 12, 20940, 16812, 500),
        }.items():
            values = reconciliation["projects"][project]
            for key, value in zip(
                (
                    "n_peptides",
                    "n_groups",
                    "n_evidence",
                    "n_runs",
                    "n_positive_run_quantities",
                    "n_zero_run_quantities",
                    "n_missing_evidence_intensities",
                ),
                expected,
            ):
                if values[key] != value:
                    raise ValueError(f"Controlled MAPP source count changed: {project}:{key}")

    def registry(self, reconciliation):
        return {
            "source_id": SOURCE_ID,
            "dataset_accessions": list(PROJECTS),
            "organism_scope": "human project organisms; HEK293 paper-level experiment context",
            "measurement_grain": "source_peptide_by_acquisition_run",
            "quantity_artifact": "typed/quantity_measurements.parquet",
            "reference_occurrence_artifact": "typed/peptide_reference_occurrences.parquet",
            "n_primary_peptide_rows": reconciliation["relation_rows"]["peptide_rows"],
            "projects": reconciliation["projects"],
            "rights": {
                "PRIDE_deposit": "EBI terms of use",
                "paper": "linked assertions; manuscript not redistributed",
            },
            "limitations": [
                "captured_signal_not_free_concentration",
                "unresolved_specimen_aliquot_pairing",
                "original_2016_reference_unavailable",
                "alpha6_gene_alias_not_source_supplied",
                "literal_TNF_vs_combined_cytokine_paper_context",
                "missing_quantities_not_negatives",
                "unmeasured_precursor_and_MS_response",
                "numbered_repeats_not_assured_paired_specimens",
            ],
        }
