"""Source-faithful primary-PBMC MAPP evidence from PXD009438.

The deposit labels acquisition runs as CTRL or LUPUS source samples.  Those
labels establish source study groups and acquisition-to-sample-label links; they
do not establish physical specimen, aliquot, donor, or cross-modal identities.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

from .mapp_controlled_evidence import SCHEMAS, match_quantity_columns
from .mapp_evidence import _integer, _number, _Sink
from .material_context import validate_material_graph

SOURCE_ID = "pxd009438"
ACCESSION = "PXD009438"
DATA = Path(__file__).parent / "data/evidence_revisions"
ASSET_RECEIPT = DATA / "mapp_pbmc_assets.json"
SUPPORT_FILES = {"contract.json": DATA / "mapp_pbmc_contract.json"}
_RUN = re.compile(r"(?P<acquisition>\d+)_(?P<group>CTRL|LUPUS)_(?P<sample>\d+)")
_EXPECTED_SAMPLES = {("CTRL", number) for number in range(1, 7)} | {
    ("LUPUS", number) for number in range(1, 9)
}


def _rows(directory, filename, sinks):
    path = directory / "assets" / "txt" / f"{filename}.txt"
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError("Missing or duplicated PXD009438 source columns")
        for number, row in enumerate(reader, 2):
            if None in row or any(value is None for value in row.values()):
                raise ValueError("Malformed PXD009438 source row")
            key = f"{SOURCE_ID}:{filename}:row:{number}"
            sinks["source_records"].add(
                {
                    "source_dataset": ACCESSION,
                    "source_record_id": key,
                    "asset_path": f"txt/{filename}.txt",
                    "row_1based": number,
                    "fields_json": json.dumps(row, sort_keys=True),
                }
            )
            yield key, row


def _source_sample(experiment):
    match = _RUN.fullmatch(experiment)
    if not match:
        raise ValueError("Unknown PXD009438 source acquisition label")
    group, number = match["group"], int(match["sample"])
    if (group, number) not in _EXPECTED_SAMPLES:
        raise ValueError("Unexpected PXD009438 source sample label")
    return match["acquisition"], group, number


def _channel_context(experiment):
    _acquisition, group, number = _source_sample(experiment)
    return {
        "capture_label_raw": None,
        "capture_role": "project_reported_proteasome_capture",
        "capture_target_gene": None,
        "treatment_label_raw": group,
        "treatment_kind": "source_labelled_clinical_group",
        "treatment_name": None,
        "treatment_context_status": "source_run_label_without_physical_specimen_identity",
        "treatment_dose": None,
        "treatment_dose_unit": None,
        "duration_hours": None,
        "time_hours": None,
        "condition": group,
        "condition_role": "source_labelled_clinical_group",
        "replicate_label": None,
        "measured_species": "Homo sapiens",
        "measured_species_scope": "project_organism",
        "measured_cell": "peripheral_blood_mononuclear_cell",
        "measured_cell_scope": "publication_level_cohort_context_not_physical_run_specimen",
        "measured_pool": "proteasome_associated_fragment",
        "material_link_precision": "acquisition_run_unresolved_specimen",
        "specimen_id": None,
        "aliquot_id": None,
        "pair_id": None,
        "source_sample_label_raw": f"{group}_{number}",
    }


def _write_material_graph(sinks, contexts, locators):
    project_id = f"{SOURCE_ID}:project"
    rows = [
        {
            "source_dataset": ACCESSION,
            "context_id": project_id,
            "context_kind": "source_project",
            "sample_label_raw": None,
            "condition_id": None,
            "condition_text": None,
            "condition_status": "not_applicable_source_project",
            "specimen_id": None,
            "aliquot_id": None,
            "material_link_precision": "project_context",
            "context_status": "source_project_context",
            "source_locator": "project.json",
            "notes": "Project identity does not establish a physical specimen or aliquot.",
        }
    ]
    links = []
    for group in ("CTRL", "LUPUS"):
        arm_id = f"{SOURCE_ID}:arm:{group}"
        members = sorted(
            experiment for experiment, item in contexts.items() if item["condition"] == group
        )
        arm_locator = ";".join(locators[experiment] for experiment in members)
        rows.append(
            {
                "source_dataset": ACCESSION,
                "context_id": arm_id,
                "context_kind": "study_arm",
                "sample_label_raw": group,
                "condition_id": arm_id,
                "condition_text": group,
                "condition_status": "source_run_label",
                "specimen_id": None,
                "aliquot_id": None,
                "material_link_precision": "study_arm",
                "context_status": "source_labelled_clinical_group_without_physical_material_identity",
                "source_locator": arm_locator,
                "notes": "CTRL and LUPUS are literal deposited source labels, not specimen IDs.",
            }
        )
        links.append(
            {
                "source_dataset": ACCESSION,
                "claim_id": f"{arm_id}:in_project",
                "source_context_id": arm_id,
                "target_context_id": project_id,
                "assertion_predicate": "study_arm_in_project",
                "assertion_status": "source_reported",
                "material_link_precision": "project_context",
                "source_locator": arm_locator,
                "notes": "The deposited run labels establish group membership in this project only.",
            }
        )
        for experiment in members:
            _acquisition, _group, number = _source_sample(experiment)
            sample_id = f"{SOURCE_ID}:source-sample:{group}_{number}"
            channel_id = f"{SOURCE_ID}:run:{experiment}"
            rows.append(
                {
                    "source_dataset": ACCESSION,
                    "context_id": sample_id,
                    "context_kind": "source_sample",
                    "sample_label_raw": f"{group}_{number}",
                    "condition_id": arm_id,
                    "condition_text": group,
                    "condition_status": "source_run_label",
                    "specimen_id": None,
                    "aliquot_id": None,
                    "material_link_precision": "source_acquisition",
                    "context_status": "source_labelled_sample_without_physical_specimen_or_aliquot_identity",
                    "source_locator": locators[experiment],
                    "notes": "A source sample label is not promoted to a physical specimen identifier.",
                }
            )
            rows.append(
                {
                    "source_dataset": ACCESSION,
                    "context_id": channel_id,
                    "context_kind": "source_acquisition",
                    "sample_label_raw": f"{group}_{number}",
                    "condition_id": arm_id,
                    "condition_text": group,
                    "condition_status": "source_run_label",
                    "specimen_id": None,
                    "aliquot_id": None,
                    "material_link_precision": "source_acquisition",
                    "context_status": "source_acquisition_unresolved_specimen_aliquot",
                    "source_locator": locators[experiment],
                    "notes": "The acquisition is linked to a deposited source sample label only.",
                }
            )
            links.extend(
                [
                    {
                        "source_dataset": ACCESSION,
                        "claim_id": f"{sample_id}:in_arm",
                        "source_context_id": sample_id,
                        "target_context_id": arm_id,
                        "assertion_predicate": "source_sample_in_study_arm",
                        "assertion_status": "source_reported",
                        "material_link_precision": "study_arm",
                        "source_locator": locators[experiment],
                        "notes": "The source sample label carries this literal group label.",
                    },
                    {
                        "source_dataset": ACCESSION,
                        "claim_id": f"{channel_id}:source_sample",
                        "source_context_id": channel_id,
                        "target_context_id": sample_id,
                        "assertion_predicate": "source_acquisition_to_source_sample_label",
                        "assertion_status": "source_reported",
                        "material_link_precision": "source_acquisition",
                        "source_locator": locators[experiment],
                        "notes": "This is not a specimen or aliquot pairing claim.",
                    },
                ]
            )
    validate_material_graph(rows, links)
    for row in rows:
        sinks["material_contexts"].add(row)
    for row in links:
        sinks["material_links"].add(row)


def write_pbmc_tables(source_dir):
    """Materialize lossless source rows and replay deposited peptide intensities."""
    source_dir = Path(source_dir)
    (source_dir / "typed").mkdir()
    sinks = {
        name: _Sink(source_dir / "typed", name, schema=schema) for name, schema in SCHEMAS.items()
    }
    try:
        runs, contexts, locators = {}, {}, {}
        for locator, row in _rows(source_dir, "summary", sinks):
            experiment = row["Experiment"]
            if not experiment:
                continue
            if experiment in runs or row["Raw file"] in {
                value["Raw file"] for value in runs.values()
            }:
                raise ValueError("Duplicate PXD009438 source acquisition")
            acquisition = _source_sample(experiment)[0]
            if acquisition not in row["Raw file"]:
                raise ValueError("PXD009438 raw-file/acquisition label mismatch")
            runs[experiment], contexts[experiment], locators[experiment] = (
                row,
                _channel_context(experiment),
                locator,
            )
            context = contexts[experiment]
            sinks["channels"].add(
                {
                    "source_dataset": ACCESSION,
                    "channel_id": f"{SOURCE_ID}:run:{experiment}",
                    "source_record_id": locator,
                    "experiment_raw": experiment,
                    "raw_file": row["Raw file"],
                    "search_enzyme_mode_raw": row["Enzyme mode"],
                    **{
                        key: value
                        for key, value in context.items()
                        if key != "source_sample_label_raw"
                    },
                }
            )
        observed_samples = {
            (_source_sample(experiment)[1], _source_sample(experiment)[2]) for experiment in runs
        }
        if observed_samples != _EXPECTED_SAMPLES or len(runs) != len(_EXPECTED_SAMPLES):
            raise ValueError("PXD009438 source sample design changed")
        _write_material_graph(sinks, contexts, locators)

        peptides, peptide_locators = {}, {}
        for locator, row in _rows(source_dir, "peptides", sinks):
            peptide_id = row["id"]
            if not peptide_id.isdigit() or peptide_id in peptides:
                raise ValueError("Invalid or repeated PXD009438 peptide identity")
            peptides[peptide_id], peptide_locators[peptide_id] = row, locator
            sinks["peptide_rows"].add(
                {
                    "source_dataset": ACCESSION,
                    "peptide_row_id": f"{SOURCE_ID}:peptide:{peptide_id}",
                    "source_record_id": locator,
                    "peptide_id_raw": peptide_id,
                    "sequence_raw": row["Sequence"],
                    "parent_labels_raw": row["Proteins"],
                    "leading_parent_raw": row["Leading razor protein"],
                    "group_ids_raw": row["Protein group IDs"],
                    "n_flank_raw": row["Amino acid before"],
                    "c_flank_raw": row["Amino acid after"],
                    "start_1based_reported": _integer(row["Start position"]),
                    "end_1based_reported": _integer(row["End position"]),
                    "pep": None if row["PEP"] in ("", "NaN") else _number(row["PEP"]),
                    "pep_raw": row["PEP"],
                    "pep_status": "unreported_source_nan"
                    if row["PEP"] == "NaN"
                    else "unreported"
                    if not row["PEP"]
                    else "reported",
                    "reverse": row["Reverse"] == "+",
                    "potential_contaminant": row["Potential contaminant"] == "+",
                    "measured_pool": "proteasome_associated_fragment",
                    "material_link_precision": "acquisition_run_unresolved_specimen",
                }
            )

        groups = {}
        for locator, row in _rows(source_dir, "proteinGroups", sinks):
            group_id = row["id"]
            if group_id.isdigit() and group_id not in groups:
                groups[group_id] = row
                status = "source_group"
            else:
                status = "quarantined_missing_duplicate_or_nonnumeric_id"
            sinks["groups"].add(
                {
                    "source_dataset": ACCESSION,
                    "group_record_id": locator,
                    "source_record_id": locator,
                    "group_id_raw": group_id,
                    "protein_ids_raw": row["Protein IDs"],
                    "peptide_ids_raw": row["Peptide IDs"],
                    "status": status,
                    "reverse": row["Reverse"] == "+",
                    "potential_contaminant": row["Potential contaminant"] == "+",
                }
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
        for locator, row in _rows(source_dir, "evidence", sinks):
            peptide_id, experiment, evidence_id = row["Peptide ID"], row["Experiment"], row["id"]
            if (
                not evidence_id.isdigit()
                or evidence_id in seen
                or peptide_id not in peptides
                or experiment not in runs
            ):
                raise ValueError("Invalid PXD009438 evidence relationship")
            if (
                row["Sequence"] != peptides[peptide_id]["Sequence"]
                or row["Raw file"] != runs[experiment]["Raw file"]
            ):
                raise ValueError("PXD009438 evidence sequence/acquisition mismatch")
            if any(
                group_id not in groups
                for group_id in row["Protein group IDs"].split(";")
                if group_id
            ):
                raise ValueError("PXD009438 evidence references missing protein group")
            seen.add(evidence_id)
            reciprocal[peptide_id].add(evidence_id)
            key, value = (peptide_id, experiment), _number(row["Intensity"])
            totals[key] += value or 0.0
            counts[key] += 1
            numeric[key] += value is not None
            msms = _integer(row["MS/MS Count"])
            is_direct = bool(msms and row["Type"] != "MULTI-MATCH")
            direct[key] += is_direct
            positive_direct[key] += bool(is_direct and value is not None and value > 0)
            matched[key] += row["Type"] == "MULTI-MATCH"
            source_evidence_id = f"{SOURCE_ID}:evidence:{evidence_id}"
            sinks["evidence"].add(
                {
                    "source_dataset": ACCESSION,
                    "evidence_id": source_evidence_id,
                    "source_record_id": locator,
                    "peptide_row_id": f"{SOURCE_ID}:peptide:{peptide_id}",
                    "channel_id": f"{SOURCE_ID}:run:{experiment}",
                    "sequence_raw": row["Sequence"],
                    "modified_sequence_raw": row["Modified sequence"],
                    "modifications_raw": row["Modifications"],
                    "type_raw": row["Type"],
                    "group_ids_raw": row["Protein group IDs"],
                    "intensity_raw": row["Intensity"],
                    "intensity": value,
                    "intensity_status": "unreported" if value is None else "reported",
                    "msms_count": msms,
                    "pep": None if row["PEP"] in ("", "NaN") else _number(row["PEP"]),
                    "pep_raw": row["PEP"],
                    "pep_status": "unreported_source_nan"
                    if row["PEP"] == "NaN"
                    else "unreported"
                    if not row["PEP"]
                    else "reported",
                    "reverse": row["Reverse"] == "+",
                    "potential_contaminant": row["Potential contaminant"] == "+",
                }
            )
            sinks["derivation_inputs"].add(
                {
                    "source_dataset": ACCESSION,
                    "derived_measurement_id": f"{SOURCE_ID}:peptide:{peptide_id}:run:{experiment}:evidence_sum",
                    "input_measurement_id": source_evidence_id,
                    "input_role": "source_evidence_intensity",
                }
            )

        columns = [name for name in next(iter(peptides.values())) if name.startswith("Intensity ")]
        aliases = match_quantity_columns(columns, runs)
        for column, (experiment, rule) in aliases.items():
            sinks["label_aliases"].add(
                {
                    "source_dataset": ACCESSION,
                    "alias_id": f"{SOURCE_ID}:run:{experiment}:quantity_alias",
                    "channel_id": f"{SOURCE_ID}:run:{experiment}",
                    "quantity_column_raw": column,
                    "summary_experiment_raw": experiment,
                    "match_rule": rule,
                    "source_locator": "txt/peptides.txt:header;txt/summary.txt",
                }
            )

        maximum_error, n_positive, n_zero = 0.0, 0, 0
        for peptide_id, row in peptides.items():
            if set(filter(None, row["Evidence IDs"].split(";"))) != reciprocal[peptide_id]:
                raise ValueError("PXD009438 peptide/evidence reciprocal relation changed")
            for group_id in filter(None, row["Protein group IDs"].split(";")):
                if group_id not in groups or peptide_id not in groups[group_id][
                    "Peptide IDs"
                ].split(";"):
                    raise ValueError("PXD009438 peptide/group reciprocal relation changed")
            for column, (experiment, _rule) in aliases.items():
                result_id = f"{SOURCE_ID}:peptide:{peptide_id}:run:{experiment}"
                key = peptide_id, experiment
                reported, replay = _number(row[column]), totals[key]
                if reported is None or not math.isclose(
                    reported, replay, rel_tol=1e-4, abs_tol=1.0
                ):
                    raise ValueError("PXD009438 reported peptide intensity does not replay")
                maximum_error = max(
                    maximum_error, abs(reported - replay) / max(reported, replay, 1.0)
                )
                n_positive += reported > 0
                n_zero += reported == 0
                sinks["quantity_results"].add(
                    {
                        "source_dataset": ACCESSION,
                        "result_id": result_id,
                        "peptide_row_id": f"{SOURCE_ID}:peptide:{peptide_id}",
                        "channel_id": f"{SOURCE_ID}:run:{experiment}",
                        "measured_pool": "proteasome_associated_fragment",
                        "endpoint": "captured_fragment_MS_intensity",
                        "material_link_precision": "acquisition_run_unresolved_specimen",
                        "n_source_evidence": counts[key],
                        "n_numeric_evidence": numeric[key],
                        "n_direct_msms_evidence": direct[key],
                        "n_positive_direct_msms_evidence": positive_direct[key],
                        "n_matched_evidence": matched[key],
                    }
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
                        "txt/evidence.txt",
                    ),
                    (
                        "reported_run_intensity",
                        reported,
                        "reported_positive" if reported > 0 else "reported_zero_not_absence",
                        replay,
                        peptide_locators[peptide_id] + ":" + column,
                    ),
                ):
                    sinks["quantity_measurements"].add(
                        {
                            "source_dataset": ACCESSION,
                            "measurement_id": result_id + ":" + representation,
                            "result_id": result_id,
                            "peptide_row_id": f"{SOURCE_ID}:peptide:{peptide_id}",
                            "representation": representation,
                            "value_status": status,
                            "unit": "arbitrary_intensity",
                            "quantity_scale": "relative_label_free_signal",
                            "source_locator": locator,
                            "derivation_rule": "sum_reported_evidence_intensities;missing_values_not_measured_zero",
                            "value": value,
                            "recomputed_value": reconstructed,
                            "absolute_replay_error": abs(value - reconstructed)
                            if reconstructed is not None
                            else None,
                        }
                    )
                sinks["derivation_inputs"].add(
                    {
                        "source_dataset": ACCESSION,
                        "derived_measurement_id": result_id + ":reported_run_intensity",
                        "input_measurement_id": result_id + ":evidence_sum",
                        "input_role": "arithmetic_replay",
                    }
                )

        for locator, row in _rows(source_dir, "parameters", sinks):
            if set(row) != {"Parameter", "Value"}:
                raise ValueError("Unexpected PXD009438 MaxQuant parameter columns")
            sinks["source_assertions"].add(
                {
                    "source_dataset": ACCESSION,
                    "assertion_id": locator + ":parameter",
                    "subject_id": SOURCE_ID,
                    "predicate": row["Parameter"],
                    "value_raw": row["Value"],
                    "normalized_value": None,
                    "assertion_status": "source_reported",
                    "source_locator": locator,
                    "scope": "deposited_search_parameters",
                }
            )
        contract = json.loads((source_dir / "contracts/contract.json").read_text())
        for index, row in enumerate(contract["assertions"]):
            sinks["source_assertions"].add(
                {"source_dataset": ACCESSION, "assertion_id": f"{SOURCE_ID}:curated:{index}", **row}
            )
    finally:
        for sink in sinks.values():
            sink.close()
    return {
        "n_peptides": len(peptides),
        "n_groups": len(groups),
        "n_evidence": len(seen),
        "n_runs": len(runs),
        "n_positive_run_quantities": n_positive,
        "n_zero_run_quantities": n_zero,
        "n_missing_evidence_intensities": sum(counts.values()) - sum(numeric.values()),
        "maximum_relative_replay_error": maximum_error,
        "relation_rows": {name: sink.n_rows for name, sink in sinks.items()},
    }


class PBMCMAPPAdapter:
    source_id = SOURCE_ID
    schema_version = 1
    asset_receipt_path = ASSET_RECEIPT
    support_files = SUPPORT_FILES

    def write(self, source_dir):
        source_dir = Path(source_dir)
        contract = json.loads((source_dir / "contracts/contract.json").read_text())
        receipt = source_dir / "contracts/source_assets.json"
        if hashlib.sha256(receipt.read_bytes()).hexdigest() != contract["source_receipt_sha256"]:
            raise ValueError("PXD009438 contract receipt hash changed")
        return write_pbmc_tables(source_dir)

    def verify(self, source_dir, reconciliation):
        for name, schema in SCHEMAS.items():
            parquet = pq.ParquetFile(Path(source_dir) / "typed" / f"{name}.parquet")
            if (
                parquet.schema_arrow != schema
                or parquet.metadata.num_rows != reconciliation["relation_rows"][name]
            ):
                raise ValueError(f"PXD009438 typed relation mismatch: {name}")
        expected = {
            "n_peptides": 985,
            "n_groups": 283,
            "n_evidence": 8851,
            "n_runs": 14,
            "n_positive_run_quantities": 7653,
            "n_zero_run_quantities": 6137,
            "n_missing_evidence_intensities": 185,
            "maximum_relative_replay_error": 6.500425353919898e-05,
        }
        expected_rows = {
            "source_records": 10207,
            "peptide_rows": 985,
            "groups": 283,
            "channels": 14,
            "evidence": 8851,
            "quantity_results": 13790,
            "quantity_measurements": 27580,
            "derivation_inputs": 22641,
            "source_assertions": 65,
            "material_contexts": 31,
            "material_links": 30,
            "label_aliases": 14,
        }
        for key, value in expected.items():
            if reconciliation.get(key) != value:
                raise ValueError(f"Pinned PXD009438 source count changed: {key}")
        if reconciliation["relation_rows"] != expected_rows:
            raise ValueError("Pinned PXD009438 relation count changed")

    def registry(self, reconciliation):
        return {
            "source_id": SOURCE_ID,
            "dataset_accession": ACCESSION,
            "organism_scope": "human_project_scope",
            "measured_pools": ["proteasome_associated_fragment"],
            "measurement_grain": "deposited_peptide_row_by_source_acquisition_run",
            "training_admission": "deferred_no_endpoint_specific_recipe",
            "quantity_artifact": "typed/quantity_results.parquet",
            "material_context_artifact": "typed/material_contexts.parquet",
            "material_link_artifact": "typed/material_links.parquet",
            "rights": {"PRIDE_deposit": "EBI terms of use"},
            "n_primary_peptide_rows": reconciliation["n_peptides"],
            "limitations": [
                "captured_pool_not_free_concentration",
                "source_sample_labels_not_physical_specimen_identity",
                "unresolved_specimens_and_aliquots",
                "no_cross_modal_or_cross_patient_pairing",
                "project_metadata_blood_serum_conflicts_with_publication_PBMC_description",
            ],
        }
