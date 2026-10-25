"""Source-addressed nominal standards, protocols and precision-qualified context."""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET

from .dilution_observer_compiler import identity, number
from .material_context import validate_material_graph


def assertion(
    compiler,
    predicate,
    value,
    record,
    locator,
    *,
    subject="project",
    status="source_reported",
    details=None,
):
    compiler.add(
        "assertions",
        {
            "claim_id": identity("assertion", record, subject, predicate, locator),
            "source_record_id": record,
            "context_id": identity("context", subject),
            "subject_id": identity("context", subject),
            "source_scope": subject,
            "assertion_predicate": predicate,
            "assertion_value": value,
            "assertion_status": status,
            "source_locator": locator,
            "details_json": json.dumps(details or {}, sort_keys=True),
        },
    )


def compile_metadata(compiler):
    c = compiler
    project_name = c.contract["project"]
    project = json.loads((c.source / project_name).read_text())
    if project["accession"] != c.contract["source_dataset"]:
        raise ValueError("Project accession differs")
    c.meta_records = {}
    for key, value in project.items():
        literal = value if isinstance(value, str) else json.dumps(value, ensure_ascii=True)
        locator = project_name + "#/" + key
        record = c.record(
            project_name,
            locator,
            ["key", "value_json"],
            [key, json.dumps(value, ensure_ascii=True)],
        )
        c.meta_records[key] = record
        assertion(c, "project." + key, literal, record, locator)

    inventory_name = c.contract["inventory"]
    inventory = json.loads((c.source / inventory_name).read_text())
    c.raw_names = {row["fileName"] for row in inventory}
    for index, row in enumerate(inventory):
        c.record(
            inventory_name,
            f"{inventory_name}#/{index}",
            list(row),
            [json.dumps(v, ensure_ascii=True) for v in row.values()],
        )

    article_name = c.contract["article"]
    article = ET.parse(c.source / article_name).getroot()
    table = article.find(".//table-wrap[@id='T1']")
    if table is None:
        raise ValueError("Original nominal-level Table1 is missing")
    rows = table.findall(".//tbody/tr")
    if len(rows) != c.contract["n_nominal_levels"]:
        raise ValueError("Nominal-level panel differs")
    for index, row in enumerate(rows, 1):
        cells = ["".join(cell.itertext()) for cell in row.findall("td")]
        if len(cells) != 3:
            raise ValueError("Unexpected nominal-level Table1 schema")
        factor, sample, concentration = cells
        value, unit = concentration.split()
        sample_number = re.fullmatch(r"No\.\s*(\d+)", sample)
        if sample_number is None or number(value)[1] != "positive":
            raise ValueError("Nominal source level is not interpretable")
        locator = f"{article_name}#table-wrap:T1/tbody/tr:{index}"
        record = c.record(
            article_name, locator, ["dilution_factor", "sample", "concentration"], cells
        )
        level = {
            "nominal_level_id": identity("level", sample),
            "source_record_id": record,
            "sample_label_raw": sample,
            "dilution_factor_raw": factor,
            "nominal_concentration_raw": value,
            "nominal_concentration": number(value)[0],
            "measurement_unit": unit,
            "nominal_status": "reported_nominal_solution_not_injected_amount",
        }
        key = sample_number.group(1)
        if key in c.levels:
            raise ValueError("Ambiguous nominal sample number")
        c.levels[key] = level
        c.add("nominal_levels", level)
        assertion(
            c,
            "nominal_dilution_grid",
            json.dumps(cells, ensure_ascii=True),
            record,
            locator,
            status="source_reported_conflict_retained",
        )
    for index, paragraph in enumerate(article.findall(".//body//p"), 1):
        text = "".join(paragraph.itertext())
        locator = f"{article_name}#body/p:{index}"
        record = c.record(article_name, locator, ["text"], [text])
        assertion(c, "article.paragraph", text, record, locator)
        if "1:10" in text:
            assertion(
                c,
                "nominal_dilution_grid",
                text,
                record,
                locator,
                status="source_reported_conflict_retained",
            )

    protocol = project["sampleProcessingProtocol"]
    panel = re.findall(r"[A-Z]+\[L\(13C6; 15N\)\][A-Z]+", protocol)
    if len(panel) != c.contract["n_nominal_standards"] or len(set(panel)) != len(panel):
        raise ValueError("Intended standard panel differs")
    for form in panel:
        left, right = form.split("[L(13C6; 15N)]")
        standard = {
            "standard_id": identity("standard", form),
            "source_record_id": c.meta_records["sampleProcessingProtocol"],
            "peptide_sequence": left + "L" + right,
            "reported_standard_form": form,
            "expected_modified_sequence": left + "L(UniMod:695)" + right,
            "standard_group_form": left + "L(Label:13C(6)15N(1))" + right,
            "label_position_1based": len(left) + 1,
            "nominal_status": "intended_synthetic_panel_not_measured_restriction",
        }
        c.standards.append(standard)
        c.add("nominal_standards", standard)

    # Retain original source code, including its argument wiring, without claiming
    # that a reference default proves the actual study's executable configuration.
    for asset in c.contract["assets"]:
        if asset["role"] != "pipeline_script":
            continue
        for line, text in enumerate(
            (c.source / asset["filename"]).read_text().splitlines(keepends=True), 1
        ):
            locator = f"{asset['filename']}:line:{line}"
            record = c.record(asset["filename"], locator, ["source_code_line"], [text])
            if any(
                token in text
                for token in (
                    "pyprophet export",
                    "max_rs_peakgroup_qvalue",
                    "max_global_peptide_qvalue",
                    "max_global_protein_qvalue",
                    "alignTargetedRuns",
                    "mv DIAlignR.tsv",
                )
            ):
                assertion(
                    c,
                    "reference_pipeline_command",
                    text,
                    record,
                    locator,
                    status="pinned_reference_code_not_actual_run_configuration",
                )

    c.contexts, c.context_links = [], []
    for role, kind, label, description in [
        ("project", "source_project", c.contract["source_dataset"], project.get("title")),
        ("matrix", "study_arm", "JY HLA-I eluate matrix", project.get("projectDescription")),
        ("standards", "study_arm", "post-isolation synthetic standard panel", protocol),
    ]:
        c.contexts.append(
            {
                "context_id": identity("context", role),
                "context_kind": kind,
                "sample_label_raw": label,
                "condition_text": description,
                "condition_status": "source_protocol_context",
                "material_link_precision": "project_context",
                "context_status": "physical_specimen_and_aliquot_unqualified",
                "source_locator": project_name,
                "notes": "Source context does not create physical material identity",
            }
        )
    description = project.get("projectDescription", "")
    for predicate, value, key in [
        ("cell_line_name", "JY" if "JY" in description else None, "projectDescription"),
        ("mhc_class", "I" if "HLA-I" in description else None, "projectDescription"),
        (
            "cellular_mhc_genotype",
            json.dumps(re.findall(r"HLA-[ABC]\*\d+:\d+", description)),
            "projectDescription",
        ),
        ("organism", json.dumps(project.get("organisms", [])), "organisms"),
        ("instrument", json.dumps(project.get("instruments", [])), "instruments"),
    ]:
        if key in c.meta_records:
            assertion(
                c,
                predicate,
                value,
                c.meta_records[key],
                project_name + "#/" + key,
                subject="matrix",
                status="source_reported_matrix_context_not_standard_restriction",
            )
    for predicate in (
        "per_run_injection_volume",
        "certified_standard_purity",
        "standard_lot",
        "physical_aliquot_lineage",
        "per_run_recovery",
        "actual_executable_parameters",
        "APC_machinery_state",
    ):
        assertion(
            c,
            predicate,
            None,
            c.meta_records["sampleProcessingProtocol"],
            "curation_contract:" + predicate,
            status="unqualified_in_this_source_revision",
            details={
                "scope": "No claim that the value is absent from all literature or supplements"
            },
        )

    for claim in c.contract.get("protocol_assertions", []):
        key = claim["project_field"]
        if claim["evidence_text"] not in project[key]:
            raise ValueError("Protocol assertion lacks its pinned source text")
        assertion(
            c,
            claim["predicate"],
            claim["value_raw"],
            c.meta_records[key],
            project_name + "#/" + key,
            subject=claim["subject"],
            status="reported_protocol_not_individual_run_receipt",
            details={
                "unit": claim["unit"],
                "evidence_text": claim["evidence_text"],
                "applicability": claim["applicability"],
            },
        )


def add_acquisition(compiler, run, run_id, record):
    c = compiler
    if run in c.acquisitions:
        acquisition = c.acquisitions[run]
        if acquisition["source_run_id"] != run_id:
            raise ValueError("One run label has conflicting source run IDs")
        return acquisition["acquisition_id"]
    match = re.search(r"_No(\d+)_Dilution[^_]+_[^_]+fmol-ul_DIA#([12])_", run)
    if match is None or match.group(1) not in c.levels:
        raise ValueError("Unqualified acquisition-to-nominal-level filename")
    level = c.levels[match.group(1)]
    raw = run.replace("#", "_") + ".raw"
    context_id = identity("context", "acquisition", run)
    acquisition = {
        "acquisition_id": identity("acquisition", run, run_id),
        "context_id": context_id,
        "source_record_id": record,
        "raw_file": raw if raw in c.raw_names else None,
        "source_run_id": run_id,
        "source_run_label": run,
        "nominal_level_id": level["nominal_level_id"],
        "dia_index_from_name": int(match.group(2)),
        "method_link_precision": "filename_derived",
        "material_link_precision": "source_acquisition",
        "replicate_status": "different_named_DIA_acquisitions_not_assumed_exchangeable_replicates",
    }
    c.acquisitions[run] = acquisition
    c.add("acquisitions", acquisition)
    c.contexts.append(
        {
            "context_id": context_id,
            "context_kind": "source_acquisition",
            "sample_label_raw": run,
            "condition_id": level["nominal_level_id"],
            "condition_status": "filename_derived_nominal_level",
            "material_link_precision": "source_acquisition",
            "context_status": "physical_material_unqualified",
            "source_locator": record,
            "notes": "DIA index is a method label; no independent specimen or replicate inference",
        }
    )
    for subject in ("matrix", "standards"):
        c.context_links.append(
            {
                "claim_id": identity("context_link", run, subject),
                "source_context_id": context_id,
                "target_context_id": identity("context", subject),
                "assertion_predicate": "source_protocol_material_context",
                "assertion_status": "source_protocol_not_physical_pairing",
                "material_link_precision": "study_arm",
                "source_locator": c.contract["project"] + "#/sampleProcessingProtocol",
            }
        )
    return acquisition["acquisition_id"]


def finish_contexts(compiler):
    validate_material_graph(compiler.contexts, compiler.context_links)
    for row in compiler.contexts:
        compiler.add("material_contexts", row)
    for row in compiler.context_links:
        compiler.add("material_links", row)
