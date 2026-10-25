"""Original native-cell evidence; source interpretation never selects a cohort."""

from __future__ import annotations

import hashlib
import json
import re
import xml.etree.ElementTree as ET
import zipfile
from collections import defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path

from openpyxl.styles.numbers import BUILTIN_FORMATS, is_date_format

from .goldberg_mapp import _confidence, _fields, _json, _number, _tsv
from .intact_degradomics_workbook import workbook_rows
from .mapp_evidence import _Sink
from .material_context import validate_material_graph
from .native_peptidomics_reference import Dependencies, candidate_spans, fasta_records
from .native_peptidomics_schema import SCHEMAS


def _tokens(value):
    return [token for token in (value or "").split(";") if token]


def _decimal(value):
    try:
        result = Decimal(str(value))
        return result if result.is_finite() else None
    except (InvalidOperation, ValueError):
        return None


class Compiler:
    def __init__(self, assets, destination, contract):
        self.assets, self.destination, self.contract = assets, destination, contract
        destination.mkdir(parents=True)
        self.sinks = {
            name: _Sink(destination, name, schema=schema) for name, schema in SCHEMAS.items()
        }
        self.common = {
            k: contract[k] for k in ("source_dataset", "source_dataset_precision", "source_panel")
        }
        self.contexts, self.material_links = [], []
        self.references, self.reference_accessions = [], defaultdict(list)
        self.peptides, self.forms, self.identifications = {}, {}, {}
        self.originals, self.acquisitions, self.xml_records = {}, {}, {}
        self.dependencies = Dependencies()

    def key(self, kind, *parts):
        return (
            self.contract["id_prefix"]
            + ":"
            + kind
            + ":"
            + hashlib.sha256(_json(parts).encode()).hexdigest()
        )

    def add(self, table, row):
        self.sinks[table].add({**row, **self.common})

    def record(self, filename, locator, values, index=None):
        key = self.key("source", filename, locator)
        self.add(
            "source_records",
            {
                "source_record_id": key,
                "source_file": filename,
                "source_locator": locator,
                "source_row_index": index,
                "source_row_json": _json(values),
                "source_asset_id": self.key("asset", filename),
            },
        )
        return key

    def rows(self, filename):
        with (self.assets / filename).open(encoding="utf-8-sig", newline="") as stream:
            for index, headers, values in _tsv(stream):
                record = self.record(
                    filename, f"row:{index}", {"headers": headers, "values": values}, index
                )
                yield record, _fields(headers, values)

    def assertion(
        self, record, subject, predicate, value, locator, status="reported", scope="source"
    ):
        self.add(
            "assertions",
            {
                "claim_id": self.key("assertion", record, subject, predicate, locator),
                "source_record_id": record,
                "subject_id": subject,
                "source_scope": scope,
                "assertion_predicate": predicate,
                "assertion_value": None if value is None else str(value),
                "assertion_status": status,
                "source_locator": locator,
            },
        )

    def reconcile(self, record, subject, target, predicate, status, locator, details):
        self.add(
            "reconciliations",
            {
                "claim_id": self.key("reconciliation", record, subject, target, predicate, locator),
                "source_record_id": record,
                "source_subject_id": subject,
                "target_subject_id": target,
                "assertion_predicate": predicate,
                "assertion_status": status,
                "source_locator": locator,
                "details_json": _json(details),
            },
        )

    def context(self, key, kind, label, locator, condition=None):
        precision = {
            "source_project": "project_context",
            "study_arm": "study_arm",
            "source_acquisition": "source_acquisition",
        }[kind]
        self.contexts.append(
            {
                "context_id": key,
                "context_kind": kind,
                "sample_label_raw": label,
                "condition_id": condition,
                "condition_text": label,
                "condition_status": "source_reported_condition",
                "specimen_id": None,
                "aliquot_id": None,
                "material_link_precision": precision,
                "context_status": "source_context_not_inferred_physical_identity",
                "source_locator": locator,
                "notes": "Numbered acquisitions do not establish pairing or biological replicate independence",
            }
        )

    def material_link(self, source, target, precision, locator):
        self.material_links.append(
            {
                "claim_id": self.key("material", source, target),
                "source_context_id": source,
                "target_context_id": target,
                "assertion_predicate": "source_context_membership",
                "assertion_status": "reported",
                "material_link_precision": precision,
                "source_locator": locator,
                "notes": "Not an exact specimen or aliquot link",
            }
        )

    def metadata(self):
        document_records = {}
        for filename in self.contract["metadata_json"]:
            document_records[filename] = self.record(
                filename, "document", json.loads((self.assets / filename).read_text())
            )
        for filename in self.contract["metadata_xml"]:
            tree = ET.parse(self.assets / filename)

            def visit(node, path, filename=filename):
                record = self.record(
                    filename,
                    path,
                    {
                        "tag": node.tag,
                        "text": node.text,
                        "tail": node.tail,
                        "attributes": node.attrib,
                    },
                )
                self.xml_records[filename, path] = record
                for index, child in enumerate(node):
                    visit(child, f"{path}/{child.tag}[{index}]")

            visit(tree.getroot(), "/")
            document_records[filename] = self.xml_records[filename, "/"]
        self.project_context = self.key("context", "project")
        self.context(
            self.project_context,
            "source_project",
            self.contract["source_dataset"],
            self.contract["project_metadata"],
        )
        for claim in self.contract["assertions"]:
            subject = (
                self.key("condition", claim["condition"])
                if claim.get("condition")
                else self.project_context
            )
            self.assertion(
                document_records[claim["source_file"]],
                subject,
                claim["predicate"],
                claim.get("value"),
                claim["locator"],
                claim["status"],
                claim["scope"],
            )
        design = list(self.rows(self.contract["design"]))
        design_map = {r["Experiment"]: r["Name"] for _, r in design}
        configured = {r["sample_label"]: r["raw_file"] for r in self.contract["acquisitions"]}
        mqpar = ET.parse(self.assets / self.contract["mqpar"])
        raw_names = [
            Path(node.text.replace("\\", "/")).stem for node in mqpar.findall("filePaths/string")
        ]
        labels = [node.text for node in mqpar.findall("experiments/string")]
        if (
            len(design_map) != len(design)
            or len(labels) != len(set(labels))
            or dict(zip(labels, raw_names, strict=True)) != configured
            or design_map != configured
        ):
            raise ValueError(
                "Original acquisition design/configuration differs from the pinned contract"
            )
        summaries = list(self.rows("summary.txt"))
        summary_map = {
            r["Experiment"]: r["Raw file"] for _, r in summaries if r["Raw file"] != "Total"
        }
        if summary_map != configured:
            raise ValueError("Original summary acquisition mapping differs")
        # Preserve all parameter rows, including duplicate parameter labels.
        list(self.rows("parameters.txt"))
        conditions = set()
        record_by_sample = {r["Experiment"]: record for record, r in design}
        for acquisition in self.contract["acquisitions"]:
            sample, raw = acquisition["sample_label"], acquisition["raw_file"]
            condition = self.key("condition", acquisition["condition"])
            if condition not in conditions:
                conditions.add(condition)
                self.context(
                    condition,
                    "study_arm",
                    acquisition["condition"],
                    self.contract["design"],
                    condition,
                )
                self.material_link(
                    condition, self.project_context, "project_context", self.contract["design"]
                )
            key = self.key("acquisition", raw)
            self.context(key, "source_acquisition", sample, self.contract["design"], condition)
            self.material_link(key, condition, "study_arm", self.contract["design"])
            row = {
                "acquisition_id": key,
                "context_id": key,
                "source_record_id": record_by_sample[sample],
                "raw_file": raw,
                "assay_name_raw": self.contract["source_panel"],
                "sample_label_raw": sample,
                "cell_line_name": self.contract["cell_line"],
                "measured_pool": "native_intracellular_peptide_extract",
                "condition_status": "reported_control_or_knockdown_not_numeric_activity",
                "specimen_id": None,
                "aliquot_id": None,
                "material_link_precision": "source_acquisition",
                "component_state_status": "numeric_remaining_activity_unreported",
                "condition_id": condition,
                "replicate_status": "source_numbered_acquisition_independence_and_pairing_unresolved",
            }
            self.acquisitions[sample] = row
            self.add("acquisitions", row)
        self.samples = tuple(configured)
        for path in self.contract["search_paths"]:
            node = mqpar.find(path)
            value = (
                None
                if node is None
                else _json([child.text for child in node])
                if len(node)
                else node.text
            )
            self.assertion(
                document_records[self.contract["mqpar"]],
                self.project_context,
                "search:" + path,
                value,
                self.contract["mqpar"] + ":" + path,
                "unreported" if node is None else "reported",
                "search_configuration",
            )

    def reference(self):
        filename = self.contract["reference"]
        for index, (header, sequence) in enumerate(fasta_records(self.assets / filename)):
            if not sequence or not sequence.isalpha() or sequence != sequence.upper():
                raise ValueError("Invalid deposited reference sequence")
            parts = header.split("|", 2)
            if len(parts) != 3:
                raise ValueError("Unrecognized deposited reference accession grammar")
            accession = parts[1]
            record = self.record(
                filename, f"record:{index + 1}", {"header": header, "sequence": sequence}
            )
            taxon = re.search(r"(?:^| )OX=(\d+)(?: |$)", header)
            construct = self.contract["engineered_references"].get(accession)
            if construct and not header.startswith(construct["header_prefix"]):
                raise ValueError("Engineered reference assertion does not match source header")
            role = (
                "engineered_construct"
                if construct
                else "deposited_reference_endogenous_status_not_guaranteed_by_taxonomy"
            )
            row = {
                "reference_record_id": self.key("reference", index, header),
                "source_record_id": record,
                "accession_raw": accession,
                "header_raw": header,
                "organism_taxon_id": taxon[1] if taxon else None,
                "taxonomy_status": "reported_header_not_endogenous_proof"
                if taxon
                else "unreported",
                "reference_sequence": sequence,
                "reference_status": "exact_deposited_sequence",
                "source_slot_index": index + 1,
                "reference_sequence_sha256": hashlib.sha256(sequence.encode()).hexdigest(),
                "reference_role": role,
                "n_reference_residues": len(sequence),
            }
            self.references.append(row)
            self.reference_accessions[accession].append(row)
            self.add("references", row)
            if construct:
                self.assertion(
                    record,
                    row["reference_record_id"],
                    "reference_construct",
                    construct["description"],
                    construct["locator"],
                    "reported",
                    "reference_record",
                )

    def parents(self, record, observation, peptide, raw):
        for index, token in enumerate(_tokens(raw)):
            alternatives = self.reference_accessions.get(token) or [None]
            for alternative in alternatives:
                reference_id = alternative["reference_record_id"] if alternative else None
                self.add(
                    "declared_parents",
                    {
                        "parent_claim_id": self.key(
                            "parent", record, observation, index, reference_id
                        ),
                        "source_record_id": record,
                        "peptide_id": peptide,
                        "observation_id": observation,
                        "parent_accession": token,
                        "reference_record_id": reference_id,
                        "reference_link_status": "exact_deposited_accession"
                        if alternative
                        else "accession_not_in_deposited_reference",
                    },
                )
                parent = (
                    "sequence:" + alternative["reference_sequence_sha256"]
                    if alternative
                    else "source_parent:" + token
                )
                self.dependencies.link(peptide, parent)

    def quantity(
        self,
        record,
        observation,
        peptide,
        source,
        field,
        kind,
        sample=None,
        form=None,
        identification=None,
    ):
        value, status = _number(source[field])
        lfq = field.startswith("LFQ intensity")
        acquisition = self.acquisitions[sample]["acquisition_id"] if sample is not None else None
        scope = {
            "peptide_sequence": "sequence_aggregates_original_forms",
            "modification_specific_peptide": "source_modification_specific_form",
            "feature": "source_modified_sequence_charge_and_feature",
        }[kind]
        row = {
            "measurement_id": self.key("quantity", observation, field),
            "source_record_id": record,
            "observation_id": observation,
            "peptide_id": peptide,
            "acquisition_id": acquisition,
            "source_field": field,
            "source_cell": None,
            "quantity_kind": "source_lfq_intensity" if lfq else "source_intensity",
            "quantitative_raw": source[field],
            "value_status": status,
            "measurement_unit": "source_signal_arbitrary_units",
            "quantity_scale": "linear_source_signal",
            "identification_type": source.get(
                "Type", source.get("Identification type " + sample) if sample else None
            ),
            "experiment_raw": sample,
            "acquisition_link_status": "original_run_mapping"
            if sample
            else "total_across_source_acquisitions_not_one_run",
            "quantitative_value": value,
            "observation_kind": kind,
            "form_id": form,
            "identification_id": identification,
            "normalization_status": "source_lfq_export_algorithm_not_reconstructed"
            if lfq
            else "raw_intensity_column_not_lfq_or_molar_calibration",
            "quantity_form_scope": scope,
        }
        self.add("quantities", row)
        return row

    def search_tables(self):
        for record, source in self.rows("peptides.txt"):
            native = source["id"]
            if not native or native in self.peptides:
                raise ValueError("Peptide IDs must be supplied and unique")
            pep, status = _confidence(source["PEP"])
            row = {
                "peptide_id": self.key("peptide", native),
                "source_record_id": record,
                "native_id": native,
                "peptide_sequence": source["Sequence"],
                "parent_label_raw": source["Proteins"],
                "source_group_ids_raw": source.get("Protein group IDs"),
                "source_evidence_ids_raw": source["Evidence IDs"],
                "source_form_ids_raw": source["Mod. peptide IDs"],
                "start_raw": source.get("Start position"),
                "end_raw": source.get("End position"),
                "flank_n_raw": source.get("Amino acid before"),
                "flank_c_raw": source.get("Amino acid after"),
                "reverse_raw": source["Reverse"],
                "contaminant_raw": source["Potential contaminant"],
                "source_organism_status": "complete_reference_alternatives_separate_from_source_claims",
                "confidence_raw": source["PEP"],
                "confidence_status": status,
                "posterior_error_probability": pep,
                "gene_names_raw": source.get("Gene names"),
                "protein_names_raw": source.get("Protein names"),
            }
            self.peptides[native] = row
            self.originals["peptides", native] = source
            self.add("peptides", row)
            self.parents(record, row["peptide_id"], row["peptide_id"], source["Proteins"])
            for field, sample in [
                ("Intensity", None),
                *[
                    (kind + " " + s, s)
                    for kind in ("Intensity", "LFQ intensity")
                    for s in self.samples
                ],
            ]:
                self.quantity(
                    record,
                    row["peptide_id"],
                    row["peptide_id"],
                    source,
                    field,
                    "peptide_sequence",
                    sample,
                )
        for record, source in self.rows("modificationSpecificPeptides.txt"):
            native, peptide_native = source["id"], source["Peptide ID"]
            if not native or native in self.forms or peptide_native not in self.peptides:
                raise ValueError("Form IDs or original peptide relationship unresolved")
            peptide = self.peptides[peptide_native]
            row = {
                "form_id": self.key("form", native),
                "source_record_id": record,
                "native_id": native,
                "peptide_id": peptide["peptide_id"],
                "peptide_sequence": source["Sequence"],
                "parent_label_raw": source["Proteins"],
                "peptide_modifications_raw": source["Modifications"],
                "source_group_ids_raw": source.get("Protein group IDs"),
                "source_evidence_ids_raw": source["Evidence IDs"],
                "reverse_raw": source["Reverse"],
                "contaminant_raw": source["Potential contaminant"],
                "confidence_raw": source["PEP"],
            }
            self.forms[native] = row
            self.originals["forms", native] = source
            self.add("forms", row)
            self.parents(record, row["form_id"], row["peptide_id"], source["Proteins"])
            for field, sample in [
                ("Intensity", None),
                *[("Intensity " + s, s) for s in self.samples],
            ]:
                self.quantity(
                    record,
                    row["form_id"],
                    row["peptide_id"],
                    source,
                    field,
                    "modification_specific_peptide",
                    sample,
                    form=row["form_id"],
                )
        for record, source in self.rows("evidence.txt"):
            native, peptide_native, form_native, sample = (
                source["id"],
                source["Peptide ID"],
                source["Mod. peptide ID"],
                source["Experiment"],
            )
            if (
                not native
                or native in self.identifications
                or peptide_native not in self.peptides
                or form_native not in self.forms
            ):
                raise ValueError("Feature IDs or original entity relationships unresolved")
            if (
                sample not in self.acquisitions
                or source["Raw file"] != self.acquisitions[sample]["raw_file"]
            ):
                raise ValueError("Feature acquisition identity differs from the original design")
            peptide, form = self.peptides[peptide_native], self.forms[form_native]
            pep, status = _confidence(source["PEP"])
            row = {
                "identification_id": self.key("identification", native),
                "source_record_id": record,
                "native_id": native,
                "peptide_id": peptide["peptide_id"],
                "form_id": form["form_id"],
                "acquisition_id": self.acquisitions[sample]["acquisition_id"],
                "experiment_raw": sample,
                "peptide_sequence": source["Sequence"],
                "parent_label_raw": source["Proteins"],
                "peptide_modifications_raw": source["Modifications"],
                "modified_sequence_raw": source["Modified sequence"],
                "identification_type": source["Type"],
                "source_group_ids_raw": source.get("Protein group IDs"),
                "reverse_raw": source["Reverse"],
                "contaminant_raw": source["Potential contaminant"],
                "confidence_raw": source["PEP"],
                "confidence_status": status,
                "posterior_error_probability": pep,
                "charge_raw": source.get("Charge"),
                "charge_state": _number(source.get("Charge"))[0],
            }
            self.identifications[native] = row
            self.originals["identifications", native] = source
            self.add("identifications", row)
            self.parents(record, row["identification_id"], row["peptide_id"], source["Proteins"])
            self.quantity(
                record,
                row["identification_id"],
                row["peptide_id"],
                source,
                "Intensity",
                "feature",
                sample,
                form["form_id"],
                row["identification_id"],
            )

    def identifier_reconciliation(self):
        evidence_by_peptide, forms_by_peptide, evidence_by_form = (
            defaultdict(set),
            defaultdict(set),
            defaultdict(set),
        )
        for native, row in self.identifications.items():
            source = self.originals["identifications", native]
            evidence_by_peptide[source["Peptide ID"]].add(native)
            evidence_by_form[source["Mod. peptide ID"]].add(native)
            form, peptide = (
                self.forms[source["Mod. peptide ID"]],
                self.peptides[source["Peptide ID"]],
            )
            agrees = (
                row["peptide_sequence"] == form["peptide_sequence"] == peptide["peptide_sequence"]
                and form["peptide_id"] == peptide["peptide_id"]
            )
            self.reconcile(
                row["source_record_id"],
                row["identification_id"],
                form["form_id"],
                "feature_form_peptide_identity",
                "agree" if agrees else "conflicting",
                "native IDs",
                {},
            )
        for native, row in self.forms.items():
            source = self.originals["forms", native]
            forms_by_peptide[source["Peptide ID"]].add(native)
            peptide = self.peptides[source["Peptide ID"]]
            self.reconcile(
                row["source_record_id"],
                row["form_id"],
                peptide["peptide_id"],
                "form_peptide_sequence",
                "agree"
                if row["peptide_sequence"] == peptide["peptide_sequence"]
                else "conflicting",
                "native IDs",
                {},
            )
            self.reconcile(
                row["source_record_id"],
                row["form_id"],
                None,
                "form_evidence_membership",
                "agree"
                if set(_tokens(source["Evidence IDs"])) == evidence_by_form[native]
                else "conflicting",
                "Evidence IDs",
                {
                    "source": _tokens(source["Evidence IDs"]),
                    "replayed": sorted(evidence_by_form[native]),
                },
            )
        for native, row in self.peptides.items():
            source = self.originals["peptides", native]
            for field, members in (
                ("Evidence IDs", evidence_by_peptide[native]),
                ("Mod. peptide IDs", forms_by_peptide[native]),
            ):
                self.reconcile(
                    row["source_record_id"],
                    row["peptide_id"],
                    None,
                    "peptide_" + field,
                    "agree" if set(_tokens(source[field])) == members else "conflicting",
                    field,
                    {"source": _tokens(source[field]), "replayed": sorted(members)},
                )

    def publisher(self):
        filename = self.contract["publisher_workbook"]
        namespace = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
        with zipfile.ZipFile(self.assets / filename) as archive:
            styles = ET.fromstring(archive.read("xl/styles.xml"))
        formats = dict(BUILTIN_FORMATS)
        formats.update(
            {
                int(node.get("numFmtId")): node.get("formatCode")
                for node in styles.findall(namespace + "numFmts/" + namespace + "numFmt")
            }
        )
        date_styles = {
            str(index): formats.get(int(style.get("numFmtId", "0")), "")
            for index, style in enumerate(styles.find(namespace + "cellXfs"))
            if is_date_format(formats.get(int(style.get("numFmtId", "0")), ""))
        }
        by_sequence = defaultdict(list)
        for row in self.peptides.values():
            by_sequence[row["peptide_sequence"]].append(row)
        headers = {}
        for sheet, index, cells in workbook_rows(self.assets / filename):
            if index == 1:
                headers[sheet] = {c["source_column_index"]: c["cell_value_resolved"] for c in cells}
            values = {c["source_column_index"]: c["cell_value_resolved"] for c in cells}
            record = self.record(
                filename,
                f"{sheet}!row:{index}",
                {"columns": headers[sheet], "values": values},
                index,
            )
            for cell in cells:
                self.add(
                    "source_cells",
                    {
                        **cell,
                        "cell_id": self.key("cell", filename, sheet, cell["source_cell"]),
                        "source_record_id": record,
                        "source_file": filename,
                        "sheet_name": sheet,
                    },
                )
            if index == 1:
                continue
            row = {field: values.get(column) for column, field in headers[sheet].items()}
            candidates = by_sequence.get(row.get("Sequence"), [])
            if len(candidates) != 1:
                self.reconcile(
                    record,
                    record,
                    None,
                    "publisher_peptide_identity",
                    "unresolved_or_ambiguous",
                    f"{sheet}!row:{index}",
                    {"candidates": [r["peptide_id"] for r in candidates]},
                )
                continue
            peptide = candidates[0]
            original = self.originals["peptides", peptide["native_id"]]
            self.parents(record, peptide["peptide_id"], peptide["peptide_id"], row.get("Proteins"))
            for column, field in headers[sheet].items():
                left, right = values.get(column), original.get(field)
                if left in (None, "") and right in (None, ""):
                    status = "both_blank"
                elif left == right or (
                    _decimal(left) is not None and _decimal(left) == _decimal(right)
                ):
                    status = "exact_value_agreement"
                else:
                    status = "source_representation_or_value_difference"
                details = {"publisher_value": left, "original_text_value": right}
                cell = next((c for c in cells if c["source_column_index"] == column), {})
                if (
                    field == "Gene names"
                    and cell.get("cell_style_raw") in date_styles
                    and _decimal(left) is not None
                ):
                    status = "date_typed_gene_annotation_original_text_retained"
                    details.update(
                        source_format=date_styles[cell["cell_style_raw"]],
                        intended_gene_not_inferred_from_date=True,
                    )
                self.reconcile(
                    record,
                    record,
                    peptide["peptide_id"],
                    "publisher_field:" + field,
                    status,
                    f"{sheet}!row:{index}:column:{column}",
                    details,
                )

    def aggregation(self):
        sums = defaultdict(lambda: Decimal(0))
        missing = defaultdict(int)
        for (kind, _), row in self.originals.items():
            if kind == "identifications":
                observations = [(row["Experiment"], row["Intensity"])]
            elif kind == "forms":
                observations = [(sample, row["Intensity " + sample]) for sample in self.samples]
            else:
                continue
            for sample, literal in observations:
                key = kind, row["Peptide ID"], sample
                value = _decimal(literal)
                if value is None:
                    missing[key] += 1
                else:
                    sums[key] += value
        for native, peptide in self.peptides.items():
            source = self.originals["peptides", native]
            for sample in self.samples:
                target = _decimal(source["Intensity " + sample])
                for kind in ("identifications", "forms"):
                    key = kind, native, sample
                    difference = None if target is None else sums[key] - target
                    status = (
                        "missing_contributor_values"
                        if missing[key]
                        else "exact_decimal_agreement"
                        if difference == 0
                        else "numeric_difference_retained"
                    )
                    self.reconcile(
                        peptide["source_record_id"],
                        peptide["peptide_id"],
                        self.acquisitions[sample]["acquisition_id"],
                        kind + "_intensity_sum",
                        status,
                        "Intensity " + sample,
                        {
                            "source_value": source["Intensity " + sample],
                            "contributor_sum": str(sums[key]),
                            "residual": None if difference is None else str(difference),
                            "n_missing_contributor_values": missing[key],
                            "rounding_explanation_proven": False,
                        },
                    )

    def origins(self):
        for peptide, reference, start, end, kind in candidate_spans(
            self.references, self.peptides.values()
        ):
            self.add(
                "origins",
                {
                    "origin_id": self.key(
                        "origin",
                        peptide["peptide_id"],
                        reference["reference_record_id"],
                        start,
                        end,
                    ),
                    "source_record_id": peptide["source_record_id"],
                    "peptide_id": peptide["peptide_id"],
                    "reference_record_id": reference["reference_record_id"],
                    "reference_sequence_sha256": reference["reference_sequence_sha256"],
                    "candidate_start_1based": start,
                    "candidate_end_1based": end,
                    "origin_match_type": kind,
                    "reference_role": reference["reference_role"],
                },
            )
            self.dependencies.link(
                peptide["peptide_id"], "sequence:" + reference["reference_sequence_sha256"]
            )
        for peptide, group in sorted(
            self.dependencies.groups([p["peptide_id"] for p in self.peptides.values()]).items()
        ):
            self.add(
                "dependency_groups",
                {
                    "peptide_id": peptide,
                    "dependency_group_id": group,
                    "dependency_policy": "all_reference_sequence_and_all_reported_parent_alternatives_before_eligibility",
                },
            )

    def run(self):
        try:
            for asset in self.contract["assets"]:
                self.add(
                    "assets",
                    {
                        "source_asset_id": self.key("asset", asset["filename"]),
                        "source_file": asset["filename"],
                        "source_asset_url": asset["source_url"],
                        "source_asset_role": asset["role"],
                        "asset_sha256": asset["sha256"],
                        "size_bytes": asset["size_bytes"],
                        "license_status": asset["license_status"],
                        "source_member_provenance_json": _json(asset.get("member_provenance")),
                    },
                )
            self.metadata()
            self.reference()
            self.search_tables()
            self.identifier_reconciliation()
            self.publisher()
            self.aggregation()
            self.origins()
            validate_material_graph(self.contexts, self.material_links)
            for table, rows in (
                ("material_contexts", self.contexts),
                ("material_links", self.material_links),
            ):
                for row in rows:
                    self.add(table, row)
            return {"n_" + name + "_rows": sink.n_rows for name, sink in self.sinks.items()}
        finally:
            for sink in self.sinks.values():
                sink.close()


def compile_native_source(assets, destination, contract):
    return Compiler(assets, destination, contract).run()
