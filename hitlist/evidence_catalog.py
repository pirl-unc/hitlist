"""Versioned semantic catalog for measured evidence and linked context.

This catalog describes *logical* v2 fields and the native v1 materializations
that will be migrated. It neither reads assay rows nor certifies training
eligibility for an individual row. Later migrations map native columns to these
roles and implement task-specific eligibility decisions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

SCHEMA_VERSION = 1
FIELD_ROLES = frozenset(
    {
        "identity",
        "molecular_input",
        "biological_context",
        "target",
        "protocol",
        "provenance",
        "lineage",
        "relation",
    }
)
VALUE_KINDS = frozenset({"identifier", "text", "number", "boolean", "category", "set", "sequence"})
EVIDENCE_KINDS = frozenset({"observation", "projection", "relation", "context"})

NULL_STATES = {
    "unreported": "The source did not establish a value; absence is not implied.",
    "explicit_absence": "The source explicitly reports that the factor was absent.",
    "not_applicable": "The field has no meaning for this assay or material.",
    "unresolved": "Available claims do not resolve to one value.",
    "conflicting": "Source-backed claims disagree and both remain preserved.",
}


@dataclass(frozen=True)
class Field:
    name: str
    role: str
    value_kind: str
    meaning: str
    null_states: tuple[str, ...] = ("unreported",)


@dataclass(frozen=True)
class Evidence:
    name: str
    kind: str
    family: str
    grain: str
    native_artifact: str
    source_population: str
    endpoints: tuple[str, ...]
    fields: tuple[str, ...]
    index_required: tuple[str, ...]
    native_selector: str = ""
    source_view: str = ""
    assay_families: tuple[str, ...] = ()
    status: str = "v2_contract_only"


@dataclass(frozen=True)
class Task:
    name: str
    evidence: tuple[str, ...]
    required: tuple[str, ...]
    label_rule: str
    status: str = "v2_contract_only"


@dataclass(frozen=True)
class Baseline:
    population: str
    generation: str
    n_rows: int
    count_unit: str
    note: str


def _field(name, role, value_kind, meaning, *null_states):
    return Field(name, role, value_kind, meaning, null_states or ("unreported",))


FIELDS = (
    _field("observation_id", "identity", "identifier", "One typed result at its declared grain"),
    _field("source_record_id", "provenance", "identifier", "Original source row locator"),
    _field("source_assay_id", "provenance", "identifier", "Reported source assay identifier"),
    _field("source_reference_id", "provenance", "identifier", "Source publication or deposit"),
    _field("source_dataset", "provenance", "identifier", "Pinned source release"),
    _field("assay_family", "protocol", "category", "Method/readout family, not a training label"),
    _field("endpoint", "target", "category", "Measured endpoint with task-specific meaning"),
    _field("qualitative_result", "target", "category", "Reported outcome, with original tier"),
    _field(
        "outcome_status",
        "target",
        "category",
        "Admission status distinct from a possibly blank reported outcome",
    ),
    _field("quantitative_raw", "target", "text", "Unmodified numeric source cell"),
    _field("quantitative_value", "target", "number", "Parsed numeric value"),
    _field("measurement_unit", "target", "category", "Reported unit, before any conversion"),
    _field("measurement_inequality", "target", "category", "Reported censoring or comparison"),
    _field("peptide_sequence", "molecular_input", "sequence", "Directly reported peptide sequence"),
    _field("protein_id", "molecular_input", "identifier", "Source protein identity"),
    _field("gene_id", "molecular_input", "identifier", "Source gene identity"),
    _field("transcript_id", "molecular_input", "identifier", "Source transcript identity"),
    _field("tested_mhc", "molecular_input", "identifier", "MHC molecule tested in the assay"),
    _field(
        "candidate_mhc_set",
        "biological_context",
        "set",
        "Reported or curated presenting-sample candidates, not peptide restriction",
        "unreported",
        "unresolved",
    ),
    _field(
        "peptide_mhc_restriction",
        "biological_context",
        "identifier",
        "Peptide-level restriction claim at its evidenced precision",
        "unreported",
        "unresolved",
        "conflicting",
    ),
    _field(
        "cellular_mhc_genotype",
        "biological_context",
        "set",
        "Independent cellular typing; incomplete loci remain unknown",
        "unreported",
        "unresolved",
    ),
    _field("restriction_evidence", "provenance", "category", "Basis for a restriction claim"),
    _field("material_id", "biological_context", "identifier", "Sample or experimental material"),
    _field("cell_type", "biological_context", "category", "Reported or curated cell identity"),
    _field("condition_id", "biological_context", "identifier", "Experimental arm"),
    _field(
        "gene_perturbation",
        "biological_context",
        "set",
        "Concurrent gene edits or expression interventions",
        "unreported",
        "explicit_absence",
        "not_applicable",
        "unresolved",
    ),
    _field(
        "cytokine_exposure",
        "biological_context",
        "set",
        "Cytokines applied to biological material, not a release assay outcome",
        "unreported",
        "explicit_absence",
        "not_applicable",
        "unresolved",
    ),
    _field("exposure_dose", "biological_context", "text", "Reported dose with unit"),
    _field("exposure_duration", "biological_context", "text", "Reported duration with unit"),
    _field("assay_method", "protocol", "category", "Reported assay method"),
    _field("assay_readout", "protocol", "category", "Reported response measured"),
    _field("capture_method", "protocol", "category", "Ligand or protein capture"),
    _field("digestion_enzyme", "protocol", "category", "Protein digestion enzyme"),
    _field("instrument", "protocol", "text", "MS instrument identity"),
    _field("acquisition_mode", "protocol", "category", "MS acquisition mode"),
    _field("fractionation", "protocol", "text", "Fractionation protocol"),
    _field("normalization", "protocol", "category", "Quantification scale or normalization"),
    _field(
        "donor_id",
        "lineage",
        "identifier",
        "Source-backed physical donor identity",
        "unreported",
        "unresolved",
        "conflicting",
    ),
    _field(
        "specimen_id",
        "lineage",
        "identifier",
        "Source-backed physical specimen identity",
        "unreported",
        "unresolved",
        "conflicting",
    ),
    _field("acquisition_id", "lineage", "identifier", "Source-backed acquisition identity"),
    _field("lineage_status", "lineage", "category", "Resolution per lineage axis"),
    _field("mapping_position", "relation", "number", "Peptide position in one source protein"),
    _field("n_flank", "relation", "sequence", "N-terminal sequence of one mapping"),
    _field("c_flank", "relation", "sequence", "C-terminal sequence of one mapping"),
    _field("proteome", "relation", "identifier", "Reference in which mapping was found"),
    _field("contributor_status", "relation", "category", "Source-to-observation relationship"),
)
_FIELD_BY_NAME = {field.name: field for field in FIELDS}

_SOURCE = ("observation_id", "source_record_id", "source_dataset", "source_reference_id")
_ASSAY = ("assay_family", "assay_method", "assay_readout", "endpoint")
_MEASUREMENT = (
    "qualitative_result",
    "quantitative_raw",
    "quantitative_value",
    "measurement_unit",
    "measurement_inequality",
)
_MHC = (
    "tested_mhc",
    "candidate_mhc_set",
    "peptide_mhc_restriction",
    "cellular_mhc_genotype",
    "restriction_evidence",
)
_MATERIAL = ("material_id", "cell_type", "condition_id", "gene_perturbation", "cytokine_exposure")
_MS_PROTOCOL = (
    "capture_method",
    "digestion_enzyme",
    "instrument",
    "acquisition_mode",
    "fractionation",
)
_TCELL_SOURCE = ("source_assay_id", "source_reference_id", "source_dataset")
_TCELL_CONTEXT = (
    "cell_type",
    "candidate_mhc_set",
    "peptide_mhc_restriction",
    "restriction_evidence",
)

EVIDENCE = (
    Evidence(
        "mhc_positive_ms",
        "observation",
        "mhc_ligand_ms",
        "assay_result",
        "observations.parquet",
        "mhc_v4",
        ("positive_mhc_ligand_ms",),
        (
            *_SOURCE,
            *_ASSAY,
            *_MEASUREMENT,
            "outcome_status",
            "peptide_sequence",
            *_MHC,
            *_MATERIAL,
            *_MS_PROTOCOL,
        ),
        ("observation_id", "source_dataset", "peptide_sequence", "endpoint"),
    ),
    Evidence(
        "mhc_binding",
        "observation",
        "mhc_binding",
        "assay_result",
        "binding.parquet",
        "mhc_v4",
        ("qualitative_binding", "affinity", "kinetics", "stability"),
        _SOURCE + _ASSAY + _MEASUREMENT + ("peptide_sequence",) + _MHC,
        ("observation_id", "source_dataset", "endpoint"),
    ),
    Evidence(
        "mhc_other",
        "observation",
        "mhc_other",
        "assay_result",
        "other_assays.parquet",
        "mhc_v4",
        ("negative_or_unestablished_ms", "structural", "non_ms_ligand", "other"),
        _SOURCE + _ASSAY + _MEASUREMENT + ("outcome_status", "peptide_sequence") + _MHC,
        ("observation_id", "source_dataset", "endpoint"),
        assay_families=("ms", "structural", "non_ms_ligand", "unresolved"),
    ),
    Evidence(
        "tcell_availability",
        "observation",
        "tcell",
        "source_assay_id",
        "tcell_assay_metadata/assays.parquet",
        "tcell_2026_10_20",
        ("qualitative_assay_outcome", "quantitative_reported_readout"),
        _TCELL_SOURCE + _ASSAY + _MEASUREMENT + _TCELL_CONTEXT,
        ("source_assay_id", "source_dataset", "assay_family"),
        assay_families=(
            "soluble_factor_release",
            "activation_or_proliferation",
            "cytotoxicity",
            "pmhc_multimer",
            "other_tcell_assay",
        ),
    ),
    Evidence(
        "tcell_functional_peptide",
        "projection",
        "tcell",
        "source_assay_id",
        "human_tcell_training/training.parquet",
        "tcell_2026_10_20",
        ("cellular_functional_response",),
        _TCELL_SOURCE + _ASSAY + _MEASUREMENT + _TCELL_CONTEXT + ("peptide_sequence",),
        ("source_assay_id", "peptide_sequence", "endpoint"),
        source_view="tcell_availability",
        assay_families=(
            "soluble_factor_release",
            "activation_or_proliferation",
            "cytotoxicity",
        ),
    ),
    Evidence(
        "tcell_multimer_peptide",
        "projection",
        "tcell",
        "source_assay_id",
        "human_tcell_multimer_training/training.parquet",
        "tcell_2026_10_20",
        ("pmhc_multimer_cell_binding",),
        _TCELL_SOURCE + _ASSAY + _MEASUREMENT + _TCELL_CONTEXT + ("peptide_sequence",),
        ("source_assay_id", "peptide_sequence", "endpoint"),
        source_view="tcell_availability",
        assay_families=("pmhc_multimer",),
    ),
    Evidence(
        "bulk_ms_peptide",
        "observation",
        "bulk_proteomics",
        "peptide_material_protocol",
        "bulk_proteomics.parquet",
        "bulk_v4",
        ("bulk_peptide_detection", "peptide_intensity"),
        (
            *_SOURCE,
            "peptide_sequence",
            "protein_id",
            "material_id",
            "endpoint",
            *_MEASUREMENT,
            *_MS_PROTOCOL,
        ),
        ("source_dataset", "peptide_sequence", "material_id"),
        native_selector="granularity=peptide",
    ),
    Evidence(
        "bulk_ms_protein",
        "observation",
        "bulk_proteomics",
        "protein_material_protocol",
        "bulk_proteomics.parquet",
        "bulk_v4",
        ("protein_abundance",),
        (
            *_SOURCE,
            "protein_id",
            "gene_id",
            "material_id",
            "endpoint",
            *_MEASUREMENT,
            *_MS_PROTOCOL,
        ),
        ("source_dataset", "protein_id", "material_id"),
        native_selector="granularity=protein",
    ),
    Evidence(
        "line_expression_gene",
        "observation",
        "line_expression",
        "gene_profile",
        "line_expression.parquet",
        "line_expression_v4",
        ("gene_expression",),
        (
            *_SOURCE,
            "gene_id",
            "material_id",
            "quantitative_value",
            "measurement_unit",
            "normalization",
        ),
        ("source_dataset", "gene_id", "material_id"),
        native_selector="granularity=gene",
    ),
    Evidence(
        "line_expression_transcript",
        "observation",
        "line_expression",
        "transcript_profile",
        "line_expression.parquet",
        "line_expression_v4",
        ("transcript_expression",),
        (
            *_SOURCE,
            "gene_id",
            "transcript_id",
            "material_id",
            "quantitative_value",
            "measurement_unit",
            "normalization",
        ),
        ("source_dataset", "transcript_id", "material_id"),
        native_selector="granularity=transcript",
    ),
    Evidence(
        "peptide_protein_occurrence",
        "relation",
        "mapping",
        "peptide_protein_position",
        "peptide_mappings.parquet",
        "mhc_v4",
        (),
        (
            "peptide_sequence",
            "protein_id",
            "gene_id",
            "transcript_id",
            "mapping_position",
            "n_flank",
            "c_flank",
            "proteome",
        ),
        ("peptide_sequence", "protein_id", "mapping_position", "proteome"),
    ),
    Evidence(
        "source_contributor",
        "relation",
        "provenance",
        "source_observation_link",
        "observation_contributors.parquet",
        "mhc_v4",
        (),
        ("observation_id", "source_record_id", "source_dataset", "contributor_status"),
        ("observation_id", "source_record_id"),
    ),
    Evidence(
        "curated_material_arm",
        "context",
        "material",
        "study_arm",
        "hitlist/data/pmid_overrides.yaml",
        "curation_v1",
        (),
        ("source_reference_id", *_MATERIAL, *_MHC, *_MS_PROTOCOL),
        ("source_reference_id", "material_id"),
    ),
    Evidence(
        "reviewed_lineage",
        "relation",
        "lineage",
        "identity_context",
        "hitlist/data/specimen_lineage.yaml",
        "curation_v1",
        (),
        ("condition_id", "donor_id", "specimen_id", "acquisition_id", "lineage_status"),
        ("condition_id", "lineage_status"),
    ),
)
_EVIDENCE_BY_NAME = {item.name: item for item in EVIDENCE}

TASKS = (
    Task(
        "mhc_positive_presentation",
        ("mhc_positive_ms",),
        ("peptide_sequence", "endpoint", "outcome_status"),
        "Positive MS admission or reviewed observed-MS supplement only; other MS outcomes excluded.",
    ),
    Task(
        "mhc_quantitative_binding",
        ("mhc_binding",),
        (
            "peptide_sequence",
            "tested_mhc",
            "endpoint",
            "quantitative_value",
            "measurement_unit",
        ),
        "Endpoint, unit and inequality determine the target; IC50, KD and half-life stay distinct.",
    ),
    Task(
        "tcell_functional_response",
        ("tcell_functional_peptide",),
        (
            "source_assay_id",
            "peptide_sequence",
            "assay_method",
            "assay_readout",
            "qualitative_result",
        ),
        "Directly tested single-peptide cellular functional outcome; multimer binding excluded.",
    ),
    Task(
        "tcell_multimer_cell_binding",
        ("tcell_multimer_peptide",),
        (
            "source_assay_id",
            "peptide_sequence",
            "assay_method",
            "assay_readout",
            "qualitative_result",
        ),
        "Source multimer cell-binding outcome; not functional activation or MHC affinity.",
    ),
    Task(
        "bulk_peptide_detection",
        ("bulk_ms_peptide",),
        ("peptide_sequence", "material_id", "digestion_enzyme", "endpoint"),
        "Observed bulk MS detection under the recorded preparation; absent peptides are not measured negatives.",
    ),
    Task(
        "gene_expression",
        ("line_expression_gene",),
        ("gene_id", "material_id", "quantitative_value", "normalization"),
        "Measured expression on its reported scale; surrogate material links require separate policy.",
    ),
)
_TASK_BY_NAME = {task.name: task for task in TASKS}

BASELINES = (
    Baseline(
        "mhc_positive_ms",
        "corrected-corpus-v4:2026-10-16",
        4_486_991,
        "assay_rows",
        "Positive ligand MS",
    ),
    Baseline(
        "mhc_binding",
        "corrected-corpus-v4:2026-10-16",
        814_268,
        "assay_rows",
        "Binding, affinity and stability",
    ),
    Baseline(
        "mhc_other",
        "corrected-corpus-v4:2026-10-16",
        26_537,
        "assay_rows",
        "Other and unestablished MHC assays",
    ),
    Baseline(
        "tcell_2026_10_20",
        "tcell-availability:2026-10-20",
        578_529,
        "distinct_iedb_assay_ids",
        "CEDAR's 157,263 IDs all overlap",
    ),
    Baseline(
        "tcell_functional_peptide",
        "tcell-training:2026-10-21",
        47_421,
        "training_examples",
        "Bounded direct-peptide functional subset",
    ),
    Baseline(
        "tcell_multimer_peptide",
        "tcell-training:2026-10-21",
        8_423,
        "training_examples",
        "Bounded direct-peptide multimer subset",
    ),
    Baseline(
        "bulk_v4",
        "corrected-corpus-v4:2026-10-16",
        2_269_288,
        "measurement_rows",
        "Protein and peptide rows together",
    ),
    Baseline(
        "line_expression_v4",
        "corrected-corpus-v4:2026-10-16",
        42_909,
        "measurement_rows",
        "Gene and transcript rows together",
    ),
)


def _validate_catalog() -> None:
    if len(_FIELD_BY_NAME) != len(FIELDS) or len(_EVIDENCE_BY_NAME) != len(EVIDENCE):
        raise ValueError("Duplicate evidence catalog field or evidence name")
    if len(_TASK_BY_NAME) != len(TASKS):
        raise ValueError("Duplicate evidence catalog task name")
    for field in FIELDS:
        if field.role not in FIELD_ROLES or field.value_kind not in VALUE_KINDS:
            raise ValueError(f"Invalid field role or value kind: {field.name}")
        if any(state not in NULL_STATES for state in field.null_states):
            raise ValueError(f"Unknown null state on {field.name}")
    for item in EVIDENCE:
        if item.kind not in EVIDENCE_KINDS:
            raise ValueError(f"Unknown evidence kind: {item.kind}")
        if len(set(item.fields)) != len(item.fields):
            raise ValueError(f"Duplicate field on {item.name}")
        if len(set(item.assay_families)) != len(item.assay_families):
            raise ValueError(f"Duplicate assay family on {item.name}")
        if set(item.fields) - _FIELD_BY_NAME.keys() or set(item.index_required) - set(item.fields):
            raise ValueError(f"Invalid fields on {item.name}")
        if item.source_view and (
            item.source_view not in _EVIDENCE_BY_NAME
            or _EVIDENCE_BY_NAME[item.source_view].source_population != item.source_population
        ):
            raise ValueError(f"Invalid source view on {item.name}")
    for task in TASKS:
        if len(set(task.required)) != len(task.required):
            raise ValueError(f"Duplicate task requirement: {task.name}")
        if any(key not in _EVIDENCE_BY_NAME for key in task.evidence):
            raise ValueError(f"Unknown evidence on task {task.name}")
        for key in task.evidence:
            if set(task.required) - set(_EVIDENCE_BY_NAME[key].fields):
                raise ValueError(f"Task fields absent from {key}: {task.name}")
    populations = set(_EVIDENCE_BY_NAME) | {item.source_population for item in EVIDENCE}
    for baseline in BASELINES:
        if baseline.population not in populations or baseline.n_rows < 0:
            raise ValueError(f"Invalid baseline population or count: {baseline.population}")


_validate_catalog()


def list_evidence(*, kind: str | None = None) -> tuple[str, ...]:
    """List logical evidence types without reading any assay or source file."""
    if kind is not None and kind not in EVIDENCE_KINDS:
        raise ValueError(f"Unknown evidence kind: {kind}")
    return tuple(item.name for item in EVIDENCE if kind is None or item.kind == kind)


def list_tasks() -> tuple[str, ...]:
    """List declared task contracts; each remains a v2 contract until migrated."""
    return tuple(task.name for task in TASKS)


def describe_evidence(name: str, *, fields: tuple[str, ...] | None = None) -> dict:
    """Describe one evidence type, rejecting unknown logical field requests."""
    try:
        item = _EVIDENCE_BY_NAME[name]
    except KeyError as error:
        raise KeyError(f"Unknown evidence type: {name}") from error
    selected = item.fields if fields is None else tuple(fields)
    if len(selected) != len(set(selected)):
        raise ValueError(f"Duplicate requested fields on {name}")
    unknown = sorted(set(selected) - set(item.fields))
    if unknown:
        raise ValueError(f"Fields unavailable on {name}: {unknown}")
    result = asdict(item)
    result["fields"] = [asdict(_FIELD_BY_NAME[field]) for field in selected]
    return result


def describe_task(name: str) -> dict:
    """Return task requirements without claiming any row currently satisfies them."""
    try:
        return asdict(_TASK_BY_NAME[name])
    except KeyError as error:
        raise KeyError(f"Unknown task: {name}") from error


def describe_catalog() -> dict:
    """Return a deterministic JSON-serializable declaration of the current catalog."""
    return {
        "schema_version": SCHEMA_VERSION,
        "null_states": dict(NULL_STATES),
        "evidence": [describe_evidence(item.name) for item in EVIDENCE],
        "tasks": [describe_task(task.name) for task in TASKS],
        "baselines": [asdict(baseline) for baseline in BASELINES],
    }
