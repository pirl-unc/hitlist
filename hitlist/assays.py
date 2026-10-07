"""Assay modality is positive evidence, not the complement of binding (#644)."""

import re
from functools import lru_cache

import pandas as pd

ASSAY_COLUMNS = (
    "qualitative_measurement",
    "assay_comments",
    "assay_method",
    "response_measured",
    "source",
)
ANNOTATION_COLUMNS = (
    "assay_modality",
    "assay_modality_source",
    "is_binding_assay",
    "is_ms_observation",
)
_MS_METHOD = r"mass[ -]+spectrometry"
_LIGAND_RESPONSE = r"^(?:ligand presentation|mhc ligand elution|elution)?$"
_STRUCTURE = r"3d structure|crystallograph|electron microscopy|nuclear magnetic resonance"
_BINDING_RESPONSE = (
    r"binding|inhibitory concentration|effective concentration|dissociation constant|"
    r"association constant|half[ -]?life|dissociation temperature|melting temperature|"
    r"^(?:on|off|association|dissociation) rate$"
)
_BINDING_METHOD = r"binding assay|mhc/(?:direct|competitive)|refolding|microarray|phage display"
_NON_MS_LIGAND_METHOD = r"edman|coelution|t cell recognition"


def _text(value):
    return str(value).strip().lower() if isinstance(value, str) else ""


def _matches(value, pattern, *, negate=False):
    """The same predicates operate on scanner strings and Arrow expressions."""
    if isinstance(value, str):
        result = re.search(pattern, value) is not None
        return not result if negate else result
    import pyarrow.compute as pc

    result = pc.match_substring_regex(value, pattern)
    return ~result if negate else result


def _structured_rules(method, response, source):
    ms = _matches(method, _MS_METHOD)
    compatible = _matches(response, _LIGAND_RESPONSE)
    return (
        (ms & _matches(response, _LIGAND_RESPONSE, negate=True), "unknown", "conflicting_fields"),
        (
            _matches(method, _STRUCTURE) | _matches(response, _STRUCTURE),
            "structural",
            "structured_method_or_response",
        ),
        (ms & compatible, "ms", "assay_method"),
        (
            _matches(method, _NON_MS_LIGAND_METHOD) & compatible,
            "non_ms_ligand",
            "assay_method",
        ),
        (
            _matches(response, _BINDING_RESPONSE) | _matches(method, _BINDING_METHOD),
            "binding",
            "structured_method_or_response",
        ),
        (
            _matches(method, r"^$") & compatible & _matches(source, r"^supplement$"),
            "ms",
            "curated_ms_supplement",
        ),
    )


@lru_cache(maxsize=16384)
def _classify(outcome, comments, method, response, source):
    for matched, modality, basis in _structured_rules(method, response, source):
        if matched:
            return modality, basis
    # Older deposits can identify a binding format only in comments/tier.
    # This fallback never supplies positive evidence of MS presentation.
    from .curation import _legacy_is_binding_assay

    if _legacy_is_binding_assay(outcome.title(), comments):
        return "binding", "legacy_binding_heuristic"
    return ("other", "unrecognized_fields") if method or response else ("unknown", "missing_fields")


def assay_annotations(
    qualitative_measurement="", assay_comments="", assay_method="", response_measured="", source=""
):
    """Preserve modality independently of result tier, polarity and HLA evidence."""
    outcome, comments, method, response, source = map(
        _text, (qualitative_measurement, assay_comments, assay_method, response_measured, source)
    )
    modality, basis = _classify(outcome, comments, method, response, source)
    return {
        "assay_modality": modality,
        "assay_modality_source": basis,
        "is_binding_assay": modality == "binding",
        "is_ms_observation": modality == "ms" and not outcome.startswith("negative"),
    }


def annotate_assays(frame):
    """Refresh assay annotations in place; keep every original evidence field."""
    values = [
        frame.get(column, pd.Series("", index=frame.index)).astype("string").fillna("")
        for column in ASSAY_COLUMNS
    ]
    # Cache the small modality vocabulary, never an unbounded corpus of comments.
    modalities, bases, binding, ms = [], [], [], []
    for row in zip(*values):
        record = assay_annotations(*row)
        modalities.append(record["assay_modality"])
        bases.append(record["assay_modality_source"])
        binding.append(record["is_binding_assay"])
        ms.append(record["is_ms_observation"])
    frame["assay_modality"] = pd.Categorical(modalities)
    frame["assay_modality_source"] = pd.Categorical(bases)
    frame["is_binding_assay"] = pd.Series(binding, index=frame.index, dtype=bool)
    frame["is_ms_observation"] = pd.Series(ms, index=frame.index, dtype=bool)
    return frame


def positive_ms_mask(frame):
    """Recompute admission from source evidence, including on legacy exports."""
    inputs = frame.reindex(columns=ASSAY_COLUMNS)
    return annotate_assays(inputs)["is_ms_observation"]


def assay_expression(columns, kind):
    """Filter before pandas materialization, even on legacy projected reads."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.dataset as ds

    def field(name):
        value = ds.field(name).cast(pa.string()) if name in columns else ds.scalar("")
        return pc.utf8_lower(pc.utf8_trim_whitespace(pc.coalesce(value, ds.scalar(""))))

    rules = _structured_rules(field("assay_method"), field("response_measured"), field("source"))
    accepted = (
        pc.coalesce(ds.field("is_binding_assay"), ds.scalar(False))
        if kind == "binding" and "is_binding_assay" in columns
        else ds.scalar(False)
    )
    for condition, modality, _ in reversed(rules):
        accepted = pc.if_else(condition, ds.scalar(modality == kind), accepted)
    if kind == "ms":
        accepted = accepted & _matches(field("qualitative_measurement"), r"^negative", negate=True)
    return accepted


def positive_ms_expression(columns):
    return assay_expression(columns, "ms")
