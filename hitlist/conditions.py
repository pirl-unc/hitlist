"""Flat experimental-condition columns for curated ``ms_samples`` arms.

A consumer selecting "every ERAP2 knockout arm" or "every vehicle control"
had to parse prose: the only machine-readable condition fields were
``perturbation`` (the curated string with an ``unperturbed`` prefix removed)
and ``condition_category`` (one coarse bucket per arm).  Both are lossy in
ways that matter for training data:

* A coarse bucket cannot separate two arms inside it.  ``IFN-gamma 100
  IU/mL 24h`` and ``IFN-gamma 100 ng/ml 72h`` are one ``IFN_gamma_treatment``
  value, so nothing downstream can tell them apart or refuse to.
* One bucket per arm cannot hold two simultaneous factors.  ``TAP1 knockout
  + Mycobacterium tuberculosis H37Rv infection`` categorizes as
  ``TAP_perturbation``; the infection is not anywhere in the export.
* ``simplify_condition`` blanks everything after ``unperturbed — ``, which
  is right for a culture-medium annotation and wrong for the HLA-DM
  co-transfection that 42 arms carry and 4 arms explicitly lack.  Both
  report ``condition_category == "unperturbed"``.

This module declares the columns that fix that, as one ordered registry
used for validation, the empty-frame schemas, export propagation, the
training-table defaults, and the documentation.  The columns are authored
directly on ``ms_samples`` entries — flat, scalar, and named exactly as
they are exported, so curation and consumption share one vocabulary.

Nothing here re-derives a value from prose.  :func:`condition_columns_for_sample`
reads explicit curation and returns blanks when there is none; the legacy
``perturbation`` / ``condition_category`` / ``apm_*`` classifiers keep their
documented meanings alongside (#450).

Missing-value contract
----------------------
``""`` means *not established* — the source does not say.  It never means
untreated, wild type, or known absence.  ``none`` is a positive claim of
absence and is only permitted where a field's own vocabulary declares it.
``unspecified`` means the intervention happened and its target is unnamed.
Those three are different facts and a consumer that conflates them turns a
silence into a control arm, which is the one direction that cancels the
perturbed-vs-control contrast (the same failure #392 fixed for
``apm_perturbed``).
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from functools import lru_cache
from os.path import dirname, join
from types import MappingProxyType

from .curation_yaml import load_curation_yaml

#: Completeness of *this record's* categorical annotation.
#:
#: It describes the annotation, not confidence that a peptide belongs to the
#: arm — that is ``sample_attribution``'s job, and reporting one as evidence
#: for the other is the mistake the two vocabularies exist to prevent.
CONDITION_STATUS_VALUES = (
    # Every fact the reviewed text states is represented in some column.
    # Not a claim that the paper reported every experimental variable.
    "annotated",
    # The text states a fact no column captures at its stated precision.
    "partial",
    # The record combines alternative conditions that the source does not
    # separate.  Only facts true of every contributing condition are filled.
    "mixed",
    # Nothing about the experimental condition has been annotated yet.
    "unreported",
)

#: Where the annotation came from.  ``curated_text`` is the existing curated
#: ``condition`` wording normalized into columns — it inherits whatever the
#: original curation got right.  ``primary_source`` means someone read the
#: paper, and then ``condition_reference`` must name where.
CONDITION_EVIDENCE_VALUES = ("curated_text", "primary_source")

#: Explicit experimental-control role.  Distinct from the legacy
#: ``is_control_arm``, which is ``condition_category == "unperturbed"`` —
#: a derived baseline flag, not a curated statement that this arm is the
#: comparator for another.
CONDITION_CONTROL_VALUES = (
    "untreated",
    "vehicle",
    "mock",
    "empty_vector",
    "non_targeting",
    "isotype",
    "positive",
    "other",
)

#: How the documented interventions relate to each other.
CONDITION_COMBINATION_VALUES = (
    # Exactly one intervention.
    "single",
    # Several applied together.
    "simultaneous",
    # Several applied in a stated order.  The order lives in the original
    # `condition` wording: the token lists here are sorted, so reading order
    # out of them would invent it.
    "sequential",
    # Several interventions, relationship not stated.
    "unspecified",
)

#: Coarse antigen-exposure context.
CONDITION_ANTIGEN_EXPOSURE_VALUES = (
    "peptide_pulse",
    "cross_presentation",
    "apoptotic_cell_feeding",
)

#: How the MHC being sampled is expressed or captured.  These are properties
#: of the measurement system, not treatments.
CONDITION_MHC_CONTEXT_VALUES = (
    # One allele expressed in an HLA-null / HLA-low host.
    "monoallelic",
    # MHC introduced into a host line (mono-allelic or not).
    "mhc_transfectant",
    # Soluble MHC recovered from supernatant, plasma, serum or a biofluid.
    "soluble_mhc",
    # MHC refolded in vitro around a peptide pool.
    "refolded_mhc",
    # An additional MHC allele co-expressed alongside the endogenous ones.
    "mhc_coexpression",
)

#: Reported physical state of the material.
CONDITION_MATERIAL_VALUES = (
    "direct_ex_vivo",
    "cultured",
    "fresh",
    "frozen",
    "biofluid",
    "in_vivo",
)

#: Columns whose cells hold sorted, unique, ``;``-separated tokens.
#:
#: Several tokens mean *all of them apply to this row* — never "one of
#: these".  A union of alternative conditions in one cell would read as a
#: combination treatment nobody performed, which is why ``mixed`` records
#: keep only what every contributing condition shares.
MULTI_VALUE_CONDITION_COLUMNS = frozenset(
    {
        "condition_control_for",
        "condition_knockout_genes",
        "condition_knockdown_genes",
        "condition_overexpression_genes",
        "condition_genetic_variants",
        "condition_transfection",
        "condition_transduction",
        "condition_cytokines",
        "condition_drugs",
        "condition_infection",
        "condition_stimulation",
        "condition_antigen_exposure",
        "condition_background",
        "condition_mhc_context",
        "condition_material",
        "condition_labeling",
    }
)

#: Columns restricted to a declared vocabulary.  Everything else is open:
#: gene symbols, compound names and organism designations come from the
#: sources, and an unknown reported entity must stay representable rather
#: than being dropped for missing a list.
CLOSED_CONDITION_VOCABULARIES = MappingProxyType(
    {
        "condition_status": CONDITION_STATUS_VALUES,
        "condition_evidence": CONDITION_EVIDENCE_VALUES,
        "condition_control": CONDITION_CONTROL_VALUES,
        "condition_combination": CONDITION_COMBINATION_VALUES,
        "condition_antigen_exposure": CONDITION_ANTIGEN_EXPOSURE_VALUES,
        "condition_mhc_context": CONDITION_MHC_CONTEXT_VALUES,
        "condition_material": CONDITION_MATERIAL_VALUES,
    }
)

#: The columns that name an intervention someone performed.
#:
#: ``condition_combination`` describes the relationship *between* these, so it
#: is only meaningful when at least one is filled. Without that rule a record
#: can say "several interventions, relationship not stated" while naming none,
#: which a featurizer reads as a multi-agent arm with unknown agents rather
#: than as an unresolvable record.
INTERVENTION_CONDITION_COLUMNS = frozenset(
    {
        "condition_knockout_genes",
        "condition_knockdown_genes",
        "condition_overexpression_genes",
        "condition_genetic_variants",
        "condition_transfection",
        "condition_transduction",
        "condition_cytokines",
        "condition_drugs",
        "condition_infection",
        "condition_stimulation",
        "condition_antigen_exposure",
    }
)

#: Columns that describe *one arm's own record* rather than a fact about the
#: material, and so cannot survive onto a row that reached no arm.
#:
#: The rest of the block can. If every candidate arm was cultured in RPMI-1640
#: then so was whichever one this peptide came from, and withholding that would
#: lose a fact the evidence does support. But ``condition_status: annotated``
#: says *this record's* annotation is complete, and an unattributed row has no
#: record — so keeping it while the consensus blanks the agent columns exports
#: "fully annotated, no knockout", which is the ""-means-absence conflation
#: this module exists to prevent. Same shape as the ``effective_override_origin``
#: rule in ``_consensus_meta``: a statement about a specific arm must not
#: outlive the arm (#373).
ARM_SPECIFIC_CONDITION_COLUMNS = frozenset(
    {
        "condition_id",
        "condition_status",
        "condition_evidence",
        "condition_reference",
        "condition_control_for",
    }
)

#: Columns where ``none`` is a permitted value: an explicit, positive claim
#: that this intervention was *not* applied.
#:
#: Only the intervention columns. Absence is a real experimental fact there —
#: the wild-type arm of a knockout study, or the four arms curated as "no
#: HLA-DM co-transfection" against the 42 that have it — and it is a
#: different claim from ``""``. Absence of a *context* (no culture medium, no
#: material state) is not something a source asserts, so those columns leave
#: it unsaid rather than offering a value that would read as a finding.
NONE_PERMITTED_CONDITION_COLUMNS = frozenset(
    {
        "condition_knockout_genes",
        "condition_knockdown_genes",
        "condition_overexpression_genes",
        "condition_genetic_variants",
        "condition_transfection",
        "condition_transduction",
        "condition_cytokines",
        "condition_drugs",
        "condition_infection",
        "condition_stimulation",
    }
)

#: Columns holding bare gene symbols.  Validated as *designations* — shape
#: only — never against :data:`hitlist.apm.APM_GENES`.  The APM set is a
#: filtering subset; a knockout of a gene outside it is still a knockout,
#: and rejecting it here would discard a source fact to satisfy a filter.
GENE_CONDITION_COLUMNS = frozenset(
    {
        "condition_knockout_genes",
        "condition_knockdown_genes",
        "condition_overexpression_genes",
    }
)

#: A gene designation: HGNC-style symbols plus the hyphenated and
#: locus-suffixed forms the sources actually use (``HLA-DM``, ``H2-K1``,
#: ``TAX1BP1``).  Deliberately permissive about content and strict about
#: shape, so a typo like ``ERAP1 KO`` fails while ``HLA-DOA`` passes.
_GENE_SYMBOL_RE = re.compile(r"^[A-Z][A-Z0-9]*(-[A-Z0-9]+)*$")

#: An open-vocabulary token: no separator, no leading/trailing space, and
#: no empty content.  Compound names, organism designations and construct
#: names all live here.
_OPEN_TOKEN_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 ._/+()'-]*$")

#: Columns holding free text rather than categorical tokens.  A source
#: locator is prose — "PMC6823859 Methods, 'Cell lines and antibodies'" — and
#: running it through the token shape check would reject every useful one.
FREE_TEXT_CONDITION_COLUMNS = frozenset({"condition_reference"})

#: A ``condition_id``: lowercase, short, and free of the separators the
#: multi-value cells use, so a control reference can never be ambiguous.
_CONDITION_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_]*$")

#: Every condition column, in export order, mapped to what it means.
#:
#: This is the single registry.  ``curation.MS_SAMPLE_FIELDS`` merges it so
#: the loader accepts the keys; ``export`` splices it into the samples row,
#: the empty-frame schema, the observation ``meta_cols`` and the training
#: defaults.  Adding a column here reaches all of them; adding one anywhere
#: else reaches exactly one, which is how ``_SAMPLE_PROVENANCE_COLUMNS``
#: drifted twice.
CONDITION_FIELDS = MappingProxyType(
    {
        "condition_id": (
            "persistent short ID for this curated context, unique within the "
            "study. Assigned once in curation and frozen — never derived at "
            "export time from a label, row position or category, so renaming "
            "a display label cannot silently reassign an observation"
        ),
        "condition_status": (
            "completeness of this record's categorical annotation; one of CONDITION_STATUS_VALUES"
        ),
        "condition_evidence": (
            "whether the annotation came from the curated text or from the "
            "primary source; one of CONDITION_EVIDENCE_VALUES"
        ),
        "condition_reference": (
            "source locator for a primary_source annotation — URL or accession "
            "plus the section, figure, table, sheet or sample identifier that "
            "establishes the facts. Required when condition_evidence is "
            "primary_source"
        ),
        "condition_control": (
            "explicit experimental-control role; one of CONDITION_CONTROL_VALUES. "
            "Distinct from the derived is_control_arm baseline flag"
        ),
        "condition_control_for": (
            "condition_ids in this study that this record is the control for, "
            "curated only where the comparison is documented"
        ),
        "condition_combination": (
            "how the documented interventions relate; one of CONDITION_COMBINATION_VALUES"
        ),
        "condition_knockout_genes": "genes explicitly knocked out or deleted",
        "condition_knockdown_genes": (
            "genes explicitly knocked down (shRNA, siRNA, degron). A knockdown "
            "is not a knockout and must not be relabelled as one"
        ),
        "condition_overexpression_genes": "genes explicitly overexpressed",
        "condition_genetic_variants": (
            "reported variant designations as `GENE variant`, or `GENE "
            "unspecified` when the axis is a variant comparison whose alleles "
            "are not named"
        ),
        "condition_transfection": (
            "introduced gene or construct, or `unspecified` when only transfection is stated"
        ),
        "condition_transduction": (
            "introduced gene or construct delivered by a viral vector, or "
            "`unspecified`. A vector name is not an infection"
        ),
        "condition_cytokines": "documented cytokine exposures, as HGNC symbols",
        "condition_drugs": (
            "documented compounds, as canonical names; a reported drug class "
            "(`EZH2_inhibitor`) where the compound is unnamed"
        ),
        "condition_infection": (
            "organism designations for a documented infection, at the precision the source states"
        ),
        "condition_stimulation": ("other documented stimuli, activation or expansion contexts"),
        "condition_antigen_exposure": (
            "coarse antigen-exposure context; one of CONDITION_ANTIGEN_EXPOSURE_VALUES"
        ),
        "condition_background": (
            "intrinsic genetic or functional background of the material, as "
            "distinct from an intervention introduced by this study"
        ),
        "condition_mhc_context": (
            "how the sampled MHC is expressed or captured; one of CONDITION_MHC_CONTEXT_VALUES"
        ),
        "condition_culture": (
            "reported medium or culture category, including `standard_culture` "
            "when that is all the source states"
        ),
        "condition_material": (
            "reported physical state of the material; one of CONDITION_MATERIAL_VALUES"
        ),
        "condition_labeling": (
            "metabolic labeling used as a *design* factor (SILAC heavy donor "
            "cells versus light recipients). Distinct from the acquisition "
            "`labeling` field, which records the run's quantification scheme"
        ),
    }
)

#: The ordered column registry.  Export order, documentation order, and the
#: order every empty-frame schema uses.
CONDITION_COLUMNS: tuple[str, ...] = tuple(CONDITION_FIELDS)


def _vocabulary_path() -> str:
    return join(dirname(__file__), "data", "condition_vocabulary.yaml")


@lru_cache(maxsize=1)
def load_condition_vocabulary() -> Mapping[str, Mapping[str, str]]:
    """Alias → canonical mappings for the open-vocabulary columns.

    Kept as YAML rather than Python literals: gene, drug and organism names
    are curation data, and ``AGENTS.md`` puts curation data in YAML.  The
    prose classifiers in :mod:`hitlist.apm` and
    :mod:`hitlist.condition_categories` embed their keyword lists in Python
    and are deliberately left alone — they are a separate, legacy contract
    (#450).

    Returns
    -------
    Mapping[str, Mapping[str, str]]
        Column name → ``{alias: canonical}``.  Aliases are matched
        case-insensitively; canonical values are returned verbatim.
    """
    raw = load_curation_yaml(_vocabulary_path()) or {}
    aliases = raw.get("aliases") or {}
    return MappingProxyType(
        {
            column: MappingProxyType(
                {str(k).casefold(): str(v) for k, v in (mapping or {}).items()}
            )
            for column, mapping in aliases.items()
        }
    )


def canonical_condition_token(column: str, token: str) -> str:
    """Map one open-vocabulary token to its canonical spelling.

    Unknown tokens come back unchanged: a reported entity with no alias
    entry is still a fact, and dropping it would lose exactly the source
    detail these columns exist to keep.
    """
    return load_condition_vocabulary().get(column, {}).get(token.casefold(), token)


def split_condition_tokens(value: str | None) -> list[str]:
    """Split a multi-value cell into its tokens.

    The consumer-side half of the ``;`` contract: membership tests and
    multi-hot encodings split, rather than treating ``ERAP1;ERAP2`` as a
    third agent distinct from either.
    """
    # `str(value or "")` rather than `value or ""`: a pandas NaN is truthy, so
    # the bare form returned the float and raised AttributeError on .split —
    # on the one input an exported frame most plausibly hands this.
    if value is None or value != value:  # NaN is the only value unequal to itself
        return []
    return [t for t in str(value).split(";") if t]


def _describe(pmid: object, index: int) -> str:
    return f"PMID {pmid}: ms_samples[{index}]"


def _validate_token(column: str, token: str, where: str) -> None:
    if token != token.strip():
        raise ValueError(
            f"{where} {column}={token!r} has leading or trailing whitespace inside a "
            f"';'-separated cell.  Tokens are compared exactly, so ' ERAP1' and "
            f"'ERAP1' would be two different agents."
        )
    if column in GENE_CONDITION_COLUMNS:
        if not _GENE_SYMBOL_RE.match(token):
            raise ValueError(
                f"{where} {column} token {token!r} is not a gene designation.  This "
                f"column holds bare symbols (ERAP1, HLA-DOA, H2-K1); the mechanism is "
                f"already in the column name and any dose, clone or timing detail "
                f"belongs in `condition`."
            )
        return
    if not _OPEN_TOKEN_RE.match(token):
        raise ValueError(
            f"{where} {column} token {token!r} is malformed.  Tokens must be non-empty "
            f"and free of ';'."
        )


def _reject_alias(column: str, token: str, where: str) -> None:
    """Refuse a token the vocabulary maps to a different canonical spelling.

    Load-time rejection rather than a silent rewrite, so the YAML always shows
    what a consumer will filter on.
    """
    canonical = canonical_condition_token(column, token)
    if canonical != token:
        raise ValueError(
            f"{where} {column} token {token!r} is an alias for {canonical!r}.  "
            f"Curate the canonical spelling so a membership filter finds this "
            f"row; condition_vocabulary.yaml records the mapping."
        )


def _validate_column(column: str, raw: object, where: str) -> str:
    """Validate one condition cell and return its exported string form."""
    if raw is None:
        return ""
    if not isinstance(raw, str):
        raise ValueError(
            f"{where} {column}={raw!r} must be a string.  Every condition column is a "
            f"categorical string; a bare list or number would export as prose."
        )
    value = raw.strip()
    if not value:
        return ""

    if column in FREE_TEXT_CONDITION_COLUMNS:
        return value

    allowed = CLOSED_CONDITION_VOCABULARIES.get(column)
    if value == "none" and column not in NONE_PERMITTED_CONDITION_COLUMNS:
        raise ValueError(
            f"{where} {column}={value!r} uses 'none', which this column does not "
            f"support.  'none' asserts that an intervention was not applied; for a "
            f"context field the source simply did not say, and '' is how that is "
            f"recorded."
        )
    if column in MULTI_VALUE_CONDITION_COLUMNS:
        if ";;" in value or value.startswith(";") or value.endswith(";"):
            raise ValueError(
                f"{where} {column}={value!r} has an empty token.  Use '' for "
                f"'not established', not a dangling separator."
            )
        tokens = value.split(";")
        if "none" in tokens:
            if len(tokens) > 1:
                raise ValueError(
                    f"{where} {column}={value!r} mixes 'none' with {sorted(set(tokens) - {'none'})}.  "
                    f"'none' is a claim that the column's intervention was absent; it "
                    f"cannot hold alongside one that was present."
                )
            return value
        for token in tokens:
            _validate_token(column, token, where)
            if allowed is not None and token not in allowed:
                raise ValueError(
                    f"{where} {column} token {token!r} is not in the declared vocabulary {allowed}."
                )
            _reject_alias(column, token, where)
        if sorted(set(tokens)) != tokens:
            raise ValueError(
                f"{where} {column}={value!r} must be sorted and unique.  A canonical "
                f"cell is what makes equality comparable across rows — and the sort is "
                f"why token order never encodes a sequence; use "
                f"condition_combination=sequential and keep the order in `condition`."
            )
        return value

    if allowed is not None and value not in allowed:
        raise ValueError(f"{where} {column}={value!r} is invalid; expected one of {allowed}.")
    _validate_token(column, value, where)
    # `condition_culture` is the one open-vocabulary single-valued column, so
    # without this its aliases were declared and unenforceable: `RPMI` loaded
    # happily beside the rows curated `RPMI-1640` and a membership filter
    # missed it.
    _reject_alias(column, value, where)
    return value


def validate_sample_conditions(sample: Mapping[str, object], pmid: object, index: int) -> None:
    """Validate one ``ms_samples`` record's condition block.

    Raises ``ValueError`` with the offending PMID and record index, matching
    the house style of the other loader guards.
    """
    where = _describe(pmid, index)
    values = {
        column: _validate_column(column, sample.get(column), where) for column in CONDITION_COLUMNS
    }

    condition_id = values["condition_id"]
    if not condition_id:
        raise ValueError(
            f"{where} has no condition_id.  Every curated arm needs a persistent "
            f"identifier: it is what an attributed observation reports, and without "
            f"one the only handle on the arm is its display label, which is free to "
            f"change (#450)."
        )
    if not _CONDITION_ID_RE.match(condition_id):
        raise ValueError(
            f"{where} condition_id={condition_id!r} is malformed; expected lowercase "
            f"letters, digits and underscores (it is a key, not a label)."
        )

    status = values["condition_status"]
    if not status:
        raise ValueError(
            f"{where} has no condition_status.  A blank status is indistinguishable "
            f"from a fully annotated one, which is how an unannotated arm reads as a "
            f"complete description of the experiment; use 'unreported' to say so."
        )

    evidence = values["condition_evidence"]
    if status == "unreported":
        if evidence:
            raise ValueError(
                f"{where} is condition_status='unreported' but claims "
                f"condition_evidence={evidence!r}.  Nothing has been annotated, so "
                f"there is no annotation for a source to support."
            )
    elif not evidence:
        raise ValueError(
            f"{where} is condition_status={status!r} without condition_evidence.  "
            f"Normalizing the existing curated string and reading the paper are "
            f"different claims and the audit reports them separately."
        )

    interventions = [c for c in INTERVENTION_CONDITION_COLUMNS if values[c] and values[c] != "none"]
    if values["condition_combination"] and not interventions:
        raise ValueError(
            f"{where} sets condition_combination="
            f"{values['condition_combination']!r} but names no intervention.  The "
            f"value describes how the documented interventions relate to each "
            f"other, so with none documented it asserts agents the record does "
            f"not have."
        )

    if evidence == "primary_source" and not values["condition_reference"]:
        raise ValueError(
            f"{where} claims condition_evidence='primary_source' with no "
            f"condition_reference.  A primary-source claim names where in the source "
            f"the facts are — section, figure, table, sheet or sample identifier."
        )
    if values["condition_reference"] and evidence != "primary_source":
        raise ValueError(
            f"{where} has a condition_reference but condition_evidence={evidence!r}.  "
            f"The reference documents a source that was read; recording one against "
            f"mechanically normalized text overstates the evidence."
        )


def validate_study_conditions(entry: Mapping[str, object]) -> None:
    """Validate a study's condition block: per-record rules, then the study-scoped ones.

    Uniqueness and control references are study-scoped because
    ``(pmid, condition_id)`` is the identity: a duplicate would make two
    arms indistinguishable to every join, and a dangling
    ``condition_control_for`` would point at nothing while still reading as
    a documented comparison.
    """
    pmid = entry.get("pmid", entry.get("submission_id"))
    samples = entry.get("ms_samples") or []

    # Opt-in per study and all-or-none, the rule `sample_group` already uses
    # (#359).  A study that curates no condition block at all is uncurated
    # and says so; one that curates half is the silent case — the annotated
    # arms are filterable, the rest are indistinguishable from arms whose
    # condition nobody could establish, and no error marks the difference.
    # A record that genuinely has nothing to say uses
    # `condition_status: unreported`, which is a statement, not a silence.
    annotated = [i for i, s in enumerate(samples) if any(c in s for c in CONDITION_COLUMNS)]
    if not annotated:
        return
    if len(annotated) != len(samples):
        missing = [
            samples[i].get("sample_label", "?")
            for i in range(len(samples))
            if i not in set(annotated)
        ]
        raise ValueError(
            f"PMID {pmid}: the condition columns are curated on {len(annotated)} of "
            f"{len(samples)} ms_samples; it must be all or none.  Missing on {missing}.  "
            f"A half-curated study exports blanks that are indistinguishable from "
            f"'nobody could establish this', so the gap is invisible to the coverage "
            f"audit.  Use condition_status: unreported to say an arm has nothing "
            f"recorded (#450)."
        )

    seen: dict[str, int] = {}
    for index, sample in enumerate(samples):
        validate_sample_conditions(sample, pmid, index)
        condition_id = str(sample.get("condition_id") or "").strip()
        if condition_id in seen:
            raise ValueError(
                f"PMID {pmid}: condition_id={condition_id!r} is used by ms_samples"
                f"[{seen[condition_id]}] and ms_samples[{index}].  It identifies the "
                f"curated context, so two arms sharing one are the same arm to every "
                f"consumer that joins on it."
            )
        seen[condition_id] = index

    for index, sample in enumerate(samples):
        for target in split_condition_tokens(
            str(sample.get("condition_control_for") or "").strip()
        ):
            if target not in seen:
                raise ValueError(
                    f"PMID {pmid}: ms_samples[{index}] condition_control_for names "
                    f"{target!r}, which is not a condition_id in this study "
                    f"(known: {sorted(seen)}).  A control points at the arm it is the "
                    f"control for."
                )
            if seen[target] == index:
                raise ValueError(
                    f"PMID {pmid}: ms_samples[{index}] condition_control_for names its "
                    f"own condition_id {target!r}.  An arm is not its own comparator."
                )


def condition_columns_for_sample(sample: Mapping[str, object]) -> dict[str, str]:
    """The condition column block for one curated ``ms_samples`` record.

    Reads explicit curation only.  An uncurated field stays ``""`` rather
    than being guessed from the ``condition`` prose — a guess here would be
    indistinguishable from curation in every export, and the whole point of
    the block is that a consumer can tell those apart.
    """
    return {column: str(sample.get(column) or "").strip() for column in CONDITION_COLUMNS}


def empty_condition_columns() -> dict[str, str]:
    """The blank block, for binding rows and unresolved observations.

    Binding evidence has no MS sample, so it has no experimental condition
    to report.  ``""`` says that; any other default would assert an
    untreated arm for every predicted binder in the training table.
    """
    return dict.fromkeys(CONDITION_COLUMNS, "")
