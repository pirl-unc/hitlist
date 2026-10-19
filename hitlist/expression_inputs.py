"""Conventional expression inputs, with explicit choices for real ambiguities."""

import re
from pathlib import Path

_FILES = (
    "expression.tsv",
    "expression.csv",
    "expression.tab",
    "expression.sf",
    "quant.sf",
    "abundance.tsv",
    "*.genes.results",
    "*.isoforms.results",
)
_GENE_COLUMNS = (
    "ensembl_gene_id",
    "gene_id",
    "geneid",
    "gene",
    "symbol",
    "gene_symbol",
    "gene_name",
    "external_gene_name",
)
_TRANSCRIPT_COLUMNS = (
    "ensembl_transcript_id",
    "transcript_id",
    "transcriptid",
    "transcript",
    "tx_id",
)
_NEUTRAL_COLUMNS = ("target_id", "targetid", "target", "name", "id")
_RSEM_COLUMNS = {"gene_id", "length", "effective_length", "expected_count", "tpm"}
_RSEM_AUXILIARY_TPM = {
    "pme_tpm",
    "isopct_from_pme_tpm",
    "tpm_ci_lower_bound",
    "tpm_ci_upper_bound",
    "tpm_coefficient_of_quartile_variation",
}


def expression_input_path(path=None):
    """Use an explicit file, or one conventional file in the chosen directory."""
    path = Path.cwd() if path is None else Path(path)
    if not path.is_dir():
        return path, False
    matches = sorted(
        {
            candidate
            for pattern in _FILES
            for suffix in ("", ".gz")
            for candidate in path.glob(pattern + suffix)
            if candidate.is_file()
        }
    )
    if len(matches) != 1:
        detail = ", ".join(candidate.name for candidate in matches) or "none"
        raise ValueError(
            f"Expected one conventional expression file in {path}; found {detail}. "
            "Use --expression PATH (or place expression.tsv, expression.csv, quant.sf, "
            "abundance.tsv or one RSEM results file here)."
        )
    return matches[0], True


def _normalized(name):
    return re.sub(r"[\s-]+", "_", str(name).strip().casefold())


def _named_column(frame, candidates):
    for candidate in candidates:
        matches = [name for name in frame if _normalized(name) == candidate]
        if len(matches) > 1:
            raise ValueError(
                f"Ambiguous identifier columns {matches}; use --id-column with the exact name"
            )
        if matches:
            return matches[0]
    return None


def _quantifier_identifier(frame):
    columns = {_normalized(name) for name in frame}
    if {"name", "length", "effectivelength", "tpm", "numreads"} <= columns:
        return "name", "transcript"
    if {"target_id", "length", "eff_length", "est_counts", "tpm"} <= columns:
        return "target_id", "transcript"
    if columns >= _RSEM_COLUMNS:
        return (
            ("transcript_id", "transcript") if "transcript_id" in columns else ("gene_id", "gene")
        )
    return None, None


def infer_expression_columns(frame, *, id_column=None, tpm_column=None, level="auto"):
    """Resolve the same auditable defaults for CLI and Python consumers.

    Stable gene IDs precede symbols. Standard quantifier schemas establish row
    level; generic tables with both identity axes require a choice. A neutral
    Name/target_id column defaults to genes unless its values are Ensembl
    transcript IDs. Mixed gene/transcript values require an explicit level.
    Numeric values never establish units: only TPM-labelled columns qualify.
    """
    if level not in {"auto", "gene", "transcript"}:
        raise ValueError("Expression level must be auto, gene or transcript")
    inferred = {
        "id_column": id_column is None,
        "tpm_column": tpm_column is None,
        "level": level == "auto",
    }
    profile_column, profile_level = _quantifier_identifier(frame)
    if id_column is None:
        if level == "auto" and profile_column:
            id_column = _named_column(frame, (profile_column,))
        elif level in {"gene", "transcript"}:
            candidates = _GENE_COLUMNS if level == "gene" else _TRANSCRIPT_COLUMNS
            id_column = _named_column(frame, candidates) or _named_column(frame, _NEUTRAL_COLUMNS)
        else:
            gene = _named_column(frame, _GENE_COLUMNS)
            transcript = _named_column(frame, _TRANSCRIPT_COLUMNS)
            if gene and transcript:
                raise ValueError(
                    "Both gene and transcript columns are present; use --expression-level or --id-column"
                )
            id_column = gene or transcript or _named_column(frame, _NEUTRAL_COLUMNS)
        if id_column is None:
            raise ValueError(
                f"Cannot infer identifier column from {list(frame.columns)}; use --id-column"
            )
    if id_column not in frame:
        raise ValueError(
            f"Identifier column {id_column!r} is missing; available: {list(frame.columns)}"
        )
    if tpm_column is None:
        # RSEM's posterior estimates and uncertainty fields describe the same
        # sample. Default to its ordinary TPM; explicit overrides remain valid.
        auxiliary = (
            _RSEM_AUXILIARY_TPM if {_normalized(name) for name in frame} >= _RSEM_COLUMNS else set()
        )
        matches = [
            name
            for name in frame
            if _normalized(name) not in auxiliary
            and (
                _normalized(name) == "tpm"
                or _normalized(name).startswith("tpm_")
                or _normalized(name).endswith("_tpm")
            )
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Expected one TPM-labelled column; found {matches}. Use --tpm-column to choose a TPM column; counts/FPKM are not TPM"
            )
        tpm_column = matches[0]
    if tpm_column not in frame or id_column == tpm_column:
        raise ValueError(
            "Specify distinct identifier and TPM columns present in the expression table"
        )
    if level == "auto":
        name = _normalized(id_column)
        if name in _TRANSCRIPT_COLUMNS:
            level = "transcript"
        elif name in _GENE_COLUMNS:
            level = "gene"
        else:
            values = frame[id_column].astype("string").fillna("").str.strip()
            values = values[values.ne("")]
            transcripts = values.str.fullmatch(r"ENST\d+(?:\.\d+)?", case=False)
            genes = values.str.fullmatch(r"ENSG\d+(?:\.\d+)?", case=False)
            if genes.any():
                if not genes.all():
                    raise ValueError("Mixed gene and other identifiers; use --expression-level")
                return id_column, tpm_column, "gene", inferred
            if transcripts.any() and not transcripts.all():
                raise ValueError("Mixed transcript and other identifiers; use --expression-level")
            if name == profile_column:
                return id_column, tpm_column, profile_level, inferred
            level = "transcript" if len(values) and transcripts.all() else "gene"
    return id_column, tpm_column, level, inferred
