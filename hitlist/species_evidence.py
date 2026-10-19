"""Portable evidence over an explicit species reference, without live providers."""

from __future__ import annotations

import io
import json
import re
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path

import pandas as pd

from .assays import annotate_assays
from .detectability import _fasta_records
from .lineage import attach_lineage, validate_lineage
from .provenance import CONTRIBUTOR_COLUMNS, file_digest
from .species_expression import (
    REFERENCE_FIELDS,
    canonical_json,
    expression_audit,
    json_digest,
    normal_expression_coverage,
    sequence_id,
    validate_expression,
    validate_policy,
)
from .tissue_blacklist import build_tissue_blacklist

KIND = "species_expression"
SCHEMA_VERSION = 1
LIMITS = {
    "max_input_bytes": 256 * 1024**2,
    "max_json_bytes": 64 * 1024**2,
    "max_table_bytes": 128 * 1024**2,
    "max_records": 250000,
    "max_reference_bytes": 512 * 1024**2,
    "max_protein_residues": 1000000,
    "max_total_residues": 100000000,
    "max_peptides": 50000,
    "max_mapping_rows": 250000,
    "max_output_bytes": 512 * 1024**2,
}
INPUT_NAMES = {
    "reference",
    "expression",
    "candidates",
    "observations",
    "contributors",
    "lineage",
    "normal",
}
TABLE_NAMES = {
    "mappings",
    "peptides",
    "presentation",
    "excluded_observations",
    "tissue_evidence",
    "tissue_risk",
}
JSON_NAMES = {"expression_audit", "group_expression", "peptide_expression"}
MAPPING_COLUMNS = [
    "peptide",
    "match_kind",
    "protein_id",
    "protein_sequence_id",
    "start",
    "end",
    "occurrence_id",
    "gene_id",
    "transcript_id",
    "annotation_status",
    "candidate_group",
]
PEPTIDE_COLUMNS = [
    "peptide",
    "n_exact_proteins",
    "n_exact_occurrences",
    "n_unannotated_proteins",
    "candidate_specific_in_reference",
    "n_il_equivalent_proteins",
    "n_exact_ms_observations",
    "n_native_ms_observations",
    "n_heterologous_ms_observations",
    "n_unknown_context_ms_observations",
    "il_identity_status",
    "normal_expression_counterevidence",
    "normal_expression_unresolved",
    "normal_ms_coverage",
    "n_normal_ms_donors",
    "blacklisted",
]
OBSERVATION_FIELDS = {
    "peptide",
    "provenance_id",
    "source",
    "pmid",
    "mhc_restriction",
    "species",
    "host",
    "presenting_species",
    "mhc_species",
    "source_taxon",
    "presenting_taxon",
    "mhc_taxon",
    "condition_id",
    "sample_attribution",
    "attributed_sample_label",
    "assay_method",
    "response_measured",
    "qualitative_measurement",
}


def _established_ms(frame):
    """Species schema 1: modality and established outcome are separate gates.

    Blank outcomes in the existing curated observed-ligand-table contract are
    source-supported observations. A structured method alone is insufficient;
    explicit unknown outcomes never qualify (#665). Human replay is unchanged.
    """
    frame = annotate_assays(frame.copy())
    outcome = frame.qualitative_measurement.astype("string").fillna("").str.strip().str.casefold()
    positive = outcome.isin(["positive", "positive-high", "positive-intermediate", "positive-low"])
    curated = outcome.eq("") & frame.assay_modality_source.eq("curated_ms_supplement")
    frame["ms_outcome_status"] = "unestablished"
    frame.loc[outcome.str.startswith("negative"), "ms_outcome_status"] = "negative"
    frame.loc[positive, "ms_outcome_status"] = "positive_result"
    frame.loc[curated, "ms_outcome_status"] = "curated_ligand_table"
    frame["is_ms_observation"] &= positive | curated
    return frame


def _json(path, max_bytes):
    if path.stat().st_size > max_bytes:
        raise ValueError("JSON exceeds max_json_bytes")

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def invalid(value):
        raise ValueError(f"Non-finite JSON value: {value}")

    return json.loads(path.read_text(), object_pairs_hook=pairs, parse_constant=invalid)


class _BoundedWriter(io.BufferedWriter):
    """Reserve bytes before writing, sharing one budget across every artifact."""

    def __init__(self, path, budget):
        super().__init__(io.FileIO(path, "wb"))
        self.budget = budget

    def write(self, data):
        if len(data) > self.budget["remaining"]:
            raise ValueError("Bundle exceeds max_output_bytes")
        self.budget["remaining"] -= len(data)
        return super().write(data)


def _write_json(path, value, budget):
    encoder = json.JSONEncoder(sort_keys=True, separators=(",", ":"), allow_nan=False)
    with _BoundedWriter(path, budget) as stream:
        for chunk in encoder.iterencode(value):
            stream.write(chunk.encode())
        stream.write(b"\n")


def _limits(overrides):
    if set(overrides) - LIMITS.keys():
        raise ValueError("Unknown species bundle resource limit")
    result = {**LIMITS, **overrides}
    if any(type(v) is not int or v < 1 for v in result.values()):
        raise ValueError("Resource limits must be positive integers")
    return result


def _local(root, value):
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("Input paths must be relative local files")
    path = root / path
    if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Input path must remain inside the manifest directory")
    return path


def _table(path, limits):
    import pyarrow.parquet as pq

    source = pq.ParquetFile(path)
    meta = source.metadata
    n_bytes = sum(meta.row_group(i).total_byte_size for i in range(meta.num_row_groups))
    if meta.num_rows > limits["max_records"] or n_bytes > limits["max_table_bytes"]:
        raise ValueError("Parquet exceeds max_records or max_table_bytes")
    return source.read().to_pandas()


def _validate_manifest(manifest, root, limits):
    if manifest.get("schema_version") != 1 or set(manifest.get("files", {})) != INPUT_NAMES:
        raise ValueError("Species input manifest requires schema 1 and all explicit input files")
    reference = manifest.get("reference", {})
    if (
        set(reference) != REFERENCE_FIELDS
        or type(reference.get("taxon")) is not int
        or reference["taxon"] < 1
    ):
        raise ValueError(
            "Explicit reference assembly, annotation, source, asset hashes and taxon required"
        )
    for field in ("assembly_accession", "annotation_release", "source_version"):
        if not isinstance(reference[field], str) or not reference[field].strip():
            raise ValueError("Reference version fields must be nonempty strings")
    hashes = reference["asset_hashes"]
    if (
        not isinstance(hashes, dict)
        or not hashes
        or any(
            not isinstance(v, str) or not re.fullmatch("[0-9a-f]{64}", v) for v in hashes.values()
        )
    ):
        raise ValueError("Reference asset hashes must be SHA256 strings")
    if manifest.get("reference_complete") is not True or not manifest.get("reference_scope"):
        raise ValueError("Complete reference and explicit reference_scope declaration required")
    if manifest["files"]["reference"].get("sha256") not in hashes.values():
        raise ValueError("Protein FASTA digest is absent from reference asset hashes")
    validate_policy(manifest.get("policy", {}))
    normal = manifest.get("normal_source", {})
    if normal.get("taxon") != reference["taxon"]:
        raise ValueError("Normal source taxon must match the reference taxon")
    if normal.get("coverage") not in {"missing", "scoped"} or any(
        not isinstance(normal.get(k), str) or not normal[k].strip()
        for k in ("source", "version", "coverage_note")
    ):
        raise ValueError("Explicit versioned normal source and coverage declaration required")
    if type(manifest.get("include_il_equivalent", False)) is not bool:
        raise ValueError("include_il_equivalent must be a boolean")
    paths, n_bytes = {}, 0
    for name, record in manifest["files"].items():
        path = _local(root, record["path"])
        if type(record.get("size_bytes")) is not int or record["size_bytes"] < 0:
            raise ValueError("Pinned size_bytes required for every input")
        n_bytes += path.stat().st_size
        if n_bytes > limits["max_input_bytes"]:
            raise ValueError("Inputs exceed max_input_bytes")
        if file_digest(path) != {k: record.get(k) for k in ("sha256", "size_bytes")}:
            raise ValueError(f"Input checksum/size mismatch: {name}")
        paths[name] = path
    return paths


def _observations(paths, limits, taxon):
    frame = _table(paths["observations"], limits)
    if not set(frame) >= OBSERVATION_FIELDS:
        raise ValueError(f"Observation fields missing: {sorted(OBSERVATION_FIELDS - set(frame))}")
    if frame[list(OBSERVATION_FIELDS)].isna().any().any():
        raise ValueError(
            "Observation context must be explicit (empty strings/zero taxon for unknown)"
        )
    for col in ("source_taxon", "presenting_taxon", "mhc_taxon"):
        if (
            not frame[col]
            .map(lambda v: isinstance(v, int) and not isinstance(v, bool) and v >= 0)
            .all()
        ):
            raise ValueError(
                "Observation taxon fields must be nonnegative integers; zero means unknown"
            )
    contributors = _table(paths["contributors"], limits)
    if not set(CONTRIBUTOR_COLUMNS) <= set(contributors):
        raise ValueError("Full contributor records required")
    if contributors[["provenance_id", "source_record_id", "source_dataset"]].isna().any().any():
        raise ValueError("Contributor identities cannot be missing")
    if any(
        contributors[c].astype(str).str.strip().eq("").any()
        for c in ("provenance_id", "source_record_id", "source_dataset")
    ):
        raise ValueError("Contributor identities cannot be empty")
    if (
        set(frame.provenance_id) != set(contributors.provenance_id)
        or frame.provenance_id.eq("").any()
    ):
        raise ValueError("Exact contributor coverage required for all observations")
    if contributors.duplicated(["provenance_id", "source_record_id"]).any():
        raise ValueError("Duplicate contributor record identity")
    observed_strings = frame.groupby("provenance_id").peptide.agg(set).to_dict()
    for row in contributors.itertuples(index=False):
        fields = json.loads(row.original_fields)
        if (
            not isinstance(fields, dict)
            or not isinstance(json.loads(row.source_row_values), list)
            or not isinstance(json.loads(row.relationships), list)
        ):
            raise ValueError("Contributor raw source fields/relationships have invalid JSON types")
        reported = fields.get("peptide")
        if (
            isinstance(reported, str)
            and re.fullmatch("[ACDEFGHIKLMNPQRSTVWY]+", reported)
            and observed_strings[row.provenance_id] != {reported}
        ):
            raise ValueError("Contributor peptide contradicts its linked observation")
    frame = _established_ms(frame)
    valid_sequence = frame.peptide.astype("string").str.fullmatch("[ACDEFGHIKLMNPQRSTVWY]{5,50}")
    frame["exclusion_reason"] = ""
    frame.loc[~frame.is_ms_observation, "exclusion_reason"] = "not_positive_ms"
    frame.loc[
        frame.assay_modality.eq("ms") & frame.ms_outcome_status.eq("unestablished"),
        "exclusion_reason",
    ] = "unestablished_ms_outcome"
    frame.loc[~valid_sequence, "exclusion_reason"] = "invalid_or_unresolved_sequence"
    frame["evidence_kind"] = frame.is_ms_observation.map({True: "ms", False: "other"})
    frame = attach_lineage(
        frame, registry=validate_lineage(_json(paths["lineage"], limits["max_json_bytes"]))
    )
    if "evidence_row_id" not in frame:
        identity_fields = sorted(OBSERVATION_FIELDS)
        frame["evidence_row_id"] = [
            "species-observation:" + json_digest(row)
            for row in json.loads(frame[identity_fields].to_json(orient="records"))
        ]
    if (
        frame.evidence_row_id.isna().any()
        or frame.evidence_row_id.eq("").any()
        or frame.evidence_row_id.duplicated().any()
    ):
        raise ValueError("Observation evidence_row_id must be unique and nonempty")
    # These explicit taxon axes accompany, and never overwrite, the reported
    # source/host/MHC names and restriction fields.
    frame["presentation_context"] = "unknown"
    native = frame.source_taxon.eq(taxon) & frame.presenting_taxon.eq(taxon)
    heterologous = (
        frame.source_taxon.gt(0)
        & frame.presenting_taxon.gt(0)
        & ~native
        & frame.mhc_taxon.eq(taxon)
    )
    frame.loc[native, "presentation_context"] = "native"
    frame.loc[heterologous, "presentation_context"] = "heterologous_mhc"
    frame["il_identity_status"] = "reported_string_only"
    # No input can turn the default unresolved MS I/L identity into an exact
    # molecular identification merely by asserting it in a derived column.
    return frame.sort_values("evidence_row_id").reset_index(drop=True)


def _mappings(path, peptides, occurrences, candidates, limits, include_il):
    by_protein, seeds = defaultdict(list), defaultdict(list)
    for row in occurrences.values():
        by_protein[row["protein_id"]].append(row)
    for peptide in sorted(peptides):
        seeds[peptide[:5]].append((peptide, peptide, "exact"))
    il_seeds = defaultdict(list)
    if include_il:
        for peptide in sorted(peptides):
            normalized = peptide.replace("I", "L")
            il_seeds[normalized[:5]].append((peptide, normalized, "il_equivalent"))
    candidate_ids = {c["sequence_id"] for c in candidates}
    mappings, seen_occurrences, unannotated = [], set(), set()
    n_residues, n_proteins = 0, 0
    for protein, _, sequence in _fasta_records(
        path,
        max_protein_residues=limits["max_protein_residues"],
        max_reference_bytes=limits["max_reference_bytes"],
    ):
        n_proteins += 1
        n_residues += len(sequence)
        if n_residues > limits["max_total_residues"]:
            raise ValueError("Reference exceeds max_total_residues")
        sid = sequence_id(sequence)
        annotations = by_protein.get(protein, [])
        for row in annotations:
            if row["sequence"] != sequence:
                raise ValueError(f"Occurrence sequence disagrees with reference protein {protein}")
            seen_occurrences.add(row["occurrence_id"])
        if not annotations:
            unannotated.add(sid)
        scans = [(sequence, seeds)]
        if include_il:
            scans.append((sequence.replace("I", "L"), il_seeds))
        for scanned, index in scans:
            for start in range(max(0, len(scanned) - 4)):
                for peptide, pattern, kind in index.get(scanned[start : start + 5], ()):
                    end = start + len(pattern)
                    if not scanned.startswith(pattern, start) or (
                        kind == "il_equivalent" and sequence[start:end] == peptide
                    ):
                        continue
                    for annotation in annotations or [{}]:
                        if len(mappings) >= limits["max_mapping_rows"]:
                            raise ValueError("Mappings exceed max_mapping_rows")
                        mappings.append(
                            {
                                "peptide": peptide,
                                "match_kind": kind,
                                "protein_id": protein,
                                "protein_sequence_id": sid,
                                "start": start,
                                "end": end,
                                "occurrence_id": annotation.get("occurrence_id", ""),
                                "gene_id": annotation.get("gene_id", ""),
                                "transcript_id": annotation.get("transcript_id", ""),
                                "annotation_status": "resolved" if annotation else "unresolved",
                                "candidate_group": sid in candidate_ids,
                            }
                        )
    if not n_proteins or seen_occurrences != set(occurrences):
        raise ValueError("Reference is empty or does not contain every supplied occurrence")
    return pd.DataFrame(mappings, columns=MAPPING_COLUMNS).sort_values(
        ["peptide", "match_kind", "protein_id", "start", "occurrence_id"]
    ).reset_index(drop=True), unannotated


def _normal(paths, manifest, limits):
    frame = _table(paths["normal"], limits)
    required = {
        "source_taxon",
        "assay_method",
        "response_measured",
        "qualitative_measurement",
        "source",
    }
    if (
        not required <= set(frame)
        or not frame.source_taxon.eq(manifest["reference"]["taxon"]).all()
    ):
        raise ValueError("Normal observations need matching source taxon and original assay fields")
    if frame.empty != (manifest["normal_source"]["coverage"] == "missing"):
        raise ValueError(
            "Empty normal evidence must declare missing coverage; nonempty evidence is scoped"
        )
    if "source_record_id" not in frame or frame.source_record_id.duplicated().any():
        raise ValueError("Normal source_record_id must identify unique source rows")
    if "is_cell_line" not in frame or not frame.is_cell_line.map(lambda v: type(v) is bool).all():
        raise ValueError("Normal is_cell_line must be an explicit boolean")
    frame = _established_ms(frame)
    frame["assay_modality"] = frame.is_ms_observation.map(
        {True: "mass_spectrometry", False: "not_positive_ms"}
    )
    risk, audit = build_tissue_blacklist(
        frame, min_donors=2, policy=manifest["policy"]["tissue_blacklist"]
    )
    return risk, audit.sort_values(["peptide", "source_record_id"]).reset_index(drop=True)


def _derive(root, manifest, limits):
    paths = _validate_manifest(manifest, root, limits)
    expression = _json(paths["expression"], limits["max_json_bytes"])
    candidates = _json(paths["candidates"], limits["max_json_bytes"])
    occurrences, samples, allocations, groups = validate_expression(
        expression, manifest["reference"], candidates, manifest["policy"], limits["max_records"]
    )
    observations = _observations(paths, limits, manifest["reference"]["taxon"])
    risk, tissue = _normal(paths, manifest, limits)
    peptides = manifest.get("peptides", [])
    if not isinstance(peptides, list) or any(
        not isinstance(p, str) or not re.fullmatch("[ACDEFGHIKLMNPQRSTVWY]{5,50}", p)
        for p in peptides
    ):
        raise ValueError("Requested peptides must be exact unmodified 5-50 amino-acid strings")
    peptides = set(peptides) | set(
        observations.loc[
            observations.peptide.astype("string").str.fullmatch("[ACDEFGHIKLMNPQRSTVWY]{5,50}"),
            "peptide",
        ]
    )
    if len(peptides) > limits["max_peptides"]:
        raise ValueError("Peptides exceed max_peptides")
    mappings, unannotated = _mappings(
        paths["reference"],
        peptides,
        occurrences,
        candidates,
        limits,
        manifest.get("include_il_equivalent", False),
    )
    expression_rows, group_rows = expression_audit(
        occurrences, samples, allocations, groups, candidates, manifest["policy"], unannotated
    )
    expression_by_group = defaultdict(list)
    for row in expression_rows:
        expression_by_group[row["sequence_id"]].append(row)
    missing_expression_coverage = {}
    for sid, rows in expression_by_group.items():
        by_namespace = defaultdict(list)
        for row in rows:
            by_namespace[canonical_json(row["namespace"])].append(row)
        missing_expression_coverage[sid] = any(
            len(donors) < manifest["policy"]["min_normal_donors"]
            for namespace_rows in by_namespace.values()
            for donors in normal_expression_coverage(
                namespace_rows, samples, manifest["policy"]
            ).values()
        )
    peptide_expression = []
    expression_flags = defaultdict(lambda: {"counterevidence": False, "unresolved": False})
    for hit in (
        mappings.loc[mappings.match_kind.eq("exact"), ["peptide", "protein_sequence_id"]]
        .drop_duplicates()
        .itertuples(index=False)
    ):
        rows = expression_by_group[hit.protein_sequence_id]
        if missing_expression_coverage.get(hit.protein_sequence_id, True):
            expression_flags[hit.peptide]["unresolved"] = True
        for row in rows:
            sample = samples[row["sample_id"]]
            normal = (
                sample.get("health") == "healthy"
                and sample.get("tissue")
                and sample["tissue"] not in manifest["policy"]["allowed_tissues"]
            )
            counter = bool(
                normal
                and row["unit_matches_policy"]
                and row["lower"] is not None
                and row["lower"] > manifest["policy"]["normal_max"]
            )
            unresolved = not row["unit_matches_policy"] or row["ambiguous"] or row["upper"] is None
            expression_flags[hit.peptide]["counterevidence"] |= counter
            expression_flags[hit.peptide]["unresolved"] |= unresolved
            if len(peptide_expression) >= limits["max_mapping_rows"]:
                raise ValueError("Expression links exceed max_mapping_rows")
            peptide_expression.append(
                {
                    "peptide": hit.peptide,
                    **row,
                    "normal_counterevidence": counter,
                    "measurement_target": "full_sequence_group_not_peptide_abundance",
                }
            )
    presentation = observations.loc[observations.exclusion_reason.eq("")].reset_index(drop=True)
    excluded = observations.loc[observations.exclusion_reason.ne("")].reset_index(drop=True)
    mapping_groups = dict(iter(mappings.groupby("peptide", sort=False)))
    observed_groups = dict(iter(presentation.groupby("peptide", sort=False)))
    tissue_risk = risk.set_index("peptide").to_dict(orient="index")
    summaries = []
    for peptide in sorted(peptides):
        hits = mapping_groups.get(peptide, mappings.iloc[:0])
        exact = hits[hits.match_kind.eq("exact")]
        observed = observed_groups.get(peptide, presentation.iloc[:0])
        normal = tissue_risk.get(peptide, {})
        summaries.append(
            {
                "peptide": peptide,
                "n_exact_proteins": exact.protein_id.nunique(),
                "n_exact_occurrences": exact.loc[
                    exact.occurrence_id.ne(""), "occurrence_id"
                ].nunique(),
                "n_unannotated_proteins": exact.loc[
                    exact.annotation_status.eq("unresolved"), "protein_id"
                ].nunique(),
                "candidate_specific_in_reference": bool(len(exact))
                and bool(exact.candidate_group.all())
                and bool(exact.annotation_status.eq("resolved").all()),
                "n_il_equivalent_proteins": hits.loc[
                    hits.match_kind.eq("il_equivalent"), "protein_id"
                ].nunique(),
                "n_exact_ms_observations": len(observed),
                "n_native_ms_observations": int(observed.presentation_context.eq("native").sum()),
                "n_heterologous_ms_observations": int(
                    observed.presentation_context.eq("heterologous_mhc").sum()
                ),
                "n_unknown_context_ms_observations": int(
                    observed.presentation_context.eq("unknown").sum()
                ),
                "il_identity_status": "reported_string_only"
                if len(observed)
                else "no_exact_reported_ms",
                "normal_expression_counterevidence": expression_flags[peptide]["counterevidence"],
                "normal_expression_unresolved": expression_flags[peptide]["unresolved"]
                or not len(exact),
                "normal_ms_coverage": manifest["normal_source"]["coverage"],
                "n_normal_ms_donors": normal.get("n_donors", 0),
                "blacklisted": normal.get("blacklisted", False),
            }
        )
    return {
        "mappings": mappings,
        "peptides": pd.DataFrame(summaries, columns=PEPTIDE_COLUMNS),
        "presentation": presentation,
        "excluded_observations": excluded,
        "tissue_evidence": tissue,
        "tissue_risk": risk,
        "expression_audit": expression_rows,
        "group_expression": group_rows,
        "peptide_expression": peptide_expression,
    }


def write_species_evidence_bundle(
    directory, *, input_manifest="species-evidence.json", limits=None
):
    """Freeze and audit explicit species inputs into a fresh offline bundle.

    No downloads, global index, human CTA provider or normal-tissue fallback.
    See ``docs/species-evidence.md`` for the versioned manifest contract. Bounds
    are positive integer overrides of :data:`LIMITS`; exceeding one publishes
    nothing. Existing destinations are never overwritten.
    """
    from . import __version__

    directory, source = Path(directory), Path(input_manifest)
    if directory.exists():
        raise FileExistsError(directory)
    limits = _limits(limits or {})
    manifest = _json(source, min(limits["max_json_bytes"], 1024**2))
    paths = _validate_manifest(manifest, source.parent, limits)
    directory.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".species-evidence-", dir=directory.parent))
    budget = {"remaining": limits["max_output_bytes"]}
    try:
        (staging / "inputs").mkdir()
        frozen = json.loads(canonical_json(manifest))
        for name, path in paths.items():
            suffix = (
                ".fasta.gz"
                if name == "reference" and path.name.endswith(".gz")
                else ".fasta"
                if name == "reference"
                else ".parquet"
                if name in {"observations", "contributors", "normal"}
                else ".json"
            )
            relative = "inputs/" + name + suffix
            with (
                path.open("rb") as source_stream,
                _BoundedWriter(staging / relative, budget) as destination_stream,
            ):
                shutil.copyfileobj(source_stream, destination_stream, length=1024**2)
            frozen["files"][name]["path"] = relative
        _write_json(staging / "input.json", frozen, budget)
        derived = _derive(staging, frozen, limits)
        for name in sorted(TABLE_NAMES):
            with _BoundedWriter(staging / (name + ".parquet"), budget) as stream:
                derived[name].to_parquet(stream, index=False)
        for name in sorted(JSON_NAMES):
            _write_json(staging / (name + ".json"), derived[name], budget)
        forbidden = (
            derived["tissue_risk"].loc[derived["tissue_risk"].blacklisted, "peptide"].tolist()
        )
        with _BoundedWriter(staging / "forbidden_sequences.txt", budget) as stream:
            stream.write("".join(p + "\n" for p in forbidden).encode())
        del derived
        files = {
            str(p.relative_to(staging)): file_digest(p)
            for p in sorted(staging.rglob("*"))
            if p.is_file()
        }
        result = {
            "schema_version": SCHEMA_VERSION,
            "kind": KIND,
            "hitlist_version": __version__,
            "reference_key": json_digest(frozen["reference"]),
            "limits": limits,
            "files": files,
        }
        _write_json(staging / "manifest.json", result, budget)
        if (
            sum(p.stat().st_size for p in staging.rglob("*") if p.is_file())
            > limits["max_output_bytes"]
        ):
            raise ValueError("Bundle exceeds max_output_bytes")
        verify_species_evidence_bundle(staging)
        if directory.exists():
            raise FileExistsError(directory)
        staging.rename(directory)
        return result
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def verify_species_evidence_bundle(directory):
    """Verify bytes and replay derivations using only captured local inputs."""
    from .evidence_bundle import _equal_frame

    root = Path(directory)
    manifest = _json(root / "manifest.json", 1024**2)
    if manifest.get("kind") != KIND or manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported species evidence bundle")
    limits = _limits(manifest.get("limits", {}))
    if sum(p.stat().st_size for p in root.rglob("*") if p.is_file()) > limits["max_output_bytes"]:
        raise ValueError("Bundle exceeds max_output_bytes")
    for name, fingerprint in manifest["files"].items():
        if file_digest(_local(root, name)) != fingerprint:
            raise ValueError(f"Species evidence checksum mismatch: {name}")
    frozen = _json(root / "input.json", min(limits["max_json_bytes"], 1024**2))
    expected = (
        {"input.json", "forbidden_sequences.txt"}
        | {n + ".parquet" for n in TABLE_NAMES}
        | {n + ".json" for n in JSON_NAMES}
        | {v["path"] for v in frozen["files"].values()}
    )
    if set(manifest["files"]) != expected or manifest["reference_key"] != json_digest(
        frozen["reference"]
    ):
        raise ValueError("Species evidence artifact/reference contract mismatch")
    derived = _derive(root, frozen, limits)
    for name in sorted(TABLE_NAMES):
        _equal_frame(pd.read_parquet(root / (name + ".parquet")), derived[name], name)
    for name in sorted(JSON_NAMES):
        if _json(root / (name + ".json"), limits["max_output_bytes"]) != derived[name]:
            raise ValueError(f"Species evidence replay mismatch: {name}")
    forbidden = derived["tissue_risk"].loc[derived["tissue_risk"].blacklisted, "peptide"].tolist()
    if (root / "forbidden_sequences.txt").read_text().splitlines() != forbidden:
        raise ValueError("Species evidence forbidden sequences mismatch")
    return manifest
