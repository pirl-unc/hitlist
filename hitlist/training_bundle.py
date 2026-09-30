"""Shared, content-bound training artifacts for provenance and split audits."""

from __future__ import annotations

import inspect
import json
import shutil
import tempfile
from importlib.metadata import version
from pathlib import Path

import pandas as pd

from .lineage import AXES, LINEAGE_COLUMNS, lineage_path, load_lineage
from .provenance import (
    CONTRIBUTOR_COLUMNS,
    _contributor_contract,
    contributors_path,
    file_digest,
)
from .split_audit import POLICIES, audit_splits

SCHEMA_VERSION = 1
FILES = {
    "training": "training.parquet",
    "identities": "identities.parquet",
    "contributors": "contributors.parquet",
    "lineage": "lineage.json",
}
IDENTITY_COLUMNS = [
    "evidence_kind",
    "evidence_row_id",
    "evidence_source_id",
    "provenance_id",
    "provenance_status",
    "peptide",
    "mhc_restriction",
    "has_peptide_level_allele",
    "pmid",
    "condition_id",
    "sample_attribution",
    "attributed_sample_label",
    "assay_iri",
    "reference_iri",
    *LINEAGE_COLUMNS,
]
_CONTRIBUTOR_SUMMARY_COLUMNS = [
    "provenance_id",
    "source_record_id",
    "source_dataset",
    "relationship_status",
    "original_pmid",
]


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _inputs(*, proteome_release=None):
    from .builder import _cache_meta, _curation_fingerprints
    from .downloads import data_dir

    # Record both the build's curation and the curation used now to enrich
    # training rows. An old index must never acquire a fictitious new build hash.
    paths = {
        name: data_dir() / name
        for name in (
            "observations.parquet",
            "binding.parquet",
            "observations_meta.json",
            "peptide_mappings.parquet",
            "peptide_mappings_meta.json",
            "observation_contributors.parquet",
            "line_expression.parquet",
        )
    }
    paths["specimen_lineage.yaml"] = lineage_path()
    from .line_expression import _packaged_input_paths

    paths.update(
        {f"line_expression:{name}": path for name, path in _packaged_input_paths().items()}
    )
    if proteome_release is not None:
        from pyensembl import EnsemblRelease

        cache = Path(EnsemblRelease(proteome_release).download_cache.cache_directory_path)
        paths.update(
            {
                f"transcript_cache:{path.relative_to(cache)}": path
                for path in sorted(cache.rglob("*"))
                if path.is_file()
            }
        )
    curation = _curation_fingerprints(fetch_missing_assets=False)
    for name, entry in curation.items():
        if entry.get("path"):
            paths[name] = Path(entry["path"])
    return {
        "files": {
            name: {"path": str(path), **file_digest(path)}
            if path.exists()
            else {"path": str(path), "missing": True}
            for name, path in paths.items()
        },
        "build_metadata": _cache_meta(),
        "current_curation": curation,
    }


def _identity_table(training):
    columns = [c for c in IDENTITY_COLUMNS if c in training]
    identities = training[columns].drop_duplicates().reset_index(drop=True)
    # Empty/legacy evidence IDs can collide; do not silently discard differing
    # biological contexts under the same ID and then certify their split.
    if identities["evidence_row_id"].duplicated().any():
        raise ValueError("Conflicting contexts for one evidence_row_id; rebuild observations")
    return identities


def _stream_contributors(ids, destination):
    """Copy selected raw records in bounded batches; keep only compact audit fields."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    contract = _contributor_contract(verify_hashes=True)
    wanted = pd.Index(sorted(set(ids) - {""}))
    if contract is None:
        if len(wanted):
            raise ValueError("Provenance metadata missing; rebuild observations")
        empty = pd.DataFrame(columns=CONTRIBUTOR_COLUMNS)
        empty.to_parquet(destination, index=False)
        return empty[_CONTRIBUTOR_SUMMARY_COLUMNS]
    source = pq.ParquetFile(contributors_path())
    summaries = []
    with pq.ParquetWriter(destination, source.schema_arrow) as writer:
        if len(wanted):
            for batch in source.iter_batches(batch_size=10000):
                mask = wanted.get_indexer(batch.column("provenance_id").to_pandas()) >= 0
                selected = pa.Table.from_batches([batch]).filter(mask)
                if len(selected):
                    writer.write_table(selected)
                    summaries.append(selected.select(_CONTRIBUTOR_SUMMARY_COLUMNS).to_pandas())
    contributors = (
        pd.concat(summaries, ignore_index=True)
        if summaries
        else pd.DataFrame(columns=_CONTRIBUTOR_SUMMARY_COLUMNS)
    )
    if set(wanted) != set(contributors.provenance_id):
        raise ValueError(
            "Missing source contributors for exported observations; rebuild observations"
        )
    return contributors


def _contributor_identities(identities, contributors):
    """Contributor uncertainty and every original PMID survive retained-row selection."""
    captured = set(contributors.provenance_id)
    uncertain = set(
        contributors.loc[contributors.relationship_status.eq("overlap_unresolved"), "provenance_id"]
    )
    identities["contributor_resolution"] = identities.provenance_id.map(
        lambda value: (
            "overlap_unresolved"
            if value in uncertain
            else "captured"
            if value in captured
            else "legacy_missing"
        )
    )
    pmids = pd.to_numeric(contributors.original_pmid, errors="coerce").astype("Int64")
    missing = set(contributors.loc[pmids.isna(), "provenance_id"])
    work = pd.DataFrame({"provenance_id": contributors.provenance_id, "pmid": pmids})
    paper_sets = (
        work.dropna()
        .drop_duplicates()
        .groupby("provenance_id")["pmid"]
        .agg(lambda values: json.dumps(sorted(str(value) for value in values)))
    )
    identities["contributor_pmids"] = identities.provenance_id.map(paper_sets).fillna("[]")
    identities["contributor_study_status"] = identities.provenance_id.map(
        lambda value: "resolved" if value in captured and value not in missing else "unknown"
    )
    return identities


def _coverage(identities, contributors):
    captured = set(contributors.provenance_id)
    return {
        "n_observations": len(identities),
        "n_observations_with_contributors": int(identities.provenance_id.isin(captured).sum()),
        "n_contributor_links": len(contributors),
        "n_source_records": int(contributors.source_record_id.nunique()),
        "n_unresolved_contributor_links": int(
            contributors.relationship_status.eq("overlap_unresolved").sum()
        ),
        "lineage": {
            axis: {
                "n_resolved_observations": int(identities[f"{axis}_status"].eq("resolved").sum()),
                "n_unresolved_observations": int(identities[f"{axis}_status"].ne("resolved").sum()),
                "n_distinct_resolved_identities": len(
                    {
                        value
                        for ids in identities.loc[
                            identities[f"{axis}_status"].eq("resolved"), f"{axis}_ids"
                        ]
                        for value in json.loads(ids)
                    }
                ),
            }
            for axis in AXES
        },
    }


def write_training_bundle(directory, *, split_policy="report_only", **training_options) -> Path:
    """Write a new reproducible training bundle using the ordinary export API.

    A requested column projection affects only ``training.parquet``; the
    identities table retains the fields needed for audits. Contributor links
    never multiply training rows. The destination must not already exist.
    Legacy indexes explicitly report missing contributor coverage.
    """
    from .export import _project_training_columns, generate_training_table
    from .version import __version__

    destination = Path(directory)
    if destination.exists():
        raise FileExistsError(f"Training bundle already exists: {destination}")
    if split_policy not in POLICIES:
        raise ValueError(f"Unknown split policy: {split_policy}")
    bound = inspect.signature(generate_training_table).bind(**training_options)
    bound.apply_defaults()
    options = bound.arguments
    # Fail before generating a large table or writing anything. Arbitrary
    # backend instances cannot be reproduced from their repr/string address.
    json.dumps(options, allow_nan=False)
    if options.get("cancer_type_backend") is not None:
        raise ValueError("Bundles require the default reproducible cancer_type_backend")
    input_options = (
        {"proteome_release": options["proteome_release"]} if options["with_peptide_origin"] else {}
    )
    inputs = _inputs(**input_options)
    # Match current-curation fingerprints even in a long-lived interpreter.
    from .curation import _clear_curation_caches

    _clear_curation_caches()
    if options["with_peptide_origin"]:
        from . import line_expression

        for cached in (
            line_expression._load_anchors_yaml,
            line_expression._load_sources_yaml,
            line_expression._alias_to_expression_key,
            line_expression._load_packaged_csv,
            line_expression._gene_name_map,
            line_expression._fingerprint_at,
            line_expression._packaged_rows,
            line_expression._expression_sources_at,
            line_expression._composed_rows,
            line_expression._carry_plan,
            line_expression._index_status,
        ):
            cached.cache_clear()
    complete_options = {**options, "columns": None}
    training = generate_training_table(**complete_options)
    identities = _identity_table(training)
    ids = identities.provenance_id.unique().tolist()
    projected = _project_training_columns(training, options["columns"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    try:
        projected.to_parquet(staging / FILES["training"], index=False)
        contributors = _stream_contributors(ids, staging / FILES["contributors"])
        identities = _contributor_identities(identities, contributors)
        identities.to_parquet(staging / FILES["identities"], index=False)
        _write_json(staging / FILES["lineage"], load_lineage())
        if _inputs(**input_options) != inputs:
            raise ValueError("Training inputs changed during export; retry from stable indexes")
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "hitlist_version": __version__,
            "dependency_versions": {
                package: version(package)
                for package in ("pandas", "pyarrow", "mhcgnomes", "numpy", "PyYAML", "pyensembl")
            },
            "training_options": options,
            "split_policy": split_policy,
            "randomness": {"used": False, "seed": None},
            "inputs": inputs,
            "artifacts": {
                key: {"file": filename, **file_digest(staging / filename)}
                for key, filename in FILES.items()
            },
            "n_rows": len(projected),
            "coverage": _coverage(identities, contributors),
        }
        _write_json(staging / "manifest.json", manifest)
        del training, projected, identities, contributors
        verify_training_bundle(staging)
        # rename must never overwrite an existing destination, even if another
        # producer created an empty directory while this export was running.
        destination.mkdir()
        try:
            for path in staging.iterdir():
                if path.name != "manifest.json":
                    path.replace(destination / path.name)
            (staging / "manifest.json").replace(destination / "manifest.json")
        except BaseException:
            shutil.rmtree(destination)
            raise
    finally:
        shutil.rmtree(staging)
    return destination / "manifest.json"


def verify_training_bundle(directory) -> dict:
    """Verify schema, bytes, contributor coverage and observation relationships."""
    from .lineage import validate_lineage

    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported training bundle schema")
    if set(manifest.get("artifacts", {})) != set(FILES):
        raise ValueError("Incomplete training bundle")
    for key, filename in FILES.items():
        entry = manifest["artifacts"][key]
        if entry["file"] != filename or file_digest(directory / filename) != {
            k: entry[k] for k in ("sha256", "size_bytes")
        }:
            raise ValueError(f"Training bundle artifact mismatch: {filename}")
    validate_lineage(json.loads((directory / FILES["lineage"]).read_text()))
    training = pd.read_parquet(
        directory / FILES["training"],
        columns=["evidence_row_id", "evidence_source_id", "evidence_kind", "provenance_id"],
    )
    identities = pd.read_parquet(directory / FILES["identities"])
    contributors = pd.read_parquet(
        directory / FILES["contributors"], columns=_CONTRIBUTOR_SUMMARY_COLUMNS
    )
    derived = _contributor_identities(identities.copy(), contributors)
    for column in ("contributor_resolution", "contributor_pmids", "contributor_study_status"):
        if identities[column].tolist() != derived[column].tolist():
            raise ValueError(f"Contributor identity mismatch: {column}")
    if identities.evidence_row_id.duplicated().any():
        raise ValueError("Duplicate observation identities in bundle")
    if set(training.evidence_row_id) != set(identities.evidence_row_id):
        raise ValueError("Training/identity observation mismatch")
    for column in ("provenance_id", "evidence_source_id", "evidence_kind"):
        expected = training.evidence_row_id.map(identities.set_index("evidence_row_id")[column])
        if training[column].tolist() != expected.tolist():
            raise ValueError(f"Training/identity {column} mismatch")
    expected = set(identities.provenance_id) - {""}
    if set(contributors.provenance_id) != expected:
        raise ValueError("Training/contributor observation mismatch")
    snapshots = manifest["inputs"]["build_metadata"].get("provenance", {}).get("sources", {})
    if not set(contributors.source_dataset) <= set(snapshots):
        raise ValueError("Contributor source snapshot missing")
    if (
        len(training) != manifest["n_rows"]
        or _coverage(identities, contributors) != manifest["coverage"]
    ):
        raise ValueError("Training bundle counts/coverage mismatch")
    return manifest


def audit_training_bundles(partitions, *, policy="report_only") -> dict:
    """Audit named verified bundles, preserving their output hashes in the report."""
    frames, manifests, registries = {}, {}, []
    for name, directory in partitions.items():
        directory = Path(directory)
        manifests[name] = verify_training_bundle(directory)
        identities = pd.read_parquet(directory / FILES["identities"])
        training = pd.read_parquet(directory / FILES["training"], columns=["evidence_row_id"])
        # Use the complete identity context even for narrow projections, while
        # keeping mapping expansion multiplicity available to the audit.
        frames[name] = training[["evidence_row_id"]].merge(
            identities, on="evidence_row_id", validate="many_to_one"
        )
        registries.append(json.loads((directory / FILES["lineage"]).read_text()))
    if any(registry != registries[0] for registry in registries[1:]):
        raise ValueError(
            "Bundles use different lineage registries; export against one reviewed registry"
        )
    report = audit_splits(frames, policy=policy, registry=registries[0] if registries else None)
    report["bundles"] = {
        name: {
            "manifest": file_digest(Path(directory) / "manifest.json"),
            "artifacts": manifests[name]["artifacts"],
        }
        for name, directory in partitions.items()
    }
    return report
