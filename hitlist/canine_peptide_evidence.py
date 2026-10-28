"""Frozen, source-faithful canine bulk evidence from released ORFdata artifacts."""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from dataclasses import dataclass
from importlib.metadata import version
from itertools import islice
from pathlib import Path

from .canine_orfdata import _registry_identity
from .evidence_snapshot import snapshot_file
from .managed_assets import cache_bytes
from .provenance import file_digest

# Selecting this provider explicitly selects the larger, complete evidence artifact.
# Legacy providers retain their existing defaults. Acquisition remains opt-in.
DEFAULT_LIMITS = {
    "max_input_bytes": 8 * 2**30,
    "max_upstream_bytes": 32 * 2**30,
    "max_output_bytes": 32 * 2**30,
    "max_records": 100_000_000,
}


def _require_provider():
    if sys.version_info < (3, 11):
        raise RuntimeError("The orfdata_peptides provider requires Python 3.11+ and ORFdata 0.40+")
    from packaging.version import Version

    if Version(version("orfdata")) < Version("0.40.0"):
        raise RuntimeError("The orfdata_peptides provider requires ORFdata 0.40+")


def _check_release(source, manifest, download, reference):
    """Check the reviewed generation and its relationship to the original summary."""
    pin = source.get("orfdata_peptides")
    if source["adapter"] != "pfizer" or not pin:
        raise ValueError("Source has no reviewed orfdata_peptides provider")
    if (manifest["dataset_id"], manifest["version"]) != (
        pin["dataset_id"],
        pin["version"],
    ) or _registry_identity(manifest, download) != pin["registry_sha256"]:
        raise ValueError("orfdata peptide registry disagrees with the pinned release")
    refs = manifest["references"]
    expected = source["assets"]["reference"]
    if (
        len(refs) != 1
        or refs[0]["taxon_id"] != source["source_taxon"]
        or refs[0]["sequence_sha256"] != expected["sha256"]
        or reference["reference"] != refs[0]
        or reference["source"]["sha256"] != expected["sha256"]
        or reference["source"]["size"] != expected["size_bytes"]
        or reference["expected_sequences"] != manifest["coverage"]["search_sequences"]
        or reference["dataset_id"] != manifest["dataset_id"]
        or reference["dataset_version"] != manifest["version"]
    ):
        raise ValueError("orfdata peptide searched reference disagrees with source")
    baseline = manifest["processing"]["protein_summary"]
    if (
        baseline["source_sha256"] != source["assets"]["table"]["sha256"]
        or baseline["dataset_id"] != source["orfdata"]["dataset_id"]
        or baseline["version"] != source["orfdata"]["version"]
        or manifest["assay_kinds"] != ["digest_ms"]
    ):
        raise ValueError("orfdata peptide evidence disagrees with the same-study protein summary")


def _inventory(dataset, native):
    counts = dataset.counts
    return {
        "peptides": counts["measurement"],
        "mappings": counts["mapping"],
        "groups": counts["target_relation"],
        "samples": counts["sample"],
        "assays": counts["assay"],
        **{
            f"raw_{name}": count
            for name, count in native.metadata["reconciliation"]["counts"].items()
        },
    }


def _check_dataset(dataset, native, manifest, reference):
    if dataset.manifest.to_dict() != manifest:
        raise ValueError("Frozen orfdata peptide dataset manifest mismatch")
    if native.search_reference.to_dict() != reference:
        raise ValueError("Frozen orfdata peptide search reference mismatch")
    if native.metadata["reconciliation"]["counts"] != manifest["coverage"]["native_counts"]:
        raise ValueError("orfdata peptide native source counts mismatch")
    if native.metadata["protein_summary"] != manifest["processing"]["protein_summary"]:
        raise ValueError("orfdata peptide protein-summary reconciliation mismatch")
    if (
        native.metadata["attribution_counts"]["searched_proteins"]
        != reference["expected_sequences"]
    ):
        raise ValueError("orfdata peptide complete searched reference count mismatch")


def curate_orfdata_peptides(source, target, bounds, allow_download, cache_root):
    """Snapshot the complete released artifact; never extract vendor sources."""
    _require_provider()
    from orfdata import DatasetRegistry
    from orfdata.sources.pfizer_attribution import CanineEvidence

    from .version import __version__

    pin = source.get("orfdata_peptides")
    if source["adapter"] != "pfizer" or not pin:
        raise ValueError("Source has no reviewed orfdata_peptides provider")
    registry = DatasetRegistry(cache_root=cache_root)
    name, release = pin["dataset_id"], pin["version"]
    manifest = registry.info(name, release).to_dict()
    download = registry.download_metadata(name, release)
    (reference,) = registry.search_references(name, release)
    reference = reference.to_dict()
    _check_release(source, manifest, download, reference)
    artifact = download["assets"]["dataset.sqlite"]
    if artifact["size"] > bounds["max_upstream_bytes"]:
        raise ValueError("orfdata peptide installed artifact exceeds max_upstream_bytes")
    if download["distribution"]["download_size"] > bounds["max_input_bytes"]:
        raise ValueError("orfdata peptide download exceeds max_input_bytes")
    if artifact["size"] + 2 * 2**20 > bounds["max_output_bytes"]:
        raise ValueError("orfdata peptide bundle exceeds max_output_bytes")
    if sum(manifest["coverage"]["native_counts"].values()) > bounds["max_records"]:
        raise ValueError("orfdata peptide native records exceed max_records")
    if allow_download:
        registry.download(name, release)
    dataset = registry.open(name, release)
    native = CanineEvidence(dataset.path)
    _check_dataset(dataset, native, manifest, reference)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}-", dir=target.parent))
    try:
        snapshot_file(dataset.path, staging / "dataset.sqlite")
        receipt = {
            "schema_version": 2,
            "kind": "canine_bulk",
            "hitlist_version": __version__,
            "source": source,
            "limits": bounds,
            "presentation_coverage": "missing",
            "n_records_by_table": _inventory(dataset, native),
            "artifacts": {
                "dataset.sqlite": {"sha256": artifact["sha256"], "size_bytes": artifact["size"]}
            },
            "upstream": {
                "provider": "orfdata_peptides",
                "package_version": version("orfdata"),
                "dataset_manifest": manifest,
                "download": download,
                "search_reference": reference,
                "native_metadata": native.metadata,
            },
        }
        (staging / "manifest.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        if cache_bytes(staging) > bounds["max_output_bytes"]:
            raise ValueError("orfdata peptide bundle exceeds max_output_bytes")
        verify_peptide_evidence(staging)
        # Refuse to replace an output that appeared during the snapshot.
        target.mkdir()
        try:
            for name in ("dataset.sqlite", "manifest.json"):
                (staging / name).replace(target / name)
        except BaseException:
            shutil.rmtree(target)
            raise
        return target / "manifest.json"
    finally:
        shutil.rmtree(staging)


def _manifest(directory):
    manifest = json.loads((Path(directory) / "manifest.json").read_text())
    if (
        manifest.get("kind") != "canine_bulk"
        or manifest.get("schema_version") != 2
        or manifest.get("presentation_coverage") != "missing"
        or manifest.get("upstream", {}).get("provider") != "orfdata_peptides"
        or set(manifest.get("artifacts", {})) != {"dataset.sqlite"}
    ):
        raise ValueError("Unsupported canine peptide evidence schema/scope")
    upstream = manifest["upstream"]
    _check_release(
        manifest["source"],
        upstream["dataset_manifest"],
        upstream["download"],
        upstream["search_reference"],
    )
    asset = upstream["download"]["assets"]["dataset.sqlite"]
    if manifest["artifacts"]["dataset.sqlite"] != {
        "sha256": asset["sha256"],
        "size_bytes": asset["size"],
    }:
        raise ValueError("Frozen canine peptide artifact differs from its release pin")
    return manifest


@dataclass(frozen=True)
class CaninePeptideEvidence:
    """Common streaming queries plus explicit native records, controls and definitions.

    ``dataset`` exposes ORFdata measurements/mappings/relations. ``native`` exposes
    its documented CanineEvidence API. Neither acquires files or infers DLA labels.
    """

    manifest: dict
    dataset: object
    native: object


def open_canine_peptide_evidence(directory, *, verify=True):
    """Open a frozen schema-2 bundle offline, verifying its full bytes by default.

    Explicit ``verify=False`` avoids rehashing a previously verified immutable
    bundle. It still checks frozen release identity, source context and counts.
    No current registry or separately installed search FASTA is needed.
    """
    _require_provider()
    from orfdata import QuantitativeDataset
    from orfdata.sources.pfizer_attribution import CanineEvidence

    root = Path(directory)
    manifest = _manifest(root)
    if verify and file_digest(root / "dataset.sqlite") != manifest["artifacts"]["dataset.sqlite"]:
        raise ValueError("Bulk evidence artifact mismatch: dataset.sqlite")
    dataset = QuantitativeDataset(root / "dataset.sqlite")
    native = CanineEvidence(dataset.path)
    upstream = manifest["upstream"]
    _check_dataset(dataset, native, upstream["dataset_manifest"], upstream["search_reference"])
    if (
        native.metadata != upstream["native_metadata"]
        or _inventory(dataset, native) != manifest["n_records_by_table"]
    ):
        raise ValueError("Frozen canine peptide evidence reconciliation mismatch")
    return CaninePeptideEvidence(manifest, dataset, native)


def verify_peptide_evidence(directory):
    return open_canine_peptide_evidence(directory).manifest


def _rows(evidence, table, tissue, peptide):
    from orfdata import ScopedTarget

    dataset, native = evidence.dataset, evidence.native
    samples = {sample.sample_id: sample for sample in dataset.samples()}
    selected = {
        key
        for key, sample in samples.items()
        if tissue is None or (sample.tissue or "").casefold() == tissue.casefold()
    }
    if table == "peptides":
        (reference,) = dataset.manifest.references
        target = (
            ScopedTarget(reference, "peptide", peptide.upper(), "amino-acid-sequence")
            if peptide
            else None
        )
        queries = ({"sample_id": key} for key in sorted(selected)) if tissue else ({},)
        for query in queries:
            for value in dataset.measurements(target=target, **query):
                row = value.to_dict()
                sample = samples[value.sample_id]
                role = sample.attributes.get("role")
                row.update(
                    peptide=value.target.identifier,
                    source_tissue=sample.tissue,
                    sample_role=role,
                    donor_id=sample.attributes.get("donor_id"),
                    donor_status="pooled" if role == "tissue_pool" else "not_applicable",
                    sample_context=sample.to_dict(),
                    peptide_modifications_raw=value.provenance.get("native_modifications"),
                )
                yield row
    elif table in ("samples", "assays"):
        if peptide:
            raise ValueError(f"Table {table!r} does not support a peptide filter")
        for value in getattr(dataset, table)():
            if value.sample_id in selected:
                yield {**value.to_dict(), "source_tissue": samples[value.sample_id].tissue}
    else:
        if tissue:
            raise ValueError(
                f"Table {table!r} does not support a tissue filter; use assay/sample links"
            )
        if table == "mappings":
            for value in dataset.mappings(source_identifier=peptide.upper() if peptide else None):
                yield value.to_dict()
        elif table == "groups":
            if peptide:
                raise ValueError("Group relations use protein targets; use dataset.relations()")
            for value in dataset.relations():
                yield value.to_dict()
        elif table.startswith("raw_") and table[4:] in native.tables:
            name = table[4:]
            if peptide and "Sequence" not in native.tables[name][1]:
                raise ValueError(f"Table {table!r} does not support a peptide filter")
            yield from native.records(name, sequence=peptide.upper() if peptide else None)
        else:
            raise ValueError(f"Unknown evidence table {table!r}")


def load_peptide_evidence(directory, *, table, tissue, peptide, max_rows):
    """Bounded dataframe convenience; the reader's dataset API streams large queries."""
    import pandas as pd

    for name, value in (("tissue", tissue), ("peptide", peptide)):
        if value is not None and (not isinstance(value, str) or not value):
            raise ValueError(f"{name} must be a nonempty string")
    evidence = open_canine_peptide_evidence(directory, verify=False)
    rows = list(islice(_rows(evidence, table, tissue, peptide), max_rows + 1))
    if len(rows) > max_rows:
        raise ValueError("Selection exceeds max_rows; narrow the query or use the streaming reader")
    frame = pd.DataFrame(rows)
    source = evidence.manifest["source"]
    # Native tables are exact source cells, including decoys and opaque vector bytes.
    if not table.startswith("raw_"):
        for key, value in {
            "source_id": source["source_id"],
            "source_taxon": source["source_taxon"],
            "source": "bulk_proteomics",
            "evidence_kind": "bulk",
            "assay_modality": "bulk_proteomics",
            "is_ms_observation": False,
            "mhc_restriction": "",
            "instrument": source["instrument"],
        }.items():
            frame[key] = value
    return frame
