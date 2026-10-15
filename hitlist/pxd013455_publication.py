"""Audited, portable publication of the recovered PXD013455 search scope."""

import gzip
import hashlib
import json
import shutil
import tarfile
from pathlib import Path

import datacache
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .detectability import (
    _RESIDUE_MASS,
    DetectabilitySearchSpace,
    _fasta_records,
    export_detectability_training_set,
)
from .downloads import data_dir
from .provenance import file_digest
from .pxd013455 import build_pxd013455_observations
from .version import __version__


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _sources(paths):
    return {
        name: {
            "path": path,
            "sha256": file_digest(path)["sha256"],
            "size": Path(path).stat().st_size,
        }
        for name, path in paths.items()
    }


def _candidate_id(catalog, row):
    return f"{catalog['dataset_id']}:{row.uniprot_acc}:{row.start_position}:{row.end_position}"


def _attach_lineage(source, destination, observations, catalog):
    evidence = observations.set_index("peptide")
    with pq.ParquetFile(source) as reader:
        writer = None
        try:
            for batch in reader.iter_batches(batch_size=25000):
                frame = batch.to_pandas()
                frame["candidate_id"] = [_candidate_id(catalog, row) for row in frame.itertuples()]
                frame["source_observation_id"] = frame.peptide.map(evidence.evidence_row_id).astype(
                    "string"
                )
                frame["label_evidence"] = frame.peptide.map(evidence.identification_type).fillna(
                    "not observed in scope"
                )
                table = pa.Table.from_pandas(frame, preserve_index=False).replace_schema_metadata(
                    None
                )
                if writer is None:
                    writer = pq.ParquetWriter(destination, table.schema, compression="zstd")
                writer.write_table(table)
        finally:
            if writer is not None:
                writer.close()


def audit_pxd013455_candidates(directory):
    """Check every exported occurrence independently of the digestion iterator.

    This audit uses sequence coordinates and direct K/R boundary checks rather
    than invoking the candidate generator again. The pinned source census is a
    separate completeness check. It does not establish biological independence.
    """
    root = Path(directory)
    catalog = json.loads((root / "source_registry.json").read_text())
    observations = pd.read_parquet(root / "observations.parquet")
    parents = pd.read_parquet(root / "proteins.parquet")
    expected = catalog["expected_counts"]
    contract = DetectabilitySearchSpace.read(catalog["search_space"])
    datacache.validate_file(root / "searched.fasta", contract.fasta_sha256)
    if (
        len(observations) != expected["n_observations"]
        or len(parents) != expected["n_observed_parents"]
    ):
        raise ValueError("Source observation/parent census mismatch")
    reference = {
        accession: sequence
        for accession, _, sequence in _fasta_records(
            root / "searched.fasta", max_protein_residues=1000000, max_reference_bytes=256 * 2**20
        )
    }
    if len(reference) != expected["n_reference_sequences"]:
        raise ValueError("Reference inventory differs from the pinned census")
    if observations.peptide.duplicated().any() or observations.evidence_row_id.duplicated().any():
        raise ValueError("Duplicate source observation identity")
    evidence = observations.set_index("peptide").to_dict("index")
    parent_ids = set(parents.uniprot_acc)
    if not parent_ids <= reference.keys() or parent_ids != set(observations.uniprot_acc):
        raise ValueError("Observed parent inventory disagrees with source/reference")
    for row in observations.itertuples():
        if row.peptide not in reference[row.uniprot_acc]:
            raise ValueError("Source peptide absent from its reported leading-razor parent")
    candidate_ids, observed_sequences = set(), set()
    n_candidates = n_observed = 0
    with pq.ParquetFile(root / "candidates.parquet") as reader:
        for batch in reader.iter_batches(batch_size=25000):
            for row in batch.to_pandas().itertuples():
                n_candidates += 1
                sequence = reference[row.uniprot_acc]
                start, end = row.start_position - 1, row.end_position
                if row.uniprot_acc not in parent_ids or not row.protein_observed:
                    raise ValueError("Candidate lacks same-scope observed parent")
                if (
                    not 0 <= start < end <= len(sequence)
                    or sequence[start:end] != row.peptide
                    or not 7 <= len(row.peptide) <= 30
                    or (start > 0 and sequence[start - 1] not in "KR")
                    or (end < len(sequence) and sequence[end - 1] not in "KR")
                    or row.n_missed_cleavages != sum(aa in "KR" for aa in row.peptide[:-1])
                    or row.n_missed_cleavages > 2
                ):
                    raise ValueError("Candidate coordinates or search digestion are inconsistent")
                if (
                    row.n_flank != sequence[max(0, start - 15) : start]
                    or row.c_flank != sequence[end : end + 15]
                ):
                    raise ValueError("Candidate flank does not match the searched reference")
                mass = 18.010565 + sum(
                    _RESIDUE_MASS[aa] + contract.fixed_residue_modifications.get(aa, 0)
                    for aa in row.peptide
                )
                if (
                    not 0 < row.peptide_mass_da <= catalog["search_space"]["max_peptide_mass_da"]
                    or abs(row.peptide_mass_da - mass) > 1e-6
                ):
                    raise ValueError("Candidate is outside the searched mass window")
                for key, value in {
                    **catalog["scope"],
                    **catalog["acquisition"],
                    "search_space_id": contract.identifier,
                }.items():
                    if getattr(row, key) != value:
                        raise ValueError(f"Candidate has conflicting scope: {key}")
                candidate_id = _candidate_id(catalog, row)
                if row.candidate_id != candidate_id or candidate_id in candidate_ids:
                    raise ValueError("Duplicate or unstable candidate identity")
                candidate_ids.add(candidate_id)
                original = evidence.get(row.peptide)
                if bool(row.observed) != (original is not None):
                    raise ValueError(
                        "Candidate presence disagrees with scope-wide sequence evidence"
                    )
                if original is not None:
                    n_observed += 1
                    observed_sequences.add(row.peptide)
                    if (
                        pd.isna(row.source_observation_id)
                        or row.source_observation_id != original["evidence_row_id"]
                        or row.label_evidence != original["identification_type"]
                        or not pd.isna(row.n_replicates_detected)
                    ):
                        raise ValueError(
                            "Observed candidate lost its evidence or invented replication"
                        )
                elif (
                    not pd.isna(row.source_observation_id)
                    or row.label_evidence != "not observed in scope"
                    or row.n_replicates_detected != 0
                ):
                    raise ValueError("Unobserved candidate has invented measurement evidence")
                if not pd.isna(row.first_seen_at_n_fractions) or row.n_replicates_possible != 1:
                    raise ValueError("Unsupported graded detection label")
    counts = {
        "n_candidate_occurrences": n_candidates,
        "n_observed_candidate_occurrences": n_observed,
        "n_distinct_observed_candidate_sequences": len(observed_sequences),
    }
    for key, value in counts.items():
        if value != expected[key]:
            raise ValueError(f"Candidate census mismatch: {key}={value}, expected {expected[key]}")
    return {
        "status": "passed",
        "checks": [
            "every occurrence coordinate and flank",
            "Trypsin/P boundaries and missed cleavages",
            "search length and mass limits",
            "source-wide sequence presence",
            "positive source observation linkage",
            "unique candidate identities",
            "nullable depth and biological replication",
            "independently recorded source and candidate census",
        ],
        **counts,
        "n_reference_sequences": len(reference),
        "n_observations_outside_candidate_window": len(evidence) - len(observed_sequences),
        "partition_status": "not_assigned",
        "leakage_audit_status": "requires_actual_partitions",
        "biological_independence": "unresolved",
        "training_readiness": "not_established",
    }


def _readme(catalog, audit):
    return f"""# PXD013455 HeLa, 46-fraction Trypsin/P search scope

This is a reconstruction of the Jarnuczak et al. reanalysis of the
Bekker-Jensen et al. experiment. It is not the original PXD004452 search output.

- Original: PMID 28601559, https://doi.org/10.1016/j.cels.2017.05.009
- Reanalysis: PMID 33893311, https://doi.org/10.1038/s41597-021-00890-2
- Exact experiment: `{catalog["experiment"]}`
- Cell context: cultured HeLa epithelial cervical carcinoma; see source registry
  for paper-level conditions, authentication limitations and supplement cells.
- Assay: label-free shotgun LC-MS/MS; MHC allele sets are not applicable.
- Search: effective MaxQuant 1.6.0.16, Trypsin/P, two missed cleavages,
  minimum length 7, maximum mass 4,600 Da, fixed carbamidomethyl-C.
- Fractionation: 46 raw files and 46 fractions. Reported buffer C pH is 8;
  this corrects the prior unsupported pH 10 default, and is not measured eluate pH.

## Inspect the data

| File | Purpose |
| --- | --- |
| observations.parquet | 172,033 retained source sequences, source IDs, original intensities and identification evidence |
| proteins.parquet | 12,287 observed leading-razor parents and within-scope abundance proxy |
| candidates.parquet | {audit["n_candidate_occurrences"]:,} eligible occurrences, coordinates, flanks, labels and observation links |
| source_registry.json | Citable sources, exact byte pins, cell context, search settings and limitations |
| raw_runs.tsv | Every selected raw-file and fraction assignment from effective results |
| *.xlsx | Original supplementary tables used for biological and scope annotations |
| searched.fasta | Exact searched reference: 93,588 accessions including isoforms |
| parameters.txt / summary.txt / mqpar-celllines.xml | Effective results and deposited configuration, including their differing version reports |
| *.receipt.json | HTTP ZIP acquisition receipts with archive validators and member hashes |
| label_policy.json | Endpoint, units, evidence, absence and missing-label policy |
| coverage.json / validation.json | Source census and independent all-occurrence audit |
| manifest.json | SHA-256 and byte inventory for every artifact |

Join positive candidates' `source_observation_id` to observations' `evidence_row_id`.
Candidate IDs identify reference occurrences; shared observed sequences remain
positive at alternative parent occurrences. These links are audit metadata,
not model features. Intensities are arbitrary-unit source measurements, not
calibrated probabilities. Protein abundance is a rank of assigned intensity sums.

## Labels and limitations

The endpoint is observed in this selected search scope, conditional on an observed
leading-razor parent. Zero means not observed in this scope. It does not establish
intrinsic undetectability. Positive evidence includes 150,855 MS/MS identifications
and 21,178 matching identifications; `label_evidence` preserves the distinction.
{audit["n_distinct_observed_candidate_sequences"]:,} observed sequences occur in the
7-30 residue candidate window. Source observations outside it remain in their table.

Biological specimen, donor and replicate identity are unresolved. One detection
opportunity is not a biological replicate count. Depth and graded reproducibility
labels are missing. No partitions or held-out independence claim are supplied by
this single-scope publication; Presto must freeze and audit actual membership.
Training readiness and prospective/cross-tissue performance remain unestablished.

## Reproduce and reuse

Use `hitlist.build_pxd013455_bundle()` to recover pinned public sources and rebuild
the complete generation under the shared Hitlist datacache directory. Use
`hitlist.export_pxd013455_archive()` to create its portable archive. A forced
rebuild is explicit (`force=True`) and preserves previous published generations.
`hitlist.verify_pxd013455_bundle(path, expected_sha256=MANIFEST_SHA256)` verifies
the public inventory; `audit_pxd013455_candidates(path)` reruns the scientific audit.
Download/install the archive with `datacache.install_archive` and its published
SHA-256/size; never substitute a current FASTA or mutable live indexes.

The prior missing archive was not recovered. This publication has new hashes,
source receipts, lineage columns and corrected metadata. Counts matching the
earlier receipt do not imply identical bytes. See Hitlist #681, #688 and #678.
"""


def build_pxd013455_bundle(*, data_root=None, force=False):
    """Build, audit and atomically publish the complete scoped evidence bundle."""
    observed = build_pxd013455_observations(data_root=data_root, force=force)
    catalog = json.loads(Path(observed["source_registry.json"]).read_text())
    root = Path(data_root or data_dir()) / "scoped-proteomics" / "pxd013455-hela-tryp46"
    output_names = [
        *observed,
        "candidates.parquet",
        "candidates.manifest.json",
        "label_policy.json",
        "validation.json",
        "README.md",
        "manifest.json",
    ]

    def builder(inputs, outputs):
        directory = Path(outputs["manifest.json"]).parent
        for name in observed:
            shutil.copyfile(inputs[name], outputs[name])
        frame = pd.read_parquet(inputs["observations.parquet"])
        temporary = directory / "candidate-export"
        export_detectability_training_set(
            temporary,
            search_fasta=inputs["searched.fasta"],
            search_space=catalog["search_space"],
            peptide_observations=frame,
            protein_observations=pd.read_parquet(inputs["proteins.parquet"]),
            protocol_id=catalog["scope"]["protocol_id"],
            fractionation_ph=catalog["scope"]["fractionation_ph"],
        )
        _attach_lineage(
            temporary / "candidates.parquet", outputs["candidates.parquet"], frame, catalog
        )
        candidate_manifest = json.loads((temporary / "manifest.json").read_text())
        candidate_manifest["artifact"] = {
            "name": "candidates.parquet",
            **file_digest(outputs["candidates.parquet"]),
        }
        candidate_manifest["search_reference"]["path"] = "searched.fasta"
        candidate_manifest["source_linkage"] = (
            "source_observation_id joins observations.evidence_row_id by scope-wide peptide sequence"
        )
        _write_json(outputs["candidates.manifest.json"], candidate_manifest)
        shutil.rmtree(temporary)
        _write_json(
            outputs["label_policy.json"],
            {
                "version": 1,
                "endpoint": "observed_in_selected_search_scope",
                "units": "binary",
                "censoring": "not_applicable",
                "positive": "positive experiment intensity after reverse/contaminant exclusion",
                "negative": "unobserved candidate sequence within this search scope, conditional on observed parent",
                "matching_policy": "retain separately from direct MS/MS; no donor run inferred",
                "replicate_policy": "one aggregate detection opportunity; biological replication unresolved",
                "depth_policy": "missing; no comparable depth series established",
                "source_intensity_units": "MaxQuant arbitrary intensity units",
                "protein_abundance_proxy": "within-scope percentile of summed leading-razor assigned intensities",
                "candidate_window": {
                    "length": [7, 30],
                    "max_missed_cleavages": 2,
                    "flank_length": 15,
                },
                "limitations": catalog["limitations"],
            },
        )
        audit = audit_pxd013455_candidates(directory)
        _write_json(outputs["validation.json"], audit)
        Path(outputs["README.md"]).write_text(_readme(catalog, audit))
        _write_json(
            outputs["manifest.json"],
            {
                "schema_version": 1,
                "dataset_id": catalog["dataset_id"],
                "producer": {
                    "hitlist_version": __version__,
                    "publication_recipe_sha256": file_digest(Path(__file__))["sha256"],
                    "source_recipe_sha256": file_digest(Path(__file__).with_name("pxd013455.py"))[
                        "sha256"
                    ],
                    "pandas_version": pd.__version__,
                    "pyarrow_version": pa.__version__,
                },
                "artifacts": {
                    name: file_digest(path)
                    for name, path in outputs.items()
                    if name != "manifest.json"
                },
                "training_readiness": "not_established",
            },
        )

    return datacache.materialize(
        root / "publication",
        _sources(observed),
        transform={
            "version": "pxd013455-publication-v1",
            "options": {"recipe_sha256": file_digest(Path(__file__))["sha256"]},
        },
        outputs={name: {} for name in output_names},
        builder=builder,
        force=force,
    )


def verify_pxd013455_bundle(directory, *, expected_sha256=None):
    """Verify the frozen public inventory without source downloads or rebuilds."""
    root = Path(directory)
    if (root / "manifest.json").is_symlink():
        raise ValueError("Publication manifest must not be a symlink")
    manifest_path = root / "manifest.json"
    datacache.validate_file(manifest_path)
    with manifest_path.open("rb") as handle:
        payload = handle.read(2**20 + 1)
    if len(payload) > 2**20:
        raise ValueError("Publication manifest exceeds 1 MiB")
    digest = hashlib.sha256(payload).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256.lower():
        raise ValueError("Publication manifest does not match pinned SHA-256")
    manifest = json.loads(payload)
    required = {
        "candidates.parquet",
        "observations.parquet",
        "proteins.parquet",
        "searched.fasta",
        "source_registry.json",
        "label_policy.json",
        "validation.json",
        "coverage.json",
        "README.md",
    }
    if (
        manifest.get("schema_version") != 1
        or manifest.get("dataset_id") != "pxd013455-hela-tryp46-v1"
        or not required <= manifest.get("artifacts", {}).keys()
    ):
        raise ValueError("Incomplete or unsupported PXD013455 publication")
    for name, spec in manifest["artifacts"].items():
        if Path(name).name != name or name in ("", ".", "..", "manifest.json") or "\\" in name:
            raise ValueError("Invalid publication artifact path")
        if (root / name).is_symlink():
            raise ValueError("Publication artifacts must not be symlinks")
        datacache.validate_file(root / name, spec["sha256"], spec["size_bytes"])
    if file_digest(manifest_path)["sha256"] != digest:
        raise ValueError("Publication manifest changed during verification")
    return manifest


def export_pxd013455_archive(*, data_root=None, force=False):
    """Create a deterministic portable archive through datacache materialization."""
    paths = build_pxd013455_bundle(data_root=data_root, force=force)
    root = Path(data_root or data_dir()) / "scoped-proteomics" / "pxd013455-hela-tryp46"
    name = "pxd013455-hela-tryp46-v1.tar.gz"

    def builder(inputs, outputs):
        with (
            open(outputs[name], "wb") as raw,
            gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed,
            tarfile.open(fileobj=compressed, mode="w|") as archive,
        ):
            for filename, path in sorted(inputs.items()):
                info = archive.gettarinfo(path, arcname=filename)
                info.uid = info.gid = info.mtime = 0
                info.uname = info.gname = ""
                info.mode = 0o644
                with open(path, "rb") as source:
                    archive.addfile(info, source)

    return datacache.materialize(
        root / "portable",
        _sources(paths),
        transform={"version": "pxd013455-portable-v1"},
        outputs={name: {}},
        builder=builder,
        force=force,
    )[name]
