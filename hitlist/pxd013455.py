"""Reproduce the pinned PXD013455 HeLa search scope from public source bytes."""

import csv
import hashlib
import json
import math
import shutil
import xml.etree.ElementTree as ET
import zipfile
import zlib
from collections import Counter, defaultdict
from pathlib import Path, PureWindowsPath

import datacache
import pandas as pd
import yaml

from .detectability import DetectabilitySearchSpace
from .downloads import data_dir
from .provenance import file_digest

CATALOG = Path(__file__).parent / "data" / "pxd013455.yaml"


def _catalog():
    with CATALOG.open() as handle:
        return yaml.safe_load(handle)


def recover_pxd013455_sources(*, data_root=None):
    """Acquire and hash-verify source files in the shared datacache namespace.

    The member stores retain discovery receipts. Independent catalog pins are
    checked with datacache.validate_file on every call, so discovery artifacts
    can be verified and reused without downloading their large bytes again.
    Nothing is admitted to a curated build before those checks succeed.
    """
    catalog = _catalog()
    root = Path(data_root or data_dir()) / "scoped-proteomics" / "pxd013455-hela-tryp46"
    paths = {}
    for name, spec in {**catalog["files"], **catalog["supporting_files"]}.items():
        paths[name] = datacache.fetch_file(
            spec["url"],
            destination=root / "source-files" / spec.get("original_name", name),
            raw=True,
            expected_sha256=spec["sha256"],
            expected_size=spec["size_bytes"],
            max_bytes=spec["size_bytes"],
            record_provenance=True,
        )
    for name, spec in catalog["members"].items():
        member = datacache.install_zip_member(
            root / "result-members" / name,
            spec["archive_url"],
            spec["member"],
            expected_size=spec["size_bytes"],
            max_bytes=spec["size_bytes"],
            max_compressed_bytes=spec["compressed_size_bytes"],
        )
        datacache.validate_file(member["payload"], spec["sha256"], spec["size_bytes"])
        datacache.validate_file(
            member["compressed"], spec["compressed_sha256"], spec["compressed_size_bytes"]
        )
        receipt = json.loads(Path(member["receipt.json"]).read_text())
        if receipt["crc32"] != spec["crc32"]:
            raise ValueError(f"Unexpected {name} CRC32 receipt")
        paths[f"{name}.txt"] = member["payload"]
        paths[f"{name}.deflate"] = member["compressed"]
        paths[f"{name}.receipt.json"] = member["receipt.json"]

    tables = catalog["supplement_tables"]

    def extract(inputs, outputs):
        for name, path in outputs.items():
            with zipfile.ZipFile(inputs[tables[name]["archive"]]) as archive:
                info = archive.getinfo(name)
                if info.file_size != tables[name]["size_bytes"]:
                    raise ValueError(f"Unexpected supplement member size: {name}")
                with archive.open(info) as source, open(path, "xb") as target:
                    shutil.copyfileobj(source, target)

    archives = {value["archive"] for value in tables.values()}
    paths.update(
        datacache.materialize(
            root / "paper-supplements" / "selected-tables",
            {
                name: {
                    "path": paths[name],
                    "sha256": catalog["supporting_files"][name]["sha256"],
                    "size": catalog["supporting_files"][name]["size_bytes"],
                }
                for name in archives
            },
            transform={"version": "pxd013455-selected-supplements-v1", "options": tables},
            outputs={
                name: {"sha256": spec["sha256"], "size": spec["size_bytes"]}
                for name, spec in tables.items()
            },
            builder=extract,
        )
    )
    return paths


def _deflate_lines(path, spec, *, max_line_bytes=8 * 2**20):
    """Read the complete retained stream with bounded expansion and line size."""
    digest = hashlib.sha256()
    decompressor = zlib.decompressobj(-15)
    pending = b""
    n_bytes = crc = 0
    with open(path, "rb") as handle:
        while compressed := handle.read(64 * 1024):
            while True:
                chunk = decompressor.decompress(compressed, 2**20)
                compressed = decompressor.unconsumed_tail
                if decompressor.unused_data:
                    raise ValueError("Trailing bytes in peptide deflate stream")
                n_bytes += len(chunk)
                if n_bytes > spec["size_bytes"]:
                    raise ValueError("Peptide stream exceeds pinned expanded size")
                digest.update(chunk)
                crc = zlib.crc32(chunk, crc)
                lines = (pending + chunk).split(b"\n")
                pending = lines.pop()
                for line in lines:
                    if len(line) > max_line_bytes:
                        raise ValueError("Peptide source line exceeds max_line_bytes")
                    yield line.decode("utf-8") + "\n"
                if len(pending) > max_line_bytes:
                    raise ValueError("Peptide source line exceeds max_line_bytes")
                # zlib can consume the final input while retaining output at
                # the expansion limit. Drain that bounded output with b"".
                if not compressed and len(chunk) < 2**20:
                    break
        if pending:
            yield pending.decode("utf-8")
    if (
        not decompressor.eof
        or n_bytes != spec["size_bytes"]
        or digest.hexdigest() != spec["sha256"]
        or f"{crc:08x}" != spec["crc32"]
    ):
        raise ValueError("Incomplete or corrupt expanded peptide source")


def _protocol(paths, catalog):
    """Cross-check effective results, XML and design without inferring replicates."""
    experiment = catalog["experiment"]
    with open(paths["parameters.txt"], newline="") as handle:
        parameters = dict(list(csv.reader(handle, delimiter="\t"))[1:])
    for name, expected in {
        "Version": "1.6.0.16",
        "Fixed modifications": "Carbamidomethyl (C)",
        "Min. peptide Length": "7",
        "Max. peptide mass [Da]": "4600",
        "Match between runs": "True",
    }.items():
        if parameters.get(name) != expected:
            raise ValueError(f"Effective search parameter mismatch: {name}")
    if (
        PureWindowsPath(parameters["Fasta file"]).name
        != catalog["files"]["searched.fasta"]["original_name"]
    ):
        raise ValueError("Effective results name a different search FASTA")
    with open(paths["summary.txt"], newline="") as handle:
        summary = list(csv.DictReader(handle, delimiter="\t"))
    selected = [row for row in summary if row["Experiment"] == experiment]
    expected = catalog["expected_counts"]
    if len(selected) != expected["n_raw_files"]:
        raise ValueError("Unexpected number of effective search runs")
    membership = {(row["Raw file"], row["Fraction"]) for row in selected}
    if (
        len(membership) != len(selected)
        or len({row["Fraction"] for row in selected}) != catalog["scope"]["n_fractions_in_run"]
    ):
        raise ValueError("Ambiguous selected raw-file or fraction membership")
    for row in selected:
        if (
            row["Enzyme"] != catalog["search_space"]["enzyme"]
            or int(row["Max. missed cleavages"]) != catalog["search_space"]["max_missed_cleavages"]
            or row["Enzyme mode"] != "Specific"
            or row["LC-MS run type"] != "Standard"
        ):
            raise ValueError("Effective selected-run digestion settings disagree")
    xml = ET.parse(paths["mqpar-celllines.xml"]).getroot()
    columns = [
        [child.text for child in xml.find(tag)]
        for tag in ("filePaths", "experiments", "fractions", "paramGroupIndices")
    ]
    if len({len(column) for column in columns}) != 1:
        raise ValueError("XML run metadata columns have unequal lengths")
    xml_rows = [row for row in zip(*columns) if row[1] == experiment]
    if {(PureWindowsPath(row[0]).stem, row[2]) for row in xml_rows} != membership or len(
        xml_rows
    ) != len(selected):
        raise ValueError("Effective results and standalone XML membership disagree")
    groups = list(xml.find("parameterGroups"))
    for row in xml_rows:
        group = groups[int(row[3])]
        if [item.text for item in group.find("enzymes")] != [
            catalog["search_space"]["enzyme"]
        ] or int(group.findtext("maxMissedCleavages")) != catalog["search_space"][
            "max_missed_cleavages"
        ]:
            raise ValueError("Effective digestion settings disagree with standalone XML")
    with open(paths["experimentalDesignTemplate-celllines.txt"], newline="") as handle:
        design = [
            row for row in csv.DictReader(handle, delimiter="\t") if row["Experiment"] == experiment
        ]
    if {(row["Name"], row["Fraction"]) for row in design} != membership or len(design) != len(
        selected
    ):
        raise ValueError("Experimental design and effective results membership disagree")
    return selected, {
        "effective_maxquant_version": parameters["Version"],
        "standalone_xml_maxquant_version": xml.findtext("maxQuantVersion"),
        "n_raw_files": len(selected),
        "n_fractions": len({row["Fraction"] for row in selected}),
        "biological_replicate_identity": "unresolved",
    }


def _observations(compressed_path, catalog):
    """Project one scope and retain original measured row identities and flags."""
    scope = catalog["experiment"]
    spec = catalog["members"]["peptides"]
    old_limit = csv.field_size_limit(8 * 2**20)
    counts = Counter()
    observations = []
    parent_counts = Counter()
    parent_intensities = defaultdict(float)
    seen_ids, seen_sequences = set(), set()
    try:
        reader = csv.DictReader(_deflate_lines(compressed_path, spec), delimiter="\t")
        required = {
            "Sequence",
            "Leading razor protein",
            "Proteins",
            "Start position",
            "End position",
            "Gene names",
            "id",
            "Reverse",
            "Potential contaminant",
            "PEP",
            "Score",
            f"Intensity {scope}",
            f"Identification type {scope}",
        }
        if (
            not reader.fieldnames
            or not required.issubset(reader.fieldnames)
            or len(set(reader.fieldnames)) != len(reader.fieldnames)
        ):
            raise ValueError("Peptide source is missing required or has duplicate columns")
        for row in reader:
            counts["n_source_rows"] += 1
            if None in row or any(value is None for value in row.values()):
                raise ValueError("Malformed peptide source row")
            intensity = float(row[f"Intensity {scope}"] or 0)
            if not math.isfinite(intensity) or intensity < 0:
                raise ValueError("Invalid source intensity")
            if intensity == 0:
                continue
            counts["n_positive_intensity_rows"] += 1
            if any(row[flag] not in ("", "+") for flag in ("Reverse", "Potential contaminant")):
                raise ValueError("Unrecognized source exclusion flag")
            if row["Reverse"] == "+" or row["Potential contaminant"] == "+":
                counts["n_reverse_or_contaminant_rows"] += 1
                continue
            peptide, parent, source_id = row["Sequence"], row["Leading razor protein"], row["id"]
            if (
                not parent
                or ";" in parent
                or not source_id
                or source_id in seen_ids
                or peptide in seen_sequences
            ):
                raise ValueError("Ambiguous observation or leading-razor parent identity")
            if not peptide or set(peptide) - set("ACDEFGHIKLMNPQRSTVWY"):
                raise ValueError("Nonstandard observed sequence needs explicit curation")
            seen_ids.add(source_id)
            seen_sequences.add(peptide)
            identification = row[f"Identification type {scope}"]
            if identification not in ("By MS/MS", "By matching"):
                raise ValueError("Unknown scope identification type")
            counts[
                "n_msms_observations" if identification == "By MS/MS" else "n_matching_observations"
            ] += 1
            parent_counts[parent] += 1
            parent_intensities[parent] += intensity
            observations.append(
                {
                    "peptide": peptide,
                    "uniprot_acc": parent,
                    "evidence_row_id": f"PXD013455:txt-celllines.zip:peptides:{source_id}",
                    "source_table_row_id": source_id,
                    "source_file_sha256": spec["sha256"],
                    "reported_proteins": row["Proteins"],
                    "reported_gene_names": row["Gene names"],
                    "reported_start_position": row["Start position"],
                    "reported_end_position": row["End position"],
                    "identification_type": identification,
                    "source_intensity": intensity,
                    "source_intensity_units": "MaxQuant intensity (arbitrary units)",
                    "search_global_pep": row["PEP"],
                    "search_global_score": row["Score"],
                    "source_experiment_id": scope,
                }
            )
    finally:
        csv.field_size_limit(old_limit)
    counts["n_observations"] = len(observations)
    counts["n_observed_parents"] = len(parent_counts)
    for name in (
        "n_source_rows",
        "n_positive_intensity_rows",
        "n_reverse_or_contaminant_rows",
        "n_observations",
        "n_msms_observations",
        "n_matching_observations",
        "n_observed_parents",
    ):
        value = counts[name]
        if value != catalog["expected_counts"][name]:
            raise ValueError(f"Unexpected {name}: {value}")
    proteins = pd.DataFrame(
        [
            {
                "uniprot_acc": parent,
                "n_peptides": n_peptides,
                "assigned_intensity_sum": parent_intensities[parent],
            }
            for parent, n_peptides in sorted(parent_counts.items())
        ]
    )
    proteins["abundance_percentile"] = proteins.assigned_intensity_sum.rank(pct=True)
    frame = pd.DataFrame(observations)
    contract = DetectabilitySearchSpace.read(catalog["search_space"])
    for table in (frame, proteins):
        for name, value in {
            **catalog["scope"],
            **catalog["acquisition"],
            "search_space_id": contract.identifier,
            "search_enzyme": contract.enzyme,
        }.items():
            table[name] = value
    frame["n_replicates_detected"] = pd.Series(pd.NA, index=frame.index, dtype="Int64")
    frame["n_replicates_possible"] = 1
    frame["replicate_count_status"] = "biological_replicate_identity_unresolved"
    return frame, proteins, dict(counts)


def build_pxd013455_observations(*, data_root=None, force=False):
    """Publish a reproducible, human-inspectable scoped observation generation."""
    catalog = _catalog()
    paths = recover_pxd013455_sources(data_root=data_root)
    root = Path(data_root or data_dir()) / "scoped-proteomics" / "pxd013455-hela-tryp46"
    # Compressed peptide bytes are the dependency: avoid another 3 GB expanded copy.
    copied_names = [
        *catalog["files"],
        *catalog["supplement_tables"],
        "parameters.txt",
        "summary.txt",
        *(f"{name}.receipt.json" for name in catalog["members"]),
    ]
    names = [*copied_names, "peptides.deflate"]
    sources = {
        name: {
            "path": paths[name],
            "sha256": file_digest(paths[name])["sha256"],
            "size": Path(paths[name]).stat().st_size,
        }
        for name in names
    }
    sources["catalog.yaml"] = {"path": str(CATALOG), "sha256": file_digest(CATALOG)["sha256"]}
    output_names = [
        "observations.parquet",
        "proteins.parquet",
        "raw_runs.tsv",
        "coverage.json",
        "source_registry.json",
        "search_space.json",
        *copied_names,
    ]

    def builder(inputs, outputs):
        runs, protocol = _protocol(inputs, catalog)
        frame, proteins, counts = _observations(inputs["peptides.deflate"], catalog)
        frame.to_parquet(outputs["observations.parquet"], index=False)
        proteins.to_parquet(outputs["proteins.parquet"], index=False)
        pd.DataFrame(runs).to_csv(outputs["raw_runs.tsv"], sep="\t", index=False)
        for name, value in {
            "coverage.json": {**counts, "protocol": protocol},
            "source_registry.json": catalog,
            "search_space.json": catalog["search_space"],
        }.items():
            Path(outputs[name]).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
        for name in copied_names:
            shutil.copyfile(inputs[name], outputs[name])

    return datacache.materialize(
        root / "observations",
        sources,
        transform={
            "version": "pxd013455-observations-v1",
            "options": {"recipe_sha256": file_digest(Path(__file__))["sha256"]},
        },
        outputs={name: {} for name in output_names},
        builder=builder,
        force=force,
    )
