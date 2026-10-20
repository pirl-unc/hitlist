"""Audit source-file context for the frozen, donor-disjoint Atlas cohort.

The SDRF is a community annotation. A donor/tissue/class match is a candidate
context, not proof that a particular run passed QC or produced a peptide.
"""

import argparse
import csv
import gzip
import hashlib
import io
import json
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlsplit

from datacache import install_bundle

from hitlist.atlas import CATALOG_PATH as PRIMARY_CATALOG
from hitlist.atlas import atlas_sources
from hitlist.curation_yaml import load_curation_yaml
from hitlist.downloads import data_dir
from hitlist.provenance import file_digest

CATALOG = Path(__file__).resolve().parents[1] / "hitlist/data/atlas_community_crosswalk.yaml"
SEED = "atlas-donor-holdout-v1"
SDRF_CLASS = {
    "Human leukocyte antigen class I": "HLA-I",
    "Human leukocyte antigen class II": "HLA-II",
}
SOURCE_COLUMNS = (
    "source name",
    "characteristics[individual]",
    "characteristics[organism part]",
    "factor value[human leukocyte antigen]",
    "comment[data file]",
    "comment[file uri]",
    "comment[technical replicate]",
)


def pinned_sources(*, data_root=None):
    """Acquire the two complete, content-pinned community sources together."""
    catalog = load_curation_yaml(CATALOG)
    root = Path(data_root or data_dir())
    return install_bundle(
        root / "hla-ligand-atlas/2020.12/community-crosswalk",
        catalog["files"],
        download_options={"timeout": 60, "show_progress": False},
    )


def check_pins(paths, catalog_path=CATALOG):
    for name, spec in load_curation_yaml(catalog_path)["files"].items():
        path = Path(paths[name])
        digest = file_digest(path)
        if digest["sha256"] != spec["sha256"] or path.stat().st_size != spec["size"]:
            raise ValueError(f"Community source pin mismatch: {name}")


def check_primary_pin(path):
    spec = load_curation_yaml(PRIMARY_CATALOG)["files"]["sample_hits.tsv.gz"]
    digest = file_digest(path)
    if digest["sha256"] != spec["sha256"] or Path(path).stat().st_size != spec["size"]:
        raise ValueError("Primary Atlas sample-hit pin mismatch")


def key(donor, tissue, hla_class):
    # The release's tissue spellings differ only in case and whitespace from
    # the SDRF. Refuse collisions before using this candidate-context key.
    return donor, "".join(tissue.casefold().split()), hla_class


def read_primary_groups(path):
    counts = Counter()
    tissue_labels = defaultdict(set)
    with gzip.open(path, "rt", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            group = key(row["donor"], row["tissue"], row["hla_class"])
            counts[group] += 1
            tissue_labels[group].add(row["tissue"])
    if any(len(labels) != 1 for labels in tissue_labels.values()):
        raise ValueError("Primary Atlas tissue normalization collision")
    return counts, {group: next(iter(labels)) for group, labels in tissue_labels.items()}


def read_sdrf(path):
    with Path(path).open(newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if not set(SOURCE_COLUMNS) <= set(reader.fieldnames or []):
            raise ValueError("Unexpected community SDRF schema")
        return list(reader)


def read_inventory(path):
    data = json.loads(Path(path).read_text())
    if not isinstance(data, list):
        raise ValueError("Unexpected PRIDE inventory schema")
    names = [row["fileName"] for row in data]
    accessions = [row["accession"] for row in data]
    if len(set(names)) != len(names) or len(set(accessions)) != len(accessions):
        raise ValueError("Duplicate PRIDE inventory name or accession")
    return data


def donor_partitions(donors):
    ordered = sorted(
        donors,
        key=lambda donor: hashlib.sha256(
            (SEED + "\nhla-ligand-atlas:" + donor).encode()
        ).hexdigest(),
    )
    return {
        donor: "test" if i < 3 else "val" if i < 6 else "train" for i, donor in enumerate(ordered)
    }


def build_crosswalk(sdrf_rows, inventory, primary_counts, tissue_labels):
    inventory_by_name = {row["fileName"]: (i, row) for i, row in enumerate(inventory, 1)}
    if len(inventory_by_name) != len(inventory):
        raise ValueError("Duplicate PRIDE inventory filename")
    source_names = defaultdict(set)
    source_files = defaultdict(list)
    file_rows = []
    seen_files = set()
    for row_number, row in enumerate(sdrf_rows, 1):
        file_name = row["comment[data file]"]
        if file_name in seen_files:
            raise ValueError(f"Duplicate SDRF data filename: {file_name}")
        seen_files.add(file_name)
        if Path(urlsplit(row["comment[file uri]"]).path).name != file_name:
            raise ValueError(f"SDRF file URI disagrees with data file: {row_number}")
        if row["factor value[human leukocyte antigen]"] not in SDRF_CLASS:
            raise ValueError(f"Unknown SDRF HLA class: {row_number}")
        group = key(
            row["characteristics[individual]"],
            row["characteristics[organism part]"],
            SDRF_CLASS[row["factor value[human leukocyte antigen]"]],
        )
        name = row["source name"]
        source_names[name].add(group)
        source_files[group].append(file_name)
        candidate = inventory_by_name.get(file_name)
        if candidate is None or candidate[1]["fileCategory"]["value"] != "PEAK":
            raise ValueError(f"SDRF acquisition lacks unique PRIDE PEAK record: {file_name}")
        inventory_row, record = candidate
        raw_name = file_name.removesuffix(".mzML") + ".raw"
        raw = inventory_by_name.get(raw_name)
        raw_match = raw is not None and raw[1]["fileCategory"]["value"] == "RAW"
        file_rows.append(
            {
                "sdrf_row": row_number,
                **row,
                "inventory_row": inventory_row,
                "inventory_accession": record["accession"],
                "file_match_status": "exact_unique_peak",
                "raw_file_name": raw_name if raw_match else "",
                "raw_inventory_row": raw[0] if raw_match else "",
                "raw_match_status": "exact_stem" if raw_match else "unresolved",
                "primary_group_status": (
                    "candidate_context" if group in primary_counts else "no_primary_group"
                ),
            }
        )
    if any(len(groups) != 1 for groups in source_names.values()):
        raise ValueError("SDRF source name spans donor/tissue/class groups")
    # No source file may point to two donor groups, and no primary group is
    # assigned a run as its measured origin by this relationship.
    group_sources = defaultdict(set)
    for name, groups in source_names.items():
        group_sources[next(iter(groups))].add(name)
    partitions = donor_partitions({group[0] for group in primary_counts})
    group_rows = []
    for group, n_observations in sorted(primary_counts.items()):
        sources = sorted(group_sources[group])
        group_rows.append(
            {
                "donor": group[0],
                "tissue": tissue_labels[group],
                "hla_class": group[2],
                "partition": partitions[group[0]],
                "n_observations": n_observations,
                "n_candidate_source_names": len(sources),
                "n_candidate_peak_files": len(source_files[group]),
                "candidate_source_names": json.dumps(sources),
                "candidate_peak_files": json.dumps(sorted(source_files[group])),
                "candidate_status": (
                    "no_sdrf_context"
                    if not sources
                    else "multiple_source_names"
                    if len(sources) > 1
                    else "one_source_name"
                ),
                "qc_inclusion_status": "unknown",
                "observation_acquisition_status": "unknown",
                "specimen_status": "unknown",
            }
        )
    sdrf_index = {row["comment[data file]"]: row["sdrf_row"] for row in file_rows}
    inventory_rows = [
        {
            "inventory_row": i,
            "accession": row["accession"],
            "file_name": row["fileName"],
            "category": row["fileCategory"]["value"],
            "sdrf_row": sdrf_index.get(row["fileName"], ""),
            "sdrf_status": (
                "exact_data_file" if row["fileName"] in sdrf_index else "not_sdrf_data_file"
            ),
        }
        for i, row in enumerate(inventory, 1)
    ]
    coverage = defaultdict(Counter)
    donor_coverage = defaultdict(Counter)
    for row in group_rows:
        for bucket in (
            coverage[(row["partition"], row["hla_class"])],
            donor_coverage[(row["donor"], row["hla_class"])],
        ):
            bucket["n_observations"] += row["n_observations"]
            bucket["n_reported_sample_groups"] += 1
            bucket["n_candidate_context_observations"] += (
                row["n_observations"] if row["n_candidate_source_names"] else 0
            )
            bucket["n_multiple_source_name_observations"] += (
                row["n_observations"] if row["n_candidate_source_names"] > 1 else 0
            )
            bucket["n_unknown_qc_inclusion_observations"] += row["n_observations"]
            bucket["n_unknown_acquisition_observations"] += row["n_observations"]
            bucket["n_unknown_specimen_observations"] += row["n_observations"]
    file_partitions = defaultdict(set)
    for row in file_rows:
        group = key(
            row["characteristics[individual]"],
            row["characteristics[organism part]"],
            SDRF_CLASS[row["factor value[human leukocyte antigen]"]],
        )
        if group in primary_counts:
            file_partitions[row["comment[data file]"]].add(partitions[group[0]])
    summary = {
        "source_accession": "PXD019643",
        "n_sdrf_rows": len(file_rows),
        "n_sdrf_source_names": len(source_names),
        "n_sdrf_donor_tissue_class_groups": len(group_sources),
        "n_inventory_records": len(inventory_rows),
        "n_exact_peak_matches": len(file_rows),
        "n_exact_raw_stem_matches": sum(
            row["raw_match_status"] == "exact_stem" for row in file_rows
        ),
        "n_unresolved_raw_aliases": sum(
            row["raw_match_status"] == "unresolved" for row in file_rows
        ),
        "n_unannotated_peak_records": sum(
            row["category"] == "PEAK" and row["sdrf_status"] != "exact_data_file"
            for row in inventory_rows
        ),
        "n_inventory_result_records_without_qc_join": sum(
            row["category"] == "RESULT" for row in inventory_rows
        ),
        "n_primary_groups": len(group_rows),
        "n_primary_groups_with_candidate_context": sum(
            row["n_candidate_source_names"] > 0 for row in group_rows
        ),
        "n_primary_groups_with_multiple_source_names": sum(
            row["n_candidate_source_names"] > 1 for row in group_rows
        ),
        "n_candidate_peak_files_in_primary_groups": len(file_partitions),
        "n_candidate_peak_files_in_multiple_partitions": sum(
            len(splits) > 1 for splits in file_partitions.values()
        ),
        "coverage_by_partition_and_class": {
            f"{part}/{hla_class}": dict(counts)
            for (part, hla_class), counts in sorted(coverage.items())
        },
        "coverage_by_donor_and_class": {
            f"{donor}/{hla_class}": dict(counts)
            for (donor, hla_class), counts in sorted(donor_coverage.items())
        },
        "verdicts": {
            "sdrf_to_pride_peak_file": "pass",
            "primary_group_to_sdrf_context": (
                "complete"
                if all(row["n_candidate_source_names"] for row in group_rows)
                else "partial"
            ),
            "qc_included_run_to_primary_group": "inconclusive",
            "observation_to_acquisition": "inconclusive",
            "physical_specimen_identity": "inconclusive",
            "candidate_peak_file_partition_reuse": (
                "none_observed"
                if all(len(splits) == 1 for splits in file_partitions.values())
                else "observed"
            ),
        },
        "limits": [
            "The community SDRF and PRIDE inventory are separate from the primary Atlas release.",
            "A matching donor/tissue/class is contextual; it does not identify peptide-producing runs.",
            "The primary sample-hit table contains no source-name, result-file or run reference.",
            "PRIDE includes runs excluded by upstream QC; result-file inventory presence is not inclusion.",
            "Source names and technical replicate labels do not establish physical specimen identity.",
            "Candidate file sets are donor-disjoint in this Atlas cohort; actual acquisition reuse remains unknown.",
        ],
    }
    return file_rows, inventory_rows, group_rows, summary


def write_tsv_gz(path, rows):
    if not rows:
        raise ValueError("Crosswalk output cannot be empty")
    with (
        Path(path).open("wb") as raw,
        gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped,
        io.TextIOWrapper(zipped, encoding="utf-8", newline="") as text,
    ):
        writer = csv.DictWriter(text, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_snapshot(path, source):
    with (
        Path(path).open("wb") as raw,
        gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=9) as zipped,
        Path(source).open("rb") as stream,
    ):
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            zipped.write(chunk)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--community-dir", type=Path)
    parser.add_argument("--primary-dir", type=Path)
    args = parser.parse_args()
    community = (
        {name: args.community_dir / name for name in ("community.sdrf.tsv", "files-all.json")}
        if args.community_dir
        else pinned_sources(data_root=args.data_root)
    )
    check_pins(community)
    primary = (
        {"sample_hits.tsv.gz": args.primary_dir / "sample_hits.tsv.gz"}
        if args.primary_dir
        else atlas_sources(data_root=args.data_root)
    )
    check_primary_pin(primary["sample_hits.tsv.gz"])
    counts, labels = read_primary_groups(primary["sample_hits.tsv.gz"])
    files, inventory, groups, summary = build_crosswalk(
        read_sdrf(community["community.sdrf.tsv"]),
        read_inventory(community["files-all.json"]),
        counts,
        labels,
    )
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in (
        ("sdrf-file-crosswalk.tsv.gz", files),
        ("inventory-crosswalk.tsv.gz", inventory),
        ("primary-group-crosswalk.tsv.gz", groups),
    ):
        write_tsv_gz(output / name, rows)
    for name, path in community.items():
        write_snapshot(output / f"{name}.gz", path)
    summary["inputs"] = {
        name: file_digest(path)
        for name, path in {
            **community,
            "sample_hits.tsv.gz": primary["sample_hits.tsv.gz"],
            "catalog.yaml": CATALOG,
        }.items()
    }
    summary["outputs"] = {
        path.name: file_digest(path)
        for path in sorted(output.iterdir())
        if path.name != "summary.json"
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
