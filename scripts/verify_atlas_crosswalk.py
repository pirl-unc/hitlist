"""Independently replay the published Atlas crosswalk from pinned source bytes."""

import argparse
import csv
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

PINNED = {
    "community.sdrf.tsv": "baa1907cc847abc382683492970eef9da6fbbb24ef4448dab6dbe76357c67fb2",
    "files-all.json": "3bac909670e737f9f71d8bb5aad1f5e26b33f68f2c3b8bace1a350f8d6b987b0",
    "sample_hits.tsv.gz": "0e8771113cae27c620b2b9619f035675ddcc87b8d7dfac50750e71ab60eb0167",
}
HLA_CLASSES = {
    "Human leukocyte antigen class I": "HLA-I",
    "Human leukocyte antigen class II": "HLA-II",
}


def digest(path):
    value = hashlib.sha256()
    opener = (
        gzip.open if Path(path).name in {"community.sdrf.tsv.gz", "files-all.json.gz"} else open
    )
    with opener(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def tsv(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", newline="") as stream:
        yield from csv.DictReader(stream, delimiter="\t")


def group(donor, tissue, hla_class):
    return donor, "".join(tissue.casefold().split()), hla_class


def verify(audit_dir, primary_path):
    audit_dir = Path(audit_dir)
    summary = json.loads((audit_dir / "summary.json").read_text())
    for name, expected in summary["outputs"].items():
        path = audit_dir / name
        assert path.stat().st_size == expected["size_bytes"], name
        with path.open("rb") as stream:
            content = hashlib.sha256()
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                content.update(chunk)
        assert content.hexdigest() == expected["sha256"], name
    paths = {
        "community.sdrf.tsv": audit_dir / "community.sdrf.tsv.gz",
        "files-all.json": audit_dir / "files-all.json.gz",
        "sample_hits.tsv.gz": Path(primary_path),
    }
    for name, path in paths.items():
        assert digest(path) == PINNED[name], name
        assert summary["inputs"][name]["sha256"] == PINNED[name], name
    sdrf = list(tsv(paths["community.sdrf.tsv"]))
    with gzip.open(paths["files-all.json"], "rt") as stream:
        inventory = json.load(stream)
    files = list(tsv(audit_dir / "sdrf-file-crosswalk.tsv.gz"))
    recorded_inventory = list(tsv(audit_dir / "inventory-crosswalk.tsv.gz"))
    recorded_groups = list(tsv(audit_dir / "primary-group-crosswalk.tsv.gz"))
    assert len(sdrf) == len(files) == summary["n_sdrf_rows"]
    assert len(inventory) == len(recorded_inventory) == summary["n_inventory_records"]
    names = {record["fileName"]: (i, record) for i, record in enumerate(inventory, 1)}
    assert len(names) == len(inventory)
    sdrf_names = set()
    source_names = defaultdict(set)
    context_files = defaultdict(set)
    raw_matches = 0
    for i, (source, recorded) in enumerate(zip(sdrf, files), 1):
        assert all(recorded[column] == value for column, value in source.items()), i
        assert recorded["sdrf_row"] == str(i)
        name = source["comment[data file]"]
        assert name not in sdrf_names
        sdrf_names.add(name)
        inventory_row, item = names[name]
        assert item["fileCategory"]["value"] == "PEAK"
        assert recorded["inventory_row"] == str(inventory_row)
        assert recorded["inventory_accession"] == item["accession"]
        assert recorded["file_match_status"] == "exact_unique_peak"
        raw_name = name.removesuffix(".mzML") + ".raw"
        raw = names.get(raw_name)
        matched = raw is not None and raw[1]["fileCategory"]["value"] == "RAW"
        assert recorded["raw_match_status"] == ("exact_stem" if matched else "unresolved")
        assert recorded["raw_file_name"] == (raw_name if matched else "")
        assert recorded["raw_inventory_row"] == (str(raw[0]) if matched else "")
        raw_matches += matched
        identity = group(
            source["characteristics[individual]"],
            source["characteristics[organism part]"],
            HLA_CLASSES[source["factor value[human leukocyte antigen]"]],
        )
        source_names[identity].add(source["source name"])
        context_files[identity].add(name)
    assert (
        len({name for nameset in source_names.values() for name in nameset})
        == summary["n_sdrf_source_names"]
    )
    assert len(source_names) == summary["n_sdrf_donor_tissue_class_groups"]
    assert raw_matches == summary["n_exact_raw_stem_matches"]
    assert len(files) - raw_matches == summary["n_unresolved_raw_aliases"]
    for i, (item, recorded) in enumerate(zip(inventory, recorded_inventory), 1):
        assert recorded == {
            "inventory_row": str(i),
            "accession": item["accession"],
            "file_name": item["fileName"],
            "category": item["fileCategory"]["value"],
            "sdrf_row": next(
                (
                    str(j)
                    for j, row in enumerate(sdrf, 1)
                    if row["comment[data file]"] == item["fileName"]
                ),
                "",
            ),
            "sdrf_status": "exact_data_file"
            if item["fileName"] in sdrf_names
            else "not_sdrf_data_file",
        }, i
    counts = Counter()
    for row in tsv(paths["sample_hits.tsv.gz"]):
        counts[group(row["donor"], row["tissue"], row["hla_class"])] += 1
    assert len(counts) == len(recorded_groups) == summary["n_primary_groups"]
    for source, recorded in zip(sdrf, files):
        identity = group(
            source["characteristics[individual]"],
            source["characteristics[organism part]"],
            HLA_CLASSES[source["factor value[human leukocyte antigen]"]],
        )
        assert recorded["primary_group_status"] == (
            "candidate_context" if identity in counts else "no_primary_group"
        )
    donors = sorted(
        {identity[0] for identity in counts},
        key=lambda donor: hashlib.sha256(
            ("atlas-donor-holdout-v1\nhla-ligand-atlas:" + donor).encode()
        ).hexdigest(),
    )
    splits = {
        donor: "test" if i < 3 else "val" if i < 6 else "train" for i, donor in enumerate(donors)
    }
    coverage = defaultdict(Counter)
    donor_coverage = defaultdict(Counter)
    seen_groups = set()
    candidate_file_splits = defaultdict(set)
    for row in recorded_groups:
        identity = group(row["donor"], row["tissue"], row["hla_class"])
        assert identity not in seen_groups
        seen_groups.add(identity)
        assert int(row["n_observations"]) == counts[identity]
        assert row["partition"] == splits[row["donor"]]
        assert json.loads(row["candidate_source_names"]) == sorted(source_names[identity])
        assert json.loads(row["candidate_peak_files"]) == sorted(context_files[identity])
        assert int(row["n_candidate_source_names"]) == len(source_names[identity])
        assert int(row["n_candidate_peak_files"]) == len(context_files[identity])
        assert row["candidate_status"] == (
            "no_sdrf_context"
            if not source_names[identity]
            else "multiple_source_names"
            if len(source_names[identity]) > 1
            else "one_source_name"
        )
        for file_name in context_files[identity]:
            candidate_file_splits[file_name].add(row["partition"])
        assert row["qc_inclusion_status"] == "unknown"
        assert row["observation_acquisition_status"] == "unknown"
        assert row["specimen_status"] == "unknown"
        for bucket in (
            coverage[f"{row['partition']}/{row['hla_class']}"],
            donor_coverage[f"{row['donor']}/{row['hla_class']}"],
        ):
            bucket["n_observations"] += counts[identity]
            bucket["n_reported_sample_groups"] += 1
            bucket["n_candidate_context_observations"] += (
                counts[identity] if source_names[identity] else 0
            )
            bucket["n_multiple_source_name_observations"] += (
                counts[identity] if len(source_names[identity]) > 1 else 0
            )
            for axis in ("qc_inclusion", "acquisition", "specimen"):
                bucket[f"n_unknown_{axis}_observations"] += counts[identity]
    assert seen_groups == set(counts)
    assert {name: dict(value) for name, value in coverage.items()} == summary[
        "coverage_by_partition_and_class"
    ]
    assert {name: dict(value) for name, value in donor_coverage.items()} == summary[
        "coverage_by_donor_and_class"
    ]
    assert sum(counts.values()) == 967449
    assert summary["n_primary_groups_with_candidate_context"] == sum(
        bool(source_names[identity]) for identity in counts
    )
    assert summary["n_primary_groups_with_multiple_source_names"] == sum(
        len(source_names[identity]) > 1 for identity in counts
    )
    assert summary["n_candidate_peak_files_in_primary_groups"] == len(candidate_file_splits)
    assert summary["n_candidate_peak_files_in_multiple_partitions"] == sum(
        len(splits) > 1 for splits in candidate_file_splits.values()
    )
    assert summary["n_unannotated_peak_records"] == sum(
        item["fileCategory"]["value"] == "PEAK" and item["fileName"] not in sdrf_names
        for item in inventory
    )
    assert summary["n_inventory_result_records_without_qc_join"] == sum(
        item["fileCategory"]["value"] == "RESULT" for item in inventory
    )
    return {
        "verdict": "pass",
        "n_primary_observations": sum(counts.values()),
        "n_primary_groups": len(counts),
        "n_exact_peak_matches": len(files),
        "n_unknown_acquisition_observations": sum(counts.values()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("audit_dir", type=Path)
    parser.add_argument("primary_sample_hits", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.audit_dir, args.primary_sample_hits), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
