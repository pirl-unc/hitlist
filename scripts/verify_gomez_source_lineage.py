"""Replay Gomez-Zepeda original ZIP rows without importing Hitlist builders."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sqlite3
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.dataset as ds
import pyarrow.parquet as pq

PARENT_SHA = "58ac75009b6c6452cf19b317908f6043f1be414c81dc8ae9e9423a185467721d"
DERIVED = {
    "JY": ("gomez_zepeda_2024_jy.csv", 20_983),
    "Raji": ("gomez_zepeda_2024_raji.csv", 24_325),
    "HeLa": ("gomez_zepeda_2024_hela.csv", 10_356),
    "SK-MEL-37": ("gomez_zepeda_2024_skmel37.csv", 11_491),
    "Plasma": ("gomez_zepeda_2024_plasma.csv", 1_397),
}


def digest(path):
    sha = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            sha.update(block)
    return {"sha256": sha.hexdigest(), "size_bytes": Path(path).stat().st_size}


def _rows(parquet):
    for batch in pq.ParquetFile(parquet).iter_batches(batch_size=8192):
        yield from batch.to_pylist()


def replay(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    parent_path = directory / "manifests/gomez_parent.json"
    parent = json.loads(parent_path.read_text())
    registry = json.loads((directory / "gomez_source/registry.json").read_text())
    assert (manifest["schema_version"], manifest["catalog_schema_version"]) == (11, 16)
    assert (parent["schema_version"], parent["catalog_schema_version"]) == (10, 15)
    assert digest(parent_path)["sha256"] == manifest["parent_manifest_sha256"] == PARENT_SHA
    assert set(parent["files"]) <= set(manifest["files"])
    assert all(parent["files"][name] == manifest["files"][name] for name in parent["files"])
    actual_files = {
        str(path.relative_to(directory))
        for path in directory.rglob("*")
        if path.is_file() and path != directory / "manifest.json"
    }
    assert actual_files == set(manifest["files"])
    for name, expected in manifest["files"].items():
        assert digest(directory / name) == expected, name

    derived_table = ds.dataset(directory / "native/observations.parquet").to_table(
        columns=["provenance_id", "peptide", "supplementary_file"],
        filter=ds.field("pmid") == 38480730,
    )
    derived = {
        (row["supplementary_file"], row["peptide"]): row["provenance_id"]
        for row in derived_table.to_pylist()
    }
    assert len(derived) == derived_table.num_rows == 68_552
    source_result_rows = _rows(directory / "gomez_source/results.parquet")
    groups = sqlite3.connect(":memory:")
    groups.execute(
        "CREATE TABLE members (id TEXT PRIMARY KEY, payload TEXT NOT NULL, "
        "first_id TEXT NOT NULL, n_source_result_rows INTEGER NOT NULL)"
    )
    counts = Counter()
    unions = defaultdict(set)
    linked = set()
    condition_counts = Counter()
    outside = Counter()
    for source in registry["sources"]:
        archive_path = directory / "gomez_source" / source["archive"]
        assert digest(archive_path)["sha256"] == source["archive_sha256"]
        with zipfile.ZipFile(archive_path) as archive:
            assert archive.namelist() == [source["member"]]
            sha = hashlib.sha256()
            with archive.open(source["member"]) as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    sha.update(block)
            assert sha.hexdigest() == source["member_sha256"]
            with archive.open(source["member"]) as handle:
                reader = csv.DictReader(io.TextIOWrapper(handle, encoding="utf-8-sig", newline=""))
                assert reader.fieldnames and len(reader.fieldnames) == len(set(reader.fieldnames))
                for row_index, raw in enumerate(reader, 1):
                    result = next(source_result_rows)
                    assert all(result[name] == value for name, value in raw.items())
                    assert result["source_phase"] == source["id"]
                    assert result["source_row_index"] == row_index
                    assert result["source_asset_id"] == source["archive"]
                    assert result["source_pmid"] == "38480730"
                    assert result["source_dataset"] == "gomez_zepeda_2024_original"
                    assert result["peptide_sequence"] == raw["Sequence"]
                    assert result["cell_line_name"] == raw["Cell.Line"]
                    assert result["condition_text"] == raw["Condition"]
                    assert result["biological_replicate_label"] == raw["BioReplicate"]
                    assert result["sample_label_raw"] == raw["Sample"]
                    assert result["technical_replicate_label"] == raw["TechReplicate"]
                    assert result["acquisition_id"] == raw["Source_File_ID"]
                    assert result["peptide_modifications_raw"] == raw["PTM"]
                    assert result["reported_peptidoform_raw"] == raw["Peptide_charge"]
                    assert result["charge_raw"] == raw["Charge"]
                    assert result["identification_method_raw"] == raw["Found_By"]
                    assert result["identification_confidence_raw"] == raw["inv10lgP"]
                    assert result["identification_q_value_raw"] == raw.get("mokapot_q_value")
                    assert result["tims_ramp_raw"] == raw.get("TIMS_ramp")
                    assert result["polygon_raw"] == raw.get("Polygon")
                    assert result["quantitative_raw"] == raw["Area"]
                    assert result["log2_area_raw"] == raw["Log2Area"]
                    assert result["measurement_unit"] == "source_peak_area_arbitrary_units"
                    record_id = f"gomez:{source['id']}:{row_index}"
                    assert result["source_record_id"] == record_id
                    key = (
                        source["id"],
                        raw["Cell.Line"],
                        raw["Condition"],
                        raw["BioReplicate"],
                        raw["Sample"],
                        raw["Source_File_ID"],
                        raw["Sequence"],
                    )
                    encoded = json.dumps(key, ensure_ascii=False, separators=(",", ":"))
                    member_id = "gomez:membership:" + hashlib.sha256(encoded.encode()).hexdigest()
                    assert result["membership_id"] == member_id
                    if 8 <= len(raw["Sequence"]) <= 15:
                        cell_file = DERIVED[raw["Cell.Line"]][0]
                        derived_id = derived[(cell_file, raw["Sequence"])]
                        status = "linked_exact_sequence_and_cell_source"
                        linked.add(derived_id)
                        unions[raw["Cell.Line"]].add(raw["Sequence"])
                    else:
                        derived_id = None
                        status = "outside_derived_length_8_15"
                        outside[raw["Cell.Line"]] += 1
                    assert (result["derived_observation_id"], result["derivation_link_status"]) == (
                        derived_id,
                        status,
                    )
                    payload = json.dumps((key, derived_id, status), ensure_ascii=False)
                    groups.execute(
                        "INSERT INTO members VALUES (?, ?, ?, 1) "
                        "ON CONFLICT(id) DO UPDATE SET "
                        "n_source_result_rows=n_source_result_rows+1",
                        (member_id, payload, record_id),
                    )
                    counts[source["id"]] += 1
                    condition_counts[(source["id"], raw["Condition"])] += 1
        assert counts[source["id"]] == source["n_rows"]
    try:
        next(source_result_rows)
    except StopIteration:
        pass
    else:
        raise AssertionError("Extra source result rows")
    assert linked == set(derived.values())
    assert {name: len(values) for name, values in unions.items()} == {
        name: count for name, (_, count) in DERIVED.items()
    }
    member_rows = _rows(directory / "gomez_source/memberships.parquet")
    n_memberships = 0
    n_grouped_results = 0
    for member_id, payload, first_id, count in groups.execute("SELECT * FROM members ORDER BY id"):
        actual = next(member_rows)
        key, derived_id, status = json.loads(payload)
        phase, cell, condition, biological, sample, acquisition, sequence = key
        expected = {
            "membership_id": member_id,
            "source_phase": phase,
            "source_asset_id": next(s["archive"] for s in registry["sources"] if s["id"] == phase),
            "peptide_sequence": sequence,
            "cell_line_name": cell,
            "condition_text": condition,
            "biological_replicate_label": biological,
            "sample_label_raw": sample,
            "acquisition_id": acquisition,
            "first_source_record_id": first_id,
            "n_source_result_rows": count,
            "derived_observation_id": derived_id,
            "derivation_link_status": status,
        }
        assert actual == expected, member_id
        n_memberships += 1
        n_grouped_results += count
    try:
        next(member_rows)
    except StopIteration:
        pass
    else:
        raise AssertionError("Extra membership rows")
    groups.close()
    assert n_grouped_results == sum(counts.values()) == 594_916
    assert manifest["reconciliation"] == {
        "n_source_result_rows": 594_916,
        "n_source_memberships": n_memberships,
        "n_derived_observations_linked": 68_552,
        "source_rows": dict(counts),
        "derived_sequence_unions": {name: len(unions[name]) for name in sorted(unions)},
    }
    return {
        "manifest_sha256": digest(directory / "manifest.json")["sha256"],
        "n_predecessor_files": len(parent["files"]),
        "n_successor_files": len(manifest["files"]),
        "n_source_result_rows": sum(counts.values()),
        "n_source_memberships": n_memberships,
        "n_derived_observations_linked": len(linked),
        "source_rows": dict(counts),
        "condition_rows": {
            f"{phase}:{condition}": count
            for (phase, condition), count in sorted(condition_counts.items())
        },
        "outside_derived_length_rows": dict(outside),
        "derived_sequence_unions": {name: len(unions[name]) for name in sorted(unions)},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = replay(args.package)
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(payload)
    print(payload)


if __name__ == "__main__":
    main()
