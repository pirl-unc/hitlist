"""Replay every original digestion MS1/raw/final arithmetic result without Hitlist."""

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import shutil
import statistics
import tempfile
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

SID = "soh2024_digestion"
TOLERANCES = {"raw": 1e-11, "means": 1e-11, "final": 1e-10}


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_rows(source_dir, table, condition):
    result = pq.read_table(
        source_dir / "typed" / (table + ".parquet"),
        columns=["fields_json", "source_record_id"],
        filters=[("condition_id", "=", condition)],
    ).to_pylist()
    return [(json.loads(r["fields_json"]), r["source_record_id"]) for r in result]


def number(text):
    if text in (None, "", "NA", "NaN", "#N/A"):
        return None
    x = float(text)
    return x if math.isfinite(x) else None


def condition_replay(archive, name, failures, source_dir):
    condition = f"{SID}:condition:{name}"
    raw = read_rows(source_dir, "raw_quantities", condition)
    filtered = read_rows(source_dir, "filtered_vectors", condition)
    final = read_rows(source_dir, "final_vectors", condition)
    stats = {
        "condition": name,
        "raw_rows": len(raw),
        "filtered_rows": len(filtered),
        "final_rows": len(final),
    }
    errors, maxima = Counter(), defaultdict(float)

    def compare(kind, expected, observed, record_id, detail=None):
        tolerance = TOLERANCES[
            "raw" if kind == "raw" else "means" if kind.startswith("mean") else "final"
        ]
        error = (
            0.0
            if expected is None and observed is None
            else math.inf
            if expected is None or observed is None
            else abs(expected - observed) / max(1.0, abs(observed))
        )
        maxima[kind] = max(maxima[kind], error)
        if error > tolerance:
            errors[kind] += 1
            failures.write(
                json.dumps(
                    {
                        "condition": name,
                        "kind": kind,
                        "source_record_id": record_id,
                        "expected": expected,
                        "observed": observed,
                        "relative_error": error if math.isfinite(error) else None,
                        "detail": detail,
                    },
                    allow_nan=False,
                )
                + "\n"
            )

    assignment_entries = [e for e in archive.namelist() if Path(e).name == "ASSIGNMENTS.csv"]
    ms1_entries = [e for e in archive.namelist() if Path(e).name == "MS1_HPR.csv"]
    if len(assignment_entries) != 1 or len(ms1_entries) != 1:
        raise ValueError(f"Expected one assignment and MS1 table in {name}")
    ids = Counter()
    with archive.open(assignment_entries[0]) as stream:
        for r in csv.DictReader(io.TextIOWrapper(stream)):
            if number(r["spectralAngle"]) is not None:
                ids[r["ID"]] += 1
    sources = {r["source"] for r, _ in raw}
    peptides = {r["pepSeq"] for r, _ in raw}
    totals, unique = defaultdict(float), set()
    source_rows = 0
    with archive.open(ms1_entries[0]) as stream:
        for r in csv.DictReader(io.TextIOWrapper(stream)):
            source_rows += 1
            source = r["FileName"].removesuffix(".raw")
            if (
                r["FragmentIon"] != "precursor"
                or r["ProteinName"] not in ids
                or source not in sources
                or r["PeptideSequence"] not in peptides
            ):
                continue
            # getQuant's numeric conversion precedes distinct-row reduction.
            numeric = tuple(
                number(r[k])
                for k in (
                    "TotalAreaMs1",
                    "TotalBackgroundMs1",
                    "ProductCharge",
                    "ProductMz",
                    "AverageMeasuredRetentionTime",
                )
            )
            key = (
                r["ProteinName"],
                source,
                r["PeptideSequence"],
                r["FragmentIon"],
                r["PeptideModifiedSequence"],
                *numeric,
            )
            if key in unique:
                continue
            unique.add(key)
            totals[source, r["PeptideSequence"]] += (
                (numeric[0] or 0.0) + (numeric[1] or 0.0)
            ) * ids[r["ProteinName"]]
    stats.update(
        {
            "original_ms1_rows": source_rows,
            "distinct_precursor_rows": len(unique),
            "assignment_ids": len(ids),
            "multiply_joined_assignment_ids": sum(n > 1 for n in ids.values()),
            "unique_raw_keys": len({(r["source"], r["pepSeq"]) for r, _ in raw}),
        }
    )
    tech, bio = defaultdict(list), defaultdict(list)
    for r, rid in raw:
        actual = number(r["intensity"])
        compare("raw", totals.get((r["source"], r["pepSeq"])), actual, rid)
        t = float(r["digestTime"])
        if actual is not None and (t == 0 or actual != 0):
            tech[r["pepSeq"], t, r["biological_replicate"]].append(actual)
            bio[r["pepSeq"], t].append(actual)
    for r, rid in raw:
        t = float(r["digestTime"])
        tt = tech[r["pepSeq"], t, r["biological_replicate"]]
        bb = bio[r["pepSeq"], t]
        compare(
            "mean_technical",
            statistics.mean(tt) if tt else 0.0,
            number(r["mean_techRep"]),
            rid,
        )
        compare(
            "mean_biological",
            statistics.mean(bb) if bb else 0.0,
            number(r["mean_bioRep"]),
            rid,
        )

    original = []
    for r, rid in filtered:
        for t, v in zip(r["times_pasted"].split(";"), r["int_pasted"].split(";"), strict=True):
            amount = number(v)
            if amount is None:
                raise ValueError(f"Nonfinite filtered value: {name}:{rid}")
            original.append(
                (
                    r["pepSeq"],
                    r["biological_replicate"],
                    int(r["technical_replicate"]),
                    float(t),
                    amount,
                )
            )
    biological = {r[1] for r in original}
    has_zero = {r[1] for r in original if r[3] == 0}
    missing_zero = biological - has_zero
    # Source script inserts one mock zero for each peptide/bio/tech when an
    # entire biological replicate has no zero-time records. Never repair RF1.
    mock = set()
    for p, b, a, _, _ in original:
        if b in missing_zero:
            mock.add((p, b, a, 0.0, 0.0))
    original.extend(sorted(mock))
    il = defaultdict(float)
    for p, b, a, t, v in original:
        il[p.replace("I", "L"), b, a, t] += v
    expanded = [
        (p, b, 1 if a == 3 else 2 if a == 4 else a, t, il[p.replace("I", "L"), b, a, t])
        for p, b, a, t, v in original
    ]
    baseline = {}
    for p, b, a, t, v in expanded:
        if t == 0:
            baseline[p, b, a] = max(baseline.get((p, b, a), -math.inf), v)
    if not baseline:
        raise ValueError(f"No source baseline in {name}")
    noise = statistics.median(baseline.values())
    corrected = defaultdict(list)
    censored = 0
    absent_baseline = 0
    for p, b, a, t, v in expanded:
        if t == 0:
            corrected[p, b, t].append(0.0)
        elif (p, b, a) not in baseline:
            absent_baseline += 1
        else:
            residual = v - max(noise, baseline[p, b, a])
            if residual > 0:
                corrected[p, b, t].append(residual)
            else:
                censored += 1
    slots, missing, zeros, annotation = 0, 0, 0, 0
    for r, rid in final:
        if r["biological_replicate"] == "NA":
            annotation += 1
            if r["digestTimes"] != "NA" or r["intensities"] != "NA":
                raise ValueError(f"Annotation-only vector has values: {name}:{rid}")
            continue
        for slot, (t, v) in enumerate(
            zip(r["digestTimes"].split(";"), r["intensities"].split(";"), strict=True)
        ):
            group = corrected[r["pepSeq"], r["biological_replicate"], float(t)]
            expected = statistics.mean(group) if group else None
            observed = number(v)
            compare(
                "final",
                expected,
                observed,
                rid,
                {
                    "slot": slot,
                    "time": t,
                    "peptide": r["pepSeq"],
                    "biological_replicate": r["biological_replicate"],
                    "n_finite_contributors": len(group),
                },
            )
            slots += 1
            missing += observed is None
            zeros += observed == 0
    stats.update(
        {
            "expanded_filtered_values_with_mock": len(original),
            "source_mock_zero_rows": len(mock),
            "source_mock_zero_bioreplicates": sorted(missing_zero),
            "baseline_median": noise,
            "below_background_contributors": censored,
            "missing_baseline_contributors": absent_baseline,
            "final_slots": slots,
            "source_missing_slots": missing,
            "source_zero_slots": zeros,
            "annotation_only_rows": annotation,
            "errors": dict(errors),
            "max_relative_errors": {k: v if math.isfinite(v) else None for k, v in maxima.items()},
            "passed": not errors,
        }
    )
    return stats


def main(package, output_dir):
    start = time.perf_counter()
    package, output_dir = Path(package), Path(output_dir)
    source_dir = package / "external/soh2024_digestion"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((package / "manifest.json").read_text())
    checked = 0
    for name, expected in manifest["files"].items():
        if name.startswith("external/soh2024_digestion/"):
            if sha(package / name) != expected["sha256"]:
                raise ValueError(f"Original source or typed file changed: {name}")
            checked += 1
    results = []
    with (
        zipfile.ZipFile(source_dir / "assets/aSPIRE_outputs_zip.zip") as outer,
        tempfile.TemporaryDirectory() as tmp,
        gzip.open(output_dir / "discrepancies.jsonl.gz", "wt") as failures,
    ):
        local = Path(tmp) / "condition.zip"
        for entry in outer.infolist():
            if entry.is_dir():
                continue
            with outer.open(entry) as src, local.open("wb") as dst:
                shutil.copyfileobj(src, dst, 1024 * 1024)
            with zipfile.ZipFile(local) as inner:
                stats = condition_replay(inner, Path(entry.filename).stem, failures, source_dir)
            results.append(stats)
            (output_dir / "conditions.json").write_text(
                json.dumps(results, indent=2, sort_keys=True, allow_nan=False) + "\n"
            )
            print(json.dumps(stats, sort_keys=True), flush=True)
    summary = {
        "n_conditions": len(results),
        "n_passing": sum(r["passed"] for r in results),
        "failed_conditions": [r["condition"] for r in results if not r["passed"]],
        "checked_source_files": checked,
        "raw_rows": sum(r["raw_rows"] for r in results),
        "final_slots": sum(r["final_slots"] for r in results),
        "wall_seconds": time.perf_counter() - start,
        "source_package_manifest_sha256": sha(package / "manifest.json"),
        "source_scope": "Original arithmetic only; no claim of independent search/mapping/synthesis-removal validation or training admission",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    if (
        summary["n_conditions"] != 17
        or summary["n_passing"] != 17
        or summary["raw_rows"] != 545664
        or summary["final_slots"] != 280664
    ):
        raise ValueError("Original digestion arithmetic or source census changed")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_package", type=Path)
    parser.add_argument("output_directory", type=Path)
    args = parser.parse_args()
    main(args.source_package, args.output_directory)
