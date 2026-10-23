"""Independently replay the controlled contrast from a source-verified run export."""

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("package", type=Path)
parser.add_argument("base", type=Path)
parser.add_argument("contrast", type=Path)
args = parser.parse_args()
package, base, contrast = args.package, args.base, args.contrast


def read(directory, name):
    return json.loads((directory / f"{name}.json").read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


for new_name, old_name in (
    ("components", "components"),
    ("run_examples", "examples"),
    ("run_dispositions", "dispositions"),
    ("source_tables", "source_tables"),
):
    assert sha(contrast / f"{new_name}.json") == sha(base / f"{old_name}.json")

peptides = pq.read_table(
    package / "external/wolflevy2018_controlled/typed/peptide_rows.parquet",
    columns=["peptide_row_id", "source_dataset", "sequence_raw"],
).to_pylist()
assert len(peptides) == 9206
assignments = {row["peptide_row_id"]: row for row in read(base, "components")}
assert len(assignments) == len(peptides)
run_rows = read(base, "examples")
assert len(run_rows) == 21607
by_peptide = defaultdict(lambda: defaultdict(list))
for row in run_rows:
    context = row["context"]
    assert context["pair_id"] is None
    if row["source_dataset"] == "PXD009508" and context["capture_label_raw"] == "alpha6":
        by_peptide[row["peptide_row_id"]][context["treatment_label_raw"]].append(row)

expected_dispositions = []
expected_examples = []
reason_counts = Counter()
for peptide in sorted(peptides, key=lambda row: row["peptide_row_id"]):
    pid = peptide["peptide_row_id"]
    reasons = []
    arms = {}
    if peptide["source_dataset"] != "PXD009508":
        reasons.append("outside_inhibitor_project")
    else:
        for arm in ("ut", "epoxo"):
            values = sorted(by_peptide[pid][arm], key=lambda row: row["context"]["replicate_label"])
            labels = [row["context"]["replicate_label"] for row in values]
            if labels != [1, 2, 3]:
                reasons.append("fewer_than_three_positive_runs_" + arm)
            if not any(row["evidence_counts"]["n_direct_msms_evidence"] > 0 for row in values):
                reasons.append("no_direct_msms_" + arm)
            arms[arm] = values
    common = {
        **assignments[pid],
        "source_dataset": peptide["source_dataset"],
        "peptide_sequence": peptide["sequence_raw"],
    }
    expected_dispositions.append(
        {**common, "status": "excluded" if reasons else "admitted", "reasons": reasons}
    )
    reason_counts.update(reasons)
    if reasons:
        continue
    example = dict(common)
    for arm, label in (("ut", "untreated"), ("epoxo", "epoxomicin")):
        rows = arms[arm]
        values = [row["reported_intensity"] for row in rows]
        assert all(value > 0 for value in values)
        example[label + "_intensities"] = values
        example[label + "_measurement_ids"] = [row["measurement_id"] for row in rows]
        example[label + "_mean_intensity"] = math.fsum(values) / len(values)
    expected_examples.append(example)

assert expected_dispositions == read(contrast, "dispositions")
assert expected_examples == read(contrast, "examples")
summary = read(contrast, "summary")
splits = Counter(row["split"] for row in expected_examples)
assert dict(sorted(splits.items())) == {"test": 14, "train": 59, "val": 14}
assert summary["n_source_peptides"] == len(peptides)
assert summary["n_examples"] == len(expected_examples) == 87
assert summary["n_excluded_peptides"] == len(peptides) - len(expected_examples)
assert summary["n_contributing_runs"] == 6 * len(expected_examples)
assert summary["reason_counts"] == dict(sorted(reason_counts.items()))
assert summary["examples_by_split"] == dict(sorted(splits.items()))
for split in ("train", "val", "test"):
    expected = len({row["component_id"] for row in expected_examples if row["split"] == split})
    assert summary["components_by_split"][split] == expected

print(
    json.dumps(
        {
            "passed": True,
            "n_source_peptides": len(peptides),
            "n_run_examples": len(run_rows),
            "n_contrast_examples": len(expected_examples),
            "examples_by_split": dict(sorted(splits.items())),
            "n_contributing_runs": 6 * len(expected_examples),
            "n_excluded_peptides": len(peptides) - len(expected_examples),
            "embedded_base_payloads_exact": 4,
            "held_out_response_metrics": "none",
        },
        sort_keys=True,
    )
)
