"""Independently verify every serialized count against source rows and inventory."""

import argparse
import csv
import gzip
import hashlib
import itertools
import json
import math
from collections import Counter
from pathlib import Path

from datacache import install_bundle

from hitlist.curation_yaml import load_curation_yaml
from hitlist.provenance import file_digest


def verify(paths, originals):
    registry = load_curation_yaml(paths["registry.yaml"])
    audit = json.loads(Path(paths["audit.json"]).read_text())
    summary_spec = registry["provenance"]["members"]["summary.txt"]
    assert file_digest(paths["summary.txt"]) == summary_spec
    with Path(paths["summary.txt"]).open() as stream:
        summary = list(csv.DictReader(stream, delimiter="\t"))
    lookup = {}
    for scope in registry["scopes"]:
        selected = [row for row in summary if row["Experiment"] in scope["experiment_ids"]]
        raw = sorted({row["Raw file"] for row in selected})
        assert len(raw) == len(selected) == scope["n_acquisition_runs"]
        assert {row["Experiment"] for row in selected} == set(scope["experiment_ids"])
        assert hashlib.sha256("\n".join(raw).encode()).hexdigest() == scope["raw_file_ids_sha256"]
        assert len(scope["experiment_ids"]) == scope["n_replicates_possible"]
        key = (
            scope["cell_line_name"],
            scope["reported_digestion_enzyme"],
            str(scope["n_fractions_in_run"]),
            scope["enrichment"],
            float(scope["legacy_fractionation_ph"]),
        )
        assert key not in lookup
        lookup[key] = scope
    counts, per_scope = Counter(), Counter()
    maxima = {}
    # This first pass independently replays the prior observed-maximum behavior.
    with gzip.open(originals["bekker_jensen_2017_peptides.csv.gz"], "rt", newline="") as stream:
        for row in csv.DictReader(stream):
            key = (
                row["cell_line"],
                row["digestion_enzyme"],
                row["n_fractions_in_run"],
                row["enrichment"],
                float(row["fractionation_ph"]),
            )
            protocol = lookup[key]["protocol_id"]
            maxima[protocol] = max(maxima.get(protocol, 0), int(row["n_replicates_detected"]))
    assert maxima == audit["historical_maxima_by_scope"]
    with (
        gzip.open(originals["bekker_jensen_2017_peptides.csv.gz"], "rt", newline="") as old,
        gzip.open(paths["peptides.csv.gz"], "rt", newline="") as new,
    ):
        for before, after in itertools.zip_longest(csv.DictReader(old), csv.DictReader(new)):
            assert before is not None and after is not None
            for field in before:
                if field not in {"n_replicates_detected", "fractionation_ph", "reference"}:
                    assert before[field] == after[field], field
            assert after["reported_n_replicates_detected"] == before["n_replicates_detected"]
            assert after["reported_n_replicates_possible"] == ""
            assert after["reported_reference"] == before["reference"]
            assert after["reference"] == "PMID:28601559"
            assert float(after["legacy_fractionation_ph"]) == float(before["fractionation_ph"])
            key = (
                before["cell_line"],
                before["digestion_enzyme"],
                before["n_fractions_in_run"],
                before["enrichment"],
                float(before["fractionation_ph"]),
            )
            scope = lookup[key]
            protocol = scope["protocol_id"]
            assert after["replicate_protocol_id"] == after["fractionation_protocol_id"] == protocol
            ph = registry["fractionation_ph_evidence"][scope["fractionation_ph_evidence_id"]]
            value = float(after["fractionation_ph"]) if after["fractionation_ph"] else None
            assert value == scope["fractionation_ph"]
            assert after["fractionation_ph_basis"] == ph["basis"]
            assert after["fractionation_ph_reference"] == ph["reference"]
            assert after["fractionation_ph_evidence_id"] == scope["fractionation_ph_evidence_id"]
            total = len(scope["experiment_ids"])
            detected = int(before["n_replicates_detected"])
            available = scope["replicate_count_status"] == "available" and 0 <= detected <= total
            for name in ("n_experiments_possible", "n_replicates_possible"):
                assert int(after[name]) == total
            for name in ("n_experiments_detected", "n_replicates_detected"):
                assert after[name] == (str(detected) if available else "")
            assert after["replicate_count_status"] == scope["replicate_count_status"]
            assert after["n_biological_replicates"] == ""
            assert int(after["n_acquisition_runs"]) == scope["n_acquisition_runs"]
            assert json.loads(after["replicate_experiment_ids"]) == scope["experiment_ids"]
            assert after["replicate_count_unit"] == "MaxQuant experiment"
            assert after["replicate_count_source_sha256"] == summary_spec["sha256"]
            assert after["replicate_acquisition_ids_sha256"] == scope["raw_file_ids_sha256"]
            assert (
                after["replicate_count_reference"]
                == registry["provenance"]["archive"] + "#summary.txt"
            )
            assert int(after["legacy_inferred_n_replicates_possible"]) == maxima[protocol]
            legacy_fraction = detected / maxima[protocol]
            assert math.isclose(
                float(after["legacy_inferred_reproducibility_fraction"]),
                legacy_fraction,
                rel_tol=0,
                abs_tol=1e-15,
            )
            if available:
                assert math.isclose(
                    float(after["reproducibility_fraction"]),
                    detected / total,
                    rel_tol=0,
                    abs_tol=1e-15,
                )
            else:
                assert after["reproducibility_fraction"] == ""
            assert after["denominator_changed"] == str(total != maxima[protocol])
            values = {
                "n_rows": 1,
                "n_unknown_numerators": int(not available),
                "n_changed_denominators": int(total != maxima[protocol]),
            }
            for cutoff, label in [(0, "zero"), (0.5, "half"), (1, "all")]:
                values["n_legacy_pass_" + label] = int(legacy_fraction >= cutoff)
                values["n_qualified_pass_" + label] = int(available and detected / total >= cutoff)
            for name, count in values.items():
                counts[name] += count
                per_scope[(protocol, name)] += count
    assert dict(counts) == audit["counts"]
    with Path(paths["scope-counts.tsv"]).open() as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            for name in counts:
                assert int(row[name]) == per_scope[(row["protocol_id"], name)]
    return {
        "verified": True,
        "n_reviewed_scopes": len(lookup),
        "counts": dict(counts),
        "all_original_rows_order_and_non_count_values_preserved": True,
        "primary_summary_experiments_and_acquisition_hashes_verified": True,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    receipt = json.loads(args.receipt.read_text())
    paths = receipt.get("downloaded", receipt.get("paths"))
    for name, path in paths.items():
        expected = receipt["files"][name]
        actual = file_digest(path)
        assert actual["sha256"] == expected["sha256"]
        assert actual["size_bytes"] == expected.get("size_bytes", expected.get("size"))
    sources = json.loads(Path(paths["source-registry.json"]).read_text())
    originals = install_bundle(
        Path.home() / ".hitlist/bulk-replicates/verification-sources",
        sources,
        download_options={"show_progress": False},
    )
    result = verify(paths, originals)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
