"""Independently compare every serialized corrected row with its pinned original."""

import argparse
import csv
import gzip
import itertools
import json
from collections import Counter
from pathlib import Path

from datacache import install_bundle

from hitlist.curation_yaml import load_curation_yaml
from hitlist.provenance import file_digest


def verify(paths, originals):
    registry = load_curation_yaml(paths["registry.yaml"])
    lookup = {}
    for scope in registry["scopes"]:
        key = (
            scope["cell_line_name"],
            scope["reported_digestion_enzyme"],
            str(scope["n_fractions_in_run"]),
            scope["enrichment"],
            float(scope["legacy_fractionation_ph"]),
        )
        assert key not in lookup
        lookup[key] = scope
    audit = json.loads(Path(paths["audit.json"]).read_text())
    report = {}
    for kind, filename in (
        ("peptides", "bekker_jensen_2017_peptides.csv.gz"),
        ("proteins", "bekker_jensen_2017_protein_abundance.csv.gz"),
    ):
        counts = Counter()
        with (
            gzip.open(originals[filename], "rt", newline="") as old,
            gzip.open(paths[kind + ".csv.gz"], "rt", newline="") as new,
        ):
            for before, after in itertools.zip_longest(csv.DictReader(old), csv.DictReader(new)):
                assert before is not None and after is not None
                assert all(
                    before[k] == after[k]
                    for k in before
                    if k not in {"reference", "fractionation_ph"}
                )
                assert after["reported_reference"] == before["reference"]
                assert after["reference"] == "PMID:28601559"
                original_ph = float(before["fractionation_ph"])
                assert float(after["legacy_fractionation_ph"]) == original_ph
                key = (
                    before["cell_line"],
                    before["digestion_enzyme"],
                    before["n_fractions_in_run"],
                    before["enrichment"],
                    original_ph,
                )
                scope = lookup[key]
                evidence = registry["fractionation_ph_evidence"][
                    scope["fractionation_ph_evidence_id"]
                ]
                value = float(after["fractionation_ph"]) if after["fractionation_ph"] else None
                assert value == scope["fractionation_ph"]
                assert after["fractionation_protocol_id"] == scope["protocol_id"]
                assert (
                    after["fractionation_ph_evidence_id"] == scope["fractionation_ph_evidence_id"]
                )
                assert after["fractionation_ph_basis"] == evidence["basis"]
                assert after["fractionation_ph_reference"] == evidence["reference"]
                counts["n_rows"] += 1
                counts["n_changed_ph_rows"] += value != original_ph
                counts["n_unknown_ph_rows"] += value is None
        assert all(value == audit["tables"][kind][key] for key, value in counts.items())
        report[kind] = dict(counts)
    return {"tables": report, "all_serialized_rows_and_other_fields_verified": True}


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
        Path.home() / ".hitlist/bulk-fractionation/verification-sources",
        sources,
        download_options={"show_progress": False},
    )
    result = verify(paths, originals)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
