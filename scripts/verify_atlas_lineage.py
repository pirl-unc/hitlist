"""Independently verify all native Atlas rows and partitions without Hitlist imports."""

import argparse
import csv
import gzip
import hashlib
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import quote


def pin(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"sha256": digest.hexdigest(), "size_bytes": Path(path).stat().st_size}


def read(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", newline="") as stream:
        yield from csv.DictReader(stream, delimiter="\t")


def verify(receipt):
    paths = {name: Path(path) for name, path in receipt["paths"].items()}
    for name, path in paths.items():
        assert pin(path) == receipt["files"][name], name
    peptides = {
        r["peptide_sequence_id"]: (r["peptide_sequence"], i)
        for i, r in enumerate(read(paths["peptides.tsv.gz"]), 1)
    }
    typing = defaultdict(set)
    for row in read(paths["donors.tsv.gz"]):
        typing[row["donor"]].add(row["hla_allele"])
    policy = json.loads(paths["split-policy.json"].read_text())
    ids = ["hla-ligand-atlas:" + donor for donor in typing]
    ordered = sorted(
        ids, key=lambda d: hashlib.sha256(("atlas-donor-holdout-v1\n" + d).encode()).hexdigest()
    )
    expected_assignments = {
        d: "test" if i < 3 else "val" if i < 6 else "train" for i, d in enumerate(ordered)
    }
    assert policy["assignments"] == expected_assignments and len(ordered) == 21
    observed_ids, donors, groups, seqs, classes, tissues = (
        set(),
        defaultdict(set),
        defaultdict(set),
        defaultdict(set),
        defaultdict(Counter),
        defaultdict(set),
    )
    n_by_split, n_by_donor, n_by_group = Counter(), Counter(), Counter()
    group_metadata = {}
    donor_sequences, donor_tissues, donor_groups = (
        defaultdict(set),
        defaultdict(set),
        defaultdict(set),
    )
    group_sequences = defaultdict(set)
    n_rows = 0
    for i, pair in enumerate(
        itertools.zip_longest(
            read(paths["sample_hits.tsv.gz"]), read(paths["observations.tsv.gz"])
        ),
        1,
    ):
        raw, row = pair
        assert raw is not None and row is not None, "source/output row count mismatch"
        assert all(row[key] == value for key, value in raw.items()), i
        donor_id = "hla-ligand-atlas:" + raw["donor"]
        group = "hla-ligand-atlas:2020.12:sample-group:" + ":".join(
            quote(raw[key], safe="") for key in ["donor", "tissue", "hla_class"]
        )
        identity = group + ":peptide:" + quote(raw["peptide_sequence_id"], safe="")
        peptide, peptide_row = peptides[raw["peptide_sequence_id"]]
        split = expected_assignments[donor_id]
        expected = {
            "sample_hits_row": str(i),
            "peptide": peptide,
            "peptide_row": str(peptide_row),
            "peptide_source:peptide_sequence": peptide,
            "source_tissue": raw["tissue"],
            "source_dataset": "hla-ligand-atlas:2020.12",
            "source_record_id": f"hla-ligand-atlas:2020.12:sample_hits:{i}",
            "donor_id": donor_id,
            "donor_status": "resolved",
            "tissue_status": "nonmalignant",
            "assay_modality": "mass_spectrometry",
            "is_cell_line": "False",
            "mhc_class": {"HLA-I": "I", "HLA-II": "II"}[raw["hla_class"]],
            "mhc_restriction": "",
            "sample_alleles": ";".join(sorted(typing[raw["donor"]])),
            "pmid": "33858848",
            "reported_sample_group_id": group,
            "evidence_row_id": identity,
            "donor_evidence_basis": "primary_sample_hit_donor_reference",
            "reported_sample_group_basis": "source_donor_tissue_hla_class_triplet",
            "specimen_ids": "[]",
            "specimen_status": "unknown",
            "acquisition_ids": "[]",
            "acquisition_status": "unknown",
            "assay_method": "cellular MHC/mass spectrometry",
            "label_kind": "observed_eluted_ligand",
            "label_policy": "positive_observation_only_no_unobserved_negative",
            "source_reference": "https://doi.org/10.1136/jitc-2020-002071",
            "source_url": "https://hla-ligand-atlas.org/data",
            "source_accession": "PXD019643",
            "cell_type": "",
            "cell_condition": "direct_ex_vivo",
            "cell_condition_basis": "primary_publication_tissue_ligand_isolation",
            "partition": split,
        }
        assert set(row) == set(raw) | set(expected), set(row) - set(raw) - set(expected)
        assert all(row[key] == value for key, value in expected.items()), i
        assert identity not in observed_ids
        observed_ids.add(identity)
        donors[split].add(donor_id)
        groups[split].add(group)
        seqs[split].add(peptide)
        classes[split][row["mhc_class"]] += 1
        tissues[split].add(raw["tissue"])
        n_by_split[split] += 1
        n_by_donor[donor_id] += 1
        n_by_group[group] += 1
        donor_sequences[donor_id].add(peptide)
        donor_tissues[donor_id].add(raw["tissue"])
        donor_groups[donor_id].add(group)
        group_sequences[group].add(peptide)
        group_metadata[group] = (donor_id, raw["tissue"], row["mhc_class"], split)
        n_rows = i
    assert n_rows == 967449
    coverage = json.loads(paths["coverage.json"].read_text())
    for split in n_by_split:
        assert coverage[split] == {
            "n_observations": n_by_split[split],
            "n_peptides": len(seqs[split]),
            "n_donors": len(donors[split]),
            "n_reported_sample_groups": len(groups[split]),
            "n_tissues": len(tissues[split]),
            "n_unknown_donor_observations": 0,
            "n_unknown_specimen_observations": n_by_split[split],
            "n_unknown_acquisition_observations": n_by_split[split],
            "n_observations_by_mhc_class": dict(classes[split]),
        }
    donor_rows = list(read(paths["donors.tsv"]))
    assert len(donor_rows) == len(ordered)
    for row in donor_rows:
        d = row["donor_id"]
        assert row["partition"] == expected_assignments[d]
        for key, value in {
            "n_observations": n_by_donor[d],
            "n_peptides": len(donor_sequences[d]),
            "n_tissues": len(donor_tissues[d]),
            "n_reported_sample_groups": len(donor_groups[d]),
        }.items():
            assert int(row[key]) == value
        assert row["sample_alleles"] == ";".join(
            sorted(typing[d.removeprefix("hla-ligand-atlas:")])
        )
    group_rows = list(read(paths["sample-groups.tsv"]))
    assert len(group_rows) == len(n_by_group)
    for row in group_rows:
        g = row["reported_sample_group_id"]
        assert int(row["n_observations"]) == n_by_group[g]
        assert int(row["n_peptides"]) == len(group_sequences[g])
        assert (
            tuple(row[k] for k in ["donor_id", "source_tissue", "mhc_class", "partition"])
            == group_metadata[g]
        )
    shared = {(r["left"], r["right"], r["peptide"]) for r in read(paths["shared-peptides.tsv.gz"])}
    expected_shared = set()
    overlap = json.loads(paths["overlap.json"].read_text())
    for pair in overlap["pairs"]:
        a, b = pair["left"], pair["right"]
        common = seqs[a] & seqs[b]
        assert not donors[a] & donors[b] and not groups[a] & groups[b]
        assert pair == {
            "left": a,
            "right": b,
            "n_shared_donors": 0,
            "n_shared_sample_groups": 0,
            "n_shared_observations": 0,
            "n_shared_peptides": len(common),
        }
        expected_shared.update((a, b, p) for p in common)
    assert shared == expected_shared and len(overlap["pairs"]) == 3
    assert overlap["donor_disjoint_verdict"] == "pass"
    assert overlap["peptide_disjoint_verdict"] == "fail"
    assert (
        overlap["specimen_identity_verdict"]
        == overlap["acquisition_identity_verdict"]
        == "inconclusive"
    )
    return {
        "verified": True,
        "n_source_rows_verified": n_rows,
        "n_donors": len(ordered),
        "n_reported_sample_groups": len(n_by_group),
        "n_shared_peptide_pair_rows": len(shared),
        "n_artifacts_verified": len(paths),
        "donor_disjoint_verdict": "pass",
        "peptide_disjoint_verdict": "fail",
        "physical_specimen_identity": "unknown",
        "method": "independent standard-library raw-row join, identity, partition, coverage and overlap verification; no Hitlist imports",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(json.loads(args.receipt.read_text()))
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
