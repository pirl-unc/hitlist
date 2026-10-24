"""PXD009438 retains source sample labels without claiming physical identity."""

import csv
import json

import pyarrow.parquet as pq
import pytest

from hitlist.mapp_pbmc_evidence import write_pbmc_tables
from hitlist.material_context import links_with_minimum_precision


def _write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _fixture(root):
    (root / "contracts").mkdir(parents=True)
    (root / "contracts/contract.json").write_text(json.dumps({"assertions": []}))
    labels = [
        f"{985 + index}_{group}_{number}"
        for index, (group, number) in enumerate(
            [("CTRL", number) for number in range(1, 7)]
            + [("LUPUS", number) for number in range(1, 9)]
        )
    ]
    directory = root / "assets/txt"
    _write(
        directory / "summary.txt",
        [
            {
                "Raw file": f"raw_{label.split('_', 1)[0]}",
                "Experiment": label,
                "Enzyme mode": "Unspecific",
            }
            for label in labels
        ]
        + [{"Raw file": "Total", "Experiment": "", "Enzyme mode": ""}],
    )
    peptides = []
    evidence = []
    for peptide_id in ("0", "1"):
        peptide = {
            "id": peptide_id,
            "Sequence": "REPEATED",
            "Proteins": "P1",
            "Leading razor protein": "P1",
            "Protein group IDs": "0",
            "Amino acid before": "-",
            "Amino acid after": "A",
            "Start position": "1",
            "End position": "8",
            "PEP": "0.001",
            "Reverse": "",
            "Potential contaminant": "",
            "Evidence IDs": ";".join(
                str(len(labels) * int(peptide_id) + index) for index in range(len(labels))
            ),
            **{"Intensity " + label: str(int(peptide_id) + 1) for label in labels},
        }
        peptides.append(peptide)
        for index, label in enumerate(labels):
            evidence_id = str(len(labels) * int(peptide_id) + index)
            evidence.append(
                {
                    "id": evidence_id,
                    "Peptide ID": peptide_id,
                    "Experiment": label,
                    "Sequence": "REPEATED",
                    "Raw file": f"raw_{label.split('_', 1)[0]}",
                    "Protein group IDs": "0",
                    "Intensity": str(int(peptide_id) + 1),
                    "Type": "MSMS",
                    "MS/MS Count": "1",
                    "Modified sequence": "_REPEATED_",
                    "Modifications": "Unmodified",
                    "PEP": "0.001",
                    "Reverse": "",
                    "Potential contaminant": "",
                }
            )
    _write(directory / "peptides.txt", peptides)
    _write(
        directory / "proteinGroups.txt",
        [
            {
                "id": "0",
                "Protein IDs": "P1",
                "Peptide IDs": "0;1",
                "Reverse": "",
                "Potential contaminant": "",
            }
        ],
    )
    _write(directory / "evidence.txt", evidence)
    _write(directory / "parameters.txt", [{"Parameter": "Protein FDR", "Value": "1"}])
    return directory


def _table(root, name):
    return pq.read_table(root / "typed" / f"{name}.parquet").to_pylist()


def test_source_labels_are_queryable_without_physical_identity(tmp_path):
    _fixture(tmp_path)
    result = write_pbmc_tables(tmp_path)
    assert result["n_peptides"] == 2
    assert result["n_evidence"] == 28
    assert result["n_runs"] == 14
    assert result["relation_rows"]["quantity_results"] == 28
    assert result["relation_rows"]["material_contexts"] == 31
    assert result["relation_rows"]["material_links"] == 30
    assert [row["sequence_raw"] for row in _table(tmp_path, "peptide_rows")] == [
        "REPEATED",
        "REPEATED",
    ]
    contexts = _table(tmp_path, "material_contexts")
    links = _table(tmp_path, "material_links")
    samples = [row for row in contexts if row["context_kind"] == "source_sample"]
    assert len(samples) == 14
    assert {row["condition_text"] for row in samples} == {"CTRL", "LUPUS"}
    assert all(row["specimen_id"] is row["aliquot_id"] is None for row in contexts)
    assert not links_with_minimum_precision(links, "exact_specimen")
    assert len(links_with_minimum_precision(links, "study_arm")) == 14


def test_unreplayable_quantity_fails_before_admission(tmp_path):
    directory = _fixture(tmp_path)
    path = directory / "peptides.txt"
    with path.open() as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    first = next(name for name in rows[0] if name.startswith("Intensity "))
    rows[0][first] = "99"
    _write(path, rows)
    with pytest.raises(ValueError, match="does not replay"):
        write_pbmc_tables(tmp_path)


def test_source_adapter_is_registered_separately_from_other_mapp_panels():
    from hitlist.evidence_bindings import materialized_bindings
    from hitlist.external_evidence import _adapters

    assert "pxd009438" in _adapters()
    bindings = materialized_bindings(9, appended_source_ids=["pxd009438"])
    assert "pbmc_mapp_channel" in bindings
    assert "external_pbmc_proteasome_fragment_quantity" in bindings
    assert "mapp_channel" not in bindings
