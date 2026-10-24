"""MAPP preserves selection, original identities and measurement scope."""

import csv
import json
import math

import openpyxl
import pyarrow.parquet as pq
import pytest

from hitlist.mapp_evidence import write_mapp_tables
from hitlist.material_context import links_with_minimum_precision


def _tsv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _source(root):
    assets = root / "assets"
    (assets / "txt").mkdir(parents=True)
    (assets / "publisher").mkdir()
    (root / "contracts").mkdir()
    (root / "contracts/contract.json").write_text(json.dumps({"assertions": []}))
    runs = []
    for condition in ("UT", "TNF+IFN", "CHX", "NO-DSP", "NO-INH", "BEADS"):
        for replicate in (1, 2, 3):
            experiment = f"001_CP-HEK-T+I_{condition}_{replicate}"
            runs.append(
                {
                    "Experiment": experiment,
                    "Raw file": f"{condition}-{replicate}",
                    "Enzyme mode": "Unspecific",
                }
            )
    evidence, peptides = [], []
    for pid, sequence in [("0", "PEPTIDE"), ("1", "SEQUENCE")]:
        row = {
            "id": pid,
            "Sequence": sequence,
            "Proteins": "P1;P1-2",
            "Leading razor protein": "P1",
            "Protein group IDs": "0",
            "Amino acid before": "-",
            "Amino acid after": "A",
            "Start position": "1",
            "End position": str(len(sequence)),
            "PEP": "0.01",
            "Reverse": "",
            "Potential contaminant": "",
            "Evidence IDs": "",
        }
        ids = []
        for run in runs:
            experiment = run["Experiment"]
            # Three background values 4; run value 6 is filtered, 10 is retained.
            value = 4 if "_BEADS_" in experiment else (6 if "_CHX_" in experiment else 10)
            if "_BEADS_" not in experiment:
                row["Intensity " + experiment.replace("T+I_", "T+I-MAPPbeadFilt_")] = str(
                    0 if value == 6 or pid == "1" else value
                )
            if pid == "1":
                continue
            eid = str(len(evidence))
            ids.append(eid)
            evidence.append(
                {
                    "id": eid,
                    "Peptide ID": pid,
                    "Sequence": sequence,
                    "Experiment": experiment,
                    "Raw file": run["Raw file"],
                    "Intensity": str(value),
                    "Modified sequence": "_" + sequence + "_",
                    "Modifications": "Unmodified",
                    "Type": "MULTI-MATCH",
                    "MS/MS Count": "0",
                    "PEP": "0.01",
                    "Protein group IDs": "0",
                    "Reverse": "",
                    "Potential contaminant": "",
                }
            )
        row["Evidence IDs"] = ";".join(ids)
        peptides.append(row)
    _tsv(assets / "txt/summary.txt", runs)
    _tsv(assets / "txt/peptides.txt", peptides)
    evidence[0]["PEP"] = "NaN"
    _tsv(assets / "txt/evidence.txt", evidence)
    _tsv(
        assets / "txt/proteinGroups.txt",
        [
            {
                "id": "0",
                "Protein IDs": "P1;P1-2",
                "Peptide IDs": "0;1",
                "Reverse": "",
                "Potential contaminant": "",
            },
            {
                "id": "",
                "Protein IDs": "",
                "Peptide IDs": "",
                "Reverse": "",
                "Potential contaminant": "",
            },
        ],
    )
    _tsv(
        assets / "txt/parameters.txt",
        [{"Parameter": "Fasta file", "Value": "Human_UP_2016_02.fasta"}],
    )
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "Sheet1"
    sheet.append(["title"])
    sheet.append([None, None, "Log10 Transformed", None, "Imputed"])
    sheet.append(["Sequence", "Gene", "Median UT", "Median TNF+IFN", "Median UT", "Median TNF+IFN"])
    sheet.append([None])
    sheet.append(["PEPTIDE", "G1", math.log10(10), math.log10(10), 99.0, 12.0])
    sheet.append(["SEQUENCE", "G1", None, None, 9.0, 9.0])
    workbook.save(assets / "publisher/suppdata2.xlsx")
    workbook = openpyxl.Workbook()
    workbook.active.append(["selected gene panel"])
    workbook.save(assets / "publisher/suppdata3.xlsx")
    return assets


def _table(root, name):
    return pq.read_table(root / "typed" / f"{name}.parquet").to_pylist()


def test_source_zeros_and_imputation_are_not_biological_absence(tmp_path):
    _source(tmp_path)
    counts = write_mapp_tables(tmp_path)
    assert counts["n_filtered_zeros_with_positive_evidence"] == 3
    assert counts["n_quarantined_groups"] == 1
    quantities = _table(tmp_path, "quantity_measurements")
    assert sum(q["value_status"] == "filtered_zero" for q in quantities) == 3
    assert sum(q["value_status"] == "no_reported_evidence" for q in quantities) == 18
    assert sum(q["value_status"] == "reported_zero_no_positive_evidence" for q in quantities) == 15
    assert all(q["value"] == 0 for q in quantities if q["value_status"] == "filtered_zero")
    assert all(
        q["absolute_replay_error"] == 0
        for q in quantities
        if q["representation"] == "bead_filtered"
    )
    publisher = _table(tmp_path, "publisher_quantities")
    assert len(publisher) == 8
    imputed = [q for q in publisher if q["representation"] == "imputed_header_conflict"]
    assert len(imputed) == 4
    assert {q["condition"] for q in imputed} == {None}
    assert {q["value_status"] for q in imputed} == {"imputed_conflicting_condition_header"}
    assert [q["value"] for q in imputed] == [99.0, 12.0, 9.0, 9.0]
    assert all(q["recomputed_value"] is None for q in imputed)
    assert all(
        c["material_link_precision"] == "acquisition_run_unresolved_specimen"
        for c in _table(tmp_path, "channels")
    )
    assert {
        c["condition_role"] for c in _table(tmp_path, "channels") if c["condition"] == "NO-INH"
    } == {"procedural_control"}
    assert {e["type_raw"] for e in _table(tmp_path, "evidence")} == {"MULTI-MATCH"}
    assert _table(tmp_path, "evidence")[0]["pep"] is None
    assert _table(tmp_path, "evidence")[0]["pep_raw"] == "NaN"
    assert _table(tmp_path, "evidence")[0]["pep_status"] == "unreported_source_nan"
    assert len(_table(tmp_path, "quantity_results")) == 36
    raw = _table(tmp_path, "source_records")
    invalid = [
        r
        for r in raw
        if r["asset_path"] == "txt/proteinGroups.txt" and json.loads(r["fields_json"])["id"] == ""
    ]
    assert len(invalid) == 1
    assert {p["parent_labels_raw"] for p in _table(tmp_path, "peptide_rows")} == {"P1;P1-2"}
    references = {e["evidence_id"] for e in _table(tmp_path, "evidence")} | {
        q["measurement_id"] for q in quantities
    }
    assert all(
        row["input_measurement_id"] in references for row in _table(tmp_path, "derivation_inputs")
    )
    contexts = _table(tmp_path, "material_contexts")
    links = _table(tmp_path, "material_links")
    assert len(contexts) == 25
    assert len(links) == 24
    assert {row["condition_text"] for row in contexts if row["context_kind"] == "study_arm"} == {
        "UT",
        "TNF+IFN",
        "CHX",
        "NO-DSP",
        "NO-INH",
        "BEADS",
    }
    acquisition = [row for row in contexts if row["context_kind"] == "source_acquisition"]
    assert len(acquisition) == 18
    assert all(row["specimen_id"] is row["aliquot_id"] is None for row in acquisition)
    assert not links_with_minimum_precision(links, "exact_specimen")
    assert len(links_with_minimum_precision(links, "study_arm")) == 18


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("filter", "bead filter"),
        ("identity", "relationship mismatch"),
        ("condition", "run condition"),
        ("numeric", "numeric source value"),
    ],
)
def test_unexpected_source_changes_fail(tmp_path, mutation, match):
    assets = _source(tmp_path)
    name = (
        "summary"
        if mutation == "condition"
        else ("peptides" if mutation == "filter" else "evidence")
    )
    path = assets / f"txt/{name}.txt"
    with path.open() as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    if mutation == "filter":
        column = next(k for k in rows[0] if k.startswith("Intensity "))
        rows[0][column] = "999"
    elif mutation == "identity":
        rows[0]["Sequence"] = "OTHER"
    elif mutation == "condition":
        rows[0]["Experiment"] = rows[0]["Experiment"].replace("_UT_", "_UNKNOWN_")
    else:
        rows[0]["Intensity"] = "nan"
    _tsv(path, rows)
    with pytest.raises(ValueError, match=match):
        write_mapp_tables(tmp_path)
