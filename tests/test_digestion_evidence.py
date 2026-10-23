"""Source semantics survive decoding, normalization and canonical querying."""

import csv
import gzip
import io
import json
import zipfile

import openpyxl
import pyarrow.parquet as pq
import pytest

from hitlist import digestion_evidence as digestion


def _csv(rows):
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def _fixture(root, monkeypatch):
    """RData payloads here are fake CSV; real R decoding is a source-run gate."""
    assets = root / "assets"
    (assets / "substrate-fastas").mkdir(parents=True)
    (root / "contracts").mkdir()
    (assets / "substrate-fastas/H4.fasta").write_text(">test unknown species\nACDEFG\n")
    book = openpyxl.Workbook()
    sheet = book.active
    sheet.title = "Main_searches"
    sheet.append(
        ["raw_file", "protein_name", "digestTime", "biological_replicate", "Mass spectrometry"]
    )
    sheet.append(["control.raw", "H4 substrate only", 0, 1, "instrument"])
    sheet = book.create_sheet("Contaminant_DB_generation")
    sheet.append(["raw_file", "Enzyme"])
    sheet.append(["sample_Try_01.raw", "Chymotrypsin"])
    book.save(assets / "Identifier.xlsx")
    contract = json.loads((digestion.DATA / "digestion_contract.json").read_text())
    contract["construct_files"] = {"aSPIRE_H4": "H4.fasta"}
    contract["expected"] = {
        "n_runs": 1,
        "n_metadata_rows": 2,
        "n_constructs": 1,
        "n_contexts": 1,
        "n_metadata_enzyme_conflicts": 1,
        "n_archives": 1,
        "n_archive_members": 6,
        "n_ms1_rows": 1,
        "n_unlinked_ms1_rows": 0,
        "n_raw_quantities": 1,
        "n_filtered_vectors": 1,
        "n_identifications": 1,
        "n_final_rows": 2,
        "n_final_values": 4,
        "n_annotation_only_rows": 1,
        "n_time_conflicts": 1,
    }
    (root / "contracts/contract.json").write_text(json.dumps(contract))
    raw = {
        "substrateSeq": "ACDEFG",
        "source": "control",
        "digestTime": "1",
        "pepSeq": "ACD",
        "biological_replicate": "b1",
        "intensity": "12.5",
        "assignments": "one;two",
        "frac_background": "0.1",
        "noise": "2",
        "mean_techRep": "12.5",
        "mean_bioRep": "NA",
    }
    filtered = {
        "pepSeq": "ACD",
        "biological_replicate": "b1",
        "technical_replicate": "2",
        "times_pasted": "0;0;1;2",
        "int_pasted": "1;2;3;4",
        "productType": "spliced",
        "spliceType": "cis",
        "positions": "1-2;4-4",
    }
    final = {
        "substrateSeq": "ACDEFG",
        "pepSeq": "ACD",
        "biological_replicate": "b1",
        "digestTimes": "0;0;1;2",
        "intensities": "0;0;NaN;10",
        "productType": "spliced",
        "spliceType": "cis",
        "positions": "1-2;4-4",
    }
    ms1 = {
        "FileName": "control.raw",
        "TotalAreaMs1": "0",
        "TotalBackgroundMs1": "NA",
        "PeptideSequence": "ACD",
        "PeptideModifiedSequence": "AC[57]D",
        "ProteinName": "one",
        "FragmentIon": "precursor",
        "PrecursorCharge": "2",
    }
    identification = {
        "source": "control",
        "pepSeq": "ACD",
        "modifiedSequence": "AC[57]D",
        "ID": "one",
        "productType": "spliced",
        "positions": "1-2;4-4",
        "qValue": "0.01",
        "postErrProb": "NaN",
    }
    inner = io.BytesIO()
    with zipfile.ZipFile(inner, "w") as archive:
        # Deliberately final before conflicting raw: context must still block admission.
        archive.writestr(
            "output/finalKinetics.csv",
            _csv(
                [
                    final,
                    {
                        **final,
                        "biological_replicate": "NA",
                        "digestTimes": "NA",
                        "intensities": "NA",
                    },
                ]
            ),
        )
        archive.writestr("output/QUANTITIES_raw.RData", _csv([raw]))
        archive.writestr("output/QUANTITIES_filtered.RData", _csv([filtered]))
        archive.writestr("output/MS1_HPR.csv", _csv([ms1]))
        archive.writestr("output/ASSIGNMENTS.csv", _csv([identification]))
        archive.writestr("output/log.txt", "Retained original-only method details\n")
    with zipfile.ZipFile(assets / "aSPIRE_outputs_zip.zip", "w") as archive:
        archive.writestr("aSPIRE_H4.zip", inner.getvalue())

    def fake_decode(rscript, exporter, original, exported):
        with gzip.open(exported, "wb") as stream:
            stream.write(original.read_bytes())

    monkeypatch.setattr(digestion, "_decode_rdata", fake_decode)
    monkeypatch.setattr(digestion.shutil, "which", lambda _: "/fake/Rscript")
    monkeypatch.setattr(digestion.subprocess, "check_output", lambda *a, **k: "fixture R\n")
    return root


def _rows(root, name):
    return pq.read_table(root / "typed" / f"{name}.parquet").to_pylist()


def test_source_states_conflicts_controls_and_modifications_survive(tmp_path, monkeypatch):
    root = _fixture(tmp_path, monkeypatch)
    result = digestion.write_tables(root)
    digestion.DigestionAdapter().verify(root, result)
    quantity = _rows(root, "raw_quantities")[0]
    assert quantity["quantity_time_hours"] == 1 and quantity["metadata_time_hours"] == 0
    assert quantity["control_role"] == "substrate_only"
    assert quantity["biological_replicate_raw"] == "b1"
    assert quantity["metadata_biological_replicate_raw"] == "1"
    assert quantity["time_status"] == "source_metadata_conflict"
    assert quantity["measurement_unit"] == digestion.MS1_UNIT
    assert quantity["quantity_scale"] == "aSPIRE_raw_peptide_run_quantity"
    context = _rows(root, "contexts")[0]
    assert context["training_status"] == "unresolved_source_time_conflict"
    assert context["enzyme_species"] == "Homo sapiens"
    assert context["substrate_species"] is None
    assert len(_rows(root, "conflicts")) == 2
    members = _rows(root, "members")
    assert len(members) == 6
    assert {row["materialization_status"] for row in members} == {
        "typed_raw_quantities",
        "typed_filtered_vectors",
        "typed_ms1",
        "typed_identifications",
        "typed_final_vectors",
        "original_asset_only",
    }
    vectors = _rows(root, "final_vectors")
    assert [r["row_status"] for r in vectors] == ["reported_vector", "annotation_only"]
    assert vectors[0]["product_type_raw"] == "spliced"
    values = _rows(root, "final_values")
    assert [r["slot_0based"] for r in values] == [0, 1, 2, 3]
    assert [r["intensity"] for r in values] == [0, 0, None, 10]
    assert [r["value_status"] for r in values] == [
        "postprocessing_forced_zero",
        "postprocessing_forced_zero",
        "postprocessing_missing_unresolved_contributors",
        "source_positive",
    ]
    assert {r["measurement_unit"] for r in values} == {digestion.MS1_UNIT}
    assert {r["quantity_scale"] for r in values} == {"aSPIRE_background_adjusted_biological_mean"}
    assert all(r["time_status"] == "repeated_source_time_slots" for r in values)
    assert len({r["source_record_id"] for r in values}) == 4
    assert _rows(root, "ms1")[0]["modified_sequence"] == "AC[57]D"
    assert _rows(root, "ms1")[0]["run_link_status"] == "matched_main_search"
    assert _rows(root, "ms1")[0]["quantity_scale"] == "Skyline_MS1_peak_area"
    assert _rows(root, "ms1")[0]["ms1_background"] is None
    assert _rows(root, "identifications")[0]["run_link_status"] == "matched_main_search"
    assert _rows(root, "identifications")[0]["posterior_error_probability"] is None
    metadata = json.loads(_rows(root, "metadata")[0]["fields_json"])
    assert metadata["columns"][0] == "raw_file"
    assert metadata["values"][0] == "control.raw"
    assert len(json.loads((root / "rdata_decoding.json").read_text())["exports"]) == 2


@pytest.mark.parametrize(
    "value,expected",
    [
        ("0", (0.0, "source_zero")),
        ("NaN", (None, "source_missing")),
        ("NA", (None, "source_missing")),
        ("Inf", (None, "source_nonfinite")),
        ("-1", (-1.0, "source_negative")),
        ("3.5", (3.5, "source_positive")),
    ],
)
def test_missing_is_not_zero(value, expected):
    assert digestion.numeric(value) == expected


def test_bad_time_vectors_and_unknown_runs_reject():
    assert digestion.run_link_status("", None) == "blank_original_source_name"
    assert digestion.run_link_status("other", None) == "outside_main_search_roster"
    assert digestion.run_link_status("known", {"run_id": "x"}) == "matched_main_search"
    with pytest.raises(ValueError, match="lengths differ"):
        digestion.vector_values(
            {"biological_replicate": "1", "digestTimes": "0;1", "intensities": "1"}
        )
    with pytest.raises(ValueError, match="Annotation-only"):
        digestion.vector_values(
            {"biological_replicate": "NA", "digestTimes": "0", "intensities": "1"}
        )
    with pytest.raises(ValueError, match="invalid time"):
        digestion.vector_values(
            {"biological_replicate": "1", "digestTimes": "-1", "intensities": "1"}
        )
    with pytest.raises(ValueError, match="Unknown digestion acquisition"):
        digestion.raw_quantity({"source": "unregistered"}, {})


def test_receipt_change_and_missing_R_fail_before_creating_tables(tmp_path, monkeypatch):
    root = _fixture(tmp_path, monkeypatch)
    (root / "contracts/source_assets.json").write_text("tampered")
    with pytest.raises(ValueError, match="original-asset contract"):
        digestion.DigestionAdapter().write(root)
    monkeypatch.setattr(digestion.shutil, "which", lambda _: None)
    with pytest.raises(RuntimeError, match="requires Rscript"):
        digestion.write_tables(root)
    assert not (root / "typed").exists()


def test_optional_source_catalog_covers_native_relations_and_links():
    from hitlist.evidence_bindings import materialized_bindings
    from hitlist.evidence_catalog import DIGESTION_RELATIONS, _validate_catalog

    _validate_catalog()
    assert "digestion_raw_quantities" not in materialized_bindings(9)
    bindings = materialized_bindings(9, appended_source_ids=[digestion.SOURCE_ID])
    for name, (_, aliases) in DIGESTION_RELATIONS.items():
        assert "digestion_" + name in bindings
        assert set(aliases.values()) <= set(digestion.SCHEMAS[name].names)
