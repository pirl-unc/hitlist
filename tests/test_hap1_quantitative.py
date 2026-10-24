"""Synthetic source fixtures exercise curation semantics, never predictive fitting."""

import csv
import io
import json
import zipfile

import openpyxl
import pyarrow.parquet as pq
import pytest

from hitlist import hap1_quantitative as hap1
from hitlist.evidence_store import EvidenceStore
from hitlist.provenance import file_digest


def _csv(headers, rows, delimiter=","):
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, delimiter=delimiter)
    writer.writerow(headers)
    writer.writerows(rows)
    return stream.getvalue()


def _workbook():
    book = openpyxl.Workbook()
    book.remove(book.active)
    for name, rows in {
        "immunopep_library_generation": [("wt", 1, "wt_dda", 1)],
        "immunopep_library_search": [
            ("wt", 1, "wt_dia1", 1),
            ("wt", 1, "wt_dia2", 2),
            ("TAP1", 1, "ko_dia", 1),
        ],
        "proteomics": [("wt", 1, "wt_p_1", None), ("TAP1", 1, "ko_p_1", None)],
    }.items():
        sheet = book.create_sheet(name)
        sheet.append(
            ["knockout_condition", "biological_replicate", "raw_file", "technical_replicate"]
        )
        for row in rows:
            sheet.append(row)
    stream = io.BytesIO()
    book.save(stream)
    return stream.getvalue()


@pytest.fixture
def source(tmp_path, monkeypatch):
    assets = tmp_path / "source"
    assets.mkdir()
    (assets / "article.xml").write_text("<article/>")
    (assets / "project.json").write_text("{}")
    (assets / "files-page0.json").write_text(
        json.dumps(
            [
                {"fileName": name + ".raw"}
                for name in ["wt_dda", "wt_dia1", "wt_dia2", "ko_dia", "wt-p-1", "ko-p-1"]
            ]
        )
    )
    (assets / "files-page1.json").write_text("[]")
    (assets / "samples.sdrf.tsv").write_text(
        "source name\tcomment[sdrf template]\tcomment[sdrf template]\nwt\tfirst\tsecond\n"
    )
    with zipfile.ZipFile(assets / "supplements.zip", "w") as archive:
        archive.writestr("mmc5.xlsx", _workbook())
        archive.writestr(
            "mmc6.csv",
            _csv(
                [
                    "peptide",
                    "mapped_proteins",
                    "wt_1_.sampling_density",
                    "TAP1_1_.sampling_density",
                ],
                [
                    ["AAAA", "sp|P1|TEST_HUMAN", "100", "NA"],
                    ["ACDE", "Biognosys|iRT", "0", "2"],
                    ["WWWW", "unknown", "NA", "4"],
                ],
            ),
        )
        archive.writestr(
            "mmc7.csv",
            _csv(
                ["peptide", "mapped_proteins", "wt_1_.sampling_density"],
                [["AAAA", "sp|P1|TEST_HUMAN", "50"]],
            ),
        )
        archive.writestr(
            "mmc8.csv",
            _csv(
                [
                    "protein_id",
                    "wt_1.intensity",
                    "TAP1_1.intensity",
                    "wt_1_.sampling_density",
                    "wt_1.protein_coverage_proteomics",
                ],
                [["sp|P1|TEST_HUMAN", "NA", "20", "0.5", "0.8"]],
            ),
        )
    with zipfile.ZipFile(assets / "Search.zip", "w") as archive:
        archive.writestr("Raw_file_list.xlsx", _workbook())
        archive.writestr(
            "Search_results_DIA.tsv",
            _csv(
                [
                    "Run",
                    "Stripped.Sequence",
                    "Modified.Sequence",
                    "Protein.Ids",
                    "Precursor.Id",
                    "Precursor.Charge",
                    "Precursor.Quantity",
                    "Precursor.Normalised",
                    "Q.Value",
                    "Global.Q.Value",
                    "PEP",
                ],
                [
                    [
                        "wt_dia1",
                        "AAAA",
                        "AAAA",
                        "P1",
                        "AAAA2",
                        "2",
                        "1000",
                        "200",
                        ".001",
                        ".02",
                        ".004",
                    ],
                    [
                        "unknown_run",
                        "WWWW",
                        "WWWW",
                        "unknown",
                        "WWWW3",
                        "3",
                        "2",
                        "1",
                        ".01",
                        ".01",
                        ".02",
                    ],
                ],
                delimiter="\t",
            ),
        )
        archive.writestr(
            "search_results_proteomics.tsv",
            _csv(
                [
                    "Protein",
                    "Combined Total Peptides",
                    "Indistinguishable Proteins",
                    "wt_p_1 Intensity",
                    "wt_p_1 MaxLFQ Intensity",
                    "ko_p_1 Intensity",
                    "ko_p_1 MaxLFQ Intensity",
                ],
                [["sp|P1|TEST_HUMAN", "1", "sp|P2|OTHER_HUMAN", "900", "100", "800", "20"]],
                delimiter="\t",
            ),
        )
        archive.writestr(
            "original.fasta",
            ">sp|P1|TEST_HUMAN OS=Homo sapiens OX=9606\nAAAAA\n>sp|P2|OTHER_HUMAN OX=9606\nCAAAA\n>Biognosys|iRT\nACDE\n>rev_Biognosys|iRT\nEDCA\n",
        )
    receipt_path = tmp_path / "assets.json"
    receipt_path.write_text(
        json.dumps(
            {
                "assets": [
                    {
                        "filename": path.name,
                        "source_relative_path": path.name,
                        "url": "https://example.invalid/fixture/" + path.name,
                        **file_digest(path),
                    }
                    for path in sorted(assets.iterdir())
                ]
            }
        )
    )
    contract = json.loads(hap1.CONTRACT.read_text())
    contract["contexts"] = {
        key: value for key, value in contract["contexts"].items() if key in ("wt", "TAP1")
    }
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps(contract))
    monkeypatch.setattr(hap1, "ASSETS", receipt_path)
    monkeypatch.setattr(hap1, "CONTRACT", contract_path)
    return assets


def test_public_package_preserves_quantities_missingness_context_and_origins(source, tmp_path):
    destination = tmp_path / "package"
    hap1.write_hap1_quantitative_package(source, destination)
    store = EvidenceStore.open(destination)
    assert set(store.bindings) == {"hap1_" + name for name in hap1.SCHEMAS}
    assert (
        store.catalog()["source_registry"]["shapiro2025_hap1"]["training_status"]
        == "deferred_endpoint_recipe"
    )
    rows = store.query(
        "hap1_quantities",
        select=[
            "quantitative_raw",
            "quantitative_value",
            "value_status",
            "native.quantity_kind",
            "sample_id",
            "native.source_field",
        ],
        limit=None,
    ).table.to_pylist()
    assert len(rows) == 11
    assert sum(row["value_status"] == "source_missing_NA" for row in rows) == 3
    zero = next(row for row in rows if row["value_status"] == "source_zero")
    assert zero["quantitative_value"] == 0
    assert all(row["quantitative_value"] is None for row in rows if row["quantitative_raw"] == "NA")
    protein_na = next(row for row in rows if row["native.source_field"] == "wt_1.intensity")
    deposited = store.query(
        "hap1_protein_quantities", select=["quantitative_value", "native.source_field"], limit=None
    ).table.to_pylist()
    assert protein_na["quantitative_value"] is None
    assert {row["native.source_field"]: row["quantitative_value"] for row in deposited}[
        "wt_p_1 MaxLFQ Intensity"
    ] == 100
    runs = store.query(
        "hap1_runs",
        select=["native.source_run_raw", "sample_id", "native.deposit_link_status"],
        limit=None,
    ).table.to_pylist()
    assert (
        next(row for row in runs if row["native.source_run_raw"] == "unknown_run")["sample_id"]
        is None
    )
    assert (
        next(row for row in runs if row["native.source_run_raw"] == "wt_p_1")[
            "native.deposit_link_status"
        ]
        == "unique_punctuation_candidate"
    )
    samples = store.query(
        "hap1_samples", select=["sample_id", "specimen_id", "aliquot_id"], limit=None
    ).table.to_pylist()
    assert len(samples) == 4
    assert all(row["specimen_id"] is None and row["aliquot_id"] is None for row in samples)
    references = store.query(
        "hap1_references", select=["reference_role"], limit=None
    ).table.to_pylist()
    assert {row["reference_role"] for row in references} == {
        "human_target",
        "spiked_standard",
        "reverse_decoy",
    }
    occurrences = store.query(
        "hap1_occurrences",
        select=["peptide_sequence", "mapping_position", "mapping_status"],
        limit=None,
    ).table.to_pylist()
    assert sum(row["peptide_sequence"] == "AAAA" for row in occurrences) == 6
    assert (
        next(row for row in occurrences if row["peptide_sequence"] == "WWWW")["mapping_status"]
        == "no_exact_reference_occurrence"
    )
    precursor = store.query(
        "hap1_precursors",
        select=["native.raw_quantity", "native.normalized_quantity", "native.confidence_json"],
        limit=1,
    ).table.to_pylist()[0]
    assert precursor["native.raw_quantity"] == 1000
    assert precursor["native.normalized_quantity"] == 200
    assert json.loads(precursor["native.confidence_json"])["Global.Q.Value"] == ".02"
    contextual = store.query(
        "hap1_quantities",
        select=["peptide_sequence", "context.condition_knockout_genes", "context.mhc_genotype"],
        joins=["context"],
        limit=None,
    ).table.to_pylist()
    assert {row["context.condition_knockout_genes"] for row in contextual} == {"none", "TAP1"}
    assert all("HLA-A*02:01" in row["context.mhc_genotype"] for row in contextual)
    density = store.query(
        "hap1_quantities",
        select=["context.measured_pool"],
        where={"native.quantity_kind": "protein_sampling_density"},
        joins=["context"],
    ).table.to_pylist()
    assert density == [{"context.measured_pool": "HLA_I_immunoprecipitated_peptides"}]
    sdrf = store.query(
        "hap1_source_records",
        select=["source_row_json"],
        where={"source_file": "samples.sdrf.tsv"},
    ).table.to_pylist()
    assert json.loads(sdrf[0]["source_row_json"]) == {
        "headers": ["source name", "comment[sdrf template]", "comment[sdrf template]"],
        "values": ["wt", "first", "second"],
    }


def test_replay_rejects_semantic_tampering_even_with_updated_artifact_hash(source, tmp_path):
    destination = tmp_path / "package"
    hap1.write_hap1_quantitative_package(source, destination)
    path = destination / "hap1/typed/quantities.parquet"
    table = pq.read_table(path)
    rows = table.to_pylist()
    rows[0]["quantitative_value"] = 999
    import pyarrow as pa

    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), path)
    manifest_path = destination / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"][str(path.relative_to(destination))] = file_digest(path)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="differs from source"):
        hap1.verify_hap1_quantitative_package(destination)


def test_changed_source_is_rejected_without_publishing_partial_package(source, tmp_path):
    (source / "article.xml").write_text("changed")
    with pytest.raises(ValueError, match="source mismatch"):
        hap1.write_hap1_quantitative_package(source, tmp_path / "package")
    assert not (tmp_path / "package").exists()
    assert not list(tmp_path.glob(".package-*"))


@pytest.mark.parametrize(
    "raw,status",
    [
        ("NA", "source_missing_NA"),
        ("0", "source_zero"),
        ("<10", "unparsed_source_text"),
        ("nan", "source_nonfinite"),
    ],
)
def test_numeric_status_does_not_invent_bounds(raw, status):
    assert hap1._number(raw)[1] == status


def test_ambiguous_filename_alias_does_not_select_a_run():
    aliases, status = hap1._candidate_aliases("a_b_c", {"a-b_c", "a_b-c"})
    assert len(aliases) == 2
    assert status == "ambiguous_candidates"
