"""Independent read-only replay of the original public workbooks when provisioned."""

import json
import math
import os
from collections import Counter
from pathlib import Path

import openpyxl
import pyarrow.parquet as pq
import pytest

from hitlist.evidence_recipe import verify_recipe_package
from hitlist.evidence_store import EvidenceStore
from hitlist.perturbation_contrast import RECIPE_ID
from hitlist.perturbation_quantities import (
    ROOT,
    source_contract,
    write_perturbation_quantity_package,
)


@pytest.mark.integration
def test_original_reporter_workbooks_roundtrip(tmp_path):
    source = os.environ.get("HITLIST_PERTURBATION_SOURCE_DIR")
    if not source:
        pytest.skip(
            "Original perturbation workbooks are provisioned in the source qualification job"
        )
    destination = tmp_path / "original-source-package"
    write_perturbation_quantity_package(source, destination)
    store = EvidenceStore.open(destination)
    output = tmp_path / "original-contrasts"
    store.materialize(
        {
            "schema_version": 1,
            "name": "original source qualification",
            "registered_recipe": RECIPE_ID,
        },
        output,
    )
    manifest = verify_recipe_package(output)
    assert manifest["coverage"]["n_contrasts"] == 17763
    assert manifest["coverage"]["n_automatic_training_admissions"] == 0
    typed = destination / ROOT / "typed"
    forms = {row["form_id"]: row for row in pq.read_table(typed / "forms.parquet").to_pylist()}
    quantities = pq.read_table(typed / "measurements.parquet").to_pylist()
    assert Counter(row["quantity_kind"] for row in quantities) == {
        "hipmhc_normalized_reporter": 106578,
        "raw_reporter": 15558,
        "mean_adjusted_reporter": 42765,
    }
    assert sum(row["form_status"] == "source_phosphorylated" for row in forms.values()) == 8
    by_locator = {row["source_locator"]: row for row in quantities}
    assert len(by_locator) == len(quantities)
    contrasts = {
        row["form_id"]: row for row in pq.read_table(output / "typed/contrasts.parquet").to_pylist()
    }
    assert all(row["cellular_mhc_genotype"] is None for row in contrasts.values())
    assert all(row["candidate_mhc_set"] for row in contrasts.values())
    summaries = {
        row["source_locator"]: row
        for row in pq.read_table(typed / "reported_summaries.parquet").to_pylist()
    }
    by_record = {row["source_record_id"]: row for row in forms.values()}
    records = pq.read_table(
        typed / "source_records.parquet",
        columns=["source_record_id", "source_locator", "row_disposition"],
    ).to_pylist()
    record_by_locator = {row["source_locator"]: row for row in records}
    assert sum(row["row_disposition"] == "excluded_aggregate" for row in records) == 1
    max_delta = 0.0
    n_forms = n_values = 0
    contract = source_contract()
    for asset in contract["assets"]:
        book = openpyxl.load_workbook(
            Path(source) / asset["filename"], read_only=True, data_only=True
        )
        for spec in (
            item for item in contract["comparisons"] if item["asset"] == asset["filename"]
        ):
            sheet = book[spec["sheet"]]
            n_sheet_forms = 0
            for ordinal, cells in enumerate(
                sheet.iter_rows(min_row=spec["header_row"] + 1, values_only=True),
                spec["header_row"] + 1,
            ):
                raw_form = cells[0]
                if not raw_form or raw_form == "Grand Total":
                    continue
                n_forms += 1
                n_sheet_forms += 1
                locator = asset["filename"] + "#" + sheet.title + "!"
                record = record_by_locator[locator + str(ordinal)]
                form = by_record[record["source_record_id"]]
                assert form["modified_sequence_raw"] == raw_form
                assert form["peptide_sequence"] == raw_form.replace("[", "").replace("]", "")
                normalized = []
                for quantity in spec["quantities"]:
                    for column in quantity["columns"]:
                        value = cells[openpyxl.utils.column_index_from_string(column) - 1]
                        row = by_locator[locator + column + str(ordinal)]
                        assert row["quantity_kind"] == quantity["quantity_kind"]
                        assert row["quantitative_value"] == value
                        assert value > 0 and math.isfinite(value)
                        n_values += 1
                        if quantity["quantity_kind"] == "hipmhc_normalized_reporter":
                            normalized.append(value)
                assert len(normalized) == 6
                for statistic, column in spec["statistics"].items():
                    value = cells[openpyxl.utils.column_index_from_string(column) - 1]
                    summary = summaries[locator + column + str(ordinal)]
                    assert summary["statistic_name"] == statistic
                    if isinstance(value, (int, float)) and math.isfinite(value):
                        assert summary["quantitative_value"] == value
                    else:
                        assert summary["quantitative_value"] is None
                independent_effect = math.log2(sum(normalized[3:]) / sum(normalized[:3]))
                contrast = contrasts[form["form_id"]]
                assert contrast["target_log2_fold_change"] == pytest.approx(
                    independent_effect, abs=1e-12
                )
                reported = cells[
                    openpyxl.utils.column_index_from_string(spec["statistics"]["log2_fold_change"])
                    - 1
                ]
                assert contrast["reported_log2_fold_change"] == reported
                max_delta = max(max_delta, abs(independent_effect - reported))
                assert contrast["n_control_samples"] == contrast["n_treated_samples"] == 3
            assert n_sheet_forms == spec["n_forms"]
        book.close()
    assert n_forms == 17763 and n_values == 164901
    assert max_delta < 1.1e-9
    receipt = {
        "n_form_comparisons": n_forms,
        "n_quantity_cells": n_values,
        "max_reported_log2_difference": max_delta,
        "coverage": manifest["coverage"],
        "source_assets": [
            {k: asset[k] for k in ("filename", "sha256", "size_bytes")}
            for asset in contract["assets"]
        ],
    }
    print(json.dumps(receipt, sort_keys=True))
