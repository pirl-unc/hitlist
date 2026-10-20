"""The community SDRF is contextual evidence, never a peptide-to-run join."""

from collections import Counter

import pytest
from scripts.audit_atlas_crosswalk import build_crosswalk, key


def sdrf_row(source_name, file_name, *, hla_class="I", tissue="liver", donor="D1"):
    return {
        "source name": source_name,
        "characteristics[individual]": donor,
        "characteristics[organism part]": tissue,
        "factor value[human leukocyte antigen]": f"Human leukocyte antigen class {hla_class}",
        "comment[data file]": file_name,
        "comment[file uri]": "https://example.org/" + file_name,
        "comment[technical replicate]": "1",
    }


def record(file_name, category):
    return {
        "fileName": file_name,
        "accession": f"record:{file_name}",
        "fileCategory": {"value": category},
    }


def test_exact_files_are_distinct_from_qc_and_observation_lineage():
    rows = [
        sdrf_row("Liver-early", "early.mzML"),
        sdrf_row("Liver-late", "late.mzML"),
        sdrf_row("Liver-II", "class2.mzML", hla_class="II"),
        sdrf_row("Excluded", "excluded.mzML", tissue="heart"),
    ]
    inventory = [
        record("early.mzML", "PEAK"),
        record("late.mzML", "PEAK"),
        record("class2.mzML", "PEAK"),
        record("excluded.mzML", "PEAK"),
        record("early.raw", "RAW"),
        record("late.raw", "RAW"),
        record("unannotated.mzML", "PEAK"),
        record("result.mzid.gz", "RESULT"),
    ]
    counts = Counter({key("D1", "Liver", "HLA-I"): 7, key("D1", "Liver", "HLA-II"): 11})
    labels = dict.fromkeys(counts, "Liver")
    files, records, groups, summary = build_crosswalk(rows, inventory, counts, labels)
    assert len(files) == 4
    assert files[0]["raw_match_status"] == "exact_stem"
    assert files[2]["raw_match_status"] == "unresolved"
    assert files[3]["primary_group_status"] == "no_primary_group"
    assert records[-1]["sdrf_status"] == "not_sdrf_data_file"
    assert summary["n_unannotated_peak_records"] == 1
    assert summary["n_inventory_result_records_without_qc_join"] == 1
    assert summary["n_primary_groups_with_multiple_source_names"] == 1
    assert summary["n_candidate_peak_files_in_multiple_partitions"] == 0
    assert summary["verdicts"]["observation_to_acquisition"] == "inconclusive"
    assert summary["verdicts"]["physical_specimen_identity"] == "inconclusive"
    assert all(row["qc_inclusion_status"] == "unknown" for row in groups)
    assert all(row["observation_acquisition_status"] == "unknown" for row in groups)
    assert sum(row["n_observations"] for row in groups) == 18


@pytest.mark.parametrize(
    "defect", ["duplicate_sdrf_file", "missing_peak", "wrong_category", "bad_uri"]
)
def test_rejects_non_unique_or_unsubstantiated_file_joins(defect):
    row = sdrf_row("Liver", "one.mzML")
    rows = [row]
    inventory = [record("one.mzML", "PEAK")]
    if defect == "duplicate_sdrf_file":
        rows.append(sdrf_row("Other", "one.mzML"))
    elif defect == "missing_peak":
        inventory = []
    elif defect == "wrong_category":
        inventory[0]["fileCategory"]["value"] = "RAW"
    else:
        row["comment[file uri]"] = "https://example.org/other.mzML"
    with pytest.raises(ValueError):
        build_crosswalk(rows, inventory, Counter(), {})


def test_rejects_source_name_that_spans_donors():
    rows = [sdrf_row("Same", "one.mzML"), sdrf_row("Same", "two.mzML", donor="D2")]
    inventory = [record("one.mzML", "PEAK"), record("two.mzML", "PEAK")]
    with pytest.raises(ValueError, match="source name spans"):
        build_crosswalk(rows, inventory, Counter(), {})
