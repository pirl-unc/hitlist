"""Independent synthetic arithmetic and source-quality checks, not biological fits."""

import json
import math

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from hitlist import intact_degradomics as source
from hitlist import intact_organ_transfer as organs
from hitlist.evidence_recipe import RecipeError, verify_recipe_package
from hitlist.evidence_store import EvidenceStore
from hitlist.provenance import file_digest
from tests import test_intact_degradomics as fixtures

original_source = fixtures.original


@pytest.fixture
def organ_store(original_source):
    assets, package, contract = original_source
    sheets, inventory = {}, []
    for organ_index, organ in enumerate(organs.ORGANS):
        for minutes in (0, 30, 60, 90, 120):
            for replicate in (1, 2):
                if organ == "Spleen" and minutes == 30 and replicate == 2:
                    continue
                label = f"{organ}_20S_aSyn_{minutes}min_{replicate}"
                gain = (organ_index + 1) * replicate * (minutes + 1)
                # Vary the zero-time ratios across replicates so geometric/log
                # averaging is distinguishable from an arithmetic ratio average.
                products = [
                    ("ACDEFGHIK", 1, 9, 10),
                    ("ACDE", 1, 4, 100 * replicate * 10 ** (minutes / 60)),
                ]
                if replicate == 1:
                    products.append(("CDEF", 2, 5, 50 * 10 ** (-minutes / 60)))
                if minutes == 0:
                    products.append(("DEFG", 3, 6, 30))  # Baseline only.
                else:
                    products.append(("GHIK", 6, 9, 25))  # No zero baseline.
                rows = [fixtures.HEADERS]
                for mass_id, (sequence, start, end, value) in enumerate(products, 1):
                    missing_ids = organ == "Brain" and minutes == 0 and replicate == 2
                    row = fixtures.search_rows(intensity=value * gain, missing_ids=missing_ids)[1]
                    row[6] = mass_id
                    if not missing_ids:
                        row[0] = mass_id
                    row[14], row[15], row[17] = start, end, sequence
                    rows.append(row)
                sheets[label] = rows
                inventory.append(
                    {
                        "fileName": f"F3_AlphaSyn_{organ}20S_{minutes}min_R{replicate}.RAW",
                        "fileCategory": {"value": "RAW"},
                    }
                )
    fixtures.write_workbook(assets / list(contract["search_workbooks"])[2], sheets)
    (assets / "files-project-route.json").write_text(json.dumps(inventory))
    for asset in contract["assets"]:
        asset.update(file_digest(assets / asset["filename"]))
    source.CONTRACT.write_text(json.dumps(contract))
    source.write_intact_degradomics_package(assets, package)
    return EvidenceStore.open(package)


def recipe():
    return {
        "schema_version": 1,
        "name": "organ_transfer_test",
        "registered_recipe": organs.RECIPE_ID,
    }


def rows(directory, name):
    return pq.read_table(directory / f"{name}.parquet").to_pylist()


def test_public_recipe_baseline_arithmetic_weights_and_target_separation(organ_store, tmp_path):
    assert any(
        r["contract"]["recipe_id"] == organs.RECIPE_ID
        for r in organ_store.catalog()["registered_recipes"]
    )
    output = tmp_path / "recipe"
    organ_store.materialize(recipe(), output)
    manifest = verify_recipe_package(output, source_package=organ_store.directory)
    assert not manifest["coverage"]["support_passed"]
    examples = rows(output / "typed", "examples")
    baselines = {r["assignment_id"]: r for r in rows(output / "typed", "baselines")}
    assert len(examples) == 35
    assert len(baselines) == 13  # Brain zero R2 has no source input IDs.
    assert {r["peptide_sequence"] for r in examples} == {"ACDE", "CDEF"}
    for r in examples:
        if r["peptide_sequence"] == "ACDE":
            replicate = int(r["replicate_label_raw"].removeprefix("R"))
            baseline_shift = 0 if r["organ_label"] == "Brain" else math.log10(2) / 2
            expected = r["time_minutes"] / 60 + math.log10(replicate) - baseline_shift
        else:
            expected = -r["time_minutes"] / 60
        assert r["target_change"] == pytest.approx(expected)
        ids = json.loads(r["baseline_assignment_ids_json"])
        assert len(ids) == r["n_baseline_observations"]
        assert all(baselines[i]["curve_id"] == r["curve_id"] for i in ids)
        assert all(baselines[i]["time_minutes"] == 0 for i in ids)
        assert r["baseline_log10_ratio"] == pytest.approx(
            sum(baselines[i]["target_log10_ratio"] for i in ids) / len(ids)
        )
    for split in ("test", "extrapolation"):
        selected = [r for r in examples if r["split"] == split]
        assert sum(r["weight"] for r in selected) == pytest.approx(1)
        for organ in organs.ORGANS:
            assert sum(r["weight"] for r in selected if r["organ_label"] == organ) == pytest.approx(
                1 / 3
            )
            for peptide in ("ACDE", "CDEF"):
                assert sum(
                    r["weight"]
                    for r in selected
                    if r["organ_label"] == organ and r["peptide_sequence"] == peptide
                ) == pytest.approx(1 / 6)
    covariates = pq.read_table(output / "typed/covariates.parquet")
    assert not set(covariates.column_names) & organs.LABEL_FIELDS
    assert "baseline_log10_ratio" in covariates.column_names
    decisions = rows(output / "typed", "decisions")
    assert all(
        r["quantitative_value"] is None for r in decisions if r["source_view_role"] != "organs"
    )
    late = [
        r
        for r in decisions
        if r["source_view_role"] == "organs" and r["peptide_sequence"] == "GHIK"
    ]
    assert late and all("no_eligible_zero_baseline" in json.loads(r["reasons_json"]) for r in late)
    with pytest.raises(FileExistsError):
        organ_store.materialize(recipe(), output)


@pytest.mark.parametrize(
    "mutation", ["ambiguous", "modified", "missing_parent", "duplicate", "bad_link", "mass_error"]
)
def test_assignment_failures_exclude_without_repairs(organ_store, tmp_path, monkeypatch, mutation):
    original = organs._rows

    def changed(directory, name):
        for row in original(directory, name):
            if name == "assignments":
                if mutation == "missing_parent" and row["peptide_sequence"] == "ACDEFGHIK":
                    row = {**row, "span_status": "conflict"}
                elif row["peptide_sequence"] == "ACDE":
                    changes = {
                        "ambiguous": {"n_assignments_replayed": 2},
                        "modified": {"modifications_json": "{}"},
                        "bad_link": {"mass_link_status": "unresolved"},
                        "mass_error": {"mass_error_ppm": 10.01},
                    }
                    row = {**row, **changes.get(mutation, {})}
                    if mutation == "duplicate":
                        yield {**row, "assignment_id": row["assignment_id"] + "duplicate"}
            yield row

    monkeypatch.setattr(organs, "_rows", changed)
    organs.compile_kinetics(organ_store.directory / source.ROOT / "typed", tmp_path / "changed")
    examples = rows(tmp_path / "changed", "examples")
    if mutation == "missing_parent":
        assert not examples
    else:
        assert {r["peptide_sequence"] for r in examples} == {"CDEF"}


def test_gain_response_cancellation_future_independence_and_zero_rejection(organ_store, tmp_path):
    tables = organ_store.directory / source.ROOT / "typed"
    organs.compile_kinetics(tables, tmp_path / "original")
    # A copy of typed software-fixture tables only; no scientific package mutated.
    changed = tmp_path / "tables"
    changed.mkdir()
    for name in organs.TABLES:
        table = pq.read_table(tables / f"{name}.parquet")
        if name == "quantities":
            source_rows = table.to_pylist()
            assignments = {r["measurement_id"]: r for r in rows(tables, "assignments")}
            for r in source_rows:
                a = assignments.get(r["measurement_id"])
                if a and a["peptide_sequence"] == "ACDE":
                    r["quantitative_value"] *= 37
            table = pa.Table.from_pylist(source_rows, schema=table.schema)
        pq.write_table(table, changed / f"{name}.parquet")
    organs.compile_kinetics(changed, tmp_path / "scaled")
    before = rows(tmp_path / "original", "examples")
    after = rows(tmp_path / "scaled", "examples")
    assert [r["assignment_id"] for r in before] == [r["assignment_id"] for r in after]
    assert [r["target_change"] for r in before] == pytest.approx(
        [r["target_change"] for r in after]
    )
    # Change only future quantities; supplied zero baselines must be identical.
    acq = {r["acquisition_id"]: r for r in rows(tables, "acquisitions")}
    table = pq.read_table(changed / "quantities.parquet")
    records = table.to_pylist()
    for r in records:
        a = assignments.get(r["measurement_id"])
        if a and acq.get(a["acquisition_id"], {}).get("time_minutes", 0) > 0:
            if a["peptide_sequence"] == "ACDE":
                r["quantitative_value"] *= 1000
            elif a["peptide_sequence"] == "CDEF":
                r["quantitative_value"] = 0
    pq.write_table(
        pa.Table.from_pylist(records, schema=table.schema), changed / "quantities.parquet"
    )
    organs.compile_kinetics(changed, tmp_path / "future")
    assert rows(tmp_path / "scaled", "baselines") == rows(tmp_path / "future", "baselines")
    assert {r["peptide_sequence"] for r in rows(tmp_path / "future", "examples")} == {"ACDE"}


def test_acquisition_reuse_is_excluded_regardless_of_claim_status():
    views = {
        "f2": {"view_id": "f2", "source_view_role": "kinetics", "acquisition_id": "old"},
        "f3": {"view_id": "f3", "source_view_role": "organs", "acquisition_id": "new"},
    }
    acquisitions = {
        "old": {"raw_file": "old.RAW"},
        "new": {"raw_file": "new.RAW", "organ_label": "Brain", "time_minutes": 0},
    }
    assert not organs.acquisition_scope(views, acquisitions, [])["f3"]
    claim = {
        "assertion_predicate": "repeated_acquisition_source_views",
        "source_subject_id": "f3",
        "target_subject_id": "f2",
        "assertion_status": "conflicting_values",
    }
    assert (
        "source_backed_reused_view" in organs.acquisition_scope(views, acquisitions, [claim])["f3"]
    )
    acquisitions["new"]["raw_file"] = "old.RAW"
    assert "F2_acquisition_overlap" in organs.acquisition_scope(views, acquisitions, [])["f3"]


@pytest.mark.parametrize(
    "table_name,field", [("examples", "target_change"), ("baselines", "baseline_log10_ratio")]
)
def test_rehashed_tampering_rejected(organ_store, tmp_path, table_name, field):
    output = tmp_path / "recipe"
    organ_store.materialize(recipe(), output)
    path = output / "typed" / (table_name + ".parquet")
    table = pq.read_table(path)
    records = table.to_pylist()
    records[0][field] += 1
    pq.write_table(pa.Table.from_pylist(records, schema=table.schema), path)
    manifest = json.loads((output / "manifest.json").read_text())
    manifest["files"][str(path.relative_to(output))] = file_digest(path)
    (output / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(RecipeError, match="rows differ"):
        verify_recipe_package(output, source_package=organ_store.directory)
