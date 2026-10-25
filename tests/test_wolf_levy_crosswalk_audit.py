import json
from pathlib import Path

AUDIT = (
    Path(__file__).parents[1]
    / "hitlist/data/evidence_revisions/wolf_levy_companion_crosswalk_audit.json"
)


def test_wolf_levy_crosswalk_audit_is_bounded_and_hash_pinned():
    audit = json.loads(AUDIT.read_text())
    assert audit["schema_version"] == 1
    assert audit["conclusion"] == "condition_specimen_aliquot_and_mapp_crosswalk_unresolved"
    by_project = {row["source_dataset"]: row for row in audit["project_file_inventory_audit"]}
    assert set(by_project) == {"PXD009360", "PXD009181", "PXD009112"}
    assert {project: row["n_raw_files"] for project, row in by_project.items()} == {
        "PXD009360": 9,
        "PXD009181": 24,
        "PXD009112": 14,
    }
    assert all(row["n_raw_files_with_sample_properties"] == 0 for row in by_project.values())
    assert all(len(row["files_sha256"]) == 64 for row in by_project.values())
    assert all("source backed" in rule for rule in audit["non_inference_rules"])
