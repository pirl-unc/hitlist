"""Verify and recover the already completed human export after receipt failure #695."""

from __future__ import annotations

import json
import time
from pathlib import Path

from qualify_human_cohort import OPTIONS, RUN_ID, digest, encode, publish_receipt

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT / "tasks/human-cohort-results/original-materialization.json"
REMOTE = (
    f"hitlist-qualification/runs/{RUN_ID}/human-cohort/"
    "0c0d3f8546913331a790414f545794ce75b65f91f98cf564ad1f6c824df658a9/"
    "artifacts/bundles/2026-10-16T03-49-57Z"
)


def main():
    import modal
    from datacache import materialize, validate_file

    from hitlist import verify_training_bundle
    from hitlist.downloads import data_dir

    manifest = json.loads(ORIGINAL.read_text())
    options = manifest["definition"]["transform"]["options"]
    if options["parent_run_id"] != RUN_ID or options["training_options"] != OPTIONS:
        raise ValueError("Original export does not match the registered cohort")
    manifest_pin = digest(ORIGINAL)
    destination = data_dir() / "human-cohort-recovery" / manifest_pin["sha256"]
    volume = modal.Volume.from_name("presto-data", create_if_missing=False)
    original_log = ROOT / "tasks/human-cohort-results/original-launcher.log"
    started = time.monotonic()

    def recover(inputs, outputs):
        for name, pin in manifest["outputs"].items():
            with Path(outputs[name]).open("wb") as output:
                for chunk in volume.read_file(f"{REMOTE}/{name}"):
                    output.write(chunk)
            validate_file(outputs[name], pin["sha256"], pin["size"])
        verify_training_bundle(Path(outputs["manifest.json"]).parent)

    paths = materialize(
        destination / "artifacts",
        {"original.json": {"path": str(ORIGINAL), "sha256": manifest_pin["sha256"]}},
        transform={"version": "recover-original-cohort-v1", "options": {"remote": REMOTE}},
        outputs=manifest["outputs"],
        builder=recover,
    )
    native = verify_training_bundle(Path(paths["manifest.json"]).parent)
    result = {
        "status": "recovered_verified_export",
        "original_run_status": "receipt_publication_failed",
        "original_export_producer": "hitlist 1.76.0",
        "original_export_runtime_seconds": None,
        "original_export_peak_rss_bytes": None,
        "original_materialization": manifest_pin,
        "original_recipe_sha256": options["recipe_sha256"],
        "recovery_runtime_seconds": time.monotonic() - started,
        "recovery_recipe": digest(__file__),
        "options": options,
        "native_coverage": native["coverage"],
        "coverage": json.loads(Path(paths["coverage.json"]).read_text()),
        "limitations": [
            "Original receipt failed; original runtime and RSS were not recovered.",
            "Baseline producer 1.76.0 predates the curation corrections in #694/#696.",
            "No partitions or independent-experiment training-readiness claim.",
        ],
        "artifacts": {name: {"path": str(path), **digest(path)} for name, path in paths.items()},
    }
    receipts = publish_receipt(destination / "receipts", result, original_log)
    print(
        encode(
            {"receipt": result, "receipt_files": {k: str(v) for k, v in receipts.items()}}
        ).decode()
    )


if __name__ == "__main__":
    main()
