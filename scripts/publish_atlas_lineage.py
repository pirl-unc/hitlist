"""Publish the independently verified primary Atlas donor cohort."""

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

from datacache import install_bundle, materialize

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "tasks/atlas-lineage-results"
TAG = "atlas-donor-lineage-v1"
RELEASE = "https://github.com/pirl-unc/hitlist/releases/tag/" + TAG
BASE = "https://github.com/pirl-unc/hitlist/releases/download/" + TAG + "/"
CACHE = Path.home() / ".hitlist/atlas-lineage"


def pin(path):
    path = Path(path)
    return {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "size": path.stat().st_size}


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def gh(*args):
    return subprocess.check_output(["gh", *args], text=True)


def prepare():
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    assert not subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)
    receipt = json.loads((FOLDER / "audit-receipt.json").read_text())
    verification = json.loads((FOLDER / "independent-verification.json").read_text())
    assert verification["verified"]
    files = {name: Path(path) for name, path in receipt["paths"].items()}
    for name, path in files.items():
        assert pin(path) == {
            "sha256": receipt["files"][name]["sha256"],
            "size": receipt["files"][name]["size_bytes"],
        }
    files.update(
        {
            "README.md": FOLDER / "README.md",
            "independent-verification.json": FOLDER / "independent-verification.json",
            "verify.py": ROOT / "scripts/verify_atlas_lineage.py",
            "plan.md": ROOT / "tasks/human-lineage-spec.md",
            "publish.py": Path(__file__),
            "initial_failed_recipe.py": ROOT / "scripts/audit_atlas_lineage.py",
            "initial-verification-failure.txt": FOLDER / "initial-verification-failure.txt",
            "initial-audit-receipt.json": FOLDER / "initial-audit-receipt.json",
        }
    )
    names = [*files, "publication.json", "SHA256SUMS"]

    def build(inputs, outputs):
        for name, path in inputs.items():
            shutil.copyfile(path, outputs[name])
        write(
            outputs["publication.json"],
            {
                "schema_version": 1,
                "release": RELEASE,
                "executed_audit_commit": receipt["commit"],
                "publication_commit": commit,
                "verification": verification,
                "source_policy": "Direct primary donors are partitioned without overlap. Peptide overlap and unresolved physical specimen/acquisition identities are explicit. Historical IEDB cohorts are unchanged and cannot be appended as independent evidence.",
                "files": {name: {"url": BASE + name, **pin(outputs[name])} for name in files},
            },
        )
        Path(outputs["SHA256SUMS"]).write_text(
            "".join(
                pin(outputs[name])["sha256"] + "  " + name + "\n"
                for name in names
                if name != "SHA256SUMS"
            )
        )

    paths = materialize(
        CACHE / "publication" / commit,
        {name: {"path": str(path), **pin(path)} for name, path in files.items()},
        transform={
            "version": "atlas-donor-lineage-publication-v1",
            "options": {"commit": commit},
        },
        outputs={name: {} for name in names},
        builder=build,
    )
    result = {
        "release": RELEASE,
        "publication_commit": commit,
        "actual_audit_commit": receipt["commit"],
        "files": {name: {"url": BASE + name, **pin(path)} for name, path in paths.items()},
        "paths": {name: str(path) for name, path in paths.items()},
    }
    write(FOLDER / "publication.json", result)
    print(json.dumps({"prepared": len(paths), "commit": commit}))


def publish():
    receipt = json.loads((FOLDER / "publication.json").read_text())
    for name, path in receipt["paths"].items():
        assert pin(path) == {key: receipt["files"][name][key] for key in ("sha256", "size")}
    gh(
        "release",
        "create",
        TAG,
        "--repo",
        "pirl-unc/hitlist",
        "--target",
        receipt["publication_commit"],
        "--draft",
        "--title",
        "Primary HLA Ligand Atlas donor-linked cohort and independence audit",
        "--notes-file",
        str(FOLDER / "README.md"),
        *receipt["paths"].values(),
    )
    release = next(
        item
        for item in json.loads(gh("api", "repos/pirl-unc/hitlist/releases?per_page=100"))
        if item["tag_name"] == TAG
    )
    assert {asset["name"] for asset in release["assets"]} == set(receipt["files"])
    for asset in release["assets"]:
        expected = receipt["files"][asset["name"]]
        assert asset["size"] == expected["size"]
        assert asset["digest"] == "sha256:" + expected["sha256"]
    gh(
        "api",
        "--method",
        "PATCH",
        "repos/pirl-unc/hitlist/releases/" + str(release["id"]),
        "-F",
        "draft=false",
        "-f",
        "make_latest=false",
    )
    downloaded = install_bundle(
        CACHE / "public-verification" / receipt["publication_commit"],
        receipt["files"],
        download_options={"show_progress": False},
    )
    receipt["downloaded"] = {name: str(path) for name, path in downloaded.items()}
    receipt["n_verified_public_assets"] = len(downloaded)
    write(FOLDER / "publication.json", receipt)
    print(json.dumps({"release": RELEASE, "n_verified_public_assets": len(downloaded)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "publish"])
    args = parser.parse_args()
    (prepare if args.action == "prepare" else publish)()
