# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Tests for the mirrored data-asset registry + datacache fetch (#303)."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from hitlist import downloads

_EXPECTED_EXCLUDED = {
    "bekker_jensen_2017_peptides.csv.gz",
    "bekker_jensen_2017_protein_abundance.csv.gz",
    "ccle_nusinow_2020.csv.gz",
    "strazar_2023_hla2.csv",
    "abelin_2019_maptac_class2.csv",
}
_PKG_DATA = Path(__file__).resolve().parent.parent / "hitlist" / "data"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def test_registry_well_formed():
    assets = downloads.data_assets()
    assert len(assets) >= 26
    for name, meta in assets.items():
        assert meta["sha256"] and len(meta["sha256"]) == 64
        assert meta["url"].endswith(name)
        assert meta["source"]
    # the five large files must be flagged for exclusion from the wheel
    excluded = {n for n, m in assets.items() if not m["bundled"]}
    assert excluded == _EXPECTED_EXCLUDED


def test_registry_sha256_matches_packaged_files():
    """Every registry sha256 must match the file currently in the source tree —
    catches registry drift / a forgotten regen after editing a data CSV."""
    assets = downloads.data_assets()
    checked = 0
    for sub in ("bulk_proteomics", "line_expression", "peptide_attributions", "supplementary"):
        for p in (_PKG_DATA / sub).glob("*.csv*"):
            if p.name in assets:
                assert _sha256(p) == assets[p.name]["sha256"], f"sha256 drift for {p.name}"
                checked += 1
    assert checked >= 26


def test_view_membership():
    assert "strazar_2023_hla2.csv" in downloads.EXTERNAL_DATA_ASSETS
    assert "nope.csv" not in downloads.EXTERNAL_DATA_ASSETS


def test_packaged_or_fetched_prefers_local(tmp_path):
    local = tmp_path / "x.csv"
    local.write_text("hi")
    got = downloads.packaged_or_fetched(local, "strazar_2023_hla2.csv")
    assert got == local  # never fetched — local copy wins


def test_packaged_or_fetched_fetches_when_absent(monkeypatch, tmp_path):
    called = {}

    def fake_fetch(filename, *, force=False, verbose=True):
        called["filename"] = filename
        return tmp_path / "cached.csv"

    monkeypatch.setattr(downloads, "fetch_data_asset", fake_fetch)
    got = downloads.packaged_or_fetched(tmp_path / "missing.csv", "ccle_nusinow_2020.csv.gz")
    assert called["filename"] == "ccle_nusinow_2020.csv.gz"
    assert got == tmp_path / "cached.csv"


def test_fetch_data_asset_unknown_raises():
    with pytest.raises(KeyError, match="unknown data asset"):
        downloads.fetch_data_asset("not_a_real_asset.csv")


def test_fetch_data_asset_checksum_mismatch_raises(monkeypatch, tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("corrupt")  # wrong content → sha256 won't match the registry

    monkeypatch.setattr("datacache.fetch_file", lambda *a, **k: str(bad))
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        downloads.fetch_data_asset("ccle_nusinow_2020.csv.gz", verbose=False)


def test_fetch_all_iterates_every_asset(monkeypatch, tmp_path):
    fetched = []

    def fake_fetch(filename, *, force=False, verbose=True):
        fetched.append(filename)
        return tmp_path / filename

    monkeypatch.setattr(downloads, "fetch_data_asset", fake_fetch)
    # avoid stat() on non-existent files in the summary line
    out = downloads.fetch_all_data_assets(verbose=False)
    assert set(fetched) == set(downloads.data_assets())
    assert set(out) == set(downloads.data_assets())


# ── Packaging + loader fallback for peptide_attributions (#347, #348) ──


def test_bundled_registry_assets_have_a_package_data_glob():
    """#347: setuptools drops any data subdirectory with no ``package-data``
    glob, so ``peptide_attributions/`` was absent from both the wheel and
    the sdist and a clean install raised FileNotFoundError on PMID
    31844290.  Every bundled registry asset needs a matching glob."""
    from pathlib import Path

    import pytest

    import hitlist

    tomllib = pytest.importorskip(
        "tomllib", reason="stdlib tomllib is 3.11+; the glob is verified on newer runners"
    )

    root = Path(hitlist.__file__).resolve().parent.parent
    pyproject = root / "pyproject.toml"
    if not pyproject.is_file():  # installed (non-source) checkout
        pytest.skip("pyproject.toml not present in this install")

    cfg = tomllib.loads(pyproject.read_text())
    globs = cfg["tool"]["setuptools"]["package-data"]["hitlist"]
    excluded = set(cfg["tool"]["setuptools"]["exclude-package-data"]["hitlist"])

    data_dir = Path(hitlist.__file__).resolve().parent / "data"
    for asset in data_dir.rglob("*"):
        if not asset.is_file() or asset.suffix not in (".csv", ".gz", ".yaml"):
            continue
        rel = asset.relative_to(data_dir.parent).as_posix()
        if rel.removeprefix("hitlist/") in excluded or rel in excluded:
            continue
        covered = any(
            Path(rel).match(g) or Path(rel.removeprefix("hitlist/")).match(g) for g in globs
        )
        assert covered, f"{rel} has no package-data glob — it will be dropped from the wheel"


def test_asset_path_falls_back_to_the_mirror(monkeypatch, tmp_path):
    """#348: curation._data_path is a bare join with no fallback, so a
    registry asset missing from the install raised FileNotFoundError out
    of pd.read_csv instead of being fetched, unlike bulk_proteomics."""
    from hitlist import curation

    fetched = tmp_path / "sarkizova_2020_patient_cohort.csv"
    fetched.write_text("peptide,sample_label\n")

    calls = []

    def fake_fetch(filename):
        calls.append(filename)
        return fetched

    monkeypatch.setattr("hitlist.downloads.fetch_data_asset", fake_fetch)
    monkeypatch.setattr(
        "hitlist.curation._data_path", lambda rel: str(tmp_path / "definitely-absent" / rel)
    )

    out = curation._asset_path("peptide_attributions/sarkizova_2020_patient_cohort.csv")
    assert out == str(fetched)
    # The registry is keyed by bare filename, not by the relative path.
    assert calls == ["sarkizova_2020_patient_cohort.csv"]


def test_asset_path_prefers_the_installed_copy(tmp_path):
    """When the file ships with the install, no fetch is attempted."""
    from hitlist import curation

    real = curation._asset_path("peptide_attributions/sarkizova_2020_patient_cohort.csv")
    assert real.endswith("sarkizova_2020_patient_cohort.csv")
