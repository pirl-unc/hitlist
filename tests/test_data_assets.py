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


def _packaging_config():
    """Parse the package-data include/exclude globs out of pyproject.toml."""
    try:
        import tomllib
    except ModuleNotFoundError:  # Python < 3.11
        tomli = pytest.importorskip("tomli", reason="need tomllib/tomli to read pyproject.toml")
        loads = tomli.loads
    else:
        loads = tomllib.loads
    cfg = loads((Path(__file__).resolve().parent.parent / "pyproject.toml").read_text())
    setuptools_cfg = cfg["tool"]["setuptools"]
    return (
        setuptools_cfg["package-data"]["hitlist"],
        setuptools_cfg.get("exclude-package-data", {}).get("hitlist", []),
    )


def _shipped_paths():
    """Paths setuptools would install, per the pyproject globs.

    Uses ``Path.glob`` rather than ``fnmatch`` so ``*`` does not cross a
    directory separator -- otherwise ``data/*.yaml`` would appear to cover
    files in every subdirectory.
    """
    include, exclude = _packaging_config()
    pkg_root = _PKG_DATA.parent
    included = {p for g in include for p in pkg_root.glob(g)}
    excluded = {p for g in exclude for p in pkg_root.glob(g)}
    return included - excluded


def test_bundled_flag_matches_packaging_globs():
    """Every ``bundled: true`` asset must actually be shipped, and every
    ``bundled: false`` asset must not be.

    Regression test for the packaging gap that dropped
    ``data/peptide_attributions/`` from the wheel and sdist: the directory had
    no ``package-data`` glob, so a file the registry advertised as bundled was
    absent from both published artifacts, and every peptide-level call that
    touched PMID 31844290 raised ``FileNotFoundError`` on a clean install.

    The existing sha256 test cannot catch this -- it globs the source tree,
    where the file is always present.
    """
    shipped = _shipped_paths()
    checked = 0
    for name, meta in downloads.data_assets().items():
        matches = list(_PKG_DATA.rglob(name))
        assert len(matches) == 1, f"expected exactly one {name} under {_PKG_DATA}, got {matches}"
        assert (matches[0] in shipped) is bool(meta["bundled"]), (
            f"{name}: bundled={meta['bundled']} but "
            f"{'not ' if meta['bundled'] else ''}shipped by the pyproject globs"
        )
        checked += 1
    assert checked >= 26
