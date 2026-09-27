"""Regressions for ``downloads.data_dir()``'s resolution order (#291).

hitlist historically kept everything in ``~/.hitlist`` while
``fetch_data_asset`` wrote to datacache's cache dir, so a user had two data
locations and the CLI printed only one.  #291 moved the default onto the
openvax-ecosystem convention (``datacache.get_data_dir``) *without* moving
anybody's corpus: an install that already has a populated ``~/.hitlist`` keeps
using it.

Every test here fakes ``$HOME`` — the real ``~/.hitlist`` and the real
``~/Library/Caches/hitlist`` must never be read, created or written by the
suite, and ``_isolated_home`` asserts that isolation rather than assuming it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hitlist import downloads


@pytest.fixture
def _isolated_home(tmp_path, monkeypatch):
    """Point every home-derived path at ``tmp_path`` and clear session state.

    ``legacy_data_dir()`` goes through ``Path.home()``; datacache's default
    goes through ``appdirs``, which reads ``$HOME`` (macOS) or
    ``$XDG_CACHE_HOME``/``$HOME`` (Linux).  Both are patched, and the resulting
    datacache dir is asserted to land under ``tmp_path`` so a platform whose
    cache location we did not anticipate fails loudly instead of quietly
    touching the developer's real cache.
    """
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: home)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("XDG_CACHE_HOME", str(home / ".cache"))
    monkeypatch.delenv("HITLIST_DATA_DIR", raising=False)
    monkeypatch.setattr(downloads, "_override_data_dir", None)
    monkeypatch.setattr(downloads, "_legacy_data_dir_notified", False)
    assert home in downloads.default_data_dir().parents
    return home


def _populate_legacy(home: Path, name: str = "manifest.json") -> Path:
    legacy = home / ".hitlist"
    legacy.mkdir(parents=True, exist_ok=True)
    (legacy / name).write_text("{}\n")
    return legacy


def test_populated_legacy_dir_beats_the_datacache_default(_isolated_home, capsys):
    """An existing corpus in ``~/.hitlist`` keeps being used — nobody rebuilds."""
    legacy = _populate_legacy(_isolated_home)
    assert downloads.data_dir() == legacy
    assert downloads.data_dir_origin() == "legacy"
    assert downloads.data_dir() != downloads.default_data_dir()


@pytest.mark.parametrize("marker", sorted(downloads._LEGACY_DATA_DIR_MARKERS))
def test_every_declared_marker_makes_the_legacy_dir_populated(_isolated_home, marker, capsys):
    """Each documented marker file on its own is enough to hold the location."""
    legacy = _populate_legacy(_isolated_home, marker)
    assert downloads.legacy_data_dir_is_populated()
    assert downloads.data_dir() == legacy


def test_absent_legacy_dir_does_not_win(_isolated_home):
    assert not (_isolated_home / ".hitlist").exists()
    assert not downloads.legacy_data_dir_is_populated()
    assert downloads.data_dir() == downloads.default_data_dir()
    assert downloads.data_dir_origin() == "default"


def test_empty_legacy_dir_does_not_win(_isolated_home):
    """``data_dir()`` used to ``mkdir`` on every call, so empty ones abound."""
    (_isolated_home / ".hitlist").mkdir()
    assert not downloads.legacy_data_dir_is_populated()
    assert downloads.data_dir() == downloads.default_data_dir()


def test_legacy_dir_holding_only_empty_subdirs_does_not_win(_isolated_home):
    """``_proteomes_dir()`` / ``genes._cache_path()`` create these eagerly."""
    legacy = _isolated_home / ".hitlist"
    (legacy / "proteomes").mkdir(parents=True)
    (legacy / "gene_cache").mkdir(parents=True)
    assert not downloads.legacy_data_dir_is_populated()
    assert downloads.data_dir() == downloads.default_data_dir()


def test_a_marker_that_is_a_directory_does_not_count(_isolated_home):
    """Only files prove data; a directory named ``manifest.json`` does not."""
    (_isolated_home / ".hitlist" / "manifest.json").mkdir(parents=True)
    assert not downloads.legacy_data_dir_is_populated()


def test_env_var_beats_a_populated_legacy_dir(_isolated_home, monkeypatch, tmp_path):
    _populate_legacy(_isolated_home)
    configured = tmp_path / "configured"
    monkeypatch.setenv("HITLIST_DATA_DIR", str(configured))
    assert downloads.data_dir() == configured
    assert downloads.data_dir_origin() == "env"


def test_env_var_is_used_verbatim_not_as_a_datacache_root(_isolated_home, monkeypatch, tmp_path):
    """``$HITLIST_DATA_DIR`` has always meant *this* directory.

    ``datacache.get_data_dir(subdir="hitlist", envkey=...)`` would return
    ``$HITLIST_DATA_DIR/hitlist`` instead, silently orphaning every existing
    configured corpus — so the env var is resolved before datacache sees it.
    """
    configured = tmp_path / "configured"
    monkeypatch.setenv("HITLIST_DATA_DIR", str(configured))
    assert downloads.data_dir() == configured
    assert downloads.data_dir().name != "hitlist"


def test_override_beats_everything(_isolated_home, monkeypatch, tmp_path):
    _populate_legacy(_isolated_home)
    monkeypatch.setenv("HITLIST_DATA_DIR", str(tmp_path / "from-env"))
    monkeypatch.setattr(downloads, "_override_data_dir", tmp_path / "from-override")
    assert downloads.data_dir() == tmp_path / "from-override"
    assert downloads.data_dir_origin() == "override"


def test_set_data_dir_beats_everything(_isolated_home, monkeypatch, tmp_path):
    _populate_legacy(_isolated_home)
    monkeypatch.setenv("HITLIST_DATA_DIR", str(tmp_path / "from-env"))
    monkeypatch.setattr(downloads, "_override_data_dir", None)
    downloads.set_data_dir(tmp_path / "explicit")
    assert downloads.data_dir() == tmp_path / "explicit"


def test_legacy_notice_fires_once_per_process(_isolated_home, capsys):
    legacy = _populate_legacy(_isolated_home)
    for _ in range(5):
        downloads.data_dir()
    err = capsys.readouterr().err
    assert err.count(str(legacy)) == 1
    assert "legacy data directory" in err
    assert "fully supported" in err
    # The notice has to say where to migrate to, or it is not actionable.
    assert str(downloads.default_data_dir()) in err
    assert "HITLIST_DATA_DIR" in err


def test_no_notice_when_the_legacy_dir_is_not_in_use(_isolated_home, capsys):
    downloads.data_dir()
    assert capsys.readouterr().err == ""


def test_resolution_creates_no_directory(_isolated_home):
    """Asking "where is it?" must not touch the filesystem (#579).

    A ``mkdir`` here would also make an empty ``~/.hitlist`` satisfy the
    legacy rule on the very next call.
    """
    before = sorted(p for p in _isolated_home.rglob("*"))
    resolved = downloads.data_dir()
    downloads.data_dir_origin()
    downloads.default_data_dir()
    downloads.data_asset_dir()
    downloads.legacy_data_dir_is_populated()
    assert not resolved.exists()
    assert sorted(p for p in _isolated_home.rglob("*")) == before


def test_ensure_data_dir_creates_on_the_write_path(_isolated_home):
    resolved = downloads.data_dir()
    assert not resolved.exists()
    assert downloads.ensure_data_dir() == resolved
    assert resolved.is_dir()


def test_data_asset_dir_matches_where_datacache_actually_writes(_isolated_home):
    """The asset dir is computed by datacache itself so it cannot drift."""
    import datacache

    expected = Path(datacache.expected_path(filename="x.csv", subdir="hitlist"))
    assert downloads.data_asset_dir() == expected.parent


def test_asset_dir_ignores_hitlist_data_dir(_isolated_home, monkeypatch, tmp_path):
    """datacache's ``build_path``/``expected_path`` take no ``envkey``.

    Reporting the asset dir as moving with ``$HITLIST_DATA_DIR`` would be a
    lie, and is exactly the kind of drift #291 is about.
    """
    monkeypatch.setenv("HITLIST_DATA_DIR", str(tmp_path / "configured"))
    assert downloads.data_asset_dir() == downloads.default_data_dir()


def test_fresh_install_unifies_the_two_locations(_isolated_home):
    """With nothing overridden, indexes and mirrored assets share one dir."""
    assert downloads.data_dir() == downloads.data_asset_dir()


# ── The write path now owns directory creation ──────────────────────────────
#
# ``data_dir()`` no longer mkdir's, so every public entry point that writes a
# canonical artifact has to create the directory itself. Each of these fails
# with FileNotFoundError if its writer stops doing so.


@pytest.fixture
def _unborn_data_dir(tmp_path, monkeypatch):
    """Point the data dir at a directory that does not exist yet."""
    target = tmp_path / "unborn"
    monkeypatch.setattr(downloads, "_override_data_dir", target)
    assert not target.exists()
    return target


def test_atomic_write_parquet_creates_the_data_dir(_unborn_data_dir):
    """Covers observations.parquet, binding.parquet and line_expression.parquet."""
    import pandas as pd

    from hitlist.observations import observations_path
    from hitlist.parquet_io import atomic_write_parquet

    out = observations_path()
    atomic_write_parquet(pd.DataFrame({"peptide": ["SIINFEKL"]}), out)
    assert out.is_file()
    assert pd.read_parquet(out)["peptide"].tolist() == ["SIINFEKL"]


def test_build_bulk_proteomics_creates_the_data_dir(_unborn_data_dir, monkeypatch):
    """``build_bulk_proteomics`` is public and may run before anything else."""
    from hitlist import bulk_proteomics as bp
    from hitlist.builder import _bulk_proteomics_path, build_bulk_proteomics

    empty = __import__("pandas").DataFrame()
    monkeypatch.setattr(bp, "_load_ccle", lambda *a, **k: empty)
    monkeypatch.setattr(bp, "_load_bj", lambda *a, **k: empty)
    monkeypatch.setattr(bp, "_load_bj_protein", lambda *a, **k: empty)

    build_bulk_proteomics()
    assert _bulk_proteomics_path().is_file()


def test_build_peptide_mappings_creates_the_data_dir(_unborn_data_dir):
    """``build_peptide_mappings`` writes its sidecar + meta into the data dir."""
    import pandas as pd

    from hitlist.mappings import build_peptide_mappings, mappings_meta_path

    obs = pd.DataFrame(
        {
            "peptide": ["SIINFEKL"],
            # No registry entry and no fetching, so nothing is searched and the
            # build reduces to "write an empty sidecar".
            "source_organism": ["Nonexistent organism sp."],
            "mhc_species": ["Nonexistent organism sp."],
            "pmid": [1],
        }
    )
    out = build_peptide_mappings(obs_override=obs, fetch_missing=False, verbose=False, force=True)
    assert out.is_file()
    assert mappings_meta_path().is_file()
