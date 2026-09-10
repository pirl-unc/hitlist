"""Public, quiet cache-validity predicates (#448).

``observations.is_built()`` and ``mappings.is_mappings_built()`` are
existence-only, so a consumer gating on them serves a legacy artifact
forever. These predicates expose the verdict the builders reach before
deciding to skip, with none of their side effects: no output, no writes, no
download of an externalized curation asset.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pandas as pd
import pytest

import hitlist
from hitlist import builder, downloads
from hitlist.builder import _OBSERVATIONS_ARTIFACT_VERSION, _cache_is_valid, _meta_path
from hitlist.mappings import (
    _mapping_artifact_contract,
    _obs_fingerprint,
    build_peptide_mappings,
    mappings_cache_is_current,
    mappings_meta_path,
)
from hitlist.observations import observations_cache_is_current

# ── observations ────────────────────────────────────────────────────────────


def _seed_observation_cache(tmp_path, monkeypatch, *, artifact_version=None):
    """A registered source plus the four parquets and metadata that match them."""
    monkeypatch.setattr(downloads, "_override_data_dir", tmp_path)
    for name in ("observations", "binding", "bulk_proteomics", "line_expression"):
        builder._atomic_write_parquet(
            pd.DataFrame({"peptide": ["AAAAAAAAA"]}), tmp_path / f"{name}.parquet"
        )
    source = tmp_path / "iedb.csv"
    source.write_text("source")
    paths = {"iedb": source}
    monkeypatch.setattr(builder, "_source_paths", lambda: paths)
    _meta_path().write_text(
        json.dumps(
            {
                "artifact_version": (
                    _OBSERVATIONS_ARTIFACT_VERSION if artifact_version is None else artifact_version
                ),
                "sources": builder._source_fingerprints(paths, fetch_missing_assets=False),
                "parquets": builder._parquet_fingerprints(),
            }
        )
    )
    return paths


def _assert_silent(capsys):
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_observations_predicate_is_none_without_registered_sources(tmp_path, monkeypatch, capsys):
    """Validity is unknowable with no source; build_observations would raise here."""
    monkeypatch.setattr(downloads, "_override_data_dir", tmp_path)
    monkeypatch.setattr(builder, "_source_paths", lambda: {})

    assert observations_cache_is_current() is None
    _assert_silent(capsys)


def test_observations_predicate_true_on_current_cache(tmp_path, monkeypatch, capsys):
    _seed_observation_cache(tmp_path, monkeypatch)

    assert observations_cache_is_current() is True
    _assert_silent(capsys)


def test_observations_predicate_false_on_legacy_artifact_version(tmp_path, monkeypatch, capsys):
    """The case that motivated #448: a parquet exists but predates the schema."""
    _seed_observation_cache(
        tmp_path, monkeypatch, artifact_version=_OBSERVATIONS_ARTIFACT_VERSION - 1
    )

    assert observations_cache_is_current() is False
    _assert_silent(capsys)


def test_observations_predicate_false_when_a_parquet_is_replaced(tmp_path, monkeypatch):
    _seed_observation_cache(tmp_path, monkeypatch)
    builder._atomic_write_parquet(
        pd.DataFrame({"peptide": ["AAAAAAAAA", "CCCCCCCCC"]}), tmp_path / "observations.parquet"
    )

    assert observations_cache_is_current() is False


def test_observations_predicate_false_when_nothing_is_built(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(downloads, "_override_data_dir", tmp_path)
    source = tmp_path / "iedb.csv"
    source.write_text("source")
    monkeypatch.setattr(builder, "_source_paths", lambda: {"iedb": source})

    assert observations_cache_is_current() is False
    _assert_silent(capsys)


@pytest.fixture
def curation_referencing_uncached_asset(tmp_path, monkeypatch):
    """Curation whose ``peptide_attributions`` CSV is neither packaged nor cached."""
    from hitlist import cell_name_parser, curation

    data_root = tmp_path / "curation"
    data_root.mkdir()
    for name in ("pmid_overrides.yaml", "tissue_categories.yaml", "monoallelic_lines.yaml"):
        (data_root / name).write_bytes(Path(curation._data_path(name)).read_bytes())
    (data_root / "cell_lines.yaml").write_bytes(cell_name_parser._registry_path().read_bytes())
    (data_root / "pmid_overrides.yaml").write_text(
        "- pmid: 99999999\n"
        "  restriction_evidence: experimental\n"
        "  peptide_attributions: peptide_attributions/never_fetched_448.csv\n"
    )
    monkeypatch.setattr(curation, "_data_path", lambda name: str(data_root / name))
    monkeypatch.setattr(cell_name_parser, "_registry_path", lambda: data_root / "cell_lines.yaml")

    def refuse_network(filename, **_kw):
        raise AssertionError(f"tried to download {filename}")

    monkeypatch.setattr(downloads, "fetch_data_asset", refuse_network)
    return data_root


def test_observations_predicate_never_downloads(
    curation_referencing_uncached_asset, tmp_path, monkeypatch
):
    """A wheel install with an uncached attribution CSV: stale, not fetched.

    The builder's own validity check fetches the asset because a build needs
    its bytes; the predicate must reach its verdict without touching the
    network, and that verdict is the one a build would reach (fetching
    stamps a fresh mtime, so the stored fingerprint could never match).
    """
    paths = _seed_observation_cache(tmp_path, monkeypatch)
    # A real build fetched the asset and stamped its bytes; emulate that
    # metadata, since the seed helper cannot fetch either.
    asset_key = "curation:peptide_attributions/never_fetched_448.csv"
    meta = json.loads(_meta_path().read_text())
    assert meta["sources"][asset_key] == {"missing": True}
    meta["sources"][asset_key] = {
        "path": "/cache/x.csv",
        "size": 1,
        "mtime": 0.0,
        "sha256": "0" * 64,
    }
    _meta_path().write_text(json.dumps(meta))

    with pytest.raises(AssertionError, match="never_fetched_448"):
        _cache_is_valid(paths)

    assert observations_cache_is_current() is False


def test_packaged_or_cached_finds_only_local_copies(tmp_path, monkeypatch):
    packaged = tmp_path / "packaged.csv"
    packaged.write_text("x")
    assert downloads.packaged_or_cached(packaged, "packaged.csv") == packaged

    def refuse_network(filename, **_kw):
        raise AssertionError(f"tried to download {filename}")

    monkeypatch.setattr(downloads, "fetch_data_asset", refuse_network)
    assert (
        downloads.packaged_or_cached(tmp_path / "absent.csv", "absent_448_never_cached.csv") is None
    )


# ── mappings ────────────────────────────────────────────────────────────────


def _seed_mapping_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(downloads, "_override_data_dir", tmp_path)
    for name in ("observations.parquet", "binding.parquet", "peptide_mappings.parquet"):
        (tmp_path / name).write_bytes(name.encode())
    mappings_meta_path().write_text(
        json.dumps(
            {
                "observations": _obs_fingerprint(),
                "contract": _mapping_artifact_contract(
                    release=112, use_uniprot=False, fetch_missing=True, flank=15
                ),
            }
        )
    )


def test_mappings_predicate_true_on_current_sidecar(tmp_path, monkeypatch, capsys):
    _seed_mapping_cache(tmp_path, monkeypatch)

    assert mappings_cache_is_current(flank=15) is True
    _assert_silent(capsys)


def test_mappings_predicate_false_on_contract_change(tmp_path, monkeypatch):
    _seed_mapping_cache(tmp_path, monkeypatch)

    assert mappings_cache_is_current(flank=16) is False
    assert mappings_cache_is_current(release=113, flank=15) is False


def test_mappings_predicate_false_when_observations_move(tmp_path, monkeypatch):
    """A rebuilt observations parquet orphans the sidecar even if it still exists."""
    _seed_mapping_cache(tmp_path, monkeypatch)
    (tmp_path / "observations.parquet").write_bytes(b"rebuilt with more rows")

    assert mappings_cache_is_current(flank=15) is False


def test_mappings_predicate_false_when_nothing_is_built(tmp_path, monkeypatch, capsys):
    """build_peptide_mappings raises FileNotFoundError here; the predicate answers."""
    monkeypatch.setattr(downloads, "_override_data_dir", tmp_path)

    assert mappings_cache_is_current() is False
    _assert_silent(capsys)


def test_mappings_predicate_defaults_track_the_builder():
    """The predicate answers for the build a consumer would run with no arguments."""
    predicate = inspect.signature(mappings_cache_is_current).parameters
    build = inspect.signature(build_peptide_mappings).parameters
    for name in ("release", "fetch_missing", "use_uniprot", "flank"):
        assert predicate[name].default == build[name].default, name


# ── wiring ──────────────────────────────────────────────────────────────────


def test_predicates_are_top_level_api():
    assert hitlist.observations_cache_is_current is observations_cache_is_current
    assert hitlist.mappings_cache_is_current is mappings_cache_is_current
