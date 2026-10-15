"""Admission of current inputs before expensive training-bundle generation (#679)."""

import functools
import json

import pandas as pd
import pytest

from hitlist import export, mappings, observations
from hitlist.training_bundle import write_training_bundle
from tests.test_cache_current import _seed_observation_cache


@pytest.fixture
def current_indexes(tmp_path, monkeypatch):
    _seed_observation_cache(tmp_path, monkeypatch)
    assert observations.observations_cache_is_current() is True
    return tmp_path


def _stop_at_generator(monkeypatch):
    @functools.wraps(export.generate_training_table)
    def generate(*args, **kwargs):
        raise RuntimeError("generator reached")

    monkeypatch.setattr(export, "generate_training_table", generate)


@pytest.mark.parametrize("status", [False, None])
def test_rejects_stale_or_unverifiable_before_generation(tmp_path, monkeypatch, status):
    monkeypatch.setattr(observations, "observations_cache_is_current", lambda: status)
    _stop_at_generator(monkeypatch)
    destination = tmp_path / "missing-parent" / "bundle"
    with pytest.raises(ValueError, match=r"current.*indexes"):
        write_training_bundle(destination)
    assert not destination.parent.exists()


def test_real_stale_schema_requires_rebuild(current_indexes, monkeypatch):
    metadata = current_indexes / "observations_meta.json"
    value = json.loads(metadata.read_text())
    value["artifact_version"] = 5
    metadata.write_text(json.dumps(value))
    _stop_at_generator(monkeypatch)
    with pytest.raises(ValueError, match="hitlist build observations"):
        write_training_bundle(current_indexes / "bundle")


@pytest.mark.parametrize(
    "options",
    [
        {"map_source_proteins": True},
        {"explode_mappings": True},
        {"with_peptide_origin": True},
        {"gene": "PRAME"},
        {"gene_name": "PRAME"},
        {"gene_id": "ENSG00000185686"},
    ],
)
def test_missing_consumed_mappings_fail_before_generation(current_indexes, monkeypatch, options):
    _stop_at_generator(monkeypatch)
    with pytest.raises(ValueError, match="mappings"):
        write_training_bundle(current_indexes / "bundle", **options)


def test_unused_mapping_index_is_not_required(current_indexes, monkeypatch):
    _stop_at_generator(monkeypatch)
    with pytest.raises(RuntimeError, match="generator reached"):
        write_training_bundle(current_indexes / "bundle", gene=[], explode_mappings=False)


def test_recorded_mapping_settings_and_stale_inputs(current_indexes, monkeypatch):
    pd.DataFrame({"peptide": ["AAAAAAAAA"]}).to_parquet(mappings.mappings_path())
    metadata = {
        "observations": mappings._obs_fingerprint(),
        "contract": mappings._mapping_artifact_contract(
            release=111, fetch_missing=False, use_uniprot=True, flank=12
        ),
    }
    mappings.mappings_meta_path().write_text(json.dumps(metadata))
    _stop_at_generator(monkeypatch)
    with pytest.raises(RuntimeError, match="generator reached"):
        write_training_bundle(
            current_indexes / "bundle", map_source_proteins=True, proteome_release=112
        )
    metadata["observations"] = {}
    mappings.mappings_meta_path().write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="mappings"):
        write_training_bundle(current_indexes / "bundle", map_source_proteins=True)


@pytest.mark.parametrize("contract", [None, {}, {"release": "unknown"}])
def test_invalid_mapping_contract_fails_closed(current_indexes, monkeypatch, contract):
    mappings.mappings_meta_path().write_text(json.dumps({"contract": contract}))
    _stop_at_generator(monkeypatch)
    with pytest.raises(ValueError, match="mappings"):
        write_training_bundle(current_indexes / "bundle", map_source_proteins=True)
