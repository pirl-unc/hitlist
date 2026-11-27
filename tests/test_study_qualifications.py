"""Tests for the unified YAML-backed study qualification registry (#986 Phase 2a)."""

from __future__ import annotations

import hashlib
import io
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from hitlist.bulk_proteomics import qualify_bekker_jensen_peptide_table
from hitlist.downloads import available_dataset_catalog, info, preflight
from hitlist.errors import OfflineError
from hitlist.study_qualifications import (
    _extract_remote_zip_members,
    dlbcl_supplement_contract,
    fetch_study_assets,
    get_study_qualification,
    list_study_qualifications,
    load_study_assets_manifest,
    load_study_contract,
    load_study_qualifications,
    study_assets_manifest_path,
)


def test_study_qualifications_registry_covers_all_16_legacy_contracts():
    registry = load_study_qualifications()
    ids = list_study_qualifications()
    assert len(ids) == 16
    assert ids == sorted(registry)
    for study_id in ids:
        entry = get_study_qualification(study_id)
        assert entry["study_id"] == study_id
        contract = load_study_contract(study_id)
        assert isinstance(contract, dict) and contract
        if entry.get("assets_file"):
            manifest = load_study_assets_manifest(study_id)
            assert isinstance(manifest, dict) and manifest
            assert study_assets_manifest_path(study_id).is_file()


def test_unknown_study_qualification_raises_key_error():
    with pytest.raises(KeyError, match="Unknown study qualification"):
        get_study_qualification("nonexistent_study_xyz")


def test_managed_catalog_datasets_backed_by_study_qualifications():
    catalog = available_dataset_catalog()
    expected_bundles = (
        "dlbcl-35561310-supplements",
        "pollock-2021-neotoma",
        "sanz-bravo-2018b-erap2-oe",
        "bekker-jensen-2017-pxd004452-provenance",
        "pxd052314-goldberg-mapp-metadata",
        "pxd052314-goldberg-mapp",
        "pxd004886-ms-calibration",
        "maben-2021-erap1-source-data",
        "pxd058121-intact-degradomics",
        "pxd018722-sahu-2021-psmd2-metadata",
    )
    for name in expected_bundles:
        assert name in catalog
        assert info(name)["name"] == name
        assert preflight(name)["name"] == name


def test_remote_zip_range_extraction_and_offline_refusal(tmp_path: Path):
    buf = io.BytesIO()
    payload = b"Sequence\tLength\nSIINFEKL\t8\n"
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("nested/txt/peptides.txt", payload)
    archive = buf.getvalue()

    def _range_getter(_url: str, start: int, end_inclusive: int) -> bytes:
        return archive[start : end_inclusive + 1]

    extracted = _extract_remote_zip_members(
        "https://example.org/test.zip",
        len(archive),
        {"peptides.txt": "txt/peptides.txt"},
        range_getter=_range_getter,
    )
    assert extracted == {"peptides.txt": payload}
    assert (
        hashlib.sha256(extracted["peptides.txt"]).hexdigest() == hashlib.sha256(payload).hexdigest()
    )

    with pytest.raises(OfflineError):
        fetch_study_assets("bekker_jensen_2017", dest_dir=tmp_path / "bj", offline=True)
    with pytest.raises(OfflineError):
        fetch_study_assets("erap2_oe_29769354", dest_dir=tmp_path / "erap2", offline=True)


def test_dlbcl_contract_and_non_downloadable_study_rejection(tmp_path: Path):
    contract = dlbcl_supplement_contract()
    assert contract["pmid"] == 35561310
    assert "table_3_class_i" in contract["workbooks"]
    with pytest.raises(ValueError, match="no auto-downloadable asset bundle"):
        fetch_study_assets("canine_dla_atlas", dest_dir=tmp_path)


def test_qualify_bekker_jensen_peptide_table_stamps_provenance():
    df = pd.DataFrame([{"peptide": "SIINFEKL", "log2_intensity": 20.5}])
    out = qualify_bekker_jensen_peptide_table(df)
    assert out.loc[0, "modifications_scope"] == "deposit_aggregate_peptides_txt"
    assert out.loc[0, "intensity_scale"] == "raw_ms1_intensity"
    assert out.loc[0, "experiment_scope_provenance"] == "pooled_legacy_csv"
    assert pd.isna(out.loc[0, "log2_lfq_intensity"])
