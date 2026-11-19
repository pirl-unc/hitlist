"""Unit, round-trip, and real-source qualification tests for NetMHCstabpan (#875)."""

from __future__ import annotations

import csv
import hashlib
import os
from pathlib import Path

import pandas as pd
import pytest

from hitlist import (
    build_observations,
    deduplicate_netmhcstabpan_against_binding,
    fetch_netmhcstabpan_source,
    load_contributors,
    load_netmhcstabpan_stability,
    netmhcstabpan_source_contract,
    prepare_stability_training_data,
    reconcile_netmhcstabpan_with_iedb,
    scan_netmhcstabpan_stability,
)
from hitlist.export import generate_binding_table, generate_training_table
from hitlist.observations import is_binding_built, load_all_evidence, load_binding
from hitlist.training_bundle import verify_training_bundle, write_training_bundle


def _write_synthetic_iedb_csv(path: Path, rows: list[dict]) -> Path:
    """Write a two-header IEDB CSV compatible with ``hitlist.scanner.scan``."""
    from hitlist.scanner import _COLUMN_NAMES

    keys = list(_COLUMN_NAMES)
    cat_header = []
    field_header = []
    for key in keys:
        cand = _COLUMN_NAMES[key][0]
        if " | " in cand:
            cat, fld = cand.split(" | ", 1)
        else:
            cat, fld = "", cand
        cat_header.append(cat)
        field_header.append(fld)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(cat_header)
        writer.writerow(field_header)
        for row in rows:
            writer.writerow([row.get(k, "") for k in keys])
    return path


def _synthetic_stability_file(path: Path) -> Path:
    lines = [
        "HLA\tPep\tThalf",
        "HLA-A*02:01 VTTEVAFGL 1.80",
        "HLA-A*03:01 GMQIRGFVY 0.00",
        "HLA-B*14:01(C67S) KLADMSIYC 108.40",
        "HLA-B*14:02(C67S) SLFNTIAVL 0.50",
        "HLA-B*39:06(C67S) THDGSLQRL 0.00",
        "HLA-B*99:01(Y9F) SLYNTVATL 2.40",
        "HLA-B*07:02 APRTVALTA 4.20",
        "HLA-B*15:01 FVRQCFNPM 2.90",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_source_contract_and_tamper_guards(tmp_path):
    contract = netmhcstabpan_source_contract()
    assert contract["source"]["pmid"] == 27402703
    assert contract["expected_reconciliation"]["n_observations"] == 28_166
    assert contract["expected_reconciliation"]["n_constructs"] == 75
    assert contract["expected_reconciliation"]["n_mutant_rows"] == 1_135

    fixture = _synthetic_stability_file(tmp_path / "Stability.txt")
    # Default verify_pinned_digest=True must reject any non-matching file.
    with pytest.raises(ValueError, match="checksum/size mismatch"):
        scan_netmhcstabpan_stability(fixture)

    bad_header = tmp_path / "bad_header.txt"
    bad_header.write_text("Allele\tPeptide\tValue\nHLA-A*02:01 SLYNTVATL 1.00\n")
    with pytest.raises(ValueError, match="Unexpected NetMHCstabpan header"):
        scan_netmhcstabpan_stability(bad_header, verify_pinned_digest=False)

    dup_file = tmp_path / "dup.txt"
    dup_file.write_text("HLA\tPep\tThalf\nHLA-A*02:01 SLYNTVATL 1.00\nHLA-A*02:01 SLYNTVATL 2.00\n")
    with pytest.raises(ValueError, match="Duplicate"):
        scan_netmhcstabpan_stability(dup_file, verify_pinned_digest=False)

    neg_file = tmp_path / "neg.txt"
    neg_file.write_text("HLA\tPep\tThalf\nHLA-A*02:01 SLYNTVATL -0.10\n")
    with pytest.raises(ValueError, match="Invalid negative"):
        scan_netmhcstabpan_stability(neg_file, verify_pinned_digest=False)

    short_line = tmp_path / "short.txt"
    short_line.write_text("HLA\tPep\tThalf\nHLA-A*02:01 SLYNTVATL\n")
    with pytest.raises(ValueError, match="Malformed NetMHCstabpan row"):
        scan_netmhcstabpan_stability(short_line, verify_pinned_digest=False)


def test_line_provenance_censor_semantics_constructs_and_chemistry(tmp_path):
    fixture = _synthetic_stability_file(tmp_path / "Stability.txt")
    expected_sha = hashlib.sha256(fixture.read_bytes()).hexdigest()
    df = scan_netmhcstabpan_stability(fixture, verify_pinned_digest=False)

    assert len(df) == 8
    assert not df["pinned_source_verified"].any()
    assert set(df["source_sha256"]) == {expected_sha}
    assert list(df["source_line_number"]) == [2, 3, 4, 5, 6, 7, 8, 9]
    assert df.loc[0, "source_record_id"] == "netmhcstabpan-1.0:Stability.txt:line:2"
    assert df.loc[0, "assay_iri"].endswith("/Stability.txt#line=2")
    assert set(df["measurement_units"]) == {"h"}
    assert set(df["measurement_inequality"]) == {""}
    assert set(df["qualifier_provenance"]) == {"unreported_in_source_table"}
    assert set(df["measurement_origin_status"]) == {"experimental_screen_or_inhouse_archive"}

    # Row 0: positive within 48h window, non-Cys synthetic sequence
    assert df.loc[0, "qualitative_measurement"] == "Positive"
    assert df.loc[0, "stability_value_status"] == "positive_within_48h_window"
    assert df.loc[0, "peptide_chemistry_status"] == "synthetic_unannotated_plain_sequence"

    # Row 1: literal 0.00h nondetection (never coerced to positive floor)
    assert df.loc[1, "quantitative_value"] == 0.0
    assert df.loc[1, "quantitative_measurement"] == "0.00"
    assert df.loc[1, "qualitative_measurement"] == "Negative"
    assert df.loc[1, "stability_value_status"] == "zero_hour_nondetection_or_subthreshold"
    assert df.loc[1, "censoring_evidence_status"] == "unresolved_assay_floor_zero_reported"

    # Row 2: C67S mutant construct + > 48h extrapolated + Cys-containing peptide
    assert df.loc[2, "reported_mhc_construct"] == "HLA-B*14:01(C67S)"
    assert df.loc[2, "mhc_restriction"] == "HLA-B*14:01 C67S mutant"
    assert df.loc[2, "construct_resolution_status"] == "curated_mutant_construct"
    assert bool(df.loc[2, "is_mutant_mhc"]) is True
    assert df.loc[2, "stability_value_status"] == "positive_extrapolated_beyond_48h_window"
    assert df.loc[2, "peptide_chemistry_status"] == "synthetic_unannotated_plain_sequence_with_cys"

    # Row 5: Unverified mutant construct is never stripped to wild-type HLA-B*99:01
    assert df.loc[5, "reported_mhc_construct"] == "HLA-B*99:01(Y9F)"
    assert df.loc[5, "mhc_restriction"] == "HLA-B*99:01(Y9F)"
    assert df.loc[5, "construct_resolution_status"] == "unresolved_construct"
    assert df.loc[5, "allele_resolution"] == "unresolved"
    assert bool(df.loc[5, "is_mutant_mhc"]) is True


def test_iedb_reconciliation_separates_dtu_lineage_from_cross_study_overlaps(tmp_path):
    fixture = _synthetic_stability_file(tmp_path / "Stability.txt")
    stab = scan_netmhcstabpan_stability(fixture, verify_pinned_digest=False)

    iedb_binding = pd.DataFrame(
        [
            # 1. Confirmed DTU submission duplicate (1.80 h == 108.0 min)
            {
                "peptide": "VTTEVAFGL",
                "mhc_restriction": "HLA-A*02:01",
                "response_measured": "half life",
                "quantitative_value": 108.0,
                "measurement_units": "min",
                "assay_iri": "http://www.iedb.org/assay/2145900",
                "reference_iri": "http://www.iedb.org/reference/1028285",
                "submission_id": "1000632",
                "source": "iedb",
            },
            # 2. Confirmed DTU submission with hypothetical value conflict (2.90 h vs 300.0 min = 5.0 h)
            {
                "peptide": "FVRQCFNPM",
                "mhc_restriction": "HLA-B*15:01",
                "response_measured": "half life",
                "quantitative_value": 300.0,
                "measurement_units": "min",
                "assay_iri": "http://www.iedb.org/assay/2146999",
                "reference_iri": "http://www.iedb.org/reference/1028293",
                "submission_id": "1000639",
                "source": "iedb",
            },
            # 3. Cross-study exact value match (4.20 h == 252.0 min in PMID 23543059 / ref 1026371)
            {
                "peptide": "APRTVALTA",
                "mhc_restriction": "HLA-B*07:02",
                "response_measured": "half life",
                "quantitative_value": 252.0,
                "measurement_units": "min",
                "assay_iri": "http://www.iedb.org/assay/9900001",
                "reference_iri": "http://www.iedb.org/reference/1026371",
                "submission_id": "",
                "source": "iedb",
            },
            # 4. Wild-type HLA-B*14:02 in PMID 7528217 must NOT match HLA-B*14:02(C67S),
            #    while flagging wildtype_base_allele_pmhc_in_iedb=True.
            {
                "peptide": "SLFNTIAVL",
                "mhc_restriction": "HLA-B*14:02",
                "response_measured": "half life",
                "quantitative_value": 30.0,
                "measurement_units": "min",
                "assay_iri": "http://www.iedb.org/assay/3154630",
                "reference_iri": "http://www.iedb.org/reference/315463",
                "submission_id": "",
                "source": "iedb",
            },
        ]
    )

    reconciled = reconcile_netmhcstabpan_with_iedb(stab, iedb_binding)
    by_pep = reconciled.set_index("peptide")
    assert (
        by_pep.loc["VTTEVAFGL", "iedb_overlap_status"] == "confirmed_dtu_iedb_submission_duplicate"
    )
    assert (
        by_pep.loc["FVRQCFNPM", "iedb_overlap_status"]
        == "confirmed_dtu_iedb_submission_value_conflict"
    )
    assert (
        by_pep.loc["APRTVALTA", "iedb_overlap_status"]
        == "cross_reference_exact_value_overlap_warning"
    )
    assert by_pep.loc["APRTVALTA", "iedb_overlap_assay_iris"] == "http://www.iedb.org/assay/9900001"
    assert by_pep.loc["SLFNTIAVL", "iedb_overlap_status"] == "no_iedb_half_life_overlap"
    assert bool(by_pep.loc["SLFNTIAVL", "wildtype_base_allele_pmhc_in_iedb"]) is True
    assert by_pep.loc["SLYNTVATL", "iedb_overlap_status"] == "unresolved_construct_not_matched"

    retained, summary = deduplicate_netmhcstabpan_against_binding(stab, iedb_binding)
    assert summary["n_dropped_confirmed_dtu_duplicates"] == 1
    assert len(retained) == 7
    # Value conflicts and cross-study overlap warnings are NEVER silently dropped.
    assert {"FVRQCFNPM", "APRTVALTA"} <= set(retained["peptide"])
    assert "VTTEVAFGL" not in set(retained["peptide"])

    # Convenience loader: reconcile-only vs deduplicate_confirmed_iedb=True
    reconciled_loaded = load_netmhcstabpan_stability(
        fixture,
        binding_df=iedb_binding,
        deduplicate_confirmed_iedb=False,
        verify_pinned_digest=False,
    )
    assert len(reconciled_loaded) == 8
    dedup_loaded = load_netmhcstabpan_stability(
        fixture,
        binding_df=iedb_binding,
        deduplicate_confirmed_iedb=True,
        verify_pinned_digest=False,
    )
    assert len(dedup_loaded) == 7


def test_fetch_source_and_offline_loader_guards(tmp_path, monkeypatch):
    import shutil
    from collections import namedtuple

    from hitlist import downloads

    monkeypatch.setattr(downloads, "_override_data_dir", tmp_path / "empty_data")
    with pytest.raises((KeyError, FileNotFoundError)):
        load_netmhcstabpan_stability(allow_download=False)

    captured = {}

    def _fake_download(url, dest, **kwargs):
        captured["url"] = url
        captured["dest"] = dest
        captured.update(kwargs)
        dest.write_text("HLA\tPep\tThalf\nHLA-A*02:01 SLYNTVATL 1.00\n", encoding="utf-8")
        return dest

    monkeypatch.setattr(downloads, "download_to_file", _fake_download)
    out = fetch_netmhcstabpan_source(tmp_path / "src_cache", force=True, verbose=False)
    contract = netmhcstabpan_source_contract()["source"]
    assert out == tmp_path / "src_cache" / "Stability.txt"
    assert captured["url"] == contract["source_url"]
    assert captured["expected_sha256"] == contract["sha256"]
    assert captured["expected_size"] == contract["size_bytes"]
    assert captured["force"] is True

    DiskUsage = namedtuple("DiskUsage", ["total", "used", "free"])
    monkeypatch.setattr(shutil, "disk_usage", lambda _p: DiskUsage(100, 99, 1))
    with pytest.raises(OSError, match="1 GiB free disk reserve"):
        fetch_netmhcstabpan_source(tmp_path / "low_disk")


def test_end_to_end_builder_contributors_bundle_and_model_training_round_trip(
    tmp_path, monkeypatch
):
    from hitlist import builder, downloads, supplement
    from hitlist.downloads import register
    from hitlist.parquet_io import atomic_write_parquet

    data_root = tmp_path / "data"
    monkeypatch.setattr(downloads, "_override_data_dir", data_root)
    monkeypatch.setattr(supplement, "scan_supplementary", lambda **kw: pd.DataFrame())

    def _empty_parquet(path: Path) -> pd.DataFrame:
        frame = pd.DataFrame()
        atomic_write_parquet(frame, path)
        return frame

    monkeypatch.setattr(
        builder,
        "build_bulk_proteomics",
        lambda **kw: _empty_parquet(builder._bulk_proteomics_path()),
    )
    monkeypatch.setattr(
        builder,
        "build_line_expression",
        lambda **kw: _empty_parquet(builder._line_expression_path()),
    )
    monkeypatch.setattr(builder, "_VERIFY_PINNED_NETMHCSTABPAN", False)

    stab_file = tmp_path / "Stability.txt"
    stab_file.write_text(
        "HLA\tPep\tThalf\n"
        "HLA-A*02:01 VTTEVAFGL 1.80\n"
        "HLA-A*03:01 GMQIRGFVY 0.00\n"
        "HLA-B*14:01(C67S) KLADMSIYC 2.50\n"
        "HLA-B*07:02 APRTVALTA 4.20\n",
        encoding="utf-8",
    )

    # Standalone netmhcstabpan-only build (no IEDB/CEDAR registered) also succeeds cleanly.
    solo_root = tmp_path / "solo_data"
    monkeypatch.setattr(downloads, "_override_data_dir", solo_root)
    register("netmhcstabpan", stab_file)
    build_observations(with_flanking=False, build_mappings=False)
    assert len(load_binding(source="netmhcstabpan")) == 4

    # Combined IEDB + NetMHCstabpan build exercises deduplication + contributor redirects.
    monkeypatch.setattr(downloads, "_override_data_dir", data_root)
    iedb_file = _write_synthetic_iedb_csv(
        tmp_path / "iedb.csv",
        [
            {
                "assay_iri": "http://www.iedb.org/assay/2145900",
                "ref_iri": "http://www.iedb.org/reference/1028285",
                "pmid": "",
                "submission_id": "1000632",
                "ref_title": "Large scale analysis of peptide - HLA-I stability",
                "epitope_name": "VTTEVAFGL",
                "host": "Homo sapiens",
                "qualitative_measurement": "Positive-High",
                "assay_method": "purified MHC/direct/radioactivity",
                "response_measured": "half life",
                "measurement_units": "min",
                "measurement_inequality": "=",
                "quantitative_measurement": "108.0",
                "assay_comments": "pMHC-I complex stability (PMID 21044632).",
                "mhc_restriction": "HLA-A*02:01",
                "mhc_class": "I",
            },
            {
                "assay_iri": "http://www.iedb.org/assay/9900001",
                "ref_iri": "http://www.iedb.org/reference/1026371",
                "pmid": "23543059",
                "submission_id": "",
                "ref_title": "Independent study citing SPA",
                "epitope_name": "APRTVALTA",
                "host": "Homo sapiens",
                "qualitative_measurement": "Positive",
                "assay_method": "purified MHC/direct/radioactivity",
                "response_measured": "half life",
                "measurement_units": "min",
                "measurement_inequality": "=",
                "quantitative_measurement": "252.0",
                "assay_comments": "Harndahl 2011.",
                "mhc_restriction": "HLA-B*07:02",
                "mhc_class": "I",
            },
        ],
    )

    register("iedb", iedb_file)
    register("netmhcstabpan", stab_file)
    build_observations(with_flanking=False, build_mappings=False)

    # 1. load_binding / generate_binding_table / generate_training_table / load_all_evidence
    stab_loaded = load_binding(source="netmhcstabpan")
    # VTTEVAFGL was deduplicated against DTU submission 1000632; 3 rows remain.
    assert len(stab_loaded) == 3
    assert set(stab_loaded["peptide"]) == {"GMQIRGFVY", "KLADMSIYC", "APRTVALTA"}
    exported_stab = generate_binding_table(source="netmhcstabpan", response_measured="half life")
    assert len(exported_stab) == 3
    training_table = generate_training_table(include_evidence="binding")
    assert len(training_table) == 5
    union = load_all_evidence()
    assert len(union) == 5  # 2 IEDB + 3 retained NetMHCstabpan

    # 2. Contributor graph linked PMID 27402703 onto the deduplicated DTU IEDB row!
    iedb_loaded = load_binding(source="iedb")
    vtte_prov_id = iedb_loaded.loc[iedb_loaded["peptide"] == "VTTEVAFGL", "provenance_id"].iloc[0]
    contribs = load_contributors([vtte_prov_id])
    assert set(contribs["source_dataset"]) == {"iedb", "netmhcstabpan"}
    assert "27402703" in set(contribs["original_pmid"])

    # 3. write_training_bundle -> verify_training_bundle -> prepare_stability_training_data
    bundle_dir = tmp_path / "stability_bundle"
    write_training_bundle(bundle_dir, include_evidence="binding")
    manifest = verify_training_bundle(bundle_dir)
    assert manifest["n_rows"] == 5

    identities = pd.read_parquet(bundle_dir / "identities.parquet")
    vtte_ident = identities.loc[identities["provenance_id"] == vtte_prov_id].iloc[0]
    assert vtte_ident["contributor_pmids"] == '["27402703"]'
    assert vtte_ident["contributor_resolution"] == "overlap_unresolved"

    sel_default = prepare_stability_training_data(bundle_dir)
    # 5 candidate rows:
    # - VTTEVAFGL (iedb, 108 min) -> accepted
    # - APRTVALTA (iedb, 252 min) -> accepted
    # - APRTVALTA (netmhcstabpan, 4.20 h = 252 min, cross_reference warning) -> accepted
    # - GMQIRGFVY (netmhcstabpan, 0.00 h) -> excluded (non_positive_measurement_value)
    # - KLADMSIYC (netmhcstabpan, HLA-B*14:01 C67S mutant) -> excluded (mutant_mhc_construct_excluded)
    assert len(sel_default.accepted) == 3
    assert len(sel_default.excluded) == 2
    assert set(sel_default.excluded["exclusion_reason"]) == {
        "non_positive_measurement_value",
        "mutant_mhc_construct_excluded",
    }

    sel_mut = prepare_stability_training_data(bundle_dir, allow_mutant_mhc=True)
    assert len(sel_mut.accepted) == 4
    assert len(sel_mut.excluded) == 1
    assert "HLA-B*14:01 C67S mutant" in set(sel_mut.accepted["allele"])

    # 4. Cached fetch() rejects tampered bytes on pinned dataset "netmhcstabpan"
    dest = data_root / downloads.FETCHABLE_DATASETS["netmhcstabpan"]["filename"]
    dest.write_bytes(b"x" * 799_880)
    assert (
        downloads._cached_file_problem("netmhcstabpan", dest, None)
        == "sha256 differs from the pinned source value"
    )


def _find_real_stability_txt() -> Path | None:
    env_dir = os.environ.get("HITLIST_NETMHCSTABPAN_SOURCE_DIR")
    if env_dir:
        candidate = Path(env_dir) / "Stability.txt"
        if candidate.is_file():
            return candidate
    from hitlist.downloads import get_path

    try:
        return get_path("netmhcstabpan")
    except (KeyError, FileNotFoundError):
        return None


@pytest.mark.integration
def test_real_netmhcstabpan_source_and_corpus_reconciliation():
    path = _find_real_stability_txt()
    if path is None:
        pytest.skip("Checksum-pinned NetMHCstabpan Stability.txt not installed")

    df = load_netmhcstabpan_stability(path, verify_pinned_digest=True)
    assert len(df) == 28_166
    assert df["pinned_source_verified"].all()
    assert df["reported_mhc_construct"].nunique() == 75
    assert int((df["raw_stability_hours"] > 0).sum()) == 22_487
    assert int((df["raw_stability_hours"] == 0).sum()) == 5_679
    assert int((df["raw_stability_hours"] > 48.0).sum()) == 354
    assert int(df["is_mutant_mhc"].sum()) == 1_135

    sel = prepare_stability_training_data(df)
    assert len(sel.accepted) + len(sel.excluded) == 28_166
    assert (sel.accepted["measurement_value"] > 0).all()
    assert not sel.accepted["allele_completeness"].str.startswith("mutant_").any()

    if is_binding_built():
        bdf = load_binding()
        retained, summary = deduplicate_netmhcstabpan_against_binding(df, bdf)
        assert summary["n_dropped_confirmed_dtu_duplicates"] > 0
        assert len(retained) + summary["n_dropped_confirmed_dtu_duplicates"] == 28_166
        assert "cross_reference_exact_value_overlap_warning" in set(retained["iedb_overlap_status"])
