"""End-to-end coverage for the supplementary build + observations export
flow (issue #29).

Existing tests validate ``scan_supplementary()`` against the real
shipped manifest, and ``generate_observations_table()`` against a
session-scoped pre-built parquet (which only runs when
``observations.parquet`` exists on the dev box). What was missing:

* a fixture-based integration test that exercises the full path
  (synthetic supplementary CSV → scan → parquet → load →
  ``generate_observations_table()``) without depending on the
  4.9M-row built table; and
* coverage for the parquet output path of ``hitlist export
  observations -o foo.parquet``.

These tests build small synthetic inputs in ``tmp_path`` and
monkeypatch the module-level paths / loader functions so they run
in seconds with no on-disk corpus.
"""

from __future__ import annotations

import sys

import pandas as pd

# ── Synthetic fixtures ────────────────────────────────────────────────


def _write_supplementary_fixture(tmp_path):
    """Write a tiny supplementary CSV + manifest YAML rooted at ``tmp_path``.

    Returns ``(manifest_path, supp_dir, csv_path)``. Caller monkeypatches
    ``hitlist.supplement._MANIFEST_PATH`` and ``_SUPP_DIR`` to point here.
    """
    supp_dir = tmp_path / "supplementary"
    supp_dir.mkdir()
    csv_path = supp_dir / "synthetic_2026.csv"
    csv_path.write_text(
        "peptide,mhc_restriction,mhc_class\nFAKEPEPTID,HLA-A*02:01,I\nANOTHERPEP,HLA-B*07:02,I\n"
    )
    # Manifest links the CSV to a curated PMID + a defaults block (defaults
    # populate scanner-equivalent columns that the supplementary CSV
    # itself doesn't carry — process_type, disease, etc).
    manifest_path = tmp_path / "supplementary.yaml"
    manifest_path.write_text(
        "- pmid: 99999999\n"
        '  file: "synthetic_2026.csv"\n'
        '  label: "synthetic_2026_test_fixture"\n'
        "  defaults:\n"
        '    process_type: "Cellular MHC ligand presentation"\n'
        '    disease: "Healthy"\n'
        '    cell_name: "synthetic_cell_line"\n'
        '    source_tissue: ""\n'
    )
    return manifest_path, supp_dir, csv_path


def _patch_supplement_paths(monkeypatch, manifest_path, supp_dir):
    """Point ``hitlist.supplement`` at the synthetic manifest + CSV dir."""
    monkeypatch.setattr("hitlist.supplement._MANIFEST_PATH", manifest_path)
    monkeypatch.setattr("hitlist.supplement._SUPP_DIR", supp_dir)


def _write_observations_fixture(tmp_path):
    """Synthetic observations.parquet covering both mono- and multi-
    allelic samples, plus a row whose ``mhc_restriction`` is the class
    sentinel (``HLA class I``) — the three sample-match-type branches
    in ``generate_observations_table``."""
    df = pd.DataFrame(
        {
            "peptide": [
                "MONOPEPTID",  # mono-allelic exact match
                "MULTIPEP01",  # multi-allelic, sample carries this allele
                "POOLPEPTID",  # class-only sentinel
            ],
            "mhc_restriction": [
                "HLA-A*02:01",
                "HLA-A*02:01",
                "HLA class I",
            ],
            "mhc_class": ["I", "I", "I"],
            "reference_iri": ["mono-1", "multi-1", "pool-1"],
            "pmid": pd.array([11111111, 22222222, 22222222], dtype="Int64"),
            "source": ["iedb"] * 3,
            "mhc_species": ["Homo sapiens"] * 3,
            "is_binding_assay": [False] * 3,
        }
    )
    path = tmp_path / "observations.parquet"
    df.to_parquet(path, index=False)
    return path


def _make_pmid_overrides_fixture():
    """Return a ``load_pmid_overrides()``-shaped dict with two PMIDs:

    * 11111111 — mono-allelic HeLa-like cell, single ms_sample with
      ``HLA-A*02:01`` restriction.
    * 22222222 — multi-allelic donor, ms_sample carries A*02:01 +
      B*07:02 + C*07:02.
    """
    return {
        11111111: {
            "study_label": "Mono PMID",
            "ms_samples": [
                {
                    "sample_label": "Mono-HeLa-1",
                    "mhc": "HLA-A*02:01",
                    "mhc_class": "I",
                    "is_monoallelic": True,
                    "instrument": "Q Exactive HF",
                    "instrument_type": "Orbitrap",
                    "acquisition_mode": "DDA",
                }
            ],
        },
        22222222: {
            "study_label": "Multi PMID",
            # Two samples — single_sample_fallback only fires for PMIDs
            # with exactly one sample, so a second sample here forces
            # class-only rows down the pmid_class_pool branch instead.
            "ms_samples": [
                {
                    "sample_label": "Multi-donor-A",
                    "mhc": "HLA-A*02:01 HLA-B*07:02 HLA-C*07:02",
                    "mhc_class": "I",
                    "is_monoallelic": False,
                    "instrument": "Orbitrap Fusion Lumos",
                    "instrument_type": "Orbitrap",
                    "acquisition_mode": "DDA",
                },
                {
                    "sample_label": "Multi-donor-B",
                    "mhc": "HLA-A*03:01 HLA-B*44:02 HLA-C*05:01",
                    "mhc_class": "I",
                    "is_monoallelic": False,
                    "instrument": "Orbitrap Fusion Lumos",
                    "instrument_type": "Orbitrap",
                    "acquisition_mode": "DDA",
                },
            ],
        },
    }


# ── Test 1: scan_supplementary survives parquet round-trip on a fixture ──


def test_scan_supplementary_with_fixture_survives_parquet(tmp_path, monkeypatch):
    """v1.30.7 / #29: ``scan_supplementary()`` against a synthetic
    manifest + CSV produces rows that survive a parquet round-trip with
    peptide / pmid / allele / source classification preserved.

    Today's ``test_scan_supplementary_parquet_roundtrip`` exercises the
    SHIPPED manifest, which means CI that lacks the supplementary CSVs
    or that's running on a stripped-down install gets nothing. This
    fixture-based test runs unconditionally."""
    manifest_path, supp_dir, _ = _write_supplementary_fixture(tmp_path)
    _patch_supplement_paths(monkeypatch, manifest_path, supp_dir)

    from hitlist.supplement import scan_supplementary

    df = scan_supplementary()
    assert not df.empty
    # Rows from the synthetic fixture (peptides match what we wrote).
    peptides = set(df["peptide"])
    assert peptides == {"FAKEPEPTID", "ANOTHERPEP"}
    # Source classification ran (otherwise these columns would be missing).
    assert "src_cancer" in df.columns
    assert "src_cell_line" in df.columns
    # PMID survives unchanged.
    assert df["pmid"].dropna().unique().tolist() == [99999999]
    # Allele was normalized (already canonical here, but check column exists).
    assert df["mhc_restriction"].iloc[0].startswith("HLA-")

    # Parquet round-trip preserves all of this.
    out = tmp_path / "supp.parquet"
    df.to_parquet(out, index=False)
    rt = pd.read_parquet(out)
    assert len(rt) == len(df)
    assert set(rt["peptide"]) == peptides
    assert rt["pmid"].notna().all()


def test_scan_supplementary_survives_concat_with_synthetic_iedb_rows(tmp_path, monkeypatch):
    """v1.30.7 / #29: the supplementary frame must concatenate cleanly
    with scanner-shaped rows so the builder can union them. Pin that
    the column intersection is non-empty and that supplementary rows
    are still retrievable after the concat."""
    manifest_path, supp_dir, _ = _write_supplementary_fixture(tmp_path)
    _patch_supplement_paths(monkeypatch, manifest_path, supp_dir)

    from hitlist.supplement import scan_supplementary

    supp = scan_supplementary()
    # The builder tags supplementary rows with ``source="supplement"``
    # after the scan (builder.py:388); mirror that here so the merged
    # frame has the same shape downstream consumers see.
    supp["source"] = "supplement"

    # Synthetic IEDB-shape rows (only the columns shared by the scanner;
    # the builder unions by available columns, so missing columns
    # fill with NaN — this is the contract we're locking down).
    iedb_like = pd.DataFrame(
        {
            "peptide": ["IEDBPEPTID"],
            "mhc_restriction": ["HLA-A*02:01"],
            "mhc_class": ["I"],
            "pmid": [12345678],
            "source": ["iedb"],
        }
    )
    merged = pd.concat([iedb_like, supp], ignore_index=True, sort=False)
    # All three rows present.
    assert len(merged) == 3
    # Supplementary peptides are still recoverable by source.
    supp_rows = merged[merged["source"] == "supplement"]
    assert set(supp_rows["peptide"]) == {"FAKEPEPTID", "ANOTHERPEP"}


# ── Test 2: generate_observations_table joins on a synthetic fixture ──


def test_generate_observations_table_joins_mono_and_multi_allelic(tmp_path, monkeypatch):
    """v1.30.7 / #29: integration test for the sample-metadata join
    without depending on the 4.9M-row built parquet.

    Synthetic obs.parquet (3 rows: mono-allelic, multi-allelic with
    sample carrying that allele, class-only sentinel) + fixture
    pmid_overrides → assert each row gets the right
    ``sample_match_type`` and metadata."""
    obs_path = _write_observations_fixture(tmp_path)
    monkeypatch.setattr("hitlist.observations.observations_path", lambda: obs_path)

    overrides = _make_pmid_overrides_fixture()
    monkeypatch.setattr("hitlist.curation.load_pmid_overrides", lambda: overrides)
    monkeypatch.setattr("hitlist.export.load_pmid_overrides", lambda: overrides)

    from hitlist.export import generate_observations_table

    df = generate_observations_table()
    assert len(df) == 3

    # Mono-allelic row joins to its single sample by exact allele match.
    mono = df[df["peptide"] == "MONOPEPTID"].iloc[0]
    assert mono["sample_match_type"] == "allele_match"
    assert mono["sample_label"] == "Mono-HeLa-1"
    assert mono["instrument_type"] == "Orbitrap"

    # Multi-allelic row's allele appears in the sample's set.
    multi = df[df["peptide"] == "MULTIPEP01"].iloc[0]
    assert multi["sample_match_type"] == "allele_match"
    assert multi["sample_label"] == "Multi-donor-A"

    # Class-only row falls into the per-class allele pool — there's no
    # specific sample to point at, but the union of class-I alleles
    # across the PMID's samples should still be reported.
    pool = df[df["peptide"] == "POOLPEPTID"].iloc[0]
    assert pool["sample_match_type"] == "pmid_class_pool"


def test_generate_observations_table_columns_contract(tmp_path, monkeypatch):
    """v1.30.7 / #29: lock down the joined-table column contract on the
    fixture flow. Downstream consumers (predictors, exports, the
    ``training`` table planned in #136) rely on these names being
    stable."""
    obs_path = _write_observations_fixture(tmp_path)
    monkeypatch.setattr("hitlist.observations.observations_path", lambda: obs_path)

    overrides = _make_pmid_overrides_fixture()
    monkeypatch.setattr("hitlist.curation.load_pmid_overrides", lambda: overrides)
    monkeypatch.setattr("hitlist.export.load_pmid_overrides", lambda: overrides)

    from hitlist.export import generate_observations_table

    df = generate_observations_table()
    expected = {
        "peptide",
        "mhc_restriction",
        "mhc_class",
        "pmid",
        "source",
        # Sample-metadata join columns.
        "sample_label",
        "instrument",
        "instrument_type",
        "acquisition_mode",
        "sample_mhc",
        # Provenance / match-type columns.
        "sample_match_type",
        "matched_sample_count",
        "has_peptide_level_allele",
    }
    missing = expected - set(df.columns)
    assert not missing, f"missing expected columns: {missing}"


# ── Test 3: CLI ``hitlist export observations -o ...`` parquet round-trip ──


def test_cli_export_observations_to_parquet(tmp_path, monkeypatch, capsys):
    """v1.30.7 / #29: the parquet output path of ``hitlist export
    observations -o foo.parquet`` was never exercised. Round-trip via
    the actual CLI entrypoint."""
    obs_path = _write_observations_fixture(tmp_path)
    monkeypatch.setattr("hitlist.observations.observations_path", lambda: obs_path)

    overrides = _make_pmid_overrides_fixture()
    monkeypatch.setattr("hitlist.curation.load_pmid_overrides", lambda: overrides)
    monkeypatch.setattr("hitlist.export.load_pmid_overrides", lambda: overrides)

    from hitlist.cli import main

    out_parquet = tmp_path / "exported.parquet"
    monkeypatch.setattr(sys, "argv", ["hitlist", "export", "observations", "-o", str(out_parquet)])

    main()
    assert out_parquet.exists()

    rt = pd.read_parquet(out_parquet)
    assert len(rt) == 3
    # The CLI export wrote the joined table — provenance columns survive.
    assert "sample_match_type" in rt.columns
    assert set(rt["peptide"]) == {"MONOPEPTID", "MULTIPEP01", "POOLPEPTID"}


def test_cli_export_observations_to_csv_default(tmp_path, monkeypatch, capsys):
    """v1.30.7 / #29: complement to the parquet test — confirm the CSV
    output path still works on the same fixture (catches a regression
    where the suffix-detection branch in the CLI export handler picks
    the wrong writer)."""
    obs_path = _write_observations_fixture(tmp_path)
    monkeypatch.setattr("hitlist.observations.observations_path", lambda: obs_path)

    overrides = _make_pmid_overrides_fixture()
    monkeypatch.setattr("hitlist.curation.load_pmid_overrides", lambda: overrides)
    monkeypatch.setattr("hitlist.export.load_pmid_overrides", lambda: overrides)

    from hitlist.cli import main

    out_csv = tmp_path / "exported.csv"
    monkeypatch.setattr(sys, "argv", ["hitlist", "export", "observations", "-o", str(out_csv)])
    main()
    assert out_csv.exists()
    rt = pd.read_csv(out_csv)
    assert len(rt) == 3
    assert set(rt["peptide"]) == {"MONOPEPTID", "MULTIPEP01", "POOLPEPTID"}
