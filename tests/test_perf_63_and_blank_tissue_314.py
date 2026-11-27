"""Tests for #63 build-performance optimizations and #314 blank-tissue provenance."""

from __future__ import annotations

import sqlite3

import pandas as pd
import pytest

from hitlist.curation import pmid_provenance
from hitlist.provenance import ContributorCollector
from hitlist.qc import blank_tissue_provenance_audit, mhc_token_audit
from hitlist.scanner import scan
from tests.test_scanner import _write_tiny_iedb_csv


def test_mhc_token_audit_preserves_counts_duplicates_and_na_pmids():
    frame = pd.DataFrame(
        {
            "pmid": [100, 100, pd.NA, pd.NA, 200],
            "mhc_restriction": [
                "HLA-A*02:01",
                "HLA-B23;HLA-B23",
                "HLA-DR7A",
                "HLA-DR7A",
                "NOT_AN_ALLELE",
            ],
            "host_mhc_types": ["", "", "", "", ""],
            "serotype": ["", "", "", "", ""],
            "serotypes": ["", "", "", "", ""],
        }
    )
    audit = mhc_token_audit(
        evidence_frames={"ms": frame},
        curated_samples=pd.DataFrame(columns=["pmid", "sample_label", "mhc", "mhc_genotype"]),
    )
    assert len(audit) == 3
    by_tok = audit.set_index("token")
    # Duplicate token in one cell counts twice, matching prior explode() semantics
    assert by_tok.loc["HLA-B23", "n_records"] == 2
    assert by_tok.loc["HLA-B23", "status"] == "invalid_source"
    # NA pmid rows aggregate cleanly
    assert by_tok.loc["HLA-DR7A", "n_records"] == 2
    assert by_tok.loc["HLA-DR7A", "status"] == "invalid_source"
    assert pd.isna(by_tok.loc["HLA-DR7A", "pmid"])
    assert by_tok.loc["NOT_AN_ALLELE", "n_records"] == 1
    assert by_tok.loc["NOT_AN_ALLELE", "status"] == "unrecognized"


def test_provenance_batched_capture_and_dedup_preserve_exact_bytes_and_integrity(tmp_path):
    iedb = tmp_path / "iedb.csv"
    rows = []
    for i in range(600):
        r = [""] * 27
        r[0] = f"http://iedb.org/assay/{i // 2}"  # every pair shares assay_iri
        r[1] = f"http://iedb.org/ref/{i // 100}"
        r[2] = str(30640174 if (i // 2) % 2 == 0 else 36284347)
        r[5] = "CLAVEEVSL" if i % 2 == 0 else "GILGFVFTL + OX(M1)"
        r[8] = "Homo sapiens"
        r[10] = "Cellular MHC ligand presentation"
        r[14] = "Positive"
        r[19] = "HLA-A*02:01" if i % 3 else "HLA-B*07:02"
        r[20] = "I"
        r[22] = "cellular MHC/mass spectrometry"
        rows.append(r)
    _write_tiny_iedb_csv(iedb, rows)

    out_path = tmp_path / "contributors.parquet"
    with ContributorCollector(scratch_dir=tmp_path) as col:
        df = scan(iedb_path=iedb, mhc_species="Homo sapiens", provenance=col)
        meta = col.write([df], out_path)

    assert len(df) == 300
    assert meta["n_contributor_links"] == 600
    # Verify #314 per-PMID provenance was applied inside scan()
    assert set(df.loc[df["pmid"] == 30640174, "source_tissue"]) == {"Blood"}
    assert set(df.loc[df["pmid"] == 36284347, "source_tissue"]) == {"Pancreas"}

    # Duplicate row_number must still raise sqlite3.IntegrityError immediately
    with ContributorCollector(scratch_dir=tmp_path) as col:
        writer = col.csv_record_writer("iedb", {"pmid": 0})
        writer(1, ["111"])
        writer(2, ["222"])
        with pytest.raises(sqlite3.IntegrityError):
            writer(2, ["333"])


@pytest.mark.parametrize(
    ("pmid", "expected_tissue", "expected_disease"),
    [
        (31575892, "Blood", "acute myeloid leukemia"),
        (30640174, "Blood", "acute myeloid leukemia"),
        (36284347, "Pancreas", "pancreatic ductal adenocarcinoma"),
        (31921104, "Skin", "melanoma"),
        (34290406, "Skin", "melanoma"),
        (25645385, "Skin", "skin melanoma"),
        (37938771, "Peripheral Nervous System", "neuroblastoma"),
    ],
)
def test_curated_314_pmid_provenance_entries(pmid, expected_tissue, expected_disease):
    prov = pmid_provenance(pmid)
    assert prov["source_tissue"] == expected_tissue
    assert prov["disease"] == expected_disease


def test_blank_tissue_provenance_audit_distinguishes_reviewed_and_missing():
    obs = pd.DataFrame(
        [
            # Reviewed multi-line MAPTAC -> info
            {
                "pmid": 31495665,
                "species": "Homo sapiens",
                "source_tissue": "",
                "cell_name": "cell-line mixture",
                "cell_type": "",
                "disease": "healthy",
                "src_cancer": False,
                "src_cell_line": True,
            },
            # Unreviewed blank-tissue cancer line -> warn
            {
                "pmid": 99999901,
                "species": "Homo sapiens",
                "source_tissue": "Other",
                "cell_name": "SomeLine",
                "cell_type": "Epithelial",
                "disease": "carcinoma",
                "src_cancer": True,
                "src_cell_line": True,
            },
            # Properly populated source_tissue -> omitted from audit
            {
                "pmid": 30640174,
                "species": "Homo sapiens",
                "source_tissue": "Blood",
                "cell_name": "",
                "cell_type": "",
                "disease": "acute myeloid leukemia",
                "src_cancer": True,
                "src_cell_line": False,
            },
        ]
    )
    audit = blank_tissue_provenance_audit(obs)
    assert len(audit) == 2
    by_pmid = audit.set_index("pmid")
    assert by_pmid.loc[31495665, "status"] == "reviewed_non_single_tissue"
    assert by_pmid.loc[31495665, "severity"] == "info"
    assert by_pmid.loc[99999901, "status"] == "missing_source_tissue"
    assert by_pmid.loc[99999901, "severity"] == "warn"
    assert callable(blank_tissue_provenance_audit)
