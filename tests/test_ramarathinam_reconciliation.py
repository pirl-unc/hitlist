"""Tests for #632: Ramarathinam 2021 (PMID 34357683 / PXD025877) coverage reconciliation."""

from __future__ import annotations

import json
import re
from importlib.resources import files

from hitlist.curation import expand_allele_set, load_pmid_overrides

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _load_ledger() -> dict:
    path = files("hitlist").joinpath(
        "data/evidence_revisions/ramarathinam_2021_34357683_reconciliation.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))


def test_ramarathinam_reconciliation_ledger_arithmetic_and_fingerprints() -> None:
    ledger = _load_ledger()
    assert ledger["source"]["pmid"] == 34357683
    assert ledger["source"]["doi"] == "10.1002/pmic.202000160"
    assert ledger["source"]["pride_accession"] == "PXD025877"

    fp = ledger["fingerprints"]
    assert _SHA256_RE.match(fp["ci_corpus_v2_observations_parquet_sha256"])
    assert _SHA256_RE.match(fp["pride_sdrf"]["sha256"])

    mzids = fp["pride_mzid_files"]
    assert len(mzids) == 18
    for item in mzids:
        assert _SHA256_RE.match(item["sha256"]), f"bad sha256 on {item['file_name']}"
        assert item["compressed_bytes"] > 0
        assert item["n_peptide_entries"] > 0
        assert item["locus"] in {"HLA-DR", "HLA-DQ", "HLA-DP"}

    rec = ledger["reconciliation"]
    pride = rec["pride_pxd025877_mzid"]
    iedb = rec["iedb_deposited_34357683"]
    obs = rec["hitlist_observations_parquet"]
    diff = rec["iedb_to_hitlist_filter_diff"]

    # PRIDE length buckets sum to PRIDE union
    assert (
        sum(pride["n_distinct_peptides_by_length_bucket"].values())
        == pride["n_distinct_peptides_union"]
    )
    assert (
        sum(pride["n_distinct_peptides_by_locus"].values())
        == pride["n_distinct_peptides_locus_sum"]
    )
    assert sum(pride["n_files_by_locus"].values()) == pride["n_files"] == len(mzids)

    # IEDB is a 100% subset of PRIDE PXD025877
    assert iedb["n_distinct_peptides_iedb_only"] == 0
    assert iedb["n_distinct_peptides_in_pride"] == iedb["n_distinct_peptides_union"]
    assert (
        pride["n_distinct_peptides_union"] - iedb["n_distinct_peptides_union"]
        == iedb["n_distinct_peptides_pride_only"]
    )
    assert (
        sum(iedb["n_distinct_peptides_pride_only_by_length_bucket"].values())
        == iedb["n_distinct_peptides_pride_only"]
    )
    assert iedb["n_rows_unmodified"] + iedb["n_rows_modified"] == iedb["n_rows_total"]
    assert sum(iedb["n_rows_by_modification"].values()) == iedb["n_rows_modified"]
    assert sum(iedb["n_rows_by_locus"].values()) == iedb["n_rows_total"]
    assert (
        sum(iedb["n_distinct_peptides_by_locus"].values()) == iedb["n_distinct_peptides_locus_sum"]
    )
    assert (
        iedb["n_distinct_peptides_locus_sum"] - iedb["n_distinct_peptides_union"]
        == iedb["n_distinct_peptides_shared_across_loci"]
    )

    # Exact #122 length-filter accounting between IEDB and observations.parquet
    assert sum(obs["n_rows_by_locus"].values()) == obs["n_rows_total"] == 76855
    assert obs["n_distinct_peptides_union"] == 62467
    assert sum(obs["n_distinct_peptides_by_locus"].values()) == obs["n_distinct_peptides_locus_sum"]
    assert (
        obs["n_distinct_peptides_locus_sum"] - obs["n_distinct_peptides_union"]
        == obs["n_distinct_peptides_shared_across_loci"]
    )
    assert iedb["n_rows_total"] - obs["n_rows_total"] == diff["n_rows_dropped"] == 7896
    assert (
        iedb["n_distinct_peptides_union"] - obs["n_distinct_peptides_union"]
        == diff["n_distinct_peptides_dropped"]
        == 7180
    )
    assert sum(diff["n_rows_dropped_by_length"].values()) == diff["n_rows_dropped"]
    assert sum(diff["n_rows_dropped_by_locus"].values()) == diff["n_rows_dropped"]
    for locus in ("HLA-DR", "HLA-DQ", "HLA-DP"):
        assert (
            iedb["n_rows_by_locus"][locus] - diff["n_rows_dropped_by_locus"][locus]
            == obs["n_rows_by_locus"][locus]
        )
    assert diff["n_rows_ge_12_lost"] == 0
    assert diff["n_distinct_peptides_ge_12_lost"] == 0


def test_ramarathinam_retains_locus_restrictions_without_promoting_inferred_alleles() -> None:
    """DR/DQ/DP restrictions stay locus-level; candidate sets never overwrite reported MHC (#632)."""
    entry = load_pmid_overrides()[34357683]
    assert "PXD025877" in entry["note"]
    assert "LB3.1" in entry["ip_antibody"]

    for locus, expected_candidate in [
        ("HLA-DR", "HLA-DRB1*12:01"),
        ("HLA-DQ", "HLA-DQB1*03:01"),
        ("HLA-DP", "HLA-DPB1*04:01"),
    ]:
        allele_set, provenance, set_size = expand_allele_set(locus, pmid=34357683, mhc_class="II")
        assert allele_set == expected_candidate
        assert provenance == "pmid_locus_pool"
        assert set_size == 1
