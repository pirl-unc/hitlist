"""Tests for #491: pmhc --mhc-allele splitting a pasted genotype string
(space- or comma-joined, as one shell token) into individual alleles, and
the --mhc-alleles plural alias."""

from __future__ import annotations

import pandas as pd


def test_split_allele_tokens_passes_through_already_separate_tokens():
    from hitlist.cli import _split_allele_tokens

    assert _split_allele_tokens(["HLA-A*02:01", "HLA-B*07:02"]) == [
        "HLA-A*02:01",
        "HLA-B*07:02",
    ]


def test_split_allele_tokens_splits_a_single_space_joined_token():
    """The exact failure mode from #491: one quoted shell argument holding
    a whole pasted genotype string."""
    from hitlist.cli import _split_allele_tokens

    assert _split_allele_tokens(["HLA-A*02:01 HLA-B*07:02 HLA-C*05:01"]) == [
        "HLA-A*02:01",
        "HLA-B*07:02",
        "HLA-C*05:01",
    ]


def test_split_allele_tokens_splits_a_single_comma_joined_token():
    from hitlist.cli import _split_allele_tokens

    assert _split_allele_tokens(["HLA-A*02:01,HLA-B*07:02,HLA-C*05:01"]) == [
        "HLA-A*02:01",
        "HLA-B*07:02",
        "HLA-C*05:01",
    ]


def test_split_allele_tokens_handles_mixed_commas_spaces_and_repeats():
    from hitlist.cli import _split_allele_tokens

    assert _split_allele_tokens(
        ["HLA-A*02:01, HLA-B*07:02", "HLA-C*05:01", "HLA-A*03:01 , HLA-B*44:02"]
    ) == [
        "HLA-A*02:01",
        "HLA-B*07:02",
        "HLA-C*05:01",
        "HLA-A*03:01",
        "HLA-B*44:02",
    ]


def test_split_allele_tokens_empty_input():
    from hitlist.cli import _split_allele_tokens

    assert _split_allele_tokens([]) == []


def test_pmhc_cli_splits_a_pasted_genotype_string(monkeypatch):
    """End-to-end: the CLI's own argument plumbing must apply the split,
    not just the helper function in isolation."""
    from hitlist import cli, pmhc_query

    captured = {}

    def fake_query(**kwargs):
        captured.update(kwargs)
        return pd.DataFrame(columns=["gene_name", "peptide"])

    monkeypatch.setattr(pmhc_query, "query", fake_query)
    monkeypatch.setattr(
        "sys.argv",
        ["hitlist", "pmhc", "--gene", "SSX1", "--mhc-allele", "HLA-A*02:01 HLA-B*07:02"],
    )
    cli.main()

    assert captured["alleles"] == ["HLA-A*02:01", "HLA-B*07:02"]


def test_pmhc_cli_accepts_mhc_alleles_plural_alias(monkeypatch):
    """--mhc-alleles must resolve to the same destination as --mhc-allele."""
    from hitlist import cli, pmhc_query

    captured = {}

    def fake_query(**kwargs):
        captured.update(kwargs)
        return pd.DataFrame(columns=["gene_name", "peptide"])

    monkeypatch.setattr(pmhc_query, "query", fake_query)
    monkeypatch.setattr(
        "sys.argv",
        ["hitlist", "pmhc", "--gene", "SSX1", "--mhc-alleles", "HLA-A*02:01", "HLA-B*07:02"],
    )
    cli.main()

    assert captured["alleles"] == ["HLA-A*02:01", "HLA-B*07:02"]
