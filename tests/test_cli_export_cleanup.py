# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the v1.27.0 export cleanup:
- ``bulk-proteomics`` is the canonical name; ``bulk`` is an alias.
- ``peptide-counts --by class|study`` replaces ``summary`` and ``counts``.
- ``alleles`` and ``data-alleles`` emit deprecation notices.

All legacy commands continue to dispatch to the same handlers so behavior
is unchanged during the v1.x deprecation window.
"""

from __future__ import annotations

import argparse


def _build_full_parser() -> argparse.ArgumentParser:
    """Reconstruct the full CLI parser the way ``main()`` does, so we can
    drive argparse end-to-end without calling subcommand handlers."""
    from hitlist.cli import _build_data_parser, _build_top_level_build_parser

    parser = argparse.ArgumentParser(prog="hitlist")
    sub = parser.add_subparsers(dest="command")
    _build_data_parser(sub)
    _build_top_level_build_parser(sub)

    p_export = sub.add_parser("export")
    export_sub = p_export.add_subparsers(dest="export_command")
    # We only register the parsers this PR touches.  The module-level
    # main() registration is the source of truth; this is a focused subset.
    from hitlist.cli import _add_export_bulk_proteomics_args, _add_peptide_counts_args

    p_pc = export_sub.add_parser("peptide-counts")
    _add_peptide_counts_args(p_pc)

    p_bulk_prot = export_sub.add_parser("bulk-proteomics")
    _add_export_bulk_proteomics_args(p_bulk_prot)

    return parser


def test_peptide_counts_default_by_is_class():
    parser = _build_full_parser()
    args = parser.parse_args(["export", "peptide-counts"])
    assert args.export_command == "peptide-counts"
    assert args.by == "class"
    # Source defaults but is unused for --by class.
    assert args.source == "merged"


def test_peptide_counts_by_study_takes_source():
    parser = _build_full_parser()
    args = parser.parse_args(["export", "peptide-counts", "--by", "study", "--source", "iedb"])
    assert args.by == "study"
    assert args.source == "iedb"


def test_peptide_counts_by_class_takes_class_filter():
    parser = _build_full_parser()
    args = parser.parse_args(["export", "peptide-counts", "--by", "class", "--class", "I"])
    assert args.by == "class"
    assert args.mhc_class == "I"


def test_bulk_proteomics_canonical_name_parses():
    parser = _build_full_parser()
    args = parser.parse_args(
        ["export", "bulk-proteomics", "--granularity", "peptide", "--cell-line", "JY", "K562"]
    )
    assert args.export_command == "bulk-proteomics"
    assert args.granularity == "peptide"
    assert args.cell_line == ["JY", "K562"]


def test_legacy_export_subcommands_emit_deprecation_notices(monkeypatch, capsys):
    """``hitlist export {summary,counts,alleles,data-alleles,bulk}`` must
    print a stderr notice on every invocation."""
    import pandas as pd

    from hitlist import cli

    # The dispatch helpers are imported lazily inside ``_export`` from
    # ``hitlist.export``, so we patch the source module not ``hitlist.cli``.
    monkeypatch.setattr("hitlist.export.generate_species_summary", lambda **kw: pd.DataFrame())
    monkeypatch.setattr("hitlist.export.validate_mhc_alleles", lambda: pd.DataFrame())
    monkeypatch.setattr("hitlist.export.collect_alleles_from_data", lambda **kw: pd.DataFrame())
    monkeypatch.setattr("hitlist.export.count_peptides_by_study", lambda **kw: pd.DataFrame())

    cases = [
        ("summary", "hitlist export peptide-counts --by class"),
        ("counts", "hitlist export peptide-counts --by study"),
        ("alleles", "hitlist qc normalization"),
        ("data-alleles", "hitlist qc resolution"),
    ]
    for legacy_cmd, hint in cases:
        args = argparse.Namespace(
            export_command=legacy_cmd,
            mhc_class=None,
            source="merged",
            output=None,
        )
        cli._export(args)
        captured = capsys.readouterr()
        assert "deprecated" in captured.err
        assert hint in captured.err
