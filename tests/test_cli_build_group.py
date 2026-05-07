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

"""Tests for the canonical ``hitlist build`` top-level group and the
deprecation notice on the legacy ``hitlist data {build,fetch-proteomes}``
entry points.  The legacy ``index`` subcommand was removed in v1.30.41
when the per-source CSV-scan cache was obliterated; ``get_index()`` now
derives counts from ``observations.parquet`` directly so there's nothing
to "build" anymore."""

from __future__ import annotations

import argparse


def _build_full_parser() -> argparse.ArgumentParser:
    """Construct the same parser ``main()`` constructs, so we can drive
    argparse end-to-end without invoking subcommand handlers."""
    from hitlist.cli import (
        _build_data_parser,
        _build_top_level_build_parser,
    )

    parser = argparse.ArgumentParser(prog="hitlist")
    sub = parser.add_subparsers(dest="command")
    _build_data_parser(sub)
    _build_top_level_build_parser(sub)
    return parser


def test_build_group_routes_observations_args_into_namespace():
    """``hitlist build observations --proteome-release 114 --use-uniprot``
    must produce a Namespace with the expected fields, and command/build_command
    must identify the leaf so the dispatcher can route correctly."""
    parser = _build_full_parser()
    args = parser.parse_args(
        ["build", "observations", "--proteome-release", "114", "--use-uniprot"]
    )
    assert args.command == "build"
    assert args.build_command == "observations"
    assert args.proteome_release == 114
    assert args.use_uniprot is True
    assert args.no_mappings is False
    assert args.force is False


def test_build_group_routes_proteomes_args():
    parser = _build_full_parser()
    args = parser.parse_args(["build", "proteomes", "--min-observations", "500"])
    assert args.command == "build"
    assert args.build_command == "proteomes"
    assert args.min_observations == 500
    assert args.use_uniprot is False
    assert args.force is False


def test_legacy_data_subcommands_still_parse():
    """Legacy paths must still parse identically to the new paths so users
    on old scripts aren't broken until v2.0."""
    parser = _build_full_parser()
    legacy = parser.parse_args(["data", "build", "--proteome-release", "114", "--use-uniprot"])
    canonical = parser.parse_args(
        ["build", "observations", "--proteome-release", "114", "--use-uniprot"]
    )
    # Same flag-derived attributes.
    assert legacy.proteome_release == canonical.proteome_release
    assert legacy.use_uniprot == canonical.use_uniprot
    assert legacy.no_mappings == canonical.no_mappings
    # Different command identification (so dispatcher can decide).
    assert legacy.command == "data" and legacy.data_command == "build"
    assert canonical.command == "build" and canonical.build_command == "observations"


def test_legacy_data_build_emits_deprecation_notice(monkeypatch, capsys):
    """``hitlist data build`` must still work but print a stderr notice
    pointing at the new path."""
    from hitlist.cli import _handle_data

    called = {}

    def fake_data_build(_args):
        called["data_build"] = True

    monkeypatch.setattr("hitlist.cli._data_build", fake_data_build)

    args = argparse.Namespace(
        command="data",
        data_command="build",
    )
    _handle_data(args)

    captured = capsys.readouterr()
    assert "deprecated" in captured.err
    assert "hitlist build observations" in captured.err
    assert called.get("data_build") is True


def test_build_dispatcher_calls_same_handlers(monkeypatch):
    """``hitlist build observations`` must reach the same handler as
    ``hitlist data build`` so behaviour stays identical during the
    deprecation window."""
    from hitlist.cli import _handle_build

    called = {}

    def fake_data_build(_args):
        called["target"] = "data_build"

    monkeypatch.setattr("hitlist.cli._data_build", fake_data_build)

    args = argparse.Namespace(
        command="build",
        build_command="observations",
    )
    _handle_build(args)
    assert called["target"] == "data_build"
