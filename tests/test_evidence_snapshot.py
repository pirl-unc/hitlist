"""Evidence generations must own their file bytes after publication."""

import subprocess
from types import SimpleNamespace

import pytest

from hitlist import evidence_snapshot


def test_snapshot_file_cleans_partial_clone_and_falls_back_to_copy(tmp_path, monkeypatch):
    source = tmp_path / "source.txt"
    source.write_bytes(b"original evidence")
    target = tmp_path / "nested" / "target.txt"
    monkeypatch.setattr(evidence_snapshot, "sys", SimpleNamespace(platform="darwin"))

    def failed_clone(command, **_kwargs):
        target.write_bytes(b"partial clone")
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(evidence_snapshot.subprocess, "run", failed_clone)
    evidence_snapshot.snapshot_file(source, target)

    assert target.read_bytes() == b"original evidence"
    assert target.stat().st_ino != source.stat().st_ino
    source.write_bytes(b"later source edit")
    assert target.read_bytes() == b"original evidence"


def test_snapshot_file_refuses_existing_target_without_changing_it(tmp_path):
    source = tmp_path / "source.txt"
    target = tmp_path / "target.txt"
    source.write_bytes(b"source")
    target.write_bytes(b"existing")

    with pytest.raises(FileExistsError, match="already exists"):
        evidence_snapshot.snapshot_file(source, target)

    assert target.read_bytes() == b"existing"


def test_snapshot_file_resolves_source_symlink_to_independent_bytes(tmp_path):
    source = tmp_path / "source.txt"
    source.write_bytes(b"source bytes")
    link = tmp_path / "source-link.txt"
    link.symlink_to(source)
    target = tmp_path / "target.txt"

    evidence_snapshot.snapshot_file(link, target)
    source.write_bytes(b"later edit")

    assert target.read_bytes() == b"source bytes"
    assert target.stat().st_ino != source.stat().st_ino
