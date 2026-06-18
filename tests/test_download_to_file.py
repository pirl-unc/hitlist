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

"""Tests for the public ``download_to_file`` helper: cache reporting, progress
streaming, and ``.zip``/``.gz`` decompression (hitlist#341)."""

from __future__ import annotations

import gzip
import io
import zipfile
from contextlib import contextmanager

from hitlist import downloads


@contextmanager
def _fake_response(payload: bytes):
    """Mimic the context-manager object returned by urlopen() (no headers)."""
    yield io.BytesIO(payload)


def _serve(monkeypatch, payload: bytes) -> dict:
    """Monkeypatch urlopen to return *payload*; return a call counter."""
    calls = {"n": 0}

    def fake_urlopen(url, timeout=None):
        calls["n"] += 1
        return _fake_response(payload)

    monkeypatch.setattr(downloads.urllib.request, "urlopen", fake_urlopen)
    return calls


def _no_network(monkeypatch) -> None:
    """Make any urlopen call fail loudly (proves the cache short-circuit)."""

    def boom(url, timeout=None):
        raise AssertionError(f"unexpected network call to {url}")

    monkeypatch.setattr(downloads.urllib.request, "urlopen", boom)


def test_is_compressed_heuristic(tmp_path):
    assert downloads._is_compressed("http://x/f.tsv.zip", tmp_path / "f.tsv")
    assert downloads._is_compressed("http://x/f.tsv.gz", tmp_path / "f.tsv")
    # dest keeps the archive suffix -> leave compressed.
    assert not downloads._is_compressed("http://x/f.gz", tmp_path / "f.gz")
    assert not downloads._is_compressed("http://x/f.tsv", tmp_path / "f.tsv")


def test_cache_hit_short_circuits(tmp_path, monkeypatch, capsys):
    dest = tmp_path / "out.txt"
    dest.write_bytes(b"cached")
    _no_network(monkeypatch)

    out = downloads.download_to_file("http://x/out.txt", dest, label="thing")

    assert out == dest
    assert dest.read_bytes() == b"cached"
    assert "already cached" in capsys.readouterr().out


def test_fresh_download_streams_and_reports(tmp_path, monkeypatch, capsys):
    dest = tmp_path / "out.fasta"
    calls = _serve(monkeypatch, b">sp|P1\nACDEF\n")

    out = downloads.download_to_file("http://x/out.fasta", dest, label="prot")

    assert out == dest
    assert dest.read_bytes() == b">sp|P1\nACDEF\n"
    assert calls["n"] == 1
    assert "downloading from" in capsys.readouterr().out


def test_force_redownloads_over_existing(tmp_path, monkeypatch):
    dest = tmp_path / "out.txt"
    dest.write_bytes(b"stale")
    _serve(monkeypatch, b"fresh")

    downloads.download_to_file("http://x/out.txt", dest, force=True, verbose=False)

    assert dest.read_bytes() == b"fresh"


def test_verbose_false_is_silent(tmp_path, monkeypatch, capsys):
    dest = tmp_path / "out.txt"
    _serve(monkeypatch, b"data")

    downloads.download_to_file("http://x/out.txt", dest, verbose=False)

    assert capsys.readouterr().out == ""


def test_decompress_gz(tmp_path, monkeypatch):
    dest = tmp_path / "genes.tsv"
    _serve(monkeypatch, gzip.compress(b"col1\tcol2\n1\t2\n"))

    downloads.download_to_file("http://x/genes.tsv.gz", dest, decompress=True, verbose=False)

    assert dest.read_bytes() == b"col1\tcol2\n1\t2\n"
    # The compressed archive is cleaned up, only the expanded file remains.
    assert not dest.with_name(dest.name + ".gz").exists()


def test_decompress_zip_picks_named_member(tmp_path, monkeypatch):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("table.tsv", b"the wanted member\n")
        z.writestr("readme.txt", b"ignore me\n")
    dest = tmp_path / "table.tsv"
    _serve(monkeypatch, buf.getvalue())

    downloads.download_to_file("http://x/table.tsv.zip", dest, decompress=True, verbose=False)

    assert dest.read_bytes() == b"the wanted member\n"
    assert not dest.with_name(dest.name + ".zip").exists()


def test_decompress_zip_falls_back_to_largest_member(tmp_path, monkeypatch):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("small.txt", b"x")
        z.writestr("big.tsv", b"the biggest member by far\n")
    dest = tmp_path / "out.tsv"  # name matches no member -> largest wins
    _serve(monkeypatch, buf.getvalue())

    downloads.download_to_file("http://x/out.tsv.zip", dest, decompress=True, verbose=False)

    assert dest.read_bytes() == b"the biggest member by far\n"
