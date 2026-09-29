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

import requests

from hitlist import downloads


def _serve(monkeypatch, payload: bytes) -> dict:
    """Serve bytes at the HTTP boundary used by datacache."""
    calls = {"n_requests": 0}

    def fake_get(url, **kwargs):
        calls["n_requests"] += 1
        response = requests.Response()
        response.status_code = 200
        response.raw = io.BytesIO(payload)
        return response

    monkeypatch.setattr(requests, "get", fake_get)
    return calls


def _no_network(monkeypatch) -> None:
    def boom(url, **kwargs):
        raise AssertionError(f"unexpected network call to {url}")

    monkeypatch.setattr(requests, "get", boom)


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
    assert calls["n_requests"] == 1
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


def test_decompress_leaves_archive_destination_compressed(tmp_path, monkeypatch):
    dest = tmp_path / "genes.tsv.gz"
    payload = gzip.compress(b"col1\tcol2\n1\t2\n")
    _serve(monkeypatch, payload)
    downloads.download_to_file("http://x/genes.tsv.gz", dest, decompress=True, verbose=False)
    assert dest.read_bytes() == payload


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
