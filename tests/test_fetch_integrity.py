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

"""Size/checksum integrity of ``downloads.fetch`` for FETCHABLE_DATASETS (#882)."""

from __future__ import annotations

import hashlib

import pytest

from hitlist import downloads

GOOD = b">sp|P1|X\n" + b"MKV" * 400 + b"\n"


@pytest.fixture
def data_dir(tmp_path):
    downloads.set_data_dir(tmp_path)
    yield tmp_path
    downloads._override_data_dir = None


@pytest.fixture
def fake_download(monkeypatch):
    """Replace the network fetch; ``payload`` is what the next download writes."""
    state = {"payload": GOOD, "calls": 0}

    def fake(url, dest, **kwargs):
        state["calls"] += 1
        dest.write_bytes(state["payload"])
        return dest

    monkeypatch.setattr(downloads, "download_to_file", fake)
    return state


def test_every_fetchable_dataset_has_a_min_size():
    for name, ds in downloads.FETCHABLE_DATASETS.items():
        assert ds["min_size_bytes"] > 0, name


def test_fetch_records_sha256_and_size(data_dir, fake_download):
    path = downloads.fetch("hpv16")
    entry = downloads.info("hpv16")
    assert path.read_bytes() == GOOD
    assert entry["size_bytes"] == len(GOOD)
    assert entry["sha256"] == hashlib.sha256(GOOD).hexdigest()


@pytest.mark.parametrize("payload", [b"", b"<html>no</html>"])
def test_fetch_rejects_tiny_download_and_removes_it(data_dir, fake_download, payload):
    fake_download["payload"] = payload
    with pytest.raises(RuntimeError, match="empty or error body"):
        downloads.fetch("hpv16")
    assert not (data_dir / "hpv16.fasta").exists()
    assert "hpv16" not in downloads.list_datasets()


def test_cache_hit_reuses_valid_file(data_dir, fake_download):
    downloads.fetch("hpv16")
    downloads.fetch("hpv16")
    assert fake_download["calls"] == 1


def test_cache_hit_refetches_truncated_file(data_dir, fake_download):
    path = downloads.fetch("hpv16")
    path.write_bytes(GOOD[:-10])
    downloads.fetch("hpv16")
    assert fake_download["calls"] == 2
    assert path.read_bytes() == GOOD


def test_cache_hit_refetches_same_size_corruption(data_dir, fake_download):
    path = downloads.fetch("hpv16")
    path.write_bytes(b"X" * len(GOOD))
    downloads.fetch("hpv16")
    assert fake_download["calls"] == 2
    assert path.read_bytes() == GOOD


def test_cache_hit_refetches_empty_leftover_file(data_dir, fake_download):
    (data_dir / "hpv16.fasta").write_bytes(b"")
    downloads.fetch("hpv16")
    assert fake_download["calls"] == 1
    assert (data_dir / "hpv16.fasta").read_bytes() == GOOD


def test_unrecorded_cache_hit_is_adopted_with_notice(data_dir, fake_download, capsys):
    (data_dir / "hpv16.fasta").write_bytes(GOOD)
    downloads.fetch("hpv16")
    assert fake_download["calls"] == 0
    assert "adopting existing" in capsys.readouterr().err
    assert "hpv16" in downloads.list_datasets()
