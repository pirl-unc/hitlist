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

"""Locked read-modify-write of manifest.json and the gene cache (#883)."""

from __future__ import annotations

import multiprocessing as mp

import pytest

from hitlist import downloads, genes

N_WORKERS = 6
N_KEYS_PER_WORKER = 15


def _manifest_worker(data_dir: str, worker: int) -> None:
    downloads.set_data_dir(data_dir)
    for i in range(N_KEYS_PER_WORKER):
        with downloads.manifest_transaction() as manifest:
            manifest.setdefault("datasets", {})[f"w{worker}-k{i}"] = {"path": "x"}


def _gene_worker(data_dir: str, worker: int) -> None:
    downloads.set_data_dir(data_dir)
    for i in range(N_KEYS_PER_WORKER):
        genes._record_resolution(f"w{worker}-q{i}", {"symbols": [f"S{worker}_{i}"]})


def _run_in_processes(target, data_dir) -> None:
    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=target, args=(str(data_dir), w)) for w in range(N_WORKERS)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=120)
        assert p.exitcode == 0


def test_concurrent_manifest_transactions_do_not_lose_updates(tmp_path):
    downloads.set_data_dir(tmp_path)
    _run_in_processes(_manifest_worker, tmp_path)
    datasets = downloads.load_manifest()["datasets"]
    expected = {f"w{w}-k{i}" for w in range(N_WORKERS) for i in range(N_KEYS_PER_WORKER)}
    assert set(datasets) == expected


def test_manifest_transaction_does_not_save_on_exception(tmp_path):
    downloads.set_data_dir(tmp_path)
    with downloads.manifest_transaction() as manifest:
        manifest["datasets"]["keep"] = {"path": "a"}
    with pytest.raises(RuntimeError), downloads.manifest_transaction() as manifest:
        manifest["datasets"]["dropped"] = {"path": "b"}
        raise RuntimeError("boom")
    assert set(downloads.load_manifest()["datasets"]) == {"keep"}
    assert not list(tmp_path.glob(".manifest-*.tmp"))


def test_concurrent_gene_cache_writes_do_not_lose_updates(tmp_path):
    downloads.set_data_dir(tmp_path)
    _run_in_processes(_gene_worker, tmp_path)
    cache = genes._load_cache()
    expected = {f"w{w}-q{i}" for w in range(N_WORKERS) for i in range(N_KEYS_PER_WORKER)}
    assert set(cache) == expected
    assert not list(genes._cache_path().parent.glob(".hgnc-*.tmp"))
