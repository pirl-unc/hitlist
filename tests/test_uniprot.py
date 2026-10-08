"""Pinned reference identity, resource bounds, and data-management integration."""

import argparse
import hashlib
import io
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import datacache
import pytest
import requests
import yaml

from hitlist import cli, downloads, uniprot


@pytest.fixture
def references(tmp_path, monkeypatch):
    payloads = {
        "2015_10": b">sp|P1|OLD_HUMAN\nAAKCCK\n",
        "2026_03": b">sp|P2|NEW_HUMAN\nAAKDDK\n",
    }
    catalog = {
        "schema_version": 1,
        "license_statement": "Test UniProt attribution notice.\n",
        "collections": {
            "human": {
                "taxonomy_id": 9606,
                "selection": {"canonical": "all", "reviewed_isoforms": True},
                "default_release": "2015_10",
                "releases": {
                    release: {
                        "filename": "human.fasta",
                        "url": f"https://example.test/{release}.fasta",
                        "sha256": hashlib.sha256(body).hexdigest(),
                        "size_bytes": len(body),
                        "n_sequences": 1,
                        "license": "CC-BY-4.0",
                        "sources": [{"url": "https://example.test/source", "release": release}],
                    }
                    for release, body in payloads.items()
                },
            }
        },
    }
    path = tmp_path / "catalog.yaml"
    path.write_text(yaml.safe_dump(catalog))
    monkeypatch.setattr(uniprot, "_CATALOG_PATH", path)
    monkeypatch.setattr(downloads, "_override_data_dir", tmp_path / "data")
    monkeypatch.setattr(downloads, "_data_dir_cache", {})
    calls = []

    def respond(url, **kwargs):
        calls.append((url, kwargs))
        body = payloads[Path(url).stem]
        response = requests.Response()
        response.status_code = 200
        response.raw = io.BytesIO(body)
        response.headers["Content-Length"] = str(len(body))
        return response

    monkeypatch.setattr(requests, "get", respond)
    original = datacache.fetch_file
    monkeypatch.setattr(datacache, "fetch_file", lambda *a, **kw: original(*a, **kw, max_retries=0))
    return payloads, catalog, path, calls


@pytest.mark.parametrize("order", [("2015_10", "2026_03"), ("2026_03", "2015_10")])
def test_releases_coexist_and_keep_independent_identity(references, order):
    payloads, _, _, calls = references
    paths = {release: uniprot.fetch_uniprot_reference(release, verbose=False) for release in order}
    assert paths["2015_10"] != paths["2026_03"]
    for release, path in paths.items():
        assert path.read_bytes() == payloads[release]
        info = uniprot.uniprot_info(release, verify=True)
        assert info["release"] == release and info["verified"]
        assert info["sha256"] == hashlib.sha256(payloads[release]).hexdigest()
        assert info["fetched_at"]
    assert uniprot.fetch_uniprot_reference(verbose=False) == paths["2015_10"]
    assert len(calls) == 2
    assert {row["release"] for row in uniprot.list_uniprot_references()} == set(payloads)


def test_read_only_operations_do_not_create_cache_or_download(references):
    _, _, _, calls = references
    assert uniprot.uniprot_info()["status"] == "missing"
    assert len(uniprot.list_uniprot_references(verify=True)) == 2
    with pytest.raises(FileNotFoundError):
        uniprot.uniprot_path()
    assert not uniprot.remove_uniprot_reference("2015_10")
    assert not downloads.data_dir().exists()
    assert not calls


@pytest.mark.parametrize(
    "options", [{"release": "latest"}, {"release": "../2015_10"}, {"collection": "../human"}]
)
def test_unknown_references_fail_before_writes_or_network(references, options):
    with pytest.raises(ValueError, match="Unknown"):
        uniprot.fetch_uniprot_reference(**options)
    assert not downloads.data_dir().exists()
    assert not references[3]


def test_corrupt_cache_requires_explicit_repair(references):
    body = references[0]["2015_10"]
    path = uniprot.fetch_uniprot_reference(verbose=False)
    path.write_bytes(body.replace(b"AAK", b"ZZZ"))
    assert uniprot.uniprot_info()["verified"] is False
    assert uniprot.uniprot_info(verify=True)["status"] == "corrupt"
    with pytest.raises(ValueError, match="SHA-256"):
        uniprot.fetch_uniprot_reference(verbose=False)
    assert len(references[3]) == 1
    assert uniprot.fetch_uniprot_reference(force=True, verbose=False).read_bytes() == body


def test_oversized_or_wrong_hash_replacement_preserves_valid_file(references):
    body = references[0]["2015_10"]
    path = uniprot.fetch_uniprot_reference(verbose=False)
    for bad in (body * 3, body.replace(b"AAK", b"ZZZ")):
        references[0]["2015_10"] = bad
        with pytest.raises(ValueError, match=r"expected_size|SHA-256"):
            uniprot.fetch_uniprot_reference(force=True, verbose=False)
        assert path.read_bytes() == body
    assert uniprot.uniprot_info(verify=True)["verified"]


def test_interrupted_replacement_resumes_without_losing_installed_file(references, monkeypatch):
    body = references[0]["2015_10"]
    path = uniprot.fetch_uniprot_reference(verbose=False)

    def interrupt(url, **kwargs):
        response = requests.Response()
        response.status_code = 200
        response.raw = io.BytesIO()

        def chunks(**kwargs):
            yield body[:5]
            raise requests.ConnectionError("interrupted")

        response.iter_content = chunks
        return response

    monkeypatch.setattr(requests, "get", interrupt)
    with pytest.raises(requests.ConnectionError):
        uniprot.fetch_uniprot_reference(force=True, verbose=False)
    assert path.read_bytes() == body
    partials = list(path.parent.glob(".datacache-resume-*/partial"))
    assert len(partials) == 1 and partials[0].read_bytes() == body[:5]

    def resume(url, **kwargs):
        assert kwargs["headers"]["Range"] == "bytes=5-"
        response = requests.Response()
        response.status_code = 206
        response.headers["Content-Range"] = f"bytes 5-{len(body) - 1}/{len(body)}"
        response.raw = io.BytesIO(body[5:])
        return response

    monkeypatch.setattr(requests, "get", resume)
    assert uniprot.fetch_uniprot_reference(force=True, verbose=False).read_bytes() == body


def test_storage_preflight_counts_hidden_partials_and_preserves_other_files(references):
    body = references[0]["2015_10"]
    root = uniprot.uniprot_cache_dir()
    root.mkdir(parents=True)
    partial = root / ".partial"
    partial.write_bytes(b"x" * 100)
    budget = uniprot._TRANSFER_HEADROOM_BYTES + len(body) + 99
    with pytest.raises(ValueError, match="max_cache_bytes"):
        uniprot.fetch_uniprot_reference(max_cache_bytes=budget)
    assert partial.read_bytes() == b"x" * 100
    assert not references[3]


def test_force_requires_replacement_headroom(references):
    path = uniprot.fetch_uniprot_reference(verbose=False)
    budget = uniprot._cache_bytes() + uniprot._TRANSFER_HEADROOM_BYTES + path.stat().st_size - 1
    with pytest.raises(ValueError, match="max_cache_bytes"):
        uniprot.fetch_uniprot_reference(force=True, max_cache_bytes=budget)
    assert path.read_bytes() == references[0]["2015_10"]


@pytest.mark.parametrize(
    "options", [{"max_asset_bytes": 1}, {"max_cache_bytes": 0}, {"max_asset_bytes": True}]
)
def test_invalid_or_insufficient_asset_limits_do_not_download(references, options):
    with pytest.raises(ValueError):
        uniprot.fetch_uniprot_reference(**options)
    assert not references[3]


def test_concurrent_writers_cannot_overcommit_cache(references, monkeypatch):
    started, finish = Event(), Event()
    original = requests.get

    def respond(url, **kwargs):
        started.set()
        assert finish.wait(10)
        return original(url, **kwargs)

    monkeypatch.setattr(requests, "get", respond)
    budget = uniprot._TRANSFER_HEADROOM_BYTES + sum(map(len, references[0].values())) - 1
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(
            uniprot.fetch_uniprot_reference, "2015_10", max_cache_bytes=budget, verbose=False
        )
        assert started.wait(10)
        second = pool.submit(
            uniprot.fetch_uniprot_reference, "2026_03", max_cache_bytes=budget, verbose=False
        )
        finish.set()
        assert first.result().exists()
        with pytest.raises(ValueError, match="max_cache_bytes"):
            second.result()
    assert len(references[3]) == 1


@pytest.mark.parametrize(
    "component",
    ["uniprot", "uniprot/human", "uniprot/human/2015_10", "uniprot/human/2015_10/human.fasta"],
)
def test_managed_path_symlinks_cannot_redirect_writes_or_deletion(references, tmp_path, component):
    outside = tmp_path / "outside"
    outside.mkdir()
    target = downloads.data_dir() / component
    target.parent.mkdir(parents=True, exist_ok=True)
    target.symlink_to(outside, target_is_directory=True)
    for operation in (
        uniprot.fetch_uniprot_reference,
        uniprot.uniprot_path,
        uniprot.remove_uniprot_reference,
    ):
        with pytest.raises(ValueError, match="Symlinks"):
            operation("2015_10")
    assert uniprot.uniprot_info()["status"] == "corrupt"
    assert not list(outside.iterdir())


def test_remove_is_explicit_and_scoped_to_one_release(references):
    first = uniprot.fetch_uniprot_reference(verbose=False)
    second = uniprot.fetch_uniprot_reference("2026_03", verbose=False)
    other = first.with_name("manual.fasta")
    other.write_text("manual")
    with pytest.raises(ValueError, match="explicit"):
        uniprot.remove_uniprot_reference(None)
    assert uniprot.remove_uniprot_reference("2015_10")
    assert not first.exists() and second.exists() and other.exists()
    assert not Path(datacache.provenance.sidecar_path(first)).exists()
    assert not first.with_suffix(".license.txt").exists()


def test_download_keeps_required_license_statement_beside_fasta(references):
    path = uniprot.fetch_uniprot_reference(verbose=False)
    assert path.with_suffix(".license.txt").read_text() == references[1]["license_statement"]


def test_non_posix_never_silently_falls_back_to_unbounded_transfer(references, monkeypatch):
    monkeypatch.setattr(uniprot, "os", SimpleNamespace(name="nt"))
    with pytest.raises(NotImplementedError, match="POSIX"):
        uniprot.fetch_uniprot_reference()
    assert not references[3]
    assert not downloads.data_dir().exists()


def test_inventory_does_not_verify_an_escape_via_symlink(references, monkeypatch, tmp_path):
    outside = tmp_path / "outside.fasta"
    outside.write_bytes(references[0]["2015_10"])
    path = Path(uniprot.uniprot_info()["path"])
    path.parent.mkdir(parents=True)
    path.symlink_to(outside)
    monkeypatch.setattr(downloads, "data_assets", dict)
    row = downloads.list_cache_files(verify=True, include_unregistered=False)[0]
    assert row["status"] == "corrupt" and not row["verified"]
    assert "Symlinks" in row["error"]


def test_inventory_knows_versioned_hashes(references, monkeypatch):
    path = uniprot.fetch_uniprot_reference(verbose=False)
    monkeypatch.setattr(downloads, "data_assets", dict)
    rows = downloads.list_cache_files(verify=True, include_unregistered=False)
    row = next(r for r in rows if r["path"] == str(path))
    assert row["verified"] and "uniprot/human/2015_10" in row["names"]
    path.write_bytes(path.read_bytes().replace(b"AAK", b"ZZZ"))
    assert (
        downloads.list_cache_files(verify=True, include_unregistered=False)[0]["status"]
        == "corrupt"
    )


def test_cli_matches_api_and_requires_release_for_remove(references, capsys):
    parser = argparse.ArgumentParser()
    cli._build_data_parser(parser.add_subparsers(dest="command"))
    for command in ("fetch", "path", "info", "list", "remove"):
        args = parser.parse_args(
            [
                "data",
                "uniprot",
                command,
                "--json",
                *(["--release", "2015_10"] if command != "list" else []),
            ]
        )
        cli._handle_data(args)
        result = json.loads(capsys.readouterr().out)
        if command == "info":
            assert result["release"] == "2015_10"
        if command == "remove":
            assert result["removed"]
    with pytest.raises(SystemExit):
        parser.parse_args(["data", "uniprot", "remove"])


def test_descriptor_uses_exact_bytes_not_path_or_release_guess(references):
    reference = uniprot.uniprot_info()
    assert (
        uniprot.uniprot_reference_for_digest(reference["sha256"], reference["size_bytes"])[
            "release"
        ]
        == "2015_10"
    )
    assert uniprot.uniprot_reference_for_digest("0" * 64, reference["size_bytes"]) is None
    assert (
        uniprot.uniprot_reference_for_digest(reference["sha256"], reference["size_bytes"] + 1)
        is None
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("filename", "../human.fasta"),
        ("filename", "human.fasta.gz"),
        ("sha256", "bad"),
        ("size_bytes", 0),
        ("url", "http://mutable.test"),
    ],
)
def test_catalog_rejects_unsafe_or_unpinned_definitions(references, field, value):
    _, catalog, path, _ = references
    catalog["collections"]["human"]["releases"]["2015_10"][field] = value
    path.write_text(yaml.safe_dump(catalog))
    with pytest.raises(ValueError):
        uniprot.uniprot_catalog()
