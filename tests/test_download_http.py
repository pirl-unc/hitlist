"""Exercise hitlist's datacache integration against real local HTTP responses."""

import contextlib
import gzip
import hashlib
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import datacache
import pytest
import requests

from hitlist import downloads


@pytest.fixture
def http_source(monkeypatch):
    state = {"payload": b"fresh data", "requests": [], "statuses": [], "etag": '"v1"'}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            state["requests"].append(dict(self.headers))
            status = state["statuses"].pop(0) if state["statuses"] else 200
            if status != 200:
                self.send_error(status)
                return
            payload = state["payload"]
            offset = (
                int(self.headers["Range"].split("=")[1].split("-")[0])
                if "Range" in self.headers
                else 0
            )
            self.send_response(206 if offset else 200)
            self.send_header("Content-Length", str(len(payload) - offset))
            if state["etag"]:
                self.send_header("ETag", state["etag"])
            if offset:
                self.send_header(
                    "Content-Range", f"bytes {offset}-{len(payload) - 1}/{len(payload)}"
                )
            self.end_headers()
            if state.pop("interrupt", False):
                self.wfile.write(payload[: 2 * 1024 * 1024])
                self.wfile.flush()
                self.connection.shutdown(socket.SHUT_RDWR)
                self.connection.close()
                state["statuses"] = [503, 503]
            else:
                with contextlib.suppress(BrokenPipeError, ConnectionResetError):
                    self.wfile.write(payload[offset:])

    # Avoid actual retry delays; retain datacache's retry decisions.
    monkeypatch.setattr("datacache.download.time.sleep", lambda seconds: None)
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", state
    finally:
        server.shutdown()
        server.server_close()
        worker.join()


@pytest.mark.parametrize("status, n_requests", [(404, 1), (429, 3), (503, 3)])
def test_failed_refresh_preserves_file(http_source, tmp_path, status, n_requests):
    url, state = http_source
    state["statuses"] = [status] * 3
    dest = tmp_path / "old.txt"
    dest.write_bytes(b"good cache")
    with pytest.raises(RuntimeError, match="Failed to download") as raised:
        downloads.download_to_file(url, dest, force=True, verbose=False)
    assert isinstance(raised.value.__cause__, requests.HTTPError)
    assert len(state["requests"]) == n_requests
    assert dest.read_bytes() == b"good cache"


def test_retry_then_success_and_provenance(http_source, tmp_path):
    url, state = http_source
    state["statuses"] = [503]
    dest = tmp_path / "data.txt"
    downloads.download_to_file(url + "/data?secret=omitted", dest, verbose=False)
    assert len(state["requests"]) == 2
    inspection = datacache.inspect_file(dest)
    assert inspection.status == "available"
    assert inspection.source_url == url + "/data"
    assert inspection.fetched_at
    assert not inspection.verified
    before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.iterdir()}
    downloads.download_to_file(url, dest, verbose=False)
    assert before == {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.iterdir()}
    assert len(state["requests"]) == 2


@pytest.mark.parametrize(
    "suffix, payload", [(".gz", gzip.compress(b"expanded")), (".html", b"<html>raw</html>")]
)
def test_default_keeps_raw_bytes(http_source, tmp_path, suffix, payload):
    url, state = http_source
    state["payload"] = payload
    dest = tmp_path / "arbitrary.csv"
    downloads.download_to_file(url + "/source" + suffix, dest, verbose=False)
    assert dest.read_bytes() == payload


def test_literal_decompression_and_failed_transform(http_source, tmp_path):
    url, state = http_source
    dest = tmp_path / "data.txt"
    state["payload"] = gzip.compress(b"expanded")
    # Query-bearing URLs were raw in hitlist even with decompress=True.
    downloads.download_to_file(url + "/data.gz?query=1", dest, decompress=True, verbose=False)
    assert dest.read_bytes() == state["payload"]
    downloads.download_to_file(url + "/data.gz", dest, decompress=True, force=True, verbose=False)
    assert dest.read_bytes() == b"expanded"
    state["payload"] = b"invalid gzip"
    with pytest.raises(RuntimeError):
        downloads.download_to_file(
            url + "/data.gz", dest, decompress=True, force=True, verbose=False
        )
    assert dest.read_bytes() == b"expanded"


def test_empty_response_is_retried_and_rejected(http_source, tmp_path):
    url, state = http_source
    state["payload"] = b""
    dest = tmp_path / "data"
    with pytest.raises(RuntimeError, match="empty"):
        downloads.download_to_file(url, dest, verbose=False)
    assert len(state["requests"]) == 3
    assert not dest.exists()


@pytest.mark.parametrize("with_hash", [True, False])
def test_resume_across_calls_keeps_good_destination(http_source, tmp_path, with_hash):
    url, state = http_source
    payload = b"ACGT" * (1024 * 1024)
    state.update(payload=payload, interrupt=True)
    dest = tmp_path / "dna.download"
    dest.write_bytes(b"old generation")
    digest = hashlib.sha256(payload).hexdigest() if with_hash else None
    kwargs = {
        "force": True,
        "verbose": False,
        "resume": True,
        "expected_size": len(payload),
        "expected_sha256": digest,
    }
    with pytest.raises(RuntimeError):
        downloads.download_to_file(url + "/dna.gz", dest, **kwargs)
    assert dest.read_bytes() == b"old generation"
    downloads.download_to_file(url + "/dna.gz", dest, **kwargs)
    assert dest.read_bytes() == payload
    last = state["requests"][-1]
    assert last["Range"] == "bytes=2097152-"
    assert last["If-Range"] == '"v1"'
    assert datacache.inspect_file(dest).recorded_sha256 == digest


@pytest.mark.parametrize("etag", [None, 'W/"weak"'])
def test_size_only_resume_refuses_missing_or_weak_validator(http_source, tmp_path, etag):
    url, state = http_source
    state["etag"] = etag
    dest = tmp_path / "out"
    with pytest.raises(RuntimeError):
        downloads.download_to_file(
            url, dest, verbose=False, resume=True, expected_size=len(state["payload"])
        )
    assert not dest.exists()


def test_socket_timeout_is_forwarded(tmp_path, monkeypatch):
    seen = {}

    def fail(url, **kwargs):
        seen.update(kwargs)
        raise requests.Timeout("stalled")

    monkeypatch.setattr(requests, "get", fail)
    monkeypatch.setattr("datacache.download.time.sleep", lambda seconds: None)
    with pytest.raises(RuntimeError) as raised:
        downloads.download_to_file("http://example.test/data", tmp_path / "out", verbose=False)
    assert isinstance(raised.value.__cause__, requests.Timeout)
    assert seen["timeout"] == 300.0
