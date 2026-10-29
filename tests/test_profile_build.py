"""Profiler failures must retain diagnostics and restore instrumented functions."""

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/profile_build.py"
SPEC = importlib.util.spec_from_file_location("profile_build", SCRIPT)
profile_build = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(profile_build)


@pytest.mark.parametrize("fail", [False, True])
def test_profile_records_success_or_failure_and_restores_functions(tmp_path, monkeypatch, fail):
    from hitlist import builder, scanner

    def scan():
        if fail:
            raise RuntimeError("source unavailable")
        return tmp_path / "observations.parquet"

    def build(**options):
        assert options == {"force": True, "build_mappings": False}
        return scanner.scan()

    monkeypatch.setattr(scanner, "scan", scan)
    monkeypatch.setattr(builder, "build_observations", build)
    monkeypatch.setattr(profile_build, "STAGES", [("hitlist.scanner", "scan")])
    report_path = tmp_path / "report.json"
    options = {"build_mappings": False}
    if fail:
        with pytest.raises(RuntimeError, match="source unavailable"):
            profile_build.measure_build(
                report_path, options=options, profile_path=tmp_path / "build.prof"
            )
        assert (tmp_path / "build.prof").is_file()
        assert (tmp_path / "build.txt").is_file()
    else:
        assert (
            profile_build.measure_build(report_path, options=options)
            == tmp_path / "observations.parquet"
        )
        assert not (tmp_path / "build.prof").exists()
    assert scanner.scan is scan
    report = json.loads(report_path.read_text())
    assert report["status"] == ("failed" if fail else "success")
    assert report["profiled"] is fail
    assert report["stages"][0]["status"] == report["status"]
    assert report["total"]["wall_seconds"] > 0
    assert report["total"]["process_peak_rss_bytes"] > 0
    assert "not process-tree sum" in report["rss_scope"]
    assert options == {"build_mappings": False}
