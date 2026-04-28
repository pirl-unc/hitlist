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

"""Tests for the v1.28.0 ``hitlist report`` rewrite — should default to
``observations.parquet`` and only fall back to CSV scanning when explicitly
opted in.
"""

from __future__ import annotations

import pandas as pd


def _write_obs_fixture(tmp_path):
    """Minimal observations.parquet with the columns ``run_report`` projects."""
    df = pd.DataFrame(
        {
            "peptide": ["AAAAAAAAA", "BBBBBBBBB"],
            "pmid": [1, 2],
            "mhc_class": ["I", "I"],
            "mhc_restriction": ["HLA-A*02:01", "HLA-B*07:02"],
            "is_monoallelic": [False, True],
            "monoallelic_host": ["", "Expi293F"],
            "cell_line_name": ["JY", ""],
            "source_tissue": ["", "spleen"],
            "disease": ["", ""],
            "src_cancer": [False, False],
            "src_healthy_tissue": [False, True],
            "species": ["Homo sapiens"] * 2,
            "mhc_species": ["Homo sapiens"] * 2,
        }
    )
    path = tmp_path / "observations.parquet"
    df.to_parquet(path, index=False)
    return path


def test_run_report_defaults_to_observations_parquet(tmp_path, monkeypatch):
    """The default path reads ``observations.parquet`` and never touches
    ``scanner.scan`` (which is the minutes-slow CSV path)."""
    from hitlist import report as report_mod

    obs_path = _write_obs_fixture(tmp_path)
    monkeypatch.setattr("hitlist.observations.observations_path", lambda: obs_path)

    # If anything calls scanner.scan during the default path, fail the test.
    def _explode(*a, **kw):
        raise AssertionError("scanner.scan must not be called when from_csv=False")

    monkeypatch.setattr("hitlist.scanner.scan", _explode)

    text = report_mod.run_report()
    assert "HITLIST DATA QUALITY REPORT" in text
    # MHC class section appears.
    assert "MHC Class Distribution" in text


def test_run_report_from_csv_uses_scanner(tmp_path, monkeypatch):
    """``from_csv=True`` must route to the legacy scanner.scan path so users
    can still get a fresh-data sanity check before observations.parquet exists."""
    from hitlist import report as report_mod

    captured = {}

    def fake_scan(**kwargs):
        captured.update(kwargs)
        # Return a tiny DataFrame with the columns generate_report uses.
        return pd.DataFrame(
            {
                "peptide": ["X"],
                "pmid": [1],
                "mhc_class": ["I"],
                "mhc_restriction": ["HLA-A*02:01"],
                "is_monoallelic": [False],
                "monoallelic_host": [""],
                "cell_line_name": [""],
                "source_tissue": [""],
                "disease": [""],
                "src_cancer": [False],
                "src_healthy_tissue": [False],
            }
        )

    monkeypatch.setattr("hitlist.scanner.scan", fake_scan)
    # Stub away the path-resolution side of the legacy fallback so it
    # doesn't try to look up real datasets.
    monkeypatch.setattr("hitlist.downloads.get_path", lambda name: f"/tmp/fake/{name}.csv")

    text = report_mod.run_report(from_csv=True)
    assert captured.get("classify_source") is True
    assert "HITLIST DATA QUALITY REPORT" in text


def test_run_report_no_observations_parquet_prints_hint(tmp_path, monkeypatch, capsys):
    """If the parquet hasn't been built and the user hasn't passed
    --from-csv, run_report prints a one-line hint pointing at the new
    canonical build path and returns ``""``."""
    from hitlist import report as report_mod

    monkeypatch.setattr("hitlist.observations.is_built", lambda: False)
    text = report_mod.run_report()
    assert text == ""
    captured = capsys.readouterr()
    assert "hitlist build observations" in captured.err
    assert "--from-csv" in captured.err


def test_generate_report_handles_pmid_overrides_without_legacy_label_key():
    """Regression for v1.29.5: ``label`` was renamed to ``study_label`` in
    v1.7.0 but the report still indexed ``entry['label']``, so any call
    against a built parquet whose first row had a curated PMID raised
    ``KeyError: 'label'``."""
    import pandas as pd

    from hitlist import report as report_mod

    # Pick a PMID that's in pmid_overrides.yaml so the override block
    # is exercised. Marcu 2021 (33858848) has been curated since v1.0.
    df = pd.DataFrame(
        {
            "peptide": ["AAAAAAAAA"],
            "pmid": [33858848],
            "mhc_class": ["I"],
            "mhc_restriction": ["HLA-A*02:01"],
            "is_monoallelic": [False],
            "monoallelic_host": [""],
            "cell_line_name": [""],
            "source_tissue": [""],
            "disease": [""],
            "src_cancer": [False],
            "src_healthy_tissue": [True],
        }
    )
    text = report_mod.generate_report(df)
    assert "Curated Study Overrides Applied" in text
    assert "PMID 33858848" in text
