"""Conservation of curated ``ms_samples`` records from YAML to export.

Two independent leaks were removing curated sample records between the
YAML and every consumer (issues #438 and #437):

* ``load_pmid_overrides()`` built its mapping with a dict comprehension
  keyed by PMID, so a second entry for an already-curated PMID silently
  replaced the first one — including all of its ``ms_samples``.
* ``generate_ms_samples_table()`` dropped every ``n_samples: 0`` record,
  which is precisely the explicit ``profiled: false`` curation that says
  "this arm exists in the paper and was deliberately not profiled".

Both are silent: nothing raised, nothing warned, and the only symptom was
a sample count that did not match the file. These tests pin the inventory
at each hop so a future edit cannot re-open either leak.
"""

from __future__ import annotations

import pandas as pd
import pytest
import yaml

from hitlist import curation
from hitlist.curation import load_pmid_overrides, pmid_source_organism
from hitlist.export import _observation_eligible_samples, generate_ms_samples_table

#: The four records carrying explicit ``profiled: false`` curation (#437).
UNPROFILED_SAMPLES = {
    (36589698, "healthy donor PBMCs"),
    (38920720, "tumor tissue"),
    (35051231, "P1 lung — X31-infected"),
    (35051231, "moDC cross-presentation — uninfected control"),
}


def _raw_entries() -> list[dict]:
    """The YAML as written, before any de-duplication by the loader."""
    with open(curation._data_path("pmid_overrides.yaml")) as f:
        return yaml.safe_load(f)


def _raw_sample_keys() -> list[tuple[int, str]]:
    return [
        (int(entry["pmid"]), str(sample.get("sample_label", "")))
        for entry in _raw_entries()
        for sample in (entry.get("ms_samples") or [])
    ]


def _loaded_sample_keys() -> list[tuple[int, str]]:
    return [
        (pmid, str(sample.get("sample_label", "")))
        for pmid, entry in load_pmid_overrides().items()
        for sample in (entry.get("ms_samples") or [])
    ]


# ── #438: duplicate PMID entries ────────────────────────────────────────────


def test_packaged_overrides_declare_each_pmid_once():
    """A PMID curated twice means one of the two blocks is invisible."""
    seen: dict[int, int] = {}
    for entry in _raw_entries():
        seen[int(entry["pmid"])] = seen.get(int(entry["pmid"]), 0) + 1
    assert [pmid for pmid, count in seen.items() if count > 1] == []


def test_load_pmid_overrides_rejects_duplicate_pmids(tmp_path, monkeypatch):
    """Duplicate identifiers must fail loudly instead of last-wins.

    The real file had two entries each for PMID 33460454 and 28188227;
    the source-organism blocks added by #307 silently displaced the study
    blocks that carried the ``ms_samples``.
    """
    dup_yaml = tmp_path / "pmid_overrides.yaml"
    dup_yaml.write_text(
        yaml.safe_dump(
            [
                {"pmid": 12345678, "study_label": "first", "ms_samples": [{"sample_label": "a"}]},
                {"pmid": 12345678, "study_label": "second", "source_organism": "Homo sapiens"},
            ]
        )
    )
    real_data_path = curation._data_path
    monkeypatch.setattr(
        curation,
        "_data_path",
        lambda fn: str(dup_yaml) if fn == "pmid_overrides.yaml" else real_data_path(fn),
    )
    curation.load_pmid_overrides.cache_clear()
    try:
        with pytest.raises(ValueError, match="12345678"):
            curation.load_pmid_overrides()
    finally:
        curation.load_pmid_overrides.cache_clear()


def test_every_raw_sample_record_survives_loading():
    """The loader must not lose an ``ms_samples`` record to de-duplication."""
    assert sorted(_loaded_sample_keys()) == sorted(_raw_sample_keys())


@pytest.mark.parametrize(
    ("pmid", "study_label", "source_organism", "sample_labels"),
    [
        (
            33460454,
            "Gastaldello 2021",
            "Sarcophilus harrisii",
            {
                "devil fibroblasts (healthy host)",
                "DFT1 cell line 4906 + IFN-gamma",
                "DFT2 cell line Red Velvet",
            },
        ),
        (
            28188227,
            "Barnea 2017",
            "Rattus norvegicus",
            {
                "HLA-B27 transgenic rat spleen (WT)",
                "HLA-B27 transgenic rat spleen (ERAP1 KO)",
            },
        ),
    ],
)
def test_consolidated_entries_keep_samples_and_provenance(
    pmid, study_label, source_organism, sample_labels
):
    """Consolidation must preserve BOTH halves of each former duplicate.

    The study block carried the samples; the #307 block carried the
    source-organism curation the scanner reads. Losing either one is a
    regression, and the study labels on the #307 blocks named the wrong
    first author (verified against PubMed).
    """
    entry = load_pmid_overrides()[pmid]
    assert entry["study_label"].startswith(study_label)
    assert {s["sample_label"] for s in entry["ms_samples"]} == sample_labels
    assert pmid_source_organism(pmid) == (source_organism, source_organism)


# ── #437: explicitly unprofiled records ─────────────────────────────────────


def test_ms_samples_table_exports_every_loaded_record():
    """No curated sample may vanish between the loader and the export."""
    exported = {
        (int(row.pmid), str(row.sample_label)) for row in generate_ms_samples_table().itertuples()
    }
    assert set(_loaded_sample_keys()) - exported == set()


def test_unprofiled_records_are_exported_as_not_profiled():
    """``profiled: false`` is curated information, not an absence of it.

    The exporter's own docstring says ``profiled`` distinguishes
    "curated but not profiled" from "uncurated", yet its unconditional
    ``continue`` on ``n_samples == 0`` deleted exactly the rows that
    distinction exists for.
    """
    samples = generate_ms_samples_table()
    unprofiled = samples[samples["profiled"] == "false"]
    assert {
        (int(row.pmid), str(row.sample_label)) for row in unprofiled.itertuples()
    } == UNPROFILED_SAMPLES
    assert (unprofiled["n_samples"] == 0).all()


def test_observation_eligible_samples_drops_unprofiled_arms():
    """The join must not see an arm that was never run on the instrument.

    This is not hypothetical bookkeeping: attribution path 3c matches
    ``attributed_sample_label`` against ``sample_label`` directly, with no
    allele involved, and overrides every heuristic above it.  A curated
    per-row label colliding with an unprofiled arm would attribute real
    peptides to a sample the paper says was never profiled.
    """
    samples = pd.DataFrame(
        {
            "pmid": [1, 1, 2],
            "sample_label": ["profiled", "not profiled", "uncurated"],
            "profiled": ["true", "false", ""],
        }
    )
    eligible = _observation_eligible_samples(samples)
    assert list(eligible["sample_label"]) == ["profiled", "uncurated"]


def test_unprofiled_records_are_excluded_from_observation_attribution(full_observations_df):
    """Restoring the metadata must not manufacture a peptide observation.

    These arms have no MS data by construction, so no evidence row may be
    attributed to one.  None of the four is reachable through today's
    allele-based paths, so this is the corpus-level regression guard that
    stays true as their curation gains an ``mhc`` or a matching
    ``attributed_sample_label``.
    """
    attributed = {
        (int(pmid), str(label))
        # No ``strict=True``: this package supports Python 3.9.  This test is
        # integration-marked, so CI's 3.9 job skips it and never caught the
        # same mistake shipped in 1.58.6.
        for pmid, label in zip(full_observations_df["pmid"], full_observations_df["sample_label"])
        if str(label)
    }
    assert attributed & UNPROFILED_SAMPLES == set()
