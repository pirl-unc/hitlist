"""Tests for hitlist.mappings — peptide_mappings.parquet contract.

Issue #141 added ``transcript_id`` and ``is_canonical_transcript`` as
first-class columns and exposed matching filters on
``load_peptide_mappings``.  These tests pin down the schema (uniform
across FASTA + Ensembl backends) and the filter pushdown.
"""

from __future__ import annotations

import pandas as pd
import pytest

from hitlist.mappings import (
    _MAPPING_COLUMNS,
    MappingResult,
    MappingTask,
    _build_species_index,
    _build_workers,
    _flanking_rows_to_mapping_rows,
    _mapping_artifact_contract,
    _obs_fingerprint,
    _per_canonical_mapping_worker,
    _prefetch_proteomes_for_workers,
    _prefetch_worker,
    _proteome_group_key,
    _supervise_prefetch_tasks,
    load_peptide_mappings,
    mappings_meta_path,
)
from hitlist.mappings import _cache_is_valid as _mapping_cache_is_valid


def test_mapping_columns_contract_includes_transcript_fields():
    """The canonical mapping schema MUST include the new transcript fields."""
    cols = set(_MAPPING_COLUMNS)
    assert "transcript_id" in cols
    assert "is_canonical_transcript" in cols
    # Existing fields stay.
    for legacy in ("peptide", "protein_id", "gene_name", "gene_id", "position"):
        assert legacy in cols


def test_flanking_rows_to_mapping_rows_carries_transcript_columns():
    """When a transcript-aware ProteomeIndex feeds map_peptides → output,
    ``_flanking_rows_to_mapping_rows`` must propagate transcript_id and
    is_canonical_transcript onto the long-form mapping frame.
    """
    flanking = pd.DataFrame(
        {
            "peptide": ["ABCDEFGHI"],
            "protein_id": ["ENSP00000001"],
            "gene_name": ["TP53"],
            "gene_id": ["ENSG00000141510"],
            "transcript_id": ["ENST00000269305"],
            "is_canonical_transcript": [True],
            "position": [42],
            "n_flank": ["NNNNN"],
            "c_flank": ["CCCCC"],
        }
    )
    out = _flanking_rows_to_mapping_rows(
        flanking, proteome_label="Homo sapiens", proteome_source="ensembl"
    )
    assert list(out.columns) == list(_MAPPING_COLUMNS)
    row = out.iloc[0]
    assert row["transcript_id"] == "ENST00000269305"
    assert row["is_canonical_transcript"] is True or row["is_canonical_transcript"] == True  # noqa: E712


def test_flanking_rows_to_mapping_rows_legacy_input_safe_defaults():
    """Older fixtures / FASTA-only proteome indexes don't supply the new
    columns; the converter must fill safe defaults so the parquet schema
    stays uniform regardless of backend.
    """
    flanking = pd.DataFrame(
        {
            "peptide": ["ZZZZZ"],
            "protein_id": ["sp|P|A"],
            "gene_name": ["GENE"],
            "gene_id": [""],
            "position": [0],
            "n_flank": [""],
            "c_flank": [""],
        }
    )
    out = _flanking_rows_to_mapping_rows(flanking, proteome_label="custom", proteome_source="fasta")
    assert "transcript_id" in out.columns
    assert "is_canonical_transcript" in out.columns
    assert out.iloc[0]["transcript_id"] == ""
    assert bool(out.iloc[0]["is_canonical_transcript"]) is False


def test_flanking_rows_to_mapping_rows_empty_input_emits_full_schema():
    out = _flanking_rows_to_mapping_rows(pd.DataFrame(), proteome_label="x", proteome_source="x")
    assert list(out.columns) == list(_MAPPING_COLUMNS)
    assert len(out) == 0


def test_load_peptide_mappings_transcript_id_filter(tmp_path, monkeypatch):
    """``load_peptide_mappings(transcript_id=...)`` must push down the
    filter to pyarrow and return only the matching ENST rows.
    """
    rows = pd.DataFrame(
        {
            "peptide": ["AAA", "AAA", "BBB"],
            "protein_id": ["ENSP1", "ENSP2", "ENSP3"],
            "gene_name": ["TP53", "TP53", "MYC"],
            "gene_id": ["ENSG_TP53", "ENSG_TP53", "ENSG_MYC"],
            "transcript_id": ["ENST_T1", "ENST_T2", "ENST_T3"],
            "is_canonical_transcript": [True, False, True],
            "position": [1, 2, 3],
            "n_flank": ["", "", ""],
            "c_flank": ["", "", ""],
            "proteome": ["Homo sapiens"] * 3,
            "proteome_source": ["ensembl"] * 3,
        }
    )
    p = tmp_path / "peptide_mappings.parquet"
    rows.to_parquet(p, index=False)
    monkeypatch.setattr("hitlist.mappings.mappings_path", lambda: p)

    sub = load_peptide_mappings(transcript_id="ENST_T2")
    assert list(sub["transcript_id"]) == ["ENST_T2"]
    assert list(sub["protein_id"]) == ["ENSP2"]


def test_load_peptide_mappings_is_canonical_filter(tmp_path, monkeypatch):
    """``is_canonical_transcript=True`` returns only the canonical rows."""
    rows = pd.DataFrame(
        {
            "peptide": ["AAA", "AAA", "BBB"],
            "protein_id": ["ENSP1", "ENSP2", "ENSP3"],
            "gene_name": ["TP53", "TP53", "MYC"],
            "gene_id": ["ENSG_TP53", "ENSG_TP53", "ENSG_MYC"],
            "transcript_id": ["ENST_T1", "ENST_T2", "ENST_T3"],
            "is_canonical_transcript": [True, False, True],
            "position": [1, 2, 3],
            "n_flank": ["", "", ""],
            "c_flank": ["", "", ""],
            "proteome": ["Homo sapiens"] * 3,
            "proteome_source": ["ensembl"] * 3,
        }
    )
    p = tmp_path / "peptide_mappings.parquet"
    rows.to_parquet(p, index=False)
    monkeypatch.setattr("hitlist.mappings.mappings_path", lambda: p)

    canon = load_peptide_mappings(is_canonical_transcript=True)
    assert set(canon["transcript_id"]) == {"ENST_T1", "ENST_T3"}
    non_canon = load_peptide_mappings(is_canonical_transcript=False)
    assert list(non_canon["transcript_id"]) == ["ENST_T2"]


def test_load_peptide_mappings_missing_file_raises(tmp_path, monkeypatch):
    monkeypatch.setattr("hitlist.mappings.mappings_path", lambda: tmp_path / "nonexistent.parquet")
    with pytest.raises(FileNotFoundError, match="not built"):
        load_peptide_mappings(peptide="AAA")


def _seed_mapping_cache(tmp_path, monkeypatch, *, include_contract=True):
    import json

    from hitlist import downloads

    monkeypatch.setattr(downloads, "_override_data_dir", tmp_path)
    for name in ("observations.parquet", "binding.parquet", "peptide_mappings.parquet"):
        (tmp_path / name).write_bytes(name.encode())
    meta = {"observations": _obs_fingerprint()}
    if include_contract:
        meta["contract"] = _mapping_artifact_contract(
            release=112,
            use_uniprot=False,
            fetch_missing=True,
            flank=15,
        )
    mappings_meta_path().write_text(json.dumps(meta))
    return meta


def test_mapping_cache_requires_current_artifact_contract(tmp_path, monkeypatch):
    _seed_mapping_cache(tmp_path, monkeypatch)

    assert _mapping_cache_is_valid(
        release=112,
        use_uniprot=False,
        fetch_missing=True,
        flank=15,
    )


def test_mapping_cache_rejects_legacy_metadata(tmp_path, monkeypatch):
    _seed_mapping_cache(tmp_path, monkeypatch, include_contract=False)

    assert not _mapping_cache_is_valid(
        release=112,
        use_uniprot=False,
        fetch_missing=True,
        flank=15,
    )


@pytest.mark.parametrize(
    ("override", "value"),
    [
        ("release", 113),
        ("use_uniprot", True),
        ("fetch_missing", False),
        ("flank", 10),
    ],
)
def test_mapping_cache_rejects_parameter_changes(tmp_path, monkeypatch, override, value):
    _seed_mapping_cache(tmp_path, monkeypatch)
    params = {
        "release": 112,
        "use_uniprot": False,
        "fetch_missing": True,
        "flank": 15,
    }
    params[override] = value

    assert not _mapping_cache_is_valid(**params)


def test_mapping_cache_rejects_builder_version_change(tmp_path, monkeypatch):
    import json

    meta = _seed_mapping_cache(tmp_path, monkeypatch)
    meta["contract"]["artifact_version"] -= 1
    mappings_meta_path().write_text(json.dumps(meta))

    assert not _mapping_cache_is_valid(
        release=112,
        use_uniprot=False,
        fetch_missing=True,
        flank=15,
    )


def test_mapping_cache_rejects_schema_change(tmp_path, monkeypatch):
    import json

    meta = _seed_mapping_cache(tmp_path, monkeypatch)
    meta["contract"]["columns"] = meta["contract"]["columns"][:-1]
    mappings_meta_path().write_text(json.dumps(meta))

    assert not _mapping_cache_is_valid(
        release=112,
        use_uniprot=False,
        fetch_missing=True,
        flank=15,
    )


def test_mapping_cache_retries_incomplete_artifact(tmp_path, monkeypatch):
    import json

    meta = _seed_mapping_cache(tmp_path, monkeypatch)
    meta["unavailable_proteomes"] = ["Timed-out species"]
    mappings_meta_path().write_text(json.dumps(meta))

    assert not _mapping_cache_is_valid(
        release=112,
        use_uniprot=False,
        fetch_missing=True,
        flank=15,
    )


# ── #107 / v1.30.6: build-order clusters same-FASTA canonicals ─────────


def test_proteome_group_key_uniprot_same_proteome_id_clusters():
    """v1.30.6 / #107: two canonicals with the same UniProt proteome_id
    share a group key, so sorting by it lands them adjacently and the
    second one's ``from_fasta`` call hits the LRU cache. Strain-variant
    canonicals (multiple LCMV / SARS-CoV-2 / EBV strains) all share one
    underlying FASTA via ``proteome_id``."""
    e1 = {
        "kind": "uniprot",
        "proteome_id": "UP000111111",
        "canonical_species": "Strain A",
    }
    e2 = {
        "kind": "uniprot",
        "proteome_id": "UP000111111",
        "canonical_species": "Strain Z",
    }
    e3 = {
        "kind": "uniprot",
        "proteome_id": "UP000999999",
        "canonical_species": "Other species",
    }
    assert _proteome_group_key(e1) == _proteome_group_key(e2)
    assert _proteome_group_key(e1) != _proteome_group_key(e3)


def test_proteome_group_key_orders_ensembl_before_uniprot_before_other():
    """Ensembl bucket sorts before uniprot, uniprot before unrecognised,
    so the build log clusters the human pass first, then FASTA-backed
    species, then any tail of unmatchable canonicals."""
    e_ensembl = {"kind": "ensembl", "species": "human"}
    e_uniprot = {"kind": "uniprot", "proteome_id": "UP000005640"}
    e_other = {"kind": "", "canonical_species": "Mystery sp."}
    assert _proteome_group_key(e_ensembl) < _proteome_group_key(e_uniprot)
    assert _proteome_group_key(e_uniprot) < _proteome_group_key(e_other)


def test_proteome_group_key_sorts_canonicals_by_fasta_adjacency():
    """End-to-end: a list of mixed canonicals sorted by
    ``(group_key, canonical)`` puts same-FASTA canonicals next to each
    other instead of scattering them by alphabetic name."""
    canonical_to_entry = {
        "Apple virus alpha": {
            "kind": "uniprot",
            "proteome_id": "UP000ZZZ001",
            "canonical_species": "Apple virus alpha",
        },
        # Same FASTA as "Apple virus alpha" — alphabetically far apart but
        # should cluster after sorting.
        "Zebra virus omega": {
            "kind": "uniprot",
            "proteome_id": "UP000ZZZ001",
            "canonical_species": "Zebra virus omega",
        },
        "Banana virus beta": {
            "kind": "uniprot",
            "proteome_id": "UP000ZZZ002",
            "canonical_species": "Banana virus beta",
        },
    }
    canonicals = list(canonical_to_entry.keys())
    ordered = sorted(
        canonicals,
        key=lambda c: (_proteome_group_key(canonical_to_entry[c]), c),
    )
    # The two ZZZ001 entries are now adjacent; the ZZZ002 is separate.
    apple_idx = ordered.index("Apple virus alpha")
    zebra_idx = ordered.index("Zebra virus omega")
    banana_idx = ordered.index("Banana virus beta")
    assert abs(apple_idx - zebra_idx) == 1, ordered
    # Banana (different proteome_id) is not sandwiched between them.
    assert not (apple_idx < banana_idx < zebra_idx), ordered


# ── Parallel mapping helpers (#249) ──────────────────────────────────────


def test_build_workers_default_is_capped_at_four(monkeypatch):
    """Default worker count is min(4, cpu_count // 2) — bounded so peak
    RSS stays under workers x largest-single-length-index."""
    monkeypatch.delenv("HITLIST_BUILD_WORKERS", raising=False)
    n = _build_workers()
    assert 1 <= n <= 4


def test_build_workers_env_override_is_respected(monkeypatch):
    monkeypatch.setenv("HITLIST_BUILD_WORKERS", "7")
    assert _build_workers() == 7


def test_build_workers_env_override_zero_falls_back_to_default(monkeypatch):
    """0 / negative are nonsense and silently fall back to the default."""
    monkeypatch.setenv("HITLIST_BUILD_WORKERS", "0")
    assert _build_workers() >= 1
    monkeypatch.setenv("HITLIST_BUILD_WORKERS", "-3")
    assert _build_workers() >= 1


def test_build_workers_env_override_garbage_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("HITLIST_BUILD_WORKERS", "all-of-them")
    n = _build_workers()
    assert 1 <= n <= 4


def test_build_workers_explicit_one_returns_one(monkeypatch):
    """HITLIST_BUILD_WORKERS=1 forces the sequential fallback."""
    monkeypatch.setenv("HITLIST_BUILD_WORKERS", "1")
    assert _build_workers() == 1


class _FakeFlanking:
    """Stand-in for ProteomeIndex with the methods the worker calls."""

    def __init__(self, label: str):
        self._label = label

    def map_peptides(self, peptides, flank, verbose):
        # One row per input peptide, all "matched" against this proteome.
        return pd.DataFrame(
            {
                "peptide": list(peptides),
                "protein_id": [f"{self._label}_PROT"] * len(peptides),
                "gene_name": [self._label] * len(peptides),
                "gene_id": [f"{self._label}_GENE"] * len(peptides),
                "transcript_id": [f"{self._label}_TX"] * len(peptides),
                "is_canonical_transcript": [True] * len(peptides),
                "position": list(range(len(peptides))),
                "n_flank": ["NNNNN"] * len(peptides),
                "c_flank": ["CCCCC"] * len(peptides),
            }
        )


def _mapping_task(
    canonical="X",
    peptides=("ABCDEFGHI",),
    *,
    entry=None,
    flank=15,
):
    return MappingTask(
        canonical=canonical,
        entry=entry or {"kind": "uniprot", "proteome_id": "UP_TEST"},
        peptides=tuple(peptides),
        seed_lengths=(7,),
        release=112,
        flank=flank,
    )


def test_per_canonical_worker_returns_expected_shape(monkeypatch):
    """The result object gives the orchestrator named aggregation fields."""
    monkeypatch.setattr(
        "hitlist.mappings._build_species_index",
        lambda *a, **kw: _FakeFlanking("Homo sapiens"),
    )
    peptides = ["ABCDEFGHI", "JKLMNOPQR", "ABCDEFGHIJ", "ZZZZZZZZZZZZ"]
    result = _per_canonical_mapping_worker(
        _mapping_task(
            canonical="Homo sapiens",
            peptides=peptides,
            entry={"kind": "ensembl", "species": "human"},
        )
    )
    assert result.canonical == "Homo sapiens"
    # One index, one pass, one frame -- regardless of how many lengths the
    # peptides span (#398).
    assert result.mapping_frame is not None
    # Including the 12-mer, which one seed index serves like any other length.
    assert result.n_matched_peptides == 4
    assert result.n_input_peptides == 4
    assert result.proteome_available is True


def test_per_canonical_worker_builds_one_index_for_all_lengths(monkeypatch):
    """#398: one seed index serves every peptide length.

    This used to rebuild the index once per length in `lengths_in_query`,
    so a canonical spanning 8/9/10/11 paid four full builds to answer what
    one seed index answers in a single pass.
    """
    builds = {"n": 0}

    def counting_build(*a, **kw):
        builds["n"] += 1
        return _FakeFlanking("X")

    monkeypatch.setattr("hitlist.mappings._build_species_index", counting_build)
    _per_canonical_mapping_worker(
        _mapping_task(peptides=["ABCDEFGH", "ABCDEFGHI", "ABCDEFGHIJ", "A" * 20])
    )
    assert builds["n"] == 1


def test_per_canonical_worker_uses_resolved_entry_and_is_cache_only(monkeypatch):
    captured = {}

    def capture_build(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return _FakeFlanking("offline")

    monkeypatch.setattr("hitlist.mappings._build_species_index", capture_build)
    entry = {"kind": "uniprot", "proteome_id": "UP_OFFLINE"}
    task = _mapping_task(entry=entry)

    result = _per_canonical_mapping_worker(task)

    assert result.n_matched_peptides == 1
    assert captured["kwargs"]["entry"] is entry
    assert captured["kwargs"]["fetch_missing"] is False
    assert captured["kwargs"]["lengths"] == (7,)


def test_build_species_index_offline_never_downloads(monkeypatch):
    from hitlist import downloads

    captured = {}

    def cache_only_fetch(upid, **kwargs):
        captured["upid"] = upid
        captured["kwargs"] = kwargs
        return None

    monkeypatch.setattr(downloads, "fetch_proteome_by_upid", cache_only_fetch)

    result = _build_species_index(
        "Offline species",
        release=112,
        use_uniprot=True,
        verbose=False,
        entry={"kind": "uniprot", "proteome_id": "UP_OFFLINE"},
        fetch_missing=False,
    )

    assert result is None
    assert captured == {
        "upid": "UP_OFFLINE",
        "kwargs": {
            "label": "Offline species",
            "verbose": False,
            "fetch_missing": False,
        },
    }


def test_build_species_index_tolerates_corrupt_cached_fasta(tmp_path, monkeypatch):
    from hitlist import downloads
    from hitlist.proteome import ProteomeIndex

    fasta = tmp_path / "broken.fasta"
    fasta.write_text("not a FASTA")
    monkeypatch.setattr(downloads, "fetch_proteome_by_upid", lambda *_a, **_kw: fasta)

    def fail_to_index(*_args, **_kwargs):
        raise ValueError("corrupt FASTA")

    monkeypatch.setattr(ProteomeIndex, "from_fasta", fail_to_index)

    result = _build_species_index(
        "Broken cached species",
        release=112,
        use_uniprot=False,
        verbose=False,
        entry={"kind": "uniprot", "proteome_id": "UP_BROKEN"},
        fetch_missing=False,
    )

    assert result is None


def test_per_canonical_worker_survives_an_unbuildable_index(monkeypatch):
    """When _build_species_index returns None the worker degrades to a
    no-op rather than raising -- a missing FASTA or GTF must not take down
    the whole mapping pass."""
    monkeypatch.setattr("hitlist.mappings._build_species_index", lambda *a, **kw: None)

    result = _per_canonical_mapping_worker(_mapping_task(peptides=["ABCDEFGHI", "ABCDEFGHIJ"]))
    assert result.canonical == "X"
    assert result.mapping_frame is None
    assert result.n_matched_peptides == 0
    # n_total still reports what was asked for, so the stats line is honest
    # about coverage rather than silently shrinking the denominator.
    assert result.n_input_peptides == 2
    assert result.proteome_available is False


def test_per_canonical_worker_args_are_picklable():
    """ProcessPoolExecutor.map dispatches via pickle — args MUST round-trip."""
    import pickle

    task = _mapping_task(
        canonical="Homo sapiens",
        peptides=["ABCDEFGHI", "JKLMNOPQR", "ABCDEFGHIJ"],
        entry={"kind": "ensembl", "species": "human"},
    )
    assert pickle.loads(pickle.dumps(task)) == task


def test_per_canonical_worker_return_value_is_picklable(monkeypatch):
    """And so must the return value — list[DataFrame] is picklable."""
    import pickle

    monkeypatch.setattr(
        "hitlist.mappings._build_species_index",
        lambda *a, **kw: _FakeFlanking("Z"),
    )
    result = _per_canonical_mapping_worker(_mapping_task(canonical="Z", flank=10))
    round_tripped = pickle.loads(pickle.dumps(result))
    assert round_tripped.canonical == "Z"
    assert round_tripped.mapping_frame is not None
    assert round_tripped.n_matched_peptides == 1
    assert round_tripped.n_input_peptides == 1
    assert round_tripped.proteome_available is True


# ── _prefetch_proteomes_for_workers (#249) ──────────────────────────────


def _scripted_prefetch_worker(label, _entry, _cache_dir, _release):
    """Spawn-safe test worker: one named failure, all other tasks succeed."""
    if label == "BadSpecies":
        return label, False, "RuntimeError: simulated failure"
    return label, True, ""


def _blocking_prefetch_worker(label, _entry, _cache_dir, _release):
    """Spawn-safe test worker that never answers its first request."""
    import time

    time.sleep(60)
    return label, True, ""


def test_prefetch_plans_unique_canonicals_and_groups_ensembl(monkeypatch):
    captured = {}

    def capture(tasks, **kwargs):
        captured["tasks"] = tasks
        captured["kwargs"] = kwargs
        return set()

    monkeypatch.setattr("hitlist.mappings._supervise_prefetch_tasks", capture)
    pairs = [
        ("Virus", {"kind": "uniprot", "proteome_id": "UP1"}),
        ("Virus", {"kind": "uniprot", "proteome_id": "UP1"}),
        ("Virus alias", {"kind": "uniprot", "proteome_id": "UP1"}),
        ("Human alias A", {"kind": "ensembl", "species": "human"}),
        ("Human alias B", {"kind": "ensembl", "species": "human"}),
    ]

    unavailable = _prefetch_proteomes_for_workers(pairs, release=112, verbose=False)

    assert unavailable == set()
    assert captured["tasks"] == [
        (
            "Virus",
            ("Virus", "Virus alias"),
            {"kind": "uniprot", "proteome_id": "UP1"},
        ),
        (
            "Ensembl human r112",
            ("Human alias A", "Human alias B"),
            {"kind": "ensembl", "species": "human"},
        ),
    ]
    assert captured["kwargs"]["release"] == 112


def test_prefetch_worker_fetches_explicit_upid(tmp_path, monkeypatch):
    fetched = []

    fasta = tmp_path / "rare.fasta"
    fasta.write_text(">p\nPEPTIDE\n")

    def fake_fetch(upid, **kwargs):
        fetched.append((upid, kwargs))
        return fasta

    monkeypatch.setattr("hitlist.downloads.fetch_proteome_by_upid", fake_fetch)
    result = _prefetch_worker(
        "Rare species",
        {"kind": "uniprot", "proteome_id": "UP123"},
        str(tmp_path),
        release=112,
    )

    assert fetched == [
        (
            "UP123",
            {
                "label": "Rare species",
                "verbose": False,
                "fetch_missing": True,
            },
        )
    ]
    assert result == ("Rare species", True, "")


@pytest.mark.parametrize(("cached", "succeeded"), [(True, True), (False, False)])
def test_prefetch_worker_distinguishes_negative_resolution_from_transport_failure(
    tmp_path, monkeypatch, cached, succeeded
):
    from hitlist import downloads

    monkeypatch.setattr(downloads, "lookup_proteome", lambda *_a, **_kw: None)
    monkeypatch.setattr(
        downloads,
        "_uniprot_cache",
        lambda: {"Mystery organism": {"not_found": True}} if cached else {},
    )

    result = _prefetch_worker(
        "Resolve mystery",
        {"kind": "resolve", "organism": "Mystery organism"},
        str(tmp_path),
        release=112,
    )

    assert result[0] == "Resolve mystery"
    assert result[1] is succeeded
    if not succeeded:
        assert "transiently" in result[2]


def test_prefetch_supervisor_continues_after_failure(capsys):
    tasks = [
        ("BadSpecies", ("BadSpecies",), {"kind": "uniprot", "proteome_id": "UP_BAD"}),
        (
            "GoodSpecies",
            ("GoodSpecies",),
            {"kind": "uniprot", "proteome_id": "UP_GOOD"},
        ),
    ]

    unavailable = _supervise_prefetch_tasks(
        tasks,
        release=112,
        verbose=True,
        deadline_seconds=5,
        worker_target=_scripted_prefetch_worker,
    )

    out = capsys.readouterr().out
    assert unavailable == {"BadSpecies"}
    assert "[1/2] BadSpecies" in out
    assert "[2/2] GoodSpecies" in out
    assert "simulated failure" in out


def test_prefetch_supervisor_terminates_blocked_inflight_call(capsys):
    import time

    tasks = [
        ("BlockedSpecies", ("BlockedSpecies",), {"kind": "uniprot", "proteome_id": "UP1"}),
        ("NeverStarted", ("NeverStarted",), {"kind": "uniprot", "proteome_id": "UP2"}),
    ]
    started = time.monotonic()

    unavailable = _supervise_prefetch_tasks(
        tasks,
        release=112,
        verbose=True,
        deadline_seconds=0.2,
        worker_target=_blocking_prefetch_worker,
    )

    elapsed = time.monotonic() - started
    out = capsys.readouterr().out
    assert elapsed < 5
    assert unavailable == {"BlockedSpecies", "NeverStarted"}
    assert "BlockedSpecies" in out
    assert "timed out" in out
    assert "NeverStarted" not in out


@pytest.mark.parametrize("deadline_seconds", [0.0, -1.0, float("nan"), float("inf")])
def test_prefetch_supervisor_fails_safe_for_invalid_or_exhausted_deadline(deadline_seconds, capsys):
    tasks = [("NeverStarted", ("NeverStarted",), {"kind": "uniprot", "proteome_id": "UP1"})]

    unavailable = _supervise_prefetch_tasks(
        tasks,
        release=112,
        verbose=False,
        deadline_seconds=deadline_seconds,
        worker_target=_blocking_prefetch_worker,
    )

    assert unavailable == {"NeverStarted"}
    assert "invalid or already exhausted" in capsys.readouterr().out


def test_prefetch_handles_empty_input():
    assert _prefetch_proteomes_for_workers([], release=112, verbose=False) == set()


# ── End-to-end orchestrator dispatch (#249) ────────────────────────────


def _e2e_worker_for_pool_test(task):
    """Module-level (picklable) double of _per_canonical_mapping_worker for
    end-to-end ProcessPoolExecutor dispatch tests.  Returns the same shape
    so the orchestrator's aggregation can be exercised without needing a
    real proteome.  Defined at module scope so spawn-based pools can pickle.
    """
    df = pd.DataFrame(
        {
            "peptide": ["FAKEPEP"],
            "protein_id": [f"{task.canonical}_PROT"],
            "gene_name": [task.canonical],
            "gene_id": [f"{task.canonical}_GENE"],
            "transcript_id": [""],
            "is_canonical_transcript": [False],
            "position": [0],
            "n_flank": [""],
            "c_flank": [""],
            "proteome": [task.canonical],
            "proteome_source": ["species"],
        }
    )
    return MappingResult(
        canonical=task.canonical,
        mapping_frame=df,
        n_matched_peptides=len(task.peptides),
        n_input_peptides=len(task.peptides),
    )


def test_pool_map_dispatch_preserves_order_and_aggregates_results():
    """Verify ProcessPoolExecutor.map round-trips our worker contract end-to-end:
    pickle args/results, preserve task order, return all canonicals."""
    from concurrent.futures import ProcessPoolExecutor

    tasks = [_mapping_task(canonical=f"species_{i}", peptides=[f"PEP{i:05d}"]) for i in range(6)]
    with ProcessPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(_e2e_worker_for_pool_test, tasks, chunksize=2))
    # Order is preserved by pool.map (matches submission order).
    assert [result.canonical for result in results] == [task.canonical for task in tasks]
    # Each task produced exactly one DataFrame.
    assert all(result.mapping_frame is not None for result in results)
    # n_matched == n_total == 1 in all cases.
    assert all(
        result.n_matched_peptides == 1 and result.n_input_peptides == 1 for result in results
    )
