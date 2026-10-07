"""Search-space and observation controls for detectability candidates (#361)."""

import pytest

from hitlist.proteome import digest, digest_occurrences


@pytest.mark.parametrize(
    "enzyme,residue,cleaves_before_p",
    [
        ("Trypsin", "K", False),
        ("Trypsin", "R", False),
        ("Trypsin/P", "K", True),
        ("Trypsin/P", "R", True),
        ("LysC", "K", False),
        ("LysC/P", "K", True),
        ("Chymotrypsin", "F", True),
        ("Chymotrypsin", "M", False),
        ("Chymotrypsin+", "M", True),
        ("Chymotrypsin+", "L", True),
        ("GluC", "E", True),
        ("GluC", "D", False),
        ("GluC;D.P", "D", True),
    ],
)
def test_search_proline_pairs_match_cox_enzyme_table(enzyme, residue, cleaves_before_p):
    # Cox Lab's 2015 specificity-pair table, independently inspected for #654.
    sequence = f"AA{residue}PAA"
    expected = {f"AA{residue}", "PAA"} if cleaves_before_p else {sequence}
    assert digest(sequence, enzyme=enzyme, min_len=1, max_missed=0) == expected


def test_occurrences_preserve_repeats_coordinates_and_missed_cleavages():
    rows = list(digest_occurrences("AAKAAKTAIL", min_len=3, max_len=6, max_missed=1))
    assert [(r.peptide, r.start_position, r.end_position, r.n_missed_cleavages) for r in rows] == [
        ("AAK", 1, 3, 0),
        ("AAKAAK", 1, 6, 1),
        ("AAK", 4, 6, 0),
        ("TAIL", 7, 10, 0),
    ]
    assert digest("AAKAAKTAIL", min_len=3, max_len=6, max_missed=1) == {"AAK", "AAKAAK", "TAIL"}


def test_contradictory_legacy_search_name_requires_an_explicit_choice():
    with pytest.raises(ValueError, match="contradictory"):
        digest("AAKPAAK", enzyme="Trypsin/P (cleaves K/R except before P)")


@pytest.mark.parametrize(
    "options",
    [
        {"min_len": 0},
        {"max_len": -1},
        {"max_missed": -1},
        {"max_missed": True},
        {"min_len": 8, "max_len": 7},
    ],
)
def test_invalid_digest_limits_are_not_silent_empty_results(options):
    with pytest.raises(ValueError):
        list(digest_occurrences("AAKPAAR", **options))


@pytest.fixture
def search_inputs(tmp_path):
    import pandas as pd

    from hitlist.detectability import DetectabilitySearchSpace

    fasta = tmp_path / "searched.fasta"
    fasta.write_text(">sp|P1|one GN=ONE\nAAKAAKTAIL\n>sp|P2|two GN=TWO\nAAKCCK\n>P3\nDDK\n")
    contract = DetectabilitySearchSpace.from_fasta(
        fasta,
        enzyme="Trypsin/P",
        max_missed_cleavages=2,
        min_peptide_length=3,
        max_peptide_length=30,
        max_peptide_mass_da=None,
        fixed_residue_modifications={},
        provenance="test search settings",
    )
    scope = {
        "source": "test",
        "cell_line_name": "HeLa",
        "digestion_enzyme": "Trypsin/P",
        "n_fractions_in_run": 46,
        "enrichment": "none",
        "fractionation_ph": 10.0,
        "protocol_id": "test/46",
        "search_enzyme": "Trypsin/P",
        "search_space_id": contract.identifier,
        "instrument": "instrument",
        "fragmentation": "HCD",
        "acquisition_mode": "DDA",
        "labeling": "label-free",
        "search_engine": "engine",
        "detection_basis": "direct MS/MS",
        "protein_observation_basis": "same-scope peptide support",
        "n_replicates_possible": 2,
    }
    peptides = pd.DataFrame(
        [
            {**scope, "peptide": "AAK", "uniprot_acc": "P1", "n_replicates_detected": 1},
            {**scope, "peptide": "CCK", "uniprot_acc": "P2", "n_replicates_detected": 2},
        ]
    )
    proteins = pd.DataFrame(
        [
            {
                **scope,
                "uniprot_acc": "P1",
                "gene_symbol": "ONE",
                "n_peptides": 1,
                "abundance_percentile": 0.5,
            },
            {
                **scope,
                "uniprot_acc": "P2",
                "gene_symbol": "TWO",
                "n_peptides": 1,
                "abundance_percentile": 1.0,
            },
        ]
    )
    return {
        "search_fasta": fasta,
        "search_space": contract,
        "peptide_observations": peptides,
        "protein_observations": proteins,
        "length": (3, 6),
        "max_missed": 0,
        "flank": 2,
    }


def test_dataset_preserves_shared_and_repeated_occurrences(search_inputs):
    from pandas.testing import assert_frame_equal

    from hitlist.detectability import build_detectability_training_set

    before = search_inputs["peptide_observations"].copy(deep=True)
    result = build_detectability_training_set(**search_inputs)
    assert list(zip(result.peptide, result.uniprot_acc, result.start_position)) == [
        ("AAK", "P1", 1),
        ("AAK", "P1", 4),
        ("TAIL", "P1", 7),
        ("AAK", "P2", 1),
        ("CCK", "P2", 4),
    ]
    assert result.observed.tolist() == [True, True, False, True, True]
    assert result.n_replicates_detected.tolist() == [1, 1, 0, 1, 2]
    assert result.first_seen_at_n_fractions.isna().all()
    assert result.loc[1, "n_flank"] == "AK"
    assert result.loc[1, "c_flank"] == "TA"
    assert result.protein_observed.all()
    assert_frame_equal(before, search_inputs["peptide_observations"])
    assert (
        result.attrs["detectability"]["search_reference"]["sha256"]
        == search_inputs["search_space"].fasta_sha256
    )


def test_optional_unobserved_parents_are_explicit(search_inputs):
    from hitlist.detectability import build_detectability_training_set

    result = build_detectability_training_set(**search_inputs, require_protein_observed=False)
    row = result[result.uniprot_acc.eq("P3")].iloc[0]
    assert row.peptide == "DDK" and not row.observed and not row.protein_observed
    assert __import__("pandas").isna(row.protein_abundance_percentile)


def test_stream_and_atomic_export_equivalence(search_inputs, tmp_path):
    import json

    import pandas as pd
    from pandas.testing import assert_frame_equal

    from hitlist.detectability import (
        build_detectability_training_set,
        export_detectability_training_set,
        iter_detectability_training_set,
    )
    from hitlist.provenance import file_digest

    expected = build_detectability_training_set(**search_inputs)
    batches = list(iter_detectability_training_set(**search_inputs, batch_size=2))
    assert [len(b) for b in batches] == [2, 2, 1]
    assert_frame_equal(expected, pd.concat(batches, ignore_index=True))
    out = export_detectability_training_set(tmp_path / "bundle", **search_inputs, batch_size=2)
    actual = pd.read_parquet(out / "candidates.parquet")
    assert_frame_equal(expected, actual, check_dtype=False)
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["n_candidates"] == 5 and manifest["n_observed_candidates"] == 4
    assert manifest["artifact"]["sha256"] == file_digest(out / "candidates.parquet")["sha256"]
    with pytest.raises(FileExistsError):
        export_detectability_training_set(out, **search_inputs)


@pytest.mark.parametrize("limits", [{"max_candidates": 2}, {"max_output_bytes": 100}])
def test_failed_export_never_publishes_partial_dataset(search_inputs, tmp_path, limits):
    from hitlist.detectability import export_detectability_training_set

    with pytest.raises((ValueError, OSError), match=r"limit|exceeds"):
        export_detectability_training_set(
            tmp_path / "bundle", **search_inputs, batch_size=1, **limits
        )
    assert not (tmp_path / "bundle").exists()
    assert not list(tmp_path.glob(".detectability-*"))


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("search_space", None, "verified search_space"),
        ("max_missed", 3, "missed-cleavage"),
        ("length", (2, 6), "length window"),
        ("length", (3, 31), "length window"),
        ("max_reference_bytes", 1, "max_reference_bytes"),
        ("max_observation_rows", 1, "max_observation_rows"),
        ("batch_size", 0, "batch_size"),
        ("require_protein_observed", "yes", "boolean"),
    ],
)
def test_invalid_search_requests_fail(search_inputs, field, value, error):
    from hitlist.detectability import build_detectability_training_set

    search_inputs[field] = value
    with pytest.raises(ValueError, match=error):
        build_detectability_training_set(**search_inputs)


@pytest.mark.parametrize(
    "table,column,value,error",
    [
        ("peptide_observations", "search_space_id", "wrong", "search_space_id"),
        ("protein_observations", "search_space_id", "wrong", "search_space_id"),
        ("peptide_observations", "n_replicates_detected", 1.5, "integers"),
        ("peptide_observations", "n_replicates_detected", 3, "possible replicates"),
        ("protein_observations", "n_peptides", 0, "positive n_peptides"),
        ("protein_observations", "abundance_percentile", 2, "0-1"),
        ("peptide_observations", "peptide", "AAAK", "does not occur"),
        ("peptide_observations", "uniprot_acc", "missing", "missing"),
    ],
)
def test_inconsistent_evidence_fails_before_labels(search_inputs, table, column, value, error):
    from hitlist.detectability import iter_detectability_training_set

    search_inputs[table][column] = search_inputs[table][column].astype(object)
    search_inputs[table].loc[0, column] = value
    with pytest.raises(ValueError, match=error):
        next(iter_detectability_training_set(**search_inputs, batch_size=1))


def test_reference_identity_is_verified(search_inputs):
    from hitlist.detectability import build_detectability_training_set

    search_inputs["search_fasta"].write_text(">different\nAAK\n")
    with pytest.raises(ValueError, match="SHA256"):
        build_detectability_training_set(**search_inputs)


def test_empty_search_window_has_stable_schema(search_inputs):
    from hitlist.detectability import build_detectability_training_set

    search_inputs["length"] = (20, 30)
    result = build_detectability_training_set(**search_inputs)
    assert result.empty and "observed" in result and "peptide" in result
    assert str(result.n_replicates_detected.dtype) == "Int64"


def test_mass_filter_enforces_fixed_modifications(search_inputs):
    from dataclasses import replace

    from hitlist.detectability import build_detectability_training_set

    contract = replace(
        search_inputs["search_space"],
        max_peptide_mass_da=350.0,
        fixed_residue_modifications={"C": 57.021464},
    )
    search_inputs["search_space"] = contract
    for key in ("peptide_observations", "protein_observations"):
        search_inputs[key]["search_space_id"] = contract.identifier
    result = build_detectability_training_set(**search_inputs)
    assert set(result.peptide) == {"AAK"}


def _add_comparison(search_inputs):
    import json

    import pandas as pd

    selected = search_inputs["peptide_observations"].copy()
    selected["comparison_group"] = "verified"
    selected["comparison_provenance"] = "documented controlled fractionation experiment"
    selected["comparison_controls"] = json.dumps(
        {
            "lc_gradient_minutes": 30,
            "peptide_load_ug": 1,
            "sample_preparation": "identical",
            "acquisition_method": "method hash",
        }
    )
    selected["experiment_ids"] = json.dumps(["test/46/a", "test/46/b"])
    earlier = selected.iloc[[0]].copy()
    earlier["n_fractions_in_run"] = 14
    earlier["protocol_id"] = "test/14"
    earlier["experiment_ids"] = json.dumps(["test/14/a", "test/14/b"])
    search_inputs["peptide_observations"] = pd.concat([selected, earlier], ignore_index=True)


def test_depth_is_only_derived_for_valid_comparison(search_inputs):
    from hitlist.detectability import build_detectability_training_set

    _add_comparison(search_inputs)
    result = build_detectability_training_set(**search_inputs)
    assert set(result.loc[result.peptide.eq("AAK"), "first_seen_at_n_fractions"]) == {14}
    assert result.loc[result.peptide.eq("CCK"), "first_seen_at_n_fractions"].tolist() == [46]
    assert result.loc[result.peptide.eq("TAIL"), "first_seen_at_n_fractions"].isna().all()


@pytest.mark.parametrize(
    "column,value,error",
    [
        ("instrument", "different", "instrument"),
        ("search_space_id", "different", "search_space_id"),
        ("comparison_controls", "{}", "comparison_controls"),
        ("experiment_ids", '["test/46/a", "test/14/b"]', "reuses"),
        ("n_replicates_possible", 3, "n_replicates_possible"),
    ],
)
def test_uncomparable_or_reused_protocols_rejected(search_inputs, column, value, error):
    from hitlist.detectability import build_detectability_training_set

    _add_comparison(search_inputs)
    search_inputs["peptide_observations"].loc[2, column] = value
    with pytest.raises(ValueError, match=error):
        build_detectability_training_set(**search_inputs)


def test_unknown_replicate_count_remains_nullable(search_inputs):
    import pandas as pd

    from hitlist.detectability import build_detectability_training_set

    search_inputs["peptide_observations"]["n_replicates_detected"] = pd.NA
    result = build_detectability_training_set(**search_inputs)
    assert result.loc[result.observed, "n_replicates_detected"].isna().all()
    assert set(result.loc[result.observed, "replicate_count_status"]) == {"unresolved_aggregate"}
    assert set(result.loc[~result.observed, "n_replicates_detected"]) == {0}


def test_packaged_adapter_preserves_pool_and_source_provenance(tmp_path, monkeypatch):
    import pandas as pd

    from hitlist import bulk_proteomics
    from hitlist.detectability import DetectabilitySearchSpace, _packaged_observations
    from hitlist.provenance import file_digest

    common = {
        "cell_line": "HeLa",
        "digestion_enzyme": "Trypsin/P (cleaves K/R except before P)",
        "n_fractions_in_run": 46,
        "enrichment": "none",
        "fractionation_ph": 10.0,
        "source": "Bekker-Jensen_2017",
        "reference": "PMID:28591648",
        "uniprot_acc": "P1",
    }
    peptide = pd.DataFrame(
        [
            {**common, "peptide": "AAAAAAK", "n_replicates_detected": 2},
            {**common, "peptide": "AAAAAAR", "n_replicates_detected": 1, "fractionation_ph": 8.0},
        ]
    )
    protein = pd.DataFrame([{**common, "n_peptides": 1, "abundance_percentile": 1.0}])
    paths = {}
    for suffix, table in [("peptides", peptide), ("protein_abundance", protein)]:
        name = f"bekker_jensen_2017_{suffix}.csv.gz"
        paths[name] = tmp_path / name
        table.to_csv(paths[name], index=False)
    monkeypatch.setattr(bulk_proteomics, "_bulk_data_path", lambda name: paths[name])
    contract = DetectabilitySearchSpace(
        fasta_sha256="0" * 64,
        enzyme="Trypsin/P",
        max_missed_cleavages=3,
        min_peptide_length=7,
        max_peptide_length=30,
        max_peptide_mass_da=4600,
        fixed_residue_modifications={"C": 57.021464},
        provenance="synthetic contract for adapter test",
    )
    peptides, proteins, provenance = _packaged_observations(
        "HeLa", "Trypsin/P", 46, "none", 10.0, contract, 100
    )
    assert peptides.peptide.tolist() == ["AAAAAAK"]
    assert peptides.n_replicates_detected.isna().all()
    assert peptides.reported_n_replicates_detected.tolist() == [2]
    assert peptides.n_replicates_possible.tolist() == [3]
    assert peptides.reference.tolist() == ["PMID:28601559"]
    assert peptides.reported_reference.tolist() == ["PMID:28591648"]
    assert set(provenance["scope"]["experiment_ids"]) == {
        "HeLa-46fracs-IT-E1",
        "HeLa-46fracs-IT-E2",
        "Tryp-46fracs",
    }
    assert (
        provenance["peptides"]["sha256"]
        == file_digest(paths["bekker_jensen_2017_peptides.csv.gz"])["sha256"]
    )
    assert proteins.search_space_id.tolist() == [contract.identifier]
    with pytest.raises(ValueError, match="max_observation_rows"):
        _packaged_observations("HeLa", "Trypsin/P", 46, "none", 10.0, contract, 0)


def test_legacy_bulk_reference_correction_retains_original():
    import pandas as pd

    from hitlist.bulk_proteomics import _correct_bj_reference

    before = pd.DataFrame(
        {
            "source": ["Bekker-Jensen_2017", "CCLE_Nusinow_2020"],
            "reference": ["PMID:28591648", "PMID:31978347"],
            "pmid": [28591648, 31978347],
        }
    )
    before["reference"] = before.reference.astype("category")
    result = _correct_bj_reference(before)
    assert result.pmid.tolist() == [28601559, 31978347]
    assert result.reference.tolist() == ["PMID:28601559", "PMID:31978347"]
    assert result.reported_reference.tolist() == ["PMID:28591648", "PMID:31978347"]
    assert before.pmid.tolist() == [28591648, 31978347]
    assert (
        _correct_bj_reference(result).reported_reference.tolist()
        == result.reported_reference.tolist()
    )


def test_duplicate_or_missing_reference_parents_fail_before_first_batch(search_inputs):
    from dataclasses import asdict

    from hitlist.detectability import DetectabilitySearchSpace, iter_detectability_training_set

    path = search_inputs["search_fasta"]
    path.write_text(path.read_text() + ">P1\nAAK\n")
    settings = asdict(search_inputs["search_space"])
    settings.pop("fasta_sha256")
    contract = DetectabilitySearchSpace.from_fasta(path, **settings)
    search_inputs["search_space"] = contract
    for key in ("peptide_observations", "protein_observations"):
        search_inputs[key]["search_space_id"] = contract.identifier
    with pytest.raises(ValueError, match="Duplicate protein"):
        next(iter_detectability_training_set(**search_inputs, batch_size=1))


def test_partial_comparison_metadata_is_not_silently_accepted(search_inputs):
    import pandas as pd

    from hitlist.detectability import build_detectability_training_set

    _add_comparison(search_inputs)
    search_inputs["peptide_observations"].loc[0, "comparison_group"] = pd.NA
    with pytest.raises(ValueError, match="complete comparison_group"):
        build_detectability_training_set(**search_inputs)


def test_invalid_comparison_load_is_rejected(search_inputs):
    import json

    from hitlist.detectability import build_detectability_training_set

    _add_comparison(search_inputs)
    frame = search_inputs["peptide_observations"]
    controls = json.loads(frame.comparison_controls.iloc[0])
    controls["peptide_load_ug"] = -1
    frame["comparison_controls"] = json.dumps(controls)
    with pytest.raises(ValueError, match="finite and positive"):
        build_detectability_training_set(**search_inputs)


def test_narrow_length_window_bounds_large_missed_cleavage_allowance():
    rows = list(digest_occurrences("AAK" * 1000, min_len=3, max_len=3, max_missed=1000000))
    assert len(rows) == 1000
    assert all(row.peptide == "AAK" and row.n_missed_cleavages == 0 for row in rows)
