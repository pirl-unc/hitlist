"""Cellular typing is independent of experimental MHC candidates (#520)."""

import pandas as pd
import pytest

from hitlist import curation, export

GENOTYPE = "HLA-B*35:03 HLA-B*40:02 HLA-C*04:01"
TYPING = {
    "mhc_basis": "selected_restriction",
    "mhc_genotype": GENOTYPE,
    "mhc_genotype_cell": "C1R-B*40:02",
    "mhc_genotype_complete_loci": "",
    "mhc_genotype_source": "PMID 31530632, Cell Lines",
}


def _sample(label="C1R-B40", **changes):
    return {
        "sample_label": label,
        "mhc": "HLA-B*40:02",
        "mhc_class": "I",
        **TYPING,
        **changes,
    }


def _install(monkeypatch, samples, **row_columns):
    entries = {31530632: {"ms_samples": samples}}
    monkeypatch.setattr(export, "load_pmid_overrides", lambda: entries)
    monkeypatch.setattr(curation, "load_pmid_overrides", lambda: entries)
    rows = pd.DataFrame(
        {
            "peptide": ["AAAAAAAAA"],
            "pmid": [31530632],
            "mhc_restriction": ["HLA-B*40:02"],
            "mhc_class": ["I"],
            "mhc_species": ["Homo sapiens"],
            "source": ["iedb"],
            "cell_name": [""],
            "source_tissue": [""],
            "assay_comments": [""],
            "antigen_processing_comments": [""],
            "is_binding_assay": [False],
        }
    )
    for column, value in row_columns.items():
        rows[column] = [value]
    monkeypatch.setattr("hitlist.observations.load_observations", lambda **kwargs: rows.copy())


def test_cellular_background_does_not_expand_experimental_candidates(monkeypatch):
    _install(monkeypatch, [_sample()])
    sample = export.generate_ms_samples_table().iloc[0]
    observation = export.generate_observations_table().iloc[0]
    assert sample.mhc == observation.sample_mhc == "HLA-B*40:02"
    assert observation.mhc_restriction == "HLA-B*40:02"
    assert observation.sample_attribution == "allele_exact"
    for key, value in TYPING.items():
        assert sample[key] == observation[key] == value
    assert observation.mhc_genotype_reported_loci == "HLA-B;HLA-C"
    assert export.generate_ms_peptide_summary_table(
        mhc_allele="HLA-B*35:03", peptide="AAAAAAAAA"
    ).empty


def test_legacy_candidates_are_not_automatically_a_genotype(monkeypatch):
    sample = {key: value for key, value in _sample().items() if key not in TYPING}
    _install(monkeypatch, [sample])
    row = export.generate_observations_table().iloc[0]
    assert row.sample_mhc == "HLA-B*40:02"
    assert row.mhc_genotype == row.mhc_genotype_source == row.mhc_basis == ""


@pytest.mark.parametrize("changed", ["mhc_genotype", "mhc_genotype_cell"])
def test_ambiguous_cellular_typings_are_never_pooled_or_detached(monkeypatch, changed):
    other = "HLA-B*40:02" if changed == "mhc_genotype" else "Another cell"
    _install(monkeypatch, [_sample("first"), _sample("second", **{changed: other})])
    row = export.generate_observations_table().iloc[0]
    assert row.sample_label == ""
    assert row.sample_mhc == "HLA-B*40:02"
    for key in TYPING:
        if key.startswith("mhc_genotype"):
            assert row[key] == ""


def test_shared_cellular_typing_survives_unresolved_treatment(monkeypatch):
    _install(monkeypatch, [_sample("WT"), _sample("KO")])
    row = export.generate_observations_table().iloc[0]
    assert row.sample_label == ""
    assert row.mhc_genotype == GENOTYPE
    assert row.mhc_genotype_cell == "C1R-B*40:02"


def test_typing_columns_survive_projection_and_empty_samples(monkeypatch):
    _install(monkeypatch, [_sample()])
    columns = [*TYPING, "mhc_genotype_reported_loci"]
    projected = export.generate_observations_table(columns=columns)
    assert list(projected.columns) == columns
    assert projected.iloc[0].mhc_genotype == GENOTYPE
    assert set(columns) <= set(export.generate_ms_samples_table(mhc_class="II").columns)


@pytest.mark.parametrize("mode", ["ms", "binding", "both"])
def test_training_export_preserves_ms_typing_and_leaves_binding_unknown(monkeypatch, mode):
    _install(monkeypatch, [_sample()])
    monkeypatch.setattr(
        export,
        "generate_binding_table",
        lambda **kwargs: pd.DataFrame(
            {
                "peptide": ["LLLLLLLLL"],
                "pmid": [31530632],
                "mhc_restriction": ["HLA-B*40:02"],
                "source": ["iedb"],
            }
        ),
    )
    result = export.generate_training_table(
        include_evidence=mode, columns=["evidence_kind", *curation.MHC_TYPING_COLUMNS]
    )
    for row in result.to_dict("records"):
        if row["evidence_kind"] == "ms":
            assert row["mhc_genotype"] == GENOTYPE
        else:
            assert all(row[column] == "" for column in curation.MHC_TYPING_COLUMNS)


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"mhc_basis": "genotype"}, "mhc_basis"),
        ({"mhc_genotype_source": ""}, "mhc_genotype_source"),
        ({"mhc_genotype_cell": ""}, "mhc_genotype_cell"),
        ({"mhc_genotype_complete_loci": "HLA-A"}, "complete_loci"),
        ({"mhc_genotype_complete_loci": "HLA-C;HLA-B"}, "complete_loci"),
        ({"mhc_genotype": ""}, "mhc_genotype"),
        ({"mhc_genotype": False}, "mhc_genotype"),
    ],
)
def test_invalid_typing_contract_is_rejected(changes, message):
    with pytest.raises(ValueError, match=message):
        curation.sample_mhc_metadata(_sample(**changes))


def test_lorente_primary_source_separates_background_and_selected_restriction():
    samples = curation.load_pmid_overrides()[31530632]["ms_samples"]
    assert len(samples) == 5
    for sample in samples:
        assert sample["mhc"] == "HLA-B*40:02"
        assert sample["mhc_genotype"] == GENOTYPE
        assert sample["mhc_basis"] == "selected_restriction"
        assert not sample.get("mhc_genotype_complete_loci")


def test_selected_restrictions_stay_out_of_cellular_background_attribution():
    assert set(curation.sample_alleles_for_pmid(31530632).values()) == {frozenset({"HLA-B*40:02"})}


def test_sarkizova_residual_c0102_does_not_expand_other_selected_alleles():
    samples = curation.load_pmid_overrides()[31844290]["ms_samples"]
    mono = [s for s in samples if s["sample_label"].startswith("721.221-")]
    assert len(mono) == 95
    for sample in mono:
        selected = curation.sample_mhc_candidates(sample["mhc"]).exact
        genotype = curation.sample_mhc_candidates(sample["mhc_genotype"]).exact
        assert len(selected) == 1
        assert genotype == selected | {"HLA-C*01:02"}
        assert sample["mhc_basis"] == "selected_restriction"
        assert not sample.get("mhc_genotype_complete_loci")


def _assayed_loci(mhc_field, mhc_class):
    """Loci of one MHC class named by an ``mhc`` / ``mhc_genotype`` field.

    Pairs are decomposed, so ``HLA-DQA1*02:01/DQB1*02:02`` reports both
    chains rather than the pair's own spelling.
    """
    return {
        locus
        for allele in curation.sample_mhc_candidates(mhc_field).exact
        for component in curation.expand_allele_components(allele)
        if (locus := curation.allele_locus(component))
        and curation.mhc_class_of(component) == mhc_class
    }


def _class_ii_loci(mhc_field):
    return _assayed_loci(mhc_field, "II")


def test_gbm_pan_class_ii_candidates_carry_the_cells_whole_class_ii_typing():
    """HB245 is pan-HLA-II and PMID 33592498 reports DP, DQ and DR ligands, so
    every class-II molecule the cell is typed for is an experimental candidate
    and the basis is the cell's own typing (#565)."""
    samples = export.generate_ms_samples_table()
    arms = samples[samples.pmid.eq(33592498) & samples.mhc_class.eq("II")]
    assert len(arms) == 3
    for sample in arms.itertuples():
        assert sample.ip_antibody == "HB245/IVA12"
        assert sample.mhc_basis == "sample_typing"
        assert _class_ii_loci(sample.mhc) == _class_ii_loci(sample.mhc_genotype)
        # The candidates are the class-II half of the typing only: a
        # class-II pull cannot have presented the cell's class-I molecules.
        assert not {
            allele
            for allele in curation.sample_mhc_candidates(sample.mhc).exact
            if curation.mhc_class_of(allele) == "I"
        }
    hrog02 = arms[arms.sample_label.eq("HROG02 CIITA-transduced (class II)")].iloc[0]
    # Table 1 leaves HROG02's second DPA1/DPB1 slot blank, so the candidates
    # stay one DP molecule wide while DQ spans both chains' alleles.
    assert "HLA-DPA1" not in hrog02.mhc_genotype_complete_loci.split(";")
    assert "HLA-DRB3" not in hrog02.mhc_genotype_complete_loci.split(";")


@pytest.mark.parametrize(
    "sample_label, restrictions",
    [
        (
            "HROG02 CIITA-transduced (class II)",
            ("HLA-DPA1*01:03/DPB1*04:01", "HLA-DQA1*02:01/DQB1*02:02", "HLA-DRB3*01:01"),
        ),
        (
            "HROG17 CIITA-transduced (class II)",
            ("HLA-DPB1*11:01", "HLA-DQA1*01:01/DQB1*05:01", "HLA-DQA1*05:05/DQB1*03:01"),
        ),
        ("RA CIITA-transduced (class II)", ("HLA-DPA1*01:03/DPB1*04:01", "HLA-DRB4*01:03")),
    ],
)
def test_deposited_gbm_class_ii_restrictions_are_candidates_of_their_own_arm(
    sample_label, restrictions
):
    """The join keys on the deposited restriction string, and a heterodimer
    only matches chain-first when the curated pair is written the same way
    round. These are the DP/DQ/DRB3/4 restrictions IEDB carries for this
    study; before #565 none of them could reach any arm."""
    sample = next(
        s
        for s in curation.load_pmid_overrides()[33592498]["ms_samples"]
        if s["sample_label"] == sample_label
    )
    candidates = {
        component
        for allele in curation.sample_mhc_candidates(sample["mhc"]).exact
        for component in curation.expand_allele_components(allele)
    }
    assert set(restrictions) <= candidates


@pytest.mark.parametrize(
    "restriction, sample_label",
    [
        # The review case: IEDB deposits this one as a bare beta chain with no
        # alpha partner, while the curated candidate is the heterodimer
        # HLA-DPA1*01:03/DPB1*11:01. It reaches the arm because the join emits
        # a key per component, not because the pair string matches (#151).
        ("HLA-DPB1*11:01", "HROG17 CIITA-transduced (class II)"),
        # The same arm reached by a full pair, which must keep working too.
        ("HLA-DQA1*05:05/DQB1*03:01", "HROG17 CIITA-transduced (class II)"),
        ("HLA-DRB3*01:01", "HROG02 CIITA-transduced (class II)"),
    ],
)
def test_single_chain_gbm_restriction_reaches_its_arm_through_the_allele_join(
    monkeypatch, restriction, sample_label
):
    """End-to-end, not just the component set.

    The parametrized test above asserts the components the join *derives its
    keys from*, which is an input to the behaviour: were the join to stop
    applying ``_normalized_allele_components`` per candidate, that assertion
    would keep passing while 2,509 HLA-DPB1*11:01 rows silently lost their arm
    and fell back to ``pmid_class_pool`` against the study's pooled candidate
    union. Checked by mutation — this test fails on all four claims there.

    ``allele_exact`` is the claim, not ``elution_conditions``: each of these
    restrictions is typed in exactly one of the three lines, so the key is
    unambiguous and never reaches the arm tie-break. ``assay_comments`` is
    left empty for that reason — the candidate list alone has to carry it.
    """
    monkeypatch.setattr(
        "hitlist.observations.load_observations",
        lambda **kwargs: pd.DataFrame(
            [
                {
                    "peptide": "AAAAAAAAAAAAAAA",
                    "pmid": 33592498,
                    "mhc_restriction": restriction,
                    "mhc_class": "II",
                    "mhc_species": "Homo sapiens",
                    "cell_name": "Glial cell",
                    "source_tissue": "Central nervous system (CNS)",
                    "antigen_processing_comments": "",
                    "assay_comments": "",
                    "is_binding_assay": False,
                    "source": "iedb",
                }
            ]
        ),
    )
    row = export.generate_observations_table(exclude_non_peptide_ligand=False).iloc[0]
    assert row.sample_label == sample_label
    assert row.sample_match_type == "allele_match"
    assert row.sample_attribution == "allele_exact"
    assert row.mhc_basis == "sample_typing"
    # The row now reports the arm's whole class-II typing rather than the
    # study's pooled DRB1 list, which is what #565 was about.
    assert {"HLA-DP", "HLA-DQ", "HLA-DR"} <= {prefix[:6] for prefix in row.sample_mhc.split()}
    assert row.mhc_genotype_cell == sample_label.split()[0]


def test_no_sample_drops_an_assayed_locus_its_own_typing_reports():
    """A candidate list narrower than the cell's typing at the assayed class
    means ligands from the missing locus can never match their own sample —
    the #565 shape. ``selected_restriction`` is exempt by definition: there
    the experiment, not the typing, chose the molecules.

    Read its coverage honestly before trusting it as a corpus-wide guard: it
    can only compare samples that carry *both* a candidate list and an
    independent ``mhc_genotype``, which today is a small minority of the
    curated samples — most have no genotype curated yet, and the rest are
    ``selected_restriction``. ``assert_covers`` below pins that number so the
    guard cannot quietly shrink to nothing as samples are added; growing it is
    a matter of curating more genotypes, not of loosening this test.
    """
    evaluated = 0
    findings = []
    for pmid, entry in curation.load_pmid_overrides().items():
        for sample in entry.get("ms_samples") or []:
            if sample.get("mhc_basis") == "selected_restriction":
                continue
            genotype, mhc = sample.get("mhc_genotype") or "", sample.get("mhc") or ""
            mhc_class = sample.get("mhc_class")
            if not genotype or not mhc or mhc_class not in ("I", "II"):
                continue
            evaluated += 1
            typed, candidates = (_assayed_loci(field, mhc_class) for field in (genotype, mhc))
            if missing := typed - candidates:
                findings.append((pmid, sample.get("sample_label", ""), sorted(missing)))
    assert findings == []
    # Coverage, not a threshold to tune: if this drops, the guard above went
    # quiet rather than the corpus getting cleaner.
    assert evaluated >= 9, f"locus guard now evaluates only {evaluated} samples"


def test_modc_genotype_names_p4_and_never_pools_a549_feeders():
    samples = curation.load_pmid_overrides()[35051231]["ms_samples"]
    sample = next(s for s in samples if s["sample_label"].endswith("A549 feeders"))
    assert sample["mhc_genotype_cell"] == "P4"
    alleles = curation.sample_mhc_candidates(sample["mhc_genotype"]).exact
    assert {"HLA-A*01:01", "HLA-B*51:01", "HLA-B*57:01"} <= alleles
    assert not ({"HLA-A*25:01", "HLA-A*30:01", "HLA-B*44:03"} & alleles)
    assert "HLA-DQA1" not in sample["mhc_genotype_complete_loci"]


def test_genotype_contract_is_checked_on_yaml_load(tmp_path, monkeypatch):
    import yaml

    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump([{"pmid": 1, "ms_samples": [_sample(mhc_genotype_cell="")]}]))
    real_path = curation._data_path
    monkeypatch.setattr(
        curation,
        "_data_path",
        lambda name: path if name == "pmid_overrides.yaml" else real_path(name),
    )
    curation.load_pmid_overrides.cache_clear()
    try:
        with pytest.raises(ValueError, match=r"PMID 1: ms_samples\[0\].*mhc_genotype_cell"):
            curation.load_pmid_overrides()
    finally:
        curation.load_pmid_overrides.cache_clear()


def test_token_qc_reads_independent_genotype(monkeypatch):
    from hitlist import qc

    monkeypatch.setattr(
        qc,
        "load_pmid_overrides",
        lambda: {1: {"ms_samples": [_sample(mhc_genotype="HLA-B*40:02 MADEUP")]}},
    )
    monkeypatch.setattr(qc, "ms_excluded_pmids", lambda: frozenset())
    findings = qc.mhc_token_audit(evidence_frames={})
    assert findings[["field", "token"]].values.tolist() == [["mhc_genotype", "MADEUP"]]


def test_filtered_and_projected_typing_matches_complete_export(tmp_path, monkeypatch):
    from hitlist import observations

    path = tmp_path / "observations.parquet"
    rows = [
        {
            "peptide": peptide,
            "pmid": 31530632,
            "mhc_restriction": "HLA-B*40:02",
            "mhc_class": "I",
            "mhc_species": "Homo sapiens",
            "source": "iedb",
            "cell_name": "C1R",
            "source_tissue": "",
            "antigen_processing_comments": "",
            "assay_comments": "",
            "is_binding_assay": False,
        }
        for peptide in ["AAAAAAAAA", "LLLLLLLLL"]
    ]
    pd.DataFrame(rows).to_parquet(path, index=False)
    monkeypatch.setattr(observations, "observations_path", lambda: path)
    columns = ["peptide", "sample_mhc", *curation.MHC_TYPING_COLUMNS]
    full = export.generate_observations_table(columns=columns).set_index("peptide")
    selected = export.generate_observations_table(peptide="AAAAAAAAA", columns=columns)
    pd.testing.assert_frame_equal(
        selected.set_index("peptide").astype(str), full.loc[["AAAAAAAAA"]].astype(str)
    )
    assert selected.iloc[0].mhc_genotype == GENOTYPE


def test_pooled_candidates_drop_the_basis_claim(monkeypatch):
    """``mhc_basis`` describes one sample's own candidates, so it cannot
    survive onto a row whose ``sample_mhc`` is the study's class-wide union
    of two arms' selected restrictions."""
    _install(
        monkeypatch,
        [
            _sample("first", mhc="HLA-B*40:02", **{k: "" for k in TYPING if k != "mhc_basis"}),
            _sample("second", mhc="HLA-B*35:03", **{k: "" for k in TYPING if k != "mhc_basis"}),
        ],
        mhc_restriction="HLA class I",
    )
    row = export.generate_observations_table().iloc[0]
    assert row.sample_label == ""
    assert row.sample_match_type == "pmid_class_pool"
    assert row.sample_mhc == "HLA-B*35:03 HLA-B*40:02"
    assert row.mhc_basis == ""


def test_named_sample_without_candidates_keeps_its_own_typing_only(monkeypatch):
    """A row resolved to an arm that reported no candidates inherits the
    study pool in ``sample_mhc``. The cellular typing stays that arm's own
    and the basis claim does not follow the pool."""
    _install(
        monkeypatch,
        [
            _sample("armA", mhc="HLA-A*02:01 HLA-B*07:02", **dict.fromkeys(TYPING, "")),
            _sample("armB", mhc="", mhc_basis=""),
        ],
        mhc_restriction="HLA class I",
        attributed_sample_label="armB",
    )
    row = export.generate_observations_table().iloc[0]
    assert row.sample_label == "armB"
    assert row.sample_attribution == "curated_sample_label"
    assert row.sample_match_type == "pmid_class_pool"
    assert row.sample_mhc == "HLA-A*02:01 HLA-B*07:02"
    assert row.mhc_basis == ""
    assert row.mhc_genotype == GENOTYPE
    assert row.mhc_genotype_cell == "C1R-B*40:02"


@pytest.mark.parametrize(
    "changes, message",
    [
        # ``unknown`` and ``HLA class I`` are legal, truthy ``mhc`` values that
        # name no MHC entity, so a basis claim over either asserts "this
        # sample's candidate list is its typing" about a sentinel (#564).
        ({"mhc": "unknown"}, "mhc_basis requires"),
        ({"mhc": "HLA class I"}, "mhc_basis requires"),
        # A serotype-only genotype loads but exports blank reported loci, which
        # is documented as "unknown", and then makes any completeness claim
        # unrecordable because the subset check compares against an empty set.
        ({"mhc_genotype": "HLA-DR15", "mhc_genotype_complete_loci": ""}, "precisely reported"),
    ],
)
def test_sentinels_and_serotypes_are_not_typing(changes, message):
    with pytest.raises(ValueError, match=message):
        curation.sample_mhc_metadata(_sample(**changes))


def test_ploidy_audit_covers_the_field_that_claims_to_be_a_genotype():
    """``mhc_genotype`` is defined as one cell's typing, and had no allele-count
    enforcement: ``sample_mhc_metadata`` checks completeness and provenance and
    never counts per locus, so pooling two donors' typing loaded cleanly (#564).
    """
    from hitlist import qc

    pooled = {
        1: {
            "ms_samples": [
                {
                    "sample_label": "pooled typing",
                    "mhc": "HLA-A*02:01",
                    "mhc_class": "I",
                    "mhc_genotype": "HLA-B*07:02 HLA-B*08:01 HLA-B*35:01",
                    "mhc_genotype_cell": "two donors",
                    "mhc_genotype_source": "synthetic",
                }
            ]
        }
    }
    findings = qc.sample_ploidy_audit(pooled)
    assert findings[["field", "locus", "n_alleles"]].to_dict("records") == [
        {"field": "mhc_genotype", "locus": "HLA-B", "n_alleles": 3}
    ]
    # The real corpus stays clean on both fields.
    assert qc.sample_ploidy_audit().empty
