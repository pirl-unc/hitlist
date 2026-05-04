from hitlist.export import (
    _classify_instrument,
    _extract_allele_strings,
    generate_ms_observations_table,
    generate_ms_samples_table,
    generate_species_summary,
    validate_mhc_alleles,
)


def test_ms_samples_table_columns():
    df = generate_ms_samples_table()
    # Columns split into legacy (always-present) and provenance (added by
    # issue #149).  Asserted via subset rather than equality so future
    # additions don't break the test.
    legacy = {
        "species",
        "sample_label",
        "perturbation",
        "pmid",
        "study_label",
        "mhc_class",
        "n_samples",
        "notes",
        "mhc",
        "ip_antibody",
        "acquisition_mode",
        "instrument",
        "instrument_type",
        "fragmentation",
        "labeling",
        "search_engine",
        "fdr",
    }
    provenance = {
        "condition",
        "source",
        "profiled",
        "peptides",
        "reference_proteomes",
    }
    cols = set(df.columns)
    assert legacy <= cols, f"missing legacy columns: {legacy - cols}"
    assert provenance <= cols, f"missing provenance columns: {provenance - cols}"


def test_ms_samples_table_not_empty():
    df = generate_ms_samples_table()
    assert len(df) > 0


def test_ms_samples_filter_class_i():
    df_all = generate_ms_samples_table()
    df_i = generate_ms_samples_table(mhc_class="I")
    assert len(df_i) > 0
    assert len(df_i) < len(df_all)
    for cls in df_i["mhc_class"]:
        assert "I" in cls


def test_ms_samples_filter_class_ii():
    df_ii = generate_ms_samples_table(mhc_class="II")
    assert len(df_ii) > 0
    for cls in df_ii["mhc_class"]:
        assert "II" in cls


def test_ms_samples_no_zero_n():
    """Placeholder rows with n=0 should be excluded."""
    df = generate_ms_samples_table()
    for n in df["n_samples"].dropna():
        assert n > 0


def test_ms_samples_acquisition_metadata():
    """Spot-check that acquisition metadata is populated for curated studies."""
    df = generate_ms_samples_table()
    # Mommen 2014 — EThcD fragmentation
    mommen = df[df["pmid"] == 24616531]
    assert len(mommen) > 0
    assert mommen.iloc[0]["fragmentation"] == "EThcD"
    assert mommen.iloc[0]["ip_antibody"] == "W6/32"
    # Ritz 2017b — DIA acquisition
    ritz = df[df["pmid"] == 28834231]
    assert len(ritz) > 0
    assert ritz.iloc[0]["acquisition_mode"] == "DIA"
    # Pfammatter 2020 — TMT labeling
    pfammatter = df[df["pmid"] == 32502341]
    assert len(pfammatter) > 0
    assert pfammatter.iloc[0]["labeling"] == "TMT"


def test_classify_instrument():
    assert _classify_instrument("Q Exactive HF-X") == "Orbitrap"
    assert _classify_instrument("Orbitrap Fusion Lumos") == "Orbitrap"
    assert _classify_instrument("LTQ Orbitrap Elite") == "Orbitrap"
    assert _classify_instrument("timsTOF Pro") == "timsTOF"
    assert _classify_instrument("TripleTOF 5600") == "TOF"
    assert _classify_instrument("TSQ Altis Plus") == "QqQ"
    assert _classify_instrument("") == ""
    assert _classify_instrument("Some Novel Instrument") == "Some Novel Instrument"


def test_ms_samples_instrument_type():
    """instrument_type should be derived from instrument."""
    df = generate_ms_samples_table()
    qe = df[df["instrument"] == "Q Exactive"]
    assert len(qe) > 0
    assert qe.iloc[0]["instrument_type"] == "Orbitrap"


def test_ms_samples_two_level_inheritance():
    """Per-sample ip_antibody should override PMID-level default."""
    df = generate_ms_samples_table()
    # Abelin 2019 has MAPTAC (streptavidin) and tissue (L243) samples
    abelin = df[df["pmid"] == 31495665]
    if len(abelin) > 1:
        antibodies = set(abelin["ip_antibody"].dropna())
        # Should have at least 2 distinct ip_antibody values
        assert len(antibodies) > 1


def test_ms_samples_mhc_per_sample():
    """Sarkizova validation samples should have per-sample MHC alleles."""
    df = generate_ms_samples_table()
    sarkizova = df[df["pmid"] == 31844290]
    cll_a = sarkizova[sarkizova["sample_label"].str.contains("CLL A")]
    assert len(cll_a) == 1
    mhc = cll_a.iloc[0]["mhc"]
    assert "A*03:01" in mhc
    assert "B*14:02" in mhc


def test_ms_samples_mhc_class_only():
    """Pat9 ccRCC has no allele genotype but the IP antibody (W6/32) tells
    us the class.  Since 1.7.1 we record ``"HLA class I"`` instead of the
    less-informative ``"unknown"`` sentinel.
    """
    df = generate_ms_samples_table()
    sarkizova = df[df["pmid"] == 31844290]
    pat9 = sarkizova[sarkizova["sample_label"].str.contains("Pat9")]
    assert len(pat9) == 1
    assert pat9.iloc[0]["mhc"] == "HLA class I"


def test_ms_samples_strazar_2023():
    """Stražar 2023 should export one class II sample row per profiled allele."""
    df = generate_ms_samples_table()
    st = df[df["pmid"] == 37301199]
    assert len(st) == 42
    assert set(st["mhc_class"]) == {"II"}
    assert st["mhc"].nunique() == 42
    assert (st["instrument"] == "Orbitrap Exploris 480").all()
    assert (st["ip_antibody"] == "Strep-Tactin XT Sepharose (Strep-tag II)").all()
    assert (st["sample_label"].str.startswith("StrepII-")).all()


def test_ms_samples_alpizar_2017_split():
    """Alpizar 2017 should split per transfectant (B*40:02, B*39:01) plus
    a pooled entry that owns the 515 IEDB rows whose APC mentions both
    transfectants. None should carry the legacy B*40:01 (issue #207).
    """
    df = generate_ms_samples_table()
    al = df[df["pmid"] == 27920218]
    assert len(al) == 3, f"expected 3 entries (2 transfectants + pooled), got {len(al)}"
    assert "HLA-B*40:01" not in set(al["mhc"])
    labels = set(al["sample_label"])
    assert "C1R-HLA-B*40:02" in labels
    assert "C1R-HLA-B*39:01" in labels
    assert any("pooled" in lbl.lower() for lbl in labels)
    assert set(al["mhc_class"]) == {"I"}


def test_ms_samples_marcilla_2014_c1r_b4002():
    """Marcilla 2014 should export as one C1R mono-allelic B*40:02 sample."""
    df = generate_ms_samples_table()
    mc = df[df["pmid"] == 24366607]
    assert len(mc) == 1, f"expected 1 Marcilla sample, got {len(mc)}"
    row = mc.iloc[0]
    assert row["sample_label"] == "C1R-HLA-B*40:02"
    assert row["mhc"] == "HLA-B*40:02"
    assert row["mhc_class"] == "I"
    assert row["ip_antibody"] == "W6/32"


def test_ms_samples_illing_2018_split():
    """Illing 2018 should be split per HLA-B57/B58 transfectant (3 entries)."""
    df = generate_ms_samples_table()
    il = df[df["pmid"] == 30410026]
    assert len(il) == 3, f"expected 3 per-transfectant entries, got {len(il)}"
    assert set(il["mhc"]) == {"HLA-B*57:01", "HLA-B*57:03", "HLA-B*58:01"}
    assert set(il["sample_label"]) == {
        "C1R-HLA-B*57:01",
        "C1R-HLA-B*57:03",
        "C1R-HLA-B*58:01",
    }
    assert set(il["mhc_class"]) == {"I"}


def test_ms_samples_trolle_2016_split():
    """Trolle 2016 should be split per HeLa sHLA transfectant (5 entries)."""
    df = generate_ms_samples_table()
    tr = df[df["pmid"] == 26783342]
    assert len(tr) == 5, f"expected 5 per-transfectant entries, got {len(tr)}"
    assert set(tr["mhc"]) == {
        "HLA-A*01:01",
        "HLA-A*02:01",
        "HLA-A*24:02",
        "HLA-B*07:02",
        "HLA-B*51:01",
    }
    assert (tr["sample_label"].str.startswith("HeLa-sHLA-HLA-")).all()
    assert set(tr["mhc_class"]) == {"I"}


def test_ms_samples_chen_2020_hela_abc_ko():
    """Chen 2020 should encode HeLa.ABC-KO + B*51:01 with control vs ERAP1 shRNA arms."""
    df = generate_ms_samples_table()
    ch = df[df["pmid"] == 32161166]
    assert len(ch) == 2, f"expected 2 ms_sample entries, got {len(ch)}"
    assert set(ch["mhc"]) == {"HLA-B*51:01"}
    assert (ch["sample_label"].str.startswith("HeLa.ABC-KO-HLA-B*51:01")).all()
    assert set(ch["mhc_class"]) == {"I"}


def test_liepe_2016_fibroblast_not_cancer():
    """Liepe 2016 Direct Ex Vivo fibroblasts should classify as healthy, not cancer."""
    from hitlist.curation import classify_ms_row

    row = classify_ms_row(
        process_type="No immunization",
        disease="",
        culture_condition="Direct Ex Vivo",
        source_tissue="Skin",
        cell_name="Fibroblast",
        pmid=27846572,
        mhc_restriction="HLA-A*02:01",
        submission_id="",
    )
    assert row["src_cancer"] is False
    assert row["src_healthy_tissue"] is True


def test_caron_2015_pbmc_not_cancer():
    """Caron 2015 Direct Ex Vivo PBMCs should classify as healthy, not cancer."""
    from hitlist.curation import classify_ms_row

    row = classify_ms_row(
        process_type="No immunization",
        disease="",
        culture_condition="Direct Ex Vivo",
        source_tissue="Blood",
        cell_name="PBMC",
        pmid=26154972,
        mhc_restriction="HLA-A*02:01",
        submission_id="",
    )
    assert row["src_cancer"] is False
    assert row["src_healthy_tissue"] is True


def test_ms_samples_sarango_2022_dr_allele():
    """Sarango 2022 should include HLA-DRB1*01:02 in the mhc string (paper-verified)."""
    df = generate_ms_samples_table()
    sa = df[df["pmid"] == 36215666]
    assert len(sa) == 2
    assert set(sa["mhc_class"]) == {"I+II"}
    assert all("HLA-DRB1*01:02" in m for m in sa["mhc"]), (
        f"expected DRB1*01:02 in all mhc strings, got {list(sa['mhc'])}"
    )
    assert all("HLA-A*68:02" in m for m in sa["mhc"])


def test_ms_samples_weingarten_gabbay_2021_no_hbec():
    """Weingarten-Gabbay 2021: only A549-ACE2-TMPRSS2 and HEK293T-ACE2-TMPRSS2; HBECs never profiled."""
    df = generate_ms_samples_table()
    wg = df[df["pmid"] == 34171305]
    assert len(wg) == 4, f"expected 4 entries (2 cell lines x uninf/inf), got {len(wg)}"
    labels = set(wg["sample_label"])
    assert not any("HBEC" in label for label in labels), f"HBECs should be dropped, got: {labels}"
    assert sum("A549" in label for label in labels) == 2
    assert sum("HEK293T" in label for label in labels) == 2
    assert sum("uninfected" in label for label in labels) == 2
    assert sum("SARS-CoV-2-infected" in label for label in labels) == 2


def test_ms_samples_parquet_roundtrip(tmp_path):
    """ms_samples_table should write to parquet and read back identically."""
    import pandas as pd

    df = generate_ms_samples_table()
    out = tmp_path / "samples.parquet"
    df.to_parquet(out, index=False)
    assert out.exists()
    rt = pd.read_parquet(out)
    assert len(rt) == len(df)
    assert set(rt.columns) == set(df.columns)
    # Spot-check a known entry survives the round-trip
    assert 32161166 in rt["pmid"].values  # Chen 2020


def test_scan_supplementary_parquet_roundtrip(tmp_path):
    """scan_supplementary output should survive a parquet round-trip."""
    import pandas as pd

    from hitlist.supplement import scan_supplementary

    df = scan_supplementary()
    out = tmp_path / "supp.parquet"
    df.to_parquet(out, index=False)
    rt = pd.read_parquet(out)
    assert len(rt) == len(df)
    assert "peptide" in rt.columns
    assert "mhc_restriction" in rt.columns
    assert "pmid" in rt.columns
    # Every supplementary row must carry a PMID
    assert rt["pmid"].notna().all()


def test_generate_observations_table(full_observations_df):
    """Observations table should join peptides with sample metadata."""
    df = full_observations_df
    assert len(df) > 0
    # Original observation columns
    assert "peptide" in df.columns
    assert "mhc_restriction" in df.columns
    # Enriched sample metadata columns
    assert "instrument" in df.columns
    assert "instrument_type" in df.columns
    assert "sample_mhc" in df.columns
    assert "quantification_method" in df.columns


def test_generate_observations_monoallelic_filter(full_observations_df):
    """--mono-allelic / --multi-allelic should change row counts."""
    df_all = full_observations_df
    df_mono = df_all[df_all["is_monoallelic"]]
    df_multi = df_all[~df_all["is_monoallelic"]]
    # Both subsets should be non-empty and smaller than the full table
    assert len(df_mono) > 0, "Mono-allelic filter returned no rows"
    assert len(df_multi) > 0, "Multi-allelic filter returned no rows"
    assert len(df_mono) < len(df_all), "Mono-allelic filter did not reduce row count"
    assert len(df_multi) < len(df_all), "Multi-allelic filter did not reduce row count"


def test_generate_observations_provenance_columns(full_observations_df):
    """Provenance columns should be present and meaningful."""
    df = full_observations_df
    assert "sample_match_type" in df.columns
    assert "matched_sample_count" in df.columns
    assert "has_peptide_level_allele" in df.columns
    # Match types should only be these values
    valid_types = {"allele_match", "single_sample_fallback", "pmid_class_pool", "unmatched"}
    assert set(df["sample_match_type"].unique()).issubset(valid_types)
    # Most rows with alleles should have allele_match
    has_allele = df[df["has_peptide_level_allele"]]
    if len(has_allele) > 0:
        allele_matched = (has_allele["sample_match_type"] == "allele_match").sum()
        assert allele_matched > 0, "No rows have allele_match despite having alleles"


def test_generate_observations_parquet_export(tmp_path, full_observations_df):
    """Parquet export path should produce a readable file."""
    df = full_observations_df
    out = tmp_path / "obs.parquet"
    df.to_parquet(out, index=False)
    import pandas as pd

    loaded = pd.read_parquet(out)
    assert len(loaded) == len(df)
    assert set(loaded.columns) == set(df.columns)


def test_generate_observations_table_not_built():
    """Should raise FileNotFoundError when observations not built."""
    from hitlist.observations import is_built

    if is_built():
        import pytest

        pytest.skip("Observations table is built — cannot test error path")
    import pytest

    from hitlist.export import generate_observations_table

    with pytest.raises(FileNotFoundError, match="not built"):
        generate_observations_table()


def test_ms_samples_species_normalized():
    """Species column should contain canonical names, not parenthetical IEDB format."""
    df = generate_ms_samples_table()
    for s in df["species"]:
        assert "(" not in s, f"Species not normalized: {s}"


# ── Sample provenance preservation (issue #149) ────────────────────────────


def test_ms_samples_preserves_original_condition():
    """Original ``condition`` is preserved alongside the simplified
    ``perturbation`` (issue #149).  For unperturbed samples the condition
    is non-empty (e.g. ``"unperturbed"``) but perturbation is empty.
    """
    df = generate_ms_samples_table()
    # At least some rows have unperturbed samples — verify they retain the
    # original condition while perturbation is blank.
    unperturbed = df[df["perturbation"] == ""]
    assert not unperturbed.empty
    # Some unperturbed rows must have a non-empty condition string.
    assert (unperturbed["condition"].astype(str).str.len() > 0).any(), (
        "expected at least one unperturbed sample to carry a non-empty "
        "raw condition string after issue #149"
    )


def test_ms_samples_preserves_reference_proteomes():
    """``reference_proteomes`` is serialized as ``UPID:label;...``"""
    df = generate_ms_samples_table()
    assert "reference_proteomes" in df.columns
    # The Gomez-Zepeda Raji entries (#147) carry EBV reference proteomes;
    # at least one row should have a non-empty serialized string.
    has_proteome = df["reference_proteomes"].astype(str).str.contains("UP")
    assert has_proteome.any(), (
        "expected at least one ms_samples row with a serialized reference_proteomes UniProt ID"
    )


def test_ms_samples_preserves_source_field():
    """The original ``source`` field (e.g. tissue source notes) is exported."""
    df = generate_ms_samples_table()
    assert "source" in df.columns
    assert (df["source"].astype(str).str.len() > 0).any()


def test_serialize_reference_proteomes_handles_dict_list():
    from hitlist.export import _serialize_reference_proteomes

    result = _serialize_reference_proteomes(
        [
            {"uniprot": "UP000153037", "proteome_label": "Epstein-Barr virus"},
            {"uniprot": "UP000464024", "proteome_label": "SARS-CoV-2"},
        ]
    )
    assert result == "UP000153037:Epstein-Barr virus;UP000464024:SARS-CoV-2"


def test_serialize_reference_proteomes_handles_partial_entries():
    from hitlist.export import _serialize_reference_proteomes

    # Only UPID, no label.
    assert _serialize_reference_proteomes([{"uniprot": "UP000005640"}]) == "UP000005640"
    # Only label, no UPID.
    assert _serialize_reference_proteomes([{"proteome_label": "human"}]) == "human"
    # Empty / None inputs.
    assert _serialize_reference_proteomes(None) == ""
    assert _serialize_reference_proteomes([]) == ""
    # Already-serialized string passes through.
    assert _serialize_reference_proteomes("UP000005640:human") == "UP000005640:human"


def test_generate_sample_expression_table_carries_provenance():
    """Sample-expression export must include the same provenance columns
    as ``generate_ms_samples_table`` plus the expression-anchor columns
    (issue #149).
    """
    from hitlist.export import generate_sample_expression_table

    df = generate_sample_expression_table()
    assert not df.empty
    for col in (
        "sample_label",
        "pmid",
        "study_label",
        "mhc_class",
        "condition",
        "perturbation",
        "source",
        "profiled",
        "peptides",
        "reference_proteomes",
        "mhc",
        "expression_backend",
        "expression_key",
        "expression_match_tier",
        "expression_parent_key",
    ):
        assert col in df.columns, f"missing column: {col}"


def test_attach_peptide_origin_passes_cell_name_and_tissue_to_resolver(tmp_path, monkeypatch):
    """When the observations frame carries ``cell_name`` / ``source_tissue``,
    those values are passed into the expression-anchor resolver so a bare
    ``sample_label`` like ``"donor 42"`` can still resolve via the
    cell_name (issue #149).
    """
    import pandas as pd

    from hitlist.export import _attach_peptide_origin

    captured: list[dict] = []

    # Stub the resolver in line_expression so we can observe the kwargs.
    def fake_resolver(sample_label, **kwargs):
        captured.append({"sample_label": sample_label, **kwargs})

        class _Anchor:
            expression_backend = "depmap_rna"
            expression_key = "HeLa"
            expression_match_tier = 1
            expression_parent_key = None
            source_ids = ()
            reason = "fake"
            matched_alias = "hela"

        return _Anchor

    monkeypatch.setattr("hitlist.line_expression.resolve_sample_expression_anchor", fake_resolver)

    # Stub mappings + line_expression load so the rest of the pipeline runs.
    monkeypatch.setattr(
        "hitlist.mappings.load_peptide_mappings",
        lambda peptide=None, columns=None, **_: pd.DataFrame(
            [{"peptide": "P", "gene_name": "TP53", "gene_id": "", "protein_id": ""}]
            if peptide
            else []
        ),
    )
    monkeypatch.setattr(
        "hitlist.line_expression.load_line_expression",
        lambda line_key=None, **_: pd.DataFrame(
            {
                "line_key": [line_key or "HeLa"],
                "source_id": ["mock"],
                "granularity": ["gene"],
                "gene_id": [""],
                "gene_name": ["TP53"],
                "transcript_id": [""],
                "tpm": [10.0],
                "log2_tpm": [3.0],
            }
        ),
    )

    df = pd.DataFrame(
        {
            "peptide": ["P"],
            "sample_label": ["donor 42"],
            "pmid": [12345],
            "study_label": ["S"],
            "cell_name": ["HeLa cells"],
            "source_tissue": ["cervix, uterine"],
        }
    )
    _attach_peptide_origin(df)

    assert captured, "resolver was not called"
    call = captured[0]
    assert call["cell_name"] == "HeLa cells"
    assert call["lineage_tissue"] == "cervix, uterine"


# ── Class-II heterodimer component matching (issue #151) ───────────────────


def test_expand_heterodimer_components_dp_pair():
    """HLA-DP heterodimer string splits into full + beta + alpha components."""
    from hitlist.export import _expand_heterodimer_components

    out = _expand_heterodimer_components("HLA-DPB1*06:01/DPA1*01:03")
    assert out[0] == "HLA-DPB1*06:01/DPA1*01:03"
    assert "HLA-DPB1*06:01" in out
    assert "HLA-DPA1*01:03" in out


def test_expand_heterodimer_components_dq_pair():
    from hitlist.export import _expand_heterodimer_components

    out = _expand_heterodimer_components("HLA-DQB1*06:04/DQA1*01:02")
    assert out[0] == "HLA-DQB1*06:04/DQA1*01:02"
    assert "HLA-DQB1*06:04" in out
    assert "HLA-DQA1*01:02" in out


def test_expand_heterodimer_components_class_i_passthrough():
    from hitlist.export import _expand_heterodimer_components

    assert _expand_heterodimer_components("HLA-A*02:01") == ["HLA-A*02:01"]


def test_expand_heterodimer_components_empty_and_whitespace():
    from hitlist.export import _expand_heterodimer_components

    assert _expand_heterodimer_components("") == []
    assert _expand_heterodimer_components("   ") == []


def test_beta_chain_only_restriction_joins_heterodimer_sample(tmp_path, monkeypatch):
    """An observation with ``mhc_restriction="HLA-DPB1*06:01"`` must join
    to a curated MAPTAC sample whose mhc is the heterodimer
    ``"HLA-DPB1*06:01/DPA1*01:03"`` — issue #151.
    """
    import pandas as pd

    from hitlist.export import generate_observations_table

    # Monkeypatch load_pmid_overrides to return a single synthetic PMID
    # with a DP heterodimer sample, so we don't depend on the real YAML.
    def fake_overrides():
        return {
            99999001: {
                "study_label": "synthetic — DP heterodimer MAPTAC",
                "species": "Homo sapiens (human)",
                "ms_samples": [
                    {
                        "sample_label": "MAPTAC DP heterodimer",
                        "n_samples": 1,
                        "condition": "unperturbed",
                        "mhc_class": "II",
                        "mhc": "HLA-DPB1*06:01/DPA1*01:03",
                    }
                ],
            }
        }

    monkeypatch.setattr("hitlist.export.load_pmid_overrides", fake_overrides)

    obs_data = pd.DataFrame(
        {
            # Beta-chain-only restriction — should still match the sample.
            "peptide": ["BETACHAINAB"],
            "mhc_restriction": ["HLA-DPB1*06:01"],
            "mhc_class": ["II"],
            "reference_iri": ["iri:1"],
            "pmid": pd.array([99999001], dtype="Int64"),
            "source": ["iedb"],
            "mhc_species": ["Homo sapiens"],
            "is_monoallelic": [False],
            "is_binding_assay": [False],
            "qualitative_measurement": ["Positive"],
        }
    )
    obs_path = tmp_path / "observations.parquet"
    obs_data.to_parquet(obs_path, index=False)
    monkeypatch.setattr("hitlist.observations.observations_path", lambda: obs_path)

    df = generate_observations_table()
    assert len(df) == 1
    row = df.iloc[0]
    assert row["sample_label"] == "MAPTAC DP heterodimer"
    assert row["sample_match_type"] == "allele_match"


def test_heterodimer_string_still_matches_directly(tmp_path, monkeypatch):
    """An observation that reports the full heterodimer string must continue
    to match — expansion adds components WITHOUT removing the original
    full-string match.
    """
    import pandas as pd

    from hitlist.export import generate_observations_table

    def fake_overrides():
        return {
            99999002: {
                "study_label": "synthetic",
                "species": "Homo sapiens (human)",
                "ms_samples": [
                    {
                        "sample_label": "DQ heterodimer sample",
                        "n_samples": 1,
                        "condition": "unperturbed",
                        "mhc_class": "II",
                        "mhc": "HLA-DQB1*06:04/DQA1*01:02",
                    }
                ],
            }
        }

    monkeypatch.setattr("hitlist.export.load_pmid_overrides", fake_overrides)

    obs_data = pd.DataFrame(
        {
            "peptide": ["FULLPAIRAB"],
            "mhc_restriction": ["HLA-DQB1*06:04/DQA1*01:02"],
            "mhc_class": ["II"],
            "reference_iri": ["iri:2"],
            "pmid": pd.array([99999002], dtype="Int64"),
            "source": ["iedb"],
            "mhc_species": ["Homo sapiens"],
            "is_monoallelic": [False],
            "is_binding_assay": [False],
            "qualitative_measurement": ["Positive"],
        }
    )
    obs_path = tmp_path / "observations.parquet"
    obs_data.to_parquet(obs_path, index=False)
    monkeypatch.setattr("hitlist.observations.observations_path", lambda: obs_path)

    df = generate_observations_table()
    assert len(df) == 1
    row = df.iloc[0]
    assert row["sample_match_type"] == "allele_match"
    assert row["sample_label"] == "DQ heterodimer sample"


def test_alpha_chain_only_restriction_also_matches(tmp_path, monkeypatch):
    """Rare but possible: an observation reporting only the alpha chain
    of a curated DP heterodimer must still match its sample.
    """
    import pandas as pd

    from hitlist.export import generate_observations_table

    def fake_overrides():
        return {
            99999003: {
                "study_label": "synthetic",
                "species": "Homo sapiens (human)",
                "ms_samples": [
                    {
                        "sample_label": "DP heterodimer sample",
                        "n_samples": 1,
                        "condition": "unperturbed",
                        "mhc_class": "II",
                        "mhc": "HLA-DPB1*06:01/DPA1*01:03",
                    }
                ],
            }
        }

    monkeypatch.setattr("hitlist.export.load_pmid_overrides", fake_overrides)

    obs_data = pd.DataFrame(
        {
            "peptide": ["ALPHAONLYA"],
            "mhc_restriction": ["HLA-DPA1*01:03"],
            "mhc_class": ["II"],
            "reference_iri": ["iri:3"],
            "pmid": pd.array([99999003], dtype="Int64"),
            "source": ["iedb"],
            "mhc_species": ["Homo sapiens"],
            "is_monoallelic": [False],
            "is_binding_assay": [False],
            "qualitative_measurement": ["Positive"],
        }
    )
    obs_path = tmp_path / "observations.parquet"
    obs_data.to_parquet(obs_path, index=False)
    monkeypatch.setattr("hitlist.observations.observations_path", lambda: obs_path)

    df = generate_observations_table()
    assert len(df) == 1
    assert df.iloc[0]["sample_match_type"] == "allele_match"


def test_species_summary_columns():
    """Summary sources from observations.parquet (#117, v1.15.0).

    Old columns (n_studies / n_sample_types / n_samples) came from
    pmid_overrides.yaml curation and undercounted real data coverage
    by orders of magnitude. Replaced with parquet-derived counts.

    The empty-index path still returns the canonical column set — this
    test runs regardless of whether observations.parquet is built.
    """
    df = generate_species_summary()
    expected = {
        "species",
        "mhc_class",
        "n_pmids",
        "n_peptides",
        "n_observations",
    }
    assert expected == set(df.columns)


def test_species_summary_has_multiple_species():
    from hitlist.observations import is_built

    if not is_built():
        import pytest

        pytest.skip("Observations table not built")
    df = generate_species_summary()
    assert df["species"].nunique() > 1


def test_species_summary_class_filter():
    from hitlist.observations import is_built

    if not is_built():
        import pytest

        pytest.skip("Observations table not built")
    df_i = generate_species_summary(mhc_class="I")
    assert len(df_i) > 0
    assert set(df_i["mhc_class"]) == {"I"}


def test_species_summary_covers_non_curated_species():
    """Mouse has hundreds of PMIDs in the parquet; the summary must reflect that.

    Before #117, non-human species showed ``0`` in ``n_samples`` because
    ``pmid_overrides.yaml`` only has a handful of curated mouse entries.
    The parquet-derived summary must surface the full coverage.
    """
    from hitlist.observations import is_built

    if not is_built():
        import pytest

        pytest.skip("Observations table not built")
    df = generate_species_summary(mhc_class="I")
    mouse = df[df["species"] == "Mus musculus"]
    assert len(mouse) == 1, "mouse should appear exactly once for class I"
    mouse_row = mouse.iloc[0]
    # Real mouse data has at least 100 PMIDs (typically ~388 for class I).
    assert int(mouse_row["n_pmids"]) > 100, (
        f"Mus musculus I should have many PMIDs in the parquet, got {mouse_row['n_pmids']}"
    )
    assert int(mouse_row["n_peptides"]) > 10_000
    assert int(mouse_row["n_observations"]) > 10_000


def test_species_summary_counts_are_coherent():
    """n_observations >= n_peptides >= n_pmids is a structural invariant.

    Each PMID contributes at least one peptide, each peptide at least
    one observation row, and the inequalities are non-strict when every
    peptide happens to appear once in a single PMID.
    """
    from hitlist.observations import is_built

    if not is_built():
        import pytest

        pytest.skip("Observations table not built")
    df = generate_species_summary()
    assert (df["n_observations"] >= df["n_peptides"]).all()
    assert (df["n_peptides"] >= df["n_pmids"]).all()


def test_species_summary_empty_when_not_built(tmp_path, monkeypatch):
    """Unbuilt-index path returns empty frame with canonical columns."""
    from hitlist.observations import is_built

    if is_built():
        import pytest

        pytest.skip("Observations table is built — cannot test empty path")
    df = generate_species_summary()
    assert len(df) == 0
    assert set(df.columns) == {"species", "mhc_class", "n_pmids", "n_peptides", "n_observations"}


def test_validate_alleles_columns():
    df = validate_mhc_alleles()
    expected = {
        "pmid",
        "study_label",
        "allele",
        "parsed_name",
        "parsed_type",
        "species",
        "valid",
    }
    assert expected == set(df.columns)


def test_validate_alleles_most_valid():
    df = validate_mhc_alleles()
    if len(df) > 0:
        valid_pct = df["valid"].mean()
        assert valid_pct > 0.5, f"Only {valid_pct:.0%} of alleles parsed successfully"


def test_extract_allele_strings_list():
    result = _extract_allele_strings(["HLA-A*02:01", "HLA-B*07:02"])
    assert result == ["HLA-A*02:01", "HLA-B*07:02"]


def test_extract_allele_strings_dict():
    result = _extract_allele_strings({"donor_1": ["HLA-A*02:01"], "donor_2": ["HLA-B*07:02"]})
    assert "HLA-A*02:01" in result
    assert "HLA-B*07:02" in result


def test_extract_allele_strings_description():
    result = _extract_allele_strings("51 HLA-I allotypes (95% coverage)")
    assert result == []


def test_observations_join_with_synthetic_fixture(tmp_path, monkeypatch):
    """End-to-end join test using a small synthetic observations table."""
    import pandas as pd

    from hitlist.export import generate_observations_table

    # Build a tiny synthetic observations parquet
    obs_data = pd.DataFrame(
        {
            "peptide": ["AAAAAAAAA", "BBBBBBBBB", "CCCCCCCCC"],
            "mhc_restriction": ["HLA-A*02:01", "HLA-B*07:02", ""],
            "mhc_class": ["I", "I", "I"],
            "reference_iri": ["iri:1", "iri:2", "iri:3"],
            "pmid": pd.array([33858848, 33858848, 38480730], dtype="Int64"),
            "source": ["iedb", "iedb", "supplement"],
            "mhc_species": ["Homo sapiens", "Homo sapiens", "Homo sapiens"],
            "is_monoallelic": [False, False, False],
            "is_binding_assay": [False, False, False],
            "qualitative_measurement": ["Positive", "Positive", ""],
        }
    )
    obs_path = tmp_path / "observations.parquet"
    obs_data.to_parquet(obs_path, index=False)

    # Monkeypatch the observations path
    monkeypatch.setattr("hitlist.observations.observations_path", lambda: obs_path)

    df = generate_observations_table()
    assert len(df) == 3
    # Provenance columns should be present
    assert "sample_match_type" in df.columns
    assert "matched_sample_count" in df.columns
    assert "has_peptide_level_allele" in df.columns
    # Third row has no allele — flag should reflect that
    row3 = df[df["peptide"] == "CCCCCCCCC"].iloc[0]
    assert row3["has_peptide_level_allele"] is False or row3["has_peptide_level_allele"] == False  # noqa: E712


def test_generate_observations_has_peptide_level_allele_uses_resolution(tmp_path, monkeypatch):
    """Class-only / serological restrictions are not allele-level evidence."""
    import pandas as pd

    from hitlist.export import generate_observations_table

    obs_data = pd.DataFrame(
        {
            "peptide": ["AAA", "BBB", "CCC"],
            "mhc_restriction": ["HLA-A*02:01", "HLA-A2", "HLA class I"],
            "mhc_class": ["I", "I", "I"],
            "reference_iri": ["1", "2", "3"],
            "pmid": pd.array([33858848, 33858848, 33858848], dtype="Int64"),
            "source": ["iedb", "iedb", "iedb"],
            "mhc_species": ["Homo sapiens", "Homo sapiens", "Homo sapiens"],
            "is_monoallelic": [False, False, False],
            "is_binding_assay": [False, False, False],
            "qualitative_measurement": ["Positive", "Positive", "Positive"],
            "allele_resolution": ["four_digit", "serological", "class_only"],
        }
    )
    obs_path = tmp_path / "observations.parquet"
    obs_data.to_parquet(obs_path, index=False)
    monkeypatch.setattr("hitlist.observations.observations_path", lambda: obs_path)

    df = generate_observations_table()
    by_peptide = df.set_index("peptide")["has_peptide_level_allele"].to_dict()
    assert by_peptide == {"AAA": True, "BBB": False, "CCC": False}


def test_generate_observations_gene_filter_requires_mappings(tmp_path, monkeypatch):
    """Using --gene without a peptide_mappings sidecar should error clearly."""
    import pandas as pd
    import pytest

    from hitlist.export import generate_observations_table

    obs_data = pd.DataFrame(
        {
            "peptide": ["AAAAAAAAA"],
            "mhc_restriction": ["HLA-A*02:01"],
            "mhc_class": ["I"],
            "reference_iri": ["iri:1"],
            "pmid": pd.array([33858848], dtype="Int64"),
            "source": ["iedb"],
            "mhc_species": ["Homo sapiens"],
            "is_monoallelic": [False],
            "is_binding_assay": [False],
            "qualitative_measurement": ["Positive"],
        }
    )
    obs_path = tmp_path / "observations.parquet"
    obs_data.to_parquet(obs_path, index=False)
    monkeypatch.setattr("hitlist.observations.observations_path", lambda: obs_path)

    with pytest.raises(ValueError, match="mappings-built"):
        generate_observations_table(gene="PRAME")


def test_generate_observations_gene_filter_matches(tmp_path, monkeypatch):
    """--gene-name should resolve via the mappings sidecar.

    Two PRAME peptides + one MAGEA1 peptide.  The mappings sidecar holds
    the long-form (peptide, gene) data; observations.parquet only carries
    central semicolon-joined gene_names columns.
    """
    import pandas as pd

    from hitlist import downloads
    from hitlist.export import generate_observations_table

    monkeypatch.setattr(downloads, "_override_data_dir", tmp_path)

    obs_data = pd.DataFrame(
        {
            "peptide": ["AAAAAAAAA", "BBBBBBBBB", "CCCCCCCCC"],
            "mhc_restriction": ["HLA-A*02:01", "HLA-B*07:02", "HLA-A*02:01"],
            "mhc_class": ["I", "I", "I"],
            "reference_iri": ["iri:1", "iri:2", "iri:3"],
            "pmid": pd.array([33858848, 33858848, 33858848], dtype="Int64"),
            "source": ["iedb", "iedb", "iedb"],
            "mhc_species": ["Homo sapiens", "Homo sapiens", "Homo sapiens"],
            "is_monoallelic": [False, False, False],
            "is_binding_assay": [False, False, False],
            "qualitative_measurement": ["Positive", "Positive", "Positive"],
            "gene_names": ["PRAME", "MAGEA1", "PRAME"],
            "gene_ids": ["ENSG00000185686", "ENSG00000198681", "ENSG00000185686"],
            "protein_ids": ["P_PRAME", "P_MAGEA1", "P_PRAME"],
        }
    )
    obs_path = tmp_path / "observations.parquet"
    obs_data.to_parquet(obs_path, index=False)
    monkeypatch.setattr("hitlist.observations.observations_path", lambda: obs_path)

    # Build the long-form mappings sidecar
    mappings_data = pd.DataFrame(
        {
            "peptide": ["AAAAAAAAA", "BBBBBBBBB", "CCCCCCCCC"],
            "protein_id": ["P_PRAME", "P_MAGEA1", "P_PRAME"],
            "gene_name": ["PRAME", "MAGEA1", "PRAME"],
            "gene_id": ["ENSG00000185686", "ENSG00000198681", "ENSG00000185686"],
            "position": [10, 20, 30],
            "n_flank": ["NNNNN", "NNNNN", "NNNNN"],
            "c_flank": ["CCCCC", "CCCCC", "CCCCC"],
            "proteome": ["Homo sapiens", "Homo sapiens", "Homo sapiens"],
            "proteome_source": ["species", "species", "species"],
        }
    )
    mappings_path = tmp_path / "peptide_mappings.parquet"
    mappings_data.to_parquet(mappings_path, index=False)

    # By gene_name (resolved via mappings sidecar)
    df = generate_observations_table(gene_name="PRAME")
    assert len(df) == 2
    assert set(df["peptide"]) == {"AAAAAAAAA", "CCCCCCCCC"}

    # By gene_id
    df = generate_observations_table(gene_id="ENSG00000198681")
    assert len(df) == 1
    assert df.iloc[0]["peptide"] == "BBBBBBBBB"

    # Via --gene with direct symbol match
    df = generate_observations_table(gene="MAGEA1")
    assert len(df) == 1

    # Comma-separated
    df = generate_observations_table(gene_name="PRAME,MAGEA1")
    assert len(df) == 3


def test_generate_observations_mhc_allele_filter(tmp_path, monkeypatch):
    """--mhc-allele should filter exact allele matches (after normalization)."""
    import pandas as pd

    from hitlist.export import generate_observations_table

    obs_data = pd.DataFrame(
        {
            "peptide": ["AAA", "BBB", "CCC"],
            "mhc_restriction": ["HLA-A*02:01", "HLA-B*07:02", "HLA-A*02:01"],
            "mhc_class": ["I", "I", "I"],
            "reference_iri": ["1", "2", "3"],
            "pmid": pd.array([33858848] * 3, dtype="Int64"),
            "source": ["iedb"] * 3,
            "mhc_species": ["Homo sapiens"] * 3,
            "is_monoallelic": [False] * 3,
            "is_binding_assay": [False] * 3,
            "qualitative_measurement": ["Positive"] * 3,
        }
    )
    obs_path = tmp_path / "observations.parquet"
    obs_data.to_parquet(obs_path, index=False)
    monkeypatch.setattr("hitlist.observations.observations_path", lambda: obs_path)

    df = generate_observations_table(mhc_allele="HLA-A*02:01")
    assert len(df) == 2
    assert set(df["peptide"]) == {"AAA", "CCC"}

    df = generate_observations_table(mhc_allele=["HLA-A*02:01", "HLA-B*07:02"])
    assert len(df) == 3


def test_generate_observations_serotype_filter(tmp_path, monkeypatch):
    """--serotype A24 matches rows with HLA-A24 in the serotypes column."""
    import pandas as pd

    from hitlist.export import generate_observations_table

    obs_data = pd.DataFrame(
        {
            "peptide": ["AAA", "BBB", "CCC", "DDD"],
            "mhc_restriction": ["HLA-A*24:02", "HLA-B*57:01", "HLA-A*02:01", "HLA-B*07:02"],
            "mhc_class": ["I"] * 4,
            "reference_iri": ["1", "2", "3", "4"],
            "pmid": pd.array([33858848] * 4, dtype="Int64"),
            "source": ["iedb"] * 4,
            "mhc_species": ["Homo sapiens"] * 4,
            "is_monoallelic": [False] * 4,
            "is_binding_assay": [False] * 4,
            "qualitative_measurement": ["Positive"] * 4,
            "serotype": ["HLA-A24", "HLA-B57", "HLA-A2", "HLA-B7"],
            "serotypes": ["HLA-A24;HLA-Bw4", "HLA-B57;HLA-B17;HLA-Bw4", "HLA-A2", "HLA-B7;HLA-Bw6"],
        }
    )
    obs_path = tmp_path / "observations.parquet"
    obs_data.to_parquet(obs_path, index=False)
    monkeypatch.setattr("hitlist.observations.observations_path", lambda: obs_path)

    # Locus-specific match
    df = generate_observations_table(serotype="A24")
    assert len(df) == 1
    assert df.iloc[0]["peptide"] == "AAA"

    # Public epitope matches multiple loci
    df = generate_observations_table(serotype="Bw4")
    assert set(df["peptide"]) == {"AAA", "BBB"}

    # Comma-separated, HLA- prefix optional
    df = generate_observations_table(serotype="HLA-A24,B57")
    assert set(df["peptide"]) == {"AAA", "BBB"}

    # No match
    df = generate_observations_table(serotype="A99")
    assert len(df) == 0


def test_generate_ms_observations_table_alias(monkeypatch):
    import pandas as pd

    captured = {}

    def fake_generate_observations_table(**kwargs):
        captured.update(kwargs)
        return pd.DataFrame({"peptide": ["AAA"]})

    monkeypatch.setattr(
        "hitlist.export.generate_observations_table", fake_generate_observations_table
    )

    df = generate_ms_observations_table(mhc_class="I", gene=["PRAME"])
    assert list(df["peptide"]) == ["AAA"]
    assert captured["mhc_class"] == "I"
    assert captured["gene"] == ["PRAME"]


def _make_binding_fixture(tmp_path):
    """Write a small binding.parquet fixture and return its path."""
    import pandas as pd

    bd = pd.DataFrame(
        {
            "peptide": ["PEP1", "PEP2", "PEP3", "PEP4"],
            "mhc_restriction": ["HLA-A*02:01", "HLA-A*02:01", "HLA-B*07:02", "HLA-A*24:02"],
            "mhc_class": ["I", "I", "I", "I"],
            "reference_iri": ["b1", "b2", "b3", "b4"],
            "pmid": pd.array([1, 2, 3, 4], dtype="Int64"),
            "source": ["iedb", "cedar", "iedb", "iedb"],
            "mhc_species": ["Homo sapiens"] * 4,
            "is_binding_assay": [True] * 4,
            "qualitative_measurement": [
                "Positive-High",
                "Positive-Intermediate",
                "Negative",
                "Positive-Low",
            ],
            "serotype": ["HLA-A2", "HLA-A2", "HLA-B7", "HLA-A24"],
            "serotypes": ["HLA-A2", "HLA-A2", "HLA-B7;HLA-Bw6", "HLA-A24;HLA-Bw4"],
        }
    )
    path = tmp_path / "binding.parquet"
    bd.to_parquet(path, index=False)
    return path


def test_generate_binding_table_returns_binding_rows(tmp_path, monkeypatch):
    """generate_binding_table loads from binding.parquet without MS joins."""
    from hitlist.export import generate_binding_table

    bd_path = _make_binding_fixture(tmp_path)
    monkeypatch.setattr("hitlist.observations.binding_path", lambda: bd_path)

    df = generate_binding_table()
    assert len(df) == 4
    assert df["is_binding_assay"].all()
    # No sample-metadata join: these MS-only columns should not appear
    for col in ("sample_label", "perturbation", "instrument", "instrument_type"):
        assert col not in df.columns


def test_generate_binding_table_mhc_and_serotype_filters(tmp_path, monkeypatch):
    from hitlist.export import generate_binding_table

    bd_path = _make_binding_fixture(tmp_path)
    monkeypatch.setattr("hitlist.observations.binding_path", lambda: bd_path)

    df = generate_binding_table(mhc_allele="HLA-A*02:01")
    assert set(df["peptide"]) == {"PEP1", "PEP2"}

    df = generate_binding_table(serotype="Bw4")
    assert set(df["peptide"]) == {"PEP4"}

    df = generate_binding_table(source="cedar")
    assert list(df["peptide"]) == ["PEP2"]


def _make_quant_binding_fixture(tmp_path):
    """Binding fixture with the new quantitative columns populated (#148, #135)."""
    import pandas as pd

    bd = pd.DataFrame(
        {
            "peptide": ["LOWIC50", "HIGHIC50", "LOGIC50", "QUALONLY"],
            "mhc_restriction": ["HLA-A*02:01"] * 4,
            "mhc_class": ["I"] * 4,
            "reference_iri": ["b1", "b2", "b3", "b4"],
            "pmid": pd.array([1, 2, 3, 4], dtype="Int64"),
            "source": ["iedb"] * 4,
            "mhc_species": ["Homo sapiens"] * 4,
            "is_binding_assay": [True] * 4,
            "qualitative_measurement": [
                "Positive-High",
                "Positive-Low",
                "Positive",
                "Positive",
            ],
            "assay_method": [
                "purified MHC/direct/fluorescence",
                "purified MHC/direct/fluorescence",
                "purified MHC/direct/fluorescence",
                "cellular MHC/direct",
            ],
            "response_measured": [
                "qualitative binding",
                "qualitative binding",
                "qualitative binding",
                "half life",
            ],
            "measurement_units": ["nM", "nM", "log10(nM)", ""],
            "measurement_inequality": ["=", "=", "=", ""],
            "quantitative_measurement": ["12.5", "5000", "1.1", ""],
            "quantitative_value": [12.5, 5000.0, 1.1, float("nan")],
            # Issue #137: bag-expansion columns.  LOWIC50 / HIGHIC50 / LOGIC50
            # are exact 4-digit; QUALONLY is a class-only row whose donor
            # genotype carries multiple alleles → sample_allele_match.
            "mhc_allele_set": [
                "HLA-A*02:01",
                "HLA-A*02:01",
                "HLA-A*02:01",
                "HLA-A*02:01;HLA-B*07:02",
            ],
            "mhc_allele_provenance": [
                "exact",
                "exact",
                "exact",
                "sample_allele_match",
            ],
            "mhc_allele_bag_size": [1, 1, 1, 2],
        }
    )
    path = tmp_path / "binding.parquet"
    bd.to_parquet(path, index=False)
    return path


def test_generate_binding_table_quantitative_value_min_max_filters(tmp_path, monkeypatch):
    from hitlist.export import generate_binding_table

    bd_path = _make_quant_binding_fixture(tmp_path)
    monkeypatch.setattr("hitlist.observations.binding_path", lambda: bd_path)

    # IC50 ≤ 100 nM — must not include the 5000 nM weak binder, the
    # log10-unit row (1.1), or the qualitative-only row (NaN).
    df = generate_binding_table(
        measurement_units="nM",
        quantitative_value_max=100.0,
    )
    assert set(df["peptide"]) == {"LOWIC50"}


def test_generate_binding_table_has_quantitative_value_flag(tmp_path, monkeypatch):
    from hitlist.export import generate_binding_table

    bd_path = _make_quant_binding_fixture(tmp_path)
    monkeypatch.setattr("hitlist.observations.binding_path", lambda: bd_path)

    quant_rows = generate_binding_table(has_quantitative_value=True)
    assert set(quant_rows["peptide"]) == {"LOWIC50", "HIGHIC50", "LOGIC50"}

    qual_rows = generate_binding_table(has_quantitative_value=False)
    assert set(qual_rows["peptide"]) == {"QUALONLY"}


def test_generate_binding_table_assay_method_substring_match(tmp_path, monkeypatch):
    from hitlist.export import generate_binding_table

    bd_path = _make_quant_binding_fixture(tmp_path)
    monkeypatch.setattr("hitlist.observations.binding_path", lambda: bd_path)

    # Case-insensitive substring on assay_method.
    df = generate_binding_table(assay_method="purified")
    assert set(df["peptide"]) == {"LOWIC50", "HIGHIC50", "LOGIC50"}

    df = generate_binding_table(assay_method="cellular")
    assert set(df["peptide"]) == {"QUALONLY"}


def test_generate_binding_table_allele_bag_filters(tmp_path, monkeypatch):
    """Issue #137: ``mhc_allele_in_bag`` matches both literal 4-digit rows
    and class-only rows whose expanded bag contains the allele.
    ``mhc_allele_provenance`` provides a strict-vs-MIL training filter."""
    from hitlist.export import generate_binding_table

    bd_path = _make_quant_binding_fixture(tmp_path)
    monkeypatch.setattr("hitlist.observations.binding_path", lambda: bd_path)

    # Filter to bag-contains HLA-A*02:01 — both exact rows AND the
    # sample_allele_match row (whose bag is "HLA-A*02:01;HLA-B*07:02") match.
    df = generate_binding_table(mhc_allele_in_bag="HLA-A*02:01")
    assert set(df["peptide"]) == {"LOWIC50", "HIGHIC50", "LOGIC50", "QUALONLY"}

    # Filter to bag-contains HLA-B*07:02 — only the sample_allele_match row.
    df = generate_binding_table(mhc_allele_in_bag="HLA-B*07:02")
    assert set(df["peptide"]) == {"QUALONLY"}

    # Strict-resolution training: only "exact" rows.
    df = generate_binding_table(mhc_allele_provenance="exact")
    assert set(df["peptide"]) == {"LOWIC50", "HIGHIC50", "LOGIC50"}

    # MIL-friendly: exact + sample_allele_match (small trusted bags).
    df = generate_binding_table(mhc_allele_provenance=["exact", "sample_allele_match"])
    assert set(df["peptide"]) == {"LOWIC50", "HIGHIC50", "LOGIC50", "QUALONLY"}


def test_generate_binding_table_response_measured_filter(tmp_path, monkeypatch):
    """Issue #135: ``response_measured`` filter discriminates IC50/Kd
    binding rows from kinetic / stability rows even when other axes
    (assay_method, units) overlap."""
    from hitlist.export import generate_binding_table

    bd_path = _make_quant_binding_fixture(tmp_path)
    monkeypatch.setattr("hitlist.observations.binding_path", lambda: bd_path)

    binders = generate_binding_table(response_measured="qualitative binding")
    assert set(binders["peptide"]) == {"LOWIC50", "HIGHIC50", "LOGIC50"}

    kinetics = generate_binding_table(response_measured="half life")
    assert set(kinetics["peptide"]) == {"QUALONLY"}

    # Multi-value (case-insensitive) filter.
    multi = generate_binding_table(response_measured=["Qualitative Binding", "HALF LIFE"])
    assert len(multi) == 4


def test_generate_binding_table_units_filter_prevents_mixing(tmp_path, monkeypatch):
    """Selecting ``measurement_units="nM"`` must drop the log10(nM) row
    even when its numeric value happens to be in-bounds — the whole
    point of the ``measurement_units`` filter is to prevent cross-unit
    comparisons.
    """
    from hitlist.export import generate_binding_table

    bd_path = _make_quant_binding_fixture(tmp_path)
    monkeypatch.setattr("hitlist.observations.binding_path", lambda: bd_path)

    df = generate_binding_table(measurement_units="nM")
    assert set(df["peptide"]) == {"LOWIC50", "HIGHIC50"}


def test_generate_training_table_unifies_ms_and_binding(tmp_path, monkeypatch):
    import pandas as pd

    from hitlist.export import generate_training_table

    obs_data = pd.DataFrame(
        {
            "peptide": ["AAAAAAAAA", "BBBBBBBBB"],
            "mhc_restriction": ["HLA-A*02:01", ""],
            "mhc_class": ["I", "I"],
            "reference_iri": ["iri:1", "iri:2"],
            "pmid": pd.array([33858848, 38480730], dtype="Int64"),
            "source": ["iedb", "supplement"],
            "mhc_species": ["Homo sapiens", "Homo sapiens"],
            "is_monoallelic": [False, False],
            "is_binding_assay": [False, False],
            "qualitative_measurement": ["Positive", ""],
        }
    )
    obs_path = tmp_path / "observations.parquet"
    obs_data.to_parquet(obs_path, index=False)
    monkeypatch.setattr("hitlist.observations.observations_path", lambda: obs_path)

    bd_path = _make_binding_fixture(tmp_path)
    monkeypatch.setattr("hitlist.observations.binding_path", lambda: bd_path)

    df = generate_training_table(include_evidence="both")
    assert set(df["evidence_kind"]) == {"ms", "binding"}
    assert "evidence_row_id" in df.columns

    ms = df[df["evidence_kind"] == "ms"]
    binding = df[df["evidence_kind"] == "binding"]

    assert len(ms) == 2
    assert len(binding) == 4
    assert ms["sample_match_type"].ne("not_applicable").any()
    assert (binding["sample_match_type"] == "not_applicable").all()
    assert (binding["matched_sample_count"] == 0).all()
    assert (binding["sample_mhc"] == "").all()


def test_generate_training_table_threads_allele_bag_filters(tmp_path, monkeypatch):
    """Issue #136 + #137: bag columns survive concat and filters reach both
    sides of the union (ms + binding).  Without this wiring, MIL-style
    training filters (``provenance=["exact","sample_allele_match"]``) only
    work via ``generate_observations_table`` / ``generate_binding_table``
    and silently no-op on ``generate_training_table``.
    """
    import pandas as pd

    from hitlist.export import generate_training_table

    # MS fixture: 2 exact rows, 1 sample_allele_match (multi-allelic).
    obs_data = pd.DataFrame(
        {
            "peptide": ["MS_EXACT1", "MS_EXACT2", "MS_BAG"],
            "mhc_restriction": ["HLA-A*02:01", "HLA-A*02:01", "HLA-A*02:01"],
            "mhc_class": ["I", "I", "I"],
            "assay_iri": ["a:1", "a:2", "a:3"],
            "reference_iri": ["iri:1", "iri:2", "iri:3"],
            "pmid": pd.array([1, 2, 3], dtype="Int64"),
            "source": ["iedb", "iedb", "iedb"],
            "mhc_species": ["Homo sapiens"] * 3,
            "is_monoallelic": [False, False, False],
            "is_binding_assay": [False, False, False],
            "qualitative_measurement": ["Positive"] * 3,
            "mhc_allele_set": [
                "HLA-A*02:01",
                "HLA-A*02:01",
                "HLA-A*02:01;HLA-B*07:02",
            ],
            "mhc_allele_provenance": ["exact", "exact", "sample_allele_match"],
            "mhc_allele_bag_size": [1, 1, 2],
        }
    )
    obs_path = tmp_path / "observations.parquet"
    obs_data.to_parquet(obs_path, index=False)
    monkeypatch.setattr("hitlist.observations.observations_path", lambda: obs_path)

    # Binding fixture: 3 exact rows + 1 sample_allele_match (matches the
    # _make_quant_binding_fixture shape but inlined for clarity).
    bd_data = pd.DataFrame(
        {
            "peptide": ["BD_EXACT", "BD_EXACT2", "BD_EXACT3", "BD_BAG"],
            "mhc_restriction": ["HLA-A*02:01"] * 4,
            "mhc_class": ["I"] * 4,
            "reference_iri": ["b1", "b2", "b3", "b4"],
            "pmid": pd.array([10, 11, 12, 13], dtype="Int64"),
            "source": ["iedb"] * 4,
            "mhc_species": ["Homo sapiens"] * 4,
            "is_binding_assay": [True] * 4,
            "qualitative_measurement": ["Positive"] * 4,
            "mhc_allele_set": [
                "HLA-A*02:01",
                "HLA-A*02:01",
                "HLA-A*02:01",
                "HLA-A*02:01;HLA-B*07:02",
            ],
            "mhc_allele_provenance": [
                "exact",
                "exact",
                "exact",
                "sample_allele_match",
            ],
            "mhc_allele_bag_size": [1, 1, 1, 2],
        }
    )
    bd_path = tmp_path / "binding.parquet"
    bd_data.to_parquet(bd_path, index=False)
    monkeypatch.setattr("hitlist.observations.binding_path", lambda: bd_path)

    # No filter — all 7 rows survive and bag columns are present on both sides.
    df = generate_training_table(include_evidence="both")
    assert set(df["evidence_kind"]) == {"ms", "binding"}
    for col in ("mhc_allele_set", "mhc_allele_provenance", "mhc_allele_bag_size"):
        assert col in df.columns, f"bag column {col} dropped by training export"
    assert df["mhc_allele_provenance"].notna().all()

    # Strict point-label training: only "exact" rows from both sides.
    df_exact = generate_training_table(include_evidence="both", mhc_allele_provenance="exact")
    assert set(df_exact["peptide"]) == {
        "MS_EXACT1",
        "MS_EXACT2",
        "BD_EXACT",
        "BD_EXACT2",
        "BD_EXACT3",
    }

    # MIL-friendly: exact + sample_allele_match.
    df_mil = generate_training_table(
        include_evidence="both",
        mhc_allele_provenance=["exact", "sample_allele_match"],
    )
    assert len(df_mil) == 7  # all rows

    # Bag-contains: rows whose mhc_allele_set lists HLA-B*07:02 (the two
    # multi-allelic donor-typed rows, one ms + one binding).
    df_b7 = generate_training_table(include_evidence="both", mhc_allele_in_bag="HLA-B*07:02")
    assert set(df_b7["peptide"]) == {"MS_BAG", "BD_BAG"}


def test_evidence_row_id_prefers_assay_iri_when_present(tmp_path, monkeypatch):
    """assay_iri is row-level in IEDB/CEDAR (Reference IRI is paper-level).

    Per issue #146, multiple rows from the same paper must receive distinct
    evidence_row_ids when assay_iri is populated on the observations parquet.
    This is the regression proving ``evidence_row_id`` is now row-unique,
    not paper-unique, for IEDB/CEDAR-sourced rows.
    """
    import pandas as pd

    from hitlist.export import generate_training_table

    # Three distinct assay rows from the SAME paper (same reference_iri).
    obs_data = pd.DataFrame(
        {
            "peptide": ["AAAAAAAAA", "BBBBBBBBB", "CCCCCCCCC"],
            "mhc_restriction": ["HLA-A*02:01", "HLA-A*02:01", "HLA-A*02:01"],
            "mhc_class": ["I", "I", "I"],
            "assay_iri": [
                "http://iedb.org/assay/1000001",
                "http://iedb.org/assay/1000002",
                "http://iedb.org/assay/1000003",
            ],
            "reference_iri": ["http://iedb.org/reference/42"] * 3,
            "pmid": pd.array([33858848, 33858848, 33858848], dtype="Int64"),
            "source": ["iedb"] * 3,
            "mhc_species": ["Homo sapiens"] * 3,
            "is_monoallelic": [False] * 3,
            "is_binding_assay": [False] * 3,
            "qualitative_measurement": ["Positive"] * 3,
        }
    )
    obs_path = tmp_path / "observations.parquet"
    obs_data.to_parquet(obs_path, index=False)
    monkeypatch.setattr("hitlist.observations.observations_path", lambda: obs_path)

    df = generate_training_table(include_evidence="ms")
    assert df["evidence_row_id"].nunique() == 3, (
        f"expected 3 unique evidence_row_ids (one per assay_iri), got "
        f"{df['evidence_row_id'].nunique()} — reference_iri is paper-level "
        f"and should not drive evidence identity."
    )
    # All three should carry the assay IRI in the id.
    assert all(df["evidence_row_id"].str.contains("assay/"))


def test_evidence_row_id_falls_back_to_reference_iri_for_older_parquets(tmp_path, monkeypatch):
    """Parquets built before #146 don't have an assay_iri column.

    The export must fall back to reference_iri so downstream consumers
    don't crash on older indexes.  Value stability is lost (because the
    old parquet has no row-level identifier), but the contract of
    "evidence_row_id is always set and always prefixed by evidence_kind"
    must hold.
    """
    import pandas as pd

    from hitlist.export import generate_training_table

    obs_data = pd.DataFrame(
        {
            "peptide": ["AAAAAAAAA", "BBBBBBBBB"],
            "mhc_restriction": ["HLA-A*02:01", "HLA-A*02:01"],
            "mhc_class": ["I", "I"],
            # No ``assay_iri`` column — simulates a v1.15-era parquet.
            "reference_iri": ["iri:1", "iri:2"],
            "pmid": pd.array([33858848, 33858848], dtype="Int64"),
            "source": ["iedb", "iedb"],
            "mhc_species": ["Homo sapiens", "Homo sapiens"],
            "is_monoallelic": [False, False],
            "is_binding_assay": [False, False],
            "qualitative_measurement": ["Positive", "Positive"],
        }
    )
    obs_path = tmp_path / "observations.parquet"
    obs_data.to_parquet(obs_path, index=False)
    monkeypatch.setattr("hitlist.observations.observations_path", lambda: obs_path)

    df = generate_training_table(include_evidence="ms")
    assert df["evidence_row_id"].nunique() == 2
    assert set(df["evidence_row_id"]) == {"ms:iri:1", "ms:iri:2"}


def test_evidence_row_id_positional_fallback_for_missing_both(tmp_path, monkeypatch):
    """When neither assay_iri nor reference_iri is populated, a positional
    ``{kind}:row:{idx}`` sentinel is emitted — the contract stays that
    evidence_row_id is always non-empty.
    """
    import pandas as pd

    from hitlist.export import generate_training_table

    obs_data = pd.DataFrame(
        {
            "peptide": ["AAAAAAAAA"],
            "mhc_restriction": ["HLA-A*02:01"],
            "mhc_class": ["I"],
            "assay_iri": [""],
            "reference_iri": [""],
            "pmid": pd.array([33858848], dtype="Int64"),
            "source": ["iedb"],
            "mhc_species": ["Homo sapiens"],
            "is_monoallelic": [False],
            "is_binding_assay": [False],
            "qualitative_measurement": ["Positive"],
        }
    )
    obs_path = tmp_path / "observations.parquet"
    obs_data.to_parquet(obs_path, index=False)
    monkeypatch.setattr("hitlist.observations.observations_path", lambda: obs_path)

    df = generate_training_table(include_evidence="ms")
    assert df["evidence_row_id"].iloc[0].startswith("ms:row:")


def test_generate_training_table_explodes_mappings(tmp_path, monkeypatch):
    import pandas as pd

    from hitlist import downloads
    from hitlist.export import generate_training_table

    monkeypatch.setattr(downloads, "_override_data_dir", tmp_path)

    obs_data = pd.DataFrame(
        {
            "peptide": ["AAAAAAAAA", "BBBBBBBBB"],
            "mhc_restriction": ["HLA-A*02:01", "HLA-B*07:02"],
            "mhc_class": ["I", "I"],
            "reference_iri": ["iri:1", "iri:2"],
            "pmid": pd.array([33858848, 33858848], dtype="Int64"),
            "source": ["iedb", "iedb"],
            "mhc_species": ["Homo sapiens", "Homo sapiens"],
            "is_monoallelic": [False, False],
            "is_binding_assay": [False, False],
            "qualitative_measurement": ["Positive", "Positive"],
            "gene_names": ["PRAME", "MAGEA1"],
            "gene_ids": ["ENSG00000185686", "ENSG00000198681"],
            "protein_ids": ["P1;P2", "P3"],
        }
    )
    obs_path = tmp_path / "observations.parquet"
    obs_data.to_parquet(obs_path, index=False)
    monkeypatch.setattr("hitlist.observations.observations_path", lambda: obs_path)

    mappings_data = pd.DataFrame(
        {
            "peptide": ["AAAAAAAAA", "AAAAAAAAA", "BBBBBBBBB"],
            "protein_id": ["P1", "P2", "P3"],
            "gene_name": ["PRAME", "PRAME", "MAGEA1"],
            "gene_id": ["ENSG00000185686", "ENSG00000185686", "ENSG00000198681"],
            # Issue #141 added these as first-class mapping columns; the
            # synthetic fixture has to declare them for the training-export
            # column-projection request to succeed against pyarrow.
            "transcript_id": ["ENST_T1", "ENST_T2", "ENST_T3"],
            "is_canonical_transcript": [True, False, True],
            "position": [10, 25, 7],
            "n_flank": ["NNNNN", "QQQQQ", "MMMMM"],
            "c_flank": ["CCCCC", "RRRRR", "TTTTT"],
            "proteome": ["Homo sapiens", "Homo sapiens", "Homo sapiens"],
            "proteome_source": ["species", "species", "species"],
        }
    )
    mappings_path = tmp_path / "peptide_mappings.parquet"
    mappings_data.to_parquet(mappings_path, index=False)

    df = generate_training_table(include_evidence="ms", map_source_proteins=True)
    assert len(df) == 3
    assert {"protein_id", "position", "n_flank", "c_flank"} <= set(df.columns)
    assert df["evidence_row_id"].nunique() == 2
    # transcript_id and is_canonical_transcript flow through to the export.
    assert "transcript_id" in df.columns
    assert "is_canonical_transcript" in df.columns

    aa = df[df["peptide"] == "AAAAAAAAA"]
    assert set(aa["protein_id"]) == {"P1", "P2"}
    assert set(aa["n_flank"]) == {"NNNNN", "QQQQQ"}

    # Deprecated alias still works but emits DeprecationWarning.  Same input
    # parquets, same expected expansion.
    import warnings

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        df_alias = generate_training_table(include_evidence="ms", explode_mappings=True)
    assert any(issubclass(rec.category, DeprecationWarning) for rec in w)
    assert len(df_alias) == len(df)
    assert {"protein_id", "n_flank", "c_flank"} <= set(df_alias.columns)


def test_generate_training_table_exploded_mappings_respect_gene_filter(tmp_path, monkeypatch):
    import pandas as pd

    from hitlist import downloads
    from hitlist.export import generate_training_table

    monkeypatch.setattr(downloads, "_override_data_dir", tmp_path)

    obs_data = pd.DataFrame(
        {
            "peptide": ["SHAREDPEP"],
            "mhc_restriction": ["HLA-A*02:01"],
            "mhc_class": ["I"],
            "reference_iri": ["iri:1"],
            "pmid": pd.array([33858848], dtype="Int64"),
            "source": ["iedb"],
            "mhc_species": ["Homo sapiens"],
            "is_monoallelic": [False],
            "is_binding_assay": [False],
            "qualitative_measurement": ["Positive"],
            "gene_names": ["PRAME;MAGEA1"],
            "gene_ids": ["ENSG1;ENSG2"],
            "protein_ids": ["P1;P2"],
            "n_source_proteins": [2],
        }
    )
    obs_path = tmp_path / "observations.parquet"
    obs_data.to_parquet(obs_path, index=False)
    monkeypatch.setattr("hitlist.observations.observations_path", lambda: obs_path)

    pd.DataFrame(columns=obs_data.columns).to_parquet(tmp_path / "binding.parquet", index=False)

    mappings_data = pd.DataFrame(
        {
            "peptide": ["SHAREDPEP", "SHAREDPEP"],
            "protein_id": ["P1", "P2"],
            "gene_name": ["PRAME", "MAGEA1"],
            "gene_id": ["ENSG1", "ENSG2"],
            # Issue #141: required schema columns.
            "transcript_id": ["ENST_T1", "ENST_T2"],
            "is_canonical_transcript": [True, True],
            "position": [10, 20],
            "n_flank": ["NN", "MM"],
            "c_flank": ["CC", "TT"],
            "proteome": ["Homo sapiens", "Homo sapiens"],
            "proteome_source": ["species", "species"],
        }
    )
    mappings_data.to_parquet(tmp_path / "peptide_mappings.parquet", index=False)

    df = generate_training_table(
        include_evidence="ms",
        gene_name="PRAME",
        map_source_proteins=True,
    )
    assert len(df) == 1
    assert df.iloc[0]["gene_name"] == "PRAME"
    assert df.iloc[0]["protein_id"] == "P1"


def test_generate_training_table_binding_has_peptide_level_allele_uses_resolution(
    tmp_path, monkeypatch
):
    import pandas as pd

    from hitlist.export import generate_training_table

    binding_data = pd.DataFrame(
        {
            "peptide": ["P1", "P2", "P3"],
            "mhc_restriction": ["HLA-A*02:01", "HLA-A2", "HLA class I"],
            "mhc_class": ["I", "I", "I"],
            "reference_iri": ["b1", "b2", "b3"],
            "pmid": pd.array([1, 2, 3], dtype="Int64"),
            "source": ["iedb", "iedb", "iedb"],
            "mhc_species": ["Homo sapiens"] * 3,
            "is_binding_assay": [True] * 3,
            "qualitative_measurement": ["Positive", "Positive", "Positive"],
            "allele_resolution": ["four_digit", "serological", "class_only"],
        }
    )
    binding_path = tmp_path / "binding.parquet"
    binding_data.to_parquet(binding_path, index=False)
    monkeypatch.setattr("hitlist.observations.binding_path", lambda: binding_path)

    df = generate_training_table(include_evidence="binding")
    by_peptide = df.set_index("peptide")["has_peptide_level_allele"].to_dict()
    assert by_peptide == {"P1": True, "P2": False, "P3": False}


def test_generate_training_table_rejects_invalid_evidence_mode():
    import pytest

    from hitlist.export import generate_training_table

    with pytest.raises(ValueError, match="include_evidence"):
        generate_training_table(include_evidence="oops")


def test_generate_training_table_projection_preserves_evidence_identity(tmp_path, monkeypatch):
    import pandas as pd

    from hitlist.export import generate_training_table

    obs_data = pd.DataFrame(
        {
            "peptide": ["PEP"],
            "mhc_restriction": ["HLA-A*02:01"],
            "mhc_class": ["I"],
            "reference_iri": ["iri:1"],
            "pmid": pd.array([33858848], dtype="Int64"),
            "source": ["iedb"],
            "mhc_species": ["Homo sapiens"],
            "is_monoallelic": [False],
            "is_binding_assay": [False],
            "qualitative_measurement": ["Positive"],
        }
    )
    obs_path = tmp_path / "observations.parquet"
    obs_data.to_parquet(obs_path, index=False)
    monkeypatch.setattr("hitlist.observations.observations_path", lambda: obs_path)

    pd.DataFrame(columns=obs_data.columns).to_parquet(tmp_path / "binding.parquet", index=False)

    df = generate_training_table(include_evidence="ms", columns=["peptide"])
    assert df.columns.tolist() == ["peptide", "evidence_kind", "evidence_row_id"]


def test_export_training_cli_helper(monkeypatch):
    import argparse

    import pandas as pd

    from hitlist.cli import _export_training

    captured = {}

    def fake_generate_training_table(**kwargs):
        captured.update(kwargs)
        return pd.DataFrame({"evidence_kind": ["ms"], "peptide": ["AAAAAAAAA"]})

    monkeypatch.setattr("hitlist.export.generate_training_table", fake_generate_training_table)

    args = argparse.Namespace(
        include_evidence="both",
        mhc_class="I",
        species="Homo sapiens",
        source="iedb",
        instrument_type="Orbitrap",
        acquisition_mode="DDA",
        mono_allelic=True,
        min_allele_resolution="four_digit",
        mhc_allele=["HLA-A*02:01"],
        mhc_allele_in_bag=["HLA-A*02:01"],
        mhc_allele_provenance=["exact"],
        gene=["PRAME"],
        gene_name=["PRAME"],
        gene_id=["ENSG00000185686"],
        peptide=["AAAAAAAAA"],
        serotype=["A2"],
        length_min=8,
        length_max=11,
        map_source_proteins=True,
        with_peptide_origin=False,
        proteome_release=112,
    )

    df = _export_training(args)
    assert list(df["evidence_kind"]) == ["ms"]
    assert captured == {
        "include_evidence": "both",
        "mhc_class": "I",
        "species": "Homo sapiens",
        "source": "iedb",
        "instrument_type": "Orbitrap",
        "acquisition_mode": "DDA",
        "is_mono_allelic": True,
        "min_allele_resolution": "four_digit",
        "mhc_allele": ["HLA-A*02:01"],
        "mhc_allele_in_bag": ["HLA-A*02:01"],
        "mhc_allele_provenance": ["exact"],
        "gene": ["PRAME"],
        "gene_name": ["PRAME"],
        "gene_id": ["ENSG00000185686"],
        "peptide": ["AAAAAAAAA"],
        "serotype": ["A2"],
        "length_min": 8,
        "length_max": 11,
        "map_source_proteins": True,
        "with_peptide_origin": False,
        "proteome_release": 112,
    }


def test_to_list_flattens_all_three_input_shapes():
    """List filters must accept repeated, space-separated, and
    comma-separated input identically.  Pre-fix, comma-separated values
    inside list elements (from ``--gene NRAS,KRAS``) were left as a single
    string ``"NRAS,KRAS"`` and silently failed to match anything.
    """
    from hitlist.export import _to_list

    # Bare string with commas (legacy single-flag form).
    assert _to_list("NRAS,KRAS") == ["NRAS", "KRAS"]
    assert _to_list("  NRAS , KRAS ") == ["NRAS", "KRAS"]
    # List of single tokens (repeated --flag X --flag Y → ["X","Y"]
    # or space-separated nargs="+" --flag X Y → ["X","Y"]).
    assert _to_list(["NRAS", "KRAS"]) == ["NRAS", "KRAS"]
    # List with comma-bearing elements (mixed: --flag NRAS,KRAS HRAS).
    assert _to_list(["NRAS,KRAS", "HRAS"]) == ["NRAS", "KRAS", "HRAS"]
    # Empties drop, whitespace strips.
    assert _to_list(["", "NRAS", " "]) == ["NRAS"]
    assert _to_list("") == []


def test_cli_list_args_accept_space_and_comma_and_repeated_forms():
    """argparse must produce the same flat list for all three input shapes
    so users don't need to remember which form a given flag accepts.
    """
    # No public CLI parser-builder helper, so reproduce the action/nargs
    # combination the real flags use and exercise argparse directly.
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--gene", action="extend", nargs="+")
    p.add_argument("--mhc-allele", action="extend", nargs="+")

    # Repeated.
    a = p.parse_args(["--gene", "NRAS", "--gene", "KRAS"])
    assert a.gene == ["NRAS", "KRAS"]
    # Space-separated (the case that used to fail).
    a = p.parse_args(["--gene", "NRAS", "KRAS"])
    assert a.gene == ["NRAS", "KRAS"]
    # Multiple flags interleaved with space-separated values.
    a = p.parse_args(["--mhc-allele", "HLA-A*02:01", "HLA-B*07:02", "--gene", "NRAS"])
    assert a.mhc_allele == ["HLA-A*02:01", "HLA-B*07:02"]
    assert a.gene == ["NRAS"]


# ── Sample resolver tie-break (cell_name / assay-comments) ────────────────


def test_select_best_candidate_prefers_unique_cell_name_token():
    """When two samples share an allele but differ in cell type, the
    obs row's ``cell_name`` should select the matching sample.
    """
    from hitlist.export import _select_best_candidate

    candidates = [
        ("C57BL/6 hepatocytes (AAV-HC-Kd)", "", {"sample_label": "hepatocytes"}),
        ("spleen (178.3 and B6.Kd)", "", {"sample_label": "spleen"}),
    ]
    pick = _select_best_candidate(candidates, "Splenocyte", "", "", "")
    assert pick == {"sample_label": "spleen"}, (
        "Splenocyte obs row should match the spleen sample, not hepatocytes"
    )

    pick = _select_best_candidate(candidates, "Hepatocyte", "", "", "")
    assert pick == {"sample_label": "hepatocytes"}, (
        "Hepatocyte obs row should match the hepatocytes sample"
    )


def test_select_best_candidate_rarity_weights_unique_tokens():
    """When candidates share generic tokens but differ in identifying
    ones, the rarity weight should prefer the sample whose unique
    token (e.g. cell-line ID 4906) appears in the obs cell_name.
    """
    from hitlist.export import _select_best_candidate

    candidates = [
        ("DFT1 cell line 4906 + IFN-gamma", "", {"sample_label": "dft1"}),
        ("DFT2 cell line Red Velvet", "", {"sample_label": "dft2"}),
        ("devil fibroblasts (healthy host)", "", {"sample_label": "fb"}),
    ]
    # Both DFT samples share "cell" + "line" tokens; only DFT1 carries
    # 4906 — the cell_name's "4906" must outweigh the shared "cell".
    pick = _select_best_candidate(candidates, "4906-Glial cell", "", "", "")
    assert pick == {"sample_label": "dft1"}

    # Same logic in the other direction.
    pick = _select_best_candidate(candidates, "DFT2.RV-Glial cell", "", "", "")
    assert pick == {"sample_label": "dft2"}

    # Fibroblast-token case: only the fibroblasts candidate has the
    # "fibroblast" stem.
    pick = _select_best_candidate(candidates, "Fibroblast", "", "", "")
    assert pick == {"sample_label": "fb"}


def test_select_best_candidate_word_boundary_substring():
    """The substring tie-break must respect word boundaries — "treated"
    appearing as a suffix of "untreated" must not falsely pick the
    untreated candidate when the obs row says "IFNg-treated".
    """
    from hitlist.export import _select_best_candidate

    candidates = [
        ("MDA-MB-231 (untreated)", "", {"label": "untreated"}),
        (
            "MDA-MB-231 + IFN-gamma (class I)",
            "IFN-gamma 50 IU/mL 48h",
            {"label": "ifn"},
        ),
    ]
    pick = _select_best_candidate(
        candidates,
        cell_name="",
        source_tissue="",
        antigen_processing_comments="",
        assay_comments="The peptide was eluted from the following experimental conditions: IFNg-treated.",
    )
    assert pick == {"label": "ifn"}

    pick = _select_best_candidate(
        candidates,
        cell_name="",
        source_tissue="",
        antigen_processing_comments="",
        assay_comments="The peptide was eluted from the following experimental conditions: untreated.",
    )
    assert pick == {"label": "untreated"}


def test_select_best_candidate_empty_obs_returns_none():
    """No metadata to score against → no winner; caller falls back."""
    from hitlist.export import _select_best_candidate

    candidates = [
        ("foo sample", "", {"x": 1}),
        ("bar sample", "", {"x": 2}),
    ]
    assert _select_best_candidate(candidates, "", "", "", "") is None


def test_select_best_candidate_single_candidate_returns_none_without_overlap():
    """Single candidate with no overlap → None (caller may still
    propagate single-candidate meta from a higher level).
    """
    from hitlist.export import _select_best_candidate

    pick = _select_best_candidate([("foo", "", {"x": 1})], "unrelated", "", "", "")
    assert pick is None


def test_resolver_normalizes_mouse_alleles(full_observations_df):
    """Mouse alleles like ``H-2Kb`` written by curators in YAML must
    join against IEDB's canonical form ``H2-K*b`` after mhcgnomes
    normalization. Regression for the silent allele-bag mismatch on
    PMID 34428180 (Son 2021 transplant tolerance) — pre-fix, all
    ~70K rows fell through to ``pmid_class_pool`` because the YAML
    token never matched.
    """
    df = full_observations_df
    sub = df[df["pmid"] == 34428180]
    if sub.empty:
        import pytest

        pytest.skip("Son 2021 not present in this build")
    # The bulk of rows should now allele-match; if normalization
    # regressed, every row would land in ``pmid_class_pool`` again.
    n_allele = (sub["sample_match_type"] == "allele_match").sum()
    assert n_allele > 0.5 * len(sub), (
        f"expected most Son 2021 rows to allele-match after H2 normalization; "
        f"got {n_allele} / {len(sub)}"
    )


def test_resolver_disambiguates_mouse_hepatocyte_vs_spleen(full_observations_df):
    """Multiple mouse samples in PMID 34428180 share H-2Kd. The
    resolver must split splenocyte rows from hepatocyte rows by
    matching obs ``cell_name`` against each sample's label.
    """
    df = full_observations_df
    sub = df[df["pmid"] == 34428180]
    if sub.empty:
        import pytest

        pytest.skip("Son 2021 not present in this build")
    spleen_rows = sub[sub["cell_name"].str.contains("Splenocyte", na=False)]
    if spleen_rows.empty:
        import pytest

        pytest.skip("no splenocyte rows in this build")
    # The spleen ms_sample carries the word "spleen" in its sample_label;
    # the resolver should pick that over the hepatocyte sample for at
    # least the bulk of splenocyte rows.
    pick_spleen = spleen_rows["sample_label"].str.contains("spleen", case=False, na=False).sum()
    assert pick_spleen > 0.4 * len(spleen_rows), (
        f"expected majority of splenocyte rows to resolve to a 'spleen'-named "
        f"ms_sample after the cell_name tie-break; got {pick_spleen} / {len(spleen_rows)}"
    )


def test_resolver_alpizar_apc_substring_splits_b40_b39(full_observations_df):
    """Alpizar 2017 (PMID 27920218) has 515 IEDB rows tagged "HLA class I"
    on C1R cells whose ``antigen_processing_comments`` mention either
    ``HLA-B*40:02 (C1R-B*40)`` or ``HLA-B*39:01 (C1R-B*39)``. The
    resolver's allele-substring tie-break should split them onto the
    matching C1R-HLA-B transfectant. Issue #207.
    """
    df = full_observations_df
    sub = df[(df["pmid"] == 27920218) & (df["mhc_restriction"] == "HLA class I")]
    if sub.empty:
        import pytest

        pytest.skip("Alpizar 2017 not present in this build")
    # The exact split depends on per-row APC counts (~408 + 99 + a few
    # pooled), but at minimum the resolver should attribute most rows
    # to one of the two single-allele C1R-HLA-B samples (not a class-
    # only blank).
    n_resolved = sub["sample_label"].str.contains("C1R-HLA-B", case=False, na=False).sum()
    assert n_resolved > 0.9 * len(sub), (
        f"expected most of the 515 unattributed Alpizar rows to resolve "
        f"to a C1R-HLA-B sample via APC substring match; got {n_resolved}/{len(sub)}"
    )
    # Most rows mention B*40:02 in the APC; sanity check that neither
    # pole gets every row (the APC really does discriminate).
    n_b40 = sub["sample_label"].str.contains(r"B\*40:02", case=False, na=False).sum()
    n_b39 = sub["sample_label"].str.contains(r"B\*39:01", case=False, na=False).sum()
    assert n_b40 > 0 and n_b39 > 0, (
        f"expected both B*40:02 and B*39:01 to win for some rows; got n_b40={n_b40}, n_b39={n_b39}"
    )


def test_select_best_candidate_rejects_short_family_root_match():
    """``tap1`` and ``tapasin`` share the family root ``tap`` — but
    that's a 3-char overlap on neither-side-complete prefixes, which
    must NOT count as a match. Otherwise Shapiro 2025's TAPBP sample
    (with "tapasin" in its perturbation field) wins TAP1-KO obs rows
    via the spurious "tap" overlap.
    """
    from hitlist.export import _select_best_candidate

    candidates = [
        ("HAP1 TAP1 KO", "TAP1 CRISPR/Cas9 knockout", {"label": "tap1"}),
        ("HAP1 TAPBP KO", "TAPBP CRISPR/Cas9 knockout (tapasin)", {"label": "tapbp"}),
        ("HAP1 TAP2 KO", "TAP2 CRISPR/Cas9 knockout", {"label": "tap2"}),
    ]
    pick = _select_best_candidate(candidates, "HAP1 TAP1 KO", "", "", "")
    assert pick == {"label": "tap1"}, (
        f"expected TAP1 KO sample for TAP1 obs (full token match), got {pick}"
    )

    pick = _select_best_candidate(candidates, "HAP1 TAPBP KO", "", "", "")
    assert pick == {"label": "tapbp"}


def test_resolver_shapiro_2025_routes_per_ko_sample(full_observations_df):
    """Shapiro 2025 (PMID 40113210) — 12 HAP1 samples sharing one
    allele bag (A*02:01 + B*40:01 + C*03:04). The supplementary CSVs
    set ``cell_name`` to the KO label (e.g. "HAP1 B2M KO"); the
    resolver must route every CSV's peptides to its matching
    ms_sample. Issue #54.
    """
    df = full_observations_df
    sub = df[df["pmid"] == 40113210]
    if sub.empty:
        import pytest

        pytest.skip("Shapiro 2025 not present in this build")
    mismatches = (
        sub[sub["cell_name"].notna() & (sub["cell_name"] != "")]
        .groupby(["cell_name", "sample_label"])
        .size()
        .reset_index(name="n")
    )
    bad = mismatches[mismatches["cell_name"] != mismatches["sample_label"]]
    assert bad.empty, (
        f"Shapiro per-KO supplementary CSVs should route to the "
        f"matching sample_label; got {len(bad)} mismatches"
    )


def test_resolver_skips_apc_when_cell_name_varies(full_observations_df):
    """Liepe 2016 (PMID 27846572) class-I obs have varying ``cell_name``
    (B cell / Fibroblast / C1R cells-B cell / T2). Some assay_comments
    text mentions "GR lymphoblastoid cell line" referring to the
    GR-LCL sample. Without the cell_name-varies guard, the C1R
    parental sample's "lymphoblastoid" token would falsely win those
    rows via APC narrative. Verify B cell rows stay unassigned (no
    GR-LCL/JY-specific identifier in the obs metadata to discriminate
    those two samples) — and that the C1R-cell-name rows correctly
    resolve to the C1R parental sample.
    """
    df = full_observations_df
    sub = df[df["pmid"] == 27846572]
    if sub.empty:
        import pytest

        pytest.skip("Liepe 2016 not present in this build")
    # B cell rows must NOT be assigned to C1R parental (GR-LCL / JY
    # are the two B-LCL candidates and neither has a unique-and-matching
    # token, so the right answer is to leave them unassigned).
    b_cell = sub[sub["cell_name"] == "B cell"]
    if not b_cell.empty:
        c1r_b_cell = b_cell["sample_label"].str.contains("C1R", na=False).sum()
        assert c1r_b_cell == 0, (
            f"Liepe 'B cell' rows should not be assigned to a C1R sample "
            f"(those are GR-LCL/JY rows); got {c1r_b_cell}/{len(b_cell)} "
            f"misassigned to C1R parental"
        )
    # C1R cells-B cell rows MUST resolve to the C1R parental sample.
    c1r_rows = sub[sub["cell_name"] == "C1R cells-B cell"]
    if not c1r_rows.empty:
        c1r_to_c1r = c1r_rows["sample_label"].str.contains("C1R", na=False).sum()
        assert c1r_to_c1r > 0.95 * len(c1r_rows), (
            f"Liepe 'C1R cells-B cell' rows should resolve to C1R parental; "
            f"got {c1r_to_c1r}/{len(c1r_rows)}"
        )
