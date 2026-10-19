import pandas as pd
import pytest

from hitlist import cta_expression as module


@pytest.fixture
def reference(monkeypatch):
    frame = pd.DataFrame(
        {"Ensembl_Gene_ID": ["G1", "G2", "G3"], "Symbol": ["PRAME", "MAGEA3", "MAGEA4"]}
    )
    monkeypatch.setattr(module, "_cta_reference", lambda definition: (frame, {"G1", "G2", "G3"}))
    identities = {
        "PRAME": ("G1", "PRAME"),
        "MAGEA3": ("G2", "MAGEA3"),
        "MAGEA4": ("G3", "MAGEA4"),
        "G1": ("G1", "PRAME"),
        "ACTB": ("G4", "ACTB"),
    }
    monkeypatch.setattr(module, "_gene_identity", lambda name: identities.get(name, ("", "")))
    monkeypatch.setattr(
        module,
        "_transcript_identity",
        lambda name, release: ("G1", "PRAME") if name in {"ENST1", "ENST2"} else ("", ""),
    )
    monkeypatch.setattr(module, "_transcript_reference", lambda release: {"release": release})
    return frame


def test_gene_table_keeps_all_exclusion_reasons_and_mage_exception(reference, tmp_path):
    path = tmp_path / "expression.tsv"
    path.write_text("gene\ttpm\nPRAME\t10\nMAGEA3\t20\nMAGEA4\t4\nACTB\t50\nUNKNOWN\t5\n")
    audit, snapshot, metadata = module.resolve_expression_table(
        path, id_column="gene", tpm_column="tpm"
    )
    assert audit.selection_reason.tolist() == [
        "selected",
        "excluded_gene_pattern",
        "selected",
        "not_cta",
        "unresolved_identifier",
    ]
    assert audit.input_row.tolist() == [1, 2, 3, 4, 5]
    assert set(snapshot.Ensembl_Gene_ID) == {"G1", "G2", "G3"}
    assert metadata["input"]["sha256"]


def test_transcripts_keep_distinct_expression_and_versioned_input(reference, tmp_path):
    path = tmp_path / "salmon.tsv"
    path.write_text("Name\tTPM\nENST1.4\t5\nENST2.7\t0.2\n")
    audit, _, _ = module.resolve_expression_table(
        path, id_column="Name", tpm_column="TPM", level="transcript"
    )
    assert audit.transcript_id.tolist() == ["ENST1", "ENST2"]
    assert audit.original_identifier.tolist() == ["ENST1.4", "ENST2.7"]
    assert audit.selection_reason.tolist() == ["selected", "below_tpm"]


@pytest.mark.parametrize("values", ["PRAME\t2\nG1\t3\n", "PRAME\t-1\n", "PRAME\tinf\n"])
def test_duplicate_resolutions_and_invalid_expression_fail(reference, tmp_path, values):
    path = tmp_path / "x.tsv"
    path.write_text("gene\ttpm\n" + values)
    with pytest.raises(ValueError):
        module.resolve_expression_table(path, id_column="gene", tpm_column="tpm")


def test_missing_tpm_does_not_mean_selected_or_zero(reference, tmp_path):
    path = tmp_path / "x.csv"
    path.write_text("gene,tpm\nPRAME,\n")
    audit, _, _ = module.resolve_expression_table(path, id_column="gene", tpm_column="tpm")
    assert audit.iloc[0].selection_reason == "missing_tpm"
    assert pd.isna(audit.iloc[0].tpm)


def test_specificity_uses_all_reference_genes_not_selected_gene_set(monkeypatch):
    monkeypatch.setattr(module, "_canonical_mapping_gene", lambda x: x)
    mappings = pd.DataFrame(
        [
            {
                "peptide": peptide,
                "gene_id": gene,
                "protein_id": protein,
                "transcript_id": transcript,
                "proteome": "Homo sapiens",
            }
            for peptide, gene, protein, transcript in [
                ("ONLYCTA", "G1", "P1", "T1"),
                ("SHAREDCTA", "G1", "P1", "T1"),
                ("SHAREDCTA", "G2", "P2", "T2"),
                ("AVOIDSELF", "G1", "P1", "T1"),
                ("AVOIDSELF", "G4", "P4", "T4"),
                ("UNKNOWN", "G1", "P1", "T1"),
                ("UNKNOWN", "", "PX", "TX"),
            ]
        ]
    )
    result, annotated = module.summarize_cta_mappings(mappings, {"G1", "G2"})
    result = result.set_index("peptide")
    assert result.loc["ONLYCTA", "cta_specific"]
    assert result.loc["SHAREDCTA", "shared_between_ctas"]
    assert result.loc["SHAREDCTA", "cta_specific"]
    assert result.loc["AVOIDSELF", "has_non_cta_match"]
    assert not result.loc["AVOIDSELF", "cta_specific"]
    assert not result.loc["UNKNOWN", "cta_specific"]
    assert len(annotated) == len(mappings)


def test_duplicate_column_names_are_not_silently_renamed(reference, tmp_path):
    path = tmp_path / "ambiguous.tsv"
    path.write_text("gene\ttpm\ttpm\nPRAME\t10\t0\n")
    with pytest.raises(ValueError, match="unique"):
        module.resolve_expression_table(path, id_column="gene", tpm_column="tpm")


def test_transcript_selection_does_not_include_unmeasured_isoforms(reference, monkeypatch):
    from hitlist import mappings
    from hitlist.evidence_bundle import _selected_mapping_rows

    expression = pd.DataFrame([{"selected": True, "gene_id": "G1", "transcript_id": "ENST1"}])
    mapping = pd.DataFrame(
        {
            "gene_id": ["G1", "G1"],
            "transcript_id": ["ENST1", "ENST2"],
            "peptide": ["SLYNTVATL", "KLGGALQAK"],
        }
    )
    monkeypatch.setattr(mappings, "load_peptide_mappings", lambda **kw: mapping)
    monkeypatch.setattr(module, "_canonical_mapping_gene", lambda x: x)
    assert _selected_mapping_rows(expression).peptide.tolist() == ["SLYNTVATL"]


@pytest.mark.parametrize(
    "filename,contents,level,id_column,tpm_column",
    [
        ("expression.tsv", "gene\tTPM\nPRAME\t10\n", "gene", "gene", "TPM"),
        ("expression.csv", "Symbol,tumor_TPM\nPRAME,10\n", "gene", "Symbol", "tumor_TPM"),
        (
            "quant.sf",
            "Name\tLength\tEffectiveLength\tTPM\tNumReads\nENST1.4\t100\t80\t10\t20\n",
            "transcript",
            "Name",
            "TPM",
        ),
        (
            "abundance.tsv",
            "target_id\tlength\teff_length\test_counts\ttpm\nENST1.4\t100\t80\t20\t10\n",
            "transcript",
            "target_id",
            "tpm",
        ),
        (
            "sample.genes.results",
            "gene_id\ttranscript_id(s)\tlength\teffective_length\texpected_count\tTPM\tFPKM\nG1\tENST1,ENST2\t100\t80\t20\t10\t5\n",
            "gene",
            "gene_id",
            "TPM",
        ),
        (
            "sample.isoforms.results",
            "transcript_id\tgene_id\tlength\teffective_length\texpected_count\tTPM\tFPKM\nENST1.4\tG1\t100\t80\t20\t10\t5\n",
            "transcript",
            "transcript_id",
            "TPM",
        ),
    ],
)
def test_standard_expression_defaults(
    reference, tmp_path, monkeypatch, filename, contents, level, id_column, tpm_column
):
    monkeypatch.chdir(tmp_path)
    (tmp_path / filename).write_text(contents)
    audit, _, metadata = module.resolve_expression_table()
    assert audit.iloc[0].selected
    assert metadata["level"] == level
    assert metadata["id_column"] == id_column
    assert metadata["tpm_column"] == tpm_column
    assert metadata["inferred_inputs"] == {
        "path": True,
        "id_column": True,
        "tpm_column": True,
        "level": True,
    }
    assert metadata["input"]["path"] == str(tmp_path / filename)


def test_ambiguous_files_require_only_path_override(reference, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for name in ("expression.tsv", "expression.csv"):
        (tmp_path / name).write_text(
            "gene,TPM\nPRAME,10\n" if name.endswith("csv") else "gene\tTPM\nPRAME\t10\n"
        )
    with pytest.raises(ValueError, match="--expression"):
        module.resolve_expression_table()
    audit, _, meta = module.resolve_expression_table(tmp_path / "expression.csv")
    assert audit.iloc[0].selected
    assert not meta["inferred_inputs"]["path"]


def test_multiple_tpm_columns_require_sample_choice(reference, tmp_path):
    path = tmp_path / "expression.tsv"
    path.write_text("gene\ttumor_TPM\tnormal_TPM\nPRAME\t10\t0\n")
    with pytest.raises(ValueError, match="--tpm-column"):
        module.resolve_expression_table(path)
    audit, _, meta = module.resolve_expression_table(path, tpm_column="tumor_TPM")
    assert audit.iloc[0].tpm == 10
    assert not meta["inferred_inputs"]["tpm_column"]


def test_generic_gene_and_transcript_columns_need_level_choice(reference, tmp_path):
    path = tmp_path / "expression.tsv"
    path.write_text("gene_id\ttranscript_id\tTPM\nG1\tENST1\t10\n")
    with pytest.raises(ValueError, match="--expression-level"):
        module.resolve_expression_table(path)
    _, _, meta = module.resolve_expression_table(path, level="transcript")
    assert meta["id_column"] == "transcript_id"


def test_counts_are_not_implicitly_tpm(reference, tmp_path):
    path = tmp_path / "expression.tsv"
    path.write_text("gene\tcounts\tFPKM\nPRAME\t10\t5\n")
    with pytest.raises(ValueError, match="--tpm-column"):
        module.resolve_expression_table(path)


def test_gzip_directory_input_and_header_variants(reference, tmp_path):
    import gzip

    path = tmp_path / "expression.tsv.gz"
    with gzip.open(path, "wt") as stream:
        stream.write(" Gene ID \tTpM\tgene_name\nG1\t10\tWRONG\n")
    audit, _, meta = module.resolve_expression_table(tmp_path)
    assert audit.iloc[0].gene_name == "PRAME"
    assert meta["id_column"] == " Gene ID "
    assert meta["inferred_inputs"]["path"]


def test_neutral_header_infers_ensembl_transcripts(reference, tmp_path):
    path = tmp_path / "expression.tsv"
    path.write_text("Name\tTPM\nENST1.4\t10\n")
    audit, _, meta = module.resolve_expression_table(path)
    assert audit.iloc[0].transcript_id == "ENST1"
    assert meta["level"] == "transcript"


def test_gene_ids_override_quantifier_transcript_default(reference, tmp_path, monkeypatch):
    monkeypatch.setattr(module, "_gene_identity", lambda name: ("G1", "PRAME"))
    path = tmp_path / "quant.sf"
    path.write_text(
        "Name\tLength\tEffectiveLength\tTPM\tNumReads\nENSG00000185686\t100\t80\t10\t20\n"
    )
    audit, _, meta = module.resolve_expression_table(path)
    assert meta["level"] == "gene"
    assert audit.iloc[0].transcript_id == ""


def test_missing_default_does_not_choose_unrelated_csv(reference, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "unrelated.csv").write_text("gene,TPM\nPRAME,10\n")
    with pytest.raises(ValueError, match="--expression"):
        module.resolve_expression_table()


def test_case_collisions_require_exact_override(reference, tmp_path):
    path = tmp_path / "expression.tsv"
    path.write_text("gene\tGene\tTPM\nPRAME\tACTB\t10\n")
    with pytest.raises(ValueError, match="--id-column"):
        module.resolve_expression_table(path)
    audit, _, meta = module.resolve_expression_table(path, id_column="gene", level="gene")
    assert audit.iloc[0].selected
    assert not meta["inferred_inputs"]["id_column"]
    assert not meta["inferred_inputs"]["level"]


def test_mixed_identifiers_require_level_override(reference, tmp_path):
    path = tmp_path / "expression.tsv"
    path.write_text("Name\tTPM\nENST1\t10\nPRAME\t10\n")
    with pytest.raises(ValueError, match="--expression-level"):
        module.resolve_expression_table(path)


def test_explicit_unconventional_columns_are_preserved(reference, tmp_path):
    path = tmp_path / "custom.csv"
    path.write_text("feature,patient\nPRAME,10\n")
    audit, _, meta = module.resolve_expression_table(
        path, id_column="feature", tpm_column="patient", level="gene"
    )
    assert audit.iloc[0].selected
    assert not any(meta["inferred_inputs"].values())


@pytest.mark.parametrize("transcript", [False, True])
def test_rsem_auxiliary_estimates_do_not_require_sample_choice(reference, tmp_path, transcript):
    path = tmp_path / "patient.results"
    header = "gene_id\tlength\teffective_length\texpected_count\tTPM\tpme_TPM\tTPM_ci_lower_bound\tTPM_ci_upper_bound\tTPM_coefficient_of_quartile_variation"
    row = "PRAME\t100\t80\t20\t10\t12\t8\t15\t0.1"
    if transcript:
        header = "transcript_id\t" + header + "\tIsoPct_from_pme_TPM"
        row = "ENST1\t" + row + "\t100"
    path.write_text(header + "\n" + row + "\n")
    audit, _, meta = module.resolve_expression_table(path)
    assert audit.iloc[0].tpm == 10
    assert meta["tpm_column"] == "TPM"
    assert meta["level"] == ("transcript" if transcript else "gene")
    audit, _, _ = module.resolve_expression_table(path, tpm_column="pme_TPM")
    assert audit.iloc[0].tpm == 12
    path.write_text(header + "\ttumor_TPM\n" + row + "\t14\n")
    with pytest.raises(ValueError, match="--tpm-column"):
        module.resolve_expression_table(path)
