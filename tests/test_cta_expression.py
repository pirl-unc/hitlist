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
