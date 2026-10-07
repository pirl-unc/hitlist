import json

import pandas as pd
import pytest

from hitlist.evidence_bundle import (
    verify_evidence_bundle,
    write_cta_evidence_bundle,
    write_tissue_blacklist_bundle,
)
from tests import test_cta_expression, test_tissue_blacklist, test_training_bundle

reference = test_cta_expression.reference
atlas_dir = test_tissue_blacklist.atlas_dir
built_index = test_training_bundle.built_index


def rehash(directory, filename):
    from hitlist.provenance import file_digest

    path = directory / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["artifacts"][filename] = file_digest(directory / filename)
    path.write_text(json.dumps(manifest))


@pytest.fixture
def mapped_index(built_index):
    from hitlist.mappings import _mapping_artifact_contract, _obs_fingerprint

    rows = []
    for gene, protein in [("G1", "P1"), ("G4", "P4")]:
        rows.append(
            {
                "peptide": "SLYNTVATL",
                "gene_id": gene,
                "gene_name": "PRAME" if gene == "G1" else "ACTB",
                "protein_id": protein,
                "transcript_id": "T1" if gene == "G1" else "T4",
                "is_canonical_transcript": True,
                "gene_biotype": "protein_coding",
                "position": 0,
                "n_flank": "",
                "c_flank": "",
                "proteome": "Homo sapiens",
                "proteome_source": "ensembl",
            }
        )
    pd.DataFrame(rows).to_parquet(built_index / "peptide_mappings.parquet", index=False)
    metadata = {
        "observations": _obs_fingerprint(),
        "contract": _mapping_artifact_contract(
            release=112, use_uniprot=False, fetch_missing=True, flank=15
        ),
        "unavailable_proteomes": [],
    }
    (built_index / "peptide_mappings_meta.json").write_text(json.dumps(metadata))
    return built_index


def test_blacklist_bundle_is_standalone_reproducible_and_tamper_evident(atlas_dir, tmp_path):
    target = tmp_path / "blacklist"
    write_tissue_blacklist_bundle(target, atlas_dir=atlas_dir)
    manifest = verify_evidence_bundle(target)
    assert manifest["kind"] == "tissue_blacklist"
    assert (target / "forbidden_sequences.txt").read_text() == "SLYNTVATL\n"
    assert manifest["policy"]["hla_filter"] is None
    assert manifest["policy"]["tissue_status"] == "nonmalignant"
    with pytest.raises(FileExistsError):
        write_tissue_blacklist_bundle(target, atlas_dir=atlas_dir)
    (target / "forbidden_sequences.txt").write_text("TAMPER\n")
    with pytest.raises(ValueError, match="artifact mismatch"):
        verify_evidence_bundle(target)


def test_expression_bundle_preserves_shared_mapping_and_raw_contributors(
    mapped_index, reference, atlas_dir, tmp_path
):
    expression = tmp_path / "expression.tsv"
    expression.write_text("gene\ttpm\nPRAME\t10\nMAGEA3\t20\n")
    target = tmp_path / "evidence"
    write_cta_evidence_bundle(
        target, expression, atlas_dir=atlas_dir, id_column="gene", tpm_column="tpm"
    )
    manifest = verify_evidence_bundle(target)
    assert manifest["kind"] == "cta_expression"
    peptides = pd.read_parquet(target / "peptides.parquet")
    assert peptides.iloc[0].blacklisted
    assert peptides.iloc[0].has_non_cta_match
    assert not peptides.iloc[0].cta_specific
    assert peptides.iloc[0].avoid_sequence
    assert len(pd.read_parquet(target / "presentation.parquet")) == 2
    assert len(pd.read_parquet(target / "contributors.parquet")) == 3
    assert set(pd.read_parquet(target / "mappings.parquet").gene_id) == {"G1", "G4"}
    assert len(pd.read_parquet(target / "expression.parquet")) == 2


def test_stale_index_fails_before_creating_bundle(mapped_index, reference, atlas_dir, tmp_path):
    expression = tmp_path / "expression.tsv"
    expression.write_text("gene\ttpm\nPRAME\t10\n")
    (mapped_index / "peptide_mappings_meta.json").unlink()
    with pytest.raises(ValueError, match="mapping"):
        write_cta_evidence_bundle(
            tmp_path / "bad", expression, atlas_dir=atlas_dir, id_column="gene", tpm_column="tpm"
        )
    assert not (tmp_path / "bad").exists()


def test_verifier_rejects_changed_raw_source_link_even_with_refreshed_hash(atlas_dir, tmp_path):
    target = tmp_path / "blacklist"
    write_tissue_blacklist_bundle(target, atlas_dir=atlas_dir)
    evidence = pd.read_parquet(target / "tissue_evidence.parquet")
    evidence.loc[0, "source_record_id"] = "invented:row"
    evidence.to_parquet(target / "tissue_evidence.parquet", index=False)
    rehash(target, "tissue_evidence.parquet")
    with pytest.raises(ValueError, match="source"):
        verify_evidence_bundle(target)


def test_verifier_rejects_expression_relationship_tampering(
    mapped_index, reference, atlas_dir, tmp_path
):
    expression = tmp_path / "expression.tsv"
    expression.write_text("gene\ttpm\nPRAME\t10\n")
    target = tmp_path / "evidence"
    write_cta_evidence_bundle(
        target, expression, atlas_dir=atlas_dir, id_column="gene", tpm_column="tpm"
    )
    links = pd.read_parquet(target / "expression_mappings.parquet")
    links.loc[0, "expression_tpm"] = 9000
    links.to_parquet(target / "expression_mappings.parquet", index=False)
    rehash(target, "expression_mappings.parquet")
    with pytest.raises(ValueError, match="expression"):
        verify_evidence_bundle(target)


def test_longer_candidate_containing_forbidden_sequence_is_avoided(monkeypatch):
    from hitlist import cta_expression
    from hitlist.evidence_bundle import _peptide_summary
    from hitlist.tissue_blacklist import build_tissue_blacklist
    from tests.test_tissue_blacklist import observation

    monkeypatch.setattr(cta_expression, "_canonical_mapping_gene", lambda x: x)
    mappings = pd.DataFrame(
        [
            {
                "peptide": "ASLYNTVATLK",
                "gene_id": "G1",
                "protein_id": "P1",
                "transcript_id": "T1",
                "proteome": "Homo sapiens",
            }
        ]
    )
    risk, _ = build_tissue_blacklist(pd.DataFrame([observation("D1"), observation("D2")]))
    summary, _ = _peptide_summary(mappings, {"G1"}, pd.DataFrame(columns=["peptide"]), risk)
    row = summary.iloc[0]
    assert row.cta_specific
    assert not row.blacklisted  # the longer peptide itself was not observed
    assert row.contains_blacklisted_sequence and row.avoid_sequence
    assert json.loads(row.blacklist_matches) == ["SLYNTVATL"]


def test_empty_selection_keeps_global_blacklist(mapped_index, reference, atlas_dir, tmp_path):
    expression = tmp_path / "expression.tsv"
    expression.write_text("gene\ttpm\nMAGEA3\t10\n")
    target = tmp_path / "empty"
    write_cta_evidence_bundle(
        target, expression, atlas_dir=atlas_dir, id_column="gene", tpm_column="tpm"
    )
    assert pd.read_parquet(target / "peptides.parquet").empty
    assert pd.read_parquet(target / "presentation.parquet").empty
    assert (target / "forbidden_sequences.txt").read_text() == "SLYNTVATL\n"


def test_non_ms_rows_remain_auditable_but_never_count_as_presentation(
    mapped_index, reference, atlas_dir, tmp_path, monkeypatch
):
    from hitlist import export

    original = export.generate_training_table

    def non_ms(**kwargs):
        frame = original(**kwargs)
        frame["assay_method"] = frame.assay_method.astype("string")
        frame.loc[frame.index[0], "assay_method"] = "fluorescence"
        return frame

    monkeypatch.setattr(export, "generate_training_table", non_ms)
    expression = tmp_path / "expression.tsv"
    expression.write_text("gene\ttpm\nPRAME\t10\n")
    target = tmp_path / "evidence"
    write_cta_evidence_bundle(
        target, expression, atlas_dir=atlas_dir, id_column="gene", tpm_column="tpm"
    )
    assert len(pd.read_parquet(target / "presentation.parquet")) == 1
    rejected = pd.read_parquet(target / "excluded_observations.parquet")
    assert rejected.assay_method.tolist() == ["fluorescence"]
    assert len(pd.read_parquet(target / "contributors.parquet")) == 3
    assert pd.read_parquet(target / "peptides.parquet").iloc[0].n_ms_observations == 1


def test_input_change_during_verification_prevents_publication(
    mapped_index, reference, atlas_dir, tmp_path, monkeypatch
):
    from hitlist import evidence_bundle

    expression = tmp_path / "expression.tsv"
    expression.write_text("gene\ttpm\nPRAME\t10\n")
    original = evidence_bundle.verify_evidence_bundle

    def change_input(path):
        result = original(path)
        expression.write_text("gene\ttpm\nPRAME\t999\n")
        return result

    monkeypatch.setattr(evidence_bundle, "verify_evidence_bundle", change_input)
    with pytest.raises(ValueError, match="inputs changed"):
        write_cta_evidence_bundle(
            tmp_path / "changed",
            expression,
            atlas_dir=atlas_dir,
            id_column="gene",
            tpm_column="tpm",
        )
    assert not (tmp_path / "changed").exists()
    assert not list(tmp_path.glob(".changed-*"))


def test_blacklist_cli_round_trip(atlas_dir, tmp_path, monkeypatch, capsys):
    import sys

    from hitlist.cli import main

    target = tmp_path / "cli-bundle"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "hitlist",
            "export",
            "tissue-blacklist",
            "--atlas-dir",
            str(atlas_dir),
            "--bundle",
            str(target),
        ],
    )
    main()
    monkeypatch.setattr(sys, "argv", ["hitlist", "verify-evidence-bundle", str(target)])
    main()
    assert "Verified tissue_blacklist bundle" in capsys.readouterr().out


def test_negative_assay_and_unknown_modality_are_not_positive_ms():
    from hitlist.evidence_bundle import _positive_ms

    rows = pd.DataFrame(
        {
            "assay_method": ["mass spectrometry", "mass spectrometry", "", "", "fluorescence"],
            "qualitative_measurement": ["Positive", "Negative", "", "", "Positive"],
            "source": ["iedb", "iedb", "supplement", "iedb", "supplement"],
        }
    )
    assert _positive_ms(rows).tolist() == [True, False, True, False, False]


def test_verification_uses_captured_tissue_policy_after_installed_policy_changes(
    mapped_index, reference, atlas_dir, tmp_path, monkeypatch
):
    from hitlist import export, tissue_blacklist

    original = export.generate_training_table

    def essential_tissue(**kwargs):
        frame = original(**kwargs)
        frame["source_tissue"] = "Heart"
        frame["src_healthy_tissue"] = True
        frame["src_cell_line"] = False
        return frame

    monkeypatch.setattr(export, "generate_training_table", essential_tissue)
    expression = tmp_path / "expression.tsv"
    expression.write_text("gene\ttpm\nPRAME\t10\n")
    target = tmp_path / "portable"
    write_cta_evidence_bundle(
        target, expression, atlas_dir=atlas_dir, id_column="gene", tpm_column="tpm"
    )
    assert (
        pd.read_parquet(target / "peptides.parquet")
        .iloc[0]
        .n_unresolved_essential_tissue_observations
        == 2
    )
    changed_policy = tissue_blacklist.tissue_blacklist_policy()
    changed_policy["tissue_groups"]["heart"] = ["Myocardium"]
    monkeypatch.setattr(tissue_blacklist, "tissue_blacklist_policy", lambda: changed_policy)
    verify_evidence_bundle(target)


def test_cta_cli_uses_defaults_and_records_resolved_input(
    mapped_index, reference, atlas_dir, tmp_path, monkeypatch, capsys
):
    import sys

    from hitlist.cli import main

    monkeypatch.chdir(tmp_path)
    (tmp_path / "expression.tsv").write_text("gene\tTPM\nPRAME\t10\n")
    target = tmp_path / "defaults"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "hitlist",
            "export",
            "cta-evidence",
            "--atlas-dir",
            str(atlas_dir),
            "--bundle",
            str(target),
        ],
    )
    main()
    manifest = verify_evidence_bundle(target)
    assert all(manifest["expression"]["inferred_inputs"].values())
    assert manifest["expression"]["id_column"] == "gene"
    assert manifest["expression"]["tpm_column"] == "TPM"
    assert "ID=gene; TPM=TPM" in capsys.readouterr().out
    assert pd.read_parquet(target / "expression_mappings.parquet").expression_tpm.tolist() == [10]


def test_bundle_api_infers_columns_from_explicit_file(mapped_index, reference, atlas_dir, tmp_path):
    path = tmp_path / "patient.csv"
    path.write_text("gene,TPM\nPRAME,10\n")
    target = tmp_path / "api"
    write_cta_evidence_bundle(target, path, atlas_dir=atlas_dir)
    manifest = verify_evidence_bundle(target)
    assert manifest["expression"]["inferred_inputs"] == {
        "path": False,
        "id_column": True,
        "tpm_column": True,
        "level": True,
    }
