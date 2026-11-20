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


def test_ms_policy_version_is_explicit_and_legacy_bundles_remain_verifiable(
    mapped_index, reference, atlas_dir, tmp_path
):
    expression = tmp_path / "expression.tsv"
    expression.write_text("gene\ttpm\nPRAME\t10\n")
    target = tmp_path / "evidence"
    write_cta_evidence_bundle(
        target, expression, atlas_dir=atlas_dir, id_column="gene", tpm_column="tpm"
    )
    path = target / "manifest.json"
    manifest = json.loads(path.read_text())
    assert manifest.pop("ms_policy_version") == 3
    path.write_text(json.dumps(manifest))
    verify_evidence_bundle(target)
    manifest["ms_policy_version"] = 999
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Unsupported MS evidence policy"):
        verify_evidence_bundle(target)


@pytest.mark.parametrize("built_index", [True], indirect=True)
def test_built_structural_evidence_is_exported_with_contributors(
    mapped_index, reference, atlas_dir, tmp_path
):
    expression = tmp_path / "expression.tsv"
    expression.write_text("gene\ttpm\nPRAME\t10\n")
    target = tmp_path / "evidence"
    write_cta_evidence_bundle(
        target, expression, atlas_dir=atlas_dir, id_column="gene", tpm_column="tpm"
    )
    manifest = verify_evidence_bundle(target)
    assert manifest["ms_policy_version"] == 3
    assert len(pd.read_parquet(target / "presentation.parquet")) == 2
    excluded = pd.read_parquet(target / "excluded_observations.parquet")
    assert excluded.assay_modality.tolist() == ["structural"]
    assert excluded.evidence_kind.tolist() == ["other"]
    assert excluded.assay_iri.tolist() == ["http://iedb.org/assay/3"]
    assert len(pd.read_parquet(target / "contributors.parquet")) == 4
    assert pd.read_parquet(target / "peptides.parquet").iloc[0].n_ms_observations == 2


@pytest.mark.parametrize("policy_version", [1, 2, 3])
def test_unknown_outcome_new_admission_and_saved_policy_replay(
    mapped_index, reference, atlas_dir, tmp_path, monkeypatch, policy_version
):
    from hitlist import evidence_bundle, export
    from hitlist.assays import ANNOTATION_COLUMNS

    original = export.generate_training_table

    def unknown(**kwargs):
        frame = original(**kwargs)
        frame["qualitative_measurement"] = "unknown"
        return frame

    monkeypatch.setattr(export, "generate_training_table", unknown)
    expression = tmp_path / "expression.tsv"
    expression.write_text("gene\tTPM\nPRAME\t10\n")
    target = tmp_path / "evidence"
    # Simulate an export with the historical admission contract. Then remove
    # new annotations, so replay cannot depend on a current derived flag.
    with monkeypatch.context() as saved:
        saved.setattr(evidence_bundle, "MS_POLICY_VERSION", policy_version)
        if policy_version < 3:
            saved.setattr(evidence_bundle, "_positive_ms", evidence_bundle._modality_positive_ms)
        write_cta_evidence_bundle(target, expression, atlas_dir=atlas_dir)
    presentation = pd.read_parquet(target / "presentation.parquet")
    excluded = pd.read_parquet(target / "excluded_observations.parquet")
    assert len(presentation) == (2 if policy_version < 3 else 0)
    assert len(excluded) == (0 if policy_version < 3 else 2)
    assert len(pd.read_parquet(target / "contributors.parquet")) == 3
    if policy_version < 3:
        presentation.drop(columns=list(ANNOTATION_COLUMNS)).to_parquet(
            target / "presentation.parquet", index=False
        )
        rehash(target, "presentation.parquet")
    else:
        assert excluded.ms_outcome_status.tolist() == ["unestablished"] * 2
        assert excluded.exclusion_reason.tolist() == ["unestablished_ms_outcome"] * 2
    assert verify_evidence_bundle(target)["ms_policy_version"] == policy_version
    if policy_version < 3:
        path = target / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["ms_policy_version"] = 3
        path.write_text(json.dumps(manifest))
        with pytest.raises(ValueError, match="Non-MS evidence"):
            verify_evidence_bundle(target)
    else:
        excluded["exclusion_reason"] = "invented"
        excluded.to_parquet(target / "excluded_observations.parquet", index=False)
        rehash(target, "excluded_observations.parquet")
        with pytest.raises(ValueError, match="MS exclusion reasons"):
            verify_evidence_bundle(target)


@pytest.mark.parametrize("built_index", [False, True], indirect=True)
def test_archived_pre_applicability_cta_bundle_verifies_offline_and_partial_applicability_fails(
    mapped_index, reference, atlas_dir, tmp_path
):
    from hitlist.lineage import APPLICABILITY_COLUMNS
    from hitlist.provenance import file_digest

    def strip_to_pre_2_1_18(directory):
        for name in ("presentation.parquet", "excluded_observations.parquet"):
            frame = pd.read_parquet(directory / name)
            assert "assay_method" in frame.columns
            frame.drop(columns=[c for c in APPLICABILITY_COLUMNS if c in frame.columns]).to_parquet(
                directory / name, index=False
            )
            rehash(directory, name)
        identities = pd.read_parquet(directory / "identities.parquet")
        identities.drop(
            columns=[c for c in ("assay_method", *APPLICABILITY_COLUMNS) if c in identities.columns]
        ).to_parquet(directory / "identities.parquet", index=False)
        manifest_path = directory / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["coverage"].pop("applicability", None)
        manifest["artifacts"]["identities.parquet"] = file_digest(directory / "identities.parquet")
        manifest_path.write_text(json.dumps(manifest))

    # 1. Nonempty archived pre-2.1.18 bundle (both MS-only and MS + excluded other_assays).
    expression = tmp_path / "expression.tsv"
    expression.write_text("gene\ttpm\nPRAME\t10\n")
    archived = tmp_path / "archived"
    write_cta_evidence_bundle(
        archived, expression, atlas_dir=atlas_dir, id_column="gene", tpm_column="tpm"
    )
    strip_to_pre_2_1_18(archived)
    verified = verify_evidence_bundle(archived)
    assert "applicability" not in verified["coverage"]

    # 2. Empty-selection archived pre-2.1.18 bundle.
    empty_expr = tmp_path / "empty_expression.tsv"
    empty_expr.write_text("gene\ttpm\nMAGEA3\t10\n")
    archived_empty = tmp_path / "archived_empty"
    write_cta_evidence_bundle(
        archived_empty, empty_expr, atlas_dir=atlas_dir, id_column="gene", tpm_column="tpm"
    )
    strip_to_pre_2_1_18(archived_empty)
    verified_empty = verify_evidence_bundle(archived_empty)
    assert "applicability" not in verified_empty["coverage"]

    # 3. Partial applicability columns (or applicability present in observations/manifest
    # while stripped from identities) must fail verification.
    partial = tmp_path / "partial"
    write_cta_evidence_bundle(
        partial, expression, atlas_dir=atlas_dir, id_column="gene", tpm_column="tpm"
    )
    pd.read_parquet(partial / "identities.parquet").drop(
        columns=["donor_applicability"]
    ).to_parquet(partial / "identities.parquet", index=False)
    rehash(partial, "identities.parquet")
    with pytest.raises(ValueError, match="Incomplete applicability columns"):
        verify_evidence_bundle(partial)


def test_cta_policy_defaults_to_legacy_and_archived_manifest_without_policy_verifies(
    mapped_index, reference, atlas_dir, tmp_path
):
    """#852: default remains legacy historical oncoref.cta; pre-2.1.24 manifests
    without cta_policy remain verifiable."""
    expression = tmp_path / "expression.tsv"
    expression.write_text("gene\ttpm\nPRAME\t10\n")
    target = tmp_path / "legacy-policy"
    write_cta_evidence_bundle(
        target, expression, atlas_dir=atlas_dir, id_column="gene", tpm_column="tpm"
    )
    manifest = verify_evidence_bundle(target)
    policy = manifest["expression"]["cta_policy"]
    assert policy["selector"] == "legacy"
    assert policy["provider"] == "oncoref.cta"
    assert policy["policy_id"] == "oncoref.cta.legacy.strict"
    assert policy["implicit_upgrade"] is False
    assert policy["observed_peptide_blacklist_independent"] is True
    assert policy["unsafe_gene_ids"] is None
    assert "has_policy_unsafe_match" not in pd.read_parquet(target / "peptides.parquet").columns

    # Strip cta_policy to simulate a pre-2.1.24 archived bundle.
    manifest_path = target / "manifest.json"
    archived = json.loads(manifest_path.read_text())
    archived["expression"].pop("cta_policy")
    manifest_path.write_text(json.dumps(archived))
    verified = verify_evidence_bundle(target)
    assert "cta_policy" not in verified["expression"]


def test_explicit_oncoref_cta_selection_policy_records_receipt_and_keeps_other_non_forbidden(
    mapped_index, reference, atlas_dir, tmp_path, monkeypatch
):
    """#852: explicit cta_selection policy records policy_id/reference_id and
    independent profile fields, keeps Other/Unassessed out of Unsafe, and keeps
    donor-resolved observed-peptide blacklists independent."""
    import sys
    import types

    from hitlist import cta_expression

    fake_selection = types.ModuleType("oncoref.cta_selection")
    fake_selection.profiles = lambda *, species="human": pd.DataFrame(
        [
            {
                "entity_id": "prot:PRAME",
                "label": "PRAME",
                "gene_ids": ("G1",),
                "category": "CTA",
                "subtype": pd.NA,
                "classification": "CTA",
                "assessment_status": "assessed",
                "policy_id": "oncoref.cta-selection.2026-10-31.1",
                "reference_id": "ensembl-112-human",
                "source_snapshot_id": "snap-2026-10-31",
                "tumor_max_tpm": 45.0,
                "replicated_tumor_rna": True,
                "normal_max_ntpm": 0.4,
                "adult_disposable_expression": False,
                "adult_disposable_only": False,
                "developmental_measured": True,
            },
            {
                "entity_id": "prot:ACTB",
                "label": "ACTB",
                "gene_ids": ("G4",),
                "category": "Unsafe",
                "subtype": pd.NA,
                "classification": "Unsafe",
                "assessment_status": "assessed",
                "policy_id": "oncoref.cta-selection.2026-10-31.1",
                "reference_id": "ensembl-112-human",
                "source_snapshot_id": "snap-2026-10-31",
                "tumor_max_tpm": 900.0,
                "replicated_tumor_rna": True,
                "normal_max_ntpm": 850.0,
                "adult_disposable_expression": False,
                "adult_disposable_only": False,
                "developmental_measured": True,
            },
            {
                "entity_id": "prot:TCHHL1",
                "label": "TCHHL1",
                "gene_ids": ("G5",),
                "category": "Other",
                "subtype": "Nonreproductive",
                "classification": "Other:Nonreproductive",
                "assessment_status": "assessed",
                "policy_id": "oncoref.cta-selection.2026-10-31.1",
                "reference_id": "ensembl-112-human",
                "source_snapshot_id": "snap-2026-10-31",
                "tumor_max_tpm": 2.0,
                "replicated_tumor_rna": False,
                "normal_max_ntpm": 0.1,
                "adult_disposable_expression": True,
                "adult_disposable_only": True,
                "developmental_measured": False,
            },
            {
                "entity_id": "prot:OR10A4",
                "label": "OR10A4",
                "gene_ids": ("G6",),
                "category": pd.NA,
                "subtype": pd.NA,
                "classification": "Unassessed",
                "assessment_status": "unassessed",
                "policy_id": "oncoref.cta-selection.2026-10-31.1",
                "reference_id": "ensembl-112-human",
                "source_snapshot_id": "snap-2026-10-31",
                "tumor_max_tpm": 1.0,
                "replicated_tumor_rna": False,
                "normal_max_ntpm": 0.0,
                "adult_disposable_expression": False,
                "adult_disposable_only": False,
                "developmental_measured": False,
            },
        ]
    )
    fake_selection.sequence_dispositions = lambda *, species="human": pd.DataFrame(
        [
            {"entity_id": "prot:PRAME", "overlap_role": "allow", "direct_target_eligible": True},
            {"entity_id": "prot:ACTB", "overlap_role": "exclude", "direct_target_eligible": False},
            {"entity_id": "prot:TCHHL1", "overlap_role": "allow", "direct_target_eligible": False},
            {"entity_id": "prot:OR10A4", "overlap_role": "review", "direct_target_eligible": False},
        ]
    )
    fake_selection.contract = lambda *, species="human": {
        "selection_policy_id": "oncoref.cta-selection.2026-10-31.1",
        "reference_id": "ensembl-112-human",
    }
    fake_selection.coverage = lambda *, species="human": {
        "counts": {"CTA": 1, "Unsafe": 1, "Other": 1, "Unassigned": 1}
    }
    import oncoref

    monkeypatch.setitem(sys.modules, "oncoref.cta_selection", fake_selection)
    monkeypatch.setattr(oncoref, "cta_selection", fake_selection, raising=False)
    monkeypatch.setattr(
        cta_expression,
        "_gene_identity",
        lambda name: {
            "PRAME": ("G1", "PRAME"),
            "ACTB": ("G4", "ACTB"),
            "TCHHL1": ("G5", "TCHHL1"),
            "OR10A4": ("G6", "OR10A4"),
        }.get(name, ("", "")),
    )

    expression = tmp_path / "expression.tsv"
    expression.write_text("gene\ttpm\nPRAME\t10\nTCHHL1\t15\nOR10A4\t12\nACTB\t50\n")
    target = tmp_path / "cta-selection-bundle"
    write_cta_evidence_bundle(
        target,
        expression,
        atlas_dir=atlas_dir,
        id_column="gene",
        tpm_column="tpm",
        cta_policy="cta_selection",
    )
    manifest = verify_evidence_bundle(target)
    policy = manifest["expression"]["cta_policy"]
    assert policy["selector"] == "cta_selection"
    assert policy["policy_id"] == "oncoref.cta-selection.2026-10-31.1"
    assert policy["reference_id"] == "ensembl-112-human"
    assert policy["source_snapshot_id"] == "snap-2026-10-31"
    assert policy["n_cta_genes"] == 1
    assert policy["n_unsafe_genes"] == 1
    assert policy["unsafe_gene_ids"] == ["G4"]  # ONLY Unsafe; never Other (G5) or Unassessed (G6)
    assert policy["other_is_not_forbidden"] is True
    assert policy["observed_peptide_blacklist_independent"] is True

    # Independent profile attributes survive in cta_reference.parquet.
    ref = pd.read_parquet(target / "cta_reference.parquet").set_index("Ensembl_Gene_ID")
    assert ref.loc["G5", "policy_classification"] == "Other:Nonreproductive"
    assert bool(ref.loc["G5", "adult_disposable_only"]) is True
    assert bool(ref.loc["G5", "developmental_measured"]) is False
    assert ref.loc["G6", "policy_category"] == "Unassessed"
    assert ref.loc["G6", "overlap_role"] == "review"

    # Peptide SLYNTVATL maps to G1 (CTA) and G4 (Unsafe) AND is Atlas-blacklisted.
    peptides = pd.read_parquet(target / "peptides.parquet")
    row = peptides.iloc[0]
    assert bool(row.has_policy_unsafe_match) is True
    assert json.loads(row.policy_categories) == ["CTA", "Unsafe"]
    assert "policy_unsafe_reference_match" in row.avoid_reasons
    assert "essential_tissue_blacklist" in row.avoid_reasons
    assert (target / "forbidden_sequences.txt").read_text() == "SLYNTVATL\n"

    # Verifier rejects leaking Other (G5) into unsafe_gene_ids or tampering policy_id.
    manifest_path = target / "manifest.json"
    tampered = json.loads(manifest_path.read_text())
    tampered["expression"]["cta_policy"]["unsafe_gene_ids"] = ["G4", "G5"]
    tampered["expression"]["cta_policy"]["n_unsafe_genes"] = 2
    manifest_path.write_text(json.dumps(tampered))
    with pytest.raises(ValueError, match="Unsafe partition mismatch"):
        verify_evidence_bundle(target)

    tampered = json.loads(manifest_path.read_text())
    tampered["expression"]["cta_policy"]["unsafe_gene_ids"] = ["G4"]
    tampered["expression"]["cta_policy"]["n_unsafe_genes"] = 1
    tampered["expression"]["cta_policy"]["policy_id"] = "oncoref.cta-selection.tampered"
    manifest_path.write_text(json.dumps(tampered))
    with pytest.raises(ValueError, match="policy_id mismatch"):
        verify_evidence_bundle(target)


@pytest.mark.parametrize("selector", ["target_sets", "target_selection"])
def test_target_sets_and_target_selection_policies_never_treat_neither_or_other_as_unsafe(
    tmp_path, monkeypatch, selector
):
    """#852: target_sets (1.20.0) and target_selection (1.37+) keep Neither/Other
    separate from Unsafe and reject definition='extended'."""
    import sys
    import types

    import oncoref

    from hitlist import cta_expression

    if selector == "target_sets":
        fake_ts = types.ModuleType("oncoref.target_sets")
        fake_ts.POLICY_ID = "oncoref.target-sets.2026-10-25.2"
        fake_ts.results = lambda: pd.DataFrame(
            [
                {"Gene": "CTAG1B", "Gene ID": "G1", "Category": "CTA", "Detail": "Pass"},
                {"Gene": "MAGEA4", "Gene ID": "G2", "Category": "Neither", "Detail": "Variable"},
                {"Gene": "PRAME", "Gene ID": "G3", "Category": "Unsafe", "Detail": "Protein"},
            ]
        )
        fake_pol = types.ModuleType("oncoref.cta_policy")
        fake_pol.POLICY_ID = "oncoref.target-sets.2026-10-25.2"
        fake_pol.manifest = lambda: {
            "policy_id": "oncoref.target-sets.2026-10-25.2",
            "reference": "ensembl-112",
        }
        monkeypatch.setitem(sys.modules, "oncoref.target_sets", fake_ts)
        monkeypatch.setitem(sys.modules, "oncoref.cta_policy", fake_pol)
        monkeypatch.setattr(oncoref, "target_sets", fake_ts, raising=False)
        monkeypatch.setattr(oncoref, "cta_policy", fake_pol, raising=False)
    else:
        fake_tsel = types.ModuleType("oncoref.target_selection")
        fake_tsel.POLICY_ID = "oncoref.target-selection.2026-11-19.4"
        fake_tsel.definition = lambda: {"policy_id": "oncoref.target-selection.2026-11-19.4"}
        fake_tsel.metadata = lambda: {"reference_id": "ensembl-112-rebuilt"}
        fake_tsel.results = lambda: pd.DataFrame(
            [
                {
                    "protein_group_label": "CTAG1B",
                    "member_gene_ids": json.dumps(["G1"]),
                    "category": "CTA",
                    "subtype": "",
                },
                {
                    "protein_group_label": "CA_ONLY",
                    "member_gene_ids": json.dumps(["G2"]),
                    "category": "Cancer Antigen only",
                    "subtype": "",
                },
                {
                    "protein_group_label": "PRAME",
                    "member_gene_ids": json.dumps(["G3"]),
                    "category": "Unsafe",
                    "subtype": "",
                },
            ]
        )
        monkeypatch.setitem(sys.modules, "oncoref.target_selection", fake_tsel)
        monkeypatch.setattr(oncoref, "target_selection", fake_tsel, raising=False)

    monkeypatch.setattr(
        cta_expression,
        "_gene_identity",
        lambda name: {"CTAG1B": ("G1", "CTAG1B"), "G2": ("G2", "G2"), "PRAME": ("G3", "PRAME")}.get(
            name, ("", "")
        ),
    )
    expr_path = tmp_path / "expr.tsv"
    expr_path.write_text("gene\ttpm\nCTAG1B\t10\nG2\t10\nPRAME\t10\n")

    with pytest.raises(ValueError, match="only applies to cta_policy='legacy'"):
        cta_expression.resolve_expression_table(
            expr_path,
            id_column="gene",
            tpm_column="tpm",
            cta_policy=selector,
            definition="extended",
        )

    audit, ref, meta = cta_expression.resolve_expression_table(
        expr_path, id_column="gene", tpm_column="tpm", cta_policy=selector
    )
    assert meta["cta_gene_ids"] == ["G1"]
    assert meta["cta_policy"]["selector"] == selector
    assert meta["cta_policy"]["unsafe_gene_ids"] == ["G3"]
    assert "G2" not in meta["cta_policy"]["unsafe_gene_ids"]
    assert audit.set_index("gene_id")["selection_reason"].to_dict() == {
        "G1": "selected",
        "G2": "not_cta",
        "G3": "not_cta",
    }
    assert set(ref["Ensembl_Gene_ID"]) == {"G1", "G2", "G3"}


def test_cta_evidence_cli_accepts_cta_policy_flag(
    mapped_index, reference, atlas_dir, tmp_path, monkeypatch, capsys
):
    import sys

    from hitlist.cli import main

    expression = tmp_path / "expression.tsv"
    expression.write_text("gene\ttpm\nPRAME\t10\n")
    target = tmp_path / "cli-cta-policy"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "hitlist",
            "export",
            "cta-evidence",
            "--expression",
            str(expression),
            "--id-column",
            "gene",
            "--tpm-column",
            "tpm",
            "--cta-policy",
            "legacy",
            "--atlas-dir",
            str(atlas_dir),
            "--bundle",
            str(target),
        ],
    )
    main()
    out = capsys.readouterr().out
    assert "policy=oncoref.cta.legacy.strict" in out
    assert verify_evidence_bundle(target)["expression"]["cta_policy"]["selector"] == "legacy"


def test_live_installed_oncoref_cta_selection_policy_if_available():
    """Smoke-test real oncoref.cta_selection when installed (e.g. CI oncoref>=1.27)."""
    import importlib

    try:
        importlib.import_module("oncoref.cta_selection")
    except ImportError:
        pytest.skip("Installed oncoref predates oncoref.cta_selection")

    from hitlist.cta_expression import _resolve_cta_policy

    ref, cta_ids, receipt = _resolve_cta_policy("cta_selection", "strict")
    assert receipt["selector"] == "cta_selection"
    assert receipt["policy_id"]
    assert receipt["n_cta_genes"] == len(cta_ids) > 0
    assert receipt["n_unsafe_genes"] > 0
    assert set(cta_ids).isdisjoint(set(receipt["unsafe_gene_ids"]))
    assert {"Ensembl_Gene_ID", "policy_category", "policy_id", "reference_id"} <= set(ref.columns)
