"""Supporting modalities keep source grain, scales and unresolved material links."""

import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from hitlist.provenance import file_digest
from hitlist.supporting_evidence import (
    BULK_FIELDS,
    EXPRESSION_FIELDS,
    MAPPING_FIELDS,
    verify_supporting_evidence_package,
    write_supporting_evidence_package,
)
from tests.test_context_evidence import _cohort_fixture


def _add_supporting_native(source):
    assay = pq.read_table(source / "other_assays.parquet").to_pylist()
    assay[0]["peptide"] = "UNMAPPED"
    assay[1]["peptide"] = "NONPEPTIDE"
    assay[1]["is_non_peptide_ligand"] = True
    pq.write_table(pa.Table.from_pylist(assay), source / "other_assays.parquet")

    bulk = []
    for index, (source_id, grain, enzyme) in enumerate(
        [
            ("Bekker-Jensen_2017", "peptide", "Trypsin/P legacy display"),
            ("Bekker-Jensen_2017", "peptide", "Chymotrypsin"),
            ("Bekker-Jensen_2017", "protein", "Trypsin/P legacy display"),
            ("CCLE_Nusinow_2020", "protein", "Trypsin/P legacy display"),
        ]
    ):
        row = dict.fromkeys(BULK_FIELDS)
        row.update(
            {
                "evidence_kind": "bulk_proteomics",
                "granularity": grain,
                "source": source_id,
                "reference": "PMID:12345678",
                "pmid": 12345678,
                "species": "Homo sapiens",
                "cell_line_name": "HeLa",
                "sample_label": "HeLa",
                "gene_symbol": "TEST",
                "uniprot_acc": "P00001",
                "peptide": "PEPTIDE" if grain == "peptide" else "",
                "n_replicates_detected": 2 if grain == "peptide" else None,
                "log2_intensity": 3.0 if index == 2 else None,
                "abundance_log2_normalized": 4.0 if index == 3 else None,
                "abundance_percentile": 50.0 if grain == "protein" else None,
                "digestion_enzyme": enzyme,
                "n_fractions_in_run": 46 if source_id == "Bekker-Jensen_2017" else None,
                "enrichment": "none",
                "instrument": "source instrument",
            }
        )
        bulk.append(row)
    pq.write_table(pa.Table.from_pylist(bulk), source / "bulk_proteomics.parquet")

    expression = []
    for source_id, backend, line, normalization in (
        ("ENCODE_GM12878_polyA_rnaseq", "encode_rnaseq", "GM12878", "tpm"),
        ("DepMap_24Q4_HAP1_gene", "depmap_rna", "HAP1", "log2_tpm_plus_1"),
    ):
        row = dict.fromkeys(EXPRESSION_FIELDS)
        row.update(
            {
                "backend": backend,
                "source_id": source_id,
                "reference": "source reference",
                "line_key": line,
                "parent_line_key": line,
                "granularity": "gene",
                "gene_id": "ENSG000001",
                "gene_name": "TEST",
                "profile_id": "profile" if line == "HAP1" else "",
                "tpm": 3.0,
                "log2_tpm": 2.0,
                "normalization": normalization,
                "quantifier": "rsem",
                "license": "source license",
            }
        )
        expression.append(row)
    pq.write_table(pa.Table.from_pylist(expression), source / "line_expression.parquet")

    mappings = []
    for protein, position in (("ENSP1", 1), ("ENSP2", 5)):
        row = dict.fromkeys(MAPPING_FIELDS)
        row.update(
            {
                "peptide": "PEPTIDE",
                "protein_id": protein,
                "gene_name": "TEST",
                "gene_id": "ENSG000001",
                "transcript_id": "ENST1",
                "is_canonical_transcript": protein == "ENSP1",
                "position": position,
                "n_flank": "AAA",
                "c_flank": "BBB",
                "proteome": "Homo sapiens",
                "proteome_source": "species",
            }
        )
        mappings.append(row)
    pq.write_table(pa.Table.from_pylist(mappings), source / "peptide_mappings.parquet")
    (source / "peptide_mappings_meta.json").write_text('{"n_rows":2}\n')
    manifest_path = source / "observations_meta.json"
    manifest = json.loads(manifest_path.read_text())
    for name in (
        "other_assays.parquet",
        "bulk_proteomics.parquet",
        "line_expression.parquet",
        "peptide_mappings.parquet",
        "peptide_mappings_meta.json",
    ):
        manifest["files"][name] = file_digest(source / name)
    manifest_path.write_text(json.dumps(manifest))


def _source_assets(tmp_path):
    root = tmp_path / "assets"
    (root / "bulk_proteomics").mkdir(parents=True)
    (root / "line_expression").mkdir()
    (root / "bulk_proteomics" / "sources.yaml").write_text(
        """sources:
  - source_id: Bekker-Jensen_2017
    digestion_enzyme: Trypsin/P legacy display
    search_enzyme: Trypsin/P
    instrument: source instrument
    ancillary_digests:
      - digestion_enzyme: Chymotrypsin
        maxquant_spec: Chymotrypsin+
  - source_id: CCLE_Nusinow_2020
    digestion_enzyme: Trypsin/P legacy display
    instrument: source instrument
"""
    )
    (root / "line_expression" / "sources.yaml").write_text(
        """sources:
  - source_id: ENCODE_GM12878_polyA_rnaseq
    backend: encode_rnaseq
    granularity: gene
    normalization: tpm
    quantifier: rsem
    build_status: packaged
    license: source license
  - source_id: DepMap_24Q4_HAP1_gene
    backend: depmap_rna
    granularity: gene
    normalization: log2_tpm_plus_1
    quantifier: rsem
    build_status: packaged
    license: source license
"""
    )
    (root / "line_expression_anchors.yaml").write_text(
        """lines:
  - name: GM12878
    aliases: [gm12878]
    expression_backend: encode_rnaseq
    expression_key: GM12878
    source_ids: [ENCODE_GM12878_polyA_rnaseq]
  - name: HAP1
    aliases: [hap1]
    expression_backend: depmap_rna
    expression_key: HAP1
    source_ids: [DepMap_24Q4_HAP1_gene]
  - name: HAP1-KO
    aliases: [hap1-ko]
    parent_line: HAP1
    expression_backend: none
    source_ids: []
"""
    )
    receipt = tmp_path / "receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "generation": "fixture",
                "assets": {
                    name: file_digest(root / name)
                    for name in (
                        "bulk_proteomics/sources.yaml",
                        "line_expression/sources.yaml",
                        "line_expression_anchors.yaml",
                    )
                },
            }
        )
    )
    return root, receipt


def _package(tmp_path):
    mhc, cohort, old, current, lineage = _cohort_fixture(
        tmp_path, augment_source=_add_supporting_native
    )
    from hitlist.context_evidence import write_context_evidence_package

    parent = tmp_path / "context"
    write_context_evidence_package(
        mhc,
        cohort,
        parent,
        historical_curation_path=old,
        current_curation_path=current,
        lineage_file=lineage,
    )
    root, receipt = _source_assets(tmp_path)
    target = tmp_path / "supporting"
    write_supporting_evidence_package(
        parent, target, source_root=root, source_asset_receipt=receipt
    )
    return target, parent, root, receipt


def test_supporting_package_preserves_grains_scales_and_mapping_alternatives(tmp_path):
    target, parent, root, receipt = _package(tmp_path)
    manifest = verify_supporting_evidence_package(target)
    counts = manifest["reconciliation"]
    assert counts["n_bulk_results"] == 4
    assert counts["n_bulk_peptide_results"] == 2
    assert counts["n_bulk_protein_results"] == 2
    assert counts["n_bulk_measurements"] == 6
    assert counts["n_expression_results"] == 2
    assert counts["n_expression_measurements"] == 4
    assert counts["n_mapping_occurrences"] == 2
    assert counts["n_raw_input_peptides"] == 3
    assert counts["n_eligible_input_peptides"] == 2
    assert counts["n_mapped_input_peptides"] == 1
    assert counts["n_unmapped_input_peptides"] == 1
    assert counts["n_mapping_peptides_without_eligible_input"] == 0
    assert counts["n_line_anchor_claims"] == 3
    assert counts["n_line_anchors_with_v4_measurement"] == 2
    assert file_digest(target / "manifests" / "context.json") == file_digest(
        parent / "manifest.json"
    )
    assert file_digest(target / "assets" / "receipt.json") == file_digest(receipt)
    assert file_digest(target / "assets" / "bulk_proteomics" / "sources.yaml") == (
        file_digest(root / "bulk_proteomics" / "sources.yaml")
    )
    bulk = pq.read_table(target / "typed" / "bulk_results.parquet").to_pylist()
    assert all(row["material_link_status"] == "cell_line_label_only" for row in bulk)
    assert bulk[0]["search_enzyme"] == "Trypsin/P"
    assert bulk[0]["digestion_enzyme_raw"] == "Trypsin/P legacy display"
    assert bulk[1]["search_enzyme"] == "Chymotrypsin+"
    assert bulk[3]["search_enzyme_status"] == "unreported"
    measures = pq.read_table(target / "typed" / "bulk_measurements.parquet").to_pylist()
    assert {row["quantity_scale"] for row in measures} == {
        "scoped_detection",
        "label_free_source_index",
        "tmt_panel_relative",
        "within_source_rank",
    }
    assert all(
        row["value_status"] == "derived_in_source_index"
        for row in measures
        if row["representation"] == "source_derived_percentile"
    )
    assert not any(row["representation"] == "peptide_intensity" for row in measures)
    expression = pq.read_table(target / "typed" / "expression_measurements.parquet").to_pylist()
    assert sum(row["value_status"] == "source_reported" for row in expression) == 2
    coverage = pq.read_table(target / "typed" / "peptide_mapping_coverage.parquet").to_pylist()
    assert coverage == [
        {"peptide_sequence": "PEPTIDE", "n_mapping_occurrences": 2, "mapping_status": "mapped"},
        {"peptide_sequence": "UNMAPPED", "n_mapping_occurrences": 0, "mapping_status": "unmapped"},
    ]
    anchors = pq.read_table(target / "typed" / "line_anchor_claims.parquet").to_pylist()
    assert anchors[2]["material_link_precision"] == "parent_line_surrogate"
    assert not anchors[2]["has_v4_measurement"]


def test_supporting_package_rejects_source_tampering(tmp_path):
    target, _, _, _ = _package(tmp_path)
    with (target / "typed" / "bulk_results.parquet").open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ValueError, match="artifact mismatch"):
        verify_supporting_evidence_package(target)


def test_supporting_manifest_remains_verifiable_inside_successor(tmp_path):
    target, _, _, _ = _package(tmp_path)
    nested = target / "manifests" / "supporting.json"
    nested.write_bytes((target / "manifest.json").read_bytes())
    assert (
        verify_supporting_evidence_package(target, manifest_path="manifests/supporting.json")[
            "reconciliation"
        ]["n_bulk_results"]
        == 4
    )
    with pytest.raises(ValueError, match="file inventory"):
        verify_supporting_evidence_package(target)
    with pytest.raises(ValueError, match="Invalid supporting package manifest path"):
        verify_supporting_evidence_package(target, manifest_path="../manifest.json")


def test_supporting_generation_snapshots_ingestion_assets(tmp_path):
    target, _, root, receipt = _package(tmp_path)
    (root / "line_expression_anchors.yaml").write_text("lines: []\n")
    receipt.write_text("changed after publication\n")
    verify_supporting_evidence_package(target)


def test_supporting_writer_rejects_unreviewed_source_asset(tmp_path):
    _, parent, root, receipt = _package(tmp_path)
    (root / "line_expression_anchors.yaml").write_text("lines: []\n")
    with pytest.raises(ValueError, match="Supporting source asset mismatch"):
        write_supporting_evidence_package(
            parent,
            tmp_path / "tampered",
            source_root=root,
            source_asset_receipt=receipt,
        )
