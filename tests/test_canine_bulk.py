"""Offline source contracts for bulk canine evidence, not DLA ligands."""

import copy
import csv
import io
import json
from zipfile import ZipFile

import pytest


def tsv(rows):
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=list(rows[0]), delimiter="\t")
    writer.writeheader()
    writer.writerows(rows)
    return out.getvalue()


@pytest.fixture
def sources(tmp_path, monkeypatch):
    from hitlist import canine_bulk
    from hitlist.provenance import file_digest

    fasta = tmp_path / "reference.fasta"
    fasta.write_text(">P1 canine OX=9615\nMACDEFGHIK\n>P2 canine OX=9615\nMACDEFGHIK\n")
    archive = tmp_path / "atlas.zip"
    tables = {
        "sample": [
            {"sample_id": "S1", "sample_title": "Heart", "instrument_model_id": "63"},
            {"sample_id": "S2", "sample_title": "Brain", "instrument_model_id": "63"},
        ],
        "atlas_search_batch": [
            {"atlas_search_batch_id": "B1", "sample_id": "S1", "n_runs": "2"},
            {"atlas_search_batch_id": "B2", "sample_id": "S1", "n_runs": "1"},
            {"atlas_search_batch_id": "B3", "sample_id": "S2", "n_runs": "1"},
        ],
        "peptide": [{"peptide_id": "P", "peptide_sequence": "ACDEFGHIK"}],
        "peptide_instance": [
            {"peptide_instance_id": "I", "peptide_id": "P", "best_probability": "0.999"}
        ],
        "peptide_instance_search_batch": [
            {
                "peptide_instance_search_batch_id": str(i),
                "peptide_instance_id": "I",
                "atlas_search_batch_id": b,
                "n_observations": "3",
            }
            for i, b in enumerate(["B1", "B2", "B3"], 1)
        ],
        "biosequence": [
            {"biosequence_id": str(i), "biosequence_name": f"P{i}", "biosequence_seq": "MACDEFGHIK"}
            for i in [1, 2]
        ],
        "peptide_mapping": [
            {
                "peptide_mapping_id": str(i),
                "peptide_instance_id": "I",
                "matched_biosequence_id": str(i),
                "start_in_biosequence": "2",
                "end_in_biosequence": "10",
            }
            for i in [1, 2]
        ],
        "protein_identification": [
            {
                "protein_identification_id": str(i),
                "biosequence_id": str(i),
                "protein_group_number": "G1",
                "probability": "0.999",
                "n_observations": "9",
            }
            for i in [1, 2]
        ],
        "modified_peptide_instance": [
            {
                "modified_peptide_instance_id": "M",
                "peptide_instance_id": "I",
                "modified_peptide_sequence": "AC[160]DEFGHIK",
                "peptide_charge": "2",
                "best_probability": "0.999",
            }
        ],
        "modified_peptide_instance_search_batch": [
            {
                "modified_peptide_instance_search_batch_id": "1",
                "modified_peptide_instance_id": "M",
                "atlas_search_batch_id": "B1",
                "n_observations": "2",
            }
        ],
    }
    with ZipFile(archive, "w") as z:
        for name, rows in tables.items():
            z.writestr(name + ".tsv", tsv(rows))
    proteins = tmp_path / "proteins.tsv"
    proteins.write_text(
        tsv(
            [
                {
                    "Run": run,
                    "Protein": "P1",
                    "ProteinGroup": "P1",
                    "Abundance": "3.5",
                    "IBAQ": "12",
                    "rIBAQ": "0.5",
                    "sumIBAQ": "24",
                    "BioReplicate": "pool-heart",
                    "Condition": "Heart",
                    "Mixture": "Pool1",
                    "Channel": "126",
                    "TechRepMixture": rep,
                    "sample.type": "sample",
                }
                for rep, run in [("1", "Pool1_1"), ("2", "Pool1_2")]
            ]
        )
    )
    catalog = copy.deepcopy(canine_bulk.canine_bulk_catalog())
    local = {}
    for key, source in catalog["sources"].items():
        local[key] = {}
        for name, asset in source["assets"].items():
            path = (
                fasta
                if name == "reference"
                else archive
                if source["adapter"] == "peptideatlas"
                else proteins
            )
            asset.update(file_digest(path))
            local[key][name] = path
    monkeypatch.setattr(canine_bulk, "canine_bulk_catalog", lambda: catalog)
    return local


def test_atlas_preserves_aggregate_scope_and_all_alternatives(sources, tmp_path):
    from hitlist.assays import positive_ms_mask
    from hitlist.canine_bulk import curate_canine_bulk, load_bulk_evidence, verify_bulk_evidence
    from hitlist.tissue_blacklist import build_tissue_blacklist

    target = tmp_path / "atlas"
    curate_canine_bulk("labrador_2024_01", target, local_assets=sources["labrador_2024_01"])
    assert verify_bulk_evidence(target)["source"]["source_taxon"] == 9615
    peptides = load_bulk_evidence(target)
    assert len(peptides) == 3
    assert peptides.source_record_id.nunique() == 3
    assert peptides.sample_id.nunique() == 2
    assert peptides.donor_id.eq("").all()
    assert peptides.donor_status.eq("aggregate_unresolved").all()
    assert peptides.developmental_stage.str.contains("stillborn").all()
    assert peptides.health_context.str.contains("hip dysplasia").all()
    assert not positive_ms_mask(peptides).any()
    risk, audit = build_tissue_blacklist(peptides)
    assert risk.n_donors.tolist() == [0] and not risk.blacklisted.any()
    assert audit.exclusion_reason.eq("not_positive_ms").all()
    mappings = load_bulk_evidence(target, table="mappings")
    assert set(mappings.protein_id) == {"P1", "P2"}
    assert mappings.peptide_instance_id.nunique() == 1
    assert len(load_bulk_evidence(target, table="proteins")) == 2
    assert load_bulk_evidence(target, table="modifications").modified_peptide.tolist() == [
        "AC[160]DEFGHIK"
    ]
    raw = load_bulk_evidence(target, table="raw_peptide_mapping")
    assert raw.matched_biosequence_id.tolist() == ["1", "2"]


def test_pfizer_pools_are_not_donors_or_peptide_detections(sources, tmp_path):
    from hitlist.canine_bulk import curate_canine_bulk, load_bulk_evidence

    target = tmp_path / "pfizer"
    curate_canine_bulk("pfizer_2026_03", target, local_assets=sources["pfizer_2026_03"])
    frame = load_bulk_evidence(target, table="proteins")
    assert len(frame) == 2
    assert frame.sample_id.nunique() == 1
    assert frame.donor_id.eq("").all() and frame.donor_status.eq("pooled").all()
    assert frame.ibaq.tolist() == [12.0, 12.0]
    assert frame.protein_group_resolution.eq("reported_representative_only").all()
    assert load_bulk_evidence(target).empty
    assert len(load_bulk_evidence(target, table="samples")) == 1


def test_pfizer_conflicting_instrument_reports_survive_curation(sources, tmp_path):
    from hitlist.canine_bulk import curate_canine_bulk, load_bulk_evidence, verify_bulk_evidence

    target = tmp_path / "instrument-conflict"
    curate_canine_bulk("pfizer_2026_03", target, local_assets=sources["pfizer_2026_03"])
    source = verify_bulk_evidence(target)["source"]
    assert source["instrument_status"] == "conflicting_reports"
    reports = source["instrument_reports"]
    assert reports["publication"]["value"] == "Orbitrap Fusion Eclipse"
    assert reports["publication"]["url"] == "https://www.nature.com/articles/s41597-026-07512-9"
    assert reports["deposit"]["value"] == "Orbitrap Fusion Lumos"
    assert (
        reports["deposit"]["url"]
        == "https://proteomecentral.proteomexchange.org/cgi/GetDataset?ID=PXD070290"
    )
    assert source["instrument"].startswith("unresolved")
    assert "Eclipse" in source["instrument"] and "Lumos" in source["instrument"]
    assert load_bulk_evidence(target, table="proteins").instrument.eq(source["instrument"]).all()


def test_sample_annotations_follow_reviewed_catalog(sources, tmp_path):
    from hitlist.canine_bulk import (
        canine_bulk_catalog,
        curate_canine_bulk,
        load_bulk_evidence,
        verify_bulk_evidence,
    )

    source_id = "pfizer_2026_03"
    rewrite_source(sources, source_id, lambda text: text.replace("Heart", "ReviewedHeart"))
    source = canine_bulk_catalog()["sources"][source_id]
    source["tissue_groups"] = {"reviewedheart": "heart"}
    source["default_sample_context"] = "pool"
    source["sample_context_by_id"] = {"pool-heart": "reviewed"}
    source["sample_contexts"] = {
        "pool": {
            "donor_id": "",
            "donor_status": "pooled",
            "developmental_stage": "unknown",
            "health_context": "reported normal tissue pools",
            "tissue_status": "nonmalignant",
            "is_cell_line": False,
        }
    }
    source["sample_contexts"]["reviewed"] = {
        **source["sample_contexts"]["pool"],
        "developmental_stage": "reviewed stage",
        "health_context": "reviewed health context",
    }
    target = tmp_path / "reviewed"
    curate_canine_bulk(source_id, target, local_assets=sources[source_id])
    sample = load_bulk_evidence(target, table="samples").iloc[0]
    assert sample.source_tissue == "ReviewedHeart"
    assert sample.tissue_group == "heart"
    assert sample.developmental_stage == "reviewed stage"
    assert sample.health_context == "reviewed health context"
    assert not sample.is_cell_line
    frozen = verify_bulk_evidence(target)["source"]
    assert frozen["sample_contexts"] == source["sample_contexts"]
    assert frozen["sample_context_by_id"] == source["sample_context_by_id"]


def test_sample_label_does_not_infer_adult_cohort(sources, tmp_path):
    from hitlist.canine_bulk import curate_canine_bulk, load_bulk_evidence

    source_id = "labrador_2024_01"
    rewrite_source(
        sources, source_id, lambda text: text.replace("Heart", "Plasma_neonate"), "sample.tsv"
    )
    target = tmp_path / "neonate"
    curate_canine_bulk(source_id, target, local_assets=sources[source_id])
    sample = load_bulk_evidence(target, table="samples").set_index("source_sample_id").loc["S1"]
    assert "stillborn" in sample.developmental_stage
    assert "hip dysplasia" in sample.health_context
    assert sample.tissue_group == ""


@pytest.mark.parametrize("sample_id", ["10478", "10479", "10480", "10481", "10829"])
def test_reviewed_adult_sample_ids_select_context(sources, tmp_path, sample_id):
    from hitlist.canine_bulk import curate_canine_bulk, load_bulk_evidence

    source_id = "labrador_2024_01"
    for member in ("sample.tsv", "atlas_search_batch.tsv"):
        rewrite_source(sources, source_id, lambda text: text.replace("S1", sample_id), member)
    target = tmp_path / "adult"
    curate_canine_bulk(source_id, target, local_assets=sources[source_id])
    samples = load_bulk_evidence(target, table="samples").set_index("source_sample_id")
    assert samples.loc[sample_id].developmental_stage == "adult; 1-8 years"
    assert "separate from tissue donors" in samples.loc[sample_id].health_context
    assert "stillborn" in samples.loc["S2"].developmental_stage
    assert samples.donor_id.eq("").all()


def test_sample_curation_catalog_is_complete():
    from hitlist.canine_bulk import _SAMPLE_COLUMNS, canine_bulk_catalog

    for source in canine_bulk_catalog()["sources"].values():
        contexts = source["sample_contexts"]
        assert source["default_sample_context"] in contexts
        for context in contexts.values():
            assert set(context) == set(_SAMPLE_COLUMNS[4:])
            assert type(context["is_cell_line"]) is bool
            assert all(isinstance(context[name], str) for name in _SAMPLE_COLUMNS[4:-1])
        for sample_id, context_name in source.get("sample_context_by_id", {}).items():
            assert isinstance(sample_id, str) and sample_id
            assert context_name in contexts
        for tissue, group in source["tissue_groups"].items():
            assert isinstance(tissue, str) and tissue == tissue.lower()
            assert isinstance(group, str) and group


def test_checked_inputs_atomic_output_and_bounded_query(sources, tmp_path):
    from hitlist.canine_bulk import curate_canine_bulk, load_bulk_evidence

    target = tmp_path / "atlas"
    assets = sources["labrador_2024_01"]
    with pytest.raises(ValueError, match="max_input_bytes"):
        curate_canine_bulk(
            "labrador_2024_01", target, local_assets=assets, limits={"max_input_bytes": 1}
        )
    assert not target.exists()
    curate_canine_bulk("labrador_2024_01", target, local_assets=assets)
    with pytest.raises(ValueError, match="max_rows"):
        load_bulk_evidence(target, max_rows=2)
    assert len(load_bulk_evidence(target, tissue="brain", max_rows=1)) == 1
    with pytest.raises(FileExistsError):
        curate_canine_bulk("labrador_2024_01", target, local_assets=assets)
    assets["reference"].write_text("changed")
    with pytest.raises(ValueError, match=r"checksum|size|integrity"):
        curate_canine_bulk("labrador_2024_01", tmp_path / "bad", local_assets=assets)
    assert not (tmp_path / "bad").exists()


def test_bundle_tamper_detection(sources, tmp_path):
    from hitlist.canine_bulk import curate_canine_bulk, verify_bulk_evidence

    target = tmp_path / "atlas"
    curate_canine_bulk("labrador_2024_01", target, local_assets=sources["labrador_2024_01"])
    manifest = json.loads((target / "manifest.json").read_text())
    assert manifest["presentation_coverage"] == "missing"
    (target / "reference.fasta").write_text("changed")
    with pytest.raises(ValueError, match="artifact"):
        verify_bulk_evidence(target)


def rewrite_source(sources, source_id, transform, member=None):
    from hitlist.canine_bulk import canine_bulk_catalog
    from hitlist.provenance import file_digest

    path = sources[source_id]["table"]
    if member:
        with ZipFile(path) as archive:
            contents = {name: archive.read(name) for name in archive.namelist()}
        contents[member] = transform(contents[member].decode()).encode()
        with ZipFile(path, "w") as archive:
            for name, content in contents.items():
                archive.writestr(name, content)
    else:
        path.write_text(transform(path.read_text()))
    canine_bulk_catalog()["sources"][source_id]["assets"]["table"].update(file_digest(path))


def test_missing_modified_counts_remain_missing(sources, tmp_path):
    from hitlist.canine_bulk import curate_canine_bulk, load_bulk_evidence

    rewrite_source(
        sources,
        "labrador_2024_01",
        lambda s: s.replace("B1\t2", "B1\t\\N"),
        "modified_peptide_instance_search_batch.tsv",
    )
    target = tmp_path / "missing-count"
    curate_canine_bulk("labrador_2024_01", target, local_assets=sources["labrador_2024_01"])
    assert load_bulk_evidence(target, table="modifications").n_reported_observations.isna().all()


@pytest.mark.parametrize("sentinel", ["-Inf", " +Inf ", " NaN ", "\t-Inf\t"])
def test_nonfinite_abundance_retains_original_cell_without_becoming_zero(
    sources, tmp_path, sentinel
):
    from hitlist.canine_bulk import curate_canine_bulk, load_bulk_evidence

    rewrite_source(sources, "pfizer_2026_03", lambda s: s.replace("3.5", f'"{sentinel}"'))
    target = tmp_path / "nonfinite"
    curate_canine_bulk("pfizer_2026_03", target, local_assets=sources["pfizer_2026_03"])
    frame = load_bulk_evidence(target, table="proteins")
    assert frame.reported_abundance.isna().all()
    assert frame.reported_abundance_status.eq("reported_nonfinite").all()
    assert load_bulk_evidence(target, table="raw_proteins").Abundance.eq(sentinel).all()


def test_overflowing_abundance_fails_instead_of_claiming_finite_measurement(sources, tmp_path):
    from hitlist.canine_bulk import curate_canine_bulk

    rewrite_source(sources, "pfizer_2026_03", lambda s: s.replace("3.5", "1e1000"))
    with pytest.raises(ValueError, match="finite"):
        curate_canine_bulk(
            "pfizer_2026_03", tmp_path / "overflow", local_assets=sources["pfizer_2026_03"]
        )
    assert not (tmp_path / "overflow").exists()


def test_bulk_semantics_survive_rehashed_database_tamper(sources, tmp_path):
    import sqlite3

    from hitlist.canine_bulk import curate_canine_bulk, verify_bulk_evidence
    from hitlist.provenance import file_digest

    target = tmp_path / "tampered-semantics"
    curate_canine_bulk("labrador_2024_01", target, local_assets=sources["labrador_2024_01"])
    with sqlite3.connect(target / "bulk.sqlite") as db:
        sql = db.execute("SELECT sql FROM sqlite_master WHERE name='peptides'").fetchone()[0]
        db.execute("DROP VIEW peptides")
        db.execute(
            sql.replace("'bulk_proteomics' AS assay_modality", "'mhc_ligand' AS assay_modality")
        )
    manifest = json.loads((target / "manifest.json").read_text())
    manifest["artifacts"]["bulk.sqlite"] = file_digest(target / "bulk.sqlite")
    (target / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="view semantics"):
        verify_bulk_evidence(target)


def test_decoy_and_contaminant_only_matches_are_not_bulk_target_detections(sources, tmp_path):
    from hitlist.canine_bulk import canine_bulk_catalog, curate_canine_bulk, load_bulk_evidence
    from hitlist.provenance import file_digest

    rewrite_source(
        sources,
        "labrador_2024_01",
        lambda s: s.replace("P1", "DECOY_P1").replace("P2", "CONTAM_P2"),
        "biosequence.tsv",
    )
    path = sources["labrador_2024_01"]["reference"]
    path.write_text(path.read_text().replace(">P1", ">DECOY_P1").replace(">P2", ">CONTAM_P2"))
    canine_bulk_catalog()["sources"]["labrador_2024_01"]["assets"]["reference"].update(
        file_digest(path)
    )
    target = tmp_path / "controls"
    curate_canine_bulk("labrador_2024_01", target, local_assets=sources["labrador_2024_01"])
    frame = load_bulk_evidence(target)
    assert not frame.is_bulk_detection.any()
    assert frame.n_target_mappings.eq(0).all()
    assert frame.n_decoy_mappings.eq(1).all()
    assert frame.n_contaminant_mappings.eq(1).all()
    assert frame.qualitative_measurement.eq("").all()


@pytest.mark.parametrize(
    "limit", ["max_output_bytes", "max_expanded_bytes", "max_records", "max_protein_residues"]
)
def test_resource_failure_never_publishes_partial_bundle(sources, tmp_path, limit):
    from hitlist.canine_bulk import curate_canine_bulk

    target = tmp_path / "limited"
    with pytest.raises(ValueError, match=limit):
        curate_canine_bulk(
            "labrador_2024_01", target, local_assets=sources["labrador_2024_01"], limits={limit: 1}
        )
    assert not target.exists() and not list(tmp_path.glob(".limited-*"))


def test_broken_source_relationship_fails_without_silently_losing_rows(sources, tmp_path):
    from hitlist.canine_bulk import curate_canine_bulk

    rewrite_source(
        sources,
        "labrador_2024_01",
        lambda s: s.replace("I\tB1", "UNKNOWN\tB1"),
        "peptide_instance_search_batch.tsv",
    )
    with pytest.raises(ValueError, match="relationship"):
        curate_canine_bulk(
            "labrador_2024_01", tmp_path / "broken", local_assets=sources["labrador_2024_01"]
        )
    assert not (tmp_path / "broken").exists()


def test_acquisition_is_explicit_and_cache_hits_are_verified(sources, tmp_path, monkeypatch):
    from hitlist import canine_bulk, downloads

    monkeypatch.setattr(downloads, "_override_data_dir", tmp_path / "cache")
    calls = []

    def fetch(url, **options):
        calls.append(url)
        destination = options["destination"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(sources["pfizer_2026_03"]["reference"].read_bytes())
        options["progress_callback"](destination.stat().st_size, destination.stat().st_size)

    monkeypatch.setattr(canine_bulk.datacache, "fetch_file", fetch)
    with pytest.raises(ValueError, match="explicitly"):
        canine_bulk.curate_canine_bulk("pfizer_2026_03", tmp_path / "implicit")
    assert calls == []
    path = canine_bulk.fetch_canine_bulk_asset("pfizer_2026_03", "reference")
    assert canine_bulk.fetch_canine_bulk_asset("pfizer_2026_03", "reference") == path
    assert len(calls) == 1
    path.write_text("corrupt")
    with pytest.raises(ValueError, match="checksum"):
        canine_bulk.fetch_canine_bulk_asset("pfizer_2026_03", "reference")
    assert len(calls) == 1


def test_bulk_source_catalog_opt_in_preserves_existing_defaults():
    from hitlist.bulk_proteomics import load_bulk_sources

    defaults = load_bulk_sources()
    expanded = load_bulk_sources(include_canine=True)
    assert expanded[: len(defaults)] == defaults
    assert {s["source_id"] for s in expanded[len(defaults) :]} == {
        "pfizer_2026_03",
        "labrador_2024_01",
    }
