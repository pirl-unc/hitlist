"""Tests for #947 (T-cell pin/parity reconciliation, SPR separation, rich immunogenicity)
and foundational #943/#944 reference MHC genotype registry.
"""

from __future__ import annotations

import csv
import hashlib
import io
import zipfile

import pyarrow.parquet as pq
import pytest
import yaml

import hitlist
from hitlist.mhc_genotypes import (
    load_reference_mhc_genotypes,
    lookup_cell_line_genotype,
    lookup_strain_genotype,
    normalize_mhc_token,
    parse_mhc_allele_list,
    resolve_condition_mhc_genotype,
)
from hitlist.tcell_assays import (
    CEDAR_NAME,
    IEDB_NAME,
    SOURCE_FIELDS,
    assay_family,
    iter_tcell_assay_metadata,
    write_tcell_assay_metadata,
)
from hitlist.tcell_immunogenicity import (
    BIOPHYSICS_SCHEMA,
    IMMUNOGENICITY_SCHEMA,
    classify_antigen_delivery_mode,
    classify_apc_lineage,
    classify_effector_repertoire,
    classify_immunogenicity_modality,
    iter_tcell_immunogenicity_assays,
    verify_tcell_immunogenicity_bundle,
    write_tcell_immunogenicity_bundle,
)
from hitlist.tcell_training import (
    verify_human_tcell_training_bundle,
    write_human_tcell_training_bundle,
)

_EXTENDED_HEADERS = (
    "MHC Restriction | Name",
    "Epitope | Name",
    "Assay Antigen | Name",
    "Epitope | Source Organism IRI",
    "Epitope | Source Organism",
    "Epitope | Object Type",
    "Epitope | Modified residues",
    "Epitope | Modifications",
    "Epitope | Starting Position",
    "Epitope | Ending Position",
    "Epitope | Related Object",
    "Assay Antigen | Epitope Relation",
    "Assay Antigen | Source Organism IRI",
    "Assay Antigen | Source Organism",
    "Assay | Response Frequency (%)",
    "Host | MHC Present",
    "1st in vivo Process | Process Type",
    "1st in vivo Process | Disease",
    "Effector Cell | TCR Name",
)


def _write_tcell_zip(path, records, *, cedar=False):
    source = "CEDAR" if cedar else "IEDB"
    headers = [
        f"Assay ID | {source} IRI",
        f"Reference | {source} IRI",
        *SOURCE_FIELDS.values(),
        *_EXTENDED_HEADERS,
    ]
    if cedar:
        headers.append("Epitope | Mutation")
    headers = list(dict.fromkeys(headers))
    buf = io.StringIO(newline="")
    writer = csv.writer(buf)
    writer.writerow([h.split(" | ")[0] for h in headers])
    writer.writerow([h.split(" | ")[1] for h in headers])
    for assay_id, overrides in records:
        row = {
            f"Assay ID | {source} IRI": f"https://example.org/assay/{assay_id}",
            f"Reference | {source} IRI": "https://example.org/reference/100",
            "Reference | PMID": "999999",
            "Assay | Method": "ELISPOT",
            "Assay | Response measured": "IFNg release",
            "Assay | Qualitative Measurement": "Positive-High",
            "Assay | Quantitative measurement": "250",
            "Assay | Measurement Inequality": "=",
            "Assay | Units": "SFC/10^6",
            "Assay | Number of Subjects Tested": "10",
            "Assay | Number of Subjects Positive": "7",
            "Assay | Response Frequency (%)": "70.0",
            "Assay Antigen | Object Type": "Linear peptide",
            "Host | Name": "Homo sapiens",
            "MHC Restriction | Name": "HLA-A*02:01",
            "MHC Restriction | Evidence Code": "Single allele present",
            "MHC Restriction | Class": "I",
            "Epitope | Name": "GILGFVFTL",
            "Assay Antigen | Name": "GILGFVFTL",
            "Epitope | Source Organism IRI": "http://purl.obolibrary.org/obo/NCBITaxon_9606",
            "Epitope | Source Organism": "Homo sapiens",
            "Epitope | Object Type": "Linear peptide",
            "Assay Antigen | Epitope Relation": "Epitope",
            "Assay Antigen | Source Organism IRI": "http://purl.obolibrary.org/obo/NCBITaxon_9606",
            "Assay Antigen | Source Organism": "Homo sapiens",
        }
        row.update(overrides)
        writer.writerow([row.get(h, "") for h in headers])
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("tcell_full_v3.csv", buf.getvalue())


def _make_sources(tmp_path, iedb_records, cedar_records, *, drift_pin=False):
    iedb_path = tmp_path / IEDB_NAME
    cedar_path = tmp_path / CEDAR_NAME
    _write_tcell_zip(iedb_path, iedb_records, cedar=False)
    _write_tcell_zip(cedar_path, cedar_records, cedar=True)
    catalog_path = tmp_path / "catalog.yaml"
    files = {}
    for name, path in ((IEDB_NAME, iedb_path), (CEDAR_NAME, cedar_path)):
        files[name] = {
            "url": f"https://example.org/{name}",
            "sha256": ("0" * 64 if drift_pin else hashlib.sha256(path.read_bytes()).hexdigest()),
            "size": (path.stat().st_size + 42) if drift_pin else path.stat().st_size,
        }
    catalog_path.write_text(
        yaml.safe_dump({"schema_version": 1, "observed": "2026-10-20", "files": files})
    )
    return {"iedb_path": iedb_path, "cedar_path": cedar_path, "catalog_path": catalog_path}


# ── Reference MHC Genotype Registry (#943, #944, #947) ───────────────────────


def test_reference_mhc_genotypes_yaml_all_alleles_parse_as_exact():
    data = load_reference_mhc_genotypes()
    assert len(data["cell_lines"]) >= 25
    assert len(data["strains"]) >= 15
    for section in ("cell_lines", "strains"):
        for name, entry in data[section].items():
            for field in ("class_i_alleles", "class_ii_alleles", "transgenic_alleles"):
                for allele in entry.get(field) or ():
                    norm = normalize_mhc_token(allele)
                    assert norm.is_exact_molecule, f"{section}.{name}.{field}: {allele!r} not exact"
                    assert norm.canonical == allele, (
                        f"{section}.{name}.{field}: {allele!r} != canonical {norm.canonical!r}"
                    )


def test_rodent_allele_normalization_and_list_parsing():
    assert normalize_mhc_token("H2-D*b").canonical == "H2-Db"
    assert normalize_mhc_token("H2-D*b").is_exact_molecule is True
    assert normalize_mhc_token("H2-Kb").canonical == "H2-Kb"
    assert normalize_mhc_token("H2-IAb").canonical == "H2-IAb"
    assert normalize_mhc_token("H2-IAb").is_exact_molecule is True
    assert normalize_mhc_token("RT1-Al").canonical == "RT1-Al"
    assert normalize_mhc_token("RT1-Al").is_exact_molecule is True
    # Two-digit HLA group is NOT an exact molecule.
    assert normalize_mhc_token("HLA-A*02").is_exact_molecule is False

    parsed = parse_mhc_allele_list("HLA-A*02:01; HLA-DRB1*04:01; H2-D*b", mhc_class_hint="I")
    assert parsed == ("H2-Db", "HLA-A*02:01")


def test_cell_line_and_strain_genotype_safeguards():
    # Class-I null hosts must NEVER impute endogenous HLA-I alleles.
    assert lookup_cell_line_genotype("K562", mhc_class="I") is None
    assert lookup_cell_line_genotype("721.221", mhc_class="I") is None
    # 721.221 retains endogenous Class II when Class II is requested.
    match_721_ii = lookup_cell_line_genotype("721.221", mhc_class="II")
    assert match_721_ii is not None
    assert "HLA-DRB1*01:01" in match_721_ii.alleles
    # C1R trace endogenous Class I is opt-in only.
    assert lookup_cell_line_genotype("C1R", mhc_class="I") is None
    assert lookup_cell_line_genotype("C1R", mhc_class="I", allow_low_endogenous=True) is not None

    # Canonical human & rodent cell lines resolve cleanly including hybrid IEDB names.
    hela = lookup_cell_line_genotype("HeLa cells-Epithelial cell", mhc_class="I")
    assert hela is not None
    assert hela.alleles == ("HLA-A*68:02", "HLA-B*15:03", "HLA-C*12:03")

    hap1 = lookup_cell_line_genotype("HAP1", mhc_class="I")
    assert hap1 is not None
    assert hap1.alleles == ("HLA-A*02:01", "HLA-B*40:01", "HLA-C*03:04")

    # Chimeric HHD vs standard HLA-A*02:01 Tg vs inbred C57BL/6.
    hhd = lookup_strain_genotype("Mus musculus HLA-A2/HHD", mhc_class="I")
    assert hhd is not None
    assert hhd.alleles == ("HLA-A*02:01",)
    assert hhd.mhc_species == "Homo sapiens"
    assert hhd.host_species == "Mus musculus"

    b6 = lookup_strain_genotype("Mus musculus C57BL/6J", mhc_class="I")
    assert b6 is not None
    assert b6.alleles == ("H2-Db", "H2-Kb")

    lewis = lookup_strain_genotype("Rattus norvegicus Lewis", mhc_class="I")
    assert lewis is not None
    assert lewis.alleles == ("RT1-Al",)


def test_resolve_condition_mhc_genotype_never_conflates_prediction_with_experimental():
    res = resolve_condition_mhc_genotype(
        reported_mhc_name="HLA-A*02:01",
        restriction_evidence_code="MHC binding prediction",
        reported_mhc_class="I",
        host_mhc_types="HLA-A*02:01;HLA-A*24:02;HLA-B*07:02",
        host_name="Homo sapiens",
    )
    assert res.is_predicted_restriction is True
    assert res.experimental_mhc_allele is None
    assert res.mhc_allele_set == "HLA-A*02:01;HLA-A*24:02;HLA-B*07:02"
    assert res.mhc_allele_cardinality == "multi_allelic_bag"
    assert res.mhc_allele_set_provenance == "host_mhc_types"


# ── #947 Bug 1 & Bug 2: Pin/Parity Reconciliation + SPR Biophysics Routing ───


def test_assay_family_separates_non_cellular_tcr_pmhc_biophysics():
    assert assay_family("surface plasmon resonance", "dissociation constant KD") == (
        "tcr_pmhc_biophysics"
    )
    assert assay_family("biolayer interferometry", "off rate") == "tcr_pmhc_biophysics"
    assert assay_family("X-ray crystallography", "3D structure") == "tcr_pmhc_biophysics"
    assert assay_family("in vitro assay", "on rate") == "tcr_pmhc_biophysics"
    assert assay_family("multimer/tetramer", "qualitative binding") == "pmhc_multimer"
    assert assay_family("ELISPOT", "IFNg release") == "soluble_factor_release"
    # Historical v5 compatibility mode preserves legacy bucket.
    assert (
        assay_family(
            "surface plasmon resonance",
            "dissociation constant KD",
            separate_biophysics=False,
        )
        == "other_tcell_assay"
    )


def test_reconcile_mode_records_pin_drift_parity_diffs_and_cedar_only_ids(tmp_path):
    iedb = [
        ("1", {"Assay | Qualitative Measurement": "Positive-High"}),
        (
            "2",
            {
                "Assay | Qualitative Measurement": "Negative",
                "Assay | Number of Subjects Positive": "0",
            },
        ),
    ]
    cedar = [
        # Asynchronous mirror drift on assay 1 + CEDAR-only assay 99
        ("1", {"Assay | Qualitative Measurement": "Positive-Low"}),
        ("99", {}),
    ]
    paths = _make_sources(tmp_path, iedb, cedar, drift_pin=True)

    # Pinned mode remains strict by default.
    with pytest.raises(ValueError, match="Source size differs"):
        list(iter_tcell_assay_metadata(**paths))

    # Reconcile mode succeeds and records structured provenance.
    meta_out = tmp_path / "metadata"
    meta_manifest = write_tcell_assay_metadata(
        meta_out, **paths, source_verification_mode="reconcile"
    )
    assert meta_manifest["source_verification_mode"] == "reconcile"
    rec = meta_manifest["source_reconciliation"]
    assert rec["pin_status"][IEDB_NAME] == "drifted"
    assert rec["pin_status"][CEDAR_NAME] == "drifted"
    assert rec["n_parity_diffs"] == 1
    assert rec["parity_diff_assay_ids_sample"] == ["1"]
    assert rec["n_cedar_only_assays"] == 1
    assert rec["cedar_only_assay_ids_sample"] == ["99"]

    # Training bundle v1 also supports reconcile mode & verifies cleanly.
    train_out = tmp_path / "train_v1"
    train_manifest = write_human_tcell_training_bundle(
        train_out, **paths, source_verification_mode="reconcile"
    )
    assert train_manifest["source_verification_mode"] == "reconcile"
    assert train_manifest["n_training_examples"] == 2
    verified = verify_human_tcell_training_bundle(train_out, catalog_path=paths["catalog_path"])
    assert verified == train_manifest


# ── #947 Deliverable 3 + #944 Multi-Species Immunogenicity Bundle ────────────


def test_immunogenicity_classifiers():
    assert classify_immunogenicity_modality(
        "soluble_factor_release", "ELISPOT", "IFNg release"
    ) == (
        "th1_cytokine",
        "IFNG",
    )
    assert classify_immunogenicity_modality(
        "soluble_factor_release", "ELISPOT", "IL-5 release"
    ) == (
        "th2_cytokine",
        "IL5",
    )
    assert classify_immunogenicity_modality(
        "soluble_factor_release", "ELISPOT", "IL-17 release"
    ) == (
        "th17_cytokine",
        "IL17",
    )
    assert classify_immunogenicity_modality(
        "soluble_factor_release", "ELISPOT", "IL-10 release"
    ) == (
        "treg_regulatory_cytokine",
        "IL10",
    )
    assert classify_immunogenicity_modality(
        "activation_or_proliferation", "ICS", "degranulation"
    ) == ("degranulation", "CD107A")
    assert classify_effector_repertoire("DMF5", "CD8+ T cell", "") == "tcr_clone_or_transduced"
    assert classify_effector_repertoire("", "PBMC", "direct ex vivo") == "endogenous_polyclonal"
    assert classify_antigen_delivery_mode("Linear peptide", "Epitope", "") == "exogenous_peptide"
    assert classify_antigen_delivery_mode("mRNA minigene", "Source antigen", "") == (
        "endogenous_processed"
    )
    assert classify_apc_lineage("autologous moDC") == "dc"
    assert classify_apc_lineage("T2 cells") == "t2"


def test_tcell_immunogenicity_v2_bundle_multi_modality_context_and_multispecies(tmp_path):
    iedb = [
        # 1: Human CEDAR neoantigen Th1 ELISPOT with coordinate-verified WT reconstruction
        (
            "1",
            {
                "Epitope | Name": "KLVVVGADGV",
                "Assay Antigen | Name": "KLVVVGADGV",
                "Epitope | Starting Position": "5",
                "Epitope | Ending Position": "14",
                "Assay | Method": "ELISPOT",
                "Assay | Response measured": "IFNg release",
                "Assay | Qualitative Measurement": "Positive-High",
                "1st in vivo Process | Process Type": "Occurrence of cancer",
                "1st in vivo Process | Disease": "pancreatic adenocarcinoma",
                "In vitro Process | Process Type": "Restimulation in vitro",
                "Antigen Presenting Cell | Name": "autologous moDC",
                "Effector Cell | Name": "PBMC",
            },
        ),
        # 2: TCR-transduced clone Th2 IL-5 assay with predicted MHC only -> falls back to donor MHC
        (
            "2",
            {
                "Assay | Method": "ELISPOT",
                "Assay | Response measured": "IL-5 release",
                "Assay | Qualitative Measurement": "Positive-Low",
                "MHC Restriction | Evidence Code": "MHC binding prediction",
                "Host | MHC Present": "HLA-A*02:01;HLA-A*24:02",
                "Effector Cell | TCR Name": "NY-ESO-1c259",
                "Antigen Presenting Cell | Name": "T2 cells",
            },
        ),
        # 3: C57BL/6 mouse cytotoxicity assay with class-only MHC -> recovers H2-Db;H2-Kb
        (
            "3",
            {
                "Epitope | Name": "SIINFEKL",
                "Assay Antigen | Name": "SIINFEKL",
                "Epitope | Source Organism IRI": "http://purl.obolibrary.org/obo/NCBITaxon_9031",
                "Epitope | Source Organism": "Gallus gallus",
                "Assay | Method": "51Cr release",
                "Assay | Response measured": "cytotoxicity",
                "Assay | Qualitative Measurement": "Positive",
                "Host | Name": "Mus musculus C57BL/6",
                "MHC Restriction | Name": "H2-b class I",
                "MHC Restriction | Class": "I",
                "MHC Restriction | Evidence Code": "",
                "Antigen Presenting Cell | Name": "splenocyte",
            },
        ),
        # 4: Chimeric HLA-A*02:01 HHD transgenic mouse multimer assay
        (
            "4",
            {
                "Assay | Method": "multimer/tetramer",
                "Assay | Response measured": "qualitative binding",
                "Assay | Qualitative Measurement": "Positive-Intermediate",
                "Host | Name": "Mus musculus HLA-A2/HHD",
                "MHC Restriction | Name": "HLA class I",
                "MHC Restriction | Class": "I",
                "MHC Restriction | Evidence Code": "",
            },
        ),
        # 5: Cell-free TCR-pMHC SPR KD assay -> MUST route to biophysics.parquet, NOT training!
        (
            "5",
            {
                "Assay | Method": "surface plasmon resonance",
                "Assay | Response measured": "dissociation constant KD",
                "Assay | Qualitative Measurement": "Positive",
                "Assay | Quantitative measurement": "1200",
                "Assay | Units": "nM",
                "Effector Cell | TCR Name": "1G4",
            },
        ),
    ]
    cedar = [
        ("1", {**iedb[0][1], "Epitope | Mutation": "G12D"}),
    ]
    paths = _make_sources(tmp_path, iedb, cedar)

    assert hitlist.write_tcell_immunogenicity_bundle is write_tcell_immunogenicity_bundle
    assert hitlist.verify_tcell_immunogenicity_bundle is verify_tcell_immunogenicity_bundle
    assert hitlist.iter_tcell_immunogenicity_assays is iter_tcell_immunogenicity_assays

    out = tmp_path / "immuno_v2"
    manifest = write_tcell_immunogenicity_bundle(out, **paths)
    assert manifest["n_source_assays"] == 5
    assert manifest["n_training_examples"] == 4
    assert manifest["n_biophysics_assays"] == 1
    assert manifest["n_excluded_assays"] == 1
    assert manifest["exclusion_counts"] == {"non_cellular_tcr_pmhc_biophysics": 1}
    assert manifest["n_chimeric_mhc_host_examples"] == 1
    assert manifest["repertoire_coverage"] == {
        "endogenous_polyclonal": 3,
        "tcr_clone_or_transduced": 1,
    }
    assert verify_tcell_immunogenicity_bundle(out, catalog_path=paths["catalog_path"]) == manifest

    training = pq.read_table(out / "training.parquet")
    assert training.schema == IMMUNOGENICITY_SCHEMA
    rows = {r["source_assay_id"]: r for r in training.to_pylist()}
    assert set(rows) == {"1", "2", "3", "4"}

    # Check neoantigen WT reconstruction & context on assay 1
    assert rows["1"]["wt_peptide"] == "KLVVVGAGGV"
    assert rows["1"]["wt_peptide_provenance"] == "reconstructed_from_cedar_mutation_coords"
    assert rows["1"]["antigen_category"] == "mutated_neoantigen"
    assert rows["1"]["immunogenicity_modality"] == "th1_cytokine"
    assert rows["1"]["cytokine_or_readout"] == "IFNG"
    assert rows["1"]["ordinal_intensity"] == "Positive-High"
    assert rows["1"]["ordinal_intensity_rank"] == 3
    assert rows["1"]["in_vivo_process_type"] == "cancer"
    assert rows["1"]["in_vitro_process_type"] == "restimulation_in_vitro"
    assert rows["1"]["apc_lineage"] == "dc"

    # Check TCR clone + predicted MHC isolation on assay 2
    assert rows["2"]["effector_repertoire_type"] == "tcr_clone_or_transduced"
    assert rows["2"]["effector_tcr_name"] == "NY-ESO-1c259"
    assert rows["2"]["immunogenicity_modality"] == "th2_cytokine"
    assert rows["2"]["is_predicted_restriction"] is True
    assert rows["2"]["mhc_allele"] is None
    assert rows["2"]["mhc_allele_set"] == "HLA-A*02:01;HLA-A*24:02"
    assert rows["2"]["mhc_allele_cardinality"] == "multi_allelic_bag"

    # Check C57BL/6 mouse strain recovery on assay 3
    assert rows["3"]["host_species"] == "Mus musculus"
    assert rows["3"]["mhc_species"] == "Mus musculus"
    assert rows["3"]["mhc_allele_set"] == "H2-Db;H2-Kb"
    assert rows["3"]["mhc_allele_set_provenance"] == "inbred_strain_haplotype"
    assert rows["3"]["ordinal_intensity"] == "Positive-Ungraded"
    assert rows["3"]["ordinal_intensity_rank"] is None
    assert rows["3"]["immunogenicity_modality"] == "cytotoxicity"

    # Check chimeric HLA-A*02:01 HHD mouse on assay 4
    assert rows["4"]["host_species"] == "Mus musculus"
    assert rows["4"]["mhc_species"] == "Homo sapiens"
    assert rows["4"]["is_chimeric_mhc_host"] is True
    assert rows["4"]["mhc_allele_set"] == "HLA-A*02:01"
    assert rows["4"]["mhc_allele_cardinality"] == "monoallelic"
    assert rows["4"]["immunogenicity_modality"] == "pmhc_multimer"

    # Check biophysics separation on assay 5
    bio = pq.read_table(out / "biophysics.parquet")
    assert bio.schema == BIOPHYSICS_SCHEMA
    assert bio.num_rows == 1
    assert bio.to_pylist()[0]["source_assay_id"] == "5"
    assert bio.to_pylist()[0]["quantitative_value"] == 1200.0

    # Check species filtering ("mouse" only)
    mouse_rows = list(iter_tcell_immunogenicity_assays(species_filter="mouse", **paths))
    assert [r["source_assay_id"] for r in mouse_rows] == ["3", "4"]
