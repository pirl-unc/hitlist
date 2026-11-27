"""Tests for #943 (unified condition-level MHC genotype recovery across MS, perturbed MS,
binding, and T-cell frames) and #944 (multi-species allele/haplotype normalization,
cross-species APM ortholog mapping, and multi-species MS training prep).
"""

from __future__ import annotations

import pandas as pd

from hitlist.apm_orthologs import (
    canonical_apm_ortholog_group,
    is_apm_gene_absent_in_species,
    load_apm_orthologs,
    map_apm_genes_to_canonical,
    species_ortholog_symbols,
)
from hitlist.mhc_genotypes import (
    normalize_mhc_token,
    recover_mhc_genotypes_for_frame,
    summarize_mhc_genotype_recovery,
)
from hitlist.model_training import prepare_ms_training_data


def test_mhc_exact_molecule_rejects_typos_and_accepts_multispecies_alleles():
    # Rejects invented single-field rodent/NHP strings (parity with curation._names_one_molecule).
    assert normalize_mhc_token("H2-K*nonsense").is_exact_molecule is False
    assert normalize_mhc_token("H2-K*x").is_exact_molecule is False
    assert normalize_mhc_token("HLA-A*02").is_exact_molecule is False

    # Accepts canonical human, rodent, NHP, canine, bovine, porcine, avian, equine alleles.
    for raw, expected_sp in (
        ("HLA-A*02:01", "Homo sapiens"),
        ("H2-Kb", "Mus musculus"),
        ("H2-D*b", "Mus musculus"),
        ("H2-IAb", "Mus musculus"),
        ("H2-IAg7", "Mus musculus"),
        ("RT1-Al", "Rattus sp."),
        ("Mamu-A1*001:01", "Macaca mulatta"),
        ("DLA-88*501:01", "Canis sp."),
        ("BoLA-2*026:01", "Bos sp."),
        ("SLA-1*04:01", "Sus sp."),
        ("Gaga-BF2*004:01", "Gallus gallus"),
    ):
        norm = normalize_mhc_token(raw)
        assert norm.is_exact_molecule, f"Expected exact molecule for {raw!r}, got {norm}"
        assert norm.mhc_species == expected_sp


def test_recover_mhc_genotypes_for_frame_covers_ms_perturbed_and_multispecies():
    df = pd.DataFrame(
        [
            # 1: Already-resolved primary monoallelic row -> preserved verbatim
            {
                "peptide": "GILGFVFTL",
                "mhc_restriction": "HLA-A*02:01",
                "mhc_class": "I",
                "mhc_allele_set": "HLA-A*02:01",
                "mhc_allele_provenance": "exact",
                "host_mhc_types": "",
                "cell_line_name": "",
                "host": "Homo sapiens",
            },
            # 2: Class-only MS row with host_mhc_types (like PMID 29557506 / 35154160)
            {
                "peptide": "SLYNTVATL",
                "mhc_restriction": "HLA class I",
                "mhc_class": "I",
                "mhc_allele_set": "",
                "mhc_allele_provenance": "unmatched",
                "host_mhc_types": "HLA-A*02:01;HLA-A*24:02;HLA-B*07:02;HLA-DRB1*04:01",
                "cell_line_name": "",
                "host": "Homo sapiens",
            },
            # 3: Unresolved HAP1 perturbed row (like PMID 40113210)
            {
                "peptide": "YLQPRTFLL",
                "mhc_restriction": "",
                "mhc_class": "I",
                "mhc_allele_set": "",
                "mhc_allele_provenance": "unmatched",
                "host_mhc_types": "",
                "cell_line_name": "HAP1",
                "host": "Homo sapiens",
            },
            # 4: Class-only C57BL/6 mouse GILT KO row (like PMID 20485683)
            {
                "peptide": "AAKFEQALQ",
                "mhc_restriction": "H2 class II",
                "mhc_class": "II",
                "mhc_allele_set": "",
                "mhc_allele_provenance": "unmatched",
                "host_mhc_types": "",
                "cell_line_name": "",
                "host": "Mus musculus C57BL/6",
            },
            # 5: Chimeric HHD HLA-A*02:01 transgenic mouse
            {
                "peptide": "NLVPMVATV",
                "mhc_restriction": "HLA class I",
                "mhc_class": "I",
                "mhc_allele_set": "",
                "mhc_allele_provenance": "unmatched",
                "host_mhc_types": "",
                "cell_line_name": "",
                "host": "Mus musculus HLA-A2/HHD",
            },
            # 6: Class-I null 721.221 host with no allele -> stays class_only_unknown!
            {
                "peptide": "KLVVVGADGV",
                "mhc_restriction": "HLA class I",
                "mhc_class": "I",
                "mhc_allele_set": "",
                "mhc_allele_provenance": "unmatched",
                "host_mhc_types": "",
                "cell_line_name": "721.221",
                "host": "Homo sapiens",
            },
        ]
    )
    enriched = recover_mhc_genotypes_for_frame(df)

    assert enriched.loc[0, "mhc_allele_set"] == "HLA-A*02:01"
    assert enriched.loc[0, "mhc_allele_cardinality"] == "monoallelic"
    assert enriched.loc[0, "mhc_allele_set_provenance"] == "reported_single_allele"

    # Class I filter excludes HLA-DRB1*04:01 from host_mhc_types!
    assert enriched.loc[1, "mhc_allele_set"] == "HLA-A*02:01;HLA-A*24:02;HLA-B*07:02"
    assert enriched.loc[1, "mhc_allele_cardinality"] == "multi_allelic_bag"
    assert enriched.loc[1, "mhc_allele_set_provenance"] == "host_mhc_types"

    assert enriched.loc[2, "mhc_allele_set"] == "HLA-A*02:01;HLA-B*40:01;HLA-C*03:04"
    assert enriched.loc[2, "mhc_allele_set_provenance"] == "curated_cell_line_hla"

    assert enriched.loc[3, "mhc_allele_set"] == "H2-IAb"
    assert enriched.loc[3, "mhc_allele_cardinality"] == "monoallelic"
    assert enriched.loc[3, "mhc_allele_set_provenance"] == "inbred_strain_haplotype"
    assert enriched.loc[3, "mhc_species"] == "Mus musculus"

    assert enriched.loc[4, "mhc_allele_set"] == "HLA-A*02:01"
    assert enriched.loc[4, "mhc_allele_set_provenance"] == "transgenic_mhc_construct"
    assert enriched.loc[4, "mhc_species"] == "Homo sapiens"
    assert enriched.loc[4, "host_species"] == "Mus musculus"
    assert bool(enriched.loc[4, "is_chimeric_mhc_host"]) is True

    # 721.221 Class-I null host MUST NOT receive fake endogenous Class-I alleles
    assert enriched.loc[5, "mhc_allele_set"] == ""
    assert enriched.loc[5, "mhc_allele_cardinality"] == "class_only_unknown"
    assert enriched.loc[5, "mhc_allele_set_provenance"] == "unknown"

    summary = summarize_mhc_genotype_recovery(enriched)
    assert summary["n_rows"] == 6
    assert summary["n_resolved_rows"] == 5
    assert summary["n_monoallelic_rows"] == 3
    assert summary["n_multi_allelic_bag_rows"] == 2
    assert summary["n_class_only_unknown_rows"] == 1
    assert summary["n_chimeric_mhc_host_rows"] == 1


def test_prepare_ms_training_data_multispecies_and_reference_genotype_recovery():
    ms_df = pd.DataFrame(
        [
            {
                "evidence_kind": "ms",
                "evidence_row_id": "ms_hap1",
                "evidence_source_id": "src1",
                "provenance_id": "p1",
                "peptide": "GILGFVFTL",
                "mhc_restriction": "HLA class I",
                "mhc_class": "I",
                "mhc_species": "Homo sapiens",
                "mhc_allele_set": "",
                "mhc_allele_provenance": "unmatched",
                "cell_line_name": "HAP1",
                "host": "Homo sapiens",
                "pmid": "40113210",
            },
            {
                "evidence_kind": "ms",
                "evidence_row_id": "ms_b6_mouse",
                "evidence_source_id": "src2",
                "provenance_id": "p2",
                "peptide": "SIINFEKL",
                "mhc_restriction": "H2 class I",
                "mhc_class": "I",
                "mhc_species": "Mus musculus",
                "mhc_allele_set": "",
                "mhc_allele_provenance": "unmatched",
                "cell_line_name": "EL-4",
                "host": "Mus musculus C57BL/6",
                "pmid": "30084848",
            },
        ]
    )

    # Without recovery, both class-only unmatched rows are excluded.
    res_no_rec = prepare_ms_training_data(ms_df, species="all", recover_reference_genotypes=False)
    assert len(res_no_rec.accepted) == 0
    assert len(res_no_rec.excluded) == 2

    # With recover_reference_genotypes=True & species="all", BOTH human HAP1 and mouse EL-4 recover!
    res = prepare_ms_training_data(ms_df, species="all", recover_reference_genotypes=True)
    assert len(res.accepted) == 2
    acc = res.accepted.set_index("evidence_row_id")
    assert acc.loc["ms_hap1", "mhc_allele_set"] == "HLA-A*02:01;HLA-B*40:01;HLA-C*03:04"
    assert acc.loc["ms_hap1", "mhc_allele_cardinality"] == "multi_allelic_bag"
    assert acc.loc["ms_hap1", "candidate_bag_provenance"] == "curated_cell_line_hla"
    assert acc.loc["ms_b6_mouse", "mhc_allele_set"] == "H2-D*b;H2-K*b" or (
        set(acc.loc["ms_b6_mouse", "mhc_allele_set"].replace("*", "").split(";"))
        == {"H2-Db", "H2-Kb"}
    )
    assert acc.loc["ms_b6_mouse", "mhc_species"] == "Mus musculus"
    assert acc.loc["ms_b6_mouse", "host_species"] == "Mus musculus"
    assert bool(acc.loc["ms_b6_mouse", "is_chimeric_mhc_host"]) is False


def test_cross_species_apm_orthologs_registry():
    data = load_apm_orthologs()
    assert len(data["ortholog_groups"]) >= 35

    assert canonical_apm_ortholog_group("H2-DMa") == "HLA-DMA"
    assert canonical_apm_ortholog_group("H2-DMb1") == "HLA-DMB"
    assert canonical_apm_ortholog_group("H2-DMb2") == "HLA-DMB"
    assert canonical_apm_ortholog_group("H2-Oa") == "HLA-DOA"
    assert canonical_apm_ortholog_group("Ifi30-/-") == "IFI30"
    assert canonical_apm_ortholog_group("GILT") == "IFI30"
    assert canonical_apm_ortholog_group("Erap1_KO") == "ERAP1"
    assert canonical_apm_ortholog_group("Psmb8") == "PSMB8"
    assert canonical_apm_ortholog_group("LMP2") == "PSMB9"
    assert canonical_apm_ortholog_group("MECL-1") == "PSMB10"
    assert canonical_apm_ortholog_group("Cd74") == "CD74"
    assert canonical_apm_ortholog_group("Tapbp") == "TAPBP"
    assert canonical_apm_ortholog_group("Tapbpl") == "TAPBPL"

    assert species_ortholog_symbols("HLA-DMA", "mouse") == ("H2-DMa",)
    assert species_ortholog_symbols("HLA-DMB", "Mus musculus") == ("H2-DMb1", "H2-DMb2")
    assert species_ortholog_symbols("ERAP1", "Mus musculus") == ("Erap1",)

    # Rodents naturally lack ERAP2; birds lack immunoproteasome PSMB8/9/10 & HLA-DOA/B.
    assert is_apm_gene_absent_in_species("ERAP2", "Mus musculus") is True
    assert is_apm_gene_absent_in_species("ERAP2", "Rattus norvegicus") is True
    assert is_apm_gene_absent_in_species("ERAP2", "Homo sapiens") is False
    assert is_apm_gene_absent_in_species("PSMB8", "Gallus gallus") is True

    assert map_apm_genes_to_canonical("Psmb8;Psmb9;Psmb10;Ifi30") == (
        "PSMB8",
        "PSMB9",
        "PSMB10",
        "IFI30",
    )
