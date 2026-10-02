"""Independent primary-literature species-context curation."""

from hitlist.curation import load_pmid_overrides
from hitlist.export import _consensus_meta, generate_ms_samples_table
from hitlist.species_contexts import (
    ARM_SPECIFIC_SPECIES_CONTEXT_COLUMNS,
    SPECIES_CONTEXT_COLUMNS,
    generate_species_contexts_table,
    load_species_contexts,
)
from tests.test_curation_sanity_pass import _export

OPTI_PDX_STATEMENT = (
    "The epitope was detected in a patient-derived xenograft cell line that had been treated "
    "with IFN-\N{GREEK SMALL LETTER GAMMA} to up-regulate HLA class I expression. The epitope "
    "was also present in untreated cells, but greater numbers of cells were needed to detect "
    "the epitope in the absence of IFN-\N{GREEK SMALL LETTER GAMMA}."
)
OPTI_BIOPSY_STATEMENTS = (
    "The epitope was eluted from a tumor tissue sample obtained from a patient with liposarcoma.",
    "The epitope was eluted from a tumor tissue sample obtained from a patient with "
    "osteosarcoma and lung metastasis.",
    "The epitope was eluted from a tumor tissue sample obtained from a patient with small "
    "intestine carcinoma.",
)


def test_review_manifest_records_coverage_and_explicit_queue():
    payload = load_species_contexts()
    inventory = payload["inventory"]
    assert inventory["corpus"] == "ci-corpus-v2"
    assert inventory["n_reviewed_pmids"] == 21
    assert inventory["n_reviewed_flagged_rows"] == 95442
    assert inventory["n_flagged_rows"] == 96838
    assert inventory["reviewed_flagged_fraction"] == "0.9856"
    assert inventory["n_review_queue_pmids"] == 204
    assert inventory["n_missing_pmid_flagged_rows"] == 423
    assert set(inventory["reviewed_pmids"]).isdisjoint(inventory["review_queue_pmids"])


def test_complete_registry_includes_unattached_and_unresolved_reviews():
    contexts = generate_species_contexts_table()
    assert contexts.pmid.nunique() == 21
    # This review is useful before observation attribution exists: the audit
    # table carries it, while an empty condition_ids prevents PMID broadcast.
    equine = contexts[contexts.species_context_id == "mouse_p815_expressing_equine_mhc"].iloc[0]
    assert equine.condition_ids == ""
    assert equine.presenting_species == "Mus musculus"
    assert equine.introduced_mhc_species == "Equus caballus"
    assert equine.species_context_status == "partial"
    # A reviewed candidate is not promoted to supported foreign material.
    murine_mhcii = contexts[contexts.pmid == 26495903].iloc[0]
    assert murine_mhcii.reviewed_candidate_species == "Bos taurus"
    assert murine_mhcii.supported_foreign_species == ""


def test_exact_condition_links_attach_context_without_pmid_broadcast():
    samples = generate_ms_samples_table()
    assert set(SPECIES_CONTEXT_COLUMNS) <= set(samples.columns)

    mamu = samples[
        (samples.pmid == 26811146) & (samples.condition_id == "cell_line_expressing_mamu_b_008_01")
    ].iloc[0]
    assert mamu.species == "Homo sapiens"
    assert mamu.presenting_species == "Homo sapiens"
    assert mamu.introduced_mhc_species == "Macaca mulatta"
    assert mamu.species_context_kind == "mhc_transfectant"

    # PMID 29393594 has two independently established systems. The curated
    # sample records carry their own evidence even though the deposited rows
    # cannot yet be mapped to one system or the other.
    mixed = samples[samples.pmid == 29393594].set_index("condition_id")
    assert len(mixed) == 2
    assert set(mixed.condition_mhc_context) == {"mhc_transfectant", "mhc_transgenic"}
    assert mixed.loc["human_cells_expressing_hla_b27", "presenting_species"] == "Homo sapiens"
    assert (
        mixed.loc["hla_b27_transgenic_rat_spleens_lys_p2", "presenting_species"]
        == "Rattus norvegicus"
    )
    assert (
        mixed.loc["hla_b27_transgenic_rat_spleens_lys_p2", "introduced_mhc_species"]
        == "Homo sapiens"
    )


def test_xenograft_at_harvest_and_xenograft_lineage_are_distinct():
    samples = generate_ms_samples_table()
    actual = samples[
        (samples.pmid == 32502341) & (samples.condition_id == "b_all_10h080_xenograft")
    ].iloc[0]
    assert actual.presenting_species == "Homo sapiens"
    assert actual.in_vivo_host_species == "Mus musculus"
    assert actual.lineage_host_species == ""
    assert actual.species_context_kind == "xenograft"

    derived = samples[
        (samples.pmid == 39111711) & (samples.condition_id == "pdx_derived_cell_line_ifn_gamma")
    ].iloc[0]
    assert derived.presenting_species == "Homo sapiens"
    assert derived.in_vivo_host_species == ""
    assert derived.lineage_host_species == "Mus musculus"
    assert derived.species_context_kind == "xenograft_derived_culture"
    assert derived.condition_material == "cultured"
    assert derived.condition_cytokines == "IFNG"

    biopsies = samples[
        (samples.pmid == 39111711) & (samples.condition_id == "patient_tumor_biopsies")
    ].iloc[0]
    assert biopsies.condition_material == "frozen"
    assert biopsies.lineage_host_species == ""


def test_literature_context_reaches_an_exactly_attributed_observation(monkeypatch):
    result = _export(
        monkeypatch,
        [
            {
                "pmid": 39111711,
                "mhc_class": "I",
                "mhc_restriction": "HLA-A*02:01",
                "assay_comments": OPTI_BIOPSY_STATEMENTS[0],
            }
        ],
    ).iloc[0]
    assert result.condition_id == "patient_tumor_biopsies"
    assert result.sample_attribution == "elution_conditions"
    assert result.species_context_id == "native_human_tumor_biopsies"
    assert result.presenting_species == "Homo sapiens"
    assert result.lineage_host_species == ""


def test_deposited_pdx_statement_preserves_both_reported_culture_arms(monkeypatch):
    result = _export(
        monkeypatch,
        [
            {
                "pmid": 39111711,
                "mhc_class": "I",
                "mhc_restriction": "HLA-A*02:01",
                "assay_comments": OPTI_PDX_STATEMENT,
            }
        ],
    ).iloc[0]
    assert result.condition_id == ""
    assert result.sample_attribution == "pmid_ambiguous"
    assert result.arm_resolution == "multi_arm_evidence"
    assert result.species_context_id == ""
    assert result.presenting_species == "Homo sapiens"
    assert result.lineage_host_species == "Mus musculus"


def test_every_deposited_opti_prm_statement_has_an_exact_arm_mapping():
    mapping = load_pmid_overrides()[39111711]["elution_condition_ids"]
    assert mapping == {
        OPTI_PDX_STATEMENT: [
            "pdx_derived_cell_line_ifn_gamma",
            "pdx_derived_cell_line_untreated",
        ],
        **{statement: ["patient_tumor_biopsies"] for statement in OPTI_BIOPSY_STATEMENTS},
    }


def test_unmapped_mixed_study_does_not_broadcast_an_arm_context(monkeypatch):
    result = _export(
        monkeypatch,
        [
            {
                "pmid": 29393594,
                "mhc_class": "I",
                "mhc_restriction": "HLA-B*27:05",
                "host_organism": "Rattus norvegicus",
                "source_species": "Rattus norvegicus",
            }
        ],
    ).iloc[0]
    assert result.sample_attribution == "pmid_ambiguous"
    assert result.species_context_id == ""
    assert result.presenting_species == ""
    # Both reviewed arms use a human HLA-B27 molecule, so that shared fact is
    # safe even though the presenting system and record identity are not.
    assert result.introduced_mhc_species == "Homo sapiens"


def test_ambiguous_consensus_clears_review_identity_but_keeps_shared_axes():
    common = {
        "species_context_status": "resolved",
        "species_context_scope": "arm-specific scope",
        "species_context_reference": "doi:10/example, Methods",
        "presenting_species": "Homo sapiens",
        "introduced_mhc_species": "Canis lupus familiaris",
        "species_context_kind": "mhc_transfectant",
    }
    candidates = [
        ("", "", {**common, "species_context_id": "one", "species_context_note": "one"}),
        ("", "", {**common, "species_context_id": "two", "species_context_note": "two"}),
    ]
    result = _consensus_meta(candidates, list(SPECIES_CONTEXT_COLUMNS))
    for column in ARM_SPECIFIC_SPECIES_CONTEXT_COLUMNS:
        assert result[column] == ""
    assert result["presenting_species"] == "Homo sapiens"
    assert result["introduced_mhc_species"] == "Canis lupus familiaris"


def test_packaged_condition_links_are_validated_with_pmid_curation():
    # The curation loader performs the cross-registry validation after every
    # condition_id is known. Reaching this assertion proves all 29 records and
    # every linked arm passed both schema guards.
    assert 39111711 in load_pmid_overrides()
