"""Lorente 2019 experimental lines and deposited evidence (#452)."""

import pandas as pd

from hitlist.curation import load_pmid_overrides
from hitlist.export import generate_ms_samples_table, generate_observations_table


def test_lorente_has_five_lines_and_process_matched_unedited_controls():
    # PMC6823859: ERAP2 KO Generation by CRISPR; Isolation of HLA-B*40:02
    # Bound Peptides; Results/Table I. Three biological preparations per line.
    entry = load_pmid_overrides()[31530632]
    arms = {arm["condition_id"]: arm for arm in entry["ms_samples"]}
    prefix = "c1r_hla_b_40_02_"
    assert set(arms) == {prefix + name for name in ["wt", "wt1", "wt2", "erap2_ko1", "erap2_ko3"]}
    knockout_ids = {prefix + "erap2_ko1", prefix + "erap2_ko3"}
    for arm in arms.values():
        assert arm["n_samples"] == 3
        assert arm["condition_background"] == "ERAP1_hap8"
        assert arm["condition_mhc_context"] == "mhc_transfectant"
        assert arm["sample_group"] == "C1R"
        assert arm["condition_evidence"] == "primary_source"
    for name in ["wt", "wt1", "wt2"]:
        arm = arms[prefix + name]
        assert arm["condition_knockout_genes"] == "none"
        assert set(arm["condition_control_for"].split(";")) == knockout_ids
        assert arm["condition_control"] == ("untreated" if name == "wt" else "other")
    for condition_id in knockout_ids:
        assert arms[condition_id]["condition_knockout_genes"] == "ERAP2"
    samples = generate_ms_samples_table()
    samples = samples[samples.pmid == 31530632]
    assert len(samples) == 5
    controls = samples[samples.condition_id.isin({prefix + name for name in ["wt", "wt1", "wt2"]})]
    assert controls.condition_knockout_genes.eq("none").all()
    assert controls.perturbation.eq("").all()


def test_lorente_deposited_genotype_groups_do_not_identify_clones(monkeypatch):
    # These are the only three assay-comment variants in all 10,319 local rows.
    common = (
        "The HLA-B*40:02 peptidomes from wild-type and ERAP2-KO cells were compared "
        "in order to study the effects of ERAP2 depletion. This peptide was identified "
    )
    frame = pd.DataFrame(
        {
            "peptide": ["AAAAAAAAA", "CCCCCCCCC", "DDDDDDDDD"],
            "pmid": 31530632,
            "cell_name": "C1R cells-B cell",
            "source_tissue": "Lymphoid",
            "mhc_restriction": "HLA-B*40:02",
            "mhc_class": "I",
            "mhc_species": "Homo sapiens",
            "is_monoallelic": False,
            "source": "iedb",
            "is_binding_assay": False,
            "qualitative_measurement": "Positive",
            "antigen_processing_comments": (
                "Stable C1R transfectants expressing HLA-B*40:02 were used for "
                "immunopeptidomics studies. The cells were either wild-type or ERAP2 knockouts."
            ),
            "assay_comments": [
                common + ending
                for ending in [
                    "in both cells.",
                    "in the wild-type cells only.",
                    "in the ERAP2-KO cells only.",
                ]
            ],
        }
    )
    monkeypatch.setattr("hitlist.observations.load_observations", lambda **kwargs: frame.copy())
    rows = generate_observations_table(exclude_non_peptide_ligand=False)
    assert len(rows) == 3
    assert rows.sample_group.eq("C1R").all()
    assert rows.sample_attribution.eq("group_ambiguous").all()
    assert rows.condition_id.eq("").all()
    assert rows.arm_resolution.eq("arm_not_recorded").all()
    assert rows.mhc_restriction.eq("HLA-B*40:02").all()
