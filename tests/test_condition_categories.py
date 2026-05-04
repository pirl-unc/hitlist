"""Tests for hitlist.condition_categories."""

from hitlist.condition_categories import (
    OTHER_PERTURBATION,
    UNPERTURBED,
    categorize_condition,
    categorize_conditions,
)


def test_unperturbed_for_empty_input():
    assert categorize_condition("") == UNPERTURBED
    assert categorize_condition(None) == UNPERTURBED
    assert categorize_condition("   ") == UNPERTURBED


def test_erap1_phrasings_collapse_to_one_category():
    """The whole point of the categorizer: phrasing variation
    around the same biological perturbation maps to one bucket.
    """
    cat = "ERAP1_perturbation"
    for phrasing in (
        "ERAP1 deletion",
        "ERAP1 CRISPR/Cas9 knockout",
        "ERAP1 shRNA knockdown",
        "ERAP1 pharmacological inhibition",
        "ERAP1 variant comparison",
        "ERAP1+ERAP2 double CRISPR/Cas9 knockout (3 replicates)",
        "ERAP1/ERAP2 variant combinations",
    ):
        assert categorize_condition(phrasing) == cat, (
            f"phrasing '{phrasing}' did not collapse to {cat}"
        )


def test_b2m_loss_dominates_co_perturbation():
    """B2M is highest-priority — a sample tagged "B2M KO + IFN-gamma"
    must bucket as MHC-I_loss_B2M, since B2M loss collapses class I
    presentation regardless of any cytokine co-treatment.
    """
    assert categorize_condition("B2M CRISPR knockout + IFN-gamma stimulation") == "MHC-I_loss_B2M"


def test_listeria_does_not_bucket_as_viral():
    """Bacterial / parasite patterns must beat the generic
    "infection" rule.
    """
    assert categorize_condition("Listeria monocytogenes infection") == (
        "infection_bacterial_or_parasite"
    )
    assert categorize_condition("Mycobacterium tuberculosis exposure") == (
        "infection_bacterial_or_parasite"
    )
    assert categorize_condition("T. parva infection") == ("infection_bacterial_or_parasite")


def test_aav_transduction_is_not_viral_infection():
    """AAV / retroviral / lentiviral transduction is gene delivery,
    not viral infection, even though the vector is virus-derived.
    """
    assert categorize_condition("AAV-transduced allogeneic MHC-I expression") == ("transduction")
    assert categorize_condition("CMV pp65 retroviral transduction") == "transduction"
    assert categorize_condition("HIV Env transduction") == "transduction"


def test_uv_inactivated_control_beats_infection():
    """The UV-inactivated control arm is paired with an infection
    arm in the same study; tag the control distinctly.
    """
    assert categorize_condition("UV-inactivated virus control") == ("virus_inactivated_control")


def test_apm_chaperones_share_one_bucket():
    """CALR / CANX / GANAB / PDIA3 are all PLC chaperones — same
    biology for downstream training.
    """
    cat = "PLC_chaperone_perturbation"
    for phrasing in (
        "CALR CRISPR/Cas9 knockout (calreticulin)",
        "CANX CRISPR/Cas9 knockout (calnexin)",
        "GANAB CRISPR/Cas9 knockout (glucosidase II alpha)",
        "PDIA3 CRISPR/Cas9 knockout (ERp57)",
    ):
        assert categorize_condition(phrasing) == cat


def test_tap1_tap2_share_tap_perturbation_bucket():
    assert categorize_condition("TAP1 CRISPR/Cas9 knockout") == "TAP_perturbation"
    assert categorize_condition("TAP2 CRISPR/Cas9 knockout") == "TAP_perturbation"
    assert categorize_condition("TAP deficiency") == "TAP_perturbation"


def test_cytokines_each_get_own_bucket():
    assert categorize_condition("IFN-gamma 100 IU/mL 24h") == "IFN_gamma_treatment"
    assert categorize_condition("IFN-alpha treatment") == "IFN_alpha_treatment"
    assert categorize_condition("TNF-alpha + IFN-gamma") == "IFN_gamma_treatment"


def test_drug_exposure_covers_long_tail():
    assert categorize_condition("carbamazepine exposure") == "drug_exposure"
    assert categorize_condition("flucloxacillin treatment") == "drug_exposure"
    assert categorize_condition("lenalidomide treatment") == "drug_exposure"


def test_unmatched_falls_through_to_other():
    assert categorize_condition("strange unknown thing") == OTHER_PERTURBATION


def test_categorize_conditions_vectorized():
    out = categorize_conditions(["", "ERAP1 deletion", "Listeria monocytogenes infection", None])
    assert out == [
        UNPERTURBED,
        "ERAP1_perturbation",
        "infection_bacterial_or_parasite",
        UNPERTURBED,
    ]
