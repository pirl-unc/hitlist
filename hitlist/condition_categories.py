# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Coarse perturbation categories for sample-level conditions.

Curated ``ms_samples`` entries write the perturbation as free text
("ERAP1 deletion", "ERAP1 CRISPR/Cas9 knockout", "ERAP1 shRNA
knockdown", "ERAP1 pharmacological inhibition", "ERAP1 variant
comparison", "ERAP1+ERAP2 double KO", ...).  All five strings name
the same biology — ERAP1 perturbation — and consumers training a
condition-aware model want one bucket, not one bucket per
phrasing.

This module assigns each ``perturbation`` string to a
single ``condition_category`` from a stable enum.  The
categorization layers two passes:

1. The :mod:`hitlist.apm` gene classifier already detects 30+ APM
   gene names / inhibitors in free text — promote those flags to
   APM-level categories first.  When a perturbation matches more
   than one APM gene (e.g. ``ERAP1+ERAP2 double KO``), the highest-
   priority gene wins by an explicit precedence list (B2M loss
   trumps anything else; TAP family before tapasin before PLC
   chaperones, etc.) so each perturbation lands in exactly one
   category.

2. For non-APM perturbations, regex patterns match common
   experimental contexts: viral / bacterial / parasitic infection,
   drug exposure, transfection, transduction, transplant, cell
   activation, TLR stimulation, biomaterial contact, metabolic
   stress, and labeling controls.  Bacterial / parasite patterns
   are checked **before** the generic "infection" pattern so
   "Listeria monocytogenes infection" doesn't get tagged
   ``infection_viral``.

Empty / missing perturbation → ``unperturbed``.  Unmatched non-empty
perturbations → ``other_perturbation``; review these and add a
specific pattern when a category accumulates real volume.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from .apm import classify_apm_perturbations

# Mapping from APM gene flag → category. Order in this dict is also
# used to break ties when multiple APM genes match a single
# perturbation: the FIRST matching key in this dict wins.  Keep B2M
# at the top because B2M loss collapses all class-I presentation
# regardless of any other co-perturbation listed in the same string.
_APM_TO_CATEGORY: dict[str, str] = {
    # MHC-I loss (most severe APM perturbation possible).
    "b2m": "MHC-I_loss_B2M",
    # TAP family.
    "tap_deficient_line": "TAP_perturbation",
    "tap_inhibitor": "TAP_perturbation",
    "tap1": "TAP_perturbation",
    "tap2": "TAP_perturbation",
    # Tapasin / TAPBPR.
    "tapbp": "tapasin_perturbation",
    # ER aminopeptidases.
    "erap1": "ERAP1_perturbation",
    "erap2": "ERAP2_perturbation",
    "erap_inhibitor": "ERAP_inhibitor",
    # Peptide-loading-complex chaperones.
    "pdia3": "PLC_chaperone_perturbation",
    "calr": "PLC_chaperone_perturbation",
    "canx": "PLC_chaperone_perturbation",
    "ganab": "PLC_chaperone_perturbation",
    # Transcription / induction.
    "irf2": "IRF2_perturbation",
    "nlrc5": "NLRC5_perturbation",
    "sppl3": "SPPL3_perturbation",
    # Proteasome.
    "psmb5": "proteasome_subunit_perturbation",
    "psmb8": "proteasome_subunit_perturbation",
    "psmb9": "proteasome_subunit_perturbation",
    "psmb10": "proteasome_subunit_perturbation",
    "proteasome_inhibitor": "proteasome_inhibitor",
    # Class II.
    "ciita": "CIITA_transduction",
    "hla_dm": "HLA-DM_perturbation",
    "hla_do": "HLA-DO_perturbation",
    "cd74": "CD74_perturbation",
    "rfx": "RFX_BLS_perturbation",
    "bls": "RFX_BLS_perturbation",
    # Cathepsin (class II processing).
    "cathepsin": "cathepsin_perturbation",
    "cathepsin_inhibitor": "cathepsin_perturbation",
    # Cytokines and TLR stimulation — kept LAST among APM categories
    # so that a sample tagged "B2M KO + IFN-gamma" buckets as
    # MHC-I_loss_B2M (the dominant biological effect) rather than
    # IFN_gamma_treatment.
    "ifn_gamma": "IFN_gamma_treatment",
    "ifn_alpha": "IFN_alpha_treatment",
    "ifn_beta": "IFN_beta_treatment",
    "tnf_alpha": "TNF_alpha_treatment",
    "lps": "TLR_stimulation",
}

# Regex patterns for non-APM perturbations. Order matters: more
# specific patterns first.  Each tuple is ``(pattern, category)``.
_NON_APM_PATTERNS: list[tuple[str, str]] = [
    # UV / heat-inactivated virus control comes FIRST — it's a
    # control arm of an infection study, not the infection itself.
    (r"uv-?inactivated|heat-?inactivated|virus control", "virus_inactivated_control"),
    # Transduction (AAV / lentiviral / retroviral) — vector-based
    # gene delivery is NOT viral infection in the immunological
    # sense, even though the vector is virus-derived.  Caught
    # BEFORE the viral-infection patterns so "AAV-transduced
    # allogeneic MHC-I expression" and "CMV pp65 retroviral
    # transduction" land here, not in infection_viral.
    (
        r"\b(?:AAV-?|lentivir(?:us|al)|retrovir(?:us|al)|oncoretrovir|"
        r"transduc(?:ed|tion))\b",
        "transduction",
    ),
    # Bacterial / parasitic infection — checked BEFORE the generic
    # "infection" rule so Listeria etc. don't get tagged viral.
    (
        r"\b(?:listeria|salmonella|chlamydia|pseudomonas|mycobacterium|tuberculosis|"
        r"borrelia|theileria|t\.\s*parva|toxoplasma|leishmania|plasmodium)\b",
        "infection_bacterial_or_parasite",
    ),
    # Specific viral infections (named viruses).
    (
        r"\b(?:influenza|sars-?cov|coronavirus|hiv|ebv|cmv|hcv|hbv|measles|hpv|"
        r"canine distemper|adenovirus|herpes|wisconsin)\b",
        "infection_viral",
    ),
    # Generic "infection" / "infected" without a virus name still buckets viral
    # by default (most "X infection" entries in the corpus are viral).
    (r"\binfection\b|\binfected\b", "infection_viral"),
    # TAPBPR (the "tapasin-related" gene; not yet in the APM
    # vocabulary because it's mentioned in only one curated entry).
    (r"\bTAPBPR\b|tapasin-related", "tapasin_perturbation"),
    # Generic "TAP deficiency" without the T2/RMA-S marker.
    (r"\bTAP deficiency\b|\bTAP-deficient\b", "TAP_perturbation"),
    # Generic IFN treatment without alpha/beta/gamma qualifier.
    (r"\bIFN treatment\b|\binterferon treatment\b", "cytokine_treatment_generic"),
    # Drug exposure / pharmacology — covers a long tail of
    # perturbations where a small molecule was added to culture.
    (
        r"\b(?:carbamazepine|flucloxacillin|abacavir|allopurinol|lamotrigine|"
        r"amoxicillin|penicillin|ceftriaxone|trimethoprim|"
        r"lenalidomide|tazemetostat|decitabine|EZH2i|"
        r"cdk4/6|palbociclib|ribociclib|abemaciclib|"
        r"dasatinib|imatinib|nilotinib|"
        r"vorinostat|romidepsin|panobinostat|"
        r"cisplatin|carboplatin|paclitaxel|doxorubicin|fluorouracil|"
        r"olaparib|niraparib|rucaparib)\b",
        "drug_exposure",
    ),
    # Transfection (HLA / oncogene / TCR / MAPTAC mono-allelic).
    (
        r"\btransfect|\btransfection\b|MAPTAC|expression vector|mono-allelic transfectant",
        "transfection",
    ),
    # Transplant / graft.
    (r"\btransplant|graft\b", "transplant"),
    # Cell activation / stimulation (PMA, ionomycin, in vitro
    # activation, antigen-specific restimulation).
    (
        r"\bPMA\b|ionomycin|in vitro activation|restimulation|"
        r"antigen-specific|cytokine cocktail|polyclonal stimulation|"
        r"CD3/CD28",
        "cell_activation",
    ),
    # Biomaterial / surface / implant contact.
    (r"biomaterial|implant|polymer|surface contact", "biomaterial_contact"),
    # Metabolic stress / nutrient deprivation / hypoxia.
    (r"hypoxia|metabolic stress|glucose deprivation|amino acid starvation", "metabolic_stress"),
    # Labeling / methodology controls (SILAC, TMT) — not really
    # biological perturbations.
    (r"\bSILAC\b|\bTMT labeling\b|\bisotope labeling\b", "labeling_control"),
]

_COMPILED_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(pat, re.IGNORECASE), cat) for pat, cat in _NON_APM_PATTERNS
]

UNPERTURBED = "unperturbed"
OTHER_PERTURBATION = "other_perturbation"


def categorize_condition(perturbation: str | None) -> str:
    """Map a free-text perturbation string to a single category.

    Empty input → ``"unperturbed"``.  An unmatched non-empty string
    → ``"other_perturbation"`` (review and add a pattern when these
    accumulate).
    """
    if not perturbation:
        return UNPERTURBED
    text = str(perturbation).strip()
    if not text:
        return UNPERTURBED

    apm_flags = classify_apm_perturbations(text)
    matched_apm = {g for g, v in apm_flags.items() if v}
    if matched_apm:
        for gene in _APM_TO_CATEGORY:
            if gene in matched_apm:
                return _APM_TO_CATEGORY[gene]

    for rx, cat in _COMPILED_PATTERNS:
        if rx.search(text):
            return cat

    return OTHER_PERTURBATION


def categorize_conditions(perturbations: Iterable[str | None]) -> list[str]:
    """Vectorized convenience wrapper over :func:`categorize_condition`."""
    return [categorize_condition(p) for p in perturbations]
