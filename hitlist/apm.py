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

"""Antigen-processing-machinery (APM) perturbation classifier.

Each ms_sample's ``condition`` (and the parent study's ``perturbations``
list) gets parsed against a curated vocabulary of APM gene names.  We
emit one boolean flag per gene plus a union flag, so consumers can
filter the corpus to "samples where ERAP1 was knocked out" or "any
APM perturbation" with a single column query.

Why parse, not match free text?
- The YAML's ``note`` field often mentions APM genes incidentally
  (e.g. "GANAB CRISPR KO (glucosidase II alpha — glycan trimming)"
  also contains the substring "II" which would match the invariant
  chain abbreviation "Ii" under naive case-insensitive search).
- The ``condition`` and ``perturbations`` fields are deliberately
  written by curators to describe the experimental perturbation.
- Restricting to those fields makes the binary signal crisp.

Vocabulary kept to genes that actually appear in the corpus
(surveyed Apr 2026).  Add a new entry to ``APM_GENES`` when a paper
introduces a new perturbation; the columns + union flag fall out
automatically.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

# Mapping from canonical column suffix → list of regex patterns that
# identify the gene in curator-written perturbation strings. Patterns
# are word-boundary-anchored so "ERAP1" doesn't match "ERAP10" and
# "Ii" doesn't match "II".  Multi-word names (e.g. "HLA-DM",
# "invariant chain") allow more permissive matching than the
# default ``\b<token>\b`` because the hyphen / space already
# disambiguates.
APM_GENES: dict[str, tuple[str, ...]] = {
    # ── Class I APM (peptide loading complex) ──
    "b2m": (r"\bB2M\b", r"\bbeta-?2-?microglobulin\b", r"β2m"),
    "tap1": (r"\bTAP1\b",),
    "tap2": (r"\bTAP2\b",),
    "tapbp": (r"\bTAPBP\b", r"\btapasin\b"),
    # TAPBPR is a widely expressed tapasin homolog that is *not* part of
    # the peptide-loading complex — a second, independent peptide editor
    # performing exchange outside the PLC (Hermann et al., PMID 26869717;
    # Thomas & Tampe, Science 2017).  So it gets its own key rather than
    # folding into ``tapbp``.  The ``\bTAPBP\b``
    # pattern above already declines to match it (the trailing "R" is a
    # word character), which is why it went uncaptured entirely.
    "tapbpr": (r"\bTAPBPR\b", r"\bTAPBPL\b"),
    # ERAAP ("ER aminopeptidase associated with antigen processing") is
    # an established alternate name for ERAP1, used throughout the murine
    # literature — Hammer et al., Nat Immunol 2006 ("The aminopeptidase
    # ERAAP shapes the peptide repertoire displayed by MHC class I
    # molecules").  Without the alias a mouse ERAP1 knockout landed in
    # ``other_perturbation`` instead of ``ERAP1_perturbation``.
    "erap1": (r"\bERAP1\b", r"\bERAAP\b"),
    "erap2": (r"\bERAP2\b",),
    "pdia3": (r"\bPDIA3\b", r"\bERp57\b"),
    "calr": (r"\bCALR\b", r"\bcalreticulin\b"),
    "canx": (r"\bCANX\b", r"\bcalnexin\b"),
    "irf2": (r"\bIRF2\b",),
    "ganab": (r"\bGANAB\b",),
    "sppl3": (r"\bSPPL3\b",),
    "nlrc5": (r"\bNLRC5\b",),
    # ── Proteasome subunits / inhibitors ──
    # Subunit-specific perturbations (LMP2/LMP7/MECL1 are the IFN-gamma-
    # inducible immunoproteasome catalytic subunits) plus the small-
    # molecule inhibitors used to perturb the constitutive proteasome
    # in MS studies.
    "psmb5": (r"\bPSMB5\b",),
    "psmb8": (r"\bPSMB8\b", r"\bLMP7\b"),
    "psmb9": (r"\bPSMB9\b", r"\bLMP2\b"),
    "psmb10": (r"\bPSMB10\b", r"\bMECL[- ]?1\b"),
    "proteasome_inhibitor": (
        r"\bbortezomib\b",
        r"\bMG-?132\b",
        r"\bepoxomicin\b",
        r"\bcarfilzomib\b",
        r"\bixazomib\b",
        r"\blactacystin\b",
        r"\bONX-?0914\b",
        r"\bPR-957\b",
        r"\bLMP7 inhibitor\b",
        r"\bproteasom(?:e|al) inhibitor\b",
    ),
    # ── Viral / chemical inhibitors of TAP & ERAP & cathepsin ──
    "tap_inhibitor": (r"\bICP47\b", r"\bUS6\b", r"\bTAP inhibitor\b"),
    "erap_inhibitor": (r"\bERAP inhibitor\b", r"\bDG013A\b"),
    "cathepsin_inhibitor": (
        r"\bleupeptin\b",
        r"\bE-?64\b",
        r"\bpepstatin\b",
        r"\bcathepsin inhibitor\b",
    ),
    # ── Class II APM ──
    "ciita": (r"\bCIITA\b",),
    "hla_dm": (r"\bHLA-DMA?\b", r"\bHLA-DMB\b", r"\bHLA[- ]DM\b"),
    "hla_do": (r"\bHLA-DOA?\b", r"\bHLA-DOB\b", r"\bHLA[- ]DO\b"),
    "cd74": (r"\bCD74\b", r"\binvariant chain\b"),
    "cathepsin": (r"\bcathepsin\b", r"\bCTS[BLS]\b"),
    # ── Class-II loci umbrella + bare lymphocyte syndrome ──
    "rfx": (r"\bRFXANK\b", r"\bRFXAP\b", r"\bRFX5\b"),
    "bls": (r"\bbare lymphocyte\b",),
    # ── Cytokine inducers (not APM components, but APM-modulating;
    # included so a single union flag captures the studies that
    # explicitly perturb antigen-presentation expression). Keep these
    # at the END of the dict so per-gene KO studies sort first under
    # apm_genes_perturbed lexicographic listing.
    "ifn_gamma": (
        r"\bIFN-?γ\b",  # noqa: RUF001
        r"\bIFN-?gamma\b",
        r"\binterferon[- ]gamma\b",
        r"\bIFNG\b",
    ),
    "ifn_alpha": (
        r"\bIFN-?α\b",  # noqa: RUF001
        r"\bIFN-?alpha\b",
        r"\binterferon[- ]alpha\b",
    ),
    "ifn_beta": (
        r"\bIFN-?β\b",
        r"\bIFN-?beta\b",
    ),
    "tnf_alpha": (
        r"\bTNF-?α\b",  # noqa: RUF001
        r"\bTNF[- ]alpha\b",
        r"\bTNFα\b",  # noqa: RUF001
    ),
    "lps": (r"\bLPS\b", r"\blipopolysaccharide\b"),
    # ── TAP-deficient cell line lineage (T2 / RMA-S) ──
    # T2 (the .174 x CEM hybrid) and RMA-S are TAP1/TAP2-deficient by
    # deletion; any peptidome from these lines is implicitly a TAP-
    # deficient state regardless of any CRISPR-style annotation.
    "tap_deficient_line": (r"\bT2 cells?\b", r"\bT2 lymphoblast\b", r"\bRMA-S\b"),
}

# Compiled regexes — built once at import. Each gene's patterns are
# OR'ed together into a single pattern.
_GENE_REGEX: dict[str, re.Pattern[str]] = {
    gene: re.compile("|".join(pats), re.IGNORECASE) for gene, pats in APM_GENES.items()
}


def classify_apm_perturbations(*texts: str | None) -> dict[str, bool]:
    """Return ``{gene: bool}`` for every APM gene in :data:`APM_GENES`.

    Concatenates the input ``texts`` (typically the sample's
    ``condition`` plus the study-level ``perturbations`` joined with
    spaces) and scans for each gene's regex.  ``None`` / empty inputs
    are tolerated.

    The output dict is **stable**: every gene in :data:`APM_GENES`
    gets a key, regardless of whether it matched.  Consumers can
    rely on the column shape per-row.
    """
    blob = " ".join(t for t in texts if t)
    return {gene: bool(rx.search(blob)) for gene, rx in _GENE_REGEX.items()}


def apm_columns_for_sample(
    condition: str | None,
    study_perturbations: Iterable[str] | None = None,
) -> dict[str, object]:
    """Build the per-sample APM column block for the ms_samples table.

    The per-gene flags describe **this sample only** and are derived
    from its own ``condition`` string.  The parent study's
    ``perturbations`` list is reported separately.

    That separation is the whole point (issue #353).  Folding the
    study-level list into the per-gene flags meant any study curating
    a *panel* of perturbations marked every one of its arms — the
    untreated control included — as perturbed for every gene in the
    panel.  The Shapiro HAP1 CRISPR panel (PMID 40113210) made all 12
    arms claim the same 11 genes, so ``HAP1 wildtype`` reported
    ``apm_erap1_perturbed=True`` and per-gene filtering selected whole
    studies instead of perturbed samples.

    Returns a dict with one ``apm_<gene>_perturbed`` boolean per gene,
    plus:

    - ``apm_perturbed`` — union flag, ``"true"`` iff any gene matched
      this sample's own condition.  A **tri-state string**, matching
      ``is_control_arm``: on ``ms_samples`` every row is a sample so it
      is never blank, but the observation-level join leaves it ``""``
      where no arm could be resolved.  It was a plain ``bool`` until
      #392, which meant an unresolved arm was reported as a positive
      claim that nothing was perturbed — indistinguishable from a real
      WT control, and mislabeled toward the control class, the one
      direction that cancels the KO-vs-WT contrast rather than merely
      adding noise.
    - ``apm_genes_perturbed`` — semicolon-joined list of matching
      gene names (lowercase keys from :data:`APM_GENES`), empty when
      none match. Lets consumers filter to specific genes via a
      string-contains check without re-parsing.  Its ``""`` is
      ambiguous in the same way for the same reason, so read
      ``apm_perturbed`` to tell "no genes" from "no arm".
    - ``study_apm_perturbed`` / ``study_apm_genes`` — the same union
      over the parent study's ``perturbations`` list, so the panel
      context stays queryable without being mistaken for a
      sample-level fact.  See :func:`study_apm_columns`.
    """
    flags = classify_apm_perturbations(condition)
    out: dict[str, object] = {f"apm_{gene}_perturbed": v for gene, v in flags.items()}
    out["apm_perturbed"] = "true" if any(flags.values()) else "false"
    out["apm_genes_perturbed"] = ";".join(g for g, v in flags.items() if v)
    out.update(study_apm_columns(study_perturbations))
    return out


def study_apm_columns(
    study_perturbations: Iterable[str] | None,
) -> dict[str, object]:
    """The study-level APM block, from a deposit's ``perturbations`` list.

    Split out from :func:`apm_columns_for_sample` because these two
    columns are a property of the *study*, not of any sample in it.  The
    export layer joins them on PMID for exactly that reason: routing
    them through the sample join made every row whose arm was ambiguous
    report ``study_apm_perturbed=False``, denying a perturbation the
    deposit plainly records (#392).

    Unlike the per-sample block there is no unknown state — a PMID's
    ``perturbations`` list is either present or absent — so
    ``study_apm_perturbed`` stays a plain ``bool``.
    """
    flags = classify_apm_perturbations(*(study_perturbations or []))
    return {
        "study_apm_perturbed": any(flags.values()),
        "study_apm_genes": ";".join(g for g, v in flags.items() if v),
    }
