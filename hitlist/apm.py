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
    "erap1": (r"\bERAP1\b",),
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
    perturbations: Iterable[str] | None = None,
) -> dict[str, object]:
    """Build the per-sample APM column block for the ms_samples table.

    Returns a dict with one ``apm_<gene>_perturbed`` boolean per gene,
    plus:

    - ``apm_perturbed`` — union flag (``True`` iff any gene matched).
    - ``apm_genes_perturbed`` — semicolon-joined list of matching
      gene names (lowercase keys from :data:`APM_GENES`), empty when
      none match. Lets consumers filter to specific genes via a
      string-contains check without re-parsing.
    """
    perts = list(perturbations or [])
    flags = classify_apm_perturbations(condition, *perts)
    out: dict[str, object] = {f"apm_{gene}_perturbed": v for gene, v in flags.items()}
    out["apm_perturbed"] = any(flags.values())
    out["apm_genes_perturbed"] = ";".join(g for g, v in flags.items() if v)
    return out
