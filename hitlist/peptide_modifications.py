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

"""Parse IEDB PTM annotations from the Epitope|Name column.

IEDB sometimes embeds post-translational modifications inline in the
peptide name, e.g. ``LQPFPQPQLPY + DEAM(Q8)`` (deamidated Q at position
8). When stored as-is, these strings break length-based filters, model
training inputs, and amino-acid validation. This module pulls the
modification out into a structured form while preserving the
information.

Output of :func:`parse_peptide_modifications`:

- ``peptide`` — bare AA sequence (no annotation).
- ``peptide_modifications`` — canonical ``MOD:<residue><pos>;...`` string,
  empty when no PTM. Sortable, parquet-friendly, easy to filter on.
- ``has_ptm`` — boolean convenience flag.
- ``peptide_extended`` — ProForma 2.0-style string with PTM tags inline,
  e.g. ``LQPFPQPQ[Deamidated]LPY``. Suitable for PTM-aware model inputs.
"""

from __future__ import annotations

import re

# IEDB format: BARE + MOD(POSITIONS). POSITIONS is one or more
# "<residue><digits>" separated by commas (with optional whitespace).
# Examples in the corpus:
#   "LQPFPQPQLPY + DEAM(Q8)"
#   "CLGGLLTMV + MCM(G3,G4,T7,M8)"
#   "EVLYLKPLAGVYRSLKKQLE + MCM(K6, P7)"
#   "GILGFVFTL + MCM(F7, A8 )"     # note trailing space inside parens
_PTM_RE = re.compile(
    r"^(?P<bare>[ACDEFGHIKLMNPQRSTVWY]+)\s*\+\s*"
    r"(?P<mod>[A-Z][A-Z0-9]*)\((?P<positions>[^)]+)\)\s*$"
)
_POS_RE = re.compile(r"([A-Z])\s*(\d+)")

# Mapping from IEDB short codes to ProForma 2.0 monomer names. ProForma
# uses Unimod / PSI-MOD names; for codes that don't have a clean PSI-MOD
# tag we keep the IEDB short code so the extended string still encodes
# the modification type. See https://www.psidev.info/proforma.
_PROFORMA_TAG: dict[str, str] = {
    "ABA": "ABA",
    "ACET": "Acetyl",
    "AIB": "AIB",
    "AMID": "Amidated",
    "BIOT": "Biotin",
    "CITR": "Citrullination",
    "CYSTL": "Cysteinyl",
    "DEAM": "Deamidated",
    "DEHY": "Dehydrated",
    "FORM": "Formyl",
    "GAL": "Hex",
    "GLUC": "Hex",
    "GLUT": "Glutathione",
    "GLYC": "Glycosyl",
    "HYL": "Hydroxyl",
    "HYLGLUCGAL": "Hex(2)Hydroxyl",
    "IASA": "IASA",
    "INDIST": "Indistinguishable",
    "ISO": "Iso",
    "MCM": "MCM",
    "METH": "Methyl",
    "MULT": "Multiple",
    "MYRI": "Myristoyl",
    "NIT": "Nitro",
    "OTH": "Other",
    "OX": "Oxidation",
    "PALM": "Palmitoyl",
    "PHOS": "Phospho",
    "PYRE": "Pyro-glu",
    "RED": "Reduced",
    "SCM": "SCM",
    "SEC": "Selenocysteine",
    "SULF": "Sulfo",
    "UNK": "Unknown",
}


def parse_peptide_modifications(s: str) -> tuple[str, str, bool, str]:
    """Split an IEDB peptide string into bare sequence + modifications.

    Parameters
    ----------
    s
        Raw peptide string from IEDB's ``Epitope | Name`` column. May be
        a bare AA sequence (``"LQPFPQPQLPY"``) or carry a single PTM
        annotation (``"LQPFPQPQLPY + DEAM(Q8)"``).

    Returns
    -------
    (bare, modifications, has_ptm, extended)
        - ``bare`` — bare AA sequence (no annotation). Equal to the
          input when no PTM is present, or when the annotation is
          unparseable (preserves information rather than dropping).
        - ``modifications`` — canonical ``MOD:<residue><pos>`` string,
          multi-position modifications joined by ``;``. Empty string
          when no PTM.
        - ``has_ptm`` — ``True`` iff a PTM was successfully parsed.
        - ``extended`` — ProForma string with the PTM tag inline at the
          modified residue. Equal to ``bare`` when no PTM.
    """
    s = (s or "").strip()
    if not s:
        return "", "", False, ""
    if " + " not in s:
        return s, "", False, s
    m = _PTM_RE.match(s)
    if not m:
        # Unparseable PTM annotation — preserve the original string in
        # both bare and extended so we don't silently drop information.
        # Curators can grep these out via the "+" character.
        return s, "", False, s
    bare = m.group("bare")
    mod = m.group("mod")
    positions = _POS_RE.findall(m.group("positions"))
    if not positions:
        return s, "", False, s
    canonical = ";".join(f"{mod}:{r}{p}" for (r, p) in positions)
    tag = _PROFORMA_TAG.get(mod, mod)
    extended_chars = list(bare)
    # Insert tags walking right-to-left so earlier indices don't shift.
    for residue, pos1_str in sorted(positions, key=lambda rp: int(rp[1]), reverse=True):
        idx = int(pos1_str) - 1
        # Defensive: only annotate if the residue at the claimed position
        # actually matches what IEDB said it should be. Mismatches
        # indicate either the IEDB annotation is wrong or our parse is
        # off; in either case, fall through and skip the inline tag for
        # that position rather than emitting a misleading extended form.
        if 0 <= idx < len(extended_chars) and extended_chars[idx] == residue:
            extended_chars[idx] = f"{residue}[{tag}]"
    extended = "".join(extended_chars)
    return bare, canonical, True, extended
