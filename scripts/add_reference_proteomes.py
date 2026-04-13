#!/usr/bin/env python3
"""One-shot utility: add reference_proteomes to ms_samples entries in
pmid_overrides.yaml for viral / host context we can identify.

- EBV (UP000153037) → every ms_sample mentioning JY, Raji, B-LCL, EBV-LCL
- Influenza A (UP000009255) → samples mentioning "influenza" or infected
  lung samples in the Nicholas 2022 and Wu 2019 PMIDs
- SARS-CoV-2 (UP000464024) → Weingarten-Gabbay 2021, Gomez-Zepeda Raji-spike
- Vaccinia (UP000000344) → Schellens 2015 vaccinia-infected
- HIV-1 (UP000002241) → Ramarathinam 2018, Chikata 2019
- HCMV (UP000000938) → Hassan 2013 JYpp65 (CMV pp65 transduced)

Run once; re-running is safe (idempotent — skips samples that already
have reference_proteomes set).
"""
from __future__ import annotations

import sys
from pathlib import Path

from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap, CommentedSeq

YAML_PATH = Path(__file__).parent.parent / "hitlist" / "data" / "pmid_overrides.yaml"

EBV_UPID = "UP000153037"
INFLUENZA_A_UPID = "UP000009255"
SARS2_UPID = "UP000464024"
VACCINIA_UPID = "UP000000344"
HIV1_UPID = "UP000002241"
HCMV_UPID = "UP000000938"
# Note: host proteome is NOT listed in reference_proteomes — the primary
# flanking pass already tries the host via source_organism / pyensembl.
# reference_proteomes is for ADDITIONAL proteomes (viruses, transgenes,
# custom peptide DBs) to try for peptides unmatched by the primary pass.


def _make_proteome(upid: str, label: str) -> CommentedMap:
    m = CommentedMap()
    m["uniprot"] = upid
    m["label"] = label
    return m


def _ensure_reference_proteomes(
    sample: CommentedMap, proteomes: list[tuple[str, str]]
) -> bool:
    """Set sample.reference_proteomes if not already present.  Returns True if modified."""
    if "reference_proteomes" in sample:
        return False
    seq = CommentedSeq()
    for upid, label in proteomes:
        seq.append(_make_proteome(upid, label))
    sample["reference_proteomes"] = seq
    return True


def _type_matches_any(sample: CommentedMap, patterns: list[str]) -> bool:
    """Case-insensitive match across type, condition, culture_condition."""
    haystack = " ".join(
        (sample.get(k) or "") for k in ("type", "condition", "culture_condition")
    ).lower()
    return any(p.lower() in haystack for p in patterns)


# (pmid_int, [(list of patterns to match sample type), list of (upid, label)])
# If patterns is None, applies to every ms_sample for that PMID.
PATCHES: list[tuple[int, list[str] | None, list[tuple[str, str]]]] = [
    # === EBV-LCL (host handled by primary pass, EBV added here) ===
    (23481700, ["JYpp65"], [(EBV_UPID, "Epstein-Barr virus"), (HCMV_UPID, "Human cytomegalovirus")]),
    (23481700, ["HHC"], [(EBV_UPID, "Epstein-Barr virus")]),
    (24616531, None, [(EBV_UPID, "Epstein-Barr virus")]),
    (24714562, None, [(EBV_UPID, "Epstein-Barr virus")]),
    (25502872, None, [(EBV_UPID, "Epstein-Barr virus")]),
    (25576301, ["JY"], [(EBV_UPID, "Epstein-Barr virus")]),
    (26728094, ["EBV-LCL"], [(EBV_UPID, "Epstein-Barr virus")]),
    (27841757, ["EBV-LCL"], [(EBV_UPID, "Epstein-Barr virus")]),
    (27846572, ["LCL", "JY"], [(EBV_UPID, "Epstein-Barr virus")]),
    (28832583, ["EBV-LCL"], [(EBV_UPID, "Epstein-Barr virus")]),
    (28834231, ["JY"], [(EBV_UPID, "Epstein-Barr virus")]),
    (29242379, ["B-LCL"], [(EBV_UPID, "Epstein-Barr virus")]),
    (29508533, ["EBV-LCL"], [(EBV_UPID, "Epstein-Barr virus")]),
    (32350084, ["EBV-LCL"], [(EBV_UPID, "Epstein-Barr virus")]),
    (32357974, None, [(EBV_UPID, "Epstein-Barr virus")]),
    (32897882, ["JY"], [(EBV_UPID, "Epstein-Barr virus")]),
    (33298915, None, [(EBV_UPID, "Epstein-Barr virus")]),
    (34211107, None, [(EBV_UPID, "Epstein-Barr virus")]),
    (34932366, None, [(EBV_UPID, "Epstein-Barr virus")]),
    (35154160, None, [(EBV_UPID, "Epstein-Barr virus")]),
    (38480730, ["JY"], [(EBV_UPID, "Epstein-Barr virus")]),
    (38480730, ["Raji"], [(EBV_UPID, "Epstein-Barr virus"), (SARS2_UPID, "SARS-CoV-2")]),

    # === Vaccinia (EBV-LCL ± vaccinia) ===
    (26375851, ["uninfected"], [(EBV_UPID, "Epstein-Barr virus")]),
    (26375851, ["vaccinia"], [(EBV_UPID, "Epstein-Barr virus"), (VACCINIA_UPID, "Vaccinia virus")]),

    # === Influenza-infected ===
    (31253788, ["influenza"], [(INFLUENZA_A_UPID, "Influenza A virus")]),
    (35051231, ["Wisconsin", "X31"], [(INFLUENZA_A_UPID, "Influenza A virus")]),
    # UV mock is uninfected, no extras

    # === SARS-CoV-2 ===
    (34171305, ["SARS-CoV-2"], [(SARS2_UPID, "SARS-CoV-2")]),

    # === HIV-1 ===
    (29437277, None, [(HIV1_UPID, "Human immunodeficiency virus 1")]),
    (31217245, None, [(HIV1_UPID, "Human immunodeficiency virus 1")]),
]


def main() -> int:
    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.indent(mapping=2, sequence=4, offset=2)
    yaml.width = 4096  # don't wrap long lines

    with open(YAML_PATH) as f:
        docs = yaml.load(f)

    # Index PMIDs for quick lookup
    by_pmid: dict[int, CommentedMap] = {}
    for entry in docs:
        try:
            by_pmid[int(entry["pmid"])] = entry
        except (KeyError, ValueError, TypeError):
            continue

    changes: list[str] = []
    for pmid, patterns, proteomes in PATCHES:
        entry = by_pmid.get(pmid)
        if entry is None:
            print(f"  ⚠ PMID {pmid} not found, skipping", file=sys.stderr)
            continue
        samples = entry.get("ms_samples") or []
        for sample in samples:
            if patterns is not None and not _type_matches_any(sample, patterns):
                continue
            if _ensure_reference_proteomes(sample, proteomes):
                label = ", ".join(p[1] for p in proteomes)
                changes.append(
                    f"  PMID {pmid:>8d}  [{sample.get('type', '?')[:45]:45s}]  ← {label}"
                )

    if not changes:
        print("No changes — all targeted samples already have reference_proteomes")
        return 0

    print(f"Applying {len(changes)} updates:\n")
    for line in changes:
        print(line)

    with open(YAML_PATH, "w") as f:
        yaml.dump(docs, f)

    print(f"\nWrote {YAML_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
