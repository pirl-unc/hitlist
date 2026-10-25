# PXD052314 material-context audit

This is the source-fact receipt for Goldberg2025 / PXD052314 material-context
qualification in 2.0.37. It audits the pinned bytes named in
`hitlist/data/evidence_revisions/goldberg_mapp.json`, not a reconstructed
cohort. A row is an assay acquisition, not a donor, specimen, aliquot or
independent biological sample.

## Source coverage

| Source fact | Covered acquisitions | Admitted representation |
|---|---:|---|
| Deposition raw file, experiment and literal sample label | 36 / 36 | acquisition, source record, original-deposition assertions |
| Shared literal sample-label context | 36 / 36, 12 labels | `source_sample` material context and acquisition-to-context link |
| Table5 header → deposited acquisition | 36 / 36 | publisher-header assertion after exact raw-token + label-suffix bridge |
| Table5 ReadMe condition description | 36 / 36 | acquisition-scoped source assertion |
| Strictly parsed role, duration, reported replicate | 36 / 36 | `reported_condition_*` assertions with ReadMe locator |
| Physical specimen / aliquot identifier | 0 / 36 | null; no physical link emitted |
| Table3 pooled-slot run / physical sample identity | 0 slots qualified | retained ordered unresolved slots only |

The deposited labels are `Ctrl_1`, `Infect_1`, `Ctrl_2`, `Infect_2`,
`Ctrl_3`, `Infect_3`, `Ctrl4_1`, `Infect4_1`, `Ctrl4_2`, `Ctrl4_3`,
`Infect4_2` and `Infect4_3`. They recur across the whole-cell, MAPP-peptide
and proteasome-composition experiment labels. The implementation creates one
project-scoped **source-label** context per literal, then preserves the three
assay views as separate acquisitions. It does not claim that recurrence means
the same prepared material.

## Exact bridge and conflicts

Table5 headers contain a run label such as `5424_A549AMP30_Ctrl_1`; the
deposition raw name contains the same literal four-digit run token and reports
the suffix `Ctrl_1`. The bridge accepts only a unique triple of:

1. declared Table5 sheet ↔ deposited experiment mapping;
2. four-digit header token ↔ literal token in one deposited raw-file name; and
3. header suffix ↔ that row's exact deposited sample label.

The original MaxQuant `summary.txt` provides its own complete run labels for
the MAPP-peptide view. Ten of twelve agree with the above bridge. Two do not:

| Raw-file token | Deposition / Table5 label | `summary.txt` experiment label |
|---:|---|---|
| 5450 | `A549AMP30_Infect4_2` | `5450_A549AMP30_Infect4_3` |
| 5454 | `A549AMP30_Infect4_3` | `5454_A549AMP30_Infect4_4` |

Both source assertions remain attached to the same acquisition and the
publisher-header assertion is explicitly status-labelled as conflicting with
the original search run. It is not converted to an alias or silently repaired.
The 24 other deposited acquisitions have no original search summary in the
pinned MAPP archive; their condition bridge is limited to deposition plus
publisher header/ReadMe evidence.

The community SDRF is retained in its own `community_sdrf` assertion scope
when its raw name exactly matches a deposited raw file. It is not used to
replace deposition labels or create a specimen/aliquot identity.

## Query boundary

Use `mapp_source_assertion` for the source-scoped condition claims and
`mapp_material_context`/`mapp_material_link` for source-label grouping. The
source-label contexts are valid only for source grouping; `exact_specimen` and
`exact_aliquot` queries still return no PXD052314 links. Table3 must be queried
through `mapp_pooled_quantity_slot` and is intentionally outside this bridge.
