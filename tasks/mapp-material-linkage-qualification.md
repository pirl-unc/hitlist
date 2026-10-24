# MAPP material-linkage qualification

## Objective

Make the material context carried by MAPP and adjacent proteomics sources
queryable at its actual source-backed precision. Preserve every source run,
condition label, specimen/aliquot unknown and cross-modal non-link; never turn
publication-level parallelism, run ordering, replicate number, cell line or
abundance correlation into a physical-material identifier.

## Audited coverage at start

The Wolf-Levy 2018 paper lists seven PRIDE projects. Hitlist currently exposes
three captured-fragment MAPP deposits:

| Project | Current coverage | Source-run condition status |
| --- | --- | --- |
| PXD009092 | 10,673 peptide rows, 149,959 evidence rows, 18 runs | Six three-run source-label arms: UT, TNF+IFN, CHX, NO-DSP, NO-INH and BEADS |
| PXD007263 | 3,146 peptide rows, 27,196 evidence rows, 12 runs | Four three-run literal TNF exposure-time labels: 0, 1, 2 and 6 h |
| PXD009508 | 6,060 peptide rows, 19,581 evidence rows, 18 runs | alpha6/isotype/GAPDH capture labels crossed with untreated/epoxomicin, three numbered acquisitions each |

That is 19,879 deposited peptide rows (not a deduplicated peptide universe),
196,736 evidence rows and 48 MAPP acquisition runs. PXD009438 is the fourth
captured-fragment MAPP deposit (primary PBMC SLE/healthy study) and is not yet
admitted. PXD009360, PXD009181 and PXD009112 are deposited tryptic panels, not
MAPP measurements, and are not a substitute for the missing MAPP PBMC panel.

The manuscript reports a HEK293 untreated versus TNF-alpha+IFN-gamma 24-hour
comparison analyzed by MAPP and standard proteomics in parallel. This supports
a **publication-level modality/comparison assertion**. It does not identify
which PXD009360 acquisition file belongs to either arm, prove that it is one of
the stated runs, or link a MAPP run to the same specimen or aliquot. The present
PXD009360 archive has nine labels (448--456), no deposited condition map, and
no material crosswalk. Its unresolved claims remain correct at run/specimen/
aliquot precision.

## Contract

Add a common, typed material-context graph for source adapters and standalone
context packages.

- A `material_context` is an explicitly scoped source subject: acquisition run,
  study arm, source project, reported specimen or reported aliquot. It has an
  ID, source ID, scope, literal labels, optional normalized condition and exact
  evidence status/locator.
- A `material_link` connects two material contexts with a typed predicate and
  *separate* assertion status and precision. Allowed precisions are exact
  `specimen`, exact `aliquot`, source-acquisition, study-arm, project-context
  and unresolved. A study-arm or project-context link is never promoted to a
  specimen/aliquot link.
- The graph must retain negative evidence (`unreported_in_admitted_assets`) and
  candidate associations (`same_publication_candidate`) alongside established
  claims. All unknown IDs stay null.
- Conditions require a source label or an explicit source-backed translation;
  a named project, replicate number or adjacent acquisition number is not a
  condition map. A source-backed label translation records its source locator
  and scope.
- Recipe and training consumers must opt in to a minimum material-link
  precision. The default is to reject cross-modal pairing unless a source-backed
  exact link satisfies the requested precision.

## Serial implementation plan

1. **Material graph and existing MAPP facts.** Add generic evidence bindings,
   schemas, verification and query support for material contexts/links. Emit the
   already source-backed PXD009092, PXD007263 and PXD009508 run-to-arm/condition
   facts. Explicitly retain their unknown specimen/aliquot identity. No new
   training examples.
2. **Missing MAPP PBMC source.** Freeze PXD009438 asset receipts; audit its
   original search archive, supporting metadata and publication tables. Admit
   its MAPP source rows only with exact source conditions/specimen IDs where
   supplied; otherwise emit unknowns and a coverage ledger. Keep patients,
   controls and sample counts separate from run identity.
3. **Parallel tryptic recovery.** Audit PXD009360 and, where relevant,
   PXD009181/PXD009112 original submitted metadata, archive members and
   publisher material for an exact acquisition-to-condition/specimen/aliquot
   crosswalk. Add the publication-scoped HEK293 context only at its actual
   project/comparison precision. Upgrade only claims with locators and
   independently replay every join. If none is present, publish the
   negative-evidence audit and retain the graph's unresolved rows; no order-,
   condition-, peptide- or abundance-based pairing is permitted.

## Acceptance

- Every existing measurement/source row is conserved byte-for-byte or through
  a verified immutable predecessor relation.
- The real packages reproduce the counts above and expose the graph through the
  normal EvidenceStore/canonical catalog interface.
- Tests reject a populated specimen/aliquot ID without the required precision,
  prevent a study-arm link from satisfying an exact-pair query, and demonstrate
  that run order and matching labels cannot create a physical link.
- PXD009360 remains unavailable for paired precursor/MAPP training until the
  source proves the required link. The graph may make its project-level context
  queryable but may not turn that context into a training pair.
- Run format, lint, full tests, hosted checks, clean-main release and PyPI
  publication for each PR. Record released hashes and issue evidence.
