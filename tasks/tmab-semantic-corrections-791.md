# TMAB source semantics and recipe readiness (#791)

## Specification, frozen before implementation

Base main f4d3a6d0f1933162f9ac6eac42ac1b134c8a27dc; isolated branch/worktree.
The original Presto0117 source audit and canonical v1 package stay immutable.
Scope is shared Hitlist source correction, not a new curator or model fit.

### Identity

Use an explicit source-notation parser rather than uppercase-letter extraction.
Strip one complete outer tentative wrapper before recognizing acetylation. Parse
canonical residues and Mox tokens, preserving supported modification positions.
Keep identification status (reported/tentative/unidentified/unresolved annotation),
modification status and numeric exactness separately queryable. PyroE precursor
identity is ambiguous without a parent mapping; defer it rather than assuming E/Q.
Unsupported text, internal parentheses and uncertain dehydration annotations keep
raw text and yield no confident normalized sequence. Existing row/measurement IDs
and raw source values remain unchanged. Expose the correct Gelman S2 column F.

### Reuse and eligibility

Build source-independent candidate row fingerprints from original cell, sequence,
charge, observed mass and the five replicate/control ratio cells, requiring a
nonempty sequence and at least two numeric ratios. Keep the full source identity
separate. Mark groups spanning papers as possible reuse and report whether their
case/whitespace-normalized experiment labels agree. Matching fingerprints are not
proof of physical specimen identity, and unmatched rows remain unresolved rather
than source-unique. Per-slot lineage may share a candidate-row group and role/rep
while retaining each unique measurement ID. Preserve all candidate row links.

Numeric exactness alone cannot make a row sequence-ready. Distinguish confident
positive treated-ratio candidates (exact or censored) from unidentified/tentative/
unresolved sequences, nonpositive quantities, controls, averages and missing/error
states. These are source eligibility candidates, not an independence-qualified
training recipe. No original zero, bound or missing cell is replaced or clipped.

### Protocol context

Separate the publishing study's protocol instrument from the original acquisition.
Gelman Methods report G1; Dasgupta report G2. Known republished bortezomib/epoxomicin
panels cannot inherit the later instrument. Keep their acquisition unresolved until
qualified; original2012 epoxomicin source is not available in the audit.

Retain both experiment-column and average-label assertions in each context. Parse
reported incubation/harvest components only from supported annotation syntax;
unknown durations stay unknown. Distinguish reported total duration from incubation
and harvesting. The2014 new-experiment protocol retains inhibitor during washing;
its20+15minute annotation supports35minutes total, not washout. The2013 protocol
uses inhibitor in some washes, so row-level presence remains unresolved. Full
compound labels must distinguish bestatin from bestatin methyl ester. No chemical
treatment becomes KO/subunit loss or known in-cell residual activity.

### Versioning and shared surface

Publish corrected TMAB revisionv2, package schema16, catalog21, code2.0.16 subject
to a final current-version check. Reject v1 for corrected interpretation rather
than silently relabeling an existing immutable package. Historical reports retain
their pinned v1 code. Add fields to existing source/context/measurement/lineage
relations and public catalog bindings; no competing sidecar metadata system.

### Verification and delivery

- [x] Add meaningful real-example regressions and a miniature two-paper workbook
  package fixture; record failing-before behavior against unchanged code.
- [x] Implement semantic corrections and catalog/schema/version changes.
- [x] Verify package snapshot/row/quantity/identity/context/lineage through public
  writer, EvidenceStore queries, relationships and tamper detection.
- [x] Register a new Presto source-replay family before comparing full real assets;
  preserve every raw row/ratio/measurement ID and independently verify changed
  semantics against frozen source cells. No predictive split in this source audit.
- [x] Run ./format.sh, ./lint.sh, ./test.sh; inspect any failures instead of bypassing.
- [ ] Review diff, publish PR and satisfy hosted/release requirements; do not claim
  delivered until merged and published. Preserve user/shared Python environments.
- [ ] Follow with a separately specified canonical native-response recipe and real
  biological comparison. This source fix does not qualify APC concentrations.

## Review

Implementation and real-source replay are verified. Presto family
2026-10-24_0156_codex_tmab-semantic-replay preserves all source rows/quantities,
checks every normalized sequence and structured modification independently,
and reproduces all 830 candidate reuse memberships. The package manifest is
f84bd8e738a2c14d446956c719cfeaba91bc867a95aebd62dac1c91d3e3a1666.
The initial context-label comparator incorrectly expected untrimmed projections;
its failure is preserved, and separate closure verifies all 1,242 differences
against both package versions and the untouched raw rows. Source candidate
counts are 7,933 exact and 4,662 bounded treated/control ratios. Independence,
parent mapping and canonical recipe admission remain unresolved. No training
or source-value repair has occurred. Full ./test.sh passes (3,403 tests); hosted checks and release delivery remain pending.
