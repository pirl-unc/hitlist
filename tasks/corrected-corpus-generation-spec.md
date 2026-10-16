# Corrected frozen corpus generation — #678 / #680

Rebuild all observation/binding/contributor/mapping artifacts with the independently
verified released Hitlist1.78.0 producer. The existing ci-corpus-v3 and human
baseline were produced with1.76.0 and must remain unchanged. This generation must
include the shipped source-arm and class-II pair corrections, while retaining
unresolved treatment, experimental, donor/specimen and acquisition identities.

Preserve the original baseline lock; copy it to corrected-corpus-requirements,
changing only the Hitlist wheel/
sdist pin to the published1.78.0 hashes. Pin a frozen copy of the official Ensembl112
reference-stage receipt and verify both receipt and every reference artifact before
reuse. This reuses sequence/reference bytes and indexes only, never observations,
curation outputs or peptide mappings. Record the original reference producer and
versions in the new receipt; require compatible reference-library versions. Build
all source indexes and peptide mappings in a separate content-addressed datacache
root using existing atomic generation publication. Exact raw IEDB/CEDAR bytes and
paper-asset hashes remain pinned. Raw export dates stay explicitly unverified.

Preserve the original six-stage workflow, resource limits and failure receipts.
Stage1 checks the released producer and pip compatibility before data loading.
Sources verify/copy only pinned inputs; indexes rebuild with current curated data;
reference stages the verified prior frozen sequence cache; mappings rebuild against
it on local POSIX scratch; snapshot verifies and exports one complete generation.
Each stage records recipe/dependency/source hashes, elapsed time, runner-only RSS,
byte inventories and logs through shared datacache. CPU8/RAM64GiB/scratch512GiB,
90-minute function/85-minute work limit per stage; no GPU. Never overwrite old
public release tags or mutate shared environments.

Before launching, verify corrupted/missing reference receipt/artifacts fail closed,
reference metadata/version compatibility is enforced, and existing fresh-reference
construction still works. Run format/lint and focused regressions. Freeze clean
launcher/code/requirements/source pins and publish a draft PR with version1.79.0.
Run actual stages sequentially; don't repeat completed verified stages merely for
fresh timings. Archive every actual failure and repair the root cause separately.

After the complete generation succeeds, independently recover and hash-check all
files, compare observation counts/context changes with v3, and publish a new
complete downloadable release with human-readable qualification limits. Export a
separate human cohort and retain original source/cell/condition/MHC/assay metadata,
contributors and label policies. Update the Presto qualification input only in a
new prospective experiment with explicit cap-selection semantics and frozen
membership; no claims are transferred from the original baseline. Existing
within-search detectability results are a separate contract.

Acceptance is software and artifact delivery, not a fabricated scientific pass:
full format/lint/test and integration/release gates, exact-head review, merged PR,
clean-main PyPI deployment and independent public downloads. #678/#680 remain open
where biological independence or target-specific fitting evidence remains missing.
