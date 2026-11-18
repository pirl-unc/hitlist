# Final qualification of DP/DM source PR #898

Reconcile the existing source integration with the export, filter and APC fixes
without changing its ten source samples or original PSM evidence. Reserve 2.1.16
after 2.1.15; confirm actual main and published versions before release.

Review source manifest hashes and all packaged rows, exact DPA1/DPB1 pairs,
explicit DM background/control semantics, source/sample/acquisition identity,
control exclusion, and chemistry. Verify the primary supplementary methods.
Do not relabel A/B samples as independently proven physical specimens or treat
spectral support as abundance. Trace memberships through the common export and
check for contradictions with the APC changes. File and resolve in-scope bugs.

Keep the existing protected-cohort boundary: 8,338 identity-overlap rows remain
embargoed and the other 13,209 are not automatically eligible for Presto TRAIN.
This PR makes source evidence available; consumer split reconciliation remains
separate from source qualification and software correctness.

Run format/lint, guarded tests and final-head hosted unit/integration/release
checks, then review, merge and verify both public PyPI distributions. Preserve
the original failed test evidence and use an isolated environment only.
