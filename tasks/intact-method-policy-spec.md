# Separate method-supported F3 search-assignment policy

Qualification is recorded in Presto family2026-10-25_1735. Original article
PMC11874349 Methods/ID-MS Data Analysis and Statistics explicitly reports no
additional modifications in the ClipsMS search. Independent XML parsers agree.
The official ClipsMS source commit8ad829b (2022-07-22) supplies a zero-mass `None`
option, but the exact study software version/export roundtrip remain unknown.
The declaration supports a search assignment, not proof of biochemical purity.

Do not change raw nulls, the original source contract/generation, the F2 recipe,
or the strict F3 recipe. Add a separately registered
`pxd058121_organ_method_baseline_change_v1` endpoint. All existing mass/span/link,
normalizer/baseline/support/weight rules remain identical.

The derived recipe snapshots the already-pinned original article as an additional
source asset. It records a typed search-policy assertion with source hash, exact
paragraph hash, locator, reported-method status and study/search scope. Missing
source-record IDs remain missing; do not invent a canonical record ID. The original
source package remains verifiable by the unchanged source contract.

Admission accepts original explicitly-unmodified rows, or blank unlocalized fields
only when both reported localized/terminal values are zero and the original Methods
paragraph supplies the qualified no-additional-modifications declaration. Nonzero
or contradictory row values, unsupported/missing declaration and unrelated study
scope do not qualify the fallback. Decisions and covariates identify the policy and
assertion IDs used. Future response values never establish this interpretation.

Share existing organ compiler arithmetic through an internal policy implementation
hook; strict default output schemas and rows must remain identical. Extend package
snapshot/replay to optional, explicitly named source assets, defaulting to none.
Verify their hashes against the source manifest and replay the policy table.

Tests: method-supported blanks, blanks with no supporting Methods declaration,
wrong study/scope, nonzero/contrary modifications, original null preservation,
policy/target separation and rehashed article/policy tampering. Compare strict F3
outputs against immutable7c427e3 on the same synthetic fixture; preserve the prior
F2 parity check. Run required format/lint/full default tests before publication.
Version remains2.0.41 because this extends the same unmerged PR848. Keep the PR
draft until real new-cohort admission/replay verifies; ship only after CI/release.

The next Presto comparison must be separately registered before numeric admission,
with the same31frozen model files,21arms and original quality/support gates.
