# Lymph composition recipe verification

Source PR818 merged as9133736d; source release2.0.27 is separately building.
This follow-on exposes a deterministic registered endpoint without rewriting the
historical source-only source manifest. Existing source column schemas are unchanged.

Format/lint pass. Eleven recipe tests plus eight source tests pass. Full required
`TEST_SH_MAX=1 ./test.sh` passes3,477tests/392warnings in306.46seconds. The initial
sandbox preflight could not probe memory; rerun used actual host memory access,
not an override to suppress the guard. Initial test-module import and lint failures
were corrected before the registered real-source admission.

Actual source materialization/replay:32,995quantity cells,4,384source observations,
2,472admitted representative-feature measurements. Independent original workbook
area, all-source graph, selection, parent alternatives and target reconstruction
passes. Human native component support34/14/12train/val/test; DCdatabase36/1/6 and
DClibrary32/3/11. The two DC cohorts fail the prospectively declared validation
support gate and are not fitted. No split/chemistry/origin relaxation followed.

The downstream human exact/REINFORCE comparison is recorded in Presto experiment
2026-10-24_1938_codex_lymph-composition; predictive success is not claimed by source
admission. Source context, ambiguity, missingness and all excluded decisions remain
available for future independently specified analyses.
