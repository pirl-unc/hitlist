# Native source validation

Format and lint pass. All 33 focused checks pass, including nine new source tests.
The required default ./test.sh run passes 3,533 non-integration tests in 179.10s
(183.27s wall), with TEST_SH_MAX=2 and unchanged memory/disk guards. The maximum
process RSS reported by time is 887,554,048 bytes; this is not total worker memory.
The 393 warnings remain in the losslessly compressed `full-tests.log.gz`;
its exact original bytes are verified in `full-tests-log-integrity.json`. Shared Python was not modified.

Presto family 0519 independently verifies all original quantities, references,
origin alternatives, workbook cells, source-parent claims and public relationship
closure. The summary is copied here; the complete reproduction bundle and failure
receipts live in Presto experiments/2026-10-25_0519_codex_native-source-canonical.
No model was fitted and no biological predictive improvement is claimed.

Integration release gates, PR merge and verified PyPI publication remain pending.
