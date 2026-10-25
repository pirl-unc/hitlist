# Method-supported F3 policy verification

2026-10-25; Codex/GPT-6, Python3.12.6, source checkout selected with PYTHONPATH.
No shared editable-install modification. Version2.0.41 extends the unmerged PR848.

`./format.sh` and `./lint.sh` pass. First focused run:27passed/3failed because the
synthetic original-source manifest did not include the article asset. The fixture
now declares and hashes that asset. Failure log preserved; production admission
rules were not relaxed. Second focused run:30passed in27.28s, including exact
strict-F2 and strict-F3 output parity against d76c225 and7c427e3 respectively.

Full default suite:3,594passed/392warnings in332.17s;336.23s wall;
906,149,888bytes maximum resident set size. No source edited during this suite.
Integration tests are separate hosted CI checks, not covered by this local run.

Exact full-suite command (log includes shell timing):

```sh
env PATH=/Users/iskander/code/shared-virtual-env/bin:/private/tmp/presto-pr66-tools/bin:/usr/bin:/bin:/usr/sbin:/sbin PYTHONPATH=/Users/iskander/code/presto/.worktrees/hitlist-organ-transfer PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 TEST_SH_MAX=1 ./test.sh -q
```

Focused: same environment, `python -m pytest -q tests/test_intact_organ_method.py
tests/test_intact_organ_transfer.py tasks/verification/test_f2_reference_parity.py
tasks/verification/test_f3_reference_parity.py`.

Previous published7c427e3 CI40216126367 passed all Python3.10–3.14, integration,
lint and combined coverage. New code still requires new-head CI and release.
