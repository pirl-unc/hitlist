# Organ-transfer recipe validation

Issue: https://github.com/pirl-unc/hitlist/issues/847 (exact body read back).
Base: d76c225a0aa741a1a1b0aee707bff328ca467c7c; isolated feature branch.
No real F3 quantitative admission or predictive outcome evaluation preceded these
checks. Fixtures are synthetic software checks, not biological labels.

- `./format.sh`: passed; `./lint.sh`: passed.
- Focused tests plus pinned pre-change F2 compiler parity: 17 passed, 22.25 s.
- Required default `./test.sh -q`: 3,582 passed, 392 warnings, 413.12 s pytest
  (417.48 s process wall time); OS maximum resident size 738,590,720 bytes.
- One worker (`TEST_SH_MAX=1`), shared Python 3.12.6, explicit checkout PYTHONPATH,
  `PYTHONDONTWRITEBYTECODE=1`, `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`.
  Normal repository memory admission was retained; no installation changes.
- Focused attempt 1 had a fixture mapping-index error (6 passed, 10 setup errors).
  Attempt 2 had an incorrect expected baseline count (15 passed, 1 failed).
  The source adapter correctly excludes Brain nominal-zero R2 input-ID omissions;
  the fixture now reproduces that omission and checks the surviving baseline.

Raw logs: `tasks/verification/organ-transfer/{focused-1,focused-2,focused-3,full-1}.log`.
Pinned F2 comparison source: `tasks/verification/test_f2_reference_parity.py`.
Reproduce it with the same environment and:

```sh
./test.sh tasks/verification/test_f2_reference_parity.py -q \
  -k 'intact_organ_transfer or intact_kinetics or f2_reference_output_parity'
```

The full default suite excludes built-corpus integration tests by repository
policy. Deployment must still run its required release checks. Merge, deployment,
PyPI-byte verification and independent replay on the real F3 package are pending.
