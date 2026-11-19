# Explicit Test Selection & Bounded PyPI Visibility Verification — #929, #867

## Scope & Goals

1. **Issue #929 (`test.sh`)**: Honor explicit pytest file/directory/node selectors without appending the entire `tests` directory, while preserving `pyproject.toml`'s `testpaths = ["tests"]` default, `-k` and `-m` filters, Bash 3.2 compatibility, two-process `--all` isolation, and all CPU/memory preflight guards.
2. **Issue #867 (`scripts/release_artifacts.py`)**: After uploading tested distributions (or verifying an in-progress propagation window), wait within a bounded deadline (`PUBLICATION_VISIBILITY_TIMEOUT_SECONDS = 300`, `PUBLICATION_POLL_INTERVAL_SECONDS = 5`, overridable via env/kwargs) for:
   - Project JSON inventory (`https://pypi.org/pypi/hitlist/<version>/json`, no-cache) and exact `https://files.pythonhosted.org` byte/SHA-256 verification of both distributions,
   - PEP 691 Simple-index JSON inventory (`https://pypi.org/simple/hitlist/`, `Accept: application/vnd.pypi.simple.v1+json`, no-cache) for exact version wheel + sdist entries,
   - Uncached isolated installer resolution (`sys.executable -I -m pip install --dry-run --ignore-installed --no-deps --no-cache-dir --disable-pip-version-check --index-url https://pypi.org/simple/ --quiet --report - hitlist==<version>`), without modifying any shared or local Python environment.

## Detailed Contracts

### 1. `test.sh` (#929)

- Remove the unconditional positional `tests` argument from `run_pytest`; rely on `[tool.pytest.ini_options] testpaths = ["tests"]` when no explicit path/node selector is passed.
- Preserve phase marker separation when a caller supplies `-m <expr>` or `-m=<expr>` before `--`:
  - Extract caller marker expressions into `user_marker` rather than letting a trailing `-m` overwrite the phase's `-m "not integration"` or `-m "integration"` flag (which previously allowed `./test.sh -m integration` to run integration tests under the 2.5 GB unit worker budget).
  - Compose the phase marker as `(${marker}) and (${user_marker})` when `user_marker` is non-empty; reject a dangling `-m` with exit code `2`.
- In `--all` mode when caller selection arguments are present (`has_user_selection == 1`):
  - Pytest exit code `5` (`no tests collected` after marker/keyword deselection in one phase, e.g., `./test.sh --all tests/test_public_api.py` or `./test.sh --all tests/test_perturbation_quantities_real.py`) is allowed on one phase provided the other phase exits `0`.
  - If **both** phases exit `5`, `test.sh` exits `5`.
  - Any other non-zero status (`1` test failure or preflight abort, `2` probe/usage/collection error, signals, etc.) aborts immediately without starting the second phase.
  - Unfiltered `./test.sh --all` (`deploy.sh`) continues to require exit code `0` from both phases.

### 2. `scripts/release_artifacts.py` (#867)

- **No repeat uploads**: `publish_distributions` uploads `pending` files at most once per invocation; subsequent bounded polling never re-runs `twine upload`.
- **Immediate conflict rejection**: On every poll of Project JSON or PEP 691 Simple JSON, any visible file for `version` with an unexpected filename, `yanked` flag, mismatched `sha256` or `size_bytes`, non-`https://files.pythonhosted.org` URL, or mismatched downloaded bytes raises `ValueError` immediately without retrying.
- **Error classification**: Only HTTP `404` on Project JSON / Simple JSON or `No matching distribution` / `Could not find a version` from uncached `pip install --dry-run` counts as pending CDN propagation. HTTP `401`/`403`/`429`/`5xx`, invalid JSON, timeouts, or other `pip` errors fail immediately.
- **Bounded deadline & receipt**: Shared monotonic deadline across Project JSON + byte download, Simple JSON, and uncached installer dry-run, with stderr progress messages on each wait step and clear timeout diagnostics naming the missing stage/files. Writes `visibility` endpoint/timestamp metadata into `pypi-<version>-verified.json`.
