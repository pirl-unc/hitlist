#!/usr/bin/env bash
#
# Tunables (env vars):
#   DEPLOY_TEST_RETRY_DELAY_SECONDS   delay before one memory preflight retry per phase (default: 120)

set -e

DEPLOY_TEST_RETRY_DELAY_SECONDS="${DEPLOY_TEST_RETRY_DELAY_SECONDS:-120}"

VERSION=$(python -c "from hitlist.version import __version__; print(__version__)")
echo "Deploying hitlist v${VERSION}"
echo ""

echo "==> Running lint checks..."
./lint.sh

echo ""
echo "==> Running tests (--all, including integration corpus tests)..."
# Retry only a phase's memory preflight, before its tests start (#526).
# Replaying --all repeats passed tests and does not help an integration
# preflight refusal. Actual test failures must stop the release immediately.
TEST_SH_MEMORY_RETRY_DELAY_SECONDS="$DEPLOY_TEST_RETRY_DELAY_SECONDS" ./test.sh --all --retry-memory

echo ""
echo "==> Cleaning old builds..."
rm -f dist/*

echo ""
echo "==> Building distribution..."
python -m build

echo ""
echo "==> Verifying distribution license metadata..."
python scripts/check_distribution_license.py dist

echo ""
echo "==> Built artifacts:"
ls -lh dist/

echo ""
echo "==> Uploading to PyPI..."
twine upload dist/*

echo ""
echo "Deploy complete! https://pypi.org/project/hitlist/${VERSION}/"
