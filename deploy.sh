#!/usr/bin/env bash
#
# Tunables (env vars):
#   DEPLOY_TEST_RETRY_DELAY_SECONDS   delay before the one retry below (default: 120)

set -e

DEPLOY_TEST_RETRY_DELAY_SECONDS="${DEPLOY_TEST_RETRY_DELAY_SECONDS:-120}"

VERSION=$(python -c "from hitlist.version import __version__; print(__version__)")
echo "Deploying hitlist v${VERSION}"
echo ""

echo "==> Running lint checks..."
./lint.sh

echo ""
echo "==> Running tests (--all, including integration corpus tests)..."
# Retry once after a delay before giving up (#483): the memory pressure
# behind a failed/aborted run is usually ordinary desktop app usage on the
# shared machine, not anything test.sh itself did, and it often clears on
# its own within a couple of minutes. One retry, not a loop -- a second
# failure propagates for real rather than masking a genuine break.
if ! ./test.sh --all; then
    echo "" >&2
    echo "==> test.sh --all failed; waiting ${DEPLOY_TEST_RETRY_DELAY_SECONDS}s and retrying once (#483)..." >&2
    sleep "$DEPLOY_TEST_RETRY_DELAY_SECONDS"
    ./test.sh --all
fi

echo ""
echo "==> Cleaning old builds..."
rm -f dist/*

echo ""
echo "==> Building distribution..."
python -m build

echo ""
echo "==> Built artifacts:"
ls -lh dist/

echo ""
echo "==> Uploading to PyPI..."
twine upload dist/*

echo ""
echo "Deploy complete! https://pypi.org/project/hitlist/${VERSION}/"
