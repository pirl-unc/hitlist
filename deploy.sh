#!/usr/bin/env bash

set -e

VERSION=$(python -c "from hitlist.version import __version__; print(__version__)")
echo "Deploying hitlist v${VERSION}"
echo ""

echo "==> Running lint checks..."
./lint.sh

echo ""
echo "==> Running tests (--all, including integration corpus tests)..."
./test.sh --all

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
