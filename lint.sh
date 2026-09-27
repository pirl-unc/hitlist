#!/usr/bin/env bash

set -e

# No paths: ruff checks exactly pyproject.toml's [tool.ruff] include list, the
# one list format.sh and CI (which runs this script) share.

echo "Running ruff check..."
ruff check

echo "Running ruff format check..."
ruff format --check

echo "All checks passed!"
