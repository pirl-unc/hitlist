#!/usr/bin/env bash

set -e

# No paths: ruff formats exactly pyproject.toml's [tool.ruff] include list, the
# one list lint.sh and CI share.

echo "Running ruff format..."
ruff format

echo "Formatting complete!"
