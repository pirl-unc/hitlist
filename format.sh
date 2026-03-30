#!/usr/bin/env bash

set -e

SOURCES="hitlist tests"

echo "Running ruff format..."
ruff format $SOURCES

echo "Formatting complete!"
