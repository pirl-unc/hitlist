#!/usr/bin/env bash
# Default: skip integration tests (those that exercise the built
# ``observations.parquet`` corpus). Pass ``--all`` to include them.
# ``./deploy.sh`` always runs the full set so deploys remain safe.
#
# See issue #223.

set -e

filter=(-m "not integration")
extra=()
for arg in "$@"; do
    if [[ "$arg" == "--all" ]]; then
        filter=()
    else
        extra+=("$arg")
    fi
done

exec pytest -n auto "${filter[@]}" --cov=hitlist/ --cov-report=term-missing tests "${extra[@]}"
