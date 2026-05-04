#!/usr/bin/env bash
# Default: skip integration tests (those that exercise the built
# ``observations.parquet`` corpus). Pass ``--all`` to include them.
# ``./deploy.sh`` always runs the full set so deploys remain safe.
#
# See issue #223.

set -e

MARKER_FILTER='-m "not integration"'
for arg in "$@"; do
    case "$arg" in
        --all)
            MARKER_FILTER=''
            shift
            ;;
        *)
            ;;
    esac
done

eval pytest -n auto $MARKER_FILTER --cov=hitlist/ --cov-report=term-missing tests "$@"
