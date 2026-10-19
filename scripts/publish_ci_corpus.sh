#!/usr/bin/env bash
#
# Publish the locally-built observations corpus as a PUBLIC GitHub release
# asset so CI can download it and run the integration test suite (the tests
# that exercise the built ``observations.parquet`` — see
# ``.github/workflows/tests.yml`` and ``tests/conftest.py::full_observations_df``).
#
# Why this exists: CI runners don't have the ~15 GB IEDB/CEDAR source
# exports (behind terms-of-use, not in the repo), so they cannot build the
# corpus themselves. The derived output is much smaller, so we ship that instead.
#
# LICENSING NOTE: the published parquets are IEDB/CEDAR-derived and become
# PUBLICLY downloadable as a release asset.  This was a deliberate choice
# (full-fidelity CI coverage over keeping derived data private).
#
# Usage:
#   ./scripts/publish_ci_corpus.sh <version>      # e.g. v1, v2, ...
#
# Run this from a machine that has the corpus built (``hitlist build`` →
# the current datacache generation under ``~/.hitlist/dataset``).  Bump <version> whenever the builder logic or
# source data changes enough to matter for the integration tests, then bump
# ``CORPUS_VERSION`` in both test and release workflows to match so CI
# invalidates its cache and pulls the new corpus.

set -euo pipefail

VERSION="${1:?usage: publish_ci_corpus.sh <version>   (e.g. v1)}"
TAG="ci-corpus-${VERSION}"
REPO="${HITLIST_CI_CORPUS_REPO:-pirl-unc/hitlist}"
DATA_DIR="${HITLIST_DATA_DIR:-$HOME/.hitlist}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Never replace bytes under a cached corpus version.
if gh release view "$TAG" --repo "$REPO" >/dev/null 2>&1; then
    echo "error: release ${TAG} already exists; choose a new corpus version." >&2
    exit 1
fi

# Resolve one generation and retain its complete verified portable copy in
# shared datacache storage. An existing flat corpus remains explicitly archival.
snapshot_command=(python "$SCRIPT_DIR/ci_corpus.py" snapshot "$DATA_DIR")
if [[ -n "${HITLIST_CI_CORPUS_CACHE_DIR:-}" ]]; then
    snapshot_command+=(--cache-dir "$HITLIST_CI_CORPUS_CACHE_DIR")
fi
SNAPSHOT_DIR=$("${snapshot_command[@]}")
python "$SCRIPT_DIR/ci_corpus.py" verify "$SNAPSHOT_DIR"
FILES=$(python - "$SNAPSHOT_DIR" <<'PYFILES'
import json
import pathlib
import sys
manifest = json.loads((pathlib.Path(sys.argv[1]) / "observations_meta.json").read_text())
print("\n".join([*sorted(manifest["files"]), "observations_meta.json"]))
PYFILES
)
paths=()
while IFS= read -r filename; do
    paths+=("${SNAPSHOT_DIR}/${filename}")
done <<< "$FILES"

total=$(du -ch "${paths[@]}" | tail -1 | cut -f1)
echo "Publishing corpus (${total}) to ${REPO} release ${TAG}:"
printf '  %s\n' "${paths[@]}"

echo "Creating release ${TAG}..."
gh release create "$TAG" "${paths[@]}" \
    --repo "$REPO" \
    --latest=false \
    --title "CI corpus ${VERSION}" \
    --notes "Prebuilt hitlist observations corpus for CI integration tests. Includes original build contract and SHA-256 provenance for the complete versioned inventory, including contributor and mapping metadata when captured. IEDB/CEDAR-derived; see scripts/publish_ci_corpus.sh."

echo "Done. Ensure CORPUS_VERSION=${VERSION} in both test and release workflows."
