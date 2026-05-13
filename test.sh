#!/usr/bin/env bash
# Default: skip integration tests (those that exercise the built
# ``observations.parquet`` corpus). Pass ``--all`` to include them.
# ``./deploy.sh`` always runs the full set so deploys remain safe.
#
# Worker count is memory-aware, not CPU-bound: pytest-xdist's
# ``-n auto`` spawns one worker per core, but the integration tier's
# ``full_observations_df`` fixture costs ~2 GB per worker (#244, #262).
# On a 32 GB Mac with 10 cores, ``-n auto`` blows memory and thrashes.
# We pick the smaller of (cores, available_RAM / per_worker_gb).
#
# See issues #223, #244, #262.

set -e

# Per-worker memory budget. A soft heuristic, not a hard cap — tune as
# the fixture footprint changes (#262 will shrink it materially).
readonly PER_WORKER_GB=2.5

# macOS available-RAM heuristic: free + inactive + speculative pages
# are reclaimable on demand, so they count as headroom.
macos_available_bytes() {
    local page_size pages
    page_size=$(sysctl -n hw.pagesize)
    pages=$(vm_stat | awk '
        /Pages free/        { gsub(/\./, "", $3); free     = $3 }
        /Pages inactive/    { gsub(/\./, "", $3); inactive = $3 }
        /Pages speculative/ { gsub(/\./, "", $3); spec     = $3 }
        END                 { print free + inactive + spec }
    ')
    echo $(( pages * page_size ))
}

# Pick a pytest -n value that respects both CPU and available RAM.
pytest_workers() {
    local cpus avail_bytes
    cpus=$(getconf _NPROCESSORS_ONLN 2>/dev/null || sysctl -n hw.logicalcpu)
    if [[ "$(uname)" == "Darwin" ]]; then
        avail_bytes=$(macos_available_bytes)
    else
        avail_bytes=$(awk '/MemAvailable/ { print $2 * 1024 }' /proc/meminfo)
    fi
    awk -v cpus="$cpus" -v bytes="$avail_bytes" -v budget="$PER_WORKER_GB" '
        BEGIN {
            by_memory = int(bytes / 1024^3 / budget)
            n = (cpus < by_memory ? cpus : by_memory)
            print (n < 1 ? 1 : n)
        }
    '
}

filter=(-m "not integration")
extra=()
for arg in "$@"; do
    if [[ "$arg" == "--all" ]]; then
        filter=()
    else
        extra+=("$arg")
    fi
done

workers=$(pytest_workers)
echo "Running pytest with -n ${workers} (per-worker budget ≈ ${PER_WORKER_GB} GB)"
exec pytest -n "$workers" "${filter[@]}" --cov=hitlist/ --cov-report=term-missing tests "${extra[@]}"
