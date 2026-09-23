#!/usr/bin/env bash
# Run the hitlist test suite with a memory- and CPU-aware pytest-xdist
# worker count.
#
# Default: skip integration tests (those that exercise the built
# ``observations.parquet`` corpus). Pass ``--all`` to include them.
# ``./deploy.sh`` always runs the full set so deploys remain safe.
#
# See ~/code/trufflepig/test.sh for the worker-cap rationale: running
# several sibling repos' suites concurrently can fork-bomb the laptop,
# so we cap workers at min(cpu_reserve, available_RAM / PER_WORKER_GB).
#
# ``--all`` runs as TWO separate pytest processes, not one (#483):
# non-integration first, then integration. CI has done this since
# #272/#274 — a fresh process for the ~2-4 GB ``full_observations_df``
# fixture, rather than one that already carries whatever the preceding
# ~1500 non-integration tests left allocated (freed objects don't
# necessarily return pages to the OS; a long-lived interpreter's heap
# only gets more fragmented). ``deploy.sh``'s single combined pass hit
# this directly: two sessions independently got OOM-killed by the OS
# running ``--all`` as one process on an otherwise memory-constrained
# machine, even down to a single worker. Splitting doesn't fix that
# ceiling by itself, but it removes one avoidable multiplier on it, and
# the integration pass gets its own, more conservative worker budget
# instead of inheriting the light pass's.
#
# xdist is optional — fall back to serial pytest when it isn't
# installed.
#
# See issues #223, #244, #262, #440, #483.
#
# Preflight memory guard (#483): if available memory can't even cover
# TEST_SH_MIN workers at the pass's own per-worker budget, abort with a
# clear message instead of silently forcing TEST_SH_MIN anyway and letting
# the OS SIGKILL pytest later with no useful signal. Both prior OOM kills
# this issue was filed from would have hit this guard instead of burning
# 10+ minutes of test progress on an ambiguous "process vanished".
#
# Tunables (env vars):
#   PER_WORKER_GB               non-integration per-worker budget in GB (default: 2.5)
#   INTEGRATION_PER_WORKER_GB   integration per-worker budget in GB (default: 5)
#   TEST_SH_MIN                 floor on workers (default: 1); also the preflight guard's
#                               worker-count target -- lower it to relax the guard
#   TEST_SH_MAX                 hard ceiling on workers, both passes (default: unset)
#   TEST_SH_MEMORY_RETRY_DELAY_SECONDS  delay for --retry-memory (default: 120)
#
# --retry-memory retries a refused memory preflight once per phase, before
# pytest starts. A passed phase is never replayed and test failures are not retried.

set -eo pipefail

PER_WORKER_GB="${PER_WORKER_GB:-2.5}"
INTEGRATION_PER_WORKER_GB="${INTEGRATION_PER_WORKER_GB:-5}"
TEST_SH_MIN="${TEST_SH_MIN:-1}"
TEST_SH_MAX="${TEST_SH_MAX:-0}"
TEST_SH_MEMORY_RETRY_DELAY_SECONDS="${TEST_SH_MEMORY_RETRY_DELAY_SECONDS:-120}"

log() { printf '[test.sh] %s\n' "$*" >&2; }

case "$(uname -s)" in
    Darwin) OS=macos ;;
    Linux)  OS=linux ;;
    *)      OS=unknown ;;
esac

cpu_count() {
    local n=""
    if command -v getconf >/dev/null 2>&1; then
        n=$(getconf _NPROCESSORS_ONLN 2>/dev/null || true)
    fi
    if [[ -z "$n" && "$OS" == "macos" ]]; then
        n=$(sysctl -n hw.logicalcpu 2>/dev/null || true)
    fi
    if [[ -z "$n" && -r /proc/cpuinfo ]]; then
        n=$(grep -c '^processor' /proc/cpuinfo 2>/dev/null || true)
    fi
    if [[ -z "$n" || "$n" -lt 1 ]]; then n=1; fi
    echo "$n"
}

cpu_cap() {
    local c="$1"
    if   (( c <= 1 )); then echo 1
    elif (( c <= 3 )); then echo $(( c - 1 ))
    else                    echo $(( c - 2 ))
    fi
}

mac_available_bytes() {
    local page_size
    page_size=$(sysctl -n hw.pagesize 2>/dev/null) || return 1
    vm_stat 2>/dev/null | awk -v ps="$page_size" '
        /Pages free/        { gsub(/\./, "", $3); free     = $3 }
        /Pages speculative/ { gsub(/\./, "", $3); spec     = $3 }
        # Inactive pages may require eviction and swap I/O to reclaim.
        # Counting them as free overcommits a busy machine (#440).
        END { print (free + spec) * ps }
    '
}

linux_available_bytes() {
    [[ -r /proc/meminfo ]] || return 1
    awk '
        /^MemAvailable:/ { print $2 * 1024; found=1; exit }
        END              { if (!found) exit 1 }
    ' /proc/meminfo
}

available_bytes() {
    case "$OS" in
        macos) mac_available_bytes ;;
        linux) linux_available_bytes ;;
        *)     return 1 ;;
    esac
}

CPUS=$(cpu_count)
CPU_CAP=$(cpu_cap "$CPUS")

# Re-probed for each pass (not cached), since availability can shift
# meaningfully between the light pass finishing and the heavy one
# starting -- exactly the situation #483 was filed from.
worker_count() {
    local per_worker_gb="$1"
    local avail mem_cap avail_gb workers probed=1
    if avail=$(available_bytes 2>/dev/null) && [[ -n "$avail" ]]; then
        mem_cap=$(awk -v b="$avail" -v g="$per_worker_gb" 'BEGIN { print int(b / 1024^3 / g) }')
        avail_gb=$(awk -v b="$avail" 'BEGIN { printf "%.2f", b / 1024^3 }')
        mem_note="ram_free=${avail_gb}GB mem_cap=${mem_cap}"
    else
        probed=0
        mem_cap=1
        mem_note="ram_free=? (probe unavailable) mem_cap=1"
    fi
    if (( CPU_CAP < mem_cap )); then workers=$CPU_CAP; else workers=$mem_cap; fi
    if (( workers < TEST_SH_MIN )); then workers=$TEST_SH_MIN; fi
    if (( TEST_SH_MAX > 0 && workers > TEST_SH_MAX )); then workers=$TEST_SH_MAX; fi
    # Serial fallback still consumes one worker's budget (#526).
    if (( ! use_xdist )); then workers=1; fi
    # TEST_SH_MIN (and, on a low-CPU box, TEST_SH_MAX) can each force workers
    # above what mem_cap actually supports. Check the worker count that
    # would really run, after every floor/ceiling has applied, not just the
    # raw memory division -- otherwise a TEST_SH_MAX clamp that brings it
    # back into a safe range would abort anyway on the pre-clamp value.
    if (( probed )) && (( workers > mem_cap )); then
        local need_gb
        need_gb=$(awk -v g="$per_worker_gb" -v w="$workers" 'BEGIN { printf "%.1f", g * w }')
        echo "abort 0 only ${avail_gb}GB available, need ~${need_gb}GB for ${workers} worker(s) at ${per_worker_gb}GB each -- free memory and retry (or lower TEST_SH_MIN / raise PER_WORKER_GB to accept the risk)"
        return
    fi
    echo "ok ${workers} ${mem_note}"
}

# Argument parsing: --all expands to include integration tests.
run_all=0
retry_memory=0
extra=()
for arg in "$@"; do
    if [[ "$arg" == "--all" ]]; then
        run_all=1
    elif [[ "$arg" == "--retry-memory" ]]; then
        retry_memory=1
    else
        extra+=("$arg")
    fi
done

use_xdist=0
if python -c "import xdist" 2>/dev/null; then
    use_xdist=1
else
    log "platform=${OS} cpus=${CPUS} (pytest-xdist not installed; running serial)"
fi

run_pytest() {
    # $1 = per-worker GB, $2 = -m marker expression (empty = no filter),
    # remaining args = extra pytest cov/report flags for this invocation.
    # Plain positional args rather than an array-by-reference: this
    # machine's /bin/bash is 3.2, which predates nameref support (4.3+).
    local per_worker_gb=$1
    local marker=$2
    shift 2
    local filter_args=()
    if [[ -n "$marker" ]]; then
        filter_args=(-m "$marker")
    fi
    local xdist_flags=()
    local status workers mem_note attempted_retry=0
    while true; do
        read -r status workers mem_note < <(worker_count "$per_worker_gb")
        if [[ "$status" != "abort" ]]; then
            break
        fi
        log "${mem_note} (#483)"
        if (( ! retry_memory || attempted_retry )); then
            return 1
        fi
        log "Memory preflight for '${marker}' refused; waiting ${TEST_SH_MEMORY_RETRY_DELAY_SECONDS}s and retrying once (#526)"
        sleep "$TEST_SH_MEMORY_RETRY_DELAY_SECONDS"
        attempted_retry=1
    done
    log "cpus=${CPUS} cpu_cap=${CPU_CAP} ${mem_note} per_worker=${per_worker_gb}GB workers=${workers}"
    if (( use_xdist )); then
        xdist_flags=(-n "$workers")
    fi
    log "→ exec python -m pytest ${xdist_flags[*]:-} ${filter_args[*]:-} $* tests ${extra[*]:-}"
    python -m pytest "${xdist_flags[@]}" "${filter_args[@]}" "$@" tests "${extra[@]}"
}

if (( run_all )); then
    # Two processes, not one (#483): the integration pass starts with a
    # clean interpreter instead of inheriting whatever ~1500 preceding
    # non-integration tests left allocated, and gets its own (higher)
    # per-worker memory budget rather than the light pass's.
    run_pytest "$PER_WORKER_GB" "not integration" --cov=hitlist/
    run_pytest "$INTEGRATION_PER_WORKER_GB" "integration" --cov=hitlist/ --cov-append --cov-report=term-missing
else
    run_pytest "$PER_WORKER_GB" "not integration" --cov=hitlist/ --cov-report=term-missing
fi
