#!/usr/bin/env bash
# ============================================================================
# run_experiment.sh — Launch full 30+30 CC experiment on multi-core machine
#
# Usage:
#   ./run_experiment.sh                  # 30 mono + 30 CC, default settings
#   ./run_experiment.sh --dry-run        # print commands only
#   ./run_experiment.sh --mono-only      # only monolithic runs
#   ./run_experiment.sh --cc-only        # only CC runs
#   ./run_experiment.sh --workers 8      # parallel workers per run (default: 4)
#   ./run_experiment.sh --parallel 16    # parallel runs per condition (default: 16)
#
# Layout:
#   results/monolithic/run_000.json ... run_029.json
#   results/cc/run_000.json         ... run_029.json
#
# Parallelism:
#   Each run uses --workers W concurrent eval threads.
#   PARALLEL runs are launched simultaneously, each in a tmux pane / background job.
#   Total cores = PARALLEL * W. Set so that PARALLEL * W <= nproc.
#
#   Example for 64-core Threadripper:
#     --parallel 16 --workers 4   → 16 * 4 = 64 cores
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

BINARY="$SCRIPT_DIR/build/dragonchess"
TOTAL_RUNS=30
PARALLEL=16
WORKERS=4
DRY_RUN=false
RUN_MONO=true
RUN_CC=true

# ---- Parse args ----
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)     DRY_RUN=true ;;
        --mono-only)   RUN_CC=false ;;
        --cc-only)     RUN_MONO=false ;;
        --workers)     WORKERS="$2"; shift ;;
        --parallel)    PARALLEL="$2"; shift ;;
        --runs)        TOTAL_RUNS="$2"; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
    shift
done

echo "============================================"
echo " DragonchessAI Full Experiment"
echo "============================================"
echo " Total runs per condition: $TOTAL_RUNS"
echo " Parallel runs:            $PARALLEL"
echo " Workers per run:          $WORKERS"
echo " Total cores used:         $((PARALLEL * WORKERS))"
echo " CPU count available:      $(nproc)"
echo " Mono:                     $RUN_MONO"
echo " CC:                       $RUN_CC"
echo "============================================"
echo

if [[ ! -f "$BINARY" ]]; then
    echo "ERROR: binary not found at $BINARY"
    echo "Build with: mkdir -p build && cd build && cmake -DCMAKE_BUILD_TYPE=Release .. && make -j\$(nproc)"
    exit 1
fi

# ---- Helper: run a batch of N sequential runs starting at offset O ----
run_batch() {
    local SCRIPT="$1"
    local N="$2"
    local OFFSET="$3"
    local OUT="$4"
    local LOG="$5"

    local CMD="python3 -u $SCRIPT \
        --runs $N \
        --run-id-offset $OFFSET \
        --games 200 \
        --generations 300 \
        --opponent alphabeta \
        --opponent-depth 2 \
        --workers $WORKERS \
        --out $OUT"

    if $DRY_RUN; then
        echo "[DRY RUN] $CMD > $LOG 2>&1 &"
        return
    fi

    mkdir -p "$OUT" "$(dirname "$LOG")"
    echo "Launching: runs $OFFSET–$((OFFSET+N-1)) → $LOG"
    $CMD > "$LOG" 2>&1 &
}

# ---- Split TOTAL_RUNS across PARALLEL batches ----
split_runs() {
    # Outputs: "start_id count" pairs, one per batch
    local total=$1
    local parallel=$2
    local batch_size=$(( (total + parallel - 1) / parallel ))
    local start=0
    while (( start < total )); do
        local count=$(( batch_size < total - start ? batch_size : total - start ))
        echo "$start $count"
        start=$(( start + count ))
    done
}

PIDS=()

# ---- Monolithic runs ----
if $RUN_MONO; then
    echo "=== Launching Monolithic Runs ==="
    mkdir -p results/monolithic logs/mono
    while IFS=' ' read -r START COUNT; do
        run_batch train_cma.py "$COUNT" "$START" \
            results/monolithic/ \
            "logs/mono/batch_${START}.log"
        if ! $DRY_RUN; then PIDS+=($!); fi
    done < <(split_runs $TOTAL_RUNS $PARALLEL)
    echo
fi

# ---- CC runs ----
if $RUN_CC; then
    echo "=== Launching CC Runs ==="
    mkdir -p results/cc logs/cc
    while IFS=' ' read -r START COUNT; do
        run_batch train_cc.py "$COUNT" "$START" \
            results/cc/ \
            "logs/cc/batch_${START}.log"
        if ! $DRY_RUN; then PIDS+=($!); fi
    done < <(split_runs $TOTAL_RUNS $PARALLEL)
    echo
fi

if $DRY_RUN; then
    echo "[DRY RUN] Would launch $(split_runs $TOTAL_RUNS $PARALLEL | wc -l) batches per condition."
    exit 0
fi

echo "All batches launched (${#PIDS[@]} processes). Waiting for completion..."
echo "Monitor with: watch -n30 'ls results/monolithic/ results/cc/ | grep json | wc -l'"
echo "Or run:       python3 dashboard.py"
echo

# Wait for all background jobs
for pid in "${PIDS[@]}"; do
    wait "$pid" || echo "WARNING: process $pid exited with error"
done

echo
echo "============================================"
echo " All runs complete!"
echo " Monolithic: $(ls results/monolithic/run_*.json 2>/dev/null | wc -l) runs"
echo " CC:         $(ls results/cc/run_*.json 2>/dev/null | wc -l) runs"
echo "============================================"
echo "Next steps:"
echo "  python3 analyze_results.py --out figures/"
echo "  python3 evaluate_posttrain.py"
