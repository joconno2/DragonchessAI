#!/usr/bin/env bash
# ============================================================================
# run_td_depth2_sweep.sh — Depth-2 hyperparameter sweep
#
# Sweep 1 established that depth-2 selfplay dominates depth-1.
# This script sweeps λ and lr at depth-2 to fine-tune hyperparameters.
#
# Usage:
#   ./run_td_depth2_sweep.sh                  # run until killed
#   ./run_td_depth2_sweep.sh --hours 2.0      # stop after N hours
#   ./run_td_depth2_sweep.sh --workers 4      # cap workers per trainer
#   ./run_td_depth2_sweep.sh --dry-run        # print commands only
#
# Configs (all depth-2):
#   d2_baseline   λ=0.7  lr=0.05  cold-start  (reference)
#   d2_lambda_low λ=0.3  lr=0.05  cold-start
#   d2_lambda_high λ=0.9 lr=0.05  cold-start
#   d2_lr_high    λ=0.7  lr=0.15  cold-start
#   d2_warm       λ=0.7  lr=0.05  warm-start
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

BINARY="$SCRIPT_DIR/build/dragonchess"
NCPU=$(nproc)
WORKERS=$(( NCPU / 5 > 1 ? NCPU / 5 : 1 ))
MAX_HOURS=""
DRY_RUN=false
GAMES_PER_BATCH=50
EVAL_EVERY=20

while [[ $# -gt 0 ]]; do
    case "$1" in
        --workers)    WORKERS="$2";       shift ;;
        --hours)      MAX_HOURS="$2";     shift ;;
        --games)      GAMES_PER_BATCH="$2"; shift ;;
        --eval-every) EVAL_EVERY="$2";    shift ;;
        --dry-run)    DRY_RUN=true ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
    shift
done

echo "============================================"
echo " DragonchessAI — TD(λ) Depth-2 Sweep"
echo "============================================"
echo " CPUs available:   $NCPU"
echo " Workers/trainer:  $WORKERS"
echo " Games/batch:      $GAMES_PER_BATCH"
echo " Eval every:       $EVAL_EVERY batches"
[[ -n "$MAX_HOURS" ]] && echo " Max hours:        $MAX_HOURS"
echo "============================================"
echo

if [[ ! -f "$BINARY" ]]; then
    echo "ERROR: binary not found at $BINARY"
    echo "Build with: cmake --build build --parallel"
    exit 1
fi

mkdir -p results/td logs/td

CONFIGS=(d2_baseline d2_lambda_low d2_lambda_high d2_lr_high d2_warm)

launch() {
    local NAME="$1"; shift
    local EXTRA="$*"
    local OUT="results/td/$NAME"
    local LOG="logs/td/${NAME}.log"

    local CMD="python3 -u train_td.py \
        --out $OUT \
        --workers $WORKERS \
        --games-per-batch $GAMES_PER_BATCH \
        --eval-every $EVAL_EVERY \
        --eval-games 200 \
        --eval-ab-depth 2"

    [[ -n "$MAX_HOURS" ]] && CMD="$CMD --max-hours $MAX_HOURS"
    CMD="$CMD $EXTRA"

    if $DRY_RUN; then
        echo "  [dry]   $NAME"
        echo "          $CMD"
        echo "          → $LOG"
        echo
        return
    fi

    mkdir -p "$OUT"
    echo "  [start] $NAME  →  $LOG"
    $CMD > "$LOG" 2>&1 &
    echo $! > "$OUT/.pid"
}

echo "=== Launching configs ==="
echo

launch d2_baseline   --lambda 0.7 --lr 0.05 --td-depth 2
launch d2_lambda_low --lambda 0.3 --lr 0.05 --td-depth 2
launch d2_lambda_high --lambda 0.9 --lr 0.05 --td-depth 2
launch d2_lr_high    --lambda 0.7 --lr 0.15 --td-depth 2
launch d2_warm       --lambda 0.7 --lr 0.05 --td-depth 2 --warm-start

echo
if $DRY_RUN; then
    echo "[dry run] No processes launched."
    exit 0
fi

echo "All trainers launched.  Monitoring every 60s..."
echo "(Ctrl-C kills this monitor; trainers keep running in the background)"
echo
echo "Live logs:   tail -f logs/td/<name>.log"
echo

sleep 5

while true; do
    echo "--- $(date '+%H:%M:%S') ---"
    for NAME in "${CONFIGS[@]}"; do
        OUT="results/td/$NAME"
        LATEST="$OUT/latest.json"

        if [[ ! -f "$LATEST" ]]; then
            echo "  $NAME: no checkpoint yet"
            continue
        fi

        BATCHES=$(python3 -c "import json; d=json.load(open('$LATEST')); print(d.get('total_batches',0))" 2>/dev/null || echo "?")
        GAMES=$(python3 -c "import json; d=json.load(open('$LATEST')); print(d.get('total_games',0))" 2>/dev/null || echo "?")
        WR=$(python3 -c "
import json
d = json.load(open('$LATEST'))
h = d.get('win_rate_history', [])
print(f\"{h[-1]['win_rate_vs_ab']:.3f}\" if h else 'n/a')
" 2>/dev/null || echo "?")

        RUNNING=""
        PID_FILE="$OUT/.pid"
        if [[ -f "$PID_FILE" ]]; then
            PID=$(cat "$PID_FILE")
            kill -0 "$PID" 2>/dev/null && RUNNING=" [running]" || RUNNING=" [stopped]"
        fi

        echo "  $NAME: batch=$BATCHES  games=$GAMES  wr_vs_ab2=$WR$RUNNING"
    done
    echo

    ALL_DONE=true
    for NAME in "${CONFIGS[@]}"; do
        PID_FILE="results/td/$NAME/.pid"
        if [[ -f "$PID_FILE" ]]; then
            PID=$(cat "$PID_FILE")
            kill -0 "$PID" 2>/dev/null && ALL_DONE=false
        fi
    done
    $ALL_DONE && { echo "All trainers finished."; break; }

    sleep 60
done

echo
echo "============================================"
echo " Sweep 2 complete."
echo "============================================"
for NAME in "${CONFIGS[@]}"; do
    LATEST="results/td/$NAME/latest.json"
    [[ -f "$LATEST" ]] && python3 -c "
import json
d = json.load(open('$LATEST'))
h = d.get('win_rate_history', [])
best = max((e['win_rate_vs_ab'] for e in h), default=float('nan'))
print(f'  $NAME: {d.get(\"total_batches\",0)} batches  {d.get(\"total_games\",0)} games  best_wr={best:.3f}')
" 2>/dev/null || echo "  $NAME: (unreadable)"
done
