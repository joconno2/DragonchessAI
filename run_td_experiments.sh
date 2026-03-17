#!/usr/bin/env bash
# ============================================================================
# run_td_experiments.sh — Local TD(λ) experiment sweep
#
# Runs several independent TD(λ) configurations in parallel, each writing
# to its own results/td/<name>/ directory.  Kill it whenever; every trainer
# checkpoints after every batch and resumes automatically.
#
# Usage:
#   ./run_td_experiments.sh                  # all configs, auto workers
#   ./run_td_experiments.sh --workers 4      # cap workers per trainer
#   ./run_td_experiments.sh --batches 100    # stop each trainer after N batches
#   ./run_td_experiments.sh --hours 1.0      # stop each trainer after N hours
#   ./run_td_experiments.sh --dry-run        # print commands only
#   ./run_td_experiments.sh --resume         # skip configs that already have checkpoints
#
# Config sweep:
#   baseline    λ=0.7  lr=0.05  depth=1  cold-start
#   warm        λ=0.7  lr=0.05  depth=1  warm-start (Jackman material init)
#   deep        λ=0.7  lr=0.05  depth=2  cold-start  (slower, stronger search)
#   lambda_low  λ=0.3  lr=0.05  depth=1  cold-start  (shorter trace)
#   lambda_high λ=0.9  lr=0.05  depth=1  cold-start  (longer trace)
#   lr_high     λ=0.7  lr=0.15  depth=1  cold-start  (faster adaptation)
#
# Parallelism:
#   Each trainer uses --workers W threads for the C++ subprocess.
#   All trainers run simultaneously in the background.
#   Total cores ≈ N_CONFIGS * W — set --workers so this fits your machine.
#   Default: workers = max(1, nproc / 6)  (6 configs, fill all cores)
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

BINARY="$SCRIPT_DIR/build/dragonchess"
NCPU=$(nproc)
WORKERS=$(( NCPU / 6 > 1 ? NCPU / 6 : 1 ))
MAX_BATCHES=""
MAX_HOURS=""
DRY_RUN=false
RESUME=false
GAMES_PER_BATCH=50
EVAL_EVERY=20

while [[ $# -gt 0 ]]; do
    case "$1" in
        --workers)       WORKERS="$2";       shift ;;
        --batches)       MAX_BATCHES="$2";   shift ;;
        --hours)         MAX_HOURS="$2";     shift ;;
        --games)         GAMES_PER_BATCH="$2"; shift ;;
        --eval-every)    EVAL_EVERY="$2";    shift ;;
        --dry-run)       DRY_RUN=true ;;
        --resume)        RESUME=true ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
    shift
done

echo "============================================"
echo " DragonchessAI — TD(λ) Local Experiments"
echo "============================================"
echo " CPUs available:   $NCPU"
echo " Workers/trainer:  $WORKERS"
echo " Games/batch:      $GAMES_PER_BATCH"
echo " Eval every:       $EVAL_EVERY batches"
[[ -n "$MAX_BATCHES" ]] && echo " Max batches:      $MAX_BATCHES"
[[ -n "$MAX_HOURS"   ]] && echo " Max hours:        $MAX_HOURS"
echo "============================================"
echo

if [[ ! -f "$BINARY" ]]; then
    echo "ERROR: binary not found at $BINARY"
    echo "Build with: cmake --build build --parallel"
    exit 1
fi

mkdir -p results/td logs/td

# ---- Helper: build and launch one trainer ----
launch() {
    local NAME="$1";   shift
    local EXTRA="$*"   # remaining args forwarded to train_td.py

    local OUT="results/td/$NAME"
    local LOG="logs/td/${NAME}.log"

    # Skip if resuming and checkpoint already exists
    if $RESUME && [[ -f "$OUT/latest.json" ]]; then
        echo "  [skip]  $NAME  (checkpoint exists, --resume set)"
        return
    fi

    local CMD="python3 -u train_td.py \
        --out $OUT \
        --workers $WORKERS \
        --games-per-batch $GAMES_PER_BATCH \
        --eval-every $EVAL_EVERY \
        --eval-games 200 \
        --eval-ab-depth 2"

    [[ -n "$MAX_BATCHES" ]] && CMD="$CMD --max-batches $MAX_BATCHES"
    [[ -n "$MAX_HOURS"   ]] && CMD="$CMD --max-hours $MAX_HOURS"

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
    # shellcheck disable=SC2086
    $CMD > "$LOG" 2>&1 &
    echo $! > "$OUT/.pid"
}

echo "=== Launching configs ==="
echo

# --- Config sweep ---

# Baseline: standard hyperparameters, random init
launch baseline \
    --lambda 0.7 --lr 0.05 --td-depth 1

# Warm: same as baseline but init material weights from Jackman values
launch warm \
    --lambda 0.7 --lr 0.05 --td-depth 1 --warm-start

# Deep: depth-2 search during self-play (stronger play, slower games)
launch deep \
    --lambda 0.7 --lr 0.05 --td-depth 2

# Lambda sweep: shorter trace (faster credit assignment but less look-back)
launch lambda_low \
    --lambda 0.3 --lr 0.05 --td-depth 1

# Lambda sweep: longer trace (more credit assigned further back)
launch lambda_high \
    --lambda 0.9 --lr 0.05 --td-depth 1

# Learning rate: more aggressive adaptation
launch lr_high \
    --lambda 0.7 --lr 0.15 --td-depth 1

echo
if $DRY_RUN; then
    echo "[dry run] No processes launched."
    exit 0
fi

# ---- Monitor loop ----
echo "All trainers launched.  Monitoring every 60s..."
echo "(Ctrl-C kills this monitor; trainers keep running in the background)"
echo
echo "Live logs:   tail -f logs/td/<name>.log"
echo "Results:     ls results/td/"
echo "Status:      python3 train_td_status.py  (if available)"
echo

sleep 5

while true; do
    echo "--- $(date '+%H:%M:%S') ---"
    for NAME in baseline warm deep lambda_low lambda_high lr_high; do
        OUT="results/td/$NAME"
        if [[ ! -d "$OUT" ]]; then
            echo "  $NAME: not started"
            continue
        fi

        LATEST="$OUT/latest.json"
        if [[ ! -f "$LATEST" ]]; then
            echo "  $NAME: no checkpoint yet"
            continue
        fi

        BATCHES=$(python3 -c "import json; d=json.load(open('$LATEST')); print(d.get('total_batches',0))" 2>/dev/null || echo "?")
        GAMES=$(python3 -c "import json; d=json.load(open('$LATEST')); print(d.get('total_games',0))" 2>/dev/null || echo "?")

        # Latest eval win rate if available
        WR=$(python3 -c "
import json
d = json.load(open('$LATEST'))
h = d.get('win_rate_history', [])
if h:
    print(f\"{h[-1]['win_rate_vs_ab']:.3f}\")
else:
    print('n/a')
" 2>/dev/null || echo "?")

        # Check if still running
        PID_FILE="$OUT/.pid"
        RUNNING=""
        if [[ -f "$PID_FILE" ]]; then
            PID=$(cat "$PID_FILE")
            kill -0 "$PID" 2>/dev/null && RUNNING=" [running]" || RUNNING=" [stopped]"
        fi

        echo "  $NAME: batch=$BATCHES  games=$GAMES  wr_vs_ab2=$WR$RUNNING"
    done
    echo

    # Exit if all trainers have stopped
    ALL_DONE=true
    for NAME in baseline warm deep lambda_low lambda_high lr_high; do
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
echo " Experiment complete."
echo "============================================"
echo "Results in: results/td/"
echo "Checkpoints: results/td/<name>/latest.json"
for NAME in baseline warm deep lambda_low lambda_high lr_high; do
    LATEST="results/td/$NAME/latest.json"
    if [[ -f "$LATEST" ]]; then
        python3 -c "
import json
d = json.load(open('$LATEST'))
h = d.get('win_rate_history', [])
best = max((e['win_rate_vs_ab'] for e in h), default=float('nan'))
batches = d.get('total_batches', 0)
games = d.get('total_games', 0)
print(f'  $NAME: {batches} batches  {games} games  best_wr={best:.3f}')
" 2>/dev/null || echo "  $NAME: (unreadable)"
    fi
done
