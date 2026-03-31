#!/usr/bin/env bash
# ============================================================================
# run_nightly.sh — Nightly training runner (designed for systemd timer)
#
# Runs TD(λ) training from 3AM until DEADLINE_HOUR (default 11AM), then
# sends a summary to Discord via the research assistant bot.
#
# Modes:
#   sweep1      — run_td_experiments.sh  (depth-1 vs depth-2, Sweep 1)
#   sweep2      — run_td_depth2_sweep.sh (depth-2 hyperparam sweep, Sweep 2)
#   best        — single train_td.py run with best-config args
#
# Configuration: edit MODE and BEST_CONFIG_ARGS below, or pass as args:
#   ./run_nightly.sh sweep2
#   ./run_nightly.sh best
#   ./run_nightly.sh best --lambda 0.9 --lr 0.05 --td-depth 2
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

ASSISTANT_VENV="$HOME/research/assistant/.venv/bin/python3"

# ---- Configuration ----

# Default mode if not passed as argument
MODE="${1:-best}"
shift 2>/dev/null || true

# Deadline: stop by this hour (24h format). 11 = 11:00 AM.
DEADLINE_HOUR=11

# Best-config defaults (override by passing extra args on the command line)
BEST_CONFIG_ARGS=(
    --lambda 0.7
    --lr 0.05
    --td-depth 2
    --out results/td/best_nightly_v2
    --games-per-batch 50
    --eval-every 20
    --eval-games 200
    --eval-ab-depth 2
    --mixed-frac 0.3
    --snapshot-every 50
    --snapshot-keep 10
    --early-stop-patience 500
    --early-stop-threshold 0.15
)

# ---- Compute time budget ----

CURRENT_HOUR=$(date +%-H)
CURRENT_MIN=$(date +%-M)

if (( CURRENT_HOUR < DEADLINE_HOUR )); then
    REMAINING_MIN=$(( (DEADLINE_HOUR - CURRENT_HOUR) * 60 - CURRENT_MIN ))
else
    echo "[nightly] Current time $(date +%H:%M) is past ${DEADLINE_HOUR}:00 deadline. Exiting."
    exit 0
fi

# Leave 5 minutes of margin for cleanup/checkpoint saving
REMAINING_MIN=$(( REMAINING_MIN - 5 ))
if (( REMAINING_MIN < 10 )); then
    echo "[nightly] Less than 10 minutes remaining. Exiting."
    exit 0
fi

# Convert to fractional hours for --hours flag
HOURS=$(python3 -c "print(f'{${REMAINING_MIN}/60:.2f}')")

# ---- Resolve results directories for notification ----

results_dirs_for_mode() {
    case "$1" in
        sweep1)  echo results/td/baseline results/td/warm results/td/deep results/td/lambda_low results/td/lambda_high results/td/lr_high ;;
        sweep2)  echo results/td/d2_baseline results/td/d2_lambda_low results/td/d2_lambda_high results/td/d2_lr_high results/td/d2_warm ;;
        best)    echo results/td/best_nightly_v2 ;;
    esac
}

# ---- Setup logging ----

LOGFILE="$SCRIPT_DIR/logs/nightly/$(date +%Y%m%d_%H%M).log"
mkdir -p "$(dirname "$LOGFILE")"

log() { echo "$@" | tee -a "$LOGFILE"; }

log "============================================"
log " DragonchessAI — Nightly Training Run"
log "============================================"
log " Mode:      $MODE"
log " Started:   $(date)"
log " Deadline:  ${DEADLINE_HOUR}:00"
log " Budget:    ${HOURS}h (${REMAINING_MIN}min)"
log " Log:       $LOGFILE"
log "============================================"
log ""

# ---- Run training ----

TRAIN_EXIT=0

case "$MODE" in
    sweep1)
        log "[nightly] Running Sweep 1 (depth-1 vs depth-2)..."
        "$SCRIPT_DIR/run_td_experiments.sh" --hours "$HOURS" "$@" >> "$LOGFILE" 2>&1 || TRAIN_EXIT=$?
        ;;
    sweep2)
        log "[nightly] Running Sweep 2 (depth-2 hyperparam sweep)..."
        "$SCRIPT_DIR/run_td_depth2_sweep.sh" --hours "$HOURS" "$@" >> "$LOGFILE" 2>&1 || TRAIN_EXIT=$?
        ;;
    best)
        ARGS=("${BEST_CONFIG_ARGS[@]}" --max-hours "$HOURS" "$@")
        log "[nightly] Running best config: python3 train_td.py ${ARGS[*]}"
        python3 -u "$SCRIPT_DIR/train_td.py" "${ARGS[@]}" >> "$LOGFILE" 2>&1 || TRAIN_EXIT=$?
        ;;
    *)
        log "Unknown mode: $MODE"
        log "Usage: $0 {sweep1|sweep2|best} [extra args...]"
        exit 1
        ;;
esac

log ""
log "[nightly] Training finished (exit=$TRAIN_EXIT) at $(date)"

# ---- Send Discord notification ----

RESULTS_DIRS=$(results_dirs_for_mode "$MODE")
LABEL="Nightly ${MODE} run ($(date +%Y-%m-%d))"

log "[nightly] Sending Discord summary..."
if "$ASSISTANT_VENV" "$SCRIPT_DIR/notify_training.py" $RESULTS_DIRS --label "$LABEL" >> "$LOGFILE" 2>&1; then
    log "[nightly] Discord notification sent."
else
    log "[nightly] WARNING: Discord notification failed (exit=$?)."
fi

log "[nightly] Done."
