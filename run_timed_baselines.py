#!/usr/bin/env python3
"""
Time-controlled baseline matchups for DragonchessAI.

Each player gets a fixed time budget per move. Iterative deepening
searches as deep as possible within the budget.

Usage:
    python run_timed_baselines.py --out results/timed_baselines/ --time-per-move 500 --games 100
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

BINARY = Path(__file__).parent / "build" / "dragonchess"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("timed_baselines")


def extract_nn_weights(ckpt_path, out_bin_path):
    ckpt = json.loads(Path(ckpt_path).read_text())
    weights = np.array(ckpt["nn_weights"], dtype=np.float32)
    Path(out_bin_path).write_bytes(weights.tobytes())
    return len(weights)


def run_timed_matchup(gold_ai, scarlet_ai, time_per_move_ms,
                      n_games, threads, gold_nn=None, scarlet_nn=None,
                      timeout_s=7200.0):
    result_path = Path(f"/tmp/timed_{gold_ai['label']}_vs_{scarlet_ai['label']}.json")

    cmd = [
        str(BINARY), "--headless",
        "--mode", "tournament",
        "--gold-ai", gold_ai["type"],
        "--gold-depth", str(gold_ai.get("depth", 2)),
        "--scarlet-ai", scarlet_ai["type"],
        "--scarlet-depth", str(scarlet_ai.get("depth", 2)),
        "--time-per-move", str(time_per_move_ms),
        "--games", str(n_games),
        "--threads", str(threads),
        "--output-json", str(result_path),
        "--quiet",
    ]
    if gold_nn:
        cmd.extend(["--gold-nn-weights", str(gold_nn)])
    if scarlet_nn:
        cmd.extend(["--scarlet-nn-weights", str(scarlet_nn)])

    try:
        t0 = time.time()
        subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
        elapsed = time.time() - t0
        rj = json.loads(result_path.read_text())
        s = rj.get("summary", rj)
        return {
            "gold_wins": s["gold_wins"],
            "scarlet_wins": s["scarlet_wins"],
            "draws": s["draws"],
            "total": s["total_games"],
            "avg_length": s.get("avg_game_length", 0),
            "elapsed_s": elapsed,
        }
    except subprocess.TimeoutExpired:
        log.warning("TIMEOUT")
        return None
    except Exception as e:
        log.warning("ERROR: %s", e)
        return None


def wr_ci(wins, total, z=1.96):
    if total == 0:
        return 0.0, 0.0, 0.0
    p = wins / total
    denom = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denom
    margin = z * (p * (1 - p) / total + z**2 / (4 * total**2))**0.5 / denom
    return p, max(0, center - margin), min(1, center + margin)


def main():
    parser = argparse.ArgumentParser(description="Time-controlled baselines")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--time-per-move", type=float, default=500,
                        help="Milliseconds per move for all players")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--nn-sup-weights", type=Path, default=None)
    parser.add_argument("--nn-adv-weights", type=Path, default=None)
    parser.add_argument("--timeout", type=float, default=14400.0)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    tpm = args.time_per_move

    nn_sup_bin = None
    nn_adv_bin = None
    if args.nn_sup_weights and args.nn_sup_weights.exists():
        nn_sup_bin = args.out / "nn_sup_weights.bin"
        extract_nn_weights(args.nn_sup_weights, nn_sup_bin)
    if args.nn_adv_weights and args.nn_adv_weights.exists():
        nn_adv_bin = args.out / "nn_adv_weights.bin"
        extract_nn_weights(args.nn_adv_weights, nn_adv_bin)

    # All players compete as Gold against AB as Scarlet, same time budget.
    players = [
        {"type": "random", "depth": 1, "label": "Random", "nn": None},
        {"type": "greedyvalue", "depth": 1, "label": "GreedyValue", "nn": None},
        {"type": "alphabeta", "depth": 99, "label": "AB(timed)", "nn": None},
    ]
    if nn_sup_bin:
        players.append({"type": "nneval", "depth": 99, "label": "NN-Sup(timed)", "nn": str(nn_sup_bin)})
    if nn_adv_bin:
        players.append({"type": "nneval", "depth": 99, "label": "NN-Adv(timed)", "nn": str(nn_adv_bin)})

    # Opponent is always AB with same time budget
    opponent = {"type": "alphabeta", "depth": 99, "label": "AB(timed)"}

    results = []
    for i, player in enumerate(players):
        log.info("[%d/%d] %s vs %s @ %.0fms/move (%d games)...",
                 i + 1, len(players), player["label"], opponent["label"],
                 tpm, args.games)

        result = run_timed_matchup(
            player, opponent, tpm, args.games, args.threads,
            gold_nn=player.get("nn"),
            timeout_s=args.timeout)

        if result is None:
            log.info("  -> TIMEOUT/ERROR")
            results.append({"gold": player["label"], "result": "timeout"})
            continue

        wr, ci_lo, ci_hi = wr_ci(result["gold_wins"], result["total"])
        log.info("  -> WR: %.1f%% [%.1f-%.1f%%] (%d/%d/%d) avg=%.0f %.0fs",
                 wr * 100, ci_lo * 100, ci_hi * 100,
                 result["gold_wins"], result["scarlet_wins"], result["draws"],
                 result["avg_length"], result["elapsed_s"])

        results.append({
            "gold": player["label"],
            "scarlet": opponent["label"],
            "time_per_move_ms": tpm,
            "gold_wins": result["gold_wins"],
            "scarlet_wins": result["scarlet_wins"],
            "draws": result["draws"],
            "total": result["total"],
            "win_rate": wr,
            "ci_95_lo": ci_lo,
            "ci_95_hi": ci_hi,
            "avg_game_length": result["avg_length"],
            "elapsed_s": result["elapsed_s"],
        })

    # Save
    Path(args.out / "results.json").write_text(json.dumps(results, indent=2))

    # Print table
    print(f"\n{'='*60}")
    print(f"Time-controlled baselines: {tpm:.0f}ms per move, {args.games} games")
    print(f"{'='*60}")
    print(f"{'Player':<20} {'WR vs AB(timed)':>16} {'W/L/D':>12}")
    print(f"{'-'*60}")
    for r in results:
        if "win_rate" in r:
            print(f"{r['gold']:<20} {r['win_rate']*100:5.1f}% ±{(r['ci_95_hi']-r['ci_95_lo'])*50:4.1f}%"
                  f"  {r['gold_wins']:>3}/{r['scarlet_wins']:>3}/{r['draws']:>3}")
        else:
            print(f"{r['gold']:<20} {'timeout':>16}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
