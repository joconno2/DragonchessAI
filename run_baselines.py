#!/usr/bin/env python3
"""
Run exhaustive baseline matchups for DragonchessAI paper.

Produces a full results table: each player vs AB(d=1), AB(d=2), AB(d=3).
200 games per matchup for statistical significance (~±7% CI at 95%).

Usage:
    python run_baselines.py --out results/baselines/ --threads 4
    python run_baselines.py --out results/baselines/ --threads 4 --nn-sup-weights results/nn_v2_epochs_d4/best.json --nn-adv-weights results/nn_v2_adversarial/best.json
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
log = logging.getLogger("baselines")


def extract_nn_weights(ckpt_path, out_bin_path):
    """Extract NN weights from a JSON checkpoint to binary file."""
    ckpt = json.loads(Path(ckpt_path).read_text())
    weights = np.array(ckpt["nn_weights"], dtype=np.float32)
    Path(out_bin_path).write_bytes(weights.tobytes())
    return len(weights)


def run_matchup(gold_ai, gold_depth, scarlet_ai, scarlet_depth,
                n_games, threads, gold_nn_weights=None, scarlet_nn_weights=None,
                timeout_s=3600.0):
    """Run a tournament matchup and return results dict."""
    result_path = Path(f"/tmp/baseline_{gold_ai}d{gold_depth}_vs_{scarlet_ai}d{scarlet_depth}.json")

    cmd = [
        str(BINARY), "--headless",
        "--mode", "tournament",
        "--gold-ai", gold_ai,
        "--gold-depth", str(gold_depth),
        "--scarlet-ai", scarlet_ai,
        "--scarlet-depth", str(scarlet_depth),
        "--games", str(n_games),
        "--threads", str(threads),
        "--output-json", str(result_path),
        "--quiet",
    ]
    if gold_nn_weights:
        cmd.extend(["--gold-nn-weights", str(gold_nn_weights)])
    if scarlet_nn_weights:
        cmd.extend(["--scarlet-nn-weights", str(scarlet_nn_weights)])

    try:
        t0 = time.time()
        subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
        elapsed = time.time() - t0
        rj = json.loads(result_path.read_text())
        summary = rj.get("summary", rj)
        return {
            "gold_wins": summary["gold_wins"],
            "scarlet_wins": summary["scarlet_wins"],
            "draws": summary["draws"],
            "total": summary["total_games"],
            "avg_length": summary.get("avg_game_length", 0),
            "elapsed_s": elapsed,
        }
    except subprocess.TimeoutExpired:
        log.warning("TIMEOUT: %s(d=%d) vs %s(d=%d) after %.0fs",
                     gold_ai, gold_depth, scarlet_ai, scarlet_depth, timeout_s)
        return None
    except Exception as e:
        log.warning("ERROR: %s", e)
        return None


def wr_ci(wins, total, z=1.96):
    """Wilson score interval for win rate."""
    if total == 0:
        return 0.0, 0.0, 0.0
    p = wins / total
    denom = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denom
    margin = z * (p * (1 - p) / total + z**2 / (4 * total**2))**0.5 / denom
    return p, max(0, center - margin), min(1, center + margin)


def main():
    parser = argparse.ArgumentParser(description="DragonchessAI baseline matchups")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--nn-sup-weights", type=Path, default=None,
                        help="Supervised NN checkpoint (JSON)")
    parser.add_argument("--nn-adv-weights", type=Path, default=None,
                        help="Adversarial NN checkpoint (JSON)")
    parser.add_argument("--timeout", type=float, default=3600.0,
                        help="Timeout per matchup in seconds")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    # Extract NN weights if provided
    nn_sup_bin = None
    nn_adv_bin = None
    if args.nn_sup_weights and args.nn_sup_weights.exists():
        nn_sup_bin = args.out / "nn_sup_weights.bin"
        n = extract_nn_weights(args.nn_sup_weights, nn_sup_bin)
        log.info("Supervised NN: %d params -> %s", n, nn_sup_bin)
    if args.nn_adv_weights and args.nn_adv_weights.exists():
        nn_adv_bin = args.out / "nn_adv_weights.bin"
        n = extract_nn_weights(args.nn_adv_weights, nn_adv_bin)
        log.info("Adversarial NN: %d params -> %s", n, nn_adv_bin)

    # Define all players (gold_ai, gold_depth, gold_nn_weights, label)
    players = [
        ("random", 1, None, "Random"),
        ("greedyvalue", 1, None, "GreedyValue"),
        ("alphabeta", 1, None, "AB(d=1)"),
        ("alphabeta", 2, None, "AB(d=2)"),
        ("alphabeta", 3, None, "AB(d=3)"),
    ]
    if nn_sup_bin:
        players.append(("nneval", 1, str(nn_sup_bin), "NN-Sup(d=1)"))
        players.append(("nneval", 2, str(nn_sup_bin), "NN-Sup(d=2)"))
    if nn_adv_bin:
        players.append(("nneval", 1, str(nn_adv_bin), "NN-Adv(d=1)"))
        players.append(("nneval", 2, str(nn_adv_bin), "NN-Adv(d=2)"))

    # Opponents to test against
    opponents = [
        ("alphabeta", 1, "AB(d=1)"),
        ("alphabeta", 2, "AB(d=2)"),
        ("alphabeta", 3, "AB(d=3)"),
    ]

    results = []
    total_matchups = len(players) * len(opponents)
    completed = 0

    for gold_ai, gold_depth, gold_nn, gold_label in players:
        for opp_ai, opp_depth, opp_label in opponents:
            # Skip trivially slow matchups
            if gold_ai == "random" and opp_depth >= 3:
                completed += 1
                continue

            completed += 1
            log.info("[%d/%d] %s vs %s (%d games)...",
                     completed, total_matchups, gold_label, opp_label, args.games)

            result = run_matchup(
                gold_ai, gold_depth, opp_ai, opp_depth,
                args.games, args.threads,
                gold_nn_weights=gold_nn,
                timeout_s=args.timeout,
            )

            if result is None:
                log.info("  -> TIMEOUT/ERROR")
                results.append({
                    "gold": gold_label, "scarlet": opp_label,
                    "result": "timeout",
                })
                continue

            wr, ci_lo, ci_hi = wr_ci(result["gold_wins"], result["total"])
            log.info("  -> WR: %.1f%% [%.1f-%.1f%%] (%d/%d/%d) avg_len=%.0f %.1fs",
                     wr * 100, ci_lo * 100, ci_hi * 100,
                     result["gold_wins"], result["scarlet_wins"], result["draws"],
                     result["avg_length"], result["elapsed_s"])

            results.append({
                "gold": gold_label,
                "scarlet": opp_label,
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

    # Save results
    results_path = args.out / "results.json"
    results_path.write_text(json.dumps(results, indent=2))
    log.info("Results saved to %s", results_path)

    # Print summary table
    print("\n" + "=" * 80)
    print(f"{'Player':<16} {'vs AB(d=1)':>14} {'vs AB(d=2)':>14} {'vs AB(d=3)':>14}")
    print("-" * 80)
    for gold_ai, gold_depth, gold_nn, gold_label in players:
        row = f"{gold_label:<16}"
        for opp_ai, opp_depth, opp_label in opponents:
            match = [r for r in results
                     if r["gold"] == gold_label and r["scarlet"] == opp_label]
            if match and "win_rate" in match[0]:
                r = match[0]
                row += f" {r['win_rate']*100:5.1f}% ±{(r['ci_95_hi']-r['ci_95_lo'])*50:4.1f}%"
            else:
                row += f" {'---':>13}"
        print(row)
    print("=" * 80)


if __name__ == "__main__":
    main()
