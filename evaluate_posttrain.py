"""
Post-training evaluation for Dragonchess CC experiment.

Takes the best evolved weights from each run and re-evaluates them using a
stronger depth-3 alpha-beta agent against a hierarchy of opponents.  This
separates the *piece value learning* story (training, depth=1) from the
*gameplay performance* story (this script, depth=3).

Matchups run per condition:
  1. Evolved(depth=3)  vs  GreedyValue           -- sanity / easy baseline
  2. Evolved(depth=3)  vs  Jackman(depth=3)       -- apples-to-apples vs expert values
  3. Evolved(depth=3)  vs  AlphaBeta(depth=2)     -- standard strong opponent
  4. [head-to-head]    CC(depth=3) vs Mono(depth=3) -- direct comparison

Usage:
    python3 evaluate_posttrain.py
    python3 evaluate_posttrain.py --mono results/monolithic/ --cc results/cc/
    python3 evaluate_posttrain.py --games 100 --workers 16 --out results/posttrain/
"""

import subprocess
import json
import os
import argparse
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from scipy import stats

BINARY = os.path.join(os.path.dirname(__file__), "build", "dragonchess")

PIECE_NAMES = [
    "Sylph", "Griffin", "Dragon", "Oliphant", "Unicorn",
    "Hero", "Thief", "Cleric", "Mage",
    "Paladin", "Warrior", "Basilisk", "Elemental", "Dwarf"
]

# Jackman (1997) expert-tuned values — used as the depth-3 reference opponent
JACKMAN_WEIGHTS = [1.0, 2.0, 9.0, 5.0, 8.0, 5.0, 2.5, 4.5, 4.0, 8.0, 1.0, 3.0, 4.0, 2.0]


def weights_to_str(weights):
    return ",".join(f"{w:.6f}" for w in weights)


def play_match(gold_weights, gold_depth, scarlet_type, scarlet_depth,
               scarlet_weights, games):
    """Run one match; returns (gold_wins, total_games) or None on failure."""
    cmd = [
        BINARY, "--headless", "--mode", "tournament",
        "--games", str(games),
        "--gold-ai", "evolvable",
        "--gold-weights", weights_to_str(gold_weights),
        "--gold-depth", str(gold_depth),
        "--scarlet-ai", scarlet_type,
        "--scarlet-depth", str(scarlet_depth),
        "--quiet",
        "--output-json", "-",
    ]
    if scarlet_type == "evolvable":
        cmd += ["--scarlet-weights", weights_to_str(scarlet_weights)]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=1200)
        if r.returncode != 0:
            return None
        data = json.loads(r.stdout)
        s = data["summary"]
        return s["gold_wins"], s["total_games"]
    except Exception:
        return None


def evaluate_weights(weights, games, eval_depth=3):
    """Evaluate a weight vector; returns dict of win rates.

    Matchups (in increasing difficulty):
      vs_greedy    - sanity/easy baseline (fast)
      vs_ab2       - same opponent as training (generalization check)
      vs_jackman   - expert piece values at eval_depth (key paper comparison)
      vs_ab3       - one depth above training opponent (ceiling test)
    """
    results = {}

    # 1. vs GreedyValue — sanity check, fast
    r = play_match(weights, eval_depth, "greedyvalue", 0, None, games)
    results["vs_greedy"] = r[0] / r[1] if r else None

    # 2. vs AlphaBeta(depth=2) — same opponent used during training; generalization check
    r = play_match(weights, eval_depth, "alphabeta", 2, None, games)
    results["vs_ab2"] = r[0] / r[1] if r else None

    # 3. vs Jackman(depth=eval_depth) — equal-depth, expert piece values (key paper comparison)
    r = play_match(weights, eval_depth, "evolvable", eval_depth,
                   JACKMAN_WEIGHTS, games)
    results["vs_jackman"] = r[0] / r[1] if r else None

    # 4. vs AlphaBeta(depth=3) — one depth above training opponent; ceiling/transfer test
    r = play_match(weights, eval_depth, "alphabeta", 3, None, games)
    results["vs_ab3"] = r[0] / r[1] if r else None

    return results


def load_runs(directory):
    runs = []
    if not os.path.isdir(directory):
        return runs
    for fname in sorted(os.listdir(directory)):
        if fname.startswith("run_") and fname.endswith(".json"):
            with open(os.path.join(directory, fname)) as f:
                runs.append(json.load(f))
    return runs


def run_evaluation(mono_runs, cc_runs, games=50, eval_depth=3,
                   n_workers=8, out_dir=None):
    """Evaluate all runs from both conditions in parallel."""
    all_weights = (
        [("mono", r["run_id"], r["best_weights"]) for r in mono_runs] +
        [("cc",   r["run_id"], r["best_weights"]) for r in cc_runs]
    )

    print(f"Evaluating {len(all_weights)} agents at depth={eval_depth}, "
          f"{games} games/matchup, {n_workers} workers...")

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(evaluate_weights, w, games, eval_depth): (cond, rid)
            for cond, rid, w in all_weights
        }

    results = {"mono": [], "cc": []}
    for future, (cond, rid) in futures.items():
        r = future.result()
        r["run_id"] = rid
        results[cond].append(r)

    elapsed = time.time() - t0
    print(f"Completed in {elapsed:.0f}s")
    return results


def head_to_head(mono_runs, cc_runs, games=100, eval_depth=3, n_workers=8):
    """Direct CC vs Mono matchup using mean weights from each condition."""
    mono_mean = np.mean([r["best_weights"] for r in mono_runs], axis=0).tolist()
    cc_mean   = np.mean([r["best_weights"] for r in cc_runs],   axis=0).tolist()

    print(f"\nHead-to-head: CC vs Mono mean weights ({games} games)...")
    t0 = time.time()
    r = play_match(cc_mean, eval_depth, "evolvable", eval_depth, mono_mean, games)
    elapsed = time.time() - t0
    if r:
        cc_wins, total = r
        print(f"  CC(depth={eval_depth}) vs Mono(depth={eval_depth}): "
              f"{cc_wins}/{total} ({cc_wins/total:.3f}) for CC  [{elapsed:.0f}s]")
        return cc_wins / total, total
    return None, None


def print_summary(results):
    matchups = [
        ("vs_greedy",  "vs GreedyValue"),
        ("vs_ab2",     "vs AlphaBeta-D2 (train)"),
        ("vs_jackman", "vs Jackman (same depth)"),
        ("vs_ab3",     "vs AlphaBeta-D3 (ceiling)"),
    ]

    print(f"\n{'='*70}")
    print(f"Post-Training Evaluation (depth=3 agents)")
    print(f"{'='*70}")
    print(f"{'Matchup':<26} {'Monolithic':>18} {'CC-CMA-ES':>18}  {'p-val':>7}  {'d':>5}")
    print(f"{'-'*70}")

    for key, label in matchups:
        mono_wr = np.array([r[key] for r in results["mono"] if r[key] is not None])
        cc_wr   = np.array([r[key] for r in results["cc"]   if r[key] is not None])

        if len(mono_wr) == 0 or len(cc_wr) == 0:
            print(f"{label:<26}  {'—':>18}  {'—':>18}")
            continue

        m_str = f"{mono_wr.mean():.3f} ± {mono_wr.std():.3f} (n={len(mono_wr)})"
        c_str = f"{cc_wr.mean():.3f} ± {cc_wr.std():.3f} (n={len(cc_wr)})"

        _, p = stats.mannwhitneyu(mono_wr, cc_wr, alternative="two-sided")
        pooled = np.sqrt((np.std(mono_wr, ddof=1)**2 + np.std(cc_wr, ddof=1)**2) / 2)
        d = (np.mean(mono_wr) - np.mean(cc_wr)) / pooled if pooled > 0 else 0.0

        print(f"{label:<26}  {m_str:>18}  {c_str:>18}  {p:>7.3f}  {d:>5.2f}")

    print(f"{'='*70}")


def save_results(results, out_dir, eval_depth, games):
    os.makedirs(out_dir, exist_ok=True)
    summary = {
        "eval_depth": eval_depth,
        "games_per_matchup": games,
        "mono": results["mono"],
        "cc":   results["cc"],
    }
    path = os.path.join(out_dir, "posttrain_eval.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Post-training depth=3 evaluation for CC experiment")
    parser.add_argument("--mono",    default="results/monolithic/")
    parser.add_argument("--cc",      default="results/cc/")
    parser.add_argument("--out",     default="results/posttrain/")
    parser.add_argument("--games",   type=int, default=50,
                        help="Games per matchup (50 → ±7%% CI, 100 → ±5%% CI)")
    parser.add_argument("--depth",   type=int, default=3,
                        help="Eval depth for evolved agents (default: 3)")
    parser.add_argument("--workers", type=int,
                        default=max(1, (os.cpu_count() or 4) // 2))
    args = parser.parse_args()

    mono_runs = load_runs(args.mono)
    cc_runs   = load_runs(args.cc)

    if not mono_runs and not cc_runs:
        print("No results found.")
        return

    print(f"Loaded: {len(mono_runs)} monolithic, {len(cc_runs)} CC runs")

    results = run_evaluation(mono_runs, cc_runs,
                             games=args.games, eval_depth=args.depth,
                             n_workers=args.workers)
    print_summary(results)

    if mono_runs and cc_runs:
        head_to_head(mono_runs, cc_runs,
                     games=args.games * 2, eval_depth=args.depth,
                     n_workers=args.workers)

    save_results(results, args.out, args.depth, args.games)


if __name__ == "__main__":
    main()
