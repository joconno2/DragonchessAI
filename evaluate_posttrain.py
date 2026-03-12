"""
Post-training evaluation for Dragonchess CC experiment.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
from scipy import stats

from cluster.evaluator_pool import EvaluationRequest, LocalEvaluatorPool, RayEvaluatorPool


BINARY = os.path.join(os.path.dirname(__file__), "build", "dragonchess")
PIECE_NAMES = [
    "Sylph", "Griffin", "Dragon", "Oliphant", "Unicorn",
    "Hero", "Thief", "Cleric", "Mage",
    "Paladin", "Warrior", "Basilisk", "Elemental", "Dwarf",
]
JACKMAN_WEIGHTS = [1.0, 2.0, 9.0, 5.0, 8.0, 5.0, 2.5, 4.5, 4.0, 8.0, 1.0, 3.0, 4.0, 2.0]
MATCHUPS = [
    ("vs_greedy", "greedyvalue", 0, None),
    ("vs_ab2", "alphabeta", 2, None),
    ("vs_jackman", "evolvable", 3, JACKMAN_WEIGHTS),
    ("vs_ab3", "alphabeta", 3, None),
]


def build_match_request(
    gold_weights,
    *,
    gold_depth,
    scarlet_type,
    scarlet_depth,
    scarlet_weights,
    games,
    threads_per_eval,
    timeout_s,
):
    return EvaluationRequest(
        gold_ai="evolvable",
        gold_weights=list(gold_weights),
        gold_depth=gold_depth,
        scarlet_ai=scarlet_type,
        scarlet_depth=scarlet_depth,
        scarlet_weights=None if scarlet_weights is None else list(scarlet_weights),
        games=games,
        threads=threads_per_eval,
        timeout_s=timeout_s,
        quiet=True,
    )


def evaluate_weights(
    weights,
    games,
    eval_depth=3,
    *,
    evaluator=None,
    threads_per_eval=1,
    timeout_s=1200.0,
):
    owns_pool = evaluator is None
    if owns_pool:
        evaluator = LocalEvaluatorPool(
            max_workers=1,
            binary_path=BINARY,
            threads_per_eval=threads_per_eval,
        )
    try:
        requests = [
            build_match_request(
                weights,
                gold_depth=eval_depth,
                scarlet_type=scarlet_type,
                scarlet_depth=eval_depth if scarlet_type == "evolvable" else scarlet_depth,
                scarlet_weights=scarlet_weights,
                games=games,
                threads_per_eval=threads_per_eval,
                timeout_s=timeout_s,
            )
            for _, scarlet_type, scarlet_depth, scarlet_weights in MATCHUPS
        ]
        outcomes = evaluator.evaluate_many(requests)
        return {
            matchup_key: outcome.win_rate
            for (matchup_key, _, _, _), outcome in zip(MATCHUPS, outcomes)
        }
    finally:
        if owns_pool:
            evaluator.shutdown()


def load_runs(directory):
    runs = []
    if not os.path.isdir(directory):
        return runs
    for fname in sorted(os.listdir(directory)):
        if fname.startswith("run_") and fname.endswith(".json"):
            with open(os.path.join(directory, fname)) as f:
                runs.append(json.load(f))
    return runs


def run_evaluation(
    mono_runs,
    cc_runs,
    games=50,
    eval_depth=3,
    n_workers=8,
    *,
    evaluator=None,
    threads_per_eval=1,
    timeout_s=1200.0,
):
    """Evaluate all runs from both conditions using the provided backend."""
    owns_pool = evaluator is None
    if owns_pool:
        evaluator = LocalEvaluatorPool(
            max_workers=n_workers,
            binary_path=BINARY,
            threads_per_eval=threads_per_eval,
        )

    all_weights = (
        [("mono", run["run_id"], run["best_weights"]) for run in mono_runs]
        + [("cc", run["run_id"], run["best_weights"]) for run in cc_runs]
    )
    total_matchups = len(all_weights) * len(MATCHUPS)
    print(
        f"Evaluating {len(all_weights)} agents at depth={eval_depth}, "
        f"{games} games/matchup, {total_matchups} total matchups..."
    )

    t0 = time.time()
    jobs = []
    requests = []
    for condition, run_id, weights in all_weights:
        for matchup_key, scarlet_type, scarlet_depth, scarlet_weights in MATCHUPS:
            jobs.append((condition, run_id, matchup_key))
            requests.append(
                build_match_request(
                    weights,
                    gold_depth=eval_depth,
                    scarlet_type=scarlet_type,
                    scarlet_depth=eval_depth if scarlet_type == "evolvable" else scarlet_depth,
                    scarlet_weights=scarlet_weights,
                    games=games,
                    threads_per_eval=threads_per_eval,
                    timeout_s=timeout_s,
                )
            )

    try:
        outcomes = evaluator.evaluate_many(requests)
    finally:
        if owns_pool:
            evaluator.shutdown()

    grouped: dict[tuple[str, int], dict[str, float]] = {}
    for (condition, run_id, matchup_key), outcome in zip(jobs, outcomes):
        grouped.setdefault((condition, run_id), {"run_id": run_id})[matchup_key] = outcome.win_rate

    results = {"mono": [], "cc": []}
    for (condition, _), payload in grouped.items():
        results[condition].append(payload)

    for condition in results:
        results[condition].sort(key=lambda item: item["run_id"])

    elapsed = time.time() - t0
    print(f"Completed in {elapsed:.0f}s")
    return results


def head_to_head(
    mono_runs,
    cc_runs,
    games=100,
    eval_depth=3,
    n_workers=8,
    *,
    evaluator=None,
    threads_per_eval=1,
    timeout_s=1200.0,
):
    """Direct CC vs Mono matchup using mean weights from each condition."""
    mono_mean = np.mean([r["best_weights"] for r in mono_runs], axis=0).tolist()
    cc_mean = np.mean([r["best_weights"] for r in cc_runs], axis=0).tolist()

    print(f"\nHead-to-head: CC vs Mono mean weights ({games} games)...")
    t0 = time.time()
    owns_pool = evaluator is None
    if owns_pool:
        evaluator = LocalEvaluatorPool(
            max_workers=n_workers,
            binary_path=BINARY,
            threads_per_eval=threads_per_eval,
        )
    try:
        request = build_match_request(
            cc_mean,
            gold_depth=eval_depth,
            scarlet_type="evolvable",
            scarlet_depth=eval_depth,
            scarlet_weights=mono_mean,
            games=games,
            threads_per_eval=threads_per_eval,
            timeout_s=timeout_s,
        )
        outcome = evaluator.evaluate_many([request])[0]
    finally:
        if owns_pool:
            evaluator.shutdown()
    elapsed = time.time() - t0
    if outcome.total_games > 0:
        cc_wins, total = outcome.gold_wins, outcome.total_games
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
    parser.add_argument(
        "--threads-per-eval",
        type=int,
        default=1,
        help="Threads passed to each dragonchess tournament subprocess",
    )
    parser.add_argument("--ray-address", type=str, default=None,
                        help="Optional Ray address for distributed post-train evaluation")
    parser.add_argument("--workers-csv", type=str, default="workers.csv",
                        help="workers.csv path when using --ray-address")
    parser.add_argument("--repo-dir", type=str, default="~/DragonchessAI",
                        help="Repository path on each Ray worker when using --ray-address")
    args = parser.parse_args()

    mono_runs = load_runs(args.mono)
    cc_runs   = load_runs(args.cc)

    if not mono_runs and not cc_runs:
        print("No results found.")
        return

    print(f"Loaded: {len(mono_runs)} monolithic, {len(cc_runs)} CC runs")
    evaluator = None
    try:
        if args.ray_address:
            import ray

            ray.init(address=args.ray_address)
            evaluator = RayEvaluatorPool(
                workers_csv=args.workers_csv,
                repo_dir=args.repo_dir,
                threads_per_eval=args.threads_per_eval,
            )
            evaluator.start()

        results = run_evaluation(
            mono_runs,
            cc_runs,
            games=args.games,
            eval_depth=args.depth,
            n_workers=args.workers,
            evaluator=evaluator,
            threads_per_eval=args.threads_per_eval,
        )
        print_summary(results)

        if mono_runs and cc_runs:
            head_to_head(
                mono_runs,
                cc_runs,
                games=args.games * 2,
                eval_depth=args.depth,
                n_workers=args.workers,
                evaluator=evaluator,
                threads_per_eval=args.threads_per_eval,
            )

        save_results(results, args.out, args.depth, args.games)
    finally:
        if evaluator is not None:
            evaluator.shutdown()
            try:
                import ray

                ray.shutdown()
            except Exception:
                pass


if __name__ == "__main__":
    main()
