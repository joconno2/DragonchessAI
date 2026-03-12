"""
run_ray.py — Head-side Ray experiment driver with shared evaluator actors.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from cluster.evaluator_pool import RayEvaluatorPool


def parse_args():
    parser = argparse.ArgumentParser(
        description="Ray cluster launcher for Dragonchess CMA-ES experiment"
    )
    parser.add_argument("--runs", type=int, default=30, help="Independent runs per condition")
    parser.add_argument("--games", type=int, default=200, help="Games per fitness evaluation")
    parser.add_argument("--generations", type=int, default=300, help="Max generations per run")
    parser.add_argument("--opponent", type=str, default="alphabeta", help="Opponent AI type")
    parser.add_argument("--opponent-depth", type=int, default=2, help="Opponent search depth")
    parser.add_argument(
        "--parallel-runs",
        type=int,
        default=4,
        help="How many run coordinators to execute concurrently on the head node",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Deprecated alias for --parallel-runs",
    )
    parser.add_argument(
        "--threads-per-eval",
        type=int,
        default=1,
        help="Threads passed to each dragonchess tournament subprocess on evaluator nodes",
    )
    parser.add_argument("--workers-csv", type=str, default="workers.csv")
    parser.add_argument(
        "--repo-dir",
        type=str,
        default="~/DragonchessAI",
        help="Repository path on each Ray worker node",
    )
    parser.add_argument("--out-mono", type=str, default="results/monolithic/")
    parser.add_argument("--out-cc", type=str, default="results/cc/")
    parser.add_argument("--run-id-offset", type=int, default=0)
    parser.add_argument("--mono-only", action="store_true")
    parser.add_argument("--cc-only", action="store_true")
    parser.add_argument("--address", type=str, default="auto", help="Ray cluster address")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="Deprecated; results are always written by the head-side driver.",
    )
    return parser.parse_args()


def print_summary(results, label):
    if not results:
        print(f"\n[{label}] No successful results.")
        return
    win_rates = [result["best_win_rate"] for result in results if "best_win_rate" in result]
    if not win_rates:
        return
    win_rates = np.array(win_rates)
    print(f"\n[{label}] {len(win_rates)} runs completed")
    print(
        f"  Win rate: mean={win_rates.mean():.3f}  std={win_rates.std():.3f}  "
        f"[{win_rates.min():.3f}, {win_rates.max():.3f}]"
    )


def save_summary(results, out_dir):
    if not results:
        return
    win_rates = [result["best_win_rate"] for result in results if "best_win_rate" in result]
    summary = {
        "runs": results,
        "win_rate_mean": float(np.mean(win_rates)) if win_rates else None,
        "win_rate_std": float(np.std(win_rates)) if win_rates else None,
    }
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "summary.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"{os.path.basename(out_dir.rstrip('/'))} summary saved to {path}")


def main():
    args = parse_args()
    if args.workers is not None:
        args.parallel_runs = args.workers

    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_mono = os.path.join(script_dir, args.out_mono) if not os.path.isabs(args.out_mono) else args.out_mono
    out_cc = os.path.join(script_dir, args.out_cc) if not os.path.isabs(args.out_cc) else args.out_cc
    workers_csv = (
        os.path.join(script_dir, args.workers_csv)
        if not os.path.isabs(args.workers_csv)
        else args.workers_csv
    )
    run_ids = [args.run_id_offset + index for index in range(args.runs)]

    print("=" * 60)
    print(" DragonchessAI — Ray Experiment Launcher")
    print("=" * 60)
    print(f" Runs per condition : {args.runs}")
    print(f" Run IDs            : {run_ids[0]} – {run_ids[-1]}")
    print(f" Games/eval         : {args.games}")
    print(f" Max generations    : {args.generations}")
    print(f" Opponent           : {args.opponent} (depth={args.opponent_depth})")
    print(f" Parallel runs      : {args.parallel_runs}")
    print(f" Threads/eval       : {args.threads_per_eval}")
    print(f" Workers CSV        : {workers_csv}")
    print(f" Worker repo dir    : {args.repo_dir}")
    print(f" Monolithic out     : {out_mono}")
    print(f" CC out             : {out_cc}")
    print(
        f" Conditions         : "
        f"{'MONO ' if not args.cc_only else ''}"
        f"{'CC' if not args.mono_only else ''}"
    )
    print("=" * 60)

    tasks = []
    if not args.cc_only:
        tasks.extend(("mono", run_id) for run_id in run_ids)
    if not args.mono_only:
        tasks.extend(("cc", run_id) for run_id in run_ids)

    if args.dry_run:
        print("\n[DRY RUN] Would dispatch:")
        for condition, run_id in tasks:
            print(f"  {condition:<4} run_id={run_id}")
        print(f"\nTotal coordinators: {len(tasks)}")
        return

    import ray

    print(f"\nConnecting to Ray cluster at '{args.address}'...")
    ray.init(address=args.address)
    print(f"Cluster resources: {ray.cluster_resources()}\n")

    evaluator_pool = RayEvaluatorPool(
        workers_csv=workers_csv,
        repo_dir=args.repo_dir,
        threads_per_eval=args.threads_per_eval,
    )
    evaluator_pool.start()

    capacity = evaluator_pool.describe_capacity()
    total_slots = sum(capacity.values())
    print(f"Evaluator slots: {total_slots}")
    for host, slot_count in sorted(capacity.items()):
        print(f"  - {host}: {slot_count}")

    started = time.time()
    mono_results = []
    cc_results = []

    def run_one(condition: str, run_id: int):
        if condition == "mono":
            import train_cma

            _, _, result = train_cma.run_once(
                run_id=run_id,
                games_per_eval=args.games,
                max_generations=args.generations,
                opponent=args.opponent,
                opponent_depth=args.opponent_depth,
                out_dir=out_mono,
                verbose=False,
                evaluator=evaluator_pool,
                threads_per_eval=args.threads_per_eval,
            )
            return condition, run_id, result

        import train_cc

        _, _, result = train_cc.run_once(
            run_id=run_id,
            games_per_eval=args.games,
            max_generations=args.generations,
            opponent=args.opponent,
            opponent_depth=args.opponent_depth,
            out_dir=out_cc,
            verbose=False,
            evaluator=evaluator_pool,
            threads_per_eval=args.threads_per_eval,
        )
        return condition, run_id, result

    print(f"Launching {len(tasks)} run coordinators...\n")
    try:
        with ThreadPoolExecutor(max_workers=max(1, args.parallel_runs)) as pool:
            futures = [pool.submit(run_one, condition, run_id) for condition, run_id in tasks]
            for future in as_completed(futures):
                try:
                    condition, run_id, result = future.result()
                except Exception as exc:
                    print(f"[failed] {exc}", flush=True)
                    continue

                label = "MONO" if condition == "mono" else "CC"
                elapsed = time.time() - started
                print(
                    f"[{label}] run_{run_id:03d} done  "
                    f"win_rate={result['best_win_rate']:.3f}  "
                    f"gen={result['generations']}  "
                    f"({elapsed:.0f}s)",
                    flush=True,
                )
                if condition == "mono":
                    mono_results.append(result)
                else:
                    cc_results.append(result)
    finally:
        evaluator_pool.shutdown()
        ray.shutdown()

    total_elapsed = time.time() - started
    print(f"\n{'=' * 60}")
    print(f" All tasks complete  ({total_elapsed:.0f}s total)")
    print(f"{'=' * 60}")

    mono_results.sort(key=lambda result: result["run_id"])
    cc_results.sort(key=lambda result: result["run_id"])

    print_summary(mono_results, "Monolithic")
    print_summary(cc_results, "CC-CMA-ES")
    save_summary(mono_results, out_mono)
    save_summary(cc_results, out_cc)

    print("\nNext steps:")
    print("  python3 analyze_results.py --out figures/")
    print("  python3 evaluate_posttrain.py --workers 16")


if __name__ == "__main__":
    main()
