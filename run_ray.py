"""
run_ray.py — Ray cluster launcher for the CC-CMA-ES piece value experiment.

Distributes 30+30 independent training runs across a Ray cluster.
Each run is a single Ray remote task; within-run parallelism is handled
by ThreadPoolExecutor inside train_cma.run_once / train_cc.run_once.

Usage (on Ray head node):
    python3 run_ray.py                           # 30+30 with default settings
    python3 run_ray.py --mono-only               # only monolithic runs
    python3 run_ray.py --cc-only                 # only CC runs
    python3 run_ray.py --runs 5 --dry-run        # preview what would be dispatched

Required cluster setup:
    ray start --head --port=6379                 # on head node
    ray start --address=HEAD_IP:6379             # on each worker node
    python3 run_ray.py --address auto            # auto-connects to running cluster

    OR pass a specific address: --address ray://HEAD_IP:10001

Notes:
    - The dragonchess binary must be accessible at the same path on all workers.
      If you have a shared filesystem (NFS/Lustre) this is automatic.
      If not, set BINARY_PATH to a local path and ensure each worker has the binary there.
    - Results are written to --out-mono / --out-cc directories.
      These must be on a shared filesystem if running across multiple nodes.
      If there is no shared FS, set --local-only and results are returned in-memory
      and saved by this script on the head node.
    - Use --run-id-offset to fill in missing runs without overwriting completed ones.
"""

import argparse
import json
import os
import sys
import time

import numpy as np


# ---------------------------------------------------------------------------
# Argument parsing (done before importing ray so --dry-run is fast)
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Ray cluster launcher for Dragonchess CMA-ES experiment"
    )
    parser.add_argument("--runs",         type=int, default=30,
                        help="Independent runs per condition (default: 30)")
    parser.add_argument("--games",        type=int, default=200,
                        help="Games per fitness evaluation (default: 200)")
    parser.add_argument("--generations",  type=int, default=300,
                        help="Max CMA-ES generations per run (default: 300)")
    parser.add_argument("--opponent",     type=str, default="alphabeta",
                        help="Opponent AI type (default: alphabeta)")
    parser.add_argument("--opponent-depth", type=int, default=2,
                        help="Opponent search depth (default: 2)")
    parser.add_argument("--workers",      type=int, default=4,
                        help="ThreadPoolExecutor workers within each run (default: 4). "
                             "Set based on cores-per-node / concurrent-tasks-per-node.")
    parser.add_argument("--out-mono",     type=str, default="results/monolithic/",
                        help="Output directory for monolithic results")
    parser.add_argument("--out-cc",       type=str, default="results/cc/",
                        help="Output directory for CC results")
    parser.add_argument("--run-id-offset", type=int, default=0,
                        help="Add to run IDs; use to append runs without overwriting")
    parser.add_argument("--mono-only",    action="store_true",
                        help="Only run monolithic condition")
    parser.add_argument("--cc-only",      action="store_true",
                        help="Only run CC condition")
    parser.add_argument("--address",      type=str, default="auto",
                        help="Ray cluster address (default: auto)")
    parser.add_argument("--local-only",   action="store_true",
                        help="Save results on head node (no shared FS required); "
                             "results are returned from tasks and saved here")
    parser.add_argument("--dry-run",      action="store_true",
                        help="Print what would be dispatched without running")
    parser.add_argument("--cpus-per-task", type=float, default=None,
                        help="Ray CPU request per task (default: --workers value). "
                             "Overrides the automatic setting.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Ray remote task wrappers
# ---------------------------------------------------------------------------

def make_ray_tasks(args):
    """Import ray and define remote tasks. Called after dry-run check."""
    import ray

    cpus = args.cpus_per_task if args.cpus_per_task is not None else float(args.workers)

    # We import inside the function so that the module-level import of train_cma/train_cc
    # happens inside the remote task on the worker, not on the head node.
    # This avoids shipping large objects over the network.

    @ray.remote(num_cpus=cpus)
    def run_mono_task(run_id, games, generations, opponent, opponent_depth,
                      n_workers, out_dir):
        """One monolithic CMA-ES run. Returns result dict."""
        import sys, os
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import train_cma
        _, _, result = train_cma.run_once(
            run_id=run_id,
            games_per_eval=games,
            max_generations=generations,
            opponent=opponent,
            opponent_depth=opponent_depth,
            n_workers=n_workers,
            out_dir=out_dir,    # None if --local-only
            verbose=False,      # suppress per-generation output on workers
        )
        return result

    @ray.remote(num_cpus=cpus)
    def run_cc_task(run_id, games, generations, opponent, opponent_depth,
                    n_workers, out_dir):
        """One CC-CMA-ES run. Returns result dict."""
        import sys, os
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import train_cc
        _, _, result = train_cc.run_once(
            run_id=run_id,
            games_per_eval=games,
            max_generations=generations,
            opponent=opponent,
            opponent_depth=opponent_depth,
            n_workers=n_workers,
            out_dir=out_dir,
            verbose=False,
        )
        return result

    return run_mono_task, run_cc_task


# ---------------------------------------------------------------------------
# Result collection helpers
# ---------------------------------------------------------------------------

def collect_results(futures_with_meta, label, out_dir, local_only):
    """
    Wait for a list of (future, run_id) pairs; print progress; save if local_only.
    Returns list of result dicts.
    """
    import ray

    results = []
    pending = list(futures_with_meta)  # [(future, run_id), ...]
    n_total = len(pending)
    n_done = 0
    t0 = time.time()

    print(f"\n[{label}] Waiting for {n_total} tasks...")

    while pending:
        # ray.wait returns (ready, not_ready) — check one at a time for live progress
        ready_refs = [f for f, _ in pending]
        done_refs, _ = ray.wait(ready_refs, num_returns=1, timeout=30.0)

        if not done_refs:
            # Timeout — print heartbeat
            elapsed = time.time() - t0
            print(f"  [{label}] {n_done}/{n_total} done  ({elapsed:.0f}s elapsed) ...",
                  flush=True)
            continue

        # Match done ref back to (future, run_id)
        done_ref = done_refs[0]
        matched = [(f, rid) for f, rid in pending if f == done_ref]
        if not matched:
            # Shouldn't happen, but guard anyway
            pending = [(f, rid) for f, rid in pending if f not in done_refs]
            continue

        future, run_id = matched[0]
        pending = [(f, rid) for f, rid in pending if f != future]
        n_done += 1

        try:
            result = ray.get(future)
            results.append(result)
            wr = result.get("best_win_rate", float("nan"))
            gen = result.get("generations", "?")
            elapsed = time.time() - t0
            print(f"  [{label}] run_{run_id:03d} done  "
                  f"win_rate={wr:.3f}  gen={gen}  "
                  f"({n_done}/{n_total}, {elapsed:.0f}s)",
                  flush=True)

            # Save on head node if no shared FS
            if local_only and out_dir:
                os.makedirs(out_dir, exist_ok=True)
                path = os.path.join(out_dir, f"run_{run_id:03d}.json")
                with open(path, "w") as fp:
                    json.dump(result, fp, indent=2)

        except Exception as exc:
            print(f"  [{label}] run_{run_id:03d} FAILED: {exc}", flush=True)

    return results


def print_summary(results, label):
    if not results:
        print(f"\n[{label}] No successful results.")
        return
    wrs = [r["best_win_rate"] for r in results if "best_win_rate" in r]
    if not wrs:
        return
    wrs = np.array(wrs)
    print(f"\n[{label}] {len(wrs)} runs completed")
    print(f"  Win rate: mean={wrs.mean():.3f}  std={wrs.std():.3f}  "
          f"[{wrs.min():.3f}, {wrs.max():.3f}]")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Resolve output directories relative to this script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_mono = os.path.join(script_dir, args.out_mono) if not os.path.isabs(args.out_mono) \
               else args.out_mono
    out_cc   = os.path.join(script_dir, args.out_cc)   if not os.path.isabs(args.out_cc) \
               else args.out_cc

    # out_dir passed to tasks: None if local_only (task doesn't save; head node does)
    task_out_mono = None if args.local_only else out_mono
    task_out_cc   = None if args.local_only else out_cc

    run_ids = [args.run_id_offset + i for i in range(args.runs)]

    print("=" * 60)
    print(" DragonchessAI — Ray Experiment Launcher")
    print("=" * 60)
    print(f" Runs per condition : {args.runs}")
    print(f" Run IDs            : {run_ids[0]} – {run_ids[-1]}")
    print(f" Games/eval         : {args.games}")
    print(f" Max generations    : {args.generations}")
    print(f" Opponent           : {args.opponent} (depth={args.opponent_depth})")
    print(f" Workers/run        : {args.workers}")
    print(f" CPUs/task (Ray)    : {args.cpus_per_task or args.workers}")
    print(f" Monolithic out     : {out_mono}")
    print(f" CC out             : {out_cc}")
    print(f" Conditions         : "
          f"{'MONO ' if not args.cc_only else ''}"
          f"{'CC' if not args.mono_only else ''}")
    print(f" Local-only mode    : {args.local_only}")
    print("=" * 60)

    if args.dry_run:
        print("\n[DRY RUN] Would dispatch:")
        if not args.cc_only:
            for rid in run_ids:
                print(f"  mono run_id={rid}")
        if not args.mono_only:
            for rid in run_ids:
                print(f"  cc   run_id={rid}")
        total = (0 if args.cc_only else len(run_ids)) + \
                (0 if args.mono_only else len(run_ids))
        print(f"\nTotal tasks: {total}")
        return

    # --- Connect to Ray ---
    import ray
    print(f"\nConnecting to Ray cluster at '{args.address}'...")
    ray.init(address=args.address)
    print(f"Cluster resources: {ray.cluster_resources()}\n")

    run_mono_task, run_cc_task = make_ray_tasks(args)

    t_start = time.time()

    # --- Dispatch all tasks ---
    mono_futures = []
    cc_futures   = []

    if not args.cc_only:
        print(f"Dispatching {len(run_ids)} monolithic tasks...")
        for rid in run_ids:
            f = run_mono_task.remote(
                rid, args.games, args.generations,
                args.opponent, args.opponent_depth,
                args.workers, task_out_mono,
            )
            mono_futures.append((f, rid))

    if not args.mono_only:
        print(f"Dispatching {len(run_ids)} CC tasks...")
        for rid in run_ids:
            f = run_cc_task.remote(
                rid, args.games, args.generations,
                args.opponent, args.opponent_depth,
                args.workers, task_out_cc,
            )
            cc_futures.append((f, rid))

    total_tasks = len(mono_futures) + len(cc_futures)
    print(f"\nAll {total_tasks} tasks dispatched. Collecting results...\n")

    # --- Collect results (interleaved progress) ---
    mono_results = []
    cc_results   = []

    # Collect both conditions in parallel by interleaving ray.wait calls
    # Simple approach: collect them sequentially (each collect loop is non-blocking
    # on the other condition because ray.get is called only on ready futures).
    import threading

    mono_done = threading.Event()
    cc_done   = threading.Event()
    mono_results_container = []
    cc_results_container   = []

    def collect_mono():
        mono_results_container.extend(
            collect_results(mono_futures, "MONO", out_mono, args.local_only)
        )
        mono_done.set()

    def collect_cc():
        cc_results_container.extend(
            collect_results(cc_futures, "CC", out_cc, args.local_only)
        )
        cc_done.set()

    threads = []
    if mono_futures:
        t = threading.Thread(target=collect_mono, daemon=True)
        t.start()
        threads.append(t)
    else:
        mono_done.set()

    if cc_futures:
        t = threading.Thread(target=collect_cc, daemon=True)
        t.start()
        threads.append(t)
    else:
        cc_done.set()

    for t in threads:
        t.join()

    # --- Summary ---
    total_elapsed = time.time() - t_start
    print(f"\n{'=' * 60}")
    print(f" All tasks complete  ({total_elapsed:.0f}s total)")
    print(f"{'=' * 60}")
    print_summary(mono_results_container, "Monolithic")
    print_summary(cc_results_container,   "CC-CMA-ES")

    # Save combined summary JSONs
    for results, label, out_dir in [
        (mono_results_container, "monolithic", out_mono),
        (cc_results_container,   "cc",         out_cc),
    ]:
        if results and out_dir:
            wrs = [r["best_win_rate"] for r in results if "best_win_rate" in r]
            summary = {
                "runs": results,
                "win_rate_mean": float(np.mean(wrs)) if wrs else None,
                "win_rate_std":  float(np.std(wrs))  if wrs else None,
            }
            os.makedirs(out_dir, exist_ok=True)
            spath = os.path.join(out_dir, "summary.json")
            with open(spath, "w") as f:
                json.dump(summary, f, indent=2)
            print(f"\n{label} summary saved to {spath}")

    print("\nNext steps:")
    print("  python3 analyze_results.py --out figures/")
    print("  python3 evaluate_posttrain.py --workers 16")

    ray.shutdown()


if __name__ == "__main__":
    main()
