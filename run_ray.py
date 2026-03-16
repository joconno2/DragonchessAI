"""
run_ray.py — Head-side Ray experiment driver with shared evaluator actors.
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

import numpy as np

from cluster.evaluator_pool import RayEvaluatorPool
from cluster.runtime_sync import stage_runtime_working_dir

try:
    from rich.console import Console, Group
    from rich.live import Live
    from rich.panel import Panel
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        SpinnerColumn,
        TaskProgressColumn,
        TextColumn,
        TimeElapsedColumn,
    )
    from rich.table import Table

    RICH_AVAILABLE = True
except ImportError:  # pragma: no cover - plain fallback when rich is unavailable
    Console = None
    Group = None
    Live = None
    Panel = None
    Progress = None
    SpinnerColumn = None
    TextColumn = None
    BarColumn = None
    TaskProgressColumn = None
    TimeElapsedColumn = None
    MofNCompleteColumn = None
    Table = None
    RICH_AVAILABLE = False


def condition_label(condition: str) -> str:
    return "MONO" if condition == "mono" else "CC"


def make_console():
    return Console() if RICH_AVAILABLE else None


def build_live_renderable(
    progress,
    *,
    total_tasks: int,
    parallel_runs: int,
    inflight: dict,
    run_status: dict,
    queued_count: int,
    completed_count: int,
    failed_count: int,
    mono_finished: int,
    cc_finished: int,
    total_evaluated_games: int,
    started_at: float,
    mono_results,
    cc_results,
):
    elapsed = time.time() - started_at
    summary = Table(title="Ray Training Status")
    summary.add_column("Metric")
    summary.add_column("Value", justify="right")
    summary.add_row("Elapsed", f"{elapsed:.0f}s")
    summary.add_row("Running", f"{len(inflight)}/{parallel_runs}")
    summary.add_row("Queued", str(queued_count))
    summary.add_row("Completed", f"{completed_count}/{total_tasks}")
    summary.add_row("Failed", str(failed_count))
    summary.add_row("Finished MONO", str(mono_finished))
    summary.add_row("Finished CC", str(cc_finished))
    summary.add_row("Eval Games", f"{total_evaluated_games:,}")
    summary.add_row(
        "Cluster Games/s",
        f"{(total_evaluated_games / elapsed):.1f}" if elapsed > 0 else "0.0",
    )
    if mono_results:
        summary.add_row(
            "Best MONO",
            f"{max(result['best_win_rate'] for result in mono_results):.3f}",
        )
    if cc_results:
        summary.add_row(
            "Best CC",
            f"{max(result['best_win_rate'] for result in cc_results):.3f}",
        )

    running = Table(title="Active Runs")
    running.add_column("Condition")
    running.add_column("Run")
    running.add_column("Gen", justify="right")
    running.add_column("Best", justify="right")
    running.add_column("Games/s", justify="right")
    running.add_column("Elapsed", justify="right")
    if inflight:
        for condition, run_id, task_started in sorted(
            inflight.values(), key=lambda item: (item[0], item[1])
        ):
            status = run_status.get((condition, run_id), {})
            run_elapsed = max(0.0, time.time() - task_started)
            evaluated_games = int(status.get("evaluated_games", 0))
            games_per_second = (evaluated_games / run_elapsed) if run_elapsed > 0 else 0.0
            best_win_rate = status.get("best_win_rate")
            running.add_row(
                condition_label(condition),
                f"run_{run_id:03d}",
                str(status.get("generation", 0)),
                f"{best_win_rate:.3f}" if best_win_rate is not None else "-",
                f"{games_per_second:.1f}",
                f"{run_elapsed:.0f}s",
            )
    else:
        running.add_row("-", "-", "-", "-", "-", "-")

    return Group(
        Panel(progress, title="Progress", border_style="cyan"),
        summary,
        running,
    )


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
        help="Legacy worker repo path when using --no-runtime-sync",
    )
    parser.add_argument(
        "--binary-path",
        type=str,
        default="build/dragonchess",
        help="Local dragonchess binary path to include in the Ray runtime package",
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
    parser.add_argument(
        "--no-runtime-sync",
        action="store_true",
        help="Do not stage the repo via Ray runtime_env; expect --repo-dir to exist on workers",
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
    console = make_console()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_mono = os.path.join(script_dir, args.out_mono) if not os.path.isabs(args.out_mono) else args.out_mono
    out_cc = os.path.join(script_dir, args.out_cc) if not os.path.isabs(args.out_cc) else args.out_cc
    workers_csv = (
        os.path.join(script_dir, args.workers_csv)
        if not os.path.isabs(args.workers_csv)
        else args.workers_csv
    )
    binary_path = (
        os.path.join(script_dir, args.binary_path)
        if not os.path.isabs(args.binary_path)
        else args.binary_path
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
    print(
        f" Runtime sync       : "
        f"{'disabled' if args.no_runtime_sync else 'enabled'}"
    )
    if args.no_runtime_sync:
        print(f" Worker repo dir    : {args.repo_dir}")
    else:
        print(f" Local binary path  : {binary_path}")
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

    runtime_package = None
    repo_dir_for_workers = args.repo_dir
    runtime_env = None
    if not args.no_runtime_sync:
        runtime_package = stage_runtime_working_dir(
            script_dir,
            binary_path=binary_path,
        )
        repo_dir_for_workers = None
        runtime_env = {"working_dir": str(runtime_package.path)}

    print(f"\nConnecting to Ray cluster at '{args.address}'...")
    ray.init(address=args.address, runtime_env=runtime_env)
    print(f"Cluster resources: {ray.cluster_resources()}\n")

    evaluator_pool = RayEvaluatorPool(
        workers_csv=workers_csv,
        repo_dir=repo_dir_for_workers,
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
    status_lock = threading.Lock()

    def make_progress_callback(condition: str, run_id: int):
        def _callback(*, generation: int, evaluated_games: int, latest_win_rate: float, best_win_rate: float):
            with status_lock:
                status = run_status.get((condition, run_id))
                if status is None:
                    return
                status["generation"] = generation
                status["evaluated_games"] = evaluated_games
                status["latest_win_rate"] = latest_win_rate
                status["best_win_rate"] = best_win_rate

        return _callback

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
                progress_callback=make_progress_callback(condition, run_id),
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
            progress_callback=make_progress_callback(condition, run_id),
        )
        return condition, run_id, result

    print(f"Launching {len(tasks)} run coordinators...\n")
    total_tasks = len(tasks)
    mono_total = sum(1 for condition, _ in tasks if condition == "mono")
    cc_total = sum(1 for condition, _ in tasks if condition == "cc")
    completed_count = 0
    failed_count = 0
    finished_by_condition = {"mono": 0, "cc": 0}
    inflight = {}
    run_status = {}
    finished_evaluated_games = 0
    next_task_index = 0

    def submit_next(pool):
        nonlocal next_task_index
        while len(inflight) < max(1, args.parallel_runs) and next_task_index < total_tasks:
            condition, run_id = tasks[next_task_index]
            next_task_index += 1
            future = pool.submit(run_one, condition, run_id)
            inflight[future] = (condition, run_id, time.time())
            with status_lock:
                run_status[(condition, run_id)] = {
                    "generation": 0,
                    "evaluated_games": 0,
                    "latest_win_rate": None,
                    "best_win_rate": None,
                }

    progress = None
    overall_task_id = None
    mono_task_id = None
    cc_task_id = None
    if RICH_AVAILABLE:
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
            transient=False,
        )
        overall_task_id = progress.add_task("Overall", total=total_tasks)
        mono_task_id = progress.add_task("Monolithic", total=mono_total)
        cc_task_id = progress.add_task("CC-CMA-ES", total=cc_total)

    def refresh_live(live=None):
        with status_lock:
            status_snapshot = {
                key: value.copy() for key, value in run_status.items()
            }
            aggregate_evaluated_games = finished_evaluated_games + sum(
                int(value.get("evaluated_games", 0)) for value in status_snapshot.values()
            )
        if not RICH_AVAILABLE:
            return
        progress.update(overall_task_id, completed=completed_count)
        progress.update(mono_task_id, completed=finished_by_condition["mono"])
        progress.update(cc_task_id, completed=finished_by_condition["cc"])
        renderable = build_live_renderable(
            progress,
            total_tasks=total_tasks,
            parallel_runs=max(1, args.parallel_runs),
            inflight=inflight,
            run_status=status_snapshot,
            queued_count=max(0, total_tasks - completed_count - len(inflight)),
            completed_count=completed_count,
            failed_count=failed_count,
            mono_finished=finished_by_condition["mono"],
            cc_finished=finished_by_condition["cc"],
            total_evaluated_games=aggregate_evaluated_games,
            started_at=started,
            mono_results=mono_results,
            cc_results=cc_results,
        )
        if live is not None:
            live.update(renderable)

    try:
        with ThreadPoolExecutor(max_workers=max(1, args.parallel_runs)) as pool:
            submit_next(pool)
            if RICH_AVAILABLE:
                with progress, Live(
                    build_live_renderable(
                        progress,
                        total_tasks=total_tasks,
                        parallel_runs=max(1, args.parallel_runs),
                        inflight=inflight,
                        run_status={key: value.copy() for key, value in run_status.items()},
                        queued_count=total_tasks - len(inflight),
                        completed_count=completed_count,
                        failed_count=failed_count,
                        mono_finished=finished_by_condition["mono"],
                        cc_finished=finished_by_condition["cc"],
                        total_evaluated_games=0,
                        started_at=started,
                        mono_results=mono_results,
                        cc_results=cc_results,
                    ),
                    console=console,
                    refresh_per_second=4,
                ) as live:
                    while inflight:
                        done, _ = wait(
                            set(inflight.keys()),
                            timeout=1.0,
                            return_when=FIRST_COMPLETED,
                        )
                        if not done:
                            refresh_live(live)
                            continue

                        for future in done:
                            submitted_condition, submitted_run_id, _ = inflight.pop(future)
                            with status_lock:
                                status = run_status.pop((submitted_condition, submitted_run_id), None)
                            finished_evaluated_games += (
                                int(status.get("evaluated_games", 0)) if status is not None else 0
                            )
                            try:
                                condition, run_id, result = future.result()
                            except Exception as exc:
                                failed_count += 1
                                completed_count += 1
                                finished_by_condition[submitted_condition] += 1
                                console.print(
                                    f"[red][failed][/red] "
                                    f"{condition_label(submitted_condition)} "
                                    f"run_{submitted_run_id:03d}: {exc}"
                                )
                                continue

                            label = condition_label(condition)
                            elapsed = time.time() - started
                            console.print(
                                f"[green][{label}][/green] run_{run_id:03d} done  "
                                f"win_rate={result['best_win_rate']:.3f}  "
                                f"gen={result['generations']}  "
                                f"({elapsed:.0f}s)"
                            )
                            completed_count += 1
                            finished_by_condition[condition] += 1
                            if condition == "mono":
                                mono_results.append(result)
                            else:
                                cc_results.append(result)

                        submit_next(pool)
                        refresh_live(live)
            else:
                while inflight:
                    done, _ = wait(
                        set(inflight.keys()),
                        return_when=FIRST_COMPLETED,
                    )
                    for future in done:
                        submitted_condition, submitted_run_id, _ = inflight.pop(future)
                        with status_lock:
                            status = run_status.pop((submitted_condition, submitted_run_id), None)
                        finished_evaluated_games += (
                            int(status.get("evaluated_games", 0)) if status is not None else 0
                        )
                        try:
                            condition, run_id, result = future.result()
                        except Exception as exc:
                            failed_count += 1
                            completed_count += 1
                            finished_by_condition[submitted_condition] += 1
                            print(
                                f"[failed] {condition_label(submitted_condition)} "
                                f"run_{submitted_run_id:03d}: {exc}",
                                flush=True,
                            )
                            continue

                        label = condition_label(condition)
                        elapsed = time.time() - started
                        print(
                            f"[{label}] run_{run_id:03d} done  "
                            f"win_rate={result['best_win_rate']:.3f}  "
                            f"gen={result['generations']}  "
                            f"({elapsed:.0f}s)",
                            flush=True,
                        )
                        completed_count += 1
                        finished_by_condition[condition] += 1
                        if condition == "mono":
                            mono_results.append(result)
                        else:
                            cc_results.append(result)
                    submit_next(pool)
    finally:
        evaluator_pool.shutdown()
        ray.shutdown()
        if runtime_package is not None:
            runtime_package.cleanup()

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
