"""
CMA-ES training script for monolithic EvolvableAI on Dragonchess.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import cma
import numpy as np

from cluster.evaluator_pool import EvaluationRequest, LocalEvaluatorPool


BINARY = os.path.join(os.path.dirname(__file__), "build", "dragonchess")
PIECE_NAMES = [
    "Sylph", "Griffin", "Dragon", "Oliphant", "Unicorn",
    "Hero", "Thief", "Cleric", "Mage",
    "Paladin", "Warrior", "Basilisk", "Elemental", "Dwarf",
]
X0 = [0.0] * 14


def build_request(
    weights,
    *,
    games_per_eval: int,
    opponent: str,
    opponent_depth: int,
    threads_per_eval: int,
    timeout_s: float,
):
    return EvaluationRequest(
        gold_ai="evolvable",
        gold_weights=list(weights),
        gold_depth=1,
        scarlet_ai=opponent,
        scarlet_depth=opponent_depth,
        games=games_per_eval,
        threads=threads_per_eval,
        timeout_s=timeout_s,
        quiet=True,
    )


def evaluate_batch(evaluator, requests, retries: int = 1):
    last_error = None
    for attempt in range(retries + 1):
        try:
            return evaluator.evaluate_many(requests)
        except Exception as exc:
            last_error = exc
            if attempt == retries:
                raise
    raise last_error  # pragma: no cover


def evaluate(
    weights,
    games_per_eval=30,
    opponent="greedyvalue",
    opponent_depth=2,
    *,
    evaluator=None,
    threads_per_eval=1,
    timeout_s=300.0,
):
    owns_pool = evaluator is None
    if owns_pool:
        evaluator = LocalEvaluatorPool(
            max_workers=1,
            binary_path=BINARY,
            threads_per_eval=threads_per_eval,
        )
    try:
        request = build_request(
            weights,
            games_per_eval=games_per_eval,
            opponent=opponent,
            opponent_depth=opponent_depth,
            threads_per_eval=threads_per_eval,
            timeout_s=timeout_s,
        )
        result = evaluate_batch(evaluator, [request])[0]
        return -result.win_rate
    finally:
        if owns_pool:
            evaluator.shutdown()


def run_once(
    run_id=0,
    games_per_eval=30,
    max_generations=300,
    opponent="greedyvalue",
    opponent_depth=2,
    sigma0=3.0,
    n_workers=4,
    out_dir=None,
    verbose=True,
    *,
    evaluator=None,
    threads_per_eval=1,
    timeout_s=300.0,
):
    """Run one CMA-ES optimization and return (best_weights, fitness_log, result)."""
    es = cma.CMAEvolutionStrategy(
        X0,
        sigma0,
        {
            "maxiter": max_generations,
            "tolx": 1e-11,
            "tolfun": 1e-11,
            "verbose": -9,
            "seed": run_id,
        },
    )

    owns_pool = evaluator is None
    if owns_pool:
        evaluator = LocalEvaluatorPool(
            max_workers=n_workers,
            binary_path=BINARY,
            threads_per_eval=threads_per_eval,
        )

    fitness_log = []
    t0 = time.time()

    try:
        while not es.stop():
            candidates = es.ask()
            requests = [
                build_request(
                    candidate,
                    games_per_eval=games_per_eval,
                    opponent=opponent,
                    opponent_depth=opponent_depth,
                    threads_per_eval=threads_per_eval,
                    timeout_s=timeout_s,
                )
                for candidate in candidates
            ]
            outcomes = evaluate_batch(evaluator, requests)
            fitnesses = [-outcome.win_rate for outcome in outcomes]
            es.tell(candidates, fitnesses)

            best_fitness = min(fitnesses)
            fitness_log.append(-best_fitness)

            if verbose and (len(fitness_log) % 10 == 0 or len(fitness_log) == 1):
                elapsed = time.time() - t0
                print(
                    f"  Run {run_id:02d} | Gen {len(fitness_log):4d} | "
                    f"Best win rate: {-best_fitness:.3f} | "
                    f"sigma: {es.sigma:.4f} | "
                    f"elapsed: {elapsed:.0f}s"
                )

        best_weights = es.result.xbest
        final_request = build_request(
            best_weights,
            games_per_eval=games_per_eval,
            opponent=opponent,
            opponent_depth=opponent_depth,
            threads_per_eval=threads_per_eval,
            timeout_s=timeout_s,
        )
        best_win_rate = evaluate_batch(evaluator, [final_request])[0].win_rate

        if verbose:
            print(f"\n  Run {run_id:02d} DONE | Best win rate: {best_win_rate:.3f}")
            for name, weight in zip(PIECE_NAMES, best_weights):
                print(f"    {name:12s}: {weight:.4f}")

        result = {
            "run_id": run_id,
            "best_win_rate": best_win_rate,
            "best_weights": list(best_weights),
            "fitness_log": fitness_log,
            "generations": len(fitness_log),
            "opponent": opponent,
            "opponent_depth": opponent_depth,
            "games_per_eval": games_per_eval,
        }

        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            path = os.path.join(out_dir, f"run_{run_id:03d}.json")
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(result, handle, indent=2)
            if verbose:
                print(f"  Saved to {path}")

        return best_weights, fitness_log, result
    finally:
        if owns_pool:
            evaluator.shutdown()


def main():
    parser = argparse.ArgumentParser(description="CMA-ES training for Dragonchess monolithic agent")
    parser.add_argument("--runs", type=int, default=1, help="Number of independent runs")
    parser.add_argument("--games", type=int, default=200, help="Games per fitness evaluation")
    parser.add_argument("--generations", type=int, default=300, help="Max generations per run")
    parser.add_argument("--opponent", type=str, default="alphabeta", help="Opponent AI type")
    parser.add_argument("--opponent-depth", type=int, default=2, help="Opponent search depth")
    parser.add_argument("--sigma", type=float, default=3.0, help="Initial CMA-ES sigma")
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 4) // 2),
        help="Parallel local evaluator workers",
    )
    parser.add_argument(
        "--threads-per-eval",
        type=int,
        default=1,
        help="Threads passed to each dragonchess tournament subprocess",
    )
    parser.add_argument("--out", type=str, default=None, help="Output directory for run JSONs")
    parser.add_argument(
        "--run-id-offset",
        type=int,
        default=0,
        help="Add this to run IDs (for parallel batching on multi-node machines)",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    print("=== Monolithic CMA-ES Training ===")
    print(
        f"Runs: {args.runs} | Games/eval: {args.games} | "
        f"Max gen: {args.generations} | Opponent: {args.opponent} (depth={args.opponent_depth}) | "
        f"Workers: {args.workers} | Threads/eval: {args.threads_per_eval} | "
        f"Run-ID offset: {args.run_id_offset}"
    )
    print(f"Parameter space: {len(X0)}D (piece values, King fixed)\n")

    all_results = []
    for index in range(args.runs):
        run_id = args.run_id_offset + index
        print(f"\n--- Run {index + 1}/{args.runs} (id={run_id}) ---")
        _, _, result = run_once(
            run_id=run_id,
            games_per_eval=args.games,
            max_generations=args.generations,
            opponent=args.opponent,
            opponent_depth=args.opponent_depth,
            sigma0=args.sigma,
            n_workers=args.workers,
            out_dir=args.out,
            verbose=not args.quiet,
            threads_per_eval=args.threads_per_eval,
        )
        all_results.append(result)

    win_rates = [result["best_win_rate"] for result in all_results]
    print(f"\n=== Summary ({args.runs} runs) ===")
    print(
        f"Win rate: mean={np.mean(win_rates):.3f} "
        f"std={np.std(win_rates):.3f} "
        f"max={np.max(win_rates):.3f} "
        f"min={np.min(win_rates):.3f}"
    )

    if args.out:
        summary_path = os.path.join(args.out, "summary.json")
        with open(summary_path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "runs": all_results,
                    "win_rate_mean": float(np.mean(win_rates)),
                    "win_rate_std": float(np.std(win_rates)),
                },
                handle,
                indent=2,
            )
        print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
