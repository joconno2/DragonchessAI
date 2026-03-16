"""
Cooperative Coevolution (CC) CMA-ES training for Dragonchess.
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
    "Sylph", "Griffin", "Dragon",
    "Oliphant", "Unicorn", "Hero", "Thief", "Cleric", "Mage",
    "Paladin", "Warrior",
    "Basilisk", "Elemental", "Dwarf",
]
TIERS = [
    ("Sky", 0, 3, [0.0, 0.0, 0.0], 3.0),
    ("Ground", 3, 11, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 3.0),
    ("Underworld", 11, 14, [0.0, 0.0, 0.0], 3.0),
]


def assemble_weights(sky_w, ground_w, uw_w):
    return list(sky_w) + list(ground_w) + list(uw_w)


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


def utc_timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


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
    n_workers=4,
    out_dir=None,
    verbose=True,
    *,
    evaluator=None,
    threads_per_eval=1,
    timeout_s=300.0,
    progress_callback=None,
):
    """Run one CC-CMA-ES optimization and return (best_weights, fitness_log, result)."""
    es_list = []
    for _, _, _, x0, sigma0 in TIERS:
        es = cma.CMAEvolutionStrategy(
            x0,
            sigma0,
            {
                "maxiter": max_generations,
                "tolx": 1e-11,
                "tolfun": 1e-11,
                "verbose": -9,
                "seed": run_id * 10 + len(es_list),
            },
        )
        es_list.append(es)

    owns_pool = evaluator is None
    if owns_pool:
        evaluator = LocalEvaluatorPool(
            max_workers=n_workers,
            binary_path=BINARY,
            threads_per_eval=threads_per_eval,
        )

    best_per_tier = [np.array(x0) for (_, _, _, x0, _) in TIERS]
    fitness_log = []
    t0 = time.time()
    generation = 0
    evaluated_games = 0

    try:
        while generation < max_generations:
            if all(es.stop() for es in es_list):
                break

            generation += 1

            for tier_index, (es, _) in enumerate(zip(es_list, TIERS)):
                if es.stop():
                    continue

                candidates = es.ask()
                collaborators = [best_per_tier[i] for i in range(3) if i != tier_index]
                full_weights_list = []
                for candidate in candidates:
                    parts = [None, None, None]
                    parts[tier_index] = candidate
                    collaborator_iter = iter(collaborators)
                    for part_index in range(3):
                        if part_index != tier_index:
                            parts[part_index] = next(collaborator_iter)
                    full_weights_list.append(assemble_weights(*parts))

                requests = [
                    build_request(
                        weights,
                        games_per_eval=games_per_eval,
                        opponent=opponent,
                        opponent_depth=opponent_depth,
                        threads_per_eval=threads_per_eval,
                        timeout_s=timeout_s,
                    )
                    for weights in full_weights_list
                ]
                outcomes = evaluate_batch(evaluator, requests)
                evaluated_games += len(requests) * games_per_eval
                fitnesses = [-outcome.win_rate for outcome in outcomes]
                es.tell(candidates, fitnesses)

                best_index = int(np.argmin(fitnesses))
                best_per_tier[tier_index] = np.array(candidates[best_index])

            best_full = assemble_weights(*best_per_tier)
            best_request = build_request(
                best_full,
                games_per_eval=games_per_eval,
                opponent=opponent,
                opponent_depth=opponent_depth,
                threads_per_eval=threads_per_eval,
                timeout_s=timeout_s,
            )
            win_rate = evaluate_batch(evaluator, [best_request])[0].win_rate
            evaluated_games += games_per_eval
            fitness_log.append(win_rate)
            if progress_callback is not None:
                progress_callback(
                    generation=generation,
                    evaluated_games=evaluated_games,
                    latest_win_rate=win_rate,
                    best_win_rate=max(fitness_log),
                )

            if verbose and (generation % 10 == 0 or generation == 1):
                elapsed = time.time() - t0
                sigmas = [es.sigma for es in es_list if not es.stop()]
                sigma_str = "/".join(f"{sigma:.4f}" for sigma in sigmas)
                print(
                    f"  Run {run_id:02d} | Gen {generation:4d} | "
                    f"Best win rate: {win_rate:.3f} | "
                    f"sigmas: {sigma_str} | "
                    f"elapsed: {elapsed:.0f}s"
                )

        best_weights = assemble_weights(
            es_list[0].result.xbest if es_list[0].result.xbest is not None else best_per_tier[0],
            es_list[1].result.xbest if es_list[1].result.xbest is not None else best_per_tier[1],
            es_list[2].result.xbest if es_list[2].result.xbest is not None else best_per_tier[2],
        )
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
            "completed_at_utc": utc_timestamp(),
            "best_win_rate": best_win_rate,
            "best_weights": list(best_weights),
            "fitness_log": fitness_log,
            "generations": generation,
            "opponent": opponent,
            "opponent_depth": opponent_depth,
            "games_per_eval": games_per_eval,
            "tier_best": {
                "sky": list(best_per_tier[0]),
                "ground": list(best_per_tier[1]),
                "underworld": list(best_per_tier[2]),
            },
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
    parser = argparse.ArgumentParser(
        description="CC-CMA-ES training for Dragonchess (3 tier sub-agents)"
    )
    parser.add_argument("--runs", type=int, default=1, help="Independent runs")
    parser.add_argument("--games", type=int, default=200, help="Games per fitness eval")
    parser.add_argument("--generations", type=int, default=300, help="Max generations per run")
    parser.add_argument("--opponent", type=str, default="alphabeta", help="Opponent AI type")
    parser.add_argument("--opponent-depth", type=int, default=2, help="Opponent search depth")
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
    parser.add_argument("--out", type=str, default=None, help="Output directory")
    parser.add_argument(
        "--run-id-offset",
        type=int,
        default=0,
        help="Add to run IDs (for parallel batching on multi-node machines)",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    print("=== CC-CMA-ES Training (3 Tier Sub-agents) ===")
    print(
        f"Runs: {args.runs} | Games/eval: {args.games} | "
        f"Max gen: {args.generations} | Opponent: {args.opponent} (depth={args.opponent_depth}) | "
        f"Workers: {args.workers} | Threads/eval: {args.threads_per_eval} | "
        f"Run-ID offset: {args.run_id_offset}"
    )
    print("Tier dims: Sky=3D, Ground=8D, Underworld=3D (total 14D)\n")

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
