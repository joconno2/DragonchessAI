"""
CMA-ES training script for monolithic EvolvableAI on Dragonchess.

Optimizes 14 piece-value weights (King excluded, fixed at 10000).
Piece order: Sylph, Griffin, Dragon, Oliphant, Unicorn, Hero, Thief,
             Cleric, Mage, Paladin, Warrior, Basilisk, Elemental, Dwarf

Usage:
    python3 train_cma.py                        # single run, results to stdout
    python3 train_cma.py --runs 30 --out results/monolithic/
    python3 train_cma.py --runs 30 --workers 6 --out results/monolithic/
"""

import subprocess
import json
import os
import argparse
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import cma

BINARY = os.path.join(os.path.dirname(__file__), "build", "dragonchess")
PIECE_NAMES = [
    "Sylph", "Griffin", "Dragon", "Oliphant", "Unicorn",
    "Hero", "Thief", "Cleric", "Mage",
    "Paladin", "Warrior", "Basilisk", "Elemental", "Dwarf"
]

# Starting weights: zero (uninformed) so convergence curves show real learning
X0 = [0.0] * 14


def weights_to_str(weights):
    return ",".join(f"{w:.6f}" for w in weights)


def evaluate(weights, games_per_eval=30, opponent="greedyvalue", opponent_depth=2):
    """Returns negative win rate (CMA-ES minimizes)."""
    try:
        cmd = [
            BINARY, "--headless", "--mode", "tournament",
            "--games", str(games_per_eval),
            "--gold-ai", "evolvable",
            "--gold-weights", weights_to_str(weights),
            "--gold-depth", "1",   # depth=1: piece values determine outcome, not lookahead
            "--scarlet-ai", opponent,
            "--scarlet-depth", str(opponent_depth),
            "--quiet",
            "--output-json", "-",  # write JSON to stdout; no temp file needed
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=300)
        if result.returncode != 0:
            return 0.5  # neutral on failure
        data = json.loads(result.stdout)
        summary = data["summary"]
        wins = summary["gold_wins"]
        total = summary["total_games"]
        win_rate = wins / total if total > 0 else 0.0
        return -win_rate  # negate for minimization
    except Exception:
        return 0.5


def run_once(run_id=0, games_per_eval=30, max_generations=300,
             opponent="greedyvalue", opponent_depth=2, sigma0=3.0, n_workers=4,
             out_dir=None, verbose=True):
    """Run one CMA-ES optimization and return (best_weights, fitness_log, result)."""
    es = cma.CMAEvolutionStrategy(
        X0, sigma0,
        {
            "maxiter": max_generations,
            "tolx":    1e-11,   # disable premature stopping in noisy fitness landscape
            "tolfun":  1e-11,
            "verbose": -9,      # suppress cma's own output
            "seed": run_id,
        }
    )

    fitness_log = []  # best win rate per generation
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        while not es.stop():
            candidates = es.ask()
            # Evaluate all candidates in parallel
            fitnesses = list(executor.map(
                evaluate,
                candidates,
                [games_per_eval] * len(candidates),
                [opponent] * len(candidates),
                [opponent_depth] * len(candidates),
            ))
            es.tell(candidates, fitnesses)

            best_f = min(fitnesses)
            fitness_log.append(-best_f)  # store as win rate (positive)

            if verbose and (len(fitness_log) % 10 == 0 or len(fitness_log) == 1):
                elapsed = time.time() - t0
                print(f"  Run {run_id:02d} | Gen {len(fitness_log):4d} | "
                      f"Best win rate: {-best_f:.3f} | "
                      f"sigma: {es.sigma:.4f} | "
                      f"elapsed: {elapsed:.0f}s")

    best_weights = es.result.xbest
    # Final evaluation (same games_per_eval as CC for direct comparability)
    # Retry once on crash to avoid storing crash sentinel (0.5) as win rate
    _final = evaluate(best_weights, games_per_eval, opponent, opponent_depth)
    if _final == 0.5:
        _final = evaluate(best_weights, games_per_eval, opponent, opponent_depth)
    best_win_rate = -_final

    if verbose:
        print(f"\n  Run {run_id:02d} DONE | Best win rate: {best_win_rate:.3f}")
        for name, w in zip(PIECE_NAMES, best_weights):
            print(f"    {name:12s}: {w:.4f}")

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
        with open(path, "w") as f:
            json.dump(result, f, indent=2)
        if verbose:
            print(f"  Saved to {path}")

    return best_weights, fitness_log, result


def main():
    parser = argparse.ArgumentParser(description="CMA-ES training for Dragonchess monolithic agent")
    parser.add_argument("--runs",        type=int,   default=1,           help="Number of independent runs")
    parser.add_argument("--games",       type=int,   default=200,         help="Games per fitness evaluation")
    parser.add_argument("--generations", type=int,   default=300,         help="Max generations per run")
    parser.add_argument("--opponent",       type=str,   default="alphabeta",   help="Opponent AI type")
    parser.add_argument("--opponent-depth", type=int,   default=2,             help="Search depth for minimax/alphabeta opponent")
    parser.add_argument("--sigma",       type=float, default=3.0,         help="Initial CMA-ES sigma")
    parser.add_argument("--workers",     type=int,   default=max(1, (os.cpu_count() or 4) // 2),
                                                                           help="Parallel eval workers")
    parser.add_argument("--out",         type=str,   default=None,        help="Output directory for run JSONs")
    parser.add_argument("--run-id-offset", type=int, default=0,
                        help="Add this to run IDs (for parallel batching on multi-node machines)")
    parser.add_argument("--quiet",       action="store_true")
    args = parser.parse_args()

    print(f"=== Monolithic CMA-ES Training ===")
    print(f"Runs: {args.runs} | Games/eval: {args.games} | "
          f"Max gen: {args.generations} | Opponent: {args.opponent} (depth={args.opponent_depth}) | "
          f"Workers: {args.workers} | Run-ID offset: {args.run_id_offset}")
    print(f"Parameter space: {len(X0)}D (piece values, King fixed)\n")

    all_results = []
    for i in range(args.runs):
        run_id = args.run_id_offset + i
        print(f"\n--- Run {i + 1}/{args.runs} (id={run_id}) ---")
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
        )
        all_results.append(result)

    win_rates = [r["best_win_rate"] for r in all_results]
    print(f"\n=== Summary ({args.runs} runs) ===")
    print(f"Win rate: mean={np.mean(win_rates):.3f} "
          f"std={np.std(win_rates):.3f} "
          f"max={np.max(win_rates):.3f} "
          f"min={np.min(win_rates):.3f}")

    if args.out:
        summary_path = os.path.join(args.out, "summary.json")
        with open(summary_path, "w") as f:
            json.dump({"runs": all_results, "win_rate_mean": float(np.mean(win_rates)),
                       "win_rate_std": float(np.std(win_rates))}, f, indent=2)
        print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
