"""
Cooperative Coevolution (CC) CMA-ES training for Dragonchess.

Three independent CMA-ES sub-populations, one per board tier:
  Sky        (3D): Sylph, Griffin, Dragon         [weight indices 0-2]
  Ground     (8D): Oliphant, Unicorn, Hero, Thief, Cleric, Mage, Paladin, Warrior [3-10]
  Underworld (3D): Basilisk, Elemental, Dwarf     [11-13]

Fitness assignment: best-collaborator strategy.
  Each candidate is evaluated with the current best individual from the other
  two sub-populations to form a complete 14-weight vector.

Usage:
    python3 train_cc.py                        # single run, results to stdout
    python3 train_cc.py --runs 30 --out results/cc/
    python3 train_cc.py --runs 30 --workers 6 --out results/cc/
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

# Piece order (matches monolithic agent and EvolvableAI weight indices)
PIECE_NAMES = [
    "Sylph", "Griffin", "Dragon",                                    # Sky      [0-2]
    "Oliphant", "Unicorn", "Hero", "Thief", "Cleric", "Mage",        # Ground   [3-8]
    "Paladin", "Warrior",                                            # Ground   [9-10]
    "Basilisk", "Elemental", "Dwarf"                                 # UW       [11-13]
]

# Tier definitions: (name, slice_start, slice_end, x0, sigma0)
# sigma0=3.0: Jackman piece values span 1-9; sigma=3 explores that range from x0=0
TIERS = [
    ("Sky",        0,  3, [0.0, 0.0, 0.0],                           3.0),
    ("Ground",     3, 11, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 3.0),
    ("Underworld", 11, 14, [0.0, 0.0, 0.0],                          3.0),
]


def weights_to_str(weights):
    return ",".join(f"{w:.6f}" for w in weights)


def assemble_weights(sky_w, ground_w, uw_w):
    """Combine three tier weight vectors into a single 14-weight list."""
    return list(sky_w) + list(ground_w) + list(uw_w)


def evaluate(weights, games_per_eval=30, opponent="greedyvalue", opponent_depth=2):
    """Run a headless tournament and return negative win rate."""
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
            return 0.5
        data = json.loads(result.stdout)
        summary = data["summary"]
        wins = summary["gold_wins"]
        total = summary["total_games"]
        win_rate = wins / total if total > 0 else 0.0
        return -win_rate
    except Exception:
        return 0.5


def run_once(run_id=0, games_per_eval=30, max_generations=300,
             opponent="greedyvalue", opponent_depth=2, n_workers=4, out_dir=None, verbose=True):
    """Run one CC-CMA-ES optimization and return (best_weights, fitness_log, result)."""

    # Initialize one CMA-ES per tier
    es_list = []
    for name, start, end, x0, sigma0 in TIERS:
        es = cma.CMAEvolutionStrategy(
            x0, sigma0,
            {
                "maxiter": max_generations,
                "tolx":   1e-11,   # disable premature stopping in noisy fitness landscape
                "tolfun": 1e-11,
                "verbose": -9,
                "seed": run_id * 10 + len(es_list),
            }
        )
        es_list.append(es)

    # Current best for each tier (used as collaborators); start with x0
    best_per_tier = [np.array(x0) for (_, _, _, x0, _) in TIERS]

    fitness_log = []  # best composite win rate per generation
    t0 = time.time()
    generation = 0

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        while generation < max_generations:
            # Check if all sub-populations have stopped
            if all(es.stop() for es in es_list):
                break

            generation += 1

            # ---- Evaluate each tier sequentially (collaborators updated after each tier) ----
            for t_idx, (es, (name, start, end, _, _)) in enumerate(zip(es_list, TIERS)):
                if es.stop():
                    continue

                candidates = es.ask()
                collaborators = [best_per_tier[i] for i in range(3) if i != t_idx]

                # Build full weight vectors for all candidates in this tier
                full_weights_list = []
                for cand in candidates:
                    parts = [None, None, None]
                    parts[t_idx] = cand
                    c_iter = iter(collaborators)
                    for i in range(3):
                        if i != t_idx:
                            parts[i] = next(c_iter)
                    full_weights_list.append(assemble_weights(*parts))

                # Evaluate all candidates in this tier in parallel
                fitnesses = list(executor.map(
                    evaluate,
                    full_weights_list,
                    [games_per_eval] * len(candidates),
                    [opponent] * len(candidates),
                    [opponent_depth] * len(candidates),
                ))

                es.tell(candidates, fitnesses)

                # Update best collaborator for this tier
                best_idx = int(np.argmin(fitnesses))
                best_per_tier[t_idx] = np.array(candidates[best_idx])

            # Evaluate best composite this generation (logged as win rate)
            best_full = assemble_weights(*best_per_tier)
            best_fitness = evaluate(best_full, games_per_eval, opponent, opponent_depth)
            win_rate = -best_fitness
            fitness_log.append(win_rate)

            if verbose and (generation % 10 == 0 or generation == 1):
                elapsed = time.time() - t0
                sigmas = [es.sigma for es in es_list if not es.stop()]
                sigma_str = "/".join(f"{s:.4f}" for s in sigmas)
                print(f"  Run {run_id:02d} | Gen {generation:4d} | "
                      f"Best win rate: {win_rate:.3f} | "
                      f"sigmas: {sigma_str} | "
                      f"elapsed: {elapsed:.0f}s")

    # Collect final best weights from each tier
    best_weights = assemble_weights(
        es_list[0].result.xbest if es_list[0].result.xbest is not None else best_per_tier[0],
        es_list[1].result.xbest if es_list[1].result.xbest is not None else best_per_tier[1],
        es_list[2].result.xbest if es_list[2].result.xbest is not None else best_per_tier[2],
    )

    # Final evaluation (same games_per_eval as monolithic for direct comparability)
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
        "generations": generation,
        "opponent": opponent,
        "opponent_depth": opponent_depth,
        "games_per_eval": games_per_eval,
        "tier_best": {
            "sky":        list(best_per_tier[0]),
            "ground":     list(best_per_tier[1]),
            "underworld": list(best_per_tier[2]),
        },
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
    parser = argparse.ArgumentParser(
        description="CC-CMA-ES training for Dragonchess (3 tier sub-agents)")
    parser.add_argument("--runs",        type=int,   default=1,           help="Independent runs")
    parser.add_argument("--games",       type=int,   default=200,         help="Games per fitness eval")
    parser.add_argument("--generations", type=int,   default=300,         help="Max generations per run")
    parser.add_argument("--opponent",       type=str,   default="alphabeta",   help="Opponent AI type")
    parser.add_argument("--opponent-depth", type=int,   default=2,             help="Search depth for minimax/alphabeta opponent")
    parser.add_argument("--workers",     type=int,   default=max(1, (os.cpu_count() or 4) // 2),
                                                                           help="Parallel eval workers")
    parser.add_argument("--out",         type=str,   default=None,        help="Output directory")
    parser.add_argument("--run-id-offset", type=int, default=0,
                        help="Add to run IDs (for parallel batching on multi-node machines)")
    parser.add_argument("--quiet",       action="store_true")
    args = parser.parse_args()

    print(f"=== CC-CMA-ES Training (3 Tier Sub-agents) ===")
    print(f"Runs: {args.runs} | Games/eval: {args.games} | "
          f"Max gen: {args.generations} | Opponent: {args.opponent} (depth={args.opponent_depth}) | "
          f"Workers: {args.workers} | Run-ID offset: {args.run_id_offset}")
    print(f"Tier dims: Sky=3D, Ground=8D, Underworld=3D (total 14D)\n")

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
            json.dump({
                "runs": all_results,
                "win_rate_mean": float(np.mean(win_rates)),
                "win_rate_std":  float(np.std(win_rates)),
            }, f, indent=2)
        print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
