# DragonchessAI — CMA-ES Piece Value Experiment
## Handoff Guide for Research Continuation

**Status as of March 2026:** Preliminary runs complete on local hardware. Full experiment
ready to launch on Ray cluster. Read this document top to bottom before touching any code.

---

## Table of Contents

1. [Research Question](#1-research-question)
2. [Background](#2-background)
3. [Experimental Design](#3-experimental-design)
4. [Code Structure](#4-code-structure)
5. [Preliminary Results and What They Mean](#5-preliminary-results-and-what-they-mean)
6. [Running the Full Experiment on Ray](#6-running-the-full-experiment-on-ray)
7. [Analyzing Results](#7-analyzing-results)
8. [Post-Training Evaluation](#8-post-training-evaluation)
9. [Open Questions and Next Steps](#9-open-questions-and-next-steps)
10. [Design Decisions Log](#10-design-decisions-log)

---

## 1. Research Question

**Can cooperative coevolution improve the learning of piece values in Dragonchess?**

Dragonchess is a three-board variant of chess invented by Gary Gygax. It has 14 distinct piece
types spread across three tiers (Sky, Ground, Underworld), each tier with its own movement rules.
The correct relative values of these pieces are not well established — Jackman (1997) provides
expert estimates, but no empirical optimization has been done.

We optimize piece values using CMA-ES, a state-of-the-art evolutionary strategy optimizer. We
compare two problem decompositions:

- **Monolithic:** Treat all 14 piece values as a single 14-dimensional optimization problem.
- **Cooperative Coevolution (CC):** Split the 14 values into three sub-problems by board tier
  (Sky 3D, Ground 8D, Underworld 3D) and run three independent CMA-ES optimizers that
  collaborate by sharing their current-best solutions.

The hypothesis is that CC will converge faster and/or to better piece values because the
problem is naturally decomposable — Sky pieces almost never interact with Underworld pieces
mid-game, so optimizing them together just adds noise to each other's fitness signal.

---

## 2. Background

### 2.1 What is CMA-ES?

CMA-ES (Covariance Matrix Adaptation Evolution Strategy) is an evolutionary algorithm for
continuous black-box optimization. Unlike gradient descent, it does not require derivatives.
It maintains a population of candidate solutions (here: 14-dimensional weight vectors) and
iteratively updates a multivariate Gaussian from which new candidates are sampled. The key
adaptation is to the covariance matrix, which lets the algorithm automatically learn correlations
between parameters.

We use the `cma` Python package. Key parameters for this experiment:
- `sigma0 = 3.0` — initial standard deviation. Jackman values span 1–9, so σ=3 lets the
  algorithm explore the plausible range from a zero-initialized starting point.
- `maxiter = 300` — maximum generations. In practice convergence often happens before this.
- `tolx = tolfun = 1e-11` — deliberately loose tolerances to prevent premature stopping
  in a noisy fitness landscape (our win-rate estimates have statistical noise).
- Population size: CMA-ES default λ = 4 + floor(3·ln(N)). For 14D: λ ≈ 9. For 3D: λ ≈ 6.
  This means each generation evaluates ~9 candidates (mono) or ~6 per tier (CC).

### 2.2 What is Cooperative Coevolution?

Standard (monolithic) optimization treats all 14 dimensions together. Cooperative Coevolution
(Potter & De Jong 1994) splits the problem into subcomponents and evolves each independently,
with each subpopulation using the best-known solutions from the other subpopulations as fixed
collaborators when computing fitness.

**Our CC variant (3 tiers):**
- Sky sub-ES optimizes weights for {Sylph, Griffin, Dragon} (3D)
- Ground sub-ES optimizes weights for {Oliphant, Unicorn, Hero, Thief, Cleric, Mage, Paladin,
  Warrior} (8D)
- Underworld sub-ES optimizes weights for {Basilisk, Elemental, Dwarf} (3D)

Each generation: for each tier, evaluate all candidates in that tier by filling in the other
two tiers with the current best known weights from those tiers. After evaluating a tier, update
its best-known solution. This is the **best-collaborator strategy**.

**Why might CC help here?**
The three boards in Dragonchess are semi-independent. A Sylph (Sky piece) rarely interacts
directly with a Dwarf (Underworld piece) in a single game. Evaluating their values together
means the fitness signal for Sylph weights is noisy due to random Dwarf weight sampling and
vice versa. CC fixes collaborators, reducing this noise and potentially leading to faster,
more reliable convergence.

**Why might CC hurt?**
Pieces do interact across tiers (a Dragon can capture a Ground piece). If inter-tier piece
value relationships are important, decomposing them may find locally optimal within-tier
values that are globally suboptimal. This is a real risk we are testing empirically.

### 2.3 Piece Value Table

| Index | Piece      | Tier       | Jackman (1997) |
|-------|-----------|------------|----------------|
| 0     | Sylph     | Sky        | 1.0            |
| 1     | Griffin   | Sky        | 2.0            |
| 2     | Dragon    | Sky        | 9.0            |
| 3     | Oliphant  | Ground     | 5.0            |
| 4     | Unicorn   | Ground     | 8.0            |
| 5     | Hero      | Ground     | 5.0            |
| 6     | Thief     | Ground     | 2.5            |
| 7     | Cleric    | Ground     | 4.5            |
| 8     | Mage      | Ground     | 4.0            |
| 9     | Paladin   | Ground     | 8.0            |
| 10    | Warrior   | Ground     | 1.0            |
| 11    | Basilisk  | Underworld | 3.0            |
| 12    | Elemental | Underworld | 4.0            |
| 13    | Dwarf     | Underworld | 2.0            |

King is excluded (fixed at 10000 internally — losing it is instant loss).
All weights start at 0.0 and are learned from scratch.

---

## 3. Experimental Design

### 3.1 The Planned Full Experiment

| Parameter        | Value                                      |
|------------------|--------------------------------------------|
| Conditions       | Monolithic CMA-ES vs CC-CMA-ES             |
| Runs per condition | 30 independent runs (different RNG seeds)|
| Games per fitness eval | 200                                  |
| Max generations  | 300                                        |
| Opponent during training | `alphabeta` at depth 2             |
| Eval depth (training) | depth=1 (piece values drive decisions)|
| Post-training eval depth | depth=3                            |
| Significance threshold | α = 0.05 (Mann-Whitney U, two-sided) |
| Target effect size | Cohen's d ≥ 0.5 for practical interest  |

**Why `alphabeta` depth=2 as training opponent?**
Greedy opponents can be beaten with trivially learned values (just prefer capturing any piece
at all). A depth-2 alpha-beta opponent requires the agent to learn values that hold up over a
two-move lookahead, which is a more meaningful test of the piece value representation.

**Why depth=1 for the evolved agent during training?**
At depth=1, the agent's decisions are determined almost entirely by piece values (it picks
moves that maximize immediate material gain). This isolates what we want to measure: do the
learned values reflect actual piece strength? If we used depth=3 during training, we'd be
jointly learning values AND search, confounding the comparison.

**Why 200 games per eval?**
With 10 games, win rate estimates are quantized to 0.1 increments and have enormous variance.
With 200 games, a 60% win rate has a 95% CI of roughly ±7%. That's still noisy but CMA-ES is
robust to moderate noise — this is the tradeoff between evaluation quality and compute cost.

**Why 30 runs?**
Power analysis: to detect d=0.5 between conditions at α=0.05 with 80% power (two-sample
Mann-Whitney), you need ~27 runs per condition. 30 gives modest headroom.

### 3.2 What "One Run" Means

For **monolithic**: CMA-ES optimizes a 14-dimensional weight vector. Each generation evaluates
~9 candidates (CMA-ES default population size for 14D), each candidate plays 200 games as Gold
against alphabeta-D2 as Scarlet. Fitness = negative win rate (CMA-ES minimizes). Workers are
used to evaluate candidates in parallel within a generation.

For **CC**: Three CMA-ES instances run simultaneously, one per tier. Each generation:
1. For each tier in turn:
   a. Sample ~6 candidates (population size for 3D or 8D)
   b. Build each candidate's full 14-weight vector by combining it with the current best from
      the other two tiers
   c. Evaluate all candidates in parallel (200 games each)
   d. Update the tier CMA-ES; store the best candidate as the new best collaborator for this tier
2. Log the current best composite win rate

**Termination:** Early stop when all sub-ES have converged (CMA-ES internal criterion with our
loose tolerances), or when max_generations is reached.

---

## 4. Code Structure

```
DragonchessAI/
├── build/dragonchess          # Compiled C++ binary (required; see Building)
├── src/                       # C++ game engine source
│   ├── ai.h / ai.cpp          # AI classes (BaseAI, AlphaBetaAI, EvolvableAI, ...)
│   ├── headless.h / headless.cpp  # Headless tournament runner
│   ├── game.h / game.cpp      # Game state, move gen, threefold repetition
│   └── main.cpp               # CLI entry point and argument parsing
│
├── train_cma.py               # Monolithic CMA-ES training
├── train_cc.py                # Cooperative coevolution CMA-ES training
├── run_ray.py                 # Ray cluster launcher (runs both conditions)
├── analyze_results.py         # Paper-ready figures (convergence, distribution, weights)
├── dashboard.py               # Quick experiment-status dashboard
├── evaluate_posttrain.py      # Post-training depth=3 evaluation
│
├── results/
│   ├── monolithic/            # run_000.json ... run_029.json
│   ├── cc/                    # run_000.json ... run_029.json
│   └── posttrain/             # posttrain_eval.json (from evaluate_posttrain.py)
└── figures/                   # Generated figures (PDF + PNG)
```

### 4.1 Key Function Map

| Script              | Key function      | What it does                                            |
|---------------------|------------------|---------------------------------------------------------|
| `train_cma.py`      | `run_once()`     | One full monolithic CMA-ES run; returns result dict     |
| `train_cma.py`      | `evaluate()`     | Play N games, return negative win rate                  |
| `train_cc.py`       | `run_once()`     | One full CC-CMA-ES run; returns result dict             |
| `train_cc.py`       | `assemble_weights()` | Combine three tier vectors into 14-weight list      |
| `run_ray.py`        | `run_mono_ray()` | Distribute 30 mono runs across Ray cluster             |
| `run_ray.py`        | `run_cc_ray()`   | Distribute 30 CC runs across Ray cluster               |
| `analyze_results.py`| `fig_convergence()` | Figure 1: mean win rate vs generation               |
| `analyze_results.py`| `fig_final_distribution()` | Figure 2: violin+box of final win rates   |
| `analyze_results.py`| `fig_piece_values()` | Figure 3: evolved weights vs Jackman bar chart    |
| `evaluate_posttrain.py` | `evaluate_weights()` | 4-matchup depth=3 evaluation of a weight set |

### 4.2 Result JSON Format

Each `run_NNN.json` contains:
```json
{
  "run_id": 0,
  "best_win_rate": 0.62,
  "best_weights": [w0, w1, ..., w13],
  "fitness_log": [0.31, 0.45, ..., 0.62],
  "generations": 247,
  "opponent": "alphabeta",
  "opponent_depth": 2,
  "games_per_eval": 200,
  "tier_best": {              // CC runs only
    "sky": [w0, w1, w2],
    "ground": [w3, ..., w10],
    "underworld": [w11, w12, w13]
  }
}
```
`fitness_log[i]` is the best win rate seen in generation i (not negative; already flipped).

### 4.3 Building the C++ Binary

```bash
cd DragonchessAI
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
cd ..
# Verify:
./build/dragonchess --headless --mode tournament --games 5 \
    --gold-ai greedyvalue --scarlet-ai random --quiet --output-json -
```
You should see a JSON summary printed to stdout.

**Required system libraries:** SDL2, SDL2_image, SDL2_ttf (only needed for GUI; headless mode
does not use them at runtime, but CMake still links them). On Ubuntu/Debian:
```bash
sudo apt install libsdl2-dev libsdl2-image-dev libsdl2-ttf-dev
```

---

## 5. Preliminary Results and What They Mean

### 5.1 What These Runs Are

The `results/monolithic/` and `results/cc/` directories contain **pilot runs** done on a local
workstation, NOT the planned full experiment. Key differences:

| Parameter           | Preliminary (current)    | Full experiment (planned) |
|---------------------|--------------------------|---------------------------|
| Opponent            | `greedyvalue`            | `alphabeta` depth=2        |
| Games per eval      | 10                       | 200                        |
| Max generations     | 300                      | 300                        |
| Actual generations  | ~90 (converged early)    | Unknown (expect more)      |
| Runs — Monolithic   | 20                       | 30                         |
| Runs — CC           | 7                        | 30                         |

With only 10 games/eval, win rate estimates are quantized to 0.1 steps with huge variance.
The early convergence at ~90 generations is consistent with CMA-ES finding a locally good
region quickly on such a noisy surface and then stagnating. These should be treated as
"sanity checks that the pipeline works," not as scientific results.

### 5.2 Summary of Preliminary Numbers

```
Condition        n    mean    std    [min, max]
Monolithic      20   0.620   0.133  [0.40, 0.90]
CC-CMA-ES        7   0.571   0.116  [0.40, 0.80]

Mann-Whitney p = 0.345  (not significant — n too small, noise too high)
Cohen's d = 0.37  (medium effect numerically, but unreliable with n=7)
```

No conclusions should be drawn from these numbers about which method is better. The
experiment is underpowered and the opponent is too weak.

### 5.3 Preliminary Piece Weights

The evolved weights (mean across runs) differ substantially from Jackman's expert estimates:

```
Piece          Jackman    Mono(prelim)   CC(prelim)
Dragon           9.00         2.08          2.57
Unicorn          8.00         1.14          0.00
Paladin          8.00        -0.07          2.77
Sylph            1.00        -0.61         -0.83
Warrior          1.00         0.59          0.97
```

**Likely explanation:** Training against a depth-1 GreedyValue opponent that plays greedily
by immediate capture value means the fitness landscape favors weights that beat *that specific
strategy*, not weights that reflect true piece strength. Dragon and Unicorn are strong strategic
pieces but their value may not manifest in 10 short games against a weak greedy opponent.

The negative Sylph weight is interesting: Sylphs may be so easy to lose to a greedy opponent
that the learner discovers it's optimal to not defend them (and therefore gives them negative
value so the engine doesn't waste moves protecting them). This hypothesis should be tested in
the full experiment.

### 5.4 What to Check When Full Results Arrive

1. Do both conditions beat alphabeta-D2 substantially (win rate > 0.55)? If not, 200 games/eval
   may still be too few or 300 generations too few — check convergence curves.
2. Are the evolved Dragon/Unicorn values closer to Jackman with the stronger opponent?
3. Does CC converge faster (fewer generations to plateau)?
4. Does CC reach higher final win rates?
5. Are CC weights more consistent across runs (lower std)?

---

## 6. Running the Full Experiment on Ray

### 6.1 Prerequisites

On your Mac or other head machine, install the full Python dependencies:

- Python packages from `requirements-cluster.txt`

Workers only need:

- a compatible `dragonchess` binary at `~/DragonchessAI/build/dragonchess`
- the `ray` Python package

Build the headless binary once locally:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DHEADLESS_ONLY=ON
cmake --build build --parallel
```

Then distribute the repo and the prebuilt binary to the workers:

```bash
python3 setup/cluster_sync.py \
    --workers-file workers.csv \
    --repo-dir ~/DragonchessAI \
    --binary-path build/dragonchess \
    --install-python-deps
```

That script syncs the repo, copies the prebuilt binary into each worker's `build/` directory, and
optionally installs `ray` on the workers. It does not run `cmake` on the cluster nodes.

### 6.2 Starting a Ray Cluster

```bash
python3 setup/setup_ray.py \
    --workers-file workers.csv \
    --local-head \
    --head-ip YOUR_MAC_IP \
    --restart
```

This starts the Ray head on your local machine and connects the workers to it. `setup_ray.py`
starts Ray with `--num-cpus <core_usage>` on every worker, so the worker-side cluster capacity
respects the CPU budget declared in `workers.csv`.

### 6.3 Running the Experiment

```bash
# From the DragonchessAI directory on the Ray head node:
python3 run_ray.py \
    --address auto \
    --workers-csv workers.csv \
    --repo-dir ~/DragonchessAI \
    --runs 30 \
    --games 200 \
    --generations 300 \
    --opponent alphabeta \
    --opponent-depth 2 \
    --parallel-runs 4 \
    --threads-per-eval 1 \
    --out-mono results/monolithic/ \
    --out-cc   results/cc/
```

`run_ray.py` now uses a shared evaluator-actor pool across the cluster:

- each worker contributes `core_usage` evaluator slots
- each evaluator actor reserves `num_cpus=1`
- each tournament subprocess runs with `--threads 1`
- `--parallel-runs` controls how many training coordinators are active on the head node at once

This means cluster capacity comes from the evaluator pool, not from allocating whole-Ray-task CPU
bundles to individual runs.

```bash
# Monitor progress:
python3 dashboard.py   # regenerate as runs complete
watch -n60 'ls results/monolithic/*.json results/cc/*.json | wc -l'
```

### 6.4 What run_ray.py Does

`run_ray.py` now:
1. Calls `ray.init(address='auto')` to connect to the running cluster
2. Builds a shared pool of evaluator actors from `workers.csv`
3. Launches multiple monolithic and CC run coordinators on the head node
4. Lets those runs share the evaluator pool while writing results locally on the head node

Each completed run still writes the same `run_NNN.json` schema, so the downstream dashboard and
analysis scripts remain unchanged.

### 6.5 Fault Tolerance

Evaluator failures now propagate as explicit exceptions instead of being silently converted into
neutral fitness. `run_ray.py` catches per-run failures and logs them; successful runs are still
saved. Re-run with
`--run-id-offset N` to fill in missing runs without overwriting completed ones.

---

## 7. Analyzing Results

Once all 30+30 runs are complete:

```bash
# Paper-ready figures (saves to figures/)
python3 analyze_results.py --mono results/monolithic/ --cc results/cc/ --out figures/

# Quick dashboard (live-updatable during the run)
python3 dashboard.py --mono results/monolithic/ --cc results/cc/ --out figures/
```

**Figures produced:**
- `fig1_convergence.{pdf,png}` — Mean win rate vs generation with 95% CI shading
- `fig2_distribution.{pdf,png}` — Violin + box + jitter of final win rates; reports
  Mann-Whitney p-value and rank-biserial r
- `fig3_piece_values.{pdf,png}` — Bar chart comparing evolved vs Jackman weights, grouped
  by tier

**Statistics reported:**
- Mann-Whitney U (two-sided): non-parametric, appropriate for non-normal win rates
- Rank-biserial correlation r: effect size for Mann-Whitney
- Cohen's d: parametric effect size for comparison with prior literature
- 95% CI on convergence curves via t-distribution

If results are not significant (p > 0.05), that is itself informative: it means the problem
decomposition into tiers doesn't clearly help or hurt. Report it honestly.

---

## 8. Post-Training Evaluation

After the main experiment, run a more thorough evaluation of the best evolved weights.
This separates the *learning story* (training, depth=1) from the *gameplay story* (evaluation,
depth=3):

```bash
python3 evaluate_posttrain.py \
    --mono results/monolithic/ \
    --cc results/cc/ \
    --games 100 \
    --depth 3 \
    --workers 16 \
    --out results/posttrain/
```

**Four matchups evaluated per agent:**

| Matchup              | What it tells you                                           |
|----------------------|-------------------------------------------------------------|
| vs GreedyValue       | Sanity check / easy baseline                                |
| vs AlphaBeta-D2      | Generalization: was training opponent easy to overfit?      |
| vs Jackman(depth=3)  | Key comparison: do evolved values beat expert values?       |
| vs AlphaBeta-D3      | Transfer: do values hold up against a deeper opponent?      |

**Head-to-head:** CC mean weights vs Mono mean weights, both at depth=3. This is the cleanest
direct comparison of the two conditions.

Results are saved to `results/posttrain/posttrain_eval.json`.

---

## 9. Open Questions and Next Steps

These are the questions this experiment is designed to begin answering. Each bullet is a
potential future experiment or analysis.

### 9.1 Immediate (this experiment)
- [ ] Does CC-CMA-ES achieve higher win rates than monolithic after 30 runs?
- [ ] Does CC converge in fewer generations (faster learning)?
- [ ] Are CC weights more consistent across runs (lower variance)?
- [ ] Do evolved weights beat Jackman (1997) in head-to-head at depth=3?

### 9.2 Design Variations to Explore
- **Opponent curriculum:** Train first against greedy, then against alphabeta-D2. Does this
  warm-start lead to better final values?
- **Deeper training:** Use alphabeta-D3 as training opponent (much slower per eval). Do values
  improve qualitatively?
- **Alternative CC decomposition:** Instead of splitting by tier, split by piece role
  (attackers vs defenders vs support). Does this grouping make more sense strategically?
- **Larger population size:** CMA-ES default λ for 14D is ~9. Try λ=20 or 30 to explore
  more of the landscape per generation at the cost of more evaluations.
- **Noisy CMA-ES:** Use CMA-ES's built-in noise handling (`CMAEvolutionStrategy` with
  `noise_handler`) instead of manually setting loose tolerances.

### 9.3 Extensions
- **Neuroevolution:** Instead of a linear piece-value function, evolve a small neural network
  that takes board features as input and outputs a position score. Does a richer function class
  do better?
- **Self-play:** Instead of a fixed opponent, evolve against the current best agent (like SELFPL
  in AlphaZero). This avoids opponent-specific overfitting entirely.
- **Cross-game transfer:** Do piece values learned in Dragonchess transfer to a modified
  variant (e.g., different starting positions, one tier removed)?

---

## 10. Design Decisions Log

This section explains *why* things are the way they are. Before changing anything, read the
relevant entry here.

### Why start weights at zero (not Jackman values)?

Starting from zero means the convergence curves show true learning from scratch. If we
initialized at Jackman values, early generations would already be near-optimal and the curve
would look flat — making it hard to see whether CC converges faster. Zero initialization
maximizes the visible learning signal.

**Tradeoff:** Starting from zero adds noise in early generations since random weight
perturbations from σ=3 will often produce reasonable-looking weights by chance. If you want
to measure *refinement* rather than *discovery*, re-run with Jackman initialization.

### Why depth=1 for the evolved agent during training?

At depth=1, the agent's move selection reduces to: pick the move that maximizes
(captured_piece_value - lost_piece_value). The piece values literally determine every decision.
This isolates the learning signal for piece values from the confound of search quality.

At depth=3, a well-searched agent with any reasonable values can beat a greedy opponent.
Training at depth=3 would give the piece-value signal much less leverage over the outcome.

### Why alphabeta depth=2 as the training opponent (in the full experiment)?

GreedyValue is too weak — any non-negative piece values beat it easily, giving no gradient
signal to distinguish good from mediocre values. AlphaBeta-D2 thinks two moves ahead, which
is challenging enough to require the agent to learn approximately correct relative ordering of
piece values.

We did not use D3 because each evaluation is ~10× slower, making 200-game evaluations
prohibitively expensive at scale.

### Why tolx=tolfun=1e-11 (very loose CMA-ES tolerances)?

With 10 games/eval (preliminary) and even 200 games/eval (full experiment), the fitness
landscape is noisy — running the same weight vector twice gives different win rates. CMA-ES's
default convergence criterion (tolx=1e-12, tolfun=1e-11) can trigger early stopping when
the population collapses to a region where all candidates get similar noisy fitness values,
even if the true optimum is far away.

We set both tolerances to 1e-11 to relax the convergence criterion. If you see runs
consistently stopping well before 300 generations, investigate whether CMA-ES is hitting
another stop criterion (check `es.stop()` return value — it's a dict of all triggered
conditions).

### Why the best-collaborator strategy (not random collaborators)?

In CC, each tier candidate could be paired with (a) the current best from other tiers
("best-collaborator") or (b) random samples from the other tier populations ("random
collaborator"). Best-collaborator is simpler, more stable, and standard in the CC literature
for small populations. Random collaborator can be better when populations are diverse, but
with small populations like CMA-ES's (~6–9 individuals), random sampling often picks poor
collaborators that swamp the fitness signal.

### Why NOT use multiprocessing for runs (instead of ThreadPoolExecutor)?

Each training run launches subprocess calls to the C++ binary (`dragonchess`). Subprocess
calls release the GIL, so ThreadPoolExecutor works well here despite Python's GIL limitation.
We deliberately avoided multiprocessing to keep the code simple and compatible with Ray
(which handles distributed execution at the run level, while ThreadPoolExecutor handles
parallelism within a run).

### Why does the CC fitness log record the composite best, not per-tier bests?

The composite win rate (all three tier-bests assembled together) is the quantity that
matters scientifically. Per-tier fitness would be meaningless in isolation because a tier's
fitness only makes sense relative to its collaborators. Logging the composite also makes
the convergence curve directly comparable to the monolithic curve.

### On the preliminary run parameters (10 games, greedyvalue)

The preliminary runs were intentionally run with weak settings to validate the pipeline
quickly on local hardware. They confirm: (1) the binary interface works, (2) CMA-ES finds
improving solutions, (3) JSON output is correct. They are NOT scientific results and should
not be included in any paper. The results/monolithic/ and results/cc/ directories should be
cleared and repopulated with full-experiment runs on the cluster.
