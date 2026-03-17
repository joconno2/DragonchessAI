# DragonchessAI — TD(λ) Research Plan

## Status (as of 2026-03-16)

CMA-ES piece value experiments (monolithic + CC) are **complete and archived** in
`results/monolithic/` and `results/cc/`.  They showed a ~30% win rate ceiling against
AlphaBeta(depth=2) with high variance and no meaningful difference between conditions.
Root cause: 14-dimensional piece value search space is too flat (Jackman weights already
near-optimal for material eval); fixed opponent causes overfitting; 200 games/eval too
noisy for CMA-ES updates.

**Active direction: TD(λ) self-play at depth-2, hyperparameter sweep in progress.**

### Sweep 1 results — depth-1 vs depth-2 (2026-03-16, ~30 min)

| Config | λ | lr | depth | Games | Best wr vs AB2 |
|--------|---|----|-------|-------|----------------|
| `deep` | 0.7 | 0.05 | 2 | 26,900 | **0.700** |
| `lambda_low` | 0.3 | 0.05 | 1 | 64,100 | 0.120 |
| `baseline` | 0.7 | 0.05 | 1 | 59,600 | 0.115 |
| `lr_high` | 0.7 | 0.15 | 1 | 188,000 | 0.075 |
| `warm` | 0.7 | 0.05 | 1 | 175,000 | 0.070 |
| `lambda_high` | 0.9 | 0.05 | 1 | 54,000 | 0.070 |

**Finding: depth-2 selfplay is the dominant factor.** `deep` hit 70% vs AB2 — well above
the 30% CMA-ES ceiling. All depth-1 configs clustered at 7-12%. λ and lr variations were
negligible by comparison. Warm-starting hurt (material features alone are not enough signal
at depth-1; Jackman init may be biasing away from positional learning).

**Sweep 2 in progress:** depth-2 sweep varying λ and lr to fine-tune. See
`run_td_depth2_sweep.sh`.

---

## What Was Built

### C++ engine additions
| File | Purpose |
|------|---------|
| `src/td_features.h/cpp` | 40-feature extractor (Gold-positive perspective) |
| `src/ai.h/cpp` | `TDEvalAI` — AlphaBeta with feature-weight eval, overrides `evaluate_material` |
| `src/headless.h/cpp` | `run_selfplay_game`, `run_selfplay_batch` — record feature vectors + outcomes as NDJSON |
| `src/headless_main.cpp` | `--mode selfplay` CLI; `--td-weights`, `--gold/scarlet-td-weights`, `--td-depth` |

**Selfplay CLI:**
```bash
./build/dragonchess --headless --mode selfplay \
    --td-weights w0,w1,...,w39 --games 50 --threads 8
# → NDJSON to stdout: {"o": <±1/0>, "p": [[f0..f39], ...]}
```

### Python training stack
| File | Purpose |
|------|---------|
| `train_td.py` | TD(λ) trainer: AdaGrad, atomic checkpoints, SIGINT-safe, Ray-compatible |
| `run_td_experiments.sh` | Local sweep: 6 configs in parallel, built-in monitor |

**Trainer CLI:**
```bash
python train_td.py --out results/td/myrun/        # local, runs until killed
python train_td.py --out results/td/myrun/ --ray  # Ray cluster
```
Resumes from `results/td/myrun/latest.json` automatically.  Safe to kill at any time.

---

## Feature Vector (40 features, Gold-positive)

```
[0-13]  Material count diff per piece type (gold - scarlet)
        Order: Sylph, Griffin, Dragon, Oliphant, Unicorn, Hero, Thief,
               Cleric, Mage, Paladin, Warrior, Basilisk, Elemental, Dwarf
[14-16] Level piece count diff (Sky, Ground, Cavern)
[17-19] Level material value diff (Sky, Ground, Cavern)
[20]    Gold mean piece row on Ground (advancement proxy)
[21]    Scarlet mean piece advancement on Ground (7 - mean_row)
[22]    Gold mean piece row in Cavern
[23]    Scarlet mean piece advancement in Cavern
[24]    Scarlet pieces within Chebyshev-2 of gold king  (king danger)
[25]    Gold pieces within Chebyshev-2 of scarlet king  (attack pressure)
[26]    Gold frozen piece count
[27]    Scarlet frozen piece count
[28]    Cross-level piece count diff (Dragon/Griffin/Paladin/Hero/Mage)
[29]    Ground center control diff (cols 3-8)
[30]    Total piece count diff, normalized by 26
[31]    Gold king row on Ground (0 if king not on Ground)
[32]    Scarlet king advancement on Ground
[33-34] Absolute Sky material (gold, scarlet)
[35-36] Absolute Cavern material (gold, scarlet)
[37]    Game progress: min(move_count / 200, 1.0)
[38-39] Dragon presence flags (gold, scarlet)
```

**Sign convention:** `evaluate_material` returns `dot(w, features)` for Gold,
`-dot(w, features)` for Scarlet.  AlphaBeta always maximizes from AI's own perspective.
Training targets: Gold win = +1, Scarlet win = −1, draw = 0.

---

## Sweep Scripts

### Sweep 1 — `run_td_experiments.sh` (COMPLETE)
Initial depth-1 vs depth-2 comparison. Finding: depth-2 dominates.

### Sweep 2 — `run_td_depth2_sweep.sh` (IN PROGRESS)
All configs at depth-2, varying λ and lr:

| Config | λ | lr | depth | init | What it tests |
|--------|---|----|-------|------|---------------|
| `d2_baseline` | 0.7 | 0.05 | 2 | random | reference (same as `deep`) |
| `d2_lambda_low` | 0.3 | 0.05 | 2 | random | short credit trace at depth-2 |
| `d2_lambda_high` | 0.9 | 0.05 | 2 | random | long credit trace at depth-2 |
| `d2_lr_high` | 0.7 | 0.15 | 2 | random | faster adaptation at depth-2 |
| `d2_warm` | 0.7 | 0.05 | 2 | Jackman | warm-start at depth-2 |

Results: `results/td/<name>/latest.json`
Logs: `logs/td/<name>.log`

```bash
./run_td_depth2_sweep.sh --hours 2.0
./run_td_depth2_sweep.sh --hours 2.0   # always resumes internally
```

---

## What to Look For in Results

Each checkpoint's `win_rate_history` contains periodic evaluations vs AlphaBeta(depth=2).
Key signals:

1. **Is anything learning?**  Win rate should rise from ~0.3-0.5 baseline toward >0.6
   within ~100 batches.  If not, training signal is too weak.

2. **Warm vs baseline:**  If `warm` converges much faster, the material features are
   load-bearing.  If similar, the non-material features are carrying the signal.

3. **Deep vs baseline:**  If `deep` wins significantly more, depth-2 self-play is worth
   the cost for production.  If similar, depth-1 is fine (much faster = more data).

4. **Lambda sensitivity:**  Large gap between `lambda_low` / `lambda_high` suggests
   credit assignment timing matters a lot.  Flat means λ is not a critical hyperparameter.

5. **Learning rate:**  If `lr_high` converges faster without instability, raise lr in
   the full run.  If it oscillates or diverges, keep 0.05.

6. **Failure mode — flat RMSE:**  If RMSE stays high (~0.4+) and win rate doesn't move,
   features aren't providing signal.  Likely cause: evaluation too noisy (increase
   `--games-per-batch`) or feature scale mismatch (add normalization).

7. **Failure mode — win rate plateau ~0.5:**  Agent is learning to draw, not win.
   Fix: add a small mixed-opponent component (current vs frozen snapshot, or vs GreedyValue).

---

## Immediate Next Steps

1. **Let sweep 2 finish** — pick best λ/lr at depth-2
2. **Long solo run** of best config: `--hours 4+` until win rate plateaus
3. **Inspect weights** — features with near-zero weight after training are dead; replace them
4. **Write `evaluate_td_posttrain.py`** — benchmark best checkpoint vs GreedyValue, AB2, AB3, CMA-ES best
5. **Ray cluster run** with best config if local results look promising

**If sweep 2 learning is flat:**
- Increase `--games-per-batch 200` to reduce gradient noise
- Normalize features (divide material diffs by 5, advancement by 7, etc.) in `src/td_features.cpp`
- Add mixed opponent (see "Robustness improvements" below)

### Robustness improvements (high priority)

**Mixed opponent training** to prevent draw-collapse.  In `train_td.py`, add:
- 70% games: current weights vs current weights (self-play)
- 30% games: current weights vs a frozen snapshot from the checkpoint pool

Implementation sketch in `TDTrainer.generate_batch`:
```python
# Save snapshot every K batches
if self.total_batches % 10 == 0:
    self.snapshots.append(self.weights.copy())
    self.snapshots = self.snapshots[-5:]  # keep last 5

# Mix in frozen opponent for 30% of games
if self.snapshots and random.random() < 0.3:
    opp_weights = random.choice(self.snapshots)
    # generate games with gold=current, scarlet=frozen
    # these games still contribute to gold's TD update
```

**Smarter checkpoint pruning:**  Currently keeps last 20 by timestamp.  Should also
keep the checkpoint with the best `win_rate_vs_ab` seen so far (never delete the best).
Add `keep_best=True` flag to `save_checkpoint`.

### Feature improvements (medium priority)

Current 40 features are a reasonable first pass.  After seeing what's learned:
- Inspect `latest.json["weights"]` — features with near-zero weights after training
  are uninformative; consider replacing them
- High-value additions:
  - **Mobility:** actual legal move count for current player (expensive but informative)
  - **Piece-square tables:** which squares each piece type prefers (major expansion)
  - **Attack maps:** squares controlled per level (requires move-gen calls)

### Full Ray experiment (after local experiments look good)

```bash
python train_td.py \
    --out results/td_full/ \
    --ray --ray-address auto \
    --ray-workers 32 --ray-games-per-worker 100 \
    --lambda 0.7 --lr <best_from_sweep> \
    --td-depth <best_from_sweep> \
    --eval-every 50 --eval-games 500 --eval-ab-depth 3 \
    --hours 12
```

Use `--eval-ab-depth 3` for the full run (harder opponent = better calibration).

### Post-training evaluation (mirrors existing `evaluate_posttrain.py`)

Once training is done, evaluate the best checkpoint against:
1. GreedyValue — sanity check, should win >90%
2. AlphaBeta(depth=2) — training opponent
3. AlphaBeta(depth=3) — transfer test (generalization beyond training opponent)
4. Evolved best weights from CMA-ES experiment — direct comparison

A new `evaluate_td_posttrain.py` script needs to be written (extend the existing
`evaluate_posttrain.py` to accept TDEvalAI weights and run the same matchup battery).

---

## Architecture Notes

### Why TDEvalAI extends AlphaBetaAI
`AlphaBetaAI::alphabeta()` calls `evaluate_material(game_copy)` at leaf nodes.
`evaluate_material` is virtual.  TDEvalAI overrides it to compute
`dot(weights, extract_td_features(game))`.  This reuses all search infrastructure
(transposition table, move ordering, α-β pruning) with zero duplication.

### Why features are Gold-positive (not current-player-relative)
Simpler to reason about: a single weight vector predicts Gold's game value.
AlphaBeta alternates maximizer/minimizer; the sign flip in `evaluate_material`
(`return (color == GOLD) ? score : -score`) makes Scarlet minimize the same function.
TD training records everything from Gold's perspective with Gold-positive outcomes.

### Naming: td_features not features
`features.h` shadows the system `<features.h>` header that defines `__GLIBC_PREREQ`,
causing cryptic preprocessor failures.  Always name domain headers with a project prefix.

### Checkpoint format (v2)
```json
{
  "version": 2,
  "timestamp_utc": "...",
  "weights": [40 floats],
  "adagrad_G": [40 floats],
  "n_features": 40,
  "total_games": 12500,
  "total_batches": 250,
  "win_rate_history": [{"batch": 20, "win_rate_vs_ab": 0.42, ...}, ...],
  "elapsed_seconds": 3600.0,
  "config": {"lambda": 0.7, "gamma": 1.0, "lr": 0.05, "td_depth": 1, ...}
}
```

---

## Open Questions

1. **How much does search depth during self-play matter?**  **ANSWERED: depth-2 is essential.**
   Depth-1 produces noisy gradients and tops out at ~12% vs AB2. Depth-2 reached 70% in the
   same time budget. The slower game rate is more than compensated by better signal quality.

2. **Is 40 features enough?**  TD-Gammon used ~198 features; KnightCap used hand-crafted
   positional features.  Our feature set is relatively small.  If learning plateaus early,
   expand to piece-square tables.

3. **Does warm-starting hurt long-run performance?**  Jackman weights might bias the
   optimizer away from counterintuitive but better solutions.

4. **What is the ceiling against AlphaBeta(depth=3)?**  This is the real benchmark.
   CMA-ES hit ~30% vs depth-2.  TD(λ) should reach >60% vs depth-2 and >45% vs depth-3
   if the feature set is adequate.

5. **When to move to neural network eval?**  If TD(λ) with 40 linear features plateaus
   below 65% vs depth-2, the linear function class is the bottleneck.  Next step would
   be a small MLP (2-3 layers, ~128 hidden units) trained the same way (AlphaZero-lite).
