# DragonchessAI — TD(λ) Research Plan

## Status (as of 2026-03-31)

CMA-ES piece value experiments (monolithic + CC) are **complete and archived** in
`results/monolithic/` and `results/cc/`.  They showed a ~30% win rate ceiling against
AlphaBeta(depth=2) with high variance and no meaningful difference between conditions.
Root cause: 14-dimensional piece value search space is too flat (Jackman weights already
near-optimal for material eval); fixed opponent causes overfitting; 200 games/eval too
noisy for CMA-ES updates.

**Active direction: TD(λ) self-play at depth-2 with mixed-opponent training.**

### Critical Bug Fixed (2026-03-31): selfplay depth-2 was silently depth-1

`headless_main.cpp` had a bug where `--td-depth 2` in selfplay mode was silently reset
to depth 1. The code checked `if (gold_config.depth == 2) gold_config.depth = 1` to set
a "default", but couldn't distinguish between the struct's default value (2) and an
explicit `--td-depth 2`. **All nightly runs and sweeps were running depth-1 self-play.**

Evaluations (tournament mode) were NOT affected — they used `--gold-depth` which has no
such override. So sweep results like 65.5% vs AB2 came from depth-1 training + depth-2
evaluation. The high win rates were from the search depth at eval time compensating for
shallow training data.

**Fix:** track `td_depth_set` flag; only default to depth-1 if `--td-depth` was not
explicitly passed. Verified: depth-2 selfplay now ~11x slower than depth-1 (was 2x = same speed).

### Nightly v1 results (Mar 25–31): draw-collapse confirmed

7 nightly runs completed (266K batches, 13.3M games, 48.2h), all at depth-1 (due to bug).
Peak WR: 31% at batch 120, collapsed to 4.5% by batch 266K. Training in a draw equilibrium
with near-zero TD error signal. **All compute after batch ~500 was wasted.**

### Sweep Results (complete as of 2026-03-17)

**Note:** All selfplay was actually depth-1 due to the bug above. The depth-2 evaluation
explains the high WR differences between intended "depth-1" and "depth-2" configs.

**Sweep 1 (depth-1 vs depth-2 eval):**

| Config | λ | lr | eval depth | Best WR vs AB2 | Notes |
|--------|---|----|-----------|----------------|-------|
| `deep` | 0.7 | 0.05 | 2 | **70.0%** | d2 eval + lucky seed |
| `baseline` | 0.7 | 0.05 | 1 | 11.5% | d1 eval |
| all others | — | — | 1 | 7–12% | d1 eval |

**Sweep 2 (depth-2 eval hyperparam, 11h per config):**

| Config | λ | lr | Best WR vs AB2 | Notes |
|--------|---|----|----------------|-------|
| `d2_baseline` | 0.7 | 0.05 | **65.5%** | Winner |
| `d2_lambda_high` | 0.9 | 0.05 | 61.5% | |
| `d2_lambda_low` | 0.3 | 0.05 | 58.5% | |
| `d2_lr_high` | 0.7 | 0.15 | 37.0% | Unstable |
| `d2_warm` | 0.7 | 0.05 | 20.5% | Warm-start hurts |

**Key findings (revised):**
- λ=0.7, lr=0.05 is the best config
- Warm-starting from Jackman weights hurts
- All runs show draw-collapse (best WR early, degrades to ~10%)
- The depth-2 vs depth-1 gap in sweep 1 was about eval depth, not training depth

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
# Self-play (both sides same weights)
./build/dragonchess --headless --mode selfplay \
    --td-weights w0,w1,...,w39 --games 50 --threads 8 --td-depth 2

# Asymmetric (mixed-opponent training)
./build/dragonchess --headless --mode selfplay \
    --gold-td-weights w0,...,w39 --scarlet-td-weights v0,...,v39 \
    --games 50 --threads 8 --td-depth 2
```

### Python training stack
| File | Purpose |
|------|---------|
| `train_td.py` | TD(λ) trainer: AdaGrad, mixed-opponent, best-checkpoint, early-stop, Ray-compatible |
| `run_nightly.sh` | Nightly training runner (systemd timer, 4:30AM–11AM) |
| `run_td_experiments.sh` | Local sweep: 6 configs in parallel |
| `run_td_depth2_sweep.sh` | Depth-2 hyperparam sweep |
| `notify_training.py` | Discord notification after nightly runs |

**Trainer CLI (v2 — mixed-opponent + early-stop):**
```bash
python train_td.py \
    --out results/td/myrun/ \
    --lambda 0.7 --lr 0.05 --td-depth 2 \
    --mixed-frac 0.3 --snapshot-every 50 --snapshot-keep 10 \
    --early-stop-patience 500 --early-stop-threshold 0.15
```

**New features (2026-03-31):**
- `--mixed-frac 0.3` — 30% of batches use a frozen snapshot as Scarlet opponent
- `--snapshot-every 50` — save weight snapshot every 50 batches to opponent pool
- `--snapshot-keep 10` — keep last 10 snapshots in pool
- `--early-stop-patience 500` — stop if no WR improvement for 500 batches
- `--early-stop-threshold 0.15` — trigger threshold: best_wr - recent_avg_wr
- `best.json` — always preserved (never overwritten unless new best WR)

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

**Weight analysis (from 266K batch collapsed run):**
- Features 0-13 (material diffs) and 30 (total piece diff) are the only active weights (|w| > 0.1)
- Features 17-23 (positional) and 29, 31-32 (king/center) are effectively dead
- High AdaGrad G values on features 17-18, 33-36 (>1M) have frozen their learning rates
- Many material weights are NEGATIVE (pathological — artifact of draw-collapse)

---

## Nightly Automation

**Timer:** `dragonchess-nightly.timer` — 4:30 AM daily, Persistent=true
**Service:** `dragonchess-nightly.service` — wraps `run_nightly.sh` with systemd-inhibit
**Output:** `results/td/best_nightly_v2/` (fresh start with mixed-opponent)
**Logs:** `logs/nightly/YYYYMMDD_HHMM.log`
**Discord:** Auto-notification via `notify_training.py` after each run

---

## Immediate Next Steps

1. **Tonight's run (Apr 1, 4:30 AM):** First real depth-2 self-play with mixed-opponent training.
   Output: `results/td/best_nightly_v2/`. Should see depth-2 batches at ~1s each (vs ~80ms before).
   Expect ~23K batches per night at depth-2 (vs ~25K at depth-1, but with much better signal).
2. **Monitor draw-collapse resistance** — watch win rate trajectory over first 3-5 nights.
   Mixed-opponent (30% frozen snapshots) should prevent the symmetric equilibrium.
3. **Compare true depth-2 vs depth-1 training** — now that the bug is fixed, we can finally
   answer whether depth-2 selfplay actually produces better signal than depth-1.
4. **Inspect best.json after first night** — compare weights to the old collapsed run.
5. **Write `evaluate_td_posttrain.py`** — benchmark best checkpoint vs GreedyValue, AB2, AB3.
6. **Feature normalization** — if positional features remain dead after mixed-opponent,
   normalize feature scales (material ÷5, level material ÷50) in `td_features.cpp`.

### If mixed-opponent training works:
- Increase `--games-per-batch 200` for lower-variance gradients
- Try AB(depth=3) as evaluation target
- Plan Ray cluster run for production training

### If mixed-opponent training also collapses:
- Switch to Adam optimizer (AdaGrad's monotonic G accumulation may be part of the problem)
- Feature normalization (critical if AdaGrad G values diverge again)
- Neural network eval function (MLP, use RTX 4070 SUPER)

---

## What to Look For in Results

Each checkpoint's `win_rate_history` contains periodic evaluations vs AlphaBeta(depth=2).
Key signals:

1. **Is anything learning?**  Win rate should rise from ~0.3-0.5 baseline toward >0.6
   within ~100 batches.  If not, training signal is too weak.

2. **Draw-collapse resistance:**  With mixed-opponent, WR should NOT degrade monotonically
   after peaking. Sustained WR >40% over 1000+ batches = success.

3. **Depth-2 vs old depth-1:**  Compare first-night best WR to sweep 2's 65.5%.
   If true depth-2 selfplay produces better training signal, we should exceed this.

4. **Learning rate stability:**  If `lr_high` converges faster without instability, raise lr.

5. **Failure mode — flat RMSE:**  If RMSE stays high (~0.4+), features aren't providing signal.

6. **Early stopping:**  If triggered, the collapse pattern is still present — try different
   mixed-frac or opponent strategy.

---

## Build Notes

### Two build targets

| Mode | Command | Entry point | Use for |
|------|---------|-------------|---------|
| **GUI** (default) | `cmake -S . -B build` | `src/main.cpp` | Playing the game, demos |
| **Headless / Research** | `cmake -S . -B build -DHEADLESS_ONLY=ON` | `src/headless_main.cpp` | Training, experiments, CI |

**Critical:** `--td-weights`, `--mode selfplay`, and all TD training options are only
available in the headless build. Always use `HEADLESS_ONLY=ON` for training runs.

```bash
# For training
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DHEADLESS_ONLY=ON
cmake --build build --parallel $(nproc)
```

### Repository layout

Two separate GitHub repos:
- **`joconno2/DragonchessAI`** (`origin`) — game/platform
- **`joconno2/DragonchessAI-Research`** (`research`) — training code, results, PLAN.md

---

## Architecture Notes

### Why TDEvalAI extends AlphaBetaAI
`AlphaBetaAI::alphabeta()` calls `evaluate_material(game_copy)` at leaf nodes.
`evaluate_material` is virtual.  TDEvalAI overrides it to compute
`dot(weights, extract_td_features(game))`.  This reuses all search infrastructure.

### Why features are Gold-positive (not current-player-relative)
Simpler to reason about: a single weight vector predicts Gold's game value.
AlphaBeta alternates maximizer/minimizer; the sign flip in `evaluate_material`
(`return (color == GOLD) ? score : -score`) makes Scarlet minimize the same function.

### Naming: td_features not features
`features.h` shadows the system `<features.h>` header.

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
  "best_win_rate": 0.655,
  "best_batch": 120,
  "win_rate_history": [{"batch": 20, "win_rate_vs_ab": 0.42, ...}, ...],
  "elapsed_seconds": 3600.0,
  "config": {"lambda": 0.7, "gamma": 1.0, "lr": 0.05, "td_depth": 2, "mixed_frac": 0.3, ...}
}
```
Separate `best.json` preserves the weights at the best win rate (never overwritten unless beaten).

---

## Open Questions

1. **Does true depth-2 selfplay produce better signal than depth-1?** Now testable
   with the bug fixed. First real comparison starting Apr 1 nightly run.

2. **Is 40 features enough?**  TD-Gammon used ~198 features. If learning plateaus early,
   expand to piece-square tables.

3. **Does mixed-opponent training prevent draw-collapse?** First test Apr 1.

4. **What is the ceiling against AlphaBeta(depth=3)?**  CMA-ES hit ~30% vs depth-2.
   TD(λ) should reach >60% vs depth-2 and >45% vs depth-3 if adequate.

5. **When to move to neural network eval?**  If TD(λ) with 40 linear features plateaus
   below 65% vs depth-2, the linear function class is the bottleneck.

6. **AdaGrad vs Adam:**  AdaGrad G values grow monotonically, eventually freezing learning
   rates near zero. Adam's moving averages decay, maintaining plasticity. If mixed-opponent
   doesn't fix the collapse, switching optimizers is next.
