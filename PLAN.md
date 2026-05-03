# DragonchessAI Research Plan

## Status (2026-05-03)

### Running Now

**AlphaZero self-play training** on Ray cluster (200 actors, 268 CPUs).
- tmux `alphazero` on mega_knight
- 200 games/iteration, 200 MCTS sims/move, ~6.5 min/iteration
- Training on GPU (Titan X), self-play on cluster
- Target: 200 iterations

**Timed baselines** running on mega_knight.
- tmux `timed` on mega_knight
- 500ms per move, iterative deepening, 100 games per matchup
- AB(timed) vs AB(timed) as anchor, plus NN variants

### Completed Baselines (Fixed Depth, 200 games each)

| Player | vs AB(d=1) | vs AB(d=2) | vs AB(d=3) |
|--------|-----------|-----------|-----------|
| Random | 0.0% | 0.5% | - |
| GreedyValue | 47.5% | 30.0% | 19.5% |
| AB(d=1) | 47.5% | 34.5% | 20.5% |
| AB(d=2) | 66.5% | 34.5% | 29.0% |
| AB(d=3) | 78.5% | 65.0% | 54.0% |
| NN-Supervised(d=1) | 17.5% | 4.0% | 1.5% |
| NN-Supervised(d=2) | 73.5% | 20.0% | 15.5% |
| NN-Adversarial(d=1) | 40.0% | 12.5% | 10.0% |
| NN-Adversarial(d=2) | 59.5% | 31.5% | 22.5% |

**Key insight:** NN-Adversarial(d=2) gets 31.5% vs AB(d=2) on 200 games. The old 42% from 50-game evals was noisy. AB(d=3) gets 65% vs AB(d=2). That's the bar.

### What Was Built (May 2-3)

**Offline dataset generation pipeline:**
- `generate_dataset.py`: Ray-distributed AB label generation, compact binary format
- d=4 dataset: 507K positions in 5 minutes (1590 pos/s, 0 errors)
- d=6 dataset: 1M positions in 13 hours (21 pos/s, ~8% actor error rate)

**Epoch-based supervised training:**
- `train_epochs.py`: loads binary dataset, trains in epochs with shuffling, val split, cosine LR
- v2 architecture (32284 -> 512 -> 64 -> 1, 16.6M params) with ReLU (not ClippedReLU)
- d=4 100 epochs: RMSE 1.85 cp, 0.997 correlation with AB scores, but 0% WR vs AB(d=2)
- d=6 30+ epochs: similar convergence, no improvement in WR

**Adversarial TD fine-tuning:**
- Modified `train_nn.py` for v2 architecture with ReLU
- Loads supervised checkpoint, trains against AB(d=2) via TDLeaf
- Peak 42% WR vs AB(d=2) at batch 70 (50-game eval), collapsed to ~15%
- 200-game eval: 31.5% vs AB(d=2)

**AlphaZero pipeline:**
- `src/mcts.h/cpp`: MCTS with PUCT selection, Dirichlet noise, temperature scheduling
- `src/mcts.h`: DualHeadWeights (256x128 trunk + policy head + value head, 19M params)
- `train_alphazero.py`: full self-play loop with Ray cluster for game generation, GPU training
- Sparse policy loss (avoids materializing 82K-wide tensors)
- C++ `--mode mcts-selfplay` outputs NDJSON training data

**Time-controlled tournaments:**
- Iterative deepening with `--time-per-move` flag
- `choose_move_timed()` in AlphaBetaAI: searches d=1,2,3,... until time budget exhausted
- `run_timed_baselines.py`: fair comparison at equal thinking time

**Comprehensive baselines:**
- `run_baselines.py`: all players vs AB(d=1,2,3), 200 games each, Wilson CIs
- `run_timed_baselines.py`: time-controlled version

### Bugs Found and Fixed

1. **ClippedReLU vs ReLU mismatch:** Python trained with ClippedReLU (clamp [0,1]) but C++ used regular ReLU. Network couldn't output values > ~12, targets were up to 104. Fixed to ReLU everywhere.

2. **C++ `score *= 50.0f`:** Leftover from old normalized training. NN output was rescaled 50x in C++, compressing useful resolution near zero. Removed.

3. **Runtime_env packaging:** data/ directory (600MB) was included in Ray working_dir package. Added to EXCLUDE_DIRS.

### What Didn't Work

**Supervised training alone can't beat AB(d=2).** The NN predicts AB scores with 0.997 correlation and 1.85 cp RMSE, but plays passively. It knows the value of positions but won't pursue captures. The d=4 labels tell the NN "this position is slightly better" but not "capture the piece to GET to that position." Games go 300+ moves and draw.

**Deeper labels (d=6) didn't help.** Same passive play, similar convergence. The NN approximates the handcrafted eval's judgment but can't exceed it because the labels come from that eval.

**Adversarial fine-tuning collapses.** TDLeaf against AB(d=2) peaks at ~42% WR then oscillates down to 15%. The 16.6M parameter network is too sensitive to online gradient updates. Same collapse pattern as the old v1 architecture.

### Next Steps

1. **AlphaZero training** (running now). Self-play discovers strategies the handcrafted eval doesn't know. MCTS + policy/value network is the established approach for this class of game. Target: >50% WR vs AB(d=2) at equal time.

2. **Timed baseline table** (running now). Honest comparison at equal thinking time. AB iterative deepening reaches d=3-4 in 500ms. NN reaches d=1-2 in the same time. MCTS gets ~200 simulations.

3. **Scale MCTS simulations.** Current 200 sims is low. AlphaZero used 800 for chess. More sims = stronger play but slower iteration. Increase once the training loop is validated.

4. **Network architecture upgrade.** Current architecture is fully connected (sparse input). A convolutional network on the 3x8x12 board would capture spatial patterns better. Future work once the FC version shows learning.

### Paper Target

**CoG Short Paper (May 14) or AIIDE (Jun 26).** The paper needs a positive result: AlphaZero-style MCTS beating classical alpha-beta at equal time. The comprehensive baseline table provides the comparison framework. If AlphaZero doesn't work in time, this project doesn't have a paper yet.

---

## Architecture Notes

### Board: 3 x 8 x 12 = 288 squares
- Sky (layer 0): rows 0-7, cols 0-11, indices 0-95
- Ground (layer 1): indices 96-191
- Cavern (layer 2): indices 192-287
- Index = layer * 96 + row * 12 + col

### Pieces: signed int16, positive = Gold, negative = Scarlet
Sylph(1), Griffin(2), Dragon(3), Oliphant(4), Unicorn(5), Hero(6), Thief(7), Cleric(8), Mage(9), King(10), Paladin(11), Warrior(12), Basilisk(13), Elemental(14), Dwarf(15)

### Move: tuple<int, int, MoveFlag> = (from_sq, to_sq, flag)
MoveFlag: QUIET(0), CAPTURE(1), AFAR(2), AMBIGUOUS(3), THREED(4)

### Action encoding (for policy network): from_sq * 288 + to_sq = 82,944

### Build notes
- Headless binary must be built with `-DHEADLESS_ONLY=ON`
- Workers on the Ray cluster need GLIBC 2.35 compatible binary (build on grid or Ubuntu 22.04)
- mega_knight has cmake/g++ and can build directly

### Checkpoint formats
- **NN supervised:** JSON with `nn_weights` flat array (16,562,817 floats)
- **NN adversarial:** Same format, `type: "nn"`
- **AlphaZero:** Binary weights file (19M floats) + JSON metadata

### Machines
- mega_knight (136.244.224.136): Ray head, GTX Titan X x2, build tools
- Grid (136.244.224.30): SSH jump host only, 1 core
- Threadripper (136.244.224.45): 64 cores, RTX 5090
- Vertex (136.244.224.117): 8 cores, RTX 5090
- 12+ NLH214 lab machines: Ray workers (8-24 cores each)
