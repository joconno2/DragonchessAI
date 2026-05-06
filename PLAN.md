# DragonchessAI Research Plan

## Status (2026-05-06)

**Parked for summer.** CoG Short (May 14) dropped. Target: Springer chapter revision (ICAART post-publication), summer 2026.

The summer goal is a 30%+ improvement over the ICAART 2026 baseline. The adversarial TD approach already showed 42% WR vs AB(d=2) at peak (31.5% on 200-game eval). Stabilizing that is the path.

---

## What Exists

### Engine
- Full C++ Dragonchess engine with headless CLI, AB search (d=1 through d=6+), MCTS, tournament mode
- Repo: `joconno2/DragonchessAI` (engine), `joconno2/DragonchessAI-Research` (training)
- Build: `cmake -DHEADLESS_ONLY=ON`, must build on Ubuntu 22.04 for cluster GLIBC compat

### Baselines (200 games each, fixed depth)

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

### Datasets
- d=4: 507K positions, 5 min to generate (1590 pos/s)
- d=6: 1M positions, 13 hours (21 pos/s, ~8% actor error rate)
- Format: compact binary (sparse features + AB score label)
- Generated via Ray cluster with `generate_dataset.py`

### Trained Checkpoints
- **Supervised d=4 (100 epochs):** RMSE 1.85 cp, 0.997 correlation. 0% WR. Passive play.
- **Supervised d=6 (30 epochs):** Similar convergence. 0% WR.
- **Adversarial TD v5:** Peak 42% WR vs AB(d=2) at batch 70 (50-game eval). Collapsed to 15%. 200-game eval: 31.5%.
- All on mega_knight at `/home/mega_knight/DragonchessAI/results/`

### Infrastructure
- HTCondor cluster: 44 machines, 540 CPUs. Container image `aall/dragonchess` in local registry.
- Condor AlphaZero driver + worker scripts in `~/research/condor/examples/`
- Usage logger + dashboard running on mega_knight (systemd, port 8401)

---

## What Didn't Work

### Supervised training (0% WR at any depth)
NN predicts AB scores with 0.997 correlation but plays passively. Knows position values but won't pursue captures. Games go 300+ moves and draw. Deeper labels (d=6) didn't help. The NN approximates the handcrafted eval but can't exceed it because the labels come from that eval.

### Adversarial TD (collapses after peak)
TDLeaf against AB(d=2) peaks at 42% WR then oscillates down. The 16.6M param v2 network is too sensitive to online gradient updates. Same collapse pattern with the old v1 linear model (peaked 67%, collapsed due to AdaGrad accumulation). The approach works briefly but is unstable.

### AlphaZero MCTS (0% WR at all sim counts)

Two runs attempted:
1. **Ray cluster (May 3-4):** 200 sims, 200 iters, 21.5 hours. 0% WR. Policy loss converged to near-zero (memorized garbage MCTS policy from insufficient sims).
2. **Condor cluster (May 6):** 800 sims, 13 iters, ~6 hours. 0% WR at iter 10 eval.

**MCTS strength test (May 6):** Tested 200/800/3200 sims with trained and random weights vs AB(d=1) and AB(d=2). 20 games per condition. **0/240 games won across all conditions.** MCTS cannot play this game at any tested sim count.

Root cause: Dragonchess has 108 avg legal moves (vs chess 35). MCTS with a flat value function degenerates to breadth-first random search. AB(d=1) with material counting beats that trivially. The chicken-and-egg problem (need good values to play good games, need good games to learn values) doesn't bootstrap at feasible sim counts.

### Bugs found
1. ClippedReLU vs ReLU mismatch between Python training and C++ inference
2. C++ `score *= 50.0f` leftover compressing NN output near zero
3. Runtime_env packaging included 600MB data/ directory

---

## Summer Plan: Stabilize Adversarial Training

The adversarial TD approach is the only one that showed real playing strength (42% peak). The problem is instability, not capability. Plan:

### Phase 1: Stabilize TD training
- **Periodic optimizer resets.** Train for N batches, snapshot best WR, reset AdamW state, resume from best weights. Prevents gradient accumulator drift.
- **Smaller learning rate schedule.** Cosine anneal from 0.001 to 0.0001 over each reset cycle.
- **Population-based approach.** Maintain 4-8 training runs in parallel (on Condor). Select and mutate the best. This is ES-adjacent and fits the AALL thesis.
- **Smaller network.** The 16.6M param network may be overkill. Test 32284->256->32->1 (8.3M params) and 32284->128->16->1 (4.2M params).

### Phase 2: ES fine-tuning of supervised checkpoint
- Start from the d=6 supervised checkpoint (0.997 correlation, 0% WR)
- CMA-ES or OpenES on the last layer weights only (64x1 = 64 params for the small net)
- Fitness: WR vs AB(d=2) over 50 games
- Run on Condor cluster (each eval = 50 tournament games in container)
- This is a direct application of AALL methods (ES for game AI) on a novel domain

### Phase 3: Evaluation
- 200-game tournaments vs AB(d=1,2,3) at fixed depth
- Time-controlled tournaments (500ms/move) for fair comparison
- Wilson confidence intervals on all WR numbers

### Target result
- 50%+ WR vs AB(d=2) (up from ICAART baseline)
- Stable (doesn't collapse over continued training)
- Enough for the Springer chapter revision

---

## Writing Needed (Springer Chapter)

The ICAART 2026 paper was selected for Springer post-publication. The chapter needs to extend the conference paper with new results. Here's what to write:

### New sections to add
1. **Extended evaluation methodology.** The baseline table (8 players x 3 depths x 200 games) that didn't exist in the ICAART paper. Time-controlled tournaments.
2. **Neural network evaluation function.** Architecture (v2, 32K sparse features, king-bucket), supervised training pipeline, the eval-vs-play gap analysis (0.997 correlation but 0% WR).
3. **Adversarial fine-tuning.** TDLeaf training, peak results, collapse analysis, stabilization approach.
4. **AlphaZero investigation.** MCTS implementation, the branching factor problem (108 avg legal moves), the strength test showing 0/240 at all sim counts. This is important context for why classical search still dominates in high-branching variants.
5. **ES-based training** (Phase 2 results, when complete). CMA-ES on the supervised checkpoint. Direct application of EC to the eval function.

### Sections to revise from ICAART paper
- Feature representation: update to v2 architecture (32K features vs original)
- Training: replace old TD(lambda) linear model with the full NN pipeline
- Results: comprehensive baselines replace the limited original comparison

### Figures needed
- Convergence plots (supervised: loss + correlation over epochs; adversarial: WR over batches showing peak and collapse)
- Baseline WR heatmap (players x depths)
- Architecture diagram (sparse features -> trunk -> eval head)
- MCTS strength test plot (sims vs WR, flat line at 0%)
- ES training curve (when complete)

### Tables needed
- Full baseline matrix (already have data)
- Supervised training metrics (have data)
- MCTS strength test (have data)
- Architecture comparison (params, inference time, WR)
- Time-controlled results (have data from timed baselines run)

### Data that exists and is ready
- All baseline tournaments (200 games each): `run_baselines.py` output
- Supervised training logs: `results/nn_v2_epochs_d4/`, `results/nn_v2_epochs_d6/`
- Adversarial training logs: `results/nn_adversarial_v5/`
- AlphaZero logs: `results/alphazero/`, `results/az_condor_v1/`
- MCTS strength test: ran May 6, need to save output formally
- Timed baselines: `results/timed_baselines/`

### Data that needs to be generated (summer)
- ES fine-tuning results (Phase 2)
- Stabilized adversarial results (Phase 1)
- Possibly: smaller network variants

---

## Architecture Notes

### Board: 3 x 8 x 12 = 288 squares
- Sky (layer 0): rows 0-7, cols 0-11, indices 0-95
- Ground (layer 1): indices 96-191
- Cavern (layer 2): indices 192-287
- Index = layer * 96 + row * 12 + col

### Pieces: signed int16, positive = Gold, negative = Scarlet
Sylph(1), Griffin(2), Dragon(3), Oliphant(4), Unicorn(5), Hero(6), Thief(7), Cleric(8), Mage(9), King(10), Paladin(11), Warrior(12), Basilisk(13), Elemental(14), Dwarf(15)

### Move encoding: from_sq * 288 + to_sq = 82,944 actions
Average legal moves per position: 108 (range 67-140)

### v2 Network: 32284 -> 512 -> 64 -> 1 (16.6M params)
- Input: king-bucket sparse features (32284 dims)
- ReLU activations throughout
- AlphaZero variant: 32284 -> 256 -> 128 -> {policy(82944), value(1)} (19M params)

### Build
```bash
# Headless (training/experiments)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DHEADLESS_ONLY=ON
cmake --build build --parallel $(nproc)
```
Must build on Ubuntu 22.04 for GLIBC 2.35 compatibility with cluster workers.

### Machines
- mega_knight (136.244.224.136): Condor CM, GTX Titan X x2, build tools
- Condor cluster: 44 machines, 540 CPUs, container universe
- Threadripper (136.244.224.45): 64 cores, RTX 5090
- Vertex (136.244.224.117): 8 cores, RTX 5090
