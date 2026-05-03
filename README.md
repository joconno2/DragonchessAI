# DragonchessAI: Learning to Play Three-Dimensional Chess

![Board Example](assets/board_example.png)

## Overview

DragonchessAI is a research platform for training AI agents to play Gary Gygax's Dragonchess, a three-dimensional chess variant played across three 8x12 boards (Sky, Ground, Cavern) with 15 unique piece types per side. The project includes a high-performance C++ game engine, multiple AI architectures, and distributed training infrastructure.

**Current research focus:** comparing classical search (alpha-beta), neural network evaluation (NNUE-style supervised + adversarial), and AlphaZero-style MCTS self-play for Dragonchess position evaluation and move selection.

## Game Details

- **Board:** 3 levels (Sky, Ground, Cavern), each 8 rows x 12 columns = 288 total squares
- **Pieces:** 15 types per side (Sylph, Griffin, Dragon, Oliphant, Unicorn, Hero, Thief, Cleric, Mage, King, Paladin, Warrior, Basilisk, Elemental, Dwarf)
- **Mechanics:** Cross-level movement, freezing (Basilisk), ranged attacks (Dragon), and piece-specific abilities
- **Draw rules:** 100 moves without capture, 1000 move limit, stalemate
- **Branching factor:** ~50-100 legal moves per position (comparable to standard chess)

## Repository Structure

```
DragonchessAI/
├── src/                          # C++ engine and AI
│   ├── game.cpp/h               # Game state, rules, make/undo
│   ├── moves.cpp/h              # Legal move generation (15 piece types)
│   ├── bitboard.cpp/h           # Board representation, indexing
│   ├── ai.cpp/h                 # AI implementations (Random, Greedy, Minimax, AlphaBeta, NN)
│   ├── nn_eval.h                # NNUE-style neural network (sparse input, incremental accumulator)
│   ├── td_features.cpp/h        # Feature extraction (king-bucket PSQ + strategic features)
│   ├── mcts.cpp/h               # Monte Carlo Tree Search with policy+value network
│   ├── headless.cpp/h           # Headless game runner (tournaments, self-play, data gen)
│   ├── headless_main.cpp        # CLI entry point for all headless modes
│   ├── simple_ai.cpp/h          # Plugin base class for student bots
│   └── ai_plugin.cpp/h          # Dynamic library loader
│
├── train_nn.py                  # TDLeaf(lambda) adversarial trainer (NN vs AB)
├── train_nn_v2.py               # Search-supervised trainer (online, AB labels)
├── train_epochs.py              # Epoch-based supervised trainer (offline dataset)
├── train_alphazero.py           # AlphaZero self-play training loop (Ray cluster)
├── generate_dataset.py          # Offline labeled position generator (Ray cluster)
├── run_baselines.py             # Fixed-depth baseline tournament suite
├── run_timed_baselines.py       # Time-controlled baseline tournament suite
│
├── cluster/                     # Ray cluster utilities
│   ├── runtime_sync.py          # Working directory staging for Ray workers
│   ├── nn_pool.py               # Ray actor pool for distributed NN selfplay
│   ├── evaluator_pool.py        # Ray actor pool for distributed evaluation
│   └── worker_config.py         # Worker discovery and configuration
│
├── examples/                    # Educational plugin examples
│   ├── random_bot.cpp           # Random legal move
│   ├── material_bot.cpp         # Greedy material maximization
│   ├── tactical_bot.cpp         # Capture-aware tactics
│   ├── positional_bot.cpp       # Positional evaluation
│   └── student_bot.cpp          # Template for student work
│
├── results/                     # Experiment results (not in git)
├── data/                        # Generated training datasets (not in git)
├── assets/                      # Piece sprites, fonts
└── CMakeLists.txt               # Build configuration
```

## Building

### Headless (training/experiments)

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DHEADLESS_ONLY=ON
cmake --build build --parallel $(nproc)
```

### GUI (playing/demos)

Requires SDL2, SDL2_image, SDL2_ttf.

```bash
cmake -B build-gui -DCMAKE_BUILD_TYPE=Release
cmake --build build-gui --parallel $(nproc)
```

## AI Implementations

### Classical Search

| AI | Description | Typical Strength |
|----|-------------|-----------------|
| `random` | Uniform random legal moves | Floor |
| `greedyvalue` | 1-ply material maximization | Weak |
| `alphabeta` | Alpha-beta with handcrafted eval, TT, move ordering | Strong (scales with depth) |

Alpha-beta supports both fixed-depth (`--gold-depth N`) and time-controlled (`--time-per-move Nms`) iterative deepening.

### Neural Network Evaluation (NNUE-style)

**Architecture:** Sparse input (32,284 king-bucket piece-square + strategic features) -> 512 (ReLU) -> 64 (ReLU) -> 1. Total: 16.6M parameters. First layer is incrementally accumulated during search (fast).

**Feature encoding:**
- 8 king buckets (4 column groups x 2 row halves on ground board)
- 14 piece types x 288 squares per bucket = 32,256 piece-square features
- 28 strategic features (frozen pieces, king safety, pawn structure, material ratios, game phase)

**Training approaches tried:**
1. **Supervised (epoch-based):** Generate millions of labeled positions offline using AB(d=4 or d=6) with handcrafted eval. Train NN to predict search scores via MSE regression. Fast convergence (RMSE < 2 cp), but NN plays passively and can't beat AB(d=2).
2. **Adversarial TD:** Fine-tune supervised checkpoint via TDLeaf(lambda) playing against AB(d=2). Learns to seek captures. Peak 42% WR vs AB(d=2) at batch 70, then collapses to ~15% oscillation.

### AlphaZero (MCTS + Policy/Value Network)

**Architecture:** Sparse input (32,284) -> 256 (ReLU) -> 128 (ReLU) -> {policy: 82,944 actions, value: tanh scalar}. Total: 19M parameters.

**Action encoding:** from_square * 288 + to_square = 82,944 possible moves. Policy logits computed only for legal moves during MCTS (sparse, fast).

**Training loop:**
1. Self-play via MCTS (200 simulations/move) on Ray cluster (200+ actors)
2. Collect (position, MCTS_policy, game_outcome) training triples
3. Train policy head (cross-entropy with MCTS policy) + value head (MSE with outcome)
4. Repeat with updated weights

**MCTS details:**
- PUCT selection: Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
- Dirichlet noise at root for exploration (alpha=0.3, frac=0.25)
- Temperature: 1.0 for first 30 moves, near-zero after (exploit)
- Game cap: 500 moves

## Headless CLI Modes

```bash
# Tournament: two AIs play N games
./build/dragonchess --headless --mode tournament \
    --gold-ai alphabeta --gold-depth 3 \
    --scarlet-ai alphabeta --scarlet-depth 2 \
    --games 200 --threads 8 --output-json results.json

# Time-controlled tournament: both sides get equal time per move
./build/dragonchess --headless --mode tournament \
    --gold-ai alphabeta --scarlet-ai alphabeta \
    --time-per-move 500 --games 100 --threads 4

# NN evaluation: load binary weights
./build/dragonchess --headless --mode tournament \
    --gold-ai nneval --gold-depth 2 --gold-nn-weights weights.bin \
    --scarlet-ai alphabeta --scarlet-depth 2 \
    --games 200 --output-json results.json

# Generate labeled positions for supervised training
./build/dragonchess --headless --mode genlabels \
    --games 100 --label-depth 6 --random-plies 8 --threads 4

# MCTS self-play for AlphaZero training
./build/dragonchess --headless --mode mcts-selfplay \
    --games 50 --mcts-simulations 400 --mcts-nn-weights weights.bin

# Plugin tournament
./build/dragonchess --headless --mode tournament \
    --gold-ai-plugin student_bot.so \
    --scarlet-ai alphabeta --scarlet-depth 3 --games 1000
```

## Training Scripts

### Offline Dataset Generation

```bash
# Generate 1M positions labeled by AB(d=6), distributed on Ray cluster
python generate_dataset.py \
    --out data/d6_1M.bin \
    --label-depth 6 --random-plies 8 \
    --target-positions 1000000 \
    --ray-address auto --max-actors 200
```

### Epoch-based Supervised Training

```bash
# Train v2 architecture on pre-generated dataset
python train_epochs.py \
    --data data/d6_1M.bin \
    --out results/nn_v2_epochs/ \
    --lr 0.0003 --batch-size 16384 --epochs 100 \
    --eval-games 200 --eval-ab-depth 2
```

### Adversarial TD Training

```bash
# Fine-tune supervised checkpoint against AB(d=2)
python train_nn.py \
    --out results/nn_v2_adversarial/ \
    --lr 0.0001 --lambda 0.7 --td-depth 2 \
    --opponent ab --opponent-depth 2 --draw-penalty -0.3 \
    --games-per-batch 100 --eval-every 10 \
    --ray-address auto --max-actors 200
```

### AlphaZero Self-Play Training

```bash
# Full AlphaZero loop on Ray cluster
python train_alphazero.py \
    --out results/alphazero/ \
    --games-per-iter 200 --mcts-sims 200 \
    --epochs-per-iter 2 --batch-size 1024 --lr 0.001 \
    --iterations 200 --eval-every 5 \
    --ray-address auto --max-actors 200
```

### Baseline Evaluation

```bash
# Fixed-depth baselines (every player vs AB at d=1,2,3)
python run_baselines.py \
    --out results/baselines/ --games 200 --threads 4 \
    --nn-sup-weights results/nn_v2_epochs_d4/latest.json \
    --nn-adv-weights results/nn_v2_adversarial/best.json

# Time-controlled baselines (equal thinking time per move)
python run_timed_baselines.py \
    --out results/timed_baselines/ \
    --time-per-move 500 --games 100 --threads 4 \
    --nn-sup-weights results/nn_v2_epochs_d4/latest.json \
    --nn-adv-weights results/nn_v2_adversarial/best.json
```

## Baseline Results (Fixed Depth, 200 games each, Gold WR)

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

All matchups played as Gold. 95% Wilson confidence intervals are roughly +/-6-7% at 200 games. AB(d=2) vs AB(d=2) shows 34.5% Gold WR with 25.5% draw rate, indicating a slight Scarlet advantage at this depth.

## Distributed Training Infrastructure

Self-play and data generation run on a Ray cluster (268 CPUs across 16 machines). The C++ binary is packaged and distributed to workers automatically via Ray's runtime_env. Training runs on GPU (mega_knight, GTX Titan X).

Key infrastructure:
- `cluster/runtime_sync.py`: Stages the repo (excluding data/results) and binary for Ray workers
- `cluster/nn_pool.py`: Ray actor pool for distributed NN self-play and evaluation
- `generate_dataset.py`: Parallel labeled position generation using Ray actors

## Feature Vector Details

The 32,284-dimensional sparse feature vector encodes the board state relative to the friendly king's position:

**Piece-square features [0..32255]:** 8 king buckets x 14 piece types x 288 squares. Each bucket corresponds to a region of the ground board where the king might be (4 column groups x 2 row halves). For each bucket, binary features indicate where pieces of each type are located. Gold pieces = +1.0, Scarlet pieces = -1.0.

**Strategic features [32256..32283]:** Frozen piece counts, frozen material values, Basilisk proximity, king safety (Chebyshev distance rings), king home rank, material ratios, game phase (move count, no-capture counter, piece density), pawn structure (isolated/doubled/connected warriors), cross-level piece counts, material advantage.

## Plugin System

Students implement a single `choose_move()` method:

```cpp
#include "simple_ai.h"

class StudentBot : public SimpleAI {
public:
    using SimpleAI::SimpleAI;
    std::optional<Move> choose_move() override {
        auto moves = get_legal_moves();
        // Your strategy here
        return moves[0];
    }
};

extern "C" SimpleAI* create_ai(Game& game, Color color) {
    return new StudentBot(game, color);
}
```

Compile and run:
```bash
g++ -std=c++17 -fPIC -O3 -I../src -shared \
    ../src/simple_ai.cpp ../src/bitboard.cpp \
    ../src/moves.cpp ../src/game.cpp \
    student_bot.cpp -o student_bot.so

./dragonchess --headless --mode tournament \
    --gold-ai-plugin student_bot.so \
    --scarlet-ai alphabeta --scarlet-depth 2 --games 100
```

## References

Gygax, G. (1985). *Dragonchess*. TSR, Inc.

Silver, D., et al. (2018). A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play. *Science*, 362(6419), 1140-1144.

## License

This software is provided for educational and research purposes. Free to modify, extend, and redistribute with attribution.
