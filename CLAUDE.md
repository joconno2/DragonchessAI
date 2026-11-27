# Project: DragonchessAI

> **Agent rules: `~/.claude/CLAUDE.md` (global). Project state: `~/Documents/Vault/Projects/` STATE files.**
> Lab machines: `lab ls` / `lab ssh NAME` (see `~/Documents/Vault/Agent/TOOLS.md`).

## Project Overview

Dragonchess AI research project -- training evaluation functions for a 3-board chess variant
via TD(lambda) self-play. C++ game engine with Python training orchestration.

## Key Files

- `PLAN.md` -- detailed research plan, sweep results, architecture notes, next steps
- `train_td.py` -- TD(lambda) trainer (AdamW, mixed-opponent, draw penalty, depth mixing, atomic checkpoints, SIGINT-safe, Ray-compatible)
- `run_nightly.sh` -- nightly training runner (systemd timer, 3AM-11AM)
- `notify_training.py` -- sends training summary to Discord via research assistant bot
- `run_td_depth2_sweep.sh` -- depth-2 hyperparameter sweep (complete)
- `run_td_experiments.sh` -- depth-1 vs depth-2 sweep (complete)
- `src/td_features.h/cpp` -- 40-feature extractor (Gold-positive perspective)
- `src/ai.h/cpp` -- `TDEvalAI` (AlphaBeta with learned feature-weight eval)
- `src/headless.h/cpp` -- self-play game generation (NDJSON output)
- `src/headless_main.cpp` -- headless CLI entry point

## Build

Two build targets -- **do not confuse them**:

```bash
# Headless (for training/experiments) -- ALWAYS use this for TD work
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DHEADLESS_ONLY=ON
cmake --build build --parallel $(nproc)

# GUI (for playing/demos) -- separate build directory
cmake -S . -B build-gui -DCMAKE_BUILD_TYPE=Release
cmake --build build-gui --parallel $(nproc)
```

## Conventions

- Features are Gold-positive: `evaluate_material` returns `dot(w, features)` for Gold, negated for Scarlet
- Domain headers use `td_` prefix (not `features.h` -- shadows system header)
- Checkpoints are atomic (write-then-rename) and resumable
- Training results go in `results/td/<config_name>/`
- Logs go in `logs/td/` (sweep) or `logs/nightly/` (nightly runs)

## Current State (2026-04-03)

- CMA-ES experiments: complete, archived (30% ceiling vs AB2)
- TD(lambda) sweeps 1 & 2: complete. Best config: lambda=0.7, lr=0.05, depth=2 (65.5% vs AB2)
- v2 nightly (Apr 1-2): peaked at 67% vs AB2, collapsed due to AdaGrad G accumulation
- **v3 nightly (starting Apr 3):** AdamW optimizer, draw penalty (-0.3), playout cap randomization (25% deep), opponent pool seeding from disk. Output: `results/td/best_nightly_v3/`
- Checkpoint format: v3 (adam_m, adam_v, adam_t). Backward-compatible with v2 (auto-migrates)
- Linear model is intentional -- part of research point (no neural network eval planned)

## Obsidian

When making significant progress or changes, update the Obsidian note at:
`~/Documents/Vault/Personal/Games/Development/DragonchessAI.md`

## Repos

- `origin` = `joconno2/DragonchessAI` (engine/platform)
- `research` = `joconno2/DragonchessAI-Research` (this repo)
- `./release.sh vX.Y.Z` publishes to both
