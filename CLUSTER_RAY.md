# DragonchessAI Ray Cluster Guide

This guide covers the distributed experiment pipeline used by `run_ray.py`.

## Overview

The cluster workflow has three layers:

1. `setup/cluster_sync.py` copies the repository to the worker machines and distributes a prebuilt headless binary.
2. `setup/setup_ray.py` starts or stops the Ray cluster and limits each worker node to the `core_usage` capacity declared in `workers.csv`.
3. `run_ray.py` launches multiple training runs on the head node while a shared pool of Ray evaluator actors executes tournaments across the cluster.

Each evaluator actor reserves `num_cpus=1` and runs `dragonchess` with `--threads 1`, so `core_usage` maps directly to the maximum number of concurrent tournament evaluations on a machine.

## workers.csv

The distributed tooling reads these columns from `workers.csv`:

- `hostname`
- `ip-address`
- `username`
- `password`
- `env`
- `core_usage`

Optional:

- `in_cluster`: if present and false-like, that row is skipped

Behavior:

- blank `env`: run commands directly
- blank `username`: defaults to `dgezgin`
- blank `password`: defaults to `geesearebigtoddlers1`
- `env` set to `mycondaenv`: activate that conda env before running commands
- `env` set to a path such as `~/venvs/dc`: source `~/venvs/dc/bin/activate`
- `core_usage <= 0`: no evaluator slots are created for that machine

## Install Python Dependencies

On your Mac or other head machine:

```bash
python3 -m pip install --user -r requirements-cluster.txt
```

On the workers, the only Python dependency required by the current runtime is `ray`. You can let the sync script install it automatically with `--install-python-deps`.

## Build And Sync

Build the headless binary locally first:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DHEADLESS_ONLY=ON
cmake --build build --parallel
```

Then sync the repo and the prebuilt binary to the workers:

```bash
python3 setup/cluster_sync.py \
  --workers-file workers.csv \
  --repo-dir ~/DragonchessAI \
  --binary-path build/dragonchess \
  --install-python-deps
```

That command:

- copies the repo to every enabled worker
- copies the prebuilt local `build/dragonchess` to `~/DragonchessAI/build/dragonchess` on each worker
- optionally installs `ray` on the workers
- does not run `cmake` on the workers

Useful flags:

- `--build-local`: have the script build the headless binary locally before syncing
- `--clean-build`: delete the local build directory before `--build-local`
- `--include-disabled`: sync every row that has a reachable host, even if `core_usage` is zero

Important:

- the distributed binary must be compatible with the worker OS and architecture
- if you build on a machine that does not match the workers, distribute a compatible prebuilt binary instead of using `--build-local`

## Start The Ray Cluster

Start the Ray head locally on your Mac and have the workers join it:

```bash
python3 setup/setup_ray.py \
  --workers-file workers.csv \
  --local-head \
  --head-ip YOUR_MAC_IP \
  --restart
```

`setup_ray.py` starts Ray on each worker with `--num-cpus <core_usage>`, so the worker-side cluster capacity respects your per-node CPU budget.

## Run The Experiment

Run this on your Mac inside the local repo:

```bash
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
  --out-cc results/cc/
```

Notes:

- `--parallel-runs` controls how many training coordinators are active on the head at once.
- Cluster throughput comes from the shared evaluator actors, not from giving each run a private block of CPUs.
- Results are written by the head-side driver, so a shared filesystem is not required.
- Workers do not build the project; they only host Ray and run tournament evaluations against the distributed binary.

## Post-Training Evaluation On Ray

`evaluate_posttrain.py` can reuse the same evaluator pool:

```bash
python3 evaluate_posttrain.py \
  --mono results/monolithic/ \
  --cc results/cc/ \
  --games 100 \
  --depth 3 \
  --ray-address auto \
  --workers-csv workers.csv \
  --repo-dir ~/DragonchessAI \
  --threads-per-eval 1 \
  --out results/posttrain/
```

## Smoke Tests

Recommended bring-up order:

1. Build `build/dragonchess` locally and sync one worker.
2. SSH to that worker and confirm `~/DragonchessAI/build/dragonchess --headless --help` works.
3. Start Ray from your Mac and verify `python3 -c "import ray; ray.init(address='auto'); print(ray.cluster_resources())"`.
4. Run a tiny monolithic test:

```bash
python3 run_ray.py --runs 1 --games 10 --generations 2 --parallel-runs 1 --threads-per-eval 1
```

5. Confirm evaluator capacity matches the sum of `core_usage` across enabled rows.
