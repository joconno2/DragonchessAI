# DragonchessAI Ray Cluster Guide

This guide covers the distributed experiment pipeline used by `run_ray.py`.

## Overview

The cluster workflow has three layers:

1. `setup/cluster_sync.py` copies the repository to the worker machines and can build a headless binary there.
2. `setup/setup_ray.py` starts or stops the Ray cluster and limits each node to the `core_usage` capacity declared in `workers.csv`.
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

On the head node and on each worker that will run Ray actors:

```bash
python3 -m pip install --user -r requirements-cluster.txt
```

## Sync And Build

From the local checkout:

```bash
python3 setup/cluster_sync.py \
  --workers-file workers.csv \
  --repo-dir ~/DragonchessAI \
  --install-python-deps
```

That command:

- copies the repo to every enabled worker
- optionally installs the Python packages from `requirements-cluster.txt`
- builds the project with `-DHEADLESS_ONLY=ON`

Useful flags:

- `--skip-build`: only sync files
- `--clean-build`: delete the remote `build/` directory before rebuilding
- `--include-disabled`: sync every row that has a reachable host, even if `core_usage` is zero

## Start The Ray Cluster

Pick one worker as the Ray head:

```bash
python3 setup/setup_ray.py \
  --workers-file workers.csv \
  --head-hostname NL214-Lin11170 \
  --restart
```

`setup_ray.py` starts Ray with `--num-cpus <core_usage>` on each machine, so the cluster itself also respects your per-node CPU budget.

## Run The Experiment

Run this on the head node inside the synced repo:

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

1. Sync one worker and confirm `~/DragonchessAI/build/dragonchess --headless --help` works.
2. Start Ray and verify `python3 -c "import ray; ray.init(address='auto'); print(ray.cluster_resources())"`.
3. Run a tiny monolithic test:

```bash
python3 run_ray.py --runs 1 --games 10 --generations 2 --parallel-runs 1 --threads-per-eval 1
```

4. Confirm evaluator capacity matches the sum of `core_usage` across enabled rows.
