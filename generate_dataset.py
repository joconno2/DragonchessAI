#!/usr/bin/env python3
"""
Offline dataset generator for DragonchessAI search-supervised training.

Runs AB(label_depth) on random-opening games across the Ray cluster,
collects labeled positions, and saves to a compact binary file.

Usage:
    # On mega_knight (or any machine with Ray tunnel):
    python generate_dataset.py \
        --out data/d6_1M.bin \
        --label-depth 6 \
        --target-positions 1000000 \
        --auto-cluster

Binary format (little-endian):
    Header:  b'DCDT' (4B), uint32 n_positions, uint32 n_features
    Per position:
        float32 score
        uint16  nnz
        nnz x (uint16 feat_index, float32 feat_value)
"""

import argparse
import json
import logging
import os
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

BINARY = Path(__file__).parent / "build" / "dragonchess"
N_INPUT = 32284

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("generate_dataset")

MAGIC = b'DCDT'


# ---------------------------------------------------------------------------
# Binary dataset I/O
# ---------------------------------------------------------------------------

def write_dataset(path, positions, n_features=N_INPUT):
    """Write positions to binary file.

    positions: list of (score, [(index, value), ...])
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    with open(tmp, 'wb') as f:
        f.write(MAGIC)
        f.write(struct.pack('<II', len(positions), n_features))
        for score, feats in positions:
            f.write(struct.pack('<f', score))
            f.write(struct.pack('<H', len(feats)))
            for idx, val in feats:
                f.write(struct.pack('<Hf', idx, val))
    os.replace(tmp, path)
    log.info("Wrote %d positions to %s (%.1f MB)",
             len(positions), path, path.stat().st_size / 1e6)


def read_dataset(path):
    """Read binary dataset. Returns (scores, feat_lists) arrays."""
    with open(path, 'rb') as f:
        magic = f.read(4)
        assert magic == MAGIC, f"Bad magic: {magic}"
        n_pos, n_feat = struct.unpack('<II', f.read(8))
        scores = np.empty(n_pos, dtype=np.float32)
        feat_lists = []
        for i in range(n_pos):
            scores[i] = struct.unpack('<f', f.read(4))[0]
            nnz = struct.unpack('<H', f.read(2))[0]
            feats = []
            for _ in range(nnz):
                idx, val = struct.unpack('<Hf', f.read(6))
                feats.append((idx, val))
            feat_lists.append(feats)
    return scores, feat_lists, n_feat


# ---------------------------------------------------------------------------
# NDJSON parsing (same format as C++ genlabels output)
# ---------------------------------------------------------------------------

def parse_ndjson(text):
    """Parse NDJSON genlabels output into list of (score, [(idx, val), ...])."""
    positions = []
    for line in text.strip().split('\n'):
        if not line:
            continue
        d = json.loads(line)
        feats = [(idx, val) for idx, val in zip(d['i'], d['v'])
                 if 0 <= idx < N_INPUT]
        positions.append((d['s'], feats))
    return positions


# ---------------------------------------------------------------------------
# Ray actor for distributed generation
# ---------------------------------------------------------------------------

def _make_actor():
    import ray

    @ray.remote
    class GenLabelsActor:
        def __init__(self, repo_dir):
            cwd = os.getcwd()
            cwd_binary = os.path.join(cwd, "build", "dragonchess")
            repo_binary = (str(Path(repo_dir).expanduser() / "build" / "dragonchess")
                           if repo_dir else None)
            if os.path.isfile(cwd_binary):
                os.chmod(cwd_binary, 0o755)
                self.binary = cwd_binary
            elif repo_binary and os.path.isfile(repo_binary):
                self.binary = repo_binary
            else:
                self.binary = cwd_binary
            self.host = __import__("socket").gethostname()

        def ready(self):
            return {"host": self.host, "binary": self.binary}

        def genlabels(self, n_games, label_depth, random_plies):
            cmd = [
                self.binary, "--headless",
                "--mode", "genlabels",
                "--games", str(n_games),
                "--threads", "1",
                "--label-depth", str(label_depth),
                "--random-plies", str(random_plies),
            ]
            try:
                result = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=1200.0)
                if result.returncode != 0:
                    return "ERR:" + result.stderr[:200]
                return result.stdout
            except Exception as e:
                return "ERR:" + str(e)

    return GenLabelsActor


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate labeled positions dataset for DragonchessAI")
    parser.add_argument("--out", required=True, type=Path,
                        help="Output binary file path")
    parser.add_argument("--label-depth", type=int, default=6)
    parser.add_argument("--random-plies", type=int, default=8)
    parser.add_argument("--target-positions", type=int, default=1_000_000,
                        help="Stop after collecting this many positions")
    parser.add_argument("--games-per-chunk", type=int, default=4,
                        help="Games per actor request")
    parser.add_argument("--batch-games", type=int, default=400,
                        help="Games to dispatch per collection round")
    parser.add_argument("--max-hours", type=float, default=24.0)

    # Ray
    parser.add_argument("--ray-address", default=None)
    parser.add_argument("--auto-cluster", action="store_true")
    parser.add_argument("--ray-repo-dir", default=None)
    parser.add_argument("--max-actors", type=int, default=None)

    # Append to existing dataset
    parser.add_argument("--append", action="store_true",
                        help="Append to existing dataset file")

    args = parser.parse_args()

    # Load existing positions if appending
    all_positions = []
    if args.append and args.out.exists():
        scores, feat_lists, _ = read_dataset(args.out)
        for i in range(len(scores)):
            all_positions.append((float(scores[i]), feat_lists[i]))
        log.info("Loaded %d existing positions from %s",
                 len(all_positions), args.out)

    # Connect to Ray
    import ray

    if args.auto_cluster:
        from aall_cluster import connect
        state, tunnel = connect(namespace="dragonchess-datagen",
                                verbose=True, log_to_driver=False)
        log.info("Auto-connected, head=%s", state.head_ip)
    elif args.ray_address:
        from cluster.runtime_sync import stage_runtime_working_dir
        repo_root = args.ray_repo_dir or str(Path(__file__).parent)
        staged = stage_runtime_working_dir(repo_root)
        log.info("Staged working_dir: %s", staged.path)
        ray.init(address=args.ray_address, namespace="dragonchess-datagen",
                 ignore_reinit_error=True, log_to_driver=False,
                 runtime_env={"working_dir": str(staged.path)})
    else:
        log.error("Need --auto-cluster or --ray-address")
        sys.exit(1)

    cluster_cpus = int(float(ray.cluster_resources().get("CPU", 0)))
    target_actors = min(cluster_cpus, args.max_actors or cluster_cpus)
    log.info("Ray cluster: %d CPUs, using %d actors", cluster_cpus, target_actors)

    actor_cls = _make_actor()
    pool = [actor_cls.options(num_cpus=1).remote(args.ray_repo_dir)
            for _ in range(target_actors)]

    # Check which actors are alive
    ready = ray.get([a.ready.remote() for a in pool])
    hosts = {}
    for r in ready:
        hosts[r["host"]] = hosts.get(r["host"], 0) + 1
    log.info("Pool ready: %d actors across %d hosts", len(pool), len(hosts))

    # Generation loop
    start = time.time()
    max_seconds = args.max_hours * 3600
    total_errors = 0
    round_num = 0

    while len(all_positions) < args.target_positions:
        if time.time() - start > max_seconds:
            log.info("Time limit reached.")
            break

        round_num += 1
        n_chunks = args.batch_games // args.games_per_chunk
        futures = [
            pool[i % len(pool)].genlabels.remote(
                args.games_per_chunk, args.label_depth, args.random_plies)
            for i in range(n_chunks)
        ]

        results = ray.get(futures)
        round_positions = 0
        round_errors = 0
        for ndjson in results:
            if ndjson.startswith("ERR:"):
                round_errors += 1
                continue
            try:
                positions = parse_ndjson(ndjson)
                all_positions.extend(positions)
                round_positions += len(positions)
            except Exception:
                round_errors += 1

        total_errors += round_errors
        elapsed = time.time() - start
        rate = len(all_positions) / elapsed if elapsed > 0 else 0
        pct = len(all_positions) / args.target_positions * 100

        log.info("Round %3d | +%d positions | total %d (%.1f%%) | "
                 "errors %d | %.0f pos/s | %.1f min elapsed",
                 round_num, round_positions, len(all_positions), pct,
                 round_errors, rate, elapsed / 60)

        # Periodic save every 10 rounds
        if round_num % 10 == 0:
            write_dataset(args.out, all_positions)

    # Final save
    write_dataset(args.out, all_positions)
    elapsed = time.time() - start
    log.info("Done. %d positions in %.1f min (%.0f pos/s). %d total errors.",
             len(all_positions), elapsed / 60,
             len(all_positions) / elapsed if elapsed > 0 else 0,
             total_errors)


if __name__ == "__main__":
    main()
