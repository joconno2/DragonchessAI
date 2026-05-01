#!/usr/bin/env python3
"""
Search-supervised NN trainer for Dragonchess (Stockfish NNUE approach).

Instead of TDLeaf self-play, this generates training data by running
AB(depth=N) with the handcrafted eval on diverse positions, then trains
the NN to predict the search score via supervised regression.

The NN learns to approximate deep search with a single forward pass.
No self-play collapse. No draw equilibrium problem.

Architecture: 4060 -> 256 (clipped ReLU) -> 32 (clipped ReLU) -> 1
"""

import argparse
import io
import json
import logging
import os
import signal
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BINARY = Path(__file__).parent / "build" / "dragonchess"
N_INPUT = 32284
N_H1 = 512
N_H2 = 64

CHECKPOINT_PREFIX = "ckpt_"
LATEST_NAME = "latest.json"
BEST_NAME = "best.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("train_nn_v2")


# ---------------------------------------------------------------------------
# NN model (same architecture as train_nn.py)
# ---------------------------------------------------------------------------

class ClippedReLU(nn.Module):
    def forward(self, x):
        return torch.clamp(x, 0.0, 1.0)


class NNEval(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_INPUT, N_H1),
            ClippedReLU(),
            nn.Linear(N_H1, N_H2),
            ClippedReLU(),
            nn.Linear(N_H2, 1),
        )

    def forward(self, x):
        return self.net(x)

    @staticmethod
    def total_params():
        m = NNEval()
        return sum(p.numel() for p in m.parameters())

    def to_flat_numpy(self):
        return np.concatenate([p.detach().cpu().numpy().ravel()
                               for p in self.parameters()])

    def from_flat_numpy(self, arr):
        offset = 0
        for p in self.parameters():
            n = p.numel()
            p.data.copy_(torch.from_numpy(
                arr[offset:offset + n].reshape(p.shape)))
            offset += n

    def init_kaiming(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def to_binary_weights(self):
        flat = self.to_flat_numpy().astype(np.float32)
        return flat.tobytes()


def _model_weights_bytes(model):
    return model.to_binary_weights()


# ---------------------------------------------------------------------------
# Data generation: call C++ binary in genlabels mode
# ---------------------------------------------------------------------------

def _parse_ndjson_labels(ndjson_text):
    """Parse NDJSON genlabels output into feature/score arrays."""
    features_list = []
    scores_list = []
    for line in ndjson_text.strip().split('\n'):
        if not line:
            continue
        d = json.loads(line)
        dense = np.zeros(N_INPUT, dtype=np.float32)
        for idx, val in zip(d['i'], d['v']):
            if 0 <= idx < N_INPUT:
                dense[idx] = val
        features_list.append(dense)
        scores_list.append(d['s'])
    return features_list, scores_list


def _run_genlabels_local(binary, n_games, label_depth, random_plies,
                         n_threads=1, timeout_s=600.0):
    """Run C++ genlabels locally."""
    cmd = [
        str(binary), "--headless",
        "--mode", "genlabels",
        "--games", str(n_games),
        "--threads", str(n_threads),
        "--label-depth", str(label_depth),
        "--random-plies", str(random_plies),
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return ""
    if result.returncode != 0:
        return ""
    return result.stdout


def generate_labels(binary, n_games, label_depth, random_plies,
                    n_threads=1, timeout_s=600.0, ray_pool=None,
                    chunk_size=4):
    """Generate labeled positions. Uses Ray pool if available."""
    if ray_pool is not None:
        import ray
        # Distribute across Ray actors
        chunks = []
        remaining = n_games
        while remaining > 0:
            c = min(chunk_size, remaining)
            chunks.append(c)
            remaining -= c
        futures = [
            ray_pool[i % len(ray_pool)].genlabels.remote(
                c, label_depth, random_plies)
            for i, c in enumerate(chunks)
        ]
        results = ray.get(futures)
        all_features = []
        all_scores = []
        for ndjson in results:
            if ndjson.startswith("ERR:"):
                log.warning("Actor error: %s", ndjson[:200])
                continue
            fl, sl = _parse_ndjson_labels(ndjson)
            all_features.extend(fl)
            all_scores.extend(sl)
    else:
        ndjson = _run_genlabels_local(
            binary, n_games, label_depth, random_plies, n_threads, timeout_s)
        all_features, all_scores = _parse_ndjson_labels(ndjson)

    if not all_features:
        return None, None

    features = torch.tensor(np.array(all_features), dtype=torch.float32)
    scores = torch.tensor(all_scores, dtype=torch.float32).unsqueeze(1)
    return features, scores


def _make_genlabels_actor():
    """Create a Ray actor class for distributed label generation."""
    import ray

    @ray.remote
    class GenLabelsActor:
        def __init__(self, repo_dir):
            import sys, os
            # Use working_dir (cwd set by Ray runtime_env) for the binary,
            # not the repo_dir which may not exist on this worker.
            cwd = os.getcwd()
            cwd_binary = os.path.join(cwd, "build", "dragonchess")
            repo_binary = str(Path(repo_dir).expanduser() /
                              "build" / "dragonchess") if repo_dir else None
            if os.path.isfile(cwd_binary):
                os.chmod(cwd_binary, 0o755)
                self.binary = cwd_binary
            elif repo_binary and os.path.isfile(repo_binary):
                self.binary = repo_binary
            else:
                self.binary = cwd_binary  # will fail with clear error
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
                    cmd, capture_output=True, text=True, timeout=900.0)
                if result.returncode != 0:
                    return "ERR:" + result.stderr[:200]
                return result.stdout
            except Exception as e:
                return "ERR:" + str(e)

    return GenLabelsActor


# ---------------------------------------------------------------------------
# Eval: tournament vs AB(depth=N)
# ---------------------------------------------------------------------------

def evaluate_vs_ab(binary, model, n_games, ab_depth, n_threads=1,
                   timeout_s=600.0):
    """Run NN as Gold vs AB as Scarlet. Returns win rate."""
    weights_bytes = _model_weights_bytes(model)
    weights_path = Path("/tmp/dc_nn_eval_weights.bin")
    weights_path.write_bytes(weights_bytes)

    cmd = [
        str(binary), "--headless",
        "--mode", "tournament",
        "--gold-ai", "nneval",
        "--gold-depth", "2",
        "--gold-nn-weights", str(weights_path),
        "--scarlet-ai", "alphabeta",
        "--scarlet-depth", str(ab_depth),
        "--games", str(n_games),
        "--threads", str(n_threads),
        "--output-json", "/tmp/dc_nn_eval_result.json",
        "--quiet",
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        log.warning("eval timed out")
        return 0.0

    try:
        rj = json.loads(Path("/tmp/dc_nn_eval_result.json").read_text())
        total = rj.get("total_games", 0)
        gold_wins = rj.get("gold_wins", 0)
        return gold_wins / total if total > 0 else 0.0
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def _atomic_write(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(data)
    os.replace(tmp, path)


def save_checkpoint(out_dir, model, **meta):
    data = {
        "type": "nn_v2_supervised",
        "nn_weights": model.to_flat_numpy().tolist(),
        "n_params": NNEval.total_params(),
        "arch": f"{N_INPUT}x{N_H1}x{N_H2}x1",
        **meta,
    }
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    name = f"{CHECKPOINT_PREFIX}{ts}.json"
    _atomic_write(out_dir / name, json.dumps(data))
    _atomic_write(out_dir / LATEST_NAME, json.dumps(data, indent=2))
    return out_dir / name


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Search-supervised NN trainer (NNUE approach)")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-depth", type=int, default=6,
                        help="AB search depth for generating labels")
    parser.add_argument("--random-plies", type=int, default=8,
                        help="Random opening plies for position diversity")
    parser.add_argument("--games-per-batch", type=int, default=50,
                        help="Games to play per batch (each ~20-30 positions)")
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--eval-games", type=int, default=200)
    parser.add_argument("--eval-ab-depth", type=int, default=2)
    parser.add_argument("--max-hours", type=float, default=12.0)
    parser.add_argument("--grad-clip", type=float, default=10.0)
    parser.add_argument("--threads", type=int, default=0,
                        help="Threads for C++ (0=auto)")
    parser.add_argument("--cold-start", action="store_true")
    parser.add_argument("--label", default="supervised")

    # Ray distributed
    parser.add_argument("--ray-address", default=None)
    parser.add_argument("--auto-cluster", action="store_true")
    parser.add_argument("--ray-repo-dir", default=None)
    parser.add_argument("--max-actors", type=int, default=None)

    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    n_threads = args.threads if args.threads > 0 else os.cpu_count() or 4

    log.info("Device: %s", DEVICE)

    # Init or resume
    model = NNEval().to(DEVICE)
    latest_path = args.out / LATEST_NAME
    resumed = False
    total_batches = 0
    total_positions = 0
    best_wr = 0.0
    best_batch = 0
    wr_history = []
    elapsed = 0.0

    if latest_path.exists() and not args.cold_start:
        try:
            ckpt = json.loads(latest_path.read_text())
            if ckpt.get("n_params") == NNEval.total_params():
                model.from_flat_numpy(
                    np.array(ckpt["nn_weights"], dtype=np.float32))
                model = model.to(DEVICE)
                total_batches = ckpt.get("total_batches", 0)
                total_positions = ckpt.get("total_positions", 0)
                best_wr = ckpt.get("best_win_rate", 0.0)
                best_batch = ckpt.get("best_batch", 0)
                wr_history = ckpt.get("win_rate_history", [])
                elapsed = ckpt.get("elapsed_seconds", 0.0)
                resumed = True
                log.info("Resumed: %d batches, %d positions, best WR %.1f%%",
                         total_batches, total_positions, best_wr * 100)
        except Exception as e:
            log.warning("Failed to load checkpoint: %s", e)

    if not resumed:
        model.init_kaiming()
        model = model.to(DEVICE)
        log.info("Fresh start, Kaiming init (%d params)", NNEval.total_params())

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()

    # Signal handling
    stop_requested = [False]
    def handle_signal(sig, frame):
        if stop_requested[0]:
            sys.exit(1)
        log.info("Signal received, stopping after current batch.")
        stop_requested[0] = True
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    binary = args.ray_repo_dir or str(BINARY.parent.parent)
    binary_path = Path(binary) / "build" / "dragonchess" if args.ray_repo_dir else BINARY

    # Ray pool for distributed genlabels
    ray_pool = None
    if args.auto_cluster or args.ray_address:
        import ray
        if args.auto_cluster:
            from aall_cluster import connect
            state, tunnel = connect(namespace="dragonchess-supervised",
                                    verbose=True, log_to_driver=False)
            log.info("Auto-connected, head=%s", state.head_ip)
        else:
            from cluster.runtime_sync import stage_runtime_working_dir
            repo_root = args.ray_repo_dir or str(Path(__file__).parent)
            staged = stage_runtime_working_dir(repo_root)
            log.info("Staged working_dir: %s", staged.path)
            ray.init(address=args.ray_address, namespace="dragonchess-supervised",
                     ignore_reinit_error=True, log_to_driver=False,
                     runtime_env={"working_dir": str(staged.path)})

        cluster_cpus = int(float(ray.cluster_resources().get("CPU", 0)))
        log.info("Ray cluster: %d CPUs", cluster_cpus)
        target = min(cluster_cpus, args.max_actors or cluster_cpus)
        actor_cls = _make_genlabels_actor()
        ray_pool = [
            actor_cls.options(num_cpus=1).remote(args.ray_repo_dir)
            for _ in range(target)
        ]
        ready = ray.get([a.ready.remote() for a in ray_pool])
        hosts = {}
        for r in ready:
            hosts[r["host"]] = hosts.get(r["host"], 0) + 1
        log.info("Pool ready: %d actors across %d hosts (%s)",
                 len(ray_pool), len(hosts), hosts)

    log.info("Training: label_depth=%d random_plies=%d games/batch=%d "
             "lr=%.4f arch=%dx%dx%dx1",
             args.label_depth, args.random_plies, args.games_per_batch,
             args.lr, N_INPUT, N_H1, N_H2)

    max_seconds = args.max_hours * 3600
    start_wall = time.time()

    while not stop_requested[0]:
        batch_start = time.time()

        if time.time() - start_wall + elapsed > max_seconds:
            log.info("Reached max_seconds=%d, stopping.", int(max_seconds))
            break

        # Generate labeled positions
        features, scores = generate_labels(
            binary_path, args.games_per_batch, args.label_depth,
            args.random_plies, n_threads, timeout_s=600.0,
            ray_pool=ray_pool)

        if features is None:
            log.warning("No data generated, retrying...")
            continue

        total_batches += 1
        n_pos = features.shape[0]
        total_positions += n_pos

        # Train on the batch
        features_dev = features.to(DEVICE)
        scores_dev = scores.to(DEVICE)

        # Filter terminal positions (checkmate = +/-10000).
        # Keep raw scores so NN output matches engine eval scale (~[-50, 50]).
        mask = scores_dev.abs().squeeze() < 1000.0
        if mask.sum() < 10:
            log.warning("Batch %d: <10 non-terminal positions, skipping", total_batches)
            continue
        features_dev = features_dev[mask]
        scores_norm = scores_dev[mask]

        optimizer.zero_grad()
        pred = model(features_dev)
        loss = criterion(pred, scores_norm)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        batch_ms = int((time.time() - batch_start) * 1000)
        log.info("Batch %4d | positions %6d | loss %.6f | "
                 "score_range [%.1f, %.1f] | %dms",
                 total_batches, n_pos, loss.item(),
                 scores.min().item(), scores.max().item(), batch_ms)

        # Periodic eval
        if total_batches % args.eval_every == 0:
            log.info("Evaluating vs AlphaBeta(depth=%d)...", args.eval_ab_depth)
            wr = evaluate_vs_ab(
                binary_path, model, args.eval_games, args.eval_ab_depth,
                n_threads=n_threads, timeout_s=600.0)
            wr_history.append({
                "batch": total_batches,
                "win_rate_vs_ab": wr,
                "positions": total_positions,
            })
            log.info("  WR vs AB(d=%d): %.1f%%  (best: %.1f%% @ batch %d)",
                     args.eval_ab_depth, wr * 100, best_wr * 100, best_batch)

            if wr > best_wr:
                best_wr = wr
                best_batch = total_batches
                save_checkpoint(
                    args.out, model,
                    total_batches=total_batches,
                    total_positions=total_positions,
                    best_win_rate=best_wr,
                    best_batch=best_batch,
                    win_rate_history=wr_history,
                    elapsed_seconds=elapsed + (time.time() - start_wall),
                    config={k: str(v) for k, v in vars(args).items()},
                )
                log.info("  * New best: %.1f%% @ batch %d -> best.json",
                         best_wr * 100, best_batch)

        # Periodic checkpoint
        if total_batches % 50 == 0:
            save_checkpoint(
                args.out, model,
                total_batches=total_batches,
                total_positions=total_positions,
                best_win_rate=best_wr,
                best_batch=best_batch,
                win_rate_history=wr_history,
                elapsed_seconds=elapsed + (time.time() - start_wall),
                config={k: str(v) for k, v in vars(args).items()},
            )

    # Final checkpoint
    final_elapsed = elapsed + (time.time() - start_wall)
    save_checkpoint(
        args.out, model,
        total_batches=total_batches,
        total_positions=total_positions,
        best_win_rate=best_wr,
        best_batch=best_batch,
        win_rate_history=wr_history,
        elapsed_seconds=final_elapsed,
        config={k: str(v) for k, v in vars(args).items()},
    )
    log.info("Done. Batches: %d  Positions: %d  Best WR: %.1f%% @ batch %d",
             total_batches, total_positions, best_wr * 100, best_batch)


if __name__ == "__main__":
    main()
