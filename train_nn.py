#!/usr/bin/env python3
"""
TDLeaf(λ) trainer for NNUE-style neural network evaluation.
Architecture: 4060 → 256 (clipped ReLU) → 32 (clipped ReLU) → 1
Training on GPU via PyTorch. Selfplay via C++ binary (CPU).
Weights passed to C++ binary via binary file (--nn-weights).
"""

import argparse
import io
import json
import logging
import math
import os
import signal
import smtplib
import socket
import struct
import subprocess
import sys
import tempfile
import time
from email.message import EmailMessage
from pathlib import Path
from typing import Any

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
log = logging.getLogger("train_nn")


# ---------------------------------------------------------------------------
# NN model (PyTorch)
# ---------------------------------------------------------------------------

class NNEval(nn.Module):
    """32284 → 512 → 64 → 1 with ReLU (matches C++ nn_eval.h)."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_INPUT, N_H1),
            nn.ReLU(),
            nn.Linear(N_H1, N_H2),
            nn.ReLU(),
            nn.Linear(N_H2, 1),
        )

    def forward(self, x):
        """x: (batch, N_INPUT) sparse or dense tensor → (batch, 1)"""
        return self.net(x)

    def to_flat_numpy(self):
        """Serialize to flat float32 numpy array matching C++ NNWeights layout."""
        parts = []
        for layer in [self.net[0], self.net[2], self.net[4]]:
            parts.append(layer.weight.detach().cpu().numpy().ravel())
            parts.append(layer.bias.detach().cpu().numpy().ravel())
        return np.concatenate(parts).astype(np.float32)

    def from_flat_numpy(self, flat):
        """Load from flat numpy array."""
        o = 0
        for layer in [self.net[0], self.net[2], self.net[4]]:
            w_shape = layer.weight.shape
            n_w = w_shape[0] * w_shape[1]
            layer.weight.data = torch.from_numpy(
                flat[o:o + n_w].reshape(w_shape).copy()).float()
            o += n_w
            n_b = w_shape[0]
            layer.bias.data = torch.from_numpy(flat[o:o + n_b].copy()).float()
            o += n_b

    def to_binary_file(self, path):
        flat = self.to_flat_numpy()
        with open(path, "wb") as f:
            f.write(flat.tobytes())

    def init_kaiming(self):
        for layer in [self.net[0], self.net[2], self.net[4]]:
            nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
            nn.init.zeros_(layer.bias)

    @staticmethod
    def total_params():
        return N_H1 * N_INPUT + N_H1 + N_H2 * N_H1 + N_H2 + N_H2 + 1


# ---------------------------------------------------------------------------
# Build sparse batch tensor from game records
# ---------------------------------------------------------------------------

def records_to_sparse_batch(positions):
    """Convert list of sparse position dicts to a (T, N_INPUT) sparse tensor."""
    indices = []  # (2, nnz): row, col
    values = []
    for t, pos in enumerate(positions):
        for i, v in zip(pos["i"], pos["v"]):
            indices.append([t, i])
            values.append(v)
    if not indices:
        return torch.zeros(len(positions), N_INPUT, device=DEVICE)
    indices_t = torch.tensor(indices, dtype=torch.long).t()
    values_t = torch.tensor(values, dtype=torch.float32)
    return torch.sparse_coo_tensor(
        indices_t, values_t, (len(positions), N_INPUT)
    ).to(DEVICE).to_dense()


# ---------------------------------------------------------------------------
# TDLeaf(λ) training step
# ---------------------------------------------------------------------------

def _compute_eligibility_weights(deltas_np, lambd, gamma):
    """Compute per-position weights from eligibility traces.

    For position t, weight = sum_{s>=t} delta_s * (gamma*lambda)^{s-t}
    This is computed efficiently by scanning backward.
    """
    T = len(deltas_np)
    weights = np.zeros(T, dtype=np.float64)
    running = 0.0
    gl = gamma * lambd
    for t in range(T - 1, -1, -1):
        running = deltas_np[t] + gl * running
        weights[t] = running
    return weights


def tdleaf_batch_step(model, optimizer, records, lambd, gamma,
                      draw_penalty=0.0, grad_clip=10.0):
    """One TDLeaf(λ) training step over a batch of games.

    Single forward pass over all positions, then compute
    eligibility-weighted loss per game.
    """
    model.train()
    total_positions = 0
    total_delta_sq = 0.0
    outcomes = []

    # Collect all positions into one big batch
    all_indices = []  # sparse tensor indices (row, col)
    all_values = []
    game_slices = []  # (start, end, outcome) per game
    offset = 0

    for rec in records:
        positions = rec.get("p", [])
        raw_outcome = float(rec.get("o", 0.0))
        outcomes.append(raw_outcome)
        outcome = draw_penalty if (raw_outcome == 0.0 and draw_penalty != 0.0) else raw_outcome

        if not positions:
            continue

        T = len(positions)
        for t, pos in enumerate(positions):
            row = offset + t
            for i, v in zip(pos["i"], pos["v"]):
                all_indices.append([row, i])
                all_values.append(v)

        game_slices.append((offset, offset + T, outcome))
        offset += T
        total_positions += T

    if not game_slices:
        return {"n_games": len(records), "total_positions": 0,
                "rmse": 0.0, "gold_win_rate": 0.0, "draw_rate": 0.0}

    # Build single sparse tensor and forward pass
    if all_indices:
        idx_t = torch.tensor(all_indices, dtype=torch.long).t()
        val_t = torch.tensor(all_values, dtype=torch.float32)
        x = torch.sparse_coo_tensor(
            idx_t, val_t, (offset, N_INPUT)).to(DEVICE).to_dense()
    else:
        x = torch.zeros(offset, N_INPUT, device=DEVICE)

    all_values_pred = model(x).squeeze(-1)  # (total_positions,)

    # Compute per-position weights using eligibility traces (numpy, fast)
    all_weights = np.zeros(offset, dtype=np.float64)
    with torch.no_grad():
        vals_np = all_values_pred.detach().cpu().numpy()

    for start, end, outcome in game_slices:
        T = end - start
        v = vals_np[start:end]
        # TD errors
        deltas = np.zeros(T, dtype=np.float64)
        for t in range(T - 1):
            deltas[t] = gamma * v[t + 1] - v[t]
        deltas[T - 1] = outcome - v[T - 1]
        total_delta_sq += float(np.sum(deltas ** 2))
        all_weights[start:end] = _compute_eligibility_weights(deltas, lambd, gamma)

    # Weighted loss
    weights_t = torch.tensor(all_weights, dtype=torch.float32, device=DEVICE)
    optimizer.zero_grad()
    loss = -(weights_t * all_values_pred).sum()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()

    n_games = len(records)
    stats = {
        "n_games": n_games,
        "total_positions": total_positions,
        "rmse": math.sqrt(total_delta_sq / max(total_positions, 1)),
        "gold_win_rate": sum(1 for o in outcomes if o > 0) / max(n_games, 1),
        "draw_rate": sum(1 for o in outcomes if o == 0) / max(n_games, 1),
    }
    return stats


# ---------------------------------------------------------------------------
# Game generation (C++ selfplay)
# ---------------------------------------------------------------------------

def _model_weights_bytes(model) -> bytes:
    """Serialize model weights to a bytes buffer matching C++ NNWeights layout."""
    buf = io.BytesIO()
    flat = model.to_flat_numpy()
    buf.write(flat.tobytes())
    return buf.getvalue()


def generate_games(binary, model, n_games, n_threads, td_depth,
                   timeout_s=600.0, pool=None, chunk_size=None,
                   opponent="self", opponent_depth=2):
    """Run C++ selfplay with NN weights via binary file.

    If `pool` is a RayNNPool, self-play is dispatched across the cluster in
    chunks of `chunk_size` games. Otherwise the legacy local subprocess path
    is used (single binary invocation with --threads n_threads).

    opponent: "self" for NN vs NN, "ab" for NN vs AB(opponent_depth)
    """
    if pool is not None:
        weights_bytes = _model_weights_bytes(model)
        return pool.selfplay_batch(
            weights_bytes,
            total_games=int(n_games),
            chunk_size=int(chunk_size or 1),
            td_depth=int(td_depth),
            opponent=opponent,
            opponent_depth=int(opponent_depth),
            threads_per_chunk=1,
            timeout_s=float(timeout_s),
        )

    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        model.to_binary_file(f.name)
        weights_path = f.name

    try:
        if opponent == "ab":
            cmd = [
                str(binary), "--headless",
                "--mode", "selfplay",
                "--gold-nn-weights", weights_path,
                "--gold-depth", str(td_depth),
                "--scarlet-ai", "alphabeta",
                "--scarlet-depth", str(opponent_depth),
                "--games", str(n_games),
                "--threads", str(n_threads),
                "--quiet",
            ]
        else:
            cmd = [
                str(binary), "--headless",
                "--mode", "selfplay",
                "--nn-weights", weights_path,
                "--games", str(n_games),
                "--threads", str(n_threads),
                "--td-depth", str(td_depth),
                "--quiet",
            ]
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout_s
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"dragonchess exited {result.returncode}:\n{result.stderr[:500]}")
        return result.stdout
    finally:
        os.unlink(weights_path)


def parse_game_records(ndjson):
    records = []
    for line in ndjson.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            log.warning("Skipping malformed record: %s", exc)
    return records


def evaluate_vs_ab(binary, model, n_games, td_depth, ab_depth,
                   timeout_s=600.0, pool=None, chunk_size=None):
    """Evaluate NN as Gold vs AlphaBeta as Scarlet. Returns win rate.

    If `pool` is a RayNNPool, eval games are dispatched across the cluster in
    chunks of `chunk_size` games. Otherwise the legacy local subprocess path
    is used.
    """
    if pool is not None:
        weights_bytes = _model_weights_bytes(model)
        result = pool.tournament_batch(
            weights_bytes,
            total_games=int(n_games),
            chunk_size=int(chunk_size or max(1, int(n_games) // 8)),
            ab_depth=int(ab_depth),
            td_depth=int(td_depth),
            threads_per_chunk=1,
            timeout_s=float(timeout_s),
        )
        return result.win_rate

    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        model.to_binary_file(f.name)
        weights_path = f.name

    try:
        cmd = [
            str(binary), "--headless",
            "--mode", "tournament",
            "--gold-nn-weights", weights_path,
            "--scarlet-ai", "alphabeta",
            "--scarlet-depth", str(ab_depth),
            "--games", str(n_games),
            "--threads", str(max(1, os.cpu_count() or 4)),
            "--output-json", "-",
            "--quiet",
        ]
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout_s
        )
        if result.returncode != 0:
            raise RuntimeError(f"Eval failed: {result.stderr[:500]}")

        data = json.loads(result.stdout)
        summary = data.get("summary", data)
        total = summary["total_games"]
        gold_wins = summary["gold_wins"]
        return gold_wins / total if total > 0 else 0.0
    finally:
        os.unlink(weights_path)


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------

def _utc_now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# ---------------------------------------------------------------------------
# Email notification
# ---------------------------------------------------------------------------

def _send_email_notification(subject, body, to_addr):
    """Send a plain-text email via SMTP. Reads credentials from env.

    Required env vars:
      NOTIFY_EMAIL_SMTP_USER, NOTIFY_EMAIL_SMTP_PASS
    Optional env vars:
      NOTIFY_EMAIL_FROM (defaults to NOTIFY_EMAIL_SMTP_USER)
      NOTIFY_EMAIL_SMTP_HOST (default: smtp.gmail.com)
      NOTIFY_EMAIL_SMTP_PORT (default: 587)

    Logs a warning and returns False if env is not configured.
    """
    user = os.environ.get("NOTIFY_EMAIL_SMTP_USER", "")
    pw = os.environ.get("NOTIFY_EMAIL_SMTP_PASS", "")
    if not user or not pw:
        log.warning(
            "Email notification requested but NOTIFY_EMAIL_SMTP_USER / "
            "NOTIFY_EMAIL_SMTP_PASS not set — skipping."
        )
        return False

    from_addr = os.environ.get("NOTIFY_EMAIL_FROM", user)
    host = os.environ.get("NOTIFY_EMAIL_SMTP_HOST", "smtp.gmail.com")
    port = int(os.environ.get("NOTIFY_EMAIL_SMTP_PORT", "587"))

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = from_addr
    msg["To"] = to_addr
    msg.set_content(body)

    try:
        with smtplib.SMTP(host, port, timeout=30) as smtp:
            smtp.ehlo()
            smtp.starttls()
            smtp.login(user, pw)
            smtp.send_message(msg)
        log.info("Email notification sent to %s (%s)", to_addr, subject)
        return True
    except Exception as exc:
        log.warning("Email notification failed: %s", exc)
        return False


def _build_notification_body(
    *, label, out_dir, reason, total_batches, total_games,
    best_wr, best_batch, elapsed_seconds, recent_avg_wr, config,
):
    lines = [
        f"DragonchessAI training finished: {reason}",
        "",
        f"Label:          {label}",
        f"Output dir:     {out_dir}",
        f"Host:           {socket.gethostname()}",
        "",
        f"Total batches:  {total_batches}",
        f"Total games:    {total_games}",
        f"Best WR vs AB:  {best_wr * 100:.1f}% @ batch {best_batch}",
        f"Recent avg WR:  {recent_avg_wr * 100:.1f}%",
        f"Elapsed:        {elapsed_seconds / 3600:.2f} h",
        "",
        "Config:",
    ]
    for k, v in config.items():
        lines.append(f"  {k} = {v}")
    return "\n".join(lines) + "\n"


def _atomic_write(path, content):
    tmp = str(path) + ".tmp"
    Path(tmp).write_text(content, encoding="utf-8")
    os.replace(tmp, str(path))


def save_checkpoint(out_dir, model, optimizer, meta):
    data = {
        "version": 5,
        "type": "nn",
        "timestamp_utc": _utc_now(),
        "nn_weights": model.to_flat_numpy().tolist(),
        "optimizer_state": {
            k: v.cpu().tolist() if isinstance(v, torch.Tensor) else v
            for k, v in optimizer.state_dict().items()
        } if False else None,  # Skip optimizer state for now (large)
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
# Main training loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="NN TDLeaf(λ) trainer (PyTorch)")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--lambda", dest="lambd", type=float, default=0.7)
    parser.add_argument("--td-depth", type=int, default=2)
    parser.add_argument("--games-per-batch", type=int, default=50)
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--eval-games", type=int, default=200)
    parser.add_argument("--eval-ab-depth", type=int, default=2)
    parser.add_argument("--draw-penalty", type=float, default=0.0)
    parser.add_argument("--opponent", default="self", choices=["self", "ab", "ab-mix"],
                        help="Training opponent: 'self' for NN vs NN, "
                             "'ab' for NN vs AB(opponent-depth), "
                             "'ab-mix' for cycling AB d=1/2/2/2/3")
    parser.add_argument("--opponent-depth", type=int, default=2,
                        help="AB search depth for opponent (only used with --opponent ab)")
    parser.add_argument("--max-hours", type=float, default=6.0)
    parser.add_argument("--early-stop-patience", type=int, default=3000)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=10.0)

    # Ray-distributed selfplay / eval
    parser.add_argument("--ray-address", default=None,
                        help="Ray cluster address (e.g. 'auto'). If set, "
                             "selfplay and eval run across the cluster.")
    parser.add_argument("--auto-cluster", action="store_true",
                        help="Auto-discover cluster head, tunnel if needed")
    parser.add_argument("--ray-repo-dir", default=None,
                        help="Repo directory on workers (default: ~/DragonchessAI)")
    parser.add_argument("--max-actors", type=int, default=None,
                        help="Cap on Ray actor count (default: all cluster CPUs)")
    parser.add_argument("--selfplay-chunk-size", type=int, default=2,
                        help="Games per selfplay actor request (Ray mode)")
    parser.add_argument("--eval-chunk-size", type=int, default=10,
                        help="Games per eval actor request (Ray mode)")
    parser.add_argument("--cold-start", action="store_true",
                        help="Ignore any existing latest.json and train from "
                             "a fresh Kaiming init.")

    # Email notification
    parser.add_argument("--notify-email", default=None,
                        help="Send email notification on early stop / "
                             "normal completion. Set to recipient address, or "
                             "omit to disable. SMTP creds via env vars.")
    parser.add_argument("--label", default="nightly-ray",
                        help="Label used in notification subject/body.")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    log.info("Device: %s", DEVICE)

    # Init or resume
    model = NNEval().to(DEVICE)
    latest_path = args.out / LATEST_NAME
    resumed = False
    total_batches = 0
    total_games = 0
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
                total_games = ckpt.get("total_games", 0)
                best_wr = ckpt.get("best_win_rate", 0.0)
                best_batch = ckpt.get("best_batch", 0)
                wr_history = ckpt.get("win_rate_history", [])
                elapsed = ckpt.get("elapsed_seconds", 0.0)
                resumed = True
                log.info("Resumed from checkpoint: %d batches, %d games, "
                         "best WR %.1f%% @ batch %d",
                         total_batches, total_games,
                         best_wr * 100, best_batch)
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            log.warning("Failed to load checkpoint: %s", e)

    if not resumed:
        model.init_kaiming()
        model = model.to(DEVICE)
        log.info("Starting fresh with Kaiming init (%d params).%s",
                 NNEval.total_params(),
                 " [cold start requested]" if args.cold_start else "")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # ---- Optional Ray pool for distributed selfplay/eval ----
    pool = None
    if args.auto_cluster or args.ray_address:
        try:
            import ray
            from cluster.nn_pool import RayNNPool
        except ImportError as exc:
            log.error("ray/nn_pool import failed: %s", exc)
            sys.exit(2)
        if args.auto_cluster:
            from aall_cluster import connect
            state, tunnel = connect(namespace="dragonchess-nn", verbose=True, log_to_driver=False)
            log.info("Auto-connected to cluster, head=%s", state.head_ip)
        else:
            log.info("Connecting to Ray at %s", args.ray_address)
            # Stage a working_dir with the cluster/ module + binary so workers
            # can import actor classes and find the dragonchess executable.
            from cluster.runtime_sync import stage_runtime_working_dir
            repo_root = args.ray_repo_dir or str(Path(__file__).parent)
            staged = stage_runtime_working_dir(repo_root)
            log.info("Staged working_dir: %s", staged.path)
            ray.init(address=args.ray_address, namespace="dragonchess-nn",
                     ignore_reinit_error=True, log_to_driver=False,
                     runtime_env={"working_dir": str(staged.path)})
        cluster_res = ray.cluster_resources()
        log.info("Ray cluster resources: %s", dict(cluster_res))
        pool = RayNNPool(
            repo_dir=args.ray_repo_dir,
            max_actors=args.max_actors,
        )
        pool.start()
        log.info("NN pool ready: %d actors across %d hosts (%s)",
                 pool.actor_count, len(pool.describe_capacity()),
                 pool.describe_capacity())

    # Signal handling
    stop_requested = [False]
    def handle_signal(sig, frame):
        if stop_requested[0]:
            sys.exit(1)
        log.info("Signal received — will stop after current batch.")
        stop_requested[0] = True
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    session_start = time.time()
    max_seconds = args.max_hours * 3600

    log.info("Training: λ=%.2f lr=%.4f depth=%d games/batch=%d "
             "arch=%dx%dx%dx1 draw_penalty=%.2f device=%s",
             args.lambd, args.lr, args.td_depth, args.games_per_batch,
             N_INPUT, N_H1, N_H2, args.draw_penalty, DEVICE)

    stop_reason = "signal"
    while not stop_requested[0]:
        elapsed_session = time.time() - session_start
        if elapsed_session >= max_seconds:
            log.info("Reached max_seconds=%d — stopping.", int(max_seconds))
            stop_reason = "max_hours"
            break

        batch_start = time.time()

        # Generate games (C++ on CPU, local or via Ray pool)
        # Mixed-opponent mode: cycle through opponent depths for diversity
        if args.opponent == "ab-mix":
            mix_depths = [1, 2, 2, 2, 3]  # weighted toward d=2
            opp_depth = mix_depths[total_batches % len(mix_depths)]
            opp_type = "ab"
        else:
            opp_type = args.opponent
            opp_depth = args.opponent_depth
        try:
            ndjson = generate_games(
                BINARY, model, args.games_per_batch,
                max(1, os.cpu_count() or 4),
                args.td_depth,
                pool=pool,
                chunk_size=args.selfplay_chunk_size,
                opponent=opp_type,
                opponent_depth=opp_depth,
            )
            records = parse_game_records(ndjson)
        except Exception as exc:
            log.warning("Game generation failed: %s", exc)
            time.sleep(1)
            continue

        if not records:
            log.warning("Empty batch — skipping.")
            continue

        # Training step (PyTorch on GPU)
        stats = tdleaf_batch_step(
            model, optimizer, records, args.lambd, 1.0,
            args.draw_penalty, args.grad_clip,
        )

        total_batches += 1
        total_games += stats["n_games"]
        batch_ms = int((time.time() - batch_start) * 1000)

        log.info(
            "Batch %4d | games %6d | positions %6d | "
            "RMSE %.4f | gold_wr %.3f | draw %.3f | %dms",
            total_batches, stats["n_games"], stats["total_positions"],
            stats["rmse"], stats["gold_win_rate"], stats["draw_rate"],
            batch_ms,
        )

        # Evaluate
        if total_batches % args.eval_every == 0:
            log.info("Evaluating vs AlphaBeta(depth=%d) …",
                     args.eval_ab_depth)
            try:
                wr = evaluate_vs_ab(
                    BINARY, model, args.eval_games,
                    args.td_depth, args.eval_ab_depth,
                    pool=pool,
                    chunk_size=args.eval_chunk_size,
                )
            except Exception as exc:
                log.warning("Evaluation failed: %s", exc)
                wr = 0.0

            wr_history.append({
                "batch": total_batches,
                "win_rate_vs_ab": wr,
            })

            is_best = wr > best_wr
            if is_best:
                best_wr = wr
                best_batch = total_batches

            log.info("  → Win rate vs AB(d=%d): %.1f%%  "
                     "(best: %.1f%% @ batch %d)",
                     args.eval_ab_depth, wr * 100, best_wr * 100, best_batch)

            # Save checkpoint
            meta = {
                "total_games": total_games,
                "total_batches": total_batches,
                "best_win_rate": best_wr,
                "best_batch": best_batch,
                "win_rate_history": wr_history,
                "elapsed_seconds": elapsed + (time.time() - session_start),
                "config": {
                    "lambda": args.lambd,
                    "lr": args.lr,
                    "td_depth": args.td_depth,
                    "draw_penalty": args.draw_penalty,
                },
            }
            save_checkpoint(args.out, model, optimizer, meta)

            if is_best:
                best_data = {
                    "version": 5,
                    "type": "nn",
                    "nn_weights": model.to_flat_numpy().tolist(),
                    "n_params": NNEval.total_params(),
                    "best_win_rate": best_wr,
                    "best_batch": best_batch,
                }
                _atomic_write(args.out / BEST_NAME, json.dumps(best_data))
                log.info("  * New best WR: %.1f%% @ batch %d -> best.json",
                         wr * 100, total_batches)

            # Early stopping
            batches_since_best = total_batches - best_batch
            if (args.early_stop_patience > 0
                    and batches_since_best >= args.early_stop_patience):
                recent = [e["win_rate_vs_ab"] for e in wr_history[-20:]]
                avg_recent = sum(recent) / len(recent) if recent else 0
                log.warning(
                    "EARLY STOP: %d batches since best (%.1f%% @ %d), "
                    "recent avg %.1f%%",
                    batches_since_best, best_wr * 100, best_batch,
                    avg_recent * 100)
                stop_reason = "early_stop"
                break

    # Final save
    meta = {
        "total_games": total_games,
        "total_batches": total_batches,
        "best_win_rate": best_wr,
        "best_batch": best_batch,
        "win_rate_history": wr_history,
        "elapsed_seconds": elapsed + (time.time() - session_start),
    }
    save_checkpoint(args.out, model, optimizer, meta)
    log.info("Done. Total batches: %d  Total games: %d  "
             "Best WR: %.1f%% @ batch %d",
             total_batches, total_games, best_wr * 100, best_batch)

    # ---- Email notification ----
    if args.notify_email:
        recent = [e["win_rate_vs_ab"] for e in wr_history[-20:]]
        avg_recent = sum(recent) / len(recent) if recent else 0.0
        subject = (
            f"[DragonchessAI] {args.label}: {stop_reason} "
            f"— best {best_wr * 100:.1f}% @ batch {best_batch}"
        )
        body = _build_notification_body(
            label=args.label,
            out_dir=str(args.out),
            reason=stop_reason,
            total_batches=total_batches,
            total_games=total_games,
            best_wr=best_wr,
            best_batch=best_batch,
            elapsed_seconds=elapsed + (time.time() - session_start),
            recent_avg_wr=avg_recent,
            config={
                "lambda": args.lambd,
                "lr": args.lr,
                "td_depth": args.td_depth,
                "games_per_batch": args.games_per_batch,
                "eval_games": args.eval_games,
                "eval_every": args.eval_every,
                "eval_ab_depth": args.eval_ab_depth,
                "draw_penalty": args.draw_penalty,
                "early_stop_patience": args.early_stop_patience,
                "ray_address": args.ray_address,
            },
        )
        _send_email_notification(subject, body, args.notify_email)

    # ---- Shut down Ray pool ----
    if pool is not None:
        try:
            pool.shutdown()
        except Exception as exc:
            log.warning("Pool shutdown error: %s", exc)


if __name__ == "__main__":
    main()
