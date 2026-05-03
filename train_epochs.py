#!/usr/bin/env python3
"""
Epoch-based NN trainer for DragonchessAI.

Loads a pre-generated dataset (from generate_dataset.py) and trains the
v2 NNUE network in proper epochs with mini-batches, shuffling, and
train/val split. Standard supervised learning.

Usage:
    python train_epochs.py \
        --data data/d6_1M.bin \
        --out results/nn_v2_epochs/ \
        --lr 0.001 \
        --epochs 50 \
        --batch-size 8192

Evaluates vs AB(d=2) after each epoch using local C++ binary.
"""

import argparse
import json
import logging
import math
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
from torch.utils.data import Dataset, DataLoader

N_INPUT = 32284
N_H1 = 512
N_H2 = 64

BINARY = Path(__file__).parent / "build" / "dragonchess"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("train_epochs")


# ---------------------------------------------------------------------------
# Model (same architecture as train_nn_v2.py)
# ---------------------------------------------------------------------------

class NNEval(nn.Module):
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
        return self.to_flat_numpy().astype(np.float32).tobytes()


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

MAGIC = b'DCDT'


def load_dataset(path):
    """Load binary dataset into sparse representation.

    Returns:
        indices_list: list of numpy arrays (uint16 feature indices per position)
        values_list: list of numpy arrays (float32 feature values per position)
        scores: numpy array of float32 scores
        n_features: int
    """
    log.info("Loading dataset from %s ...", path)
    t0 = time.time()
    with open(path, 'rb') as f:
        magic = f.read(4)
        assert magic == MAGIC, f"Bad magic: {magic}"
        n_pos, n_feat = struct.unpack('<II', f.read(8))

        scores = np.empty(n_pos, dtype=np.float32)
        indices_list = []
        values_list = []

        for i in range(n_pos):
            scores[i] = struct.unpack('<f', f.read(4))[0]
            nnz = struct.unpack('<H', f.read(2))[0]
            if nnz > 0:
                raw = f.read(nnz * 6)
                pairs = np.frombuffer(raw, dtype=[('idx', '<u2'), ('val', '<f4')])
                indices_list.append(pairs['idx'].copy())
                values_list.append(pairs['val'].copy())
            else:
                indices_list.append(np.empty(0, dtype=np.uint16))
                values_list.append(np.empty(0, dtype=np.float32))

    dt = time.time() - t0
    log.info("Loaded %d positions (%d features) in %.1fs", n_pos, n_feat, dt)
    return indices_list, values_list, scores, n_feat


class SparsePositionDataset(Dataset):
    """PyTorch dataset wrapping sparse position data."""

    def __init__(self, indices_list, values_list, scores, max_score=1000.0):
        # Filter terminal positions (|score| >= max_score)
        mask = np.abs(scores) < max_score
        keep = np.where(mask)[0]
        self.indices_list = [indices_list[i] for i in keep]
        self.values_list = [values_list[i] for i in keep]
        self.scores = scores[keep]
        log.info("Dataset: %d positions after filtering (removed %d terminals)",
                 len(self.scores), len(scores) - len(self.scores))

    def __len__(self):
        return len(self.scores)

    def __getitem__(self, idx):
        return idx  # Return index, collate builds dense batch


def collate_sparse(batch_indices, indices_list, values_list, scores,
                   n_features=N_INPUT):
    """Build dense feature tensor from sparse data for a mini-batch."""
    batch_size = len(batch_indices)
    features = torch.zeros(batch_size, n_features, dtype=torch.float32)
    batch_scores = np.empty((batch_size, 1), dtype=np.float32)

    for b, idx in enumerate(batch_indices):
        idxs = indices_list[idx]
        vals = values_list[idx]
        if len(idxs) > 0:
            features[b, idxs.astype(np.int64)] = torch.from_numpy(vals)
        batch_scores[b, 0] = scores[idx]

    return features, torch.from_numpy(batch_scores)


# ---------------------------------------------------------------------------
# Evaluation vs AB
# ---------------------------------------------------------------------------

def evaluate_vs_ab(model, n_games=200, ab_depth=2, timeout_s=600.0):
    """Run NN as Gold vs AB as Scarlet locally. Returns win rate."""
    weights_path = Path("/tmp/dc_epoch_eval_weights.bin")
    weights_path.write_bytes(model.to_binary_weights())

    binary = BINARY
    if not binary.exists():
        log.warning("Binary not found at %s, skipping eval", binary)
        return -1.0

    cmd = [
        str(binary), "--headless",
        "--mode", "tournament",
        "--gold-ai", "nneval",
        "--gold-depth", "2",
        "--gold-nn-weights", str(weights_path),
        "--scarlet-ai", "alphabeta",
        "--scarlet-depth", str(ab_depth),
        "--games", str(n_games),
        "--threads", str(max(1, os.cpu_count() or 4)),
        "--output-json", "/tmp/dc_epoch_eval_result.json",
        "--quiet",
    ]
    try:
        subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
        rj = json.loads(Path("/tmp/dc_epoch_eval_result.json").read_text())
        total = rj.get("total_games", 0)
        gold_wins = rj.get("gold_wins", 0)
        return gold_wins / total if total > 0 else 0.0
    except Exception as e:
        log.warning("Eval failed: %s", e)
        return -1.0


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def _atomic_write(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(data)
    os.replace(tmp, path)


def save_checkpoint(out_dir, model, optimizer, epoch, best_wr, best_epoch,
                    wr_history, train_losses, val_losses, elapsed):
    data = {
        "type": "nn_v2_epochs",
        "arch": f"{N_INPUT}x{N_H1}x{N_H2}x1",
        "n_params": NNEval.total_params(),
        "nn_weights": model.to_flat_numpy().tolist(),
        "epoch": epoch,
        "best_win_rate": best_wr,
        "best_epoch": best_epoch,
        "win_rate_history": wr_history,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "elapsed_seconds": elapsed,
    }
    _atomic_write(out_dir / "latest.json", json.dumps(data, indent=2))
    if best_wr > 0 and epoch == best_epoch:
        _atomic_write(out_dir / "best.json", json.dumps(data, indent=2))
        log.info("  * New best: %.1f%% @ epoch %d -> best.json",
                 best_wr * 100, epoch)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Epoch-based NN trainer for DragonchessAI")
    parser.add_argument("--data", required=True, type=Path,
                        help="Binary dataset from generate_dataset.py")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--val-frac", type=float, default=0.05,
                        help="Fraction of data for validation")
    parser.add_argument("--eval-games", type=int, default=200)
    parser.add_argument("--eval-ab-depth", type=int, default=2)
    parser.add_argument("--grad-clip", type=float, default=10.0)
    parser.add_argument("--score-scale", type=float, default=1.0,
                        help="Divide raw scores by this before MSE. "
                             "If scores are in centipawns (~[-500,500]), "
                             "try 100.0 to normalize to ~[-5,5].")
    parser.add_argument("--lr-schedule", choices=["constant", "cosine"],
                        default="cosine")
    parser.add_argument("--cold-start", action="store_true")
    parser.add_argument("--max-hours", type=float, default=24.0)
    parser.add_argument("--num-workers", type=int, default=0,
                        help="DataLoader workers (0=main thread)")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    # Load dataset
    indices_list, values_list, scores, n_feat = load_dataset(args.data)

    # Filter terminals
    mask = np.abs(scores) < 1000.0
    keep = np.where(mask)[0]
    indices_list = [indices_list[i] for i in keep]
    values_list = [values_list[i] for i in keep]
    scores = scores[keep]
    log.info("After filtering terminals: %d positions", len(scores))

    # Score scaling
    if args.score_scale != 1.0:
        scores = scores / args.score_scale
        log.info("Scaled scores by 1/%.1f, range [%.1f, %.1f]",
                 args.score_scale, scores.min(), scores.max())
    else:
        log.info("Score range: [%.1f, %.1f]", scores.min(), scores.max())

    # Train/val split
    n = len(scores)
    perm = np.random.default_rng(42).permutation(n)
    n_val = max(1, int(n * args.val_frac))
    n_train = n - n_val

    train_idx = perm[:n_train]
    val_idx = perm[n_train:]

    train_indices = [indices_list[i] for i in train_idx]
    train_values = [values_list[i] for i in train_idx]
    train_scores = scores[train_idx]

    val_indices = [indices_list[i] for i in val_idx]
    val_values = [values_list[i] for i in val_idx]
    val_scores = scores[val_idx]

    log.info("Train: %d, Val: %d", n_train, n_val)
    log.info("Device: %s", DEVICE)

    # Model
    model = NNEval().to(DEVICE)
    latest_path = args.out / "latest.json"
    start_epoch = 0
    best_wr = 0.0
    best_epoch = 0
    wr_history = []
    train_losses = []
    val_losses = []
    elapsed = 0.0

    if latest_path.exists() and not args.cold_start:
        try:
            ckpt = json.loads(latest_path.read_text())
            if ckpt.get("n_params") == NNEval.total_params():
                model.from_flat_numpy(
                    np.array(ckpt["nn_weights"], dtype=np.float32))
                model = model.to(DEVICE)
                start_epoch = ckpt.get("epoch", 0)
                best_wr = ckpt.get("best_win_rate", 0.0)
                best_epoch = ckpt.get("best_epoch", 0)
                wr_history = ckpt.get("win_rate_history", [])
                train_losses = ckpt.get("train_losses", [])
                val_losses = ckpt.get("val_losses", [])
                elapsed = ckpt.get("elapsed_seconds", 0.0)
                log.info("Resumed from epoch %d, best WR %.1f%%",
                         start_epoch, best_wr * 100)
        except Exception as e:
            log.warning("Failed to load checkpoint: %s", e)

    if start_epoch == 0 and not latest_path.exists():
        model.init_kaiming()
        model = model.to(DEVICE)
        log.info("Fresh Kaiming init (%d params)", NNEval.total_params())

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()

    # LR schedule
    if args.lr_schedule == "cosine":
        total_steps = (args.epochs - start_epoch) * math.ceil(n_train / args.batch_size)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps, eta_min=args.lr * 0.01)
    else:
        scheduler = None

    # Signal handling
    stop_requested = [False]
    def handle_signal(sig, frame):
        if stop_requested[0]:
            sys.exit(1)
        log.info("Signal received, stopping after current epoch.")
        stop_requested[0] = True
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    session_start = time.time()
    max_seconds = args.max_hours * 3600

    log.info("Training: epochs=%d batch_size=%d lr=%.4f wd=%.1e "
             "score_scale=%.1f arch=%dx%dx%dx1",
             args.epochs, args.batch_size, args.lr, args.weight_decay,
             args.score_scale, N_INPUT, N_H1, N_H2)

    for epoch in range(start_epoch, args.epochs):
        if stop_requested[0]:
            break
        if time.time() - session_start > max_seconds:
            log.info("Time limit reached.")
            break

        epoch_start = time.time()
        model.train()

        # Shuffle training data
        epoch_perm = np.random.default_rng(epoch).permutation(n_train)
        epoch_loss = 0.0
        n_batches = 0

        for batch_start_idx in range(0, n_train, args.batch_size):
            batch_end = min(batch_start_idx + args.batch_size, n_train)
            batch_perm = epoch_perm[batch_start_idx:batch_end]

            features, targets = collate_sparse(
                batch_perm, train_indices, train_values, train_scores)
            features = features.to(DEVICE)
            targets = targets.to(DEVICE)

            optimizer.zero_grad()
            pred = model(features)
            loss = criterion(pred, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            if scheduler:
                scheduler.step()

            epoch_loss += loss.item() * (batch_end - batch_start_idx)
            n_batches += 1

        avg_train_loss = epoch_loss / n_train
        train_losses.append(avg_train_loss)

        # Validation loss
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for vb_start in range(0, n_val, args.batch_size):
                vb_end = min(vb_start + args.batch_size, n_val)
                vb_idx = np.arange(vb_start, vb_end)
                features, targets = collate_sparse(
                    vb_idx, val_indices, val_values, val_scores)
                features = features.to(DEVICE)
                targets = targets.to(DEVICE)
                pred = model(features)
                loss = criterion(pred, targets)
                val_loss += loss.item() * (vb_end - vb_start)
        avg_val_loss = val_loss / n_val
        val_losses.append(avg_val_loss)

        epoch_sec = time.time() - epoch_start

        # Eval vs AB (skip early epochs where random NN makes search too slow)
        if epoch + 1 >= 5 and ((epoch + 1) % 5 == 0 or epoch + 1 >= 20):
            # Use fewer games early to avoid timeouts from long passive games
            n_eval = min(args.eval_games, 50) if epoch + 1 < 20 else args.eval_games
            wr = evaluate_vs_ab(model, n_eval, args.eval_ab_depth,
                                timeout_s=300.0)
        else:
            wr = -1.0  # skip
        wr_history.append({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "win_rate_vs_ab": wr,
        })
        if wr > best_wr:
            best_wr = wr
            best_epoch = epoch + 1

        current_lr = optimizer.param_groups[0]['lr']
        log.info("Epoch %3d/%d | train_loss %.4f | val_loss %.4f | "
                 "WR %.1f%% (best %.1f%% @%d) | lr %.6f | %ds",
                 epoch + 1, args.epochs, avg_train_loss, avg_val_loss,
                 wr * 100 if wr >= 0 else -1,
                 best_wr * 100, best_epoch,
                 current_lr, int(epoch_sec))

        # Save checkpoint
        total_elapsed = elapsed + (time.time() - session_start)
        save_checkpoint(args.out, model, optimizer, epoch + 1,
                        best_wr, best_epoch, wr_history,
                        train_losses, val_losses, total_elapsed)

    total_elapsed = elapsed + (time.time() - session_start)
    log.info("Done. Epochs: %d  Best WR: %.1f%% @ epoch %d  Elapsed: %.1f min",
             len(train_losses), best_wr * 100, best_epoch, total_elapsed / 60)


if __name__ == "__main__":
    main()
