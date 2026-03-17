"""
TD(λ) self-play trainer for Dragonchess.

Trains a 40-dimensional feature-weight evaluation function via temporal-difference
learning with eligibility traces.  The C++ engine generates self-play game records
(NDJSON) and this script updates the weights after each batch.

Designed to be maximally robust to interruption:
  - Atomic checkpoints after every batch (write-then-rename).
  - Resumes seamlessly from the latest checkpoint in --out.
  - SIGINT / SIGTERM save a checkpoint before exiting cleanly.
  - C++ subprocess crashes are caught and logged; training continues.
  - A lock file prevents accidental concurrent trainers on the same directory.

Usage
-----
Local (all cores):
    python train_td.py --out results/td/

Local (limited):
    python train_td.py --out results/td/ --workers 4 --games-per-batch 50

Ray cluster:
    python train_td.py --out results/td/ --ray --ray-address auto \\
        --workers 16 --games-per-batch 200

Evaluate every 25 batches against AlphaBeta(depth=2):
    python train_td.py --out results/td/ --eval-every 25 --eval-games 400
"""

from __future__ import annotations

import argparse
import fcntl
import json
import logging
import math
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

# Optional Ray import — only required when --ray is passed.
try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BINARY = Path(__file__).parent / "build" / "dragonchess"
N_FEATURES = 40
CHECKPOINT_PREFIX = "ckpt_"
LATEST_NAME = "latest.json"
LOCK_NAME = ".train_td.lock"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("train_td")


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _atomic_write(path: Path, data: str) -> None:
    """Write data to a temp file beside `path`, then rename atomically."""
    tmp = path.with_suffix(".tmp")
    tmp.write_text(data, encoding="utf-8")
    tmp.rename(path)


def save_checkpoint(
    out_dir: Path,
    weights: np.ndarray,
    adagrad_G: np.ndarray,
    meta: dict[str, Any],
    keep_last: int = 20,
) -> Path:
    """Save checkpoint atomically and update latest.json.  Prunes old checkpoints."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = _utc_now()
    data = {
        "version": 2,
        "timestamp_utc": ts,
        "weights": weights.tolist(),
        "adagrad_G": adagrad_G.tolist(),
        "n_features": N_FEATURES,
        **meta,
    }
    payload = json.dumps(data, indent=2)

    ckpt_path = out_dir / f"{CHECKPOINT_PREFIX}{ts}.json"
    _atomic_write(ckpt_path, payload)
    _atomic_write(out_dir / LATEST_NAME, payload)

    # Prune old checkpoints
    ckpts = sorted(out_dir.glob(f"{CHECKPOINT_PREFIX}*.json"))
    for old in ckpts[:-keep_last]:
        try:
            old.unlink()
        except OSError:
            pass

    return ckpt_path


def load_checkpoint(out_dir: Path) -> dict[str, Any] | None:
    """Load the latest checkpoint from out_dir, or None if none exists."""
    latest = out_dir / LATEST_NAME
    if not latest.exists():
        return None
    try:
        return json.loads(latest.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("Could not read latest checkpoint (%s); starting fresh.", exc)
        return None


# ---------------------------------------------------------------------------
# Self-play game generation
# ---------------------------------------------------------------------------

def _weights_to_csv(weights: np.ndarray) -> str:
    return ",".join(f"{w:.7f}" for w in weights)


def generate_games_local(
    binary: Path,
    weights: np.ndarray,
    n_games: int,
    n_threads: int,
    td_depth: int,
    timeout_s: float = 600.0,
) -> str:
    """Run the C++ binary in selfplay mode and return raw NDJSON output."""
    csv = _weights_to_csv(weights)
    cmd = [
        str(binary), "--headless",
        "--mode", "selfplay",
        "--td-weights", csv,
        "--games", str(n_games),
        "--threads", str(n_threads),
        "--td-depth", str(td_depth),
        "--quiet",
    ]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"dragonchess exited {result.returncode}:\n{result.stderr[:500]}"
        )
    return result.stdout


def parse_game_records(ndjson: str) -> list[dict[str, Any]]:
    """Parse NDJSON output from the C++ binary into a list of game dicts."""
    records = []
    for line in ndjson.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            log.warning("Skipping malformed game record: %s", exc)
    return records


# ---------------------------------------------------------------------------
# Ray worker (only materialised when --ray is used)
# ---------------------------------------------------------------------------

def _make_ray_worker(binary_str: str, td_depth: int, timeout_s: float):
    """Return a Ray remote function that generates a batch of self-play games."""

    @ray.remote(num_cpus=1)
    def _worker(weights_list: list[float], n_games: int, n_threads: int) -> str:
        weights = np.array(weights_list, dtype=np.float64)
        csv = ",".join(f"{w:.7f}" for w in weights)
        cmd = [
            binary_str, "--headless",
            "--mode", "selfplay",
            "--td-weights", csv,
            "--games", str(n_games),
            "--threads", str(n_threads),
            "--td-depth", str(td_depth),
            "--quiet",
        ]
        import subprocess as _sp
        r = _sp.run(cmd, capture_output=True, text=True, timeout=timeout_s)
        if r.returncode != 0:
            raise RuntimeError(
                f"dragonchess exited {r.returncode}:\n{r.stderr[:500]}"
            )
        return r.stdout

    return _worker


# ---------------------------------------------------------------------------
# TD(λ) update
# ---------------------------------------------------------------------------

def td_lambda_batch_gradient(
    records: list[dict[str, Any]],
    weights: np.ndarray,
    lambd: float,
    gamma: float,
) -> tuple[np.ndarray, dict[str, float]]:
    """Compute accumulated TD(λ) gradient over a batch of game records.

    Returns (gradient, stats) where stats contains diagnostic scalars.

    The gradient (dL/dw) is the sum over all games and timesteps of:
        delta_t * e_t
    where delta_t is the TD error and e_t is the eligibility trace.

    Features are Gold-positive; outcomes are +1/−1/0 from Gold's perspective.
    """
    grad = np.zeros_like(weights)
    total_positions = 0
    total_delta_sq = 0.0
    outcomes = []

    for rec in records:
        positions = rec.get("p", [])
        outcome = float(rec.get("o", 0.0))
        outcomes.append(outcome)

        if not positions:
            continue

        phi = np.array(positions, dtype=np.float64)  # (T, n_features)
        T = len(phi)

        # Values at each timestep: V_t = phi_t · w
        V = phi @ weights  # (T,)

        e = np.zeros(N_FEATURES, dtype=np.float64)  # eligibility trace

        for t in range(T):
            if t + 1 < T:
                delta = gamma * V[t + 1] - V[t]
            else:
                delta = outcome - V[t]  # terminal

            e = gamma * lambd * e + phi[t]
            grad += delta * e
            total_delta_sq += delta * delta

        total_positions += T

    n_games = len(records)
    stats = {
        "n_games": n_games,
        "total_positions": total_positions,
        "rmse": math.sqrt(total_delta_sq / max(total_positions, 1)),
        "gold_win_rate": sum(1 for o in outcomes if o > 0) / max(n_games, 1),
        "draw_rate":     sum(1 for o in outcomes if o == 0) / max(n_games, 1),
    }
    return grad, stats


def adagrad_update(
    weights: np.ndarray,
    adagrad_G: np.ndarray,
    grad: np.ndarray,
    lr: float,
    epsilon: float = 1e-8,
    grad_clip: float = 10.0,
) -> None:
    """In-place AdaGrad update.  Clips gradient L2 norm before accumulating."""
    gnorm = float(np.linalg.norm(grad))
    if gnorm > grad_clip:
        grad = grad * (grad_clip / gnorm)
    adagrad_G += grad * grad
    weights += lr / (np.sqrt(adagrad_G) + epsilon) * grad


# ---------------------------------------------------------------------------
# Evaluation against a fixed opponent
# ---------------------------------------------------------------------------

def evaluate_vs_alphabeta(
    binary: Path,
    weights: np.ndarray,
    n_games: int,
    td_depth: int,
    ab_depth: int,
    n_threads: int,
    timeout_s: float = 600.0,
) -> float:
    """Win rate of TDEvalAI (Gold) vs AlphaBeta(ab_depth) (Scarlet)."""
    csv = _weights_to_csv(weights)
    cmd = [
        str(binary), "--headless",
        "--mode", "tournament",
        "--gold-td-weights", csv,
        "--gold-depth", str(td_depth),
        "--scarlet-ai", "alphabeta",
        "--scarlet-depth", str(ab_depth),
        "--games", str(n_games),
        "--threads", str(n_threads),
        "--output-json", "-",
        "--quiet",
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout_s
        )
        data = json.loads(result.stdout)
        gold_wins = data["summary"]["gold_wins"]
        total = data["summary"]["total_games"]
        return gold_wins / max(total, 1)
    except Exception as exc:
        log.warning("Evaluation failed: %s", exc)
        return float("nan")


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class TDTrainer:
    def __init__(
        self,
        out_dir: Path,
        binary: Path,
        *,
        lambd: float = 0.7,
        gamma: float = 1.0,
        lr: float = 0.05,
        td_depth: int = 1,
        games_per_batch: int = 50,
        workers: int = 4,
        threads_per_worker: int = 1,
        eval_every: int = 25,
        eval_games: int = 200,
        eval_ab_depth: int = 2,
        warm_start: bool = False,
        ray_mode: bool = False,
        ray_workers: int = 4,
        ray_games_per_worker: int = 50,
        timeout_s: float = 600.0,
    ):
        self.out_dir = out_dir
        self.binary = binary
        self.lambd = lambd
        self.gamma = gamma
        self.lr = lr
        self.td_depth = td_depth
        self.games_per_batch = games_per_batch
        self.workers = workers
        self.threads_per_worker = threads_per_worker
        self.eval_every = eval_every
        self.eval_games = eval_games
        self.eval_ab_depth = eval_ab_depth
        self.warm_start = warm_start
        self.ray_mode = ray_mode
        self.ray_workers = ray_workers
        self.ray_games_per_worker = ray_games_per_worker
        self.timeout_s = timeout_s

        # State (overwritten by load_or_init)
        self.weights: np.ndarray = np.zeros(N_FEATURES)
        self.adagrad_G: np.ndarray = np.full(N_FEATURES, 1e-8)
        self.total_games: int = 0
        self.total_batches: int = 0
        self.win_rate_history: list[dict] = []
        self.elapsed_seconds: float = 0.0
        self._session_start = time.time()
        self._shutdown_requested = False

        # Register signal handlers for clean exit
        signal.signal(signal.SIGINT,  self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

    # ------------------------------------------------------------------
    # Init / resume
    # ------------------------------------------------------------------

    def load_or_init(self) -> None:
        ckpt = load_checkpoint(self.out_dir)
        if ckpt is not None:
            self.weights = np.array(ckpt["weights"], dtype=np.float64)
            self.adagrad_G = np.array(
                ckpt.get("adagrad_G", np.full(N_FEATURES, 1e-8).tolist()),
                dtype=np.float64,
            )
            self.total_games = ckpt.get("total_games", 0)
            self.total_batches = ckpt.get("total_batches", 0)
            self.win_rate_history = ckpt.get("win_rate_history", [])
            self.elapsed_seconds = ckpt.get("elapsed_seconds", 0.0)
            log.info(
                "Resumed from checkpoint: %d batches, %d games, "
                "elapsed %.0fs",
                self.total_batches, self.total_games, self.elapsed_seconds,
            )
        else:
            if self.warm_start:
                self._warm_start_weights()
            else:
                rng = np.random.default_rng(42)
                self.weights = rng.normal(0, 0.01, N_FEATURES)
            log.info("Starting fresh (no checkpoint found).")

    def _warm_start_weights(self) -> None:
        """Initialise material features (indices 0-13) from Jackman proxy values.

        Index mapping matches FEAT_PIECE_TYPES in features.cpp:
        Sylph(1), Griffin(2), Dragon(3), Oliphant(4), Unicorn(5),
        Hero(6), Thief(7), Cleric(8), Mage(9), Paladin(11),
        Warrior(12), Basilisk(13), Elemental(14), Dwarf(15).
        """
        jackman = [1.0, 5.0, 8.0, 5.0, 2.5, 4.5, 4.0, 9.0, 11.0,
                   10.0, 1.0, 3.0, 4.0, 2.0]
        self.weights[:14] = jackman
        log.info("Warm-started material weights from Jackman proxy values.")

    # ------------------------------------------------------------------
    # Signal handling
    # ------------------------------------------------------------------

    def _handle_signal(self, signum, frame) -> None:
        log.warning("Signal %d received — saving checkpoint and exiting…", signum)
        self._shutdown_requested = True

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def checkpoint(self) -> None:
        meta = {
            "total_games": self.total_games,
            "total_batches": self.total_batches,
            "win_rate_history": self.win_rate_history,
            "elapsed_seconds": self.elapsed_seconds + (
                time.time() - self._session_start
            ),
            "config": {
                "lambda": self.lambd,
                "gamma": self.gamma,
                "lr": self.lr,
                "td_depth": self.td_depth,
                "games_per_batch": self.games_per_batch,
                "eval_ab_depth": self.eval_ab_depth,
            },
        }
        path = save_checkpoint(self.out_dir, self.weights, self.adagrad_G, meta)
        log.debug("Checkpoint saved → %s", path.name)

    # ------------------------------------------------------------------
    # Game generation
    # ------------------------------------------------------------------

    def _generate_batch_local(self) -> list[dict]:
        """Generate games_per_batch games locally, retrying once on failure."""
        for attempt in range(2):
            try:
                ndjson = generate_games_local(
                    self.binary,
                    self.weights,
                    self.games_per_batch,
                    self.workers * self.threads_per_worker,
                    self.td_depth,
                    self.timeout_s,
                )
                records = parse_game_records(ndjson)
                if records:
                    return records
                log.warning("Empty game batch (attempt %d); retrying.", attempt + 1)
            except Exception as exc:
                log.warning("Game generation failed (attempt %d): %s", attempt + 1, exc)
        return []

    def _generate_batch_ray(self) -> list[dict]:
        """Distribute game generation across Ray workers."""
        worker_fn = _make_ray_worker(
            str(self.binary), self.td_depth, self.timeout_s
        )
        games_per_worker = self.ray_games_per_worker
        futures = [
            worker_fn.remote(
                self.weights.tolist(),
                games_per_worker,
                self.threads_per_worker,
            )
            for _ in range(self.ray_workers)
        ]
        records = []
        for fut in ray.get(futures):
            records.extend(parse_game_records(fut))
        return records

    def generate_batch(self) -> list[dict]:
        if self.ray_mode:
            return self._generate_batch_ray()
        return self._generate_batch_local()

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self, max_batches: int | None = None, max_seconds: float | None = None) -> None:
        self.load_or_init()
        self._session_start = time.time()
        log.info(
            "Training: λ=%.2f  γ=%.2f  lr=%.4f  depth=%d  "
            "games/batch=%d  eval_every=%d",
            self.lambd, self.gamma, self.lr, self.td_depth,
            self.games_per_batch, self.eval_every,
        )

        batch = 0
        while not self._shutdown_requested:
            if max_batches is not None and batch >= max_batches:
                log.info("Reached max_batches=%d — stopping.", max_batches)
                break
            elapsed = time.time() - self._session_start
            if max_seconds is not None and elapsed > max_seconds:
                log.info("Reached max_seconds=%.0f — stopping.", max_seconds)
                break

            t_batch = time.time()

            # 1. Generate self-play games
            records = self.generate_batch()
            if not records:
                log.warning("Batch %d: no records returned — skipping.", self.total_batches)
                time.sleep(1)
                continue

            # 2. TD(λ) gradient and AdaGrad update
            grad, stats = td_lambda_batch_gradient(
                records, self.weights, self.lambd, self.gamma
            )
            adagrad_update(self.weights, self.adagrad_G, grad, self.lr)

            self.total_games += stats["n_games"]
            self.total_batches += 1
            batch += 1
            batch_ms = (time.time() - t_batch) * 1000

            log.info(
                "Batch %4d | games %6d | positions %6d | "
                "RMSE %.4f | gold_wr %.3f | %.0fms",
                self.total_batches,
                stats["n_games"],
                stats["total_positions"],
                stats["rmse"],
                stats["gold_win_rate"],
                batch_ms,
            )

            # 3. Periodic evaluation
            if self.eval_every > 0 and self.total_batches % self.eval_every == 0:
                log.info("Evaluating vs AlphaBeta(depth=%d) …", self.eval_ab_depth)
                wr = evaluate_vs_alphabeta(
                    self.binary,
                    self.weights,
                    self.eval_games,
                    self.td_depth,
                    self.eval_ab_depth,
                    self.workers * self.threads_per_worker,
                    self.timeout_s,
                )
                self.win_rate_history.append({
                    "batch": self.total_batches,
                    "total_games": self.total_games,
                    "win_rate_vs_ab": wr,
                    "ab_depth": self.eval_ab_depth,
                })
                log.info(
                    "  → Win rate vs AlphaBeta(depth=%d): %.3f",
                    self.eval_ab_depth, wr,
                )

            # 4. Checkpoint after every batch (atomic write, fast)
            self.checkpoint()

        # Final checkpoint on exit
        self.checkpoint()
        log.info(
            "Done. Total batches: %d  Total games: %d",
            self.total_batches, self.total_games,
        )


# ---------------------------------------------------------------------------
# Lock file (prevent duplicate trainers)
# ---------------------------------------------------------------------------

class TrainingLock:
    def __init__(self, out_dir: Path) -> None:
        self.path = out_dir / LOCK_NAME
        self._fp = None

    def acquire(self) -> None:
        out_dir = self.path.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        self._fp = open(self.path, "w")
        try:
            fcntl.flock(self._fp, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self._fp.close()
            raise RuntimeError(
                f"Another trainer is already running in {out_dir}.\n"
                f"If that's wrong, delete {self.path} and retry."
            )
        self._fp.write(f"pid={os.getpid()}\n")
        self._fp.flush()

    def release(self) -> None:
        if self._fp is not None:
            fcntl.flock(self._fp, fcntl.LOCK_UN)
            self._fp.close()
            try:
                self.path.unlink()
            except OSError:
                pass


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="TD(λ) self-play trainer for Dragonchess",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Output / resume
    p.add_argument("--out", default="results/td",
                   help="Output directory for checkpoints")
    p.add_argument("--binary", default=str(BINARY),
                   help="Path to dragonchess binary")

    # TD hyperparameters
    p.add_argument("--lambda", dest="lambd", type=float, default=0.7,
                   help="TD(λ) trace decay")
    p.add_argument("--gamma", type=float, default=1.0,
                   help="Discount factor (1.0 = no discount)")
    p.add_argument("--lr", type=float, default=0.05,
                   help="AdaGrad base learning rate")
    p.add_argument("--td-depth", type=int, default=1,
                   help="AlphaBeta search depth for TDEvalAI during self-play")
    p.add_argument("--warm-start", action="store_true",
                   help="Initialise material weights from Jackman proxy values")

    # Batching / parallelism
    p.add_argument("--games-per-batch", type=int, default=50,
                   help="Self-play games per weight-update step")
    p.add_argument("--workers", type=int,
                   default=max(1, (os.cpu_count() or 4)),
                   help="Local worker threads passed to the C++ binary")
    p.add_argument("--threads-per-worker", type=int, default=1,
                   help="C++ tournament threads per Ray worker (Ray mode only)")

    # Stopping conditions
    p.add_argument("--max-batches", type=int, default=None,
                   help="Stop after this many batches (default: run indefinitely)")
    p.add_argument("--max-hours", type=float, default=None,
                   help="Stop after this many hours (default: run indefinitely)")

    # Evaluation
    p.add_argument("--eval-every", type=int, default=25,
                   help="Evaluate vs AlphaBeta every N batches (0 = disable)")
    p.add_argument("--eval-games", type=int, default=200,
                   help="Games per evaluation tournament")
    p.add_argument("--eval-ab-depth", type=int, default=2,
                   help="AlphaBeta depth for evaluation opponent")

    # Ray
    p.add_argument("--ray", action="store_true",
                   help="Distribute game generation via Ray")
    p.add_argument("--ray-address", default="auto",
                   help="Ray cluster address (used with --ray)")
    p.add_argument("--ray-workers", type=int, default=8,
                   help="Number of Ray remote workers per batch")
    p.add_argument("--ray-games-per-worker", type=int, default=50,
                   help="Self-play games per Ray worker per batch")

    p.add_argument("--timeout", type=float, default=600.0,
                   help="Timeout in seconds for each C++ subprocess call")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    binary = Path(args.binary)
    if not binary.exists():
        sys.exit(
            f"Binary not found: {binary}\n"
            "Build with: cmake --build build --parallel"
        )

    out_dir = Path(args.out)

    if args.ray:
        if not RAY_AVAILABLE:
            sys.exit("Ray is not installed. Run: pip install ray")
        ray.init(address=args.ray_address, ignore_reinit_error=True)
        log.info("Ray initialised at %s", args.ray_address)

    lock = TrainingLock(out_dir)
    lock.acquire()

    try:
        trainer = TDTrainer(
            out_dir=out_dir,
            binary=binary,
            lambd=args.lambd,
            gamma=args.gamma,
            lr=args.lr,
            td_depth=args.td_depth,
            games_per_batch=args.games_per_batch,
            workers=args.workers,
            threads_per_worker=args.threads_per_worker,
            eval_every=args.eval_every,
            eval_games=args.eval_games,
            eval_ab_depth=args.eval_ab_depth,
            warm_start=args.warm_start,
            ray_mode=args.ray,
            ray_workers=args.ray_workers,
            ray_games_per_worker=args.ray_games_per_worker,
            timeout_s=args.timeout,
        )
        max_seconds = args.max_hours * 3600 if args.max_hours is not None else None
        trainer.train(max_batches=args.max_batches, max_seconds=max_seconds)
    finally:
        lock.release()
        if args.ray and RAY_AVAILABLE:
            ray.shutdown()


if __name__ == "__main__":
    main()
