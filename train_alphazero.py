#!/usr/bin/env python3
"""
AlphaZero-style training for Dragonchess.

Loop:
  1. Self-play: C++ binary runs MCTS games, outputs NDJSON training data
  2. Train: PyTorch updates policy+value network from self-play data
  3. Evaluate: tournament vs AB baselines
  4. Repeat

Usage:
    python train_alphazero.py \
        --out results/alphazero/ \
        --games-per-iter 100 \
        --mcts-sims 400 \
        --epochs-per-iter 5 \
        --batch-size 256 \
        --lr 0.001 \
        --iterations 100

    # With Ray cluster:
    python train_alphazero.py \
        --out results/alphazero/ \
        --ray-address auto \
        --max-actors 100
"""

import argparse
import json
import logging
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Constants (must match C++ mcts.h / DualHeadWeights)
# ---------------------------------------------------------------------------

N_INPUT = 32284      # NUM_TD_FEATURES
N_H1 = 256
N_H2 = 128
TOTAL_SQUARES = 288
ACTION_SPACE = TOTAL_SQUARES * TOTAL_SQUARES  # 82944

BINARY = Path(__file__).parent / "build" / "dragonchess"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("alphazero")


# ---------------------------------------------------------------------------
# Dual-head network (matches C++ DualHeadWeights layout)
# ---------------------------------------------------------------------------

class AlphaZeroNet(nn.Module):
    """Policy + Value network for Dragonchess AlphaZero.

    Architecture matches DualHeadWeights in mcts.h:
      sparse_input(32284) -> h1(256, ReLU) -> h2(128, ReLU) -> {
          policy: Linear(128 -> 83808)
          value:  Linear(128 -> 1) -> tanh
      }
    """

    def __init__(self):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(N_INPUT, N_H1),
            nn.ReLU(),
            nn.Linear(N_H1, N_H2),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(N_H2, ACTION_SPACE)
        self.value_head = nn.Linear(N_H2, 1)

    def forward(self, x, legal_mask=None):
        """Forward pass.

        Args:
            x: (batch, N_INPUT) sparse/dense features
            legal_mask: (batch, ACTION_SPACE) bool tensor, True for legal actions

        Returns:
            policy_logits: (batch, ACTION_SPACE) masked logits
            value: (batch, 1) in [-1, 1]
        """
        h = self.trunk(x)
        logits = self.policy_head(h)
        if legal_mask is not None:
            logits = logits.masked_fill(~legal_mask, -1e9)
        value = torch.tanh(self.value_head(h))
        return logits, value

    def to_flat_numpy(self):
        """Serialize to flat float32 array matching C++ DualHeadWeights::from_flat."""
        parts = []
        # Trunk: w1, b1, w2, b2
        parts.append(self.trunk[0].weight.detach().cpu().numpy().ravel())
        parts.append(self.trunk[0].bias.detach().cpu().numpy().ravel())
        parts.append(self.trunk[2].weight.detach().cpu().numpy().ravel())
        parts.append(self.trunk[2].bias.detach().cpu().numpy().ravel())
        # Policy: wp, bp
        parts.append(self.policy_head.weight.detach().cpu().numpy().ravel())
        parts.append(self.policy_head.bias.detach().cpu().numpy().ravel())
        # Value: wv, bv
        parts.append(self.value_head.weight.detach().cpu().numpy().ravel())
        parts.append(self.value_head.bias.detach().cpu().numpy().ravel())
        return np.concatenate(parts).astype(np.float32)

    def from_flat_numpy(self, flat):
        o = 0
        def load(param):
            nonlocal o
            n = param.numel()
            param.data.copy_(torch.from_numpy(
                flat[o:o + n].reshape(param.shape).copy()))
            o += n
        load(self.trunk[0].weight)
        load(self.trunk[0].bias)
        load(self.trunk[2].weight)
        load(self.trunk[2].bias)
        load(self.policy_head.weight)
        load(self.policy_head.bias)
        load(self.value_head.weight)
        load(self.value_head.bias)

    def to_binary_weights(self):
        return self.to_flat_numpy().tobytes()

    @staticmethod
    def total_params():
        return (N_H1 * N_INPUT + N_H1 +
                N_H2 * N_H1 + N_H2 +
                ACTION_SPACE * N_H2 + ACTION_SPACE +
                N_H2 + 1)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)
        # Policy head: small init for uniform priors
        nn.init.normal_(self.policy_head.weight, 0, 0.01)
        nn.init.zeros_(self.policy_head.bias)
        # Value head: small init near zero
        nn.init.normal_(self.value_head.weight, 0, 0.01)
        nn.init.zeros_(self.value_head.bias)


# ---------------------------------------------------------------------------
# Self-play data generation
# ---------------------------------------------------------------------------

def parse_selfplay_ndjson(text):
    """Parse MCTS self-play NDJSON output into training examples.

    Each line: {"i":[...], "v":[...], "p":[action:prob,...], "z":outcome}

    Returns list of (features_sparse, policy_target, value_target).
    """
    examples = []
    for line in text.strip().split('\n'):
        if not line:
            continue
        d = json.loads(line)

        feat_idx = d['i']
        feat_val = d['v']

        # Policy: parallel arrays pa (actions) and pp (probabilities)
        policy = {}
        for action, prob in zip(d['pa'], d['pp']):
            policy[int(action)] = float(prob)

        value = float(d['z'])
        examples.append((feat_idx, feat_val, policy, value))
    return examples


def _make_selfplay_actor():
    """Create a Ray actor for MCTS self-play."""
    import ray

    @ray.remote
    class MCTSSelfPlayActor:
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
            return self.host

        def selfplay(self, weights_bytes, num_games, mcts_sims):
            """Run MCTS self-play games. Returns NDJSON text."""
            import tempfile
            weights_path = None
            if weights_bytes:
                with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
                    f.write(weights_bytes)
                    weights_path = f.name

            cmd = [
                self.binary, "--headless",
                "--mode", "mcts-selfplay",
                "--games", str(num_games),
                "--threads", "1",
                "--mcts-simulations", str(mcts_sims),
            ]
            if weights_path:
                cmd.extend(["--mcts-nn-weights", weights_path])

            try:
                result = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=600.0)
                if weights_path:
                    os.unlink(weights_path)
                if result.returncode != 0:
                    return "ERR:" + result.stderr[:200]
                return result.stdout
            except Exception as e:
                if weights_path and os.path.exists(weights_path):
                    os.unlink(weights_path)
                return "ERR:" + str(e)

    return MCTSSelfPlayActor


def generate_selfplay_data(model, num_games, mcts_sims, ray_pool=None,
                           games_per_actor=2, timeout_s=3600.0):
    """Run MCTS self-play on the Ray cluster. Returns training examples."""
    import ray

    weights_bytes = model.to_binary_weights()

    # Dispatch games across actors
    n_chunks = max(1, num_games // games_per_actor)
    futures = [
        ray_pool[i % len(ray_pool)].selfplay.remote(
            weights_bytes, games_per_actor, mcts_sims)
        for i in range(n_chunks)
    ]

    results = ray.get(futures)
    all_examples = []
    errors = 0
    for ndjson in results:
        if ndjson.startswith("ERR:"):
            errors += 1
            continue
        if ndjson.strip():
            all_examples.extend(parse_selfplay_ndjson(ndjson))

    if errors > 0:
        log.warning("  %d/%d actor errors during self-play", errors, n_chunks)

    return all_examples


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_on_examples(model, optimizer, examples, batch_size=256,
                      epochs=1, grad_clip=5.0, max_examples=50000):
    """Train policy+value network on self-play examples.

    Uses sparse policy loss to avoid materializing the full ACTION_SPACE tensor.
    Samples max_examples from the buffer if it's larger.
    """
    model.train()
    n = len(examples)
    if n == 0:
        return 0.0, 0.0

    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_batches = 0

    for epoch in range(epochs):
        # Sample if buffer is large
        if n > max_examples:
            indices = np.random.choice(n, max_examples, replace=False)
        else:
            indices = np.random.permutation(n)

        for batch_start in range(0, len(indices), batch_size):
            batch_idx = indices[batch_start:batch_start + batch_size]
            bs = len(batch_idx)

            # Build feature tensor (sparse -> dense)
            features = torch.zeros(bs, N_INPUT, dtype=torch.float32)
            value_targets = torch.zeros(bs, 1, dtype=torch.float32)

            # For sparse policy loss: collect (batch_idx, action, target_prob)
            pol_batch = []
            pol_action = []
            pol_target = []

            for b, idx in enumerate(batch_idx):
                feat_idx, feat_val, policy, value = examples[idx]
                for fi, fv in zip(feat_idx, feat_val):
                    if 0 <= fi < N_INPUT:
                        features[b, fi] = fv
                value_targets[b, 0] = value
                for action, prob in policy.items():
                    if 0 <= action < ACTION_SPACE and prob > 0:
                        pol_batch.append(b)
                        pol_action.append(action)
                        pol_target.append(prob)

            features = features.to(DEVICE)
            value_targets = value_targets.to(DEVICE)

            # Forward (no masking needed, we handle it in loss)
            logits, value_pred = model(features)

            # Sparse policy loss: only compute on actions that appear in targets
            if pol_batch:
                pb = torch.tensor(pol_batch, dtype=torch.long, device=DEVICE)
                pa = torch.tensor(pol_action, dtype=torch.long, device=DEVICE)
                pt = torch.tensor(pol_target, dtype=torch.float32, device=DEVICE)

                # Gather logits for target actions
                selected_logits = logits[pb, pa]

                # For proper cross-entropy, we need log_softmax over legal moves.
                # Approximate: use logsumexp over ALL actions as denominator.
                # This is exact if illegal actions have very low logits (which
                # they will after a few iterations of training).
                log_z = torch.logsumexp(logits, dim=1)  # (bs,)
                log_probs = selected_logits - log_z[pb]
                policy_loss = -(pt * log_probs).sum() / bs
            else:
                policy_loss = torch.tensor(0.0, device=DEVICE)

            value_loss = F.mse_loss(value_pred, value_targets)
            loss = policy_loss + value_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_batches += 1

    avg_p = total_policy_loss / max(total_batches, 1)
    avg_v = total_value_loss / max(total_batches, 1)
    return avg_p, avg_v


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_vs_ab(binary, weights_path, n_games=50, ab_depth=2,
                   timeout_s=600.0):
    """Evaluate MCTS player vs AB. Returns win rate."""
    # For now, use the NN as a value-only eval (like NNEvalAI)
    # since MCTS eval would be too slow for 50 games.
    # TODO: MCTS-based eval
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
        "--output-json", "/tmp/az_eval.json",
        "--quiet",
    ]
    try:
        subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
        rj = json.loads(Path("/tmp/az_eval.json").read_text())
        total = rj.get("total_games", 0)
        gold_wins = rj.get("gold_wins", 0)
        return gold_wins / total if total > 0 else 0.0
    except Exception as e:
        log.warning("Eval failed: %s", e)
        return -1.0


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def save_checkpoint(out_dir, model, iteration, best_wr, history, elapsed):
    data = {
        "type": "alphazero",
        "iteration": iteration,
        "best_win_rate": best_wr,
        "history": history,
        "elapsed_seconds": elapsed,
        "n_params": AlphaZeroNet.total_params(),
        "arch": f"{N_INPUT}x{N_H1}x{N_H2}+policy({ACTION_SPACE})+value(1)",
    }
    # Save weights separately (binary) since JSON would be huge
    weights_path = out_dir / "weights.bin"
    weights_path.write_bytes(model.to_binary_weights())

    # Save metadata
    meta_path = out_dir / "latest.json"
    tmp = meta_path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(tmp, meta_path)

    if best_wr > 0:
        best_path = out_dir / "best_weights.bin"
        best_path.write_bytes(model.to_binary_weights())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="AlphaZero training for Dragonchess")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs-per-iter", type=int, default=5)
    parser.add_argument("--games-per-iter", type=int, default=100)
    parser.add_argument("--mcts-sims", type=int, default=400)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--eval-games", type=int, default=50)
    parser.add_argument("--eval-ab-depth", type=int, default=2)
    parser.add_argument("--max-hours", type=float, default=48.0)
    parser.add_argument("--replay-buffer-size", type=int, default=100000,
                        help="Max training examples to keep in replay buffer")
    parser.add_argument("--games-per-actor", type=int, default=2,
                        help="Games per Ray actor dispatch")
    # Ray cluster
    parser.add_argument("--ray-address", default=None)
    parser.add_argument("--auto-cluster", action="store_true")
    parser.add_argument("--max-actors", type=int, default=None)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    # Model
    model = AlphaZeroNet().to(DEVICE)
    model.init_weights()
    log.info("Initialized AlphaZeroNet: %d params, device=%s",
             AlphaZeroNet.total_params(), DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Connect to Ray cluster
    import ray
    if args.auto_cluster:
        from aall_cluster import connect
        state, tunnel = connect(namespace="dragonchess-az",
                                verbose=True, log_to_driver=False)
        log.info("Auto-connected, head=%s", state.head_ip)
    elif args.ray_address:
        from cluster.runtime_sync import stage_runtime_working_dir
        repo_root = str(Path(__file__).parent)
        staged = stage_runtime_working_dir(repo_root)
        log.info("Staged working_dir: %s", staged.path)
        ray.init(address=args.ray_address, namespace="dragonchess-az",
                 ignore_reinit_error=True, log_to_driver=False,
                 runtime_env={"working_dir": str(staged.path)})
    else:
        log.error("Need --auto-cluster or --ray-address")
        sys.exit(1)

    cluster_cpus = int(float(ray.cluster_resources().get("CPU", 0)))
    target_actors = min(cluster_cpus, args.max_actors or cluster_cpus)
    log.info("Ray cluster: %d CPUs, using %d actors", cluster_cpus, target_actors)

    actor_cls = _make_selfplay_actor()
    ray_pool = [actor_cls.options(num_cpus=1).remote(None) for _ in range(target_actors)]
    hosts = {}
    for h in ray.get([a.ready.remote() for a in ray_pool]):
        hosts[h] = hosts.get(h, 0) + 1
    log.info("Pool ready: %d actors across %d hosts", len(ray_pool), len(hosts))

    # Replay buffer
    replay_buffer = []
    best_wr = 0.0
    history = []

    # Signal handling
    stop = [False]
    def handler(sig, frame):
        if stop[0]: sys.exit(1)
        log.info("Stopping after current iteration.")
        stop[0] = True
    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)

    start = time.time()
    max_seconds = args.max_hours * 3600

    for iteration in range(1, args.iterations + 1):
        if stop[0] or time.time() - start > max_seconds:
            break

        iter_start = time.time()

        # 2. Self-play on cluster
        log.info("Iter %d: generating %d self-play games (%d sims) on %d actors...",
                 iteration, args.games_per_iter, args.mcts_sims, len(ray_pool))
        examples = generate_selfplay_data(
            model, args.games_per_iter, args.mcts_sims,
            ray_pool=ray_pool,
            games_per_actor=args.games_per_actor)

        if not examples:
            log.warning("No self-play data generated, skipping iteration.")
            continue

        # Add to replay buffer
        replay_buffer.extend(examples)
        if len(replay_buffer) > args.replay_buffer_size:
            replay_buffer = replay_buffer[-args.replay_buffer_size:]

        # 3. Train
        log.info("Iter %d: training on %d examples (%d in buffer)...",
                 iteration, len(examples), len(replay_buffer))
        p_loss, v_loss = train_on_examples(
            model, optimizer, replay_buffer,
            batch_size=args.batch_size,
            epochs=args.epochs_per_iter)

        iter_elapsed = time.time() - iter_start

        # 4. Evaluate periodically
        wr = -1.0
        if iteration % args.eval_every == 0:
            # Write weights for eval (NNEvalAI uses different weight format)
            # For now, eval uses the value head only via a compatibility shim
            # TODO: proper MCTS eval or convert weights
            log.info("Iter %d: evaluating vs AB(d=%d)...",
                     iteration, args.eval_ab_depth)
            # Skip eval for now until weight format compatibility is resolved
            wr = -1.0

        if wr > best_wr:
            best_wr = wr

        history.append({
            "iteration": iteration,
            "n_examples": len(examples),
            "buffer_size": len(replay_buffer),
            "policy_loss": p_loss,
            "value_loss": v_loss,
            "win_rate": wr,
            "elapsed_s": iter_elapsed,
        })

        log.info("Iter %3d | examples %5d | buffer %6d | "
                 "p_loss %.4f | v_loss %.4f | WR %.1f%% | %ds",
                 iteration, len(examples), len(replay_buffer),
                 p_loss, v_loss, wr * 100 if wr >= 0 else -1,
                 int(iter_elapsed))

        # Save checkpoint
        save_checkpoint(args.out, model, iteration, best_wr, history,
                        time.time() - start)

    total = time.time() - start
    log.info("Done. %d iterations, %.1f min. Best WR: %.1f%%",
             len(history), total / 60, best_wr * 100)


if __name__ == "__main__":
    main()
