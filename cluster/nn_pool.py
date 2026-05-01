"""
NN-oriented Ray actor pool for DragonchessAI.

Mirrors cluster/evaluator_pool.py but supports the NN trainer:
  - Weights are ~1.1M float32 values (4.4MB), passed to actors as raw bytes
    via the Ray object store (ray.put -> ObjectRef).
  - Each actor writes the bytes to a per-actor temp file and invokes
    `dragonchess --headless --mode selfplay|tournament --nn-weights <path>`.
  - The pool exposes two high-level fanout helpers:
        selfplay_batch(weights_bytes, total_games, chunk_size, ...)
        tournament_batch(weights_bytes, total_games, chunk_size, ...)
    Both return aggregated results (NDJSON string / win rate dict).

Design notes:
  - Weights are ray.put() ONCE per batch. All chunks share the same ObjectRef,
    so Ray transfers the object at most once per worker node per batch.
  - Actors keep a per-actor tmp weights file path and only rewrite when the
    incoming weights hash differs from the last one they wrote.
  - Actors are num_cpus=1; we create as many actors as cluster CPU capacity
    (optionally capped) so that one actor maps to one worker CPU slot.
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
import subprocess
import tempfile
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

# Lazy import: cluster.evaluator_pool is imported inside the Ray actor
# __init__ so workers can find it via repo_dir on sys.path.


class NNEvalError(RuntimeError):
    """Raised when a distributed NN self-play or eval subprocess fails."""


@dataclass(frozen=True)
class NNSelfplayRequest:
    games: int
    td_depth: int
    threads: int = 1
    timeout_s: float = 900.0
    opponent: str = "self"       # "self" = NN vs NN, "ab" = NN vs AB(depth)
    opponent_depth: int = 2


@dataclass(frozen=True)
class NNTournamentRequest:
    games: int
    ab_depth: int
    td_depth: int = 2
    threads: int = 1
    timeout_s: float = 900.0


@dataclass
class NNTournamentResult:
    gold_wins: int
    total_games: int
    summary: dict[str, Any] = field(default_factory=dict)
    host: str = ""

    @property
    def win_rate(self) -> float:
        return (self.gold_wins / self.total_games) if self.total_games > 0 else 0.0


# ---------------------------------------------------------------------------
# Subprocess helpers (also used by the Ray actor body)
# ---------------------------------------------------------------------------


def _run_selfplay_subprocess(
    binary_path: str,
    weights_path: str,
    request: NNSelfplayRequest,
) -> str:
    if request.opponent == "ab":
        cmd = [
            binary_path,
            "--headless",
            "--mode", "selfplay",
            "--gold-nn-weights", weights_path,
            "--gold-depth", str(int(request.td_depth)),
            "--scarlet-ai", "alphabeta",
            "--scarlet-depth", str(int(request.opponent_depth)),
            "--games", str(int(request.games)),
            "--threads", str(max(1, int(request.threads))),
            "--quiet",
        ]
    else:
        cmd = [
            binary_path,
            "--headless",
            "--mode", "selfplay",
            "--nn-weights", weights_path,
            "--games", str(int(request.games)),
            "--threads", str(max(1, int(request.threads))),
            "--td-depth", str(int(request.td_depth)),
            "--quiet",
        ]
    try:
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=float(request.timeout_s),
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise NNEvalError(
            f"selfplay timed out after {request.timeout_s:.0f}s: {' '.join(cmd)}"
        ) from exc
    if completed.returncode != 0:
        err = (completed.stderr or completed.stdout or "").strip()[:500]
        raise NNEvalError(
            f"selfplay exited {completed.returncode}: {err}"
        )
    return completed.stdout


def _run_tournament_subprocess(
    binary_path: str,
    weights_path: str,
    request: NNTournamentRequest,
) -> NNTournamentResult:
    cmd = [
        binary_path,
        "--headless",
        "--mode", "tournament",
        "--gold-nn-weights", weights_path,
        "--scarlet-ai", "alphabeta",
        "--scarlet-depth", str(int(request.ab_depth)),
        "--games", str(int(request.games)),
        "--threads", str(max(1, int(request.threads))),
        "--output-json", "-",
        "--quiet",
    ]
    try:
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=float(request.timeout_s),
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise NNEvalError(
            f"tournament timed out after {request.timeout_s:.0f}s: {' '.join(cmd)}"
        ) from exc
    if completed.returncode != 0:
        err = (completed.stderr or completed.stdout or "").strip()[:500]
        raise NNEvalError(
            f"tournament exited {completed.returncode}: {err}"
        )
    try:
        data = json.loads(completed.stdout)
        summary = data.get("summary", data)
        gold_wins = int(summary["gold_wins"])
        total_games = int(summary["total_games"])
    except Exception as exc:
        raise NNEvalError(
            f"failed to parse tournament JSON: {completed.stdout[:500]}"
        ) from exc
    return NNTournamentResult(
        gold_wins=gold_wins,
        total_games=total_games,
        summary=summary,
        host=socket.gethostname(),
    )


# ---------------------------------------------------------------------------
# Ray actor
# ---------------------------------------------------------------------------


def _make_ray_nn_actor():
    import ray

    @ray.remote
    class RayNNActor:
        def __init__(self, repo_dir: str | None, slot_name: str) -> None:
            import sys
            if repo_dir:
                rdir = str(Path(repo_dir).expanduser().resolve())
                if rdir not in sys.path:
                    sys.path.insert(0, rdir)
            from cluster.evaluator_pool import binary_path_from_repo, materialize_executable
            self.binary_path = materialize_executable(binary_path_from_repo(repo_dir))
            self.slot_name = slot_name
            self.host = socket.gethostname()
            self._weights_dir = Path(tempfile.mkdtemp(prefix="dc-nn-actor-"))
            self._weights_path = self._weights_dir / "weights.bin"
            self._weights_hash: str | None = None

        def ready(self) -> dict[str, str]:
            return {
                "host": self.host,
                "slot": self.slot_name,
                "binary": self.binary_path,
                "weights_dir": str(self._weights_dir),
            }

        def _ensure_weights(self, weights_bytes: bytes) -> str:
            digest = hashlib.sha1(weights_bytes).hexdigest()
            if digest != self._weights_hash or not self._weights_path.exists():
                tmp = self._weights_path.with_suffix(".bin.tmp")
                with open(tmp, "wb") as f:
                    f.write(weights_bytes)
                os.replace(tmp, self._weights_path)
                self._weights_hash = digest
            return str(self._weights_path)

        def selfplay(
            self,
            weights_bytes: bytes,
            request_dict: dict[str, Any],
        ) -> dict[str, Any]:
            weights_path = self._ensure_weights(weights_bytes)
            request = NNSelfplayRequest(**request_dict)
            ndjson = _run_selfplay_subprocess(
                self.binary_path, weights_path, request
            )
            return {"ndjson": ndjson, "host": self.host, "slot": self.slot_name}

        def tournament(
            self,
            weights_bytes: bytes,
            request_dict: dict[str, Any],
        ) -> dict[str, Any]:
            weights_path = self._ensure_weights(weights_bytes)
            request = NNTournamentRequest(**request_dict)
            result = _run_tournament_subprocess(
                self.binary_path, weights_path, request
            )
            payload = asdict(result)
            payload["host"] = self.host
            payload["slot"] = self.slot_name
            return payload

    return RayNNActor


# ---------------------------------------------------------------------------
# Pool
# ---------------------------------------------------------------------------


class RayNNPool:
    """
    Ray-backed actor pool for distributed NN self-play and eval.

    Usage:
        pool = RayNNPool(repo_dir="~/DragonchessAI", max_actors=400)
        pool.start()
        ndjson = pool.selfplay_batch(weights_bytes, total_games=400, chunk_size=2, td_depth=2)
        result = pool.tournament_batch(weights_bytes, total_games=400, chunk_size=10, ab_depth=2)
        pool.shutdown()
    """

    def __init__(
        self,
        *,
        repo_dir: str | None = None,
        max_actors: int | None = None,
        max_request_retries: int = 6,
    ) -> None:
        self.repo_dir = repo_dir
        self.max_actors = max_actors
        self.max_request_retries = max(0, int(max_request_retries))
        self.actors: list[Any] = []
        self._next_actor = 0
        self._lock = threading.Lock()
        self._hosts: dict[str, int] = {}

    # ---- lifecycle -------------------------------------------------------

    def start(self) -> None:
        import ray

        if self.actors:
            return

        cluster_cpus = int(float(ray.cluster_resources().get("CPU", 0)))
        if cluster_cpus <= 0:
            raise RuntimeError("Ray reports 0 CPUs; no actors can be created")

        target = cluster_cpus
        if self.max_actors is not None:
            target = min(target, int(self.max_actors))
        if target <= 0:
            raise RuntimeError(f"Invalid target actor count: {target}")

        actor_cls = _make_ray_nn_actor()
        for i in range(target):
            slot = f"nn-slot-{i + 1:04d}"
            actor = actor_cls.options(num_cpus=1).remote(self.repo_dir, slot)
            self.actors.append(actor)

        # Block until all actors are ready; record host distribution.
        ready_info = ray.get([actor.ready.remote() for actor in self.actors])
        for info in ready_info:
            host = str(info.get("host", "?"))
            self._hosts[host] = self._hosts.get(host, 0) + 1

    @property
    def actor_count(self) -> int:
        return len(self.actors)

    def describe_capacity(self) -> dict[str, int]:
        return dict(self._hosts)

    def shutdown(self) -> None:
        try:
            import ray
        except ImportError:
            return
        for actor in self.actors:
            try:
                ray.kill(actor, no_restart=True)
            except Exception:
                pass
        self.actors = []
        self._hosts = {}

    # ---- scheduling helpers ---------------------------------------------

    def _pick_actor(self, excluded: Sequence[Any] = ()) -> Any:
        excluded_ids = {id(actor) for actor in excluded}
        with self._lock:
            if not self.actors:
                raise RuntimeError("No NN actors available")
            count = len(self.actors)
            for offset in range(count):
                idx = (self._next_actor + offset) % count
                actor = self.actors[idx]
                if id(actor) in excluded_ids:
                    continue
                self._next_actor = (idx + 1) % count
                return actor
        raise RuntimeError("No NN actors available for retry")

    def _remove_actor(self, actor: Any) -> bool:
        with self._lock:
            before = len(self.actors)
            self.actors = [a for a in self.actors if a is not actor]
            if len(self.actors) == before:
                return False
            if self.actors:
                self._next_actor %= len(self.actors)
            else:
                self._next_actor = 0
            return True

    def _should_evict_actor(self, exc: Exception) -> bool:
        import ray

        types = tuple(
            getattr(ray.exceptions, name)
            for name in ("RayActorError", "ActorDiedError", "ActorUnavailableError")
            if hasattr(ray.exceptions, name)
        )
        return isinstance(exc, types)

    # ---- fanout primitives ----------------------------------------------

    def _fanout(
        self,
        weights_ref: Any,
        method_name: str,
        payloads: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Submit one remote call per payload, with retry on transient actor death.
        Returns list of result dicts aligned with `payloads`.
        """
        import ray

        if not payloads:
            return []
        if not self.actors:
            raise RuntimeError(
                "RayNNPool.start() must be called before submitting work"
            )

        results: list[dict[str, Any] | None] = [None] * len(payloads)
        attempts = [0] * len(payloads)
        pending: dict[Any, tuple[int, Any]] = {}

        def submit(index: int, excluded: Sequence[Any] = ()) -> None:
            actor = self._pick_actor(excluded)
            attempts[index] += 1
            method = getattr(actor, method_name)
            ref = method.remote(weights_ref, payloads[index])
            pending[ref] = (index, actor)

        for i in range(len(payloads)):
            submit(i)

        while pending:
            ready, _ = ray.wait(list(pending.keys()), num_returns=1)
            ref = ready[0]
            index, actor = pending.pop(ref)
            try:
                results[index] = ray.get(ref)
            except Exception as exc:
                excluded: tuple[Any, ...] = ()
                if self._should_evict_actor(exc):
                    self._remove_actor(actor)
                else:
                    excluded = (actor,)
                if attempts[index] <= self.max_request_retries:
                    submit(index, excluded)
                    continue
                raise NNEvalError(
                    f"{method_name} request {index} failed after "
                    f"{self.max_request_retries} retries: {exc}"
                ) from exc

        return [r for r in results if r is not None]

    @staticmethod
    def _chunk_games(total_games: int, chunk_size: int) -> list[int]:
        chunk_size = max(1, int(chunk_size))
        total_games = max(0, int(total_games))
        if total_games == 0:
            return []
        full, remainder = divmod(total_games, chunk_size)
        chunks = [chunk_size] * full
        if remainder:
            chunks.append(remainder)
        return chunks

    # ---- high-level APIs ------------------------------------------------

    def selfplay_batch(
        self,
        weights_bytes: bytes,
        *,
        total_games: int,
        chunk_size: int,
        td_depth: int,
        threads_per_chunk: int = 1,
        timeout_s: float = 900.0,
        opponent: str = "self",
        opponent_depth: int = 2,
    ) -> str:
        """
        Run `total_games` selfplay games across the cluster in chunks of
        `chunk_size`. Returns concatenated NDJSON.
        opponent: "self" for NN vs NN, "ab" for NN vs AB(opponent_depth)
        """
        import ray

        chunks = self._chunk_games(total_games, chunk_size)
        if not chunks:
            return ""

        weights_ref = ray.put(weights_bytes)
        payloads = [
            asdict(NNSelfplayRequest(
                games=n,
                td_depth=td_depth,
                threads=threads_per_chunk,
                timeout_s=timeout_s,
                opponent=opponent,
                opponent_depth=opponent_depth,
            ))
            for n in chunks
        ]
        results = self._fanout(weights_ref, "selfplay", payloads)
        return "".join(r.get("ndjson", "") for r in results)

    def tournament_batch(
        self,
        weights_bytes: bytes,
        *,
        total_games: int,
        chunk_size: int,
        ab_depth: int,
        td_depth: int = 2,
        threads_per_chunk: int = 1,
        timeout_s: float = 900.0,
    ) -> NNTournamentResult:
        """
        Run `total_games` tournament eval games vs AlphaBeta across the cluster
        in chunks of `chunk_size`. Returns aggregated NNTournamentResult.
        """
        import ray

        chunks = self._chunk_games(total_games, chunk_size)
        if not chunks:
            return NNTournamentResult(gold_wins=0, total_games=0)

        weights_ref = ray.put(weights_bytes)
        payloads = [
            asdict(NNTournamentRequest(
                games=n,
                ab_depth=ab_depth,
                td_depth=td_depth,
                threads=threads_per_chunk,
                timeout_s=timeout_s,
            ))
            for n in chunks
        ]
        results = self._fanout(weights_ref, "tournament", payloads)

        gold_wins = 0
        total = 0
        for r in results:
            gold_wins += int(r.get("gold_wins", 0))
            total += int(r.get("total_games", 0))
        return NNTournamentResult(
            gold_wins=gold_wins,
            total_games=total,
            summary={"chunks": len(results)},
        )
