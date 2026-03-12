from __future__ import annotations

import json
import os
import socket
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from cluster.worker_config import WorkerSpec, load_workers_csv


DEFAULT_BINARY_PATH = Path(__file__).resolve().parents[1] / "build" / "dragonchess"


class EvaluationError(RuntimeError):
    """Raised when a tournament evaluation fails."""


@dataclass(frozen=True)
class EvaluationRequest:
    gold_ai: str
    scarlet_ai: str
    games: int
    gold_depth: int = 1
    scarlet_depth: int = 2
    gold_weights: Sequence[float] | None = None
    scarlet_weights: Sequence[float] | None = None
    threads: int = 1
    timeout_s: float = 300.0
    max_moves: int = 1000
    quiet: bool = True


@dataclass
class EvaluationResult:
    gold_wins: int
    total_games: int
    win_rate: float
    summary: dict[str, Any] = field(default_factory=dict)
    host: str = ""
    slot: str = ""


def weights_to_str(weights: Sequence[float]) -> str:
    return ",".join(f"{weight:.6f}" for weight in weights)


def binary_path_from_repo(repo_dir: str | os.PathLike[str]) -> str:
    return str(Path(repo_dir).expanduser().resolve() / "build" / "dragonchess")


def _normalize_request(request: EvaluationRequest, default_threads: int) -> EvaluationRequest:
    if request.threads > 0:
        return request
    return EvaluationRequest(**{**asdict(request), "threads": int(default_threads)})


def build_command(binary_path: str, request: EvaluationRequest) -> list[str]:
    command = [
        binary_path,
        "--headless",
        "--mode",
        "tournament",
        "--games",
        str(int(request.games)),
        "--threads",
        str(int(request.threads)),
        "--max-moves",
        str(int(request.max_moves)),
        "--gold-ai",
        request.gold_ai,
        "--gold-depth",
        str(int(request.gold_depth)),
        "--scarlet-ai",
        request.scarlet_ai,
        "--scarlet-depth",
        str(int(request.scarlet_depth)),
        "--output-json",
        "-",
    ]
    if request.gold_weights is not None:
        command.extend(["--gold-weights", weights_to_str(request.gold_weights)])
    if request.scarlet_weights is not None:
        command.extend(["--scarlet-weights", weights_to_str(request.scarlet_weights)])
    if request.quiet:
        command.append("--quiet")
    return command


def run_evaluation_request(binary_path: str, request: EvaluationRequest) -> EvaluationResult:
    command = build_command(binary_path, request)
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=float(request.timeout_s),
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise EvaluationError(
            f"Timed out after {request.timeout_s:.1f}s running {' '.join(command)}"
        ) from exc

    if completed.returncode != 0:
        stderr = (completed.stderr or completed.stdout or "").strip()
        raise EvaluationError(
            f"Evaluator command failed with code {completed.returncode}: {stderr}"
        )

    try:
        data = json.loads(completed.stdout)
        summary = data["summary"]
        gold_wins = int(summary["gold_wins"])
        total_games = int(summary["total_games"])
    except Exception as exc:
        raise EvaluationError("Failed to parse evaluator JSON output") from exc

    win_rate = (gold_wins / total_games) if total_games > 0 else 0.0
    return EvaluationResult(
        gold_wins=gold_wins,
        total_games=total_games,
        win_rate=win_rate,
        summary=summary,
        host=socket.gethostname(),
    )


class LocalEvaluatorPool:
    def __init__(
        self,
        *,
        max_workers: int,
        binary_path: str | None = None,
        threads_per_eval: int = 1,
    ) -> None:
        self.binary_path = binary_path or str(DEFAULT_BINARY_PATH)
        self.threads_per_eval = max(1, int(threads_per_eval))
        self.executor = ThreadPoolExecutor(max_workers=max(1, int(max_workers)))

    def evaluate_many(self, requests: Iterable[EvaluationRequest]) -> list[EvaluationResult]:
        request_list = [
            _normalize_request(request, self.threads_per_eval) for request in requests
        ]
        futures = [
            self.executor.submit(run_evaluation_request, self.binary_path, request)
            for request in request_list
        ]
        return [future.result() for future in futures]

    def shutdown(self) -> None:
        self.executor.shutdown(wait=True)


def _resolve_ray_node_id(worker: WorkerSpec, nodes: list[dict[str, Any]]) -> str | None:
    candidates = {worker.ip_address, worker.hostname}
    try:
        if worker.hostname:
            candidates.add(socket.gethostbyname(worker.hostname))
    except Exception:
        pass

    for node in nodes:
        if not node.get("Alive"):
            continue
        address = str(node.get("NodeManagerAddress") or node.get("node_ip_address") or "")
        node_id = str(node.get("NodeID") or "")
        if address in candidates and node_id:
            return node_id
    return None


def _make_ray_evaluator_actor():
    import ray

    @ray.remote
    class RayTournamentEvaluator:
        def __init__(self, repo_dir: str, slot_name: str, threads_per_eval: int) -> None:
            self.binary_path = binary_path_from_repo(repo_dir)
            self.slot_name = slot_name
            self.threads_per_eval = max(1, int(threads_per_eval))
            self.host = socket.gethostname()

        def ready(self) -> dict[str, str]:
            return {"host": self.host, "slot": self.slot_name, "binary": self.binary_path}

        def evaluate(self, payload: dict[str, Any]) -> dict[str, Any]:
            request = _normalize_request(
                EvaluationRequest(**payload),
                self.threads_per_eval,
            )
            result = run_evaluation_request(self.binary_path, request)
            result.host = self.host
            result.slot = self.slot_name
            return asdict(result)

    return RayTournamentEvaluator


class RayEvaluatorPool:
    def __init__(
        self,
        *,
        workers_csv: str,
        repo_dir: str,
        threads_per_eval: int = 1,
    ) -> None:
        self.workers_csv = workers_csv
        self.repo_dir = repo_dir
        self.threads_per_eval = max(1, int(threads_per_eval))
        self.actors: list[Any] = []
        self.capacity_by_host: dict[str, int] = {}
        self._next_actor = 0
        self._lock = threading.Lock()

    def start(self) -> None:
        import ray
        from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

        if self.actors:
            return

        workers = load_workers_csv(self.workers_csv, include_disabled=False)
        nodes = ray.nodes()
        actor_cls = _make_ray_evaluator_actor()
        missing: list[tuple[str, str]] = []

        for worker in workers:
            node_id = _resolve_ray_node_id(worker, nodes)
            if not node_id:
                missing.append((worker.hostname, worker.ip_address))
                continue

            self.capacity_by_host[worker.display_name] = int(worker.core_usage)
            for slot_index in range(int(worker.core_usage)):
                slot_name = f"{worker.display_name}-slot-{slot_index + 1}"
                actor = actor_cls.options(
                    scheduling_strategy=NodeAffinitySchedulingStrategy(node_id, soft=False),
                    num_cpus=1,
                ).remote(self.repo_dir, slot_name, self.threads_per_eval)
                self.actors.append(actor)

        if missing:
            alive = [
                str(node.get("NodeManagerAddress") or node.get("node_ip_address") or "")
                for node in nodes
                if node.get("Alive")
            ]
            raise RuntimeError(
                "Could not match some workers.csv entries to Ray nodes. "
                f"Missing={missing}, alive_nodes={alive}"
            )

        if not self.actors:
            raise RuntimeError("No evaluator actors were created from workers.csv")

        ray.get([actor.ready.remote() for actor in self.actors])

    @property
    def actor_count(self) -> int:
        return len(self.actors)

    def evaluate_many(self, requests: Iterable[EvaluationRequest]) -> list[EvaluationResult]:
        import ray

        request_list = [
            _normalize_request(request, self.threads_per_eval) for request in requests
        ]
        if not request_list:
            return []
        if not self.actors:
            raise RuntimeError("RayEvaluatorPool.start() must be called before evaluate_many()")

        with self._lock:
            start = self._next_actor
            self._next_actor = (self._next_actor + len(request_list)) % len(self.actors)

        refs = []
        for index, request in enumerate(request_list):
            actor = self.actors[(start + index) % len(self.actors)]
            refs.append(actor.evaluate.remote(asdict(request)))

        payloads = ray.get(refs)
        return [EvaluationResult(**payload) for payload in payloads]

    def describe_capacity(self) -> dict[str, int]:
        return dict(self.capacity_by_host)

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
        self.capacity_by_host = {}
