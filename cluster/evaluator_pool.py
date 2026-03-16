from __future__ import annotations

import hashlib
import json
import os
import shutil
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


def binary_path_from_repo(repo_dir: str | os.PathLike[str] | None) -> str:
    if repo_dir is None or str(repo_dir).strip() == "":
        base_dir = Path.cwd()
    else:
        base_dir = Path(repo_dir).expanduser()
    return str(base_dir.resolve() / "build" / "dragonchess")


def materialize_executable(binary_path: str | os.PathLike[str]) -> str:
    source = Path(binary_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"dragonchess binary not found at {source}")

    stat = source.stat()
    cache_root = Path.home() / ".cache" / "dragonchess-ray"
    cache_root.mkdir(parents=True, exist_ok=True)

    key = f"{source}:{stat.st_size}:{stat.st_mtime_ns}"
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
    cache_dir = cache_root / digest
    cache_dir.mkdir(parents=True, exist_ok=True)

    cached_binary = cache_dir / "dragonchess"
    if not cached_binary.exists() or cached_binary.stat().st_size != stat.st_size:
        temp_binary = cache_dir / f".dragonchess.tmp-{os.getpid()}-{threading.get_ident()}"
        try:
            shutil.copy2(source, temp_binary)
            temp_binary.chmod(0o755)
            os.replace(temp_binary, cached_binary)
        finally:
            if temp_binary.exists():
                temp_binary.unlink()
    else:
        cached_binary.chmod(0o755)

    return str(cached_binary)


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


def _node_cpu_capacity(node: dict[str, Any]) -> int:
    resources = node.get("Resources") or {}
    try:
        return max(0, int(float(resources.get("CPU", 0))))
    except Exception:
        return 0


def _make_ray_evaluator_actor():
    import ray

    @ray.remote
    class RayTournamentEvaluator:
        def __init__(self, repo_dir: str | None, slot_name: str, threads_per_eval: int) -> None:
            self.binary_path = materialize_executable(binary_path_from_repo(repo_dir))
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
        repo_dir: str | None = None,
        threads_per_eval: int = 1,
       max_request_retries: int = 10,
    ) -> None:
        self.workers_csv = workers_csv
        self.repo_dir = repo_dir
        self.threads_per_eval = max(1, int(threads_per_eval))
        self.max_request_retries = max(0, int(max_request_retries))
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

            node = next(
                (
                    candidate
                    for candidate in nodes
                    if str(candidate.get("NodeID") or "") == node_id
                ),
                None,
            )
            slot_count = _node_cpu_capacity(node or {})
            if slot_count <= 0:
                missing.append((worker.hostname, worker.ip_address))
                continue

            self.capacity_by_host[worker.display_name] = slot_count
            for slot_index in range(slot_count):
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

    def _pick_actor(self, excluded: Sequence[Any] = ()) -> Any:
        excluded_ids = {id(actor) for actor in excluded}
        with self._lock:
            if not self.actors:
                raise RuntimeError("No evaluator actors are available")

            actor_count = len(self.actors)
            for offset in range(actor_count):
                index = (self._next_actor + offset) % actor_count
                actor = self.actors[index]
                if id(actor) in excluded_ids:
                    continue
                self._next_actor = (index + 1) % actor_count
                return actor

        raise RuntimeError("No evaluator actors are available for retry")

    def _remove_actor(self, actor: Any) -> bool:
        with self._lock:
            original_count = len(self.actors)
            self.actors = [candidate for candidate in self.actors if candidate != actor]
            if len(self.actors) == original_count:
                return False
            if self.actors:
                self._next_actor %= len(self.actors)
            else:
                self._next_actor = 0
            return True

    def _should_evict_actor(self, exc: Exception) -> bool:
        import ray

        exception_types = tuple(
            getattr(ray.exceptions, name)
            for name in ("RayActorError", "ActorDiedError", "ActorUnavailableError")
            if hasattr(ray.exceptions, name)
        )
        return isinstance(exc, exception_types)

    def evaluate_many(self, requests: Iterable[EvaluationRequest]) -> list[EvaluationResult]:
        import ray

        request_list = [
            _normalize_request(request, self.threads_per_eval) for request in requests
        ]
        if not request_list:
            return []
        if not self.actors:
            raise RuntimeError("RayEvaluatorPool.start() must be called before evaluate_many()")

        payloads = [None] * len(request_list)
        attempts = [0] * len(request_list)
        pending: dict[Any, tuple[int, Any]] = {}
        request_payloads = [asdict(request) for request in request_list]

        def submit_request(index: int, excluded: Sequence[Any] = ()) -> None:
            actor = self._pick_actor(excluded)
            attempts[index] += 1
            pending[actor.evaluate.remote(request_payloads[index])] = (index, actor)

        for index in range(len(request_list)):
            submit_request(index)

        while pending:
            ready, _ = ray.wait(list(pending.keys()), num_returns=1)
            ref = ready[0]
            index, actor = pending.pop(ref)

            try:
                payloads[index] = ray.get(ref)
            except Exception as exc:
                excluded = ()
                if self._should_evict_actor(exc):
                    self._remove_actor(actor)
                else:
                    excluded = (actor,)

                if attempts[index] <= self.max_request_retries:
                    submit_request(index, excluded)
                    continue

                raise RuntimeError(
                    f"Evaluation request {index} failed after "
                    f"{self.max_request_retries} retries"
                ) from exc

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
