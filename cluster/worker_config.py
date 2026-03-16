from __future__ import annotations

import csv
import getpass
import os
import shlex
import socket
from dataclasses import dataclass
from pathlib import Path

DEFAULT_CLUSTER_USERNAME = "dgezgin"
DEFAULT_CLUSTER_PASSWORD = "geesearebigtoddlers1"


def normalize(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def parse_boolish(value: object, default: bool = True) -> bool:
    text = normalize(value).lower()
    if not text:
        return default
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


def parse_int(value: object, default: int = 0) -> int:
    try:
        return int(normalize(value))
    except Exception:
        return default


def local_host_aliases() -> set[str]:
    aliases = {"localhost", "127.0.0.1"}
    try:
        hostname = socket.gethostname()
        aliases.add(hostname.lower())
        aliases.add(socket.getfqdn().lower())
        aliases.add(socket.gethostbyname(hostname))
    except Exception:
        pass
    return {alias for alias in aliases if alias}


@dataclass(frozen=True)
class WorkerSpec:
    room: str
    hostname: str
    ip_address: str
    monitor_name: str
    username: str
    password: str
    env: str
    core_usage: int
    in_cluster: bool = True

    @property
    def display_name(self) -> str:
        return self.monitor_name or self.hostname or self.ip_address

    @property
    def ssh_host(self) -> str:
        return self.ip_address or self.hostname

    @property
    def is_enabled(self) -> bool:
        return self.in_cluster and bool(self.ssh_host)

    @property
    def is_local(self) -> bool:
        aliases = local_host_aliases()
        return self.hostname.lower() in aliases or self.ip_address in aliases

    def require_remote_credentials(self) -> None:
        if self.is_local:
            return
        if not self.username:
            raise ValueError(
                f"Worker '{self.display_name}' is missing a username in workers.csv."
            )

    def command_with_env(self, command: str) -> str:
        return wrap_command_with_env(command, self.env)


def wrap_command_with_env(command: str, env_value: str) -> str:
    env_value = normalize(env_value)
    if not env_value:
        return command

    # Allow power users to supply a full shell prefix explicitly.
    if any(token in env_value for token in ("&&", ";", "source ", ". ", "conda activate ")):
        return f"{env_value} && {command}"

    # Support explicit activate script paths or venv roots.
    if env_value.endswith("/bin/activate") or env_value.endswith(".sh"):
        activate = f"source {shlex.quote(env_value)}"
        return f"{activate} && {command}"

    if "/" in env_value or env_value.startswith("~") or env_value.startswith("."):
        root = env_value.rstrip("/")
        activate = f"source {shlex.quote(root)}/bin/activate"
        return f"{activate} && {command}"

    # Otherwise interpret the value as a conda environment name.
    qenv = shlex.quote(env_value)
    conda_bootstrap = (
        'if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then '
        'source "$HOME/miniconda3/etc/profile.d/conda.sh"; '
        'elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then '
        'source "$HOME/anaconda3/etc/profile.d/conda.sh"; '
        'elif [ -f "$HOME/mambaforge/etc/profile.d/conda.sh" ]; then '
        'source "$HOME/mambaforge/etc/profile.d/conda.sh"; '
        'else echo "conda.sh not found" 1>&2; exit 127; fi'
    )
    return f"{conda_bootstrap} && conda activate {qenv} && {command}"


def load_workers_csv(path: str | os.PathLike[str], *, include_disabled: bool = True) -> list[WorkerSpec]:
    csv_path = Path(path).expanduser().resolve()
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    workers: list[WorkerSpec] = []
    for row in rows:
        worker = WorkerSpec(
            room=normalize(row.get("room")),
            hostname=normalize(row.get("hostname")),
            ip_address=normalize(row.get("ip-address")),
            monitor_name=normalize(row.get("monitor-name")),
            username=normalize(row.get("username")) or DEFAULT_CLUSTER_USERNAME,
            password=normalize(row.get("password")) or DEFAULT_CLUSTER_PASSWORD,
            env=normalize(row.get("env")),
            core_usage=parse_int(row.get("core_usage"), 0),
            in_cluster=parse_boolish(row.get("in_cluster"), True),
        )
        if include_disabled or worker.is_enabled:
            workers.append(worker)
    return workers


def default_remote_repo_dir() -> str:
    return f"~/{Path(os.getcwd()).resolve().name}"


def default_ssh_username() -> str:
    return getpass.getuser()
