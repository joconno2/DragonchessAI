#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fnmatch
import os
import posixpath
import shlex
import shutil
import socket
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

import paramiko


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cluster.worker_config import WorkerSpec, default_remote_repo_dir, load_workers_csv


LOCAL_EXCLUDE_DIRS = {
    ".git",
    ".cursor",
    ".pytest_cache",
    "__pycache__",
    "build",
    "build-headless",
    "results",
    "logs",
    "figures",
    "temp",
}
EXCLUDE_GLOBS = {"*.pyc", "*.pyo", "*.o", "*.so", "*.a", "*.tmp"}


def should_skip(name: str) -> bool:
    if name in LOCAL_EXCLUDE_DIRS:
        return True
    return any(fnmatch.fnmatch(name, pattern) for pattern in EXCLUDE_GLOBS)


def resolve_remote_repo_dir(sftp: paramiko.SFTPClient, repo_dir: str) -> str:
    if not repo_dir:
        repo_dir = default_remote_repo_dir()
    repo_dir = repo_dir.strip()
    home = sftp.normalize(".")
    if repo_dir == "~":
        return home
    if repo_dir.startswith("~/"):
        return f"{home}/{repo_dir[2:]}"
    return repo_dir


def sftp_mkdir_relative(sftp: paramiko.SFTPClient, base_dir: str, relative_dir: str) -> None:
    current = base_dir.rstrip("/")
    if not relative_dir:
        return
    for part in [part for part in relative_dir.split("/") if part]:
        current = f"{current}/{part}"
        try:
            sftp.stat(current)
        except FileNotFoundError:
            sftp.mkdir(current)


def copy_repo_to_sftp(sftp: paramiko.SFTPClient, local_root: Path, remote_root: str) -> None:
    for root, dirs, files in os.walk(local_root):
        dirs[:] = [directory for directory in dirs if not should_skip(directory)]
        rel_path = os.path.relpath(root, local_root)
        rel_posix = "" if rel_path == "." else rel_path.replace(os.sep, "/")
        remote_dir = remote_root if not rel_posix else f"{remote_root}/{rel_posix}"
        sftp_mkdir_relative(sftp, remote_root, rel_posix)

        for file_name in files:
            if should_skip(file_name):
                continue
            local_path = Path(root) / file_name
            remote_path = f"{remote_dir}/{file_name}"
            sftp.put(str(local_path), remote_path)


def copy_binary_to_sftp(
    sftp: paramiko.SFTPClient,
    local_binary_path: Path,
    remote_root: str,
) -> str:
    remote_build_dir = posixpath.join(remote_root, "build")
    sftp_mkdir_relative(sftp, remote_root, "build")
    remote_binary_path = posixpath.join(remote_build_dir, "dragonchess")
    sftp.put(str(local_binary_path), remote_binary_path)
    sftp.chmod(remote_binary_path, 0o755)
    return remote_binary_path


def copy_repo_local(local_root: Path, destination_root: Path) -> None:
    destination_root.mkdir(parents=True, exist_ok=True)
    for root, dirs, files in os.walk(local_root):
        dirs[:] = [directory for directory in dirs if not should_skip(directory)]
        rel_path = os.path.relpath(root, local_root)
        rel_posix = "" if rel_path == "." else rel_path.replace(os.sep, "/")
        target_dir = destination_root if not rel_posix else destination_root / rel_posix
        target_dir.mkdir(parents=True, exist_ok=True)
        for file_name in files:
            if should_skip(file_name):
                continue
            src = Path(root) / file_name
            dst = target_dir / file_name
            try:
                if src.resolve() == dst.resolve():
                    continue
            except FileNotFoundError:
                pass
            shutil.copy2(src, dst)


def run_local(command: str, cwd: str | None = None) -> None:
    subprocess.run(["bash", "-lc", command], cwd=cwd, check=True)


def run_remote(ssh: paramiko.SSHClient, command: str) -> tuple[int, str, str]:
    remote_cmd = f"bash -lc {shlex.quote(command)}"
    _, stdout, stderr = ssh.exec_command(remote_cmd, get_pty=False)
    out = stdout.read().decode(errors="replace")
    err = stderr.read().decode(errors="replace")
    code = stdout.channel.recv_exit_status()
    return code, out, err


def ensure_remote_repo_dir(ssh: paramiko.SSHClient, repo_dir: str) -> None:
    code, out, err = run_remote(ssh, f"mkdir -p {shlex.quote(repo_dir)}")
    if code != 0:
        raise RuntimeError((err or out).strip() or f"failed to create {repo_dir}")


def connect_ssh(worker: WorkerSpec, timeout: int) -> paramiko.SSHClient:
    worker.require_remote_credentials()
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(
        worker.ssh_host,
        username=worker.username,
        password=worker.password or None,
        timeout=timeout,
        auth_timeout=timeout,
        banner_timeout=timeout,
        allow_agent=True,
        look_for_keys=True,
    )
    return ssh


def build_local_binary(binary_path: Path, *, clean_build: bool) -> None:
    build_dir = binary_path.parent
    if clean_build and build_dir.exists():
        shutil.rmtree(build_dir)

    build_dir.parent.mkdir(parents=True, exist_ok=True)
    configure_cmd = (
        f"cmake -S {shlex.quote(str(REPO_ROOT))} "
        f"-B {shlex.quote(str(build_dir))} "
        "-DCMAKE_BUILD_TYPE=Release -DHEADLESS_ONLY=ON"
    )
    build_cmd = f"cmake --build {shlex.quote(str(build_dir))} --parallel"
    run_local(configure_cmd, cwd=str(REPO_ROOT))
    run_local(build_cmd, cwd=str(REPO_ROOT))


def worker_runtime_commands(repo_dir: str, *, install_python_deps: bool) -> list[str]:
    commands: list[str] = []
    if install_python_deps:
        commands.append("python3 -m pip install --user ray")
    return commands


def sync_worker(
    worker: WorkerSpec,
    *,
    local_repo_root: Path,
    remote_repo_dir: str,
    local_binary_path: Path,
    install_python_deps: bool,
    timeout: int,
) -> str:
    label = worker.display_name
    if worker.is_local:
        target_dir = Path(os.path.expanduser(remote_repo_dir)).resolve()
        copy_repo_local(local_repo_root, target_dir)
        build_dir = target_dir / "build"
        build_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local_binary_path, build_dir / "dragonchess")
        os.chmod(build_dir / "dragonchess", 0o755)
        for command in worker_runtime_commands(
            str(target_dir),
            install_python_deps=install_python_deps,
        ):
            run_local(worker.command_with_env(command), cwd=str(target_dir))
        return f"{label}: synced locally to {target_dir} with prebuilt binary"

    with connect_ssh(worker, timeout=timeout) as ssh:
        with ssh.open_sftp() as sftp:
            resolved_remote_dir = resolve_remote_repo_dir(sftp, remote_repo_dir)
            ensure_remote_repo_dir(ssh, resolved_remote_dir)
            copy_repo_to_sftp(sftp, local_repo_root, resolved_remote_dir)
            remote_binary_path = copy_binary_to_sftp(
                sftp,
                local_binary_path,
                resolved_remote_dir,
            )

        for command in worker_runtime_commands(
            resolved_remote_dir,
            install_python_deps=install_python_deps,
        ):
            code, out, err = run_remote(ssh, worker.command_with_env(command))
            if code != 0:
                raise RuntimeError((err or out).strip() or f"command failed on {label}")

        return f"{label}: synced to {resolved_remote_dir} with {remote_binary_path}"


def choose_targets(workers: Iterable[WorkerSpec], include_disabled: bool) -> list[WorkerSpec]:
    if include_disabled:
        return [worker for worker in workers if worker.ssh_host]
    return [worker for worker in workers if worker.is_enabled]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sync DragonchessAI to cluster machines and distribute a prebuilt headless binary."
    )
    parser.add_argument("--workers-file", default=str(REPO_ROOT / "workers.csv"))
    parser.add_argument("--repo-dir", default="~/DragonchessAI")
    parser.add_argument("--binary-path", default=str(REPO_ROOT / "build" / "dragonchess"))
    parser.add_argument(
        "--build-local",
        action="store_true",
        help="Build the headless binary locally before syncing it to workers",
    )
    parser.add_argument("--include-disabled", action="store_true")
    parser.add_argument(
        "--install-python-deps",
        action="store_true",
        help="Install minimal runtime deps on workers (currently just ray)",
    )
    parser.add_argument("--clean-build", action="store_true")
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--jobs", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    workers = load_workers_csv(args.workers_file, include_disabled=True)
    targets = choose_targets(workers, include_disabled=args.include_disabled)
    if not targets:
        raise SystemExit("No eligible workers found in workers.csv")

    local_repo_root = REPO_ROOT
    local_binary_path = Path(args.binary_path).expanduser().resolve()
    if args.build_local:
        build_local_binary(local_binary_path, clean_build=args.clean_build)

    if not local_binary_path.is_file():
        raise SystemExit(
            f"Local binary not found at {local_binary_path}. "
            "Build it locally first or pass --build-local."
        )

    max_workers = max(1, min(args.jobs, len(targets)))

    print(f"Syncing {len(targets)} workers from {local_repo_root} ...", flush=True)
    print(f"Distributing prebuilt binary: {local_binary_path}", flush=True)
    failures: list[str] = []

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(
                sync_worker,
                worker,
                local_repo_root=local_repo_root,
                remote_repo_dir=args.repo_dir,
                local_binary_path=local_binary_path,
                install_python_deps=args.install_python_deps,
                timeout=args.timeout,
            ): worker
            for worker in targets
        }

        for future in as_completed(futures):
            worker = futures[future]
            try:
                print(f"[ok] {future.result()}", flush=True)
            except Exception as exc:
                message = f"{worker.display_name}: {exc}"
                failures.append(message)
                print(f"[failed] {message}", flush=True)

    if failures:
        print("\nFailures:", flush=True)
        for failure in failures:
            print(f"  - {failure}", flush=True)
        return 1

    print("\nCluster sync complete.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
