#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shlex
import socket
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import paramiko


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cluster.worker_config import WorkerSpec, load_workers_csv


def run_local(command: str) -> None:
    subprocess.run(["bash", "-lc", command], check=True)


def run_remote(ssh: paramiko.SSHClient, command: str) -> tuple[int, str, str]:
    remote_cmd = f"bash -lc {shlex.quote(command)}"
    _, stdout, stderr = ssh.exec_command(remote_cmd, get_pty=False)
    out = stdout.read().decode(errors="replace")
    err = stderr.read().decode(errors="replace")
    code = stdout.channel.recv_exit_status()
    return code, out, err


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


def pick_head_worker(workers: list[WorkerSpec], head_hostname: str | None) -> WorkerSpec:
    enabled = [worker for worker in workers if worker.ssh_host and worker.in_cluster]
    if not enabled:
        raise RuntimeError("No in-cluster workers found in workers.csv")

    if head_hostname:
        needle = head_hostname.strip().lower()
        for worker in enabled:
            if worker.hostname.lower() == needle or worker.display_name.lower() == needle:
                return worker
        raise RuntimeError(f"Head hostname '{head_hostname}' not found in workers.csv")

    return enabled[0]


def head_start_command(worker: WorkerSpec, port: int, dashboard_host: str) -> str:
    return worker.command_with_env(
        "ray stop --force || true; "
        f"ray start --head --port={int(port)} --dashboard-host={shlex.quote(dashboard_host)} "
        f"--num-cpus={int(worker.core_usage)}"
    )


def worker_start_command(worker: WorkerSpec, head_ip: str, port: int) -> str:
    return worker.command_with_env(
        "ray stop --force || true; "
        f"ray start --address={shlex.quote(f'{head_ip}:{int(port)}')} "
        f"--num-cpus={int(worker.core_usage)}"
    )


def stop_command(worker: WorkerSpec) -> str:
    return worker.command_with_env("ray stop --force || true")


def execute_on_worker(worker: WorkerSpec, command: str, timeout: int) -> None:
    if worker.is_local:
        run_local(command)
        return

    with connect_ssh(worker, timeout=timeout) as ssh:
        code, out, err = run_remote(ssh, command)
        if code != 0:
            raise RuntimeError((err or out).strip() or f"command failed on {worker.display_name}")


def start_workers(workers: list[WorkerSpec], head_worker: WorkerSpec, port: int, timeout: int) -> list[str]:
    head_ip = head_worker.ip_address or socket.gethostbyname(head_worker.hostname)
    targets = [worker for worker in workers if worker.ssh_host and worker.in_cluster and worker.hostname != head_worker.hostname]
    if not targets:
        return []

    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=min(32, len(targets))) as pool:
        futures = {
            pool.submit(execute_on_worker, worker, worker_start_command(worker, head_ip, port), timeout): worker
            for worker in targets
        }
        for future in as_completed(futures):
            worker = futures[future]
            try:
                future.result()
                print(f"[ok] started worker {worker.display_name}", flush=True)
            except Exception as exc:
                message = f"{worker.display_name}: {exc}"
                failures.append(message)
                print(f"[failed] {message}", flush=True)
    return failures


def stop_workers(workers: list[WorkerSpec], head_worker: WorkerSpec, timeout: int) -> list[str]:
    targets = [worker for worker in workers if worker.ssh_host and worker.in_cluster and worker.hostname != head_worker.hostname]
    if not targets:
        return []

    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=min(32, len(targets))) as pool:
        futures = {
            pool.submit(execute_on_worker, worker, stop_command(worker), timeout): worker
            for worker in targets
        }
        for future in as_completed(futures):
            worker = futures[future]
            try:
                future.result()
                print(f"[ok] stopped worker {worker.display_name}", flush=True)
            except Exception as exc:
                message = f"{worker.display_name}: {exc}"
                failures.append(message)
                print(f"[failed] {message}", flush=True)
    return failures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Start or stop a DragonchessAI Ray cluster using workers.csv."
    )
    parser.add_argument("--workers-file", default=str(REPO_ROOT / "workers.csv"))
    parser.add_argument("--head-hostname")
    parser.add_argument("--port", type=int, default=6379)
    parser.add_argument("--dashboard-host", default="0.0.0.0")
    parser.add_argument("--timeout", type=int, default=20)
    parser.add_argument("--start", action="store_true")
    parser.add_argument("--stop", action="store_true")
    parser.add_argument("--restart", action="store_true")
    args = parser.parse_args()
    if not (args.start or args.stop or args.restart):
        parser.error("Provide --start, --stop, or --restart")
    return args


def main() -> int:
    args = parse_args()
    if args.restart:
        args.start = True
        args.stop = True

    workers = load_workers_csv(args.workers_file, include_disabled=True)
    head_worker = pick_head_worker(workers, args.head_hostname)

    if args.stop:
        print(f"Stopping Ray workers (head={head_worker.display_name}) ...", flush=True)
        failures = stop_workers(workers, head_worker, args.timeout)
        try:
            execute_on_worker(head_worker, stop_command(head_worker), args.timeout)
            print(f"[ok] stopped head {head_worker.display_name}", flush=True)
        except Exception as exc:
            failures.append(f"{head_worker.display_name}: {exc}")
            print(f"[failed] {head_worker.display_name}: {exc}", flush=True)
        if failures:
            return 1

    if args.start:
        print(
            f"Starting Ray head {head_worker.display_name} with {head_worker.core_usage} CPUs ...",
            flush=True,
        )
        execute_on_worker(
            head_worker,
            head_start_command(head_worker, args.port, args.dashboard_host),
            args.timeout,
        )
        print(f"[ok] started head {head_worker.display_name}", flush=True)

        failures = start_workers(workers, head_worker, args.port, args.timeout)
        if failures:
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
