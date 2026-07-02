#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
# DEFAULT_CONFIGS = (3, 2)
DEFAULT_CONFIGS = ((5, 12), (10, 6), (15, 4), (20, 3), (30, 2))
# DEFAULT_CONFIGS = ((8, 6), (4, 12))

@dataclass(frozen=True)
class SweepConfig:
    agent_count: int
    max_iterations: int

    @property
    def label(self) -> str:
        return f"{self.agent_count}x{self.max_iterations}"


def parse_configs(raw_value: str | None) -> list[SweepConfig]:
    if raw_value is None:
        if (
            isinstance(DEFAULT_CONFIGS, tuple)
            and len(DEFAULT_CONFIGS) == 2
            and all(isinstance(value, int) for value in DEFAULT_CONFIGS)
        ):
            agent_count, max_iterations = DEFAULT_CONFIGS
            if agent_count < 1 or max_iterations < 1:
                raise ValueError("DEFAULT_CONFIGS values must be positive")
            return [SweepConfig(agent_count, max_iterations)]
        return [SweepConfig(agent_count, max_iterations) for agent_count, max_iterations in DEFAULT_CONFIGS]

    configs: list[SweepConfig] = []
    for raw_entry in raw_value.split(","):
        entry = raw_entry.strip().lower()
        if not entry:
            raise ValueError("empty config entry")
        pieces = entry.split("x")
        if len(pieces) != 2:
            raise ValueError(f"invalid config {raw_entry!r}; expected format like 4x12")
        try:
            agent_count = int(pieces[0])
            max_iterations = int(pieces[1])
        except ValueError as exc:
            raise ValueError(f"invalid config {raw_entry!r}; values must be integers") from exc
        if agent_count < 1 or max_iterations < 1:
            raise ValueError(f"invalid config {raw_entry!r}; values must be positive")
        configs.append(SweepConfig(agent_count, max_iterations))
    return configs


def build_run_problem_command(
    args: argparse.Namespace,
    run_script: Path,
    *,
    port: int | None = None,
    run_suffix: str | None = None,
) -> list[str]:
    cmd = [
        sys.executable,
        str(run_script),
        "--tasks-root",
        str(args.tasks_root.resolve()),
        "--task-id",
        args.task_id,
        "--output-dir",
        str(args.output_dir.resolve()),
        "--agent-dir",
        str(args.agent_dir.resolve()),
        "--worker-image",
        args.worker_image,
        "--repo-root",
        str(args.repo_root.resolve()),
    ]
    if port is not None:
        cmd.extend(["--port", str(port)])
    if args.restart_server:
        cmd.append("--restart-server")
    if args.run_timeout_minutes is not None:
        cmd.extend(["--run-timeout-minutes", str(args.run_timeout_minutes)])
    cmd.append(f"--{'post-eval-message' if args.post_eval_message else 'no-post-eval-message'}")
    if args.no_skill:
        cmd.append("--no-skill")
    if run_suffix:
        cmd.extend(["--run-suffix", run_suffix])
    return cmd


def build_config_env(base_env: dict[str, str], config: SweepConfig) -> dict[str, str]:
    env = base_env.copy()
    env["AGENT_COUNT"] = str(config.agent_count)
    env["MAX_ITERATIONS"] = str(config.max_iterations)
    return env


def reserve_free_port() -> socket.socket:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("", 0))
        sock.listen(1)
    except Exception:
        sock.close()
        raise
    return sock


def reserve_parallel_ports(count: int) -> list[tuple[socket.socket, int]]:
    reservations: list[tuple[socket.socket, int]] = []
    try:
        for _ in range(count):
            sock = reserve_free_port()
            reservations.append((sock, int(sock.getsockname()[1])))
    except Exception:
        for sock, _ in reservations:
            sock.close()
        raise
    return reservations


def _stream_output(process: subprocess.Popen[str], *, label: str | None = None) -> tuple[str | None, int | None]:
    run_dir: str | None = None
    eval_server_port: int | None = None
    line_prefix = f"[{label}] " if label else ""
    assert process.stdout is not None
    for line in process.stdout:
        print(f"{line_prefix}{line}", end="", flush=True)
        unprefixed_line = line
        run_dir_prefix = "[run] Run directory: "
        if unprefixed_line.startswith(run_dir_prefix):
            run_dir = unprefixed_line[len(run_dir_prefix) :].strip()
        port_prefix = "[run] Eval server port: "
        if unprefixed_line.startswith(port_prefix):
            try:
                eval_server_port = int(unprefixed_line[len(port_prefix) :].strip())
            except ValueError:
                pass
    return run_dir, eval_server_port


def start_config(command: list[str], env: dict[str, str], *, cwd: Path) -> subprocess.Popen[str]:
    return subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )


def run_config(
    command: list[str], env: dict[str, str], *, cwd: Path, label: str | None = None
) -> tuple[int, str | None, int | None]:
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    run_dir, eval_server_port = _stream_output(process, label=label)
    return process.wait(), run_dir, eval_server_port


def run_configs_parallel(
    *,
    args: argparse.Namespace,
    configs: list[SweepConfig],
    run_script: Path,
    repo_root: Path,
) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    running: list[dict[str, object]] = []
    lock = threading.Lock()

    port_reservations = reserve_parallel_ports(len(configs))
    try:
        for index, (config, (_, port)) in enumerate(zip(configs, port_reservations, strict=True), start=1):
            run_suffix = f"{index:02d}-{config.label}"
            command = build_run_problem_command(args, run_script, port=port, run_suffix=run_suffix)
            print(f"[sweep] starting config {index}/{len(configs)} on reserved port {port}: {config.label}", flush=True)
            sock, _ = port_reservations[index - 1]
            sock.close()
            process = start_config(command, build_config_env(os.environ, config), cwd=repo_root)
            record: dict[str, object] = {
                "label": config.label,
                "agent_count": config.agent_count,
                "max_iterations": config.max_iterations,
                "command": command,
                "reserved_port": port,
                "eval_server_port": None,
                "process": process,
                "run_dir": None,
                "started_at": datetime.now().isoformat(timespec="seconds"),
            }

            def reader(entry: dict[str, object] = record) -> None:
                run_dir, eval_server_port = _stream_output(entry["process"], label=str(entry["label"]))  # type: ignore[arg-type]
                with lock:
                    entry["run_dir"] = run_dir
                    entry["eval_server_port"] = eval_server_port

            thread = threading.Thread(target=reader, daemon=True)
            thread.start()
            record["thread"] = thread
            running.append(record)
    finally:
        for sock, _ in port_reservations:
            try:
                sock.close()
            except OSError:
                pass

    for record in running:
        process = record["process"]
        thread = record["thread"]
        assert isinstance(process, subprocess.Popen)
        assert isinstance(thread, threading.Thread)
        returncode = process.wait()
        thread.join()
        record["returncode"] = returncode
        record["finished_at"] = datetime.now().isoformat(timespec="seconds")
        record.pop("process", None)
        record.pop("thread", None)
        entries.append(record)
        print(f"[sweep] finished config {record['label']} with return code {returncode}", flush=True)

    return entries


def write_summary(output_dir: Path, task_id: str, payload: dict[str, object]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = output_dir / f"config_sweep_{task_id}_{timestamp}.json"
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary_path


def run_sweep(args: argparse.Namespace) -> int:
    configs = parse_configs(args.configs)
    repo_root = args.repo_root.resolve()
    run_script = repo_root / "swarmresearch_reproduce" / "evaluation" / "run_problem.py"
    started_at = datetime.now().isoformat(timespec="seconds")
    entries: list[dict[str, object]] = []

    if args.parallel_configs:
        entries = run_configs_parallel(args=args, configs=configs, run_script=run_script, repo_root=repo_root)
    else:
        for index, config in enumerate(configs, start=1):
            command = build_run_problem_command(args, run_script, run_suffix=f"{index:02d}-{config.label}")
            print(f"[sweep] starting config {index}/{len(configs)}: {config.label}", flush=True)
            config_started_at = datetime.now().isoformat(timespec="seconds")
            returncode, run_dir, eval_server_port = run_config(command, build_config_env(os.environ, config), cwd=repo_root)
            config_finished_at = datetime.now().isoformat(timespec="seconds")
            entries.append(
                {
                    "label": config.label,
                    "agent_count": config.agent_count,
                    "max_iterations": config.max_iterations,
                    "command": command,
                    "eval_server_port": eval_server_port,
                    "returncode": returncode,
                    "run_dir": run_dir,
                    "started_at": config_started_at,
                    "finished_at": config_finished_at,
                }
            )
            print(f"[sweep] finished config {config.label} with return code {returncode}", flush=True)

    payload = {
        "task_id": args.task_id,
        "parallel_configs": bool(args.parallel_configs),
        "started_at": started_at,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "configs": entries,
    }
    summary_path = write_summary(args.output_dir.resolve(), args.task_id, payload)
    print(f"[sweep] summary: {summary_path}", flush=True)
    return 0 if all(entry["returncode"] == 0 for entry in entries) else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one task across a sweep of AGENT_COUNT x MAX_ITERATIONS configs.")
    parser.add_argument("--tasks-root", type=Path, required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--agent-dir", type=Path, required=True)
    parser.add_argument("--worker-image", default="task", help="Worker image name, or 'task' to reuse the task evaluator image")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help=(
            "Reserved for compatibility; ignored by config sweep. Sequential configs let run_problem.py "
            "choose a port, and parallel configs reserve distinct free ports in the sweep process."
        ),
    )
    parser.add_argument("--restart-server", action="store_true")
    parser.add_argument("--run-timeout-minutes", type=int, default=None)
    parser.add_argument(
        "--post-eval-message",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether ./task-eval prints the post-evaluation message after a successful evaluation.",
    )
    parser.add_argument("--no-skill", action="store_true", help="Forward --no-skill to each run_problem invocation.")
    parser.add_argument(
        "--parallel-configs",
        action="store_true",
        help="Run all configs concurrently. The sweep parent reserves one free port per config.",
    )
    parser.add_argument(
        "--configs",
        default=None,
        help="Comma-separated AGENT_COUNT x MAX_ITERATIONS configs, for example: 1x48,4x12,8x6",
    )
    args = parser.parse_args(argv)

    try:
        return run_sweep(args)
    except ValueError as exc:
        parser.error(str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
