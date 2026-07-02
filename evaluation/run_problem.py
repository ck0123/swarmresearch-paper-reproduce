#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

from jinja2 import Template

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from swarmresearch_reproduce.evaluation.container_eval import ContainerEvaluator
from swarmresearch_reproduce.evaluation.runner_common import (
    TASK_EVAL_COUNTER_FILENAME,
    TASK_EVAL_COUNTER_MOUNT_PATH,
    ensure_server_running,
    kill_server_by_port,
    loop_uses_unix_user_isolation,
    pick_server_port,
    run_loop_in_container,
    start_fresh_worker_container,
    stop_container,
)
from swarmresearch_reproduce.evaluation.tasks import TaskSpec, get_task_spec


def safe_run_suffix(raw_suffix: str | None) -> str | None:
    if raw_suffix is None:
        return None
    cleaned = "".join(char if char.isalnum() or char in "._-" else "-" for char in raw_suffix.strip())
    cleaned = cleaned.strip("._-")
    return cleaned or None


def render_prompt(spec: TaskSpec, *, instructions_path: Path | None = None) -> str:
    prompt = Template(spec.prompt_template).render(
        task_id=spec.task_id,
        eval_timeout_seconds=spec.evaluation.timeout_seconds,
    )
    if instructions_path is None:
        return prompt

    instructions = instructions_path.read_text(encoding="utf-8").strip()
    if not instructions:
        raise ValueError(f"instructions file is empty: {instructions_path}")
    return f"{prompt.rstrip()}\n\n---\n\n{instructions}\n"


def copy_task_workspace(spec: TaskSpec, run_dir: Path) -> None:
    for asset in spec.workspace_dir.iterdir():
        dst = run_dir / asset.name
        if asset.is_dir():
            shutil.copytree(asset, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(asset, dst)
    (run_dir / "task_id.txt").write_text(spec.task_id + "\n", encoding="utf-8")


def copy_eval_client(
    run_dir: Path,
    runtime_dir: Path,
    repo_root: Path,
    *,
    eval_timeout_seconds: int,
    score_direction: str,
    emit_post_eval_message: bool,
    task_eval_server_url: str | None = None,
    extra_task_eval_config: dict[str, object] | None = None,
) -> None:
    client_src = repo_root / "swarmresearch_reproduce" / "evaluation" / "client.py"
    client_dst = run_dir / "task-eval"
    shutil.copy2(client_src, client_dst)
    client_dst.chmod(0o755)
    counter_path = runtime_dir / TASK_EVAL_COUNTER_FILENAME
    counter_path.write_text("0\n", encoding="utf-8")
    config: dict[str, object] = {
        "eval_timeout_seconds": int(eval_timeout_seconds),
        "score_direction": score_direction,
        "task_eval_counter_path": TASK_EVAL_COUNTER_MOUNT_PATH,
        "emit_post_eval_message": bool(emit_post_eval_message),
    }
    if task_eval_server_url:
        config["task_eval_server_url"] = task_eval_server_url
    if extra_task_eval_config:
        config.update(extra_task_eval_config)
    (run_dir / "task_eval_config.json").write_text(
        json.dumps(config, indent=2)
        + "\n",
        encoding="utf-8",
    )


def prepare_run(
    *,
    tasks_root: Path,
    task_id: str,
    output_dir: Path,
    agent_dir: Path,
    worker_image: str,
    repo_root: Path,
    port: int | None = None,
    restart_server: bool = False,
    run_timeout_minutes: int | None = None,
    emit_post_eval_message: bool = True,
    no_skill: bool = False,
    run_suffix: str | None = None,
) -> Path:
    spec = get_task_spec(tasks_root, task_id)
    if worker_image == "task":
        task_image = ContainerEvaluator().ensure_image(spec)
        worker_image = task_image
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{spec.task_id}_{timestamp}"
    suffix = safe_run_suffix(run_suffix)
    if suffix:
        run_name = f"{run_name}_{suffix}"
    run_dir = output_dir / run_name
    runtime_dir = output_dir / ".agent_runtime" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    runtime_dir.mkdir(parents=True, exist_ok=True)
    copy_task_workspace(spec, run_dir)
    extra_task_eval_config_path = agent_dir / ".pi" / "task_eval_config.json"
    extra_task_eval_config: dict[str, object] | None = None
    if extra_task_eval_config_path.exists():
        loaded_config = json.loads(extra_task_eval_config_path.read_text(encoding="utf-8"))
        if not isinstance(loaded_config, dict):
            raise ValueError(f"expected object in {extra_task_eval_config_path}")
        extra_task_eval_config = loaded_config

    server_port = port if port is not None else pick_server_port()
    if restart_server:
        kill_server_by_port(server_port)
    server_port = ensure_server_running(tasks_root, server_port, repo_root)

    copy_eval_client(
        run_dir,
        runtime_dir,
        repo_root,
        eval_timeout_seconds=spec.evaluation.timeout_seconds,
        score_direction=spec.score_direction,
        emit_post_eval_message=emit_post_eval_message,
        task_eval_server_url=f"http://host.docker.internal:{server_port}",
        extra_task_eval_config=extra_task_eval_config,
    )
    instructions_path = None
    if no_skill:
        instructions_path = agent_dir / ".pi" / "instructions.md"
        if not instructions_path.exists():
            raise FileNotFoundError(f"--no-skill requires instructions file: {instructions_path}")
    (run_dir / "prompt.md").write_text(render_prompt(spec, instructions_path=instructions_path), encoding="utf-8")

    container_name = f"task-{run_name}"
    timeout_minutes = run_timeout_minutes or (24 * 60)
    start_fresh_worker_container(
        container_name=container_name,
        run_dir=run_dir.resolve(),
        runtime_dir=runtime_dir.resolve(),
        image=worker_image,
        server_port=server_port,
        agent_dir=agent_dir.resolve(),
    )
    print(f"[run] Run directory: {run_dir}")
    print(f"[run] Agent runtime directory: {runtime_dir}")
    print(f"[run] Eval server port: {server_port}")
    try:
        run_loop_in_container(
            container_name,
            timeout_minutes=timeout_minutes,
            run_as_root=loop_uses_unix_user_isolation(agent_dir / "loop.sh"),
        )
    finally:
        stop_container(container_name)
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a single task using the generic evaluation framework.")
    parser.add_argument("--tasks-root", type=Path, required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--agent-dir", type=Path, required=True)
    parser.add_argument("--worker-image", default="task", help="Worker image name, or 'task' to reuse the task evaluator image")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--restart-server", action="store_true")
    parser.add_argument("--run-timeout-minutes", type=int, default=None)
    parser.add_argument(
        "--post-eval-message",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether ./task-eval prints the post-evaluation message after a successful evaluation.",
    )
    parser.add_argument(
        "--no-skill",
        action="store_true",
        help="Append agent .pi/instructions.md to prompt.md and do not ask the agent to read a skill.",
    )
    parser.add_argument(
        "--run-suffix",
        default=None,
        help="Optional suffix appended to the timestamped run directory and worker container name.",
    )
    args = parser.parse_args()

    prepare_run(
        tasks_root=args.tasks_root.resolve(),
        task_id=args.task_id,
        output_dir=args.output_dir.resolve(),
        agent_dir=args.agent_dir.resolve(),
        worker_image=args.worker_image,
        repo_root=args.repo_root.resolve(),
        port=args.port,
        restart_server=args.restart_server,
        run_timeout_minutes=args.run_timeout_minutes,
        emit_post_eval_message=args.post_eval_message,
        no_skill=args.no_skill,
        run_suffix=args.run_suffix,
    )


if __name__ == "__main__":
    main()
