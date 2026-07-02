from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class GpuPoolConfig:
    name: str
    devices: tuple[str, ...]
    slots_per_device: int = 1


@dataclass(frozen=True)
class EvaluationConfig:
    timeout_seconds: int
    max_concurrent: int = 50
    gpu_pool: GpuPoolConfig | None = None


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    title: str
    prompt_template: str
    score_direction: str
    task_dir: Path
    workspace_dir: Path
    evaluator_dir: Path
    container_gpus: str | None
    host_mounts: tuple[Path, ...]
    container_env: dict[str, str]
    evaluation: EvaluationConfig


def load_task_spec(task_yaml: Path) -> TaskSpec:
    payload = yaml.safe_load(task_yaml.read_text(encoding="utf-8")) or {}
    task = payload.get("task", {})
    evaluation = payload.get("evaluation", {})

    task_id = str(task["id"])
    task_dir = task_yaml.parent
    workspace_dir = task_dir / str(task.get("workspace_dir", "workspace"))
    evaluator_dir = task_dir / str(task.get("evaluator_dir", "evaluator"))
    score_direction = str(task.get("score_direction", "maximize"))
    container_gpus = task.get("container_gpus")
    host_mounts = tuple(Path(str(path)).expanduser().resolve() for path in task.get("host_mounts", []))
    container_env = {str(key): str(value) for key, value in dict(task.get("container_env", {})).items()}
    gpu_pool = _load_gpu_pool_config(task_id, evaluation.get("gpu_pool"))

    if not workspace_dir.exists():
        raise FileNotFoundError(f"task {task_id!r} missing workspace dir: {workspace_dir}")
    if not evaluator_dir.exists():
        raise FileNotFoundError(f"task {task_id!r} missing evaluator dir: {evaluator_dir}")
    if score_direction not in {"maximize", "minimize"}:
        raise ValueError(
            f"task {task_id!r} has invalid score_direction {score_direction!r}; "
            "expected 'maximize' or 'minimize'"
        )

    return TaskSpec(
        task_id=task_id,
        title=str(task.get("title", task_id)),
        prompt_template=str(task["prompt"]["template"]),
        score_direction=score_direction,
        task_dir=task_dir,
        workspace_dir=workspace_dir,
        evaluator_dir=evaluator_dir,
        container_gpus=(None if container_gpus in (None, "") else str(container_gpus)),
        host_mounts=host_mounts,
        container_env=container_env,
        evaluation=EvaluationConfig(
            timeout_seconds=int(evaluation.get("timeout_seconds", 600)),
            max_concurrent=int(evaluation.get("max_concurrent", 50)),
            gpu_pool=gpu_pool,
        ),
    )


def _load_gpu_pool_config(task_id: str, payload: object) -> GpuPoolConfig | None:
    if payload in (None, ""):
        return None
    if not isinstance(payload, dict):
        raise ValueError(f"task {task_id!r} has invalid gpu_pool config; expected a mapping")

    name = str(payload.get("name", "")).strip()
    if not name:
        raise ValueError(f"task {task_id!r} gpu_pool.name is required")

    raw_devices = payload.get("devices")
    if not isinstance(raw_devices, list) or not raw_devices:
        raise ValueError(f"task {task_id!r} gpu_pool.devices must be a non-empty list")
    devices = tuple(str(device).strip() for device in raw_devices if str(device).strip())
    if not devices:
        raise ValueError(f"task {task_id!r} gpu_pool.devices must contain at least one device id")

    slots_per_device = int(payload.get("slots_per_device", 1))
    if slots_per_device < 1:
        raise ValueError(f"task {task_id!r} gpu_pool.slots_per_device must be at least 1")

    return GpuPoolConfig(
        name=name,
        devices=devices,
        slots_per_device=slots_per_device,
    )


def discover_tasks(tasks_root: Path) -> dict[str, TaskSpec]:
    specs: dict[str, TaskSpec] = {}
    for task_yaml in sorted(tasks_root.glob("*/task.yaml")):
        spec = load_task_spec(task_yaml)
        if spec.task_id in specs:
            raise ValueError(f"duplicate task id {spec.task_id!r}: {task_yaml}")
        specs[spec.task_id] = spec
    return specs


def get_task_spec(tasks_root: Path, task_id: str) -> TaskSpec:
    try:
        return discover_tasks(tasks_root)[task_id]
    except KeyError as exc:
        raise KeyError(f"unknown task id: {task_id}") from exc
