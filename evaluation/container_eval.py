from __future__ import annotations

import json
import os
import re
import subprocess
import uuid
from pathlib import Path

from swarmresearch_reproduce.evaluation.tasks import TaskSpec


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVAL_CONTAINER_CPUS = "2"
SHARED_IMAGE_BUILDS: dict[str, tuple[Path, Path]] = {
    "swarmresearch-task-base:py312": (
        REPO_ROOT / "swarmresearch_reproduce" / "evaluation" / "docker" / "task_base" / "py312.Dockerfile",
        REPO_ROOT / "swarmresearch_reproduce" / "evaluation" / "docker" / "task_base",
    ),
    "swarmresearch-task-base:py311": (
        REPO_ROOT / "swarmresearch_reproduce" / "evaluation" / "docker" / "task_base" / "py311.Dockerfile",
        REPO_ROOT / "swarmresearch_reproduce" / "evaluation" / "docker" / "task_base",
    ),
    "ale-bench-lite-worker:latest": (
        REPO_ROOT / "environments" / "ale_bench_lite" / "docker" / "Dockerfile",
        REPO_ROOT / "environments" / "ale_bench_lite" / "docker",
    ),
}


class ContainerEvaluator:
    def __init__(self) -> None:
        self._built_images: dict[str, str] = {}
        self._built_shared_images: set[str] = set()
        self._passthrough_env_names = (
            "GPUMODE_USE_MODAL",
            "GPUMODE_MODAL_GPU",
            "MODAL_TOKEN_ID",
            "MODAL_TOKEN_SECRET",
            "CUDA_VISIBLE_DEVICES",
            "NVIDIA_VISIBLE_DEVICES",
        )

    def _eval_container_cpus(self) -> str | None:
        raw_value = os.environ.get("TASK_EVAL_CONTAINER_CPUS", DEFAULT_EVAL_CONTAINER_CPUS).strip()
        if not raw_value or raw_value.lower() in {"0", "none", "false", "off"}:
            return None
        try:
            if float(raw_value) <= 0:
                return None
        except ValueError as exc:
            raise ValueError(f"TASK_EVAL_CONTAINER_CPUS must be a positive number, got {raw_value!r}") from exc
        return raw_value

    def ensure_image(self, task: TaskSpec) -> str:
        return self._ensure_image(task)

    def evaluate_workspace(
        self,
        task: TaskSpec,
        workspace_dir: Path,
        *,
        timeout_seconds: int | None = None,
        assigned_gpu: str | None = None,
    ) -> dict[str, object]:
        image = self._ensure_image(task)
        timeout = int(timeout_seconds or task.evaluation.timeout_seconds)
        container_name = f"task-eval-{task.task_id}-{uuid.uuid4().hex[:12]}"
        host_workspace_dir = workspace_dir.resolve()
        cmd = [
            "docker",
            "run",
            "--rm",
            "--name",
            container_name,
            "-v",
            f"{host_workspace_dir}:/workspace:ro",
        ]
        container_cpus = self._eval_container_cpus()
        if container_cpus is not None:
            cmd.extend(["--cpus", container_cpus])
        gpu_visibility: str | None = None
        if assigned_gpu is not None:
            cmd.extend(["--runtime", "nvidia"])
            gpu_visibility = assigned_gpu
        elif task.container_gpus:
            cmd.extend(["--runtime", "nvidia"])
            gpu_visibility = task.container_gpus
        if gpu_visibility is not None:
            cmd.extend(
                [
                    "-e",
                    f"NVIDIA_VISIBLE_DEVICES={gpu_visibility}",
                    "-e",
                    "NVIDIA_DRIVER_CAPABILITIES=compute,utility",
                ]
            )
        for host_path in task.host_mounts:
            if host_path.exists():
                cmd.extend(["-v", f"{host_path}:{host_path}"])
        for env_name in self._passthrough_env_names:
            if assigned_gpu is not None and env_name in {"CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES"}:
                continue
            env_value = os.environ.get(env_name)
            if env_value:
                cmd.extend(["-e", f"{env_name}={env_value}"])
        for env_name, env_value in task.container_env.items():
            cmd.extend(["-e", f"{env_name}={env_value}"])
        cmd.extend([image, "/workspace"])
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout + 10,
                check=False,
            )
        except subprocess.TimeoutExpired:
            subprocess.run(["docker", "rm", "-f", container_name], capture_output=True, check=False)
            return {
                "task_id": task.task_id,
                "status": "timeout",
                "score": 1e30,
                "metrics": {},
                "feedback": f"Evaluation timed out after {timeout}s",
            }

        if result.returncode != 0:
            return {
                "task_id": task.task_id,
                "status": "error",
                "score": 1e30,
                "metrics": {},
                "feedback": "Evaluator container exited non-zero",
                "artifacts": {"stderr": result.stderr[-4000:]},
            }

        return self._parse_output(task.task_id, result.stdout, result.stderr)

    def _ensure_image(self, task: TaskSpec) -> str:
        cached = self._built_images.get(task.task_id)
        if cached:
            return cached

        self._ensure_parent_image(task)
        image = f"prompt-eval-{task.task_id}:latest"
        result = subprocess.run(
            ["docker", "build", "-t", image, str(task.evaluator_dir)],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Docker build failed for {task.task_id}: {result.stderr}")
        self._built_images[task.task_id] = image
        return image

    def _ensure_parent_image(self, task: TaskSpec) -> None:
        dockerfile = task.evaluator_dir / "Dockerfile"
        if not dockerfile.exists():
            return

        match = re.search(r"^FROM\s+([^\s]+)", dockerfile.read_text(encoding="utf-8"), flags=re.MULTILINE)
        if not match:
            return

        parent_image = match.group(1)
        if parent_image not in SHARED_IMAGE_BUILDS or parent_image in self._built_shared_images:
            return
        if self._docker_image_exists(parent_image):
            self._built_shared_images.add(parent_image)
            return

        dockerfile_path, build_context = SHARED_IMAGE_BUILDS[parent_image]
        result = subprocess.run(
            [
                "docker",
                "build",
                "-f",
                str(dockerfile_path),
                "-t",
                parent_image,
                str(build_context),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Docker build failed for shared image {parent_image}: {result.stderr}")
        self._built_shared_images.add(parent_image)

    def _docker_image_exists(self, image: str) -> bool:
        result = subprocess.run(
            ["docker", "image", "inspect", image],
            capture_output=True,
            text=True,
            check=False,
        )
        return result.returncode == 0

    def _parse_output(self, task_id: str, stdout: str, stderr: str) -> dict[str, object]:
        try:
            payload = json.loads(stdout.strip())
        except json.JSONDecodeError as exc:
            return {
                "task_id": task_id,
                "status": "error",
                "score": 1e30,
                "metrics": {},
                "feedback": f"Evaluator returned invalid JSON: {exc}",
                "artifacts": {"stdout": stdout[-4000:], "stderr": stderr[-4000:]},
            }

        payload.setdefault("task_id", task_id)
        payload.setdefault("status", "success")
        payload.pop("combined_score", None)
        payload.setdefault("score", float(payload.get("metrics", {}).get("score", payload.get("metrics", {}).get("geom_mean_us", 1e30))))
        payload.setdefault("metrics", {})
        payload.setdefault("feedback", "")
        if stderr.strip():
            artifacts = payload.setdefault("artifacts", {})
            if isinstance(artifacts, dict):
                artifacts.setdefault("stderr", stderr[-4000:])
        return payload


def evaluate_workspace_once(
    task: TaskSpec,
    workspace_dir: Path,
    timeout_seconds: int | None = None,
    assigned_gpu: str | None = None,
) -> dict[str, object]:
    evaluator = ContainerEvaluator()
    return evaluator.evaluate_workspace(task, workspace_dir, timeout_seconds=timeout_seconds, assigned_gpu=assigned_gpu)
