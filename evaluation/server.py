#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
from contextlib import asynccontextmanager
import tempfile
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from swarmresearch_reproduce.evaluation.bundle import extract_workspace_archive
from swarmresearch_reproduce.evaluation.container_eval import ContainerEvaluator
from swarmresearch_reproduce.evaluation.tasks import GpuPoolConfig, discover_tasks


class EvalResponse(BaseModel):
    task_id: str
    status: str
    score: float
    feedback: str = ""
    metrics: dict[str, Any] = Field(default_factory=dict)
    artifacts: dict[str, Any] = Field(default_factory=dict)


class GpuSlotPool:
    def __init__(self, config: GpuPoolConfig) -> None:
        self.config = config
        self._slots: asyncio.Queue[str] = asyncio.Queue()
        for device in config.devices:
            for _ in range(config.slots_per_device):
                self._slots.put_nowait(device)

    @asynccontextmanager
    async def lease(self):
        device = await self._slots.get()
        try:
            yield device
        finally:
            self._slots.put_nowait(device)


def build_gpu_pools(tasks_root: Path) -> tuple[dict[str, Any], dict[str, GpuSlotPool]]:
    tasks = discover_tasks(tasks_root)
    pool_configs: dict[str, GpuPoolConfig] = {}
    for spec in tasks.values():
        gpu_pool = spec.evaluation.gpu_pool
        if gpu_pool is None:
            continue
        existing = pool_configs.get(gpu_pool.name)
        if existing is not None and existing != gpu_pool:
            raise ValueError(
                f"conflicting gpu_pool definition for {gpu_pool.name!r}: "
                f"{existing} != {gpu_pool}"
            )
        pool_configs[gpu_pool.name] = gpu_pool
    pools = {
        name: GpuSlotPool(config)
        for name, config in pool_configs.items()
    }
    return tasks, pools


def create_app(tasks_root: Path) -> FastAPI:
    tasks, gpu_pools = build_gpu_pools(tasks_root)
    semaphores = {
        task_id: asyncio.Semaphore(spec.evaluation.max_concurrent)
        for task_id, spec in tasks.items()
    }
    evaluator = ContainerEvaluator()
    app = FastAPI(title="Generic Task Eval Server")

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/ready")
    async def ready() -> dict[str, str]:
        return {"status": "ready"}

    @app.post("/tasks/{task_id}/evaluate", response_model=EvalResponse)
    async def evaluate(task_id: str, bundle: UploadFile = File(...), timeout_seconds: int | None = Form(default=None)) -> EvalResponse:
        if task_id not in tasks:
            raise HTTPException(status_code=404, detail=f"unknown task_id: {task_id!r}")

        task = tasks[task_id]
        async with semaphores[task_id]:
            with tempfile.TemporaryDirectory(prefix=f"task_eval_{task_id}_") as tmpdir:
                workspace_dir = Path(tmpdir) / "workspace"
                workspace_dir.mkdir(parents=True, exist_ok=True)
                try:
                    await asyncio.to_thread(extract_workspace_archive, bundle.file, workspace_dir)
                    gpu_pool = task.evaluation.gpu_pool
                    if gpu_pool is None:
                        result = await asyncio.to_thread(
                            evaluator.evaluate_workspace,
                            task,
                            workspace_dir,
                            timeout_seconds=timeout_seconds,
                        )
                    else:
                        async with gpu_pools[gpu_pool.name].lease() as assigned_gpu:
                            result = await asyncio.to_thread(
                                evaluator.evaluate_workspace,
                                task,
                                workspace_dir,
                                timeout_seconds=timeout_seconds,
                                assigned_gpu=assigned_gpu,
                            )
                except Exception as exc:
                    raise HTTPException(status_code=500, detail=f"evaluation failed: {exc}") from exc
        return EvalResponse(**result)

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks-root", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8865)
    args = parser.parse_args()

    import uvicorn

    uvicorn.run(create_app(args.tasks_root.resolve()), host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
