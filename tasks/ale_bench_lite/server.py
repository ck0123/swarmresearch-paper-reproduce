#!/usr/bin/env python3
"""ale_bench_lite wrapper around the generic task eval server."""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from swarmresearch_reproduce.evaluation.server import create_app


TASKS_ROOT = Path(__file__).resolve().parent / "tasks"
app = create_app(TASKS_ROOT)


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("TASK_EVAL_SERVER_PORT", "8865"))
    uvicorn.run(app, host="0.0.0.0", port=port)
