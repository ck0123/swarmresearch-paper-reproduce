#!/usr/bin/env python3
"""Re-evaluate all branch tips in an ale_bench_lite run repo and report the best."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from swarmresearch_reproduce.evaluation.container_eval import ContainerEvaluator
from swarmresearch_reproduce.evaluation.tasks import get_task_spec


TASKS_ROOT = Path(__file__).resolve().parent / "tasks"


def infer_task_id(run_dir: Path) -> str:
    task_id_file = run_dir / "task_id.txt"
    if task_id_file.exists():
        return task_id_file.read_text(encoding="utf-8").strip()
    return run_dir.name.split("_")[0]


def get_refs(run_dir: Path) -> list[str]:
    result = subprocess.run(
        ["git", "for-each-ref", "--format=%(refname:short)", "refs/heads"],
        cwd=run_dir,
        check=True,
        capture_output=True,
        text=True,
    )
    refs = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if "HEAD" not in refs:
        refs.append("HEAD")
    return sorted(set(refs))


def materialize_ref(run_dir: Path, ref: str, dest_dir: Path) -> None:
    with open(dest_dir / ".git-archive.tar", "wb") as archive_file:
        subprocess.run(["git", "archive", ref], cwd=run_dir, check=True, stdout=archive_file)
    subprocess.run(["tar", "-xf", str(dest_dir / ".git-archive.tar"), "-C", str(dest_dir)], check=True)
    (dest_dir / ".git-archive.tar").unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--task-id", default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    task_id = args.task_id or infer_task_id(run_dir)
    task = get_task_spec(TASKS_ROOT, task_id)
    evaluator = ContainerEvaluator()

    rows = []
    for ref in get_refs(run_dir):
        with tempfile.TemporaryDirectory(prefix=f"finalize_{task_id}_") as tmpdir:
            workspace_dir = Path(tmpdir) / "workspace"
            shutil.copytree(task.workspace_dir, workspace_dir)
            materialize_ref(run_dir, ref, workspace_dir)
            result = evaluator.evaluate_workspace(task, workspace_dir)
        rows.append({"ref": ref, **result})

    rows.sort(key=lambda row: row["combined_score"], reverse=True)
    payload = {"task_id": task_id, "best": rows[0] if rows else None, "results": rows}
    output = json.dumps(payload, indent=2, sort_keys=True)
    print(output)
    if args.output_json:
        args.output_json.write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
