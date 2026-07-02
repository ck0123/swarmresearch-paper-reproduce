#!/usr/bin/env python3
"""Standalone workspace-eval client copied into run directories as task-eval."""

from __future__ import annotations

import argparse
import fcntl
import io
import json
import os
import pathlib
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request
import uuid


DEFAULT_SERVER_URL = "http://host.docker.internal:8865"
DEFAULT_EVAL_TIMEOUT_SECONDS = 120
LOCAL_CONFIG_PATH = pathlib.Path("task_eval_config.json")
DEFAULT_EXCLUDES = {".git", ".codex", ".claude", ".pi", "__pycache__", ".pytest_cache"}
# POST_EVAL_MESSAGE = "You have used up your experiment's evaluation budget. Add a commit message and end immediately. Your commit message must follow this exact format: `{concise experiment description} | score = {score from standard evaluator or concise failure reason}"
# POST_EVAL_MESSAGE = "You have used up your experiment's evaluation budget. Record your results and continue to the next experiment or end immediately."
POST_EVAL_MESSAGE = "Evaluation complete. Decide whether to continue or stop. Be conservative with iterations. If you have fully implemented and evaluated one idea, stop. Do not branch into substantially different ideas. Only continue if the result clearly suggests a small, targeted fix to the same idea. If so, edit initial_program.py and rerun ./task-eval. If the idea still fails after a few attempts, stop immediately. Before finishing, create or update findings.md with the result and commit all changes."
# POST_EVAL_MESSAGE = ""
SCORE_SEPARATOR = "| score ="


def load_local_config() -> dict[str, object]:
    if not LOCAL_CONFIG_PATH.exists():
        return {}
    try:
        payload = json.loads(LOCAL_CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def create_archive(workspace_dir: pathlib.Path) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for path in sorted(workspace_dir.rglob("*")):
            relative_parts = path.relative_to(workspace_dir).parts
            if any(part in DEFAULT_EXCLUDES for part in relative_parts):
                continue
            archive.add(path, arcname=path.relative_to(workspace_dir))
    return buffer.getvalue()


def multipart_body(task_id: str, archive_bytes: bytes, timeout_seconds: int | None) -> tuple[bytes, str]:
    boundary = f"----task-eval-{uuid.uuid4().hex}"
    lines: list[bytes] = []
    if timeout_seconds is not None:
        lines.extend(
            [
                f"--{boundary}\r\n".encode(),
                b'Content-Disposition: form-data; name="timeout_seconds"\r\n\r\n',
                f"{timeout_seconds}\r\n".encode(),
            ]
        )
    lines.extend(
        [
            f"--{boundary}\r\n".encode(),
            b'Content-Disposition: form-data; name="bundle"; filename="workspace.tar.gz"\r\n',
            b"Content-Type: application/gzip\r\n\r\n",
            archive_bytes,
            b"\r\n",
            f"--{boundary}--\r\n".encode(),
        ]
    )
    return b"".join(lines), boundary


def load_timeout_seconds(config: dict[str, object] | None = None) -> int:
    payload = config if config is not None else load_local_config()
    try:
        return max(1, int(payload.get("eval_timeout_seconds", DEFAULT_EVAL_TIMEOUT_SECONDS)))
    except Exception:
        return DEFAULT_EVAL_TIMEOUT_SECONDS


def should_emit_post_eval_message(config: dict[str, object] | None = None) -> bool:
    payload = config if config is not None else load_local_config()
    raw_value = payload.get("emit_post_eval_message", True)
    if isinstance(raw_value, bool):
        return raw_value
    if isinstance(raw_value, str):
        normalized = raw_value.strip().lower()
        if normalized in {"0", "false", "no", "off"}:
            return False
        if normalized in {"1", "true", "yes", "on"}:
            return True
    return bool(raw_value)


def post_eval_message(config: dict[str, object] | None = None) -> str:
    payload = config if config is not None else load_local_config()
    configured_message = payload.get("post_eval_message")
    if isinstance(configured_message, str) and configured_message.strip():
        return configured_message.strip()
    return POST_EVAL_MESSAGE


def task_eval_server_url(config: dict[str, object] | None = None) -> str:
    configured_env_url = os.environ.get("TASK_EVAL_SERVER_URL")
    if isinstance(configured_env_url, str) and configured_env_url.strip():
        return configured_env_url.strip().rstrip("/")
    payload = config if config is not None else load_local_config()
    configured_url = payload.get("task_eval_server_url")
    if isinstance(configured_url, str) and configured_url.strip():
        return configured_url.strip().rstrip("/")
    return DEFAULT_SERVER_URL.rstrip("/")


def env_flag_enabled(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def auto_commit_enabled() -> bool:
    return env_flag_enabled("TASK_EVAL_AUTO_COMMIT")


def managed_subagent_required() -> bool:
    return env_flag_enabled("TASK_EVAL_REQUIRE_MANAGED_SUBAGENT")


def managed_subagent_active() -> bool:
    return env_flag_enabled("MANAGED_SUBAGENT_ACTIVE")


def hard_stop_after_commit_enabled() -> bool:
    return env_flag_enabled("TASK_EVAL_HARD_STOP_AFTER_COMMIT")


def hard_stop_process_group_id() -> int | None:
    raw_value = os.environ.get("TASK_EVAL_HARD_STOP_PGID", "").strip()
    if not raw_value:
        return None
    try:
        pgid = int(raw_value)
    except ValueError:
        print(f"warning: ignoring invalid TASK_EVAL_HARD_STOP_PGID={raw_value!r}", file=sys.stderr)
        return None
    if pgid <= 1:
        print(f"warning: ignoring unsafe TASK_EVAL_HARD_STOP_PGID={pgid}", file=sys.stderr)
        return None
    return pgid


def increment_task_eval_counter(counter_path: pathlib.Path | None) -> int | None:
    if counter_path is None:
        return None

    counter_path.parent.mkdir(parents=True, exist_ok=True)
    with counter_path.open("a+", encoding="utf-8") as counter_file:
        fcntl.flock(counter_file.fileno(), fcntl.LOCK_EX)
        counter_file.seek(0)
        raw_value = counter_file.read().strip()
        try:
            current_value = int(raw_value)
        except ValueError:
            current_value = 0
        new_value = current_value + 1
        counter_file.seek(0)
        counter_file.truncate()
        counter_file.write(f"{new_value}\n")
        counter_file.flush()
        os.fsync(counter_file.fileno())
        fcntl.flock(counter_file.fileno(), fcntl.LOCK_UN)
    return new_value


def infer_task_id(workspace_dir: pathlib.Path) -> str:
    task_id_file = workspace_dir / "task_id.txt"
    if task_id_file.exists():
        return task_id_file.read_text(encoding="utf-8").strip()
    raise SystemExit("error: task_id.txt not found and --task-id not provided")


def validate_commit_message(commit_message: str | None) -> str:
    if commit_message is None or not commit_message.strip():
        raise SystemExit("error: --commit-message is required when TASK_EVAL_AUTO_COMMIT is enabled")
    message = " ".join(commit_message.strip().split())
    if SCORE_SEPARATOR in message:
        raise SystemExit("error: --commit-message must not already contain '| score ='")
    return message


def format_score_value(score: object) -> str:
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise ValueError("score is not numeric")
    return f"{score:g}"


def evaluated_score(result: dict[str, object]) -> str:
    metrics = result.get("metrics")
    if isinstance(metrics, dict) and "combined_score" in metrics:
        return format_score_value(metrics["combined_score"])
    if "score" in result:
        return format_score_value(result["score"])
    raise ValueError("evaluation result did not include metrics.combined_score or numeric score")


def concise_failure_reason(result: dict[str, object]) -> str | None:
    for key in ("feedback", "detail", "error", "message"):
        value = result.get(key)
        if isinstance(value, str) and value.strip():
            return " ".join(value.strip().split())

    artifacts = result.get("artifacts")
    if isinstance(artifacts, dict):
        for key in ("error", "failure", "message", "stderr"):
            value = artifacts.get(key)
            if isinstance(value, str) and value.strip():
                return " ".join(value.strip().split())

    return None


def commit_score_suffix(result: dict[str, object]) -> str:
    if result.get("status") == "success":
        try:
            return evaluated_score(result)
        except ValueError as exc:
            raise SystemExit(f"error: {exc}") from exc

    reason = concise_failure_reason(result)
    if reason:
        return f"evaluator failure: {reason}"
    raise SystemExit("error: evaluator failed without a structured failure reason; refusing to auto-commit")


def git_commit_workspace(workspace_dir: pathlib.Path, message: str, score_suffix: str) -> None:
    subject = f"{message} {SCORE_SEPARATOR} {score_suffix}"
    try:
        subprocess.run(["git", "add", "-A"], cwd=workspace_dir, check=True)
        subprocess.run(["git", "commit", "-m", subject], cwd=workspace_dir, check=True)
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"error: auto-commit failed while running {' '.join(exc.cmd)}") from exc
    print(f"auto-commit complete: {subject}")


def launch_process_group_hard_stop_helper(pgid: int) -> None:
    helper_code = r"""
import os
import signal
import sys
import time


def group_alive(target_pgid):
    try:
        os.killpg(target_pgid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


pgid = int(sys.argv[1])
if pgid <= 1:
    print(f"warning: refusing to hard-stop unsafe process group {pgid}", file=sys.stderr)
    sys.exit(0)
if pgid == os.getpgrp():
    print(f"warning: refusing to hard-stop helper process group {pgid}", file=sys.stderr)
    sys.exit(0)

try:
    os.killpg(pgid, signal.SIGTERM)
except ProcessLookupError:
    sys.exit(0)
except PermissionError as exc:
    print(f"warning: could not hard-stop process group {pgid}: {exc}", file=sys.stderr)
    sys.exit(0)

time.sleep(1.0)
if group_alive(pgid):
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except PermissionError as exc:
        print(f"warning: could not SIGKILL process group {pgid}: {exc}", file=sys.stderr)
"""
    subprocess.Popen(
        [sys.executable, "-c", helper_code, str(pgid)],
        start_new_session=True,
        close_fds=True,
    )


def wait_for_hard_stop() -> None:
    raw_value = os.environ.get("TASK_EVAL_HARD_STOP_WAIT_SECONDS", "").strip()
    if raw_value:
        try:
            wait_seconds = max(0.0, float(raw_value))
        except ValueError:
            print(
                f"warning: ignoring invalid TASK_EVAL_HARD_STOP_WAIT_SECONDS={raw_value!r}",
                file=sys.stderr,
            )
            wait_seconds = 30.0
    else:
        wait_seconds = 30.0

    deadline = time.monotonic() + wait_seconds
    while time.monotonic() < deadline:
        time.sleep(0.1)

    raise SystemExit("error: hard-stop helper did not terminate this process group")


def launch_pi_hard_stop_helper(start_pid: int) -> None:
    helper_code = r"""
import os
import signal
import sys
import time


def parent_pid(pid):
    try:
        with open(f"/proc/{pid}/stat", "r", encoding="utf-8") as stat_file:
            return int(stat_file.read().split()[3])
    except Exception:
        return None


def is_pi_process(pid):
    try:
        with open(f"/proc/{pid}/comm", "r", encoding="utf-8") as comm_file:
            if comm_file.read().strip() == "pi":
                return True
    except Exception:
        pass
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as cmdline_file:
            parts = [part.decode(errors="ignore") for part in cmdline_file.read().split(b"\0") if part]
    except Exception:
        return False
    return bool(parts) and os.path.basename(parts[0]) == "pi"


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


pid = int(sys.argv[1])
seen = set()
while pid and pid > 1 and pid not in seen:
    seen.add(pid)
    if is_pi_process(pid):
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            sys.exit(0)
        except PermissionError as exc:
            print(f"warning: could not hard-stop pi process {pid}: {exc}", file=sys.stderr)
            sys.exit(0)
        time.sleep(1.0)
        if alive(pid):
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except PermissionError as exc:
                print(f"warning: could not SIGKILL pi process {pid}: {exc}", file=sys.stderr)
        sys.exit(0)
    next_pid = parent_pid(pid)
    if next_pid is None:
        break
    pid = next_pid

print("warning: TASK_EVAL_HARD_STOP_AFTER_COMMIT was enabled but no pi ancestor was found", file=sys.stderr)
"""
    subprocess.Popen(
        [sys.executable, "-c", helper_code, str(start_pid)],
        start_new_session=True,
        close_fds=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", default=None)
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--commit-message", default=None)
    args = parser.parse_args()

    if managed_subagent_required() and not managed_subagent_active():
        raise SystemExit("error: task-eval is disabled for the orchestrator; use managed_subagent instead")

    workspace_dir = pathlib.Path(args.workspace).resolve()
    task_id = args.task_id or infer_task_id(workspace_dir)
    auto_commit = auto_commit_enabled()
    commit_message = validate_commit_message(args.commit_message) if auto_commit else None
    config = load_local_config()
    timeout_seconds = load_timeout_seconds(config)
    counter_path_value = config.get("task_eval_counter_path")
    counter_path = pathlib.Path(counter_path_value) if isinstance(counter_path_value, str) and counter_path_value else None
    archive_bytes = create_archive(workspace_dir)
    body, boundary = multipart_body(task_id, archive_bytes, timeout_seconds)

    server_url = task_eval_server_url(config)
    url = f"{server_url}/tasks/{task_id}/evaluate"
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )

    increment_task_eval_counter(counter_path)

    try:
        with urllib.request.urlopen(req, timeout=timeout_seconds + 30) as resp:
            payload = resp.read().decode()
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode()
        print(f"error: server returned HTTP {exc.code}: {error_body}", file=sys.stderr)
        raise SystemExit(1) from exc
    except urllib.error.URLError as exc:
        print(f"error: could not reach task eval server at {url!r}: {exc.reason}", file=sys.stderr)
        raise SystemExit(1) from exc

    result = json.loads(payload)
    output = json.dumps(result, indent=2, sort_keys=True)
    print(output)
    if args.output_json:
        pathlib.Path(args.output_json).write_text(output + "\n", encoding="utf-8")
    if auto_commit:
        assert commit_message is not None
        git_commit_workspace(workspace_dir, commit_message, commit_score_suffix(result))
        if hard_stop_after_commit_enabled():
            sys.stdout.flush()
            sys.stderr.flush()
            pgid = hard_stop_process_group_id()
            if pgid is not None:
                launch_process_group_hard_stop_helper(pgid)
            else:
                launch_pi_hard_stop_helper(os.getppid())
            wait_for_hard_stop()
    elif should_emit_post_eval_message(config):
        print(post_eval_message(config))


if __name__ == "__main__":
    main()
