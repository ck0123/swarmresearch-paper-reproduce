from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import time
import urllib.request
from pathlib import Path


DEFAULT_SERVER_PORT = 8865
SERVER_START_ATTEMPTS = 3
SERVER_READY_TIMEOUT_SECONDS = 60
TASK_EVAL_COUNTER_FILENAME = "task_eval_invocation_count.txt"
TASK_EVAL_COUNTER_MOUNT_PATH = "/agent-home/task_eval_invocation_count.txt"
PI_DEFAULT_PACKAGE_EXTENSION_NAMES = ("cost-tracker", "temperature", "bash-timeout")
PI_OPTIONAL_EXTENSION_NAMES = ("turn-limit",)
PI_LOCAL_EXTENSIONS_DIR = Path(__file__).resolve().parent / "docker" / "task_base" / "pi-extensions"
FORWARDED_ENV_NAMES = [
    "AGENT_COUNT",
    "MAX_ITERATIONS",
    "PI_TURN_LIMIT",
    "TASK_EVAL_ONE_PER_AGENT_ROUND",
    "TASK_EVAL_AGENT_ROUND_STATE_PATH",
    "AWS_BEARER_TOKEN_BEDROCK",
    "AWS_PROFILE",
    "AWS_DEFAULT_REGION",
    "AWS_REGION",
    "AWS_REGION_NAME",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "AWS_SHARED_CREDENTIALS_FILE",
    "AWS_CONFIG_FILE",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "CLAUDE_CODE_USE_BEDROCK",
    "CLAUDE_CODE_BEDROCK_BASE_URL",
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "OPENAI_ORGANIZATION",
    "OPENAI_ORG_ID",
    "OPENAI_PROJECT",
    "OPENAI_PROJECT_ID",
    "OPENAI_API_BASE",
    "CODEX_API_KEY",
    "BATCH_SIZE",
    "CONTINUE_MESSAGE",
    "MAX_BATCHES_PER_ROUND",
    "MAX_STAGNANT_ROUNDS",
    "MAX_TOTAL_AGENT_CALLS",
    "ISOLATE_AGENT_USERS",
    "TASK_EVAL_HARD_STOP_WAIT_SECONDS",
]


def _pi_runtime_extension_names() -> tuple[str, ...]:
    extension_names = list(PI_DEFAULT_PACKAGE_EXTENSION_NAMES)
    if os.environ.get("PI_TURN_LIMIT", "1") == "1":
        extension_names.extend(PI_OPTIONAL_EXTENSION_NAMES)
    return tuple(extension_names)


def detect_agent_runtime(config_dir: Path) -> tuple[str, Path]:
    for name in (".codex", ".claude", ".pi"):
        runtime = config_dir / name
        if runtime.exists():
            return name.removeprefix("."), runtime
    raise FileNotFoundError(f"expected one of .codex, .claude, .pi in {config_dir}")


def derive_runtime_dir(run_dir: Path) -> Path:
    return run_dir.parent / ".agent_runtime" / run_dir.name


def loop_uses_unix_user_isolation(loop_script: Path) -> bool:
    try:
        text = loop_script.read_text(encoding="utf-8")
    except FileNotFoundError:
        return False
    return 'ISOLATE_AGENT_USERS="${ISOLATE_AGENT_USERS:-1}"' in text


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


def pick_server_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        if sock.connect_ex(("localhost", DEFAULT_SERVER_PORT)) != 0:
            return DEFAULT_SERVER_PORT
    return find_free_port()


def find_free_port_from(start_port: int) -> int:
    port = start_port
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            if sock.connect_ex(("localhost", port)) != 0:
                return port
        port += 1


def kill_server_by_port(port: int) -> None:
    subprocess.run(["bash", "-c", f"fuser -k {port}/tcp 2>/dev/null || true"], check=False)


def wait_for_server_ready(port: int) -> bool:
    for _ in range(SERVER_READY_TIMEOUT_SECONDS):
        time.sleep(1)
        try:
            health = urllib.request.urlopen(f"http://localhost:{port}/health", timeout=2)
            ready = urllib.request.urlopen(f"http://localhost:{port}/ready", timeout=2)
            if health.status == 200 and ready.status == 200:
                return True
        except Exception:
            pass
    return False


def start_server(tasks_root: Path, port: int, repo_root: Path) -> bool:
    server_script = repo_root / "swarmresearch_reproduce" / "evaluation" / "server.py"
    temp_root = Path(
        os.environ.get("BENCH_GOAL_PLUS_TMPDIR")
        or os.environ.get("TMPDIR")
        or Path.home() / ".tmp"
    ).expanduser()
    temp_root.mkdir(parents=True, exist_ok=True)
    server_log = temp_root / f"task_eval_server_{port}.log"
    with open(server_log, "w", encoding="utf-8") as log:
        process = subprocess.Popen(
            ["uv", "run", "--extra", "server", str(server_script), "--tasks-root", str(tasks_root), "--port", str(port)],
            stdout=log,
            stderr=log,
            cwd=str(repo_root),
            env=os.environ.copy(),
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    if wait_for_server_ready(port):
        return True
    try:
        os.killpg(process.pid, 15)
    except Exception:
        pass
    print(f"[run] ERROR: task eval server failed to start; check {server_log}")
    return False


def ensure_server_running(tasks_root: Path, port: int, repo_root: Path) -> int:
    try:
        health = urllib.request.urlopen(f"http://localhost:{port}/health", timeout=2)
        ready = urllib.request.urlopen(f"http://localhost:{port}/ready", timeout=2)
        if health.status == 200 and ready.status == 200:
            print(f"[run] task eval server already running on port {port}")
            return port
    except Exception:
        pass
    candidate_port = port
    for attempt in range(1, SERVER_START_ATTEMPTS + 1):
        candidate_port = port if attempt == 1 else find_free_port_from(candidate_port + 1)
        kill_server_by_port(candidate_port)
        print(f"[run] Starting task eval server on port {candidate_port} ({attempt}/{SERVER_START_ATTEMPTS})")
        if start_server(tasks_root, candidate_port, repo_root):
            return candidate_port
    raise RuntimeError("task eval server failed to start")


def _base_container_command(*, container_name: str, run_dir: Path, agent_kind: str, runtime_mount_source: Path, server_port: int) -> list[str]:
    container_home = Path("/agent-home")
    runtime_mount_target = container_home / runtime_mount_source.name
    counter_host_path = runtime_mount_source.parent / TASK_EVAL_COUNTER_FILENAME

    cmd = [
        "docker",
        "run",
        "-d",
        "--name",
        container_name,
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "-e",
        f"HOME={container_home}",
        "-e",
        "PATH=/workspace/.local/bin:/opt/cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "-e",
        f"TASK_EVAL_SERVER_URL=http://host.docker.internal:{server_port}",
        "-e",
        f"{agent_kind.upper()}_CONFIG_DIR={runtime_mount_target}",
        "-v",
        f"{run_dir}:/workspace",
        "-v",
        f"{runtime_mount_source}:{runtime_mount_target}",
        "-v",
        f"{counter_host_path}:{TASK_EVAL_COUNTER_MOUNT_PATH}",
        "--add-host",
        "host.docker.internal:host-gateway",
        "--entrypoint",
        "sleep",
    ]

    if agent_kind == "codex":
        cmd.extend(["-e", f"CODEX_HOME={runtime_mount_target}"])
    elif agent_kind == "pi":
        cmd.extend(["-e", f"PI_CODING_AGENT_DIR={runtime_mount_target}"])

    host_aws_dir = Path.home() / ".aws"
    if host_aws_dir.exists():
        cmd.extend(["-v", f"{host_aws_dir}:{container_home / '.aws'}:ro"])
    for env_name in FORWARDED_ENV_NAMES:
        env_value = os.environ.get(env_name)
        if env_value:
            cmd.extend(["-e", f"{env_name}={env_value}"])
    return cmd


def _runtime_mount_target(runtime_mount_source: Path) -> Path:
    return Path("/agent-home") / runtime_mount_source.name


def _update_pi_settings_packages(runtime_dir: Path) -> None:
    settings_path = runtime_dir / "settings.json"
    settings: dict[str, object] = {}
    if settings_path.exists():
        settings = json.loads(settings_path.read_text(encoding="utf-8"))

    raw_packages = settings.get("packages", [])
    packages = raw_packages if isinstance(raw_packages, list) else []
    package_set = {entry for entry in packages if isinstance(entry, str)}
    package_set.update(f"agent/extensions/{name}" for name in _pi_runtime_extension_names())
    package_set = {
        entry
        for entry in package_set
        if not entry.startswith("agent/extensions/")
        or (runtime_dir / "agent" / "extensions" / entry.removeprefix("agent/extensions/")).is_dir()
    }
    settings["packages"] = sorted(package_set)
    settings_path.write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")


def _pi_local_extension_names(runtime_dir: Path) -> tuple[str, ...]:
    settings_path = runtime_dir / "settings.json"
    requested_names: list[str] = []
    if settings_path.exists():
        settings = json.loads(settings_path.read_text(encoding="utf-8"))
        raw_packages = settings.get("packages", [])
        packages = raw_packages if isinstance(raw_packages, list) else []
        for entry in packages:
            if isinstance(entry, str) and entry.startswith("agent/extensions/"):
                requested_names.append(entry.removeprefix("agent/extensions/"))
    return tuple(dict.fromkeys([*_pi_runtime_extension_names(), *requested_names]))


def _mirror_pi_subagent_user_config(runtime_dir: Path) -> None:
    """Mirror project-style config into pi-subagents' hardcoded user paths."""
    agents_src = runtime_dir / "agents"
    if agents_src.is_dir():
        shutil.copytree(agents_src, runtime_dir / "agent" / "agents", dirs_exist_ok=True)


def _install_local_pi_runtime_extensions(runtime_mount_source: Path) -> None:
    extensions_dir = runtime_mount_source / "agent" / "extensions"
    extensions_dir.mkdir(parents=True, exist_ok=True)

    for extension_name in _pi_local_extension_names(runtime_mount_source):
        source = PI_LOCAL_EXTENSIONS_DIR / extension_name
        if not source.is_dir():
            continue
        shutil.copytree(source, extensions_dir / extension_name, dirs_exist_ok=True)


def _install_pi_runtime_extensions(container_name: str, runtime_mount_source: Path) -> None:
    _install_local_pi_runtime_extensions(runtime_mount_source)
    runtime_mount_target = _runtime_mount_target(runtime_mount_source)
    extension_names = " ".join(_pi_local_extension_names(runtime_mount_source))
    script = f"""
set -euo pipefail
mkdir -p {runtime_mount_target}/agent/extensions
for ext_name in {extension_names}; do
  src=/opt/pi-extensions/${{ext_name}}
  dst={runtime_mount_target}/agent/extensions/${{ext_name}}
  if [ -d "$src" ] && [ ! -d "$dst" ]; then
    cp -R "$src" "$dst"
  fi
done
"""
    subprocess.run(
        ["docker", "exec", container_name, "bash", "-lc", script],
        check=True,
    )
    _update_pi_settings_packages(runtime_mount_source)
    _mirror_pi_subagent_user_config(runtime_mount_source)


def start_fresh_worker_container(
    *,
    container_name: str,
    run_dir: Path,
    runtime_dir: Path,
    image: str,
    server_port: int,
    agent_dir: Path,
) -> None:
    agent_kind, runtime_src = detect_agent_runtime(agent_dir)
    runtime_dst = runtime_dir / runtime_src.name
    runtime_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(runtime_src, runtime_dst)

    if agent_kind == "codex":
        host_auth = Path.home() / ".codex" / "auth.json"
        if host_auth.exists():
            shutil.copy2(host_auth, runtime_dst / "auth.json")

    shutil.copy2(agent_dir / "loop.sh", run_dir / "loop.sh")
    (run_dir / "loop.sh").chmod(0o755)

    cmd = _base_container_command(
        container_name=container_name,
        run_dir=run_dir,
        agent_kind=agent_kind,
        runtime_mount_source=runtime_dst,
        server_port=server_port,
    )
    cmd.extend([image, "infinity"])
    subprocess.run(cmd, check=True)
    if agent_kind == "pi":
        _install_pi_runtime_extensions(container_name, runtime_dst)

    gitignore = ["__pycache__/", ".codex/", ".claude/", ".claude.json", ".pi/"]
    (run_dir / ".gitignore").write_text("\n".join(gitignore) + "\n", encoding="utf-8")
    for git_cmd in (
        ["git", "init"],
        ["git", "config", "user.email", "runner@task-eval.local"],
        ["git", "config", "user.name", "Task Runner"],
        ["git", "add", "."],
        ["git", "commit", "-m", "initial"],
    ):
        subprocess.run(["docker", "exec", "-w", "/workspace", container_name] + git_cmd, check=True)


def start_existing_worker_container(
    *,
    container_name: str,
    run_dir: Path,
    runtime_dir: Path,
    image: str,
    server_port: int,
    agent_dir: Path,
    loop_script: Path,
    resume_state_script: Path,
) -> None:
    agent_kind, _ = detect_agent_runtime(agent_dir)
    runtime_kind, runtime_mount_source = detect_agent_runtime(runtime_dir)
    if agent_kind != runtime_kind:
        raise RuntimeError(
            f"resume agent runtime kind {agent_kind!r} does not match persisted runtime kind {runtime_kind!r}"
        )

    cmd = _base_container_command(
        container_name=container_name,
        run_dir=run_dir,
        agent_kind=runtime_kind,
        runtime_mount_source=runtime_mount_source,
        server_port=server_port,
    )
    cmd.extend(["-v", f"{loop_script.resolve()}:/resume-loop.sh:ro"])
    cmd.extend(["-v", f"{resume_state_script.resolve()}:/resume-state.py:ro"])
    cmd.extend([image, "infinity"])
    subprocess.run(cmd, check=True)
    if runtime_kind == "pi":
        _install_pi_runtime_extensions(container_name, runtime_mount_source)


def stop_container(container_name: str) -> None:
    subprocess.run(["docker", "stop", container_name], check=False)
    subprocess.run(["docker", "rm", container_name], check=False)


def run_loop_in_container(
    container_name: str,
    timeout_minutes: int,
    *,
    loop_path: str = "loop.sh",
    run_as_root: bool = False,
) -> None:
    cmd = ["docker", "exec"]
    if run_as_root:
        cmd.extend(["--user", "root"])
    cmd.extend(["-w", "/workspace", container_name, "bash", loop_path])
    subprocess.run(cmd, check=True, timeout=timeout_minutes * 60)
