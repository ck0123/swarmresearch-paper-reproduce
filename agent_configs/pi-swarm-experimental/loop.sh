#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${PWD}"
AGENT_COUNT="${AGENT_COUNT:-4}"
MAX_ITERATIONS="${MAX_ITERATIONS:-3}"
TEMPERATURE="${PI_TEMPERATURE:-1.0}"
TEMPERATURE_FLAG="${TEMPERATURE:+--temperature ${TEMPERATURE}}"
PROMPT_FILE="${PI_SWARM_PROMPT_FILE:-${ROOT_DIR}/prompt.md}"
PI_CONFIG_DIR="${PI_CONFIG_DIR:-${ROOT_DIR}/.pi}"
PI_AGENT_DIR="${PI_AGENT_DIR:-${PI_CONFIG_DIR}/agent}"
WORKTREE_ROOT="${PI_SWARM_WORKTREE_ROOT:-${ROOT_DIR}/.worktrees}"
LOG_ROOT="${PI_SWARM_LOG_ROOT:-${ROOT_DIR}/.swarm_logs}"
EXTENSIONS_ROOT="${PI_EXTENSIONS_ROOT:-${PI_AGENT_DIR}/extensions}"
PI_TURN_LIMIT="${PI_TURN_LIMIT:-20}"
SCORE_TRACKING_PATH="${PI_SCORE_TRACKING_PATH:-${LOG_ROOT}/score_tracking.json}"

MODELS=(
  "minimax.minimax-m2.5"
  # "moonshotai.kimi-k2.5"
)

if (( AGENT_COUNT < 1 )); then
  printf 'AGENT_COUNT (%s) must be at least 1\n' "$AGENT_COUNT" >&2
  exit 1
fi

if ! [[ "${PI_TURN_LIMIT}" =~ ^[1-9][0-9]*$ ]]; then
  printf 'PI_TURN_LIMIT (%s) must be a positive integer\n' "$PI_TURN_LIMIT" >&2
  exit 1
fi

agent_model() {
  local agent_index="$1"
  local model_count="${#MODELS[@]}"
  local base_count="$((AGENT_COUNT / model_count))"
  local remainder="$((AGENT_COUNT % model_count))"
  local threshold="$(((base_count + 1) * remainder))"
  local zero_based_index="$((agent_index - 1))"
  local model_index

  if (( zero_based_index < threshold )); then
    model_index="$((zero_based_index / (base_count + 1)))"
  else
    model_index="$((remainder + ((zero_based_index - threshold) / base_count)))"
  fi

  printf '%s\n' "${MODELS[$model_index]}"
}

ensure_clean_worktree() {
  local agent_index="$1"
  local agent_slug="researcher-${agent_index}"
  local worktree_path="${WORKTREE_ROOT}/${agent_slug}"

  if [ ! -d "${worktree_path}" ]; then
    (
      flock 9
      if [ ! -d "${worktree_path}" ]; then
        git worktree add --detach "${worktree_path}" HEAD
      fi
    ) 9>"${WORKTREE_ROOT}/.worktree.lock"
  fi
}

build_prompt() {
  local worktree_path="$1"
  local prompt_path="$2"
  local round="$3"

  cat > "${prompt_path}" <<EOF
Your git worktree is \`${worktree_path}\`.
This worktree has already been created for you.
Run all git history, branch, commit, and file-editing work from that worktree only.

EOF
  if (( round > 1 )); then
    printf '%s\n\n' "Execute exactly one more experiment following these instructions." >> "${prompt_path}"
  fi
  cat "${PROMPT_FILE}" >> "${prompt_path}"
}

load_score_direction() {
  python3 - "${ROOT_DIR}/task_eval_config.json" <<'PY'
import json
import sys
from pathlib import Path

config_path = Path(sys.argv[1])
try:
    payload = json.loads(config_path.read_text(encoding="utf-8"))
except FileNotFoundError:
    raise SystemExit(f"missing task eval config: {config_path}")
except json.JSONDecodeError as exc:
    raise SystemExit(f"invalid JSON in {config_path}: {exc}")

direction = payload.get("score_direction")
if direction not in {"maximize", "minimize"}:
    raise SystemExit(
        f"{config_path} must contain score_direction of 'maximize' or 'minimize', got: {direction!r}"
    )
print(direction)
PY
}

record_round_scores() {
  local round="$1"
  local logs_dir="$2"
  local score_direction="$3"

  python3 - "${ROOT_DIR}" "${LOG_ROOT}" "${SCORE_TRACKING_PATH}" "${round}" "${logs_dir}" "${score_direction}" <<'PY'
import json
import math
import re
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

root_dir = Path(sys.argv[1]).resolve()
log_root = Path(sys.argv[2]).resolve()
tracking_path = Path(sys.argv[3]).resolve()
round_number = int(sys.argv[4])
logs_dir = Path(sys.argv[5]).resolve()
score_direction = sys.argv[6]

AUTO_COMMIT_RE = re.compile(
    r"(?:^|\\n|\n|\r)auto-commit complete:\s*(?P<subject>.*?\|\s*score\s*=\s*.*?)(?:\\n|\n|\r|$)"
)
GIT_COMMIT_RE = re.compile(r"\[[^\]\r\n]*\s(?P<hash>[0-9a-f]{7,40})\]\s+(?P<subject>.*?\|\s*score\s*=\s*.*?)(?:\\n|\n|\r|$)")
SCORE_RE = re.compile(r"\|\s*score\s*=\s*(?P<score>.*)$")
NUMERIC_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?$")


def clean_text(value: str) -> str:
    value = value.replace("\\n", "\n")
    value = value.replace('\\"', '"')
    return " ".join(value.strip().split())


def parse_score(subject: str) -> tuple[str, str, float | None]:
    match = SCORE_RE.search(subject)
    if match is None:
        return "", "failure", None
    score_raw = clean_text(match.group("score"))
    if NUMERIC_RE.fullmatch(score_raw):
        score_value = float(score_raw)
        if math.isfinite(score_value):
            return score_raw, "numeric", score_value
    return score_raw, "failure", None


def matching_commit_hash(text: str, subject: str, before_offset: int) -> str | None:
    normalized_subject = clean_text(subject)
    candidates = []
    for match in GIT_COMMIT_RE.finditer(text[:before_offset]):
        git_subject = clean_text(match.group("subject"))
        if git_subject == normalized_subject:
            candidates.append(match.group("hash"))
    return candidates[-1] if candidates else None


def parse_log(path: Path) -> list[dict[str, object]]:
    try:
        agent_id: int | str = int(path.stem)
    except ValueError:
        agent_id = path.stem

    text = path.read_text(encoding="utf-8", errors="replace")
    records = []
    for match in AUTO_COMMIT_RE.finditer(text):
        subject = clean_text(match.group("subject"))
        score_raw, score_type, score_value = parse_score(subject)
        record: dict[str, object] = {
            "round": round_number,
            "agent_id": agent_id,
            "commit_hash": matching_commit_hash(text, subject, match.start()),
            "subject": subject,
            "score_raw": score_raw,
            "score_type": score_type,
            "log_path": str(path.relative_to(root_dir)) if path.is_relative_to(root_dir) else str(path),
        }
        if score_value is not None:
            record["score_value"] = score_value
        records.append(record)
    return records


def is_better(candidate: dict[str, object] | None, incumbent: dict[str, object] | None) -> bool:
    if candidate is None:
        return False
    if incumbent is None:
        return True
    candidate_score = float(candidate["score_value"])
    incumbent_score = float(incumbent["score_value"])
    if score_direction == "maximize":
        return candidate_score > incumbent_score
    return candidate_score < incumbent_score


def best_of(records: list[dict[str, object]]) -> dict[str, object] | None:
    best = None
    for record in records:
        if record.get("score_type") != "numeric":
            continue
        if is_better(record, best):
            best = record
    return best


def load_state() -> dict[str, object]:
    if not tracking_path.exists():
        return {
            "schema_version": 1,
            "run_dir": str(root_dir),
            "log_root": str(log_root),
            "score_direction": score_direction,
            "rounds": [],
            "best": None,
        }
    try:
        state = json.loads(tracking_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"invalid existing score tracking JSON {tracking_path}: {exc}")
    if not isinstance(state, dict):
        raise SystemExit(f"invalid existing score tracking JSON {tracking_path}: root must be an object")
    return state


commits: list[dict[str, object]] = []
if logs_dir.is_dir():
    for log_path in sorted(logs_dir.glob("*.log"), key=lambda item: item.name):
        commits.extend(parse_log(log_path))

round_best = best_of(commits)
round_entry = {
    "round": round_number,
    "commits": commits,
    "best": round_best,
}

state = load_state()
state["schema_version"] = 1
state["run_dir"] = str(root_dir)
state["log_root"] = str(log_root)
state["score_direction"] = score_direction
state["updated_at"] = datetime.now(timezone.utc).isoformat()

rounds = state.get("rounds")
if not isinstance(rounds, list):
    rounds = []
rounds = [entry for entry in rounds if not (isinstance(entry, dict) and entry.get("round") == round_number)]
rounds.append(round_entry)
rounds.sort(key=lambda entry: int(entry.get("round", 0)) if isinstance(entry, dict) else 0)
state["rounds"] = rounds

run_best = None
for entry in rounds:
    if not isinstance(entry, dict):
        continue
    candidate = entry.get("best")
    if isinstance(candidate, dict) and is_better(candidate, run_best):
        run_best = candidate
state["best"] = run_best

tracking_path.parent.mkdir(parents=True, exist_ok=True)
with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=tracking_path.parent, delete=False) as handle:
    tmp_path = Path(handle.name)
    json.dump(state, handle, indent=2, sort_keys=True)
    handle.write("\n")
tmp_path.replace(tracking_path)

if round_best is None:
    print(f"[round {round_number}] best score: none direction={score_direction}")
else:
    print(
        f"[round {round_number}] best score: {round_best['score_value']:g} "
        f"commit={round_best.get('commit_hash') or 'unknown'} "
        f"agent={round_best.get('agent_id')} direction={score_direction}"
    )

if run_best is None:
    print("[run] best score so far: none")
else:
    print(
        f"[run] best score so far: {run_best['score_value']:g} "
        f"round={run_best.get('round')} commit={run_best.get('commit_hash') or 'unknown'}"
    )
PY
}

run_round() {
  local round="$1"
  local logs_dir="$2"
  local failed_agents=()
  local pids=()
  local i

  mkdir -p "${logs_dir}" "${PI_AGENT_DIR}" "${WORKTREE_ROOT}"
  git config extensions.worktreeConfig true

  for ((i = 1; i <= AGENT_COUNT; i++)); do
    ensure_clean_worktree "$i"
  done

  for ((i = 1; i <= AGENT_COUNT; i++)); do
    local continue_flag=()
    local model
    local prompt_path
    local worktree_path

    model="$(agent_model "$i")"
    worktree_path="${WORKTREE_ROOT}/researcher-${i}"
    prompt_path="$(mktemp "${worktree_path}/prompt.researcher-${i}.XXXXXX.md")"
    build_prompt "${worktree_path}" "${prompt_path}" "${round}"
    if (( round > 1 )); then
      continue_flag=(--continue)
    fi
    local extension_args=(
      --extension "${EXTENSIONS_ROOT}/cost-tracker/index.ts"
      --extension "${EXTENSIONS_ROOT}/temperature/index.ts"
      --extension "${EXTENSIONS_ROOT}/bash-timeout/index.ts"
      --extension "${EXTENSIONS_ROOT}/turn-limit/index.ts"
    )

    (
      cd "${worktree_path}"
      exec setsid bash -c '
        TASK_EVAL_HARD_STOP_PGID="$$"
        export TASK_EVAL_HARD_STOP_PGID
        exec "$@"
      ' task-eval-agent \
        env \
        PI_CODING_AGENT_DIR="${PI_AGENT_DIR}" \
        TASK_EVAL_AUTO_COMMIT=1 \
        TASK_EVAL_HARD_STOP_AFTER_COMMIT=1 \
        TASK_EVAL_AGENT_ID="${i}" \
        TASK_EVAL_ROUND="${round}" \
        PI_TURN_LIMIT="${PI_TURN_LIMIT}" \
        pi --provider amazon-bedrock \
        --model "${model}" \
        "${extension_args[@]}" \
        --mode json \
        --thinking high \
        ${TEMPERATURE_FLAG} \
        "${continue_flag[@]}" \
        -p "$(cat "${prompt_path}")"
    ) >"${logs_dir}/${i}.log" 2>&1 &
    pids+=($!)
  done

  for i in "${!pids[@]}"; do
    agent_index="$((i + 1))"
    if ! wait "${pids[$i]}"; then
      if ! grep -Eq 'auto-commit complete: .*\| score =' "${logs_dir}/${agent_index}.log"; then
        failed_agents+=("${agent_index}")
      fi
    fi
  done

  find "${WORKTREE_ROOT}" -maxdepth 2 -type f -name 'prompt.researcher-*.md' -delete
  if ((${#failed_agents[@]} > 0)); then
    printf '[round %s] agent failures: %s\n' \
      "$round" "${failed_agents[*]}" >&2
  fi
  printf '[round %s] finished for %s agents\n' "$round" "$AGENT_COUNT"
}

main() {
  local round
  local score_direction

  mkdir -p "${LOG_ROOT}"
  score_direction="$(load_score_direction)"
  printf '[run] score direction: %s\n' "${score_direction}"

  for round in $(seq 1 "$MAX_ITERATIONS"); do
    printf '[round %s/%s] starting %s agents\n' \
      "$round" "$MAX_ITERATIONS" "$AGENT_COUNT"
    run_round "$round" "${LOG_ROOT}/round_${round}"
    record_round_scores "$round" "${LOG_ROOT}/round_${round}" "${score_direction}"
  done
}

main "$@"
