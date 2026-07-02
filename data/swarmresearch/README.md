# SwarmResearch Run Data

This directory contains the SwarmResearch trajectories. One directory per task. We ran SwarmResearch for ~100$/task but only report solution at 50$ mark in paper to compare with baselines. 50$ commit cutoffs recorded in closest_50usd_commit_cutoffs.csv

Each task directory has the shape:
```text
<task>/
  .claude/
  working_directory/
  log_all_oneline.txt
```

## `.claude/`

`.claude/` contains the Claude/agent runtime logs for that task.

The most important files are under:

```text
.claude/projects/
```

These are JSONL conversation transcript files. Each JSONL file corresponds to one agent
session. 

## `working_directory/`

`working_directory/` contains the final working-directory artifacts copied from
the run.

Common files:

- `initial_program.py` or `initial_program.cpp`: final solution file in the run directory.
- `prompt.md`: task prompt shown to the agent.
- `task_id.txt`: task identifier.
- `task_eval_config.json`: evaluator/runtime configuration.


## `log_all_oneline.txt`

`log_all_oneline.txt` is a compact git history for the run, in graph/oneline form.

Use it to inspect:

- scored commits,
- branch structure,
- commit subjects describing attempted approaches,
- whether a commit was a new branch or a continuation,
- whether there were merge commits.
