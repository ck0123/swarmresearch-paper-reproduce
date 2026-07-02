# SwarmResearch Reproduction Package

This directory contains the minimal files needed to reproduce SwarmResearch task
runs and inspect the released run data.

## Directories

- `agent_configs/`: Agent runtime configurations used by the runner.
  - `pi-swarm-experimental/`: PI multi-agent sweep configuration.
  - `shepherd/`: swarmresearch skills used

- `evaluation/`: Generic task runner and evaluator harness. It copies task
  workspaces into run directories, starts the eval server, builds evaluator
  Docker images, and runs the selected agent loop.

- `tasks/`: Benchmark task definitions. Each task has a `task.yaml`, a
  starting `workspace/`, and an `evaluator/` Docker context.
  - `math/tasks/`
  - `ADRS/tasks/`
  - `ale_bench_lite/tasks/`

- `data/`: SwarmResearch run trajectories, logs, and solutions for main benchmark. See
  `data/swarmresearch/README.md` for the layout.

- `speculative_decoding/`: Speculative decoding case study environments and best solution crafted

## Example

Run the PI config sweep on `third_autocorr_ineq`:

```bash
uv run --extra server python swarmresearch_reproduce/evaluation/run_problem_config_sweep.py \
  --tasks-root swarmresearch_reproduce/tasks/math/tasks \
  --task-id third_autocorr_ineq \
  --output-dir swarmresearch_reproduce/runs/pi \
  --agent-dir swarmresearch_reproduce/agent_configs/pi-swarm-experimental \
  --worker-image task \
  --no-skill \
  --parallel-configs
```

This requires Docker, `uv`, and credentials for the model provider configured in
the selected agent config.
