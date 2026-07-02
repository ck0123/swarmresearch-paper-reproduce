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
