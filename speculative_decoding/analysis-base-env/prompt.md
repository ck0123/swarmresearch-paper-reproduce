Our eventual goal is to come up with new decoding schemes and techniques that speedup wall-clock decoding throughput. Example of effective and distinct decoding algorithms are speculative decoding, lookahead decoding, prompt lookup decoding, and token recycling. However, our current goal is to collect data and perform analysis studying decoding algorithms from different perspectives e.g. profiling GPU utilization, KV cache optimization, acceptance rate, prompt distributions, different drafting mechanisms, relationship between target verification logits and future tokens, etc. We want to perform experiments, collect data, look at decoding from many different angles with the only goal being to learn as much as diverse knowledge as possible to identify potential problems and bottlenecks. We’re going to focus on speculative decoding (canonical version implemented in initial_program.cpp). You are using the following target model and also have access to a draft model (you don't have to use it, you can come up with a target-only method too):
  - Target: gemma-4-26B-A4B-it (Q8_0 quantization, ~27 GB)
  - Draft:  gemma-4-26B-A4B-it MTP head (gemma4-assistant, Q8_0, ~461 MB) — used via mtp-spec mode

We are using a 30-item benchmark (from LiveCodeBench v6 + AIME 2026 + GPQA Diamond + HLE-MCQ) the target model passes reliably. We don't necessarily need to analyze from a token-level lossless decoding perspective, measuring downstream accuracy is another angle to look at.

The initial program contains a reference implementation that is compatible with the evaluator. 

Shepherd-specific instructions (Not important for explorers or optimizers):
- Your agent budget is 100. Stop when you've exhausted your agent budget
- All evaluations use 2 GPUs. Launch at most 4 search agents concurrently, 2 GPUs each (all 8 GPUs).
- Each agent gets a unique GPU pair and works in its own copy of this package (so concurrent
  `spec_modes` builds and `results/` don't collide).
- Use these four search-agent slots (GPU pair):
  - slot 0: GPUs 0,1
  - slot 1: GPUs 2,3
  - slot 2: GPUs 4,5
  - slot 3: GPUs 6,7

Evaluation:
- `./evaluator/task-eval` builds initial_program.cpp and runs it on the 30-item benchmark,
  reporting accuracy + tok/s + speedup for the `target` and `mtp-spec` modes.
  Flags: `--modes target,mtp-spec`, `--gpus <a,b>`, `--limit N`, `--n N`. GPU assignment is via
  `--gpus` (task-eval sets CUDA_VISIBLE_DEVICES per run).
- Always run evals on the slot's 2-GPU pair, in this command form:
  `./evaluator/task-eval --gpus <a,b>`
- For example, slot 0: `./evaluator/task-eval --gpus 0,1`; slot 2: `./evaluator/task-eval --gpus 4,5`
- The evaluation timeout is 7200 seconds
