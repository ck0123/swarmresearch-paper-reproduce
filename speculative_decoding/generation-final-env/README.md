# approx-decoding-analysis

Minimal evaluation infra for analyzing decoding methods on `gemma-4-26B-A4B-it` (Q8_0):
a 30-item benchmark the model passes reliably, a runner for two decoding modes, and a
**self-contained speculative-decoding implementation** (`initial_program.cpp`).

## Contents
```
initial_program.cpp     self-contained spec decoding (target / mtp-spec)
evaluator/
  task-eval             one command: build initial_program.cpp + run mtp-spec on benchmark + score + report
  data/ scorers/ gemma4_chat_template.jinja   the 30-item benchmark
vanilla_references/     fixed baselines (target.json) — read for the speedup column, never overwritten by runs
results/                outputs (task_eval.json, mode_mtp-spec.json, raw_*, prompts_*)
```

`evaluator/` = LiveCodeBench v6 + AIME 2026 + GPQA Diamond + HLE-MCQ (10/5/10/5), curated so the
target answers all 30 correctly on 3 independent samples. Self-contained scoring (incl. the
vendored LiveCodeBench executor); deps: `httpx`, `numpy`, `jinja2`, `pyarrow`.

## Modes (initial_program.cpp)
- **target**   — autoregressive sampling from the 26B target (baseline).
- **mtp-spec** — MTP head drafts γ tokens, target verifies, **our own rejection sampling** (lossless at temp>0).

Draft predictions use llama.cpp's native gemma-4 MTP head (`gemma4-assistant`) via the `nextn`
staging API + a shared-KV draft context; the accept/reject + sampling is implemented in
`initial_program.cpp` (not llama.cpp's default rejection sampler). Decoding: temp 1.0, top_p 0.95,
top_k 64, seed 0.

## Quick start
```bash
./evaluator/task-eval                       # build + run target & mtp-spec on all 30, report
./evaluator/task-eval --limit 4 --n 1024    # quick smoke
./evaluator/task-eval --modes mtp-spec --gpus 4,5
```
`task-eval` builds `initial_program.cpp` (recompiling if it changed) and runs it on the benchmark.
Models are referenced by path (not copied); override with `MODELS_DIR`, llama.cpp with `LLAMA_DIR`.

## Reference result (30 items, 2 GPUs/mode)
| mode | accuracy | tok/s | accept | speedup |
|---|---|---|---|---|
| target | 93.3% | 100.7 | — | 1.00× |
| **mtp-spec** | **93.3%** | 151.8 | 72.4% | **1.51×** |

`mtp-spec` is lossless (accuracy == target) at a 1.51× speedup. (temp-1.0 is stochastic, so
"lossless" = matches the target's accuracy/quality, not bit-identical sequences.)
