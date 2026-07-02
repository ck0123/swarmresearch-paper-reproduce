# Prompt and KV-cache footprint analysis

Instrumentation added to `initial_program.cpp` records rendered prompt token count, prompt byte count, newline count, MTP seed KV rows, context size, and remaining context slack under the evaluator's `-n 16384` generation budget. The evaluator preserves these fields in `results/mode_mtp-spec.json`.

`./evaluator/task-eval --gpus 2,3` completed on the 30-item benchmark with the instrumented `mtp-spec` run.

Measured rendered prompt token counts across the 30 items:

| metric | value |
| --- | ---: |
| min | 101 |
| median | 226 |
| mean | 341.63 |
| max | 1008 |

Prompt token counts by source:

| source | n | min | median | mean | max |
| --- | ---: | ---: | ---: | ---: | ---: |
| aime | 5 | 105 | 174 | 162.20 | 189 |
| gpqa | 10 | 160 | 196.5 | 223.20 | 472 |
| hle | 5 | 101 | 217 | 240.00 | 469 |
| lcb | 10 | 292 | 575.5 | 600.60 | 1008 |

In this implementation's `mtp-spec` mode, the measured initial target KV seed after prompt prefill is always `prompt_tokens - 1`, because the last prompt token is removed from the target cache before the verification loop. Across all 30 items, `kv_seed_rows` ranged from 100 to 1007.

With `ctx_size = 24576` and `max_new = 16384`, every item had non-negative context slack before any early stop. The maximum value of `prompt_tokens + max_new` was 17392, leaving a minimum recorded slack of 7184 context slots. The largest prompt was `lcb-medium-04` with 1008 prompt tokens, 2571 bytes, 98 newlines, 1007 MTP seed KV rows, and 7184 slack slots under the full generation budget.
