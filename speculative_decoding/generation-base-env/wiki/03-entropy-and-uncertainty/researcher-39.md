# Findings — Latent future-token foreshadowing in the target's single-step distribution

## Angle
A genuinely orthogonal lens (not acceptance/run-length/gamma/timing/forward-cost/output-length/
draft-geometry/prompt-structure/determinism/error-categories/single-next-token-uncertainty/
context-recoverability-copy): **how much multi-step future information is latent in a single
target forward pass.** In pure autoregressive target decoding, the next-token distribution
`D_p = softmax(logits_p)` is produced by one forward pass. We ask whether `D_p` already places
above-chance probability on tokens realized *further ahead* (`t_{p+d}`, `d>=1`), i.e. whether the
present distribution foreshadows future content — and whether that holds for *novel* tokens
(not copyable from context), which separates this lens from n-gram/copy structure.

This is the prompt-invited "relationship between target verification logits and future tokens",
generalized from the immediate next token (d=0) to a multi-step horizon.

## Method
- Instrumented `initial_program.cpp` (target mode only; mtp-spec path untouched): for each
  generated position the true-softmax (full-vocab denominator, temp=1) top-40 of `D_p` is dumped
  with the realized token sequence. New flag `--dump-horizon <path>`.
- Ran instrumented target mode on GPUs 4,5 over all 30 benchmark prompts, n=512
  (15,360 positions). Offline analysis in pure Python.
- Alignment: `D_s` is the distribution that produced `out[s]` (so d=0 is the sampled token);
  lookahead distance `d` measures `D_s`'s mass on `out[s+d]`.
- Controls: (1) **frequency baseline** = probability a *random realized token* from the same
  prompt lands in `D_s`'s top-40; (2) **novelty split** = future tokens never seen in
  `out[0:s+d]` (novel) vs already-emitted (seen), isolating genuine foreshadowing from copying.
- Standard `./evaluator/task-eval --gpus 4,5` after instrumentation: target 93.3% / 100.7 tok/s,
  mtp-spec 93.3% / 150.2 tok/s / 72.4% accept / **1.49x** (matches 1.51x reference; build/pipeline
  unaffected by instrumentation).

## Results

### Foreshadowing decay (pooled, top-40 window)
| d | hit% | novel-hit% | seen-hit% |
|---|------|-----------|-----------|
| 0 (sampled) | 100.0 | 100.0 | 100.0 |
| 1 | 44.3 | 43.1 | 44.7 |
| 2 | 29.2 | 22.7 | 31.9 |
| 3 | 22.5 | 17.9 | 24.3 |
| 4 | 21.0 | 13.1 | 24.1 |
| 5 | 18.1 | 10.8 | 20.9 |
| 8 | 17.4 | 8.5 | 20.9 |
| 10 | 16.8 | 7.4 | 20.3 |
| 15 | 16.6 | 6.8 | 20.3 |
| 20 | 17.4 | 6.7 | 21.2 |

- Frequency baseline (random realized token in `D_s` top-40): **15.1%**; novel-only baseline: **8.1%**.
- d=0 (the emitted token) has mean prob 0.872 — the model is highly peaked on what it samples.

### Key facts
- **The single forward pass foreshadows novel future tokens.** At d=1 the next *novel* token is
  in `D_s`'s top-40 43.1% of the time vs an 8.1% random-novel baseline (~5.3x lift). Because this
  is measured on tokens absent from prior context, it is not n-gram copy.
- **The foreshadow horizon is short (~3–5 tokens).** Novel-token hit decays 43.1% (d1) → 22.7%
  (d2) → 17.9% (d3) → 13.1% (d4) → 10.8% (d5), reaching the 8.1% baseline by d≈10. Seen-token hit
  stays ~20% at all d (frequency/copy effect, not foreshadowing).
- **Foreshadowed future tokens are mid-ranked, never the argmax.** Mean rank of a hit ≈ 12.
  Novel d=1 tokens land in top-1 only 0.7% of the time, top-5 14.3%, top-10 23.9%, top-40 43.1%.
  The future token is "on the radar" of the current pass but not its leading candidate.
  | dist | top1 | top5 | top10 | top40 |
  |---|---|---|---|---|
  | 1 | 0.7 | 14.3 | 23.9 | 43.1 |
  | 2 | 0.3 | 6.8 | 10.8 | 22.7 |
  | 3 | 0.2 | 5.3 | 8.8 | 17.9 |
  | 4 | 0.2 | 3.5 | 5.7 | 13.1 |
  | 5 | 0.0 | 2.9 | 4.9 | 10.8 |
- **Content-independent.** Novel d=1 foreshadow (top-40) vs random-novel baseline:
  aime 42.8% / 8.1% (5.3x), lcb 46.0% / 8.0% (5.8x), gpqa 41.2% / 7.2% (5.7x),
  hle 41.9% / 5.6% (7.5x). Effect size is uniform across math, code, and MCQ.
- Conditional probability of a hit *rises* with d (0.020 at d1 → 0.12–0.14 at d≥4): at long
  range the only future tokens still in the top-40 are high-frequency tokens that carry large
  `D_s` mass, so conditional-prob is a frequency-confounded metric; hit-rate and the
  novelty-split are the clean signals.

## Artifacts
- `initial_program.cpp` — added `--dump-horizon` + `topk_raw()` (target mode only).
- `results/horizon_target.jsonl` — 30 prompts × per-position top-40 of `D_p` + realized tokens.
- `results/task_eval.json`, `results/raw_horizon.txt` — eval + instrumented run outputs.
