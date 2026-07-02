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

## Extension — contiguous block recoverability from one target distribution

The marginal foreshadowing result is not sufficient for a decoding scheme: a useful future-token
proposal needs a contiguous ordered block, not just isolated future tokens somewhere in `D_s`'s
top-k. I re-ran target mode on GPUs 4,5 with `--dump-horizon` for all 30 prompts at n=512 and
analyzed 15,120 positions that had a full 8-token lookahead window.

### Contiguous future-prefix ceiling
For each generated position `s`, let `R_k` be the longest prefix length such that every realized
future token `out[s+1]..out[s+R_k]` is present in the same `D_s` top-k list. This is an oracle
membership ceiling: it gives credit for knowing the correct order externally and only asks whether
the required tokens are available in the one-step distribution.

| top-k | P(+1) | P(+1,+2) | P(+1..+3) | P(+1..+4) | E[extra prefix tokens] |
|---:|---:|---:|---:|---:|---:|
| 5 | 15.9% | 1.6% | 0.2% | 0.1% | 0.179 |
| 10 | 24.7% | 4.6% | 0.8% | 0.3% | 0.306 |
| 20 | 34.3% | 9.3% | 2.3% | 0.8% | 0.472 |
| 40 | 44.3% | 16.0% | 5.2% | 1.8% | 0.688 |

Top-40 has many marginal hits, but the contiguous-block ceiling collapses quickly: two future
tokens are both available only 16.0% of the time, three only 5.2%, and four only 1.8%. Even with an
oracle choosing the correct token/order from top-40, one target distribution contains only 0.688
extra contiguous future tokens on average over an 8-token horizon.

For novel-first positions (4,393 positions where `out[s+1]` was absent from prior generated
context), top-40 recovered the first novel future token 43.2% of the time, but recovered a
two-token all-novel contiguous prefix only 5.7% of the time and a three-token all-novel prefix only
0.8% of the time.

### Unordered coverage vs temporal order
Over the next 8 realized tokens, top-40 contained at least one future token in 80.9% of positions
and contained two or more future tokens in 53.9% of positions. However, the top-k rank order did not
encode temporal order: among top-40 future-hit pairs, earlier future tokens had better rank than
later future tokens only 46.3% of the time, and positions with two or more hits had fully monotone
rank order only 26.9% of the time.

Rank carried almost no offset information among top-40 hits:
`I(offset; rank bucket [1-5,6-10,11-20,21-40]) = 0.0029 bits`, versus
`H(offset among top-40 hits) = 2.9149 bits`. Conditional mean rank stayed nearly flat by offset:
d1 12.6, d2 13.4, d3 13.1, d4 13.0, d5 13.2, d6 13.8, d7 13.4, d8 13.7.

### Source split
The same top-40 contiguous-prefix pattern held across benchmark families:

| source | positions | P(+1) | P(+1,+2) | P(+1..+3) | E[extra prefix tokens] |
|---|---:|---:|---:|---:|---:|
| aime | 2520 | 51.4% | 23.0% | 8.9% | 0.915 |
| gpqa | 5040 | 42.1% | 13.1% | 4.2% | 0.627 |
| hle | 2520 | 39.1% | 12.5% | 3.6% | 0.556 |
| lcb | 5040 | 45.5% | 17.0% | 5.1% | 0.702 |

### Verification
Standard evaluator after the added offline analyzer:
`./evaluator/task-eval --gpus 4,5` → target reference 93.3% / 100.7 tok/s; mtp-spec 93.3% /
150.2 tok/s / 72.4% accept / 1.49x.

## Artifacts
- `initial_program.cpp` — added `--dump-horizon` + `topk_raw()` (target mode only).
- `analysis/foreshadow_block_analysis.py` — offline contiguous-prefix and rank-order analyzer for
  `--dump-horizon` JSONL.
- `results/horizon_target_512.jsonl` — 30 prompts × per-position top-40 of `D_p` + realized
  tokens for the block analysis.
- `results/foreshadow_block_analysis.txt`, `results/task_eval.json`, `results/raw_horizon_512.txt`
  — block-analysis output, standard eval output, and instrumented run output.
