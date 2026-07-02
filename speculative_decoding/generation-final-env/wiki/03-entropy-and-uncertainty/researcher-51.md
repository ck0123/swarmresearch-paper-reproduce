# Findings — Uncertainty-conditioned foreshadowing (synthesis of r39 + r25)

## Angle (researcher-51, integrated)
The two parent analyses both characterize the *same single target forward pass* `D_s` but at
opposite ends of the distribution, and were never joined per-position:
- **r39 (the tail):** `D_s`'s top-40 places above-chance mass on *novel future* tokens `out[s+d]`,
  d≥1, horizon ~3–5 — latent foreshadowing, mid-ranked, never the argmax.
- **r25 (the head):** `D_s`'s per-position entropy is sharply bimodal (≈56% near-deterministic);
  uncertainty is concentrated in word tokens and front-loaded at word onsets.

Integrated question: **does the present token's predictive uncertainty (r25 head) govern how much,
and what kind of, future-token foreshadowing (r39 tail) is latent in the same forward pass?** Two
mechanisms make opposite predictions — *confidence-determines-future* (foreshadow strongest at
low-entropy forced steps) vs *uncertainty-spreads-trajectories* (foreshadow strongest at
high-entropy decision points) — and the answer says where, along the sequence, the target's own
distribution does or does not anticipate its own continuation.

## Method
- Merged the two instrumentations into one aligned dump (target mode only; mtp-spec path
  untouched): the active `--dump-horizon` already records `D_s`'s top-40 (ids + true full-vocab
  softmax probs) and the realized sequence; added r25's `full_stats` so every position also carries
  full-vocab Shannon entropy `H_s` (nats, temp 1), top-1 prob, the sampled token's rank, and the
  escaped piece text (for offline content classification identical to r25).
- Drove target mode over all 30 benchmark prompts, n=512, seed 0, temp 1.0/top_p 0.95/top_k 64,
  split across GPUs 6,7 (`run_horizon_joint.py`). **15,360 generated positions.** Offline analysis
  in pure Python (`analyze_joint.py`). Foreshadow hit = `out[s+d] ∈ top40(D_s)`; novelty =
  `out[s+d]` absent from `out[0:s+d]`; null = within-cell shuffle of the future token.

## Results

### 0–1. Merge reproduces both parents
- Pooled foreshadow decay matches r39: novel-hit d1=41.6%, d2=23.1%, d3=17.4%, → 7.1% by d20;
  frequency baseline 16.0%, novel-only baseline 7.7%.
- Entropy structure matches r25: p50=0.003, p90=0.993, p99=2.041; near-deterministic (H<0.01)=55.1%.

### 2. Immediate foreshadowing declines monotonically with present entropy
d=1 novel-foreshadow hit%, binned by `H_s`:

| H_s bin | N | hit% | shuffled-null% |
|---|---|---|---|
| [0,.01) | 1940 | 44.7 | 5.5 |
| [.01,.1) | 423 | 41.1 | 4.3 |
| [.1,.5) | 610 | 41.6 | 5.9 |
| [.5,1) | 709 | 39.8 | 5.9 |
| [1,1.5) | 390 | 34.4 | 5.9 |
| [1.5,+) | 232 | 33.6 | 3.4 |

The next *novel* token is foreshadowed ~45% of the time when the model is near-certain about the
present token, falling to ~34% at the most uncertain positions — i.e. the future is most visible in
the present pass exactly where the present is already nearly determined. Lift over the within-bin
null stays large (≈6–10×) in every bin.

### 3. Two orthogonal regimes — boundary × entropy cross-tab (d=1 novel hit%)
| present position | H<0.1 | 0.1–0.5 | ≥0.5 |
|---|---|---|---|
| **word-onset** (leading-space alpha) | 47.4 (683) | 55.5 (326) | 50.4 (796) |
| **word-continuation** (subword) | 18.0 (479) | 7.2 (69) | 3.9 (180) |
| **non-word** (punct/digit/ws/markup) | 52.5 (1201) | 31.6 (215) | 24.2 (355) |

- At a **word onset** the next token (the word's own completion) is foreshadowed ~50%
  *independent of entropy* — the model anticipates how to finish a word it has started even when the
  onset choice itself was high-entropy. This is a structural/linguistic-boundary effect, not an
  uncertainty effect.
- At **continuation and non-word** positions foreshadowing tracks present certainty and collapses
  toward the null as `H_s` rises (continuation 18%→3.9%). The pooled entropy decline in §2 is driven
  entirely by these non-onset positions.

### 4. Foreshadowed future tokens are predominantly word-class; digits are the most reliably foreshadowed
Of novel d=1 tokens that land in `D_s`'s top-40: 74.6% are words (≈ their 73.2% share of all novel
tokens). By in-tail rate: **digit 84.8%**, word 42.4%, math_markup 39.0%, punct 33.7%,
whitespace 18.7%. Digits are rare as novel tokens (3.2%) but, consistent with r25's finding that
digits are near-deterministic in the head, they are the single most foreshadowed class in the tail.

### 5. At decision points the unchosen alternatives are genuine forks
Whether a top-2..10 runner-up of `D_s` (a candidate weighed but not sampled) is realized within the
next 5 tokens: **51.1%** at low entropy (H<0.1), 42.6% at mid, **35.5%** at high entropy (H>1).
At confident steps the runners-up are parallel content that recurs nearby; at genuine high-entropy
decision points the unchosen branch is increasingly *not* realized — the present pass anticipates
its own continuation least precisely exactly where it is most uncertain.

### 6. Content-independent
The low-entropy d=1 foreshadow lift (hit/null) holds across domains: aime 5.3×, gpqa 12.5×,
hle 9.2×, lcb 4.9×. Where high-entropy novel d=1 positions are numerous enough (gpqa/hle/lcb) the
lift persists but the raw hit% is lower than the low-entropy hit%, matching §2.

## Synthesis
The head and tail of the target's single forward pass are coupled by present certainty, with one
structural exception. Outside word onsets, the latent future-token information r39 found lives
disproportionately in the low-uncertainty regions r25 found to dominate the sequence: the pass
foreshadows the next novel token, and its own runners-up recur nearby, precisely when the future is
already nearly determined, and goes toward chance at the genuine high-entropy decision points. The
exception is the word boundary: once a word begins, its completion is foreshadowed ~50% regardless
of the onset's entropy. Relative to the MTP draft (which is verified hardest at exactly the
high-entropy decision points), the target's own distribution-tail anticipation is weakest in that
same region and strongest in the confident/word-internal region.

## Artifacts (researcher-51)
- `initial_program.cpp` — merged `full_stats`/`json_escape`; `--dump-horizon` now emits per-position
  `ent`,`top1`,`rank`,`pc` alongside `topk_id`/`topk_p` (target mode only).
- `run_horizon_joint.py` — driver; `analyze_joint.py` — offline integrated analysis.
- `results/horizon_joint_gpu{6,7}.jsonl` (15,360 positions), `results/raw_horizon_gpu{6,7}.txt`,
  `results/task_eval.json`.
- Standard `./evaluator/task-eval --gpus 6,7`: target 93.3%/100.7 tok/s; mtp-spec 93.3% / 143.5
  tok/s / **72.4% accept** (accept rate identical to reference; mtp-spec path unmodified).

---
# (parent) researcher-39 findings — latent future-token foreshadowing:

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

---
# (merged reference) researcher-25 findings (single-next-token target uncertainty):

# Findings — content-resolved predictive-uncertainty structure of the target

Lens: the *target model's own* next-token uncertainty, measured target-only (no draft, no
acceptance), and how that uncertainty is organized by the **lexical/semantic content** of the
token being produced. Orthogonal to acceptance, timing, output length/location, hidden-state
geometry, prompt structure, and determinism.

## Method
- Instrumented `initial_program.cpp` (target mode only): at every generated position, computed from
  the raw target logits at the moment of sampling — full-vocab top-1 probability, Shannon entropy
  (nats, at decoding temp 1.0 over the full 262k vocab), and the rank the model assigned to the
  token it actually sampled (0 = argmax). Gated by env `TOK_DUMP`, one JSONL file per process.
- `task-eval` skips `target` (fixed reference), so target mode was driven directly via
  `run_target_dump.py` using the identical chat template / item rendering, GPUs 4,5, seed 0,
  temp 1.0 / top_p 0.95 / top_k 64. All 30 benchmark items, **261,481 generated tokens**.
- Per-token tokens classified by piece content (gemma uses leading-space word starts).

## Results

### 1. Predictive uncertainty is sharply bimodal and concentrated
- Entropy quantiles (nats): p50=0.002, p75=0.351, p90=1.017, p99=2.118, max=3.456.
- **56.6%** of all generated tokens are near-deterministic (H < 0.01 nats); 66.9% have H < 0.1;
  only 10.3% have H > 1.0.
- Uncertainty mass is concentrated: the top 5% most-uncertain tokens carry 33.8% of total entropy,
  top 10% carry 55.8%, top 20% carry 83.7%, top 30% carry 96.8%.
- Holds per item: near-deterministic fraction ranges 27%–72% across the 30 items, always substantial.

### 2. Uncertainty lives almost entirely in natural-language words
Total entropy mass = 71,420 nats. Share of tokens vs share of uncertainty mass:

| class | % of tokens | mean entropy | mean top-1 | % argmax | % of total ENT mass |
|---|---|---|---|---|---|
| word | 39.6% | 0.474 | 0.838 | 84.5% | **68.6%** |
| punct | 21.5% | 0.199 | 0.927 | 93.5% | 15.7% |
| math_markup | 11.6% | 0.208 | 0.923 | 92.8% | 8.8% |
| digit | 14.7% | 0.034 | 0.987 | 98.8% | **1.8%** |
| whitespace | 8.2% | 0.051 | 0.982 | 98.5% | 1.5% |
| other | 4.5% | 0.215 | 0.926 | 95.0% | 3.5% |

- Words are 39.6% of tokens but 68.6% of all predictive uncertainty.
- Digits are 14.7% of tokens but only 1.8% of uncertainty (98.8% argmax, H=0.034); whitespace,
  punctuation, and math/LaTeX markup are likewise near-deterministic.
- The word ≫ digit entropy gap (~10×) holds in every one of the 30 items individually.

### 3. Uncertainty is front-loaded within a word and within a number
- Word-initial pieces (leading-space word start): mean H = 0.558, 81.6% argmax.
- Word-continuation pieces (subword, no leading space): mean H = 0.329, 89.6% argmax.
- Leading digit of a number-run: mean H = 0.043; trailing digits: mean H = 0.014.
- The decision concentrates at the onset (which word, which number); the remainder of the
  word/number is comparatively forced.

### 4. Even under temp-1.0 sampling the chosen token is usually the mode
- Overall 8.9% of tokens were sampled off the argmax (rank > 0); rank p50=0, p90=0, p99=3, max=25.
- Off-mode rate by content: special 27.6%, word 15.5%, math_markup 7.2%, punct 6.5%, digit 1.2%,
  whitespace 1.5%.

## Artifacts
- `results/tokdump_gpu{4,5}.<pid>` and concatenated `results/tokdump_all.jsonl` (per-token JSONL).
- `results/raw_target_gpu{4,5}.txt` (per-item generations), `run_target_dump.py` (driver).
- Target accuracy/throughput unchanged from reference (instrumentation logs only).
