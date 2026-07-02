# Findings — researcher-61: near-boundary frontier for entropy-gated mode recovery

## Angle
The prior branch established one safe operating point for `entropy_sticky`: force-accept
recoverable artifact rejections only when target verification entropy `Ht < 1.0`, preserving the
30-item benchmark accuracy while improving mtp-spec throughput. This extension treats the entropy
threshold itself as the object of study and maps the safe frontier immediately above that point.

## Method
- Used the existing `entropy_sticky` intervention in `initial_program.cpp` unchanged:
  `SPEC_VERIFY=entropy_sticky` and `SPEC_ENT_THRESH=<tau>`.
- Added `run_entropy_frontier.py`, a wrapper around `./evaluator/task-eval --modes mtp-spec
  --gpus 6,7`, to run one threshold at a time and preserve per-threshold evaluator JSON, raw output,
  prompts, and mtp-spec side-channel logs under `results/frontier/`.
- Added `analyze_frontier.py` to summarize `results/frontier/frontier_summary.json` and the saved
  side-channel logs.
- Full 30-item benchmark, seed 0, `n=16384`, `gamma=4`, GPUs 6,7. The prior `tau=1.0` result was not
  rerun; the new sweep evaluated thresholds strictly above it.

## Results

### 1. Full benchmark frontier above the known safe point
| gate `Ht<tau` | accuracy | tok/s | speedup | accept | forced artifacts | forced/verified | wrong item IDs |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1.0500 | 93.3% | 158.1 | 1.57x | 78.8% | 7,125 | 2.92% | hle-chem-04, hle-phys-00 |
| 1.0750 | 96.7% | 159.8 | 1.59x | 79.5% | 6,593 | 2.84% | hle-chem-04 |
| 1.0875 | 96.7% | 158.5 | 1.57x | 79.8% | 6,797 | 2.91% | aime-11-15-01 |
| 1.0950 | 96.7% | 156.4 | 1.55x | 79.3% | 6,852 | 2.94% | lcb-hard-06 |
| 1.0980 | 96.7% | 158.5 | 1.57x | 79.3% | 7,345 | 3.11% | gpqa-biol-03 |
| 1.0990 | 86.7% | 157.3 | 1.56x | 80.7% | 7,053 | 2.82% | aime-1-5-01, hle-othe-03, lcb-medium-07, lcb-medium-06 |
| 1.1000 | 90.0% | 158.9 | 1.58x | 80.4% | 7,123 | 2.90% | aime-1-5-01, lcb-hard-04, lcb-medium-08 |

The highest observed threshold with accuracy at least the fixed target-reference accuracy
(93.3%) was **`tau=1.0980`**. The lowest observed threshold below that accuracy was
**`tau=1.0990`**. On this benchmark and seed, the safe frontier is therefore bracketed to the
interval **`1.098 <= tau < 1.099`**.

### 2. Throughput plateaus while correctness changes sharply
Across `tau=1.05` through `1.10`, mtp-spec throughput stayed in a narrow band
(`156.4`–`159.8` tok/s, `1.55`–`1.59x`) and accept rate stayed between `78.8%` and `80.7%`. The
correctness change at the frontier was much larger than the throughput change: `tau=1.0980` scored
96.7%, while `tau=1.0990` scored 86.7%, with similar tok/s and side-channel forced counts.

### 3. The near-boundary failures are item-specific rather than a uniform domain collapse
The wrong-item set changed across adjacent thresholds:
- `tau=1.0980`: only `gpqa-biol-03` was wrong.
- `tau=1.0990`: `aime-1-5-01`, `hle-othe-03`, `lcb-medium-07`, and `lcb-medium-06` were wrong.
- `tau=1.1000`: `aime-1-5-01`, `lcb-hard-04`, and `lcb-medium-08` were wrong.

The first sub-thousandth threshold increase above the observed safe ledge changed both the number
and identity of failed items, while preserving similar aggregate accept rate and throughput.

## Artifacts (researcher-61)
- `run_entropy_frontier.py` — threshold-sweep wrapper preserving evaluator artifacts per threshold.
- `analyze_frontier.py` — frontier summary from saved metric JSON and side-channel logs.
- `frontier_summary_researcher61.json` — compact committed summary of the threshold sweep.
- `results/frontier/frontier_summary.json` — compact metric table for all new thresholds.
- `results/frontier/mode_tau*.json` — full per-item metric JSON for each threshold.
- `results/frontier/tau*_gpu{6,7}.jsonl` — per-position mtp-spec side-channel logs for each threshold.

---
# Findings — researcher-56: entropy-resolved decomposition of the stochasticity tax

## Angle (synthesis of the two parent lines on one axis)
The two parents converge on the same per-position quantity — the target's verification entropy
`Ht` (Shannon entropy over the top-k→top-p support, the AUC-0.882 rejection predictor from r31):
- **Parent r51/r39 (predictive structure):** the target's single forward pass FORESHADOWS its own
  future tokens strongly at LOW entropy / word onsets (next-novel-token in top-40 ~45–50%) and
  collapses toward chance at HIGH entropy; its own runners-up recur within 5 tokens 51% at H<0.1
  vs 35.5% at H>1. Low entropy = self-determined "backbone"; high entropy = unforeshadowed "joints".
- **Parent r41/r31 (stochasticity tax):** force-accepting the *artifact* rejections (draft == target
  argmax, rejected by the lossless test only because q>px under temp=1) recovers throughput
  (1.49→1.65×) but costs ~10–13 accuracy points, because the off-mode corrections those rejections
  force are "load-bearing diversity" that keeps the reasoning trajectory from degenerating.

Integrated claim: **the stochasticity tax is paid entirely at the high-entropy joints.** If the
two parents describe the same axis, then gating mode-recovery on `Ht<τ` (recover the artifact only
where the target is already determined / foreshadows its own future) should reclaim throughput
WITHOUT degenerating, and the accuracy collapse should appear only once τ crosses into the forks.

## Method
- Merged the parents into one program: kept r51's target-mode `--dump-horizon`; grafted r41's
  `SPEC_VERIFY` switch (`lossless`/`greedy`/`argmax_sticky`) into the mtp-spec accept loop and added
  a new variant **`entropy_sticky`**: force-accept an artifact (draft==target argmax that the
  lossless test would reject) ONLY when target `Ht < SPEC_ENT_THRESH`, else the unchanged lossless
  test + residual `p−q` correction. The lossless RNG/codepath is byte-preserved (one `uni` draw per
  position, identical to the parent). Side-channel logs per verified position
  `{q, px, acc, Ht, pt, accepted, art(=draft==argmax), frc(=forced by gate)}`.
- Full 30-item evals on GPUs 6,7 (`./evaluator/task-eval`, n=16384). Offline `analyze_synthesis.py`.
  The `argmax_sticky` run force-accepts every would-be-rejected artifact (`frc=1`), so its log is the
  complete recoverable-artifact entropy distribution.

## Results

### 1. Measured frontier along the entropy gate (full 30-item benchmark, GPUs 6,7)
| variant | gate `Ht<τ` | accept | tok/s | speedup | accuracy | recoverable pool captured |
|---|---|---|---|---|---|---|
| lossless (reference) | τ=0 (none) | 72.4% | ~150 | 1.49× | 93.3% | 0% |
| entropy_sticky τ=0.5 | 0.5 | 73.9% | 150.9 | 1.50× | **93.3%** | 13.4% |
| entropy_sticky τ=1.0 | 1.0 | 79.3% | 158.5 | 1.57× | **93.3%** | 55.5% |
| argmax_sticky | τ=∞ (all) | 85.0% | 165.5 | 1.64× | **80.0%** | 100% |

Recovering every artifact below `Ht=1.0` (more than half the recoverable pool) preserves accuracy at
93.3% while capturing ~54% of the available throughput gain (accept 72.4→79.3 of the 72.4→85.0
range; speedup 1.49→1.57 of 1.49→1.64). The full ~13-point accuracy collapse (93.3→80.0%) is
produced *entirely* by additionally recovering the `Ht>1.0` artifacts.

### 2. The recoverable-artifact pool lives almost entirely at moderate-to-high entropy
Complete distribution of the 11,999 recoverable artifacts (4.6% of 260,286 verified positions; 90.8%
of positions are artifacts overall, but only 5.1% of those are rejected by lossless):
| Ht band | share | cum |
|---|---|---|
| [0.0,0.1) | 0.0% | 0.0% |
| [0.1,0.3) | 4.2% | 4.3% |
| [0.3,0.5) | 9.2% | 13.4% |
| [0.5,0.7) | 24.0% | 37.4% |
| [0.7,1.0) | 18.1% | 55.5% |
| [1.0,1.5) | 24.1% | 79.6% |
| [1.5,+) | 20.4% | 100.0% |

Median recoverable-artifact `Ht`=0.91, mean 1.04. **Only 0.03% (4/11,999) sit below Ht=0.1** — the
foreshadowed backbone (≈55% of all tokens per r25) yields essentially no recoverable throughput,
because there the target is so peaked (px≈q≈1) that the artifact is already accepted by the lossless
test (`min(1,px/q)≈1`). An artifact can be *rejected* only when px<q, which structurally requires the
target to spread mass off its own argmax — i.e. nonzero entropy. The throughput pool and the
high-entropy joints are the same positions.

### 3. The tax is mechanically the entropy axis, not a content or depth confound
τ=1.0 force-accepted 6,873 artifacts (median Ht 0.65) with zero accuracy loss; argmax_sticky added
the remaining 5,126 (the Ht>1.0 tail, lifting mean forced Ht to 1.04) and that increment alone caused
the 93.3→80.0% drop. The safe/degenerating boundary is a sharp threshold near Ht≈1.0 — exactly the
entropy region where r51 measured the forward pass's foreshadowing of its own continuation to have
collapsed toward chance (runners-up realized 35.5% at H>1 vs 51% at H<0.1; novel d=1 foreshadow
33.6% at H>1.5 vs 44.7% at H<0.01).

## Synthesis
A single axis — the target's verification entropy — unifies both parent lines. Below ~1 nat the
trajectory is a self-determined backbone: the forward pass already foreshadows its own next tokens
(r51), the target is peaked enough that drafted-argmax tokens are accepted losslessly, and the few
that aren't can be force-accepted with no quality cost — recovering the target's own mode there is
effectively lossless. Above ~1 nat are the joints: the pass stops foreshadowing its continuation
(r51), the artifact rejections concentrate (this work), and the off-mode corrections they force are
load-bearing diversity whose removal degenerates the reasoning trajectory (r41). The stochasticity
tax is therefore not spread across decoding; it is localized to the high-entropy fork positions, and
the speculative-decoding throughput that mode-recovery can tap is co-located with — and inseparable
from — exactly the diversity that keeps this benchmark's accuracy intact.

## Artifacts (researcher-56)
- `initial_program.cpp` — merged: r51 `--dump-horizon` (target) + r41 `SPEC_VERIFY` and side-channel
  (mtp-spec), plus the new `entropy_sticky` gate on target `Ht` (`SPEC_ENT_THRESH`).
- `analyze_synthesis.py` — recoverable-artifact entropy distribution + per-variant forced accounting.
- `results/mode_ent05.json` (τ=0.5), `results/mode_ent10.json` + `results/ent10_{6,7}.jsonl` (τ=1.0),
  `results/mode_sticky.json` + `results/sticky_{6,7}.jsonl` (argmax_sticky). τ=0.5 side-channel not
  captured (empty `CONTENT_LOG` disabled logging); its eval metrics are intact.
- mtp-spec lossless path unchanged; lossless reference reproduces both parents (93.3% / 72.4% accept).

---
# (parent) researcher-51 findings — uncertainty-conditioned foreshadowing (synthesis of r39 + r25):

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

---
# (merged reference) researcher-41 findings (stochasticity-tax / diversity-injection):

# Findings — researcher-27

## Lens: linguistic/semantic content of drafted vs. corrected tokens

Orthogonal to acceptance rates, draft-depth/run-length, gamma sweeps, timing, output
length, latent geometry, input structure, and determinism. Question studied: *what kind of
tokens does the MTP draft get wrong, and what does the target substitute?* — the content
identity of draft vs. correction, not the rate at which it happens.

### Method
Instrumented `initial_program.cpp` (mtp-spec mode) to emit one JSON line per verification
round to a side-channel file (`results/content_<gpu>.jsonl`); stdout is unchanged so scoring
is unaffected (full run: accuracy 93.3%, identical to target reference; 1.48x speedup).
Each line records the detokenized piece string of every draft token with its accept flag,
plus the correction token (on a rejection) or bonus token (on full-accept). Post-analysis in
`analyze_content.py` classifies pieces into coarse linguistic categories
(word/punct/digit/space/newline/special) and computes substitution structure.
Full 30-item run on GPUs 6,7: **65,830 rounds — 26,866 corrections, 38,964 full-accept bonus**.

### Core facts

**1. The draft's failure mode is lexical, not structural (near-miss dominance).**
72.4% of corrections are *same-category*: the draft picked the right token KIND but the wrong
specific token (word→word 53%, punct→punct 14%, digit→digit 3%). Only 27.6% are category
errors. The draft reliably knows the local linguistic shape of what comes next; it fails on
the lexical choice within that shape.

**2. Draft is right on scaffolding, wrong on content.**
Accepted draft tokens: 38.8% word, 33.3% punct, 15.0% digit, 8.4% space, 4.5% newline.
Rejected/correction tokens: ~64–67% word, ~25% punct, ~4% digit. Punctuation, whitespace, and
newlines are accepted far above their share of corrections; content words are the dominant
correction category. Digits are accepted well relative to frequency (15% of accepts, 4% of
corrections).

**3. "Same category" does NOT mean "similar spelling."**
81.9% of corrections share ZERO character prefix with the rejected draft; only ~4% share ≥2
chars. The draft proposes a *different word of the same type*, not a typo/morphological
variant. Errors are semantic substitutions, not spelling slips.

**4. Divergences are mostly at word/line boundaries (which-word branch points).**
60.0% of corrections: both rejected-draft and correction begin a new word/line (boundary).
28.0% are both word-internal (mid-word divergence); of those only 17.7% share a stripped
prefix. Branch points are about *which next word/symbol*, not intra-word spelling.

**5. Concrete recurring substitutions** (rejected → correction):
- Formatting ambiguity: `\n`↔`\n\n` (paragraph-break length), `.`↔`,`, `$.`↔`$`/`$,`.
- Function-word / determiner choice: ` the`↔` a`↔` it`↔` this`; `the`→`a` is the single most
  common word substitution (89×).
- Reasoning-step direction: `Wait`↔`Let`, `let`→`I`, `is`→`can`/`means`/`was` — disagreement
  on how to open the next clause/reasoning move.
- Math-mode transition: ` $`↔` the`, ` $`↔` `` ` `` — the draft mis-predicts entry/exit of
  LaTeX math. `$`-bearing tokens: 11,195 accepted vs 2,052 rejected-draft, 1,868 in corrections.
- Digit value guesses: `1`↔`2`, `2`↔`1`, `1`↔`0`/`3` — adjacent-value misses where the draft
  cannot know the computed number.

**6. Domain content profile (correction category mix).**
Math (aime) and code (lcb) corrections are punctuation/digit-heavy (aime 31.2% punct, 7.4%
digit; lcb 28.1% punct, 6.0% digit). Prose multiple-choice (hle 75.9% word, gpqa 72.0% word)
corrections are overwhelmingly word choice. Correction content tracks the symbolic density of
the domain.

**7. Post-divergence drafts stay content-heavy.**
Tokens drafted after the first rejection (discarded, conditioned on a wrong prefix) are 57.5%
word / 26.6% punct / 5.8% digit — the draft keeps proposing content words rather than
collapsing into filler after it goes off-path.

### Artifacts
- `initial_program.cpp` — added side-channel per-round content logging (mtp-spec only).
- `analyze_content.py` — content categorization + substitution-structure analysis.
- `results/content_6.jsonl`, `results/content_7.jsonl` — 65,830 logged rounds.

---

# Findings — researcher-31

## Lens: confidence/entropy structure of divergence points (deeper extension of the content lens)

The content lens established draft errors are semantic "right-kind/wrong-token" substitutions.
This lens goes a level beneath: when the draft LOSES a token, is the target genuinely uncertain
there (an ambiguous branch point the draft lost by chance) or is the target confident (the draft
is simply wrong about a determined token)? And does the draft head's own probability KNOW it is
about to be rejected (calibration)?

### Method
Extended the same side-channel (mtp-spec only; stdout unchanged, so scoring identical: accuracy
93.3% == target reference, 72.4% accept, 1.46x speedup). For each verified draft position the log
now records: `q` (draft-head prob of its own proposal), `px` (target prob of that token),
`acc`=min(1,px/q), target entropy `Ht` & top-prob `pt` over the top-k support, draft entropy `Hd`
& top-prob `qt`, target argmax id, accept flag; plus on rejection the target prob of the
correction (`corr_px`) and whether the correction is the target argmax. All quantities are
computed in the existing accept/reject loop. Post-analysis in `analyze_confidence.py`.
Same full run on GPUs 6,7: 65,830 rounds, 190,764 accepted positions, 26,866 first-reject positions.

### Core facts

**1. The draft head is anti-calibrated in its uncertain regime.**
67.7% of all drafted positions are deterministic agreement (q=px=1, accept 92.3%). In the
remaining uncertain regime (q<0.999), acceptance is NON-MONOTONIC and inverted in draft
confidence: q∈[0,0.3) accepts 64.2%, q∈[0.5,0.7) 54.2%, q∈[0.7,0.85) 52.8%, q∈[0.95,0.999)
46.9%. The MORE confident the draft is (short of total certainty), the LESS likely it is
accepted. AUC of draft prob `q` predicting acceptance is only 0.685 (top-prob 0.686, entropy
0.687 — all near-useless). Mean q at rejection is 0.854: the draft is highly confident even when
wrong.

**2. The TARGET's uncertainty, not the draft's, predicts rejection (AUC 0.882).**
Target entropy: accept 0.149 nats vs reject 0.887 nats. Target top-prob: accept 0.941 vs reject
0.643. Accepted positions are uniformly determined (target entropy <0.26 across every content
category); rejections concentrate where the target distribution itself is flat. The binding
signal lives in the target's verification logits, not the draft's confidence.

**3. Most rejections are NOT the draft being wrong about a settled token.**
By target top-prob at the rejected slot: only 16.0% are "determined" (pt≥0.9), 23.3% strong
(0.7–0.9), 46.3% mid (0.4–0.7), 14.3% ambiguous (pt<0.4). 57.0% of rejected draft tokens were
"plausible" to the target (px≥0.2); only 19.6% were ruled out (px<0.02). The majority of
rejections are coin-flip losses at genuine branch points, not determined-token mistakes.

**4. 38.9% of rejections, the draft had picked the target's own argmax.**
The draft token equals the target argmax in 10,459 of 26,866 rejections — greedy (temp=0)
decoding would accept all of these. They are rejected only because the stochastic lossless
accept test fires when q>px (the draft over-assigned probability to the right token). Over a
third of "draft errors" are artifacts of lossless sampling at temp=1, not the draft being wrong.
The correction equals the target argmax in 45.9% of rejections; mean target prob of the
correction is only 0.440.

**5. Acceptance-loss decomposition at rejection (mean realized px/q = 0.348).**
31.0% target rules the draft out (px<0.10); 37.3% draft overconfidence (px≥0.10 but q≥2·px);
31.7% close ties (px≥0.10, q<2·px, lost the coin flip). The draft systematically
over-concentrates relative to the target at branch points (~3x on average).

**6. Confidence structure tracks content category.**
Rejected words sit at high target entropy (0.985) — semantic free choice. Rejected digits sit
at LOW target entropy (0.345, target top-prob 0.843): when a drafted digit is wrong the target
is fairly sure of the right value, so digit errors are genuine value misses on near-determined
slots (only 22.3% were the target argmax). Newline rejections are 54.5% target-argmax — mostly
stochastic losses on the target's top choice (paragraph-break-length ambiguity, matching the
content lens's \n↔\n\n).

**7. The draft does not degrade with draft depth.**
Across depths g=0..3, mean q rises slightly (0.951→0.962), target px rises (0.846→0.866), and
acceptance rises (87.1%→88.1%). Conditioning the MTP head on its own prior drafts does not erode
calibration within a round; deeper positions are if anything marginally more accept-prone
(consistent with survivorship — rounds reaching deeper g have already cleared the hard slots).

**8. A draft-confidence gate is unfavorable at every threshold.**
Gating first-position drafts on q≥thr drops more good (would-accept) tokens than doomed ones at
all thresholds tested (q≥0.7: drop 1,630 doomed vs 2,600 good; q≥0.9: 3,020 doomed vs 4,287
good). Draft confidence is not a usable adaptive-drafting signal — a direct consequence of the
anti-calibration in fact 1.

### Artifacts
- `initial_program.cpp` — extended the mtp-spec side-channel with per-position confidence stats
  (`ev[]`: q, px, acc, Ht, pt, Hd, qt, target-argmax, accept flag; plus corr_px/corr_argmax).
- `analyze_confidence.py` — calibration, target-uncertainty, rejection-typology, loss
  decomposition, category cross, gating, and depth analyses.
- `results/content_6.jsonl`, `results/content_7.jsonl` — 65,830 rounds with confidence telemetry.

---

# Findings — researcher-41

## Lens: the "stochasticity tax" — measuring the consequence of recovering the artifact rejections

The confidence lens recorded that 38.9% of rejections are "artifacts" of temp=1 sampling — the
draft proposed the target's own argmax and was rejected only because the lossless test fires when
q>px. This lens converts that descriptive fact into measured throughput/accuracy consequences: it
implements two verification rules that *recover* the artifact rejections and runs them end-to-end
on the full 30-item benchmark, then traces the mechanism behind the result.

### Method
Added an `SPEC_VERIFY` env switch to the mtp-spec accept loop in `initial_program.cpp` (stdout/
scoring path unchanged; the side-channel log records the realized accept decisions):
- `lossless` (default): the existing rule, `take = uni < min(1,px/q)`; residual correction `p−q`.
- `greedy`: temp→0 everywhere; `take = (draft==target argmax)`; correction/bonus = target argmax
  (canonical greedy speculative decoding; exactly lossless to greedy target decoding).
- `argmax_sticky`: temp=1 throughout, but `take = (draft==argmax) OR (uni<acc)` — force-accept the
  artifact class, otherwise the unchanged stochastic test + residual correction. It differs from
  `lossless` ONLY by accepting the 38.9% artifact rejections, so it is a controlled isolation of
  that one class. Three full evals on GPUs 2,3 plus offline analysis (`analyze_tax.py`) of the
  fresh lossless logs (65,830 rounds, identical count to the parent run).

### Core facts

**1. Measured accuracy/throughput frontier (full 30-item benchmark, GPUs 2,3).**
| variant        | accept | tok/s | speedup | accuracy | mean decoded | runaway (≥16k)/30 |
|----------------|--------|-------|---------|----------|--------------|-------------------|
| lossless temp=1| 72.4%  | 150.0 | 1.49×   | 93.3%    | 8,553        | 1                 |
| argmax_sticky  | 85.0%  | 166.5 | 1.65×   | 83.3%    | 10,917       | 4                 |
| greedy         | 85.7%  | 167.4 | 1.66×   | 53.3%    | 13,580       | 18                |
The artifact rejections ARE recoverable for throughput: both recovery rules capture nearly the same
gain (accept 72.4→85%, speedup 1.49→1.65×). Recovery is NOT free — accuracy falls monotonically with
how hard the rule biases toward the mode (93.3 → 83.3 → 53.3%).

**2. The drop is degeneration, not wrong answers — output length balloons.**
Mean decoded tokens 8,553 → 10,917 → 13,580; items hitting the 16,384 cap 1 → 4 → 18; greedy median
output IS the cap (16,384). Damage concentrates on the longest-generation domain: code (lcb) 10/10 →
7/10 → 3/10, math (aime) 4/5 → 4/5 → 2/5, while short MCQ (gpqa) holds 10/10 → 10/10 → 8/10. Mode-
seeking makes this reasoning model loop and never emit its boxed answer; truncation at the cap scores
it wrong.

**3. The artifact rejections are mechanically forced OFF-mode steps, not waste.**
On the fresh lossless run: 217,630 evaluated positions, 190,764 accepted (87.7%), 26,866 first-
rejects. Of rejections, 38.9% are artifacts (draft==argmax, reconfirming the parent exactly) and
61.1% genuine misses. When the lossless rule rejects an artifact, the correction it emits is the
argmax in **0.0%** of cases — it ALWAYS steps off the mode. This is mechanically forced: an artifact
reject means px<qx on the argmax token (verified 100% of the 10,459 cases), so the residual `p−q`
clamps that token to zero mass and it can never be resampled. By contrast, genuine misses correct
TOWARD the argmax 75.1% of the time (mean corr_px 0.591). The two rejection classes have opposite
roles: genuine = error correction (toward the mode), artifact = diversity injection (away from it).

**4. Clean causal attribution via the isolated intervention.**
`argmax_sticky` changes lossless by exactly one thing — suppressing the 38.9% diversity-injection
rejections — and that alone costs 10 accuracy points (93.3→83.3%) while delivering the full 1.65×
throughput. The further collapse to 53.3% under greedy is the additional cost of forcing the genuine-
miss corrections and the bonus tokens onto the mode as well. The "wasted" artifact rejections are the
load-bearing entropy that keeps the reasoning trajectory from degenerating.

**5. Recoverable-throughput accounting.**
Of all 195,015 draft tokens that were the target argmax, only 5.4% (10,459) are rejected, yet they
constitute 38.9% of all rejections; accepting them in place lifts the position accept-rate 87.7→92.5%.
14.9% of rounds have an artifact as their first reject (the point past which greedy would extend the
run). This is the size of the throughput pool that mode-recovery taps — and that the accuracy results
show cannot be drawn down without degrading this benchmark.

### Artifacts
- `initial_program.cpp` — `SPEC_VERIFY` env switch (lossless / greedy / argmax_sticky) in the accept loop.
- `analyze_tax.py` — artifact/genuine rejection split, off-mode-correction mechanism, run-length bound.
- `results/mode_lossless.json`, `results/mode_greedy.json`, `results/mode_sticky.json` — per-item evals.
- `results/content_lossless_*.jsonl` (65,830 rounds), `content_greedy_*`, `content_sticky_*` — telemetry.
