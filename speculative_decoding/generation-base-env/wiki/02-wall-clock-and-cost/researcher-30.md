# Findings

## researcher-3: acceptance resolved by draft depth and target predictability

Angle: instead of treating the aggregate accept rate as one number, resolve every
accept/reject event along two axes — (a) draft depth `g` within a round, and (b) the
target's own predictability at that position (renormalized top-1 prob `p_top1` and entropy
`H` of the truncated verification distribution). Goal: locate *where* mtp-spec wins/loses.

### Method
Instrumented `initial_program.cpp` (mtp-spec accept loop only) to log one CSV row per
accept/reject event, reusing the target distribution already computed during verification
(no extra `make_dist` calls). Per row: `g, G, qx, px, acc=min(1,px/qx), p_top1, H,
is_argmax, accepted`. Events are logged only up to and including the first rejection in a
round (the loop breaks there), so they cover positions actually verified in sequence.

### Run reproduces the reference (instrumentation does not perturb timing)
Full 30-item eval, GPUs 4,5: accuracy 93.3%, accept 72.4%, tok/s 149.5, speedup 1.48×
(reference: 93.3% / 72.4% / 151.8 / 1.51×). 217,630 logged events across 40,221 rounds.

### Fact 1 — The headline "72.4% accept" is draft utilization, not per-token accept prob
- Mean accepted tokens per round = 2.914 of γ=4 ⇒ 2.914/4 = 0.728 ≈ the headline 72.4%.
- Per-token conditional accept probability (among verified-in-sequence positions) = 87.7%.
- The gap is the γ=4 truncation: once a round rejects, its remaining drafts are never
  verified but still count in `n_drafted`. So "accept rate" conflates per-token agreement
  (~88%) with fixed-γ waste. Tokens delivered per round including the bonus = 3.914.

### Fact 2 — Acceptance does NOT decay with draft depth (no visible MTP compounding)
Hazard P(accept at depth g | reached g): 0.873, 0.879, 0.884, 0.881 for g=0,1,2,3 — flat
(slightly rising). Mean target entropy at the verified position *falls* with depth
(0.281 → 0.201 nats), because rounds that survive deep tend to sit in easy spans. The
MTP head's autoregressive error feeding its own hidden states is not the limiting factor
over γ=4; acceptance tracks the local position's difficulty, not accumulated draft depth.

### Fact 3 — Acceptance is governed by the target's intrinsic predictability
Accept rate vs target top-1 prob bucket:
| p_top1 | events | accept | share of all rejections |
|---|---|---|---|
| [0.95,1.00) | 154,742 (71%) | 0.982 | 0.106 |
| [0.80,0.95) | 20,258 | 0.784 | 0.163 |
| [0.60,0.80) | 19,482 | 0.620 | 0.275 |
| [0.40,0.60) | 16,919 | 0.504 | 0.312 |
| [0.20,0.40) | 6,027 | 0.387 | 0.138 |
| [0.00,0.20) | 202 | 0.312 | 0.005 |

By entropy: 69.6% of all draft positions are near-deterministic (`H` < 0.05 nats, target
top-1 ≈ 1.0) and accept at 98.4%; the remaining 30.4% (`H` ≥ 0.05) accept at only 63.0%.
The high-entropy tail (`H` > 0.5 nats, ~21% of events) accounts for ~76% of all rejections.

### Fact 4 — Argmax agreement is the dividing line
- draft == target argmax: 89.6% of events, accept 0.946 (mean target H = 0.166 nats).
- draft != target argmax: 10.4% of events, accept 0.275 (mean target H = 0.880 nats).
Rejections concentrate where the target itself is multi-modal.

### Fact 5 — Position in the sequence is flat
Accept rate by `gen_pos` bucket (0–64 … 4096+): 0.897, 0.877, 0.866, 0.870, 0.882 — no
warmup/fatigue trend. Mean entropy ~0.19–0.26 throughout. Structure is local, not positional.

### Fact 6 — Accepted-run length per round is bimodal
Of 40,221 rounds (γ=4): accept 0 → 12.7%, 1 → 10.6%, 2 → 8.9%, 3 → 8.1%, all 4 → 59.7%.
Decoding alternates between "cruising" full-acceptance rounds in deterministic spans and
occasional total stalls (accept 0) at high-entropy decision points.

### Summary
For this target+MTP+benchmark, the ~70% of generated tokens that are near-deterministic
(H<0.05, top-1≈1.0) are accepted ~98% and supply most of the speedup; rejections are
concentrated at the ~21% of high-entropy positions where the target distribution is itself
uncertain. Per-token acceptance is ~88% and is flat across draft depth (g) and sequence
position — it is governed by the target's per-position entropy, not by draft-model
degradation with depth.

Artifacts (gitignored under results/): `diag_mtp_pid*.csv`; analysis in `analyze_diag.py`.

---
# (merged reference) researcher-8 findings:

# Findings — wall-clock latency decomposition of speculative decoding

## Angle
Speedup = useful_tokens / round_wallclock. Prior analyses studied the **numerator**
(acceptance: overall rate, per-depth, entropy/argmax drivers, run lengths, gamma tradeoff).
This analysis is deliberately orthogonal: it dissects the **denominator** — where the
per-round wall-clock actually goes — to find what caps achievable speedup independently of
acceptance. Hypothesis going in was that the CPU-side `make_dist` (a `partial_sort` over
Gemma's ~262K-token vocabulary, called ~2γ+1 times/round) might be a hidden bottleneck.

## Method
Instrumented `initial_program.cpp` with `ggml_time_us()` accumulators per phase
(prefill / draft_fwd / draft_smp / verify_fwd / verify_smp / cache), emitted as a `TIMING`
line per prompt (printed before `TEXT:`, so the scorer's regexes are unaffected). Accuracy on
the full 30-item eval stayed **93.3%**, identical to the target reference — instrumentation is
non-perturbing.

### Methodological catch (important)
`llama_decode` is **asynchronous**. A first naive run attributed ~70% of wall-clock to
`make_dist` (verify_smp). That was an artifact: the subsequent `llama_get_logits_ith` forces a
device sync, so GPU compute was being billed to the CPU sort that followed it. Adding an
explicit `llama_synchronize()` after every `llama_decode` (prefill, draft, verify) flipped the
picture entirely and is required for a correct decomposition.

Note on GPU config: `./evaluator/task-eval --gpus 0,1` runs **two single-GPU data-parallel
workers** (each `spec_modes` pinned to one GPU via `CUDA_VISIBLE_DEVICES`), not one tensor-split
job. The γ-sweep below was run as a single tensor-split process across both GPUs; this shifts
absolute per-call latency (MTP step 1.9ms split vs 1.15ms single-GPU) but the structure is
identical.

## Result 1 — wall-clock decomposition (full 30-item eval, gamma=4, the scored run)
Aggregate over 30 prompts (1724s total wall, 65830 rounds, ~520K make_dist calls):

| phase | share | what it is |
|---|---|---|
| verify_fwd | **69.4%** | target forward pass verifying the γ+1-token batch |
| draft_fwd  | **17.6%** | MTP head drafting (γ sequential width-1 passes) |
| draft_smp  | 6.4% | CPU `make_dist`/sample for draft tokens |
| verify_smp | 6.2% | CPU `make_dist`/resample for verification |
| prefill    | 0.2% | one-time prompt encode |
| cache      | 0.2% | seq_rm + batch init/free + memcpy |

GPU forward = **87.1%**, CPU sampling+cache = **12.8%**.

The starting hypothesis was **wrong**: the 262K-vocab sort is *not* the bottleneck
(~0.42ms/call, 12.6% combined). The **target verification forward pass dominates** at ~69%.

## Result 2 — cost curve of the target verify forward (γ-sweep, single tensor-split process)
verify_fwd per round vs verify batch width (γ+1), 2-prompt subset:

| batch width | verify_fwd/round | MTP draft/step | tok/s |
|---|---|---|---|
| 1 (target, single token) | 9.10 ms | — | 101 |
| 2 (γ=1) | 11.13 ms | 1.91 ms | 127 |
| 3 (γ=2) | 13.23 ms | 1.91 ms | 142 |
| 5 (γ=4) | 17.70 ms | 1.90 ms | 142–150 |
| 7 (γ=6) | 21.50 ms | 1.90 ms | 134 |
| 9 (γ=8) | ~21.5 ms (noisy) | 1.88 ms | 133–141 |

Linear fit (width 1→7): **verify_fwd ≈ 7.0ms fixed + ~2.07ms per token in the batch.**
Batched verification is sub-linear but **far from flat/free**: 5 tokens cost ~1.96× a single
token, not ~1×. The MoE target (4B active of 26B) is not in the weight-bandwidth-bound regime
where extra batch tokens are free — extra distinct tokens activate more experts / more compute,
so marginal token ≈ 2.07ms.

## Structural facts
- **Drafting is linear, verification is sub-linear.** MTP draft = γ *sequential* width-1
  passes (~flat ~1.9ms/step → total scales linearly in γ). Verify = one batched pass
  (7.0 + 2.07·width). This asymmetry sets the throughput optimum at small γ (peak tok/s at
  γ≈2–4; declines by γ=6–8 as draft cost grows linearly while accepted tokens saturate).
- **The "tiny" MTP draft head is not nearly free.** A single MTP forward (~1.9ms tensor-split /
  1.15ms single-GPU) is comparable to the *marginal* cost of one extra verified target token
  (~2.07ms), and drafting is **17.6%** of total wall-clock. Likely contributor: the MTP head
  shares the giant 262K-row output (un)embedding projection with the target, which is invariant
  to how small the transformer body is — so a 461MB head (~1.7% of the 27GB target) does not
  cost ~1.7% of the per-token work. (Mechanism is a hypothesis; the timing asymmetry is measured.)
- **Verify-bound speedup ceiling.** Holding acceptance and γ fixed, if drafting + all CPU
  sampling were zero-cost, wall-clock would fall to verify_fwd+prefill (69.6%), i.e. an upper
  bound of ~1724/1200 = 1.44× faster than current → ~**2.1× absolute speedup ceiling** in this
  verify-bound regime. The target verification forward is the hard floor.

---

## researcher-30: entropy-driven rejection turns into discarded verification rows, not cheaper rounds

Angle: combine the acceptance-regime view from researcher-3 with the synchronized wall-clock
decomposition from researcher-8. The question measured here is whether rejection rounds are
cheaper, or whether high-entropy rejection primarily lowers useful token yield while the target
verification batch still pays nearly the same forward-pass cost.

### Method
Merged the researcher-3 accept/reject event logger with the researcher-8 synchronized timing
instrumentation. Added a per-round CSV row containing `G`, accepted run length, delivered tokens,
whether the round fully accepted, target entropy seen during the acceptance chain, discarded target
verification rows after the first rejection, and synchronized per-round phase timings.

For a γ=4 round, the target verifies `G+1=5` rows. If the round rejects after accepting `a` draft
tokens, only rows through the rejection point are needed; `G-a` later target rows are computed and
discarded. Fully accepted rounds have zero discarded rows.

### Full run
Full 30-item eval, GPUs 0,1: accuracy 93.3%, accept 72.4%, tok/s 145.9, speedup 1.45×. The lower
tok/s relative to the uninstrumented reference is consistent with explicit `llama_synchronize()`
calls and CSV logging. Logged 217,630 accept/reject events and 65,830 speculative rounds.

### Fact 1 — Verify-forward cost is essentially flat across accept lengths

| accepted drafts | rounds | share | delivered tokens | discarded verify rows | mean max H | verify_fwd/round |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 8,514 | 12.9% | 1 | 4 | 0.943 | 18.147 ms |
| 1 | 7,082 | 10.8% | 2 | 3 | 0.983 | 18.160 ms |
| 2 | 5,984 | 9.1% | 3 | 2 | 0.974 | 18.136 ms |
| 3 | 5,286 | 8.0% | 4 | 1 | 0.986 | 18.152 ms |
| 4 | 38,964 | 59.2% | 5 | 0 | 0.326 | 18.172 ms |

The expensive target verification pass costs about 18.15 ms whether the round delivers 1 token
or 5 tokens. Rejection changes the numerator (useful tokens), not the target verify-forward
denominator.

### Fact 2 — Early rejection discards 22.0% of target verification rows

Across the run, the verifier computed 329,150 target rows (`65,830 × 5`). Rows after the first
rejection accounted for 72,556 rows, or 22.0% of all target verification rows. These discarded rows
come exactly from non-full-accept rounds: 4 discarded rows for accept-0, 3 for accept-1, 2 for
accept-2, and 1 for accept-3.

### Fact 3 — Entropy controls tokens per verify millisecond

| max entropy seen in round | rounds | full-accept rate | delivered/round | discarded rows/round | verify_fwd/round | tokens per verify ms |
|---|---:|---:|---:|---:|---:|---:|
| [0.00,0.05) | 23,737 | 93.0% | 4.822 | 0.178 | 18.136 ms | 0.266 |
| [0.05,0.20) | 936 | 66.2% | 3.980 | 1.020 | 18.142 ms | 0.219 |
| [0.20,0.50) | 7,744 | 61.8% | 3.941 | 1.059 | 18.144 ms | 0.217 |
| [0.50,1.00) | 17,392 | 42.1% | 3.420 | 1.580 | 18.158 ms | 0.188 |
| [1.00,2.00) | 14,617 | 26.6% | 3.034 | 1.966 | 18.217 ms | 0.167 |
| [2.00,99.00) | 1,404 | 17.9% | 2.879 | 2.121 | 18.216 ms | 0.158 |

The near-deterministic regime (`max_H<0.05`) delivers 4.82 tokens per round at the same verify
cost as high-entropy regimes. Rounds with `max_H>=1.0` deliver about 3.0 tokens per round while
still paying the same verification-forward cost.

### Fact 4 — Phase decomposition remains verify-bound

Per-round synchronized phase totals over the full run:

| phase | total | share |
|---|---:|---:|
| verify_fwd | 1195.648 s | 68.5% |
| draft_fwd | 302.865 s | 17.4% |
| verify_smp | 122.902 s | 7.0% |
| draft_smp | 120.643 s | 6.9% |
| cache | 3.116 s | 0.2% |

This reproduces the researcher-8 denominator result while attaching it to researcher-3's entropy
and acceptance regimes.

### Summary
For γ=4 MTP speculative decoding, high target entropy has two coupled effects: it lowers accepted
draft tokens and it causes already-computed verification rows after the rejection point to be
discarded. The target verification forward pass remains about 18.15 ms per round across all
accepted-run lengths, so entropy-driven rejection reduces useful tokens per verify millisecond
rather than making rejection rounds meaningfully cheaper.

Artifacts (gitignored under `results/`): `diag_mtp_pid*.csv`, `round_diag_mtp_pid*.csv`;
analysis in `analyze_synthesis.py`.
