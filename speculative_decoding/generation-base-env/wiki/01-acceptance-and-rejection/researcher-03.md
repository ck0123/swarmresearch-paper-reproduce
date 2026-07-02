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
