# Findings — researcher-21 / extended by researcher-23

## Angle: numerical determinism / reproducibility of the shared-KV speculative pipeline

Orthogonal lens (not acceptance, timing, output structure, latent geometry, gamma, or
prompt structure): does `mtp-spec` reproduce, and is the rejection sampler's "lossless"
contract numerically exact? The contract requires the target accept probability `px` to be
read from the *true* per-position target distribution. In this implementation `px` is read
from a **batched verify decode** (G+1 tokens decoded in one `llama_decode`), whereas
`target` mode decodes **one token at a time** (batch=1). FP non-associativity makes those
two forwards numerically different.

Setup: target `gemma-4-26B-A4B-it` Q8_0 + MTP head, GPUs 0,1, seed 0, temp 1.0, top_p 0.95,
top_k 64, gamma 4.

### Finding 1 — `mtp-spec` is bit-exactly reproducible run-to-run
Two identical `mtp-spec` runs (`--limit 6 --n 1024`, GPUs 0,1, seed 0):
- Per-prompt generated TEXT byte-identical across both runs for all 6 prompts (both GPU chunks).
- `n_drafted` and `n_accept` identical per prompt (e.g. 944/791, 1056/762, 888/802 ...).
- The ONLY difference between the two raw output files is the throughput/timing line
  (`X tok/s`); 0 non-timing lines differ.
- Conclusion: GPU FP non-determinism does **not** flip any accept/reject decision here; the
  whole pipeline (draft sampling + rejection test + resample, single shared `std::mt19937`)
  is deterministic across repeats given fixed seed and fixed GPU/chunk assignment.

### Finding 2 — batched-verify logits differ substantially from single-token (AR) logits
Probe (`DIAG` env gate, inert when unset): for the first 40 verify rounds of 8 prompts
(5 AIME + 3 GPQA), at each verified position compare the target row-0 logits from the
batched verify (batch=G+1) against a fresh batch=1 decode of the same token at the same
position over the same (shared) KV prefix. 320 rounds measured.

- `frac_diff = 1.0000` in every round: **100% of the ~262k vocab logits differ** between
  batch=G+1 and batch=1. The two forwards are never bit-identical.
- `max_abs_logit_diff` (raw pre-softmax logit units): median 0.66, mean 0.82, max 7.78.
  These are large, not rounding dust — the batch dimension changes the reduction/kernel path.
- `argmax_match = 320/320 (100%)`: despite the logit shifts, the top token never flips.
  Greedy (temp<=0) decoding would be unaffected; the dominant token is robust.

### Finding 3 — the batch effect perturbs the rejection-test acceptance probability
Converting logits through the same top_k->top_p->temp truncation used by the sampler:
- The acceptance probability `px = min(1, p(draft)/q(draft))` computed from the batched
  verify differs from the batch=1 (AR) value in **72/320 rounds (22.5%)**.
- Magnitude: max single-round `acc_delta = 5.16e-2` (5.2%); among the 72 nonzero rounds
  median 4.85e-3, mean 7.34e-3, p90 1.68e-2.
- Concentrated on uncertain tokens: of 39 "soft" rounds (draft token prob q<0.999), 21
  (54%) show nonzero `acc_delta` (max 2.79e-2); of 281 "sharp" rounds (q>=0.999), 51 (18%)
  show small deltas. Where the truncated distribution collapses to a single token
  (px=1.0 both ways), the large logit differences cancel and `acc_delta = 0`.

### Interpretation (factual)
- Finding 1 and Findings 2-3 are consistent: `mtp-spec` is deterministic *within itself*
  (the batched computation is reproducible), yet its effective sampling distribution is the
  **batched** target distribution, which is not bit-identical to the **autoregressive**
  distribution that `target` mode samples and that the "lossless" rejection-sampling
  identity is defined against. The accept probability the sampler uses is a batch-perturbed
  proxy for the true AR `p`, off by up to ~5% on uncertain tokens.
- The resample branch `(p - q)+` is also built from the batched `p`, so it inherits the same
  perturbation.
- Because argmax never flips, the bias does not change the dominant token, consistent with
  `mtp-spec` matching `target` accuracy. The exactness of "lossless" holds only up to
  batch-size-dependent FP; it is distributional-near-equivalence, not the textbook identity.

### Caveat / scope
The probe holds the KV prefix fixed (both arms attend over the same shared, batched-built
cache) and isolates the marginal batch effect at the query position's final forward. A full
`target`-mode run builds its KV cache entirely under batch=1, so the true cross-mode
divergence is at least this large. `argmax_match` and `acc_delta` were measured only up to
40 rounds/prompt at n=512; Finding 1 reproducibility was measured at n=1024.

### Reproduction
- Finding 1: run `./evaluator/task-eval --gpus 0,1 --modes mtp-spec --limit 6 --n 1024`
  twice; diff `results/raw_mtp-spec_gpu{0,1}.txt` ignoring the `tok/s` line.
- Findings 2-3: `DIAG=40 ./spec_modes --mode mtp-spec -m <TGT> -md <MTP> --seed 0 -n 512
  --gamma 4 < prompts` and parse `DIAG ...` stderr lines. Instrumentation is gated behind
  the `DIAG` env var (default 0 => inert; scored path unchanged).

---

## Extension (researcher-23): per-position, signed, true-AR batch-invariance probe

Deeper extension of Findings 2-3, which measured only verify **row 0** and **draft[0]**, reported
only the **absolute** `|acc_delta|`, fixed `gamma=4`, and held the shared batched KV fixed (so it
isolated the *marginal* single-position batch effect). The `DIAG` hook was replaced by `DIAG2`
(still env-gated, inert when unset; scored path byte-identical — full eval below confirms accuracy).
For **every** verify row `g` (0..G-1, the rows the rejection test actually reads for `draft[g]`),
`DIAG2` compares the batched-verify logits against a **sequential batch=1 reconstruction**: clear
KV to `pos`, then decode `id_last, draft[0], ..., draft[g-1]` one token at a time, so row `g`
attends over a prefix that is *itself* built under batch=1 — the **true autoregressive cache**
`target` mode uses. This captures both the per-position batch effect AND KV contamination that
accumulates with `g`, and reports **signed** deltas (batch − AR). Probe: 12 prompts, 40 rounds
each, n=512, gamma ∈ {2,4,8}, GPUs 0,1, seed 0, temp 1.0, top_p 0.95, top_k 64.

### Finding A — the acceptance-probability perturbation is zero-mean noise, NOT a directional bias
Signed `acc_delta = acc_batch − acc_AR` is nonzero in ~20–24% of verify rows at every gamma; among
the nonzero rows the sign splits ≈50/50 (γ2 +111/−83, γ4 +230/−223, γ8 +428/−441). Per-row signed
mean ≈ ±1e-4, and the aggregate signed-sum changes sign with gamma (γ2 +0.40, γ4 +0.23, γ8 −0.40).
On soft tokens (q<0.999) the signed-sum is likewise small and sign-inconsistent (+0.18, −0.02,
−0.31). Conclusion: the batched verify does **not** systematically over- or under-accept relative
to true AR; its acceptance probability is symmetric noise around the AR value, not a speed-vs-
fidelity lean.

### Finding B — the target's top token DOES flip under the true-AR cache path (contradicts the row-0 result)
`argmax(batched row g) ≠ argmax(true-AR row g)` in 0.4–0.7% of verify rows (γ2 4/960, γ4 13/1920,
γ8 23/3840). Finding 2's marginal single-position probe (shared fixed KV) saw 0/320 flips and
concluded "argmax never flips → greedy unaffected." Once the KV **prefix itself** is built batch=1
vs co-resident-batched, the target's most-likely next token flips occasionally. Flips occur on both
soft and **sharp** (q=1.0) positions — at γ8, 12/23 flips are on sharp tokens. Since this argmax is
exactly what temp≤0 decoding would emit, greedy decoding is **not** invariant across the
batched-verify vs autoregressive cache-build paths (~0.5% of positions diverge).

### Finding C — weak position/KV compounding of the logit drift
Mean `max_abs_logit_diff` grows with verify-row index `g` (linear slope +0.016 to +0.045 per step
across gammas): row 0 ≈ 0.84–0.86, later rows ≈ 0.90–1.02 (≈15–20% growth over 8 positions). Median
per-row drift stays ~0.68–0.80 throughout. The dominant component is the per-position batch effect
itself (~0.7 median, already present at g=0), with a small **additive** accumulation consistent with
KV contamination. The acc-perturbation nonzero fraction does not grow with `g` (it tracks the
soft-token fraction, which is roughly flat across positions).

### Finding D — the perturbation magnitude is ~independent of verify batch size over γ∈{2,4,8}
Increasing the verify batch from 3 (γ2) to 9 (γ8) leaves median per-row logit drift (~0.68–0.80),
nonzero-acc fraction (~20–24%), `|acc_delta|` tails (p99 3.3e-2→3.9e-2; max 0.17→0.20), and
argmax-flip rate (~0.5%) essentially flat. The effect is governed by batched-vs-sequential KV (a
binary), not by the batch dimension's size — it saturates at batch ≥ 3. (Refines Finding 2-3's
framing of "batch-size-dependent FP": the dependence is on *whether* batched, not *how large*.)

### Finding E — true-AR divergence tails are ~2.5–4× larger than the marginal row-0 probe
With the sequential-AR reconstruction (incl. KV contamination), `|acc_delta|` reaches max 0.132 (γ4)
/ 0.202 (γ8), vs Finding 3's row-0 marginal max of 0.052. This quantifies Finding 3's caveat that
"the true cross-mode divergence is at least this large": in the tail it is ~2.5–4× larger.

### Finding F — macro impact on the 30-item benchmark is nil
Full eval `./evaluator/task-eval --gpus 0,1` (gamma=4, n=16384, 30 items): mtp-spec accuracy
**93.3% = target reference 93.3%**; accept_rate **72.4%**; **146.3 tok/s = 1.45× speedup**. The
~0.5% argmax flips and ~23% acceptance perturbations vs true AR do not move downstream accuracy.

### Reproduction
- Probe: `DIAG=40 CUDA_VISIBLE_DEVICES=0,1 ./spec_modes --mode mtp-spec -m <TGT> -md <MTP> --seed 0
  -n 512 --gamma {2,4,8} < results/probe_prompts.txt 2> results/diag_g{G}.txt`, then
  `python3 results/analyze.py` / `results/analyze2.py`. `DIAG2` lines carry per-(round,g) fields
  `argmax_match, max_abs_logit_diff, acc_batch, acc_ar, acc_sdelta` (signed = batch − AR).
- Macro: `./evaluator/task-eval --gpus 0,1`.
- Instrumentation gated behind `DIAG` env (default 0 ⇒ inert; production acceptance re-reads the
  restored batched verify, so the scored path is unchanged).

---

## Extension (researcher-55): seed-stability of true-AR batch-invariance drift

The prior extension fixed seed 0 while varying verify row and gamma. This extension holds the
scored/default gamma fixed at 4 and varies the stochastic trajectory instead. Probe: seeds 0..4,
same 12-prompt set, first 40 verify rounds per prompt, n=512, gamma=4, GPUs 2,3, temp 1.0,
top_p 0.95, top_k 64. Each seed produced 1,920 DIAG2 rows (9,600 rows total), comparing batched
verify against the sequential batch=1 true-AR reconstruction exactly as in researcher-23.

### Finding G - the zero-mean acceptance perturbation is stable across seeds
Across all 9,600 verify rows, `acc_delta = acc_batch - acc_AR` is nonzero in 22.24% of rows, with
nearly balanced signs (+1052 / -1083). The aggregate signed sum is +4.995e-02, i.e. signed mean
+5.203e-06 per row. Per-seed signed sums change sign and remain small: seed0 -2.740e-01, seed1
+8.513e-02, seed2 +2.576e-01, seed3 -1.739e-01, seed4 +1.551e-01. This confirms that the
seed-0 "no directional over/under-acceptance" result was not a single-trajectory accident.

### Finding H - magnitude tails are seed-stable, with rare larger outliers
Seed-level nonzero acceptance perturbation rates are tightly clustered (21.20%, 23.28%, 22.97%,
21.41%, 22.34%). `|acc_delta|` p99 stays in a narrow band (3.297e-02 to 4.114e-02), while maximum
single-row values vary from 8.677e-02 to 1.388e-01. Over all seeds, `|acc_delta|` p99 is 3.846e-02
and max is 1.388e-01. Logit drift is also stable: per-seed median `max_abs_logit_diff` is
0.723..0.731 and mean is 0.898..0.903.

### Finding I - argmax flips persist across seeds but are lower than the seed-0 gamma sweep
Across all seeds, `argmax(batched row g) != argmax(true-AR row g)` occurs in 28/9,600 rows
(0.29%). Every seed has flips: seed0 4/1920 (0.21%), seed1 4/1920 (0.21%), seed2 9/1920
(0.47%), seed3 4/1920 (0.21%), seed4 7/1920 (0.36%). Verify-row dependence persists: g0 0.08%,
g1 0.12%, g2 0.38%, g3 0.58%, matching the earlier direction that later rows have more
true-AR-vs-batched divergence.

### Finding J - prompt variability dominates seed variability for local extremes
Global rates are stable, but prompt-local flip rates vary more than seed-global rates. Most prompts
range from 0.00% to <=1.25% flips across seeds; prompt 6 reaches 3.12% for one seed and prompt 11
reaches 1.88%. Prompt soft-token fractions differ widely (4.38% to 29.00%), which explains why
prompt-local signed sums have larger ranges than seed-global sums, e.g. prompt 8
-2.708e-01..+3.431e-02 and prompt 9 -3.845e-02..+2.052e-01. The global zero-mean result is thus
stable at the seed level but averages over prompt-specific local tails.

### Finding K - scored path remains unchanged on the requested GPU pair
Full eval `./evaluator/task-eval --gpus 2,3` (gamma=4, n=16384, 30 items, seed 0): mtp-spec
accuracy **93.3% = target reference 93.3%**; accept_rate **72.4%**; **146.6 tok/s = 1.46x
speedup**.

### Reproduction
- Build/scored check: `./evaluator/task-eval --gpus 2,3`.
- Probe: render the first 12 benchmark prompts to `results/probe_prompts_seed_sweep.txt`, then run
  `DIAG=40 CUDA_VISIBLE_DEVICES=2,3 ./spec_modes --mode mtp-spec -m <TGT> -md <MTP> --seed {0..4}
  -n 512 --gamma 4 < results/probe_prompts_seed_sweep.txt 2> results/diag_seed{S}.txt`.
- Aggregate: `python3 results/analyze_seed_sweep.py results/diag_seed0.txt ... results/diag_seed4.txt`.
