# Findings — synthesis: composing the draft-temperature and make_dist optimizations

## Lens
Two orthogonal LOSSLESS optimizations were developed in separate lines and are combined here into
one binary, then measured as a 2x2 factorial to test whether their end-to-end gains compose:
- **(A) draft-proposal temperature** (this branch): a draft-only temperature `draft_temp` (env
  `DRAFT_TEMP`). Rejection sampling emits exactly the target `p` for any proposal `q`, so a
  tempered `q_tau` changes only acceptance. tau~2.25 corrects the MTP draft over-confidence and
  raises **emitted-tokens-per-round** (the throughput NUMERATOR).
- **(B) bounded-min-heap `make_dist`** (carried from researcher-50's saved reference,
  `parent_researcher50_make_dist_optimized.cpp`): the host-side top-k that builds the truncated
  sampling distribution scans logits into a 64-entry min-heap instead of materializing and
  `partial_sort`-ing the full 262K-vocab vector (plus reuse of the sampled draft probability).
  This cuts the exposed CPU `make_dist` cost -> **wall-time-per-round** (the DENOMINATOR). Toggled
  by env `MAKE_DIST=heap|partial`. (The "lazy target distributions" half of r-50 — stop building
  target dists at the first rejected row — is already present in the active verify loop, which
  breaks on first rejection; both modes here make the identical 519,944 / 498,510 `make_dist`
  calls, matching r-50's lazy-path count.)

Because (A) is a pure numerator multiplier and (B) reduces a denominator term, the composition
hypothesis is multiplicative. The same binary realizes all four cells via the two env toggles;
the heap is exactly distribution-preserving so cells differing only in `MAKE_DIST` share the same
seed-0 token trajectory (identical rounds/accept/emitted), isolating the timing effect.

## Setup
- Synthesized `initial_program.cpp`: `topk_logits` gains the bounded-min-heap path (env
  `MAKE_DIST`); `make_dist` builds are wrapped in a PURE-CPU timer (`#SPM`) — the GPU sync occurs
  in `llama_get_logits_ith`, evaluated as the call argument BEFORE the timer starts, so `make_dist_us`
  excludes leaked GPU-forward time. Sampled draft prob `q_tau(dtok)` is stored at draft time and
  reused at verify. `DRAFT_TEMP`/`DRAFT_TEMP_GRID`/`#SPP`/`#SPR` machinery preserved.
- Full 30-item benchmark, `./evaluator/task-eval --gpus 0,1`, gamma=4, seed 0, n=16384, one build.
  Aggregates below sum the two GPU chunks; `rest := wall - make_dist` is the GPU-forward+overhead
  residual. `fallback=0` on every cell (residual never degenerate; per-step losslessness intact).

## 2x2 factorial result
| cell | draft_temp | make_dist | accuracy | tok/s | speedup | accept | emitted/round | rounds | CPU make_dist | rest (GPU+ovh) |
|---|---|---|---|---|---|---|---|---|---|---|
| A baseline | 1.00 | partial | 93.3% | 147.9 | 1.47x | 72.4% | 3.898 | 65,830 | 221.0 s | 1504.8 s |
| B temp-only | 2.25 | partial | 90.0% | 150.9 | 1.50x | 74.8% | 3.993 | 62,365 | 217.0 s | 1422.1 s |
| C heap-only | 1.00 | heap | 93.3% | 156.7 | 1.56x | 72.4% | 3.898 | 65,830 | 127.8 s | 1502.5 s |
| **D both** | **2.25** | **heap** | **90.0%** | **160.2** | **1.59x** | **74.8%** | **3.993** | **62,365** | **123.5 s** | **1420.4 s** |

## Composition (tok/s)
| effect | ratio vs A | Δ |
|---|---|---|
| draft_temp (A->B) | 1.0203 | +2.03% |
| heap (A->C) | 1.0595 | +5.95% |
| **both (A->D) measured** | **1.0832** | **+8.32%** |
| multiplicative predict (1.0203x1.0595) | 1.0810 | +8.10% |
| additive predict | 1.0798 | +7.98% |

Aggregate tokens/summed-wall gives the same picture: A->B x1.022, A->C x1.059, A->D x1.085 measured
vs x1.082 multiplicative-predicted.

## Facts
1. **The two gains compose; combined speedup is the best cell (1.47x -> 1.59x).** Measured joint
   tok/s gain +8.32% matches the product of the marginal gains (+8.10%) to within 0.22pp — at or
   inside temp-1.0 trajectory noise. There is no destructive interaction: neither optimization
   eats the other's headroom.
2. **Each optimization's mechanism is invariant to the other knob.** Emitted-tokens-per-round
   (the acceptance numerator) is **identical** across make_dist modes — A=C=3.8978, B=D=3.9929 —
   so draft_temp's +2.44% acceptance lift is the same in the partial and heap builds. Conversely
   CPU `make_dist` time is cut by the same factor at both draft temps — heap/partial = 0.578
   (A->C) and 0.569 (B->D), i.e. ~-42% — so the heap's saving is independent of draft_temp.
3. **The optimizations act on DISJOINT, additive wall-time terms** (`wall = rest + make_dist`):
   - `rest` (GPU forward + overhead) is invariant to make_dist mode: A 1504.8 s ~ C 1502.5 s,
     B 1422.1 s ~ D 1420.4 s (<0.15% drift). The heap touches only host CPU, never the GPU term.
   - `rest`/round is ~constant at 22.8 ms across all four cells; draft_temp lowers TOTAL `rest`
     only by reducing the round count (65,830 -> 62,365, -5.3%, via higher acceptance), not the
     per-round GPU cost.
   - CPU `make_dist` is reduced by the heap (denominator term B) and, slightly and independently,
     by draft_temp (-4.1% make_dist calls, 519,944 -> 498,510, from fewer rounds).
   The separability is exact: predicting cell D's per-chunk wall as (B's `rest`) + (D's heap
   `make_dist`) gives 729.5 s vs the measured 728.9 s (0.08% error). The combined wall reduction
   is the SUM of the two individual reductions, which is why throughput composes (slightly
   super-additively, hence ~multiplicatively).
4. **CPU `make_dist` is ~12.8% of wall in the lean baseline (A): 221.0 s / 1725.8 s.** The heap
   removes ~42% of it (-93 s aggregate), shrinking it to ~7.9% (C). This is the realized magnitude
   of optimization B in a build without r-50's heavier dual-attribution instrumentation; the
   system remains GPU-verify-bound (rest ~87% of wall), so the heap's headline gain (+6.0%) is
   capped by that ~12.8% exposed-CPU share.
5. **Losslessness holds in every cell; the heap is exactly distribution-preserving.** `fallback=0`
   throughout. Accuracy is set only by draft_temp (the per-step output dist is `p` regardless of
   `q` or of the make_dist primitive): A=C=93.3% and B=D=90.0% — the heap reproduces each draft
   temperature's accuracy, accept rate, round count and emitted count to the token. The 90.0% at
   tau=2.25 is the draft_temp=2.25 sampled-trajectory variance documented in the section below
   (length-cap on 2 AIME + 1 HLE under temp-1.0 stochasticity), not a heap or calibration
   regression; per-step output is unchanged.

## Artifacts (this section)
- `initial_program.cpp`: synthesized binary with both optimizations toggleable (env `MAKE_DIST`
  heap/partial, env `DRAFT_TEMP`), pure-CPU `make_dist` timing emitted as `#SPM`, `qprob` reuse.
- `results/saved/{A_temp1.0_partial,B_temp2.25_partial,C_temp1.0_heap,D_temp2.25_heap}/`: the four
  2x2 cells' `task_eval.json`, `mode_mtp-spec.json`, and raw per-chunk diagnostics (`#SPM`/`#SPR`/`#SPP`).

---

# Findings — correcting the draft over-confidence: draft-only temperature recalibration

## Extension lens (this section)
The prior analysis (below) established a specific **miscalibration**: the MTP draft proposal
`q` is ~3.3× too peaked relative to the target `p` (entq≈0.07 vs entp≈0.24), and acceptance
loss tracks the `q>p` gap (cond_acc ≈ mean min(1, px/qx)), not wrong-token placement. This
section goes from *describing* that miscalibration to a **direct empirical test of correcting
it**: rescale only the draft proposal by a temperature `draft_temp`, predict the
acceptance-maximizing value analytically, then confirm its end-to-end effect.

## Why a draft-only temperature is the right probe (and is lossless)
In the rejection sampler a draft token `x ~ q` is accepted w.p. min(1, p(x)/q(x)), else a
token is drawn from the renormalized residual `(p−q)+`. The emitted distribution equals the
target `p` for **any** proposal `q` (output mass = min(p,q) + (p−q)+ = p), so replacing `q`
with a tempered `q_τ` changes only the acceptance rate (hence throughput), never the per-step
output distribution. Expected single-step acceptance under proposal `q_τ` is exactly
`E_{x~q_τ}[min(1,p/q_τ)] = Σ_x min(p(x), q_τ(x)) = 1 − TV(p, q_τ)`, maximized by moving `q_τ`
toward `p`. Because `q` is too peaked, the maximizer is a draft temperature **τ>1**.

## Instrumentation
- `DRAFT_TEMP` env: temperature applied *only* to the draft proposal (build of `q`, the
  min(1,p/q) test, and the `(p−q)+` residual all use the same `q_τ`); target dists stay at
  temp 1.0. Default = target temp, so an unset env reproduces the original binary exactly.
- `DRAFT_TEMP_GRID` env: for every verifier-tested position, accumulate `Σ_x min(p, q_τ)`
  for each candidate τ, recomputed from the stored top-k draft logits (top-k order is
  temperature-independent). One run thus yields the full acceptance-vs-τ curve, emitted as
  `#SPT` lines. `make_dist` refactored into `topk_logits` + `make_dist_from_topk` (behavior
  identical at τ=1.0).
- Full 30-item benchmark, GPUs 0,1, γ=4, seed 0, n=16384. `fallback=0` on every run (the
  residual always had positive support → no degenerate argmax path; losslessness intact).

## Predicted curve from ONE baseline (τ=1.0) run (tested-weighted over 217,630 positions)
| τ_draft | 1.00 | 1.25 | 1.50 | 1.75 | 2.00 | 2.25 | 2.50 | 3.00 | 4.00 |
|---|---|---|---|---|---|---|---|---|---|
| pred. E[accept]=Σmin(p,q_τ) | .8756 | .8792 | .8823 | .8849 | .8866 | **.8874** | .8867 | .8794 | .8274 |
| Δ vs τ=1 (pp) | 0 | +.35 | +.67 | +.92 | +1.10 | **+1.17** | +1.11 | +.38 | −4.83 |

Interior optimum at **τ≈2.25** (broad plateau 2.0–2.5); over-flattening (τ=4) is sharply
worse than baseline. The τ=1.0 prediction (.8756) matches the realized baseline cond_acc
(.8766).

## Confirmatory end-to-end runs (predicted optimum and an over-flatten control)
| draft_temp | accuracy | tok/s | accept_rate | speedup | cond_acc (real/pred) | emitted/round | mean_resid |
|---|---|---|---|---|---|---|---|
| 1.00 (baseline) | 93.3% | 146.7 | 72.4% | 1.46× | .8766 / .8756 | 3.898 | 0.585 |
| **2.25 (corrected)** | 90.0% | **150.1** | **74.8%** | **1.49×** | **.8873 / .8874** | **3.993** | 0.508 |
| 4.00 (over-flat) | 96.7% | 133.5 | 63.9% | 1.33× | .8320 / .8274 | 3.555 | 0.451 |

## Facts
1. **The predictive overlap diagnostic is accurate off-baseline.** Realized conditional
   acceptance vs the `Σmin(p,q_τ)` predicted from the τ=1.0 trajectory: τ=1.0 .8766/.8756,
   τ=2.25 .8873/.8874, τ=4.0 .8320/.8274 — agreement within ≤0.005 even though each run
   follows a different token trajectory. The curve computed from one run predicts the optimum
   and the realized acceptance at other temperatures.
2. **Correcting the over-confidence raises acceptance and throughput, for free.** A single
   scalar (draft_temp 1.0→2.25) lifts unconditional accept_rate 72.4%→74.8% (+2.4pp),
   conditional accept .8766→.8873, and tok/s 146.7→150.1 (+2.3%, 1.46×→1.49×). No extra
   compute — the draft logits are already produced; only their temperature is rescaled.
3. **Throughput tracks emitted-tokens-per-round, which tracks acceptance.** Each round costs
   ~constant time (γ drafts + 1 verify), and emitted/round = accepted/round + 1: 3.898 →
   3.993 → 3.555 for τ = 1.0/2.25/4.0. The emitted/round ratios (1.024×, 0.912× vs baseline)
   match the measured tok/s ratios (1.023×, 0.910×) to <0.2pp.
4. **The optimum is interior, not "flatter is better."** τ=4.0 over-flattens `q` past `p`,
   dropping accept_rate to 63.9% and tok/s to 133.5 (1.33×, below baseline). The acceptance
   and throughput curves both peak at the predicted ~2.25.
5. **Per-step losslessness holds; accuracy is decoupled from draft_temp.** Accuracy across
   τ=1.0/2.25/4.0 is 93.3%/90.0%/96.7% — non-monotonic and uncorrelated with draft_temp,
   as expected when the per-step output distribution is exactly `p` regardless of `q`. The
   three τ=2.25 misses were 2 AIME items truncated at the 16384-token cap (decoded=16384/16385)
   plus 1 HLE item — sampled-trajectory-length variance under temp-1.0, not a draft-quality
   regression. `fallback=0` on all runs (residual never degenerate).
6. **Conditional-on-rejection divergence shrinks monotonically with draft_temp** (mean_resid
   0.585→0.508→0.451 for τ=1.0/2.25/4.0): a flatter `q` overlaps `p` more even where it loses,
   but past the optimum the *number* of rejections rises (more rounds: 65,830→62,365→73,175),
   so total acceptance falls despite each rejection being milder.

## Artifacts (this section)
- `initial_program.cpp`: `topk_logits`/`make_dist_from_topk` refactor; `DRAFT_TEMP` proposal
  temperature; `DRAFT_TEMP_GRID` predictive `#SPT` overlap diagnostic; `#SPR` now reports
  `draft_temp`. All mtp-spec only; diagnostics emitted outside `=== PROMPT ===` blocks.
- `results/saved/run{1,2,3}_*`: baseline (τ=1.0), corrected (τ=2.25), over-flatten (τ=4.0)
  raw per-chunk diagnostics and scored summaries.

---

# Findings — conditional (per-position) acceptance within the speculative window

## Lens
Decompose mtp-spec acceptance by **draft-window position g (0..γ-1)** rather than as a
per-item scalar or an accepted-length histogram. For each position the verifier *reaches*
(i.e. all earlier draft tokens in that round were accepted), record the **conditional**
acceptance P(accept g | survived 0..g-1) and its drivers: target token prob `px`, draft
token prob `qx`, target dist entropy `entp`, draft dist entropy `entq`, and the residual
overlap deficit on rejection.

## Setup
- Instrumented `initial_program.cpp` (mtp-spec path): per-position counters emitted on stdout
  as `#SPP`/`#SPR` lines *outside* the `=== PROMPT ===` blocks, so the scorer is unaffected.
  Entropies computed over the truncated (top_k=64 → top_p=0.95 → temp=1.0) sampling dists.
- Full 30-item benchmark, GPUs 2,3, γ=4, seed 0, n=16384.
- Run result: accuracy **93.3%** (== target reference, lossless), **147.9 tok/s**,
  accept_rate **72.4%**, speedup **1.47×**. (Reference table: 1.51×; temp-1.0 stochastic.)
- Aggregates below sum both GPU chunks (65,830 verification rounds total).

## Per-position table (tested-weighted)
| g | tested | accepted | cond_acc | mean_px | mean_qx | entp | entq |
|---|--------|----------|----------|---------|---------|------|------|
| 0 | 65830 | 57316 | 0.8707 | 0.846 | 0.951 | 0.281 | 0.082 |
| 1 | 57316 | 50234 | 0.8764 | 0.854 | 0.955 | 0.247 | 0.076 |
| 2 | 50234 | 44250 | 0.8809 | 0.864 | 0.960 | 0.215 | 0.068 |
| 3 | 44250 | 38964 | 0.8805 | 0.866 | 0.962 | 0.201 | 0.063 |

Overall conditional acceptance (over tested positions): **0.8766**.

## Facts
1. **Conditional acceptance (the hazard) is flat across the γ=4 window** — 0.871, 0.876,
   0.881, 0.881 — slightly *rising*, not decaying. Over a 4-token horizon there is no
   measurable degradation in per-step accept probability from the MTP head conditioning on
   its own (drafted) hidden states.

2. **Target entropy at tested positions falls monotonically with depth** (0.281 → 0.247 →
   0.215 → 0.201 nats). A deeper slot is only reached when earlier tokens were accepted,
   which selects into lower-uncertainty contexts; this survivorship coincides with the flat
   (slightly rising) hazard.

3. **The MTP draft is systematically more peaked than the target.** Draft token prob
   `qx`≈0.95–0.96 vs target `px`≈0.85–0.87, and draft entropy `entq`≈0.06–0.08 is ~3.3×
   lower than target entropy `entp`≈0.20–0.28 at every position. The mean acceptance ratio
   min(1, px/qx) ≈ cond_acc, i.e. acceptance loss tracks the q>p gap rather than the draft
   placing mass on outright-wrong tokens.

4. **Marginal accepted tokens per slot decline through survivorship only, not quality.**
   P(reach slot g): 1.000, 0.871, 0.763, 0.672; expected accepted tokens contributed by
   each slot (= P reach g+1): 0.871, 0.763, 0.672, 0.592. The decline comes from attrition
   of surviving rounds while the per-step accept prob stays ≈0.877. Accepted length is thus
   near-geometric with α≈0.877.

5. **Per-drafted-token accounting (fixed γ=4):** 72.4% accepted, 10.2% verified-then-rejected,
   17.4% drafted downstream of an earlier rejection and never verified. Of 4 tokens drafted
   per round, 82.6% are actually verified. The 72.4% headline accept_rate is the
   *unconditional* figure over all drafted tokens; the *conditional* (verified-only) figure
   is 87.7%.

6. **Rejections are high-divergence events.** On rejection the residual overlap deficit
   (1 − Σ min(p_j, q_j) over the truncated dists, i.e. residual mass before renormalization)
   averages 0.584 — target and draft truncated dists share only ~42% of their mass at the
   positions where a token is rejected. `fallback`=0 across the run: the residual (p−q)+
   always had positive support, so the lossless correction never degenerated to argmax.

7. **Throughput identity:** each round drafts 4, accepts 2.90, and emits 1 bonus/resampled
   token → 3.90 emitted tokens per target verification pass (batch width γ+1 = 5);
   all-4-accepted in 59.2% of rounds.

## Artifacts
- `initial_program.cpp`: added `dist_entropy()` and per-position `#SPP`/`#SPR` diagnostics
  (mtp-spec only; emitted after the prompt loop, outside parsed blocks).
- Raw per-chunk diagnostics: `results/raw_mtp-spec_gpu2.txt`, `results/raw_mtp-spec_gpu3.txt`.
- `results/task_eval.json`, `results/mode_mtp-spec.json`: scored run summary.

---
# (merged reference) researcher-50 findings (make_dist heap+lazy lossless host-side speedup, ~1.56x):

# Findings — synthesis of two contradicting wall-clock analyses

This branch merges two prior independent analyses of MTP speculative decoding that reached
**opposite conclusions about where wall-clock time goes**:

- **Line A (researcher-29):** the timed GPU decode phases are only **4.4% of wall-clock**; the
  remaining **95.6%** is untimed host-side `make_dist` (an O(vocab) `partial_sort` over the
  ~262K-token Gemma vocabulary). Conclusion: the system is **sampling-bound**.
- **Line B (researcher-8):** that result is an artifact of **asynchronous `llama_decode`**. After
  inserting an explicit `llama_synchronize()` after every decode, GPU forward = **87.1%** of
  wall-clock and the target **verify forward pass dominates at 69.4%**; `make_dist` is ~12.6%.
  Conclusion: the system is **verify-bound**.

Both lines analyzed the same workload and agreed on the *outcome* of the run (≈1720 s wall,
≈149 tok/s, 93.3% accuracy, 72.4% accept). They disagreed only on the *decomposition* of that
wall-clock. The synthesis below resolves the disagreement with a single controlled measurement,
then keeps the (undisputed) acceptance-side findings both lines built on.

---

## Resolution — single-run dual-attribution timing experiment

Evaluation command: `./evaluator/task-eval --gpus 2,3` (full 30-item run). Run summary with this
instrumentation: `mtp-spec` accuracy **93.3%** (identical to the target reference — instrumentation
is non-perturbing to quality), 140.6 tok/s, speedup 1.39x, aggregate accept 72.4%. The throughput
is lower than the clean reference (151.8 tok/s) because this build carries *both* lines' overheads
at once (researcher-29's per-row overlap/entropy instrumentation **and** researcher-8's explicit
synchronize barriers); this affects absolute tok/s but not the relative decomposition that is the
subject of the dispute.

### Why the two lines disagree (root cause)

`llama_decode` is **asynchronous**: it returns after enqueuing GPU work, before that work
completes. The active program (Line A) times `draft_us`/`verify_us` by wrapping `llama_decode`
**alone**, with no synchronize. Those timers therefore capture only the host-side kernel *launch*,
not the GPU forward compute. The forward compute completes later, at the first operation that
forces a device sync — `llama_get_logits_ith` **inside the next `make_dist`**. So in Line A's
accounting the GPU forward time is silently transferred onto the `make_dist` wall-time that
follows it. Line B's `llama_synchronize()` after each decode forces the forward to complete inside
the decode timer, moving that same time back where it belongs.

### Method

The active `initial_program.cpp` was instrumented to measure, around **every** `llama_decode`,
three disjoint quantities from one set of timestamps:
- `*_launch_us` — `llama_decode` alone (Line A's method, no sync),
- `*_sync_us` — an explicit `llama_synchronize()` immediately after (captures exactly the GPU
  forward time the launch timer misses, i.e. the time that without the sync would be billed to the
  following `make_dist`),
- `*_smp_us` — `make_dist`+sample, now executing as **pure CPU** (its input logits are already
  resident).

Line A's decomposition is then `launch` for decode with `sync` folded into `make_dist`; Line B's
decomposition is `launch + sync` for the forward pass with `make_dist` as pure CPU. Both are
computed from the **same** run; the only difference is which bucket the `sync` time lands in.

### Result — both prior conclusions reproduced from one run

Aggregate over 30 prompts: 256,580 tokens, 1824.5 s summed wall, 565,634 `make_dist` calls. Timed
buckets sum to 1819.5 s (99.7% of wall; the 5.0 s remainder is cache/bookkeeping).

Raw phase totals (seconds):

| phase | launch (Line-A "decode") | sync (leaked GPU forward) | make_dist (pure CPU) |
|---|---:|---:|---:|
| prompt/prefill | 2.3 | 1.2 | — |
| draft (MTP) | 57.4 | 249.9 | 143.1 |
| verify (target) | 26.1 | 1170.8 | 168.7 |

**Reproducing Line A** (launch-only decode; sync time leaks into `make_dist`):

| bucket | time | share of measured |
|---|---:|---:|
| timed GPU decode (launch only) | 85.8 s | **4.7%** |
| host-side `make_dist` (as Line A billed it) | 1733.8 s | **95.3%** |

This matches researcher-29's recorded "timed GPU decode = 4.4% of wall, 95.6% untimed host-side
`make_dist`."

**Reproducing Line B** (decode = launch + sync = true GPU forward; `make_dist` = pure CPU):

| phase | time | share of timed |
|---|---:|---:|
| prefill | 3.5 s | 0.2% |
| draft_fwd (MTP) | 307.3 s | 16.9% |
| **verify_fwd (target)** | **1196.9 s** | **65.8%** |
| draft_smp (CPU make_dist) | 143.1 s | 7.9% |
| verify_smp (CPU make_dist) | 168.7 s | 9.3% |
| GPU forward total | 1507.7 s | **82.9%** |
| CPU `make_dist` total | 311.8 s | **17.1%** |

This matches researcher-8's recorded "GPU forward = 87.1%, target verify forward dominates at
69.4%, CPU sampling+cache = 12.8%." The residual gaps (verify_fwd 65.8% vs 69.4%; CPU 17.1% vs
12.8%) are accounted for by this build carrying researcher-29's heavier per-verify-row `make_dist`
instrumentation (a `Dist` is built for **every** verified row plus `overlap_decomp`/entropy),
which inflates the CPU `make_dist` term relative to researcher-8's lean path.

### The reclassified quantity (the entire dispute, isolated)

The disagreement is one number: the `sync` time, the GPU forward compute that Line A's launch-only
timers could not see.

- sync time mis-attributed to `make_dist` by Line A = **1422.0 s = 78.2% of measured time**.
- pure-CPU cost per `make_dist` call (262K-vocab top-64 `partial_sort` + Dist build) = **551 µs**;
  over 565,634 calls = 311.8 s.

So 78 of Line A's 95 "make_dist" percentage points are GPU forward compute waiting at the sort's
sync point; only ~17 points are the sort itself.

### Reconciled account

1. Both lines measured the same physical run (≈1820 s wall, ≈140–150 tok/s, 93.3% accuracy). The
   contradiction was never about the data, only about attribution.
2. The system is **verify-bound**, not sampling-bound: the target verification forward pass is
   **~66% of timed wall-clock** (researcher-8's ~69% on its lean build), the largest single cost.
   MTP drafting forward is the second cost (~17%).
3. Line A's "95% make_dist / sampling-bound" is an **async-attribution artifact**: launch-only
   decode timers leave the GPU forward time to be charged to the next sync, which is `make_dist`.
4. Line A's raw observation was not false, only mislabeled: `make_dist`'s wall-time really was
   ~95% of measured time — but ~78 of those points are GPU forward, not sorting.
5. The CPU sort is nonetheless **fully exposed on the critical path** and is a real cost
   (~12.8–17.1%): inserting synchronize barriers does **not** change total wall-clock
   (researcher-8: 1724 s with barriers ≈ researcher-29: 1719 s without), which shows there is no
   CPU/GPU overlap in the current codepath for the barriers to destroy — every decode's output is
   immediately consumed by a `make_dist` that forces a sync regardless. Both the GPU forward
   (~83%) and the CPU sort (~17%) are additive contributors to wall-clock. The kernel of truth in
   Line A is that the sort is not free/hidden; the correction is its magnitude (~17%, not ~95%).

### Per-op GPU forward costs (reconciled with researcher-8's γ-sweep)

From the sync-attributed forward times on the full γ=4 run: a serial MTP draft forward averages
307.3 s / draft_calls and the target verify forward averages 1196.9 s / verify_calls. researcher-8's
single-tensor-split γ-sweep fit the verify forward as **≈7.0 ms fixed + ≈2.07 ms per token in the
batch**, and an MTP draft step at ≈1.9 ms (tensor-split) / ≈1.15 ms (single-GPU). Both lines agree
on the structural asymmetry: **drafting is linear in γ (γ sequential width-1 passes), verification
is sub-linear (one batched pass), and a 5-token verify batch costs ≈1.96× a single token — far
from free**, because the MoE target (4B active of 26B) is not in the weight-bandwidth-bound regime
where extra batch tokens are nearly free.

---

# Extension — practical reducibility of the exposed CPU top-k cost

The exposed CPU `make_dist` cost above is not an irreducible consequence of speculative decoding;
it depends strongly on the host-side top-k implementation used to construct the truncated
distribution.

## Method

The original `make_dist` built a 262K-entry `(logit, token_id)` vector on every call, then used
`std::partial_sort` to order the top 64 entries. Two exact top-k alternatives were evaluated on
the same full 30-item benchmark using `./evaluator/task-eval --gpus 4,5`:

1. **Full-buffer `nth_element` + sort top-k:** still materializes the 262K-entry vector, partitions
   it with `std::nth_element`, then sorts the retained 64 entries.
2. **Bounded min-heap scan:** scans logits once, retains only the current top 64 candidates in a
   64-entry heap, then sorts those 64 entries before the same top-p/temp normalization. This avoids
   allocating and permuting the full vocabulary-sized candidate vector. The selected top-k set is
   exact except for irrelevant equal-logit tie ordering.

## Result

All three variants used the same number of distribution constructions on the benchmark:
565,634 `make_dist` calls. Accuracy and acceptance were unchanged in the two evaluated variants:
`mtp-spec` accuracy 93.3%, accept rate 72.4%.

| `make_dist` implementation | CPU `make_dist` time | per call | `mtp-spec` tok/s | speedup vs target ref |
|---|---:|---:|---:|---:|
| original full-vector `partial_sort` | 311.8 s | 551 µs | 140.6 | 1.39x |
| full-vector `nth_element` + sort top-k | 755.1 s | 1335 µs | 112.6 | 1.12x |
| bounded 64-entry min-heap scan | 141.5 s | 250 µs | 155.3 | 1.54x |

The `nth_element` substitution was worse than `partial_sort`: keeping the full 262K-entry buffer
but changing the selection primitive increased CPU distribution time by 443.3 s (+142%).

The bounded-heap version reduced exposed CPU distribution time by 170.3 s relative to the original
`partial_sort` path (311.8 s → 141.5 s, −54.6%). On the same full benchmark it decoded 256,580
tokens in 1644.1 s aggregate wall time (156.1 aggregate tok/s; evaluator mean 155.3 tok/s). The
timed buckets summed to 1640.7 s, of which CPU `make_dist` was 141.5 s = 8.6%; in the original
dual-attribution run CPU `make_dist` was 311.8 s = 17.1% of timed work.

## Interpretation

The exposed CPU cost was practically reducible by about half without changing the speculative
decoding accept/reject logic, the target/draft distributions, benchmark accuracy, or accept rate.
The decisive implementation detail was not replacing `partial_sort` with another full-buffer
selection primitive; it was avoiding the vocabulary-sized candidate vector and retaining only the
top-k frontier during the scan.

## Follow-up — avoiding dead verifier-row distribution construction

A deeper lossless reduction was tested in the same host-side cost center. The target verification
forward still computes the full `[id_last, draft...]` batch, but the CPU construction of target
sampling distributions now proceeds in acceptance order and stops at the first rejection. This
skips target `make_dist` for draft positions after the first rejected token in the round; those
rows are never reached by the rejection sampler and cannot affect the emitted token. The rejected
row itself is still built, because its target distribution is needed for the residual
`max(p-q,0)` sample.

The implementation also removes two smaller host-side costs: draft distributions are moved into
the per-round storage instead of copied after sampling, and the sampled draft probability plus the
target rank/probability of the sampled draft token are reused instead of recovered by later linear
support scans.

Two full 30-item evaluations were run on this branch using `./evaluator/task-eval --gpus 6,7`:

| variant | `make_dist` calls | CPU `make_dist`/sampling time | per call | `mtp-spec` tok/s | accuracy | accept |
|---|---:|---:|---:|---:|---:|---:|
| bounded 64-entry min-heap scan (prior result) | 565,634 | 141.5 s | 250 µs | 155.3 | 93.3% | 72.4% |
| attempted unsorted frontier + min rescan | 565,634 | 185.3 s | 328 µs | 150.6 | 93.3% | 72.4% |
| bounded heap + lazy target distributions + reuse | 519,944 | 119.7 s | 230 µs | 156.6 | 93.3% | 72.4% |

The unsorted-frontier attempt was exact but worse than the heap: recomputing the minimum slot after
each top-k replacement increased CPU distribution time by 43.8 s relative to the prior bounded
heap path.

The lazy target-distribution path skipped **45,690** target distribution constructions. This equals
`drafted - accepted - rejected_rounds = 263,320 - 190,764 - 26,866`, i.e. draft rows after the
first rejected token, excluding the rejected row itself. Combined with the smaller reuse changes,
CPU `make_dist`/sampling time fell from 141.5 s to 119.7 s (−15.4%) relative to the prior heap
path, and from 311.8 s to 119.7 s (−61.6%) relative to the original `partial_sort` path. Accuracy
remained 93.3% and accept rate remained 72.4%.

---

# Acceptance-side findings (undisputed; carried forward from both lines)

These analyses concern the **numerator** of speedup (useful tokens per round) and are independent
of the timing dispute above. They are recorded as established facts.

## Draft position reach and confidence

Later draft positions were often generated but never reached by the acceptance loop because an
earlier token in the same round rejected:

| draft position | drafted | reached | accepted | rejected | unreached | reached/drafted |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 65,830 | 65,830 | 57,316 | 8,514 | 0 | 100.00% |
| 1 | 65,830 | 57,316 | 50,234 | 7,082 | 8,514 | 87.07% |
| 2 | 65,830 | 50,234 | 44,250 | 5,984 | 15,596 | 76.31% |
| 3 | 65,830 | 44,250 | 38,964 | 5,286 | 21,580 | 67.22% |

Conditional acceptance given that a draft position was reached was similar across positions:
87.07%, 87.64%, 88.09%, and 88.05% for positions 0 through 3.

Target/draft top-1 agreement separated accepted and rejected reached drafts more strongly than
draft sampled-token probability. Among accepted reached drafts, top-1 agreement was 96.24%,
96.43%, 96.79%, 96.97% for positions 0–3; among rejected reached drafts, 50.29%, 43.93%, 40.41%,
37.80%. The sampled draft token's mean rank under the target distribution was ≈1 for accepted
drafts and ≈2 for rejected drafts.

## Distributional-overlap decomposition of acceptance loss

At each reached draft position the distributional overlap `O = sum_x min(p(x), q(x)) = 1 - TV(p,q)`
between the truncated-renormalized target `p` and draft `q` is the theoretical single-token accept
probability. Over 217,630 reached draft positions, empirical conditional acceptance was 87.66% and
mean theoretical `O` was 87.56% (empirical acceptance matches theory within each entropy bin and in
aggregate). Mean loss `1 - O` = 12.44% decomposed into **2.32% out-of-nucleus mass** (draft mass on
tokens with `p(x)=0`, structurally unacceptable) and **10.12% overconfidence mass** (`q(x)>p(x)>0`,
draft over-weighting shared tokens) — overconfidence is 4.4x the out-of-nucleus mass. Of 26,866
rejection events, 81.2% were in-nucleus and 18.8% out-of-nucleus.

The draft distribution was sharper than the target: mean target entropy over reached positions
0.2405; on accepted positions target entropy 0.1494 vs draft 0.0454; on rejected positions target
0.8872 vs draft 0.2715.

Acceptance stratified by target entropy `H(p)`:

| H(p) bin | reached | accept | theoretical O | share of reached |
|---|---:|---:|---:|---:|
| [0, 0.25) | 158,247 | 97.88% | 97.87% | 72.71% |
| [0.25, 0.5) | 13,847 | 77.92% | 77.86% | 6.36% |
| [0.5, 1.0) | 26,749 | 62.20% | 61.82% | 12.29% |
| [1.0, 2.0) | 17,328 | 45.77% | 45.51% | 7.96% |
| [2.0, 3.0) | 1,452 | 34.99% | 32.95% | 0.67% |
| [3.0, inf) | 7 | 28.57% | 24.90% | 0.00% |

72.71% of reached positions had target entropy below 0.25 and accepted at 97.88%.

## Round-level path-accept calibration

Across 65,830 rounds: 38,964 all-accepted, 26,866 rejected. Observed all-accept rate 59.19%; mean
sampled-path all-accept probability `prod_g min(1, p(draft_g)/q(draft_g))` = 58.99%; mean product
of overlaps 58.80%. The sampled-path all-accept probability separated outcomes (86.04% on
all-accepted rounds vs 19.76% on rejected). Binning by predicted path-accept, the top bin [0.98,1]
held 35.51% of rounds at 100.00% all-accept and 3 dead verify rows; the bottom bin [0,0.25) held
30.31% of rounds and 70.25% of all dead verify rows. First-rejection position over rejected rounds:
pos 0 31.69%, pos 1 26.36%, pos 2 22.27%, pos 3 19.68%.

## Drafted-pass waste and draft-side adaptive-gating counterfactual

Of 263,320 drafted tokens, 190,764 accepted, so 72,556 (27.6%) were wasted serial MTP draft passes
(the most expensive per-token GPU op); 22.04% of (cheaper) verify rows were dead. A draft-side-only
stopping rule — truncate the round at `G'(tau) = (first g with draft entropy H(q_g) > tau) + 1` —
depends only on the draft distribution observed before verification, so it is realizable online,
and truncating speculation length is lossless (the bonus token remains a target sample). At the
aggressive plateau (tau in [0.02, 0.10]) it truncates 33.56% of rounds, removes 18.20% of serial
draft passes, loses 6.52% of emitted tokens, and raises emitted-tokens-per-draft-pass from 0.9745
to 1.1136 (+14.3%); amortized decode-throughput gain 1.10x (GPU-decode model) to 1.115x
(sampling-bound model), the two models agreeing within ~1.5%.

Realizable round-level predictor by max draft entropy across the round:

| max H(q) bin | rounds | share | all-accept | mean accepted tokens |
|---|---:|---:|---:|---:|
| [0, 0.25) | 41,421 | 62.92% | 75.94% | 3.385 |
| [0.25, 0.5) | 6,949 | 10.56% | 36.64% | 2.270 |
| [0.5, 1.0) | 13,677 | 20.78% | 31.75% | 2.107 |
| [1.0, 2.0) | 3,661 | 5.56% | 16.77% | 1.601 |
| [2.0, 3.0) | 117 | 0.18% | 4.27% | 0.957 |
| [3.0, inf) | 5 | 0.01% | 0.00% | 1.200 |

The draft-side signal is realizable but noisier than the target-side path-accept predictor: the
draft top bin (max H(q) < 0.25) holds 62.92% of rounds yet only 75.94% fully accept, whereas the
target-side top bin held 35.51% of rounds at 100% all-accept — consistent with the draft
overconfidence measured in the overlap decomposition.
</content>
</invoke>
