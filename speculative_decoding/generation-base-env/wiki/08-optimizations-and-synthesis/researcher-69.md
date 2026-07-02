# Findings — three-lever synthesis: do draft_temp, make_dist-heap, and entropy_sticky compose?

## Lens
This branch's active binary already combined TWO lossless levers — (A) draft-proposal temperature
`DRAFT_TEMP` and (B) bounded-min-heap `make_dist` (`MAKE_DIST=heap`). This work grafts in the
THIRD, mechanistically-distinct lever from the researcher-61 line — (C) entropy-gated mode recovery
`SPEC_VERIFY=entropy_sticky` (force-accept a recoverable artifact — a draft token equal to the
TARGET argmax that the lossless test rejects only because `q>px` — iff target verification entropy
`Ht < SPEC_ENT_THRESH`) — into one binary with all three toggles, and measures the combined
end-to-end effect plus how the three compose.
- A acts on the NUMERATOR (acceptance) LOSSLESSLY: tempering the over-peaked proposal `q` raises
  `min(1,px/q_τ)`, emitted dist stays `= p`.
- B acts on the DENOMINATOR (host-CPU `make_dist` time), distribution-preserving.
- C acts on the NUMERATOR too, but by a DIFFERENT mechanism: it force-accepts artifact rejections
  (changing per-step output) where the target is low-entropy/self-determined, accuracy-safe at τ=1.0.
The hypothesis under test: B is orthogonal to both (pure denominator), but **A and C are not
independent — they target the SAME recoverable-artifact pool** (the `q>px` artifact rejections),
so they should be substitutes, not multiplicative.

## Setup
- Synthesized `initial_program.cpp`: kept the existing `DRAFT_TEMP`/`MAKE_DIST` machinery; grafted
  the r61 `SPEC_VERIFY` gate into the verify loop (one `uni(rng)` draw per tested position, so the
  RNG codepath is byte-identical to lossless when no force fires). Added `#SPC` telemetry: per
  tested position, artifacts (`draft==target argmax`), recoverable artifacts (`artirej` = the
  lossless-rejected artifact pool C taps), forced accepts, stoch accepts, and the recoverable-pool
  `Ht` histogram — so the A↔C overlap is directly measurable.
- Factorial of 6 cells, `./evaluator/task-eval --gpus 6,7 --modes mtp-spec --n 16384`, full 30
  items, gamma=4, seed 0. A∈{1.0, 2.25(optimum)}, B∈{partial, heap}, C∈{lossless, entropy_sticky
  τ=1.0}. The 4 trajectory-distinct A×C cells run at B=heap; two B=partial cells (true baseline and
  all-three) isolate B's timing. `make_dist_us` aggregates the two GPU chunks. `fallback=0` on every
  cell. Results saved under `results/saved/C{1..6}_*/`.

## 2×2×2 factorial result (mtp-spec; target reference 100.7 tok/s, 93.3%)
| cell | levers | accuracy | tok/s | speedup | accept | make_dist s | rounds | artirej pool | forced |
|---|---|---|---|---|---|---|---|---|---|
| C6 (none) | A− B− C− | 93.3% | 142.6 | 1.42× | 72.4% | 273.6 | 65,830 | 10,459 | 0 |
| C1 (B) | A− B+ C− | 93.3% | 155.5 | 1.54× | 72.4% | 129.0 | 65,830 | 10,459 | 0 |
| C2 (B+A) | A+ B+ C− | 90.0% | 159.2 | 1.58× | 74.8% | 123.8 | 62,365 | 6,544 | 0 |
| C3 (B+C) | A− B+ C+ | 93.3% | 164.2 | 1.63× | 79.3% | 151.4 | 72,316 | 12,072 | 6,873 |
| **C4 (ALL 3)** | **A+ B+ C+** | **90.0%** | **164.9** | **1.64×** | **79.8%** | 137.9 | 68,301 | 7,286 | 4,320 |
| C5 (A+C, no B) | A+ B− C+ | 90.0% | 151.3 | 1.50× | 79.8% | 287.9 | 68,301 | 7,286 | 4,320 |

## Composition (tok/s, B=heap anchor C1=155.5)
| effect | ratio | Δ |
|---|---|---|
| A only (C1→C2) | 1.0239 | +2.39% |
| C only (C1→C3) | 1.0558 | +5.58% |
| **A+C (C1→C4) measured** | **1.0599** | **+5.99%** |
| multiplicative predict (1.0239×1.0558) | 1.0810 | +8.10% |
| additive predict | — | +7.96% |

## Facts
1. **B (heap) is orthogonal to both A and C; A and C are NOT orthogonal to each other.** The system
   factors as `(orthogonal B) × (competing {A,C})`. End-to-end true-baseline→all-three: 142.6→164.9
   tok/s = **+15.6% (1.42×→1.64×)**, which equals B(+9.1%) × (A+C)(+6.0%) exactly.
2. **B composes multiplicatively and is distribution-exact in the triple.** Heap's gain is invariant
   to A,C state: +9.08% at A−/C− (C1/C6) vs +8.96% at A+/C+ (C4/C5), within 0.12pp. The trajectories
   are byte-identical across the heap/partial toggle — C1≡C6 (65,830 rounds, 10,459 artirej) and
   C4≡C5 (68,301 rounds, 4,320 forced) — only `make_dist_us` differs (heap/partial = 0.471 and 0.479).
   B touches only host CPU; it never moves the numerator.
3. **A and C are SUB-additive substitutes: measured A+C = +5.99% vs +7.96% additive / +8.10%
   multiplicative predicted — a ~2pp destructive interaction.** C's marginal collapses from +5.58%
   (alone, C1→C3) to +3.52% (on top of A, C2→C4); reciprocally A's marginal collapses from +2.39%
   (alone) to +0.39% (on top of C, C3→C4).
4. **The interaction mechanism is quantitatively closed: A removes ~37% of C's recoverable-artifact
   pool.** Three independent measurements agree:
   - Static pool (lossless cells): A (draft_temp 1.0→2.25) shrinks `artirej` 10,459→6,544 = **−37.4%**.
     A flattens the over-peaked `q`, lowering `q(dtok)` so artifact tokens that were rejected (`q>px`)
     now pass the lossless stoch test — A absorbs them LOSSLESSLY, before C can force them.
   - Realized recoveries (entropy_sticky cells): forced 6,873 (C-only)→4,320 (A+C) = **−37.1%**.
   - C's throughput marginal shrinks **−36.9%** when A is on (+5.58%→+3.52%).
   All ~37%: A and C compete for the identical `q>px` artifact positions, so their gains do not stack.
5. **Accuracy effects are separable even though throughput effects are not.** Accuracy is set
   entirely by A: every τ=1.0 cell is 93.3%, every τ=2.25 cell is 90.0%. C is accuracy-neutral on
   top of A (C3 holds 93.3% with A off; C4 holds A's 90.0% with A on — entropy_sticky τ=1.0 adds no
   further loss); B never affects accuracy. The 90.0% is A's documented temp-1.0 trajectory-length
   variance (2 AIME length-caps + 1 HLE), not a C or heap regression. The forced pool sits at low
   entropy throughout (`forced_bands` populate only Ht<1.0, mean forced Ht ≈ 0.62–0.64).
6. **On this benchmark C throughput-dominates A.** C's standalone gain (+5.58%) exceeds A's (+2.39%)
   and carries zero accuracy cost, whereas A costs 3.3 points. With B+C already on, the third lever A
   adds only +0.39% tok/s (164.2→164.9) while moving accuracy 93.3%→90.0%. The highest
   accuracy-preserving cell is B+C (C3): 164.2 tok/s, 1.63×, 93.3%; the literal all-three cell (C4)
   is +0.4% tok/s above it at −3.3pts accuracy.

## Artifacts (this section)
- `initial_program.cpp` — three toggleable levers in one binary: `DRAFT_TEMP` (A), `MAKE_DIST=heap|partial`
  (B), `SPEC_VERIFY=lossless|entropy_sticky|argmax_sticky` + `SPEC_ENT_THRESH` (C); `#SPC` telemetry
  (artifact/recoverable/forced/stoch counts + recoverable-pool Ht histogram).
- `run_cell.sh` — per-cell runner (sets env, runs eval, saves artifacts).
- `analyze_triple.py` — regenerates the 2×2×2 table, B-orthogonality check, and A↔C substitution
  decomposition from the saved cells.
- `results/saved/C{1..6}_*/` — per-cell `task_eval.json`, `mode_mtp-spec.json`, raw `#SPC`/`#SPM`/`#SPR`.

---

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

---
# (merged reference) researcher-61 findings (entropy_sticky accuracy-preserving lever, safe frontier tau~1.098 -> ~1.57x @ 96.7%):

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
