# Findings — adaptive draft-temperature policies do not materially beat the global optimum

## Extension lens (this section)
The prior section found a single lossless proposal-temperature correction: `DRAFT_TEMP=2.25`
maximized the predicted overlap `Σ min(p, q_tau)` and raised mtp-spec throughput. This section
tests whether the same knob has remaining headroom when made conditional rather than global,
using only draft-side information available before proposal sampling.

## Instrumentation
- `DRAFT_TEMP_BY_G`: optional comma-list of proposal temperatures by speculative-window slot
  `g`; the last value repeats if `gamma` is longer than the list.
- `DRAFT_TEMP_QMAX_EDGES` + `DRAFT_TEMP_QMAX_TEMPS`: optional confidence-conditional proposal
  policy. The condition is the draft model's pre-top-p, top-k, temp-1 top-token probability
  before the proposal is sampled. Edge bins are `<= edge`, and `temps` must have `edges+1`
  values. This policy overrides `DRAFT_TEMP_BY_G` when present.
- `DRAFT_TEMP_QMAX_BINS`: diagnostic bins for the same pre-top-p qmax statistic. Combined with
  `DRAFT_TEMP_GRID`, it emits `#SPQ` lines with tested-weighted predicted acceptance for every
  candidate temperature in every confidence bin.
- The actual proposal distribution used for sampling, the `min(1,p/q)` acceptance test, and the
  `(p-q)+` residual correction remain identical within each conditional branch, so the sampler
  is still lossless for any policy.

## Predictor result: slot and confidence conditioning
- Full 30-item diagnostic run, GPUs 4,5, γ=4, seed 0, n=16384, actual proposal τ=1.0, candidate
  grid `1.0,1.25,1.5,1.75,2.0,2.25,2.5,2.75,3.0,3.5,4.0`.
- Slot-wise conditioning had no headroom: every slot `g=0..3` had the same predicted optimum
  τ=2.25. Tested-weighted predicted accept stayed `0.887370`, identical to the global optimum
  at the precision measured here.
- Top-p-truncated qmax was a poor conditioning statistic: 189,932 / 217,630 tested positions
  fell into the highest bin because top-p truncation often collapsed the draft distribution to
  one token.
- Pre-top-p qmax exposed more structure, but the predicted gain remained very small:

| pre-top-p qmax bin | tested | best τ | best pred. accept | pred. at τ=2.25 |
|---|---:|---:|---:|---:|
| ≤0.50 | 2,205 | 1.75 | .47702 | .46828 |
| 0.50–0.70 | 8,089 | 2.00 | .60050 | .59833 |
| 0.70–0.85 | 7,845 | 2.25 | .61291 | .61291 |
| 0.85–0.93 | 6,895 | 2.25 | .63011 | .63011 |
| 0.93–0.97 | 6,599 | 2.25 | .64376 | .64376 |
| 0.97–0.99 | 7,675 | 2.50 | .67020 | .66946 |
| 0.99–0.995 | 4,594 | 2.50 | .69983 | .69965 |
| 0.995–0.999 | 9,856 | 2.50 | .73625 | .73553 |
| >0.999 | 163,872 | 1.00 | .96572 | .96565 |

The bin-best policy predicted tested-weighted conditional accept `0.887655`, only +0.000285
absolute (+0.028 percentage points) above the global τ=2.25 prediction `0.887370`.

## End-to-end measurement on GPUs 4,5
The selected adaptive policy used qmax edges
`0.50,0.70,0.85,0.93,0.97,0.99,0.995,0.999` and bin temperatures
`1.75,2.0,2.25,2.25,2.25,2.5,2.5,2.5,1.0`.

| policy | accuracy | mean tok/s | aggregate tok/s | accept_rate | cond_acc | emitted/round | mean_resid | fallback |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| τ=1.0 diagnostic baseline | 93.3% | 145.2 | 145.8 | 72.4% | .87655 | 3.898 | .5844 | 0 |
| global τ=2.25 control | 90.0% | 148.5 | 149.6 | 74.8% | .88729 | 3.993 | .5085 | 0 |
| qmax-adaptive policy | 93.3% | 148.5 | 150.5 | 75.5% | .89077 | 4.019 | .5033 | 0 |

## Facts
1. **The implementable adaptive policy did not improve the evaluator's headline mean tok/s**
   over the same-GPU global τ=2.25 control: 148.514 → 148.522 tok/s (+0.005%). Aggregate
   token/time improved 149.55 → 150.51 tok/s (+0.64%), matching the small emitted/round lift.
2. **Adaptive confidence conditioning raised acceptance counters more than the one-trajectory
   predictor suggested**, likely because the changed proposal followed a different token
   trajectory: accept_rate 74.8%→75.5%, cond_acc .88729→.89077, emitted/round 3.993→4.019.
3. **The global scalar was already nearly saturated.** The best pre-top-p-qmax bin policy
   predicted only +0.028pp conditional accept over τ=2.25, and the full end-to-end run converted
   higher acceptance into no measurable mean-tok/s gain on the benchmark's reported metric.
4. **The conditional policy remained lossless under the rejection sampler.** The residual had
   positive support throughout (`fallback=0`), and accuracy stayed in the same stochastic band
   as the target/sample runs (global/adaptive: 90.0%/93.3%).
5. **Most tested draft positions are extremely high-confidence before top-p truncation.**
   The `qmax>0.999` bin contained 163,872 / 217,630 tested positions (75.3%), and its predicted
   acceptance was essentially flat from τ=1.0 through τ=2.25 (.96572 vs .96565).

## Artifacts (this section)
- `initial_program.cpp`: adds `DRAFT_TEMP_BY_G`, `DRAFT_TEMP_QMAX_EDGES`,
  `DRAFT_TEMP_QMAX_TEMPS`, `DRAFT_TEMP_QMAX_BINS`, pre-top-p qmax diagnostics, and
  per-slot `mean_qmax1` / `mean_used_temp` in `#SPP`.
- `results/adaptive_temp/baseline_grid/`: first full qmax-bin diagnostic using top-p-truncated
  qmax, showing the collapse of that statistic.
- `results/adaptive_temp/pretop_qmax_grid/`: full pre-top-p qmax diagnostic and scored summary.
- `results/adaptive_temp/global_2p25/`: same-GPU global τ=2.25 end-to-end control.
- `results/adaptive_temp/qmax_policy/`: selected qmax-adaptive policy end-to-end run.

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
