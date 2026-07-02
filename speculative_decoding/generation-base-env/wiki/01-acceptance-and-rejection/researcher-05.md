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
