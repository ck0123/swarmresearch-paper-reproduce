# Acceptance rate & rejection-sampling mechanics

How the speculative accept/reject test behaves: the headline acceptance rate, how it
decomposes by draft depth and benchmark source, the distributional-overlap that bounds it,
and how the sampler's temperature reshapes it.

**Cross-cutting result.** The 72.4% headline acceptance is *draft utilization* (delivered/drafted);
the *conditional* per-token acceptance is ~87–88% and is essentially **flat across draft depth**
(no MTP error compounding). Acceptance is governed by the overlap between the truncated target `p`
and draft `q` distributions — empirical 87.7% ≈ theoretical overlap 87.6% — and the loss splits into
a small out-of-nucleus component (~2.3%) and a dominant draft-overconfidence component (~10.1%).

| File | Hook |
|---|---|
| [researcher-01](researcher-01.md) | Per-depth acceptance decay & the γ throughput ceiling: marginal conditional accept ~0.86–0.88 flat across depth; decay is geometric from a constant per-step hazard, not rising difficulty; ceiling ~6–7 tokens/round as γ→∞. |
| [researcher-02](researcher-02.md) | Acceptance & throughput by benchmark source: 72.4% accept, 1.49× speedup; per-source AIME 81.7% > LCB 75.6% > GPQA 68.6% > HLE 61.2%; per-item accept↔tok/s Pearson 0.99. |
| [researcher-03](researcher-03.md) | Acceptance resolved by depth × target predictability: 72.4% utilization vs 87.7% conditional; flat across depth; ~70% near-deterministic positions accept 98.4%, a 21% high-entropy tail drives 76% of rejections (argmax agreement is the divide). |
| [researcher-04](researcher-04.md) | Observational acceptance & round-outcome histograms: bimodal — 59.2% full-accept rounds vs 12.9% zero-accept; stable per-source ordering; 93.3% accuracy. |
| [researcher-05](researcher-05.md) | Conditional (per-position) acceptance within the window: flat 0.871–0.881; draft is more peaked than target (q≈0.95 vs p≈0.86); rejections are high-divergence events sharing only ~42% mass. |
| [researcher-09](researcher-09.md) | Phase profile + distributional-overlap decomposition: 87% conditional accept per reached position; overlap O=87.6% (theory) ≈ 87.7% (empirical). |
| [researcher-26](researcher-26.md) | Overlap decomposition + round-level path calibration: loss = 2.32% out-of-nucleus vs 10.12% overconfidence (4.4×); 59.2% all-accept rounds; sampled-path probability calibrates to outcomes; low-accept rounds hold most dead verify rows. |
| [researcher-29](researcher-29.md) | Per-position overlap + a realizable round-level predictor from draft entropy: 72.7% of positions have target entropy <0.25 and accept at 97.9%; draft overconfidence is the dominant limiter. |
| [researcher-42](researcher-42.md) | Sampler support-overlap lens: draft support ~97.7% contained in target support; target mass outside draft support ~9.5%; mean total-variation distance 0.124. |
| [researcher-45](researcher-45.md) | Temperature regime of the rejection sampler (temp ≤ 0 / "greedy"): the accept-coin and resampling still run at temp=1, so "greedy" mtp-spec is non-deterministic & non-lossless vs greedy target; target greedy collapses to 46.7% (loops to cap). |
| [researcher-54](researcher-54.md) | Acceptance = conditional overlap × γ-depth compounding: ~87–88% per-position × γ=4 → 72.4%; per-item accept↔overlap correlation 0.996; lowest-overlap item 54.6% vs highest 83.1%. |
