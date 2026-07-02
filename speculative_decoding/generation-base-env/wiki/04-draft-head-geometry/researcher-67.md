# Findings — researcher-15

## Lens: representational geometry of the MTP draft head's hidden-state feedback loop

Orthogonal to acceptance-rate / timing / length analyses. Instead of asking *how often*
drafts are accepted, this asks *what the vectors flowing through the MTP feedback loop look
like* and how they relate to the target model's own hidden (residual-stream) states.

In `mtp-spec`, the MTP head drafts autoregressively by feeding its own predicted hidden state
(`llama_get_embeddings_nextn_ith`) back as the next input (`dbatch.embd`). Step `g=0` is seeded
by the target's TRUE hidden for `id_last`; steps `g>=1` are self-fed by the MTP's own output.

### Method
Instrumented `initial_program.cpp` (mtp-spec only, gated on env `SPEC_DUMP`; absent => normal
run, no perturbation). Per draft step within every round, dumped:
- `d_g`  = MTP-predicted hidden for `draft[g]` (the vector fed back).
- `t_{g+1}` = target's TRUE hidden for the same token/position (from the verify decode).
- `t_g`, the input hidden `ing` (= seed at g=0, else `d_{g-1}`), and norms.
- Derived: `cos_dt = cos(d_g, t_{g+1})`, `l2_dt`, `cos_tt = cos(t_g, t_{g+1})`,
  `cos_in = cos(d_g, ing)`, all norms, plus whether the step was accepted.

Data: full 30-item eval, GPUs 6,7, gamma=4 => 263,320 draft-step records.
Eval result with instrumentation: accuracy 93.3%, accept 72.4%, 1.48x (matches reference;
instrumentation is non-perturbing). Analysis scripts: `analyze_geom.py`.

### Result 1 — the MTP hidden lives in a basis ~orthogonal (slightly anti-aligned) to the target's
`cos_dt` (MTP hidden vs target-true hidden at the SAME position): mean **-0.076**, sd 0.052.
- 92.7% of all draft steps have `cos_dt < 0`; 68% lie within |0.10| of zero; **none exceed 0.30.**
- This is not a scale artifact: norms are comparable (`d_norm` mean 174.7 vs `t_norm` 156.9,
  ratio 1.17). The mismatch is directional.
- Baseline for "what aligned looks like": the target's OWN consecutive hidden states have
  `cos_tt` mean **+0.592**. So the target manifold is moderately self-coherent, yet the MTP
  hidden is essentially orthogonal to it.
The speculative system is lossless and accepts 72% of drafts even though the draft head emits a
hidden vector that does not approximate the target's residual stream — the token signal is in the
MTP logits, but the fed-back hidden is a parallel, differently-oriented representation.

### Result 2 — the MTP evolves on its OWN self-consistent manifold at target-like speed
`cos_in` (MTP output vs its own input):
- g=0 (input = target-true seed): **-0.094** — same orthogonal relation as `cos_dt`. Crossing
  from target-space into MTP-space is a near-orthogonal rotation.
- g>=1 (input = MTP's own prior output): **+0.556 / +0.564 / +0.554** — flat.
Once inside its own feedback regime, the MTP's step-to-step coherence (~0.56) matches the
target's true-manifold coherence (`cos_tt` ~0.59). The head maps the target hidden into its own
latent manifold (one orthogonal rotation at the seam) and then advances within that manifold with
target-like temporal coherence.

### Result 3 — no representational drift / collapse with feedback depth
Across g=0..3 the d-vs-target geometry is stationary:
- `cos_dt`: -0.088, -0.076, -0.071, -0.069 (essentially flat).
- `l2_dt` : 245, 248, 245, 243 (flat). `cos_in` (g>=1) flat at ~0.56.
The per-step accepted fraction falls with depth (0.871, 0.763, 0.672, 0.592), but this decline is
NOT mirrored by any growth in hidden-state divergence from the target manifold. In representation
geometry the feedback loop does not wander further from the target with depth and does not
collapse toward a fixed point — the cross-manifold relationship is fixed at the seam and stable.

### Result 4 — only the MTP hidden's NORM separates accepted from rejected steps
Within each depth (removing the g=0 seam confound), accepted vs rejected draft steps:
- `d_norm`: accepted consistently LOWER, e.g. g0 169.6 vs 182.6, g3 170.8 vs 178.1 (gap ~7-13 at
  every depth).
- `l2_dt` : accepted CLOSER to target, e.g. g0 241 vs 270, g3 234 vs 256 — but since `cos_dt` is
  identical for accepted vs rejected, this l2 gap is driven by the norm difference, not direction.
- `cos_dt`: identical accepted vs rejected at every depth (directional alignment to the target
  hidden carries no accept/reject signal).
- `cos_in`: identical within depth (0.556 vs 0.561); the pooled accept/reject `cos_in` gap was
  entirely a depth confound.
Summary: among the measured geometric quantities, the magnitude (norm) of the MTP hidden vector is
the only one that distinguishes accepted from rejected draft steps; its direction relative to the
target manifold does not.

## Extension — researcher-18: held-out linear map between the MTP and target hidden manifolds

Follow-up to the same representation-geometry lens. The prior scalar dump established that
`d_g` is nearly orthogonal to the target hidden `t_{g+1}` in the raw coordinate basis. This
extension asks whether that orthogonality hides a stable linear relationship between the two
spaces.

### Method
Added a second gated instrumentation path in `initial_program.cpp`, active only when
`SPEC_PAIR_DUMP` is set. It samples full float32 vector pairs into binary files:
- `d_g` = MTP-predicted hidden after each draft step.
- `t_{g+1}` = target-true hidden for the same drafted token/position.
- metadata: prompt-local index, round, depth `g`, accepted count, step outcome, norms, raw
  `cos_dt`, and `l2_dt`.

The sample stride was set to 67 so it does not phase-lock with `gamma=4`. Full 30-item eval on
GPUs 6,7 produced 3,931 sampled vector pairs, balanced across depths:
`g=0: 983`, `g=1: 982`, `g=2: 983`, `g=3: 983`. Eval result with pair-vector sampling:
accuracy 93.3%, accept 72.4%, 146.6 tok/s, 1.46x.

Analysis script: `analyze_pair_map.py`. It holds out prompt-local indices where `p % 3 == 0`,
fits PCA-ridge linear maps from sampled `d_g` to `t_{g+1}` on the remaining records, and evaluates
on held-out records. Metrics are reported both in raw cosine and centered target-hidden space.

### Result 5 — raw orthogonality coexists with a partial held-out linear map
On the balanced sample, raw MTP-vs-target cosine remains near the previous full scalar result:
`cos_dt` mean -0.0769, sd 0.0519. On held-out records, unmapped `d_g` vs `t_{g+1}` had mean
cosine -0.0765.

Despite this, a low-rank linear map from `d_g` to `t_{g+1}` recovered a reproducible part of the
target hidden state on held-out prompts:
- target-mean baseline: raw cosine +0.6656, centered cosine 0.0000, R2 0.0000.
- scalar/depth baseline (`d_norm` plus depth one-hot): raw cosine +0.6692, centered cosine
  +0.0917, R2 +0.0172.
- PCA-ridge map, rank 64: raw cosine +0.7384, centered cosine +0.4608, R2 +0.2146.
- PCA-ridge map, rank 256: raw cosine +0.7457, centered cosine +0.4876, R2 +0.2341.

The raw coordinate systems are therefore nearly orthogonal, but the MTP hidden contains a
nontrivial linearly recoverable component of the target hidden manifold. This linear component is
much larger than what is explained by the scalar norm/depth features alone, but it is still
partial: about 23% of centered target-hidden variance in this sampled held-out test.

### Result 6 — the linearly recoverable component is stable across feedback depth
Separate depth-specific held-out maps gave similar R2 values:
- `g=0`: best reported rank 64 R2 +0.1744.
- `g=1`: best reported rank 64 R2 +0.1958.
- `g=2`: best reported rank 64 R2 +0.1756.
- `g=3`: best reported rank 64 R2 +0.1833.

The MTP-to-target linear relation is not only a `g=0` seed-transition artifact. It remains present
after the draft head is self-feeding its own hidden states.

### Result 7 — after linear mapping, rejected steps still have larger target-space residuals
Using the best all-depth map (rank 256), held-out accepted and rejected steps had similar mapped
raw cosine:
- accepted: mean +0.7443.
- rejected: mean +0.7487.

Centered cosine was lower for rejected steps:
- accepted: mean +0.5035.
- rejected: mean +0.4530.

Mapped residual norm separated outcomes more clearly:
- accepted residual: mean 102.55.
- rejected residual: mean 118.70.

The prior scalar analysis found that raw directional `cos_dt` did not distinguish accepted from
rejected steps and that the visible `l2_dt` gap was largely norm-driven. In the mapped target
space, rejected steps still have larger residuals, so the sampled full vectors contain accept/reject
information beyond raw-basis cosine, although the mapped raw cosine itself remains weakly
separating.

## Extension — researcher-67: geometry conditioned on target local predictive uncertainty

Follow-up within the same representation-geometry lens. The prior results were aggregate over all
verified draft positions. This extension asks whether the MTP/target hidden relationship changes
with the target verifier's local predictive uncertainty at the exact distribution used to accept or
reject each drafted token.

### Method
Extended the existing gated instrumentation in `initial_program.cpp`:
- For every verified draft step, recorded target-side uncertainty statistics from the verifier
  distribution `p_g`: entropy, normalized entropy, max probability, top-1/top-2 margin, effective
  support size, target probability of the drafted token, draft probability of the same token, and
  the speculative acceptance probability.
- The fields are written into both `SPEC_DUMP` scalar records and `SPEC_PAIR_DUMP` sampled
  vector-pair metadata.
- When instrumentation is active, target verifier distributions are precomputed for all `G`
  drafted positions before the accept/reject loop, so draft steps after the first rejection also
  receive uncertainty labels. The normal non-dump path keeps the original acceptance codepath.

Data: full 30-item eval on GPUs 0,1 with
`SPEC_DUMP=results/hdump SPEC_PAIR_DUMP=results/pairdumps ./evaluator/task-eval --gpus 0,1`.
Result: accuracy 93.3%, accept 72.4%, 147.0 tok/s, 1.46x. The run produced 263,320 scalar
draft-step records and 3,931 sampled full-vector pairs. Analysis script:
`analyze_uncertainty_geometry.py`.

Because 65.4% of target verifier distributions have exactly zero entropy after the active
top-k/top-p truncation, uncertainty bands were defined as:
- `zero`: entropy 0.0.
- `low+`, `mid+`, `high+`: tertiles over positive-entropy positions, with cutpoints 0.5402 and
  0.9400 nats in the full scalar dump.

### Result 8 — raw MTP-vs-target direction is almost invariant to target uncertainty
Across uncertainty bands, raw `cos_dt` stays near the same orthogonal/weakly anti-aligned value:
- `zero`: -0.0751.
- `low+`: -0.0789.
- `mid+`: -0.0791.
- `high+`: -0.0746.

This remains true after depth control. At `g=3`, for example, `cos_dt` is -0.0687, -0.0700,
-0.0698, -0.0659 for `zero`, `low+`, `mid+`, `high+`. Target uncertainty therefore does not
expose a hidden directional alignment regime in the raw coordinate basis.

### Result 9 — uncertainty strongly tracks MTP hidden norm, absolute mismatch, and acceptance
The scalar geometric quantities that are sensitive to uncertainty are magnitudes and residuals,
not raw direction:
- acceptance fraction: `zero` 0.865, `low+` 0.600, `mid+` 0.456, `high+` 0.317.
- `d_norm`: `zero` 170.94, `low+` 176.77, `mid+` 180.01, `high+` 188.24.
- `l2_dt`: `zero` 234.71, `low+` 255.24, `mid+` 261.25, `high+` 277.56.
- target probability of the drafted token: `zero` 0.9798, `low+` 0.7581, `mid+` 0.5387,
  `high+` 0.3379.

The same ordering persists within each feedback depth. At `g=0`, `d_norm` rises from 165.91
(`zero`) to 187.46 (`high+`); at `g=3`, it rises from 170.27 to 187.63. The prior aggregate
accepted/rejected norm gap is therefore partly a target-uncertainty effect: high-uncertainty
verifier positions are where the MTP hidden becomes larger and farther from the target hidden in
absolute distance.

### Result 10 — the held-out MTP-to-target linear map degrades with target uncertainty
Using the same held-out prompt split as the prior pair-map analysis, depth-pooled PCA-ridge maps
fit separately inside each uncertainty band gave:
- `zero`: train 1,883 / test 681, best rank 128, centered cosine +0.4856, R2 +0.2489.
- `low+`: train 323 / test 133, best rank 32, centered cosine +0.3329, R2 +0.1078.
- `mid+`: train 288 / test 167, best rank 32, centered cosine +0.3040, R2 +0.0828.
- `high+`: train 317 / test 139, best rank 16, centered cosine +0.2125, R2 +0.0380.

The aggregate all-depth map's previously reported R2 of about +0.234 is therefore concentrated in
low-uncertainty / deterministic verifier positions. In the highest target-uncertainty positions,
the MTP hidden remains raw-orthogonal to the target hidden and also contains much less linearly
recoverable target-hidden information.

### Result 11 — a single global map hides uncertainty-dependent residual size
The best global rank-256 map still has similar mapped raw cosine across held-out bands, but its
target-space residual grows with uncertainty:
- `zero`: mapped raw cosine +0.7455, centered cosine +0.5158, residual 100.12, acceptance 0.858.
- `low+`: mapped raw cosine +0.7325, centered cosine +0.4360, residual 117.22, acceptance 0.556.
- `mid+`: mapped raw cosine +0.7520, centered cosine +0.4522, residual 114.59, acceptance 0.449.
- `high+`: mapped raw cosine +0.7517, centered cosine +0.4417, residual 126.89, acceptance 0.252.

Mapped raw cosine is not a reliable uncertainty-sensitive diagnostic because it remains dominated
by the target-hidden mean direction. Centered cosine, residual norm, and band-specific held-out R2
show the uncertainty dependence more clearly.
