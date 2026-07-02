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
