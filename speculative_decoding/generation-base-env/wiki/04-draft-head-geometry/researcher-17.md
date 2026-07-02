# Findings — latent-state carrier geometry in MTP speculative decoding (researcher-17)

## Lens
Orthogonal to acceptance/timing/length analyses: examine the **hidden-state vector that is
the literal memory/state carrier between target and draft**. In `mtp-spec`, the MTP head is
not fed tokens alone — at each draft step it consumes a `dbatch.embd` hidden vector
(`n_embd = 2816`). At draft depth 0 this is the target's true `nextn` latent
(`llama_get_embeddings_nextn_ith(ctx_tgt, ...)`, the vector `h_last`); at depths ≥1 the MTP
recursively re-feeds *its own* produced hidden (`llama_get_embeddings_nextn_ith(ctx_dft, 0)`).
During verification the target recomputes the *true* latent at each drafted position, so the
MTP's self-produced hidden at depth `k-1` can be compared directly to the target's true latent
at verification row `k`.

Instrumentation added to `initial_program.cpp` (mtp-spec branch): per draft round/depth it
dumps cosine, relative-L2, vector norms, max-abs, and a control (`cos` of consecutive true
target latents) to `results/hidden_dump_pid*.txt`; plus raw vectors for the first rounds to
`results/hidden_raw_pid*.txt`. Diagnostics do not alter decoding output.

## Setup / correctness
- Full 30-item eval (GPUs 2,3) with instrumentation: accuracy 93.3%, 146.8 tok/s, accept 72.4%,
  1.46× — matches the reference table; diagnostics are non-perturbing.
- Alignment sanity (round 0, depth 0): the `h_last` that seeds drafting vs the target's recomputed
  row-0 latent → cos = **0.996**, rel-L2 = 0.07 (n=30). Comparison pipeline is sound.
- Data: 329,150 dump rows; 263,320 depth≥1 latent comparisons across all 30 prompts / 4 domains.

## Result 1 — MTP self-hidden is raw-orthogonal to the target's true latent
The hidden the MTP produces and recursively re-feeds is essentially uncorrelated (slightly
anti-aligned) with the target's true latent at the same position, at every draft depth:

| draft depth | n      | cos(MTP, target_true) | control cos(target_t, target_{t-1}) | rel-L2 |
|-------------|--------|-----------------------|-------------------------------------|--------|
| 1           | 65,830 | −0.088                | 0.599                               | 1.59   |
| 2           | 65,830 | −0.076                | 0.593                               | 1.63   |
| 3           | 65,830 | −0.071                | 0.589                               | 1.61   |
| 4           | 65,830 | −0.069                | 0.587                               | 1.60   |

- Overall (depth≥1): cos mean = −0.076, p5 = −0.161, **p95 = 0.011**; 29.2% of comparisons have
  |cos| < 0.05. The orthogonality holds even at depth 1, where the MTP was fed the *exact* target
  latent — so its *output* latent does not reconstruct the target's next-position latent.
- The control shows the target's own latent trajectory is strongly self-correlated
  (cos ≈ 0.59, stable across depth), so the ~0 reading is not a high-variance / random-direction
  artifact of the space.

## Result 2 — not an affine offset; the two latents are linearly related in different bases
Tested on 120 raw (MTP, target) vector pairs (dim 2816):
- Raw paired cos = −0.031.
- Centering by each space's own global mean → cos = **+0.020** (still orthogonal): the mismatch is
  **not** a constant/mean offset. The two space-means are themselves anti-aligned (cos = −0.135).
- A single fixed least-squares linear map MTP→target, fit on half and evaluated on the held-out
  half, recovers **cos = 0.50** (≈ the 0.59 intrinsic position-to-position correlation).
- Conclusion: the MTP head emits a **linear re-encoding** of the target latent in its own basis,
  not the target latent itself; the spec loop feeds this re-encoded vector straight back as
  `dbatch.embd` and remains self-consistent in MTP-space while never re-anchoring to target-space.

## Result 3 — massive-activation structure and norm stability
- Both latents carry a dominant "massive-activation" dimension. Per-sample max|coord|/‖vector‖ ≈
  0.23 (target latent), stable across the whole generation. In the mean vectors, the single top
  dimension holds ~9.8% of energy for the target latent vs ~5.0% for the MTP output — the target
  concentrates more energy in outlier dims.
- MTP-output norm is ~17% larger than the target latent it stands in for (mean ratio 1.173;
  ‖MTP‖≈174.6 vs ‖target‖≈156.9).
- Carrier (target latent) norm is **stable over generation length** — binned by round index it
  stays ≈162–183 from the first rounds out to 400+ rounds (overall mean 163, p5 108, p95 239);
  no runaway norm growth and the dominant-dim fraction stays ~0.23. The state passed between
  models does not drift in magnitude over long decodes.
