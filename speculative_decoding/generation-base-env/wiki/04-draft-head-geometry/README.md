# MTP draft-head representational geometry

What the hidden-state vectors flowing through the MTP feedback loop look like, as opposed to *how
often* drafts are accepted. The MTP head is fed back its own hidden state across draft depths; these
branches measure that carrier's geometry relative to the target's latent.

**Cross-cutting result.** The MTP hidden state is nearly **orthogonal** to the target's hidden state
(mean cosine ≈ −0.03 to −0.08), yet it evolves coherently on its own manifold (~0.56–0.59 self-cosine)
and a held-out **linear map recovers ~23%** of the target's centered hidden variance (rank 256). Only the
hidden's *norm*, not its direction, separates accepted from rejected steps, and the recoverable component
is stable across feedback depth — but **degrades with target uncertainty** (R² 0.25→0.04 from low to high
entropy), tying this theme back to entropy structure.

| File | Hook |
|---|---|
| [researcher-15](researcher-15.md) | Representational geometry of the MTP feedback loop: MTP hidden ≈ orthogonal to target (cos −0.076) but coherent on its own manifold (~0.56); only norm (not direction) separates accept/reject; no drift with depth. |
| [researcher-16](researcher-16.md) | Hidden-state handoff geometry (263K comparisons): MTP-vs-target predictor-row cosine ≈ 0 (−0.031), relative L2 1.56; target's own consecutive states far more coherent (cos +0.594); relationship is directional, not scale. |
| [researcher-17](researcher-17.md) | Latent-state carrier geometry across depths 1–4: self-cosine −0.076 stable with depth; a linear map recovers ~23% of centered target variance; MTP output norm ~17% larger, stable over generation length. |
| [researcher-18](researcher-18.md) | Held-out linear maps (rank 64–256) MTP→target: despite raw orthogonality, PCA-ridge reaches R²=+0.23 (rank 256); rejected steps keep larger residuals; recoverable component stable across feedback depth. |
| [researcher-67](researcher-67.md) | Geometry conditioned on target uncertainty: raw directional alignment invariant across uncertainty bands (~−0.075), but magnitudes/residuals scale with uncertainty; linear recoverability degrades R² 0.248→0.038 low→high entropy. |
