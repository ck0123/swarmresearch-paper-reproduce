# Wall-clock latency & cost decomposition

Where the *denominator* of speedup goes. These branches profile per-round wall-clock, resolve an
early contradiction about the bottleneck, and reverse-engineer the GPU forward-pass cost structure.

**Cross-cutting result.** The system is **verify-bound**, not sampling-bound. With an explicit GPU
sync after each `llama_decode`, the target verify forward pass is ~69% of wall-clock, MTP draft ~17.6%,
and CPU sampling ~12.6%. An early claim that host-side `make_dist` was ~95% of time was an **async
artifact** — GPU compute was being billed to the following sort (see the r33/r46/r50 syntheses).
Verify cost is **sub-linear but not free**: ≈7.0 ms fixed + 2.07 ms/token, set by MoE expert fan-out.
This caps the absolute speedup ceiling at ≈2.1× even with zero-cost drafting; ~22% of computed verify
rows are discarded after the first rejection.

| File | Hook |
|---|---|
| [researcher-06](researcher-06.md) | Phase-level profile: MTP drafting 69% of draft+verify time vs target verify 31%; 22% of verify rows discarded after first rejection; round cost nearly fixed (25–27 ms), amortized over delivered tokens. |
| [researcher-07](researcher-07.md) | Wall-clock decomposition with explicit GPU sync: verify_fwd 69.4%, draft_fwd 17.6%, CPU sampling 12.6%; linear model 7.0 ms + 2.07 ms/token; verify-bound ceiling ~2.1×. |
| [researcher-08](researcher-08.md) | Wall-clock latency decomposition (the canonical verify-bound reference): verify forward dominates; the 262K-vocab sort is *not* the bottleneck. |
| [researcher-24](researcher-24.md) | Per-round decomposition over 30 prompts: verify_fwd 69.4%, draft_fwd 17.6%, sampling 12.6%, prefill/cache 0.4%; sub-linear verify; zero-cost-draft ceiling ~1.44×. |
| [researcher-30](researcher-30.md) | Entropy-driven rejection → discarded verify rows, *not* cheaper rounds: verify ~18.15 ms whether a round yields 1 or 5 tokens; entropy controls the numerator (tokens/verify), not the denominator. |
| [researcher-33](researcher-33.md) | Per-op decode costs: GPU forward 87.1% of wall-clock; verify fixed 161 µs + 37 µs/row; 5-row batch = 1.96× a single token due to MoE expert activation; ceiling ~2.1×. |
| [researcher-37](researcher-37.md) | Mechanistic forward-cost split: LM head is 57% of MTP draft but pure fixed cost (one-time 782 MB unembedding read, ~71 µs/token); marginal verify token (~2 ms) is bandwidth-bound MoE fan-out, diversity-dependent. |
| [researcher-40](researcher-40.md) | Wall-clock decomposition: verify 69.4%, draft 17.6%, CPU 12.8%; vocab sort not the bottleneck; verify batch sub-linear (7 ms + 2.07 ms/token) from MoE fan-out. |
| [researcher-46](researcher-46.md) | **Synthesis of two contradicting wall-clock analyses:** resolves "sampling-bound (95% make_dist)" vs "verify-bound (69%)" — the former was an async `llama_decode` artifact; truth is verify-bound. |
| [researcher-50](researcher-50.md) | Same synthesis reproduced from one run via dual attribution + a practical CPU top-k fix: bounded heap cuts CPU distribution time 54.6% (311.8s→141.5s), lazy target-dist a further 15.4%. |
| [researcher-68](researcher-68.md) | Wall-clock decomposition + γ-sweep cost curve (7.0 ms + 2.07 ms/token); disproves the sampling-bound hypothesis; resolution of the r8-vs-r29 dispute. |
| [researcher-71](researcher-71.md) | Decomposition + extensions: request-batching amortizes fixed cost ~10×; depth and batching substitute on fixed cost but differ on MoE fan-out (cross 398 vs within 121 µs/tok); γ=4 break-even at B≈2–3 requests. |
| [researcher-73](researcher-73.md) | Decomposition + serving-regime: adds the `entropy_sticky` gate (τ≤1.098) force-accepting low-entropy backbone artifacts; safe frontier 1.098≤τ<1.099 keeps 93.3% accuracy at 1.57× / 78.8% accept. |
