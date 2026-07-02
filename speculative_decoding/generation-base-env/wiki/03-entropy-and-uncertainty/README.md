# Target entropy & predictive-uncertainty structure

The largest theme. Where the target model's uncertainty lives, how it drives rejection, what the
high-entropy decision points concretely *are*, how much future content a single forward pass
foreshadows, and the "stochasticity tax" — the load-bearing diversity that lossless temp=1 sampling
must keep.

**Cross-cutting result.** Target verification entropy `Ht` is the master variable. ~56–70% of tokens
are near-deterministic "backbone"; uncertainty concentrates in natural-language words and at
word-onsets/fork points. `Ht` predicts rejection (AUC ~0.88), and the MTP draft's support coverage is
a *strict decreasing function* of `Ht` (top-1 ~98.5% at Ht<0.1 → ~26.5% at Ht≥1.5). So the high-entropy
joints are simultaneously where the draft cannot help (support-miss) **and** where the target must keep
diversity for quality — the same wall. ~38.9% of rejections are temp=1 "artifacts" (draft picked the
target's argmax but the lossless test still rejected); recovering them lifts accept/speedup but spends
accuracy on reasoning tasks.

| File | Hook |
|---|---|
| [researcher-25](researcher-25.md) | Content-resolved predictive uncertainty: 56.6% of tokens near-deterministic (H<0.01); words carry 68.6% of entropy mass on 39.6% of tokens; word-initial pieces 1.7× more uncertain; temp=1 still 89% argmax. |
| [researcher-27](researcher-27.md) | Linguistic content of drafted vs corrected tokens: 72.4% of corrections are same-category substitutions (word→word); errors are semantic substitutions at word/line boundaries (60% at boundaries), not spelling. |
| [researcher-31](researcher-31.md) | Confidence/entropy at divergence points: the draft is anti-calibrated in the uncertain regime (AUC 0.685); target entropy predicts rejection AUC 0.882; 38.9% of rejections are stochastic losses where the draft picked the target's own argmax. |
| [researcher-32](researcher-32.md) | Target uncertainty at verification, by content: accepted drafts mean entropy 0.190 (96.7% rank-0) vs rejected 1.026 (38.9% rank-0); high-uncertainty rejections are mostly words; same-category lexical error persists across all buckets. |
| [researcher-36](researcher-36.md) | Discarded verifier work behind stochastic rejections: 22% of verify rows computed then discarded; counterfactual argmax-rescue would recover ~12% of accepted draft positions. |
| [researcher-39](researcher-39.md) | Latent future-token foreshadowing: novel d=1 future tokens appear in top-40 at 43.1% vs 8.1% random (5.3× lift); horizon is short (decays by d≈10); content-independent but weak per-rank. |
| [researcher-44](researcher-44.md) | Foreshadowing (extended): how much multi-step future is latent in one forward pass; same ~5–8× top-40 lift across domains; separates novel (non-copyable) foreshadowing from n-gram/copy structure. |
| [researcher-41](researcher-41.md) | The "stochasticity tax": 38.9% of rejections are temp=1 artifacts; recovering them lifts accept 72.4→85% and speedup 1.49→1.65× but drops accuracy 93.3→83.3% — artifact rejections are load-bearing entropy. |
| [researcher-47](researcher-47.md) | Interior of the stochasticity-tax frontier: probabilistic recovery λ∈[0,1]; λ=0.5–0.75 improves throughput with no accuracy loss (93.3% at λ=0.5, 78.1% accept); sharp degradation only at λ>0.9, in long-gen code/math. |
| [researcher-51](researcher-51.md) | Uncertainty-conditioned foreshadowing (r39+r25): d=1 foreshadowing declines with present entropy (45%→34%) except at word onsets (~50% regardless); future info concentrates where present is determined. |
| [researcher-56](researcher-56.md) | Entropy-resolved decomposition of the stochasticity tax: entropy-gated mode recovery (Ht<1.0) keeps 93.3% accuracy and captures 54% of the gain (1.49→1.57×); collapse to 80% only above Ht=1.0. |
| [researcher-61](researcher-61.md) | Near-boundary frontier for entropy-gated recovery: safe frontier bracketed at 1.098 ≤ τ < 1.099 (93.3% held); a sharp phase boundary in entropy space, failures item-specific not domain-wide. |
| [researcher-62](researcher-62.md) | **Unifying synthesis — the MTP draft's support IS the target's entropy:** top-1 support coverage strictly decreases with Ht; backbone ≈70% of tokens; speculative speedup = mean backbone-run-length between joints. |
| [researcher-64](researcher-64.md) | Sequential dynamics of the latent: joints are temporally **clustered** (lift 3.53× at lag-1, bursts of 10–20 tokens); prior-round trailing entropy predicts next-round acceptance; verify-bound wall still caps adaptive payoff. |
| [researcher-66](researcher-66.md) | What the high-entropy positions concretely are: 77.6% are pivotal decision points (discourse/content/numeric), 51.8% tight binary choices (not diffuse clouds); rejection rate rises with fork type (discourse forks 41.5%). |
| [researcher-72](researcher-72.md) | Acceptance-rate tax at the concrete decision points: H≥0.5 slots carry 82.5% of expected loss on 26.9% of slots; pivotal forks carry 65.1% of the tax; concrete forks like `Let|Wait` dominate; tax multiplicative with depth. |
