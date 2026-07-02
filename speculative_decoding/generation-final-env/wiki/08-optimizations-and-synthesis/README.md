# Optimizations & multi-lever synthesis

The branches that actually *build candidate schemes to beat the baseline* (reference ~1.63× vs vanilla
AR) — tuning γ, recalibrating the draft, reshaping drafts into trees, repairing draft support, and
composing multiple lossless levers — plus the factorial studies of how those levers interact. **This
group is the closest to the project's actual objective** (a novel training-free decoding scheme that is
faster than vanilla speculative decoding while holding 28/30 accuracy); speedups quoted below are each
branch's measured delta over its own local baseline.

**Cross-cutting result.** Three lossless levers emerge: **(A) draft-temperature** `DRAFT_TEMP≈2.25`
(flattens the over-confident draft, +~2% throughput), **(B) heap `make_dist`** (replaces the O(vocab)
sort, +~6%, orthogonal/multiplicative), and **(C) entropy_sticky** (force-accept low-entropy artifacts,
see [entropy theme](../03-entropy-and-uncertainty/README.md)). A and B compose multiplicatively (+8.3%);
adding C reaches ~1.64× but A and C are **substitutes** competing for the same `q>p` artifact pool
(subadditive, A removes ~37% of C's pool). Structural ideas mostly *fail* against the verify-bound wall:
**tree drafts** lose to depth-chains at the optimum (76% of chain breaks are support-misses no tree can
catch), and **direct support repair** lifted only 0.23% of accepts. γ tuning peaks at 3–4.

| File | Hook |
|---|---|
| [researcher-11](researcher-11.md) | Speculation-depth (γ) sweep: inverted-U throughput peak at γ=3–4 (151 tok/s, 1.50×), declining for γ≥6; constant per-token hazard p=0.877; round cost linear (11.6 ms + 3.3 ms/draft). |
| [researcher-34](researcher-34.md) | Direct online draft-entropy gate test: an in-process gate truncating low-yield rounds; per-round efficiency rises but throughput does not (verify-call overhead and timing variance eat the predicted gain). |
| [researcher-52](researcher-52.md) | Draft-only temperature recalibration: the MTP draft is ~3.3× too peaked; `DRAFT_TEMP=2.25` maximizes Σmin(p,q_τ) and lifts accept 72.4→74.8% and tok/s +2.3% at zero extra compute; over-flattening (τ=4) is sharply worse. |
| [researcher-57](researcher-57.md) | Branching (tree) drafts vs linear chains: tree-node cost = chain-node cost (no fan-out discount); MTP top-1 already covers ~86%, only 4–5pp branching headroom; 76% of breaks are support-misses; depth-chains win at the optimum. |
| [researcher-59](researcher-59.md) | Composes draft-temp + heap make_dist as one binary, 2×2 factorial: verify-bound facts reconfirmed; the two lossless gains combine end-to-end. |
| [researcher-60](researcher-60.md) | Direct support repair (mix copy support into the MTP proposal, α=0.05): diagnoses 109,928 outside-support tokens but only 429 (0.23% of accepts) come from repair; overall accept drops 72.4→71.0% — precision too low. |
| [researcher-63](researcher-63.md) | Adaptive draft-temperature policies do not beat the global optimum: qmax-based adaptation adds only +0.028pp over global τ=2.25; heap make_dist cuts CPU make_dist 17.1→8.6% of wall-clock. |
| [researcher-65](researcher-65.md) | Per-item concentration of the composed (draft-temp + make_dist) gains: heap speedup uniform (1.061–1.071× across domains); draft-temp gain heterogeneous; combined gains have a broad floor plus source-dependent accept lift. |
| [researcher-69](researcher-69.md) | Three-lever 2×2×2 factorial (draft_temp × heap × entropy_sticky): heap is orthogonal (+9.1%); draft_temp and entropy_sticky are **substitutes** (compete for the q>p pool); all-three ≈1.64× but A+C subadditive (−2pp). |
| [researcher-70](researcher-70.md) | 2×2 factorial of draft-temp × heap: perfectly multiplicative (+8.32% = 1.0203×1.0595); each lever invariant to the other; gains land on disjoint wall-time terms. |
