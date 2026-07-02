# Sequential dynamics of the latent (researcher-64) — the joints are temporally CLUSTERED, a free target-side signal predicts them, yet the verify-bound wall caps the payoff

## Angle
The unified theory (r-62 and parents) reduces everything to one per-position latent — the target's
verification entropy `Ht` — and its slogan is "throughput = mean **backbone-run-length between
joints**." But every prior branch measured only the **marginal** of `Ht`: the per-band miss rate,
break rate, and recoverable-artifact pool. A run-length is a statement about the **sequential
arrangement** of joints along the stream, which the marginal cannot determine. This branch pushes the
theory to the concrete consequence it never established: **is the joint process memoryless (Bernoulli)
or temporally clustered (autocorrelated)?** — and the operational question that follows directly: if
joints cluster, the **just-verified** target entropy (a free, already-computed, target-side signal —
distinct from the draft-confidence gate r-31 fact-8 already refuted, AUC 0.685) should let an
**adaptive draft-length (γ)** policy beat fixed γ by drafting long on the backbone and short at joints.

## Method (one read-only side-channel + a lossless per-round γ knob; reference path byte-unchanged)
- **SEQENT** (gated by env `SEQENT_LOG`): per speculative round, log the ordered `Ht` of every
  emitted position (the `accepted` accepted-draft positions, then the trailing joint/bonus token),
  computed on the already-built `Dist` (no extra GPU/`make_dist`). Reconstructs the full per-position
  entropy time-series, letting offline `analyze_seqent.py` measure autocorrelation, run-lengths, and
  next-round-yield-vs-previous-trailing-`Ht`.
- **ADAPT_GAMMA="glo,ghi,thr"** (mtp-spec only): set THIS round's draft length from the PREVIOUS
  round's trailing `Ht` — `ghi` if prev `Ht<thr` (predicted backbone), else `glo` (predicted joint).
  **Lossless:** the rejection-sampling accept/reject rule is byte-for-byte unchanged; only *how many*
  tokens are speculated varies per round, so each variant is a valid sample from the target.
- Non-perturbation: full 30-item eval (GPUs 2,3, SEQENT on, fixed γ=4) = **93.3% accuracy (== target),
  72.4% accept, 1.49× (150.1 tok/s)** — identical to reference. 256,594 emitted positions / 65,830 rounds.

## Result D1 — the joint process is strongly CLUSTERED, not memoryless (rejects the Bernoulli null)
Per-position `Ht` time-series (joint := `Ht≥1` nat; marginal joint rate 8.3%, backbone `Ht<0.1` = 70.8%):

| lag | P(joint_t \| joint_{t−lag}) | P(joint_t \| ¬joint_{t−lag}) | lift | corr(Ht_t, Ht_{t−lag}) |
|---:|---:|---:|---:|---:|
| 1  | 0.242 | 0.069 | **3.53×** | 0.300 |
| 2  | 0.217 | 0.071 | 3.07× | 0.259 |
| 3  | 0.183 | 0.074 | 2.48× | 0.205 |
| 5  | 0.158 | 0.076 | 2.07× | 0.162 |
| 10 | 0.138 | 0.078 | 1.76× | 0.120 |
| 20 | 0.129 | 0.079 | 1.64× | 0.097 |

The memoryless null is lift=1 / corr=0 at every lag. Measured: a joint at `t−1` makes a joint at `t`
**3.53× more likely** than after a non-joint, decaying slowly and still **1.64×** at lag 20 — joints
arrive in **bursts spanning ~10–20 tokens**, not independently. (The sign is non-trivial: r-25's
onset→continuation alternation would give *negative* lag-1 entropy correlation; the observed *positive*
0.300 means turbulent stretches cross word boundaries — reasoning forks come in clusters.) The
backbone-run-length distribution (consecutive `Ht<0.1`) is correspondingly **over-dispersed vs
geometric**: mean 2.42 but heavy-tailed — P(run>10)=5.1%, P(run>20)=2.0%, P(run>50)=0.4%. The theory's
"backbone-run-length between joints" is real and **its variance, not just its mean, is large**.

## Result D2 — the previous round's trailing `Ht` is a strong, free predictor of next-round yield
Next round's outcome, conditioned on the band of the **previous** round's trailing `Ht` (the exact
signal ADAPT_GAMMA keys on), fixed γ=4 reference, 65,800 linked round-pairs:

| prev trailing `Ht` band | rounds | mean accepted (of 4) | P(full-accept) | P(break at depth ≤1) |
|---|---:|---:|---:|---:|
| [0,0.1)   | 32,392 | **3.374** | 74.6% | 12.3% |
| [0.1,0.3) |  2,804 | 2.744 | 53.7% | 27.1% |
| [0.3,0.5) |  4,029 | 2.743 | 53.2% | 27.2% |
| [0.5,0.7) |  7,793 | 2.733 | 52.4% | 27.0% |
| [0.7,1.0) |  5,957 | 2.402 | 43.4% | 35.8% |
| [1.0,1.5) |  7,988 | 2.204 | 37.2% | 40.1% |
| [1.5,+)   |  4,837 | **1.941** | 30.7% | 47.8% |
| **ALL**   | 65,800 | 2.897 | 59.2% | 23.7% |

Monotone across all 7 bands: a round that follows the **backbone** yields **3.37** accepted (74.6%
full), a round that follows a **high joint** yields **1.94** (30.7% full, ~48% break by depth 1). The
clustering of D1 is thus directly usable one round ahead — the entropy you already paid to compute is a
**1.7×-spread predictor of the next round's accepted-token yield**, with zero extra work and no use of
the (uninformative) draft confidence.

## Result D3 — yet adaptive γ does NOT beat the fixed optimum: the verify-bound wall caps it
Full 30-item evals, all on GPUs 2,3 in one session (so the ±~5% GPU-contention band is shared):

| policy | accept | tok/s | speedup | accuracy* |
|---|---:|---:|---:|---:|
| fixed γ=4 (reference) | 72.4% | 150.1 | 1.49× | 93.3% |
| fixed γ=8 | 53.9% | 142.4 | 1.41× | 96.7% |
| **adaptive γ 2↔8, thr Ht=0.5** | 65.0% | **151.1** | **1.50×** | 96.7% |
| adaptive γ 2↔4, thr Ht=0.7 | 76.3% | 143.7 | 1.43× | 86.7% |

Despite D2's large, monotone acceptance signal, **net throughput is flat** across the policies (all
within the noise band of the fixed γ=4 optimum). Two concrete facts:
- **Fixed deep drafting is penalized** (γ=8: 142.4 < 150.1) because draft cost is *linear* in γ
  (~1.9 ms/token, r-40) while accepted tokens saturate — the established γ≈2–4 optimum.
- **The adaptive 2↔8 policy recovers the ENTIRE fixed-γ=8 deficit** (142.4 → 151.1, +6.1%) and lands
  back at the γ=4 optimum, by spending the deep draft (≈60% of rounds, prev `Ht<0.5`) only where it
  pays — the backbone — and γ=2 at predicted joints. But it does **not surpass** the optimum (151.1 vs
  150.1 is within noise). Predictability of the latent buys **robustness to γ misspecification, not a
  higher ceiling.**
- **Cutting too aggressively underperforms** (2↔4 @0.7: 143.7 < 150.1): capping γ=2 after a predicted
  joint forfeits the upside that the D1 clustering *decays* within a round — a prev-joint round often
  recovers onto the backbone, and γ=2 truncates that recovery while still paying the ~7 ms fixed verify.

## Integrated account — predictability ≠ exploitability under the verify-bound wall
The unified theory said the latent `Ht` governs draft success and the wall is the high-entropy joints.
This branch establishes the **sequential** consequence the marginal could not: the joints are **strongly
autocorrelated** (D1) and therefore **one-round-ahead predictable from a free target-side signal** (D2,
yield 3.37→1.94). The natural operational corollary — adaptive γ should convert that prediction into
throughput — **fails to beat the fixed optimum** (D3). The reason is exactly the branch's verify-bound
invariant: throughput is set by **draft-cost + verify-width per accepted token**, and acceptance is *not*
the binding constraint at the optimum. Reallocating draft **depth** along the (now-predictable) entropy
trajectory moves throughput only inside the flat top of the verify-bound curve. The one realized benefit
is robustness: adaptive γ makes the throughput **insensitive to the γ setting** (it turns a mis-set γ=8
back into the γ=4 optimum). So the high-entropy joints are an even harder wall than the parents found —
they cap throughput even when their *arrival times are predictable in advance*: the speedup ceiling on
this benchmark is robust to optimal, oracle-of-the-recent-past draft-length allocation.

## Caveats
- **Lossless by construction**: only the per-round γ changes; the accept/reject rule and residual
  `p−q` are byte-unchanged. The accuracy spread (86.7–96.7% on 30 items) is sample-path noise — varying
  γ re-segments the RNG stream into a different valid sample — consistent with the lossless copy/hybrid
  modes (r-48) that also swung to 100%/96.7%; it is not a quality change.
- `Ht` is entropy over the truncated top-k→top-p support (r-31/r-56 convention), computed on the same
  `Dist` the verify loop already builds; SEQENT/ADAPT add no GPU work.
- D2/D3's per-round survival is read at the reference γ; the cost intuition uses r-40/r-44 constants
  (verify ≈ 7.0 + 2.07·(γ+1) ms, draft ≈ 1.9·γ ms). D3's tok/s are direct measurements; D1/D2 are
  direct measurements on the realized trajectory.
- Adaptive thr=0.5 drafts γ_hi on ≈60% of rounds (prev `Ht<0.5`), γ_lo on ≈40% (estimated from D2).

## Artifacts (researcher-64)
- `initial_program.cpp` — SEQENT per-round ordered-`Ht` side-channel (env `SEQENT_LOG`) + ADAPT_GAMMA
  realized-entropy-conditioned per-round draft length (env `ADAPT_GAMMA="glo,ghi,thr"`, mtp-spec),
  ADAPTSTAT footer. Reference/target/copy/hybrid paths byte-unchanged when both envs are unset.
- `analyze_seqent.py` — autocorrelation, run-length, and next-round-yield-by-prev-band analysis.
- `results/seqent_ref.{2,3}` (65,830 rounds), `results/task_eval.json`.

---

# Synthesis (researcher-62) — one axis unifies both parents: the MTP draft's support IS the target's entropy

## Angle
This branch merges two prior *synthesis* lines and integrates them on a single per-position question
neither could close because each measured only one of the two needed quantities:
- **Parent A (r-57, support-miss / tree line):** the acceptance ceiling is set by the MTP draft's
  **support**, not by verification geometry. ~10% of realized tokens fall outside the MTP's whole
  top-k/top-p support; **76.4% of chain breaks are support-misses** (target wants a token the MTP
  assigned ~0 probability), uncatchable by any tree. It measured the *rank of the realized target
  token within the draft q* but never the target's own uncertainty at those positions.
- **Parent B (r-56, entropy / diversity-ceiling line):** a single quantity — the target's
  **verification entropy `Ht`** (Shannon entropy over the top-k→top-p support; r-31's AUC-0.882
  rejection predictor) — governs everything: below ~1 nat is a self-determined, self-foreshadowed
  **backbone**; above it are the **high-entropy joints** where artifact rejections concentrate and
  the off-mode corrections are load-bearing diversity. It measured `Ht` per position but never the
  draft's support/rank there.

Integrated question: **are A's "support-misses" the same positions as B's "high-entropy joints"?**
I.e. is the MTP draft's support a function of the target's entropy — does the draft have support
exactly where the target is determined and lose it exactly where the target spreads?

## Method (joins both parents' instrumentation in one read-only pass; decoding path byte-unchanged)
Added `ENTCOV` to the active program: at every realized verification position already logged by
r-57's `TREE_STATS` (each accepted draft token, and the resampled token at the chain break), also
record the **target entropy band** of `p_g` (`dist_entropy`, 7 bands on edges
0.1/0.3/0.5/0.7/1.0/1.5) alongside the **rank of the realized target token within the MTP draft
`q_g`**. Cross-tabulates rank-coverage by `Ht` band, separately for ALL realized positions and for
the BREAK position only. No extra GPU/`make_dist` work (entropy is computed on the already-built
`Dist`; gated by `TREE_STATS`). Tooling: `analyze_entcov.py`.
Non-perturbation confirmed: full 30-item eval (GPUs 2,3, TREE_STATS on) = **93.3% accuracy
(== target), 72.4% accept, 1.49× (150.5 tok/s)** — identical to reference.

## Result U1 — the MTP draft's in-support coverage is a strict decreasing function of target entropy
Per `Ht` band over all 217,630 realized positions, P(realized target token ∈ MTP top-b | band):

| Ht band | N | share | top1 | top2 | top3 | in-support | MISS |
|---|---:|---:|---:|---:|---:|---:|---:|
| [0,0.1)   | 151,663 | 69.7% | 98.5% | 99.1% | 99.2% | 99.2% | 0.8% |
| [0.1,0.3) |   9,603 |  4.4% | 84.1% | 88.8% | 89.5% | 89.7% | 10.3% |
| [0.3,0.5) |  10,828 |  5.0% | 75.3% | 82.2% | 82.9% | 83.0% | 17.0% |
| [0.5,0.7) |  15,927 |  7.3% | 59.2% | 72.9% | 73.7% | 74.1% | 25.9% |
| [0.7,1.0) |  10,822 |  5.0% | 51.1% | 64.4% | 66.4% | 67.0% | 33.0% |
| [1.0,1.5) |  12,387 |  5.7% | 39.1% | 52.1% | 55.9% | 57.2% | 42.8% |
| [1.5,+)   |   6,400 |  2.9% | 26.5% | 38.0% | 42.9% | 45.8% | 54.2% |
| **ALL**   | 217,630 | 100%  | **86.0%** | 89.7% | 90.3% | **90.6%** | **9.4%** |

The ALL-row marginals (top-1 86.0%, in-support 90.6%, MISS 9.4%) reproduce r-57's independently
measured T2 ceiling exactly. Resolved by entropy: on the **backbone band (Ht<0.1), which is 69.7% of
all positions, the draft covers the realized token at top-1 98.5% and is in-support 99.2%** — a
near-oracle. Coverage then falls monotonically to top-1 26.5% / in-support 45.8% at Ht≥1.5.
(Caveat: an *accepted* draft is in q by construction, so the in-support column is partly mechanical;
the unconfounded cuts are the break table U2 and the per-band break rate U3.)

## Result U2 — the break diagnosis is entropy-stratified: high-entropy breaks are nearly tree-proof
At the 26,866 chain-break positions (the resampled token, marginally a draw from the target's true
`p_g`), rank within `q` by `Ht` band:

| Ht band | breaks | share | top1 | top2 | in-support | MISS |
|---|---:|---:|---:|---:|---:|---:|
| [0,0.1)   | 2,409 |  9.0% | 15.9% | 43.4% | 48.5% | 51.5% |
| [0.1,0.3) | 1,488 |  5.5% |  7.7% | 28.6% | 33.3% | 66.7% |
| [0.3,0.5) | 2,511 |  9.3% |  5.9% | 24.0% | 26.9% | 73.1% |
| [0.5,0.7) | 5,548 | 20.7% |  1.7% | 23.2% | 25.5% | 74.5% |
| [0.7,1.0) | 4,564 | 17.0% |  1.1% | 17.5% | 21.9% | 78.1% |
| [1.0,1.5) | 6,396 | 23.8% |  0.3% | 10.8% | 17.2% | 82.8% |
| [1.5,+)   | 3,950 | 14.7% |  0.0% |  5.4% | 12.2% | 87.8% |
| **ALL**   | 26,866 | 100% | **3.0%** | 18.8% | **23.6%** | **76.4%** |

ALL-row reproduces r-57's T3 exactly (top-1 3.0%, in-support 23.6%, MISS 76.4%). Resolved by
entropy: the **support-miss fraction of breaks rises monotonically 51.5% → 87.8% with `Ht`**, and the
width-2 tree rescue (top-2) collapses from 43.4% at Ht<0.1 to 5.4% at Ht≥1.5. The thin
"rank-misorder / unlucky-sample" band a tree could rescue is itself concentrated at LOW entropy
(where breaks are rare); at the high-entropy joints breaks are almost entirely support-misses.

## Result U3 — entropy predicts where the chain breaks (composition); 80% of misses are joints
Per-band break rate P(position is a chain break | `Ht` band) = break N / all N:

| Ht band | [0,0.1) | [0.1,0.3) | [0.3,0.5) | [0.5,0.7) | [0.7,1.0) | [1.0,1.5) | [1.5,+) |
|---|---:|---:|---:|---:|---:|---:|---:|
| break rate | 1.6% | 15.5% | 23.2% | 34.8% | 42.2% | 51.6% | 61.7% |

The chain breaks at **1.6%** of backbone positions and **61.7%** of the highest-entropy positions —
the per-position realization of r-31's AUC-0.882 (target entropy predicts rejection), now mechanized:
breaks rise with `Ht` *because* draft support falls with `Ht` (U1/U2). Of all 20,531 support-misses,
**80.2% sit at Ht≥0.5 and 42.7% at Ht≥1.0**; only 0.8% of backbone positions are misses.

## Integrated account — the MTP draft is a backbone-tracker; the joints are an unavoidable wall from both sides
A single latent variable, the target's verification entropy `Ht`, unifies both parent lines:
- **The draft has support exactly where the target is determined.** On the low-entropy backbone
  (~70% of tokens, Ht<0.1) the MTP head is a near-oracle (98.5% top-1); it cannot represent the
  multi-modal high-entropy distributions at the joints and there collapses to a single, usually
  wrong, mode — so the target's realized token falls outside its top-k/top-p support. r-57's
  "support-miss" and r-56's "high-entropy joint" are therefore **the same positions** (U1–U3).
- **This pins Parent A's acceptance ceiling to Parent B's axis.** A cleverer verification geometry
  (tree/rank reorder) can only touch the in-support sliver, which U2 shows is concentrated at low
  entropy where breaks are already rare; the 90.6% in-support marginal is an upper bound on
  per-position acceptance for any trick that keeps this draft's support. Raising it requires a draft
  with *wider support at high `Ht`* — i.e. a draft that is itself uncertain where the target is
  uncertain.
- **But that is the same diversity Parent B showed is load-bearing.** r-56 found mode-recovery is
  lossless below Ht~1.0 and degenerates accuracy above it, because the off-mode corrections at the
  joints keep the trajectory from looping. So the high-entropy joints are simultaneously (a) where
  the draft cannot raise throughput (support-miss forces a full target step) and (b) where the target
  must keep its true distribution for quality. A draft that perfectly covered the joints would *be*
  the target (no speedup). The verify-bound floor (Parent A) and the accuracy floor (Parent B) are
  the same wall — the high-entropy joints — and the target's forward there is irreducible on both
  counts. Speculative speedup on this benchmark is structurally a measure of mean backbone-run-length
  between joints: ~70% of tokens are near-free to draft, and throughput is capped by how often the
  stream hits a joint, where one flat-cost target verify forward yields ~1 token.

## Caveats
- `Ht` is entropy over the truncated top-k→top-p support (r-31/r-56 convention), computed on `p_g`
  at the same positions r-57's `TREE_STATS` already scored; the join is per-position exact.
- ALL-position in-support (U1) is partly mechanical (accepted draft ⊂ q by construction); the
  unconfounded measurements are the BREAK table (U2, resampled token ∉ q is possible) and the
  per-band break rate (U3). The top-1 ALL column is informative (not pinned to 100%).
- Coverage `c_b` is measured along the realized linear-chain path (the prior's depth-stationarity
  approximation for sibling/tree paths carries over). Eval reproduced reference metrics exactly.

---

# Branching (tree) drafts vs linear chains (researcher-57) — can spending verify-width on BREADTH beat DEPTH?

## Angle
Every draft structure this branch has tested is a **linear single chain** verified as one contiguous
sequence: the canonical MTP chain, the copy chain (copy-spec), and the MTP+copy hybrid. The branch's
central established invariant is that throughput is **verify-bound** — each round pays a target verify
forward of ~`7.0ms + 2.07ms·W` (W = batch width; the 2.07ms/token is MoE expert fan-out), *regardless
of how many drafts are accepted* — so the binding quantity is **verify-width-per-accepted-token**, and
cheapening the draft side (copy-spec made it free) bought ~0×.

This extension explores the one structurally different drafting arrangement the prior work never
touched: a **branching token tree** (SpecInfer/Medusa/EAGLE geometry). Instead of spending the
verify-width budget on *depth* (one γ-token chain), spend some on *breadth* — propose the top-b draft
candidates as siblings at a position, so a near-miss (target picks the draft's 2nd choice) still
continues the path. A tree is the canonical way to raise acceptance-per-width, so it is the sharpest
possible test of the branch's invariant. The cost/benefit question: a tree of W nodes costs the same
verify forward as a chain of W nodes — does branching buy enough extra accepted tokens to beat depth?

## Method (both sides measured; decoding path byte-for-byte unchanged)
- **COST (probe arm `tree_top2`):** built a width-W verify batch laid out as top-2 MTP siblings per
  spine position (`q_0[0],q_0[1],q_1[0],q_1[1],…`) and timed its forward vs the depth chain
  (`actual_mtp`), arbitrary `prompt_distinct`, and the degenerate `same_last`/`copy_cont`. Reports
  within-batch distinct-token count `ndist` (the prior's established MoE-fan-out cost driver).
- **BENEFIT (`TREESTATS`, read-only, env-gated, zero extra GPU/make_dist):** during normal mtp-spec
  decoding, for each realized target token at draft depth g, record its **rank within the MTP draft
  distribution q_g** (q_g.ids is prob-sorted, so rank is a ≤top_k scan). The realized token at an
  accepted depth is the accepted draft; at the chain's **break** it is the resampled token, which is
  marginally a sample from the target's true p_g — so its rank in q_g answers exactly *"is the target's
  actual next token among the MTP's top-b at the position the chain just died?"* Aggregated per depth
  over the full eval. Non-perturbation confirmed: full 30-item eval (GPUs 4,5, TREE_STATS on) =
  **93.3% accuracy (== target), 72.4% accept, 1.44× (144.6 tok/s; within the documented 144–151
  GPU-contention band).** Tooling: `analyze_tree.py`.

## Result T1 (COST) — a tree node costs the SAME as a chain node; branching gets NO fan-out discount
Probe, width-9 live batches, body-only k0 (MoE body, no LM head), avg over 6 real prompts:

| batch (width 9) | ndist | body_k0 (ms) | vs same_last |
|---|---:|---:|---:|
| same_last (1 token ×9) | 1 | 16.5 | — |
| copy_cont (collapsed repetition) | 3 | 16.6 | +0.1 |
| **tree_top2 (top-2 MTP siblings/depth)** | **7–8** | **~20.2** | **+3.7** |
| actual_mtp (depth chain) | 8 | ~20.3 | +3.8 |
| prompt_distinct | 9 | ~20.4 | +3.9 |

Top-2 candidates of one MTP next-token distribution are **genuinely distinct tokens** (ndist stays
7–8 of 9), so a branching batch routes to nearly the full expert union and pays the full ~3.7ms
fan-out tax — indistinguishable from a depth chain or arbitrary distinct tokens. The only cheap
batches remain the degenerate low-diversity ones (`same_last`, `copy_cont`). **A tree of W nodes
therefore costs the same `7.0 + 2.07·W` verify forward as a chain of W nodes** — the cost/benefit
reduces entirely to expected-accepted-tokens at equal width.

## Result T2 (BENEFIT CEILING) — the MTP draft is too sharp: branching headroom is only ~4 points
Per-depth coverage `c_b(g) = P(realized target token ∈ MTP top-b | reached depth g)`, full eval:

| depth | reached | top-1 | top-2 | top-3 | top-5 | in-support (any b) |
|---|---:|---:|---:|---:|---:|---:|
| d0 | 65,830 | 0.851 | 0.892 | 0.899 | 0.901 | **0.902** |
| d1 | 57,316 | 0.859 | 0.897 | 0.903 | 0.906 | **0.906** |
| d2 | 50,234 | 0.866 | 0.900 | 0.907 | 0.909 | **0.909** |
| d3 | 44,250 | 0.868 | 0.900 | 0.905 | 0.907 | **0.907** |

The MTP **top-1 alone already covers ~85–87%** of realized tokens. The entire branching headroom —
top-1 to the in-support ceiling (any branch width whatsoever) — is **only ~4–5 percentage points**
(0.86 → 0.90). **~10% of realized tokens fall outside the MTP's whole top-k/top-p support and are
uncatchable by ANY tree** built from this draft.

## Result T3 (BREAK DIAGNOSIS) — 76% of chain breaks are draft misses no tree can fix
At the depth where the linear chain dies, rank of the target's realized token within q at that
position (26,866 breaks):

| | top-1 | top-2 | top-3 | top-5 | in-support | **MISS (outside support)** |
|---|---:|---:|---:|---:|---:|---:|
| P(target token ∈ …) | 0.030 | 0.188 | 0.220 | 0.232 | 0.236 | **0.764** |

**76.4% of breaks happen because the target wants a token the MTP head assigned ~0 probability**
(outside its entire top-k/top-p). A width-2 node rescues only **18.8%** of breaks, width-3 **22.0%**,
and the absolute ceiling at infinite width is **23.6%**. The MTP top-1 is the break token only **3.0%**
of the time — i.e. breaks are almost never "right token, unlucky sample"; they are the draft model
being categorically wrong. **3 of 4 chain breaks are fundamental draft-model misses that no tree
shape can fix; branching can address at most ~1 in 4.**

## Result T4 (COST/BENEFIT) — at equal verify-width, depth beats breadth in the useful regime
Conditional-independence model fed the measured `c_b(g)` and the measured cost
(`verify=7.0+2.07·W`, `draft=1.9ms/spine-depth`, tree-node cost == chain-node cost from T1).
tok/s proxy = (E_accepted+1)/(verify_ms+draft_ms); best topology per verify-width W:

| W | chain b1 | tree b2 | tree b3 | winner |
|---:|---|---|---|---|
| 4 | D4 E2.76 **164 t/s** | D2 E1.69 141 | D1 E0.90 111 | **chain** |
| 5 | D5 E3.24 **158** | D2 E1.69 127 | — | **chain** |
| 6 | D6 E3.65 **151** | D3 E2.41 136 | D2 E1.71 117 | **chain** |
| 8 | D8 E4.33 **137** | D4 E3.06 130 | D2 E1.71 99 | **chain** |
| 10 | D10 E4.83 **125** | D5 E3.64 125 | D3 E2.45 103 | tie |
| 12 | D12 E5.21 114 | D6 E4.17 **120** | D4 E3.11 104 | tree (past optimum) |

(Model validation: chain E at W=4 = 2.76 vs the measured 2.90 accepted/round at γ=4 — ~5% under,
consistent with the conditional-independence + depth-stationarity approximations.) Chain wins for all
W up to ~10. Each extra **depth** node carries ~86% marginal survival (c_top1); each extra **breadth**
node only converts the ~4-point gap between top-1 and the support ceiling. Trees edge ahead only at
large W (≥12), which is already **past the throughput optimum** the branch established (peak tok/s at
γ≈2–4 / W≈3–5). **Oracle bound:** even an infinite-branch tree (per-node coverage = in-support 0.90),
*charging nothing for the extra branches*, gains only +0.14 / +0.37 / +0.93 accepted tokens over the
chain at depth D=2 / 4 / 8 — and charging the branches their measured ~2ms/node erases that at useful W.

## Integrated conclusion
A branching tree — the canonical technique for raising acceptance-per-width — does not beat the
linear chain in this regime, and the mechanism is now pinned down. Branching is **not** expensive in
itself (T1: a tree node costs exactly what a chain node costs; the fan-out tax is identical), so the
failure is purely on the benefit side: the MTP draft head's error structure is **support-misses, not
rank-misorderings** (T2/T3). When the head is right it is right at top-1 (~86%); when it is wrong the
target's token is usually outside its entire support (~76% of breaks). Trees only help in the
"almost-right" band (target in top-2..k), which is a ~4-point sliver here. This sharpens the branch's
invariant: the acceptance numerator in `verify-width-per-accepted-token` **cannot be raised by a
cleverer verification geometry** — only by a draft with wider, better-calibrated support. Spending the
scarce verify-width budget on depth dominates spending it on breadth.

## Caveats
- The benefit model assumes per-depth coverage is conditionally independent across depths and roughly
  depth-stationary beyond d3 (γ=4 only realizes depths 0–3); `c_b(g)` is measured along the linear
  chain's path, an approximation for a tree path that takes a sibling. T2/T3 (coverage and the 76%
  miss rate) are direct measurements; only the T4 tok/s extrapolation inherits these assumptions.
- T1 is a flattened-token cost proxy: tokens are laid out causally in one sequence; a true tree
  attention mask changes only the smaller attention term, not the dominant per-position MoE-body
  fan-out (the established marginal driver). `ndist` is exact. Cost constants (7.0/2.07/1.9 ms) are
  the prior's 2-GPU tensor-split figures; the probe re-confirmed tree-node cost == chain-node cost.

---

# End-to-end realization (researcher-48) — does the "free copy drafter" pay off in wall-clock?

## Angle
The prior synthesis (below) measured, in offline probes, that the prompt-lookup/copy drafter is
(a) **free** to run (a CPU suffix lookup, no GPU draft forward — the draft forward is 17.7% of
mtp-spec wall-clock), (b) produces **low within-batch diversity** batches that a probe found
~3.7 ms cheaper to verify than the MTP batch, and (c) is **complementary** to MTP (token union
83.9%). It never closed the loop: it never built a decoder that drafts with copy and measured the
real throughput. This extension does exactly that — it realizes the copy-index tooling as two
actual, **lossless** decoding modes and runs the full 30-item eval, to test whether the predicted
"free + cheap-to-verify" advantages convert into end-to-end speedup.

## Method — two new lossless modes in `initial_program.cpp` (mtp-spec/target paths byte-for-byte unchanged)
- **`copy-spec`** (target-only): draft up to γ tokens by a free CPU longest-suffix (k=3→2→1) copy
  chain from the running context (`CopyIndex`); **no draft model is loaded or run at all**. Each
  copy token's proposal `q` is a point mass (q=1 on the copied id), so the existing target
  rejection sampler stays lossless: accept prob = min(1, p_target(copy)); on rejection the residual
  is the target distribution with the copied token removed. When the copy chain is empty the round
  verifies width-1 (a plain target step).
- **`hybrid-spec`**: the normal MTP draft of γ tokens, then **append up to `HYBRID_COPY_EXT`=4
  free copy tokens** continuing the drafted suffix (point-mass q, same lossless verify). Tests
  whether lengthening the speculative window *for free* yields more accepted tokens/round.
- Evaluator passes `-md MTP` for hybrid-spec; copy-spec gets no draft model. Decoded text for all
  three modes is coherent and on-task (lossless paths differ token-for-token by construction).

## Result — the predicted payoff does NOT materialize (full 30-item eval, γ=4, GPUs 0,1)
| mode | accuracy | tok/s | accept | speedup | draft_fwd | verify_fwd | accepted/round |
|---|---:|---:|---:|---:|---:|---:|---:|
| target (ref) | 93.3% | 100.7 | — | 1.00× | — | — | — |
| mtp-spec | 93.3% | 150.5 | 72.4% | 1.49× | 17.7% | 70.4% | 2.90 |
| **copy-spec** | 100.0% | **98.7** | 24.1% | **0.98×** | **0.0%** | 95.6% | 0.86 |
| **hybrid-spec** | 100.0% | **148.5** | 45.1% | **1.47×** | 15.0% | 73.8% | 3.52 |

(Accuracy 100% for copy/hybrid vs 93.3% is lossless sample-path noise on 30 items — both land on
the 2 items target/mtp missed — not a quality claim; it only confirms no quality loss.)

## Mechanism — throughput is set by accepted-tokens-per-verify-forward, not by draft cost
- **copy-spec eliminates the entire draft head (draft_fwd 17.7% → 0.0%) yet is no faster than plain
  autoregressive decoding (0.98×).** The free draft saving is erased — and slightly overshot — by
  the verify side: chained-copy acceptance is only **24.1%**, so the decoder accepts **0.86 drafts
  per round** and needs **137,434 rounds** to do what mtp-spec does in 65,830, each round still
  paying a near-width-5 target verify forward. Total verify wall-clock **doubles** (2460 s vs
  1195 s). In the verify-bound regime, the binding cost is the target forward, which you pay per
  round regardless of how cheaply the draft was produced.
- **The copy batch is NOT meaningfully cheaper to verify in real decoding.** copy-spec verify cost
  is **17.90 ms/round at avg width 4.57** vs mtp-spec **18.15 ms/round at width 5.00** — about the
  same per token, no low-diversity discount. The synthesis's offline probe finding (copy_cont
  ~3.7 ms cheaper, ndist≈3) was the *degenerate collapse-into-repetition* of a greedy copy
  predictor on a single prompt; **real chained copy spans over generated math/code are diverse
  enough to pay nearly full MoE fan-out.** The "within-batch low diversity" axis does not survive
  into end-to-end decoding.
- **hybrid-spec's free extension adds tokens but cannot pay the width tax.** The appended copy
  tokens are accepted only **18.6%** of the time (COPYEXT: 38,676 accepted of 207,513 drafted) —
  far below MTP's 72.4%, because they sit at deep positions conditioned on the whole MTP block
  being accepted first. They do raise accepted/round (2.90 → 3.52) and cut rounds (65,830 →
  54,631), but they widen the verify batch to **8.80 (22.30 ms/round)**; the extra ~2 ms/token MoE
  cost of the mostly-rejected extension tokens almost exactly cancels the extra accepted tokens.
  Net **148.5 tok/s ≈ mtp-spec 150.5** (within run-to-run noise).

## Integrated conclusion
The prior synthesis's two facts about the copy drafter are individually correct but jointly
inert end-to-end: in this **verify-bound** MoE regime the invariant that caps throughput is
**verify-batch-width per accepted token**, and (i) the copy drafter's "free draft" reduces a cost
(draft_fwd) that is not the binding one, while (ii) its tokens still cost the full ~2 ms marginal
MoE verify each and are accepted too rarely (chained 24%, deep-extension 19%) to lower
width-per-accept. A useful copy-based speedup would require raising the *acceptance* of the copy
continuation (the numerator), not exploiting its draft-side cheapness; the draft-side cheapness the
synthesis identified is real but, alone, worth ~0× here.

---

# Synthesis (researcher-43) — does context-redundancy buy MoE route-locality? Two redundancy axes

## Angle
This branch merges two prior independent analyses and integrates them on a single question they
jointly raise but neither could close:
- **Parent A (verify-cost / parent-40):** the dominant cost is the target verify forward; its
  ~2ms marginal-per-token term is **MoE expert fan-out** — diverse tokens in the verify batch
  route to a larger union of experts, pulling more weight bandwidth. It explicitly left open that
  *"the only 'free' extra verify token would be a redundant one routing to already-resident
  experts; real batches are diverse, so each token is ~2ms."*
- **Parent B (copyability / parent-35):** ~49.5% of the target's confirmed tokens are recoverable
  by a ≤3-gram suffix lookup against the running context (prompt-lookup / induction redundancy).

The integrated question: **are parent-B's context-recoverable tokens the cheap "redundant" tokens
parent-A hypothesized?** I.e. does the cross-time copy redundancy that makes half the stream
recoverable also translate into within-forward MoE route-locality (cheaper verification)?

## Method (both parents' instrumentation, non-perturbing)
Carried parent-B's `CopyIndex` (longest-suffix k=3→2→1 prompt-lookup) into the active program and
fused it with parent-A's synchronized per-round `verify_fwd` timing and `probe` mode.
- **Exp 2 (live, mtp-spec):** per round, count how many of the γ MTP draft tokens are
  context-recoverable (read-only lookahead over prompt+confirmed gen), then bucket the *same*
  synchronized verify-forward cost by that count → `COPYCOST` line. Also track copyable-accepted.
- **Exp 1 (controlled, probe):** added two width-W verify batches — `copy_cont` (the continuation a
  prompt-lookup drafter would emit, from `CopyIndex`) and `verbatim_slice` (a contiguous recurring
  prompt block) — alongside the existing same/MTP/distinct/reversed arms, and emitted each batch's
  **within-batch distinct-token count `ndist`**.
- Non-perturbation: full 30-item eval reproduced **93.3% accuracy (== target), 72.4% accept**
  across reruns (tok/s 144–151, varies with GPU contention; generation is seed-deterministic).

## Result S1 (Exp 2) — context-redundancy of the draft does NOT lower verify cost (flat)
Full 30-item eval, 65,830 rounds, verify_fwd bucketed by #context-recoverable draft tokens/round:

| copy-recoverable drafts in round | rounds | share | mean verify_fwd/round |
|---:|---:|---:|---:|
| 0 | 10,428 | 15.8% | 18.035 ms |
| 1 | 15,297 | 23.2% | 18.156 ms |
| 2 | 17,415 | 26.5% | 18.202 ms |
| 3 | 14,794 | 22.5% | 18.190 ms |
| 4 |  7,896 | 12.0% | 18.125 ms |

Spread is **~0.17 ms (~0.9%)** — verify cost is invariant to how copy-redundant the draft block is,
mirroring parent-A's E4 finding that it is also invariant to how many drafts are *accepted*. The
target verify forward has paid the same width-5 cost regardless of either the acceptance outcome or
the informational redundancy of the batch.
- **47.9%** of all MTP drafts are context-recoverable (echoes parent-B's 49.5% on confirmed tokens).
- Copy-recoverable drafts are accepted at **81.4%** vs **72.4%** overall — context-recoverable
  drafts agree with the target more often (consistent with parent-B's high-confidence-copy slice),
  but this is an acceptance (numerator) effect, not a verify-cost (denominator) effect.

## Result S2 (Exp 1) — within-batch token diversity is the cost driver; the copy *predictor* produces low-diversity batches
Width-9 verify-batch micro-benchmark, 6 real prompts (body_k0 = MoE body, no LM head):

| batch | mean ndist | body_k0 (ms) | full kW (ms) |
|---|---:|---:|---:|
| same_last (1 token ×9) | 1.0 | 16.54 | 18.26 |
| **copy_cont** (prompt-lookup drafter) | **3.0** | **16.70** | **18.42** |
| actual_reversed | 8.0 | 19.21 | 20.94 |
| verbatim_slice (contiguous prompt block) | 8.8 | 19.83 | 21.56 |
| actual_mtp | 8.0 | 20.38 | 22.11 |
| prompt_distinct | 9.0 | 20.51 | 22.23 |

- The copy-predictor continuation verifies **~3.7 ms cheaper than the actual MTP batch** and tracks
  the degenerate same-token batch (Δ ~0.16 ms), across all 6 prompts (n_copy_pred = 8/8 every time).
- The cause is measured directly: `copy_cont` has **ndist = 3** distinct tokens in the width-9 batch
  vs **8–9** for MTP/distinct/verbatim. Body cost is monotone in ndist; the small expert union of a
  low-diversity batch is the cheap regime parent-A's fan-out mechanism predicts.
- **Crucial control:** `verbatim_slice` is built from *context tokens* yet is high-diversity
  (ndist ≈ 9) and as expensive as `prompt_distinct`. So it is **not** "tokens drawn from context"
  that are cheap — it is **within-batch repetition**. The greedy copy predictor is cheap because it
  chases recurrence and collapses into a short cycle of few distinct tokens, not because its tokens
  are context-recoverable per se.

## Integrated picture — two orthogonal "redundancy" axes
There are two distinct notions of redundancy, and they are different things:
- **Cross-time / context-recoverability** (parent-B): a token repeats something earlier in the
  stream. This raises *acceptance* (81.4% vs 72.4%) but leaves *verify cost* untouched (S1, flat),
  because a context-copy token is still a distinct token *within* the verify batch, routing to its
  own experts.
- **Within-forward token diversity** (parent-A): the count of distinct tokens *inside one verify
  batch* (`ndist`) sets the expert-union size and thus the MoE-fan-out cost (S2, monotone in ndist).

Parent-A's open "redundant token routes to resident experts → free" is therefore resolved: the
redundancy that lowers verify cost is **within-batch token repetition (small expert union)**, which
the parent-B copy *predictor* happens to generate by collapsing into recurrence — NOT the cross-time
context-recoverability parent-B measured (verbatim context blocks stay diverse and expensive). The
two axes coincide only through the copy drafter's collapse-into-repetition behaviour.

Caveat: the probe's `expected_accept` field is computed for the actual MTP batch and reused as a
display constant; it is not a per-arm acceptance estimate, so S2 reports `copy_cont`'s verify *cost*
only, not its acceptance. Probe absolute µs are 2-GPU tensor-split (per parent-A's convention).

---

# Findings — wall-clock latency decomposition of speculative decoding

## Angle
Speedup = useful_tokens / round_wallclock. Prior analyses studied the **numerator**
(acceptance: overall rate, per-depth, entropy/argmax drivers, run lengths, gamma tradeoff).
This analysis is deliberately orthogonal: it dissects the **denominator** — where the
per-round wall-clock actually goes — to find what caps achievable speedup independently of
acceptance. Hypothesis going in was that the CPU-side `make_dist` (a `partial_sort` over
Gemma's ~262K-token vocabulary, called ~2γ+1 times/round) might be a hidden bottleneck.

## Method
Instrumented `initial_program.cpp` with `ggml_time_us()` accumulators per phase
(prefill / draft_fwd / draft_smp / verify_fwd / verify_smp / cache), emitted as a `TIMING`
line per prompt (printed before `TEXT:`, so the scorer's regexes are unaffected). Accuracy on
the full 30-item eval stayed **93.3%**, identical to the target reference — instrumentation is
non-perturbing.

### Methodological catch (important)
`llama_decode` is **asynchronous**. A first naive run attributed ~70% of wall-clock to
`make_dist` (verify_smp). That was an artifact: the subsequent `llama_get_logits_ith` forces a
device sync, so GPU compute was being billed to the CPU sort that followed it. Adding an
explicit `llama_synchronize()` after every `llama_decode` (prefill, draft, verify) flipped the
picture entirely and is required for a correct decomposition.

Note on GPU config: `./evaluator/task-eval --gpus 0,1` runs **two single-GPU data-parallel
workers** (each `spec_modes` pinned to one GPU via `CUDA_VISIBLE_DEVICES`), not one tensor-split
job. The γ-sweep below was run as a single tensor-split process across both GPUs; this shifts
absolute per-call latency (MTP step 1.9ms split vs 1.15ms single-GPU) but the structure is
identical.

## Result 1 — wall-clock decomposition (full 30-item eval, gamma=4, the scored run)
Aggregate over 30 prompts (1724s total wall, 65830 rounds, ~520K make_dist calls):

| phase | share | what it is |
|---|---|---|
| verify_fwd | **69.4%** | target forward pass verifying the γ+1-token batch |
| draft_fwd  | **17.6%** | MTP head drafting (γ sequential width-1 passes) |
| draft_smp  | 6.4% | CPU `make_dist`/sample for draft tokens |
| verify_smp | 6.2% | CPU `make_dist`/resample for verification |
| prefill    | 0.2% | one-time prompt encode |
| cache      | 0.2% | seq_rm + batch init/free + memcpy |

GPU forward = **87.1%**, CPU sampling+cache = **12.8%**.

The starting hypothesis was **wrong**: the 262K-vocab sort is *not* the bottleneck
(~0.42ms/call, 12.6% combined). The **target verification forward pass dominates** at ~69%.

## Result 2 — cost curve of the target verify forward (γ-sweep, single tensor-split process)
verify_fwd per round vs verify batch width (γ+1), 2-prompt subset:

| batch width | verify_fwd/round | MTP draft/step | tok/s |
|---|---|---|---|
| 1 (target, single token) | 9.10 ms | — | 101 |
| 2 (γ=1) | 11.13 ms | 1.91 ms | 127 |
| 3 (γ=2) | 13.23 ms | 1.91 ms | 142 |
| 5 (γ=4) | 17.70 ms | 1.90 ms | 142–150 |
| 7 (γ=6) | 21.50 ms | 1.90 ms | 134 |
| 9 (γ=8) | ~21.5 ms (noisy) | 1.88 ms | 133–141 |

Linear fit (width 1→7): **verify_fwd ≈ 7.0ms fixed + ~2.07ms per token in the batch.**
Batched verification is sub-linear but **far from flat/free**: 5 tokens cost ~1.96× a single
token, not ~1×. The MoE target (4B active of 26B) is not in the weight-bandwidth-bound regime
where extra batch tokens are free — extra distinct tokens activate more experts / more compute,
so marginal token ≈ 2.07ms.

## Structural facts
- **Drafting is linear, verification is sub-linear.** MTP draft = γ *sequential* width-1
  passes (~flat ~1.9ms/step → total scales linearly in γ). Verify = one batched pass
  (7.0 + 2.07·width). This asymmetry sets the throughput optimum at small γ (peak tok/s at
  γ≈2–4; declines by γ=6–8 as draft cost grows linearly while accepted tokens saturate).
- **The "tiny" MTP draft head is not nearly free.** A single MTP forward (~1.9ms tensor-split /
  1.15ms single-GPU) is comparable to the *marginal* cost of one extra verified target token
  (~2.07ms), and drafting is **17.6%** of total wall-clock. Likely contributor: the MTP head
  shares the giant 262K-row output (un)embedding projection with the target, which is invariant
  to how small the transformer body is — so a 461MB head (~1.7% of the 27GB target) does not
  cost ~1.7% of the per-token work. (Mechanism is a hypothesis; the timing asymmetry is measured.)
- **Verify-bound speedup ceiling.** Holding acceptance and γ fixed, if drafting + all CPU
  sampling were zero-cost, wall-clock would fall to verify_fwd+prefill (69.6%), i.e. an upper
  bound of ~1724/1200 = 1.44× faster than current → ~**2.1× absolute speedup ceiling** in this
  verify-bound regime. The target verification forward is the hard floor.

---

# Extension — mechanistic decomposition of the forward cost (what physically costs the time)

## Angle
The decomposition above measured the *symptom* (verify ≈ 7.0ms fixed + 2.07ms/token; draft ≈1.9ms)
and left two **labelled hypotheses** for the *cause*: (H1) the 2.07ms marginal verify token is
"MoE expert fan-out", (H2) the tiny MTP draft is expensive because it shares the 262K LM head.
This extension arbitrates both by physically separating the two candidate cost centers — the
shared 262K-row LM-head (un)embedding matmul vs the MoE transformer body — using two levers the
γ-sweep never touched. New `--mode probe` in `initial_program.cpp`; analysis is non-perturbing
(it is an isolated branch; scored `target`/`mtp-spec` paths are byte-for-byte unchanged — smoke:
mtp-spec 1.60× / 81% accept, healthy).

Architecture (Q8_0, from gguf meta): 30 layers, hidden 2816, **128 experts, top-8 used**,
expert-FFN 704, vocab **262144**. All probe numbers are **single-GPU** (CUDA_VISIBLE_DEVICES=4),
medians of 40 reps after 8 warmups, on real prefilled prompts (3 prompts, results consistent).

## Method — two orthogonal levers
- **Logit-count k:** decode a width-W batch but compute logits (the 262K LM-head matmul) on only
  the last k positions. The body (attn+MoE) runs for all W regardless, so cost(W,k)−cost(W,0)
  isolates the LM head, and its slope in k splits it into a fixed weight-read vs per-token compute.
- **Token diversity:** width-W batch of W *distinct* tokens vs W copies of one token, k fixed.
  At fixed batch width the CUDA kernels are identical, so the (distinct − same) gap isolates the
  extra expert weight bandwidth pulled in by routing diversity (8-of-128 experts/token).
- **MTP draft:** same logits-on vs logits-off lever on the draft forward → LM-head share of draft.

## Result E1 — the 262K LM head is almost pure FIXED cost, NOT the per-token driver (refutes the naive reading of H1)
Logit-count sweep at W=9 (avg µs): k0=21044 → k1=22245 (**+1201**) → k9=22817.
- LM-head **fixed weight read** (k0→k1): **~1201 µs** (one-time, ≈782MB Q8 unembedding read).
- LM-head **per-token compute** (k1→k9 slope): **~71 µs/token**.
So 17 of every 18 LM-head microseconds are the one-time weight read; each *extra* verified token
adds only ~71µs of LM-head work. The 262K vocab is a big **per-forward fixed** cost, not the
2.07ms marginal. (CPU make_dist over 262K, from the prior section, is a separate ~0.42ms/call.)

## Result E2 — the marginal verify token IS the MoE body, and expert fan-out is a large, growing share (confirms + quantifies H1)
Body-only cost (k0, no logits), avg µs, distinct vs identical tokens:

| W | same_k0 | distinct_k0 | fan-out gap | distinct marginal/tok |
|---|---|---|---|---|
| 1 | 7775 | 7760 | −15 | — |
| 2 | 9651 | 9844 | 193 | |
| 3 | 11306 | 11916 | 610 | |
| 5 | 14343 | 16106 | 1763 | |
| 7 | 16794 | 19711 | 2917 | ~1992 (W1→7) |
| 9 | 15203 | 21022 | 5819 | ~1658 (W1→9, flattening) |

- At fixed batch width (identical kernels), **distinct-token batches cost progressively more than
  identical-token batches** — the fan-out gap grows 0→5819µs across W=1→9. This is the direct
  fingerprint of MoE expert fan-out: each new *distinct* token routes to experts not already
  resident, adding weight bandwidth (~1.5GB of expert weights/token if disjoint).
- The distinct marginal/token **flattens** toward W=9 (1992→1658 µs/tok) — consistent with the
  expert union approaching the 128-expert ceiling (8×9=72 of 128 → fewer *new* experts per token).
  This is the mechanism behind the prior section's "sub-linear but not free" verify curve.
- The full-logit distinct marginal (kW, W1→7) ≈ **2.0ms/token**, reproducing the prior 2.07ms,
  now decomposed: ~71µs LM-head compute + ~900µs irreducible body (attention over the growing
  prefix + dense layers + the experts a lone token needs) + ~500–1000µs **expert fan-out**.

## Result E3 — the "tiny" MTP draft is 57% shared LM head (confirms H2)
MTP draft forward (avg µs): body (logits off) = **401** + LM head (off→on) = **524** = **926** total.
- The draft transformer body is genuinely tiny (~401µs, one nextn layer). But producing a 262K
  draft distribution forces the **same** ~782MB unembedding read the target pays (implied
  ~1492 GB/s, ≈ HBM bandwidth) → **57% of the draft forward is the shared LM head**, invariant to
  how small the head's body is. This is why a 461MB head (~1.7% of the target) is not ~1.7% of
  per-token work. H2 confirmed.

## Unifying picture
Every structural fact in the prior section follows from one cause — **weight bandwidth, split into
a fixed and a marginal term**:
- **Fixed per forward:** LM-head read (~0.8–1.2ms) + always-on dense/attention weights → the
  ~7ms verify floor and the ~0.4ms of the 0.9ms draft.
- **Marginal per token:** MoE expert fan-out, bandwidth-bound and **diversity-dependent**,
  saturating as the 128-expert union fills. The only "free" extra token would be a redundant one
  routing to already-resident experts; real verification batches are diverse, so each token is
  ~2ms. This is why verification is sub-linear-but-not-free and throughput peaks at small γ.

## Measurement caveats
- The identical-token ("same") body cost is **non-monotonic** in W (W=9 < W=7), a CUDA
  kernel/graph tiling effect at specific batch sizes; the fan-out claim therefore rests on the
  *within-W* (distinct − same) gap, which holds batch size — and thus kernel selection — fixed.
- The LM-head fixed read measures ~1201µs in the target k0→k1 delta but ~524µs in the single-token
  MTP path; the target figure carries extra per-batch graph/output overhead, so the MTP single-
  token number (~524µs, ~1.5TB/s) is the cleaner weight-read estimate. Either way the LM head is a
  large fixed cost and a negligible per-token one.
- Single-GPU absolute µs (data-parallel per-worker config); the prior γ-sweep was 2-GPU
  tensor-split, so absolute values differ but the structure matches (single-GPU width-1 target
  8.9ms here ≈ prior 9.1ms).

---

# Extension 2 — live speculative batches pay before acceptance is known

## Angle
The prior extension established that token diversity makes the MoE body expensive. This extension
tests the operational consequence for real speculative decoding rounds: whether high-acceptance
MTP continuations have enough route locality to be cheaper than arbitrary distinct-token batches,
and whether the number of accepted draft tokens predicts the target verify cost already paid.

## Method
Added two non-perturbing measurements to `initial_program.cpp`.
- Normal `mtp-spec` emits `ACCEPTCOST`, bucketed by accepted-prefix length per round, using the
  already-measured synchronized target verify forward time.
- `probe` now builds a real width-9 speculative batch from an argmax MTP continuation after a
  real prefill, estimates its expected accepted prefix under the same top-k/top-p/temp
  distributions, and benchmarks that exact batch against same-token, prompt-distinct, and
  actual-token-reversed controls.

Validation run: `./evaluator/task-eval --gpus 0,1` on the full 30 items gave **93.3% accuracy**,
**148.6 tok/s**, **72.4% accept**, **1.48× speedup**. Timing shares remained structurally
unchanged: verify_fwd **69.5%**, draft_fwd **17.6%**, CPU sampling **12.5%**, cache+prefill
**0.35%**.

Probe run: `CUDA_VISIBLE_DEVICES=0,1 PROBE_REPS=12 PROBE_PROMPTS=4 PROBE_G=8 ./spec_modes
--mode probe ...` on four real prompts. These absolute probe numbers are 2-GPU tensor-split,
not the prior single-GPU probe configuration.

## Result E4 — accepted-prefix length does not change the verify cost already paid
Full 30-item eval, γ=4, 65,830 speculative rounds:

| accepted drafts in round | rounds | share of rounds | mean verify_fwd/round |
|---:|---:|---:|---:|
| 0 | 8,514 | 12.9% | 18.145 ms |
| 1 | 7,082 | 10.8% | 18.150 ms |
| 2 | 5,984 | 9.1% | 18.131 ms |
| 3 | 5,286 | 8.0% | 18.148 ms |
| 4 | 38,964 | 59.2% | 18.162 ms |

The worst and best acceptance buckets differ by only **0.031 ms** per target verify forward
(~0.17%). Full-accept rounds are 59.2% of rounds and consume 59.2% of verify time; zero-accept
rounds are 12.9% of rounds and consume 12.9% of verify time. The target has already paid the same
width-5 verification cost before rejection sampling reveals whether the round emits 1 token or 5
tokens.

## Result E5 — real high-accept MTP batches are not cheaper than arbitrary distinct batches
Width-9 live probe averages over four prompts; expected accepted prefix for the actual MTP batch
was **6.141 of 8 draft tokens**:

| batch | body only k=0 | full logits k=W |
|---|---:|---:|
| same last token repeated | 16.507 ms | 18.240 ms |
| actual MTP continuation | 20.324 ms | 22.057 ms |
| prompt-distinct tokens | 20.533 ms | 22.273 ms |
| actual continuation reversed after first token | 19.116 ms | 20.847 ms |

The actual MTP batch is **3.82 ms** slower than the same-token control and only **0.21 ms**
faster than arbitrary prompt-distinct tokens. Thus a useful/high-accept speculative continuation
still pays nearly the full distinct-token MoE fan-out cost. Reversing the actual drafted suffix
reduced body cost by **1.21 ms**, so the cost is not only the token multiset; causal order and the
hidden states induced by that order also affect routing cost.

The same probe reproduced the earlier decomposition under the 2-GPU tensor-split configuration:
W=9 distinct body **20.729 ms** vs same body **15.435 ms** (fan-out gap **5.294 ms**);
LM-head k0→k1 fixed increment **1.182 ms** and k1→k9 slope **~68 µs/logit token**. MTP draft
logits-off/on averaged **0.982 ms → 1.510 ms**; the shared LM-head increment was **0.529 ms**
(35% of this 2-GPU draft forward).

---
# (merged reference) researcher-35 findings (context-copy/recoverability):

# Findings — context-recoverability ("copyability") of the target's generation

## Lens
A genuinely orthogonal angle to the saturated lenses (acceptance, per-depth/run-length,
gamma, timing, output length, hidden-state geometry, prompt structure, determinism,
draft-error taxonomies, single-next-token uncertainty): the **self-referential / copy
structure of the target's output stream**. For each confirmed token I ask whether it is
recoverable by a longest-suffix n-gram lookup ("induction" / prompt-lookup) against the
*running context* (prompt + generation so far), and how much of the target's own truncated
predictive mass lands on that context-recoverable candidate. This characterizes the
information-redundancy of the generation itself, independent of the speculative machinery.

## Method
- Instrumented `initial_program.cpp` with **read-only** logging in `mtp-spec` mode
  (`CopyIndex`: incremental 1/2/3-gram suffix→follower maps over prompt+generation; logs one
  JSON record per confirmed token). Decoding path is byte-for-byte unchanged.
- Verified non-perturbation: instrumented full 30-item run reproduces the reference exactly —
  **accuracy 93.3% (== target), accept 72.4%, speedup 1.47×** (ref 1.51×, within seed noise).
- 256,594 confirmed-token records (GPUs 6,7). `drafted=1` = token was an accepted MTP draft;
  `drafted=0` = resampled/bonus token (an MTP "miss"). Source mapping reconstructed from the
  interleaved `items[i::2]` chunking. Analysis: `/tmp/analyze_copy.py`.

## Results

### 1. Half the generation is context-recoverable by a trivial suffix lookup
Longest-match (≤3-gram) copy predictor over the running context:
- **Copy-coverage = 49.5%** of all confirmed tokens equal the looked-up candidate.
- A candidate exists for 94.4% of tokens; precision rises with suffix length:
  1-gram-only 27.7% hit, 2-gram-only 44.7%, 3-gram available 60.8% (hit|seen).
- Per source (copy-coverage): **aime 54.4% > lcb 50.4% > gpqa 48.4% > hle 42.5%**.
  Math/code (repeated identifiers, equations, digit runs) are most self-referential; the
  open-ended HLE prose is least.

### 2. The copy structure aligns with the target's own beliefs (not just surface repetition)
- Mean target truncated mass on the longest copy candidate = **0.524**; the candidate sits
  inside the target's top-k/top-p support **59.8%** of the time.
- Confidence-gating on the candidate's target mass is near-deterministic:
  | gate (target mass on candidate) | fires on | precision (hit) |
  |---|---|---|
  | ≥0.0 | 94.4% | 52.4% |
  | ≥0.5 | 49.4% | 97.4% |
  | ≥0.9 | 45.1% | 99.8% |
  | ≥0.99 | 43.4% | 100.0% |
  → Whenever the context-recoverable candidate is one the target is confident in, it is almost
  always the realized token. ~45% of all tokens fall in this "confident-and-copyable" regime.

### 3. Copy structure and MTP drafting overlap but do not coincide
Cross-tabulating longest-match copyability vs MTP capture (`drafted`):
- copyable & MTP-got (redundant): **40.0%**
- copyable & MTP-missed: **9.5%**
- non-copyable & MTP-got (MTP-unique): **34.4%**
- non-copyable & MTP-missed (hard): **16.1%**
- Marginals: copy-coverage 49.5%, MTP-coverage 74.3%, **union 83.9%**.
- Among the 25.7% of tokens MTP misses (resample/bonus), **37.2% are copyable**.
- The 9.5% copyable-but-MTP-missed tokens are high-confidence for the target: mean mass on the
  correct copy candidate **0.919** (84.5% have mass >0.9). I.e. MTP's misses include a
  near-certain, context-recoverable slice that a 3-gram lookup reproduces; conversely 34.4% of
  tokens are captured by MTP yet are *not* recoverable by suffix lookup (MTP's unique reach).

## Summary
On this benchmark ~half of the target's generated tokens are reproducible by a ≤3-gram
suffix match against their own running context, the redundancy is highest for math and lowest
for open-ended prose, and context-recoverability coincides almost perfectly (99.8%) with the
realized token whenever the candidate carries high target mass. MTP drafting and copy structure
are partially complementary: their captured-token sets overlap on 40% of tokens, each reaches a
distinct ~10/34% the other does not, and together they cover 83.9% of confirmed tokens.

---
# (merged reference) researcher-56 findings (entropy/diversity ceiling synthesis):

# Findings — researcher-56: entropy-resolved decomposition of the stochasticity tax

## Angle (synthesis of the two parent lines on one axis)
The two parents converge on the same per-position quantity — the target's verification entropy
`Ht` (Shannon entropy over the top-k→top-p support, the AUC-0.882 rejection predictor from r31):
- **Parent r51/r39 (predictive structure):** the target's single forward pass FORESHADOWS its own
  future tokens strongly at LOW entropy / word onsets (next-novel-token in top-40 ~45–50%) and
  collapses toward chance at HIGH entropy; its own runners-up recur within 5 tokens 51% at H<0.1
  vs 35.5% at H>1. Low entropy = self-determined "backbone"; high entropy = unforeshadowed "joints".
- **Parent r41/r31 (stochasticity tax):** force-accepting the *artifact* rejections (draft == target
  argmax, rejected by the lossless test only because q>px under temp=1) recovers throughput
  (1.49→1.65×) but costs ~10–13 accuracy points, because the off-mode corrections those rejections
  force are "load-bearing diversity" that keeps the reasoning trajectory from degenerating.

Integrated claim: **the stochasticity tax is paid entirely at the high-entropy joints.** If the
two parents describe the same axis, then gating mode-recovery on `Ht<τ` (recover the artifact only
where the target is already determined / foreshadows its own future) should reclaim throughput
WITHOUT degenerating, and the accuracy collapse should appear only once τ crosses into the forks.

## Method
- Merged the parents into one program: kept r51's target-mode `--dump-horizon`; grafted r41's
  `SPEC_VERIFY` switch (`lossless`/`greedy`/`argmax_sticky`) into the mtp-spec accept loop and added
  a new variant **`entropy_sticky`**: force-accept an artifact (draft==target argmax that the
  lossless test would reject) ONLY when target `Ht < SPEC_ENT_THRESH`, else the unchanged lossless
  test + residual `p−q` correction. The lossless RNG/codepath is byte-preserved (one `uni` draw per
  position, identical to the parent). Side-channel logs per verified position
  `{q, px, acc, Ht, pt, accepted, art(=draft==argmax), frc(=forced by gate)}`.
- Full 30-item evals on GPUs 6,7 (`./evaluator/task-eval`, n=16384). Offline `analyze_synthesis.py`.
  The `argmax_sticky` run force-accepts every would-be-rejected artifact (`frc=1`), so its log is the
  complete recoverable-artifact entropy distribution.

## Results

### 1. Measured frontier along the entropy gate (full 30-item benchmark, GPUs 6,7)
| variant | gate `Ht<τ` | accept | tok/s | speedup | accuracy | recoverable pool captured |
|---|---|---|---|---|---|---|
| lossless (reference) | τ=0 (none) | 72.4% | ~150 | 1.49× | 93.3% | 0% |
| entropy_sticky τ=0.5 | 0.5 | 73.9% | 150.9 | 1.50× | **93.3%** | 13.4% |
| entropy_sticky τ=1.0 | 1.0 | 79.3% | 158.5 | 1.57× | **93.3%** | 55.5% |
| argmax_sticky | τ=∞ (all) | 85.0% | 165.5 | 1.64× | **80.0%** | 100% |

Recovering every artifact below `Ht=1.0` (more than half the recoverable pool) preserves accuracy at
93.3% while capturing ~54% of the available throughput gain (accept 72.4→79.3 of the 72.4→85.0
range; speedup 1.49→1.57 of 1.49→1.64). The full ~13-point accuracy collapse (93.3→80.0%) is
produced *entirely* by additionally recovering the `Ht>1.0` artifacts.

### 2. The recoverable-artifact pool lives almost entirely at moderate-to-high entropy
Complete distribution of the 11,999 recoverable artifacts (4.6% of 260,286 verified positions; 90.8%
of positions are artifacts overall, but only 5.1% of those are rejected by lossless):
| Ht band | share | cum |
|---|---|---|
| [0.0,0.1) | 0.0% | 0.0% |
| [0.1,0.3) | 4.2% | 4.3% |
| [0.3,0.5) | 9.2% | 13.4% |
| [0.5,0.7) | 24.0% | 37.4% |
| [0.7,1.0) | 18.1% | 55.5% |
| [1.0,1.5) | 24.1% | 79.6% |
| [1.5,+) | 20.4% | 100.0% |

Median recoverable-artifact `Ht`=0.91, mean 1.04. **Only 0.03% (4/11,999) sit below Ht=0.1** — the
foreshadowed backbone (≈55% of all tokens per r25) yields essentially no recoverable throughput,
because there the target is so peaked (px≈q≈1) that the artifact is already accepted by the lossless
test (`min(1,px/q)≈1`). An artifact can be *rejected* only when px<q, which structurally requires the
target to spread mass off its own argmax — i.e. nonzero entropy. The throughput pool and the
high-entropy joints are the same positions.

### 3. The tax is mechanically the entropy axis, not a content or depth confound
τ=1.0 force-accepted 6,873 artifacts (median Ht 0.65) with zero accuracy loss; argmax_sticky added
the remaining 5,126 (the Ht>1.0 tail, lifting mean forced Ht to 1.04) and that increment alone caused
the 93.3→80.0% drop. The safe/degenerating boundary is a sharp threshold near Ht≈1.0 — exactly the
entropy region where r51 measured the forward pass's foreshadowing of its own continuation to have
collapsed toward chance (runners-up realized 35.5% at H>1 vs 51% at H<0.1; novel d=1 foreshadow
33.6% at H>1.5 vs 44.7% at H<0.01).

## Synthesis
A single axis — the target's verification entropy — unifies both parent lines. Below ~1 nat the
trajectory is a self-determined backbone: the forward pass already foreshadows its own next tokens
(r51), the target is peaked enough that drafted-argmax tokens are accepted losslessly, and the few
that aren't can be force-accepted with no quality cost — recovering the target's own mode there is
effectively lossless. Above ~1 nat are the joints: the pass stops foreshadowing its continuation
(r51), the artifact rejections concentrate (this work), and the off-mode corrections they force are
load-bearing diversity whose removal degenerates the reasoning trajectory (r41). The stochasticity
tax is therefore not spread across decoding; it is localized to the high-entropy fork positions, and
the speculative-decoding throughput that mode-recovery can tap is co-located with — and inseparable
from — exactly the diversity that keeps this benchmark's accuracy intact.

## Artifacts (researcher-56)
- `initial_program.cpp` — merged: r51 `--dump-horizon` (target) + r41 `SPEC_VERIFY` and side-channel
  (mtp-spec), plus the new `entropy_sticky` gate on target `Ht` (`SPEC_ENT_THRESH`).
- `analyze_synthesis.py` — recoverable-artifact entropy distribution + per-variant forced accounting.
- `results/mode_ent05.json` (τ=0.5), `results/mode_ent10.json` + `results/ent10_{6,7}.jsonl` (τ=1.0),
  `results/mode_sticky.json` + `results/sticky_{6,7}.jsonl` (argmax_sticky). τ=0.5 side-channel not
  captured (empty `CONTENT_LOG` disabled logging); its eval metrics are intact.
- mtp-spec lossless path unchanged; lossless reference reproduces both parents (93.3% / 72.4% accept).

---
# (parent) researcher-51 findings — uncertainty-conditioned foreshadowing (synthesis of r39 + r25):

## Angle (researcher-51, integrated)
The two parent analyses both characterize the *same single target forward pass* `D_s` but at
opposite ends of the distribution, and were never joined per-position:
- **r39 (the tail):** `D_s`'s top-40 places above-chance mass on *novel future* tokens `out[s+d]`,
  d≥1, horizon ~3–5 — latent foreshadowing, mid-ranked, never the argmax.
- **r25 (the head):** `D_s`'s per-position entropy is sharply bimodal (≈56% near-deterministic);
  uncertainty is concentrated in word tokens and front-loaded at word onsets.

Integrated question: **does the present token's predictive uncertainty (r25 head) govern how much,
and what kind of, future-token foreshadowing (r39 tail) is latent in the same forward pass?** Two
mechanisms make opposite predictions — *confidence-determines-future* (foreshadow strongest at
low-entropy forced steps) vs *uncertainty-spreads-trajectories* (foreshadow strongest at
high-entropy decision points) — and the answer says where, along the sequence, the target's own
distribution does or does not anticipate its own continuation.

## Method
- Merged the two instrumentations into one aligned dump (target mode only; mtp-spec path
  untouched): the active `--dump-horizon` already records `D_s`'s top-40 (ids + true full-vocab
  softmax probs) and the realized sequence; added r25's `full_stats` so every position also carries
  full-vocab Shannon entropy `H_s` (nats, temp 1), top-1 prob, the sampled token's rank, and the
  escaped piece text (for offline content classification identical to r25).
- Drove target mode over all 30 benchmark prompts, n=512, seed 0, temp 1.0/top_p 0.95/top_k 64,
  split across GPUs 6,7 (`run_horizon_joint.py`). **15,360 generated positions.** Offline analysis
  in pure Python (`analyze_joint.py`). Foreshadow hit = `out[s+d] ∈ top40(D_s)`; novelty =
  `out[s+d]` absent from `out[0:s+d]`; null = within-cell shuffle of the future token.

## Results

### 0–1. Merge reproduces both parents
- Pooled foreshadow decay matches r39: novel-hit d1=41.6%, d2=23.1%, d3=17.4%, → 7.1% by d20;
  frequency baseline 16.0%, novel-only baseline 7.7%.
- Entropy structure matches r25: p50=0.003, p90=0.993, p99=2.041; near-deterministic (H<0.01)=55.1%.

### 2. Immediate foreshadowing declines monotonically with present entropy
d=1 novel-foreshadow hit%, binned by `H_s`:

| H_s bin | N | hit% | shuffled-null% |
|---|---|---|---|
| [0,.01) | 1940 | 44.7 | 5.5 |
| [.01,.1) | 423 | 41.1 | 4.3 |
| [.1,.5) | 610 | 41.6 | 5.9 |
| [.5,1) | 709 | 39.8 | 5.9 |
| [1,1.5) | 390 | 34.4 | 5.9 |
| [1.5,+) | 232 | 33.6 | 3.4 |

The next *novel* token is foreshadowed ~45% of the time when the model is near-certain about the
present token, falling to ~34% at the most uncertain positions — i.e. the future is most visible in
the present pass exactly where the present is already nearly determined. Lift over the within-bin
null stays large (≈6–10×) in every bin.

### 3. Two orthogonal regimes — boundary × entropy cross-tab (d=1 novel hit%)
| present position | H<0.1 | 0.1–0.5 | ≥0.5 |
|---|---|---|---|
| **word-onset** (leading-space alpha) | 47.4 (683) | 55.5 (326) | 50.4 (796) |
| **word-continuation** (subword) | 18.0 (479) | 7.2 (69) | 3.9 (180) |
| **non-word** (punct/digit/ws/markup) | 52.5 (1201) | 31.6 (215) | 24.2 (355) |

- At a **word onset** the next token (the word's own completion) is foreshadowed ~50%
  *independent of entropy* — the model anticipates how to finish a word it has started even when the
  onset choice itself was high-entropy. This is a structural/linguistic-boundary effect, not an
  uncertainty effect.
- At **continuation and non-word** positions foreshadowing tracks present certainty and collapses
  toward the null as `H_s` rises (continuation 18%→3.9%). The pooled entropy decline in §2 is driven
  entirely by these non-onset positions.

### 4. Foreshadowed future tokens are predominantly word-class; digits are the most reliably foreshadowed
Of novel d=1 tokens that land in `D_s`'s top-40: 74.6% are words (≈ their 73.2% share of all novel
tokens). By in-tail rate: **digit 84.8%**, word 42.4%, math_markup 39.0%, punct 33.7%,
whitespace 18.7%. Digits are rare as novel tokens (3.2%) but, consistent with r25's finding that
digits are near-deterministic in the head, they are the single most foreshadowed class in the tail.

### 5. At decision points the unchosen alternatives are genuine forks
Whether a top-2..10 runner-up of `D_s` (a candidate weighed but not sampled) is realized within the
next 5 tokens: **51.1%** at low entropy (H<0.1), 42.6% at mid, **35.5%** at high entropy (H>1).
At confident steps the runners-up are parallel content that recurs nearby; at genuine high-entropy
decision points the unchosen branch is increasingly *not* realized — the present pass anticipates
its own continuation least precisely exactly where it is most uncertain.

### 6. Content-independent
The low-entropy d=1 foreshadow lift (hit/null) holds across domains: aime 5.3×, gpqa 12.5×,
hle 9.2×, lcb 4.9×. Where high-entropy novel d=1 positions are numerous enough (gpqa/hle/lcb) the
lift persists but the raw hit% is lower than the low-entropy hit%, matching §2.

## Synthesis
The head and tail of the target's single forward pass are coupled by present certainty, with one
structural exception. Outside word onsets, the latent future-token information r39 found lives
disproportionately in the low-uncertainty regions r25 found to dominate the sequence: the pass
foreshadows the next novel token, and its own runners-up recur nearby, precisely when the future is
already nearly determined, and goes toward chance at the genuine high-entropy decision points. The
exception is the word boundary: once a word begins, its completion is foreshadowed ~50% regardless
of the onset's entropy. Relative to the MTP draft (which is verified hardest at exactly the
high-entropy decision points), the target's own distribution-tail anticipation is weakest in that
same region and strongest in the confident/word-internal region.

## Artifacts (researcher-51)
- `initial_program.cpp` — merged `full_stats`/`json_escape`; `--dump-horizon` now emits per-position
  `ent`,`top1`,`rank`,`pc` alongside `topk_id`/`topk_p` (target mode only).
- `run_horizon_joint.py` — driver; `analyze_joint.py` — offline integrated analysis.
- `results/horizon_joint_gpu{6,7}.jsonl` (15,360 positions), `results/raw_horizon_gpu{6,7}.txt`,
  `results/task_eval.json`.
- Standard `./evaluator/task-eval --gpus 6,7`: target 93.3%/100.7 tok/s; mtp-spec 93.3% / 143.5
  tok/s / **72.4% accept** (accept rate identical to reference; mtp-spec path unmodified).

---
# (parent) researcher-39 findings — latent future-token foreshadowing:

# Findings — Latent future-token foreshadowing in the target's single-step distribution

## Angle
A genuinely orthogonal lens (not acceptance/run-length/gamma/timing/forward-cost/output-length/
draft-geometry/prompt-structure/determinism/error-categories/single-next-token-uncertainty/
context-recoverability-copy): **how much multi-step future information is latent in a single
target forward pass.** In pure autoregressive target decoding, the next-token distribution
`D_p = softmax(logits_p)` is produced by one forward pass. We ask whether `D_p` already places
above-chance probability on tokens realized *further ahead* (`t_{p+d}`, `d>=1`), i.e. whether the
present distribution foreshadows future content — and whether that holds for *novel* tokens
(not copyable from context), which separates this lens from n-gram/copy structure.

This is the prompt-invited "relationship between target verification logits and future tokens",
generalized from the immediate next token (d=0) to a multi-step horizon.

## Method
- Instrumented `initial_program.cpp` (target mode only; mtp-spec path untouched): for each
  generated position the true-softmax (full-vocab denominator, temp=1) top-40 of `D_p` is dumped
  with the realized token sequence. New flag `--dump-horizon <path>`.
- Ran instrumented target mode on GPUs 4,5 over all 30 benchmark prompts, n=512
  (15,360 positions). Offline analysis in pure Python.
- Alignment: `D_s` is the distribution that produced `out[s]` (so d=0 is the sampled token);
  lookahead distance `d` measures `D_s`'s mass on `out[s+d]`.
- Controls: (1) **frequency baseline** = probability a *random realized token* from the same
  prompt lands in `D_s`'s top-40; (2) **novelty split** = future tokens never seen in
  `out[0:s+d]` (novel) vs already-emitted (seen), isolating genuine foreshadowing from copying.
- Standard `./evaluator/task-eval --gpus 4,5` after instrumentation: target 93.3% / 100.7 tok/s,
  mtp-spec 93.3% / 150.2 tok/s / 72.4% accept / **1.49x** (matches 1.51x reference; build/pipeline
  unaffected by instrumentation).

## Results

### Foreshadowing decay (pooled, top-40 window)
| d | hit% | novel-hit% | seen-hit% |
|---|------|-----------|-----------|
| 0 (sampled) | 100.0 | 100.0 | 100.0 |
| 1 | 44.3 | 43.1 | 44.7 |
| 2 | 29.2 | 22.7 | 31.9 |
| 3 | 22.5 | 17.9 | 24.3 |
| 4 | 21.0 | 13.1 | 24.1 |
| 5 | 18.1 | 10.8 | 20.9 |
| 8 | 17.4 | 8.5 | 20.9 |
| 10 | 16.8 | 7.4 | 20.3 |
| 15 | 16.6 | 6.8 | 20.3 |
| 20 | 17.4 | 6.7 | 21.2 |

- Frequency baseline (random realized token in `D_s` top-40): **15.1%**; novel-only baseline: **8.1%**.
- d=0 (the emitted token) has mean prob 0.872 — the model is highly peaked on what it samples.

### Key facts
- **The single forward pass foreshadows novel future tokens.** At d=1 the next *novel* token is
  in `D_s`'s top-40 43.1% of the time vs an 8.1% random-novel baseline (~5.3x lift). Because this
  is measured on tokens absent from prior context, it is not n-gram copy.
- **The foreshadow horizon is short (~3–5 tokens).** Novel-token hit decays 43.1% (d1) → 22.7%
  (d2) → 17.9% (d3) → 13.1% (d4) → 10.8% (d5), reaching the 8.1% baseline by d≈10. Seen-token hit
  stays ~20% at all d (frequency/copy effect, not foreshadowing).
- **Foreshadowed future tokens are mid-ranked, never the argmax.** Mean rank of a hit ≈ 12.
  Novel d=1 tokens land in top-1 only 0.7% of the time, top-5 14.3%, top-10 23.9%, top-40 43.1%.
  The future token is "on the radar" of the current pass but not its leading candidate.
  | dist | top1 | top5 | top10 | top40 |
  |---|---|---|---|---|
  | 1 | 0.7 | 14.3 | 23.9 | 43.1 |
  | 2 | 0.3 | 6.8 | 10.8 | 22.7 |
  | 3 | 0.2 | 5.3 | 8.8 | 17.9 |
  | 4 | 0.2 | 3.5 | 5.7 | 13.1 |
  | 5 | 0.0 | 2.9 | 4.9 | 10.8 |
- **Content-independent.** Novel d=1 foreshadow (top-40) vs random-novel baseline:
  aime 42.8% / 8.1% (5.3x), lcb 46.0% / 8.0% (5.8x), gpqa 41.2% / 7.2% (5.7x),
  hle 41.9% / 5.6% (7.5x). Effect size is uniform across math, code, and MCQ.
- Conditional probability of a hit *rises* with d (0.020 at d1 → 0.12–0.14 at d≥4): at long
  range the only future tokens still in the top-40 are high-frequency tokens that carry large
  `D_s` mass, so conditional-prob is a frequency-confounded metric; hit-rate and the
  novelty-split are the clean signals.

## Artifacts
- `initial_program.cpp` — added `--dump-horizon` + `topk_raw()` (target mode only).
- `results/horizon_target.jsonl` — 30 prompts × per-position top-40 of `D_p` + realized tokens.
- `results/task_eval.json`, `results/raw_horizon.txt` — eval + instrumented run outputs.

---
# (merged reference) researcher-25 findings (single-next-token target uncertainty):

# Findings — content-resolved predictive-uncertainty structure of the target

Lens: the *target model's own* next-token uncertainty, measured target-only (no draft, no
acceptance), and how that uncertainty is organized by the **lexical/semantic content** of the
token being produced. Orthogonal to acceptance, timing, output length/location, hidden-state
geometry, prompt structure, and determinism.

## Method
- Instrumented `initial_program.cpp` (target mode only): at every generated position, computed from
  the raw target logits at the moment of sampling — full-vocab top-1 probability, Shannon entropy
  (nats, at decoding temp 1.0 over the full 262k vocab), and the rank the model assigned to the
  token it actually sampled (0 = argmax). Gated by env `TOK_DUMP`, one JSONL file per process.
- `task-eval` skips `target` (fixed reference), so target mode was driven directly via
  `run_target_dump.py` using the identical chat template / item rendering, GPUs 4,5, seed 0,
  temp 1.0 / top_p 0.95 / top_k 64. All 30 benchmark items, **261,481 generated tokens**.
- Per-token tokens classified by piece content (gemma uses leading-space word starts).

## Results

### 1. Predictive uncertainty is sharply bimodal and concentrated
- Entropy quantiles (nats): p50=0.002, p75=0.351, p90=1.017, p99=2.118, max=3.456.
- **56.6%** of all generated tokens are near-deterministic (H < 0.01 nats); 66.9% have H < 0.1;
  only 10.3% have H > 1.0.
- Uncertainty mass is concentrated: the top 5% most-uncertain tokens carry 33.8% of total entropy,
  top 10% carry 55.8%, top 20% carry 83.7%, top 30% carry 96.8%.
- Holds per item: near-deterministic fraction ranges 27%–72% across the 30 items, always substantial.

### 2. Uncertainty lives almost entirely in natural-language words
Total entropy mass = 71,420 nats. Share of tokens vs share of uncertainty mass:

| class | % of tokens | mean entropy | mean top-1 | % argmax | % of total ENT mass |
|---|---|---|---|---|---|
| word | 39.6% | 0.474 | 0.838 | 84.5% | **68.6%** |
| punct | 21.5% | 0.199 | 0.927 | 93.5% | 15.7% |
| math_markup | 11.6% | 0.208 | 0.923 | 92.8% | 8.8% |
| digit | 14.7% | 0.034 | 0.987 | 98.8% | **1.8%** |
| whitespace | 8.2% | 0.051 | 0.982 | 98.5% | 1.5% |
| other | 4.5% | 0.215 | 0.926 | 95.0% | 3.5% |

- Words are 39.6% of tokens but 68.6% of all predictive uncertainty.
- Digits are 14.7% of tokens but only 1.8% of uncertainty (98.8% argmax, H=0.034); whitespace,
  punctuation, and math/LaTeX markup are likewise near-deterministic.
- The word ≫ digit entropy gap (~10×) holds in every one of the 30 items individually.

### 3. Uncertainty is front-loaded within a word and within a number
- Word-initial pieces (leading-space word start): mean H = 0.558, 81.6% argmax.
- Word-continuation pieces (subword, no leading space): mean H = 0.329, 89.6% argmax.
- Leading digit of a number-run: mean H = 0.043; trailing digits: mean H = 0.014.
- The decision concentrates at the onset (which word, which number); the remainder of the
  word/number is comparatively forced.

### 4. Even under temp-1.0 sampling the chosen token is usually the mode
- Overall 8.9% of tokens were sampled off the argmax (rank > 0); rank p50=0, p90=0, p99=3, max=25.
- Off-mode rate by content: special 27.6%, word 15.5%, math_markup 7.2%, punct 6.5%, digit 1.2%,
  whitespace 1.5%.

## Artifacts
- `results/tokdump_gpu{4,5}.<pid>` and concatenated `results/tokdump_all.jsonl` (per-token JSONL).
- `results/raw_target_gpu{4,5}.txt` (per-item generations), `run_target_dump.py` (driver).
- Target accuracy/throughput unchanged from reference (instrumentation logs only).

---
# (merged reference) researcher-41 findings (stochasticity-tax / diversity-injection):

# Findings — researcher-27

## Lens: linguistic/semantic content of drafted vs. corrected tokens

Orthogonal to acceptance rates, draft-depth/run-length, gamma sweeps, timing, output
length, latent geometry, input structure, and determinism. Question studied: *what kind of
tokens does the MTP draft get wrong, and what does the target substitute?* — the content
identity of draft vs. correction, not the rate at which it happens.

### Method
Instrumented `initial_program.cpp` (mtp-spec mode) to emit one JSON line per verification
round to a side-channel file (`results/content_<gpu>.jsonl`); stdout is unchanged so scoring
is unaffected (full run: accuracy 93.3%, identical to target reference; 1.48x speedup).
Each line records the detokenized piece string of every draft token with its accept flag,
plus the correction token (on a rejection) or bonus token (on full-accept). Post-analysis in
`analyze_content.py` classifies pieces into coarse linguistic categories
(word/punct/digit/space/newline/special) and computes substitution structure.
Full 30-item run on GPUs 6,7: **65,830 rounds — 26,866 corrections, 38,964 full-accept bonus**.

### Core facts

**1. The draft's failure mode is lexical, not structural (near-miss dominance).**
72.4% of corrections are *same-category*: the draft picked the right token KIND but the wrong
specific token (word→word 53%, punct→punct 14%, digit→digit 3%). Only 27.6% are category
errors. The draft reliably knows the local linguistic shape of what comes next; it fails on
the lexical choice within that shape.

**2. Draft is right on scaffolding, wrong on content.**
Accepted draft tokens: 38.8% word, 33.3% punct, 15.0% digit, 8.4% space, 4.5% newline.
Rejected/correction tokens: ~64–67% word, ~25% punct, ~4% digit. Punctuation, whitespace, and
newlines are accepted far above their share of corrections; content words are the dominant
correction category. Digits are accepted well relative to frequency (15% of accepts, 4% of
corrections).

**3. "Same category" does NOT mean "similar spelling."**
81.9% of corrections share ZERO character prefix with the rejected draft; only ~4% share ≥2
chars. The draft proposes a *different word of the same type*, not a typo/morphological
variant. Errors are semantic substitutions, not spelling slips.

**4. Divergences are mostly at word/line boundaries (which-word branch points).**
60.0% of corrections: both rejected-draft and correction begin a new word/line (boundary).
28.0% are both word-internal (mid-word divergence); of those only 17.7% share a stripped
prefix. Branch points are about *which next word/symbol*, not intra-word spelling.

**5. Concrete recurring substitutions** (rejected → correction):
- Formatting ambiguity: `\n`↔`\n\n` (paragraph-break length), `.`↔`,`, `$.`↔`$`/`$,`.
- Function-word / determiner choice: ` the`↔` a`↔` it`↔` this`; `the`→`a` is the single most
  common word substitution (89×).
- Reasoning-step direction: `Wait`↔`Let`, `let`→`I`, `is`→`can`/`means`/`was` — disagreement
  on how to open the next clause/reasoning move.
- Math-mode transition: ` $`↔` the`, ` $`↔` `` ` `` — the draft mis-predicts entry/exit of
  LaTeX math. `$`-bearing tokens: 11,195 accepted vs 2,052 rejected-draft, 1,868 in corrections.
- Digit value guesses: `1`↔`2`, `2`↔`1`, `1`↔`0`/`3` — adjacent-value misses where the draft
  cannot know the computed number.

**6. Domain content profile (correction category mix).**
Math (aime) and code (lcb) corrections are punctuation/digit-heavy (aime 31.2% punct, 7.4%
digit; lcb 28.1% punct, 6.0% digit). Prose multiple-choice (hle 75.9% word, gpqa 72.0% word)
corrections are overwhelmingly word choice. Correction content tracks the symbolic density of
the domain.

**7. Post-divergence drafts stay content-heavy.**
Tokens drafted after the first rejection (discarded, conditioned on a wrong prefix) are 57.5%
word / 26.6% punct / 5.8% digit — the draft keeps proposing content words rather than
collapsing into filler after it goes off-path.

### Artifacts
- `initial_program.cpp` — added side-channel per-round content logging (mtp-spec only).
- `analyze_content.py` — content categorization + substitution-structure analysis.
- `results/content_6.jsonl`, `results/content_7.jsonl` — 65,830 logged rounds.

---

# Findings — researcher-31

## Lens: confidence/entropy structure of divergence points (deeper extension of the content lens)

The content lens established draft errors are semantic "right-kind/wrong-token" substitutions.
This lens goes a level beneath: when the draft LOSES a token, is the target genuinely uncertain
there (an ambiguous branch point the draft lost by chance) or is the target confident (the draft
is simply wrong about a determined token)? And does the draft head's own probability KNOW it is
about to be rejected (calibration)?

### Method
Extended the same side-channel (mtp-spec only; stdout unchanged, so scoring identical: accuracy
93.3% == target reference, 72.4% accept, 1.46x speedup). For each verified draft position the log
now records: `q` (draft-head prob of its own proposal), `px` (target prob of that token),
`acc`=min(1,px/q), target entropy `Ht` & top-prob `pt` over the top-k support, draft entropy `Hd`
& top-prob `qt`, target argmax id, accept flag; plus on rejection the target prob of the
correction (`corr_px`) and whether the correction is the target argmax. All quantities are
computed in the existing accept/reject loop. Post-analysis in `analyze_confidence.py`.
Same full run on GPUs 6,7: 65,830 rounds, 190,764 accepted positions, 26,866 first-reject positions.

### Core facts

**1. The draft head is anti-calibrated in its uncertain regime.**
67.7% of all drafted positions are deterministic agreement (q=px=1, accept 92.3%). In the
remaining uncertain regime (q<0.999), acceptance is NON-MONOTONIC and inverted in draft
confidence: q∈[0,0.3) accepts 64.2%, q∈[0.5,0.7) 54.2%, q∈[0.7,0.85) 52.8%, q∈[0.95,0.999)
46.9%. The MORE confident the draft is (short of total certainty), the LESS likely it is
accepted. AUC of draft prob `q` predicting acceptance is only 0.685 (top-prob 0.686, entropy
0.687 — all near-useless). Mean q at rejection is 0.854: the draft is highly confident even when
wrong.

**2. The TARGET's uncertainty, not the draft's, predicts rejection (AUC 0.882).**
Target entropy: accept 0.149 nats vs reject 0.887 nats. Target top-prob: accept 0.941 vs reject
0.643. Accepted positions are uniformly determined (target entropy <0.26 across every content
category); rejections concentrate where the target distribution itself is flat. The binding
signal lives in the target's verification logits, not the draft's confidence.

**3. Most rejections are NOT the draft being wrong about a settled token.**
By target top-prob at the rejected slot: only 16.0% are "determined" (pt≥0.9), 23.3% strong
(0.7–0.9), 46.3% mid (0.4–0.7), 14.3% ambiguous (pt<0.4). 57.0% of rejected draft tokens were
"plausible" to the target (px≥0.2); only 19.6% were ruled out (px<0.02). The majority of
rejections are coin-flip losses at genuine branch points, not determined-token mistakes.

**4. 38.9% of rejections, the draft had picked the target's own argmax.**
The draft token equals the target argmax in 10,459 of 26,866 rejections — greedy (temp=0)
decoding would accept all of these. They are rejected only because the stochastic lossless
accept test fires when q>px (the draft over-assigned probability to the right token). Over a
third of "draft errors" are artifacts of lossless sampling at temp=1, not the draft being wrong.
The correction equals the target argmax in 45.9% of rejections; mean target prob of the
correction is only 0.440.

**5. Acceptance-loss decomposition at rejection (mean realized px/q = 0.348).**
31.0% target rules the draft out (px<0.10); 37.3% draft overconfidence (px≥0.10 but q≥2·px);
31.7% close ties (px≥0.10, q<2·px, lost the coin flip). The draft systematically
over-concentrates relative to the target at branch points (~3x on average).

**6. Confidence structure tracks content category.**
Rejected words sit at high target entropy (0.985) — semantic free choice. Rejected digits sit
at LOW target entropy (0.345, target top-prob 0.843): when a drafted digit is wrong the target
is fairly sure of the right value, so digit errors are genuine value misses on near-determined
slots (only 22.3% were the target argmax). Newline rejections are 54.5% target-argmax — mostly
stochastic losses on the target's top choice (paragraph-break-length ambiguity, matching the
content lens's \n↔\n\n).

**7. The draft does not degrade with draft depth.**
Across depths g=0..3, mean q rises slightly (0.951→0.962), target px rises (0.846→0.866), and
acceptance rises (87.1%→88.1%). Conditioning the MTP head on its own prior drafts does not erode
calibration within a round; deeper positions are if anything marginally more accept-prone
(consistent with survivorship — rounds reaching deeper g have already cleared the hard slots).

**8. A draft-confidence gate is unfavorable at every threshold.**
Gating first-position drafts on q≥thr drops more good (would-accept) tokens than doomed ones at
all thresholds tested (q≥0.7: drop 1,630 doomed vs 2,600 good; q≥0.9: 3,020 doomed vs 4,287
good). Draft confidence is not a usable adaptive-drafting signal — a direct consequence of the
anti-calibration in fact 1.

### Artifacts
- `initial_program.cpp` — extended the mtp-spec side-channel with per-position confidence stats
  (`ev[]`: q, px, acc, Ht, pt, Hd, qt, target-argmax, accept flag; plus corr_px/corr_argmax).
- `analyze_confidence.py` — calibration, target-uncertainty, rejection-typology, loss
  decomposition, category cross, gating, and depth analyses.
- `results/content_6.jsonl`, `results/content_7.jsonl` — 65,830 rounds with confidence telemetry.

---

# Findings — researcher-41

## Lens: the "stochasticity tax" — measuring the consequence of recovering the artifact rejections

The confidence lens recorded that 38.9% of rejections are "artifacts" of temp=1 sampling — the
draft proposed the target's own argmax and was rejected only because the lossless test fires when
q>px. This lens converts that descriptive fact into measured throughput/accuracy consequences: it
implements two verification rules that *recover* the artifact rejections and runs them end-to-end
on the full 30-item benchmark, then traces the mechanism behind the result.

### Method
Added an `SPEC_VERIFY` env switch to the mtp-spec accept loop in `initial_program.cpp` (stdout/
scoring path unchanged; the side-channel log records the realized accept decisions):
- `lossless` (default): the existing rule, `take = uni < min(1,px/q)`; residual correction `p−q`.
- `greedy`: temp→0 everywhere; `take = (draft==target argmax)`; correction/bonus = target argmax
  (canonical greedy speculative decoding; exactly lossless to greedy target decoding).
- `argmax_sticky`: temp=1 throughout, but `take = (draft==argmax) OR (uni<acc)` — force-accept the
  artifact class, otherwise the unchanged stochastic test + residual correction. It differs from
  `lossless` ONLY by accepting the 38.9% artifact rejections, so it is a controlled isolation of
  that one class. Three full evals on GPUs 2,3 plus offline analysis (`analyze_tax.py`) of the
  fresh lossless logs (65,830 rounds, identical count to the parent run).

### Core facts

**1. Measured accuracy/throughput frontier (full 30-item benchmark, GPUs 2,3).**
| variant        | accept | tok/s | speedup | accuracy | mean decoded | runaway (≥16k)/30 |
|----------------|--------|-------|---------|----------|--------------|-------------------|
| lossless temp=1| 72.4%  | 150.0 | 1.49×   | 93.3%    | 8,553        | 1                 |
| argmax_sticky  | 85.0%  | 166.5 | 1.65×   | 83.3%    | 10,917       | 4                 |
| greedy         | 85.7%  | 167.4 | 1.66×   | 53.3%    | 13,580       | 18                |
The artifact rejections ARE recoverable for throughput: both recovery rules capture nearly the same
gain (accept 72.4→85%, speedup 1.49→1.65×). Recovery is NOT free — accuracy falls monotonically with
how hard the rule biases toward the mode (93.3 → 83.3 → 53.3%).

**2. The drop is degeneration, not wrong answers — output length balloons.**
Mean decoded tokens 8,553 → 10,917 → 13,580; items hitting the 16,384 cap 1 → 4 → 18; greedy median
output IS the cap (16,384). Damage concentrates on the longest-generation domain: code (lcb) 10/10 →
7/10 → 3/10, math (aime) 4/5 → 4/5 → 2/5, while short MCQ (gpqa) holds 10/10 → 10/10 → 8/10. Mode-
seeking makes this reasoning model loop and never emit its boxed answer; truncation at the cap scores
it wrong.

**3. The artifact rejections are mechanically forced OFF-mode steps, not waste.**
On the fresh lossless run: 217,630 evaluated positions, 190,764 accepted (87.7%), 26,866 first-
rejects. Of rejections, 38.9% are artifacts (draft==argmax, reconfirming the parent exactly) and
61.1% genuine misses. When the lossless rule rejects an artifact, the correction it emits is the
argmax in **0.0%** of cases — it ALWAYS steps off the mode. This is mechanically forced: an artifact
reject means px<qx on the argmax token (verified 100% of the 10,459 cases), so the residual `p−q`
clamps that token to zero mass and it can never be resampled. By contrast, genuine misses correct
TOWARD the argmax 75.1% of the time (mean corr_px 0.591). The two rejection classes have opposite
roles: genuine = error correction (toward the mode), artifact = diversity injection (away from it).

**4. Clean causal attribution via the isolated intervention.**
`argmax_sticky` changes lossless by exactly one thing — suppressing the 38.9% diversity-injection
rejections — and that alone costs 10 accuracy points (93.3→83.3%) while delivering the full 1.65×
throughput. The further collapse to 53.3% under greedy is the additional cost of forcing the genuine-
miss corrections and the bonus tokens onto the mode as well. The "wasted" artifact rejections are the
load-bearing entropy that keeps the reasoning trajectory from degenerating.

**5. Recoverable-throughput accounting.**
Of all 195,015 draft tokens that were the target argmax, only 5.4% (10,459) are rejected, yet they
constitute 38.9% of all rejections; accepting them in place lifts the position accept-rate 87.7→92.5%.
14.9% of rounds have an artifact as their first reject (the point past which greedy would extend the
run). This is the size of the throughput pool that mode-recovery taps — and that the accuracy results
show cannot be drawn down without degrading this benchmark.

### Artifacts
- `initial_program.cpp` — `SPEC_VERIFY` env switch (lossless / greedy / argmax_sticky) in the accept loop.
- `analyze_tax.py` — artifact/genuine rejection split, off-mode-correction mechanism, run-length bound.
- `results/mode_lossless.json`, `results/mode_greedy.json`, `results/mode_sticky.json` — per-item evals.
- `results/content_lossless_*.jsonl` (65,830 rounds), `content_greedy_*`, `content_sticky_*` — telemetry.
