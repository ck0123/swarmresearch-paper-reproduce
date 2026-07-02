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
