# Speculative Decoding Analysis — Findings Wiki

A consolidated wiki of **all 73 researchers'** findings from `.worktrees/researcher-*/findings.md`.

**Objective of the effort.** Design a *novel, training-free* decoding scheme that **maximizes
wall-clock throughput while preserving benchmark accuracy** — i.e. something genuinely faster than
vanilla speculative decoding. The reference `initial_program.cpp` (MTP speculative decoding) is the
**baseline to beat: ~1.63× speedup** over vanilla autoregressive target-only decoding, and any new
scheme **must hold 28/30 = 93.3% accuracy**. The system: **target** `gemma-4-26B-A4B-it` (Q8_0)
verifying drafts from its native **MTP head** (`gemma4-assistant`) under a lossless rejection sampler,
on a 30-item benchmark (LiveCodeBench v6 + AIME 2026 + GPQA Diamond + HLE-MCQ), 1 GPU per run.

The 73 findings below are **reconnaissance toward that design** — each probes the baseline from a
different angle to locate the bottleneck, the headroom, and which ideas can or cannot move
wall-clock without breaking accuracy. (Individual branches report their *own* measured speedups,
typically ~1.46–1.64× depending on setup/levers; read those as relative deltas against each
branch's local baseline, not against the ~1.63× reference.)

> Each `researcher-NN.md` in the subdirectories is the **raw, verbatim** `findings.md` copied from that
> researcher's worktree (renamed and grouped, content unchanged). Each subdirectory's `README.md`
> summarizes the group and links every raw file. Each original worktree (`.worktrees/researcher-NN/`)
> also holds the analysis scripts and any modified `initial_program.cpp` behind that finding.

## The one-paragraph synthesis (what it means for the design)

The system is **verify-bound**: the target's forward pass is ~69% of wall-clock and costs ~7 ms +
2.07 ms/token (MoE expert fan-out), capping the absolute speedup at ~2.1× even with free drafting —
**so a faster scheme must raise accepted-tokens-per-verify, not cheapen drafting.** Acceptance
(~88% conditional, flat across draft depth) is governed by the overlap between target `p` and the
over-confident draft `q`, and that overlap collapses exactly where the **target's entropy `Ht`** is
high. The high-entropy "joints" (~30% of tokens, temporally clustered) are simultaneously where the
draft's support misses, where verification is most wasted, and where the target must keep diversity
for output quality — the same wall, viewed three ways, and the binding constraint on any
accuracy-preserving speedup. The levers that *do* move wall-clock without breaking accuracy are small
and lossless — draft-temperature recalibration, heap `make_dist`, and entropy-gated artifact recovery
(`entropy_sticky`, safe only below `Ht≈1.0`) — and they compose (best observed ~1.64× in-branch);
structural ideas (copy/prompt-lookup drafting, tree drafts, support repair) mostly **fail** against
the verify-bound ceiling. These levers are the most promising building blocks for the novel scheme.

## Where is what

| # | Group | What's in it | Files |
|---|---|---|---|
| 01 | [Acceptance & rejection sampling](01-acceptance-and-rejection/README.md) | Headline accept rate, per-depth/per-source decomposition, distributional overlap, sampler temperature | 11 |
| 02 | [Wall-clock latency & cost](02-wall-clock-and-cost/README.md) | Where time goes; verify-bound resolution; MoE/LM-head forward-cost structure; serving/batching | 13 |
| 03 | [Entropy & uncertainty](03-entropy-and-uncertainty/README.md) | Where target uncertainty lives; rejection drivers; foreshadowing; the "stochasticity tax" | 16 |
| 04 | [Draft-head geometry](04-draft-head-geometry/README.md) | Geometry of the MTP hidden-state feedback loop vs the target latent | 5 |
| 05 | [Prompt, KV-cache & determinism](05-prompt-kv-and-determinism/README.md) | Prompt/prefill distribution, attention horizon, numerical reproducibility & cache correctness | 8 |
| 06 | [Context-recoverability & copy drafting](06-context-recoverability-and-copy/README.md) | Copyability / prompt-lookup; free-copy drafter; context-redundancy vs MoE cost | 4 |
| 07 | [Output shape & scoring](07-output-shape-and-scoring/README.md) | Generation length, answer commitment, termination, early-stop safety | 6 |
| 08 | [Optimizations & synthesis](08-optimizations-and-synthesis/README.md) | γ tuning, draft-temp, tree drafts, support repair, multi-lever factorial composition | 10 |

## Key takeaways by theme

- **Verify-bound, ~2.1× ceiling.** Target verification is the hard floor; draft-side cheapness is inert
  for throughput. An early "sampling-bound (95% make_dist)" claim was an async-timing artifact
  (reconciled in [r46](02-wall-clock-and-cost/researcher-46.md) / [r50](02-wall-clock-and-cost/researcher-50.md)).
- **Acceptance ≈ distributional overlap, flat across depth.** No MTP error compounding; the limiter is
  draft over-confidence (~10% loss) far more than out-of-nucleus support miss (~2%).
- **Entropy `Ht` is the master variable.** It predicts rejection (AUC 0.88), and the draft's support
  coverage is a strict decreasing function of it ([r62](03-entropy-and-uncertainty/researcher-62.md)).
- **The stochasticity tax is real.** ~39% of rejections are temp=1 artifacts; recovering them gains
  speed but spends accuracy — safe only below `Ht≈1.0` (entropy_sticky).
- **MTP hidden ⟂ target hidden**, but ~23% linearly recoverable; only norm separates accept/reject.
- **Copy / prompt-lookup drafting doesn't pay off** in the verify-bound regime (0.98–1.47×).
- **Output commits late; early-stopping is unsafe** (28→19/30 if truncated at the first marker).
- **Best composed result ≈ 1.6×** from three lossless levers; two of them are substitutes, not additive.

## Full researcher → group index

| Researcher | Group | Researcher | Group | Researcher | Group |
|---|---|---|---|---|---|
| r1 | 01 | r26 | 01 | r51 | 03 |
| r2 | 01 | r27 | 03 | r52 | 08 |
| r3 | 01 | r28 | 07 | r53 | 05 |
| r4 | 01 | r29 | 01 | r54 | 01 |
| r5 | 01 | r30 | 02 | r55 | 05 |
| r6 | 02 | r31 | 03 | r56 | 03 |
| r7 | 02 | r32 | 03 | r57 | 08 |
| r8 | 02 | r33 | 02 | r58 | 07 |
| r9 | 01 | r34 | 08 | r59 | 08 |
| r10 | 07 | r35 | 06 | r60 | 08 |
| r11 | 08 | r36 | 03 | r61 | 03 |
| r12 | 07 | r37 | 02 | r62 | 03 |
| r13 | 07 | r38 | 06 | r63 | 08 |
| r14 | 07 | r39 | 03 | r64 | 03 |
| r15 | 04 | r40 | 02 | r65 | 08 |
| r16 | 04 | r41 | 03 | r66 | 03 |
| r17 | 04 | r42 | 01 | r67 | 04 |
| r18 | 04 | r43 | 06 | r68 | 02 |
| r19 | 05 | r44 | 03 | r69 | 08 |
| r20 | 05 | r45 | 01 | r70 | 08 |
| r21 | 05 | r46 | 02 | r71 | 02 |
| r22 | 05 | r47 | 03 | r72 | 03 |
| r23 | 05 | r48 | 06 | r73 | 02 |
| r24 | 02 | r49 | 05 | | |
| r25 | 03 | r50 | 02 | | |

*Groups: 01 acceptance · 02 wall-clock · 03 entropy · 04 geometry · 05 prompt/KV/determinism · 06 copy · 07 output · 08 optimizations.*

## Notes on provenance

- Many findings are **cumulative branch files**: a later researcher prepends a parent's findings, then
  adds its own `## researcher-N:` section. Grouping here is by each file's **own** contribution
  (verified against the H1 title and the file's final section), so a file's group may differ from a
  parent it quotes.
- Notable lineages: the *researcher-3* acceptance-depth line (r3→r7→r11→r30), the *researcher-27*
  content/entropy line (r27→r31→r32→r36→r41→r47), the *wall-clock contradiction* synthesis
  (r33/r46/r50), the *entropy_sticky* line (r41→r47→r56→r61→r62), and the *lever-composition* line
  (r52/r50→r59→r65→r69/r70).
