# Output shape, termination & scoring

The output side, orthogonal to proposal agreement and timing: how long generations are, where the
answer actually commits, how termination behaves, and whether early-stopping rules are safe.

**Cross-cutting result.** mtp-spec scores at parity with the target (28/30, 93.3%) and generates slightly
*fewer* tokens (256,580 vs 261,481; Pearson ~0.88). Correct non-code tasks commit their answer **very late**
(median ~99.9% of characters) and terminate with a short tail; LCB code emits the first passing block near
the end. Only 1 of 30 items hits the 16,384-token budget (and it is the single incorrect one). Marker-aware
early stopping is **not safe**: it saves ~63K tokens but drops accuracy 28→19/30 (LCB 10→1/10) from premature
code truncation — termination is orthogonal to correctness in this regime.

| File | Hook |
|---|---|
| [researcher-10](researcher-10.md) | Output termination & budget alignment: mtp-spec generates 4,901 fewer tokens than target (Pearson 0.877); only 1/30 hits the budget (incorrect); correct non-code tasks stop right after the final answer marker. |
| [researcher-12](researcher-12.md) | Commitment-trajectory analysis: text answers have a single very late commitment (~99.95% of chars), only 1/20 changed value; 6/10 LCB tasks have the first passing block as the last; no task fails after passing. |
| [researcher-13](researcher-13.md) | Output commitment + commitment-trajectory (explicit markers vs incidental matches): where Boxed{}/Answer: markers appear and how stable the committed value is. |
| [researcher-14](researcher-14.md) | Output-shape & scoring: 28/30 correct (parity with target reference); 256,580 tokens decoded; only one item hits the 16,384 cap; one MCQ mismatch (hle-chem-04). |
| [researcher-28](researcher-28.md) | Output-budget & termination with marker-aware stopping: saves 63,398 tokens but drops accuracy 28→19/30 (LCB 10→1/10) via premature code truncation — early stop is unsafe here. |
| [researcher-58](researcher-58.md) | Output-shape & scoring + a seed-stability extension: length-by-source distributions, stop/extraction outcomes, and item-level stability across seeds. |
