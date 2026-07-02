# Findings

## Output commitment analysis

Evaluation command: `./evaluator/task-eval --gpus 2,3`.

The `mtp-spec` run completed all 30 benchmark items with accuracy `0.9333333333333333` and mean throughput `150.04933333333332` tok/s. The fixed target reference in `vanilla_references/target.json` has accuracy `0.9333333333333333` and mean throughput `100.714` tok/s.

`analyze_output_commitment.py` parses `results/raw_mtp-spec_gpu*.txt` and reuses the existing benchmark scorers. For AIME and MCQ tasks it evaluates scorer state at answer-affecting character offsets. For LCB tasks it evaluates prefixes ending at complete fenced-code blocks. Character offsets are measured on detokenized output text, not tokenizer boundaries.

### AIME, GPQA, and HLE text-answer tasks

There were 20 text-answer tasks. Eighteen were finally correct and two were finally incorrect.

For the 18 finally correct text-answer tasks, the first correct-scoring prefix appeared much earlier than the final stable correct-scoring prefix. The median first-correct position was `0.14924606649871283` of the output characters. The median stable-correct position was `0.9989442648328797` of the output characters. The median tail after the stable-correct position was `16.5` characters; the total tail after stable-correct positions across those 18 tasks was `1217` characters, with maximum `404` characters.

Both finally incorrect text-answer tasks had a correct-scoring prefix before the final output became incorrect under the benchmark scorer:

- `aime-11-15-01`: first correct-scoring prefix ended at character `3246` of `33084`.
- `hle-chem-04`: first correct-scoring prefix ended at character `7610` of `39274`.

Examples of early first-correct but late stable-correct text-answer outputs:

- `hle-phys-00`: first correct at character `449` of `22041`; stable correct at character `21637`.
- `hle-othe-03`: first correct at character `601` of `21080`; stable correct at character `21070`.
- `hle-math-02`: first correct at character `1676` of `36505`; stable correct at character `36491`.

### LiveCodeBench tasks

There were 10 LCB tasks and all 10 final outputs passed the LCB scorer.

Each LCB output contained multiple complete fenced-code blocks. Fence-block counts were `[3, 5, 5, 7, 8, 10, 11, 12, 14, 14]`.

All 10 LCB outputs had at least one passing complete fenced-code prefix. The median first-passing fenced-code prefix position was `0.999160903008178` of output characters. Tail lengths after the first passing fenced-code prefix were `[7, 10, 10, 13, 20, 23, 2421, 3939, 4873, 8756]` characters. The median tail after the last fenced-code close was `10.0` characters.

## Commitment-trajectory analysis (explicit commitments vs incidental matches)

Evaluation command: `./evaluator/task-eval --gpus 2,3`. This run reported `mtp-spec` accuracy `0.9333333333333333`, mean throughput `150.09533333333331` tok/s, acceptance `72.4%`, speedup `1.49x`.

The prior output-commitment analysis used the scorer's loose matching: the AIME scorer falls back to "last integer mentioned" anywhere in the text, and the MCQ analysis matched any standalone `[A-J]`. `analyze_commitment_trajectory.py` instead tracks only *explicit* answer commitments using the scorer's marker priority (`\boxed{...}` for AIME; `Answer: X` lines and `\boxed{X}` for MCQ), records the ordered sequence of committed values, and measures the genuine lock-in offset (the offset after which every explicit commitment equals the final committed value). Token tails are estimated with a linear character-to-token proxy from each item's decoded token count.

### Text-answer tasks: a single, very late, stable commitment

Of 20 text tasks, 19 produced explicit answer commitments and 1 produced none. The median number of explicit commitments per task was `1`, the median number of committed-value changes was `0`, and only `1` of 20 tasks ever changed its committed value.

Across the 19 tasks with a commitment, the first explicit commitment appeared at median character fraction `0.9995256166982922` of the output (every per-task first-commitment fraction was `>= 0.9988` except the one task noted below). The genuine lock-in fraction equaled the first-commitment fraction for all single-commitment tasks. Estimated tokens generated after lock-in summed to `67` tokens out of `156832` decoded text tokens (`0.00043` of decoded text tokens).

This places the model's explicit answer inside the final ~0.2% of the output. The earlier "first correct at median character fraction `0.149`" reported in the prior section came from the scorers' fallback matching catching incidental integers and `[A-J]` letters in the reasoning, not from an explicit commitment.

Specific tasks:

- `hle-math-00` was the only text task with a committed-value change. It wrote `D` at character `21734` of `26834` (mid-reasoning line "This would make the answer D") and then committed `B` at character `26822` ("Answer: B"); gold is `B` and the final output is correct. A policy that exited at the first explicit marker would have scored `D` (incorrect) here; this is the only first-marker early-exit verdict flip among the 19 tasks with a commitment.
- `hle-chem-04` made a single explicit commitment `G` (gold `F`); it never explicitly committed the correct answer. The prior section's "first correct-scoring prefix at character `7610`" for this task corresponds to an incidental `F` letter in the reasoning under loose matching.
- `aime-11-15-01` produced zero explicit commitments. Its `33084`-character output contains no `\boxed{...}` and ends mid-sentence ("...must satisfy $Q_1(S) -"), i.e. generation reached the token cap without committing an answer.

No text task explicitly committed the correct answer and then revised away from it (`correct_then_revised` was empty).

### LiveCodeBench tasks: many dissimilar blocks, late single pass

The 10 LCB tasks had a median of `9` complete fenced-code blocks and a median of `1` passing block. Scoring each complete block individually (each block extracted as the last block of its own text) gave these per-block pass trajectories:

- `lcb-hard-01`: `[F,F,F,F,F,T,T]` (7 blocks)
- `lcb-hard-04`: `[F,F,F,F,F,F,F,F,F,F,F,F,F,T]` (14)
- `lcb-hard-06`: `[F,F,F,F,F,F,T,F,T,T]` (10)
- `lcb-medium-00`: `[F,F,F,F,F,F,F,T]` (8)
- `lcb-medium-01`: `[F,T,T]` (3)
- `lcb-medium-02`: `[F,F,F,F,F,F,F,F,F,F,F,T]` (12)
- `lcb-medium-04`: `[F,F,F,F,F,F,F,F,F,F,T]` (11)
- `lcb-medium-06`: `[F×13,T]` (14)
- `lcb-medium-07`: `[F,F,F,T,T]` (5)
- `lcb-medium-08`: `[F,F,F,F,T]` (5)

In 6 of 10 tasks the first passing block was the last block (no earlier passing block). The 4 tasks with an earlier passing block had estimated post-first-pass tails of `2129`, `3749`, `813`, and `1693` tokens (`lcb-hard-01`, `lcb-hard-06`, `lcb-medium-01`, `lcb-medium-07`). Estimated tokens after the first passing block summed to `8384` of `99748` decoded LCB tokens (`0.084`).

No task had an earlier passing block followed by a failing final block (`tasks_fail_after_pass` was empty); the last block passed in all 10. `lcb-hard-06` emitted a failing block (index 7) after a passing block (index 6) and then passed again.

The median similarity between consecutive code blocks (difflib ratio) was `0.2255`; consecutive blocks were mostly dissimilar except `lcb-medium-01` (`0.856`). The failing blocks preceding the final passing block are largely distinct code rather than re-emissions of the same solution.
