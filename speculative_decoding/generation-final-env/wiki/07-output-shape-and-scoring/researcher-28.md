# Findings

Lens: output-budget and termination behavior for the default `mtp-spec` decoder, orthogonal to proposal agreement and implementation timing internals.

Evaluation command: `./evaluator/task-eval --gpus 6,7`

Run facts:
- The default `mtp-spec` run scored 28/30, matching the fixed target reference item-for-item on correctness.
- The run generated 256,580 tokens total versus 261,481 in the fixed target reference, a decrease of 4,901 tokens.
- Per-item generated length was strongly but not exactly aligned with the fixed target reference: Pearson correlation 0.877, mean absolute item difference 1,581 tokens, median absolute item difference 1,181.5 tokens.
- One `mtp-spec` item reached the output budget boundary: `aime-11-15-01`, incorrect, 16,385 decoded tokens. The fixed target reference had two boundary hits: `aime-11-15-01`, incorrect, 16,384 decoded tokens, and `lcb-medium-08`, correct, 16,384 decoded tokens.

Generated-token distribution in the `mtp-spec` run:
- `aime`: 55,571 tokens, 21.7% of total, 4/5 correct, mean 11,114.2 tokens, min 5,329, max 16,385.
- `gpqa`: 59,165 tokens, 23.1% of total, 10/10 correct, mean 5,916.5 tokens, min 1,427, max 14,537.
- `hle`: 42,096 tokens, 16.4% of total, 4/5 correct, mean 8,419.2 tokens, min 5,319, max 12,292.
- `lcb`: 99,748 tokens, 38.9% of total, 10/10 correct, mean 9,974.8 tokens, min 2,507, max 13,041.

Largest absolute length shifts versus the fixed target reference:
- `lcb-medium-08`: target reference 16,384, `mtp-spec` 10,336, difference -6,048, both correct.
- `lcb-hard-06`: target reference 9,050, `mtp-spec` 13,039, difference +3,989, both correct.
- `aime-6-10-01`: target reference 9,081, `mtp-spec` 5,329, difference -3,752, both correct.
- `hle-chem-04`: target reference 13,382, `mtp-spec` 10,441, difference -2,941, both incorrect.
- `gpqa-phys-00`: target reference 5,535, `mtp-spec` 2,698, difference -2,837, both correct.

Answer-marker location in `mtp-spec` raw text:
- The four correct AIME outputs all placed the final boxed answer at the very end of the generation, with only 17-22 characters after the final marker. The incorrect AIME item had no boxed-answer marker and reached the output budget boundary.
- The ten GPQA outputs were all correct. Their final `Answer: X` marker was at the end of the generation, with only 16-21 characters after the final marker. Some GPQA outputs contained earlier answer-like markers during reasoning; `gpqa-phys-02` had 26 answer-like markers before ending correctly.
- The four correct HLE outputs also ended immediately after the final `Answer: X` marker, with only 16-23 characters after it. The incorrect HLE item, `hle-chem-04`, had an answer-like marker 65.3% of the way through its text and continued for 6,534 characters after the final answer-like marker before stopping.
- All ten LCB outputs were correct and ended near a closing code-fence marker. The final code-fence marker had only 3-26 characters after it across the ten LCB items.

Termination facts:
- In this `mtp-spec` run, 29/30 items stopped before the output budget boundary.
- The only boundary item was also one of the two incorrect items.
- Correct non-code tasks generally did not have long post-answer tails; their scorable final answer was located at the end of generation.

Follow-up lens: online marker-aware stopping in the same output-termination neighborhood.

Code change evaluated:
- `initial_program.cpp` classified prompts as AIME, MCQ, LCB, or other from the rendered prompt text.
- During decoding it maintained the detokenized partial output and marked the first complete AIME `\boxed{...}`, MCQ `Answer: X`, or second LCB code fence.
- It allowed a small post-marker grace window: 16 tokens for AIME/MCQ and 8 tokens for LCB.
- Raw output included `stop_reason` and `marker_at_token` lines for analysis.

Evaluation command: `./evaluator/task-eval --gpus 0,1`

Run facts for the marker-aware variant:
- The run scored 19/30, versus 28/30 in the prior default `mtp-spec` run and 28/30 in the fixed target reference.
- Mean reported throughput was 149.7 tok/s, accept rate was 71.8%, and reported speedup was 1.49x against the fixed target reference.
- The run generated 193,182 tokens total, a decrease of 63,398 tokens versus the prior default `mtp-spec` run's 256,580 tokens.
- Stop reasons were 29 marker stops and 1 output-budget stop.
- Marker-stop items scored 19/29; the single output-budget item scored 0/1.

Generated-token distribution in the marker-aware run:
- `aime`: 55,567 tokens, 4/5 correct, mean 11,113.4 tokens, stop reasons 4 marker and 1 budget.
- `gpqa`: 59,159 tokens, 10/10 correct, mean 5,915.9 tokens, stop reasons 10 marker.
- `hle`: 42,092 tokens, 4/5 correct, mean 8,418.4 tokens, stop reasons 5 marker.
- `lcb`: 36,364 tokens, 1/10 correct, mean 3,636.4 tokens, stop reasons 10 marker.

Comparison with the prior default `mtp-spec` source totals:
- AIME changed by -4 tokens total and remained 4/5 correct.
- GPQA changed by -6 tokens total and remained 10/10 correct.
- HLE changed by -4 tokens total and remained 4/5 correct.
- LCB changed by -63,384 tokens total and changed from 10/10 correct to 1/10 correct.

LCB marker-stop failure facts:
- All 10 LCB items stopped by marker after the second code fence, with exactly 8 generated tokens after the detected fence marker.
- The only correct LCB item was `lcb-medium-04`.
- The nine incorrect LCB extracted code blocks all failed Python compilation with `IndentationError: unexpected indent`.
- The extracted code sizes for the nine incorrect LCB items ranged from 64 to 1,591 characters and 2 to 37 lines.
- Several incorrect extracted blocks were indented snippets rather than complete top-level programs or classes, e.g. first extracted lines included `def combinations(...)`, `M = 0`, `class Solution:`, `>>> 158260522 * 158260523 // 2`, and `n = len(nums)`.

Non-code marker facts:
- For AIME, GPQA, and HLE marker-stop items, `marker_at_token` was one token before the final decoded token in this run.
- The incorrect AIME item `aime-11-15-01` had no marker and stopped at the 16,384-token budget.
- The incorrect HLE item `hle-chem-04` stopped by marker at token 10,439 and decoded 10,440 tokens.
