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
