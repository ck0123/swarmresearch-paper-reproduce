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
