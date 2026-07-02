# Output-shape and scoring analysis

Ran `./evaluator/task-eval --gpus 0,1` on branch `researcher-14` with the default 30-item benchmark and default `mtp-spec` settings.

## Score parity

- `mtp-spec` scored 28/30 correct.
- The fixed target reference also scores 28/30 correct.
- There were no per-item correctness differences between this `mtp-spec` run and the fixed target reference.
- The two incorrect items in both runs were `aime-11-15-01` and `hle-chem-04`.

## Decoded output length by source

| source | n | correct | total decoded | mean decoded | median decoded | min | max | at/over 16384-token cap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| aime | 5 | 4 | 55571 | 11114.2 | 11355 | 5329 | 16385 | 1 |
| gpqa | 10 | 10 | 59165 | 5916.5 | 3841.0 | 1427 | 14537 | 0 |
| hle | 5 | 4 | 42096 | 8419.2 | 8072 | 5319 | 12292 | 0 |
| lcb | 10 | 10 | 99748 | 9974.8 | 10487.0 | 2507 | 13041 | 0 |

The `mtp-spec` run decoded 256580 total tokens across the 30 items. The fixed target reference records 261481 total decoded tokens.

## Stop and extraction outcomes

- `aime-11-15-01` decoded 16385 tokens in the `mtp-spec` run, had no `\boxed{...}` answer, and was incorrect. The fixed target reference decoded 16384 tokens for the same item and was also incorrect.
- `hle-chem-04` decoded 10441 tokens, ended with an explicit `Answer: G` followed by an EOS marker, and was incorrect because the gold answer is `F`.
- The other four AIME items each had exactly one boxed final integer, and each matched its gold answer.
- All 15 GPQA/HLE multiple-choice outputs contained an explicit answer line. Fourteen matched gold; `hle-chem-04` did not.
- All 10 LiveCodeBench outputs contained an extractable final fenced code block. The extracted code blocks ranged from 35 to 89 lines, and all 10 passed.

## Length differences versus target reference

Largest absolute decoded-token differences between this `mtp-spec` run and the fixed target reference:

| uid | mtp-spec decoded | target reference decoded | mtp-spec correct | target reference correct |
|---|---:|---:|---|---|
| lcb-medium-08 | 10336 | 16384 | true | true |
| lcb-hard-06 | 13039 | 9050 | true | true |
| aime-6-10-01 | 5329 | 9081 | true | true |
| hle-chem-04 | 10441 | 13382 | false | false |
| gpqa-phys-00 | 2698 | 5535 | true | true |
