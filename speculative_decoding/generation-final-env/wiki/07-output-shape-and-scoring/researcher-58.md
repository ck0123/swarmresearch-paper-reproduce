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

# Seed-stability extension

Ran three full default `mtp-spec` evaluations against the same fixed target reference using `./evaluator/task-eval --gpus 2,3 --seed N` for seeds 0, 1, and 2. Result snapshots are in `analysis_runs/seed_stability/seed_{0,1,2}/`.

## Aggregate stability

| seed | correct | accuracy | mean tok/s | accept rate | total decoded | differs from target reference | incorrect items |
|---:|---:|---:|---:|---:|---:|---|---|
| 0 | 28/30 | 93.3% | 150.5 | 72.4% | 256580 | none | `aime-11-15-01`, `hle-chem-04` |
| 1 | 29/30 | 96.7% | 148.9 | 72.3% | 267753 | `aime-11-15-01`, `hle-chem-04`, `hle-math-02` | `hle-math-02` |
| 2 | 29/30 | 96.7% | 149.1 | 72.0% | 258194 | `hle-othe-03`, `aime-11-15-01`, `hle-chem-04` | `hle-othe-03` |

The seed-0 `mtp-spec` correctness vector exactly matched the fixed target reference. The seed-1 and seed-2 `mtp-spec` correctness vectors did not exactly match the fixed target reference.

## Item-level changes

| uid | target reference | seed 0 | seed 1 | seed 2 | decoded tokens target/0/1/2 |
|---|---|---|---|---|---|
| `aime-11-15-01` | wrong | wrong | correct | correct | 16384/16385/12808/14411 |
| `hle-chem-04` | wrong | wrong | correct | correct | 13382/10441/11270/13368 |
| `hle-math-02` | correct | correct | wrong | correct | 10017/12292/16385/11879 |
| `hle-othe-03` | correct | correct | correct | wrong | 4782/5972/5296/4510 |

The two fixed-target-reference failures, `aime-11-15-01` and `hle-chem-04`, were also failures for `mtp-spec` seed 0 and were correct for `mtp-spec` seeds 1 and 2. In each of seeds 1 and 2, one HLE item that was correct in the target reference was incorrect for `mtp-spec`; the affected HLE item differed by seed.
