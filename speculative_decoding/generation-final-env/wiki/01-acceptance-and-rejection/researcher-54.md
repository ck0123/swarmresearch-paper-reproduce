# Findings

## 2026-06-23: MTP acceptance and throughput by benchmark source

Command run from `.worktrees/researcher-2`:

```bash
./evaluator/task-eval --gpus 2,3
```

The run evaluated `mtp-spec` on all 30 benchmark items with `gamma=4`, `seed=0`, and `n=16384`.
The fixed target reference in `vanilla_references/target.json` reports 93.3% accuracy and 100.714 mean tok/s.
This `mtp-spec` run reported 93.3% accuracy, 150.017 mean tok/s, 72.446% aggregate acceptance, and 1.49x speedup.

Per-source aggregates from `results/mode_mtp-spec.json`:

| source | n | accuracy | mean tok/s | accept rate | mean decoded | max decoded |
|---|---:|---:|---:|---:|---:|---:|
| aime | 5 | 80.0% | 165.22 | 81.72% | 11114.2 | 16385 |
| gpqa | 10 | 100.0% | 145.85 | 68.59% | 5916.5 | 14537 |
| hle | 5 | 80.0% | 132.84 | 61.16% | 8419.2 | 12292 |
| lcb | 10 | 100.0% | 155.17 | 75.59% | 9974.8 | 13041 |

Across the 30 items, per-item acceptance rate and per-item tok/s had Pearson correlation 0.9895.
Per-item decoded length and per-item tok/s had Pearson correlation 0.2587.
Per-item decoded length and per-item acceptance rate had Pearson correlation 0.3735.

The eight slowest items by tok/s were also the eight lowest-acceptance items:

| uid | source | correct | decoded | tok/s | accept rate |
|---|---|---:|---:|---:|---:|
| hle-othe-03 | hle | true | 5972 | 119.77 | 50.84% |
| hle-phys-00 | hle | true | 5319 | 121.98 | 52.13% |
| gpqa-chem-01 | gpqa | true | 3702 | 127.11 | 54.74% |
| hle-chem-04 | hle | false | 10441 | 128.46 | 57.56% |
| gpqa-biol-02 | gpqa | true | 4952 | 133.86 | 59.35% |
| gpqa-phys-06 | gpqa | true | 6136 | 134.50 | 60.95% |
| gpqa-biol-00 | gpqa | true | 1427 | 135.88 | 58.55% |
| gpqa-biol-01 | gpqa | true | 3677 | 136.92 | 61.64% |

The two incorrect items in this run were `aime-11-15-01` and `hle-chem-04`.
`aime-11-15-01` decoded 16385 tokens, one token above the configured `n=16384` cap reported to the evaluator.

## 2026-06-24: MTP acceptance decomposes into conditional distribution overlap and gamma-depth compounding

Command run from `.worktrees/researcher-54` after adding diagnostic counters to `initial_program.cpp`:

```bash
./evaluator/task-eval --gpus 2,3
```

The run evaluated `mtp-spec` on all 30 benchmark items with `gamma=4`, `seed=0`, and `n=16384`.
The fixed target reference reports 93.3% accuracy and 100.7 mean tok/s.
This instrumented `mtp-spec` run reported 93.3% accuracy, 146.5 mean tok/s, 72.4% aggregate acceptance, and 1.45x speedup.

The instrumentation measured, for each verified draft position, the overlap
`sum_x min(p_target(x), q_draft(x))` between the target distribution and the MTP draft distribution after the same top-k/top-p truncation.
For a draft token sampled from `q_draft`, this overlap is the expected single-position rejection-sampling acceptance probability.

Across all 30 items, the conditional acceptance probability at a reached verification position was about 0.87-0.88 at each of the four draft depths:

| draft depth | reached attempts | observed conditional accept | mean overlap | target mass on draft support | draft mass on target support | sampled draft outside target support | top-1 agreement |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 65830 | 87.07% | 86.96% | 90.16% | 98.33% | 1.68% | 90.30% |
| 2 | 57316 | 87.64% | 87.41% | 90.40% | 97.80% | 2.23% | 89.95% |
| 3 | 50234 | 88.09% | 88.08% | 90.93% | 97.37% | 2.59% | 90.07% |
| 4 | 44250 | 88.05% | 88.08% | 90.75% | 96.93% | 3.07% | 89.90% |

The aggregate 72.4% acceptance reported by the evaluator is lower than the 87-88% conditional single-position overlap because `gamma=4` compounds early rejections.
Using the observed conditional accept rates, the expected accepted drafts per round are:

`0.870667 + 0.870667*0.876439 + 0.870667*0.876439*0.880877 + 0.870667*0.876439*0.880877*0.880542 = 2.897828`

Dividing by four drafted tokens gives 72.4457%, matching the evaluator's aggregate acceptance.
The observed accept-depth histogram over 65,830 speculative rounds was:

| accepted drafts in round | rounds | fraction |
|---:|---:|---:|
| 0 | 8514 | 12.93% |
| 1 | 7082 | 10.76% |
| 2 | 5984 | 9.09% |
| 3 | 5286 | 8.03% |
| 4 | 38964 | 59.19% |

Per-item aggregate acceptance had Pearson correlation 0.996 with weighted mean target/draft overlap.
Per-item tok/s had Pearson correlation 0.983 with weighted mean target/draft overlap.
Per-item aggregate acceptance also correlated with top-1 agreement (0.983), target entropy (-0.974), draft entropy (-0.980), draft mass on target support (0.895), and sampled drafts outside target support (-0.891).

The five lowest-overlap items had 0.7696 weighted mean overlap, 54.55% aggregate acceptance, 123.90 mean tok/s, and 35.71% full-accept rounds.
The five highest-overlap items had 0.9252 weighted mean overlap, 83.13% aggregate acceptance, 165.15 mean tok/s, and 73.38% full-accept rounds.

| group | n | mean tok/s | aggregate acceptance | weighted overlap | full-accept rounds | avg accepted drafts/round | avg output tokens/verify round |
|---|---:|---:|---:|---:|---:|---:|---:|
| lowest-overlap 5 | 5 | 123.90 | 54.55% | 76.96% | 35.71% | 2.1818 | 3.1818 |
| middle 20 | 20 | 147.48 | 73.50% | 88.10% | 60.53% | 2.9398 | 3.9398 |
| highest-overlap 5 | 5 | 165.15 | 83.13% | 92.52% | 73.38% | 3.3252 | 4.3252 |

The lowest-overlap items were the same neighborhood as the slowest/lowest-acceptance items from the earlier analysis.
Their per-position overlaps were still mostly in the mid-0.7s, but compounding across four drafted tokens produced aggregate acceptance near 0.5-0.6:

| uid | source | tok/s | aggregate acceptance | weighted overlap | full-accept rounds | per-depth overlap |
|---|---|---:|---:|---:|---:|---|
| hle-othe-03 | hle | 117.78 | 50.84% | 74.89% | 31.49% | 0.743, 0.753, 0.757, 0.744 |
| hle-phys-00 | hle | 119.25 | 52.13% | 75.35% | 33.18% | 0.761, 0.744, 0.749, 0.760 |
| gpqa-chem-01 | gpqa | 123.68 | 54.74% | 77.14% | 35.23% | 0.774, 0.779, 0.771, 0.754 |
| gpqa-biol-00 | gpqa | 132.67 | 58.55% | 78.65% | 39.58% | 0.776, 0.805, 0.772, 0.796 |
| hle-chem-04 | hle | 126.14 | 57.56% | 78.69% | 39.37% | 0.792, 0.781, 0.784, 0.788 |

The low-overlap group had mean target probability 0.7344 on sampled draft tokens and mean draft probability 0.9085 on those same sampled tokens.
The high-overlap group had mean target probability 0.9136 and mean draft probability 0.9754.
The mean target entropy minus draft entropy gap was 0.2705 in the low-overlap group and 0.1004 in the high-overlap group.
This run therefore showed that lower-acceptance items corresponded to target distributions being less concentrated and less aligned with the sharper MTP draft distribution.
