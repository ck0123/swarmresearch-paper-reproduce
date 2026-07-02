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
