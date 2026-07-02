# Findings

## Hidden-state handoff geometry

Instrumentation added to `initial_program.cpp` records the geometry of the MTP assistant hidden payloads produced by `llama_get_embeddings_nextn_ith(ctx_dft, 0)`. For each proposed token, the run compares that payload with two target-side vectors from the verification decode:

- `hidden_handoff_predictor`: target `nextn` embedding row `g`, the row whose logits score the proposed token.
- `hidden_handoff`: target `nextn` embedding row `g + 1`, the following row used as the continuation-side target comparison.

A target-only control, `hidden_target_redecode`, compares the carried target `nextn` payload with target row `0` when the same boundary is redecoded.

Evaluation command: `./evaluator/task-eval --gpus 0,1`.

Final artifact: `results/mode_mtp-spec.json`.

Across 263,320 assistant/target vector comparisons:

| comparison | count | mean cosine | min cosine | mean relative L2 | mean absolute log norm ratio |
|---|---:|---:|---:|---:|---:|
| assistant vs target predictor row | 263,320 | -0.030706 | -0.300848 | 1.562374 | 0.208245 |
| assistant vs target following row | 263,320 | -0.075933 | -0.312371 | 1.607864 | 0.230315 |
| target carried payload vs target redecode control | 65,830 | 0.594498 | 0.170991 | 0.922109 | 0.174925 |

By source, assistant-vs-target predictor-row mean cosine stayed near zero or negative:

| source | count | mean cosine | mean relative L2 | mean absolute log norm ratio |
|---|---:|---:|---:|---:|
| aime | 52,076 | -0.024614 | 1.630183 | 0.238338 |
| gpqa | 63,224 | -0.037047 | 1.507425 | 0.191482 |
| hle | 48,860 | -0.043982 | 1.492806 | 0.179645 |
| lcb | 99,160 | -0.023322 | 1.596076 | 0.217222 |

For the following-row comparison, mean cosine by assistant rollout depth was also negative:

| rollout depth | count | mean cosine | min cosine | mean relative L2 |
|---:|---:|---:|---:|---:|
| 0 | 65,830 | -0.088030 | -0.312371 | 1.590048 |
| 1 | 65,830 | -0.075888 | -0.281720 | 1.627429 |
| 2 | 65,830 | -0.071150 | -0.270817 | 1.611921 |
| 3 | 65,830 | -0.068665 | -0.262896 | 1.602057 |

Per-item mean cosine ranges:

| comparison | minimum per-item mean | maximum per-item mean |
|---|---:|---:|
| assistant vs target predictor row | -0.060456 | -0.014041 |
| assistant vs target following row | -0.096667 | -0.055922 |

