# Findings

## Sampler support-overlap lens

Lens status: genuinely orthogonal to the saturated lenses listed in the task. This run measured the overlap between the target and MTP draft truncated sampling distributions during target verification, not acceptance-rate decomposition, depth/run-length behavior, gamma sweeps, timing/cost mechanisms, output length, hidden-state geometry, prompt structure, determinism/KV replay, linguistic error categories, target-only uncertainty, or copy/recycling behavior.

Instrumentation added to `initial_program.cpp` records, for each verified speculative position, four distribution-support quantities after the shared top-k/top-p/temp truncation:

- `support_p_mass_on_q`: target probability mass lying on the draft sampler support.
- `support_q_mass_on_p`: draft probability mass lying on the target sampler support.
- `support_tv`: total variation distance between the two truncated distributions over the union of their supports.
- `support_jaccard`: set Jaccard overlap of the two truncated supports.

Evaluation command: `./evaluator/task-eval --gpus 6,7`.

Evaluator result on 30 items:

- target reference: accuracy 93.3%, 100.714 tok/s.
- `mtp-spec`: accuracy 93.3%, 149.326 tok/s, speedup 1.48x, evaluator-reported accept 72.4%.

Support-overlap measurements covered 217,630 verified speculative positions across 30 prompts.

Step-weighted aggregate support metrics:

- `support_p_mass_on_q`: 0.905210.
- `support_q_mass_on_p`: 0.976817.
- `support_tv`: 0.124371.
- `support_jaccard`: 0.819613.

Unweighted per-prompt aggregate support metrics:

- mean `support_p_mass_on_q`: 0.898293; range 0.810498 to 0.969535; median 0.913372.
- mean `support_q_mass_on_p`: 0.973808; range 0.924495 to 0.993164; median 0.979336.
- mean `support_tv`: 0.134100; range 0.041519 to 0.251082; median 0.110570.
- mean `support_jaccard`: 0.807987; range 0.646389 to 0.931553; median 0.830081.

Lowest per-prompt total variation cases:

- `gpqa-phys-00`: `support_tv` 0.041519, `support_p_mass_on_q` 0.969535, `support_q_mass_on_p` 0.992683, `support_jaccard` 0.931553.
- `aime-6-10-01`: `support_tv` 0.062879, `support_p_mass_on_q` 0.952077, `support_q_mass_on_p` 0.991376, `support_jaccard` 0.899613.
- `aime-1-5-01`: `support_tv` 0.074376, `support_p_mass_on_q` 0.941813, `support_q_mass_on_p` 0.989157, `support_jaccard` 0.885518.

Highest per-prompt total variation cases:

- `hle-othe-03`: `support_tv` 0.251082, `support_p_mass_on_q` 0.819290, `support_q_mass_on_p` 0.924495, `support_jaccard` 0.671563.
- `hle-phys-00`: `support_tv` 0.246502, `support_p_mass_on_q` 0.810498, `support_q_mass_on_p` 0.963065, `support_jaccard` 0.646389.
- `gpqa-chem-01`: `support_tv` 0.228588, `support_p_mass_on_q` 0.821782, `support_q_mass_on_p` 0.950455, `support_jaccard` 0.684002.

The draft sampler support was mostly contained in target support by probability mass (`support_q_mass_on_p` weighted mean 0.976817), while target mass outside draft support was larger (`1 - support_p_mass_on_q` weighted mean 0.094790).
