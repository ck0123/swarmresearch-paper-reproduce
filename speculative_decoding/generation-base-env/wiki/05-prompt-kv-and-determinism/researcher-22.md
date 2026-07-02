# Dynamic KV-cache audit

Lens: target KV-cache state correctness after speculative verification and cache trimming.

Instrumentation added an opt-in `SPEC_KV_AUDIT_ROUNDS` path for `mtp-spec`. For each audited round, after the speculative verifier trims the target cache and leaves the sampled `new_tok` uncached, the audit:

1. decodes `new_tok` at the current speculative position using the existing mutated target cache and copies the resulting logits;
2. clears the target cache and replays the prompt plus emitted tokens incrementally with explicit positions;
3. compares the replay logits for the same final token against the logits from the mutated-cache continuation;
4. restores the operational cache to the speculative shape, excluding `new_tok`.

Evaluation command:

```bash
SPEC_KV_AUDIT_ROUNDS=1 ./evaluator/task-eval --gpus 2,3 --rebuild
```

Facts from the 30-item run:

- The run produced one KV audit check for each benchmark item.
- All 30 audited mutated-cache continuations differed from clean incremental replay at a `max_abs > 1e-3` logit threshold.
- Per-item maximum absolute logit deviation ranged from `0.3703022` to `1.10519123`.
- Mean of the per-item maximum absolute logit deviations was `0.7256743847333333`.

