# Findings — temperature regime of the truncated rejection sampler (temp ≤ 0 / "greedy")

Lens: sampling **temperature** as it reshapes the rejection sampler in `initial_program.cpp`,
focused on the `temp ≤ 0` ("greedy") regime. Temperature is not covered by the previously
saturated lenses (acceptance/decomposition/support, per-depth/run-length, gamma, timing/forward-cost,
length/termination/score-parity, draft-head geometry, prompt structure, FP-determinism/batch-vs-AR/
KV-replay, draft-vs-target content, target token-uncertainty, context-recovery/copy, foreshadowing).

## Code mechanism (read from initial_program.cpp)
- `make_dist(...)` clamps the softmax temperature: `float t = temp > 0 ? temp : 1.0f;`. So at
  `temp = 0` every probability used by the accept/reject step is computed at **temperature 1**.
- Token *selection* is greedy at `temp ≤ 0`: draft uses `q.ids[0]` (MTP argmax); the all-accepted
  bonus token uses `p_b.ids[0]` (target argmax).
- The accept coin is unconditionally stochastic: `acc = min(1, px/qx)` with `px,qx` = temp-1 probs,
  decided by `uni(rng) < acc`.
- The reject branch resamples unconditionally: `new_tok = sample_dist(res, rng)` — there is **no
  `temp ≤ 0` guard**, unlike the draft/bonus selection.
- Consequence: at `temp = 0` the draft *selects* the argmax (point mass), but the rejection math
  still treats the proposal as the full temp-1 softmax `qd`. Standard speculative sampling is
  lossless for an arbitrary proposal only when the draft is actually sampled from that proposal;
  here selection (point mass at argmax) ≠ the assumed proposal (`qd`), so the emitted distribution
  is not the target distribution. With `qx < 1`, `min(1, px/qx)` over-weights the draft model's
  preferred token, and the residual `max(0, p − qd)` subtracts the full draft softmax.

## Empirical results (GPUs 6,7; top_p 0.95, top_k 64, seed 0 unless noted)

### Reproducibility across RNG seed at temp=0 (limit 4, n 1024; seed 0 vs seed 1)
| mode | seed0 vs seed1 |
|---|---|
| target   | byte-identical on all 4 items (RNG unused; pure argmax) |
| mtp-spec | differs on all 4 items; first divergence at char 26–89 |

This isolates the source of stochasticity to the spec algorithm (accept coin + reject resampling),
not to target/FP nondeterminism: the target greedy path is RNG-invariant, the spec path is not.

### Greedy text equality at temp=0 (limit 4, n 1024)
mtp-spec text ≠ target text on every item, diverging within the first 26–444 characters. Correct
greedy speculative decoding would be byte-identical to greedy target (both deterministic argmax).

### Full benchmark at temp=0 (30 items, n 16384)
| mode | accuracy | tok/s | accept | speedup |
|---|---|---|---|---|
| target (reference, temp 1.0) | 93.3% | 100.7 | — | 1.00× |
| target (temp 0) | 46.7% | 98.9 | — | 0.98× |
| mtp-spec (temp 0) | 93.3% | 149.3 | 72.3% | 1.48× |

- Greedy target collapses to 46.7%. 21/30 target items decode exactly the 16384-token cap (non-
  terminating); the failures are all capped runs. Per source (target vs spec): aime 40%→100%,
  gpqa 70%→100%, lcb 10%→100%, hle 80%→60%.
- mtp-spec at temp=0 (93.3% / 1.48× / 72.3% accept) matches the temp-1.0 reference in README
  (93.3% / 1.51× / 72.4% accept), not its own temp-0 target baseline (46.7%).

## Summary
At `temp = 0` the two modes diverge by 46.6 accuracy points in opposite directions under the same
flag: `target` is deterministic greedy (46.7%, loops to the token cap on most reasoning items),
while `mtp-spec` is stochastic and tracks temp-1 behavior (93.3%). The `temp ≤ 0` path of the spec
sampler makes only draft selection and the all-accepted bonus token greedy; the accept coin and the
reject resampling run at temperature 1 and consume RNG, so "greedy" mtp-spec is not greedy and is
not lossless with respect to the temp-0 target it is nominally configured to match.

Harness note: `evaluator/task-eval` was extended with `--temp`, `--top-p`, `--top-k` passthrough,
and the explicit `target` mode is no longer filtered out of the run loop (so a fresh target run can
be produced at non-default temperatures; `vanilla_references/target.json` is untouched).
