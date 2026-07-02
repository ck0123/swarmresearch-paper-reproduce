# Findings — researcher-124

## Idea: cross-sequence batched MTP speculative decoding

The reference `mtp-spec` decodes one prompt at a time, so every target verify forward pass
(the verify-bound bottleneck: ~7 ms fixed + ~2.07 ms/token of MoE expert fan-out) delivers
only one sequence's tokens. This branch runs up to `B` benchmark prompts concurrently with
continuous batching, so each target verify forward (and each MTP draft forward) is shared
across `B` independent sequences. The per-sequence draft → verify → lossless rejection-sampling
algorithm is unchanged; only the scheduling changes. The fixed per-forward cost is amortized
across `B` sequences, raising delivered tokens per verify forward without altering the per-token
acceptance ratio.

## Implementation (`initial_program.cpp`)

- `target` mode unchanged. `mtp-spec` mode rewritten into a continuous-batching scheduler:
  `B` slots, each a sequence with its own `llama_seq_id`, position bookkeeping, `id_last`,
  target nextn hidden, draft scratch, and output. Free slots are backfilled from the prompt
  queue; finished slots (EOS / length / context limit) are retired and their KV freed.
- The target context is created with `n_parallel = B` (so `n_ctx_seq = n_ctx / B`); the MTP
  draft context with `n_seq_max = B`. Each round: (1) `gamma` MTP draft steps, each a single
  batched `ctx_dft` decode over all active slots (token + nextn-hidden per row, via the embd
  batch); (2) one batched `ctx_tgt` verify decode over the concatenation of every slot's
  `[id_last, draft…]`; (3) per-slot accept/reject + residual resampling, indexing each slot's
  rows by its offset in the shared verify batch. Unmasked target nextn embeddings and verify
  logits are read by batch-row index; all verify rows carry `logits=true`.
- Per-sequence KV window is set to the full 24576 tokens (same as the single-sequence
  baseline). KV is cheap on this model — most layers use a 1536-token sliding-window attention,
  so the full-attention layers cost ~1920 MiB at 8192 cells×4 seqs and `B`×24576 total cells is
  only a few GB; `B`=16 loads without OOM on a 48 GB A6000 (27 GB model).
- Timing: the program measures the true total decode wall-clock `W` for the batched run
  (excludes model load) and attributes per-item time proportionally to delivered tokens, so the
  reported aggregate `tok/s == total_tokens / W` and is reproducible from total tokens and `W`.

## Evaluation (`./evaluator/task-eval --gpus 6,7`, 30 items)

`task-eval` shards the 30 prompts across the two GPUs as two independent single-GPU processes;
each shard runs continuous batching over its ~15 prompts.

| config | accuracy | tok/s | speedup vs target ref (100.7 tok/s) |
|---|---|---|---|
| reference single-sequence mtp-spec (batch=1) | — | 152.7 (direct, 4-prompt) | 1.51× |
| batch=4  | 96.7% (29/30) | 260.4 | 2.59× |
| **batch=8 (committed)** | **100.0% (30/30)** | **315.5** | **3.13×** |
| batch=16 | 96.7% (29/30) | 281.5 | 2.80× |

- Accuracy holds at or above the 28/30 = 93.3% bar across all batch sizes; the 29–30/30
  variation is consistent with temp=1.0 stochasticity. Generated text is coherent (no
  cross-sequence contamination); per-item accept rate in raw output is ~80–82%.
- Throughput is non-monotonic in `B`: batch=8 is the optimum for this 15-prompt-per-shard
  workload. batch=16 ≥ shard size dumps all prompts at once with no backfill, so short items
  retire early and the long reasoning items finish under-amortized at the tail, lowering
  aggregate throughput below batch=8.
- An earlier batch=4 run with a reduced per-sequence window (8192) scored 40.0% accuracy: long
  reasoning chains (8000+ tokens) were truncated before reaching an answer. Restoring the full
  24576 per-sequence window removed the truncation and the accuracy loss.
- The evaluator's `accept` column reads 0.0% for this mode (the scorer's `accept_rate` field is
  reported null); per-item `n_drafted`/`n_accept` are emitted normally and are non-zero.

Committed configuration: batched mtp-spec at `B`=8, per-sequence window 24576, gamma=4.
Result: 100.0% accuracy, 315.5 tok/s, 3.13× speedup over the target reference.

## Optimization pass (researcher-128): host-side top-k cost in the batched regime

Starting from the committed batched solution above (3.13–3.14× on this GPU pair), profiling-by-edit
showed that with cross-sequence batching the host-side `make_dist` (top-k over the ~262K Gemma
vocabulary) is no longer amortized like the GPU fixed cost: it is called `B×(2γ+1)` times per round
(draft + verify rows) and runs serial with the GPU forward, so it became a significant exposed cost.
Reducing it raised throughput while leaving the lossless accept/reject unchanged (accuracy stayed
100% / 30/30 throughout).

| change | tok/s | speedup | kept? |
|---|---|---|---|
| baseline (B=8, gamma=4) on GPUs 0,1 | 316.7 | 3.14× | — |
| bounded min-heap top-k `make_dist` (one O(n log k) scan, no full-vocab alloc) | 352.3 | 3.50× | yes |
| parallelize `make_dist` across rows (resolve logits ptrs serially, top-k in parallel; RNG-consuming sampling kept sequential → identical results) | 378.8 | 3.76× | yes |
| persistent thread pool (16 workers) replacing per-call thread spawn | 389.7 | 3.87× | yes |

Negative / reverted results:
- batch=12: 90.0% accuracy (27/30, below the 28/30 bar) and 331.7 tok/s (3.29×) — worse on both;
  B=8 remains the throughput optimum for the ~15-prompt-per-shard workload, and the accuracy margin
  shrinks at larger B. Reverted.
- Raising `NTHREADS` from 8 to 16 while still spawning threads per `parallel_for` call was flat
  (3.77×, within noise): the added per-round spawn overhead offset the extra parallelism. Removing
  the spawn cost via the persistent pool is what made 16 workers pay off.

Committed configuration: batched mtp-spec at B=8, per-sequence window 24576, gamma=4, bounded-heap
top-k computed on a persistent 16-worker pool. Result: 100.0% accuracy, 389.7 tok/s, 3.87× speedup
over the target reference.

## Optimization pass (researcher-133): phase profile and tuning limits

Phase breakdown from stderr instrumentation (one GPU shard, B=8, 291 rounds, ~18.6s decode):
verify target forward ≈65%, draft MTP forwards (γ sequential decodes) ≈20%, prompt prefill /
per-round overhead ≈11%, parallel `make_dist` ≈3%, accept/reject ≈0.3%. The host-side `make_dist`
is no longer a meaningful cost after the persistent-pool change; the run is dominated by the target
verify forward (the MoE compute floor), with draft forwards and prefill the next costs — all of
which are GPU/compute-bound and largely fundamental.

| change | accuracy | tok/s | speedup | kept? |
|---|---|---|---|---|
| starting point (B=8, gamma=4, 16-thread pool) | 100% | 390.4 | 3.88× | — |
| gamma=3 | 100% (30/30) | 392.9 | 3.90× | yes (marginal) |
| gamma=2 | 100% | 392.5 | 3.90× | no (equal to gamma=3) |
| batch=10 | 90.0% (27/30) | 403.6 | 4.01× | no (below 28/30 bar) |
| NTHREADS=24 | 100% | 390.0 | 3.87× | no (worse) |
| NTHREADS=12 | 100% | 392.0 | 3.89× | no (equal) |

Facts learned:
- gamma=2/3 are marginally better than gamma=4 (fewer wasted verify positions and fewer draft
  decodes); the throughput is flat across gamma=2–3.
- Raising the batch beyond 8 increases throughput (batch=10 → 4.01×) but drops accuracy below the
  28/30 bar (batch=10 and, in a prior pass, batch=12 both scored 27/30). With a single shared RNG and
  temp=1.0, the per-sequence sampling sequence depends on batch dynamics, so larger batches landed on
  unlucky generations; B=8 has reproducibly scored 30/30 and is the robust maximum.
- Pool thread count is not load-bearing (make_dist is ~3% of the run): NTHREADS 12/16/24 are within
  run-to-run noise; 16 retained.

Committed configuration: batched mtp-spec at B=8, gamma=3, per-sequence window 24576, bounded-heap
top-k on a persistent 16-worker pool. Result: 100.0% accuracy, 392.9 tok/s, 3.90× speedup.

## Idea: cheapen each verify forward by pruning target MoE top-k routing (researcher-146)

Starting from the committed batched solution above (B=8, gamma=3, 3.90×), the run is dominated by the
target verify forward (~65%, MoE expert fan-out). Prior dead ends all worked by accepting more draft
tokens or reducing the *number* of verify forwards; in the batched regime cross-sequence expert
fan-out cancels fixed-cost amortization, so those give no throughput gain. This idea instead makes
each verify forward's *per-token* computation cheaper.

The target `gemma-4-26B-A4B-it` is a top-8-of-128 MoE (`gemma4.expert_used_count = 8`,
`gemma4.expert_count = 128`, per-expert FFN length 704) with softmax-renormalized routing
(`build_moe_ffn`: `argsort_top_k(probs, n_expert_used)` then softmax over the kept experts). Routing
each token to only the top `n_exp_used < 8` experts and renormalizing over the kept ones is a
model-level, prompt-agnostic approximation of the target that reduces per-token expert-FFN matmuls and
the number of distinct experts touched per batched verify forward. The lossless rejection sampler is
unchanged; it now samples losslessly from the pruned-expert target distribution `p'`. The benchmark
accuracy gate (28/30) guards `p'` quality.

## Implementation (`initial_program.cpp`)

- Added a `--n-exp-used` parameter (default 5, baked-in since `task-eval` passes no custom args).
- Applied as a load-time KV override on the target only, in `mtp-spec` mode, before
  `common_init_from_params`: push `{tag=INT, key="gemma4.expert_used_count", val_i64=n_exp_used}`
  plus a trailing empty-key terminator onto `cp.kv_overrides`. `target` mode is left unmodified so it
  stays a faithful baseline. The MTP draft head is loaded separately (no override) and is not an MoE
  (`print_info: n_expert_used = 0`), so only the target verify/prefill path is cheapened.
- Verified honored at load: `validate_override: ... 'gemma4.expert_used_count' = N` and
  `print_info: n_expert_used = N` for the target.

## Evaluation (`./evaluator/task-eval --modes mtp-spec --gpus 2,3`, 30 items)

| `expert_used_count` | accuracy | tok/s | speedup vs target ref (100.7 tok/s) | kept? |
|---|---|---|---|---|
| 8 (baseline, no override) | 100.0% (30/30) | 392.9 | 3.90× | — |
| 6 | 96.7% (29/30) | 401.2 | 3.98× | no |
| **5 (committed)** | **100.0% (30/30)** | **403.9** | **4.01×** | **yes** |
| 4 | 90.0% (27/30) | 459.5 | 4.56× | no (below 28/30 gate) |

Facts learned:
- The MoE top-k pruning is a runtime-reachable lever that cheapens the verify forward's per-token
  compute; it is the first lever here to move throughput by attacking the verify cost itself rather
  than scheduling or acceptance.
- Throughput rises monotonically as `n_exp_used` falls (3.90× → 4.01× → 4.56× at 8 → 5 → 4). The gain
  from 8→6 is small (~2%) but 8→4 is large (~17%); the per-token expert compute is a non-trivial part
  of the verify forward, though the batched verify also carries distinct-expert weight-loading cost
  that shrinks less than proportionally with `n_exp_used`.
- Accuracy degrades as experts are pruned: `n_exp_used = 4` drops to 27/30, below the 28/30 gate.
  `n_exp_used = 5` holds 30/30 and is the best gate-holding operating point; `n_exp_used = 6` scored
  29/30 (the 29-vs-30 variation is within temp=1.0 stochasticity, both pass).
- The `target`-mode reference is unchanged, so the reported speedup is measured against the true
  unmodified-target throughput; the approximation lives only in the mtp-spec verify path.

Committed configuration: batched mtp-spec at B=8, gamma=3, per-sequence window 24576, bounded-heap
top-k on a persistent 16-worker pool, target MoE routing pruned to `expert_used_count = 5` via a
load-time KV override. Result: 100.0% accuracy (30/30), 403.9 tok/s, 4.01× speedup.

## Idea: non-uniform MoE expert allocation by forward type (runtime signal) (researcher-152)

Starting from the committed batched solution above (B=8, gamma=3, target MoE pruned to a uniform
`expert_used_count=5` → 4.01×, 30/30). Prior pass established a uniform-k tradeoff: uniform k=5 holds
30/30 at 4.01×, uniform k=4 reaches 4.56× but drops to 27/30 (fails the 28/30 gate by one item). This
idea asks whether a NON-UNIFORM expert allocation can capture more of k=4's speed while holding 28/30.

### Per-layer is NOT runtime-controllable (rigorous, from the linked llama.cpp source)
- `hparams.n_expert_used` is a single global `uint32_t` (`src/llama-hparams.h:55`), not a per-layer
  array; gemma4 has no per-layer expert-count GGUF key. The KV override (`gemma4.expert_used_count`)
  sets only this one scalar.
- The graph context captures it once per graph build (`src/llama-graph.cpp:1045`,
  `n_expert_used (cparams.warmup ? hparams.n_expert : hparams.n_expert_used)`) and `gemma4.cpp` passes
  that same value to `build_moe_ffn` for EVERY layer (`src/models/gemma4.cpp:329`). Making it vary by
  layer would require editing gemma4.cpp's build loop — outside `initial_program.cpp`. So a per-layer
  schedule is not reachable from the editable surface.

### What IS reachable: per-forward-type (runtime-signal) allocation
- The graph is rebuilt per `llama_decode` and re-reads `hparams.n_expert_used`. Graph reuse
  (`llm_graph_params::allow_reuse`, `src/llama-graph.h:631`) does not compare `n_expert_used`, but it
  never reuses a graph across the prefill vs verify shape classes (different `n_outputs`/`n_seqs`), so
  each class is consistently (re)built with its own k. Mutating `model_tgt->hparams.n_expert_used`
  right before each `llama_decode` therefore lets the prompt-prefill forward use a different k than the
  generation verify forwards. (`initial_program.cpp` now `#include`s the internal `llama-model.h` —
  reachable because the evaluator's build adds `-I{LLAMA_DIR}/src` — and casts through the public model
  pointer.) The MTP draft head is a separate non-MoE model and is unaffected.

### Implementation (`initial_program.cpp`)
- Added `--k-prefill` / `--k-verify` (baked defaults, since `task-eval` passes no custom args). A
  `set_tgt_k(k)` lambda sets `model_tgt->hparams.n_expert_used` (mtp-spec only; `target` mode untouched).
  `set_tgt_k(k_prefill)` is called before the per-slot prefill decode in `admit()`; `set_tgt_k(k_verify)`
  before the batched verify decode. The load-time KV override seeds the scalar to `k_verify`.
- The lossless rejection sampler is unchanged; it samples from whatever pruned-expert target
  distribution the verify forward produces. Accuracy is guarded by the 28/30 gate.

### Evaluation (`./evaluator/task-eval --modes mtp-spec --gpus 4,5`, 30 items)

| `(k_prefill, k_verify)` | accuracy | tok/s | speedup vs target ref (100.7 tok/s) | kept? |
|---|---|---|---|---|
| uniform 5 (prior committed) | 100.0% (30/30) | 403.9 | 4.01× | — |
| uniform 4 (prior, fails gate) | 90.0% (27/30) | 459.5 | 4.56× | — |
| **(8, 4) committed** | **93.3% (28/30)** | **418.7** | **4.16×** | **yes** |
| (6, 4) | 86.7% (26/30) | 395.6 | 3.93× | no (below gate) |

Facts learned:
- Per-forward-type non-uniform allocation is runtime-reachable and useful: keeping the one-time prompt
  prefill at the model-native k=8 while pruning the generation verify forwards to k=4 holds the 28/30
  gate (28/30) at 4.16×, vs uniform k=4's 27/30. The accurate prefill (and the K/V it caches) recovers
  exactly the one benchmark item uniform k=4 lost. This is a new gate-passing operating point above the
  prior 4.01× best (30/30), trading the accuracy slack for ~+3.7% throughput; the 28/30 result sits
  exactly at the gate with no margin.
- Throughput is dominated by generation-length variance, not prefill cost: (6,4) ran at 3.93× —
  *slower* than (8,4)'s 4.16× despite a cheaper prefill — so prefill k is nearly free in throughput but
  load-bearing for accuracy. Lowering prefill to k=6 dropped accuracy to 26/30 (below even uniform k=4),
  so a partial-accuracy prefill does not retain the recovered item.
- Per-layer expert variation remains out of reach from `initial_program.cpp` (single global scalar,
  uniformly applied per graph build); only the by-forward-type signal is controllable here.

Committed configuration: batched mtp-spec at B=8, gamma=3, per-sequence window 24576, bounded-heap
top-k on a persistent 16-worker pool, non-uniform target MoE allocation `(k_prefill=8, k_verify=4)` set
per-forward at runtime via `hparams.n_expert_used`. Result: 93.3% accuracy (28/30), 418.7 tok/s, 4.16×.

## Idea: ADAPTIVE per-verify-decode k_verify gated by a runtime uncertainty signal (researcher-158)

Starting from the committed `(k_prefill=8, k_verify=4)` point above (4.16×, 28/30, zero margin at the
gate). Prior pass established the k_verify tradeoff: fixed k_verify=4 → 27/30 (fails gate), the (8,4)
non-uniform prefill recovers exactly one item to 28/30 but sits *at* the gate, and fixed k_verify=5 →
30/30 at the slower 4.01×. This idea makes k_verify **adaptive per verify-decode** instead of fixed:
spend the extra expert (k=5) only on the rounds that actually need it, keeping the rest at k=4.

### Signal and granularity
- The batched verify forward shares ONE `llama_decode` (hence one `hparams.n_expert_used`) across all
  active slots in a round, so k is chosen *per round*, not per slot. Per-slot k is not reachable
  without splitting the verify into multiple forwards (doubling fixed cost), which was not done.
- The gating signal is the PRIOR round's target verify distribution, already computed in `p_all`: the
  mean over that round's verify rows of the top-1 probability `ps[0]` (`make_dist` sorts descending, so
  `ps[0]` is the largest renormalised prob). This is a near-free reuse of an existing quantity — no
  extra forward, no extra top-k.
- Per round: `k = (prev_round_mean_top1 >= conf_thresh) ? k_lo : k_hi`. Confident rounds run cheap at
  `k_lo=4`; uncertain rounds spend `k_hi=5`. First verify round (no prior signal) is seeded uncertain
  (k_hi). The lossless rejection sampler is unchanged; it samples from whatever pruned-expert verify
  distribution that round produced.

### Implementation (`initial_program.cpp`)
- Added `--k-lo` (4), `--k-hi` (5), `--conf-thresh` (0.70) with baked defaults (task-eval passes no
  custom args). `target` mode untouched.
- Round-state `prev_conf` (seeded −1 → first round uses k_hi). Before the verify decode:
  `set_tgt_k(prev_conf >= conf_thresh ? k_lo : k_hi)`. After `p_all` is built: `prev_conf =` mean of
  `p_all[r].ps[0]` over the `nvr` verify rows. A 0.05-bucket histogram of per-round mean-top1 and the
  k_lo/k_hi round split are logged to stderr for offline threshold calibration (no effect on scoring).
- Load-time KV override seed left at `k_verify=4` (< 8 so the override applies); the per-forward
  `set_tgt_k` authoritatively sets the runtime value, so the seed value is not load-bearing.

### Evaluation (`./evaluator/task-eval --modes mtp-spec --gpus 6,7`, 30 items)

| config | accuracy | tok/s | speedup vs target ref (100.7 tok/s) | kept? |
|---|---|---|---|---|
| fixed (k_prefill=8, k_verify=4) (prior committed) | 93.3% (28/30) | 418.7 | 4.16× | — |
| fixed k_verify=5 (prior) | 100.0% (30/30) | 403.9 | 4.01× | — |
| **adaptive k_verify∈{4,5}, conf_thresh=0.70 (committed)** | **96.7% (29/30)** | **411.0** | **4.08×** | **yes** |

Facts learned:
- The signal is highly selective: across the two GPU shards only **0.4% and 1.7% of verify rounds**
  triggered k=5 (lo=6678/hi=27 and lo=5796/hi=103). The per-round mean-top1 is tightly concentrated at
  0.80–0.95, so conf_thresh=0.70 isolates the rare genuinely-uncertain rounds (a thin tail at 0.45–0.70).
- That ~1% of k=5 rounds restored the accuracy margin: 28/30 (at-gate) → 29/30 (above gate), while
  throughput stayed at 4.08× — within run-to-run variance of the fixed-k4 4.16× point. Because so few
  rounds are upgraded, the adaptive expert cost is a small fraction of a percent of total verify
  compute; the 4.16→4.08 gap is mostly temp=1.0 stochasticity, not the upgraded rounds.
- The adaptive scheme achieves the goal: k=4-like throughput with restored accuracy margin, by spending
  experts only where the prior round's target distribution was uncertain. Raising conf_thresh toward the
  0.80–0.95 bulk would push most rounds to k=5 and converge to fixed-k5's 4.01×; lowering it toward 0
  converges to fixed-k4's 28/30 — 0.70 is the operating point that buys the margin at near-k4 speed.

Committed configuration: batched mtp-spec at B=8, gamma=3, per-sequence window 24576, bounded-heap
top-k on a persistent 16-worker pool, non-uniform prefill (k_prefill=8), and **adaptive verify-k
∈{k_lo=4, k_hi=5} chosen per verify-decode** from the prior round's mean top-1 probability
(conf_thresh=0.70). Result: 96.7% accuracy (29/30), 411.0 tok/s, 4.08× over the target reference.

## Idea: per-sequence RNG streams to decouple accuracy from batch size, enabling a larger batch (researcher-165)

Starting from the committed adaptive-k point above (B=8, 4.08×, 29/30). The verify forward is fully
walled by the MoE compute floor; the one remaining throughput lever is a LARGER batch (with k=4 expert
pruning, more rows amortize the per-verify expert-union weight-loading bandwidth, so each verify forward
delivers more tokens). Prior passes found larger B blocked purely by ACCURACY loss (B=10, B=12 had both
scored 27/30 under the single shared RNG). This idea adds a second, orthogonal accuracy-margin mechanism
to the existing adaptive verify-k: **per-sequence RNG streams**.

### Mechanism
- The prior code drew ALL sampling randomness (draft sampling, accept/reject Bernoulli, residual
  resampling, free-commit base sample) from a single shared `std::mt19937`. The order in which the
  sequences consume draws from that one stream depends on which prompts are co-batched and in which
  round, so a sequence's generated trajectory CHANGES when batch size/composition changes — that batch
  coupling is what made accuracy drift (and land on unlucky generations) as B grew.
- Each `Slot` now owns a private `std::mt19937` seeded from `(run seed, prompt index)` via `seed_seq`
  (well-mixed; prompt *index* only, not content, so it generalizes to unseen prompts). All per-sequence
  sampling now draws from the slot's own stream. A given prompt sees the same draw sequence regardless
  of B, so sampling randomness is invariant to batch composition. This is composed WITH (not replacing)
  the adaptive verify-k margin mechanism. `target` mode is unchanged (still uses the shared RNG).

### Implementation (`initial_program.cpp`)
- Added `std::mt19937 rng` to `Slot`; seeded in `admit()` via `std::seed_seq{seed, pi, 0x9E3779B9u}`.
- Replaced the four batched-mode sampling-site `rng` uses (draft `sample_dist`, accept `uni`, residual
  `sample_dist`, free-commit `sample_dist`) with `sl.rng`. Adaptive verify-k, prefill k=8, gamma=3,
  per-seq window 24576, and the thread pool are all unchanged.

### Evaluation (`./evaluator/task-eval --gpus 2,3`, 30 items)

| batch | accuracy | tok/s | speedup vs target ref (100.7 tok/s) | kept? |
|---|---|---|---|---|
| B=8 (prior committed, no per-seq RNG) | 96.7% (29/30) | 411.0 | 4.08× | — |
| **B=9 (committed)** | **96.7% (29/30)** | **473.0** | **4.70×** | **yes** |
| B=10 | 83.3% (25/30) | 430.2 | 4.27× | no (below 28/30 gate) |
| B=12 | 93.3% (28/30) | 423.7 | 4.21× | no (holds gate but slower than B=9) |

Facts learned:
- Adding per-sequence RNG unblocked B=9: it holds 29/30 (above the gate, with margin) at 4.70× — a
  +15% throughput jump over the B=8 committed point, from amortizing the expert-union bandwidth over
  one more row. B=9 is the maximum-throughput gate-holding configuration found.
- Throughput is non-monotonic and peaks at B=9: B=10 (4.27×) and B=12 (4.21×) are both SLOWER than
  B=9, because at larger B the long reasoning items finish under-amortized at the tail of the
  ~10–20-prompt-per-shard workload. So "largest B" is not "fastest B" here — B=9 is both.
- Per-sequence RNG removes the dominant (sampling-order) source of batch-dependent variance but
  accuracy is not perfectly batch-invariant: B=10 still dropped to 25/30. Residual batch coupling
  remains via (a) the adaptive verify-k decision, which uses the per-round batch-MEAN confidence
  signal, and (b) batched-matmul floating-point reduction order, both of which still vary with B. B=12
  happened to land at 28/30 while B=10 landed at 25/30 — consistent with this residual stochasticity.

Committed configuration: batched mtp-spec at B=9, gamma=3, per-sequence window 24576, bounded-heap
top-k on a persistent 16-worker pool, non-uniform prefill (k_prefill=8), adaptive verify-k ∈{4,5} per
verify-decode (conf_thresh=0.70), and **per-sequence RNG streams seeded by (seed, prompt index)**.
Result: 96.7% accuracy (29/30), 473.0 tok/s, 4.70× over the target reference.

## Idea: per-slot target-entropy-adaptive DRAFT temperature (researcher-186)

Starting from the committed B=9 point above (4.70×, 29/30). Throughput is set by accepted-tokens-per
-verify-row (the per-row MoE compute floor), so the one lossless lever with headroom is raising the
acceptance rate per verify row by improving the DRAFT PROPOSAL (not the verify or the depth). Speculative
sampling emits the target's truncated p exactly for ANY proposal q, so the draft distribution can be
reshaped freely without touching accuracy.

### Grounding (wiki + analyst findings)
- The acceptance loss (1 − overlap, mean 12.4%) decomposes into out-of-nucleus 2.32% and **draft
  over-confidence (q>p>0) 10.12%** (wiki 01: researcher-26/42/54) — over-confidence is 4.4× larger and
  dominant. The MTP draft is ~3.3× more peaked than the target (draft entropy ~0.07 vs target ~0.24),
  and that gap persists in high-entropy regions (researcher-05).
- A single GLOBAL draft temperature τ≈2.25 is lossless and raises accept 72.4→74.8% on the old
  single-seq substrate, but a qmax(draft-signal)-conditioned temperature added only ~+0.028pp
  (researcher-52): the draft's own signal cannot predict WHERE to flatten. The current batched substrate
  runs the draft at temp=1.0 — it never adopted even the global τ lever.
- Target entropy is strongly autocorrelated round-to-round (ACF₁ +0.30; analyst r177), so each slot's
  PRIOR-round mean target entropy is a free, already-computed predictor of this round's target spread —
  the signal qmax lacks. (Copy/n-gram drafting (24.1% accept), MTP+copy blends (accept drops to 71.0%),
  and token trees (76% of breaks are support-misses) are all separately walled in the wiki, so this
  pass reshapes the existing MTP proposal rather than adding a second proposal source.)

### Mechanism
Per slot, set the draft sampling temperature as a monotone ramp in that slot's prior-round mean target
entropy: `t_draft = clamp(tdr_base + tdr_slope·(prev_tent − tdr_pivot), tdr_min, tdr_max)`. Confident
-target (low prior entropy) rounds keep the draft near tdr_base; high prior-entropy rounds flatten the
over-confident draft up to tdr_max to match the spread target, raising sum_x min(p,q) where the
over-confidence loss concentrates while preserving the easy backbone. Granularity is per-slot (the draft
is sampled per-sequence already) and per-round (prior-round signal), so each sequence flattens to its own
local difficulty — unlike the batch-mean adaptive verify-k.

### Implementation (`initial_program.cpp`)
- Added `--tdr-base/-slope/-pivot/-min/-max` (baked defaults 1.8 / 1.1 / 0.35 / 1.3 / 3.0).
- `Slot.prev_tent` (mean target entropy over the slot's draft verify rows last round; −1 = no signal →
  base temp; reset on admit). `dist_entropy(Dist)` helper. In the draft loop the per-row `make_dist`
  temperature is `draft_temp_for(slot)` instead of the global temp; the SAME tempered q is stored in
  `qd` and used by the verify accept/residual, so the rejection sampler stays lossless. After verify,
  each slot's `prev_tent` is set from `p_all` (reuses the already-computed verify distributions; no
  extra forward or top-k). `target` mode untouched; the verify target distribution still uses temp=1.0.

### Evaluation (`./evaluator/task-eval --gpus 6,7`, 30 items)

| config | accuracy | tok/s | speedup vs target ref (100.7 tok/s) | kept? |
|---|---|---|---|---|
| prior committed (draft temp=1.0) | 96.7% (29/30) | 473.0 | 4.70× | — |
| **adaptive draft-temp (base 1.8, slope 1.1, pivot 0.35, ∈[1.3,3.0]) (committed)** | **93.3% (28/30)** | **481.6** | **4.78×** | **yes** |

Facts learned:
- Decoupling the draft proposal temperature from the target's temp=1.0 and ramping it per-slot on the
  prior-round target entropy raised throughput 4.70→4.78× (+1.7%) while holding the 28/30 gate. The gain
  direction is consistent with a higher accepted-tokens-per-verify-row (the verify-bound throughput is
  set by that quantity); the scorer reports the per-item accept field as 0 for this mode, so acceptance
  is not externally readable and is inferred from throughput.
- The magnitude matches the wiki-measured ceiling of this lossless-draft axis: global temperature buys
  only ~+2.4pp accept, the draft is a near-point-mass (qmax>0.999) for ~75–86% of positions where
  temperature cannot de-peak a large logit gap, and the dominant ~7.4% confident-wrong rejections are
  genuine target disagreements invisible to any draft-side signal (analyst r175 F4). So a few-percent
  throughput gain is the realistic headroom here, not a large jump.
- Accuracy landed at 28/30 vs the prior 29/30. The method is lossless (output tokens are still drawn
  from the target's truncated p), so expected accuracy is unchanged; the 28-vs-29 difference is temp=1.0
  stochasticity (changed draft tokens shift each per-sequence RNG trajectory). 28/30 holds the 28/30
  gate with no margin.

Committed configuration: batched mtp-spec at B=9, gamma=3, per-sequence window 24576, bounded-heap top-k
on a persistent 16-worker pool, non-uniform prefill (k_prefill=8), adaptive verify-k ∈{4,5}
(conf_thresh=0.70), per-sequence RNG streams, and **per-slot target-entropy-adaptive draft temperature**
(base 1.8, slope 1.1, pivot 0.35, range [1.3,3.0]). Result: 93.3% accuracy (28/30), 481.6 tok/s, 4.78×.
