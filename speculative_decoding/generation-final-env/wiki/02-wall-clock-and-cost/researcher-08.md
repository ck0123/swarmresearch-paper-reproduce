# Findings — wall-clock latency decomposition of speculative decoding

## Angle
Speedup = useful_tokens / round_wallclock. Prior analyses studied the **numerator**
(acceptance: overall rate, per-depth, entropy/argmax drivers, run lengths, gamma tradeoff).
This analysis is deliberately orthogonal: it dissects the **denominator** — where the
per-round wall-clock actually goes — to find what caps achievable speedup independently of
acceptance. Hypothesis going in was that the CPU-side `make_dist` (a `partial_sort` over
Gemma's ~262K-token vocabulary, called ~2γ+1 times/round) might be a hidden bottleneck.

## Method
Instrumented `initial_program.cpp` with `ggml_time_us()` accumulators per phase
(prefill / draft_fwd / draft_smp / verify_fwd / verify_smp / cache), emitted as a `TIMING`
line per prompt (printed before `TEXT:`, so the scorer's regexes are unaffected). Accuracy on
the full 30-item eval stayed **93.3%**, identical to the target reference — instrumentation is
non-perturbing.

### Methodological catch (important)
`llama_decode` is **asynchronous**. A first naive run attributed ~70% of wall-clock to
`make_dist` (verify_smp). That was an artifact: the subsequent `llama_get_logits_ith` forces a
device sync, so GPU compute was being billed to the CPU sort that followed it. Adding an
explicit `llama_synchronize()` after every `llama_decode` (prefill, draft, verify) flipped the
picture entirely and is required for a correct decomposition.

Note on GPU config: `./evaluator/task-eval --gpus 0,1` runs **two single-GPU data-parallel
workers** (each `spec_modes` pinned to one GPU via `CUDA_VISIBLE_DEVICES`), not one tensor-split
job. The γ-sweep below was run as a single tensor-split process across both GPUs; this shifts
absolute per-call latency (MTP step 1.9ms split vs 1.15ms single-GPU) but the structure is
identical.

## Result 1 — wall-clock decomposition (full 30-item eval, gamma=4, the scored run)
Aggregate over 30 prompts (1724s total wall, 65830 rounds, ~520K make_dist calls):

| phase | share | what it is |
|---|---|---|
| verify_fwd | **69.4%** | target forward pass verifying the γ+1-token batch |
| draft_fwd  | **17.6%** | MTP head drafting (γ sequential width-1 passes) |
| draft_smp  | 6.4% | CPU `make_dist`/sample for draft tokens |
| verify_smp | 6.2% | CPU `make_dist`/resample for verification |
| prefill    | 0.2% | one-time prompt encode |
| cache      | 0.2% | seq_rm + batch init/free + memcpy |

GPU forward = **87.1%**, CPU sampling+cache = **12.8%**.

The starting hypothesis was **wrong**: the 262K-vocab sort is *not* the bottleneck
(~0.42ms/call, 12.6% combined). The **target verification forward pass dominates** at ~69%.

## Result 2 — cost curve of the target verify forward (γ-sweep, single tensor-split process)
verify_fwd per round vs verify batch width (γ+1), 2-prompt subset:

| batch width | verify_fwd/round | MTP draft/step | tok/s |
|---|---|---|---|
| 1 (target, single token) | 9.10 ms | — | 101 |
| 2 (γ=1) | 11.13 ms | 1.91 ms | 127 |
| 3 (γ=2) | 13.23 ms | 1.91 ms | 142 |
| 5 (γ=4) | 17.70 ms | 1.90 ms | 142–150 |
| 7 (γ=6) | 21.50 ms | 1.90 ms | 134 |
| 9 (γ=8) | ~21.5 ms (noisy) | 1.88 ms | 133–141 |

Linear fit (width 1→7): **verify_fwd ≈ 7.0ms fixed + ~2.07ms per token in the batch.**
Batched verification is sub-linear but **far from flat/free**: 5 tokens cost ~1.96× a single
token, not ~1×. The MoE target (4B active of 26B) is not in the weight-bandwidth-bound regime
where extra batch tokens are free — extra distinct tokens activate more experts / more compute,
so marginal token ≈ 2.07ms.

## Structural facts
- **Drafting is linear, verification is sub-linear.** MTP draft = γ *sequential* width-1
  passes (~flat ~1.9ms/step → total scales linearly in γ). Verify = one batched pass
  (7.0 + 2.07·width). This asymmetry sets the throughput optimum at small γ (peak tok/s at
  γ≈2–4; declines by γ=6–8 as draft cost grows linearly while accepted tokens saturate).
- **The "tiny" MTP draft head is not nearly free.** A single MTP forward (~1.9ms tensor-split /
  1.15ms single-GPU) is comparable to the *marginal* cost of one extra verified target token
  (~2.07ms), and drafting is **17.6%** of total wall-clock. Likely contributor: the MTP head
  shares the giant 262K-row output (un)embedding projection with the target, which is invariant
  to how small the transformer body is — so a 461MB head (~1.7% of the 27GB target) does not
  cost ~1.7% of the per-token work. (Mechanism is a hypothesis; the timing asymmetry is measured.)
- **Verify-bound speedup ceiling.** Holding acceptance and γ fixed, if drafting + all CPU
  sampling were zero-cost, wall-clock would fall to verify_fwd+prefill (69.6%), i.e. an upper
  bound of ~1724/1200 = 1.44× faster than current → ~**2.1× absolute speedup ceiling** in this
  verify-bound regime. The target verification forward is the hard floor.
