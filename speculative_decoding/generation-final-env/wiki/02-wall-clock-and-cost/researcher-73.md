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

---

# Extension — mechanistic decomposition of the forward cost (what physically costs the time)

## Angle
The decomposition above measured the *symptom* (verify ≈ 7.0ms fixed + 2.07ms/token; draft ≈1.9ms)
and left two **labelled hypotheses** for the *cause*: (H1) the 2.07ms marginal verify token is
"MoE expert fan-out", (H2) the tiny MTP draft is expensive because it shares the 262K LM head.
This extension arbitrates both by physically separating the two candidate cost centers — the
shared 262K-row LM-head (un)embedding matmul vs the MoE transformer body — using two levers the
γ-sweep never touched. New `--mode probe` in `initial_program.cpp`; analysis is non-perturbing
(it is an isolated branch; scored `target`/`mtp-spec` paths are byte-for-byte unchanged — smoke:
mtp-spec 1.60× / 81% accept, healthy).

Architecture (Q8_0, from gguf meta): 30 layers, hidden 2816, **128 experts, top-8 used**,
expert-FFN 704, vocab **262144**. All probe numbers are **single-GPU** (CUDA_VISIBLE_DEVICES=4),
medians of 40 reps after 8 warmups, on real prefilled prompts (3 prompts, results consistent).

## Method — two orthogonal levers
- **Logit-count k:** decode a width-W batch but compute logits (the 262K LM-head matmul) on only
  the last k positions. The body (attn+MoE) runs for all W regardless, so cost(W,k)−cost(W,0)
  isolates the LM head, and its slope in k splits it into a fixed weight-read vs per-token compute.
- **Token diversity:** width-W batch of W *distinct* tokens vs W copies of one token, k fixed.
  At fixed batch width the CUDA kernels are identical, so the (distinct − same) gap isolates the
  extra expert weight bandwidth pulled in by routing diversity (8-of-128 experts/token).
- **MTP draft:** same logits-on vs logits-off lever on the draft forward → LM-head share of draft.

## Result E1 — the 262K LM head is almost pure FIXED cost, NOT the per-token driver (refutes the naive reading of H1)
Logit-count sweep at W=9 (avg µs): k0=21044 → k1=22245 (**+1201**) → k9=22817.
- LM-head **fixed weight read** (k0→k1): **~1201 µs** (one-time, ≈782MB Q8 unembedding read).
- LM-head **per-token compute** (k1→k9 slope): **~71 µs/token**.
So 17 of every 18 LM-head microseconds are the one-time weight read; each *extra* verified token
adds only ~71µs of LM-head work. The 262K vocab is a big **per-forward fixed** cost, not the
2.07ms marginal. (CPU make_dist over 262K, from the prior section, is a separate ~0.42ms/call.)

## Result E2 — the marginal verify token IS the MoE body, and expert fan-out is a large, growing share (confirms + quantifies H1)
Body-only cost (k0, no logits), avg µs, distinct vs identical tokens:

| W | same_k0 | distinct_k0 | fan-out gap | distinct marginal/tok |
|---|---|---|---|---|
| 1 | 7775 | 7760 | −15 | — |
| 2 | 9651 | 9844 | 193 | |
| 3 | 11306 | 11916 | 610 | |
| 5 | 14343 | 16106 | 1763 | |
| 7 | 16794 | 19711 | 2917 | ~1992 (W1→7) |
| 9 | 15203 | 21022 | 5819 | ~1658 (W1→9, flattening) |

- At fixed batch width (identical kernels), **distinct-token batches cost progressively more than
  identical-token batches** — the fan-out gap grows 0→5819µs across W=1→9. This is the direct
  fingerprint of MoE expert fan-out: each new *distinct* token routes to experts not already
  resident, adding weight bandwidth (~1.5GB of expert weights/token if disjoint).
- The distinct marginal/token **flattens** toward W=9 (1992→1658 µs/tok) — consistent with the
  expert union approaching the 128-expert ceiling (8×9=72 of 128 → fewer *new* experts per token).
  This is the mechanism behind the prior section's "sub-linear but not free" verify curve.
- The full-logit distinct marginal (kW, W1→7) ≈ **2.0ms/token**, reproducing the prior 2.07ms,
  now decomposed: ~71µs LM-head compute + ~900µs irreducible body (attention over the growing
  prefix + dense layers + the experts a lone token needs) + ~500–1000µs **expert fan-out**.

## Result E3 — the "tiny" MTP draft is 57% shared LM head (confirms H2)
MTP draft forward (avg µs): body (logits off) = **401** + LM head (off→on) = **524** = **926** total.
- The draft transformer body is genuinely tiny (~401µs, one nextn layer). But producing a 262K
  draft distribution forces the **same** ~782MB unembedding read the target pays (implied
  ~1492 GB/s, ≈ HBM bandwidth) → **57% of the draft forward is the shared LM head**, invariant to
  how small the head's body is. This is why a 461MB head (~1.7% of the target) is not ~1.7% of
  per-token work. H2 confirmed.

## Unifying picture
Every structural fact in the prior section follows from one cause — **weight bandwidth, split into
a fixed and a marginal term**:
- **Fixed per forward:** LM-head read (~0.8–1.2ms) + always-on dense/attention weights → the
  ~7ms verify floor and the ~0.4ms of the 0.9ms draft.
- **Marginal per token:** MoE expert fan-out, bandwidth-bound and **diversity-dependent**,
  saturating as the 128-expert union fills. The only "free" extra token would be a redundant one
  routing to already-resident experts; real verification batches are diverse, so each token is
  ~2ms. This is why verification is sub-linear-but-not-free and throughput peaks at small γ.

## Measurement caveats
- The identical-token ("same") body cost is **non-monotonic** in W (W=9 < W=7), a CUDA
  kernel/graph tiling effect at specific batch sizes; the fan-out claim therefore rests on the
  *within-W* (distinct − same) gap, which holds batch size — and thus kernel selection — fixed.
- The LM-head fixed read measures ~1201µs in the target k0→k1 delta but ~524µs in the single-token
  MTP path; the target figure carries extra per-batch graph/output overhead, so the MTP single-
  token number (~524µs, ~1.5TB/s) is the cleaner weight-read estimate. Either way the LM head is a
  large fixed cost and a negligible per-token one.
- Single-GPU absolute µs (data-parallel per-worker config); the prior γ-sweep was 2-GPU
  tensor-split, so absolute values differ but the structure matches (single-GPU width-1 target
  8.9ms here ≈ prior 9.1ms).

---

# Extension 2 — a structurally different way to amortize the fixed per-forward cost: request batching vs speculative depth

## Angle
The prior sections established the dominant cost is **weight bandwidth = a large FIXED per-forward
term + a marginal MoE term**, and that the fixed term is largely invariant to how many tokens a
forward processes. All prior amortization happened along **one** axis: speculative DEPTH — packing
W candidate tokens of a *single* sequence into one verify forward. That axis is capped by
acceptance/γ (peak tok/s at γ≈2–4) and gives a single-sequence ~2.1× ceiling.

But the weights are read **once per forward regardless of which sequence the tokens belong to**.
So there is an orthogonal amortization axis the prior work never touched: **request batching** —
pack B tokens from B *independent* sequences (concurrent requests), each decoding its own next
token. This amortizes the *same* fixed cost but is **not capped by acceptance**. This extension
measures both axes head-to-head and asks: (Q1) does request batching amortize the fixed cost as
well as speculative depth? (Q2) where does the two differ? (Q3) what does this do to the prior
2.1× ceiling?

## Method — new `--mode batch` (isolated branch, non-perturbing)
New isolated `--mode batch` in `initial_program.cpp` (returns before the scored generation loop;
sets `n_parallel`/`n_seq_max` only for this mode). Scored `target`/`mtp-spec` paths are byte-for-
byte unchanged — smoke: **mtp-spec 1.61× / 81.3% accept**, matching prior.
A controlled comparison at matched width **X** and fixed prefix length **L**, tokens drawn from a
shared real-vocab pool (valid embeddings):
- **cross(X):** X sequences × 1 query token each (request batching) — KV holds X disjoint L-prefixes.
- **within(X):** 1 sequence × X query tokens (speculative depth, the prior axis) — 1 shared L-prefix.
Both issue X query tokens. Levers reused from the probe: **logits off (k0, body only) vs on (kX,
all X queries get a 262K row)** isolates LM-head amortization; **distinct vs identical content**
isolates MoE fan-out. Single-GPU (CUDA_VISIBLE_DEVICES=2/3), medians of 30 reps after 8 warmups.
Primary L=256; robustness L=64. (X=1 width-1 forward 9.0ms ≈ prior single-GPU 8.9ms.)

## Result B1 — request batching amortizes the fixed cost ~10×, with NO acceptance cap (answers Q1, Q3)
Realistic case (cross, distinct content, logits-on), per-forward and **per-token** µs, L=256:

| B (concurrent reqs) | forward (µs) | per-token (µs) | per-token speedup vs B=1 |
|---|---|---|---|
| 1 | 9012 | 9012 | 1.0× |
| 2 | 11231 | 5616 | 1.6× |
| 4 | 16248 | 4062 | 2.2× |
| 8 | 27465 | 3433 | 2.6× |
| 16 | 33323 | 2083 | 4.3× |
| 32 | 41432 | 1295 | 7.0× |
| 64 | 58315 | **911** | **9.9×** |

The width-1 forward is ~88% fixed weight read (body k0 7843µs + LM-head fixed ~1.1ms of the 1169µs
k0→kX gap); batching B independent requests spreads that fixed cost over B tokens, so per-token
cost falls 9.0ms→0.91ms (**9.9×**) by B=64. This is throughput the prior 2.1×-ceiling analysis did
not see: **the ceiling is the B=1 endpoint.** Request batching reaches spec-dec's ~1.6× by B≈2–3
and ~10× by B=64, unbounded by acceptance.

## Result B2 — the two axes amortize the FIXED cost identically; they differ only in the MARGINAL MoE term (answers Q2)
Compare cross(X) vs within(X) at matched X, split by content (L=256, body-only k0 µs):

| X | cross_dist | within_dist | cross_same | within_same |
|---|---|---|---|---|
| 1 | 7843 | 7928 | 7845 | 7860 |
| 8 | 25225 | 22752 | 15174 | 18958 |
| 64 | 54277 | 35043 | 29193 | 27346 |

- **Identical content: cross ≈ within at every X** (X=64: 29193 vs 27346). With routing held fixed,
  the fixed weight-read amortizes *axis-independently* — request batching and speculative depth are
  the same amortization of the same fixed cost. (The small residual is KV layout, see B3.)
- **Distinct content: cross ≫ within** and the gap grows with X (X=64: 54277 vs 35043, +19ms).
  The divergence is **not** attention/KV bandwidth — it is **MoE expert fan-out**. Independent
  concurrent requests have decorrelated hidden states → route to a *larger union* of the 128
  experts → pull more expert weight bandwidth than same-prefix speculative candidates do.
- Body marginal per added token (k0 slope, X=1→64):
  - cross_dist **737 µs/tok**, cross_same 339 → cross-sequence fan-out ≈ **398 µs/tok**
  - within_dist 430 µs/tok, within_same 309 → within-sequence fan-out ≈ **121 µs/tok**
  Cross-sequence (concurrent-request) fan-out is **~3.3× larger** than within-sequence
  (speculative) fan-out at matched width: diverse real requests hit the expert-bandwidth wall
  far harder than a single sequence's speculative continuation.
- LM-head term reproduces prior E1 independent of axis: k0→kX gap = ~1123µs **fixed** + ~46µs/row
  (cross X=1 gap 1169µs; X=64 gap 4038µs over 64 rows). At B=64 the 1.1ms fixed LM-head read is
  shared by 64 requests → ~18µs/request.

## Result B3 — the cross−within gap is mostly L-invariant (rules out KV bandwidth as the cause)
Re-run at short prefix L=64 (4× less KV): cross_dist_kX at X=64 = 52822µs (L=256: 58315);
within_dist_kX at X=64 = 39002µs (L=256: 39006). A 4× change in KV size moves the gap only
~14ms→19ms (k0, X=64). The bulk of the cross>within divergence survives at tiny L, so it is the
content/fan-out term, not the disjoint-prefix attention reads (which contribute the small,
L-growing remainder).

## Unifying picture (extends the prior "weight bandwidth = fixed + marginal" model)
- **Fixed per forward (LM-head ~1.1ms read + always-on dense/attention weights):** amortizes
  identically along *either* axis — speculative depth OR concurrent requests. Speculative depth is
  acceptance-capped (~γ≈2–4); request batching is not, so it drives the fixed-cost-per-token toward
  zero (9.0ms→0.91ms at B=64) and the 2.1× single-sequence ceiling is just the B=1 point.
- **Marginal per token (MoE expert fan-out):** the floor the amortization runs into, and it is
  **larger for request batching than for speculative depth** — independent requests are
  routing-decorrelated and saturate the 128-expert union faster (~398 vs ~121 µs/tok fan-out).
- **Consequence:** the two axes are *substitutes* for amortizing the fixed cost. Once a server runs
  at batch B, the fixed cost is already amortized, so a speculative token added on top no longer
  rides a "nearly-free" fixed forward — it costs the full MoE fan-out marginal (and the
  cross-sequence measurement shows diverse batches pay that marginal heavily). The premise that
  makes speculative decoding win at B=1 (extra verify tokens are nearly free against the fixed cost)
  erodes as serving batch size grows.

## Measurement caveats (Extension 2)
- Single-GPU data-parallel config; absolute µs differ from the 2-GPU tensor-split γ-sweep but the
  structure matches. Default `kv_unified` and `n_batch`; `n_seq_max=80`.
- The "same content" baseline (identical L-prefix + identical query token across all sequences)
  is a *lower bound* on fan-out (minimal routing diversity); real concurrent requests are at least
  as diverse as the "distinct" pool used here.
- "cross" stores B disjoint L-prefixes (KV = B·L) while "within" stores one (KV = L); the
  L=64↔256 comparison isolates this as a small, L-scaling residual on top of the fan-out gap.
- The distinct token pool is built from the 30 benchmark prompts, tiled to cover B·L tokens; for
  B·L beyond the pool's ~6k tokens the windows repeat, mildly *reducing* cross-sequence diversity
  at the largest B (so the measured high-B fan-out is, if anything, a lower bound).

---

# Extension 3 — interaction surface: batching and speculative depth together

## Angle
Extension 2 showed that request batching and speculative depth both amortize the same fixed
per-forward weight-read cost, so they should partially substitute for each other. This extension
measures the full two-axis surface instead of only the pure endpoints: **B independent sequences ×
D query tokens per sequence** in one target verification forward. The factual question is how the
benefit of increasing one axis changes as the other axis is already large.

## Method — new `--mode interact`
Added isolated `--mode interact` in `initial_program.cpp`. It pre-fills B independent L-token
sequences, then verifies D positions per sequence in one forward (total query tokens N=B·D).
The scored `target`/`mtp-spec` paths are unchanged. Same levers as the earlier probes:
- `dist` vs `same` content separates real routing diversity from minimal expert fan-out.
- logits off/on separates body cost from LM-head rows.

Run: `CUDA_VISIBLE_DEVICES=0,1 PROBE_REPS=20 PROBE_L=256 ./spec_modes --mode interact ...` using
all 30 benchmark prompts as the token pool. Times below are medians in microseconds.

Full scored verification after adding the probe, using `./evaluator/task-eval --gpus 0,1`:
**93.3% accuracy**, **149.2 tok/s**, **72.4% accept**, **1.48× speedup** for `mtp-spec`.

## Result I1 — per-token cost surface (distinct content, logits on)

| B \ D | 1 | 2 | 4 | 8 |
|---:|---:|---:|---:|---:|
| 1  | 8982 | 5622 | 3890 | 3158 |
| 2  | 5647 | 4008 | 3185 | 1626 |
| 4  | 4170 | 3339 | 1800 | 1004 |
| 8  | 3431 | 1955 | 1078 | 707 |
| 16 | 2022 | 1159 | 751 | 569 |
| 32 | 1254 | 804 | 601 | 389 |
| 64 | 890 | 645 | 414 | **301** |

Combined axes reduce target verify cost from **8982µs/token** at B=1,D=1 to
**301µs/token** at B=64,D=8 (**29.8×** per-token throughput increase for this isolated forward).
The pure endpoint gains are 10.1× for B=1→64 at D=1 and 2.84× for D=1→8 at B=1; their product
predicts 28.7×, close to the measured 29.8×. At the coarse throughput level the two axes combine
close to multiplicatively.

## Result I2 — the substitution is visible in absolute marginal savings
Increasing D from 1→8 saves fewer microseconds/token once B is already large:

| B | D=1→8 relative gain | absolute saved µs/token |
|---:|---:|---:|
| 1  | 2.84× | 5824 |
| 8  | 4.85× | 2724 |
| 16 | 3.55× | 1453 |
| 32 | 3.22× | 865 |
| 64 | 2.95× | **588** |

So the fixed-cost substitution is real in absolute terms: the first speculative-depth expansion
at B=1 removes milliseconds/token of fixed-cost burden, while the same D=1→8 expansion at B=64
removes only hundreds of microseconds/token because request batching has already shared most of
the fixed read.

The reverse direction is less diminished in this grid: B=1→64 gives 10.1× at D=1, 8.7× at D=2,
9.4× at D=4, and 10.5× at D=8. Batch scaling remains strong even after depth is present because
it increases total forward width from D to 64D and continues to share fixed LM-head/dense reads.

## Result I3 — depth amortizes expert fan-out inside batched requests
The body-only fan-out gap (`dist_k0 - same_k0`) is large for request batching, but its **per-token**
cost shrinks as D grows inside a fixed B:

| B | fan-out µs/token at D=1 | fan-out µs/token at D=8 |
|---:|---:|---:|
| 8  | 1294 | 235 |
| 16 | 897 | 213 |
| 32 | 539 | 124 |
| 64 | 369 | **67** |

This refines the "axes are substitutes" statement. They substitute for the fixed per-forward
read, but adding depth on top of a request batch also spreads the expert-union fan-out over more
tokens in the same forward. That is why relative D gains do not collapse at high B even though
the absolute fixed-cost savings are much smaller.

## Measurement caveats (Extension 3)
- This is a controlled verification-forward benchmark, not an end-to-end serving benchmark:
  acceptance, draft cost, scheduling gaps, and queueing latency are excluded.
- D is the number of target query positions per sequence in the controlled forward. In real
  speculative decoding, useful committed tokens are acceptance-limited.
- Absolute times use visible GPUs 0,1 for one process; earlier single-GPU probe numbers are not
  directly comparable in milliseconds, but the decomposition levers are identical.

---

# Extension 4 — practical break-even threshold for depth once batching exists

## Angle
Extension 3 established that request batching and speculative depth share the same fixed-cost
amortization pool. This extension converts that interaction surface into a concrete decision
threshold: for the current `gamma=4` path (target verifies `D=5` positions), at what request batch
size does speculative depth stop paying for itself after measured acceptance and measured
draft/sampling overhead are included?

## Method
Added a small non-perturbing control hook to `--mode interact`: `PROBE_BATCHES` and
`PROBE_DEPTHS` can restrict the sweep without changing scored generation. Measured the
decision-relevant surface directly at `D=5` rather than interpolating between `D=4` and `D=8`.

Primary threshold surface below is a single-GPU worker probe (`CUDA_VISIBLE_DEVICES=0`,
`PROBE_REPS=10`, `PROBE_L=256`, `PROBE_DEPTHS=1,5`, `PROBE_BATCHES=1,2,4,8,16,32,64`) because
`./evaluator/task-eval --gpus 0,1` runs two single-GPU workers. A two-GPU tensor-split repeat
gave the same threshold class (B=1,2 positive; B>=4 non-positive with current overhead).

Full scored verification after adding the probe hook, using `./evaluator/task-eval --gpus 0,1`:
**93.3% accuracy**, **144.4 tok/s**, **72.4% accept**, **1.43x speedup** for `mtp-spec`.

From the full eval timing counters:
- rounds = 65,830; drafted = 263,320; accepted = 190,764; decoded = 256,580.
- useful tokens/round = **3.898** for `gamma=4`.
- non-target overhead/round = draft_fwd + draft_smp + verify_smp + cache =
  **8,656us** (4,572 + 2,029 + 2,010 + 45).

Break-even condition for `gamma=4` at serving batch B:

`5 * V(B,5) + H < U * V(B,1)`

where `V(B,D)` is measured target verify time per query token, `U=3.898` useful tokens per
speculative round, and `H` is all non-target overhead per sequence-round. Equivalently, the
maximum tolerable overhead is `U * V(B,1) - 5 * V(B,5)`.

## Result T1 — measured overhead budget for current `gamma=4`

| B | target D=1 us/token | target D=5 us/query | max overhead us/round | current net delta us/token | decision |
|---:|---:|---:|---:|---:|---|
| 1  | 8988.0 | 3474.8 | 17658 | +2310 | worthwhile |
| 2  | 5610.5 | 2252.3 | 10606 | +500 | worthwhile |
| 4  | 3944.0 | 1522.2 | 7761  | -230 | not worthwhile |
| 8  | 3330.0 | 967.6  | 8141  | -132 | not worthwhile |
| 16 | 2022.4 | 698.6  | 4390  | -1095 | not worthwhile |
| 32 | 1281.4 | 476.5  | 2612  | -1551 | not worthwhile |
| 64 | 922.6  | 373.5  | 1728  | -1777 | not worthwhile |

The current implementation's 8,656us non-target overhead fits comfortably under the B=1 budget
and narrowly under B=2. It exceeds the measured budget from B=4 onward. The practical threshold
for this implementation is therefore **between B=2 and B=4 concurrent requests per worker**.
Beyond that point, request batching has already amortized enough fixed target cost that adding
`gamma=4` speculative depth is neutral to negative in this measured cost model.

## Result T2 — target-only depth remains useful, but not enough to cover the draft path
Ignoring draft/sampling overhead, `D=5` still reduces target verify cost at all measured B:

| B | target-only gain from D=1 to D=5 |
|---:|---:|
| 1  | 2.02x |
| 2  | 1.94x |
| 4  | 2.02x |
| 8  | 2.68x |
| 16 | 2.25x |
| 32 | 2.10x |
| 64 | 1.93x |

So the threshold is not caused by depth failing to amortize the target forward. It is caused by
the absolute value of the saved target time shrinking to the same scale as the draft forward plus
two CPU sampling phases. At B=64, `gamma=4` has only 1.73ms/round of overhead budget while the
measured current overhead is 8.66ms/round.

## Measurement caveats (Extension 4)
- This is a cost-model threshold using measured forward timings, measured acceptance, and measured
  per-round overhead from the current implementation; it is not a multi-request end-to-end server
  benchmark.
- `H` is the current unbatched per-sequence MTP draft plus CPU sampling path. The table's "max
  overhead" column is independent of that implementation detail and states the break-even budget
  imposed by the measured target surface.
- The B=4 and B=8 negative margins are small (hundreds of microseconds/token), so the exact
  crossing point is noisy. The B=16,32,64 points are well below break-even under the same measured
  overhead.
