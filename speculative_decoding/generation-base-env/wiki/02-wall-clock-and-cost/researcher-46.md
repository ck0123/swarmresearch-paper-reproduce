# Findings — synthesis of two contradicting wall-clock analyses

This branch merges two prior independent analyses of MTP speculative decoding that reached
**opposite conclusions about where wall-clock time goes**:

- **Line A (researcher-29):** the timed GPU decode phases are only **4.4% of wall-clock**; the
  remaining **95.6%** is untimed host-side `make_dist` (an O(vocab) `partial_sort` over the
  ~262K-token Gemma vocabulary). Conclusion: the system is **sampling-bound**.
- **Line B (researcher-8):** that result is an artifact of **asynchronous `llama_decode`**. After
  inserting an explicit `llama_synchronize()` after every decode, GPU forward = **87.1%** of
  wall-clock and the target **verify forward pass dominates at 69.4%**; `make_dist` is ~12.6%.
  Conclusion: the system is **verify-bound**.

Both lines analyzed the same workload and agreed on the *outcome* of the run (≈1720 s wall,
≈149 tok/s, 93.3% accuracy, 72.4% accept). They disagreed only on the *decomposition* of that
wall-clock. The synthesis below resolves the disagreement with a single controlled measurement,
then keeps the (undisputed) acceptance-side findings both lines built on.

---

## Resolution — single-run dual-attribution timing experiment

Evaluation command: `./evaluator/task-eval --gpus 2,3` (full 30-item run). Run summary with this
instrumentation: `mtp-spec` accuracy **93.3%** (identical to the target reference — instrumentation
is non-perturbing to quality), 140.6 tok/s, speedup 1.39x, aggregate accept 72.4%. The throughput
is lower than the clean reference (151.8 tok/s) because this build carries *both* lines' overheads
at once (researcher-29's per-row overlap/entropy instrumentation **and** researcher-8's explicit
synchronize barriers); this affects absolute tok/s but not the relative decomposition that is the
subject of the dispute.

### Why the two lines disagree (root cause)

`llama_decode` is **asynchronous**: it returns after enqueuing GPU work, before that work
completes. The active program (Line A) times `draft_us`/`verify_us` by wrapping `llama_decode`
**alone**, with no synchronize. Those timers therefore capture only the host-side kernel *launch*,
not the GPU forward compute. The forward compute completes later, at the first operation that
forces a device sync — `llama_get_logits_ith` **inside the next `make_dist`**. So in Line A's
accounting the GPU forward time is silently transferred onto the `make_dist` wall-time that
follows it. Line B's `llama_synchronize()` after each decode forces the forward to complete inside
the decode timer, moving that same time back where it belongs.

### Method

The active `initial_program.cpp` was instrumented to measure, around **every** `llama_decode`,
three disjoint quantities from one set of timestamps:
- `*_launch_us` — `llama_decode` alone (Line A's method, no sync),
- `*_sync_us` — an explicit `llama_synchronize()` immediately after (captures exactly the GPU
  forward time the launch timer misses, i.e. the time that without the sync would be billed to the
  following `make_dist`),
- `*_smp_us` — `make_dist`+sample, now executing as **pure CPU** (its input logits are already
  resident).

Line A's decomposition is then `launch` for decode with `sync` folded into `make_dist`; Line B's
decomposition is `launch + sync` for the forward pass with `make_dist` as pure CPU. Both are
computed from the **same** run; the only difference is which bucket the `sync` time lands in.

### Result — both prior conclusions reproduced from one run

Aggregate over 30 prompts: 256,580 tokens, 1824.5 s summed wall, 565,634 `make_dist` calls. Timed
buckets sum to 1819.5 s (99.7% of wall; the 5.0 s remainder is cache/bookkeeping).

Raw phase totals (seconds):

| phase | launch (Line-A "decode") | sync (leaked GPU forward) | make_dist (pure CPU) |
|---|---:|---:|---:|
| prompt/prefill | 2.3 | 1.2 | — |
| draft (MTP) | 57.4 | 249.9 | 143.1 |
| verify (target) | 26.1 | 1170.8 | 168.7 |

**Reproducing Line A** (launch-only decode; sync time leaks into `make_dist`):

| bucket | time | share of measured |
|---|---:|---:|
| timed GPU decode (launch only) | 85.8 s | **4.7%** |
| host-side `make_dist` (as Line A billed it) | 1733.8 s | **95.3%** |

This matches researcher-29's recorded "timed GPU decode = 4.4% of wall, 95.6% untimed host-side
`make_dist`."

**Reproducing Line B** (decode = launch + sync = true GPU forward; `make_dist` = pure CPU):

| phase | time | share of timed |
|---|---:|---:|
| prefill | 3.5 s | 0.2% |
| draft_fwd (MTP) | 307.3 s | 16.9% |
| **verify_fwd (target)** | **1196.9 s** | **65.8%** |
| draft_smp (CPU make_dist) | 143.1 s | 7.9% |
| verify_smp (CPU make_dist) | 168.7 s | 9.3% |
| GPU forward total | 1507.7 s | **82.9%** |
| CPU `make_dist` total | 311.8 s | **17.1%** |

This matches researcher-8's recorded "GPU forward = 87.1%, target verify forward dominates at
69.4%, CPU sampling+cache = 12.8%." The residual gaps (verify_fwd 65.8% vs 69.4%; CPU 17.1% vs
12.8%) are accounted for by this build carrying researcher-29's heavier per-verify-row `make_dist`
instrumentation (a `Dist` is built for **every** verified row plus `overlap_decomp`/entropy),
which inflates the CPU `make_dist` term relative to researcher-8's lean path.

### The reclassified quantity (the entire dispute, isolated)

The disagreement is one number: the `sync` time, the GPU forward compute that Line A's launch-only
timers could not see.

- sync time mis-attributed to `make_dist` by Line A = **1422.0 s = 78.2% of measured time**.
- pure-CPU cost per `make_dist` call (262K-vocab top-64 `partial_sort` + Dist build) = **551 µs**;
  over 565,634 calls = 311.8 s.

So 78 of Line A's 95 "make_dist" percentage points are GPU forward compute waiting at the sort's
sync point; only ~17 points are the sort itself.

### Reconciled account

1. Both lines measured the same physical run (≈1820 s wall, ≈140–150 tok/s, 93.3% accuracy). The
   contradiction was never about the data, only about attribution.
2. The system is **verify-bound**, not sampling-bound: the target verification forward pass is
   **~66% of timed wall-clock** (researcher-8's ~69% on its lean build), the largest single cost.
   MTP drafting forward is the second cost (~17%).
3. Line A's "95% make_dist / sampling-bound" is an **async-attribution artifact**: launch-only
   decode timers leave the GPU forward time to be charged to the next sync, which is `make_dist`.
4. Line A's raw observation was not false, only mislabeled: `make_dist`'s wall-time really was
   ~95% of measured time — but ~78 of those points are GPU forward, not sorting.
5. The CPU sort is nonetheless **fully exposed on the critical path** and is a real cost
   (~12.8–17.1%): inserting synchronize barriers does **not** change total wall-clock
   (researcher-8: 1724 s with barriers ≈ researcher-29: 1719 s without), which shows there is no
   CPU/GPU overlap in the current codepath for the barriers to destroy — every decode's output is
   immediately consumed by a `make_dist` that forces a sync regardless. Both the GPU forward
   (~83%) and the CPU sort (~17%) are additive contributors to wall-clock. The kernel of truth in
   Line A is that the sort is not free/hidden; the correction is its magnitude (~17%, not ~95%).

### Per-op GPU forward costs (reconciled with researcher-8's γ-sweep)

From the sync-attributed forward times on the full γ=4 run: a serial MTP draft forward averages
307.3 s / draft_calls and the target verify forward averages 1196.9 s / verify_calls. researcher-8's
single-tensor-split γ-sweep fit the verify forward as **≈7.0 ms fixed + ≈2.07 ms per token in the
batch**, and an MTP draft step at ≈1.9 ms (tensor-split) / ≈1.15 ms (single-GPU). Both lines agree
on the structural asymmetry: **drafting is linear in γ (γ sequential width-1 passes), verification
is sub-linear (one batched pass), and a 5-token verify batch costs ≈1.96× a single token — far
from free**, because the MoE target (4B active of 26B) is not in the weight-bandwidth-bound regime
where extra batch tokens are nearly free.

---

# Extension — practical reducibility of the exposed CPU top-k cost

The exposed CPU `make_dist` cost above is not an irreducible consequence of speculative decoding;
it depends strongly on the host-side top-k implementation used to construct the truncated
distribution.

## Method

The original `make_dist` built a 262K-entry `(logit, token_id)` vector on every call, then used
`std::partial_sort` to order the top 64 entries. Two exact top-k alternatives were evaluated on
the same full 30-item benchmark using `./evaluator/task-eval --gpus 4,5`:

1. **Full-buffer `nth_element` + sort top-k:** still materializes the 262K-entry vector, partitions
   it with `std::nth_element`, then sorts the retained 64 entries.
2. **Bounded min-heap scan:** scans logits once, retains only the current top 64 candidates in a
   64-entry heap, then sorts those 64 entries before the same top-p/temp normalization. This avoids
   allocating and permuting the full vocabulary-sized candidate vector. The selected top-k set is
   exact except for irrelevant equal-logit tie ordering.

## Result

All three variants used the same number of distribution constructions on the benchmark:
565,634 `make_dist` calls. Accuracy and acceptance were unchanged in the two evaluated variants:
`mtp-spec` accuracy 93.3%, accept rate 72.4%.

| `make_dist` implementation | CPU `make_dist` time | per call | `mtp-spec` tok/s | speedup vs target ref |
|---|---:|---:|---:|---:|
| original full-vector `partial_sort` | 311.8 s | 551 µs | 140.6 | 1.39x |
| full-vector `nth_element` + sort top-k | 755.1 s | 1335 µs | 112.6 | 1.12x |
| bounded 64-entry min-heap scan | 141.5 s | 250 µs | 155.3 | 1.54x |

The `nth_element` substitution was worse than `partial_sort`: keeping the full 262K-entry buffer
but changing the selection primitive increased CPU distribution time by 443.3 s (+142%).

The bounded-heap version reduced exposed CPU distribution time by 170.3 s relative to the original
`partial_sort` path (311.8 s → 141.5 s, −54.6%). On the same full benchmark it decoded 256,580
tokens in 1644.1 s aggregate wall time (156.1 aggregate tok/s; evaluator mean 155.3 tok/s). The
timed buckets summed to 1640.7 s, of which CPU `make_dist` was 141.5 s = 8.6%; in the original
dual-attribution run CPU `make_dist` was 311.8 s = 17.1% of timed work.

## Interpretation

The exposed CPU cost was practically reducible by about half without changing the speculative
decoding accept/reject logic, the target/draft distributions, benchmark accuracy, or accept rate.
The decisive implementation detail was not replacing `partial_sort` with another full-buffer
selection primitive; it was avoiding the vocabulary-sized candidate vector and retaining only the
top-k frontier during the scan.

---

# Acceptance-side findings (undisputed; carried forward from both lines)

These analyses concern the **numerator** of speedup (useful tokens per round) and are independent
of the timing dispute above. They are recorded as established facts.

## Draft position reach and confidence

Later draft positions were often generated but never reached by the acceptance loop because an
earlier token in the same round rejected:

| draft position | drafted | reached | accepted | rejected | unreached | reached/drafted |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 65,830 | 65,830 | 57,316 | 8,514 | 0 | 100.00% |
| 1 | 65,830 | 57,316 | 50,234 | 7,082 | 8,514 | 87.07% |
| 2 | 65,830 | 50,234 | 44,250 | 5,984 | 15,596 | 76.31% |
| 3 | 65,830 | 44,250 | 38,964 | 5,286 | 21,580 | 67.22% |

Conditional acceptance given that a draft position was reached was similar across positions:
87.07%, 87.64%, 88.09%, and 88.05% for positions 0 through 3.

Target/draft top-1 agreement separated accepted and rejected reached drafts more strongly than
draft sampled-token probability. Among accepted reached drafts, top-1 agreement was 96.24%,
96.43%, 96.79%, 96.97% for positions 0–3; among rejected reached drafts, 50.29%, 43.93%, 40.41%,
37.80%. The sampled draft token's mean rank under the target distribution was ≈1 for accepted
drafts and ≈2 for rejected drafts.

## Distributional-overlap decomposition of acceptance loss

At each reached draft position the distributional overlap `O = sum_x min(p(x), q(x)) = 1 - TV(p,q)`
between the truncated-renormalized target `p` and draft `q` is the theoretical single-token accept
probability. Over 217,630 reached draft positions, empirical conditional acceptance was 87.66% and
mean theoretical `O` was 87.56% (empirical acceptance matches theory within each entropy bin and in
aggregate). Mean loss `1 - O` = 12.44% decomposed into **2.32% out-of-nucleus mass** (draft mass on
tokens with `p(x)=0`, structurally unacceptable) and **10.12% overconfidence mass** (`q(x)>p(x)>0`,
draft over-weighting shared tokens) — overconfidence is 4.4x the out-of-nucleus mass. Of 26,866
rejection events, 81.2% were in-nucleus and 18.8% out-of-nucleus.

The draft distribution was sharper than the target: mean target entropy over reached positions
0.2405; on accepted positions target entropy 0.1494 vs draft 0.0454; on rejected positions target
0.8872 vs draft 0.2715.

Acceptance stratified by target entropy `H(p)`:

| H(p) bin | reached | accept | theoretical O | share of reached |
|---|---:|---:|---:|---:|
| [0, 0.25) | 158,247 | 97.88% | 97.87% | 72.71% |
| [0.25, 0.5) | 13,847 | 77.92% | 77.86% | 6.36% |
| [0.5, 1.0) | 26,749 | 62.20% | 61.82% | 12.29% |
| [1.0, 2.0) | 17,328 | 45.77% | 45.51% | 7.96% |
| [2.0, 3.0) | 1,452 | 34.99% | 32.95% | 0.67% |
| [3.0, inf) | 7 | 28.57% | 24.90% | 0.00% |

72.71% of reached positions had target entropy below 0.25 and accepted at 97.88%.

## Round-level path-accept calibration

Across 65,830 rounds: 38,964 all-accepted, 26,866 rejected. Observed all-accept rate 59.19%; mean
sampled-path all-accept probability `prod_g min(1, p(draft_g)/q(draft_g))` = 58.99%; mean product
of overlaps 58.80%. The sampled-path all-accept probability separated outcomes (86.04% on
all-accepted rounds vs 19.76% on rejected). Binning by predicted path-accept, the top bin [0.98,1]
held 35.51% of rounds at 100.00% all-accept and 3 dead verify rows; the bottom bin [0,0.25) held
30.31% of rounds and 70.25% of all dead verify rows. First-rejection position over rejected rounds:
pos 0 31.69%, pos 1 26.36%, pos 2 22.27%, pos 3 19.68%.

## Drafted-pass waste and draft-side adaptive-gating counterfactual

Of 263,320 drafted tokens, 190,764 accepted, so 72,556 (27.6%) were wasted serial MTP draft passes
(the most expensive per-token GPU op); 22.04% of (cheaper) verify rows were dead. A draft-side-only
stopping rule — truncate the round at `G'(tau) = (first g with draft entropy H(q_g) > tau) + 1` —
depends only on the draft distribution observed before verification, so it is realizable online,
and truncating speculation length is lossless (the bonus token remains a target sample). At the
aggressive plateau (tau in [0.02, 0.10]) it truncates 33.56% of rounds, removes 18.20% of serial
draft passes, loses 6.52% of emitted tokens, and raises emitted-tokens-per-draft-pass from 0.9745
to 1.1136 (+14.3%); amortized decode-throughput gain 1.10x (GPU-decode model) to 1.115x
(sampling-bound model), the two models agreeing within ~1.5%.

Realizable round-level predictor by max draft entropy across the round:

| max H(q) bin | rounds | share | all-accept | mean accepted tokens |
|---|---:|---:|---:|---:|
| [0, 0.25) | 41,421 | 62.92% | 75.94% | 3.385 |
| [0.25, 0.5) | 6,949 | 10.56% | 36.64% | 2.270 |
| [0.5, 1.0) | 13,677 | 20.78% | 31.75% | 2.107 |
| [1.0, 2.0) | 3,661 | 5.56% | 16.77% | 1.601 |
| [2.0, 3.0) | 117 | 0.18% | 4.27% | 0.957 |
| [3.0, inf) | 5 | 0.01% | 0.00% | 1.200 |

The draft-side signal is realizable but noisier than the target-side path-accept predictor: the
draft top bin (max H(q) < 0.25) holds 62.92% of rounds yet only 75.94% fully accept, whereas the
target-side top bin held 35.51% of rounds at 100% all-accept — consistent with the draft
overconfidence measured in the overlap decomposition.
</content>
</invoke>
