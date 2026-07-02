# Findings

## Angle: per-depth draft acceptance decay and the γ throughput ceiling (mtp-spec)

Method: instrumented `initial_program.cpp` (mtp-spec branch) to emit, per round,
`DEPTH_DRAFTED[g]` (rounds a draft token existed at depth g), `DEPTH_ACCEPT[g]`
(rounds the depth-g draft was accepted), and `ACCEPT_HIST` (accepted-run-length
histogram 0..γ). Aggregated with `analyze_depth.py`. Drafts always reach full γ
(no early break observed), so `DEPTH_DRAFTED[g] == n_rounds` for all g.
Full 30-item benchmark, GPUs 0,1, temp 1.0 / top_p 0.95 / top_k 64 / seed 0.

### Headline numbers
| γ | accuracy | tok/s | speedup | overall accept | mean accepted/round | tokens/target-pass | n_rounds |
|---|---|---|---|---|---|---|---|
| 4 | 93.3% | 148.4 | 1.47× | 72.45% | 2.898 | 3.898 | 65,830 |
| 8 | 96.7% | 137.9 | 1.37× | 53.86% | 4.308 | 5.308 | 48,048 |

(target reference: 93.3% acc, 100.7 tok/s. γ=4 reproduces README's 72.4% accept /
~1.5× speedup. γ=8 accuracy 96.7% vs 93.3% is stochastic — temp=1.0 with different γ
draws a different RNG sequence; rejection sampling is lossless w.r.t. the target
distribution at any γ.)

### Per-depth marginal conditional acceptance is flat (no depth degradation)
Marginal conditional accept at depth g = accepted_at[g] / accepted_at[g-1]
(probability the depth-g draft is accepted *given* the prefix was accepted).

γ=4:
```
depth | accept-up-to | marginal-cond
  0   |   87.07%     |   87.07%
  1   |   76.31%     |   87.64%
  2   |   67.22%     |   88.09%
  3   |   59.19%     |   88.05%
```
γ=8:
```
depth | accept-up-to | marginal-cond
  0   |   84.87%     |   84.87%
  1   |   72.83%     |   85.81%
  2   |   63.25%     |   86.84%
  3   |   54.93%     |   86.86%
  4   |   48.00%     |   87.39%
  5   |   41.50%     |   86.46%
  6   |   35.41%     |   85.33%
  7   |   30.05%     |   84.84%
```
The marginal conditional accept rate is constant in depth (~0.86–0.88) out to depth 7;
it does not decay. The MTP draft chain feeds its own hidden state forward yet does not
lose calibration with depth over the measured range. The steep *unconditional* decay
(γ=4: 87→76→67→59%; γ=8: 85→73→63→…→30%) is the geometric product of a near-constant
per-step rate, not rising per-step difficulty.

### Geometric/constant-p model is essentially exact
With per-step accept p, mean accepted run length for block γ = Σ_{k=1..γ} p^k.
Using p≈0.86 (the γ=8 marginal): Σ_{k=1..8} 0.86^k = 4.30, vs measured 4.308.
Using p≈0.88 (γ=4 marginals): Σ_{k=1..4} 0.88^k = 2.935, vs measured 2.898.
Implied ceiling as γ→∞: p/(1-p) ≈ 0.86/0.14 ≈ 6–7 accepted draft tokens/round.

### Accepted-run-length distribution is bimodal
γ=4: full-accept (=4) 59.2%, zero-accept (=0) 12.9%, middle lengths 8–11% each.
γ=8: full-accept (=8) 30.0%, zero-accept (=0) 15.1%, decaying tail in between.
Rounds either accept the whole block or reject early; intermediate lengths are the
minority.

### Throughput falls from γ=4 to γ=8 despite more tokens/pass
γ=8 produces more tokens per target verification pass (5.308 vs 3.898, +36%) but
tok/s drops (148.4 → 137.9). Per-round cost ≈ T_target_verify + γ·t_draft (γ sequential
MTP decodes + one target verify of γ+1 tokens). Solving
(5.308)/(T+8t) < (3.898)/(T+4t) gives T < 7.1·t, i.e. one MTP draft decode costs more
than ~1/7 of a target block-verify despite the draft model being 461 MB vs the 27 GB
target. The doubling of sequential draft decodes (8 vs 4) outweighs the +1.4
extra accepted tokens. For this setup, γ=4 yields higher throughput than γ=8.

### Per-domain per-token accept rate (stable ordering across γ)
| source | γ=4 accept | γ=8 accept |
|---|---|---|
| aime (math)      | 81.7% | 65.9% |
| lcb (code)       | 75.6% | 58.5% |
| gpqa             | 68.6% | 48.4% |
| hle (MCQ)        | 61.2% | 41.0% |
Ordering aime > lcb > gpqa > hle holds at both γ. AIME math reasoning drafts most
predictably; HLE-MCQ least.

Instrumentation and aggregation: `initial_program.cpp` (DEPTH_*/ACCEPT_HIST lines,
parser-ignored), `analyze_depth.py`.
