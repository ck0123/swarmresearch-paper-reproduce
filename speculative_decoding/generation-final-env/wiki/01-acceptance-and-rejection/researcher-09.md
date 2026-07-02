# Findings

## Phase-level speculative profile

Evaluation command: `./evaluator/task-eval --gpus 6,7`

The run completed on the 30-item benchmark with `mtp-spec` accuracy 93.3%, mean throughput 146.9 tok/s, speedup 1.46x over the fixed target reference, and aggregate accept rate 72.45%.

Instrumentation added to `initial_program.cpp` records per-prompt speculative decoding phase time, target verification rows, discarded target verification rows after first rejection, and per-draft-position confidence/outcome aggregates.

Across the 30 items, `mtp-spec` decoded 256,580 output tokens, drafted 263,320 tokens, and accepted 190,764 drafted tokens.

There were 65,830 speculative rounds. With `gamma=4`, target verification decoded 329,150 rows total. Of those rows, 72,556 were after the first rejection in their round and were later discarded, equal to 22.04% of target verification rows and 1.10 discarded target rows per speculative round.

Round outcomes were 38,964 all-accepted rounds and 26,866 rejected rounds. This is 59.19% all-accepted rounds and 40.81% rejected rounds.

Summed across prompts, measured prompt decode time was 2.006 s, serial MTP draft decode time was 59.983 s, and batched target verification decode time was 26.810 s. Within the measured draft+verify decode phases, MTP drafting accounted for 69.11% of time and target verification accounted for 30.89%.

## Draft position reach and confidence

Later draft positions were often generated but never reached by the acceptance loop because an earlier token in the same round rejected:

| draft position | drafted | reached | accepted | rejected | unreached | reached/drafted |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 65,830 | 65,830 | 57,316 | 8,514 | 0 | 100.00% |
| 1 | 65,830 | 57,316 | 50,234 | 7,082 | 8,514 | 87.07% |
| 2 | 65,830 | 50,234 | 44,250 | 5,984 | 15,596 | 76.31% |
| 3 | 65,830 | 44,250 | 38,964 | 5,286 | 21,580 | 67.22% |

Conditional acceptance given that a draft position was reached was similar across positions: 87.07%, 87.64%, 88.09%, and 88.05% for positions 0 through 3.

Mean draft probability assigned to the sampled token remained high even for rejected or unreached drafts:

| draft position | accepted mean q(sample) | rejected mean q(sample) | unreached mean q(sample) |
|---:|---:|---:|---:|
| 0 | 0.9654 | 0.8524 | - |
| 1 | 0.9696 | 0.8492 | 0.9072 |
| 2 | 0.9738 | 0.8577 | 0.9100 |
| 3 | 0.9759 | 0.8605 | 0.9132 |

Target/draft top-1 agreement separated accepted and rejected reached drafts more strongly than draft sampled-token probability. Among accepted reached drafts, target/draft top-1 agreement was 96.24%, 96.43%, 96.79%, and 96.97% for positions 0 through 3. Among rejected reached drafts, it was 50.29%, 43.93%, 40.41%, and 37.80%.

The sampled draft token's mean rank under the target distribution was near 1 for accepted reached drafts and around 2 for rejected reached drafts. Mean accepted ranks by position were 1.050, 1.044, 1.038, and 1.036. Mean rejected ranks by position were 2.046, 2.187, 2.228, and 2.312.

## Distributional-overlap decomposition of acceptance loss

Evaluation command: `./evaluator/task-eval --gpus 6,7`. Run summary: `mtp-spec` accuracy 93.3%, mean throughput 148.7 tok/s, speedup 1.48x, aggregate accept 72.4%.

Instrumentation added to `initial_program.cpp` computes, at each reached draft position, the distributional overlap `O = sum_x min(p(x), q(x)) = 1 - TV(p,q)` between the truncated-renormalized target distribution `p` and draft distribution `q` (the same top_k -> top_p -> temp distributions used by the rejection sampler). `O` is the theoretical single-token acceptance probability. The loss `1 - O = sum_x max(0, q(x) - p(x))` is split exactly into out-of-nucleus mass (draft mass on tokens with `p(x) = 0`, i.e. outside the target's truncated support) and overconfidence mass (`q(x) > p(x) > 0`, draft over-weighting tokens the target also keeps). Target entropy `H(p)` and draft entropy `H(q)` are recorded per reached position; rejections are split by whether the rejected draft token had `p > 0` (in-nucleus) or `p = 0` (out-of-nucleus).

Over 217,630 reached draft positions, empirical conditional acceptance was 87.66% and the mean theoretical overlap `O` was 87.56%. The mean loss `1 - O` of 12.44% decomposed into 2.32% out-of-nucleus mass and 10.12% overconfidence mass (overconfidence was 4.4x the out-of-nucleus mass). Mean `O` was 0.9404 on positions that empirically accepted and 0.4156 on positions that empirically rejected.

Per draft position 0-3:

| g | reached | emp accept | O (theory) | O accepted | O rejected | exc_oon | exc_overconf | oon share of loss |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 65,830 | 87.07% | 0.8696 | 0.9307 | 0.4577 | 0.0167 | 0.1137 | 12.8% |
| 1 | 57,316 | 87.64% | 0.8741 | 0.9385 | 0.4169 | 0.0220 | 0.1039 | 17.5% |
| 2 | 50,234 | 88.09% | 0.8808 | 0.9462 | 0.3977 | 0.0263 | 0.0929 | 22.0% |
| 3 | 44,250 | 88.05% | 0.8808 | 0.9506 | 0.3664 | 0.0307 | 0.0885 | 25.8% |

Out-of-nucleus mass rose with draft position (0.0167 -> 0.0307) while overconfidence mass fell (0.1137 -> 0.0885); the out-of-nucleus share of total loss rose from 12.8% at position 0 to 25.8% at position 3.

The draft distribution was sharper than the target at matched positions. Mean target entropy over reached positions was 0.2405; on accepted positions target entropy was 0.1494 and draft entropy 0.0454; on rejected positions target entropy was 0.8872 and draft entropy 0.2715.

Of 26,866 rejection events, 21,824 (81.2%) were in-nucleus (rejected draft token had `p > 0`) and 5,042 (18.8%) were out-of-nucleus (`p = 0`, structurally unacceptable).

Acceptance stratified by target entropy `H(p)` over all reached positions:

| H(p) bin | reached | accept | theoretical O | share of reached |
|---|---:|---:|---:|---:|
| [0, 0.25) | 158,247 | 97.88% | 97.87% | 72.71% |
| [0.25, 0.5) | 13,847 | 77.92% | 77.86% | 6.36% |
| [0.5, 1.0) | 26,749 | 62.20% | 61.82% | 12.29% |
| [1.0, 2.0) | 17,328 | 45.77% | 45.51% | 7.96% |
| [2.0, 3.0) | 1,452 | 34.99% | 32.95% | 0.67% |
| [3.0, inf) | 7 | 28.57% | 24.90% | 0.00% |

72.71% of reached draft positions had target entropy below 0.25 and accepted at 97.88%. Empirical acceptance matched the theoretical overlap `O` within each entropy bin and in aggregate.
