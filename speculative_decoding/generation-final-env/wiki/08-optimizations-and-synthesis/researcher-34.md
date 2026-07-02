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

## Round-level path acceptance calibration

Evaluation command: `./evaluator/task-eval --gpus 2,3`. Run summary: `mtp-spec` accuracy 93.3%, mean throughput 148.6 tok/s, speedup 1.48x, aggregate accept 72.4%.

Instrumentation added to `initial_program.cpp` computes, for each speculative round, the product of the actual accept probabilities `min(1, p(draft_g) / q(draft_g))` over all four drafted tokens in the verified path. This product is the conditional probability that the sampled draft path accepts fully. The instrumentation also records the product of per-position distributional overlaps `O`, the minimum token accept probability in the round, mean target entropy across the four verified draft positions, predicted-all-accept bins, and first-rejection position counts.

Across 65,830 rounds, 38,964 all-accepted and 26,866 rejected. The observed all-accept rate was 59.19%; the mean sampled-path all-accept probability was 58.99%. Mean product of theoretical overlaps was 58.80%, mean minimum token accept probability per round was 62.84%, and mean per-round target entropy was 0.2797.

The sampled-path all-accept probability separated successful and failed rounds. It averaged 86.04% on all-accepted rounds and 19.76% on rejected rounds.

Predicted all-accept probability bins:

| predicted path accept bin | rounds | share of rounds | observed all-accept | mean predicted all-accept | dead verify rows | share of dead rows | mean target entropy |
|---|---:|---:|---:|---:|---:|---:|---:|
| [0, 0.25) | 19,956 | 30.31% | 7.07% | 6.98% | 50,974 | 70.25% | 0.5796 |
| [0.25, 0.5) | 7,656 | 11.63% | 38.24% | 37.07% | 12,301 | 16.95% | 0.4528 |
| [0.5, 0.75) | 6,924 | 10.52% | 62.45% | 62.50% | 6,770 | 9.33% | 0.2938 |
| [0.75, 0.9) | 4,794 | 7.28% | 83.10% | 82.94% | 2,064 | 2.84% | 0.1729 |
| [0.9, 0.98) | 3,124 | 4.75% | 94.17% | 93.55% | 444 | 0.61% | 0.0953 |
| [0.98, 1] | 23,376 | 35.51% | 100.00% | 100.00% | 3 | 0.00% | 0.0095 |

The lowest predicted path-accept bin, `[0, 0.25)`, contained 30.31% of rounds and 70.25% of all dead verification rows. The highest bin, `[0.98, 1]`, contained 35.51% of rounds, all but one of those rounds fully accepted, and it accounted for 3 dead verification rows.

First rejection positions over the 26,866 rejected rounds were:

| first rejected draft position | rounds | share of rejected rounds |
|---:|---:|---:|
| 0 | 8,514 | 31.69% |
| 1 | 7,082 | 26.36% |
| 2 | 5,984 | 22.27% |
| 3 | 5,286 | 19.68% |

## Wall-clock cost decomposition and per-op decode costs

Evaluation command: `./evaluator/task-eval --gpus 2,3` (full 30-item run; `mtp-spec` accuracy 93.3%, mean throughput 148.5 tok/s, speedup 1.47x, aggregate accept 72.4%). Calibration runs at `--gamma 1 --limit 10 --n 1024` and `--gamma 2 --limit 10 --n 1024` on GPUs 2,3 provided verify-batch sizes of 2 and 3 rows alongside the full run's 5-row batches. Instrumentation added to `initial_program.cpp` emits per-prompt `spec_cost` aggregates (draft/verify decode microseconds and call/row counts).

The two timed phases (`llama_decode` for MTP drafting and target verification) wrapped by the existing instrumentation are not the wall-clock bottleneck. Over the full run, summed measured wall decode time was 1,719.4 s (256,580 tokens, 149.2 tok/s), of which timed GPU decode was 75.7 s = 4.4% of wall (draft 51.2 s, verify 22.6 s, prompt 1.9 s). The remaining 1,643.8 s (95.6% of wall) is untimed host-side work, dominated by the truncated-sampling routine `make_dist`, which does an O(vocab) `partial_sort` over the ~256k-token vocabulary once per draft pass and once per verify row (the calibration/overlap instrumentation builds a target `Dist` for every verified row). The previously recorded "drafting = 69% of decode time" describes the split *within* the 75.7 s timed-GPU slice (draft 51.2 s / 75.7 s = 67.6%), not wall-clock.

Per-op GPU decode costs, fit across the gamma=1/2/4 runs (verify call time vs rows-per-call, linear least squares):

| op | cost |
|---|---:|
| MTP draft pass (1 token, serial) | 194.4 us |
| target verify, per-call fixed | 161.1 us |
| target verify, per-row | 37.3 us |
| target verify, 5-row call (gamma=4) | 347.6 us (measured 343.1) |

A serial MTP draft pass (194 us) costs about 5.2x a verify row (37 us), and the verify call's fixed overhead (161 us) is 53.7% of a 5-row verify call. Draft-pass cost rose with context length (124 us at limit-10 gamma=1 vs 194 us on the full long-context run).

## Drafted-pass waste and draft-side adaptive-gating counterfactual

Reframing waste by the dominant GPU op: of 263,320 drafted tokens, 190,764 were accepted, so 72,556 drafted tokens (27.6%) were wasted serial MTP draft passes — the most expensive per-token GPU operation — versus the previously recorded 22.04% of (cheaper) verify rows being dead.

Instrumentation added to `initial_program.cpp` evaluates, in-process over every round, a draft-side-only stopping rule: truncate the round at G'(tau) = (first drafted position g whose draft entropy H(q_g) exceeds tau) + 1, else G. The gate depends only on the draft distribution, observed serially before any target verification, so it is realizable as an online policy; truncating speculation length does not change the per-token output distribution (the bonus token remains a target sample), so the policy is lossless. For each tau the instrumentation accumulates emitted tokens (min(accepted, G') + 1), draft passes (G'), verify rows (G'+1), rounds truncated, and "good" tokens lost to early stopping (max(0, accepted - G')). Per-round acceptance structure is held fixed at the realized values.

Amortized throughput is reconstructed two ways: a GPU-decode model (time = draft_passes x 194.4 us + rounds x 161.1 us + verify_rows x 37.3 us) and a sampling-bound model (cost proportional to make_dist calls ~ draft_passes + verify_rows). Both agree within ~1.5%.

| tau (max draft H) | emitted | Δemit | draft passes | Δdraft | rounds truncated | good tok lost | emit/pass | speedup (GPU) | speedup (sampling) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.02 | 239,863 | -6.52% | 215,389 | -18.20% | 33.56% | 16,731 | 1.1136 | 1.100x | 1.115x |
| 0.10 | 239,867 | -6.52% | 215,429 | -18.19% | 33.54% | 16,727 | 1.1134 | 1.100x | 1.115x |
| 0.20 | 240,437 | -6.30% | 217,288 | -17.48% | 32.42% | 16,157 | 1.1065 | 1.095x | 1.109x |
| 0.30 | 242,823 | -5.37% | 223,641 | -15.07% | 28.32% | 13,771 | 1.0858 | 1.080x | 1.093x |
| 0.50 | 247,018 | -3.73% | 234,469 | -10.96% | 21.00% | 9,576 | 1.0535 | 1.058x | 1.067x |
| 0.75 | 253,787 | -1.09% | 252,047 | -4.28% | 8.51% | 2,807 | 1.0069 | 1.025x | 1.028x |
| 1.00 | 255,403 | -0.46% | 257,689 | -2.14% | 4.29% | 1,191 | 0.9911 | 1.013x | 1.015x |
| inf (baseline) | 256,594 | 0% | 263,320 | 0% | 0% | 0 | 0.9745 | 1.000x | 1.000x |

Thresholds tau in [0.02, 0.10] reach the same operating point because few draft tokens have entropy between 0.02 and 0.10 (the draft is either near-deterministic or clearly uncertain). At that plateau the policy truncates 33.56% of rounds, removing 18.20% of serial draft passes while losing 6.52% of emitted tokens (16,731 tokens that would have been accepted), raising emitted-tokens-per-draft-pass from 0.9745 to 1.1136 (+14.3%). The amortized decode-throughput gain peaks at the most aggressive thresholds: 1.10x (GPU-decode model) to 1.115x (sampling-bound model). The 62.9% of rounds whose drafts stay below the gate keep their full gamma=4 speculation.

## Realizable round-level predictor from draft entropy

The same instrumentation bins each round by the maximum draft entropy across its drafted positions (the worst-case draft uncertainty, known before verification) and records the full-accept rate:

| max H(q) bin | rounds | share | all-accept | mean accepted tokens |
|---|---:|---:|---:|---:|
| [0, 0.25) | 41,421 | 62.92% | 75.94% | 3.385 |
| [0.25, 0.5) | 6,949 | 10.56% | 36.64% | 2.270 |
| [0.5, 1.0) | 13,677 | 20.78% | 31.75% | 2.107 |
| [1.0, 2.0) | 3,661 | 5.56% | 16.77% | 1.601 |
| [2.0, 3.0) | 117 | 0.18% | 4.27% | 0.957 |
| [3.0, inf) | 5 | 0.01% | 0.00% | 1.200 |

The draft model's own max entropy stratifies rounds, but more weakly than the previously recorded target-side path-accept predictor. The draft-side top bin (max H(q) < 0.25) holds 62.92% of rounds yet only 75.94% of them fully accept, whereas the target-side predictor's top bin [0.98, 1] held 35.51% of rounds at 100.00% all-accept. The draft-side signal is realizable but noisier: the draft is confident (low entropy) on a substantial fraction of rounds it does not carry, consistent with the previously recorded draft overconfidence.

## Direct online draft-entropy gate test

Evaluation command: `./evaluator/task-eval --modes mtp-spec,mtp-spec-gate --gpus 0,1`.

`initial_program.cpp` now includes an actual online `mtp-spec-gate` mode, distinct from the prior in-process counterfactual. In this mode the MTP draft loop stops after the first drafted token whose draft entropy exceeds `tau`; the triggering token is still included in the verification batch. The default `mtp-spec-gate` threshold is `tau=0.10`, matching the aggressive plateau from the counterfactual table.

Full 30-item evaluation:

| mode | accuracy | mean tok/s | speedup vs target ref | accept rate |
|---|---:|---:|---:|---:|
| target reference | 93.3% | 100.7 | 1.00x | - |
| mtp-spec | 93.3% | 148.5 | 1.47x | 72.4% |
| mtp-spec-gate tau=0.10 | 96.7% | 144.1 | 1.43x | 80.0% |

The direct online gate did not realize the counterfactual throughput gain in this instrumented full run. Mean throughput decreased from 148.5 tok/s to 144.1 tok/s (-3.0%) even though accepted/drafted rose from 72.45% to 79.96%.

Aggregate mechanism counters from the same run:

| metric | mtp-spec | mtp-spec-gate tau=0.10 | change |
|---|---:|---:|---:|
| decoded tokens | 256,580 | 271,722 | +5.90% |
| drafted tokens / draft calls | 263,320 | 245,503 | -6.77% |
| accepted draft tokens | 190,764 | 196,313 | +2.91% |
| speculative rounds / verify calls | 65,830 | 75,423 | +14.57% |
| verify rows | 329,150 | 320,926 | -2.50% |
| dead verify rows | 72,556 | 49,190 | -32.20% |
| all-accepted rounds | 38,964 | 49,219 | +26.32% |
| rejected rounds | 26,866 | 26,204 | -2.46% |
| emitted tokens per draft call | 0.9744 | 1.1068 | +13.59% |

The gate truncated 31,162 of 75,423 rounds (41.32%). Its actual average draft length was 3.255 tokens per round instead of 4.000, and its average verify batch was 4.255 rows per round instead of 5.000. Because the run emitted more tokens and used more rounds, total draft calls fell by 6.77%, much less than the 18.19% draft-pass reduction predicted by the prior fixed-trace counterfactual.

Measured timed decode phases differed from the counterfactual cost model. MTP draft decode cost stayed similar: 192.9 us/call for `mtp-spec` and 196.7 us/call for `mtp-spec-gate`. Target verification timing increased sharply under the gated policy in this run: 363.5 us/verify call and 72.7 us/verify row for `mtp-spec`, versus 3310.5 us/verify call and 778.0 us/verify row for `mtp-spec-gate`. The elevated gated verify timing was spread across multiple prompts rather than a single outlier; several gated prompts reported 10-16 s of target verify decode time.
