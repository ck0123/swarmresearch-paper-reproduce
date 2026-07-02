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
