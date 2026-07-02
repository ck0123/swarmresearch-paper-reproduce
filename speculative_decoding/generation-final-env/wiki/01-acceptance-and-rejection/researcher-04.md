# Findings

- Added observational counters to `mtp-spec` for verification rounds, zero-accept rounds, full-accept rounds, resampling rounds, and accepted-token histograms. The counters are printed before `TEXT:` and do not alter sampling or scoring.
- Ran `./evaluator/task-eval --gpus 6,7` on the full 30-item benchmark after adding counters. The run reported 93.3% accuracy, 147.6 tok/s, 72.4% aggregate acceptance, and 1.47x speedup versus the fixed target reference.
- Across the 30 prompts, `mtp-spec` decoded 256580 tokens over 65830 verification rounds. It drafted 263320 tokens and accepted 190764 of them.
- With `gamma=4`, accepted-token histogram by verification round was: 0 accepted = 8514 rounds, 1 accepted = 7082 rounds, 2 accepted = 5984 rounds, 3 accepted = 5286 rounds, 4 accepted = 38964 rounds.
- The corresponding round shares were 12.93%, 10.76%, 9.09%, 8.03%, and 59.19%. Thus, most rounds accepted the full draft window, but 40.81% of rounds resampled before the bonus token.
- Mean accepted draft tokens per verification round was 2.898. Mean generated tokens per verification round was 3.898, including the sampled bonus token.
- Aggregate acceptance varied by source in this run: AIME 81.7%, LiveCodeBench 75.6%, GPQA 68.6%, and HLE 61.2%.
- The lowest per-item acceptance rates in this run were `hle-othe-03` at 50.8%, `hle-phys-00` at 52.1%, `gpqa-chem-01` at 54.7%, `hle-chem-04` at 57.6%, and `gpqa-biol-00` at 58.6%.
- The highest per-item acceptance rates in this run were `gpqa-phys-00` at 89.5%, `aime-6-10-01` at 85.8%, `aime-1-5-01` at 83.8%, `aime-11-15-01` at 81.3%, and `aime-11-15-00` at 80.9%.
