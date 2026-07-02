# Findings — content-resolved predictive-uncertainty structure of the target

Lens: the *target model's own* next-token uncertainty, measured target-only (no draft, no
acceptance), and how that uncertainty is organized by the **lexical/semantic content** of the
token being produced. Orthogonal to acceptance, timing, output length/location, hidden-state
geometry, prompt structure, and determinism.

## Method
- Instrumented `initial_program.cpp` (target mode only): at every generated position, computed from
  the raw target logits at the moment of sampling — full-vocab top-1 probability, Shannon entropy
  (nats, at decoding temp 1.0 over the full 262k vocab), and the rank the model assigned to the
  token it actually sampled (0 = argmax). Gated by env `TOK_DUMP`, one JSONL file per process.
- `task-eval` skips `target` (fixed reference), so target mode was driven directly via
  `run_target_dump.py` using the identical chat template / item rendering, GPUs 4,5, seed 0,
  temp 1.0 / top_p 0.95 / top_k 64. All 30 benchmark items, **261,481 generated tokens**.
- Per-token tokens classified by piece content (gemma uses leading-space word starts).

## Results

### 1. Predictive uncertainty is sharply bimodal and concentrated
- Entropy quantiles (nats): p50=0.002, p75=0.351, p90=1.017, p99=2.118, max=3.456.
- **56.6%** of all generated tokens are near-deterministic (H < 0.01 nats); 66.9% have H < 0.1;
  only 10.3% have H > 1.0.
- Uncertainty mass is concentrated: the top 5% most-uncertain tokens carry 33.8% of total entropy,
  top 10% carry 55.8%, top 20% carry 83.7%, top 30% carry 96.8%.
- Holds per item: near-deterministic fraction ranges 27%–72% across the 30 items, always substantial.

### 2. Uncertainty lives almost entirely in natural-language words
Total entropy mass = 71,420 nats. Share of tokens vs share of uncertainty mass:

| class | % of tokens | mean entropy | mean top-1 | % argmax | % of total ENT mass |
|---|---|---|---|---|---|
| word | 39.6% | 0.474 | 0.838 | 84.5% | **68.6%** |
| punct | 21.5% | 0.199 | 0.927 | 93.5% | 15.7% |
| math_markup | 11.6% | 0.208 | 0.923 | 92.8% | 8.8% |
| digit | 14.7% | 0.034 | 0.987 | 98.8% | **1.8%** |
| whitespace | 8.2% | 0.051 | 0.982 | 98.5% | 1.5% |
| other | 4.5% | 0.215 | 0.926 | 95.0% | 3.5% |

- Words are 39.6% of tokens but 68.6% of all predictive uncertainty.
- Digits are 14.7% of tokens but only 1.8% of uncertainty (98.8% argmax, H=0.034); whitespace,
  punctuation, and math/LaTeX markup are likewise near-deterministic.
- The word ≫ digit entropy gap (~10×) holds in every one of the 30 items individually.

### 3. Uncertainty is front-loaded within a word and within a number
- Word-initial pieces (leading-space word start): mean H = 0.558, 81.6% argmax.
- Word-continuation pieces (subword, no leading space): mean H = 0.329, 89.6% argmax.
- Leading digit of a number-run: mean H = 0.043; trailing digits: mean H = 0.014.
- The decision concentrates at the onset (which word, which number); the remainder of the
  word/number is comparatively forced.

### 4. Even under temp-1.0 sampling the chosen token is usually the mode
- Overall 8.9% of tokens were sampled off the argmax (rank > 0); rank p50=0, p90=0, p99=3, max=25.
- Off-mode rate by content: special 27.6%, word 15.5%, math_markup 7.2%, punct 6.5%, digit 1.2%,
  whitespace 1.5%.

## Artifacts
- `results/tokdump_gpu{4,5}.<pid>` and concatenated `results/tokdump_all.jsonl` (per-token JSONL).
- `results/raw_target_gpu{4,5}.txt` (per-item generations), `run_target_dump.py` (driver).
- Target accuracy/throughput unchanged from reference (instrumentation logs only).
