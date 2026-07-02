# Findings — context-recoverability ("copyability") of the target's generation

## Lens
A genuinely orthogonal angle to the saturated lenses (acceptance, per-depth/run-length,
gamma, timing, output length, hidden-state geometry, prompt structure, determinism,
draft-error taxonomies, single-next-token uncertainty): the **self-referential / copy
structure of the target's output stream**. For each confirmed token I ask whether it is
recoverable by a longest-suffix n-gram lookup ("induction" / prompt-lookup) against the
*running context* (prompt + generation so far), and how much of the target's own truncated
predictive mass lands on that context-recoverable candidate. This characterizes the
information-redundancy of the generation itself, independent of the speculative machinery.

## Method
- Instrumented `initial_program.cpp` with **read-only** logging in `mtp-spec` mode
  (`CopyIndex`: incremental 1/2/3-gram suffix→follower maps over prompt+generation; logs one
  JSON record per confirmed token). Decoding path is byte-for-byte unchanged.
- Verified non-perturbation: instrumented full 30-item run reproduces the reference exactly —
  **accuracy 93.3% (== target), accept 72.4%, speedup 1.47×** (ref 1.51×, within seed noise).
- 256,594 confirmed-token records (GPUs 6,7). `drafted=1` = token was an accepted MTP draft;
  `drafted=0` = resampled/bonus token (an MTP "miss"). Source mapping reconstructed from the
  interleaved `items[i::2]` chunking. Analysis: `/tmp/analyze_copy.py`.

## Results

### 1. Half the generation is context-recoverable by a trivial suffix lookup
Longest-match (≤3-gram) copy predictor over the running context:
- **Copy-coverage = 49.5%** of all confirmed tokens equal the looked-up candidate.
- A candidate exists for 94.4% of tokens; precision rises with suffix length:
  1-gram-only 27.7% hit, 2-gram-only 44.7%, 3-gram available 60.8% (hit|seen).
- Per source (copy-coverage): **aime 54.4% > lcb 50.4% > gpqa 48.4% > hle 42.5%**.
  Math/code (repeated identifiers, equations, digit runs) are most self-referential; the
  open-ended HLE prose is least.

### 2. The copy structure aligns with the target's own beliefs (not just surface repetition)
- Mean target truncated mass on the longest copy candidate = **0.524**; the candidate sits
  inside the target's top-k/top-p support **59.8%** of the time.
- Confidence-gating on the candidate's target mass is near-deterministic:
  | gate (target mass on candidate) | fires on | precision (hit) |
  |---|---|---|
  | ≥0.0 | 94.4% | 52.4% |
  | ≥0.5 | 49.4% | 97.4% |
  | ≥0.9 | 45.1% | 99.8% |
  | ≥0.99 | 43.4% | 100.0% |
  → Whenever the context-recoverable candidate is one the target is confident in, it is almost
  always the realized token. ~45% of all tokens fall in this "confident-and-copyable" regime.

### 3. Copy structure and MTP drafting overlap but do not coincide
Cross-tabulating longest-match copyability vs MTP capture (`drafted`):
- copyable & MTP-got (redundant): **40.0%**
- copyable & MTP-missed: **9.5%**
- non-copyable & MTP-got (MTP-unique): **34.4%**
- non-copyable & MTP-missed (hard): **16.1%**
- Marginals: copy-coverage 49.5%, MTP-coverage 74.3%, **union 83.9%**.
- Among the 25.7% of tokens MTP misses (resample/bonus), **37.2% are copyable**.
- The 9.5% copyable-but-MTP-missed tokens are high-confidence for the target: mean mass on the
  correct copy candidate **0.919** (84.5% have mass >0.9). I.e. MTP's misses include a
  near-certain, context-recoverable slice that a 3-gram lookup reproduces; conversely 34.4% of
  tokens are captured by MTP yet are *not* recoverable by suffix lookup (MTP's unique reach).

## Summary
On this benchmark ~half of the target's generated tokens are reproducible by a ≤3-gram
suffix match against their own running context, the redundancy is highest for math and lowest
for open-ended prose, and context-recoverability coincides almost perfectly (99.8%) with the
realized token whenever the candidate carries high target mass. MTP drafting and copy structure
are partially complementary: their captured-token sets overlap on 40% of tokens, each reaches a
distinct ~10/34% the other does not, and together they cover 83.9% of confirmed tokens.
