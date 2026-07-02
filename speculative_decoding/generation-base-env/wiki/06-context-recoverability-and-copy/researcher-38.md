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
  `drafted=0` = the round-boundary token, now split by round reconstruction into either
  full-accept bonus or true rejection/resample. Source mapping reconstructed from the interleaved
  `items[i::2]` chunking. Analysis: `analyze_copy.py`.

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

### 3. Copy structure and MTP accepted drafts overlap but do not coincide
Cross-tabulating longest-match copyability vs MTP capture (`drafted=1`):
- copyable & MTP-got (redundant): **40.0%**
- copyable & non-drafted boundary: **9.5%**
- non-copyable & MTP-got (MTP-unique): **34.4%**
- non-copyable & non-drafted boundary: **16.1%**
- Marginals: copy-coverage 49.5%, MTP-coverage 74.3%, **union 83.9%**.
- Among the 25.7% of non-drafted boundary tokens, **37.2% are copyable**.
- The 9.5% copyable-but-non-drafted tokens are high-confidence for the target: mean mass on the
  correct copy candidate **0.919** (84.5% have mass >0.9). However, the round-level split below
  shows this bucket is dominated by full-accept bonus tokens rather than rejection misses.

### 4. Round reconstruction: most copyable non-drafted tokens are full-accept bonuses
Because each speculative round logs accepted draft tokens followed by one non-drafted boundary
token, the accepted-prefix length reconstructs boundary type (`gamma=4`):
- accepted draft tokens: **190,764 (74.3%)**; copyable **53.8%**; high-confidence copy **49.6%**.
- full-accept bonus tokens: **38,964 (15.2%)**; copyable **55.4%**; high-confidence copy **51.4%**.
- true rejection/resample tokens: **26,866 (10.5%)**; copyable **10.6%**; high-confidence copy
  **2.3%**.
- The non-drafted boundary bucket is **59.2% bonus / 40.8% true rejection**.
- The copyable non-drafted bucket (24,461 tokens, 9.5% of all tokens) splits into
  **21,603 full-accept bonus tokens (8.4% of all)** and only **2,858 true rejection tokens
  (1.1% of all)**.
- Target mass on copyable true-rejection tokens is much lower than on copyable bonus tokens:
  true-rejection copy mean mass **0.540** with **22.0% >0.9**; bonus copy mean mass **0.969**
  with **92.7% >0.9**.
- True rejection boundaries become slightly more copyable as more drafts were accepted before the
  rejection: accepted=0/1/2/3 copyable rates are **9.2% / 10.1% / 11.4% / 12.8%**. The
  full-accept bonus boundary (accepted=4) is **55.4%** copyable.

### 5. Copyability is run-structured, but high-confidence runs rarely start at true rejections
- Copyable tokens form **48,151 runs** covering 127,063 tokens (49.5% of all); mean run length
  **2.64**. Of copy-run tokens, **83.8%** are in runs of length ≥2 and **52.5%** in runs of
  length ≥4.
- High-confidence copy tokens form **47,242 runs** covering 115,362 tokens (45.0% of all); mean
  run length **2.44**. Of high-confidence-copy tokens, **80.7%** are in runs of length ≥2 and
  **48.2%** in runs of length ≥4.
- Only **629** high-confidence copy runs start on a true rejection boundary (**2.3%** of rejection
  boundaries). Their following high-confidence-copy run length averages **2.39** tokens; **47.7%**
  last at least two tokens and **19.4%** last at least four.

## Summary
On this benchmark ~half of the target's generated tokens are reproducible by a ≤3-gram
suffix match against their own running context, the redundancy is highest for math and lowest
for open-ended prose, and context-recoverability coincides almost perfectly (99.8%) with the
realized token whenever the candidate carries high target mass. Copyability is strongly
run-structured, but the most actionable-looking prior bucket changes under round reconstruction:
copyable non-drafted tokens are mostly full-accept bonus tokens, while true rejection/resample
tokens are only 10.6% copyable and only 2.3% high-confidence copyable.
