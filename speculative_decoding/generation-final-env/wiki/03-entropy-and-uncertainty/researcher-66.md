# Findings — researcher-66

## Lens: what the target's highest-uncertainty verification positions concretely ARE, beyond coarse content classes

Deeper extension of the prior content×verification-uncertainty neighborhood. Prior work showed
uncertainty mass lives in "words" and rejections cluster at high target entropy. "Word" is a
coarse bucket. This run cracks it open by logging, at every target verification distribution, the
**top competing candidate tokens** (piece + truncated prob) — i.e. WHAT the model is actually
torn between at an uncertain position — and classifying the top-2 competition into a concrete
typology.

### Method
- Added `append_cands()` to `initial_program.cpp` (mtp-spec): each per-round content record now
  carries the top-6 candidates of the already-computed target `Dist` for every draft token and for
  the corr/bonus token. Reuses the existing truncated distribution, so ~zero extra compute; the
  cand list is naturally short for confident positions (top_p truncation) and up to 6 at uncertain
  ones.
- `analyze_uncertainty_positions.py` classifies the top-1|top-2 competition into: `discourse_fork`
  (a reasoning-move connective competes: Wait/Let/So/If/Therefore...), `function_swap` (closed-class
  the/a/is/it...), `numeric`, `format` (punct/whitespace incl. `\n`↔`\n\n`), `morph_variant`
  (shared ≥4-char stem), `casing_space_var` (same token modulo leading space/case), `content_alt`
  (two different content words), `mixed_class` (top-2 differ in coarse class — structural fork),
  `single` (near-deterministic, one candidate). It also measures distribution geometry (top-2 gap,
  top-2 mass, effective #candidates), interchangeable-vs-pivotal entropy mass, a concrete token and
  fork inventory, and rejection rate per fork type.
- Full benchmark `./evaluator/task-eval --gpus 4,5`: accuracy **93.3%** (lossless, matches target
  reference), acceptance **72.4%**. Throughput **76.3 tok/s / 0.76x** is instrumentation cost
  (full-vocab entropy in the verify loop), not a decoding result. Universe = **217,630** on-path
  verified draft positions (accepted + first-rejected; discarded off-path drafts excluded),
  **63,861 nats** total entropy, 12.3% rejected.

### Core facts

**1. The hardest positions are pivotal decision points, not interchangeable paraphrase.**
At the highest-uncertainty positions (H ≥ 1.0: 24,007 positions carrying 36,158 nats = **57% of all
uncertainty**), the competition is **77.6% of positions / 79.2% of entropy mass "pivotal"**
(discourse/numeric/content/cross-class — the choice changes content or direction) vs only
**22.4% / 20.8% "interchangeable"** (format/function-swap/morph/casing). The breakdown by mass:
content_alt 37.2%, mixed_class 21.3%, discourse_fork 20.5%, function_swap 12.2%, format 7.5%.
Discourse forks carry the highest mean entropy (1.65 nats).

**2. High uncertainty is mostly a TIGHT BINARY choice, not a diffuse cloud.**
Of H ≥ 1.0 positions, **51.8% are tight binary forks** (top-2 mass ≥ 0.66 and 2nd candidate ≥ 0.20
— two genuine competing options); only **18.0% are diffuse many-way** (top1 < 0.45, top2 < 0.25).
Mean effective #candidates ≈ 3.2; top-1 prob p50 = 0.49. The model is usually deciding between
exactly two concrete continuations.

**3. Concrete token inventory of the hardest positions.** The top-1 (leading) tokens at H ≥ 1.0,
ranked by summed entropy (nats): ` the` (1651), `Wait` (1094), `The` (951), ` $` (947), ` is`
(887), `\n` (636), `$` (610), `.` (561), `Let` (555), `If` (517), ` let` (405), ` a` (402), `:`
(377). The recurring identities are: clause/sentence onsets (` the`/`The`/`This`), the
self-correction token `Wait`, math-mode entry/exit (` $`/`$`), and reasoning openers
(`Let`/`If`/`So`/`let`).

**4. Concrete fork inventory — what the model is specifically torn between.** Most frequent
top-1|top-2 competitions at H ≥ 1.0: `Let`|`Wait` (171) and `Wait`|`Let` (130) — **continue
solving vs backtrack/recheck**, the single dominant high-uncertainty fork; ` the`|` $` (143) and
` $`|` the` (120) — prose vs math-mode entry; ` let`|` I` (88), `If`|`Wait` (54), `The`|`Wait`
(49), `*`|`Wait` (49) — reasoning-control branching; ` the`|` a` (63), `\n\n`|`\n` (54), `.`|`,`
(52) — interchangeable function/format. The `Wait`-family (backtracking) forks are pervasive and
carry 45–65% rejection.

**5. The MTP draft fails most on the pivotal forks.** Rejection rate by fork type:
discourse_fork **41.5%** (mean H 1.28, the highest), content_alt **40.4%**, function_swap 38.6%,
mixed_class 35.6%, numeric 35.1%, format 30.7%, casing_space_var 33.3%; vs `single`
near-deterministic positions **1.6%**. Lost acceptance is concentrated at genuine reasoning-move
and content-word decision points — particularly the continue-vs-backtrack (`Wait`/`Let`) choice —
which the draft fundamentally cannot anticipate, not at recoverable formatting noise.

**6. Pivotal dominance holds across domains.** Pivotal share of H ≥ 0.5 entropy mass: aime 68.0%,
gpqa 75.7%, hle 75.9%, lcb 71.9%. Discourse-fork mass is highest in math (aime 19.8%) — math
reasoning carries the most backtracking decisions.

### Artifacts
- `initial_program.cpp` — adds top-candidate (`cand`) logging to the mtp-spec verification content
  log (reuses the existing truncated distribution).
- `analyze_uncertainty_positions.py` — fork-typology, geometry, token/fork inventory, rejection
  linkage.
- `results/content_4.jsonl`, `results/content_5.jsonl` — per-round logs with candidate lists.
- `results/uncertainty_positions_summary.txt`, `results/task_eval.json`.

---

# Findings — researcher-27

## Lens: linguistic/semantic content of drafted vs. corrected tokens

Orthogonal to acceptance rates, draft-depth/run-length, gamma sweeps, timing, output
length, latent geometry, input structure, and determinism. Question studied: *what kind of
tokens does the MTP draft get wrong, and what does the target substitute?* — the content
identity of draft vs. correction, not the rate at which it happens.

### Method
Instrumented `initial_program.cpp` (mtp-spec mode) to emit one JSON line per verification
round to a side-channel file (`results/content_<gpu>.jsonl`); stdout is unchanged so scoring
is unaffected (full run: accuracy 93.3%, identical to target reference; 1.48x speedup).
Each line records the detokenized piece string of every draft token with its accept flag,
plus the correction token (on a rejection) or bonus token (on full-accept). Post-analysis in
`analyze_content.py` classifies pieces into coarse linguistic categories
(word/punct/digit/space/newline/special) and computes substitution structure.
Full 30-item run on GPUs 6,7: **65,830 rounds — 26,866 corrections, 38,964 full-accept bonus**.

### Core facts

**1. The draft's failure mode is lexical, not structural (near-miss dominance).**
72.4% of corrections are *same-category*: the draft picked the right token KIND but the wrong
specific token (word→word 53%, punct→punct 14%, digit→digit 3%). Only 27.6% are category
errors. The draft reliably knows the local linguistic shape of what comes next; it fails on
the lexical choice within that shape.

**2. Draft is right on scaffolding, wrong on content.**
Accepted draft tokens: 38.8% word, 33.3% punct, 15.0% digit, 8.4% space, 4.5% newline.
Rejected/correction tokens: ~64–67% word, ~25% punct, ~4% digit. Punctuation, whitespace, and
newlines are accepted far above their share of corrections; content words are the dominant
correction category. Digits are accepted well relative to frequency (15% of accepts, 4% of
corrections).

**3. "Same category" does NOT mean "similar spelling."**
81.9% of corrections share ZERO character prefix with the rejected draft; only ~4% share ≥2
chars. The draft proposes a *different word of the same type*, not a typo/morphological
variant. Errors are semantic substitutions, not spelling slips.

**4. Divergences are mostly at word/line boundaries (which-word branch points).**
60.0% of corrections: both rejected-draft and correction begin a new word/line (boundary).
28.0% are both word-internal (mid-word divergence); of those only 17.7% share a stripped
prefix. Branch points are about *which next word/symbol*, not intra-word spelling.

**5. Concrete recurring substitutions** (rejected → correction):
- Formatting ambiguity: `\n`↔`\n\n` (paragraph-break length), `.`↔`,`, `$.`↔`$`/`$,`.
- Function-word / determiner choice: ` the`↔` a`↔` it`↔` this`; `the`→`a` is the single most
  common word substitution (89×).
- Reasoning-step direction: `Wait`↔`Let`, `let`→`I`, `is`→`can`/`means`/`was` — disagreement
  on how to open the next clause/reasoning move.
- Math-mode transition: ` $`↔` the`, ` $`↔` `` ` `` — the draft mis-predicts entry/exit of
  LaTeX math. `$`-bearing tokens: 11,195 accepted vs 2,052 rejected-draft, 1,868 in corrections.
- Digit value guesses: `1`↔`2`, `2`↔`1`, `1`↔`0`/`3` — adjacent-value misses where the draft
  cannot know the computed number.

**6. Domain content profile (correction category mix).**
Math (aime) and code (lcb) corrections are punctuation/digit-heavy (aime 31.2% punct, 7.4%
digit; lcb 28.1% punct, 6.0% digit). Prose multiple-choice (hle 75.9% word, gpqa 72.0% word)
corrections are overwhelmingly word choice. Correction content tracks the symbolic density of
the domain.

**7. Post-divergence drafts stay content-heavy.**
Tokens drafted after the first rejection (discarded, conditioned on a wrong prefix) are 57.5%
word / 26.6% punct / 5.8% digit — the draft keeps proposing content words rather than
collapsing into filler after it goes off-path.

### Artifacts
- `initial_program.cpp` — added side-channel per-round content logging (mtp-spec only).
- `analyze_content.py` — content categorization + substitution-structure analysis.
- `results/content_6.jsonl`, `results/content_7.jsonl` — 65,830 logged rounds.

---
# (merged reference) researcher-25 findings:

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

---

# Findings — researcher-32

## Lens: target uncertainty at the exact MTP verification point, resolved by token content

Integrated angle combining the prior content-error analysis with the prior target-uncertainty
analysis. Instead of measuring target uncertainty in target-only decoding, this run measured
full-vocab target entropy/top-1/rank at each MTP speculative verification distribution, while
retaining the content identity of accepted drafts, first rejected drafts, corrections, all-accept
bonus tokens, and discarded post-rejection drafts.

### Method
- Instrumented `initial_program.cpp` (mtp-spec mode) to add verifier uncertainty fields to the
  existing side-channel `results/content_<gpu>.jsonl` records:
  - for each drafted token: full-vocab target entropy, top-1 probability, rank of the draft under
    target logits, truncated target probability `p`, truncated draft probability `q`, and
    rejection-sampling acceptance probability `ar`;
  - for each correction/bonus token: full-vocab target entropy, top-1 probability, target rank,
    and truncated target probability.
- Added `analyze_spec_uncertainty.py` for the integrated summary. Updated `analyze_content.py` to
  infer source labels from the current GPU-pair result files.
- Full benchmark run: `./evaluator/task-eval --gpus 4,5`. Accuracy remained **93.3%** with
  **72.4%** acceptance. Throughput was **72.6 tok/s / 0.72x** because this run computes
  full-vocab entropy inside the verification loop; the throughput is instrumentation cost, not a
  decoding-algorithm result.
- Dataset: **65,830 rounds**, **217,630 actually verified draft tokens**, **190,764 accepted**,
  **26,866 first rejected**, **38,964 full-accept bonus tokens**, **45,690 discarded post-rejection
  draft tokens**.

### Core facts

**1. Rejections occur at much higher target uncertainty than accepted drafts.**
Accepted draft tokens: mean entropy **0.190**, p50 **0.001**, p90 **0.717**, rank-0 under the target
**96.7%**, mean target probability **0.932**, mean accept probability **0.950**. First rejected
draft tokens: mean entropy **1.026**, p50 **0.945**, p90 **1.874**, rank-0 **38.9%**, mean target
probability **0.317**, mean accept probability **0.348**. Correction tokens share the same verifier
entropy distribution as the rejected position, with target rank-0 **45.9%** and mean target
probability **0.440**.

**2. Target entropy strongly stratifies empirical rejection.**
Among actually verified draft tokens:
- H < 0.01: **117,372 tokens**, **0.7%** rejected, mean accept probability **0.993**.
- 0.01 <= H < 0.1: **22,999 tokens**, **3.4%** rejected, mean accept probability **0.966**.
- 0.1 <= H < 0.5: **26,604 tokens**, **13.3%** rejected, mean accept probability **0.867**.
- 0.5 <= H < 1.0: **26,648 tokens**, **34.3%** rejected, mean accept probability **0.654**.
- H >= 1.0: **24,007 tokens**, **52.4%** rejected, mean accept probability **0.472**.

**3. The high-uncertainty rejection region is mostly word content.**
The word share rises with uncertainty: **30.0%** of verified draft tokens at H < 0.01, **42.5%** at
0.01 <= H < 0.1, **48.8%** at 0.1 <= H < 0.5, **56.2%** at 0.5 <= H < 1.0, and **76.2%** at H >=
1.0. This directly links the prior target-only fact that uncertainty mass lives in words with the
prior MTP-content fact that first rejections are lexical/content-heavy.

**4. Content class explains much of the verifier risk gradient.**
Among actually verified draft tokens, rejection rates by content class were:
- word: **91,228 tokens**, **19.0%** rejected, mean entropy **0.474**, mean accept probability
  **0.809**.
- punct: **70,498 tokens**, **10.0%** rejected, mean entropy **0.224**, mean accept probability
  **0.899**.
- digit: **29,766 tokens**, **3.6%** rejected, mean entropy **0.039**, mean accept probability
  **0.964**.
- space: **16,521 tokens**, **2.9%** rejected, mean entropy **0.060**, mean accept probability
  **0.971**.
- newline: **9,528 tokens**, **10.0%** rejected, mean entropy **0.275**, mean accept probability
  **0.900**.

**5. Same-category lexical near-misses persist across uncertainty buckets.**
Corrections remain mostly same-category substitutions even at high uncertainty:
H < 0.01: **79.1%** same-category; 0.01 <= H < 0.1: **77.7%**; 0.1 <= H < 0.5: **73.1%**;
0.5 <= H < 1.0: **70.5%**; H >= 1.0: **72.9%**. The earlier "right kind, wrong token" result is
not confined to low-uncertainty or formatting cases.

**6. Post-rejection discarded drafts are less uncertain than first rejected drafts but still
content-heavy.**
Discarded drafts after the first rejection: mean entropy **0.568**, p50 **0.345**, p90 **1.521**,
rank-0 **76.2%**, mean target probability **0.701**, mean accept probability **0.735**. Their
content mix was **57.5% word**, **26.6% punct**, **5.8% digit**, **5.6% space**, **4.5% newline**.

**7. Domain-level rejection follows word/uncertainty density.**
Verified-token rejection and mean verifier entropy by source:
- aime: **46,290 tokens**, **8.1%** rejected, mean H **0.184**, **32.0%** word.
- gpqa: **50,676 tokens**, **14.4%** rejected, mean H **0.340**, **50.8%** word.
- hle: **36,718 tokens**, **18.6%** rejected, mean H **0.425**, **58.1%** word.
- lcb: **83,946 tokens**, **10.7%** rejected, mean H **0.268**, **34.9%** word.

### Artifacts
- `initial_program.cpp` — adds target verifier uncertainty fields to the existing mtp-spec
  side-channel content log.
- `analyze_spec_uncertainty.py` — summarizes target uncertainty by speculative role and content.
- `analyze_content.py` — source mapping generalized to the current GPU-pair result files.
- `results/content_4.jsonl`, `results/content_5.jsonl` — integrated per-round logs.
- `results/spec_uncertainty_summary.txt`, `results/content_summary.txt`,
  `results/task_eval.json` — analysis and evaluation summaries.
