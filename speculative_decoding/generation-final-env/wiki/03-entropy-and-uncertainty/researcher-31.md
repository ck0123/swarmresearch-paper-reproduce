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

# Findings — researcher-31

## Lens: confidence/entropy structure of divergence points (deeper extension of the content lens)

The content lens established draft errors are semantic "right-kind/wrong-token" substitutions.
This lens goes a level beneath: when the draft LOSES a token, is the target genuinely uncertain
there (an ambiguous branch point the draft lost by chance) or is the target confident (the draft
is simply wrong about a determined token)? And does the draft head's own probability KNOW it is
about to be rejected (calibration)?

### Method
Extended the same side-channel (mtp-spec only; stdout unchanged, so scoring identical: accuracy
93.3% == target reference, 72.4% accept, 1.46x speedup). For each verified draft position the log
now records: `q` (draft-head prob of its own proposal), `px` (target prob of that token),
`acc`=min(1,px/q), target entropy `Ht` & top-prob `pt` over the top-k support, draft entropy `Hd`
& top-prob `qt`, target argmax id, accept flag; plus on rejection the target prob of the
correction (`corr_px`) and whether the correction is the target argmax. All quantities are
computed in the existing accept/reject loop. Post-analysis in `analyze_confidence.py`.
Same full run on GPUs 6,7: 65,830 rounds, 190,764 accepted positions, 26,866 first-reject positions.

### Core facts

**1. The draft head is anti-calibrated in its uncertain regime.**
67.7% of all drafted positions are deterministic agreement (q=px=1, accept 92.3%). In the
remaining uncertain regime (q<0.999), acceptance is NON-MONOTONIC and inverted in draft
confidence: q∈[0,0.3) accepts 64.2%, q∈[0.5,0.7) 54.2%, q∈[0.7,0.85) 52.8%, q∈[0.95,0.999)
46.9%. The MORE confident the draft is (short of total certainty), the LESS likely it is
accepted. AUC of draft prob `q` predicting acceptance is only 0.685 (top-prob 0.686, entropy
0.687 — all near-useless). Mean q at rejection is 0.854: the draft is highly confident even when
wrong.

**2. The TARGET's uncertainty, not the draft's, predicts rejection (AUC 0.882).**
Target entropy: accept 0.149 nats vs reject 0.887 nats. Target top-prob: accept 0.941 vs reject
0.643. Accepted positions are uniformly determined (target entropy <0.26 across every content
category); rejections concentrate where the target distribution itself is flat. The binding
signal lives in the target's verification logits, not the draft's confidence.

**3. Most rejections are NOT the draft being wrong about a settled token.**
By target top-prob at the rejected slot: only 16.0% are "determined" (pt≥0.9), 23.3% strong
(0.7–0.9), 46.3% mid (0.4–0.7), 14.3% ambiguous (pt<0.4). 57.0% of rejected draft tokens were
"plausible" to the target (px≥0.2); only 19.6% were ruled out (px<0.02). The majority of
rejections are coin-flip losses at genuine branch points, not determined-token mistakes.

**4. 38.9% of rejections, the draft had picked the target's own argmax.**
The draft token equals the target argmax in 10,459 of 26,866 rejections — greedy (temp=0)
decoding would accept all of these. They are rejected only because the stochastic lossless
accept test fires when q>px (the draft over-assigned probability to the right token). Over a
third of "draft errors" are artifacts of lossless sampling at temp=1, not the draft being wrong.
The correction equals the target argmax in 45.9% of rejections; mean target prob of the
correction is only 0.440.

**5. Acceptance-loss decomposition at rejection (mean realized px/q = 0.348).**
31.0% target rules the draft out (px<0.10); 37.3% draft overconfidence (px≥0.10 but q≥2·px);
31.7% close ties (px≥0.10, q<2·px, lost the coin flip). The draft systematically
over-concentrates relative to the target at branch points (~3x on average).

**6. Confidence structure tracks content category.**
Rejected words sit at high target entropy (0.985) — semantic free choice. Rejected digits sit
at LOW target entropy (0.345, target top-prob 0.843): when a drafted digit is wrong the target
is fairly sure of the right value, so digit errors are genuine value misses on near-determined
slots (only 22.3% were the target argmax). Newline rejections are 54.5% target-argmax — mostly
stochastic losses on the target's top choice (paragraph-break-length ambiguity, matching the
content lens's \n↔\n\n).

**7. The draft does not degrade with draft depth.**
Across depths g=0..3, mean q rises slightly (0.951→0.962), target px rises (0.846→0.866), and
acceptance rises (87.1%→88.1%). Conditioning the MTP head on its own prior drafts does not erode
calibration within a round; deeper positions are if anything marginally more accept-prone
(consistent with survivorship — rounds reaching deeper g have already cleared the hard slots).

**8. A draft-confidence gate is unfavorable at every threshold.**
Gating first-position drafts on q≥thr drops more good (would-accept) tokens than doomed ones at
all thresholds tested (q≥0.7: drop 1,630 doomed vs 2,600 good; q≥0.9: 3,020 doomed vs 4,287
good). Draft confidence is not a usable adaptive-drafting signal — a direct consequence of the
anti-calibration in fact 1.

### Artifacts
- `initial_program.cpp` — extended the mtp-spec side-channel with per-position confidence stats
  (`ev[]`: q, px, acc, Ht, pt, Hd, qt, target-argmax, accept flag; plus corr_px/corr_argmax).
- `analyze_confidence.py` — calibration, target-uncertainty, rejection-typology, loss
  decomposition, category cross, gating, and depth analyses.
- `results/content_6.jsonl`, `results/content_7.jsonl` — 65,830 rounds with confidence telemetry.
