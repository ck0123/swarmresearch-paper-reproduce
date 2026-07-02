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

---

# Findings — researcher-41

## Lens: the "stochasticity tax" — measuring the consequence of recovering the artifact rejections

The confidence lens recorded that 38.9% of rejections are "artifacts" of temp=1 sampling — the
draft proposed the target's own argmax and was rejected only because the lossless test fires when
q>px. This lens converts that descriptive fact into measured throughput/accuracy consequences: it
implements two verification rules that *recover* the artifact rejections and runs them end-to-end
on the full 30-item benchmark, then traces the mechanism behind the result.

### Method
Added an `SPEC_VERIFY` env switch to the mtp-spec accept loop in `initial_program.cpp` (stdout/
scoring path unchanged; the side-channel log records the realized accept decisions):
- `lossless` (default): the existing rule, `take = uni < min(1,px/q)`; residual correction `p−q`.
- `greedy`: temp→0 everywhere; `take = (draft==target argmax)`; correction/bonus = target argmax
  (canonical greedy speculative decoding; exactly lossless to greedy target decoding).
- `argmax_sticky`: temp=1 throughout, but `take = (draft==argmax) OR (uni<acc)` — force-accept the
  artifact class, otherwise the unchanged stochastic test + residual correction. It differs from
  `lossless` ONLY by accepting the 38.9% artifact rejections, so it is a controlled isolation of
  that one class. Three full evals on GPUs 2,3 plus offline analysis (`analyze_tax.py`) of the
  fresh lossless logs (65,830 rounds, identical count to the parent run).

### Core facts

**1. Measured accuracy/throughput frontier (full 30-item benchmark, GPUs 2,3).**
| variant        | accept | tok/s | speedup | accuracy | mean decoded | runaway (≥16k)/30 |
|----------------|--------|-------|---------|----------|--------------|-------------------|
| lossless temp=1| 72.4%  | 150.0 | 1.49×   | 93.3%    | 8,553        | 1                 |
| argmax_sticky  | 85.0%  | 166.5 | 1.65×   | 83.3%    | 10,917       | 4                 |
| greedy         | 85.7%  | 167.4 | 1.66×   | 53.3%    | 13,580       | 18                |
The artifact rejections ARE recoverable for throughput: both recovery rules capture nearly the same
gain (accept 72.4→85%, speedup 1.49→1.65×). Recovery is NOT free — accuracy falls monotonically with
how hard the rule biases toward the mode (93.3 → 83.3 → 53.3%).

**2. The drop is degeneration, not wrong answers — output length balloons.**
Mean decoded tokens 8,553 → 10,917 → 13,580; items hitting the 16,384 cap 1 → 4 → 18; greedy median
output IS the cap (16,384). Damage concentrates on the longest-generation domain: code (lcb) 10/10 →
7/10 → 3/10, math (aime) 4/5 → 4/5 → 2/5, while short MCQ (gpqa) holds 10/10 → 10/10 → 8/10. Mode-
seeking makes this reasoning model loop and never emit its boxed answer; truncation at the cap scores
it wrong.

**3. The artifact rejections are mechanically forced OFF-mode steps, not waste.**
On the fresh lossless run: 217,630 evaluated positions, 190,764 accepted (87.7%), 26,866 first-
rejects. Of rejections, 38.9% are artifacts (draft==argmax, reconfirming the parent exactly) and
61.1% genuine misses. When the lossless rule rejects an artifact, the correction it emits is the
argmax in **0.0%** of cases — it ALWAYS steps off the mode. This is mechanically forced: an artifact
reject means px<qx on the argmax token (verified 100% of the 10,459 cases), so the residual `p−q`
clamps that token to zero mass and it can never be resampled. By contrast, genuine misses correct
TOWARD the argmax 75.1% of the time (mean corr_px 0.591). The two rejection classes have opposite
roles: genuine = error correction (toward the mode), artifact = diversity injection (away from it).

**4. Clean causal attribution via the isolated intervention.**
`argmax_sticky` changes lossless by exactly one thing — suppressing the 38.9% diversity-injection
rejections — and that alone costs 10 accuracy points (93.3→83.3%) while delivering the full 1.65×
throughput. The further collapse to 53.3% under greedy is the additional cost of forcing the genuine-
miss corrections and the bonus tokens onto the mode as well. The "wasted" artifact rejections are the
load-bearing entropy that keeps the reasoning trajectory from degenerating.

**5. Recoverable-throughput accounting.**
Of all 195,015 draft tokens that were the target argmax, only 5.4% (10,459) are rejected, yet they
constitute 38.9% of all rejections; accepting them in place lifts the position accept-rate 87.7→92.5%.
14.9% of rounds have an artifact as their first reject (the point past which greedy would extend the
run). This is the size of the throughput pool that mode-recovery taps — and that the accuracy results
show cannot be drawn down without degrading this benchmark.

### Artifacts
- `initial_program.cpp` — `SPEC_VERIFY` env switch (lossless / greedy / argmax_sticky) in the accept loop.
- `analyze_tax.py` — artifact/genuine rejection split, off-mode-correction mechanism, run-length bound.
- `results/mode_lossless.json`, `results/mode_greedy.json`, `results/mode_sticky.json` — per-item evals.
- `results/content_lossless_*.jsonl` (65,830 rounds), `content_greedy_*`, `content_sticky_*` — telemetry.

---

# Findings — researcher-47

## Lens: the interior of the stochasticity-tax frontier

The stochasticity-tax lens mapped the endpoints: lossless temp=1 preserves accuracy at lower
throughput, while fully recovering artifact rejections (`argmax_sticky`) gains throughput but loses
accuracy. This lens measures the space between those endpoints by recovering only a controlled
fraction of artifact rejections.

### Method
Added `SPEC_VERIFY=artifact_p` to `initial_program.cpp`. The rule first performs the unchanged
lossless accept/reject draw. If that draw rejects a draft token that equals the target argmax, it
force-accepts that token with probability `SPEC_ARTIFACT_P=lambda`; otherwise it emits the unchanged
lossless residual correction. Thus `lambda=0` is lossless and `lambda=1` has the same accept decisions
as `argmax_sticky` for artifact rejections. Full 30-item evals used GPUs 2,3 with
`./evaluator/task-eval --gpus 2,3`; outputs were saved under `results/mode_artifact_p*.json` and
`results/content_artifact_p*_*.jsonl`. Post-analysis in `analyze_artifact_sweep.py`.

### Core facts

**1. The interior frontier is not linear; most of the endpoint damage appears near very high artifact
recovery.**
| artifact recovery lambda | accept | tok/s | speedup | accuracy | mean decoded | cap (>=16k)/30 |
|--------------------------|--------|-------|---------|----------|--------------|----------------|
| 0.25 | 75.0% | 150.1 | 1.49x | 100.0% | 8,876 | 1 |
| 0.50 | 78.1% | 153.9 | 1.53x | 93.3% | 9,107 | 2 |
| 0.75 | 81.2% | 158.5 | 1.57x | 96.7% | 9,258 | 1 |
| 0.90 | 84.0% | 161.5 | 1.60x | 86.7% | 10,209 | 5 |
The prior endpoint `argmax_sticky` was 85.0% accept, 166.5 tok/s, 1.65x, 83.3% accuracy, mean
decoded 10,917, and 4 capped items. The sweep reaches most of the accept-rate increase by
`lambda=0.75` without reproducing the endpoint accuracy drop; the sharp degradation appears between
`lambda=0.75` and `lambda=0.90`.

**2. Residual rejections show the intervention drained the intended class.**
Residual first-reject artifacts fell monotonically as lambda increased: 8,150 (lambda=0.25), 5,482
(0.50), 2,626 (0.75), 1,198 (0.90). Residual genuine first-rejections stayed essentially constant:
16,805, 16,589, 16,670, 16,717. The rule is selectively removing artifact/diversity-injection
rejections while leaving real draft misses in place.

**3. The average accepted run length rises smoothly even though quality breaks late.**
Mean accepted draft tokens per round rose 2.999 -> 3.126 -> 3.247 -> 3.360 as lambda increased from
0.25 to 0.90. Full-accept rounds rose 41,638 -> 44,153 -> 46,109 -> 52,332. Throughput and accept-rate
therefore move smoothly, while downstream correctness and capped-output behavior change abruptly only
near high artifact recovery.

**4. The failure pattern at high lambda is length/cap dominated and domain-skewed.**
At lambda=0.90, capped items increased to 5/30 and wrong items were `hle-othe-03`, `lcb-hard-01`,
`aime-11-15-01`, and `lcb-medium-08`. Source-level scores at lambda=0.90 were AIME 4/5, GPQA 10/10,
HLE 4/5, LCB 8/10; LCB had 3 capped items and mean decoded length 12,859. This matches the endpoint
finding that mode-recovery damage concentrates in long-generation tasks rather than uniformly across
short MCQ tasks.

**5. Low-to-mid artifact recovery can improve measured throughput without lowering benchmark
accuracy in this run.**
Compared with the prior lossless endpoint (72.4% accept, 150.0 tok/s, 1.49x, 93.3% accuracy), the
lambda=0.50 run measured 78.1% accept, 153.9 tok/s, 1.53x, and 93.3% accuracy; lambda=0.75 measured
81.2% accept, 158.5 tok/s, 1.57x, and 96.7% accuracy. The higher accuracy values are stochastic
benchmark outcomes, but the accept-rate and decoded-length movement confirms these are interior
operating points rather than endpoint replicas.

### Artifacts
- `initial_program.cpp` — added `SPEC_VERIFY=artifact_p` and `SPEC_ARTIFACT_P` partial artifact
  recovery.
- `analyze_artifact_sweep.py` — evaluator and telemetry summary for the lambda sweep.
- `results/mode_artifact_p025.json`, `mode_artifact_p050.json`, `mode_artifact_p075.json`,
  `mode_artifact_p090.json` — per-item full evals.
- `results/content_artifact_p*_*.jsonl` — per-round telemetry for the four sweep points.
- `results/artifact_sweep_summary.txt` — printed summary from `analyze_artifact_sweep.py`.
