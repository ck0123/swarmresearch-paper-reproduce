# Findings — researcher-21

## Angle: numerical determinism / reproducibility of the shared-KV speculative pipeline

Orthogonal lens (not acceptance, timing, output structure, latent geometry, gamma, or
prompt structure): does `mtp-spec` reproduce, and is the rejection sampler's "lossless"
contract numerically exact? The contract requires the target accept probability `px` to be
read from the *true* per-position target distribution. In this implementation `px` is read
from a **batched verify decode** (G+1 tokens decoded in one `llama_decode`), whereas
`target` mode decodes **one token at a time** (batch=1). FP non-associativity makes those
two forwards numerically different.

Setup: target `gemma-4-26B-A4B-it` Q8_0 + MTP head, GPUs 0,1, seed 0, temp 1.0, top_p 0.95,
top_k 64, gamma 4.

### Finding 1 — `mtp-spec` is bit-exactly reproducible run-to-run
Two identical `mtp-spec` runs (`--limit 6 --n 1024`, GPUs 0,1, seed 0):
- Per-prompt generated TEXT byte-identical across both runs for all 6 prompts (both GPU chunks).
- `n_drafted` and `n_accept` identical per prompt (e.g. 944/791, 1056/762, 888/802 ...).
- The ONLY difference between the two raw output files is the throughput/timing line
  (`X tok/s`); 0 non-timing lines differ.
- Conclusion: GPU FP non-determinism does **not** flip any accept/reject decision here; the
  whole pipeline (draft sampling + rejection test + resample, single shared `std::mt19937`)
  is deterministic across repeats given fixed seed and fixed GPU/chunk assignment.

### Finding 2 — batched-verify logits differ substantially from single-token (AR) logits
Probe (`DIAG` env gate, inert when unset): for the first 40 verify rounds of 8 prompts
(5 AIME + 3 GPQA), at each verified position compare the target row-0 logits from the
batched verify (batch=G+1) against a fresh batch=1 decode of the same token at the same
position over the same (shared) KV prefix. 320 rounds measured.

- `frac_diff = 1.0000` in every round: **100% of the ~262k vocab logits differ** between
  batch=G+1 and batch=1. The two forwards are never bit-identical.
- `max_abs_logit_diff` (raw pre-softmax logit units): median 0.66, mean 0.82, max 7.78.
  These are large, not rounding dust — the batch dimension changes the reduction/kernel path.
- `argmax_match = 320/320 (100%)`: despite the logit shifts, the top token never flips.
  Greedy (temp<=0) decoding would be unaffected; the dominant token is robust.

### Finding 3 — the batch effect perturbs the rejection-test acceptance probability
Converting logits through the same top_k->top_p->temp truncation used by the sampler:
- The acceptance probability `px = min(1, p(draft)/q(draft))` computed from the batched
  verify differs from the batch=1 (AR) value in **72/320 rounds (22.5%)**.
- Magnitude: max single-round `acc_delta = 5.16e-2` (5.2%); among the 72 nonzero rounds
  median 4.85e-3, mean 7.34e-3, p90 1.68e-2.
- Concentrated on uncertain tokens: of 39 "soft" rounds (draft token prob q<0.999), 21
  (54%) show nonzero `acc_delta` (max 2.79e-2); of 281 "sharp" rounds (q>=0.999), 51 (18%)
  show small deltas. Where the truncated distribution collapses to a single token
  (px=1.0 both ways), the large logit differences cancel and `acc_delta = 0`.

### Interpretation (factual)
- Finding 1 and Findings 2-3 are consistent: `mtp-spec` is deterministic *within itself*
  (the batched computation is reproducible), yet its effective sampling distribution is the
  **batched** target distribution, which is not bit-identical to the **autoregressive**
  distribution that `target` mode samples and that the "lossless" rejection-sampling
  identity is defined against. The accept probability the sampler uses is a batch-perturbed
  proxy for the true AR `p`, off by up to ~5% on uncertain tokens.
- The resample branch `(p - q)+` is also built from the batched `p`, so it inherits the same
  perturbation.
- Because argmax never flips, the bias does not change the dominant token, consistent with
  `mtp-spec` matching `target` accuracy. The exactness of "lossless" holds only up to
  batch-size-dependent FP; it is distributional-near-equivalence, not the textbook identity.

### Caveat / scope
The probe holds the KV prefix fixed (both arms attend over the same shared, batched-built
cache) and isolates the marginal batch effect at the query position's final forward. A full
`target`-mode run builds its KV cache entirely under batch=1, so the true cross-mode
divergence is at least this large. `argmax_match` and `acc_delta` were measured only up to
40 rounds/prompt at n=512; Finding 1 reproducibility was measured at n=1024.

### Reproduction
- Finding 1: run `./evaluator/task-eval --gpus 0,1 --modes mtp-spec --limit 6 --n 1024`
  twice; diff `results/raw_mtp-spec_gpu{0,1}.txt` ignoring the `tok/s` line.
- Findings 2-3: `DIAG=40 ./spec_modes --mode mtp-spec -m <TGT> -md <MTP> --seed 0 -n 512
  --gamma 4 < prompts` and parse `DIAG ...` stderr lines. Instrumentation is gated behind
  the `DIAG` env var (default 0 => inert; scored path unchanged).
