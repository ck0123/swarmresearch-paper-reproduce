# Findings — attention-span (context-horizon) of the target's next-token decision

Branch: researcher-49. Lens: a **causal attention-context ablation** of the target
(gemma-4-26B-A4B-it, Q8_0). Speculative decoding's MTP head drafts from a single target
hidden state while the target verifies with the full KV cache; this study asks how much KV
context the target's *own* next-token decision actually depends on, and disentangles
attention-reach from attention-sink reliance. This is orthogonal to acceptance/timing/
uncertainty/geometry lenses: it intervenes on the attention span and measures decision
preservation.

## Method
Added `--mode ctx-probe` to `initial_program.cpp` (no draft model; target only).
Per prompt: generate on-policy (temp 1.0, top_k 64, top_p 0.95, seed 0), recording the
full-context argmax token at each query position `q`. Then at strided positions re-decode the
same query under three truncated-context conditions and check whether the full-context argmax
is still the argmax (`agree`) and its logit-rank (`#logits > logit[argmax]`):
- `recent`     : window tokens `[q-W+1 .. q]` at their TRUE absolute positions (pure reach).
- `shift`      : same window renumbered to start at position 0 (reach + position shift).
- `shift+sink` : first `sink=4` tokens + window, contiguous (adds the attention sink back).
So `shift+sink − shift` isolates the attention-sink contribution (matched layout) and
`shift − recent` isolates the absolute-position-shift contribution.

Run: 30-item benchmark, GPU 4, `-n 512 --probe-stride 12 --probe-sink 4`,
windows `{8,16,32,64,128,256,512,2048}`. 1286 PROBE records.
Artifacts: `analysis/probe_full.out`, `analysis/probe_summary.txt`, `analysis/analyze.py`,
`analysis/make_prompts.py`. `W=2048` ≥ every context length here, so it is the full-context
reference; it scores 99.6% (the residual ~0.4% is full-prefill-vs-incremental-decode argmax
disagreement on near-ties — the measurement floor, not analyzed here).

## 1. Context-horizon curve (cond=recent, all domains)
Fraction of next-token decisions preserved vs attention window W:

| W | 8 | 16 | 32 | 64 | 128 | 256 | 512 | full |
|---|---|----|----|----|-----|-----|-----|------|
| agree% | 16.8 | 27.1 | 43.7 | 58.5 | 72.5 | 85.2 | 94.8 | 99.6 |

- Decisions are far from locally determined: ~56% require >32 tokens, ~27% require >128,
  ~15% require >256, ~5% require >512 tokens of attention context.
- The curve is smooth with no sharp knee inside the tested range; preserving 95% of the
  target's own decisions needs ~512 tokens of reach.

## 2. Attention-sink contribution (shift+sink − shift, matched)
Restoring the first 4 tokens recovers decisions, concentrated at short windows:

| W | 8 | 16 | 32 | 64 | 128 | 256 | 512 |
|---|---|----|----|----|-----|-----|-----|
| Δagree (pp) | +7.1 | +11.5 | +7.8 | +6.1 | +5.2 | +4.1 | +1.4 |

The sink tokens carry decision-relevant signal that a pure recent window drops; their value
peaks at W≈16 (+11.5pp) and decays to ~0 once the window is long (W=512: +1.4pp). Mean-rank
for `shift+sink` is sometimes higher than `shift` despite higher agreement: the sink flips
many near-ties to rank 0 but leaves a heavy tail of large-rank cases.

## 3. Absolute-position-shift contribution (shift − recent, matched)
| W | 8 | 16 | 32 | 64 | 128 | 256 | 512 |
|---|---|----|----|----|-----|-----|-----|
| Δagree (pp) | −0.4 | −1.0 | −1.4 | +0.3 | −0.0 | −2.4 | −4.3 |

Renumbering the window toward the origin (compressing the gap to position 0) is nearly free
for short windows but costs up to −4.3pp at W=512: longer-range decisions are more sensitive
to absolute position, not only to which tokens are in the window.

## 4. Cross-domain context horizon (cond=recent, agree%)
| W | aime | gpqa | hle | lcb |
|---|------|------|-----|-----|
| 8   | 23.7 | 13.5 | 14.9 | 17.7 |
| 32  | 49.3 | 38.8 | 36.3 | 49.5 |
| 64  | 66.0 | 56.5 | 50.2 | 60.9 |
| 128 | 79.5 | 68.8 | 65.6 | 76.0 |
| 256 | 93.5 | 82.6 | 80.0 | 86.3 |
| 512 | 98.6 | 95.1 | 95.3 | 92.3 |

Strong domain dependence of the attention horizon:
- **aime (math)** decisions are the most LOCAL: 66% preserved at W=64, 93.5% at W=256, and
  mean-rank collapses to ~4 by W=64. Math token choices are largely determined by nearby context.
- **gpqa / hle (knowledge/reasoning MCQ)** are the most long-range: W=8 only 13–15%, and
  mean-rank stays high far out (gpqa W=256 mean-rank ≈ 193).
- **lcb (code)** is moderate at short/mid windows but has the longest tail — it is the only
  domain still below 93% at W=512 (92.3%, mean-rank 7.5), consistent with occasional very
  long-range dependencies (distant identifiers/definitions).

## 5. Position within the generation (cond=recent, agree%)
| W | 64 | 128 | 256 | 512 |
|---|----|-----|-----|-----|
| first half  | 62.0 | 74.7 | 89.7 | 96.2 |
| second half | 54.9 | 70.2 | 80.5 | 93.3 |

The context horizon lengthens as generation proceeds: at every window the later half of the
output is less recoverable from a fixed-size recent window than the earlier half (e.g. W=256:
89.7% → 80.5%). Later decisions integrate more accumulated context.

## Summary of facts
- A large minority of the target's next-token decisions depend on long-range attention
  (~15% need >256 tokens; ~5% need >512); decisions are not locally determined.
- The first few tokens (attention sink) hold decision-relevant signal worth up to +11.5pp of
  decisions when the recent window is short; the effect vanishes for long windows.
- Absolute position matters increasingly with range: shifting a 512-token window to the origin
  costs −4.3pp.
- Context horizon is domain-structured: math is short-horizon, knowledge-MCQ (gpqa/hle) is the
  longest-horizon, code has the heaviest long-range tail; and horizon grows with output position.

# Findings — retained prompt vs older generated tokens under packed-context ablation

Branch: researcher-53. Lens: a retained-context slice extension of the previous causal
attention-span probe. The prior run showed that a packed recent window (`shift`) loses many
full-context next-token argmax decisions. This run asks what kind of omitted context restores
those decisions when all retained tokens are packed contiguously in a matched layout.

## Method
Added `--mode ctx-keep-probe` to `initial_program.cpp` (target only). Per prompt, generate
on-policy with the same sampling setup and record the full-context argmax token at each query
position. At strided query positions, re-decode packed retained slices and check whether the
full-context argmax remains rank 0.

Conditions:
- `recent-packed`: recent W tokens only, renumbered to positions 0..W-1.
- `sink+recent-packed`: first 4 prompt tokens plus recent W tokens, packed.
- `prompt+recent-packed`: full original prompt plus recent W tokens, packed.
- `prompt+genhead+recent-packed`: full prompt, first 64 generated tokens, and recent W tokens,
  packed.
- `prompt+anchors+recent-packed`: full prompt, every 64th older generated token before the
  recent window, and recent W tokens, packed.

Run: 30-item benchmark, `CUDA_VISIBLE_DEVICES=4,5`, `-n 384 --probe-stride 48`,
windows `{64,128,256,512}`, `sink=4`, `gen_keep=64`, `anchor_step=64`. 240 sampled query
positions per condition/window. Artifacts: `analysis/keep_probe_full.out`,
`analysis/keep_probe_full.err`, `analysis/keep_probe_summary.txt`, `analysis/analyze_keep.py`.
The first attempted true-position retained-slice run failed because llama.cpp rejects batches
with holes in the position sequence; the completed run therefore uses packed positions for all
conditions.

Evaluator verification: `./evaluator/task-eval --gpus 4,5` completed on the 30-item benchmark
with `mtp-spec` accuracy 93.3%, mean throughput 144.4 tok/s, accept rate 72.4%, and speedup
1.43x versus the fixed target reference.

## 1. Retaining the full prompt recovers most lost decisions
Agreement with the full-context next-token argmax:

| W | recent-packed | sink+recent | prompt+recent | prompt+genhead+recent | prompt+anchors+recent |
|---|---------------|-------------|---------------|-----------------------|-----------------------|
| 64  | 68.3 | 71.7 | 90.4 | 93.8 | 92.1 |
| 128 | 76.7 | 82.1 | 93.3 | 97.5 | 95.4 |
| 256 | 87.5 | 89.2 | 98.3 | 99.6 | 98.8 |
| 512 | 95.4 | 97.5 | 99.6 | 99.6 | 99.6 |

Prompt retention gives much larger recovery than retaining only the first 4 prompt tokens:

| W | sink+recent − recent | prompt+recent − recent |
|---|----------------------|------------------------|
| 64  | +3.3pp | +22.1pp |
| 128 | +5.4pp | +16.7pp |
| 256 | +1.7pp | +10.8pp |
| 512 | +2.1pp | +4.2pp |

The missing long-range signal is therefore not just the first few attention-sink tokens; much
of it is distributed across the original prompt.

## 2. Older generated tokens add a smaller residual after the prompt is retained
Additional recovery over `prompt+recent-packed`:

| W | + first 64 generated | + sparse generated anchors |
|---|----------------------|----------------------------|
| 64  | +3.3pp | +1.7pp |
| 128 | +4.2pp | +2.1pp |
| 256 | +1.3pp | +0.4pp |
| 512 | +0.0pp | +0.0pp |

Once the full prompt and recent window are present, retaining a small amount of older generated
history gives a measurable but much smaller gain than retaining the prompt itself.

## 3. Domain dependence of prompt recovery
`prompt+recent-packed − recent-packed` agreement deltas:

| W | aime | gpqa | hle | lcb |
|---|------|------|-----|-----|
| 64  | +10.0pp | +26.2pp | +20.0pp | +25.0pp |
| 128 | +10.0pp | +16.2pp | +25.0pp | +16.2pp |
| 256 | +5.0pp  | +12.5pp | +12.5pp | +11.3pp |
| 512 | +0.0pp  | +2.5pp  | +2.5pp  | +8.8pp |

The full prompt is especially important for gpqa/hle/lcb at short and middle windows. AIME has
the smallest prompt-retention delta, matching the previous finding that math decisions are more
local.

## 4. Recovery by omitted generated-token gap
`prompt+recent-packed − recent-packed` agreement deltas by number of older generated tokens
omitted between the prompt and the recent window:

| gap omitted | W=64 | W=128 | W=256 |
|-------------|------|-------|-------|
| none   | +6.7pp  | +13.3pp | +11.7pp |
| 1-64   | +36.7pp | +28.3pp | +10.0pp |
| 65-256 | +22.5pp | +12.2pp | +6.7pp |
| >256   | +36.7pp | - | - |

Prompt retention improves agreement even when the recent window has not yet omitted generated
tokens (`gap=none`), because it restores earlier prompt tokens outside the recent window. When
the recent window has also dropped generated history, the prompt still accounts for a large
fraction of the missing decision signal.

## Summary of retained-slice facts
- Under packed-context ablation, retaining the full original prompt plus recent W tokens raises
  agreement from 68.3% to 90.4% at W=64 and from 76.7% to 93.3% at W=128.
- The first 4 sink tokens recover only a small part of this effect in the same packed layout
  (+3.3pp to +5.4pp at W=64/128), so the prompt contribution is distributed beyond the sink.
- After the full prompt is retained, a first-64 generated prefix or sparse generated anchors add
  only 0.4-4.2pp at W<=256 and nothing measurable at W=512 in this run.
- Prompt recovery is domain-structured: weakest for AIME, strongest for gpqa/hle/lcb at short
  and middle windows.
