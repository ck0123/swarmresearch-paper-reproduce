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
