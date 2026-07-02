# Prompt distribution, KV-cache & numerical determinism

The input/infrastructure side: how the 30 prompts tokenize and fill the context window, how much KV
context the next-token decision actually needs, and whether the shared-KV speculative pipeline is
numerically reproducible and correct after cache trimming.

**Cross-cutting result.** Prefill is tiny relative to the window (<4%), with a 10× spread across items
(LCB dominates). The target's next-token decision is often **long-horizon** — ~56% of decisions need
>32 tokens, ~15% need >256 — and the horizon lengthens as generation proceeds (math is short-horizon,
knowledge-MCQ longest). The pipeline is **bit-exactly reproducible** but uses batched-verify logits
(batch=G+1) that differ from true batch=1 AR in 100% of rounds (median ~0.66 logits); this perturbs
acceptance probabilities as **zero-mean noise** and flips the target argmax in ~0.3–0.7% of rows, with
**no** downstream accuracy impact (93.3% preserved across seeds).

| File | Hook |
|---|---|
| [researcher-19](researcher-19.md) | Prompt & KV-cache footprint: prefill 101–1008 tokens (10× spread), LCB = 59% of prefill on 33% of items; min context slack 7,184 slots; no item exhausts the 24,576 window. |
| [researcher-20](researcher-20.md) | Prefill distribution, chat-template overhead, tokenizer density: gemma-4 scaffold is exactly 16 special tokens; density varies (LCB 3.27 vs GPQA 3.96 chars/tok); fixed frame is 45–48% of short prompts but <8% of long. |
| [researcher-21](researcher-21.md) | Numerical determinism / reproducibility: bit-exact run-to-run, but batched-verify logits differ from batch=1 AR in 100% of rounds (median 0.66); accept prob perturbed in 22.5% of rounds, argmax never flips, 93.3% held. |
| [researcher-22](researcher-22.md) | Dynamic KV-cache audit after trim: all 30 audited rounds show mutated-cache logits differ from clean replay at >1e-3; per-item max deviation 0.37–1.11, mean 0.726. |
| [researcher-23](researcher-23.md) | Per-position batch-invariance probe + true-AR reconstruction: acceptance perturbation is zero-mean symmetric noise (~±1e-4) but top token flips 0.4–0.7% of rows; macro eval shows no accuracy impact. |
| [researcher-49](researcher-49.md) | Attention-span / context-horizon via causal ablation: 15% of decisions need >256 tokens, 5% >512; attention sink (first 4 tokens) recovers +11.5pp at short windows; math short-horizon, code heaviest long tail. |
| [researcher-53](researcher-53.md) | Attention-span + prompt retention: ~56% of decisions need >32 tokens, ~27% >128; retaining the full prompt recovers 22% of lost decisions vs 4% from the sink alone; prompt contribution is broad, not just first tokens. |
| [researcher-55](researcher-55.md) | Seed-stability of the batch-vs-AR drift: across 5 seeds, acceptance perturbation is zero-mean (+1052/−1083 split), argmax flips ~0.3% but seed-stable, logit-drift p99 narrow (3.3–4.1e-2); macro impact nil. |
