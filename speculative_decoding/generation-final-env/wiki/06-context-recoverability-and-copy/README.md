# Context-recoverability ("copyability") & copy drafting

Whether the target's own generation can be predicted by cheap n-gram suffix lookup against the running
context (prompt-lookup / token-recycling style), and whether a "free" CPU copy drafter pays off in
wall-clock. Also tests whether context-redundancy buys cheaper MoE verification.

**Cross-cutting result.** ~49.5% of tokens are context-recoverable, and copy structure aligns with target
beliefs (59.8% of candidates in top-k/top-p support); copy and MTP partially overlap (40% redundant, 34%
MTP-only, 9.5% copyable-but-missed) covering 83.9% together. **But the "free" copy draft does not translate
to speedup** in the verify-bound regime: copy-spec hits only 24.1% acceptance (0.98× speedup), and hybrid
MTP+copy lands ~1.47× (≈ plain mtp-spec) because deep copy-extension tokens accept at only ~18%. Crucially,
context-recoverability raises acceptance (81.4% vs 72.4%) but leaves verify cost **flat** (~18 ms/round) —
it's within-batch token *diversity* (ndist), not cross-time recoverability, that drives MoE cost.

| File | Hook |
|---|---|
| [researcher-35](researcher-35.md) | Context-recoverability ("copyability"): 49.5% of tokens recoverable by longest-suffix lookup; 59.8% of candidates in target support; copy/MTP overlap covers 83.9% (40% redundant, 34% MTP-only, 9.5% copyable-but-missed). |
| [researcher-38](researcher-38.md) | Copyability (extended) — how copy structure relates to the target's beliefs and where it complements vs duplicates the MTP draft. |
| [researcher-43](researcher-43.md) | **Synthesis — does context-redundancy buy MoE route-locality?** Two orthogonal axes: recoverability raises acceptance (81.4% vs 72.4%) but verify cost stays flat (~18 ms/round); within-batch diversity (ndist), not context-source, drives MoE cost. |
| [researcher-48](researcher-48.md) | End-to-end "free copy drafter": copy-spec eliminates draft forward but throughput drops to 0.98× (24.1% accept, verify doubles); hybrid MTP+4 copy tokens ≈1.47× because deep copy tokens accept at 18.6%; verify-bound makes draft cheapness inert. |
