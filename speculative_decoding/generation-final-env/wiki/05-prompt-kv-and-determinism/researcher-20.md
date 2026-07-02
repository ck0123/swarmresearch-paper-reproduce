# Findings — Input/prompt side: prefill-token distribution, chat-template overhead, tokenizer density

Lens: the **inputs themselves** (the 30 rendered prompts and their tokenization), deliberately
orthogonal to acceptance, depth/run-length, gamma, timing, output/extraction, and draft-head
geometry. No quantity below depends on the decoding mode — these are static properties of the
benchmark inputs and the gemma-4 tokenizer/chat template.

## Method
- Rendered all 30 prompts with `evaluator/gemma4_chat_template.jinja`
  (`add_generation_prompt=True`, `enable_thinking=True`, `bos_token=""`), matching `task-eval`.
- Added one stderr line to `initial_program.cpp` (`n_prompt[i] N`) reporting the exact prefill
  token count from `common_tokenize(ctx_tgt, prompt, true, true)`. Ran the real binary on GPUs 0,1
  (`--mode target`) to read exact token counts (`task-eval` truncates stderr to 2000 chars, so the
  binary was driven directly). Probe prompts with empty/known content isolate the fixed overhead.
- Artifacts: `results/prompt_token_counts.json`, `results/scaffold_costs.json`.

## Facts

### 1. Prefill-token distribution is highly skewed (10× spread), LCB-dominated
- Per-prompt prefill length: min 101, median 226, max 1008 tokens → **10.0× spread**.
- Total prefill across 30 prompts = **10,249 tokens**.
- Per-source totals / medians (n, sum, median tok):
  - aime  : n=5,  sum=811,  median=174  (min 105, max 189)
  - gpqa  : n=10, sum=2232, median=196  (min 160, max 472)
  - hle   : n=5,  sum=1200, median=217  (min 101, max 469)
  - lcb   : n=10, sum=6006, median=575  (min 292, max 1008)
- The 10 LCB items are 33% of the set but **59% (6006/10249) of all prefill tokens**;
  median LCB prompt (575) is ~3.3× the median AIME prompt (174).

### 2. The gemma-4 chat scaffold is exactly 16 atomic special tokens
- Rendering empty system + empty user content tokenizes to **16 tokens**. Adding a one-char user
  message adds exactly 1 token (17). The role delimiters `<|turn>`, `<turn|>`, `<|think|>` are each
  a single dedicated special token (16 = BOS + `<|turn>`/`<turn|>`/`<|think|>` markers + the literal
  role words `system`/`user`/`model` + newlines for the system+user+model frame).
- System-message overhead (scaffold + per-source system prompt, empty user):
  aime 58, gpqa 48, hle 48, lcb 49 tokens. So fixed (template+system) overhead is ~48–58 tokens,
  identical across the items of a source.
- Consequence for short prompts: hle-othe-03 (101 tok) and aime-6-10-01 (105 tok) carry only
  ~53 / ~47 tokens of actual user content; the fixed scaffold+system frame is **~45–48% of the
  prompt**. For the longest LCB prompts the same fixed frame is <8%.

### 3. Tokenizer density (chars/token) varies by domain; code packs densest
- Mean chars/token per source: lcb **3.27**, aime 3.67, hle 3.75, gpqa 3.96.
- LCB code prompts fragment into ~17% more tokens per character than GPQA prose
  (3.27 vs 3.96 chars/tok), i.e. code is the least tokenizer-efficient input domain here.
- Extremes: lcb-medium-04 = 2.55 chars/tok (densest, heavy code/structure);
  gpqa-biol-00 = 4.98 chars/tok (prose-heavy, sparsest).
- Content-only chars/token (subtracting the per-source system frame) preserves the ordering, so
  the density gap is a property of the user content domain, not the template.

### 4. Lexical composition is heterogeneous across the four sources (rendered text)
- aime: short, LaTeX-dense (\\boxed, $...$), no MCQ scaffold; 397–709 chars.
- gpqa/hle: multiple-choice (5 option markers each), mixed prose + occasional LaTeX; 337–1734 chars.
- lcb: longest, code-block / function-signature heavy ({}();, def/return/print), 1093–2571 chars,
  39–99 lines (vs 7–22 lines for the other sources).

### 5. KV headroom
- `n_ctx = 24576`. Worst case here = longest prompt (1008) + largest generation budget
  (AIME `max_tokens` 16384) = 17392 < 24576, so no single item can exhaust the context window;
  prefill occupies at most ~4% of n_ctx before generation begins.

## Notes
- The instrumentation is stderr-only (`n_prompt[i] N`) and does not alter decoding behavior or the
  scored stdout blocks; the `--limit 30 --n 32` run reported accuracy 0.0% only because `-n 32`
  truncates generations before any answer is emitted (counts were the sole purpose of that run).
- clang in-editor diagnostics on `initial_program.cpp` are missing-llama-include-path noise; the
  g++ build via `task-eval` succeeds.
