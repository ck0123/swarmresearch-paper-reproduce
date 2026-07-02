// Two decoding modes on a shared codepath, self-contained sampling + speculative algorithm.
//   --mode target    : autoregressive sampling from the 26B target (baseline)
//   --mode mtp-spec  : MTP drafts GAMMA, target verifies, our rejection sampling (lossless)
//
// Draft predictions come from llama.cpp's native gemma-4 MTP head (gemma4-assistant) via the
// nextn staging API; the accept/reject + sampling is implemented here (not llama.cpp's).
//
// Prompts: read from stdin, already chat-templated, delimited by "\n<<END_PROMPT>>\n".
// Per-prompt output block (parsed by run_modes.py):
//   === PROMPT i BEGIN ===
//   decoded N tokens in T s = X tok/s
//   n_drafted N
//   n_accept  N
//   TEXT: <detokenized generation>
//   === PROMPT i END ===
#include "common.h"
#include "llama.h"
#include "llama-ext.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <string>
#include <vector>
#include <random>
#include <algorithm>
#include <iostream>
#include <iterator>

// ---------------- truncated sampling distribution (top_k -> top_p -> temp) ----------------
// A `Dist` is a *sparse* categorical distribution over the vocabulary: it stores only the
// tokens that survived truncation (`ids`) together with their normalized probabilities (`ps`),
// which sum to 1. Tokens not present implicitly have probability 0. This is exactly the set of
// tokens the model could actually emit under the (top_k, top_p, temp) sampling settings.
struct Dist {
    std::vector<int>   ids;   // surviving token ids, in descending-probability order
    std::vector<float> ps;    // matching probabilities, normalized to sum to 1
    // Probability the distribution assigns to `id` (0 if it was truncated away).
    // Linear scan — fine because `ids` is small (<= top_k).
    float p_of(int id) const {
        for (size_t i = 0; i < ids.size(); ++i) if (ids[i] == id) return ps[i];
        return 0.0f;
    }
};

// Turn a raw logit vector into the truncated sampling distribution, applying the standard
// llama.cpp sampler pipeline in order: top_k -> softmax(temp) -> top_p (nucleus) -> renormalize.
static Dist make_dist(const float * logits, int n_vocab, int top_k, float top_p, float temp) {
    // Pair every logit with its token id so we can sort while remembering which token is which.
    std::vector<std::pair<float,int>> v(n_vocab);
    for (int i = 0; i < n_vocab; ++i) v[i] = {logits[i], i};
    // top_k: keep only the k highest-logit tokens (k = whole vocab if top_k disabled).
    int k = (top_k > 0 && top_k < n_vocab) ? top_k : n_vocab;
    std::partial_sort(v.begin(), v.begin() + k, v.end(),
                      [](auto& a, auto& b){ return a.first > b.first; });
    v.resize(k);
    // Softmax with temperature over the surviving k logits. Subtract the max (mx) first for
    // numerical stability; higher temp flattens the distribution, temp<=0 is treated as 1.
    float mx = v[0].first, t = temp > 0 ? temp : 1.0f;
    double sum = 0.0; std::vector<double> pr(k);
    for (int i = 0; i < k; ++i) { pr[i] = std::exp((v[i].first - mx) / t); sum += pr[i]; }
    for (int i = 0; i < k; ++i) pr[i] /= sum;
    // top_p (nucleus): walk down in probability order, keeping tokens until the cumulative mass
    // first reaches top_p, then stop. This drops the long low-probability tail.
    Dist d; double cum = 0.0;
    for (int i = 0; i < k; ++i) {
        d.ids.push_back(v[i].second); d.ps.push_back((float) pr[i]); cum += pr[i];
        if (top_p > 0 && top_p < 1.0f && cum >= top_p) break;
    }
    // Renormalize the kept probabilities so they sum to 1 again (the tail we dropped removed mass).
    double s2 = 0.0; for (float p : d.ps) s2 += p;
    for (float & p : d.ps) p = (float)(p / s2);
    return d;
}

// Draw one token from `d` by inverse-CDF sampling: pick r ~ U(0,1) and return the first token
// whose running cumulative probability reaches r. Falls back to the last id on FP rounding.
static int sample_dist(const Dist & d, std::mt19937 & rng) {
    std::uniform_real_distribution<float> u(0.0f, 1.0f);
    float r = u(rng), c = 0.0f;
    for (size_t i = 0; i < d.ids.size(); ++i) { c += d.ps[i]; if (r <= c) return d.ids[i]; }
    return d.ids.back();
}

// Slurp all of stdin and split it into individual prompts on the "\n<<END_PROMPT>>\n" delimiter.
// The trailing segment (no delimiter after it) is kept too, as long as it is non-empty.
static std::vector<std::string> read_prompts() {
    std::string data((std::istreambuf_iterator<char>(std::cin)), std::istreambuf_iterator<char>());
    std::vector<std::string> out;
    const std::string delim = "\n<<END_PROMPT>>\n";
    size_t pos = 0, f;
    while ((f = data.find(delim, pos)) != std::string::npos) { out.push_back(data.substr(pos, f - pos)); pos = f + delim.size(); }
    if (pos < data.size()) { std::string tail = data.substr(pos); if (!tail.empty()) out.push_back(tail); }
    return out;
}

int main(int argc, char ** argv) {
    // ---- defaults for all tunables (overridable on the command line) ----
    std::string mode = "target",   // "target" (baseline) or "mtp-spec" (speculative)
                tgt_path,           // -m  : path to the 26B target model
                mtp_path;           // -md : path to the MTP draft head (mtp-spec only)
    int   top_k = 64, gamma = 4, max_new = 1024;  // gamma = draft length per round
    float top_p = 0.95f, temp = 1.0f;
    uint32_t seed = 0;
    // Minimal hand-rolled "--flag value" parser. `nx()` consumes and returns the next argv token.
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto nx = [&](){ return std::string(argv[++i]); };
        if      (a == "--mode")  mode = nx();
        else if (a == "-m")      tgt_path = nx();
        else if (a == "-md")     mtp_path = nx();
        else if (a == "--temp")  temp = std::stof(nx());
        else if (a == "--top-p") top_p = std::stof(nx());
        else if (a == "--top-k") top_k = std::stoi(nx());
        else if (a == "--gamma") gamma = std::stoi(nx());
        else if (a == "--seed")  seed = (uint32_t) std::stoul(nx());
        else if (a == "-n")      max_new = std::stoi(nx());
    }
    const bool use_mtp = (mode == "mtp-spec");
    if (tgt_path.empty() || (use_mtp && mtp_path.empty())) {
        fprintf(stderr, "usage: %s --mode target|mtp-spec -m TGT [-md MTP] [...]\n", argv[0]); return 1;
    }

    // ---- load the target model (used in both modes) ----
    llama_backend_init();
    common_params cp;
    cp.model.path = tgt_path; cp.n_gpu_layers = 99; cp.n_ctx = 24576; cp.cpuparams.n_threads = 8;
    auto init = common_init_from_params(cp);
    if (!init || init->context() == nullptr) { fprintf(stderr, "target load failed\n"); return 1; }
    llama_model   * model_tgt = init->model();
    llama_context * ctx_tgt   = init->context();
    const llama_vocab * vocab = llama_model_get_vocab(model_tgt);
    const int n_vocab = llama_vocab_n_tokens(vocab);   // ~262K for Gemma

    // ---- load the MTP draft head and wire it to the target (mtp-spec only) ----
    llama_model   * model_dft = nullptr;
    llama_context * ctx_dft   = nullptr;
    int n_embd = 0;             // width of the hidden-state vector passed target <-> draft head
    llama_batch dbatch{};       // reused batch for single-token draft decodes (carries an embd row)
    if (use_mtp) {
        llama_model_params mp = llama_model_default_params(); mp.n_gpu_layers = 99;
        model_dft = llama_model_load_from_file(mtp_path.c_str(), mp);
        if (!model_dft) { fprintf(stderr, "mtp load failed\n"); return 1; }
        n_embd = llama_model_n_embd_out(model_dft);
        // The draft context is a special "MTP" context that shares the target's KV cache
        // (ctx_other = ctx_tgt) — that is how the draft head sees the already-decoded prefix.
        llama_context_params cparams = llama_context_default_params();
        cparams.n_ctx = 24576; cparams.n_batch = 512;
        cparams.ctx_type = LLAMA_CONTEXT_TYPE_MTP; cparams.ctx_other = ctx_tgt; cparams.n_rs_seq = 0;
        ctx_dft = llama_init_from_model(model_dft, cparams);
        if (!ctx_dft) { fprintf(stderr, "mtp ctx init failed\n"); return 1; }
        // Tell both contexts to expose the "nextn" hidden state: the target produces it, the
        // draft head both consumes the target's hidden and produces its own for the next step.
        llama_set_embeddings_nextn(ctx_tgt, true, false);
        llama_set_embeddings_nextn(ctx_dft, true, true);
        // Allocate the draft batch with room for one embedding row (n_embd floats) per token.
        dbatch = llama_batch_init(512, n_embd, 1);
        dbatch.token = (llama_token *) malloc(sizeof(llama_token) * 512);
    }

    auto prompts = read_prompts();
    fprintf(stderr, "mode=%s prompts=%zu gamma=%d temp=%.2f top_p=%.2f top_k=%d\n",
            mode.c_str(), prompts.size(), gamma, temp, top_p, top_k);
    std::mt19937 rng(seed);   // single RNG shared across prompts; seeded for reproducibility

    for (size_t pi = 0; pi < prompts.size(); ++pi) {
        // Fresh start per prompt: wipe both KV caches (sequence 0, all positions).
        llama_memory_seq_rm(llama_get_memory(ctx_tgt), 0, 0, -1);
        if (ctx_dft) llama_memory_seq_rm(llama_get_memory(ctx_dft), 0, 0, -1);

        std::vector<llama_token> inp = common_tokenize(ctx_tgt, prompts[pi], true, true);
        int n = (int) inp.size();
        std::vector<llama_token> out;               // generated tokens (excludes the prompt)
        int n_drafted = 0, n_accept = 0;            // speculative-decoding counters (for stats)
        auto t0 = ggml_time_us();

        // Prefill: run the whole prompt through the target in one pass, populating the KV cache.
        llama_decode(ctx_tgt, llama_batch_get_one(inp.data(), n));   // full prompt
        llama_token id_last = inp.back();
        // Next-token distribution given the prompt (logits at the last prompt position, n-1).
        Dist p_dist = make_dist(llama_get_logits_ith(ctx_tgt, n - 1), n_vocab, top_k, top_p, temp);

        if (mode == "target") {
            // ---- Baseline: plain autoregressive sampling, one token per target forward pass ----
            int n_past = n;
            while ((int) out.size() < max_new) {
                // temp<=0 means greedy (argmax = most-probable id); otherwise sample.
                int tok = (temp <= 0) ? p_dist.ids[0] : sample_dist(p_dist, rng);
                out.push_back(tok);
                if (llama_vocab_is_eog(vocab, tok)) break;   // stop on end-of-generation token
                // Feed the chosen token back in and read the distribution for the next step.
                llama_decode(ctx_tgt, llama_batch_get_one(&tok, 1));
                n_past++;
                p_dist = make_dist(llama_get_logits_ith(ctx_tgt, 0), n_vocab, top_k, top_p, temp);
            }
        } else {
            // ---- Speculative decoding: draft gamma tokens cheaply, verify them in one target pass ----
            //
            // Each round does three things:
            //   (1) the small MTP head DRAFTS up to `gamma` candidate tokens (cheap, sequential),
            //   (2) the target VERIFIES all of them in a single batched forward pass,
            //   (3) speculative-sampling ACCEPT/REJECT decides how many drafts to keep and emits
            //       exactly one extra "correction"/bonus token — so output is distributed exactly
            //       as if it had been sampled from the target alone (lossless).
            //
            // Position bookkeeping invariant, true at the top of every round:
            //   `pos`    = the position id_last WOULD occupy; it is NOT yet in the KV cache.
            //   the KV cache holds exactly the confirmed prefix, positions [0 .. pos-1].
            //   `h_last` = the target's hidden ("nextn") state for id_last; the draft head needs it.
            std::vector<float> h_last(n_embd);
            std::memcpy(h_last.data(), llama_get_embeddings_nextn_ith(ctx_tgt, n - 1), sizeof(float)*n_embd);
            int pos = n - 1;
            llama_memory_seq_rm(llama_get_memory(ctx_tgt), 0, pos, -1);   // drop id_last; verify re-decodes it

            while ((int) out.size() < max_new) {
                // (1) DRAFT: run the MTP head gamma times, each step feeding it the previous token
                //     plus its hidden state. `draft[g]` is the g-th proposed token and `qd[g]` is
                //     the draft distribution it was sampled from (needed later for accept/reject).
                //     Note: every draft step decodes at the same scratch position `pos`; the
                //     sequential dependency flows through the hidden state `hcur`, not the position.
                std::vector<llama_token> draft; std::vector<Dist> qd;
                llama_token cur = id_last; std::vector<float> hcur = h_last;
                for (int g = 0; g < gamma; ++g) {
                    common_batch_clear(dbatch);
                    common_batch_add(dbatch, cur, pos, { 0 }, true);      // token + position
                    std::memcpy(dbatch.embd, hcur.data(), sizeof(float)*n_embd);  // + its hidden state
                    if (llama_decode(ctx_dft, dbatch) != 0) break;        // bail this round on error
                    Dist q = make_dist(llama_get_logits_ith(ctx_dft, 0), n_vocab, top_k, top_p, temp);
                    int dtok = (temp <= 0) ? q.ids[0] : sample_dist(q, rng);
                    draft.push_back(dtok); qd.push_back(q);
                    // Carry this step's hidden state forward to seed the next draft step.
                    std::memcpy(hcur.data(), llama_get_embeddings_nextn_ith(ctx_dft, 0), sizeof(float)*n_embd);
                    cur = dtok;
                }
                int G = (int) draft.size();   // tokens actually drafted (== gamma unless decode failed)
                if (G == 0) break;
                n_drafted += G;
                llama_memory_seq_rm(llama_get_memory(ctx_tgt), 0, pos, -1);   // clear draft's scratch writes to the shared cache

                // (2) VERIFY: feed [id_last, draft[0..G-1]] to the target in ONE batched pass at
                //     positions pos..pos+G. Row g of the output gives the target's true next-token
                //     distribution AFTER consuming id_last and draft[0..g-1] — i.e. the distribution
                //     the target itself would have used to produce draft[g]. Row G is the bonus
                //     distribution after all G drafts (used if every draft is accepted).
                std::vector<llama_token> seq; seq.push_back(id_last);
                for (int g = 0; g < G; ++g) seq.push_back(draft[g]);
                llama_batch vb = llama_batch_init((int)seq.size(), 0, 1);
                for (int i = 0; i < (int) seq.size(); ++i) common_batch_add(vb, seq[i], pos + i, { 0 }, true);
                llama_decode(ctx_tgt, vb);

                // (3) ACCEPT/REJECT via speculative sampling.
                //     For each draft in order: accept with probability min(1, p(x)/q(x)) where p is
                //     the target dist and q the draft dist. On the FIRST rejection, resample the
                //     correction token from the normalized residual (p - q)+ and stop. If all G are
                //     accepted, sample one bonus token from the row-G distribution. Either way the
                //     round emits `accepted` draft tokens followed by exactly one fresh token.
                int accepted = 0; llama_token new_tok = -1;
                {
                    std::uniform_real_distribution<float> uni(0.0f, 1.0f);
                    bool resampled = false;
                    for (int g = 0; g < G; ++g) {
                        Dist p_g = make_dist(llama_get_logits_ith(ctx_tgt, g), n_vocab, top_k, top_p, temp);
                        float px = p_g.p_of(draft[g]), qx = qd[g].p_of(draft[g]);
                        float acc = (qx > 0.0f) ? std::min(1.0f, px/qx) : 0.0f;
                        if (uni(rng) < acc) { accepted++; continue; }    // accept this draft, move on
                        // Rejected: build the residual distribution (p - q)+ over p's support and
                        // sample the correction token from it (this is what makes the scheme exact).
                        Dist res; double s = 0.0;
                        for (size_t j = 0; j < p_g.ids.size(); ++j) {
                            float diff = p_g.ps[j] - qd[g].p_of(p_g.ids[j]);
                            if (diff > 0) { res.ids.push_back(p_g.ids[j]); res.ps.push_back(diff); s += diff; }
                        }
                        if (s <= 0) new_tok = p_g.ids[0];                // degenerate residual: take argmax
                        else { for (float & pp : res.ps) pp = (float)(pp / s); new_tok = sample_dist(res, rng); }
                        resampled = true; break;                         // stop at first rejection
                    }
                    if (!resampled) {
                        // All G drafts accepted -> sample the bonus token from the row-G distribution.
                        Dist p_b = make_dist(llama_get_logits_ith(ctx_tgt, G), n_vocab, top_k, top_p, temp);
                        new_tok = (temp <= 0) ? p_b.ids[0] : sample_dist(p_b, rng);
                    }
                }
                n_accept += accepted;

                // The last CONFIRMED token sits at verify-batch row `accepted`; grab its hidden
                // state to seed next round's drafting. (If a draft was rejected, `accepted` points
                // at id_last+accepted drafts; if all accepted, it points at the last draft.)
                std::memcpy(h_last.data(), llama_get_embeddings_nextn_ith(ctx_tgt, accepted), sizeof(float)*n_embd);
                // Trim the cache to the confirmed prefix: [0..pos-1] + id_last + accepted drafts.
                int keep = pos + 1 + accepted;                                  // prefix + id_last + accepted drafts
                llama_memory_seq_rm(llama_get_memory(ctx_tgt), 0, keep, -1);

                // Emit the accepted draft tokens, then the new (corrected/bonus) token. Stop early
                // on an end-of-generation token or once we hit max_new.
                bool eog = false;
                for (int g = 0; g < accepted && (int) out.size() < max_new; ++g) {
                    out.push_back(draft[g]);
                    if (llama_vocab_is_eog(vocab, draft[g])) { eog = true; break; }
                }
                out.push_back(new_tok);
                if (llama_vocab_is_eog(vocab, new_tok)) eog = true;
                // Re-establish the invariant: new_tok becomes id_last at position `keep`, not yet
                // cached (it gets re-decoded as the seed of next round's verify batch).
                id_last = new_tok;
                pos = keep;                                                    // new_tok not cached; decoded next round
                llama_batch_free(vb);
                if (eog || (int) out.size() >= max_new) break;
            }
        }

        // ---- per-prompt report (format is parsed by run_modes.py) ----
        double dt = (ggml_time_us() - t0) / 1e6;          // wall-clock seconds for this prompt
        std::string text;
        for (auto t : out) text += common_token_to_piece(ctx_tgt, t);   // detokenize the output
        printf("\n=== PROMPT %zu BEGIN ===\n", pi);
        printf("decoded %d tokens in %.3fs = %.2f tok/s\n", (int) out.size(), dt, out.size()/dt);
        printf("n_drafted %d\n", n_drafted);
        printf("n_accept %d\n", n_accept);
        printf("TEXT: %s\n", text.c_str());
        printf("=== PROMPT %zu END ===\n", pi);
        fflush(stdout);
    }

    // ---- teardown (only the draft side was allocated by us; the target is owned by `init`) ----
    if (ctx_dft) { if (dbatch.token) { free(dbatch.token); dbatch.token = nullptr; } llama_batch_free(dbatch);
                   llama_free(ctx_dft); llama_model_free(model_dft); }
    llama_backend_free();
    return 0;
}
