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
struct Dist {
    std::vector<int>   ids;
    std::vector<float> ps;
    float p_of(int id) const {
        for (size_t i = 0; i < ids.size(); ++i) if (ids[i] == id) return ps[i];
        return 0.0f;
    }
};

static Dist make_dist(const float * logits, int n_vocab, int top_k, float top_p, float temp) {
    std::vector<std::pair<float,int>> v(n_vocab);
    for (int i = 0; i < n_vocab; ++i) v[i] = {logits[i], i};
    int k = (top_k > 0 && top_k < n_vocab) ? top_k : n_vocab;
    std::partial_sort(v.begin(), v.begin() + k, v.end(),
                      [](auto& a, auto& b){ return a.first > b.first; });
    v.resize(k);
    float mx = v[0].first, t = temp > 0 ? temp : 1.0f;
    double sum = 0.0; std::vector<double> pr(k);
    for (int i = 0; i < k; ++i) { pr[i] = std::exp((v[i].first - mx) / t); sum += pr[i]; }
    for (int i = 0; i < k; ++i) pr[i] /= sum;
    Dist d; double cum = 0.0;
    for (int i = 0; i < k; ++i) {
        d.ids.push_back(v[i].second); d.ps.push_back((float) pr[i]); cum += pr[i];
        if (top_p > 0 && top_p < 1.0f && cum >= top_p) break;
    }
    double s2 = 0.0; for (float p : d.ps) s2 += p;
    for (float & p : d.ps) p = (float)(p / s2);
    return d;
}

static int sample_dist(const Dist & d, std::mt19937 & rng) {
    std::uniform_real_distribution<float> u(0.0f, 1.0f);
    float r = u(rng), c = 0.0f;
    for (size_t i = 0; i < d.ids.size(); ++i) { c += d.ps[i]; if (r <= c) return d.ids[i]; }
    return d.ids.back();
}

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
    std::string mode = "target", tgt_path, mtp_path;
    int   top_k = 64, gamma = 4, max_new = 1024;
    float top_p = 0.95f, temp = 1.0f;
    uint32_t seed = 0;
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

    llama_backend_init();
    common_params cp;
    cp.model.path = tgt_path; cp.n_gpu_layers = 99; cp.n_ctx = 24576; cp.cpuparams.n_threads = 8;
    auto init = common_init_from_params(cp);
    if (!init || init->context() == nullptr) { fprintf(stderr, "target load failed\n"); return 1; }
    llama_model   * model_tgt = init->model();
    llama_context * ctx_tgt   = init->context();
    const llama_vocab * vocab = llama_model_get_vocab(model_tgt);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    llama_model   * model_dft = nullptr;
    llama_context * ctx_dft   = nullptr;
    int n_embd = 0;
    llama_batch dbatch{};
    if (use_mtp) {
        llama_model_params mp = llama_model_default_params(); mp.n_gpu_layers = 99;
        model_dft = llama_model_load_from_file(mtp_path.c_str(), mp);
        if (!model_dft) { fprintf(stderr, "mtp load failed\n"); return 1; }
        n_embd = llama_model_n_embd_out(model_dft);
        llama_context_params cparams = llama_context_default_params();
        cparams.n_ctx = 24576; cparams.n_batch = 512;
        cparams.ctx_type = LLAMA_CONTEXT_TYPE_MTP; cparams.ctx_other = ctx_tgt; cparams.n_rs_seq = 0;
        ctx_dft = llama_init_from_model(model_dft, cparams);
        if (!ctx_dft) { fprintf(stderr, "mtp ctx init failed\n"); return 1; }
        llama_set_embeddings_nextn(ctx_tgt, true, false);
        llama_set_embeddings_nextn(ctx_dft, true, true);
        dbatch = llama_batch_init(512, n_embd, 1);
        dbatch.token = (llama_token *) malloc(sizeof(llama_token) * 512);
    }

    auto prompts = read_prompts();
    fprintf(stderr, "mode=%s prompts=%zu gamma=%d temp=%.2f top_p=%.2f top_k=%d\n",
            mode.c_str(), prompts.size(), gamma, temp, top_p, top_k);
    std::mt19937 rng(seed);

    for (size_t pi = 0; pi < prompts.size(); ++pi) {
        llama_memory_seq_rm(llama_get_memory(ctx_tgt), 0, 0, -1);
        if (ctx_dft) llama_memory_seq_rm(llama_get_memory(ctx_dft), 0, 0, -1);

        std::vector<llama_token> inp = common_tokenize(ctx_tgt, prompts[pi], true, true);
        int n = (int) inp.size();
        std::vector<llama_token> out;
        int n_drafted = 0, n_accept = 0;
        auto t0 = ggml_time_us();

        llama_decode(ctx_tgt, llama_batch_get_one(inp.data(), n));   // full prompt
        llama_token id_last = inp.back();
        Dist p_dist = make_dist(llama_get_logits_ith(ctx_tgt, n - 1), n_vocab, top_k, top_p, temp);

        if (mode == "target") {
            int n_past = n;
            while ((int) out.size() < max_new) {
                int tok = (temp <= 0) ? p_dist.ids[0] : sample_dist(p_dist, rng);
                out.push_back(tok);
                if (llama_vocab_is_eog(vocab, tok)) break;
                llama_decode(ctx_tgt, llama_batch_get_one(&tok, 1));
                n_past++;
                p_dist = make_dist(llama_get_logits_ith(ctx_tgt, 0), n_vocab, top_k, top_p, temp);
            }
        } else {
            // Invariant per round: `pos` = position of id_last (NOT in cache); cache holds the
            // confirmed prefix [0..pos-1]. h_last = target hidden for id_last.
            std::vector<float> h_last(n_embd);
            std::memcpy(h_last.data(), llama_get_embeddings_nextn_ith(ctx_tgt, n - 1), sizeof(float)*n_embd);
            int pos = n - 1;
            llama_memory_seq_rm(llama_get_memory(ctx_tgt), 0, pos, -1);   // drop id_last; verify re-decodes it

            while ((int) out.size() < max_new) {
                // (1) DRAFT G tokens via MTP at scratch position `pos`
                std::vector<llama_token> draft; std::vector<Dist> qd;
                llama_token cur = id_last; std::vector<float> hcur = h_last;
                for (int g = 0; g < gamma; ++g) {
                    common_batch_clear(dbatch);
                    common_batch_add(dbatch, cur, pos, { 0 }, true);
                    std::memcpy(dbatch.embd, hcur.data(), sizeof(float)*n_embd);
                    if (llama_decode(ctx_dft, dbatch) != 0) break;
                    Dist q = make_dist(llama_get_logits_ith(ctx_dft, 0), n_vocab, top_k, top_p, temp);
                    int dtok = (temp <= 0) ? q.ids[0] : sample_dist(q, rng);
                    draft.push_back(dtok); qd.push_back(q);
                    std::memcpy(hcur.data(), llama_get_embeddings_nextn_ith(ctx_dft, 0), sizeof(float)*n_embd);
                    cur = dtok;
                }
                int G = (int) draft.size();
                if (G == 0) break;
                n_drafted += G;
                llama_memory_seq_rm(llama_get_memory(ctx_tgt), 0, pos, -1);   // clear draft writes

                // (2) verify [id_last, draft...] on target at positions pos..pos+G
                std::vector<llama_token> seq; seq.push_back(id_last);
                for (int g = 0; g < G; ++g) seq.push_back(draft[g]);
                llama_batch vb = llama_batch_init((int)seq.size(), 0, 1);
                for (int i = 0; i < (int) seq.size(); ++i) common_batch_add(vb, seq[i], pos + i, { 0 }, true);
                llama_decode(ctx_tgt, vb);

                int accepted = 0; llama_token new_tok = -1;
                {
                    std::uniform_real_distribution<float> uni(0.0f, 1.0f);
                    bool resampled = false;
                    for (int g = 0; g < G; ++g) {
                        Dist p_g = make_dist(llama_get_logits_ith(ctx_tgt, g), n_vocab, top_k, top_p, temp);
                        float px = p_g.p_of(draft[g]), qx = qd[g].p_of(draft[g]);
                        float acc = (qx > 0.0f) ? std::min(1.0f, px/qx) : 0.0f;
                        if (uni(rng) < acc) { accepted++; continue; }
                        Dist res; double s = 0.0;
                        for (size_t j = 0; j < p_g.ids.size(); ++j) {
                            float diff = p_g.ps[j] - qd[g].p_of(p_g.ids[j]);
                            if (diff > 0) { res.ids.push_back(p_g.ids[j]); res.ps.push_back(diff); s += diff; }
                        }
                        if (s <= 0) new_tok = p_g.ids[0];
                        else { for (float & pp : res.ps) pp = (float)(pp / s); new_tok = sample_dist(res, rng); }
                        resampled = true; break;
                    }
                    if (!resampled) {
                        Dist p_b = make_dist(llama_get_logits_ith(ctx_tgt, G), n_vocab, top_k, top_p, temp);
                        new_tok = (temp <= 0) ? p_b.ids[0] : sample_dist(p_b, rng);
                    }
                }
                n_accept += accepted;

                // hidden for next round = target hidden of last confirmed token (vb row `accepted`)
                std::memcpy(h_last.data(), llama_get_embeddings_nextn_ith(ctx_tgt, accepted), sizeof(float)*n_embd);
                int keep = pos + 1 + accepted;                                  // prefix + id_last + accepted drafts
                llama_memory_seq_rm(llama_get_memory(ctx_tgt), 0, keep, -1);

                bool eog = false;
                for (int g = 0; g < accepted && (int) out.size() < max_new; ++g) {
                    out.push_back(draft[g]);
                    if (llama_vocab_is_eog(vocab, draft[g])) { eog = true; break; }
                }
                out.push_back(new_tok);
                if (llama_vocab_is_eog(vocab, new_tok)) eog = true;
                id_last = new_tok;
                pos = keep;                                                    // new_tok not cached; decoded next round
                llama_batch_free(vb);
                if (eog || (int) out.size() >= max_new) break;
            }
        }

        double dt = (ggml_time_us() - t0) / 1e6;
        std::string text;
        for (auto t : out) text += common_token_to_piece(ctx_tgt, t);
        printf("\n=== PROMPT %zu BEGIN ===\n", pi);
        printf("decoded %d tokens in %.3fs = %.2f tok/s\n", (int) out.size(), dt, out.size()/dt);
        printf("n_drafted %d\n", n_drafted);
        printf("n_accept %d\n", n_accept);
        printf("TEXT: %s\n", text.c_str());
        printf("=== PROMPT %zu END ===\n", pi);
        fflush(stdout);
    }

    if (ctx_dft) { if (dbatch.token) { free(dbatch.token); dbatch.token = nullptr; } llama_batch_free(dbatch);
                   llama_free(ctx_dft); llama_model_free(model_dft); }
    llama_backend_free();
    return 0;
}
