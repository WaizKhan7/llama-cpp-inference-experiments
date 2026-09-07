#include "llama.h"
#include "ggml-backend.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

struct params {
    std::string model;
    std::string prompt_file;
    int context_tokens = 0;
    int ctx_size = 0;
    int predict = 50;
    int gpu_layers = 99;
    int batch_size = 0;
    int ubatch_size = 0;
    int warmup = 0;
    int runs = 1;
    llama_flash_attn_type flash_attn = LLAMA_FLASH_ATTN_TYPE_ENABLED;
};

struct result {
    std::vector<llama_token> ids;
    double decode_ms = 0.0;
    int decode_steps = 0;
};

static void usage(const char * p) {
    std::fprintf(stderr,
        "Usage: %s --model MODEL.gguf --prompt-file PROMPT.txt [options]\n"
        "  --context-tokens N  exact prompt token count (padding is token-level)\n"
        "  --ctx-size N        allocated context capacity; independent of prompt length\n"
        "  --predict N         generated greedy tokens; default 50\n"
        "  --gpu-layers N      GPU-offloaded layers; default 99\n"
        "  --batch-size N      logical prompt batch size; default: prompt length\n"
        "  --ubatch-size N     physical prompt batch size; default: prompt length\n"
        "  --warmup N          untimed repeats; default 0\n"
        "  --runs N            timed repeats; default 1\n"
        "  --flash-attn on|off default on\n", p);
}

static bool positive(const char * s, int & dst, bool allow_zero = false) {
    char * end = nullptr;
    const long n = std::strtol(s, &end, 10);
    if (end == s || *end != '\0' || n < (allow_zero ? 0 : 1) ||
        n > std::numeric_limits<int>::max()) return false;
    dst = static_cast<int>(n);
    return true;
}

static bool parse(int argc, char ** argv, params & p) {
    for (int i = 1; i < argc; ++i) {
        const char * a = argv[i];
        if (std::strcmp(a, "--model") == 0 && ++i < argc) p.model = argv[i];
        else if (std::strcmp(a, "--prompt-file") == 0 && ++i < argc) p.prompt_file = argv[i];
        else if (std::strcmp(a, "--context-tokens") == 0 && ++i < argc) { if (!positive(argv[i], p.context_tokens)) return false; }
        else if (std::strcmp(a, "--ctx-size") == 0 && ++i < argc) { if (!positive(argv[i], p.ctx_size)) return false; }
        else if (std::strcmp(a, "--predict") == 0 && ++i < argc) { if (!positive(argv[i], p.predict)) return false; }
        else if (std::strcmp(a, "--gpu-layers") == 0 && ++i < argc) { if (!positive(argv[i], p.gpu_layers, true)) return false; }
        else if (std::strcmp(a, "--batch-size") == 0 && ++i < argc) { if (!positive(argv[i], p.batch_size)) return false; }
        else if (std::strcmp(a, "--ubatch-size") == 0 && ++i < argc) { if (!positive(argv[i], p.ubatch_size)) return false; }
        else if (std::strcmp(a, "--warmup") == 0 && ++i < argc) { if (!positive(argv[i], p.warmup, true)) return false; }
        else if (std::strcmp(a, "--runs") == 0 && ++i < argc) { if (!positive(argv[i], p.runs)) return false; }
        else if (std::strcmp(a, "--flash-attn") == 0 && ++i < argc) {
            if (std::strcmp(argv[i], "on") == 0) p.flash_attn = LLAMA_FLASH_ATTN_TYPE_ENABLED;
            else if (std::strcmp(argv[i], "off") == 0) p.flash_attn = LLAMA_FLASH_ATTN_TYPE_DISABLED;
            else return false;
        } else return false;
    }
    return !p.model.empty() && !p.prompt_file.empty();
}

static bool read_text(const std::string & path, std::string & text) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    text.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
    return true;
}

static bool tokenize(const llama_vocab * vocab, const std::string & text,
                     bool add_special, std::vector<llama_token> & tokens) {
    const int n = -llama_tokenize(vocab, text.data(), (int32_t) text.size(),
                                  nullptr, 0, add_special, true);
    if (n <= 0) return false;
    tokens.resize(n);
    return llama_tokenize(vocab, text.data(), (int32_t) text.size(), tokens.data(),
                          (int32_t) tokens.size(), add_special, true) >= 0;
}

static bool exact_prompt(const llama_vocab * vocab, const std::string & text,
                         int requested, std::vector<llama_token> & prompt) {
    if (!tokenize(vocab, text, true, prompt)) return false;
    if (!requested) return true;
    if ((int) prompt.size() > requested) {
        std::fprintf(stderr, "Prompt has %zu tokens; target is %d.\n", prompt.size(), requested);
        return false;
    }
    std::vector<llama_token> filler;
    if (!tokenize(vocab, " GPU memory", false, filler) || filler.empty()) return false;
    for (size_t i = 0; (int) prompt.size() < requested; ++i) {
        prompt.push_back(filler[i % filler.size()]);
    }
    return true;
}

static bool generate(llama_context * ctx, const llama_vocab * vocab,
                     const std::vector<llama_token> & prompt, int n_predict,
                     bool timed, result & out) {
    llama_memory_clear(llama_get_memory(ctx), false);
    llama_sampler * sampler = llama_sampler_init_greedy();
    if (!sampler) return false;

    llama_batch batch = llama_batch_get_one(
        const_cast<llama_token *>(prompt.data()), (int32_t) prompt.size());
    if (llama_decode(ctx, batch) != 0) {
        llama_sampler_free(sampler);
        return false;
    }

    llama_token token = llama_sampler_sample(sampler, ctx, -1);
    if (!llama_vocab_is_eog(vocab, token)) {
        out.ids.push_back(token);
        llama_sampler_accept(sampler, token);
        batch = llama_batch_get_one(&token, 1);
    }

    const auto start = std::chrono::steady_clock::now();
    while ((int) out.ids.size() < n_predict) {
        if (llama_decode(ctx, batch) != 0) {
            llama_sampler_free(sampler);
            return false;
        }
        ++out.decode_steps;
        token = llama_sampler_sample(sampler, ctx, -1);
        if (llama_vocab_is_eog(vocab, token)) break;
        out.ids.push_back(token);
        llama_sampler_accept(sampler, token);
        batch = llama_batch_get_one(&token, 1);
    }
    if (timed) {
        out.decode_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count();
    }
    llama_sampler_free(sampler);
    return true;
}

int main(int argc, char ** argv) {
    params p;
    if (!parse(argc, argv, p)) {
        usage(argv[0]);
        return 2;
    }

    std::string prompt_text;
    if (!read_text(p.prompt_file, prompt_text)) {
        std::fprintf(stderr, "Could not read prompt: %s\n", p.prompt_file.c_str());
        return 2;
    }

    llama_backend_init();
    ggml_backend_load_all();

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = p.gpu_layers;
    llama_model * model = llama_model_load_from_file(p.model.c_str(), mp);
    if (!model) {
        std::fprintf(stderr, "Could not load model: %s\n", p.model.c_str());
        llama_backend_free();
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    std::vector<llama_token> prompt;
    if (!exact_prompt(vocab, prompt_text, p.context_tokens, prompt)) {
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    const int minimum_ctx = static_cast<int>(prompt.size()) + p.predict;
    if (p.ctx_size != 0 && p.ctx_size < minimum_ctx) {
        std::fprintf(stderr, "Requested context capacity %d is smaller than prompt plus generation (%d).\n", p.ctx_size, minimum_ctx);
        llama_model_free(model);
        llama_backend_free();
        return 2;
    }

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = static_cast<uint32_t>(p.ctx_size != 0 ? p.ctx_size : minimum_ctx);
    cp.n_batch = static_cast<uint32_t>(p.batch_size != 0 ? p.batch_size : prompt.size());
    cp.n_ubatch = static_cast<uint32_t>(p.ubatch_size != 0 ? p.ubatch_size : prompt.size());
    cp.flash_attn_type = p.flash_attn;
    cp.no_perf = true;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) {
        std::fprintf(stderr, "Could not create context.\n");
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    for (int i = 0; i < p.warmup; ++i) {
        result warm;
        if (!generate(ctx, vocab, prompt, p.predict, false, warm)) {
            std::fprintf(stderr, "Warmup failed.\n");
            llama_free(ctx); llama_model_free(model); llama_backend_free();
            return 1;
        }
    }

    result first;
    std::vector<double> timings;
    for (int i = 0; i < p.runs; ++i) {
        result current;
        if (!generate(ctx, vocab, prompt, p.predict, true, current)) {
            std::fprintf(stderr, "Generation failed.\n");
            llama_free(ctx); llama_model_free(model); llama_backend_free();
            return 1;
        }
        if (i == 0) first = current;
        else if (current.ids != first.ids) {
            std::fprintf(stderr, "Greedy token IDs changed between runs.\n");
            llama_free(ctx); llama_model_free(model); llama_backend_free();
            return 1;
        }
        timings.push_back(current.decode_ms);
    }

    const double sum = std::accumulate(timings.begin(), timings.end(), 0.0);
    const double mean = sum / timings.size();
    const double tps = sum > 0.0
        ? (double) first.decode_steps * timings.size() / (sum / 1000.0) : 0.0;

    std::printf("PROMPT_TOKEN_COUNT=%zu\n", prompt.size());
    std::printf("CONTEXT_CAPACITY=%u\n", cp.n_ctx);
    std::printf("PROMPT_BATCH_SIZE=%u\n", cp.n_batch);
    std::printf("PROMPT_UBATCH_SIZE=%u\n", cp.n_ubatch);
    std::printf("GENERATED_TOKEN_COUNT=%zu\n", first.ids.size());
    std::printf("GENERATED_TOKEN_IDS=");
    for (size_t i = 0; i < first.ids.size(); ++i) {
        std::printf("%s%d", i ? "," : "", first.ids[i]);
    }
    std::printf("\nTIMED_DECODE_STEPS=%d\nTIMED_RUNS=%d\n", first.decode_steps, p.runs);
    std::printf("DECODE_MEAN_MS=%.6f\nDECODE_TOKENS_PER_SECOND=%.6f\n", mean, tps);

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return first.ids.size() == (size_t) p.predict ? 0 : 3;
}
