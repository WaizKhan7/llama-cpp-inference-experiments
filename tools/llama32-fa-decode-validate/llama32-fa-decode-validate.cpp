#include "llama.h"
#include "ggml-backend.h"

#include <chrono>
#include <cmath>
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
    bool trace_logits = false;
    std::string teacher_token_ids;
    int score_prefill_tokens = 0;
    int score_tokens = 0;
    int score_start_token = 0;
    llama_flash_attn_type flash_attn = LLAMA_FLASH_ATTN_TYPE_ENABLED;
};

struct logit_record {
    int step = 0;
    llama_token top1_id = -1;
    llama_token top2_id = -1;
    float top1_logit = -std::numeric_limits<float>::infinity();
    float top2_logit = -std::numeric_limits<float>::infinity();
    bool finite = true;
};

struct result {
    std::vector<llama_token> ids;
    std::vector<logit_record> logits;
    double decode_ms = 0.0;
    int decode_steps = 0;
    double negative_log_likelihood = 0.0;
    int scored_tokens = 0;
    bool scoring_finite = true;
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
        "  --trace-logits      print raw top-1/top-2 logits for diagnosis\n"
        "  --teacher-token-ids CSV  force this comma-separated token history\n"
        "  --score-prefill-tokens N  held-out tokens used for prefill\n"
        "  --score-tokens N   held-out tokens scored one at a time\n"
        "  --score-start-token N  held-out token index of the first scored token\n"
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
        else if (std::strcmp(a, "--trace-logits") == 0) p.trace_logits = true;
        else if (std::strcmp(a, "--teacher-token-ids") == 0 && ++i < argc) p.teacher_token_ids = argv[i];
        else if (std::strcmp(a, "--score-prefill-tokens") == 0 && ++i < argc) { if (!positive(argv[i], p.score_prefill_tokens)) return false; }
        else if (std::strcmp(a, "--score-tokens") == 0 && ++i < argc) { if (!positive(argv[i], p.score_tokens)) return false; }
        else if (std::strcmp(a, "--score-start-token") == 0 && ++i < argc) { if (!positive(argv[i], p.score_start_token)) return false; }
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

    // Match llama-cli --file: remove one final newline before tokenization.
    if (!text.empty() && text.back() == char(10)) {
        text.pop_back();
    }
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

static bool parse_teacher_token_ids(
        const std::string & text, const llama_vocab * vocab,
        std::vector<llama_token> & tokens) {
    if (text.empty()) return true;
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    size_t begin = 0;
    while (begin < text.size()) {
        const size_t end = text.find(',', begin);
        const std::string item = text.substr(begin, end == std::string::npos
            ? std::string::npos : end - begin);
        char * parsed_end = nullptr;
        const long value = std::strtol(item.c_str(), &parsed_end, 10);
        if (item.empty() || parsed_end == item.c_str() || *parsed_end != '\0' ||
            value < 0 || value >= n_vocab) return false;
        tokens.push_back(static_cast<llama_token>(value));
        if (end == std::string::npos) break;
        begin = end + 1;
    }
    return !tokens.empty();
}

static bool record_logits(llama_context * ctx, const llama_vocab * vocab,
                          int step, result & out) {
    const float * values = llama_get_logits_ith(ctx, -1);
    if (!values) return false;

    logit_record record;
    record.step = step;
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);

    for (int32_t id = 0; id < n_vocab; ++id) {
        const float value = values[id];
        record.finite = record.finite && std::isfinite(value);
        if (value > record.top1_logit) {
            record.top2_logit = record.top1_logit;
            record.top2_id = record.top1_id;
            record.top1_logit = value;
            record.top1_id = id;
        } else if (value > record.top2_logit) {
            record.top2_logit = value;
            record.top2_id = id;
        }
    }

    out.logits.push_back(record);
    return record.top1_id >= 0 && record.top2_id >= 0;
}

static bool score_target_token(
        llama_context * ctx, const llama_vocab * vocab, llama_token target,
        result & out) {
    const float * logits = llama_get_logits_ith(ctx, -1);
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    if (!logits || target < 0 || target >= n_vocab) return false;

    double maximum = -std::numeric_limits<double>::infinity();
    for (int32_t id = 0; id < n_vocab; ++id) {
        const double value = static_cast<double>(logits[id]);
        if (!std::isfinite(value)) {
            out.scoring_finite = false;
            return false;
        }
        maximum = std::fmax(maximum, value);
    }

    double denominator = 0.0;
    for (int32_t id = 0; id < n_vocab; ++id) {
        denominator += std::exp(static_cast<double>(logits[id]) - maximum);
    }
    if (!std::isfinite(denominator) || denominator <= 0.0) {
        out.scoring_finite = false;
        return false;
    }

    out.negative_log_likelihood +=
        maximum + std::log(denominator) - static_cast<double>(logits[target]);
    ++out.scored_tokens;
    return std::isfinite(out.negative_log_likelihood);
}

static bool generate(llama_context * ctx, const llama_vocab * vocab,
                     const std::vector<llama_token> & prompt, int n_predict,
                     bool trace_logits, const std::vector<llama_token> * teacher_tokens,
                     bool score_teacher_tokens, bool timed, result & out) {
    llama_memory_clear(llama_get_memory(ctx), false);
    llama_sampler * sampler = teacher_tokens == nullptr
        ? llama_sampler_init_greedy() : nullptr;
    if (teacher_tokens == nullptr && !sampler) return false;

    llama_batch batch = llama_batch_get_one(
        const_cast<llama_token *>(prompt.data()), (int32_t) prompt.size());
    if (llama_decode(ctx, batch) != 0) {
        if (sampler) llama_sampler_free(sampler);
        return false;
    }
    if (trace_logits && !record_logits(ctx, vocab, 0, out)) {
        if (sampler) llama_sampler_free(sampler);
        return false;
    }
    if (score_teacher_tokens &&
        (teacher_tokens == nullptr || teacher_tokens->empty() ||
         !score_target_token(ctx, vocab, (*teacher_tokens)[0], out))) {
        if (sampler) llama_sampler_free(sampler);
        return false;
    }

    const auto start = std::chrono::steady_clock::now();
    if (teacher_tokens != nullptr) {
        for (size_t step = 0; step < teacher_tokens->size(); ++step) {
            llama_token token = (*teacher_tokens)[step];
            out.ids.push_back(token);
            if (step + 1 == teacher_tokens->size()) break;
            batch = llama_batch_get_one(&token, 1);
            if (llama_decode(ctx, batch) != 0) return false;
            ++out.decode_steps;
            if (score_teacher_tokens &&
                !score_target_token(ctx, vocab, (*teacher_tokens)[step + 1], out)) {
                return false;
            }
            if (trace_logits && !record_logits(
                    ctx, vocab, static_cast<int>(step + 1), out)) return false;
        }
    } else {
        llama_token token = llama_sampler_sample(sampler, ctx, -1);
        if (!llama_vocab_is_eog(vocab, token)) {
            out.ids.push_back(token);
            llama_sampler_accept(sampler, token);
            batch = llama_batch_get_one(&token, 1);
        }
        while ((int) out.ids.size() < n_predict) {
            if (llama_decode(ctx, batch) != 0) {
                llama_sampler_free(sampler);
                return false;
            }
            ++out.decode_steps;
            if (trace_logits && !record_logits(
                    ctx, vocab, static_cast<int>(out.ids.size()), out)) {
                llama_sampler_free(sampler);
                return false;
            }
            token = llama_sampler_sample(sampler, ctx, -1);
            if (llama_vocab_is_eog(vocab, token)) break;
            out.ids.push_back(token);
            llama_sampler_accept(sampler, token);
            batch = llama_batch_get_one(&token, 1);
        }
    }
    if (timed) out.decode_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
    if (sampler) llama_sampler_free(sampler);
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
    const bool scoring_mode =
        p.score_prefill_tokens != 0 || p.score_tokens != 0 ||
        p.score_start_token != 0;
    if (scoring_mode &&
        (p.score_prefill_tokens == 0 || p.score_tokens == 0 ||
         p.score_start_token == 0 || p.context_tokens != 0 ||
         !p.teacher_token_ids.empty())) {
        std::fprintf(stderr,
            "Scoring mode requires prefill length, score length, score start, and no prompt padding or teacher IDs.\n");
        llama_model_free(model);
        llama_backend_free();
        return 2;
    }

    std::vector<llama_token> prompt;
    std::vector<llama_token> teacher_tokens;
    int sequence_tokens = p.predict;

    if (scoring_mode) {
        std::vector<llama_token> heldout_tokens;
        if (!tokenize(vocab, prompt_text, true, heldout_tokens) ||
            p.score_start_token < p.score_prefill_tokens ||
            static_cast<int>(heldout_tokens.size()) <
                p.score_start_token + p.score_tokens) {
            std::fprintf(stderr,
                "Held-out text or score start cannot supply the requested scoring window.\n");
            llama_model_free(model);
            llama_backend_free();
            return 2;
        }
        prompt.assign(
            heldout_tokens.begin() + p.score_start_token - p.score_prefill_tokens,
            heldout_tokens.begin() + p.score_start_token);
        teacher_tokens.assign(
            heldout_tokens.begin() + p.score_start_token,
            heldout_tokens.begin() + p.score_start_token + p.score_tokens);
        sequence_tokens = p.score_tokens;
    } else {
        if (!exact_prompt(vocab, prompt_text, p.context_tokens, prompt)) {
            llama_model_free(model);
            llama_backend_free();
            return 1;
        }
        if (!parse_teacher_token_ids(p.teacher_token_ids, vocab, teacher_tokens)) {
            std::fprintf(stderr, "Invalid --teacher-token-ids value.\n");
            llama_model_free(model);
            llama_backend_free();
            return 2;
        }
        if (!teacher_tokens.empty() &&
            static_cast<int>(teacher_tokens.size()) != sequence_tokens) {
            std::fprintf(stderr, "Teacher token count must equal --predict.\n");
            llama_model_free(model);
            llama_backend_free();
            return 2;
        }
    }

    const int minimum_ctx = static_cast<int>(prompt.size()) + sequence_tokens;
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
        if (!generate(ctx, vocab, prompt, sequence_tokens, p.trace_logits,
                      teacher_tokens.empty() ? nullptr : &teacher_tokens,
                      scoring_mode, false, warm)) {
            std::fprintf(stderr, "Warmup failed.\n");
            llama_free(ctx); llama_model_free(model); llama_backend_free();
            return 1;
        }
    }

    result first;
    std::vector<double> timings;
    for (int i = 0; i < p.runs; ++i) {
        result current;
        if (!generate(ctx, vocab, prompt, sequence_tokens, p.trace_logits,
                      teacher_tokens.empty() ? nullptr : &teacher_tokens,
                      scoring_mode, true, current)) {
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
    std::printf("\n");
    for (const logit_record & record : first.logits) {
        std::printf(
            "LOGIT_STEP=%d,TOP1_ID=%d,TOP1_LOGIT=%.9g,"
            "TOP2_ID=%d,TOP2_LOGIT=%.9g,MARGIN=%.9g,FINITE=%d\n",
            record.step, record.top1_id, record.top1_logit,
            record.top2_id, record.top2_logit,
            record.top1_logit - record.top2_logit,
            record.finite ? 1 : 0);
    }
    if (scoring_mode) {
        std::printf("SCORE_START_TOKEN=%d\n", p.score_start_token);
        const double mean_nll = first.negative_log_likelihood / first.scored_tokens;
        std::printf("SCORED_TOKEN_COUNT=%d\n", first.scored_tokens);
        std::printf("NEGATIVE_LOG_LIKELIHOOD=%.12g\n", first.negative_log_likelihood);
        std::printf("MEAN_NEGATIVE_LOG_LIKELIHOOD=%.12g\n", mean_nll);
        std::printf("PERPLEXITY=%.12g\n", std::exp(mean_nll));
        std::printf("SCORING_FINITE=%d\n", first.scoring_finite ? 1 : 0);
    }
    std::printf("TIMED_DECODE_STEPS=%d\nTIMED_RUNS=%d\n", first.decode_steps, p.runs);
    std::printf("DECODE_MEAN_MS=%.6f\nDECODE_TOKENS_PER_SECOND=%.6f\n", mean, tps);

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return first.ids.size() == static_cast<size_t>(sequence_tokens) &&
        (!scoring_mode || first.scored_tokens == sequence_tokens && first.scoring_finite)
        ? 0 : 3;
}
