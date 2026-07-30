// asr_service.cpp — Automatic Speech Recognition service (SenseVoice.cpp).
//
// Wraps SenseVoice.cpp (ggml-based ASR) into a synchronous recognize() call.
// Audio format: 16kHz mono, passed as int16 samples.

#include "asr_service.h"

#include "sense-voice.h"
#include "silero-vad.h"  // full definition of silero_vad for proper cleanup

#include <cstdio>
#include <string>
#include <vector>

#include <ggml.h>

namespace winefox {
namespace world {

// Suppress SenseVoice's ggml log output (loaded by sense_voice_small_init).
// SenseVoice ships its own ggml fork that is statically linked into the exe,
// so it has a separate log state from llama.cpp's ggml (in winefox_core.dll).
// Without this, sense_voice prints verbose model-loading logs that flood
// the console during startup.
static void asr_silent_log_cb(ggml_log_level level, const char* text, void* /*user_data*/) {
    // Only forward genuine errors; swallow INFO/DEBUG/WARN.
    if (level >= GGML_LOG_LEVEL_ERROR) {
        std::fputs(text, stderr);
    }
}

bool AsrService::init(const std::string& model_path, const std::string& language,
                      int threads, bool use_itn, bool flash_attn) {
    // Store configuration up front so it is available even if init fails later.
    language_  = language.empty() ? std::string("auto") : language;
    threads_   = threads > 0 ? threads : 4;
    use_itn_   = use_itn;
    flash_attn_ = flash_attn;

    // Guard against empty model path — SenseVoice crashes on empty path.
    if (model_path.empty()) {
        std::fprintf(stderr, "[asr] model_path is empty\n");
        return false;
    }

    // Silence SenseVoice's logs before model loading. SenseVoice has TWO
    // independent log systems: ggml's (via ggml_log_set) and its own
    // SENSE_VOICE_LOG_* macros (via sense_voice_log_set). Both must be
    // suppressed, otherwise model-loading info floods the console.
    ggml_log_set(asr_silent_log_cb, nullptr);
    sense_voice_log_set(asr_silent_log_cb, nullptr);

    // Build SenseVoice context params. GPU is disabled — this service runs on
    // the CPU for portability inside the world process.
    sense_voice_context_params cparams = sense_voice_context_default_params();
    cparams.use_gpu    = false;
    cparams.flash_attn = flash_attn_;
    cparams.use_itn    = use_itn_;

    sense_voice_context* ctx = sense_voice_small_init_from_file_with_params(
        model_path.c_str(), cparams);
    if (ctx == nullptr) {
        return false;
    }

    // Pin the language on the context when an explicit language is requested.
    // "auto" leaves language_id at 0 so SenseVoice auto-detects.
    if (language_ != "auto") {
        int lid = sense_voice_lang_id(language_.c_str());
        if (lid < 0) {
            // Unknown language code — release what we allocated and bail out.
            sense_voice_free_state(ctx->state);
            delete ctx->model.model->encoder;
            delete ctx->model.model;
            delete ctx->vad_model.model;
            delete ctx;
            return false;
        }
        ctx->language_id = lid;
    }

    ctx_ = ctx;
    return true;
}

std::string AsrService::recognize(const int16_t* samples, int n) {
    sense_voice_context* ctx = reinterpret_cast<sense_voice_context*>(ctx_);
    if (ctx == nullptr || samples == nullptr || n <= 0) {
        return std::string();
    }

    // SenseVoice expects int16 PCM normalized to double in [-1.0, 1.0].
    std::vector<double> pcmf32;
    pcmf32.resize(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        pcmf32[static_cast<size_t>(i)] = double(samples[i]) / 32768.0;
    }

    // Greedy decoding, single segment — we feed one utterance per call.
    sense_voice_full_params wparams = sense_voice_full_default_params(
        SENSE_VOICE_SAMPLING_GREEDY);
    wparams.language       = language_.c_str();
    wparams.n_threads      = threads_;
    wparams.print_progress = false;
    wparams.no_timestamps  = true;
    wparams.single_segment = true;

    int rc = sense_voice_full_parallel(ctx, wparams, pcmf32,
                                       int(pcmf32.size()), 1);
    if (rc != 0) {
        return std::string();
    }

    // Extract the transcript text.
    //
    // SenseVoice emits 4 prefix tokens before the actual transcript:
    //   [0] language  (e.g. <|zh|>)
    //   [1] emotion   (e.g. <|NEUTRAL|>)
    //   [2] event     (e.g. <|Speech|>)
    //   [3] itn       (e.g. <|withitn|> or <|noitn|>)
    // We skip these and only emit the transcript. Duplicate consecutive
    // tokens are collapsed (matches sense_voice_print_output behavior)
    // and the padding token (id==0) is skipped.
    std::string text;
    const auto& ids = ctx->state->ids;
    const size_t start = 4;  // skip the 4 prefix tokens
    for (size_t i = start; i < ids.size(); ++i) {
        int id = ids[i];
        if (i > 0 && id == ids[i - 1]) continue;  // collapse duplicates
        if (id == 0) continue;                     // skip padding
        auto it = ctx->vocab.id_to_token.find(id);
        if (it != ctx->vocab.id_to_token.end()) {
            text += it->second;
        }
    }
    return text;
}

void AsrService::close() {
    sense_voice_context* ctx = reinterpret_cast<sense_voice_context*>(ctx_);
    if (ctx == nullptr) {
        return;
    }
    sense_voice_free_state(ctx->state);
    delete ctx->model.model->encoder;
    delete ctx->model.model;
    delete ctx->vad_model.model;
    delete ctx;
    ctx_ = nullptr;
}

AsrService::~AsrService() {
    close();
}

} // namespace world
} // namespace winefox
