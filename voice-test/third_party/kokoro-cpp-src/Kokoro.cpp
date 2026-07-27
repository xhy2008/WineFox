#include "Kokoro.h"
#include "Tokenizer.h"
#include <fstream>
#include <sstream>
#include <regex>
#include <numeric>
#include <cstring>
#include <chrono>
#include <thread>

namespace {

// Resolve thread count: 0 = auto, >0 = explicit.
int resolve_threads(int n) {
    if (n > 0) return n;
    int hw = static_cast<int>(std::thread::hardware_concurrency());
    if (hw <= 0) return 4;
    // Use logical cores, clamped to [4, 8].
    //
    // Benchmark on dev machine (4C8T, INT8-static decoder, 15.65s audio):
    //   t=4 -> RTF 0.75 (1.33x realtime)
    //   t=6 -> RTF 0.65 (1.53x realtime)  <- best
    //   t=8 -> RTF 0.67 (1.50x realtime)
    //
    // Unlike ggml (where SMT siblings hurt due to per-node barrier
    // overhead), ORT's thread pool benefits from SMT on the conv-heavy
    // decoder because conv ops parallelize well and ORT's work stealing
    // absorbs the SMT contention. So we do NOT divide by 2 here.
    if (hw < 4) return hw;
    if (hw > 8) return 8;
    return hw;
}

// Build a SessionOptions with the given thread count and full graph opt.
Ort::SessionOptions make_session_options(int intra_op_threads) {
    Ort::SessionOptions so;
    so.SetIntraOpNumThreads(intra_op_threads);
    so.SetInterOpNumThreads(1);
    so.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_ALL);
    // Enable memory pattern & arena. Even though input shapes vary with
    // text length, ORT's arena allocator still wins on net by reusing
    // small-block allocations across runs (measured ~30% slower if
    // disabled).
    so.EnableCpuMemArena();
    so.EnableMemPattern();
    return so;
}

#ifdef _WIN32
// Helper: open an ONNX session from a UTF-8 path on Windows.
Ort::Session open_session(Ort::Env& env, const std::string& path,
                          const Ort::SessionOptions& so) {
    std::wstring wpath(path.begin(), path.end());
    return Ort::Session(env, wpath.c_str(), so);
}
#else
Ort::Session open_session(Ort::Env& env, const std::string& path,
                          const Ort::SessionOptions& so) {
    return Ort::Session(env, path.c_str(), so);
}
#endif

}  // namespace

// Helper to trim audio (simple amplitude based silence removal)
std::vector<float> trim_audio(const std::vector<float>& audio, int sample_rate, float threshold_db = 60.0f) {
    return audio;
}

// Static input/output names for ORT sessions. Defined once here so each
// _create_audio_split call does not reconstruct char* arrays on the stack.
const char* Kokoro::kEncInputNames[3]  = {"input_ids", "ref_s", "speed"};
const char* Kokoro::kEncOutputNames[5] = {"asr", "F0_pred", "N_pred", "style_dec", "pred_dur"};
const char* Kokoro::kDecInputNames[4]  = {"asr", "F0_pred", "N_pred", "style_dec"};
const char* Kokoro::kDecOutputNames[1] = {"audio"};

// ---------------------------------------------------------------------------
// Split-mode constructor (encoder FP32 + decoder INT8/FP32, multi-threaded)
// ---------------------------------------------------------------------------
Kokoro::Kokoro(const std::string& encoder_path,
               const std::string& decoder_path,
               const std::string& voices_path,
               const std::string& vocab_path,
               int n_threads)
    : env_(ORT_LOGGING_LEVEL_WARNING, "Kokoro")
    , split_mode_(true)
    , n_threads_(resolve_threads(n_threads))
{
    // Encoder: BERT + LSTM, precision-sensitive, FP32.
    // Use fewer threads (BERT parallelism is limited); cap at 4.
    int enc_threads = std::min(n_threads_, 4);
    auto enc_so = make_session_options(enc_threads);
    enc_session_ = open_session(env_, encoder_path, enc_so);

    // Decoder: conv-heavy, parallelizes well, INT8 quantized.
    // Use full thread budget.
    auto dec_so = make_session_options(n_threads_);
    dec_session_ = open_session(env_, decoder_path, dec_so);

    load_voices(voices_path);

    // Load vocab + tokenizer (shared init)
    std::map<std::string, int> vocab;
    std::ifstream in(vocab_path);
    if (in.is_open()) {
        std::string line;
        while (std::getline(in, line)) {
            size_t tab = line.find('\t');
            if (tab != std::string::npos) {
                std::string token = line.substr(0, tab);
                std::string id_str = line.substr(tab + 1);
                size_t pos = 0;
                while((pos = token.find("\\n", pos)) != std::string::npos) { token.replace(pos, 2, "\n"); pos += 1; }
                pos = 0;
                while((pos = token.find("\\r", pos)) != std::string::npos) { token.replace(pos, 2, "\r"); pos += 1; }
                pos = 0;
                while((pos = token.find("\\t", pos)) != std::string::npos) { token.replace(pos, 2, "\t"); pos += 1; }
                try { vocab[token] = std::stoi(id_str); } catch (...) {}
            }
        }
        std::cout << "Loaded " << vocab.size() << " tokens from " << vocab_path << std::endl;
    } else {
        std::cerr << "Warning: Failed to open vocab file: " << vocab_path << std::endl;
    }
    tokenizer_ = std::make_unique<Tokenizer>(TokenizerConfig{}, vocab);

    std::cout << "Kokoro split mode: enc_threads=" << enc_threads
              << " dec_threads=" << n_threads_ << std::endl;

    // Warmup: run a short dummy forward through encoder + decoder so that
    // ORT compiles the execution plan, primes the memory arena, and JITs
    // any shape-specific kernels before the first user-facing call. The
    // first run is typically 2-3x slower than subsequent runs on the same
    // shape due to plan compilation; this hides that latency at load time.
    warmup_split();
}

// ---------------------------------------------------------------------------
// Merged-mode constructor (legacy, single-threaded for backwards compat)
// ---------------------------------------------------------------------------
Kokoro::Kokoro(const std::string& model_path, const std::string& voices_path, const std::string& vocab_path)
    : env_(ORT_LOGGING_LEVEL_WARNING, "Kokoro")
    , split_mode_(false)
    , n_threads_(1)
{
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    merged_session_ = open_session(env_, model_path, session_options);

    load_voices(voices_path);

    std::map<std::string, int> vocab;
    std::ifstream in(vocab_path);
    if (in.is_open()) {
        std::string line;
        while (std::getline(in, line)) {
            size_t tab = line.find('\t');
            if (tab != std::string::npos) {
                std::string token = line.substr(0, tab);
                std::string id_str = line.substr(tab + 1);
                size_t pos = 0;
                while((pos = token.find("\\n", pos)) != std::string::npos) { token.replace(pos, 2, "\n"); pos += 1; }
                pos = 0;
                while((pos = token.find("\\r", pos)) != std::string::npos) { token.replace(pos, 2, "\r"); pos += 1; }
                pos = 0;
                while((pos = token.find("\\t", pos)) != std::string::npos) { token.replace(pos, 2, "\t"); pos += 1; }
                try { vocab[token] = std::stoi(id_str); } catch (...) {}
            }
        }
        std::cout << "Loaded " << vocab.size() << " tokens from " << vocab_path << std::endl;
    } else {
        std::cerr << "Warning: Failed to open vocab file: " << vocab_path << std::endl;
    }
    tokenizer_ = std::make_unique<Tokenizer>(TokenizerConfig{}, vocab);
}

Kokoro::~Kokoro() {}

void Kokoro::load_voices(const std::string& voices_path) {
    std::ifstream in(voices_path, std::ios::binary);
    if (!in.is_open()) {
        std::cerr << "Failed to open voices file: " << voices_path << std::endl;
        return;
    }

    char magic[4];
    in.read(magic, 4);
    if (std::strncmp(magic, "VOIC", 4) != 0) {
        std::cerr << "Invalid voices file format (Magic header mismatch). Expected 'VOIC'." << std::endl;
        std::cerr << "Please run scripts/export_voices.py to convert voices.npy to voices.bin" << std::endl;
        return;
    }

    uint32_t version;
    in.read(reinterpret_cast<char*>(&version), 4);
    if (version != 1) {
        std::cerr << "Unsupported voices file version: " << version << std::endl;
        return;
    }

    uint32_t num_voices;
    in.read(reinterpret_cast<char*>(&num_voices), 4);

    for (uint32_t i = 0; i < num_voices; ++i) {
        uint32_t name_len;
        in.read(reinterpret_cast<char*>(&name_len), 4);

        std::string name(name_len, '\0');
        in.read(&name[0], name_len);

        uint32_t dim;
        in.read(reinterpret_cast<char*>(&dim), 4);

        std::vector<float> style(dim);
        in.read(reinterpret_cast<char*>(style.data()), dim * sizeof(float));

        voices_[name] = style;
    }

    std::cout << "Loaded " << voices_.size() << " voices from " << voices_path << std::endl;
}

std::vector<float> Kokoro::get_voice_style(const std::string& name) {
    if (voices_.find(name) != voices_.end()) {
        return voices_.at(name);
    }
    std::cerr << "Voice " << name << " not found. Using default." << std::endl;
    if (!voices_.empty()) return voices_.begin()->second;
    return std::vector<float>(256, 0.0f);
}

std::vector<std::string> Kokoro::_split_phonemes(const std::string& phonemes) {
    // Split phonemes into batches for streaming synthesis.
    //
    // Strategy: tokenize by punctuation. Sentence-ending punctuation
    // (.!?) terminates the current batch immediately (low-TTFB streaming).
    // Clause punctuation (,;) stays in the current batch. If a batch
    // exceeds MAX_PHONEME_LENGTH it is flushed early to respect the
    // model's context limit (510 tokens).
    std::vector<std::string> batches;
    std::regex re("([.,!?;])");
    std::sregex_token_iterator it(phonemes.begin(), phonemes.end(), re, {-1, 0});
    std::sregex_token_iterator end;

    std::string current_batch;

    auto flush = [&]() {
        if (!current_batch.empty()) {
            batches.push_back(current_batch);
            current_batch.clear();
        }
    };

    for (; it != end; ++it) {
        std::string part = *it;
        part = std::regex_replace(part, std::regex("^\\s+|\\s+$"), "");

        if (part.empty()) continue;

        bool is_punct = part.size() == 1 &&
                        std::string(".,!?;").find(part[0]) != std::string::npos;
        bool is_sentence_end = is_punct &&
                               std::string(".!?").find(part[0]) != std::string::npos;

        if (is_punct) {
            current_batch += part;
            if (is_sentence_end) {
                flush();
            }
        } else {
            // Hard cap: if adding this part would exceed the model context,
            // flush first to avoid truncation.
            if (!current_batch.empty() &&
                current_batch.length() + 1 + part.length() >= MAX_PHONEME_LENGTH) {
                flush();
            }
            if (!current_batch.empty()) current_batch += " ";
            current_batch += part;
        }
    }
    flush();

    return batches;
}

// ---------------------------------------------------------------------------
// Warmup: run a minimal forward pass through encoder + decoder to let ORT
// compile shape-specific execution plans, prime the memory arena, and JIT
// kernels. The first inference on a given shape is 2-3x slower than steady
// state; calling this in the constructor hides that latency from the first
// user-facing synthesis.
// ---------------------------------------------------------------------------
void Kokoro::warmup_split() {
    if (!split_mode_ || !enc_session_ || !dec_session_) return;

    // Minimal input: 4 tokens [0, 1, 2, 0] (BOS + 2 phonemes + EOS).
    // Use stack-local buffers; this runs once at construction so reuse
    // is unnecessary. mem_info_ and kEnc/kDec names are reused.
    std::vector<int64_t> tokens = {0, 1, 2, 0};
    std::vector<int64_t> ids_shape = {1, (int64_t)tokens.size()};
    std::vector<float> ref_s(256, 0.0f);
    std::vector<int64_t> ref_s_shape = {1, 256};
    std::vector<float> speed_tensor = {1.0f};
    std::vector<int64_t> speed_shape = {1};

    std::vector<Ort::Value> enc_inputs;
    enc_inputs.reserve(3);
    enc_inputs.push_back(Ort::Value::CreateTensor<int64_t>(
        mem_info_, tokens.data(), tokens.size(),
        ids_shape.data(), ids_shape.size()));
    enc_inputs.push_back(Ort::Value::CreateTensor<float>(
        mem_info_, ref_s.data(), ref_s.size(),
        ref_s_shape.data(), ref_s_shape.size()));
    enc_inputs.push_back(Ort::Value::CreateTensor<float>(
        mem_info_, speed_tensor.data(), speed_tensor.size(),
        speed_shape.data(), speed_shape.size()));

    try {
        auto enc_outputs = enc_session_.Run(
            Ort::RunOptions{nullptr},
            kEncInputNames, enc_inputs.data(), enc_inputs.size(),
            kEncOutputNames, 5);

        std::vector<Ort::Value> dec_inputs;
        dec_inputs.reserve(4);
        for (int i = 0; i < 4; ++i) dec_inputs.push_back(std::move(enc_outputs[i]));
        dec_session_.Run(
            Ort::RunOptions{nullptr},
            kDecInputNames, dec_inputs.data(), dec_inputs.size(),
            kDecOutputNames, 1);
        std::cerr << "[warmup] OK" << std::endl;
    } catch (const Ort::Exception& e) {
        std::cerr << "[warmup] failed (non-fatal): " << e.what() << std::endl;
    }
}

// ---------------------------------------------------------------------------
// Split-mode: run encoder then decoder as separate ONNX sessions.
//   Encoder inputs:  input_ids [B, T_ids], ref_s [B, 256], speed [B]
//   Encoder outputs: asr [B, 512, T_frm], F0_pred [B, T_frm*2],
//                    N_pred [B, T_frm*2], style_dec [B, 128], pred_dur [B, T_ids]
//   Decoder inputs:  asr, F0_pred, N_pred, style_dec
//   Decoder output:  audio [B, T_audio]
// ---------------------------------------------------------------------------
std::pair<std::vector<float>, int> Kokoro::_create_audio_split(
    const std::string& phonemes,
    const std::vector<float>& voice,
    float speed
) {
    namespace chr = std::chrono;
    auto t0 = chr::steady_clock::now();

    std::string truncated_phonemes = phonemes;
    if (phonemes.length() > MAX_PHONEME_LENGTH) {
        truncated_phonemes = phonemes.substr(0, MAX_PHONEME_LENGTH);
    }

    std::vector<int> tokens_raw = tokenizer_->tokenize(truncated_phonemes);

    // Build token buffer: [BOS, ...tokens, EOS]. Reuse the member buffer to
    // avoid heap alloc on every chunk in streaming mode (capacity stabilizes
    // after the first long chunk).
    tokens_buf_.clear();
    tokens_buf_.reserve(tokens_raw.size() + 2);
    tokens_buf_.push_back(0);  // BOS
    for (int t : tokens_raw) tokens_buf_.push_back(static_cast<int64_t>(t));
    tokens_buf_.push_back(0);  // EOS

    // Prepare voice style: voices.bin stores a stack indexed by token length.
    // Reuse style_buf_ (always 256 floats, never reallocates after first call).
    constexpr int STYLE_DIM = 256;
    if (voice.size() > STYLE_DIM) {
         size_t index = tokens_raw.size();
         if (index * STYLE_DIM + STYLE_DIM <= voice.size()) {
             style_buf_.assign(voice.begin() + index * STYLE_DIM,
                               voice.begin() + (index + 1) * STYLE_DIM);
         } else {
             std::cerr << "Warning: Style index out of bounds. Using first style." << std::endl;
             style_buf_.assign(voice.begin(), voice.begin() + STYLE_DIM);
         }
    } else {
        style_buf_ = voice;
    }

    // Update shape vectors (capacity stable; only the ids length changes).
    ids_shape_[1] = static_cast<int64_t>(tokens_buf_.size());
    speed_buf_[0] = speed;

    std::vector<Ort::Value> enc_inputs;
    enc_inputs.reserve(3);
    enc_inputs.push_back(Ort::Value::CreateTensor<int64_t>(
        mem_info_, tokens_buf_.data(), tokens_buf_.size(),
        ids_shape_.data(), ids_shape_.size()));
    enc_inputs.push_back(Ort::Value::CreateTensor<float>(
        mem_info_, style_buf_.data(), style_buf_.size(),
        ref_shape_.data(), ref_shape_.size()));
    enc_inputs.push_back(Ort::Value::CreateTensor<float>(
        mem_info_, speed_buf_.data(), speed_buf_.size(),
        speed_shape_.data(), speed_shape_.size()));

    auto t_enc_start = chr::steady_clock::now();
    auto enc_outputs = enc_session_.Run(
        Ort::RunOptions{nullptr},
        kEncInputNames,
        enc_inputs.data(),
        enc_inputs.size(),
        kEncOutputNames,
        5
    );
    auto t_enc_end = chr::steady_clock::now();

    // Encoder outputs become decoder inputs (zero-copy: ORT holds the buffers).
    // asr:       [1, 512, T_frm]      float32
    // F0_pred:   [1, T_frm*2]         float32
    // N_pred:    [1, T_frm*2]         float32
    // style_dec: [1, 128]             float32
    std::vector<Ort::Value> dec_inputs;
    dec_inputs.reserve(4);
    dec_inputs.push_back(std::move(enc_outputs[0]));  // asr
    dec_inputs.push_back(std::move(enc_outputs[1]));  // F0_pred
    dec_inputs.push_back(std::move(enc_outputs[2]));  // N_pred
    dec_inputs.push_back(std::move(enc_outputs[3]));  // style_dec

    auto t_dec_start = chr::steady_clock::now();
    auto dec_outputs = dec_session_.Run(
        Ort::RunOptions{nullptr},
        kDecInputNames,
        dec_inputs.data(),
        dec_inputs.size(),
        kDecOutputNames,
        1
    );
    auto t_dec_end = chr::steady_clock::now();

    float* floatarr = dec_outputs[0].GetTensorMutableData<float>();
    size_t output_len = dec_outputs[0].GetTensorTypeAndShapeInfo().GetElementCount();

    std::vector<float> audio(floatarr, floatarr + output_len);

    auto ms = [](chr::steady_clock::time_point a, chr::steady_clock::time_point b) {
        return chr::duration<double, std::milli>(b - a).count();
    };
    std::fprintf(stderr,
        "[perf] tokens=%zu audio=%zuSR=%.2fs pre=%6.1fms enc=%6.1fms dec=%6.1fms total=%6.1fms\n",
        tokens_buf_.size(), output_len, output_len / (double)SAMPLE_RATE,
        ms(t0, t_enc_start), ms(t_enc_start, t_enc_end),
        ms(t_dec_start, t_dec_end), ms(t0, t_dec_end));

    return {audio, SAMPLE_RATE};
}

// ---------------------------------------------------------------------------
// Merged-mode: single ONNX session (legacy)
// ---------------------------------------------------------------------------
std::pair<std::vector<float>, int> Kokoro::_create_audio_merged(
    const std::string& phonemes,
    const std::vector<float>& voice,
    float speed
) {
    std::string truncated_phonemes = phonemes;
    if (phonemes.length() > MAX_PHONEME_LENGTH) {
        truncated_phonemes = phonemes.substr(0, MAX_PHONEME_LENGTH);
    }

    std::vector<int> tokens_raw = tokenizer_->tokenize(truncated_phonemes);

    std::vector<int64_t> tokens = {0};
    for (int t : tokens_raw) tokens.push_back(t);
    tokens.push_back(0);

    std::vector<int64_t> input_shape = {1, (int64_t)tokens.size()};

    const int STYLE_DIM = 256;
    std::vector<float> selected_style;
    if (voice.size() > STYLE_DIM) {
         size_t index = tokens_raw.size();
         if (index * STYLE_DIM + STYLE_DIM <= voice.size()) {
             auto start = voice.begin() + index * STYLE_DIM;
             selected_style.assign(start, start + STYLE_DIM);
         } else {
             std::cerr << "Warning: Style index out of bounds. Using first style." << std::endl;
             selected_style.assign(voice.begin(), voice.begin() + STYLE_DIM);
         }
    } else {
        selected_style = voice;
    }

    std::vector<int64_t> style_shape = {1, (int64_t)selected_style.size()};
    std::vector<int64_t> speed_shape = {1};
    std::vector<float> speed_tensor = {speed};

    const char* input_names[] = {"tokens", "style", "speed"};
    const char* input_names_new[] = {"input_ids", "style", "speed"};

    bool use_new_schema = false;
    size_t num_inputs = merged_session_.GetInputCount();
    for(size_t i=0; i<num_inputs; i++) {
        auto name_ptr = merged_session_.GetInputNameAllocated(i, allocator_);
        if (std::string(name_ptr.get()) == "input_ids") {
            use_new_schema = true;
        }
    }

    std::vector<const char*> inputs;
    if (use_new_schema) {
        inputs = {"input_ids", "style", "speed"};
    } else {
        inputs = {"tokens", "style", "speed"};
    }

    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    std::vector<Ort::Value> input_tensors;
    input_tensors.push_back(Ort::Value::CreateTensor<int64_t>(
        memory_info, tokens.data(), tokens.size(), input_shape.data(), input_shape.size()));
    input_tensors.push_back(Ort::Value::CreateTensor<float>(
        memory_info, selected_style.data(), selected_style.size(), style_shape.data(), style_shape.size()));

    if (use_new_schema) {
        static int speed_int = (int)speed;
        input_tensors.push_back(Ort::Value::CreateTensor<int>(
            memory_info, &speed_int, 1, speed_shape.data(), speed_shape.size()));
    } else {
        input_tensors.push_back(Ort::Value::CreateTensor<float>(
            memory_info, speed_tensor.data(), speed_tensor.size(), speed_shape.data(), speed_shape.size()));
    }

    auto out_name_ptr = merged_session_.GetOutputNameAllocated(0, allocator_);
    std::vector<const char*> output_names_vec = {out_name_ptr.get()};

    auto output_tensors = merged_session_.Run(
        Ort::RunOptions{nullptr},
        inputs.data(),
        input_tensors.data(),
        input_tensors.size(),
        output_names_vec.data(),
        1
    );

    float* floatarr = output_tensors[0].GetTensorMutableData<float>();
    size_t output_len = output_tensors[0].GetTensorTypeAndShapeInfo().GetElementCount();

    std::vector<float> audio(floatarr, floatarr + output_len);
    return {audio, SAMPLE_RATE};
}

// ---------------------------------------------------------------------------
// Non-streaming synthesis
// ---------------------------------------------------------------------------
std::pair<std::vector<float>, int> Kokoro::create(
    const std::string& text,
    const std::vector<float>& voice_style,
    float speed,
    bool is_phonemes,
    bool trim
) {
    std::string phonemes = text;
    if (!is_phonemes) {
        phonemes = tokenizer_->phonemize(text);
    }

    auto batched_phonemes = _split_phonemes(phonemes);
    std::vector<float> full_audio;

    for (const auto& batch : batched_phonemes) {
        auto [audio_part, sr] = _create_audio(batch, voice_style, speed);
        if (trim) {
            audio_part = trim_audio(audio_part, sr);
        }
        full_audio.insert(full_audio.end(), audio_part.begin(), audio_part.end());
    }

    return {full_audio, SAMPLE_RATE};
}

std::pair<std::vector<float>, int> Kokoro::create(
    const std::string& text,
    const std::string& voice_name,
    float speed,
    bool is_phonemes,
    bool trim
) {
    return create(text, get_voice_style(voice_name), speed, is_phonemes, trim);
}

// ---------------------------------------------------------------------------
// Streaming synthesis: emit audio per phoneme batch.
// ---------------------------------------------------------------------------
void Kokoro::create_stream(
    const std::string& text,
    const std::string& voice_name,
    const std::function<bool(const std::vector<float>&, int)>& on_audio_chunk,
    float speed,
    bool is_phonemes
) {
    const auto& voice_style = get_voice_style(voice_name);

    std::string phonemes = text;
    if (!is_phonemes) {
        phonemes = tokenizer_->phonemize(text);
    }

    auto batched_phonemes = _split_phonemes(phonemes);
    for (const auto& batch : batched_phonemes) {
        auto [audio_part, sr] = _create_audio(batch, voice_style, speed);
        if (!on_audio_chunk(audio_part, sr)) {
            break;  // caller requested early stop
        }
    }
}
