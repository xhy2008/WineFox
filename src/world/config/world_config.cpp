#include "world_config.h"

#include <nlohmann/json.hpp>
#include <fstream>
#include <cstdio>

using json = nlohmann::json;

namespace winefox {
namespace world {

bool WorldConfig::load(const std::string& path) {
    std::ifstream ifs(path);
    if (!ifs) {
        std::fprintf(stderr, "[world] config not found: %s (using defaults)\n",
                     path.c_str());
        return false;
    }
    json j;
    try {
        ifs >> j;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[world] config parse error: %s\n", e.what());
        return false;
    }

    // voice.vad
    if (j.contains("voice") && j["voice"].contains("vad")) {
        auto& v = j["voice"]["vad"];
        if (v.contains("threshold"))   vad.threshold   = v["threshold"];
        if (v.contains("hop_size"))    vad.hop_size    = v["hop_size"];
        if (v.contains("min_speech"))  vad.min_speech  = v["min_speech"];
        if (v.contains("min_silence")) vad.min_silence = v["min_silence"];
        if (v.contains("max_speech"))  vad.max_speech  = v["max_speech"];
    }

    // voice.asr
    if (j.contains("voice") && j["voice"].contains("asr")) {
        auto& a = j["voice"]["asr"];
        if (a.contains("model_path")) asr.model_path = a["model_path"];
        if (a.contains("language"))   asr.language   = a["language"];
        if (a.contains("threads"))    asr.threads    = a["threads"];
        if (a.contains("use_itn"))    asr.use_itn    = a["use_itn"];
        if (a.contains("flash_attn")) asr.flash_attn = a["flash_attn"];
    }

    // voice.tts
    if (j.contains("voice") && j["voice"].contains("tts")) {
        auto& t = j["voice"]["tts"];
        if (t.contains("encoder_path")) tts.encoder_path = t["encoder_path"];
        if (t.contains("decoder_path")) tts.decoder_path = t["decoder_path"];
        if (t.contains("voices_path"))  tts.voices_path  = t["voices_path"];
        if (t.contains("vocab_path"))   tts.vocab_path   = t["vocab_path"];
        if (t.contains("dict_dir"))     tts.dict_dir     = t["dict_dir"];
        if (t.contains("voice_name"))   tts.voice_name   = t["voice_name"];
        if (t.contains("speed"))        tts.speed        = t["speed"];
        if (t.contains("threads"))      tts.threads      = t["threads"];
    }

    // voice.aec
    if (j.contains("voice") && j["voice"].contains("aec")) {
        auto& a = j["voice"]["aec"];
        if (a.contains("enabled")) aec.enabled = a["enabled"];
        if (a.contains("level"))   aec.level   = a["level"];
    }

    // window
    if (j.contains("window")) {
        auto& w = j["window"];
        if (w.contains("width"))  window.width  = w["width"];
        if (w.contains("height")) window.height = w["height"];
        if (w.contains("title"))  window.title  = w["title"];
    }

    return true;
}

WorldConfig WorldConfig::defaults() { return {}; }

} // namespace world
} // namespace winefox
