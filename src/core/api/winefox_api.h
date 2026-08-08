// winefox_api.h — C ABI for winefox_core.dll
//
// This is the ONLY exported surface from winefox_core.dll. It wraps the
// entire LLM + memory + conversation pipeline behind an opaque handle
// and a small set of C functions, so the world exe can use the AI core
// without linking llama.cpp/ggml directly (avoiding ggml target conflicts
// with SenseVoice.cpp).
//
// Memory ownership: strings returned by winefox_chat / winefox_get_memory_info
// are allocated inside the DLL and MUST be freed via winefox_free_string.
// Do not call free() or delete on them from the exe — they may use a
// different CRT heap (each /MT binary has its own CRT instance).

#ifndef WINEFOX_API_H
#define WINEFOX_API_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// DLL import/export macros
#ifdef WINEFOX_CORE_EXPORTS
  #define WF_API __declspec(dllexport)
#else
  #define WF_API __declspec(dllimport)
#endif

// Opaque handle representing the full AI core (LLM + memory + conversation).
typedef struct WineFoxCore WineFoxCore;

// Performance data from the last chat() call.
typedef struct {
    int32_t n_eval;          // generated tokens
    double  tokens_per_sec;  // decode speed
    double  t_prefill_ms;    // prefill latency
} WineFoxPerf;

// Callback for streaming tokens. Return 0 to stop generation immediately
// (hard interrupt), non-zero to continue.
// `user` is passed through from winefox_chat().
typedef int (*WineFoxTokenCallback)(const char* token, void* user);

// Initialise the AI core: loads config, opens SQLite DB, loads LLM +
// embedder models, attaches LoRA, creates conversation manager, pre-warms
// KV cache. Returns NULL on failure (error printed to stderr).
//
// config_path:        path to winefox.json (NULL = "./winefox.json")
// system_prompt_path: path to persona file (NULL = config's value)
WF_API WineFoxCore* winefox_init(const char* config_path,
                                  const char* system_prompt_path);

// Shut down and free all resources. Safe to call with NULL.
WF_API void winefox_shutdown(WineFoxCore* core);

// Process one user turn. Streams the fox's reply (text only, without the
// [emotion] tag) via on_token. Returns the full reply text (caller must
// free with winefox_free_string). Returns NULL on error.
//
// If on_token returns 0, generation is aborted immediately (hard interrupt).
// The partial reply is still returned.
WF_API const char* winefox_chat(WineFoxCore* core,
                                 const char* user_input,
                                 WineFoxTokenCallback on_token,
                                 void* user);

// Get the emotion tag from the last chat() call.
// Returns "neutral" if no chat has been made yet.
WF_API const char* winefox_last_emotion(WineFoxCore* core);

// Get a human-readable log of the long-term memories recalled for the last
// chat() call (one line per hit: file id, score, title, content snippet).
// Returns an empty string if no memories were recalled.
WF_API const char* winefox_last_recall(WineFoxCore* core);

// Get performance data from the last chat() call.
WF_API void winefox_last_perf(WineFoxCore* core, WineFoxPerf* out);

// --- Commands ---
WF_API const char* winefox_get_memory_info(WineFoxCore* core); // /memory
WF_API void        winefox_reset(WineFoxCore* core);            // /reset
WF_API long long   winefox_force_distill(WineFoxCore* core);    // /distill
WF_API long long   winefox_session_id(WineFoxCore* core);
WF_API void        winefox_close_session(WineFoxCore* core, long long session_id);

// Free a string returned by any winefox_* function. Safe with NULL.
WF_API void winefox_free_string(const char* s);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // WINEFOX_API_H
