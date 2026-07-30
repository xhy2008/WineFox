// world_main.cpp — Entry point for winefox_world.exe
//
// This is a pure GUI program (WIN32 subsystem in Release, no console window).
// In Debug builds, a console window is attached for printf debugging.
//
// Startup sequence (async for fast window responsiveness):
//   1. Load winefox.json configuration
//   2. Create SDL3 window with Vulkan surface (immediate — window appears)
//   3. Enter main loop immediately (window stays responsive)
//   4. Background: winefox_init (LLM) and pipeline.init_models (TTS/VAD/ASR)
//      run in parallel. When both complete, init_audio_io + pipeline.start.
//
// Exit: Escape key, window close button, or process termination.

#include "../config/world_config.h"
#include "../ipc/message_queue.h"
#include "../platform/sdl_window.h"
#include "../render/vulkan_context.h"
#include "../voice/voice_pipeline.h"
#include "winefox_api.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <future>
#include <string>
#include <thread>

#include <windows.h>

int main(int argc, char* argv[])
{
    // Switch working directory to the exe's directory so that all relative
    // paths in winefox.json (models/, dict/, llm-finetune/, etc.) resolve
    // correctly regardless of where the exe is launched from.
    {
        char exe_path[MAX_PATH] = {};
        GetModuleFileNameA(nullptr, exe_path, MAX_PATH);
        std::string exe_dir(exe_path);
        size_t pos = exe_dir.find_last_of("\\/");
        if (pos != std::string::npos) exe_dir = exe_dir.substr(0, pos);
        SetCurrentDirectoryA(exe_dir.c_str());
    }

    // Set console to UTF-8 so Chinese text displays correctly in Debug builds.
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);

    std::fprintf(stderr, "=== WineFox World ===\n");

    // --- Load configuration ---
    winefox::world::WorldConfig wcfg = winefox::world::WorldConfig::defaults();
    wcfg.load("winefox.json");

    // --- IPC queues ---
    winefox::ipc::SpscQueue<winefox::ipc::RenderEvent, 256> render_events;

    // --- SDL3 window (created immediately so it appears on screen) ---
    winefox::world::SdlWindow window;
    if (!window.init(wcfg.window.width, wcfg.window.height, wcfg.window.title)) {
        std::fprintf(stderr, "[world] FATAL: window init failed\n");
        return 1;
    }

    // --- Vulkan context (device + swapchain, no rendering) ---
    winefox::world::VulkanContext vk;
    if (!vk.init(window.window(), window.width(), window.height())) {
        std::fprintf(stderr, "[world] WARNING: Vulkan init failed — continuing without rendering\n");
    }

    // --- Async initialization ---
    // winefox_init (LLM/embedder/memory) and pipeline.init_models (TTS/VAD/ASR)
    // are independent — run them in parallel on background threads. The main
    // thread enters the event loop immediately so the window stays responsive.
    winefox::world::VoicePipeline pipeline;
    WineFoxCore* core = nullptr;

    std::atomic<bool> init_done{false};
    std::atomic<bool> init_failed{false};

    auto t_init_start = std::chrono::steady_clock::now();

    // Use a promise so we can join the init thread cleanly at shutdown.
    std::thread init_thread([&]() {
        // Parallel: winefox_init (LLM) || pipeline.init_models (voice models)
        auto core_future = std::async(std::launch::async, [&]() -> WineFoxCore* {
            auto t0 = std::chrono::steady_clock::now();
            WineFoxCore* c = winefox_init("winefox.json", nullptr);
            auto t1 = std::chrono::steady_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            std::fprintf(stderr, "[world] winefox_init: %.0f ms\n", ms);
            return c;
        });

        auto models_future = std::async(std::launch::async, [&]() -> bool {
            auto t0 = std::chrono::steady_clock::now();
            bool ok = pipeline.init_models(wcfg, &render_events);
            auto t1 = std::chrono::steady_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            std::fprintf(stderr, "[world] pipeline.init_models: %.0f ms\n", ms);
            return ok;
        });

        // Wait for both to complete.
        core = core_future.get();
        bool models_ok = models_future.get();

        if (!core) {
            std::fprintf(stderr, "[world] FATAL: winefox_init failed\n");
            init_failed.store(true);
            return;
        }
        if (!models_ok) {
            std::fprintf(stderr, "[world] FATAL: pipeline init_models failed\n");
            init_failed.store(true);
            return;
        }

        // Step 2 + 3: set core, init audio I/O (fast).
        pipeline.set_core(core);
        if (!pipeline.init_audio_io()) {
            std::fprintf(stderr, "[world] FATAL: audio I/O init failed\n");
            init_failed.store(true);
            return;
        }
        if (!pipeline.start()) {
            std::fprintf(stderr, "[world] FATAL: pipeline start failed\n");
            init_failed.store(true);
            return;
        }

        auto t_init_end = std::chrono::steady_clock::now();
        double total_ms = std::chrono::duration<double, std::milli>(t_init_end - t_init_start).count();
        std::fprintf(stderr, "[world] init complete: %.0f ms\n", total_ms);

        init_done.store(true);
    });

    // --- Main loop (starts immediately, window is responsive) ---
    bool running = true;
    while (running) {
        // 1. Poll SDL events (window close, Escape key).
        if (!window.poll_events()) {
            running = false;
            break;
        }

        // 2. Present a Vulkan frame (just clears the screen).
        if (vk.ready()) {
            vk.present_frame();
        }

        // 3. Check init status.
        if (init_failed.load()) {
            std::fprintf(stderr, "[world] init failed, exiting\n");
            // Wait a moment so the user can see the error in console.
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            running = false;
            break;
        }

        // 4. Drain render events from the AI thread.
        winefox::ipc::RenderEvent e;
        while (render_events.try_pop(e)) {
            switch (e.kind) {
            case winefox::ipc::RenderEventKind::StateChange: {
                const char* state_names[] = {
                    "IDLE", "LISTENING", "RECOGNIZING", "THINKING", "SPEAKING"
                };
                int idx = e.int_val;
                if (idx >= 0 && idx <= 4) {
                    std::fprintf(stderr, "[state] %s\n", state_names[idx]);
                }
                break;
            }
            case winefox::ipc::RenderEventKind::Subtitle:
                std::fprintf(stderr, "[subtitle] %s\n", e.text);
                break;
            case winefox::ipc::RenderEventKind::Emotion:
                std::fprintf(stderr, "[emotion] %s\n", e.text);
                break;
            case winefox::ipc::RenderEventKind::Perf:
                std::fprintf(stderr, "[perf] tokens=%d tps=%.1f\n",
                             e.int_val, e.float_val);
                break;
            case winefox::ipc::RenderEventKind::Error:
                std::fprintf(stderr, "[error] %s\n", e.text);
                break;
            }
        }

        // 5. While init is in progress, yield to avoid busy-spinning.
        //    Once init is done, the loop runs at full speed for event polling.
        if (!init_done.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(16));  // ~60fps
        }
    }

    // --- Shutdown ---
    std::fprintf(stderr, "[world] shutting down...\n");

    // Wait for init thread to finish before shutting down (it may still be
    // running if the user closed the window during startup).
    if (init_thread.joinable()) {
        init_thread.join();
    }

    // Only stop pipeline and shutdown core if init completed successfully.
    if (init_done.load()) {
        pipeline.stop();
    }
    if (core) winefox_shutdown(core);

    std::fprintf(stderr, "[world] done.\n");
    return 0;
}
