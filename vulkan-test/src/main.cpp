// main.cpp — Entry point for the Vulkan 3D rendering test.
//
// Creates an SDL3 window with Vulkan, initializes the VulkanApp rendering
// pipeline, and runs the main loop. Inside the enclosed room:
//   Left-drag   : look around (rotate camera)
//   W/A/S/D     : move forward / left / back / right
//   Q / E       : move down / up
//   Mouse wheel : adjust movement speed
//   ESC / close : quit

#include <cstdio>
#include <cstdlib>

#include <SDL3/SDL.h>

#include "vulkan_app.h"

int main() {
    // Disable stderr buffering so logs are visible even on a hard crash.
    setvbuf(stderr, NULL, _IONBF, 0);

    const int width  = 1280;
    const int height = 720;

    if (!SDL_Init(SDL_INIT_VIDEO | SDL_INIT_EVENTS)) {
        std::fprintf(stderr, "[main] SDL_Init failed: %s\n", SDL_GetError());
        return 1;
    }

    SDL_Window* window = SDL_CreateWindow("vulkan-test — Vulkan 3D Rendering",
                                          width, height, SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE);
    if (!window) {
        std::fprintf(stderr, "[main] SDL_CreateWindow failed: %s\n", SDL_GetError());
        SDL_Quit();
        return 1;
    }

    vkt::VulkanApp app;
    if (!app.init(window, width, height)) {
        std::fprintf(stderr, "[main] VulkanApp init failed\n");
        SDL_DestroyWindow(window);
        SDL_Quit();
        return 1;
    }

    std::fprintf(stderr, "[main] Controls: WASD=move, QE=up/down, "
                         "mouse=look (captured), wheel=speed, ESC=release/quit\n");

    // --- Mouse capture (relative mode) ---
    // Locks pointer to window center, hides cursor, and emits relative motion
    // events even when the mouse leaves the window. FPS-style mouse control.
    SDL_SetWindowRelativeMouseMode(window, true);
    bool mouse_captured = true;

    // --- Mouse state for fallback (non-relative) drag-look ---
    bool  left_down = false;
    float last_x = 0.0f, last_y = 0.0f;

    Uint64 last_ticks = SDL_GetTicks();
    bool running = true;

    // --- FPS counter (updated once per second) ---
    int    frame_count   = 0;
    Uint64 fps_last_time = last_ticks;
    char   title_buf[160];

    while (running) {
        SDL_Event e;
        while (SDL_PollEvent(&e)) {
            switch (e.type) {
            case SDL_EVENT_QUIT:
                running = false;
                break;

            case SDL_EVENT_KEY_DOWN:
                if (e.key.key == SDLK_ESCAPE) {
                    if (mouse_captured) {
                        // First ESC: release mouse pointer.
                        SDL_SetWindowRelativeMouseMode(window, false);
                        mouse_captured = false;
                    } else {
                        // Second ESC: quit.
                        running = false;
                    }
                }
                break;

            case SDL_EVENT_WINDOW_CLOSE_REQUESTED:
                running = false;
                break;

            case SDL_EVENT_WINDOW_RESIZED:
                app.on_resize(e.window.data1, e.window.data2);
                break;

            case SDL_EVENT_MOUSE_BUTTON_DOWN:
                if (!mouse_captured) {
                    // Recapture on click after ESC release.
                    SDL_SetWindowRelativeMouseMode(window, true);
                    mouse_captured = true;
                    // Reset drag origin so we don't jump.
                    last_x = e.button.x;
                    last_y = e.button.y;
                }
                if (e.button.button == SDL_BUTTON_LEFT) {
                    left_down = true;
                    last_x = e.button.x;
                    last_y = e.button.y;
                }
                break;

            case SDL_EVENT_MOUSE_BUTTON_UP:
                if (e.button.button == SDL_BUTTON_LEFT) left_down = false;
                break;

            case SDL_EVENT_MOUSE_MOTION:
                if (mouse_captured) {
                    // Relative mode: use xrel/yrel (delta from last position,
                    // pointer is locked to window center by SDL).
                    app.camera().rotate((float)e.motion.xrel, (float)e.motion.yrel);
                } else if (left_down) {
                    // Fallback: drag-look with visible cursor.
                    float dx = e.motion.x - last_x;
                    float dy = e.motion.y - last_y;
                    last_x = e.motion.x;
                    last_y = e.motion.y;
                    app.camera().rotate(dx, dy);
                }
                break;

            case SDL_EVENT_MOUSE_WHEEL:
                app.camera().adjust_speed(e.wheel.y);
                break;
            }
        }

        // Frame time (seconds), clamped to avoid big jumps on stutter.
        Uint64 now = SDL_GetTicks();
        float dt = (now - last_ticks) / 1000.0f;
        last_ticks = now;
        if (dt > 0.1f) dt = 0.1f;

        // Continuous movement via held keys.
        const bool* keys = SDL_GetKeyboardState(NULL);
        if (keys[SDL_SCANCODE_W]) app.camera().move_forward(dt);
        if (keys[SDL_SCANCODE_S]) app.camera().move_forward(-dt);
        if (keys[SDL_SCANCODE_A]) app.camera().move_right(-dt);
        if (keys[SDL_SCANCODE_D]) app.camera().move_right(dt);
        if (keys[SDL_SCANCODE_Q]) app.camera().move_up(-dt);
        if (keys[SDL_SCANCODE_E]) app.camera().move_up(dt);

        app.draw_frame();

        // Update FPS in the window title once per second.
        frame_count++;
        Uint64 fps_now = SDL_GetTicks();
        if (fps_now - fps_last_time >= 1000) {
            float fps = frame_count * 1000.0f / (float)(fps_now - fps_last_time);
            std::snprintf(title_buf, sizeof(title_buf),
                          "vulkan-test | FPS: %.1f | speed: %.1f | pos: (%.1f, %.1f, %.1f) | %s",
                          fps, app.camera().speed(),
                          app.camera().position().x, app.camera().position().y,
                          app.camera().position().z,
                          mouse_captured ? "captured" : "free");
            SDL_SetWindowTitle(window, title_buf);
            frame_count = 0;
            fps_last_time = fps_now;
        }
    }

    // Restore mouse before quitting so the OS cursor is visible.
    SDL_SetWindowRelativeMouseMode(window, false);
    SDL_DestroyWindow(window);
    SDL_Quit();
    return 0;
}
