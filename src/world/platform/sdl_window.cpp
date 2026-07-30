#include "sdl_window.h"

#include <cstdio>

namespace winefox {
namespace world {

bool SdlWindow::init(int width, int height, const std::string& title) {
    if (!SDL_Init(SDL_INIT_VIDEO | SDL_INIT_AUDIO | SDL_INIT_EVENTS)) {
        std::fprintf(stderr, "[sdl] SDL_Init failed: %s\n", SDL_GetError());
        return false;
    }

    window_ = SDL_CreateWindow(title.c_str(), width, height,
                               SDL_WINDOW_VULKAN);
    if (!window_) {
        std::fprintf(stderr, "[sdl] SDL_CreateWindow failed: %s\n", SDL_GetError());
        return false;
    }
    width_  = width;
    height_ = height;
    return true;
}

SdlWindow::~SdlWindow() {
    if (window_) SDL_DestroyWindow(window_);
    SDL_Quit();
}

bool SdlWindow::poll_events() {
    SDL_Event e;
    while (SDL_PollEvent(&e)) {
        if (e.type == SDL_EVENT_QUIT) return false;
        if (e.type == SDL_EVENT_KEY_DOWN && e.key.key == SDLK_ESCAPE) {
            return false;
        }
    }
    return true;
}

} // namespace world
} // namespace winefox
