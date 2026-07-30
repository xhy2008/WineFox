// sdl_window.h — SDL3 window + Vulkan surface creation.
//
// Creates a borderless SDL3 window and exposes the window handle for
// Vulkan surface creation. No rendering is done here — the VulkanContext
// takes the SDL window handle and creates its own surface.

#pragma once

#include <SDL3/SDL.h>
#include <string>

namespace winefox {
namespace world {

class SdlWindow {
public:
    bool init(int width, int height, const std::string& title);
    ~SdlWindow();

    // Poll events. Returns false if the window should close (SDL_EVENT_QUIT
    // or Escape key pressed).
    bool poll_events();

    SDL_Window* window() const { return window_; }
    int width() const { return width_; }
    int height() const { return height_; }

private:
    SDL_Window* window_ = nullptr;
    int         width_  = 0;
    int         height_ = 0;
};

} // namespace world
} // namespace winefox
