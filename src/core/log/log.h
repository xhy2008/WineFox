#pragma once

// WineFox logging facade.
//
// Design (PLAN.md 11.2.7): WF_LOG_* macros are no-ops in Release builds and
// emit to stderr in DEBUG builds. Keeping the implementation dependency-free
// (no spdlog) avoids pulling a third party just for debug output; we can swap
// the backend later without touching call sites.

#include <cstdarg>
#include <cstdio>

namespace winefox {
namespace log {

inline void raw(const char* level, const char* fmt, ...) {
    char buf[1024];
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    std::fprintf(stderr, "[%s] %s\n", level, buf);
    std::fflush(stderr);
}

} // namespace log
} // namespace winefox

#ifdef DEBUG
  #define WF_LOG_INFO(...)  ::winefox::log::raw("INFO",  __VA_ARGS__)
  #define WF_LOG_WARN(...)  ::winefox::log::raw("WARN",  __VA_ARGS__)
  #define WF_LOG_ERROR(...) ::winefox::log::raw("ERROR", __VA_ARGS__)
  #define WF_LOG_DEBUG(...) ::winefox::log::raw("DEBUG", __VA_ARGS__)
#else
  #define WF_LOG_INFO(...)  ((void)0)
  #define WF_LOG_WARN(...)  ((void)0)
  #define WF_LOG_ERROR(...) ((void)0)
  #define WF_LOG_DEBUG(...) ((void)0)
#endif
