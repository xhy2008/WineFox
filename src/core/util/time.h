#pragma once

// High-resolution timer for performance instrumentation.
// Works on Windows (QueryPerformanceCounter) and POSIX (clock_gettime).

#include <chrono>
#include <cstdint>
#include <string>

namespace winefox {
namespace time {

using clock = std::chrono::steady_clock;

inline int64_t now_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               clock::now().time_since_epoch())
        .count();
}

inline int64_t now_us() {
    return std::chrono::duration_cast<std::chrono::microseconds>(
               clock::now().time_since_epoch())
        .count();
}

// Simple RAII stopwatch. Reports elapsed ms to WF_LOG_INFO on destruction
// (DEBUG builds only; no-op in Release).
class scope_timer {
public:
    explicit scope_timer(const char* label)
        : label_(label), start_(clock::now()) {}
    ~scope_timer() {
#ifdef DEBUG
        auto end = clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start_).count();
        WF_LOG_DEBUG("scope_timer[%s] = %lld ms", label_, (long long)ms);
#else
        (void)0;
#endif
    }
    scope_timer(const scope_timer&) = delete;
    scope_timer& operator=(const scope_timer&) = delete;

    int64_t elapsed_ms() const {
        return std::chrono::duration_cast<std::chrono::milliseconds>(
                   clock::now() - start_)
            .count();
    }

private:
    const char* label_;
    clock::time_point start_;
};

} // namespace time
} // namespace winefox
