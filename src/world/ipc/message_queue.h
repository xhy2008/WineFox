// message_queue.h — Single-producer single-consumer lock-free ring buffer.
//
// Used for AI-thread → Render-thread event passing and Render → AI commands.
// Capacity is a power of two for fast modulo (mask instead of %).
// Designed for extreme low latency: no locks, no allocations, no syscalls.

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <new>

namespace winefox {
namespace ipc {

template<typename T, size_t Capacity>
class SpscQueue {
    static_assert((Capacity & (Capacity - 1)) == 0,
                  "Capacity must be a power of two");
    static_assert(std::is_trivially_copyable_v<T>,
                  "T must be trivially copyable for lock-free safety");

public:
    SpscQueue() : head_(0), tail_(0) {}

    // Producer side: try to push. Returns false if full.
    bool try_push(const T& item) {
        const size_t h = head_.load(std::memory_order_relaxed);
        const size_t next = (h + 1) & kMask;
        if (next == tail_.load(std::memory_order_acquire)) return false;
        buf_[h] = item;
        head_.store(next, std::memory_order_release);
        return true;
    }

    // Consumer side: try to pop. Returns false if empty.
    bool try_pop(T& out) {
        const size_t t = tail_.load(std::memory_order_relaxed);
        if (t == head_.load(std::memory_order_acquire)) return false;
        out = buf_[t];
        tail_.store((t + 1) & kMask, std::memory_order_release);
        return true;
    }

    size_t size() const {
        const size_t h = head_.load(std::memory_order_acquire);
        const size_t t = tail_.load(std::memory_order_acquire);
        return (h - t) & kMask;
    }

    bool empty() const { return size() == 0; }

private:
    static constexpr size_t kMask = Capacity - 1;

    // Align to cache line to prevent false sharing between head/tail.
    alignas(64) std::atomic<size_t> head_;
    alignas(64) std::atomic<size_t> tail_;
    alignas(64) T buf_[Capacity];
};

// ---------------------------------------------------------------------------
// Event types: AI thread → Render thread
// ---------------------------------------------------------------------------
enum class RenderEventKind : uint8_t {
    Subtitle,         // new subtitle text (partial or final)
    Emotion,          // emotion tag changed
    StateChange,      // pipeline state changed
    Perf,             // performance data updated
    Error,            // error message
};

struct RenderEvent {
    RenderEventKind kind;
    int32_t  int_val = 0;    // state enum, perf n_eval, etc.
    char     text[256] = {}; // subtitle text, emotion tag, error msg
    double   float_val = 0;  // tokens_per_sec, etc.
};

// ---------------------------------------------------------------------------
// Command types: Render thread → AI thread
// ---------------------------------------------------------------------------
enum class AiCommandKind : uint8_t {
    Interrupt,        // user wants to interrupt (VAD triggered during speaking)
    Quit,             // app is shutting down
};

struct AiCommand {
    AiCommandKind kind;
};

} // namespace ipc
} // namespace winefox
