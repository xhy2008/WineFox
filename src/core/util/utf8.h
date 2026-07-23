#pragma once

// UTF-8 string helpers: codepoint length, char counting, sentence splitting.
// All operations are byte-correct for UTF-8; no encoding conversion.

#include <cstdint>
#include <string>
#include <vector>

namespace winefox {
namespace utf8 {

// Number of bytes consumed by the UTF-8 codepoint starting at `c`.
// Returns 0 for invalid lead bytes.
inline int char_len(unsigned char c) {
    if (c < 0x80) return 1;
    if ((c & 0xE0) == 0xC0) return 2;
    if ((c & 0xF0) == 0xE0) return 3;
    if ((c & 0xF8) == 0xF0) return 4;
    return 0;
}

// Count codepoints in a UTF-8 string.
inline size_t length(const std::string& s) {
    size_t n = 0;
    for (size_t i = 0; i < s.size();) {
        int cl = char_len(static_cast<unsigned char>(s[i]));
        if (cl == 0) { ++i; ++n; continue; } // skip invalid byte, count as 1
        i += cl;
        ++n;
    }
    return n;
}

// Slice by codepoint count (returns bytes [byte_start, byte_end) covering the
// first `n` codepoints starting at `byte_start`).
inline std::string take(const std::string& s, size_t n) {
    size_t i = 0, cnt = 0;
    while (i < s.size() && cnt < n) {
        int cl = char_len(static_cast<unsigned char>(s[i]));
        if (cl == 0) { ++i; ++cnt; continue; }
        i += cl;
        ++cnt;
    }
    return s.substr(0, i);
}

// Split text into sentences for TTS streaming. Supports Chinese and ASCII
// punctuation: . ! ? , ; : and their fullwidth counterparts.
// Empty/whitespace-only segments are dropped.
inline std::vector<std::string> split_sentences(const std::string& text) {
    std::vector<std::string> out;
    std::string cur;
    auto flush = [&]() {
        if (!cur.empty()) {
            // trim trailing whitespace
            while (!cur.empty() &&
                   (cur.back() == ' ' || cur.back() == '\t' || cur.back() == '\n' ||
                    cur.back() == '\r')) {
                cur.pop_back();
            }
            if (!cur.empty()) out.push_back(cur);
            cur.clear();
        }
    };

    for (size_t i = 0; i < text.size();) {
        unsigned char c = static_cast<unsigned char>(text[i]);
        int cl = char_len(c);
        if (cl <= 1) {
            cur.push_back(static_cast<char>(c));
            if (c == '.' || c == '!' || c == '?' || c == ',' || c == ';' || c == ':') {
                flush();
            }
            ++i;
        } else {
            // Decode codepoint (only need the 3-byte / fullwidth cases).
            uint32_t cp = 0;
            if (cl == 3) {
                cp = ((c & 0x0F) << 12) |
                     ((static_cast<unsigned char>(text[i + 1]) & 0x3F) << 6) |
                     (static_cast<unsigned char>(text[i + 2]) & 0x3F);
            }
            // Fullwidth: U+3002 (。) U+FF01 (！) U+FF1F (？) U+FF0C (，)
            //           U+FF1B (；) U+FF1A (：) U+3001 (、)
            if (cp == 0x3002 || cp == 0xFF01 || cp == 0xFF1F || cp == 0xFF0C ||
                cp == 0xFF1B || cp == 0xFF1A || cp == 0x3001) {
                cur.append(text, i, cl);
                flush();
            } else {
                cur.append(text, i, cl);
            }
            i += cl;
        }
    }
    flush();
    return out;
}

} // namespace utf8
} // namespace winefox
