#include "PinyinFinder.h"

#include <fstream>
#include <regex>
#include <vector>
#include <locale>
#include <limits>
#include <cstring>
#include <cstdio>
#include <chrono>

PinyinFinder::PinyinFinder() {}
PinyinFinder::~PinyinFinder() {}

// Fast UTF-8 → UTF-16 conversion (replaces BasicStringUtil::u8tou16).
// The original uses std::wstring_convert which constructs a locale facet
// per call — ~40μs each × 411k calls = 16 seconds. This manual version
// is ~100ns per call.
static inline void fast_u8tou16(const char* src, size_t len, std::u16string& dst) {
    dst.clear();
    dst.reserve(len);
    for (size_t i = 0; i < len; ) {
        unsigned char c = (unsigned char)src[i];
        uint32_t cp = 0;
        size_t adv = 0;
        if (c < 0x80) { cp = c; adv = 1; }
        else if ((c & 0xE0) == 0xC0 && i + 1 < len) {
            cp = ((uint32_t)(c & 0x1F) << 6) | ((uint32_t)(unsigned char)src[i+1] & 0x3F);
            adv = 2;
        } else if ((c & 0xF0) == 0xE0 && i + 2 < len) {
            cp = ((uint32_t)(c & 0x0F) << 12) | ((uint32_t)(unsigned char)src[i+1] & 0x3F) << 6
               | ((uint32_t)(unsigned char)src[i+2] & 0x3F);
            adv = 3;
        } else if ((c & 0xF8) == 0xF0 && i + 3 < len) {
            cp = ((uint32_t)(c & 0x07) << 18) | ((uint32_t)(unsigned char)src[i+1] & 0x3F) << 12
               | ((uint32_t)(unsigned char)src[i+2] & 0x3F) << 6
               | ((uint32_t)(unsigned char)src[i+3] & 0x3F);
            adv = 4;
        } else { i++; continue; }

        if (cp <= 0xFFFF) {
            dst.push_back((char16_t)cp);
        } else {
            cp -= 0x10000;
            dst.push_back((char16_t)(0xD800 + (cp >> 10)));
            dst.push_back((char16_t)(0xDC00 + (cp & 0x3FF)));
        }
        i += adv;
    }
}

bool PinyinFinder::init(const std::string& singleCharacterDictPath,
          const std::string& wordsDictPath) {

  auto t0 = std::chrono::steady_clock::now();
  auto ms_since = [&]() {
    return std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - t0).count();
  };

  // Pre-reserve for ~500k phrase entries + ~20k single chars.
  word_pinyin_dict_.reserve(600000);

  // --- Single character dict (pinyin.txt, ~0.56MB) ---
  // Format: U+XXXX: pinyin # comment
  {
    std::ifstream ifs(singleCharacterDictPath, std::ios::binary | std::ios::ate);
    if (!ifs.is_open()) {
      LOG_WARNING << "Failed to open file: " << singleCharacterDictPath << std::endl;
      return false;
    }
    auto sz = ifs.tellg();
    ifs.seekg(0, std::ios::beg);
    std::string content(sz, '\0');
    ifs.read(&content[0], sz);

    int tn = 0;
    size_t pos = 0;
    while (pos < content.size()) {
      size_t line_end = content.find('\n', pos);
      if (line_end == std::string::npos) line_end = content.size();
      size_t line_len = line_end - pos;
      if (line_len > 0 && content[pos + line_len - 1] == '\r') line_len--;
      if (line_len == 0) { pos = line_end + 1; continue; }
      if (content[pos] == '#') { pos = line_end + 1; continue; }
      if (line_len < 3 || content[pos] != 'U' || content[pos+1] != '+') { pos = line_end + 1; continue; }

      const char* base = content.data() + pos;
      const char* colon = (const char*)memchr(base, ':', line_len);
      if (!colon) { pos = line_end + 1; continue; }

      std::string hexStr(base + 2, colon - (base + 2));
      uint32_t unicode = 0;
      try { unicode = std::stoul(hexStr, nullptr, 16); } catch(...) { pos = line_end + 1; continue; }

      const char* pinyinStart = colon + 1;
      while (pinyinStart < base + line_len && (*pinyinStart == ' ' || *pinyinStart == '\t')) pinyinStart++;
      const char* line_endp = base + line_len;
      const char* hash = (const char*)memchr(pinyinStart, '#', line_endp - pinyinStart);
      const char* pinyinEnd = hash ? hash : line_endp;
      while (pinyinEnd > pinyinStart && (*(pinyinEnd-1) == ' ' || *(pinyinEnd-1) == '\t')) pinyinEnd--;
      if (pinyinEnd <= pinyinStart) { pos = line_end + 1; continue; }

      std::string pinyin(pinyinStart, pinyinEnd - pinyinStart);

      UnicodeStr ustr;
      ustr.append(1, static_cast<UnicodeCharT>(unicode));

      // Take first comma-separated variant (no vector alloc)
      size_t comma = pinyin.find(',');
      if (comma != std::string::npos) pinyin = pinyin.substr(0, comma);
      word_pinyin_dict_[ustr] = std::move(pinyin);
      tn += 1;
      pos = line_end + 1;
    }
    std::fprintf(stderr, "[PinyinFinder] char dict: %.0f ms (%d entries)\n", ms_since(), tn);
  }

  // --- Phrase dict (pinyin_phrase.txt, ~9MB) ---
  // Format: word: pinyin1 pinyin2
  {
    std::ifstream ifs(wordsDictPath, std::ios::binary | std::ios::ate);
    if (!ifs.is_open()) {
      LOG_WARNING << "Failed to open file: " << wordsDictPath << std::endl;
      return false;
    }
    auto sz = ifs.tellg();
    ifs.seekg(0, std::ios::beg);
    std::string content(sz, '\0');
    ifs.read(&content[0], sz);

    int pc = 0;
    size_t pos = 0;
    while (pos < content.size()) {
      size_t line_end = content.find('\n', pos);
      if (line_end == std::string::npos) line_end = content.size();
      size_t line_len = line_end - pos;
      if (line_len > 0 && content[pos + line_len - 1] == '\r') line_len--;
      if (line_len == 0) { pos = line_end + 1; continue; }
      if (content[pos] == '#') { pos = line_end + 1; continue; }

      const char* base = content.data() + pos;
      // Strip comments
      const char* comment = (const char*)memchr(base, '#', line_len);
      size_t effective_len = comment ? (size_t)(comment - base) : line_len;

      const char* colon = (const char*)memchr(base, ':', effective_len);
      if (!colon) { pos = line_end + 1; continue; }

      // Parse Word (trim trailing whitespace before colon)
      const char* wordEnd = colon;
      while (wordEnd > base && (*(wordEnd-1) == ' ' || *(wordEnd-1) == '\t')) wordEnd--;
      std::string word(base, wordEnd - base);
      if (word.empty()) { pos = line_end + 1; continue; }

      // Parse Pinyin (trim trailing whitespace)
      const char* pinyinStart = colon + 1;
      while (pinyinStart < base + effective_len && (*pinyinStart == ' ' || *pinyinStart == '\t')) pinyinStart++;
      const char* pinyinEnd = base + effective_len;
      while (pinyinEnd > pinyinStart && (*(pinyinEnd-1) == ' ' || *(pinyinEnd-1) == '\t')) pinyinEnd--;
      if (pinyinEnd <= pinyinStart) { pos = line_end + 1; continue; }
      std::string pinyin(pinyinStart, pinyinEnd - pinyinStart);

      UnicodeStr ustr;
      fast_u8tou16(word.c_str(), word.size(), ustr);
      word_pinyin_dict_[ustr] = std::move(pinyin);
      pc += 1;
      pos = line_end + 1;
    }
    std::fprintf(stderr, "[PinyinFinder] phrase dict: %.0f ms (%d entries)\n", ms_since(), pc);
  }
  return true;
}

void PinyinFinder::find_best_pinyin(const std::string& phrasestr, std::vector<std::string>& pinyins) {
  UnicodeStr phrase;
  BasicStringUtil::u8tou16(phrasestr.c_str(), phrasestr.size(), phrase);
  int n = phrase.size();
  if (n == 0) return;

  std::vector<std::vector<int>> dp(n, std::vector<int>(n, std::numeric_limits<int>::max()));
  std::vector<std::vector<int>> opts(n, std::vector<int>(n, -1));
  
  for (int length = 1; length <= n; ++length) {
    for (int i = 0; i <= n - length; ++i) {
      int j = i + length - 1;
      if (length == 1) {
        dp[i][j] = 1;
        opts[i][j] = j;
      } else {
        int maxtry = length;
        if (length > kMaxChars) {
          maxtry = kMaxChars;
        }
        for (int k = maxtry; k >= 1; k--) {
          int to = i + k - 1;
          UnicodeStr sub = phrase.substr(i, k);
          
          if (word_pinyin_dict_.find(sub) != word_pinyin_dict_.end()) {
            if (to == j) {
              dp[i][j] = k == 1 ? 1 : 0; // Preference for longer matches
              opts[i][j] = j;
            } else {
              // Cost calculation logic from original code
              // k == 1 means we took a single char
              int cost = (k == 1) ? (dp[to + 1][j] + 1) : dp[to + 1][j];
              
              if (dp[i][j] > cost) {
                dp[i][j] = cost;
                opts[i][j] = to;
              }
            }
          }
        }
      }
    }
  }

  // Construct the best pinyin string using dp and opts
  int i = 0;
  int j = n - 1;
  while (i <= j) {
    int opt = opts[i][j];
    if(opt == -1){
      opt = i;
    }
    UnicodeStr sub = phrase.substr(i, opt - i + 1);
    auto it = word_pinyin_dict_.find(sub);
    
    if(it == word_pinyin_dict_.end()){
      std::string tstr;
      BasicStringUtil::u16tou8(sub.data(), sub.size(), tstr);
      pinyins.emplace_back(tstr);
    } else {
      std::vector<std::string> tmps;
      BasicStringUtil::SplitString(it->second.c_str(), it->second.size(), ' ', &tmps);
      for(auto& str: tmps){
        pinyins.emplace_back(str);
      }
    }
    i = opt + 1;
  }
}
