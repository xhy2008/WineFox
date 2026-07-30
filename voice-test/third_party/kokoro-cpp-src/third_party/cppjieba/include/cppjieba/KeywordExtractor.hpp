#ifndef CPPJIEBA_KEYWORD_EXTRACTOR_H
#define CPPJIEBA_KEYWORD_EXTRACTOR_H

#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include "MixSegment.hpp"

namespace cppjieba {

/*utf8*/
class KeywordExtractor {
 public:
  struct Word {
    std::string word;
    std::vector<size_t> offsets;
    double weight;
  }; // struct Word

  KeywordExtractor(const std::string& dictPath, 
        const std::string& hmmFilePath, 
        const std::string& idfPath, 
        const std::string& stopWordPath, 
        const std::string& userDict = "") 
    : segment_(dictPath, hmmFilePath, userDict) {
    LoadIdfDict(idfPath);
    LoadStopWordDict(stopWordPath);
  }
  KeywordExtractor(const DictTrie* dictTrie, 
        const HMMModel* model,
        const std::string& idfPath, 
        const std::string& stopWordPath) 
    : segment_(dictTrie, model) {
    LoadIdfDict(idfPath);
    LoadStopWordDict(stopWordPath);
  }
  ~KeywordExtractor() {
  }

  void Extract(const std::string& sentence, std::vector<std::string>& keywords, size_t topN) const {
    std::vector<Word> topWords;
    Extract(sentence, topWords, topN);
    for (size_t i = 0; i < topWords.size(); i++) {
      keywords.push_back(topWords[i].word);
    }
  }

  void Extract(const std::string& sentence, std::vector<pair<std::string, double> >& keywords, size_t topN) const {
    std::vector<Word> topWords;
    Extract(sentence, topWords, topN);
    for (size_t i = 0; i < topWords.size(); i++) {
      keywords.push_back(pair<std::string, double>(topWords[i].word, topWords[i].weight));
    }
  }

  void Extract(const std::string& sentence, std::vector<Word>& keywords, size_t topN) const {
    std::vector<std::string> words;
    segment_.Cut(sentence, words);

    std::map<std::string, Word> wordmap;
    size_t offset = 0;
    for (size_t i = 0; i < words.size(); ++i) {
      size_t t = offset;
      offset += words[i].size();
      if (IsSingleWord(words[i]) || stopWords_.find(words[i]) != stopWords_.end()) {
        continue;
      }
      wordmap[words[i]].offsets.push_back(t);
      wordmap[words[i]].weight += 1.0;
    }
    if (offset != sentence.size()) {
      XLOG(ERROR) << "words illegal";
      return;
    }

    keywords.clear();
    keywords.reserve(wordmap.size());
    for (std::map<std::string, Word>::iterator itr = wordmap.begin(); itr != wordmap.end(); ++itr) {
      std::unordered_map<std::string, double>::const_iterator cit = idfMap_.find(itr->first);
      if (cit != idfMap_.end()) {
        itr->second.weight *= cit->second;
      } else {
        itr->second.weight *= idfAverage_;
      }
      itr->second.word = itr->first;
      keywords.push_back(itr->second);
    }
    topN = min(topN, keywords.size());
    std::partial_sort(keywords.begin(), keywords.begin() + topN, keywords.end(), Compare);
    keywords.resize(topN);
  }
 private:
  void LoadIdfDict(const std::string& idfPath) {
    // Fast path: read entire file, parse in-place (avoid per-line vector alloc).
    std::ifstream ifs(idfPath, std::ios::binary | std::ios::ate);
    XCHECK(ifs.is_open()) << "open " << idfPath << " failed";
    auto sz = ifs.tellg();
    ifs.seekg(0, std::ios::beg);
    std::string content(sz, '\0');
    ifs.read(&content[0], sz);

    idfMap_.reserve(200000);
    double idfSum = 0.0;
    size_t lineno = 0;
    size_t pos = 0;
    while (pos < content.size()) {
      size_t line_end = content.find('\n', pos);
      if (line_end == std::string::npos) line_end = content.size();
      size_t line_len = line_end - pos;
      if (line_len > 0 && content[pos + line_len - 1] == '\r') line_len--;
      if (line_len == 0) { pos = line_end + 1; lineno++; continue; }

      const char* base = content.data() + pos;
      const char* sp = (const char*)memchr(base, ' ', line_len);
      if (!sp) { pos = line_end + 1; lineno++; continue; }

      std::string word(base, sp - base);
      std::string idf_str(sp + 1, base + line_len - sp - 1);
      double idf = atof(idf_str.c_str());
      idfMap_[word] = idf;
      idfSum += idf;
      lineno++;
      pos = line_end + 1;
    }

    assert(lineno);
    idfAverage_ = idfSum / lineno;
    assert(idfAverage_ > 0.0);
  }
  void LoadStopWordDict(const std::string& filePath) {
    std::ifstream ifs(filePath.c_str());
    XCHECK(ifs.is_open()) << "open " << filePath << " failed";
    std::string line ;
    while (getline(ifs, line)) {
      stopWords_.insert(line);
    }
    assert(stopWords_.size());
  }

  static bool Compare(const Word& lhs, const Word& rhs) {
    return lhs.weight > rhs.weight;
  }

  MixSegment segment_;
  std::unordered_map<std::string, double> idfMap_;
  double idfAverage_;

  std::unordered_set<std::string> stopWords_;
}; // class KeywordExtractor

inline std::ostream& operator << (std::ostream& os, const KeywordExtractor::Word& word) {
  return os << "{\"word\": \"" << word.word << "\", \"offset\": " << word.offsets << ", \"weight\": " << word.weight << "}"; 
}

} // namespace cppjieba

#endif
