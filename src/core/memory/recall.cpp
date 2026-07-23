#include "recall.h"

#include "../log/log.h"
#include "../util/time.h"

#include <algorithm>
#include <cstring>

namespace winefox {
namespace memory {

namespace {

// Candidate hit during retrieval.
struct SegmentHit {
    long long              seg_id;
    long long              file_id;
    std::string            content;
    std::vector<float>     embedding;
    int64_t                created_at;
    float                  score;
};

} // namespace

bool RecallService::init(storage::SqliteDb* db, embedder::EmbedderService* embedder) {
    if (!db || !embedder) return false;
    db_ = db;
    embedder_ = embedder;
    return true;
}

std::string RecallService::encode_embedding_(const std::vector<float>& v) {
    std::string blob(v.size() * sizeof(float), '\0');
    std::memcpy(blob.data(), v.data(), blob.size());
    return blob;
}

std::vector<float> RecallService::decode_embedding_(const std::string& blob) {
    if (blob.empty()) return {};
    size_t n = blob.size() / sizeof(float);
    std::vector<float> v(n);
    std::memcpy(v.data(), blob.data(), n * sizeof(float));
    return v;
}

float RecallService::cosine_(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size() || a.empty()) return 0.0f;
    // Embeddings are already L2-normalised by EmbedderService, so dot product
    // == cosine similarity. We still guard against zero norms defensively.
    float dot = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) dot += a[i] * b[i];
    return dot;
}

std::vector<Recall> RecallService::recall(const std::string& query, int top_k) {
    std::vector<Recall> out;
    if (!db_ || !embedder_ || !embedder_->ready() || query.empty()) return out;

    std::vector<float> q = embedder_->embed(query);
    if (q.empty()) return out;

    // Load every segment. For Phase 1 the memory footprint is small enough
    // that a linear scan is fine; revisit with an ANN index once it matters.
    std::vector<SegmentHit> hits;
    db_->query(
        "SELECT id, recall_file_id, content, embedding, created_at "
        "FROM recall_segments",
        {},
        [&](sqlite3_stmt* stmt) {
            SegmentHit h;
            h.seg_id    = sqlite3_column_int64(stmt, 0);
            h.file_id   = sqlite3_column_int64(stmt, 1);
            h.content   = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2));
            const void* blob = sqlite3_column_blob(stmt, 3);
            int blob_sz      = sqlite3_column_bytes(stmt, 3);
            if (blob && blob_sz > 0) {
                h.embedding = decode_embedding_(
                    std::string(static_cast<const char*>(blob), blob_sz));
            }
            h.created_at = sqlite3_column_int64(stmt, 4);
            h.score      = cosine_(q, h.embedding);
            hits.push_back(std::move(h));
            return true;
        });

    if (hits.empty()) return out;

    std::sort(hits.begin(), hits.end(),
              [](const SegmentHit& a, const SegmentHit& b) { return a.score > b.score; });

    // Time-decay: blend similarity with recency so stale-but-similar memories
    // do not completely drown out fresher ones. decay = 0.7*sim + 0.3*recency.
    int64_t now = winefox::time::now_ms() / 1000;
    int64_t newest = 0, oldest = 0;
    for (const auto& h : hits) {
        if (newest == 0 || h.created_at > newest) newest = h.created_at;
        if (oldest == 0 || h.created_at < oldest) oldest = h.created_at;
    }
    int64_t span = (newest > oldest) ? (newest - oldest) : 1;

    // Take top_k * 2 candidates before aggregation, then aggregate by file_id.
    int cand = std::min<int>(static_cast<int>(hits.size()), std::max(top_k * 2, top_k));
    std::map<long long, Recall> by_file;
    for (int i = 0; i < cand; ++i) {
        const auto& h = hits[i];
        float recency = static_cast<float>(h.created_at - oldest) / span;
        float blended = 0.7f * h.score + 0.3f * recency;

        auto it = by_file.find(h.file_id);
        if (it == by_file.end()) {
            Recall r;
            r.file_id    = h.file_id;
            r.content    = h.content;
            r.score      = blended;
            r.created_at = h.created_at;
            by_file[h.file_id] = std::move(r);
        } else {
            // Aggregate: append content, keep max score.
            it->second.content += "\n" + h.content;
            if (blended > it->second.score) it->second.score = blended;
        }
    }

    out.reserve(by_file.size());
    for (auto& kv : by_file) out.push_back(std::move(kv.second));
    std::sort(out.begin(), out.end(),
              [](const Recall& a, const Recall& b) { return a.score > b.score; });
    if (static_cast<int>(out.size()) > top_k) out.resize(top_k);

    // Bump recall_count / last_recalled_at on the touched files.
    for (const auto& r : out) {
        db_->exec("UPDATE recall_files SET recall_count = recall_count + 1, "
                  "last_recalled_at = " +
                  std::to_string(now) +
                  " WHERE id = " + std::to_string(r.file_id));
    }
    return out;
}

long long RecallService::commit_recall_file(const RecallFile& file) {
    if (!db_ || !embedder_) return -1;

    // Batch-embed segments + file content first (PLAN.md 9).
    std::vector<std::string> texts;
    texts.push_back(file.content);
    for (const auto& s : file.segments) texts.push_back(s.content);
    auto embs = embedder_->embed_batch(texts);
    if (embs.empty() || embs[0].empty()) {
        WF_LOG_ERROR("Recall: embed_batch returned empty for commit");
        return -1;
    }

    int64_t now = winefox::time::now_ms() / 1000;
    std::string now_str = std::to_string(now);
    std::string file_emb_blob = encode_embedding_(embs[0]);

    db_->exec("BEGIN");
    long long file_id = db_->insert(
        "INSERT INTO recall_files(title, content, summary, embedding, created_at) "
        "VALUES(?,?,?,?,?)",
        {file.title, file.content, file.summary, file_emb_blob, now_str});
    if (file_id < 0) {
        db_->exec("ROLLBACK");
        return -1;
    }
    for (size_t i = 0; i < file.segments.size(); ++i) {
        const auto& s = file.segments[i];
        std::string seg_emb = encode_embedding_(embs[i + 1]);
        db_->insert(
            "INSERT INTO recall_segments(recall_file_id, content, embedding, created_at) "
            "VALUES(?,?,?,?)",
            {std::to_string(file_id), s.content, seg_emb, now_str});
    }
    for (const auto& res : file.resources) {
        db_->insert(
            "INSERT OR IGNORE INTO recall_file_resources(recall_file_id, resource_path, resource_type) "
            "VALUES(?,?,?)",
            {std::to_string(file_id), res, ""});
    }
    db_->exec("COMMIT");
    WF_LOG_INFO("Recall: committed file %lld (%zu segments)", file_id, file.segments.size());
    return file_id;
}

void RecallService::upsert_profile(const std::string& key, const std::string& value) {
    if (!db_) return;
    int64_t now = winefox::time::now_ms() / 1000;
    db_->exec("INSERT INTO profile(key, value, updated_at) VALUES('" + key + "','" +
              value + "','" + std::to_string(now) +
              "') ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at");
}

std::string RecallService::get_profile(const std::string& key) const {
    if (!db_) return "";
    std::string result;
    db_->query("SELECT value FROM profile WHERE key='" + key + "'", {},
               [&](sqlite3_stmt* stmt) {
                   const unsigned char* v = sqlite3_column_text(stmt, 0);
                   if (v) result = reinterpret_cast<const char*>(v);
                   return true;
               });
    return result;
}

std::map<std::string, std::string> RecallService::get_all_profile() const {
    std::map<std::string, std::string> out;
    if (!db_) return out;
    db_->query("SELECT key, value FROM profile", {},
               [&](sqlite3_stmt* stmt) {
                   const unsigned char* k = sqlite3_column_text(stmt, 0);
                   const unsigned char* v = sqlite3_column_text(stmt, 1);
                   if (k && v) out[reinterpret_cast<const char*>(k)] = reinterpret_cast<const char*>(v);
                   return true;
               });
    return out;
}

long long RecallService::log_raw_message(const Message& msg, long long session_id) {
    if (!db_) return -1;
    int64_t now = msg.timestamp ? msg.timestamp : (winefox::time::now_ms() / 1000);
    return db_->insert(
        "INSERT INTO raw_messages(role, content, emotion, created_at, session_id) "
        "VALUES(?,?,?,?,?)",
        {msg.role, msg.content, msg.emotion, std::to_string(now), std::to_string(session_id)});
}

long long RecallService::start_session() {
    if (!db_) return -1;
    int64_t now = winefox::time::now_ms() / 1000;
    return db_->insert("INSERT INTO sessions(started_at) VALUES(?)",
                       {std::to_string(now)});
}

void RecallService::close_session(long long session_id) {
    if (!db_ || session_id < 0) return;
    int64_t now = winefox::time::now_ms() / 1000;
    db_->exec("UPDATE sessions SET ended_at=" + std::to_string(now) +
              " WHERE id=" + std::to_string(session_id));
}

std::vector<Message> RecallService::get_session_messages(long long session_id) const {
    std::vector<Message> out;
    if (!db_) return out;
    db_->query(
        "SELECT role, content, emotion, created_at FROM raw_messages "
        "WHERE session_id=? ORDER BY id",
        {std::to_string(session_id)},
        [&](sqlite3_stmt* stmt) {
            Message m;
            const unsigned char* r = sqlite3_column_text(stmt, 0);
            const unsigned char* c = sqlite3_column_text(stmt, 1);
            const unsigned char* e = sqlite3_column_text(stmt, 2);
            if (r) m.role = reinterpret_cast<const char*>(r);
            if (c) m.content = reinterpret_cast<const char*>(c);
            if (e) m.emotion = reinterpret_cast<const char*>(e);
            m.timestamp = sqlite3_column_int64(stmt, 3);
            out.push_back(std::move(m));
            return true;
        });
    return out;
}

// ---------------------------------------------------------------------------
// Diagnostic queries (for /memory command)
// ---------------------------------------------------------------------------

int RecallService::count_recall_files() const {
    if (!db_) return 0;
    int count = 0;
    db_->query("SELECT COUNT(*) FROM recall_files", {},
               [&](sqlite3_stmt* stmt) {
                   count = sqlite3_column_int(stmt, 0);
                   return true;
               });
    return count;
}

int RecallService::count_recall_segments() const {
    if (!db_) return 0;
    int count = 0;
    db_->query("SELECT COUNT(*) FROM recall_segments", {},
               [&](sqlite3_stmt* stmt) {
                   count = sqlite3_column_int(stmt, 0);
                   return true;
               });
    return count;
}

std::vector<RecallFile> RecallService::list_recall_files() const {
    std::vector<RecallFile> out;
    if (!db_) return out;
    db_->query(
        "SELECT id, title, content, summary, created_at, last_recalled_at, recall_count "
        "FROM recall_files ORDER BY id",
        {},
        [&](sqlite3_stmt* stmt) {
            RecallFile f;
            f.id                = sqlite3_column_int64(stmt, 0);
            const unsigned char* t = sqlite3_column_text(stmt, 1);
            const unsigned char* c = sqlite3_column_text(stmt, 2);
            const unsigned char* s = sqlite3_column_text(stmt, 3);
            if (t) f.title   = reinterpret_cast<const char*>(t);
            if (c) f.content = reinterpret_cast<const char*>(c);
            if (s) f.summary = reinterpret_cast<const char*>(s);
            f.created_at        = sqlite3_column_int64(stmt, 4);
            f.last_recalled_at  = sqlite3_column_int64(stmt, 5);
            f.recall_count      = sqlite3_column_int(stmt, 6);
            out.push_back(std::move(f));
            return true;
        });
    return out;
}

std::vector<RecallSegment> RecallService::list_recall_segments() const {
    std::vector<RecallSegment> out;
    if (!db_) return out;
    db_->query(
        "SELECT id, recall_file_id, content, created_at "
        "FROM recall_segments ORDER BY id",
        {},
        [&](sqlite3_stmt* stmt) {
            RecallSegment s;
            s.id              = sqlite3_column_int64(stmt, 0);
            s.recall_file_id  = sqlite3_column_int64(stmt, 1);
            const unsigned char* c = sqlite3_column_text(stmt, 2);
            if (c) s.content  = reinterpret_cast<const char*>(c);
            s.created_at      = sqlite3_column_int64(stmt, 3);
            out.push_back(std::move(s));
            return true;
        });
    return out;
}

} // namespace memory
} // namespace winefox
