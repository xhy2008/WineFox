#pragma once

// Long-term recall: stores distilled memory episodes and retrieves the most
// relevant ones for the current user query via cosine similarity over
// embeddings. Also owns the profile key/value store.
//
// PLAN.md 3.4: recall() embeds the query, scores recall_segments, aggregates
// to recall_files, and returns the top-k hits. commit_recall_file() batches
// all segment embeddings and inserts them in one transaction.

#include "../embedder/embedder_service.h"
#include "../storage/sqlite_db.h"
#include "message.h"

#include <map>
#include <string>
#include <vector>

namespace winefox {
namespace memory {

class RecallService {
public:
    bool init(storage::SqliteDb* db, embedder::EmbedderService* embedder);

    // Retrieve up to top_k memory hits relevant to `query`.
    std::vector<Recall> recall(const std::string& query, int top_k = 5);

    // Persist a distilled memory episode. Embeds every segment (and the file
    // itself) in one batch, then inserts everything inside a transaction.
    // Returns the new recall_files.id, or -1 on failure.
    long long commit_recall_file(const RecallFile& file);

    // Profile (long-stable user facts: name, preferences, ...).
    void        upsert_profile(const std::string& key, const std::string& value);
    std::string get_profile(const std::string& key) const;
    std::map<std::string, std::string> get_all_profile() const;

    // Append a raw message to the durable log (used by the distiller + CLI).
    long long log_raw_message(const Message& msg, long long session_id);

    // True if the embedder backend is loaded and ready.
    bool embedder_ready() const { return embedder_ && embedder_->ready(); }

    // Start / close a session.
    long long start_session();
    void      close_session(long long session_id);

    // Read raw messages of a session (for the distiller).
    std::vector<Message> get_session_messages(long long session_id) const;

    // --- Diagnostic / debug queries (for /memory command) ---

    // Count of recall_files (distilled episodes).
    int count_recall_files() const;

    // Count of recall_segments (searchable chunks).
    int count_recall_segments() const;

    // Return all recall_files (id, title, summary, created_at) for inspection.
    // Does NOT load embeddings (keeps the result small).
    std::vector<RecallFile> list_recall_files() const;

    // Return all recall_segments (id, file_id, content, created_at) for inspection.
    // Does NOT load embeddings.
    std::vector<RecallSegment> list_recall_segments() const;

private:
    storage::SqliteDb*          db_      = nullptr;
    embedder::EmbedderService*  embedder_ = nullptr;

    // float vector <-> little-endian BLOB
    static std::string            encode_embedding_(const std::vector<float>& v);
    static std::vector<float>     decode_embedding_(const std::string& blob);
    static float                  cosine_(const std::vector<float>& a, const std::vector<float>& b);
};

} // namespace memory
} // namespace winefox
