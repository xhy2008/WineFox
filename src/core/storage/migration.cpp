#include "migration.h"

#include "../log/log.h"

namespace winefox {
namespace storage {

// schema.sql content inlined so we do not depend on a runtime file path.
// Keep this in sync with src/core/storage/schema.sql.
namespace {

const char* kSchemaSql = R"SQL(
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS profile (
    key        TEXT PRIMARY KEY,
    value      TEXT NOT NULL,
    updated_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS recall_files (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    title           TEXT NOT NULL,
    content         TEXT NOT NULL,
    summary         TEXT,
    embedding       BLOB,
    created_at      INTEGER NOT NULL,
    last_recalled_at INTEGER,
    recall_count    INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS recall_segments (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    recall_file_id INTEGER NOT NULL,
    content       TEXT NOT NULL,
    embedding     BLOB,
    created_at    INTEGER NOT NULL,
    FOREIGN KEY (recall_file_id) REFERENCES recall_files(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_recall_segments_file
    ON recall_segments(recall_file_id);

CREATE TABLE IF NOT EXISTS raw_messages (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    role       TEXT NOT NULL,
    content    TEXT NOT NULL,
    emotion    TEXT,
    created_at INTEGER NOT NULL,
    session_id INTEGER NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_raw_messages_session
    ON raw_messages(session_id, id);

CREATE TABLE IF NOT EXISTS sessions (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at INTEGER NOT NULL,
    ended_at   INTEGER
);
)SQL";

} // namespace

bool Migration::migrate(SqliteDb& db) {
    if (!db.exec(kSchemaSql)) {
        WF_LOG_ERROR("Migration: failed to apply base schema");
        return false;
    }
    // Record the schema version (idempotent upsert).
    db.exec("INSERT INTO meta(key, value) VALUES('schema_version', '" +
            std::to_string(current_version()) +
            "') ON CONFLICT(key) DO UPDATE SET value=excluded.value;");
    WF_LOG_DEBUG("Migration: schema at version %d", current_version());
    return true;
}

} // namespace storage
} // namespace winefox
