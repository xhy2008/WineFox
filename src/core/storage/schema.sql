-- WineFox memory schema. Aligned with PLAN.md section 3.4.
-- All timestamps are Unix epoch seconds (INTEGER).
-- Embeddings are stored as little-endian float32 BLOBs.

PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;

-- schema_version is tracked in the singleton meta table.
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- Long-term user profile (key/value, e.g. name, preferences, important dates).
CREATE TABLE IF NOT EXISTS profile (
    key        TEXT PRIMARY KEY,
    value      TEXT NOT NULL,
    updated_at INTEGER NOT NULL
);

-- recall_files: a distilled memory "episode" (one distillation run = one file).
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

-- recall_segments: finer-grained chunks under a recall_file for retrieval.
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

-- recall_file_resources: files mentioned in a memory (images, docs, ...).
CREATE TABLE IF NOT EXISTS recall_file_resources (
    recall_file_id INTEGER NOT NULL,
    resource_path  TEXT NOT NULL,
    resource_type  TEXT,
    PRIMARY KEY (recall_file_id, resource_path),
    FOREIGN KEY (recall_file_id) REFERENCES recall_files(id) ON DELETE CASCADE
);

-- raw_messages: append-only conversation log (user + assistant + emotion).
-- Used by the distiller to reconstruct a session for summarisation.
CREATE TABLE IF NOT EXISTS raw_messages (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    role       TEXT NOT NULL,          -- 'user' | 'assistant' | 'system'
    content    TEXT NOT NULL,
    emotion    TEXT,                   -- nullable; assistant replies carry [emotion]
    created_at INTEGER NOT NULL,
    session_id INTEGER NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_raw_messages_session
    ON raw_messages(session_id, id);

-- sessions: a conversation session.
CREATE TABLE IF NOT EXISTS sessions (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at INTEGER NOT NULL,
    ended_at   INTEGER
);
