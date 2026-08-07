#pragma once

// Thin RAII wrapper around a SQLite handle. Provides exec() for statements
// without result rows and a bound-parameter query() helper that streams rows
// to a callback. Keeps the API surface small; call sites use sqlite3_stmt*
// directly for column access via sqlite3_column_*.

#include <functional>
#include <sqlite3.h>
#include <string>
#include <vector>

namespace winefox {
namespace storage {

// A single bind parameter. Set is_blob=true when `data` holds raw binary
// (e.g. an embedding vector); it is then bound with sqlite3_bind_blob so the
// value is stored/retrieved as BLOB instead of being truncated at the first
// null byte like sqlite3_bind_text would.
struct SqliteParam {
    std::string data;
    bool        is_blob = false;

    SqliteParam() = default;
    SqliteParam(const std::string& d, bool blob = false) : data(d), is_blob(blob) {}
    SqliteParam(const char* d, bool blob = false) : data(d ? d : ""), is_blob(blob) {}
};

class SqliteDb {
public:
    SqliteDb() = default;
    ~SqliteDb();
    SqliteDb(const SqliteDb&) = delete;
    SqliteDb& operator=(const SqliteDb&) = delete;
    SqliteDb(SqliteDb&& other) noexcept;
    SqliteDb& operator=(SqliteDb&& other) noexcept;

    // Open (or create) a database file. Automatically enables foreign_keys and
    // WAL journal mode. Creates the parent directory if missing.
    bool open(const std::string& path);
    void close();

    // Execute one or more semicolon-separated statements with no result rows.
    // Returns false on error and writes the SQLite message to `err` if present.
    bool exec(const std::string& sql, std::string* err = nullptr);

    // Run a parameterised query. `params` are bound by position (1-based);
    // a param with is_blob=true is bound as raw binary (BLOB).
    // `cb` is invoked for each row; returning false stops iteration.
    // Returns false on prepare/step error.
    using RowCallback = std::function<bool(sqlite3_stmt*)>;
    bool query(const std::string& sql,
               const std::vector<SqliteParam>& params,
               RowCallback cb);

    // Convenience for a single INSERT/UPDATE returning last_insert_rowid().
    // Returns -1 on failure.
    long long insert(const std::string& sql, const std::vector<SqliteParam>& params);

    sqlite3* handle() { return db_; }
    bool is_open() const { return db_ != nullptr; }

private:
    sqlite3* db_ = nullptr;
};

} // namespace storage
} // namespace winefox
