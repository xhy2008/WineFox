#include "sqlite_db.h"

#include "../log/log.h"

#include <cstdio>
#include <sys/stat.h>

#ifdef _WIN32
  #include <direct.h>
  #define MKDIR(p) _mkdir(p)
#else
  #include <unistd.h>
  #define MKDIR(p) mkdir(p, 0755)
#endif

namespace winefox {
namespace storage {

namespace {

// Create parent directory of `path` if it does not exist.
void ensure_parent_dir(const std::string& path) {
    size_t slash = path.find_last_of("/\\");
    if (slash == std::string::npos) return;
    std::string dir = path.substr(0, slash);
    if (dir.empty()) return;
    struct stat info;
    if (stat(dir.c_str(), &info) != 0) {
        // Recursively create (simple two-level attempt is enough for our paths).
        size_t sub = dir.find_last_of("/\\");
        if (sub != std::string::npos) {
            std::string parent = dir.substr(0, sub);
            if (!parent.empty() && stat(parent.c_str(), &info) != 0) {
                MKDIR(parent.c_str());
            }
        }
        MKDIR(dir.c_str());
    }
}

} // namespace

SqliteDb::~SqliteDb() { close(); }

SqliteDb::SqliteDb(SqliteDb&& other) noexcept : db_(other.db_) { other.db_ = nullptr; }

SqliteDb& SqliteDb::operator=(SqliteDb&& other) noexcept {
    if (this != &other) {
        close();
        db_ = other.db_;
        other.db_ = nullptr;
    }
    return *this;
}

bool SqliteDb::open(const std::string& path) {
    close();
    ensure_parent_dir(path);
    int rc = sqlite3_open(path.c_str(), &db_);
    if (rc != SQLITE_OK) {
        WF_LOG_ERROR("sqlite3_open failed: %s (%s)", sqlite3_errmsg(db_), path.c_str());
        sqlite3_close(db_);
        db_ = nullptr;
        return false;
    }
    // Recommended pragmas (PLAN.md 9: PRAGMA foreign_keys = ON).
    exec("PRAGMA foreign_keys = ON;");
    exec("PRAGMA journal_mode = WAL;");
    exec("PRAGMA synchronous = NORMAL;");
    // 50ms busy timeout so concurrent distiller writes do not fail instantly.
    exec("PRAGMA busy_timeout = 50000;");
    return true;
}

void SqliteDb::close() {
    if (db_) {
        sqlite3_close(db_);
        db_ = nullptr;
    }
}

bool SqliteDb::exec(const std::string& sql, std::string* err) {
    if (!db_) return false;
    char* msg = nullptr;
    int rc = sqlite3_exec(db_, sql.c_str(), nullptr, nullptr, &msg);
    if (rc != SQLITE_OK) {
        if (err) *err = msg ? msg : "unknown error";
        WF_LOG_ERROR("sqlite exec failed: %s", msg ? msg : "(null)");
        sqlite3_free(msg);
        return false;
    }
    return true;
}

bool SqliteDb::query(const std::string& sql,
                     const std::vector<std::string>& params,
                     RowCallback cb) {
    if (!db_) return false;
    sqlite3_stmt* stmt = nullptr;
    if (sqlite3_prepare_v2(db_, sql.c_str(), -1, &stmt, nullptr) != SQLITE_OK) {
        WF_LOG_ERROR("sqlite prepare failed: %s | sql: %s", sqlite3_errmsg(db_), sql.c_str());
        return false;
    }
    for (size_t i = 0; i < params.size(); ++i) {
        // Use params[i].size() instead of -1 so that embedded NUL bytes are
        // preserved. Embeddings are stored as raw float blobs reinterpreted
        // as std::string, and -1 would truncate at the first '\0'.
        sqlite3_bind_text(stmt, static_cast<int>(i + 1),
                          params[i].data(), static_cast<int>(params[i].size()),
                          SQLITE_TRANSIENT);
    }
    bool ok = true;
    while (true) {
        int rc = sqlite3_step(stmt);
        if (rc == SQLITE_ROW) {
            if (cb && !cb(stmt)) break;
        } else if (rc == SQLITE_DONE) {
            break;
        } else {
            WF_LOG_ERROR("sqlite step failed: %s", sqlite3_errmsg(db_));
            ok = false;
            break;
        }
    }
    sqlite3_finalize(stmt);
    return ok;
}

long long SqliteDb::insert(const std::string& sql, const std::vector<std::string>& params) {
    if (!db_) return -1;
    sqlite3_stmt* stmt = nullptr;
    if (sqlite3_prepare_v2(db_, sql.c_str(), -1, &stmt, nullptr) != SQLITE_OK) {
        WF_LOG_ERROR("sqlite insert prepare failed: %s", sqlite3_errmsg(db_));
        return -1;
    }
    for (size_t i = 0; i < params.size(); ++i) {
        sqlite3_bind_text(stmt, static_cast<int>(i + 1),
                          params[i].data(), static_cast<int>(params[i].size()),
                          SQLITE_TRANSIENT);
    }
    long long rowid = -1;
    int rc = sqlite3_step(stmt);
    if (rc == SQLITE_DONE) {
        rowid = sqlite3_last_insert_rowid(db_);
    } else {
        WF_LOG_ERROR("sqlite insert step failed: %s", sqlite3_errmsg(db_));
    }
    sqlite3_finalize(stmt);
    return rowid;
}

} // namespace storage
} // namespace winefox
