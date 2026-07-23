#pragma once

// Schema version management. On startup MemoryService calls migrate() to make
// sure every table exists; future schema bumps apply incremental ALTERs here.

#include "sqlite_db.h"

namespace winefox {
namespace storage {

class Migration {
public:
    // Idempotent: creates tables if missing, records schema_version in meta.
    // Returns false on failure.
    static bool migrate(SqliteDb& db);

    // Current schema version this binary expects.
    static int current_version() { return 1; }
};

} // namespace storage
} // namespace winefox
