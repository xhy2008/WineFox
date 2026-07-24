cmake_minimum_required(VERSION 3.20)

project(winefox C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MSVC)
    add_compile_options(/utf-8 /permissive- /W3)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS NOMINMAX WIN32_LEAN_AND_MEAN)
    # Reduce noise from llama.cpp / sqlite
    add_compile_options(/wd4244 /wd4267 /wd4305 /wd4996)
else()
    add_compile_options(-Wall -Wno-unused-parameter)
endif()

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(${CMAKE_SOURCE_DIR}/cmake/WineFoxOptions.cmake)

# ---------------------------------------------------------------------------
# Third-party: llama.cpp (git submodule, pinned to release tag b10069)
# ---------------------------------------------------------------------------
# Build everything static so we can link the whole stack into one binary.
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
# Disable the bits we do not need; LLAMA_STANDALONE is OFF automatically when
# added via add_subdirectory, so most of these default OFF already, but we make
# it explicit to avoid surprises across llama.cpp versions.
# LLAMA_BUILD_COMMON is ON: we need common_chat_templates_apply to pass
# enable_thinking=false to the model's jinja chat template (equivalent to
# `llama-cli -rea off`), which prevents Qwen3.5 from emitting <think> blocks.
set(LLAMA_BUILD_COMMON    ON CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS     OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS     OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES  OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_APP       OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_MTMD      ON CACHE BOOL "" FORCE)
set(LLAMA_BUILD_UI        OFF CACHE BOOL "" FORCE)
set(LLAMA_USE_PREBUILT_UI OFF CACHE BOOL "" FORCE)
set(LLAMA_CURL            OFF CACHE BOOL "" FORCE)
set(LLAMA_OPENSSL         OFF CACHE BOOL "" FORCE)
set(LLAMA_LLGUIDANCE      OFF CACHE BOOL "" FORCE)
add_subdirectory(third_party/llama.cpp)

# ---------------------------------------------------------------------------
# Third-party: SQLite amalgamation
# ---------------------------------------------------------------------------
add_library(winefox_sqlite STATIC third_party/sqlite/sqlite3.c)
target_include_directories(winefox_sqlite PUBLIC third_party/sqlite)
if(WIN32)
    target_link_libraries(winefox_sqlite PUBLIC ws2_32)
endif()

# ---------------------------------------------------------------------------
# Third-party: nlohmann/json (header-only)
# ---------------------------------------------------------------------------
# Use the json copy shipped with llama.cpp's vendor directory so that the
# version matches what llama-common expects (avoids JSON_SKIP_LIBRARY_VERSION_CHECK).
add_library(winefox_json INTERFACE)
target_include_directories(winefox_json INTERFACE third_party/llama.cpp/vendor)

# ---------------------------------------------------------------------------
# winefox_util: logging macros + small string/utf8/time helpers (header-only)
# ---------------------------------------------------------------------------
add_library(winefox_util INTERFACE)
target_include_directories(winefox_util INTERFACE src)

# ---------------------------------------------------------------------------
# winefox_core: cross-platform core services (Phase 1+)
# ---------------------------------------------------------------------------
# Source list is appended by each module's CMakeLists fragment.
set(WINEFOX_CORE_SOURCES
    src/core/storage/sqlite_db.cpp
    src/core/storage/migration.cpp
    src/core/config/config.cpp
    src/core/embedder/embedder_service.cpp
    src/core/memory/short_term.cpp
    src/core/memory/recall.cpp
    src/core/memory/distiller.cpp
    src/core/llm/llm_service.cpp
    src/core/llm/vision_service.cpp
    src/core/pipeline/conversation_manager.cpp
)

# Only define winefox_core / winefox_cli once every Phase 1 source file exists.
# This lets the skeleton build succeed before Phase 1 code is written.
set(_winefox_core_all_exist TRUE)
foreach(_f ${WINEFOX_CORE_SOURCES} src/app/cli_main.cpp)
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${_f}")
        set(_winefox_core_all_exist FALSE)
        break()
    endif()
endforeach()
set(WINEFOX_CORE_SOURCES_AND_HEADERS_EXIST ${_winefox_core_all_exist})
if(NOT _winefox_core_all_exist)
    message(STATUS "WineFox: Phase 1 sources incomplete; building skeleton only (llama.cpp + sqlite + util).")
endif()

if(WINEFOX_CORE_SOURCES_AND_HEADERS_EXIST)
    add_library(winefox_core STATIC ${WINEFOX_CORE_SOURCES})
    target_link_libraries(winefox_core PUBLIC
        winefox_util
        winefox_sqlite
        winefox_json
        llama
        llama-common
        mtmd
    )
    target_include_directories(winefox_core PUBLIC src)

    if(WINEFOX_VOICE_ENABLED)
        # Phase 2+: sherpa-onnx + webrtc-apm wired in here.
        # target_link_libraries(winefox_core PUBLIC sherpa-onnx ...)
    endif()
endif()

# ---------------------------------------------------------------------------
# winefox_cli: text conversation entry point
# ---------------------------------------------------------------------------
if(WINEFOX_BUILD_CLI AND WINEFOX_CORE_SOURCES_AND_HEADERS_EXIST)
    add_executable(winefox_cli src/app/cli_main.cpp)
    target_link_libraries(winefox_cli PRIVATE winefox_core)
    if(WIN32)
        target_link_libraries(winefox_cli PRIVATE userenv)
    endif()
    set_target_properties(winefox_cli PROPERTIES OUTPUT_NAME winefox)
endif()

# ---------------------------------------------------------------------------
# Tests (Phase 1 MVP skips these; enable with -DWINEFOX_BUILD_TESTS=ON later)
# ---------------------------------------------------------------------------
if(WINEFOX_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
