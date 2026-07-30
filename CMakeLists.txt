cmake_minimum_required(VERSION 3.20)

project(winefox C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MSVC)
    add_compile_options(/utf-8 /permissive- /W3)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS NOMINMAX WIN32_LEAN_AND_MEAN _USE_MATH_DEFINES)
    # Reduce noise from llama.cpp / sqlite
    add_compile_options(/wd4244 /wd4267 /wd4305 /wd4996)
else()
    add_compile_options(-Wall -Wno-unused-parameter)
endif()

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(${CMAKE_SOURCE_DIR}/cmake/WineFoxOptions.cmake)

# ---------------------------------------------------------------------------
# MSVC runtime library: when building the 3D World frontend OR the core DLL
# that the world exe links against, we must use the static runtime (/MT)
# because Kokoro's prebuilt onnxruntime.lib is compiled with /MT. This must
# be set BEFORE any add_subdirectory so all targets (llama.cpp, winefox_core,
# SDL3, etc.) inherit the same runtime. The final exe must not depend on the
# VC++ redistributable.
# ---------------------------------------------------------------------------
if((WINEFOX_BUILD_WORLD OR WINEFOX_BUILD_CORE_DLL) AND MSVC)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded")
    message(STATUS "WINEFOX_BUILD_WORLD/CORE_DLL=ON: forcing /MT (static runtime)")
endif()

# ---------------------------------------------------------------------------
# Third-party: llama.cpp (git submodule, pinned to release tag b10069)
#
# GATED: when WINEFOX_BUILD_WORLD=ON we do NOT add llama.cpp here, because
# the world exe links SenseVoice.cpp (which vendors its own ggml fork).
# Both define the same CMake targets (ggml-base, ggml, ggml-cpu) and
# CMake fails with CMP0002. The world exe instead links a pre-built
# winefox_core.dll (built separately with WINEFOX_BUILD_CORE_DLL=ON) whose
# ggml symbols are hidden inside the DLL.
# ---------------------------------------------------------------------------
if(NOT WINEFOX_BUILD_WORLD)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
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
endif() # NOT WINEFOX_BUILD_WORLD

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

if(WINEFOX_CORE_SOURCES_AND_HEADERS_EXIST AND NOT WINEFOX_BUILD_WORLD)
    # When WINEFOX_BUILD_CORE_DLL=ON, build as SHARED with C ABI exports so
    # the world exe can link it without pulling in llama.cpp's ggml targets.
    # The C ABI (winefox_api.h) is the only exported surface; all ggml/llama
    # symbols remain internal to the DLL, avoiding conflicts with
    # SenseVoice.cpp's vendored ggml in the world exe.
    if(WINEFOX_BUILD_CORE_DLL)
        add_library(winefox_core SHARED ${WINEFOX_CORE_SOURCES} src/core/api/winefox_api.cpp)
        target_compile_definitions(winefox_core PRIVATE WINEFOX_CORE_EXPORTS)
        # Do NOT use WINDOWS_EXPORT_ALL_SYMBOLS — we want only the explicit
        # WF_API functions exported, hiding all ggml/llama internals.
        set_target_properties(winefox_core PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS OFF)
    else()
        add_library(winefox_core STATIC ${WINEFOX_CORE_SOURCES})
    endif()

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
        # Phase 2+: voice front-end (onnxruntime VAD/TTS + ggml ASR) wired in here.
        # target_link_libraries(winefox_core PUBLIC vad tts sensevoice ...)
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

# ---------------------------------------------------------------------------
# winefox-world: 3D World frontend (Vulkan + SDL3, Phase 4, Windows only)
# Built only when explicitly requested via -DWINEFOX_BUILD_WORLD=ON.
# Pulls in SDL3, voice deps (ten-vad, SenseVoice, Kokoro), and winefox_core.
# ---------------------------------------------------------------------------
if(WINEFOX_BUILD_WORLD)
    add_subdirectory(src/world)
endif()

# ---------------------------------------------------------------------------
# voice-test: standalone voice front-end benchmark sandbox (Phase 2).
# Independent of the winefox_core build; pulls in its own deps
# (onnxruntime, ten-vad, SenseVoice.cpp, Kokoro) via its own CMakeLists.
# Built only when explicitly requested to keep the heavy deps opt-in.
# ---------------------------------------------------------------------------
if(WINEFOX_BUILD_VOICE_TEST)
    add_subdirectory(voice-test)
endif()
