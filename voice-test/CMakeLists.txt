# voice-test: standalone voice front-end benchmark sandbox (Phase 2).
#
# New architecture (sherpa-onnx-free):
#   VAD  : ten-vad-ggml (pure C++17 port, GGML model — no onnxruntime)
#   ASR  : SenseVoice.cpp (ggml, shares ggml with llama.cpp)
#   TTS  : Kokoro  (onnxruntime, G2P frontend from kokoro.cpp)
#
# This project is self-contained: it pulls in its own third-party deps
# via add_subdirectory or find_package, and produces a `voice_test`
# executable with subcommands: smoke, vad, asr, tts, stream.
#
# Build (from winefox root):
#   cmake -B build-voice-test -DWINEFOX_BUILD_VOICE_TEST=ON -DCMAKE_BUILD_TYPE=Release
#   cmake --build build-voice-test --config Release --target voice_test
#
# Stages are implemented incrementally; the current skeleton provides the
# smoke subcommand only. Each component (vad/asr/tts/stream) is gated
# behind its own option so partial implementations can be built and
# tested independently.

cmake_minimum_required(VERSION 3.20)
project(voice_test C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MSVC)
    add_compile_options(/utf-8 /permissive- /W3)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS NOMINMAX WIN32_LEAN_AND_MEAN _USE_MATH_DEFINES)
    add_compile_options(/wd4244 /wd4267 /wd4305)

    # Kokoro's prebuilt onnxruntime.lib is compiled with /MT (static runtime).
    # When TTS is enabled we must compile voice_test and all its sources with
    # the same static runtime, otherwise we get LNK2038 RuntimeLibrary mismatch.
    # This only affects TTS-enabled builds; VAD/ASR-only builds stay on the
    # default /MD (dynamic runtime) for compatibility with llama.cpp/ggml.
    if(VOICE_TEST_ENABLE_TTS)
        set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded")
        message(STATUS "VOICE_TEST_ENABLE_TTS=ON: forcing /MT (static runtime) for onnxruntime.lib compatibility")
    endif()
else()
    add_compile_options(-Wall -Wno-unused-parameter)
endif()

# ---------------------------------------------------------------------------
# Component toggles. Each can be turned on once its deps are wired in.
# ---------------------------------------------------------------------------
option(VOICE_TEST_ENABLE_VAD  "Enable ten-vad-ggml VAD benchmark"          OFF)
option(VOICE_TEST_ENABLE_ASR  "Enable SenseVoice.cpp ASR benchmark"        OFF)
option(VOICE_TEST_ENABLE_TTS  "Enable Kokoro TTS benchmark"                OFF)
option(VOICE_TEST_ENABLE_STREAM "Enable full streaming VAD+ASR benchmark" OFF)

# NOTE: VOICE_TEST_ENABLE_TTS_GGML has been removed. The ggml-based Kokoro
# decoder was deleted on 2026-07-27 because ggml is not optimized for the
# conv-heavy iSTFTNet decoder (RTF=3.07 vs ONNX RTF=0.73).
# Use VOICE_TEST_ENABLE_TTS (ONNX Runtime split mode) instead.

# ---------------------------------------------------------------------------
# Third-party deps. Resolved lazily by each component; declared here so
# the option summary is centralized.
#
#   ten-vad-ggml : git submodule at third_party/ten-vad-ggml (TEN_VAD_GGML_ROOT).
#                  Pure C++17 reimplementation; reads a GGML model file
#                  (models/vad/ten-vad-ggml.bin) at runtime. No onnxruntime,
#                  no DLL, no shared runtime dependency.
#   onnxruntime  : prebuilt package pointed by ONNXRUNTIME_ROOT.
#                  Required by TTS (Kokoro) only.
#   SenseVoice.cpp : add_subdirectory(third_party/sensevoice-cpp)
#                    (shares ggml with llama.cpp; see notes below)
#   Kokoro G2P   : header-only sources vendored under third_party/kokoro-cpp
# ---------------------------------------------------------------------------
set(TEN_VAD_GGML_ROOT "" CACHE PATH "Path to TEN-VAD-GGML.cpp source root (contains cpp/CMakeLists.txt). Empty = use bundled third_party/ten-vad-ggml")
set(ONNXRUNTIME_ROOT "" CACHE PATH "Path to onnxruntime prebuilt package root (TTS only; contains include/ and lib/)")
set(SENSEVOICE_ROOT   "" CACHE PATH "Path to SenseVoice.cpp source root. Empty = use bundled third_party/sensevoice-cpp")

# Fall back to the bundled copy when the cache value is empty.
# The ten-vad-ggml submodule lives next to voice-test/ (sibling), i.e.
# <winefox>/third_party/ten-vad-ggml, whether this CMakeLists is included
# from the winefox root or configured standalone from voice-test/.
if(NOT TEN_VAD_GGML_ROOT)
    set(TEN_VAD_GGML_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/ten-vad-ggml")
endif()
if(NOT SENSEVOICE_ROOT)
    set(SENSEVOICE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/third_party/sensevoice-cpp")
endif()

# ---------------------------------------------------------------------------
# voice_test executable. Source list grows as components are enabled.
# ---------------------------------------------------------------------------
add_executable(voice_test
    src/main.cpp
    src/smoke.cpp
)

target_include_directories(voice_test PRIVATE src)

# ---------------------------------------------------------------------------
# Component: VAD (TEN-VAD-GGML — pure C++17 port, GGML model)
#   Vendored as a git submodule at third_party/ten-vad-ggml. The library
#   loads a GGML model file (models/vad/ten-vad-ggml.bin) at runtime; no
#   DLL and no onnxruntime. We build the static lib and link it directly.
# ---------------------------------------------------------------------------
if(VOICE_TEST_ENABLE_VAD)
    if(NOT EXISTS "${TEN_VAD_GGML_ROOT}/cpp/CMakeLists.txt")
        message(FATAL_ERROR "VOICE_TEST_ENABLE_VAD: TEN-VAD-GGML.cpp not found under TEN_VAD_GGML_ROOT=${TEN_VAD_GGML_ROOT}")
    endif()

    set(TEN_VAD_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(TEN_VAD_BUILD_CLI   OFF CACHE BOOL "" FORCE)
    add_subdirectory(${TEN_VAD_GGML_ROOT}/cpp ${CMAKE_BINARY_DIR}/ten-vad-ggml EXCLUDE_FROM_ALL)

    target_sources(voice_test PRIVATE src/vad_test.cpp)
    target_include_directories(voice_test PRIVATE ${TEN_VAD_GGML_ROOT}/cpp/include)
    target_link_libraries(voice_test PRIVATE ten_vad_ggml)
    target_compile_definitions(voice_test PRIVATE VOICE_TEST_HAS_VAD=1)
endif()

# ---------------------------------------------------------------------------
# Component: ASR (SenseVoice.cpp + ggml)
#   SenseVoice.cpp ships its own ggml fork under sense-voice/csrc/third-party/.
#   For now we build it as a standalone static subproject (sense-voice-core +
#   ggml). A future refactor will redirect its ggml dependency to llama.cpp's
#   ggml so both share one ggml.dll.
# ---------------------------------------------------------------------------
if(VOICE_TEST_ENABLE_ASR)
    if(NOT EXISTS "${SENSEVOICE_ROOT}/CMakeLists.txt")
        message(FATAL_ERROR "VOICE_TEST_ENABLE_ASR: SenseVoice.cpp CMakeLists.txt not found under SENSEVOICE_ROOT=${SENSEVOICE_ROOT}")
    endif()

    # SenseVoice.cpp option overrides: build only the static lib, no examples,
    # no tests, no shared libs. Force these before add_subdirectory so the
    # nested ggml build picks them up too.
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
    set(SENSE_VOICE_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(SENSE_VOICE_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
    set(SENSE_VOICE_FATAL_WARNINGS OFF CACHE BOOL "" FORCE)
    set(GGML_BLAS                  OFF CACHE BOOL "" FORCE)

    add_subdirectory(${SENSEVOICE_ROOT} ${CMAKE_BINARY_DIR}/sensevoice-cpp EXCLUDE_FROM_ALL)

    target_sources(voice_test PRIVATE src/asr_test.cpp)
    target_include_directories(voice_test PRIVATE
        ${SENSEVOICE_ROOT}/sense-voice/csrc
    )
    target_link_libraries(voice_test PRIVATE sense-voice-core ggml)
    target_compile_definitions(voice_test PRIVATE VOICE_TEST_HAS_ASR=1)
endif()

# ---------------------------------------------------------------------------
# Component: TTS (Kokoro ONNX + G2P frontend from kokoro.cpp)
#
#   kokoro.cpp (https://github.com/koth/kokoro.cpp) provides:
#     - G2P frontend: ZHFrontend, ZHG2P, JiebaProcessor, PinyinFinder,
#       ToneSandhi, EnG2P, Tokenizer
#     - Kokoro class: orchestrates G2P + ONNX inference
#     - dict/ data files: jieba dict, pinyin tables, cmudict, vocab.txt
#
#   We compile the kokoro sources directly into voice_test (no separate
#   static lib) so we share the same compiler settings and runtime library.
#   onnxruntime is linked as a static lib (shipped in the kokoro.cpp repo
#   as a zip under third_party/onnxruntime/lib/windows_x64/).
#
#   The dict/ directory must be accessible at runtime (relative to CWD or
#   pointed to by --dict-dir). We copy it next to the executable.
# ---------------------------------------------------------------------------
if(VOICE_TEST_ENABLE_TTS)
    set(KOKORO_SRC_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/third_party/kokoro-cpp-src")

    if(NOT EXISTS "${KOKORO_SRC_ROOT}/Kokoro.cpp")
        message(FATAL_ERROR "VOICE_TEST_ENABLE_TTS: kokoro.cpp sources not found under ${KOKORO_SRC_ROOT}")
    endif()

    # onnxruntime: use the prebuilt static lib.
    # Looks for onnxruntime.lib under ONNXRUNTIME_ROOT (either directly or in lib/).
    if(NOT ONNXRUNTIME_ROOT)
        set(ONNXRUNTIME_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime-static")
    endif()
    set(_ORT_LIB "${ONNXRUNTIME_ROOT}/onnxruntime.lib")
    if(NOT EXISTS "${_ORT_LIB}")
        set(_ORT_LIB "${ONNXRUNTIME_ROOT}/lib/onnxruntime.lib")
    endif()
    if(NOT EXISTS "${_ORT_LIB}")
        message(FATAL_ERROR "VOICE_TEST_ENABLE_TTS: onnxruntime.lib not found under ${ONNXRUNTIME_ROOT}. "
            "Download static lib from https://github.com/csukuangfj/onnxruntime-libs/releases")
    endif()

    # Kokoro G2P + engine sources (compiled directly into voice_test).
    target_sources(voice_test PRIVATE
        src/tts_test.cpp
        ${KOKORO_SRC_ROOT}/Kokoro.cpp
        ${KOKORO_SRC_ROOT}/Tokenizer.cpp
        ${KOKORO_SRC_ROOT}/ZHG2P.cpp
        ${KOKORO_SRC_ROOT}/PinyinFinder.cpp
        ${KOKORO_SRC_ROOT}/ToneSandhi.cpp
        ${KOKORO_SRC_ROOT}/ZHFrontend.cpp
    )

    target_include_directories(voice_test PRIVATE
        ${KOKORO_SRC_ROOT}
        ${KOKORO_SRC_ROOT}/third_party/cppjieba/include
        ${KOKORO_SRC_ROOT}/third_party/cppjieba/deps/limonp/include
        ${ONNXRUNTIME_ROOT}/include
    )

    if(WIN32)
        target_link_libraries(voice_test PRIVATE ${_ORT_LIB})
        # onnxruntime static lib needs these Windows system libs
        target_link_libraries(voice_test PRIVATE
            Advapi32 Ws2_32 Shlwapi userenv Synchronization
        )
    else()
        find_library(ONNXRUNTIME_LIB onnxruntime HINTS ${ONNXRUNTIME_ROOT})
        target_link_libraries(voice_test PRIVATE ${ONNXRUNTIME_LIB})
    endif()

    # Copy the dict/ directory next to the executable so G2P can find
    # jieba.dict.utf8, pinyin.txt, etc. at runtime.
    add_custom_command(TARGET voice_test POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
                "${KOKORO_SRC_ROOT}/dict" "$<TARGET_FILE_DIR:voice_test>/dict"
        COMMENT "Copying kokoro dict/ to output directory")

    target_compile_definitions(voice_test PRIVATE VOICE_TEST_HAS_TTS=1)
endif()

# ---------------------------------------------------------------------------
# Component: full streaming pipeline (VAD + ASR)
# ---------------------------------------------------------------------------
if(VOICE_TEST_ENABLE_STREAM)
    if(NOT VOICE_TEST_ENABLE_VAD)
        message(FATAL_ERROR "VOICE_TEST_ENABLE_STREAM requires VOICE_TEST_ENABLE_VAD")
    endif()
    if(NOT VOICE_TEST_ENABLE_ASR)
        message(FATAL_ERROR "VOICE_TEST_ENABLE_STREAM requires VOICE_TEST_ENABLE_ASR")
    endif()
    target_sources(voice_test PRIVATE src/stream_test.cpp)
    target_compile_definitions(voice_test PRIVATE VOICE_TEST_HAS_STREAM=1)
endif()

# ---------------------------------------------------------------------------
# Output to <binary_dir>/voice-test for consistency with old layout.
# ---------------------------------------------------------------------------
set_target_properties(voice_test PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/voice-test"
)

message(STATUS "voice-test components:")
message(STATUS "  VOICE_TEST_ENABLE_VAD        = ${VOICE_TEST_ENABLE_VAD}")
message(STATUS "  VOICE_TEST_ENABLE_ASR        = ${VOICE_TEST_ENABLE_ASR}")
message(STATUS "  VOICE_TEST_ENABLE_TTS        = ${VOICE_TEST_ENABLE_TTS}")
message(STATUS "  VOICE_TEST_ENABLE_STREAM     = ${VOICE_TEST_ENABLE_STREAM}")
message(STATUS "voice-test binary -> ${CMAKE_BINARY_DIR}/voice-test")
