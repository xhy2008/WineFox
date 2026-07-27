# voice-test: standalone voice front-end benchmark sandbox (Phase 2).
#
# New architecture (sherpa-onnx-free):
#   VAD  : ten-vad  (onnxruntime, official prebuilt lib)
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
option(VOICE_TEST_ENABLE_VAD  "Enable ten-vad VAD benchmark"               OFF)
option(VOICE_TEST_ENABLE_ASR  "Enable SenseVoice.cpp ASR benchmark"        OFF)
option(VOICE_TEST_ENABLE_TTS  "Enable Kokoro TTS benchmark"                OFF)
option(VOICE_TEST_ENABLE_STREAM "Enable full streaming VAD+ASR benchmark" OFF)

# NOTE: VOICE_TEST_ENABLE_TTS_GGML has been removed. The ggml-based Kokoro
# decoder was archived to _archive_ggml/ on 2026-07-27 because ggml is not
# optimized for the conv-heavy iSTFTNet decoder (RTF=3.07 vs ONNX RTF=0.73).
# Use VOICE_TEST_ENABLE_TTS (ONNX Runtime split mode) instead.

# ---------------------------------------------------------------------------
# Third-party deps. Resolved lazily by each component; declared here so
# the option summary is centralized.
#
#   ten-vad      : prebuilt lib pointed by TEN_VAD_ROOT.
#                  The official prebuilt DLL is SELF-CONTAINED: the ONNX
#                  model + onnxruntime are baked into ten_vad.dll, so VAD
#                  does NOT need a separate onnxruntime install.
#   onnxruntime  : prebuilt package pointed by ONNXRUNTIME_ROOT.
#                  Required by TTS (Kokoro) only.
#   SenseVoice.cpp : add_subdirectory(third_party/sensevoice-cpp)
#                    (shares ggml with llama.cpp; see notes below)
#   Kokoro G2P   : header-only sources vendored under third_party/kokoro-cpp
# ---------------------------------------------------------------------------
set(TEN_VAD_ROOT      "" CACHE PATH "Path to ten-vad prebuilt package root (contains include/ and lib/). Empty = use bundled third_party/ten-vad")
set(ONNXRUNTIME_ROOT "" CACHE PATH "Path to onnxruntime prebuilt package root (TTS only; contains include/ and lib/)")
set(SENSEVOICE_ROOT   "" CACHE PATH "Path to SenseVoice.cpp source root. Empty = use bundled third_party/sensevoice-cpp")

# Fall back to the bundled prebuilt copy when the cache value is empty.
if(NOT TEN_VAD_ROOT)
    set(TEN_VAD_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/third_party/ten-vad")
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
# Component: VAD (ten-vad official prebuilt lib, self-contained)
#   The prebuilt ten_vad.dll embeds the ONNX model + onnxruntime, so no
#   external onnxruntime install is needed for VAD. We only link the
#   import lib and copy the DLL next to the executable at build time.
# ---------------------------------------------------------------------------
if(VOICE_TEST_ENABLE_VAD)
    if(NOT EXISTS "${TEN_VAD_ROOT}/include/ten_vad.h")
        message(FATAL_ERROR "VOICE_TEST_ENABLE_VAD: ten_vad.h not found under TEN_VAD_ROOT=${TEN_VAD_ROOT}")
    endif()

    target_sources(voice_test PRIVATE src/vad_test.cpp)
    target_include_directories(voice_test PRIVATE ${TEN_VAD_ROOT}/include)

    if(WIN32)
        if(NOT EXISTS "${TEN_VAD_ROOT}/lib/ten_vad.lib")
            message(FATAL_ERROR "VOICE_TEST_ENABLE_VAD: ten_vad.lib not found under ${TEN_VAD_ROOT}/lib")
        endif()
        target_link_libraries(voice_test PRIVATE ${TEN_VAD_ROOT}/lib/ten_vad.lib)
        # Copy the self-contained DLL next to the executable so it loads at runtime.
        set(_ten_vad_dll "${TEN_VAD_ROOT}/lib/ten_vad.dll")
        add_custom_command(TARGET voice_test POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${_ten_vad_dll}" "$<TARGET_FILE_DIR:voice_test>")
    else()
        if(NOT EXISTS "${TEN_VAD_ROOT}/lib/libten_vad.so")
            message(FATAL_ERROR "VOICE_TEST_ENABLE_VAD: libten_vad.so not found under ${TEN_VAD_ROOT}/lib")
        endif()
        target_link_libraries(voice_test PRIVATE ${TEN_VAD_ROOT}/lib/libten_vad.so)
        set_target_properties(voice_test PROPERTIES
            BUILD_RPATH "${TEN_VAD_ROOT}/lib")
    endif()

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

    # onnxruntime: use the prebuilt static lib extracted from kokoro.cpp's zip
    if(NOT ONNXRUNTIME_ROOT)
        set(ONNXRUNTIME_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime-prebuilt")
    endif()
    if(NOT EXISTS "${ONNXRUNTIME_ROOT}/onnxruntime.lib")
        message(FATAL_ERROR "VOICE_TEST_ENABLE_TTS: onnxruntime.lib not found under ${ONNXRUNTIME_ROOT}. "
            "Extract kokoro-cpp-src/third_party/onnxruntime/lib/windows_x64/onnxruntime.zip first.")
    endif()

    # Kokoro G2P + engine sources (compiled directly into voice_test).
    # NOTE: missing_symbols.cpp is intentionally excluded — it provides
    # __std_find_first_of_trivial_pos_{1,2} which are already present in
    # libcpmt.lib when building with /MT (static runtime), so including it
    # causes LNK2005 duplicate symbol errors.
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
        ${KOKORO_SRC_ROOT}/third_party/onnxruntime/include
        ${ONNXRUNTIME_ROOT}/../kokoro-cpp-src/third_party/onnxruntime/include
    )

    if(WIN32)
        target_link_libraries(voice_test PRIVATE ${ONNXRUNTIME_ROOT}/onnxruntime.lib)
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
