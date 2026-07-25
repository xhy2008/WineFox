# voice-test: standalone sherpa-onnx integration sandbox.
#
# This project is intentionally separate from winefox.exe so that voice
# front-end experiments (ASR / VAD / TTS) can be built and iterated on
# without touching the main CLI. See PLAN.md Phase 2 / TODO.md 2.x.
#
# Two build modes are supported:
#
#   1. Integrated (preferred when working inside the winefox solution):
#        cmake -B build -DWINEFOX_BUILD_VOICE_TEST=ON
#        cmake --build build --target voice_test
#      The root CMakeLists.txt includes this directory and shares the
#      sherpa-onnx targets already added there.
#
#   2. Standalone (for isolated experiments):
#        cmake -B voice-test/build -S voice-test
#        cmake --build voice-test/build
#      This file adds third_party/sherpa-onnx itself when CMAKE_SOURCE_DIR
#      points at voice-test, so the project can be configured without the
#      winefox root.
#
# Default model paths assume the winefox layout:
#   models/asr/sense-voice-small-int8.onnx
#   models/asr/tokens.txt
#   models/vad/ten-vad.onnx
# Override at runtime with --asr-model / --asr-tokens / --vad-model.

cmake_minimum_required(VERSION 3.20)

# ---- standalone mode bootstrap -------------------------------------------
# When configured directly (cmake -S voice-test), pull in sherpa-onnx here so
# the project is self-contained. In integrated mode the root CMakeLists.txt
# has already added sherpa-onnx before include()-ing this file.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(voice_test C CXX)

    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_EXTENSIONS OFF)

    if(MSVC)
        add_compile_options(/utf-8 /permissive- /W3)
        add_compile_definitions(_CRT_SECURE_NO_WARNINGS NOMINMAX WIN32_LEAN_AND_MEAN)
        add_compile_options(/wd4244 /wd4267 /wd4305 /wd4996)
    else()
        add_compile_options(-Wall -Wno-unused-parameter)
    endif()

    set(CMAKE_POSITION_INDEPENDENT_CODE ON)

    # sherpa-onnx expects a build type for its onnxruntime FetchContent.
    if(NOT CMAKE_BUILD_TYPE)
        set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
    endif()

    # Point at the winefox submodule from the parent directory.
    set(VOICE_TEST_ROOT ${CMAKE_CURRENT_SOURCE_DIR})
    set(SHERPA_ONNX_DIR ${VOICE_TEST_ROOT}/../third_party/sherpa-onnx)

    # Match sherpa-onnx CRT expectations: static /MT (Release) /MTd (Debug)
    # so the downloaded static onnxruntime lib matches our runtime.
    set(SHERPA_ONNX_USE_STATIC_CRT ON CACHE BOOL "" FORCE)
    # We only need the C API + core. Disable everything else to keep the
    # build lean.
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
    set(SHERPA_ONNX_ENABLE_PYTHON OFF CACHE BOOL "" FORCE)
    set(SHERPA_ONNX_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
    set(SHERPA_ONNX_ENABLE_PORTAUDIO OFF CACHE BOOL "" FORCE)
    set(SHERPA_ONNX_ENABLE_JNI OFF CACHE BOOL "" FORCE)
    set(SHERPA_ONNX_ENABLE_WEBSOCKET OFF CACHE BOOL "" FORCE)
    set(SHERPA_ONNX_ENABLE_BINARY OFF CACHE BOOL "" FORCE)
    set(SHERPA_ONNX_BUILD_C_API_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(SHERPA_ONNX_ENABLE_GPU OFF CACHE BOOL "" FORCE)
    set(SHERPA_ONNX_ENABLE_DIRECTML OFF CACHE BOOL "" FORCE)
    set(SHERPA_ONNX_ENABLE_TTS ON CACHE BOOL "" FORCE)
    set(SHERPA_ONNX_ENABLE_SPEAKER_DIARIZATION OFF CACHE BOOL "" FORCE)

    add_subdirectory(${SHERPA_ONNX_DIR} ${CMAKE_BINARY_DIR}/sherpa-onnx)
else()
    # Integrated mode: sherpa-onnx targets already exist.
    message(STATUS "voice-test: integrated mode (reusing sherpa-onnx from root)")
endif()

# ---- voice_test target ---------------------------------------------------

add_executable(voice_test
    src/main.cpp
    src/smoke.cpp
    src/asr_test.cpp
    src/vad_test.cpp
    src/tts_test.cpp
)

target_include_directories(voice_test PRIVATE src)
target_link_libraries(voice_test PRIVATE sherpa-onnx-c-api)

# Output to <binary_dir>/voice-test so it sits next to the model dir layout
# when run from the winefox root. CMAKE_RUNTIME_OUTPUT_DIRECTORY is already
# set by sherpa-onnx; override just for this target.
set_target_properties(voice_test PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/voice-test"
)

if(WIN32)
    # sherpa-onnx static link pulls in onnxruntime's Win32 deps.
    target_link_libraries(voice_test PRIVATE advapi32)
endif()

# Convenience: print where the binary lands.
message(STATUS "voice-test binary -> ${CMAKE_BINARY_DIR}/voice-test")
