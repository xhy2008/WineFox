# src/world/CMakeLists.txt — winefox_world.exe
#
# Standalone 3D World frontend build. Linked against a pre-built
# winefox_core.dll (C ABI) to avoid ggml target conflicts with
# SenseVoice.cpp's vendored ggml.
#
# Build (two-step, from winefox root):
#   Step 1 — core DLL:
#     cmake -B build-core -DWINEFOX_BUILD_CORE_DLL=ON -DCMAKE_BUILD_TYPE=Release
#     cmake --build build-core --config Release
#   Step 2 — world exe:
#     cmake -B build-world -DWINEFOX_BUILD_WORLD=ON \
#           -DWINEFOX_CORE_DLL_DIR=build-core/Release
#     cmake --build build-world --config Release --target winefox_world

cmake_minimum_required(VERSION 3.20)

# ---------------------------------------------------------------------------
# winefox_core.dll import lib + header
# ---------------------------------------------------------------------------
if(NOT WINEFOX_CORE_DLL_DIR OR WINEFOX_CORE_DLL_DIR STREQUAL "")
    message(FATAL_ERROR
        "WINEFOX_BUILD_WORLD requires WINEFOX_CORE_DLL_DIR to point to the "
        "directory containing winefox_core.lib + winefox_core.dll.\n"
        "Build the core DLL first:\n"
        "  cmake -B build-core -DWINEFOX_BUILD_CORE_DLL=ON\n"
        "  cmake --build build-core --config Release\n"
        "Then configure the world exe:\n"
        "  cmake -B build-world -DWINEFOX_BUILD_WORLD=ON "
        "-DWINEFOX_CORE_DLL_DIR=build-core/Release")
endif()

set(_core_lib "${WINEFOX_CORE_DLL_DIR}/winefox_core.lib")
set(_core_dll "${WINEFOX_CORE_DLL_DIR}/winefox_core.dll")
if(NOT EXISTS "${_core_lib}")
    message(FATAL_ERROR "winefox_core.lib not found at: ${_core_lib}")
endif()

# ---------------------------------------------------------------------------
# Paths to voice deps (shared with voice-test)
# ---------------------------------------------------------------------------
set(_voice_root "${CMAKE_SOURCE_DIR}/voice-test/third_party")
set(SENSEVOICE_ROOT "${_voice_root}/sensevoice-cpp")
set(KOKORO_SRC_ROOT  "${_voice_root}/kokoro-cpp-src")
set(ONNXRUNTIME_ROOT "${_voice_root}/onnxruntime-static")
set(TEN_VAD_ROOT     "${_voice_root}/ten-vad")

foreach(_d SENSEVOICE_ROOT KOKORO_SRC_ROOT ONNXRUNTIME_ROOT TEN_VAD_ROOT)
    if(NOT EXISTS "${${_d}}")
        message(FATAL_ERROR "${_d} not found: ${${_d}}")
    endif()
endforeach()

# ---------------------------------------------------------------------------
# SDL3 (static build — no external DLL dependency)
# ---------------------------------------------------------------------------
set(SDL_STATIC ON CACHE BOOL "" FORCE)
set(SDL_SHARED OFF CACHE BOOL "" FORCE)
set(SDL_TEST   OFF CACHE BOOL "" FORCE)
set(SDL_AUDIO  ON  CACHE BOOL "" FORCE)
set(SDL_VIDEO  ON  CACHE BOOL "" FORCE)
set(SDL_RENDER OFF CACHE BOOL "" FORCE)
set(SDL_EVENTS ON  CACHE BOOL "" FORCE)
add_subdirectory(${CMAKE_SOURCE_DIR}/third_party/SDL3 sdl3-build)

# ---------------------------------------------------------------------------
# Vulkan
# ---------------------------------------------------------------------------
find_package(Vulkan REQUIRED)

# ---------------------------------------------------------------------------
# SenseVoice.cpp (voice-test method: add_subdirectory with its own ggml)
# ---------------------------------------------------------------------------
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(SENSE_VOICE_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(SENSE_VOICE_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(SENSE_VOICE_FATAL_WARNINGS OFF CACHE BOOL "" FORCE)
set(GGML_BLAS                  OFF CACHE BOOL "" FORCE)
add_subdirectory(${SENSEVOICE_ROOT} ${CMAKE_BINARY_DIR}/sensevoice-cpp EXCLUDE_FROM_ALL)

# ---------------------------------------------------------------------------
# winefox_world executable
#
# Debug build: CONSOLE subsystem (printf visible).
# Release build: WIN32 subsystem (pure GUI, no console window).
# ---------------------------------------------------------------------------
add_executable(winefox_world
    app/world_main.cpp
    config/world_config.cpp
    platform/sdl_window.cpp
    platform/audio_io.cpp
    voice/vad_service.cpp
    voice/asr_service.cpp
    voice/tts_service.cpp
    voice/aec_service.cpp
    voice/voice_pipeline.cpp
    render/vulkan_context.cpp
)

# Debug = CONSOLE subsystem (main entry, printf visible in terminal).
# Release = WIN32 subsystem (WinMain entry, pure GUI, no console window).
# TEMPORARY: Release also uses CONSOLE for TTS diagnostic comparison.
set_target_properties(winefox_world PROPERTIES
    WIN32_EXECUTABLE OFF
)

target_include_directories(winefox_world PRIVATE
    ${CMAKE_SOURCE_DIR}/src/core/api        # winefox_api.h
    ${CMAKE_SOURCE_DIR}/src                  # core headers (config etc.)
    ${CMAKE_SOURCE_DIR}/third_party/json     # nlohmann/json (standalone)
    ${CMAKE_CURRENT_SOURCE_DIR}              # world headers
    ${SENSEVOICE_ROOT}/sense-voice/csrc      # sense-voice.h
    ${KOKORO_SRC_ROOT}                       # Kokoro.h
    ${KOKORO_SRC_ROOT}/third_party/cppjieba/include
    ${KOKORO_SRC_ROOT}/third_party/cppjieba/deps/limonp/include
    ${ONNXRUNTIME_ROOT}/include
    ${TEN_VAD_ROOT}/include
)

# --- winefox_core.dll (C ABI) ---
target_link_libraries(winefox_world PRIVATE ${_core_lib})

# --- SDL3 (static) ---
target_link_libraries(winefox_world PRIVATE SDL3::SDL3-static SDL3::Headers)

# --- Vulkan ---
target_link_libraries(winefox_world PRIVATE Vulkan::Vulkan)

# --- SenseVoice (ASR) ---
target_link_libraries(winefox_world PRIVATE sense-voice-core ggml)

# --- Kokoro (TTS) — sources compiled directly into the exe ---
target_sources(winefox_world PRIVATE
    ${KOKORO_SRC_ROOT}/Kokoro.cpp
    ${KOKORO_SRC_ROOT}/Tokenizer.cpp
    ${KOKORO_SRC_ROOT}/ZHG2P.cpp
    ${KOKORO_SRC_ROOT}/PinyinFinder.cpp
    ${KOKORO_SRC_ROOT}/ToneSandhi.cpp
    ${KOKORO_SRC_ROOT}/ZHFrontend.cpp
)

# --- onnxruntime (static, for Kokoro TTS) ---
if(WIN32)
    target_link_libraries(winefox_world PRIVATE ${ONNXRUNTIME_ROOT}/lib/onnxruntime.lib)
    target_link_libraries(winefox_world PRIVATE
        Advapi32 Ws2_32 Shlwapi userenv Synchronization
    )
endif()

# --- ten-vad (VAD, self-contained prebuilt) ---
if(WIN32)
    target_link_libraries(winefox_world PRIVATE ${TEN_VAD_ROOT}/lib/ten_vad.lib)
endif()

# --- Windows system libs ---
if(WIN32)
    target_link_libraries(winefox_world PRIVATE
        userenv ole32 imm32 version winmm setupapi dwmapi
    )
endif()

target_compile_definitions(winefox_world PRIVATE
    NOMINMAX WIN32_LEAN_AND_MEAN _USE_MATH_DEFINES
    _CRT_SECURE_NO_WARNINGS
)

# ---------------------------------------------------------------------------
# Post-build: copy DLLs and data next to the exe
# ---------------------------------------------------------------------------
set(_out_dir "$<TARGET_FILE_DIR:winefox_world>")

# winefox_core.dll
add_custom_command(TARGET winefox_world POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${_core_dll}" "${_out_dir}"
    COMMENT "Copying winefox_core.dll")

# ten_vad.dll (self-contained)
if(EXISTS "${TEN_VAD_ROOT}/lib/ten_vad.dll")
    add_custom_command(TARGET winefox_world POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${TEN_VAD_ROOT}/lib/ten_vad.dll" "${_out_dir}"
        COMMENT "Copying ten_vad.dll")
endif()

# Kokoro dict/ (jieba, pinyin, cmudict, vocab)
add_custom_command(TARGET winefox_world POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${KOKORO_SRC_ROOT}/dict" "${_out_dir}/dict"
    COMMENT "Copying kokoro dict/")

# winefox.json (config) — only copy if it exists in the source root; the
# file may live elsewhere (e.g. already in the output dir) and a missing
# source must not break the build.
if(EXISTS "${CMAKE_SOURCE_DIR}/winefox.json")
    add_custom_command(TARGET winefox_world POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${CMAKE_SOURCE_DIR}/winefox.json" "${_out_dir}"
        COMMENT "Copying winefox.json")
endif()

# models/ — create a junction (Windows directory symlink that doesn't
# require admin privileges) so the exe can load model files without copying
# multi-GB of data.
if(WIN32)
    add_custom_command(TARGET winefox_world POST_BUILD
        COMMAND cmd /c "if not exist \"${_out_dir}\\models\" mklink /J \"${_out_dir}\\models\" \"${CMAKE_SOURCE_DIR}\\models\""
        COMMENT "Creating directory junction for models/")
    # llm-finetune/ — needed at runtime for system_prompt.txt (small text files only).
    add_custom_command(TARGET winefox_world POST_BUILD
        COMMAND cmd /c "if not exist \"${_out_dir}\\llm-finetune\" mklink /J \"${_out_dir}\\llm-finetune\" \"${CMAKE_SOURCE_DIR}\\llm-finetune\""
        COMMENT "Creating directory junction for llm-finetune/")
endif()

# Output to <binary_dir>/world for consistency
set_target_properties(winefox_world PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/world"
)

message(STATUS "winefox_world binary -> ${CMAKE_BINARY_DIR}/world")
