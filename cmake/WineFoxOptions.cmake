# WineFox build options. Include'd from the root CMakeLists.txt.
#
# Option summary:
#   WINEFOX_VOICE_ENABLED  - pull in voice front-end (onnxruntime VAD/TTS + ggml ASR) (Phase 2+)
#   WINEFOX_WITH_AEC       - enable WebRTC AEC3 module (Phase 3)
#   WINEFOX_BUILD_GUI      - build the GLES3 GUI (Phase 4, Windows/Android)
#   WINEFOX_MINIMAL        - arm32 extreme-compat preset (no embedding/TTS/AEC)
#   WINEFOX_DEBUG          - enable WF_LOG_* + link spdlog
#   WINEFOX_BUILD_TESTS    - build GoogleTest unit tests
#   WINEFOX_BUILD_CLI      - build the CLI entry point

option(WINEFOX_VOICE_ENABLED "Enable voice front-end (onnxruntime VAD/TTS + ggml ASR)" OFF)
option(WINEFOX_WITH_AEC      "Enable WebRTC AEC3 echo cancellation" OFF)
option(WINEFOX_BUILD_GUI     "Build the GLES3 GUI target" OFF)
option(WINEFOX_MINIMAL       "Extreme-compat preset (arm32: disable embedding/TTS/AEC)" OFF)
option(WINEFOX_DEBUG         "Enable WF_LOG_* macros and spdlog linking" OFF)
option(WINEFOX_BUILD_TESTS   "Build GoogleTest unit tests" OFF)
option(WINEFOX_BUILD_CLI     "Build the winefox CLI entry point" ON)
option(WINEFOX_BUILD_VOICE_TEST "Build the voice-test benchmark sandbox (Phase 2)" OFF)

# WINEFOX_MINIMAL forces the optional subsystems off.
if(WINEFOX_MINIMAL)
    set(WINEFOX_VOICE_ENABLED OFF CACHE BOOL "" FORCE)
    set(WINEFOX_WITH_AEC      OFF CACHE BOOL "" FORCE)
    set(WINEFOX_BUILD_GUI     OFF CACHE BOOL "" FORCE)
    add_compile_definitions(WINEFOX_MINIMAL)
endif()

if(WINEFOX_DEBUG)
    add_compile_definitions(DEBUG)
endif()

# Voice front-end pulls in AEC implicitly when the user opts in.
if(WINEFOX_WITH_AEC AND NOT WINEFOX_VOICE_ENABLED)
    message(WARNING "WINEFOX_WITH_AEC requires WINEFOX_VOICE_ENABLED; forcing it ON.")
    set(WINEFOX_VOICE_ENABLED ON CACHE BOOL "" FORCE)
endif()

# Inject compile definitions so source files can branch on these flags.
if(WINEFOX_VOICE_ENABLED)
    add_compile_definitions(WINEFOX_VOICE_ENABLED)
endif()
if(WINEFOX_WITH_AEC)
    add_compile_definitions(WINEFOX_WITH_AEC)
endif()
if(WINEFOX_BUILD_GUI)
    add_compile_definitions(WINEFOX_BUILD_GUI)
endif()

message(STATUS "WineFox options:")
message(STATUS "  WINEFOX_VOICE_ENABLED = ${WINEFOX_VOICE_ENABLED}")
message(STATUS "  WINEFOX_WITH_AEC      = ${WINEFOX_WITH_AEC}")
message(STATUS "  WINEFOX_BUILD_GUI     = ${WINEFOX_BUILD_GUI}")
message(STATUS "  WINEFOX_MINIMAL       = ${WINEFOX_MINIMAL}")
message(STATUS "  WINEFOX_DEBUG         = ${WINEFOX_DEBUG}")
message(STATUS "  WINEFOX_BUILD_TESTS   = ${WINEFOX_BUILD_TESTS}")
message(STATUS "  WINEFOX_BUILD_CLI     = ${WINEFOX_BUILD_CLI}")
