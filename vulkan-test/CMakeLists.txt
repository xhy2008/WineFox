# vulkan-test: standalone Vulkan 3D rendering sandbox.
#
# Self-contained test bed for Vulkan rendering (graphics pipeline, mesh,
# depth buffer, MVP push constants, orbit camera). Does NOT depend on
# winefox_core.dll or any voice components — only SDL3 + Vulkan SDK + glm.
#
# Build (from winefox root):
#   cmake -B build-vulkan-test -S vulkan-test -DCMAKE_BUILD_TYPE=Release
#   cmake --build build-vulkan-test --config Release

cmake_minimum_required(VERSION 3.20)
project(vulkan_test C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MSVC)
    add_compile_options(/utf-8 /permissive- /W3)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS NOMINMAX WIN32_LEAN_AND_MEAN _USE_MATH_DEFINES)
    add_compile_options(/wd4244 /wd4267 /wd4305)
else()
    add_compile_options(-Wall -Wno-unused-parameter)
endif()

# ---------------------------------------------------------------------------
# SDL3 (static build — reuse the submodule at repo root)
# ---------------------------------------------------------------------------
set(SDL_STATIC ON CACHE BOOL "" FORCE)
set(SDL_SHARED OFF CACHE BOOL "" FORCE)
set(SDL_TEST   OFF CACHE BOOL "" FORCE)
set(SDL_AUDIO  OFF CACHE BOOL "" FORCE)
set(SDL_VIDEO  ON  CACHE BOOL "" FORCE)
set(SDL_RENDER OFF CACHE BOOL "" FORCE)
set(SDL_EVENTS ON  CACHE BOOL "" FORCE)
add_subdirectory(${CMAKE_SOURCE_DIR}/../third_party/SDL3 sdl3-build)

# ---------------------------------------------------------------------------
# Vulkan SDK (provides Vulkan headers, loader, glslc, and glm)
# ---------------------------------------------------------------------------
find_package(Vulkan REQUIRED)

# ---------------------------------------------------------------------------
# Compile GLSL shaders → SPIR-V at build time
# ---------------------------------------------------------------------------
find_program(GLSLC glslc HINTS "$ENV{VULKAN_SDK}/Bin" REQUIRED)

set(SHADER_DIR  "${CMAKE_CURRENT_SOURCE_DIR}/shaders")
set(SPIRV_DIR   "${CMAKE_CURRENT_BINARY_DIR}/shaders")

set(SHADERS
    "${SHADER_DIR}/cube.vert"
    "${SHADER_DIR}/cube.frag"
)

set(SPIRV_OUTPUTS "")
foreach(_shader ${SHADERS})
    get_filename_component(_fname ${_shader} NAME)
    set(_spv "${SPIRV_DIR}/${_fname}.spv")
    list(APPEND SPIRV_OUTPUTS ${_spv})
    add_custom_command(
        OUTPUT ${_spv}
        COMMAND ${CMAKE_COMMAND} -E make_directory "${SPIRV_DIR}"
        COMMAND ${GLSLC} ${_shader} -o ${_spv}
        DEPENDS ${_shader}
        COMMENT "Compiling ${_shader} → ${_fname}.spv"
    )
endforeach()

add_custom_target(compile_shaders DEPENDS ${SPIRV_OUTPUTS})

# ---------------------------------------------------------------------------
# vulkan_test executable
# ---------------------------------------------------------------------------
add_executable(vulkan_test
    src/main.cpp
    src/vulkan_app.cpp
)

target_include_directories(vulkan_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(vulkan_test PRIVATE
    SDL3::SDL3-static
    SDL3::Headers
    Vulkan::Vulkan
)

# Add Vulkan SDK include path for glm (shipped with the SDK)
target_include_directories(vulkan_test PRIVATE "$ENV{VULKAN_SDK}/Include")

# Ensure shaders are compiled before the executable
add_dependencies(vulkan_test compile_shaders)

# Copy SPIR-V shaders next to the executable at build time
add_custom_command(TARGET vulkan_test POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${SPIRV_DIR}" "$<TARGET_FILE_DIR:vulkan_test>/shaders"
    COMMENT "Copying SPIR-V shaders to output directory")

if(WIN32)
    target_link_libraries(vulkan_test PRIVATE
        userenv ole32 imm32 version winmm setupapi dwmapi
    )
endif()

set_target_properties(vulkan_test PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/vulkan-test"
)

message(STATUS "vulkan-test binary -> ${CMAKE_BINARY_DIR}/vulkan-test")
