# build-world.ps1 — Two-step build for WineFox World (core DLL + world exe)
#
# Usage:
#   .\build-world.ps1              # Release build (default)
#   .\build-world.ps1 -Debug       # Debug build (with console + validation layers)
#   .\build-world.ps1 -Clean       # Clean build directories first
#
# Prerequisites:
#   - CMake 3.20+
#   - MSVC (Visual Studio 2022 or Build Tools)
#   - Vulkan SDK installed (VULKAN_SDK env var set)
#   - All git submodules initialized (third_party/llama.cpp, third_party/SDL3, etc.)
#   - voice-test/third_party deps extracted (onnxruntime zip, etc.)

param(
    [switch]$Debug,
    [switch]$Clean
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path

$BuildType = if ($Debug) { "Debug" } else { "Release" }
$BuildDirCore  = Join-Path $Root "build-core"
$BuildDirWorld = Join-Path $Root "build-world"

Write-Host "=== WineFox World Build ===" -ForegroundColor Cyan
Write-Host "Configuration: $BuildType"
Write-Host "Root: $Root"
Write-Host ""

# --- Clean ---
if ($Clean) {
    Write-Host "[1/4] Cleaning build directories..." -ForegroundColor Yellow
    if (Test-Path $BuildDirCore)  { Remove-Item -Recurse -Force $BuildDirCore }
    if (Test-Path $BuildDirWorld) { Remove-Item -Recurse -Force $BuildDirWorld }
    Write-Host "    Cleaned."
}

# --- Step 1: Configure core DLL ---
Write-Host "[2/4] Configuring winefox_core.dll..." -ForegroundColor Yellow
$coreConfigArgs = @(
    "-B", $BuildDirCore,
    "-DWINEFOX_BUILD_CORE_DLL=ON",
    "-DCMAKE_BUILD_TYPE=$BuildType"
)
& cmake $coreConfigArgs
if ($LASTEXITCODE -ne 0) {
    Write-Host "FATAL: core DLL configuration failed" -ForegroundColor Red
    exit 1
}

# --- Step 2: Build core DLL ---
Write-Host "[3/4] Building winefox_core.dll..." -ForegroundColor Yellow
& cmake --build $BuildDirCore --config $BuildType --target winefox_core
if ($LASTEXITCODE -ne 0) {
    Write-Host "FATAL: core DLL build failed" -ForegroundColor Red
    exit 1
}

# Locate the built DLL + import lib.
$CoreDllDir = if ($Debug) {
    Join-Path $BuildDirCore "Debug"
} else {
    Join-Path $BuildDirCore "Release"
}
if (-not (Test-Path (Join-Path $CoreDllDir "winefox_core.lib"))) {
    # Fallback: search for it.
    $found = Get-ChildItem -Path $BuildDirCore -Filter "winefox_core.lib" -Recurse | Select-Object -First 1
    if ($found) {
        $CoreDllDir = Split-Path -Parent $found.FullName
    } else {
        Write-Host "FATAL: winefox_core.lib not found after build" -ForegroundColor Red
        exit 1
    }
}
Write-Host "    Core DLL dir: $CoreDllDir"

# --- Step 3: Configure world exe ---
Write-Host "[4/4] Configuring + building winefox_world.exe..." -ForegroundColor Yellow
$worldConfigArgs = @(
    "-B", $BuildDirWorld,
    "-DWINEFOX_BUILD_WORLD=ON",
    "-DWINEFOX_CORE_DLL_DIR=`"$CoreDllDir`"",
    "-DCMAKE_BUILD_TYPE=$BuildType"
)
& cmake $worldConfigArgs
if ($LASTEXITCODE -ne 0) {
    Write-Host "FATAL: world exe configuration failed" -ForegroundColor Red
    exit 1
}

& cmake --build $BuildDirWorld --config $BuildType --target winefox_world
if ($LASTEXITCODE -ne 0) {
    Write-Host "FATAL: world exe build failed" -ForegroundColor Red
    exit 1
}

# --- Done ---
$WorldExe = Join-Path $BuildDirWorld "world/$BuildType/winefox_world.exe"
if (-not (Test-Path $WorldExe)) {
    # Search for it.
    $found = Get-ChildItem -Path $BuildDirWorld -Filter "winefox_world.exe" -Recurse | Select-Object -First 1
    if ($found) { $WorldExe = $found.FullName }
}

Write-Host ""
Write-Host "=== Build Complete ===" -ForegroundColor Green
Write-Host "Output: $WorldExe"
Write-Host ""
Write-Host "To run: cd to the output directory and execute winefox_world.exe"
Write-Host "Make sure winefox.json and all model files are in the same directory."
