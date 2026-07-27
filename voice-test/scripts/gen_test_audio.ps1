# Generate test WAV files for voice-test using Windows SAPI (Microsoft Huihui, zh-CN).
#
# Produces:
#   test-data/asr_01.wav .. asr_05.wav   Chinese speech with known reference text
#   test-data/asr_reference.txt           Ground-truth transcripts (one per line)
#   test-data/vad_mixed.wav               3 speech segments separated by silence
#   test-data/vad_reference.txt           Ground-truth segment timestamps
#
# Output format: 16 kHz, 16-bit mono PCM (SenseVoice/TEN-VAD native rate).
# Requires .NET System.Speech (Windows Desktop).
#
# NOTE: This script is ASCII-only. Chinese text is loaded from sentences.txt
# and vad_phrases.txt (UTF-8) so PowerShell's ANSI .ps1 parser is not confused.

[CmdletBinding()]
param(
    [string]$OutDir = (Join-Path $PSScriptRoot "..\test-data")
)

$ErrorActionPreference = "Stop"
if (-not (Test-Path $OutDir)) { New-Item -ItemType Directory -Path $OutDir | Out-Null }
$OutDir = (Resolve-Path $OutDir).Path
Write-Host "Output directory: $OutDir"

# ---- Load Chinese text from UTF-8 data files ---------------------------
$sentencePath = Join-Path $PSScriptRoot "sentences.txt"
$vadPhrasePath = Join-Path $PSScriptRoot "vad_phrases.txt"
if (-not (Test-Path $sentencePath)) { Write-Error "Missing $sentencePath"; exit 1 }
if (-not (Test-Path $vadPhrasePath)) { Write-Error "Missing $vadPhrasePath"; exit 1 }

# Get-Content -Encoding UTF8 decodes UTF-8 (with or without BOM) correctly.
$sentences = Get-Content -Path $sentencePath -Encoding UTF8 | Where-Object { $_.Trim().Length -gt 0 }
$vadPhrases = Get-Content -Path $vadPhrasePath -Encoding UTF8 | Where-Object { $_.Trim().Length -gt 0 }
Write-Host ("Loaded {0} ASR sentences, {1} VAD phrases" -f $sentences.Count, $vadPhrases.Count)

# ---- SAPI voice synthesis -----------------------------------------------
Add-Type -AssemblyName System.Speech

$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
try {
    $synth.SelectVoice("Microsoft Huihui Desktop")
} catch {
    Write-Error "Microsoft Huihui Desktop (zh-CN) voice not installed."
    exit 1
}
$synth.Rate = 0
$synth.Volume = 100

# 16 kHz, 16-bit, mono - matches SenseVoice/TEN-VAD expected input rate.
$fmt = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(
    16000,
    [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen,
    [System.Speech.AudioFormat.AudioChannel]::Mono)

# ---- Generate ASR clips -------------------------------------------------
$refPath = Join-Path $OutDir "asr_reference.txt"
# Write reference file as UTF-8 (with BOM so notepad/Python both read it).
[System.IO.File]::WriteAllLines($refPath, $sentences, (New-Object System.Text.UTF8Encoding $true))
Write-Host "Wrote reference transcripts -> $refPath"

for ($i = 0; $i -lt $sentences.Count; $i++) {
    $idx = ($i + 1).ToString("00")
    $wav = Join-Path $OutDir ("asr_" + $idx + ".wav")
    $synth.SetOutputToWaveFile($wav, $fmt)
    $synth.Speak($sentences[$i])
    $synth.SetOutputToNull()
    Write-Host ("  generated " + $wav + "  text=[" + $sentences[$i] + "]")
}

# ---- VAD test file: 3 speech segments separated by silence -------------
# Layout (seconds): 1.0 silence | speech0 | 0.5 silence | speech1 |
#                   0.8 silence | speech2 | 1.0 silence

$segmentWavs = @()
for ($i = 0; $i -lt $vadPhrases.Count; $i++) {
    $tmp = Join-Path $OutDir ("_vad_seg_" + $i + ".wav")
    $synth.SetOutputToWaveFile($tmp, $fmt)
    $synth.Speak($vadPhrases[$i])
    $synth.SetOutputToNull()
    $segmentWavs += $tmp
}

$bytesPerSample = 2
$rate = 16000
$silencePreBytes  = [int]([Math]::Round(1.0 * $rate)) * $bytesPerSample
$silenceMid1Bytes = [int]([Math]::Round(0.5 * $rate)) * $bytesPerSample
$silenceMid2Bytes = [int]([Math]::Round(0.8 * $rate)) * $bytesPerSample
$silencePostBytes = [int]([Math]::Round(1.0 * $rate)) * $bytesPerSample

function Read-PcmSamples([string]$path) {
    $all = [System.IO.File]::ReadAllBytes($path)
    $off = 12
    $dataOff = 0; $dataLen = 0
    while ($off -lt $all.Length) {
        $id = [System.Text.Encoding]::ASCII.GetString($all, $off, 4)
        $sz = [BitConverter]::ToInt32($all, $off + 4)
        if ($id -eq "data") { $dataOff = $off + 8; $dataLen = $sz; break }
        $off += 8 + $sz
    }
    $samples = [byte[]]::new($dataLen)
    [Array]::Copy($all, $dataOff, $samples, 0, $dataLen)
    return [byte[]]$samples
}

# Use a MemoryStream to avoid PowerShell byte[] -> Object[] unwrapping issues
# that break List<byte>.AddRange.
$ms = New-Object System.IO.MemoryStream
$writeSilence = {
    param($n)
    $zeros = [byte[]]::new($n)
    $ms.Write($zeros, 0, $n)
}
$writeBytes = {
    param([byte[]]$b)
    $ms.Write($b, 0, $b.Length)
}

& $writeSilence $silencePreBytes
$s0 = Read-PcmSamples $segmentWavs[0]
& $writeBytes $s0
& $writeSilence $silenceMid1Bytes
$s1 = Read-PcmSamples $segmentWavs[1]
& $writeBytes $s1
& $writeSilence $silenceMid2Bytes
$s2 = Read-PcmSamples $segmentWavs[2]
& $writeBytes $s2
& $writeSilence $silencePostBytes

$pcm = $ms.ToArray()

# Ground-truth segment timestamps.
$seg0Start = 1.0
$seg0End   = $seg0Start + ($s0.Length / $bytesPerSample / $rate)
$seg1Start = $seg0End + 0.5
$seg1End   = $seg1Start + ($s1.Length / $bytesPerSample / $rate)
$seg2Start = $seg1End + 0.8
$seg2End   = $seg2Start + ($s2.Length / $bytesPerSample / $rate)

$vadRefPath = Join-Path $OutDir "vad_reference.txt"
$vadRefLines = @(
    "# Ground-truth VAD segments: start_s,end_s,duration_s,text"
    ("{0:F3},{1:F3},{2:F3},{3}" -f $seg0Start, $seg0End, ($seg0End - $seg0Start), $vadPhrases[0])
    ("{0:F3},{1:F3},{2:F3},{3}" -f $seg1Start, $seg1End, ($seg1End - $seg1Start), $vadPhrases[1])
    ("{0:F3},{1:F3},{2:F3},{3}" -f $seg2Start, $seg2End, ($seg2End - $seg2Start), $vadPhrases[2])
)
[System.IO.File]::WriteAllLines($vadRefPath, $vadRefLines, (New-Object System.Text.UTF8Encoding $true))
Write-Host "Wrote VAD reference -> $vadRefPath"
Write-Host ("  seg0: " + $seg0Start + " -> " + $seg0End)
Write-Host ("  seg1: " + $seg1Start + " -> " + $seg1End)
Write-Host ("  seg2: " + $seg2Start + " -> " + $seg2End)

# Build final WAV with proper RIFF header.
$totalLen = $pcm.Length
$header = New-Object byte[] 44
[Array]::Copy([System.Text.Encoding]::ASCII.GetBytes("RIFF"), 0, $header, 0, 4)
[BitConverter]::GetBytes([Int32]($totalLen + 36)).CopyTo($header, 4)
[Array]::Copy([System.Text.Encoding]::ASCII.GetBytes("WAVE"), 0, $header, 8, 4)
[Array]::Copy([System.Text.Encoding]::ASCII.GetBytes("fmt "), 0, $header, 12, 4)
[BitConverter]::GetBytes([Int32]16).CopyTo($header, 16)
[BitConverter]::GetBytes([Int16]1).CopyTo($header, 20)
[BitConverter]::GetBytes([Int16]1).CopyTo($header, 22)
[BitConverter]::GetBytes([Int32]$rate).CopyTo($header, 24)
[BitConverter]::GetBytes([Int32]($rate * $bytesPerSample)).CopyTo($header, 28)
[BitConverter]::GetBytes([Int16]$bytesPerSample).CopyTo($header, 32)
[BitConverter]::GetBytes([Int16]16).CopyTo($header, 34)
[Array]::Copy([System.Text.Encoding]::ASCII.GetBytes("data"), 0, $header, 36, 4)
[BitConverter]::GetBytes([Int32]$totalLen).CopyTo($header, 40)

$vadOut = Join-Path $OutDir "vad_mixed.wav"
$fs = [System.IO.File]::Create($vadOut)
$fs.Write($header, 0, 44)
$fs.Write($pcm, 0, $totalLen)
$fs.Close()
$totalSeconds = $totalLen / $bytesPerSample / $rate
Write-Host ("Wrote VAD mixed audio -> " + $vadOut + " (" + $totalSeconds.ToString("F2") + "s)")

# Clean up temp segment files.
foreach ($t in $segmentWavs) { Remove-Item $t -ErrorAction SilentlyContinue }

$synth.Dispose()
Write-Host ""
Write-Host ("Done. Generated " + $sentences.Count + " ASR clips + 1 VAD mixed clip in " + $OutDir)
