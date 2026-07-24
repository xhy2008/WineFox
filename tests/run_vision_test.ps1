# Run winefox vision test with scripted input, capturing all output.
$ErrorActionPreference = 'Continue'
$exe = 'e:\winefox\build\Release\winefox.exe'
$inputFile = 'e:\winefox\tests\test_vision_input.txt'
$logFile = 'e:\winefox\tests\test_vision.log'

Remove-Item $logFile -ErrorAction SilentlyContinue
Remove-Item ($logFile + '.err') -ErrorAction SilentlyContinue

Set-Location 'e:\winefox'

$global:outLines = [System.Collections.ArrayList]::new()
$global:errLines = [System.Collections.ArrayList]::new()

$psi = New-Object System.Diagnostics.ProcessStartInfo
$psi.FileName = $exe
$psi.WorkingDirectory = 'e:\winefox'
$psi.UseShellExecute = $false
$psi.RedirectStandardInput = $true
$psi.RedirectStandardOutput = $true
$psi.RedirectStandardError = $true
$psi.CreateNoWindow = $true
$psi.StandardOutputEncoding = [System.Text.Encoding]::UTF8
$psi.StandardErrorEncoding = [System.Text.Encoding]::UTF8

$proc = New-Object System.Diagnostics.Process
$proc.StartInfo = $psi

$outEvt = Register-ObjectEvent -InputObject $proc -EventName 'OutputDataReceived' -MessageData $global:outLines -Action {
    if ($EventArgs.Data -ne $null) {
        $Event.MessageData.Add($EventArgs.Data) | Out-Null
    }
}
$errEvt = Register-ObjectEvent -InputObject $proc -EventName 'ErrorDataReceived' -MessageData $global:errLines -Action {
    if ($EventArgs.Data -ne $null) {
        $Event.MessageData.Add($EventArgs.Data) | Out-Null
    }
}

$proc.Start() | Out-Null
$proc.BeginOutputReadLine()
$proc.BeginErrorReadLine()

$lines = Get-Content $inputFile -Encoding UTF8
foreach ($line in $lines) {
    Start-Sleep -Milliseconds 1500
    Write-Host "SENDING: [$line]"
    $proc.StandardInput.WriteLine($line)
    $proc.StandardInput.Flush()
}

$proc.StandardInput.Close()

# Wait up to 3 minutes for vision processing
$proc.WaitForExit(180000) | Out-Null
if (-not $proc.HasExited) {
    Write-Host "Process timed out, killing..."
    $proc.Kill()
    $proc.WaitForExit(5000) | Out-Null
}

Start-Sleep -Milliseconds 500

Unregister-Event -SourceIdentifier $outEvt.Name
Unregister-Event -SourceIdentifier $errEvt.Name

$global:outLines -join "`r`n" | Out-File -FilePath $logFile -Encoding UTF8
$global:errLines -join "`r`n" | Out-File -FilePath ($logFile + '.err') -Encoding UTF8

Write-Host "Process exited with code: $($proc.ExitCode)"
Write-Host "stdout lines: $($global:outLines.Count)"
Write-Host "stderr lines: $($global:errLines.Count)"
Write-Host "Output written to: $logFile"
