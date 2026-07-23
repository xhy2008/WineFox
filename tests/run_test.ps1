# Run winefox with scripted input, capturing all output.
$ErrorActionPreference = 'Continue'
$exe = 'e:\winefox\build\Release\winefox.exe'
$inputFile = 'e:\winefox\tests\test_distill_recall.txt'
$logFile = 'e:\winefox\tests\test_output.log'

# Remove old log + db so we start fresh
Remove-Item $logFile -ErrorAction SilentlyContinue
Remove-Item ($logFile + '.err') -ErrorAction SilentlyContinue
Remove-Item 'e:\winefox\winefox.db' -ErrorAction SilentlyContinue
Remove-Item 'e:\winefox\winefox.db-wal' -ErrorAction SilentlyContinue
Remove-Item 'e:\winefox\winefox.db-shm' -ErrorAction SilentlyContinue

Set-Location 'e:\winefox'

# Use a global synchronized StringBuilder so event handlers can access it
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
# Ensure UTF-8 for stdout so Chinese characters render correctly
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

# Feed input lines one by one, with a delay so the process can consume them
$lines = Get-Content $inputFile -Encoding UTF8
foreach ($line in $lines) {
    $proc.StandardInput.WriteLine($line)
    # Small delay between lines; getline in winefox blocks so this is fine
    Start-Sleep -Milliseconds 200
}

# Close stdin to signal EOF (in case /quit didn't fire)
$proc.StandardInput.Close()

# Wait for process to exit (up to 5 minutes)
$proc.WaitForExit(300000) | Out-Null
if (-not $proc.HasExited) {
    Write-Host "Process timed out, killing..."
    $proc.Kill()
    $proc.WaitForExit(5000) | Out-Null
}

# Give async readers a moment to flush
Start-Sleep -Milliseconds 500

Unregister-Event -SourceIdentifier $outEvt.Name
Unregister-Event -SourceIdentifier $errEvt.Name

# Write captured output
$global:outLines -join "`r`n" | Out-File -FilePath $logFile -Encoding UTF8
$global:errLines -join "`r`n" | Out-File -FilePath ($logFile + '.err') -Encoding UTF8

Write-Host "Process exited with code: $($proc.ExitCode)"
Write-Host "stdout lines: $($global:outLines.Count)"
Write-Host "stderr lines: $($global:errLines.Count)"
Write-Host "Output written to: $logFile"
