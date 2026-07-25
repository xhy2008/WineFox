# Run vision color test N times, deleting DB before each run.
# Usage: .\run_color_test.ps1 -MinTokens 32 -Runs 3
param(
    [int]$Runs = 3
)

$exe = 'e:\winefox\build\Release\winefox.exe'
$inputFile = 'e:\winefox\tests\test_color_input.txt'
$dbFiles = @('E:\winefox\winefox.db','E:\winefox\winefox.db-wal','E:\winefox\winefox.db-shm')

Set-Location 'e:\winefox'

for ($i = 1; $i -le $Runs; $i++) {
    Write-Host "`n===== RUN $i / $Runs =====" -ForegroundColor Cyan

    # Kill any lingering winefox.exe so DB file is released
    Get-Process -Name 'winefox' -ErrorAction SilentlyContinue | Stop-Process -Force
    Start-Sleep -Seconds 3
    # Delete DB to ensure clean state
    foreach ($f in $dbFiles) {
        Remove-Item $f -ErrorAction SilentlyContinue
    }
    # Verify deletion
    $stillThere = Test-Path 'E:\winefox\winefox.db'
    Write-Host "DB exists after delete: $stillThere"

    $logFile = "E:\winefox\tests\test_color_$i.log"
    $errFile = "E:\winefox\tests\test_color_$i.err"
    Remove-Item $logFile,$errFile -ErrorAction SilentlyContinue

    $outLines = [System.Collections.ArrayList]::new()
    $errLines = [System.Collections.ArrayList]::new()

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

    $outEvt = Register-ObjectEvent -InputObject $proc -EventName 'OutputDataReceived' -MessageData $outLines -Action {
        if ($EventArgs.Data -ne $null) { $Event.MessageData.Add($EventArgs.Data) | Out-Null }
    }
    $errEvt = Register-ObjectEvent -InputObject $proc -EventName 'ErrorDataReceived' -MessageData $errLines -Action {
        if ($EventArgs.Data -ne $null) { $Event.MessageData.Add($EventArgs.Data) | Out-Null }
    }

    $proc.Start() | Out-Null
    $proc.BeginOutputReadLine()
    $proc.BeginErrorReadLine()

    $lines = Get-Content $inputFile -Encoding UTF8
    foreach ($line in $lines) {
        Start-Sleep -Milliseconds 1500
        $proc.StandardInput.WriteLine($line)
        $proc.StandardInput.Flush()
    }
    $proc.StandardInput.Close()

    $proc.WaitForExit(120000) | Out-Null
    if (-not $proc.HasExited) { $proc.Kill(); $proc.WaitForExit(5000) | Out-Null }

    Start-Sleep -Milliseconds 500
    Unregister-Event -SourceIdentifier $outEvt.Name
    Unregister-Event -SourceIdentifier $errEvt.Name

    $outLines -join "`r`n" | Out-File -FilePath $logFile -Encoding UTF8
    $errLines -join "`r`n" | Out-File -FilePath $errFile -Encoding UTF8

    Write-Host "Exit: $($proc.ExitCode), stdout=$($outLines.Count) stderr=$($errLines.Count)"
}
