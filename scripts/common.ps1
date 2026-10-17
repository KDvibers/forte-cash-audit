# Shared helpers for Windows PowerShell 5.1 and later. Paths are repository-relative.

function Resolve-AuditPath {
    param([Parameter(Mandatory = $true)][string]$ProjectRoot,
          [Parameter(Mandatory = $true)][string]$Path)
    if ([string]::IsNullOrWhiteSpace($Path)) { throw 'A non-empty path is required.' }
    if ([System.IO.Path]::IsPathRooted($Path)) {
        return [System.IO.Path]::GetFullPath($Path)
    }
    return [System.IO.Path]::GetFullPath((Join-Path $ProjectRoot $Path))
}

function Get-AuditPython {
    param([Parameter(Mandatory = $true)][string]$ProjectRoot,
          [Parameter(Mandatory = $true)][string]$VenvPath)
    $venvRoot = Resolve-AuditPath -ProjectRoot $ProjectRoot -Path $VenvPath
    $python = Join-Path $venvRoot 'Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
        throw "Python was not found in '$venvRoot'. Run scripts/demo.ps1 -Install or scripts/check.ps1 -Install with the same -VenvPath. If needed, add -PythonPath with the full path to Python 3.12."
    }
    return $python
}

function Invoke-AuditPython {
    param([Parameter(Mandatory = $true)][string]$Python,
          [Parameter(Mandatory = $true)][string[]]$Arguments,
          [Parameter(Mandatory = $true)][string]$Action)
    $previousUtf8 = [Environment]::GetEnvironmentVariable('PYTHONUTF8', 'Process')
    $previousIoEncoding = [Environment]::GetEnvironmentVariable('PYTHONIOENCODING', 'Process')
    $previousConsoleEncoding = [Console]::OutputEncoding
    try {
        $env:PYTHONUTF8 = '1'
        $env:PYTHONIOENCODING = 'utf-8'
        [Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
        & $Python @Arguments
        $commandExitCode = $LASTEXITCODE
    } finally {
        [Environment]::SetEnvironmentVariable('PYTHONUTF8', $previousUtf8, 'Process')
        [Environment]::SetEnvironmentVariable('PYTHONIOENCODING', $previousIoEncoding, 'Process')
        [Console]::OutputEncoding = $previousConsoleEncoding
    }
    if ($commandExitCode -ne 0) { throw "$Action failed (exit code $commandExitCode)." }
}

function Get-AuditEnvironment {
    $snapshot = @{}
    foreach ($entry in [Environment]::GetEnvironmentVariables('Process').GetEnumerator()) {
        if ([string]$entry.Key -like 'CASH_AUDIT_*') { $snapshot[[string]$entry.Key] = [string]$entry.Value }
    }
    return $snapshot
}

function Clear-AuditEnvironment {
    $snapshot = Get-AuditEnvironment
    foreach ($name in $snapshot.Keys) { [Environment]::SetEnvironmentVariable($name, $null, 'Process') }
}

function Restore-AuditEnvironment {
    param([Parameter(Mandatory = $true)][hashtable]$Snapshot)
    Clear-AuditEnvironment
    foreach ($name in $Snapshot.Keys) { [Environment]::SetEnvironmentVariable($name, $Snapshot[$name], 'Process') }
}
