param(
    [ValidateSet('real', 'mock')][string]$Mode = 'real',
    [ValidateSet('clean', 'one', 'multiple', 'error')][string]$Scenario = 'one',
    [ValidateRange(1, 65535)][int]$Port = 8000,
    [string]$BindAddress = '127.0.0.1',
    [string]$PythonPath,
    [switch]$Install,
    [switch]$Dev,
    [switch]$SetupOnly
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$venvPython = Join-Path $projectRoot '.venv\Scripts\python.exe'
Set-Location -LiteralPath $projectRoot

function Assert-CommandSucceeded([string]$Action) {
    if ($LASTEXITCODE -ne 0) {
        throw "$Action failed (exit code $LASTEXITCODE)."
    }
}

if (-not (Test-Path -LiteralPath $venvPython)) {
    if (-not ($Install -or $Dev)) {
        throw 'First run: add -Install. Python 3.12 is recommended; use -PythonPath if Python is not on PATH.'
    }
    $parentPython = Join-Path (Split-Path -Parent $projectRoot) '.venv\Scripts\python.exe'
    if ($PythonPath) {
        & $PythonPath -m venv (Join-Path $projectRoot '.venv')
    } elseif (Test-Path -LiteralPath $parentPython) {
        & $parentPython -m venv (Join-Path $projectRoot '.venv')
    } elseif (Get-Command py -ErrorAction SilentlyContinue) {
        & py -3.12 -m venv (Join-Path $projectRoot '.venv')
    } elseif (Get-Command python -ErrorAction SilentlyContinue) {
        & python -m venv (Join-Path $projectRoot '.venv')
    } else {
        throw 'Install Python 3.12 or pass -PythonPath with the full path to python.exe.'
    }
    Assert-CommandSucceeded 'Virtual environment creation'
}

if ($Install -or $Dev) {
    & $venvPython -m pip install -r (Join-Path $PSScriptRoot 'requirements.txt')
    Assert-CommandSucceeded 'Server dependency installation'
}
if ($Dev) {
    & $venvPython -m pip install -r (Join-Path $PSScriptRoot 'requirements-dev.txt')
    Assert-CommandSucceeded 'Test dependency installation'
}
& $venvPython -c 'import fastapi, uvicorn, multipart, PIL, imageio_ffmpeg'
Assert-CommandSucceeded 'Dependency check (use -Install to install dependencies)'
if ($SetupOnly) {
    Write-Host 'Environment ready. Run again without -SetupOnly to start the server.'
    return
}

$env:CASH_AUDIT_MODE = $Mode
$env:CASH_AUDIT_MOCK_SCENARIO = $Scenario
Write-Host "API: http://${BindAddress}:$Port | mode: $Mode"
if ($Mode -eq 'mock') {
    Write-Host "MOCK: synthetic results only; scenario: $Scenario. No model inference."
}
Write-Host 'Health: /health | Stop: Ctrl+C | One server worker, no auto-reload.'
Write-Host 'Default API key file: backend/.local/api-key.txt (never commit this file).'
& $venvPython -m uvicorn backend.main:app --host $BindAddress --port $Port --workers 1
Assert-CommandSucceeded 'Server'
