param(
    [ValidateSet('real', 'mock')][string]$Mode = 'real',
    [ValidateSet('clean', 'one', 'multiple', 'error')][string]$Scenario = 'one',
    [ValidateRange(1, 65535)][int]$Port = 8000,
    [string]$PythonPath,
    [switch]$Install
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Write-Host 'Starting the integrated interface and API. Use -Mode mock for synthetic test results.'
& (Join-Path $projectRoot 'backend\start.ps1') -Mode $Mode -Scenario $Scenario -Port $Port -PythonPath $PythonPath -Install:$Install
