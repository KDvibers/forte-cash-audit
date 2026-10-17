[CmdletBinding()]
param(
    [switch]$Install,
    [string]$PythonPath,
    [ValidateNotNullOrEmpty()][string]$VenvPath = '.venv-app',
    [string]$NodePath
)

$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'common.ps1')
$projectRoot = Split-Path -Parent $PSScriptRoot
Push-Location -LiteralPath $projectRoot
try {
    if ($Install) {
        $setup = @{ Install = $true; Dev = $true; SetupOnly = $true; VenvPath = $VenvPath }
        if ($PythonPath) { $setup.PythonPath = $PythonPath }
        & (Join-Path $projectRoot 'backend\start.ps1') @setup
    }
    $python = Get-AuditPython -ProjectRoot $projectRoot -VenvPath $VenvPath
    $arguments = @('-m', 'backend.selfcheck')
    if ($NodePath) { $arguments += @('--node', $NodePath) }
    Write-Host 'Checking backend tests, frontend tests, and a temporary mock HTTP server.'
    Invoke-AuditPython -Python $python -Arguments $arguments -Action 'Application self-check'
} finally {
    Pop-Location
}
