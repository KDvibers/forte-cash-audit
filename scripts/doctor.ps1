[CmdletBinding()]
param(
    [ValidateSet('mock', 'real')][string]$Mode = 'mock',
    [ValidateRange(1, 65535)][int]$Port = 8000,
    [ValidateNotNullOrEmpty()][string]$DataDir = 'backend\.local\demo',
    [switch]$SkipPort,
    [ValidateNotNullOrEmpty()][string]$VenvPath = '.venv-app'
)

$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'common.ps1')
$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Get-AuditPython -ProjectRoot $projectRoot -VenvPath $VenvPath
$resolvedData = Resolve-AuditPath -ProjectRoot $projectRoot -Path $DataDir
$arguments = @('-m', 'backend.doctor', '--mode', $Mode, '--port', [string]$Port, '--data-dir', $resolvedData)
if ($SkipPort) { $arguments += '--skip-port' }
Push-Location -LiteralPath $projectRoot
try {
    Invoke-AuditPython -Python $python -Arguments $arguments -Action 'Environment diagnosis'
} finally {
    Pop-Location
}
