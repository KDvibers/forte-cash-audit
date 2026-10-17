[CmdletBinding()]
param(
    [ValidateSet('mock', 'real')][string]$Mode = 'mock',
    [ValidateSet('clean', 'one', 'multiple', 'error')][string]$Scenario = 'multiple',
    [ValidateRange(1, 65535)][int]$Port = 8010,
    [ValidateRange(0, 240)][double]$DelaySeconds = 2,
    [string]$CloudflaredPath = '.tools\cloudflared\cloudflared.exe',
    [ValidateNotNullOrEmpty()][string]$VenvPath = '.venv-app',
    [string]$PythonPath,
    [switch]$Install
)

$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'common.ps1')
$projectRoot = Split-Path -Parent $PSScriptRoot
$savedEnvironment = Get-AuditEnvironment
Push-Location -LiteralPath $projectRoot
try {
    Clear-AuditEnvironment
    if ($Install) {
        $setup = @{ Mode = $Mode; Install = $true; SetupOnly = $true; VenvPath = $VenvPath }
        if ($PythonPath) { $setup.PythonPath = $PythonPath }
        & (Join-Path $projectRoot 'backend\start.ps1') @setup
        & (Join-Path $PSScriptRoot 'install-tunnel.ps1')
    }
    $python = Get-AuditPython -ProjectRoot $projectRoot -VenvPath $VenvPath
    $tunnel = Resolve-AuditPath -ProjectRoot $projectRoot -Path $CloudflaredPath
    $arguments = @('-m', 'backend.public_tunnel', '--mode', $Mode, '--port', [string]$Port,
                   '--cloudflared', $tunnel, '--scenario', $Scenario,
                   '--delay', $DelaySeconds.ToString([Globalization.CultureInfo]::InvariantCulture))
    Invoke-AuditPython -Python $python -Arguments $arguments -Action 'Public access'
} finally {
    Restore-AuditEnvironment -Snapshot $savedEnvironment
    Pop-Location
}
