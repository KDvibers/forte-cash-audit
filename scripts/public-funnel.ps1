[CmdletBinding()]
param(
    [ValidateSet('mock', 'real')][string]$Mode = 'mock',
    [ValidateSet('clean', 'one', 'multiple', 'error')][string]$Scenario = 'multiple',
    [ValidateRange(1, 65535)][int]$Port = 8020,
    [ValidateRange(0, 240)][double]$DelaySeconds = 2,
    [string]$TailscalePath,
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
        # Install application packages only; Tailscale login remains with the owner.
        $setup = @{ Mode = $Mode; Install = $true; SetupOnly = $true; VenvPath = $VenvPath }
        if ($PythonPath) { $setup.PythonPath = $PythonPath }
        & (Join-Path $projectRoot 'backend\start.ps1') @setup
    }
    $python = Get-AuditPython -ProjectRoot $projectRoot -VenvPath $VenvPath
    $arguments = @('-m', 'backend.public_funnel', '--mode', $Mode, '--port', [string]$Port,
                   '--scenario', $Scenario,
                   '--delay', $DelaySeconds.ToString([Globalization.CultureInfo]::InvariantCulture))
    if ($TailscalePath) {
        $tailscale = Resolve-AuditPath -ProjectRoot $projectRoot -Path $TailscalePath
        $arguments += @('--tailscale', $tailscale)
    }
    Invoke-AuditPython -Python $python -Arguments $arguments -Action 'Tailscale Funnel public access'
} finally {
    Restore-AuditEnvironment -Snapshot $savedEnvironment
    Pop-Location
}
