[CmdletBinding()]
param(
    [ValidateSet('mock', 'real')][string]$Mode = 'mock',
    [ValidateSet('clean', 'one', 'multiple', 'error')][string]$Scenario = 'multiple',
    [ValidateRange(1, 65535)][int]$Port = 8020,
    [ValidateRange(0, 240)][double]$DelaySeconds = 2,
    [string]$TailscalePath,
    [ValidateNotNullOrEmpty()][string]$VenvPath = '.venv-app',
    [string]$PythonPath,
    [switch]$Install,
    [switch]$CheckOnly
)

$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'common.ps1')
$projectRoot = Split-Path -Parent $PSScriptRoot
$savedEnvironment = Get-AuditEnvironment
Push-Location -LiteralPath $projectRoot
try {
    Clear-AuditEnvironment
    if ($Install -and $CheckOnly) {
        throw '-CheckOnly never installs packages. Run installation separately, then repeat -CheckOnly.'
    }
    if ($Install) {
        if ($Mode -eq 'real') {
            $requestedVenv = Resolve-AuditPath -ProjectRoot $projectRoot -Path $VenvPath
            if (Test-Path -LiteralPath $requestedVenv) {
                throw 'Real-mode -Install requires a NEW environment directory. Existing environments are never changed. Use -VenvPath .venv-real for a new environment, or omit -Install to use your working environment.'
            }
        }
        # Tailscale login and model downloads remain with the laptop owner.
        $setup = @{ Mode = $Mode; Install = $true; SetupOnly = $true; VenvPath = $VenvPath }
        if ($PythonPath) { $setup.PythonPath = $PythonPath }
        & (Join-Path $projectRoot 'backend\start.ps1') @setup
        if ($Mode -eq 'real') {
            $newPython = Get-AuditPython -ProjectRoot $projectRoot -VenvPath $VenvPath
            Invoke-AuditPython -Python $newPython -Arguments @('-m', 'pip', 'install', '-r',
                (Join-Path $projectRoot 'backend\requirements-real.txt')) -Action 'Combined API/analyzer dependencies'
        }
    }
    $python = Get-AuditPython -ProjectRoot $projectRoot -VenvPath $VenvPath
    $arguments = @('-m', 'backend.public_funnel', '--mode', $Mode, '--port', [string]$Port,
                   '--scenario', $Scenario,
                   '--delay', $DelaySeconds.ToString([Globalization.CultureInfo]::InvariantCulture))
    if ($TailscalePath) {
        $tailscale = Resolve-AuditPath -ProjectRoot $projectRoot -Path $TailscalePath
        $arguments += @('--tailscale', $tailscale)
    }
    if ($CheckOnly) { $arguments += '--check-only' }
    Invoke-AuditPython -Python $python -Arguments $arguments -Action 'Tailscale Funnel public access'
} finally {
    Restore-AuditEnvironment -Snapshot $savedEnvironment
    Pop-Location
}
