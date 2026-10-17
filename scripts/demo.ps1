[CmdletBinding()]
param(
    [ValidateSet('clean', 'one', 'multiple', 'error')][string]$Scenario = 'multiple',
    [ValidateRange(1, 65535)][int]$Port = 8000,
    [ValidateRange(0, 240)][double]$DelaySeconds = 15,
    [string]$PythonPath,
    [switch]$Install,
    [ValidateNotNullOrEmpty()][string]$VenvPath = '.venv-app'
)

$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'common.ps1')
$projectRoot = Split-Path -Parent $PSScriptRoot
$startScript = Join-Path $projectRoot 'backend\start.ps1'
$demoData = Join-Path $projectRoot 'backend\.local\demo'
$sample = Join-Path $demoData 'sample.mov'
$savedEnvironment = Get-AuditEnvironment
Push-Location -LiteralPath $projectRoot
try {
    Clear-AuditEnvironment
    $env:CASH_AUDIT_DATA_DIR = $demoData
    $env:CASH_AUDIT_MOCK_DELAY_SECONDS = $DelaySeconds.ToString([Globalization.CultureInfo]::InvariantCulture)
    if ($Install) {
        $setup = @{ Mode = 'mock'; Install = $true; SetupOnly = $true; VenvPath = $VenvPath }
        if ($PythonPath) { $setup.PythonPath = $PythonPath }
        & $startScript @setup
    }
    $python = Get-AuditPython -ProjectRoot $projectRoot -VenvPath $VenvPath
    Invoke-AuditPython -Python $python -Arguments @('-m', 'backend.doctor', '--mode', 'mock', '--port', [string]$Port, '--data-dir', $demoData) -Action 'Demo preflight'
    Invoke-AuditPython -Python $python -Arguments @('-m', 'backend.make_sample', '--output', $sample) -Action 'Synthetic MOV creation'
    Write-Host "MOCK DEMO ONLY: synthetic results, no model inference. Scenario: $Scenario."
    Write-Host "Open this URL manually: http://127.0.0.1:$Port"
    Write-Host "Synthetic sample: $sample"
    Write-Host "Analysis delay: $DelaySeconds seconds. Data: $demoData"
    Write-Host 'The server stays in this terminal. Stop with Ctrl+C.'
    & $startScript -Mode mock -Scenario $Scenario -Port $Port -BindAddress '127.0.0.1' -VenvPath $VenvPath
} finally {
    Restore-AuditEnvironment -Snapshot $savedEnvironment
    Pop-Location
}
