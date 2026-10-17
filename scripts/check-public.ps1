[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][ValidateNotNullOrEmpty()][string]$BaseUrl,
    [Parameter(Mandatory = $true)][ValidateSet('mock', 'real')][string]$ExpectMode,
    [Parameter(Mandatory = $true)][ValidateNotNullOrEmpty()][string]$KeyFile,
    [ValidateNotNullOrEmpty()][string]$VenvPath = '.venv-app'
)

$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'common.ps1')
$projectRoot = Split-Path -Parent $PSScriptRoot
$keyPath = Resolve-AuditPath -ProjectRoot $projectRoot -Path $KeyFile
$python = Get-AuditPython -ProjectRoot $projectRoot -VenvPath $VenvPath
Push-Location -LiteralPath $projectRoot
try {
    Write-Host 'Checking public HTTPS access using two neutral synthetic videos.'
    $arguments = @('-m', 'backend.check_public', '--base-url', $BaseUrl,
                   '--expect-mode', $ExpectMode, '--key-file', $keyPath)
    Invoke-AuditPython -Python $python -Arguments $arguments -Action 'Public HTTPS check'
} finally {
    Pop-Location
}
