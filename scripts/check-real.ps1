[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][ValidateNotNullOrEmpty()][string]$BaseUrl,
    [Parameter(Mandatory = $true)][ValidateNotNullOrEmpty()][string]$KeyFile,
    [Parameter(Mandatory = $true)][ValidateNotNullOrEmpty()][string]$Video,
    [string]$SecondVideo,
    [ValidateSet('api', 'ui')][string]$Channel = 'api',
    [switch]$RequireEvents,
    [string]$Report,
    [ValidateNotNullOrEmpty()][string]$VenvPath = '.venv-app'
)

$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'common.ps1')
$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Get-AuditPython -ProjectRoot $projectRoot -VenvPath $VenvPath
$keyPath = Resolve-AuditPath -ProjectRoot $projectRoot -Path $KeyFile
$videoPath = Resolve-AuditPath -ProjectRoot $projectRoot -Path $Video
$arguments = @('-m', 'backend.check_real', '--base-url', $BaseUrl, '--key-file', $keyPath,
               '--video', $videoPath, '--channel', $Channel)
if ($SecondVideo) {
    $arguments += @('--second-video', (Resolve-AuditPath -ProjectRoot $projectRoot -Path $SecondVideo))
}
if ($RequireEvents) { $arguments += '--require-events' }
if ($Report) { $arguments += @('--report', (Resolve-AuditPath -ProjectRoot $projectRoot -Path $Report)) }
Push-Location -LiteralPath $projectRoot
try {
    Invoke-AuditPython -Python $python -Arguments $arguments -Action 'Explicit real HTTPS acceptance check'
} finally {
    Pop-Location
}
