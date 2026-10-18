[CmdletBinding()]
param(
    [ValidateRange(1, 65535)][int]$Port = 8020,
    [ValidateNotNullOrEmpty()][string]$VenvPath = '.venv-app',
    [string]$TailscalePath,
    [string]$PythonPath,
    [switch]$Install,
    [switch]$CheckOnly
)

$ErrorActionPreference = 'Stop'
# Models must already be running locally. This command never owns or stops them.
$options = @{ Mode = 'real'; Port = $Port; VenvPath = $VenvPath;
              Install = $Install; CheckOnly = $CheckOnly }
if ($TailscalePath) { $options.TailscalePath = $TailscalePath }
if ($PythonPath) { $options.PythonPath = $PythonPath }
& (Join-Path $PSScriptRoot 'public-funnel.ps1') @options
