param(
    [ValidateRange(1, 65535)][int]$Port = 4173,
    [string]$PythonPath
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$localPython = Join-Path $projectRoot '.venv\Scripts\python.exe'
$parentPython = Join-Path (Split-Path -Parent $projectRoot) '.venv\Scripts\python.exe'
$serverArgs = @('-m', 'http.server', "$Port", '--bind', '127.0.0.1', '--directory', $PSScriptRoot)

Write-Host "Forte Audit interface: http://127.0.0.1:$Port"
Write-Host 'Stage 3 preview: files stay in the browser; analysis is not connected.'
Write-Host 'Only the frontend folder is served. Stop with Ctrl+C.'
if ($PythonPath) {
    & $PythonPath @serverArgs
} elseif (Test-Path -LiteralPath $localPython) {
    & $localPython @serverArgs
} elseif (Test-Path -LiteralPath $parentPython) {
    & $parentPython @serverArgs
} elseif (Get-Command py -ErrorAction SilentlyContinue) {
    & py -3 @serverArgs
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
    & python @serverArgs
} else {
    throw 'Python was not found. Pass -PythonPath with the full path to python.exe.'
}
if ($LASTEXITCODE -ne 0) { throw "Preview server stopped with exit code $LASTEXITCODE." }
