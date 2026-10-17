param(
    [int]$Port = 8080,
    [int]$ContextSize = 8192,
    [int]$ImageMaxTokens = 512,
    [switch]$Background
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$runtimePath = Join-Path $projectRoot '.cache\llama-b11669\llama-server.exe'
$modelPath = Join-Path $projectRoot 'models\qwen3-vl-2b\Qwen3VL-2B-Instruct-Q4_K_M.gguf'
$visionPath = Join-Path $projectRoot 'models\qwen3-vl-2b\mmproj-Qwen3VL-2B-Instruct-F16.gguf'
foreach ($requiredPath in @($runtimePath, $modelPath, $visionPath)) {
    if (-not (Test-Path -LiteralPath $requiredPath -PathType Leaf)) {
        throw "Required local asset missing: $requiredPath. Run bootstrap_windows.py first."
    }
}
if (Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue) {
    throw "Port $Port is already in use. Existing processes were not changed."
}
$serverArguments = @(
    '-m', $modelPath, '--mmproj', $visionPath,
    '-ngl', 'all', '-c', "$ContextSize", '-np', '1',
    '--image-max-tokens', "$ImageMaxTokens",
    '--host', '127.0.0.1', '--port', "$Port", '--alias', 'forte-qwen'
)
if ($Background) {
    $logDirectory = Join-Path $projectRoot 'runs\model-server'
    New-Item -ItemType Directory -Force -Path $logDirectory | Out-Null
    $quotedArguments = $serverArguments | ForEach-Object {
        if ($_ -match '\s') { '"' + $_ + '"' } else { $_ }
    }
    $serverProcess = Start-Process -FilePath $runtimePath -ArgumentList $quotedArguments `
        -WorkingDirectory $projectRoot -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput (Join-Path $logDirectory 'stdout.log') `
        -RedirectStandardError (Join-Path $logDirectory 'stderr.log')
    $serverProcess.Id | Set-Content -LiteralPath (Join-Path $logDirectory 'pid.txt')
    Write-Output "Model server started, PID $($serverProcess.Id). Readiness must be checked at http://127.0.0.1:$Port/health."
} else {
    & $runtimePath @serverArguments
    exit $LASTEXITCODE
}
