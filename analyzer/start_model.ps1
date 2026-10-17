param(
    [ValidateRange(1, 65535)][int]$Port = 8080,
    [ValidateRange(2048, 65536)][int]$ContextSize = 8192,
    [ValidateSet('2b', '4b', '8b')][string]$ModelSize = '2b',
    [ValidateRange(0, 8192)][int]$ImageMinTokens = 1024,
    [int]$ImageMaxTokens = 1280,
    [ValidateSet('f16', 'q8_0')][string]$CacheType = 'f16',
    [ValidateSet('auto', 'on', 'off')][string]$FlashAttention = 'auto',
    [switch]$Background
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$runtimePath = Join-Path $projectRoot '.cache\llama-b11669\llama-server.exe'
$modelVariant = $ModelSize.ToUpperInvariant()
$modelPath = Join-Path $projectRoot "models\qwen3-vl-$ModelSize\Qwen3VL-$modelVariant-Instruct-Q4_K_M.gguf"
$visionPath = Join-Path $projectRoot "models\qwen3-vl-$ModelSize\mmproj-Qwen3VL-$modelVariant-Instruct-F16.gguf"
if ($ImageMaxTokens -lt 1 -or $ImageMinTokens -gt $ImageMaxTokens) {
    throw 'ImageMaxTokens must be positive and at least ImageMinTokens.'
}
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
    '--image-min-tokens', "$ImageMinTokens",
    '--image-max-tokens', "$ImageMaxTokens",
    '--cache-type-k', $CacheType, '--cache-type-v', $CacheType,
    '--flash-attn', $FlashAttention,
    '--host', '127.0.0.1', '--port', "$Port", '--alias', 'forte-qwen'
)
if ($Background) {
    $logName = if ($Port -eq 8080) { 'runs\model-server' } else { "runs\model-server-$Port" }
    $logDirectory = Join-Path $projectRoot $logName
    New-Item -ItemType Directory -Force -Path $logDirectory | Out-Null
    $quotedArguments = $serverArguments | ForEach-Object {
        if ($_ -match '\s') { '"' + $_ + '"' } else { $_ }
    }
    $serverProcess = Start-Process -FilePath $runtimePath -ArgumentList $quotedArguments `
        -WorkingDirectory $projectRoot -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput (Join-Path $logDirectory 'stdout.log') `
        -RedirectStandardError (Join-Path $logDirectory 'stderr.log')
    $serverProcess.Id | Set-Content -LiteralPath (Join-Path $logDirectory 'pid.txt')
    @{
        model_size = $ModelSize; model_path = $modelPath; vision_path = $visionPath;
        port = $Port; context_size = $ContextSize; image_min_tokens = $ImageMinTokens;
        cache_type_k = $CacheType; cache_type_v = $CacheType; flash_attention = $FlashAttention;
        image_max_tokens = $ImageMaxTokens; process_id = $serverProcess.Id;
        started_utc = [DateTime]::UtcNow.ToString('o')
    } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $logDirectory 'launch.json') -Encoding UTF8
    Write-Output "Model server started, PID $($serverProcess.Id). Readiness must be checked at http://127.0.0.1:$Port/health."
} else {
    & $runtimePath @serverArguments
    exit $LASTEXITCODE
}
