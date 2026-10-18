[CmdletBinding()]
param(
    [ValidateRange(1, 65535)][int]$Port = 8000,
    [string]$PythonPath,
    [ValidateNotNullOrEmpty()][string]$VenvPath = '.venv-app',
    [switch]$Install,
    [switch]$StartModels,
    [ValidateRange(5, 300)][int]$ModelWaitSeconds = 90,
    [switch]$CheckOnly
)

$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'common.ps1')
$projectRoot = Split-Path -Parent $PSScriptRoot
$venvRoot = Resolve-AuditPath -ProjectRoot $projectRoot -Path $VenvPath
$sourceVenv = Join-Path $projectRoot '.venv'
$python = Join-Path $venvRoot 'Scripts\python.exe'
$realData = Join-Path $projectRoot 'backend\.local\real'
$savedEnvironment = Get-AuditEnvironment

function Assert-RealVenv {
    # The research environment is read only here, including when -Install is used.
    $sourceRoot = [System.IO.Path]::GetFullPath($sourceVenv).TrimEnd('\', '/')
    $targetRoot = [System.IO.Path]::GetFullPath($venvRoot).TrimEnd('\', '/')
    if ($targetRoot.Equals($sourceRoot, [StringComparison]::OrdinalIgnoreCase) -or
        $targetRoot.StartsWith($sourceRoot + '\', [StringComparison]::OrdinalIgnoreCase)) {
        throw 'Use a separate application environment, for example -VenvPath .venv-app. The analyzer .venv is not modified.'
    }
    if ((Test-Path -LiteralPath $venvRoot) -and
        ((Get-Item -LiteralPath $venvRoot).Attributes -band [IO.FileAttributes]::ReparsePoint)) {
        throw 'VenvPath must be a direct directory, not a link or junction.'
    }
}

function Initialize-RealVenv {
    Assert-RealVenv
    if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
        if (-not $Install) {
            throw 'Application Python was not found. Run scripts/real.ps1 -Install; add -PythonPath for Python 3.12 if needed.'
        }
        $sourcePython = Join-Path $sourceVenv 'Scripts\python.exe'
        $bootstrapArguments = @()
        if ($PythonPath) {
            $bootstrapPython = $PythonPath
        } elseif (Test-Path -LiteralPath $sourcePython -PathType Leaf) {
            $bootstrapPython = $sourcePython
        } elseif (Get-Command py -ErrorAction SilentlyContinue) {
            $bootstrapPython = (Get-Command py).Source
            $bootstrapArguments = @('-3.12')
        } elseif (Get-Command python -ErrorAction SilentlyContinue) {
            $bootstrapPython = (Get-Command python).Source
        } else {
            throw 'Install Python 3.12 or pass -PythonPath with its full executable path.'
        }
        Invoke-AuditPython -Python $bootstrapPython -Arguments ($bootstrapArguments + @(
            '-c', 'import sys, struct; assert sys.version_info[:2] == (3, 12) and struct.calcsize(chr(80)) == 8'
        )) -Action 'Python 3.12 x64 check'
        Invoke-AuditPython -Python $bootstrapPython -Arguments ($bootstrapArguments + @('-m', 'venv', $venvRoot)) -Action 'Application environment creation'
    }
    if ($Install) {
        Invoke-AuditPython -Python $python -Arguments @(
            '-m', 'pip', 'install', '-r', (Join-Path $projectRoot 'backend\requirements-real.txt')
        ) -Action 'Real application dependency installation'
    }
}

function Start-MissingLocalModels {
    $missing = @()
    foreach ($model in @(
        @{ Size = '2b'; Port = 8080; Context = 2048 },
        @{ Size = '4b'; Port = 8081; Context = 4096 }
    )) {
        if (Get-NetTCPConnection -State Listen -LocalPort $model.Port -ErrorAction SilentlyContinue) {
            Write-Host "Reusing listener on local model port $($model.Port); model identity will be checked."
        } else {
            $missing += $model
        }
    }
    # Validate all missing assets before starting the first new helper.
    foreach ($model in $missing) {
        $variant = $model.Size.ToUpperInvariant()
        foreach ($asset in @(
            '.cache\llama-b11669\llama-server.exe',
            "models\qwen3-vl-$($model.Size)\Qwen3VL-$variant-Instruct-Q4_K_M.gguf",
            "models\qwen3-vl-$($model.Size)\mmproj-Qwen3VL-$variant-Instruct-F16.gguf"
        )) {
            if (-not (Test-Path -LiteralPath (Join-Path $projectRoot $asset) -PathType Leaf)) {
                throw "Required local model asset missing: $asset. See analyzer/README.md for model setup."
            }
        }
    }
    foreach ($model in $missing) {
        & (Join-Path $projectRoot 'analyzer\start_model.ps1') -ModelSize $model.Size -Port $model.Port `
            -ContextSize $model.Context -ImageMinTokens 1024 -ImageMaxTokens 1280 `
            -CacheType f16 -FlashAttention auto -Background
    }
}

function Wait-LocalModels {
    $deadline = [DateTime]::UtcNow.AddSeconds($ModelWaitSeconds)
    do {
        $bothHealthy = $true
        foreach ($modelPort in @(8080, 8081)) {
            try {
                $health = Invoke-RestMethod -Uri "http://127.0.0.1:$modelPort/health" -TimeoutSec 2
                if ($health.status -ne 'ok') { $bothHealthy = $false }
            } catch {
                $bothHealthy = $false
            }
        }
        if ($bothHealthy) { return }
        if ([DateTime]::UtcNow -ge $deadline) { break }
        Start-Sleep -Milliseconds 500
    } while ([DateTime]::UtcNow -lt $deadline)
    throw "Both local models did not become healthy within $ModelWaitSeconds seconds. Check runs/model-server/ and runs/model-server-8081/. No existing process was stopped."
}

Push-Location -LiteralPath $projectRoot
try {
    # A prior demo/public session must not turn this local launcher into another mode.
    Clear-AuditEnvironment
    $env:CASH_AUDIT_MODE = 'real'
    $env:CASH_AUDIT_DATA_DIR = $realData
    Initialize-RealVenv
    if ($StartModels) {
        Start-MissingLocalModels
        Wait-LocalModels
    }
    # Includes analyzer.service.check_ready(): both model roles must match their weights.
    # This diagnostic makes no inference requests and has bounded child-process checks.
    Invoke-AuditPython -Python $python -Arguments @(
        '-m', 'backend.doctor', '--mode', 'real', '--port', [string]$Port, '--data-dir', $realData
    ) -Action 'Real application preflight'
    if ($CheckOnly) {
        Write-Host 'Real application preflight passed. No application server was started.'
        return
    }
    Write-Host "Real local analysis: http://127.0.0.1:$Port"
    Write-Host 'One application worker; no automatic mock fallback or reload.'
    Write-Host "Local data: $realData"
    Write-Host 'The API key is stored in backend/.local/real/api-key.txt; it is not printed.'
    Write-Host 'Stop the application with Ctrl+C. Model servers remain loaded for the next run.'
    Invoke-AuditPython -Python $python -Arguments @(
        '-m', 'uvicorn', 'backend.main:app', '--host', '127.0.0.1', '--port', [string]$Port, '--workers', '1'
    ) -Action 'Real application server'
} finally {
    Restore-AuditEnvironment -Snapshot $savedEnvironment
    Pop-Location
}
