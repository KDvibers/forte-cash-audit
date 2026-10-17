[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$toolDirectory = Join-Path $projectRoot '.tools\cloudflared'
$destination = Join-Path $toolDirectory 'cloudflared.exe'
$version = '2026.10.1'
$expectedHash = '4d2490e74082c5d48dbfedaf92576fa5bd1702379ab88eb7faa0cfbe8dc891b6'
$downloadUrl = "https://github.com/cloudflare/cloudflared/releases/download/$version/cloudflared-windows-amd64.exe"

if (Test-Path -LiteralPath $destination -PathType Leaf) {
    if ((Get-FileHash -LiteralPath $destination -Algorithm SHA256).Hash.ToLowerInvariant() -eq $expectedHash) {
        Write-Host "cloudflared $version is already verified."
        return
    }
    throw 'Existing cloudflared.exe has a different checksum. Keep it separately or select a verified executable with -CloudflaredPath; no file was overwritten.'
}

New-Item -ItemType Directory -Path $toolDirectory -Force | Out-Null
$temporary = Join-Path $toolDirectory ('.download-' + [Guid]::NewGuid().ToString('N'))
$previousTls = [Net.ServicePointManager]::SecurityProtocol
try {
    [Net.ServicePointManager]::SecurityProtocol = $previousTls -bor [Net.SecurityProtocolType]::Tls12
    Write-Host "Downloading cloudflared $version from its official GitHub release..."
    Invoke-WebRequest -UseBasicParsing -Uri $downloadUrl -OutFile $temporary -TimeoutSec 180
    $actualHash = (Get-FileHash -LiteralPath $temporary -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actualHash -ne $expectedHash) { throw 'SHA256 verification failed. The downloaded executable will not be used.' }
    Move-Item -LiteralPath $temporary -Destination $destination
    Write-Host "Verified cloudflared ${version}: $destination"
} finally {
    [Net.ServicePointManager]::SecurityProtocol = $previousTls
    if (Test-Path -LiteralPath $temporary -PathType Leaf) { Remove-Item -LiteralPath $temporary }
}
