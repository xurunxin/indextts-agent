# Windows entry point for machines without Python, pip, uv or the CLI.
[CmdletBinding()]
param(
    [string]$Package = (Split-Path -Parent $PSScriptRoot),
    [string]$HomeDir = (Join-Path $env:USERPROFILE '.indextts-agent'),
    [string]$InstallDir,
    [ValidateSet('cuda', 'cpu')][string]$Device = 'cuda',
    [switch]$SkipModels,
    [switch]$SkipInit
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$uvVersion = '0.12.9'
$toolName = 'itt'
$resolvedHome = [IO.Path]::GetFullPath($HomeDir)
$uvCommand = Get-Command uv -ErrorAction SilentlyContinue
if ($uvCommand) {
    $uvPath = $uvCommand.Source
} else {
    $toolDir = Join-Path $resolvedHome 'bootstrap-tools'
    [IO.Directory]::CreateDirectory($toolDir) | Out-Null
    $uvPath = Join-Path $toolDir 'uv.exe'
    if (!(Test-Path -LiteralPath $uvPath)) {
        $platformSuffix = if ([Runtime.InteropServices.RuntimeInformation]::ProcessArchitecture -eq 'Arm64') { 'win_arm64.whl' } else { 'win_amd64.whl' }
        $metadata = Invoke-RestMethod -Uri "https://pypi.org/pypi/uv/$uvVersion/json"
        $asset = @($metadata.urls | Where-Object { $_.filename.EndsWith($platformSuffix) })
        if ($asset.Count -ne 1) { throw 'No matching uv binary for this Windows platform.' }
        $archivePath = Join-Path $toolDir $asset[0].filename
        Invoke-WebRequest -Uri $asset[0].url -OutFile $archivePath
        $actualHash = (Get-FileHash -LiteralPath $archivePath -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($actualHash -ne $asset[0].digests.sha256) { throw 'uv archive checksum mismatch.' }
        $archive = [IO.Compression.ZipFile]::OpenRead($archivePath)
        try {
            $entry = @($archive.Entries | Where-Object { $_.Name -eq 'uv.exe' })
            if ($entry.Count -ne 1) { throw 'uv.exe missing from the verified wheel.' }
            [IO.Compression.ZipFileExtensions]::ExtractToFile($entry[0], $uvPath, $false)
        } finally { $archive.Dispose() }
    }
}
& $uvPath --version
if ($LASTEXITCODE -ne 0) { throw 'uv cannot run on this machine.' }
& $uvPath tool install --force --python 3.11 $Package
if ($LASTEXITCODE -ne 0) { throw 'CLI installation failed; check the network and package path.' }
$binDir = & $uvPath tool dir --bin
if ($LASTEXITCODE -ne 0) { throw 'Could not locate the installed CLI.' }
$cliPath = Join-Path $binDir "$toolName.exe"
if (!$SkipInit) {
    $cliArgs = @('--home', $resolvedHome, 'init', '--device', $Device)
    if ($InstallDir) { $cliArgs += @('--install-dir', [IO.Path]::GetFullPath($InstallDir)) }
    if ($SkipModels) { $cliArgs += '--skip-models' }
    & $cliPath @cliArgs
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
Write-Output "Installed CLI: $cliPath"
