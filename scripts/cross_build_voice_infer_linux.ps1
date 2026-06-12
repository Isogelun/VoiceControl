param(
    [ValidateSet("aarch64", "x86_64")]
    [string]$Arch = "aarch64",
    [string]$OrtVersion = "1.23.0",
    [switch]$WithModels,
    [string]$OutputDir = "",
    [switch]$Clean
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
$VoiceInferDir = Join-Path $ProjectRoot "voice-infer"
$ThirdPartyDir = Join-Path $ProjectRoot "third_party"
$DistRoot = Join-Path $ProjectRoot "dist"

if ($Arch -eq "aarch64") {
    $RustTarget = "aarch64-unknown-linux-gnu"
    $OrtArch = "aarch64"
}
else {
    $RustTarget = "x86_64-unknown-linux-gnu"
    $OrtArch = "x64"
}

if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $OutputDir = Join-Path $DistRoot "voice-infer-ubuntu2204-$Arch"
}
elseif (![System.IO.Path]::IsPathRooted($OutputDir)) {
    $OutputDir = Join-Path $ProjectRoot $OutputDir
}

function Assert-Command {
    param(
        [string]$Name,
        [string]$InstallHint
    )

    if (!(Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "$Name is required. $InstallHint"
    }
}

function Add-WinGetLinksToPath {
    $links = Join-Path $env:LOCALAPPDATA "Microsoft\WinGet\Links"
    if ((Test-Path $links) -and (($env:PATH -split ';') -notcontains $links)) {
        $env:PATH = "$links;$env:PATH"
    }
}

function Write-Utf8Lf {
    param(
        [string]$Path,
        [string]$Content
    )

    $normalized = $Content -replace "`r`n", "`n" -replace "`r", "`n"
    $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllText($Path, $normalized, $utf8NoBom)
}

Add-WinGetLinksToPath

Assert-Command "cargo" "Install Rust from https://rustup.rs/"
Assert-Command "rustup" "Install Rust from https://rustup.rs/"
Assert-Command "zig" "Install Zig and add it to PATH: https://ziglang.org/download/"

if (!(cargo zigbuild --help 2>$null)) {
    throw "cargo-zigbuild is required. Install it with: cargo install cargo-zigbuild"
}

New-Item -ItemType Directory -Force $ThirdPartyDir, $DistRoot | Out-Null

$OrtName = "onnxruntime-linux-$OrtArch-$OrtVersion"
$OrtTgz = Join-Path $ThirdPartyDir "$OrtName.tgz"
$ProjectOrtDir = Join-Path $ProjectRoot $OrtName
$ThirdPartyOrtDir = Join-Path $ThirdPartyDir $OrtName
$OrtDir = $ThirdPartyOrtDir
if (Test-Path (Join-Path $ProjectOrtDir "lib\libonnxruntime.so.$OrtVersion")) {
    $OrtDir = $ProjectOrtDir
}
$OrtLib = Join-Path $OrtDir "lib\libonnxruntime.so.$OrtVersion"
$OrtUrl = "https://github.com/microsoft/onnxruntime/releases/download/v$OrtVersion/$OrtName.tgz"

if (!(Test-Path $OrtLib)) {
    $OrtDir = $ThirdPartyOrtDir
    $OrtLib = Join-Path $OrtDir "lib\libonnxruntime.so.$OrtVersion"
    if (!(Test-Path $OrtTgz)) {
        Write-Host "Downloading $OrtUrl"
        Invoke-WebRequest -Uri $OrtUrl -OutFile $OrtTgz
    }
    tar -xzf $OrtTgz -C $ThirdPartyDir
}

if (!(Test-Path $OrtLib)) {
    throw "ONNX Runtime library not found: $OrtLib"
}

Write-Host "Adding Rust target: $RustTarget"
rustup target add $RustTarget

Write-Host "Cross-building voice-infer for $RustTarget"
Push-Location $VoiceInferDir
try {
    cargo zigbuild --release --target $RustTarget
}
finally {
    Pop-Location
}

$BuiltBinary = Join-Path $VoiceInferDir "target\$RustTarget\release\voice-infer"
if (!(Test-Path $BuiltBinary)) {
    throw "Built binary not found: $BuiltBinary"
}

if ($Clean -and (Test-Path $OutputDir)) {
    $resolvedOutput = [System.IO.Path]::GetFullPath($OutputDir)
    $resolvedDist = [System.IO.Path]::GetFullPath($DistRoot)
    if (!$resolvedOutput.StartsWith($resolvedDist, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to clean outside dist/: $resolvedOutput"
    }
    Remove-Item -LiteralPath $OutputDir -Recurse -Force
}

New-Item -ItemType Directory -Force $OutputDir | Out-Null
New-Item -ItemType Directory -Force (Join-Path $OutputDir "models") | Out-Null

Copy-Item -LiteralPath $BuiltBinary -Destination (Join-Path $OutputDir "voice-infer") -Force
Copy-Item -LiteralPath $OrtLib -Destination (Join-Path $OutputDir "libonnxruntime.so.$OrtVersion") -Force
$ProviderShared = Join-Path $OrtDir "lib\libonnxruntime_providers_shared.so"
if (Test-Path $ProviderShared) {
    Copy-Item -LiteralPath $ProviderShared -Destination (Join-Path $OutputDir "libonnxruntime_providers_shared.so") -Force
}

if ($WithModels) {
    Copy-Item -LiteralPath (Join-Path $ProjectRoot "models\asr") -Destination (Join-Path $OutputDir "models\asr") -Recurse -Force
    Copy-Item -LiteralPath (Join-Path $ProjectRoot "models\nlu") -Destination (Join-Path $OutputDir "models\nlu") -Recurse -Force
}
else {
    Write-Utf8Lf -Path (Join-Path $OutputDir "models\README.txt") -Content @"
Place or symlink model directories here before running:
  models/asr
  models/nlu
"@
}

Write-Utf8Lf -Path (Join-Path $OutputDir "start.sh") -Content (@'
#!/usr/bin/env bash
set -euo pipefail

DIR="\$(cd "\$(dirname "\${BASH_SOURCE[0]}")" && pwd)"
export ORT_DYLIB_PATH="\$DIR/libonnxruntime.so.__ORT_VERSION__"
export LD_LIBRARY_PATH="\$DIR:\${LD_LIBRARY_PATH:-}"

exec "\$DIR/voice-infer" \
  --asr-model-dir "\$DIR/models/asr" \
  --nlu-model-dir "\$DIR/models/nlu" \
  --host "\${VOICE_INFER_HOST:-0.0.0.0}" \
  --asr-port "\${VOICE_INFER_ASR_PORT:-8000}" \
  --nlu-port "\${VOICE_INFER_NLU_PORT:-8001}" \
  "\$@"
'@ -replace "__ORT_VERSION__", $OrtVersion)

Write-Utf8Lf -Path (Join-Path $OutputDir "verify.sh") -Content @'
#!/usr/bin/env bash
set -euo pipefail

HOST="\${VOICE_INFER_HOST:-127.0.0.1}"
ASR_PORT="\${VOICE_INFER_ASR_PORT:-8000}"
NLU_PORT="\${VOICE_INFER_NLU_PORT:-8001}"

curl -fsS "http://\${HOST}:\${ASR_PORT}/health"
echo
curl -fsS "http://\${HOST}:\${NLU_PORT}/health"
echo
curl -fsS -X POST "http://\${HOST}:\${NLU_PORT}/nlu" \
  -H 'Content-Type: application/json' \
  -d '{"text":"\u5411\u524d\u8d70\u4e09\u6b65"}'
echo
'@

Write-Utf8Lf -Path (Join-Path $OutputDir "README.txt") -Content @"
voice-infer Linux package

Target arch: $Arch
Rust target: $RustTarget
ONNX Runtime: $OrtName

On Ubuntu 22.04 target:
  chmod +x voice-infer start.sh verify.sh
  ./start.sh

In another shell:
  ./verify.sh
"@

Write-Host ""
Write-Host "Cross-compiled package created:"
Write-Host "  $OutputDir"
Write-Host ""
Write-Host "Copy this directory to the Ubuntu 22.04 target, then run:"
Write-Host "  chmod +x voice-infer start.sh verify.sh"
Write-Host "  ./start.sh"
