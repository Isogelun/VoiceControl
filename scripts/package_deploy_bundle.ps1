param(
    [string]$OutputDir = "",
    [switch]$Clean,
    [switch]$WithModels,
    [switch]$BuildRustWithWsl,
    [switch]$BuildRustCross,
    [ValidateSet("aarch64", "x86_64")]
    [string]$RustArch = "aarch64",
    [string]$OrtVersion = "1.23.0",
    [ValidateSet("rust", "mixed")]
    [string]$Backend = "rust",
    [string]$WslDistro = "",
    [string]$RustPackageDir = ""
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir

if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $OutputDir = Join-Path $ProjectRoot "dist\deploy-bundle-$stamp"
}
elseif (![System.IO.Path]::IsPathRooted($OutputDir)) {
    $OutputDir = Join-Path $ProjectRoot $OutputDir
}

function Invoke-RobocopyChecked {
    param(
        [string]$Source,
        [string]$Destination,
        [string[]]$ExtraArgs = @()
    )

    if (!(Test-Path $Source)) {
        Write-Host "Skip missing path: $Source"
        return
    }

    New-Item -ItemType Directory -Force $Destination | Out-Null
    $args = @(
        $Source,
        $Destination,
        "/E",
        "/NFL",
        "/NDL",
        "/NJH",
        "/NJS",
        "/NP",
        "/XD", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", "target", ".git", ".venv", "venv", "third_party", "dist", "output",
        "/XF", "*.pyc", "*.pyo", "*.pdb", "*.log"
    ) + $ExtraArgs

    & robocopy @args | Out-Host
    if ($LASTEXITCODE -gt 7) {
        throw "robocopy failed: $Source -> $Destination, exit code $LASTEXITCODE"
    }
}

function Convert-ToWslPath {
    param([string]$WindowsPath)
    $converted = & wsl wslpath -a $WindowsPath
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($converted)) {
        throw "Failed to convert path for WSL: $WindowsPath"
    }
    return $converted.Trim()
}

function Set-Utf8LfContent {
    param(
        [string]$Path,
        [string]$Content
    )

    $normalized = $Content -replace "`r`n", "`n" -replace "`r", "`n"
    $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllText($Path, $normalized, $utf8NoBom)
}

if ($Clean -and (Test-Path $OutputDir)) {
    $resolvedOutput = [System.IO.Path]::GetFullPath($OutputDir)
    $resolvedDist = [System.IO.Path]::GetFullPath((Join-Path $ProjectRoot "dist"))
    if (!$resolvedOutput.StartsWith($resolvedDist, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to clean outside dist/: $resolvedOutput"
    }
    Remove-Item -LiteralPath $resolvedOutput -Recurse -Force
}

New-Item -ItemType Directory -Force $OutputDir | Out-Null

if ($BuildRustWithWsl) {
    $wslProjectRoot = Convert-ToWslPath $ProjectRoot
    # The final deploy bundle keeps models only at the bundle root. The target
    # installer links rust/models back to ../models to avoid duplicating GBs.
    $bashCommand = "cd '$wslProjectRoot' && ORT_VERSION=$OrtVersion WITH_MODELS=0 bash scripts/package_voice_infer_linux.sh"

    Write-Host "Building Linux Rust package in WSL..."
    if ([string]::IsNullOrWhiteSpace($WslDistro)) {
        & wsl bash -lc $bashCommand
    }
    else {
        & wsl -d $WslDistro bash -lc $bashCommand
    }
    if ($LASTEXITCODE -ne 0) {
        throw "WSL Rust packaging failed with exit code $LASTEXITCODE"
    }
}

if ($BuildRustCross) {
    $crossArgs = @(
        "-ExecutionPolicy", "Bypass",
        "-File", (Join-Path $ScriptDir "cross_build_voice_infer_linux.ps1"),
        "-Arch", $RustArch,
        "-OrtVersion", $OrtVersion,
        "-Clean"
    )
    Write-Host "Cross-building Linux Rust package from Windows..."
    & powershell @crossArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Rust cross-build failed with exit code $LASTEXITCODE"
    }
}

if ([string]::IsNullOrWhiteSpace($RustPackageDir)) {
    $rustPackages = Get-ChildItem -Path (Join-Path $ProjectRoot "dist") -Directory -Filter "voice-infer-ubuntu2204-*" -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending
    if ($rustPackages.Count -gt 0) {
        $RustPackageDir = $rustPackages[0].FullName
    }
}
elseif (![System.IO.Path]::IsPathRooted($RustPackageDir)) {
    $RustPackageDir = Join-Path $ProjectRoot $RustPackageDir
}

$runtimeDirs = @(
    "asr",
    "nlu",
    "pipeline",
    "unitree_webrtc_connect",
    "scripts",
    "docs",
    "audio"
)

foreach ($dir in $runtimeDirs) {
    Invoke-RobocopyChecked -Source (Join-Path $ProjectRoot $dir) -Destination (Join-Path $OutputDir $dir)
}

if ($WithModels) {
    Invoke-RobocopyChecked -Source (Join-Path $ProjectRoot "models") -Destination (Join-Path $OutputDir "models")
}
else {
    New-Item -ItemType Directory -Force (Join-Path $OutputDir "models") | Out-Null
    @"
Place model directories here before deployment:
  models/asr
  models/nlu
  models/kws    optional, only if KWS wake word is used
"@ | Set-Content -Encoding UTF8 (Join-Path $OutputDir "models\README.txt")
}

$files = @(
    "README.md",
    "config.yaml",
    "pyproject.toml",
    "requirements-robot.txt",
    "requirements-server-py38.txt",
    "run.py",
    "uv.lock"
)

foreach ($file in $files) {
    $src = Join-Path $ProjectRoot $file
    if (Test-Path $src) {
        Copy-Item -LiteralPath $src -Destination (Join-Path $OutputDir $file) -Force
    }
}

$bundleConfigPath = Join-Path $OutputDir "config.yaml"
$deployConfigPath = Join-Path $OutputDir "config.deploy.yaml"
$deployConfig = Get-Content -Raw $bundleConfigPath
$managedRustConfig = @"
inference:
  backend: $Backend
  rust_binary: rust/voice-infer
  rust_ort_dylib: rust/libonnxruntime.so.$OrtVersion
  rust_ort_opt: disable
  rust_asr_max_new_tokens: 16
  rust_threads: 1

"@
$deployConfig = [regex]::Replace($deployConfig, "(?ms)^inference:\r?\n(?:^[ \t].*\r?\n)*", $managedRustConfig, 1)
if ($deployConfig -notmatch "(?m)^inference:") {
    $deployConfig = $managedRustConfig + $deployConfig
}
Set-Utf8LfContent -Path $bundleConfigPath -Content $deployConfig
Set-Utf8LfContent -Path $deployConfigPath -Content $deployConfig

if (![string]::IsNullOrWhiteSpace($RustPackageDir) -and (Test-Path $RustPackageDir)) {
    Invoke-RobocopyChecked -Source $RustPackageDir -Destination (Join-Path $OutputDir "rust") -ExtraArgs @("/XD", "models")
}
else {
    New-Item -ItemType Directory -Force (Join-Path $OutputDir "rust") | Out-Null
    @"
No Rust Linux package was copied.

Build one on Ubuntu 22.04 with:
  WITH_MODELS=1 bash scripts/package_voice_infer_linux.sh

Or rerun this script from Windows with:
  powershell -ExecutionPolicy Bypass -File scripts\package_deploy_bundle.ps1 -BuildRustWithWsl
"@ | Set-Content -Encoding UTF8 (Join-Path $OutputDir "rust\README.txt")
}

@'
#!/usr/bin/env bash
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ ! -x "$DIR/rust/voice-infer" ]]; then
  echo "rust/voice-infer not found or not executable. Build/copy the Rust package first." >&2
  exit 1
fi

cd "$DIR"
exec python3 run.py --config "$DIR/config.deploy.yaml" "$@"
'@ | ForEach-Object {
    Set-Utf8LfContent -Path (Join-Path $OutputDir "start_python_managed_rust.sh") -Content $_
    Set-Utf8LfContent -Path (Join-Path $OutputDir "start_rust_and_pipeline.sh") -Content $_
}

@'
# VoiceControl Deploy Bundle

This directory contains Python pipeline code and, when available, the Linux Rust `voice-infer` package.

## On the Ubuntu 22.04 target

One-command target install:

```bash
chmod +x scripts/install_robot_target.sh
scripts/install_robot_target.sh --mode webrtc
```

Install to `/opt/voice-control` and enable systemd:

```bash
chmod +x scripts/install_robot_target.sh
sudo -E scripts/install_robot_target.sh --install-dir /opt/voice-control --mode webrtc --systemd
```

Install Python runtime dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-robot.txt
```

If models were not packaged, copy them to:

```text
models/asr
models/nlu
models/kws
```

The deploy bundle stores models only in the root `models/` directory. The target
installer creates `rust/models` symlinks when needed.

If you want to run Rust directly after installation:

```bash
cd rust
./start.sh
```

Or start Rust plus Python pipeline together:

```bash
chmod +x start_python_managed_rust.sh
./start_python_managed_rust.sh --webrtc
```

For onboard microphone:

```bash
./start_python_managed_rust.sh --onboard
```

For serial microphone array:

```bash
./start_python_managed_rust.sh --hardware-serial
```

`start_python_managed_rust.sh` runs Python first. Python reads `config.deploy.yaml`, starts `rust/voice-infer`, waits for ASR/NLU health checks, then enters the selected pipeline mode.

## Backend selection

`config.deploy.yaml` supports:

```yaml
inference:
  backend: rust     # rust / mixed / python / external
```

- `rust` (default): Rust handles both ASR and NLU. Python only runs the pipeline.
- `mixed`: Python ASR + Rust NLU. Fallback if Rust ASR crashes on the target.
- `external`: ASR/NLU run elsewhere; configure `services.asr_url` and `services.nlu_url`.

If Rust ASR fails on your target (e.g. ORT INT4 crash on some aarch64 boards), switch to `mixed`.
'@ | ForEach-Object {
    Set-Utf8LfContent -Path (Join-Path $OutputDir "DEPLOY.md") -Content $_
}

Write-Host ""
Write-Host "Deploy bundle created:"
Write-Host "  $OutputDir"
Write-Host ""
if (![string]::IsNullOrWhiteSpace($RustPackageDir) -and (Test-Path $RustPackageDir)) {
    Write-Host "Included Rust package:"
    Write-Host "  $RustPackageDir"
}
else {
    Write-Host "Rust package was not included. See rust\\README.txt in the bundle."
}
