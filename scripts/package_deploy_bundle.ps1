param(
    [string]$OutputDir = "",
    [switch]$Clean,
    [switch]$WithModels,
    [ValidateSet("python", "external")]
    [string]$Backend = "python"
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
        "/XD", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".git", ".venv", "venv", "third_party", "dist", "output",
        "/XF", "*.pyc", "*.pyo", "*.pdb", "*.log"
    ) + $ExtraArgs

    & robocopy @args | Out-Host
    if ($LASTEXITCODE -gt 7) {
        throw "robocopy failed: $Source -> $Destination, exit code $LASTEXITCODE"
    }
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
    Remove-Item -LiteralPath $OutputDir -Recurse -Force
}

New-Item -ItemType Directory -Force $OutputDir | Out-Null

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
"@ | Set-Utf8LfContent -Path (Join-Path $OutputDir "models\README.txt")
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
$inferenceConfig = @"
inference:
  backend: $Backend

"@
$deployConfig = [regex]::Replace($deployConfig, "(?ms)^inference:\r?\n(?:^[ \t].*\r?\n)*", $inferenceConfig, 1)
if ($deployConfig -notmatch "(?m)^inference:") {
    $deployConfig = $inferenceConfig + $deployConfig
}
Set-Utf8LfContent -Path $bundleConfigPath -Content $deployConfig
Set-Utf8LfContent -Path $deployConfigPath -Content $deployConfig

@'
#!/usr/bin/env bash
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

if [[ -d .venv ]]; then
  source .venv/bin/activate
fi

exec python3 run.py --config "$DIR/config.deploy.yaml" "$@"
'@ | Set-Utf8LfContent -Path (Join-Path $OutputDir "start_robot.sh")

@'
# VoiceControl Deploy Bundle

This bundle contains the Python VoiceControl runtime.

## Target install

```bash
chmod +x scripts/install_robot_target.sh
scripts/install_robot_target.sh --mode webrtc
```

Install to `/opt/voice-control` and enable systemd:

```bash
chmod +x scripts/install_robot_target.sh
sudo -E scripts/install_robot_target.sh --install-dir /opt/voice-control --mode webrtc --systemd
```

## Models

If models were not packaged, copy them to:

```text
models/asr
models/nlu
models/kws
```

## Backend selection

`config.deploy.yaml` supports:

```yaml
inference:
  backend: python    # python / external
```

- `python`: start local Python ASR/NLU services and then the pipeline.
- `external`: connect to ASR/NLU services running on another machine via `services.asr_url` and `services.nlu_url`.
'@ | Set-Utf8LfContent -Path (Join-Path $OutputDir "DEPLOY.md")

Write-Host ""
Write-Host "Deploy bundle created:"
Write-Host "  $OutputDir"
