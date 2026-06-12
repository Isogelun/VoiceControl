param(
    [ValidateSet("aarch64", "x86_64")]
    [string]$Arch = "aarch64",
    [string]$OrtVersion = "1.23.0",
    [ValidateSet("rust", "mixed")]
    [string]$Backend = "rust",
    [switch]$NoModels,
    [string]$OutputDir = "",
    [switch]$Clean
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir

if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $OutputDir = Join-Path $ProjectRoot "dist\robot-deploy-$Arch-$stamp"
}

function Test-Command {
    param([string]$Name)
    return [bool](Get-Command $Name -ErrorAction SilentlyContinue)
}

$wingetLinks = Join-Path $env:LOCALAPPDATA "Microsoft\WinGet\Links"
if ((Test-Path $wingetLinks) -and (($env:PATH -split ';') -notcontains $wingetLinks)) {
    $env:PATH = "$wingetLinks;$env:PATH"
}

if (!(Test-Command "cargo")) {
    throw "cargo not found. Install Rust first: https://rustup.rs/"
}

if (!(Test-Command "rustup")) {
    throw "rustup not found. Install Rust first: https://rustup.rs/"
}

if (!(Test-Command "zig")) {
    throw "zig not found. Install Zig, add zig.exe to PATH, then rerun this script: https://ziglang.org/download/"
}

if (!(cargo zigbuild --help 2>$null)) {
    throw "cargo-zigbuild not found. Install it with: cargo install cargo-zigbuild"
}

$innerArgs = @(
    "-ExecutionPolicy", "Bypass",
    "-File", (Join-Path $ScriptDir "package_deploy_bundle.ps1"),
    "-BuildRustCross",
    "-RustArch", $Arch,
    "-OrtVersion", $OrtVersion,
    "-Backend", $Backend,
    "-OutputDir", $OutputDir
)

if (!$NoModels) {
    $innerArgs += "-WithModels"
}

if ($Clean) {
    $innerArgs += "-Clean"
}

Write-Host "Building robot deploy bundle..."
Write-Host "  Arch: $Arch"
Write-Host "  Backend: $Backend"
Write-Host "  ONNX Runtime: $OrtVersion"
Write-Host "  Output: $OutputDir"
Write-Host "  Include root models: $(!$NoModels)"
Write-Host "  Rust package models: excluded; target installer creates rust/models symlinks"
Write-Host ""

& powershell @innerArgs
if ($LASTEXITCODE -ne 0) {
    throw "deploy bundle build failed with exit code $LASTEXITCODE"
}

Write-Host ""
Write-Host "Done. Copy this directory to the robot/Ubuntu target:"
Write-Host "  $OutputDir"
Write-Host ""
Write-Host "On target:"
Write-Host "  cd <copied-dir>"
Write-Host "  chmod +x scripts/install_robot_target.sh"
Write-Host "  scripts/install_robot_target.sh --mode onboard"
