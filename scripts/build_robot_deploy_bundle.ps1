param(
    [ValidateSet("python", "external")]
    [string]$Backend = "python",
    [switch]$NoModels,
    [string]$OutputDir = "",
    [switch]$Clean
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir

if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $OutputDir = Join-Path $ProjectRoot "dist\robot-deploy-$stamp"
}

$innerArgs = @(
    "-ExecutionPolicy", "Bypass",
    "-File", (Join-Path $ScriptDir "package_deploy_bundle.ps1"),
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
Write-Host "  Backend: $Backend"
Write-Host "  Output: $OutputDir"
Write-Host "  Include models: $(!$NoModels)"
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
