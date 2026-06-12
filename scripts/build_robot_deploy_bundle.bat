@echo off
setlocal

set SCRIPT_DIR=%~dp0
powershell -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_DIR%build_robot_deploy_bundle.ps1" %*

if errorlevel 1 (
  echo.
  echo Build failed.
  exit /b %errorlevel%
)

echo.
echo Build finished.
