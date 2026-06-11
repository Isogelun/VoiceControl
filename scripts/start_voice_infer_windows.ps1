param(
    [switch]$AsrOnly,
    [switch]$NluOnly,
    [string]$HostAddress = "127.0.0.1",
    [int]$AsrPort = 9000,
    [int]$NluPort = 9001,
    [string]$OrtVersion = "1.19.2",
    [string]$OrtOpt = "disable",
    [int]$AsrMaxNewTokens = 32
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
$VoiceInferDir = Join-Path $ProjectRoot "voice-infer"
$OrtDll = Join-Path $ProjectRoot "third_party\onnxruntime-win-x64-$OrtVersion\lib\onnxruntime.dll"

if (!(Test-Path $OrtDll)) {
    throw "ONNX Runtime DLL not found: $OrtDll. Download onnxruntime-win-x64-$OrtVersion.zip into third_party first."
}

Push-Location $VoiceInferDir
try {
    cargo build

    $env:ORT_DYLIB_PATH = $OrtDll
    $env:VOICE_INFER_ORT_OPT = $OrtOpt
    $env:QWEN_ASR_MAX_NEW_TOKENS = "$AsrMaxNewTokens"

    $args = @(
        "--asr-model-dir", (Join-Path $ProjectRoot "models\asr"),
        "--nlu-model-dir", (Join-Path $ProjectRoot "models\nlu"),
        "--nlu-tokenizer-dir", (Join-Path $ProjectRoot "models\nlu\tokenizer"),
        "--host", $HostAddress,
        "--asr-port", "$AsrPort",
        "--nlu-port", "$NluPort"
    )

    if ($AsrOnly) {
        $args += "--asr-only"
    }
    if ($NluOnly) {
        $args += "--nlu-only"
    }

    & ".\target\debug\voice-infer.exe" @args
}
finally {
    Pop-Location
}
