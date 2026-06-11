# Windows Rust 端到端验证

本文档用于在 Windows 本地启动 `voice-infer`，验证 Rust ONNX 推理服务。

## 1. ONNX Runtime

需要 Windows x64 ONNX Runtime 1.19.2：

```powershell
New-Item -ItemType Directory -Force third_party
Invoke-WebRequest `
  -Uri https://github.com/microsoft/onnxruntime/releases/download/v1.19.2/onnxruntime-win-x64-1.19.2.zip `
  -OutFile third_party\onnxruntime-win-x64-1.19.2.zip
Expand-Archive `
  -Path third_party\onnxruntime-win-x64-1.19.2.zip `
  -DestinationPath third_party `
  -Force
```

DLL 路径应为：

```text
third_party\onnxruntime-win-x64-1.19.2\lib\onnxruntime.dll
```

## 2. 启动 NLU

```powershell
powershell -ExecutionPolicy Bypass -File scripts\start_voice_infer_windows.ps1 -NluOnly
```

另开一个 PowerShell：

```powershell
curl.exe -fsS http://127.0.0.1:9001/health
curl.exe -fsS -X POST http://127.0.0.1:9001/nlu `
  -H "Content-Type: application/json" `
  -d "{\"text\":\"\u5411\u524d\u8d70\u4e09\u6b65\"}"
```

预期 intent：

```text
move_forward
```

## 3. 启动 ASR

```powershell
powershell -ExecutionPolicy Bypass -File scripts\start_voice_infer_windows.ps1 -AsrOnly -AsrMaxNewTokens 8
```

另开一个 PowerShell：

```powershell
curl.exe -fsS http://127.0.0.1:9000/health
curl.exe -fsS -X POST http://127.0.0.1:9000/asr `
  -F "audio=@tests\fixtures\test_1s.wav" `
  -F "language=auto" `
  -F "use_itn=true"
```

当前测试音频的预期文本：

```text
odicologyThe.
```

## 4. 已知设置

Windows 本地默认使用：

- `ORT_DYLIB_PATH=third_party\onnxruntime-win-x64-1.19.2\lib\onnxruntime.dll`
- `VOICE_INFER_ORT_OPT=disable`

`VOICE_INFER_ORT_OPT=disable` 用于避免 Windows 本地图优化加载过慢；后续可以逐步尝试 `level1`、`level2`、`level3`。
