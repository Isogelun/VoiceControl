# VoiceControl

面向宇树 Go2 机器狗的语音控制系统。主链路是：

```text
唤醒词检测 -> VAD 语音切分 -> ASR 语音识别 -> NLU 意图理解 -> 指令 JSON -> 机器狗动作/语音反馈
```

项目当前包含两套推理实现：

- Python 推理服务：现有主实现，包含 ASR、NLU、Pipeline、Go2 WebRTC/串口/本机麦克风接入。
- Rust 推理服务：位于 `voice-infer/`，目标是用单二进制替代 Python ASR+NLU，降低 Ubuntu 22.04 机器狗端部署复杂度。

## 当前状态

- Python 全链路仍是主要可运行路径。
- Rust `voice-infer` 已能编译并通过默认测试：`33 passed, 6 ignored`。
- Rust NLU 真实 ONNX 推理、HTTP `/nlu` 已在 Windows 本地通过验证。
- Rust ASR HTTP `/asr` 已能跑通测试音频并返回与 Python 参考一致的文本。
- Rust ASR 仍有一个 `decoder_init` logits 严格数值对齐测试未完全收口，端到端文本结果已经可用。
- 大模型、ONNX Runtime 下载包、虚拟环境和编译产物不入库，见 `.gitignore`。

更详细的 Rust 进度见：

- `docs/rust-rewrite-status.md`
- `docs/windows-rust-e2e.md`
- `docs/ubuntu2204-rust-deploy.md`
- `voice-infer/README.md`

## 目录结构

```text
VoiceControl/
├── run.py                         # Python 统一启动入口
├── config.yaml                    # 默认运行配置
├── pyproject.toml                 # Python 3.10 依赖
├── requirements-robot.txt         # 机器狗端 Python 3.8 依赖
├── requirements-server-py38.txt   # 上位机/服务端 Python 3.8 依赖
├── asr/                           # Python ASR 服务与推理
├── nlu/                           # Python NLU 服务与推理
├── pipeline/                      # 唤醒、VAD、ASR/NLU 客户端、动作分发
├── unitree_webrtc_connect/        # Unitree Go2 WebRTC 接入
├── voice-infer/                   # Rust ASR+NLU 推理服务
├── scripts/                       # 部署、参考数据导出、对比脚本
├── tests/                         # Python 测试与参考数据
├── docs/                          # 设计、部署、Rust 重写文档
├── audio/                         # 反馈音频资源
└── models/                        # 模型目录，本地放置，不提交 Git
```

## 模型文件

模型文件默认放在 `models/`，该目录已被 `.gitignore` 忽略。

```text
models/
├── asr/
│   ├── encoder.int4.onnx
│   ├── decoder*
│   ├── decoder_weights.int4.data
│   ├── embed_tokens.bin
│   └── tokenizer.json
├── nlu/
│   ├── encoder.onnx
│   ├── decoder.onnx
│   └── tokenizer/
└── kws/
    └── sherpa-onnx wake word models
```

不要把 `models/`、`.venv/`、`third_party/`、`voice-infer/target/` 提交到 GitHub。

## Python 环境

推荐开发环境使用 Python 3.10：

```bash
pip install -e .
```

如果目标环境只有 Python 3.8，可以使用拆分依赖：

```bash
# 推理服务端
pip install -r requirements-server-py38.txt

# 机器狗端 pipeline
pip install -r requirements-robot.txt
```

## 启动方式

默认读取项目根目录的 `config.yaml`。

```bash
# 启动 ASR + NLU + pipeline，默认音频源由 config.yaml 的 audio.source 决定
python run.py

# 只启动 ASR HTTP 服务，默认端口 8000
python run.py --serve-asr

# 只启动 NLU HTTP 服务，默认端口 8001
python run.py --serve-nlu

# 只启动 pipeline，ASR/NLU 使用外部已启动服务
python run.py --pipeline-only
```

音频源模式：

```bash
# 本机麦克风
python run.py --onboard

# Unitree Go2 WebRTC 音频
python run.py --webrtc

# 串口麦克风阵列/硬件唤醒
python run.py --hardware-serial
```

常用调试参数：

```bash
python run.py --config config.local.yaml
python run.py --gpu
python run.py --denoise
python run.py --vad-mode silence --vad-silence-timeout-ms 500
python run.py --preflight-only --webrtc
```

## 配置

`config.yaml` 是主要配置入口。常用字段：

```yaml
server:
  host: 0.0.0.0
  asr_port: 8000
  nlu_port: 8001
  gpu: false

inference:
  backend: python   # python / rust / external
  rust_binary:
  rust_ort_dylib:

audio:
  source: onboard   # onboard / webrtc / hardware_serial

models:
  asr: models/asr
  nlu: models/nlu
  nlu_tokenizer: models/nlu/tokenizer

services:
  asr_timeout: 3
  nlu_timeout: 1.2

robot:
  ip: 10.10.20.66
  connection_method: LocalSTA

vad:
  mode: silence
  silence_timeout_ms: 300
  command_listen_timeout_ms: 8000

command:
  output_dir: output
  service_url: http://127.0.0.1:8090/api/v1/local/motion
```

命令行参数优先级高于配置文件。部署到真实机器狗前，建议把本地 IP、动作服务地址、唤醒词、反馈音频路径放到 `config.local.yaml`，避免把个人环境配置提交到仓库。

`inference.backend` 用于选择推理后端：

- `python`：默认值，由 `run.py` 启动 Python ASR/NLU 服务。
- `rust`：由 `run.py` 启动 `voice-infer`，pipeline 自动连接 Rust ASR/NLU。
- `external`：`run.py` 不启动推理服务，只使用 `services.asr_url` 和 `services.nlu_url`，适合 systemd 单独托管 Rust 服务。

## Rust 推理服务

Rust 服务位于 `voice-infer/`，HTTP 接口目标是兼容 Python 版 ASR/NLU：

- `POST /asr`
- `POST /nlu`
- `GET /health`

Windows 本地验证示例：

```powershell
.\scripts\start_voice_infer_windows.ps1
```

手动启动示例：

```powershell
$env:ORT_DYLIB_PATH="E:\HuaJianCode\VoiceControl\third_party\onnxruntime-win-x64-1.19.2\lib\onnxruntime.dll"
cd voice-infer
cargo run -- --asr-model-dir ..\models\asr --nlu-model-dir ..\models\nlu
```

Ubuntu 22.04 打包：

```bash
bash scripts/package_voice_infer_linux.sh
```

详细说明见 `docs/ubuntu2204-rust-deploy.md`。

## 测试

Python 测试：

```bash
pytest
```

Rust 默认测试：

```bash
cd voice-infer
cargo test
```

Rust ONNX Runtime 真实推理测试默认是 `ignored`，需要先配置 `ORT_DYLIB_PATH`，再显式运行指定测试。

```bash
cd voice-infer
ORT_DYLIB_PATH=/path/to/libonnxruntime.so.1.19.2 cargo test test_predict_forward_matches_reference -- --ignored
```

Windows PowerShell：

```powershell
$env:ORT_DYLIB_PATH="E:\path\to\onnxruntime.dll"
cargo test test_predict_forward_matches_reference -- --ignored
```

## GitHub 提交注意事项

应该提交：

- Python 源码：`asr/`、`nlu/`、`pipeline/`、`unitree_webrtc_connect/`
- Rust 源码：`voice-infer/src/`、`voice-infer/Cargo.toml`、`voice-infer/Cargo.lock`
- 文档：`README.md`、`docs/`
- 脚本：`scripts/`
- 小型测试 fixture：`tests/fixtures/`、`voice-infer/tests/resources/`

不应该提交：

- `models/`
- `.venv/`
- `third_party/`
- `voice-infer/target/`
- `output/`
- `__pycache__/`
- `*.pdb`、日志、本地临时文件

如果模型需要对外分发，建议使用 GitHub Releases、对象存储或单独的模型下载脚本，不要直接放进 Git 仓库。
