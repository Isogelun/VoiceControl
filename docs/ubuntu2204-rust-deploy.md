# Ubuntu 22.04 Rust 推理部署

本文档用于机器狗或上位机的 Ubuntu 22.04 环境，目标是部署 `voice-infer` Rust ASR/NLU 服务，并完成端到端验证。

## 1. 目标平台

先在目标机确认架构：

```bash
uname -m
```

支持两类：

- `x86_64`: 使用 `onnxruntime-linux-x64-1.19.2.tgz`
- `aarch64`: 使用 `onnxruntime-linux-aarch64-1.19.2.tgz`

注意：Windows `.venv` 里的 `onnxruntime.dll` 不能作为机器狗 Ubuntu 部署运行时。

## 2. 打包

在 Ubuntu 22.04 目标机或同架构 Linux 机器上执行：

```bash
sudo apt-get update
sudo apt-get install -y build-essential curl tar ca-certificates
bash scripts/package_voice_infer_linux.sh
```

如果希望把模型一起复制进部署目录：

```bash
WITH_MODELS=1 bash scripts/package_voice_infer_linux.sh
```

脚本会完成：

- 自动识别 `x86_64` 或 `aarch64`
- 下载 ONNX Runtime `1.19.2`
- 构建 `voice-infer --release`
- 生成 `dist/voice-infer-ubuntu2204-<arch>/`
- 写入 `start.sh` 和 `verify.sh`

## 3. 部署目录

生成目录结构：

```text
dist/voice-infer-ubuntu2204-<arch>/
├── voice-infer
├── libonnxruntime.so.1.19.2
├── libonnxruntime.so -> libonnxruntime.so.1.19.2
├── start.sh
├── verify.sh
└── models/
    ├── asr/
    └── nlu/
```

如果打包时没有使用 `WITH_MODELS=1`，需要手动把 `models/asr` 和 `models/nlu` 放进部署目录。

## 4. 启动

```bash
cd dist/voice-infer-ubuntu2204-<arch>
./start.sh
```

默认端口：

- ASR: `8000`
- NLU: `8001`

可用环境变量覆盖：

```bash
VOICE_INFER_ASR_PORT=9000 VOICE_INFER_NLU_PORT=9001 ./start.sh
```

如果首次加载模型耗时过长，可以先关闭 ORT 图优化排查：

```bash
VOICE_INFER_ORT_OPT=disable ./start.sh
```

可选值：`disable`、`level1`、`level2`、`level3`、`all`。

专项测试如果不在仓库根目录运行，可以显式指定项目根目录：

```bash
VOICE_CONTROL_ROOT=/path/to/VoiceControl \
ORT_DYLIB_PATH=/path/to/libonnxruntime.so.1.19.2 \
cargo test test_encoder_hidden_head_matches_reference -- --ignored --nocapture
```

## 5. 验证

另开一个 shell：

```bash
cd dist/voice-infer-ubuntu2204-<arch>
./verify.sh
```

`verify.sh` 会检查：

- `GET /health` for ASR
- `GET /health` for NLU
- `POST /nlu` 基础推理

ASR 音频端到端验证可以继续用：

```bash
curl -fsS -X POST "http://127.0.0.1:8000/asr" \
  -F "audio=@tests/fixtures/test_1s.wav" \
  -F "language=auto" \
  -F "use_itn=true"
```

## 6. 机器狗联调

机器狗端 pipeline 只需要把 ASR/NLU 服务地址指向 Rust 服务：

- ASR endpoint: `http://<rust-host>:8000/asr`
- NLU endpoint: `http://<rust-host>:8001/nlu`

建议联调顺序：

1. 先在目标机本地跑 `./verify.sh`
2. 再从机器狗端 `curl http://<rust-host>:8000/health`
3. 最后启动 pipeline-only 模式接入 Rust 服务

## 7. 当前限制

当前 Windows 开发机不能代表最终运行环境。端到端推理验收应以 Ubuntu 22.04 目标机上的 `libonnxruntime.so.1.19.2` 为准。
