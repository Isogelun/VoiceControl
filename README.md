# VoiceControl

VoiceControl 是面向 Unitree Go2 的语音控制项目，流程如下：

```text
唤醒词/语音输入 -> VAD -> ASR -> NLU -> 指令 JSON -> 机器狗动作/语音反馈
```

当前仓库只保留 Python 运行链路。之前的 Rust `voice-infer` 路线已经移除，原因是真机 aarch64 环境上 ONNX Runtime 会在初始化或建 session 阶段崩溃，Rust 与 Python ORT 都受影响。

## 当前后端

`config.yaml` 通过 `inference.backend` 选择推理方式：

```yaml
inference:
  backend: python    # python / external
```

- `python`：`run.py` 在本机启动 Python ASR/NLU 服务，再启动 pipeline。
- `external`：`run.py` 不启动 ASR/NLU，只连接 `services.asr_url` 和 `services.nlu_url`，适合让机器狗只跑控制链路，推理放到上位机。

### ASR 引擎选择

在 `inference.backend=python` 模式下，可通过 `asr.engine` 选择 ASR 推理引擎：

```yaml
asr:
  engine: qwen3               # qwen3 | ncnn
  ncnn_model_dir: models/asr_ncnn  # asr.engine=ncnn 时的模型目录
```

| 引擎 | 说明 | 依赖 |
|------|------|------|
| `qwen3`（默认） | Qwen3-ASR ONNX，精度高，中英混合 | `onnxruntime`（已含在依赖中） |
| `ncnn` | Sherpa-NCNN ConvEmformer Transducer，轻量、int8 量化，中英双语 | 需额外安装：`pip install sherpa-ncnn` |

也可以通过命令行参数切换：

```bash
python run.py --onboard --asr-engine ncnn
```

NCNN 模型文件需下载到 `models/asr_ncnn/`：

```text
models/asr_ncnn/
├── encoder.ncnn.param
├── encoder.ncnn.bin
├── decoder.ncnn.param
├── decoder.ncnn.bin
├── joiner.ncnn.param
├── joiner.ncnn.bin
└── tokens.txt
```

推荐模型：[csukuangfj/sherpa-ncnn-conv-emformer-transducer-2022-12-06](https://huggingface.co/csukuangfj/sherpa-ncnn-conv-emformer-transducer-2022-12-06)（中英双语，含 int8 量化版）。

两个引擎共享同一端口（默认 8000），同一时刻只运行一个，切换只需改配置重启。

## 目录结构

```text
VoiceControl/
├── run.py                         # 统一启动入口
├── config.yaml                    # 默认配置
├── asr/                           # Python ASR 服务
│   ├── engine.py                  # Qwen3-ASR ONNX 引擎（默认）
│   ├── server.py                  # Qwen3-ASR HTTP 服务
│   ├── ncnn_engine.py             # Sherpa-NCNN 引擎（第二方案）
│   └── ncnn_server.py             # Sherpa-NCNN HTTP 服务
├── nlu/                           # Python NLU 服务
├── pipeline/                      # 唤醒、VAD、ASR/NLU 客户端、动作分发
├── unitree_webrtc_connect/        # Unitree Go2 WebRTC 接入
├── scripts/                       # 打包、部署、麦克风/扬声器工具
├── docs/                          # 项目文档
├── tests/                         # Python 测试
├── audio/                         # 反馈音频
└── models/                        # 本地模型目录，不入库
```

## 快速开始

安装 Python 依赖：

```bash
pip install -e .
pip install -r requirements-server-py38.txt
```

启动完整本地链路：

```bash
python run.py --onboard
```

只启动服务：

```bash
python run.py --serve-asr
python run.py --serve-nlu
```

只启动 pipeline，连接外部 ASR/NLU：

```bash
export ASR_URL=http://<host>:8000/asr
export NLU_URL=http://<host>:8001/nlu
python run.py --pipeline-only --webrtc
```

## 模型目录

默认模型路径：

```text
models/
├── asr/
│   ├── encoder.int4.onnx
│   ├── decoder*
│   ├── decoder_weights.int4.data
│   ├── embed_tokens.bin
│   └── tokenizer.json
├── asr_ncnn/                  # Sherpa-NCNN 模型（asr.engine=ncnn 时需要）
│   ├── encoder.ncnn.param
│   ├── encoder.ncnn.bin
│   ├── decoder.ncnn.param
│   ├── decoder.ncnn.bin
│   ├── joiner.ncnn.param
│   ├── joiner.ncnn.bin
│   └── tokens.txt
├── nlu/
│   ├── encoder.onnx
│   ├── decoder.onnx
│   └── tokenizer/
└── kws/
    └── sherpa-onnx wake word models
```

`models/` 体积较大，已被 `.gitignore` 忽略，不要直接提交。

## 真机部署

Windows 上生成部署包：

```powershell
scripts\build_robot_deploy_bundle.bat
```

不打包模型：

```powershell
scripts\build_robot_deploy_bundle.bat -NoModels
```

在机器狗或 Ubuntu 目标机上：

```bash
cd <deploy-dir>
chmod +x scripts/install_robot_target.sh
scripts/install_robot_target.sh --mode webrtc
./start_robot.sh
```

完整部署说明见 [docs/robot-production-deploy.md](docs/robot-production-deploy.md)。

## 测试

```bash
python -m pytest
python -m py_compile run.py
```

## 不要提交的内容

- `models/`
- `.venv/`
- `venv/`
- `third_party/`
- `dist/`
- `output/`
- `__pycache__/`
