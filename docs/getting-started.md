# Getting Started

本文面向第一次拿到代码的人，说明本项目需要准备什么，以及如何在开发机和机器狗上跑起来。

## 1. 基本环境

开发机建议：

- Python 3.10 或 3.11
- Git
- PowerShell 5+ 或 PowerShell 7
- `uv` 可选，不强制

机器狗/Ubuntu 目标机建议：

- Ubuntu 22.04 或厂商系统自带 Ubuntu 环境
- Python 3.8+，优先使用设备上已经验证可用的 Python
- 可访问 ASR/NLU 模型目录

不再需要 Rust、Zig、cargo 或 cargo-zigbuild。

## 2. 拉取代码

```bash
git clone <repo-url>
cd VoiceControl
```

## 3. 准备模型

把模型放到项目根目录：

```text
models/
├── asr/
├── nlu/
└── kws/    # 可选
```

最少需要：

- `models/asr`：ASR ONNX 模型与 tokenizer。
- `models/nlu`：NLU ONNX 模型与 tokenizer。
- `models/kws`：只有使用 KWS 唤醒时才需要。

`models/` 不提交到 Git。

## 4. 安装依赖

开发机：

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .
pip install -r requirements-server-py38.txt
```

如果只在机器狗上跑 pipeline 并连接外部推理服务，可以只装：

```bash
pip install -r requirements-robot.txt
```

## 5. 配置后端

`config.yaml`：

```yaml
inference:
  backend: python
```

支持值：

- `python`：本机启动 Python ASR/NLU 服务。
- `external`：连接外部 ASR/NLU 服务，机器狗只跑控制链路。

如果使用 `external`：

```yaml
inference:
  backend: external

services:
  asr_url: http://<host>:8000/asr
  nlu_url: http://<host>:8001/nlu
```

也可以用环境变量覆盖：

```bash
export ASR_URL=http://<host>:8000/asr
export NLU_URL=http://<host>:8001/nlu
```

## 6. 本地运行

完整启动：

```bash
python run.py --onboard
```

只启动 ASR：

```bash
python run.py --serve-asr
```

只启动 NLU：

```bash
python run.py --serve-nlu
```

只启动 pipeline：

```bash
python run.py --pipeline-only --webrtc
```

## 7. 打包部署

Windows 生成部署包：

```powershell
scripts\build_robot_deploy_bundle.bat
```

不包含模型：

```powershell
scripts\build_robot_deploy_bundle.bat -NoModels
```

生成 `external` 后端部署包：

```powershell
scripts\build_robot_deploy_bundle.bat -Backend external
```

输出目录在：

```text
dist/robot-deploy-<timestamp>/
```

## 8. 机器狗安装

复制部署包到机器狗后：

```bash
cd <deploy-dir>
chmod +x scripts/install_robot_target.sh
scripts/install_robot_target.sh --mode webrtc
./start_robot.sh
```

安装成 systemd：

```bash
sudo -E scripts/install_robot_target.sh --install-dir /opt/voice-control --mode webrtc --systemd
```

查看日志：

```bash
journalctl -u voice-control -f
```

## 9. 验证

代码级验证：

```bash
python -m py_compile run.py
python -m pytest
```

服务健康检查：

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8001/health
```
