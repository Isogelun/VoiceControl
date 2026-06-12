# Robot Production Deploy

本文说明如何把 VoiceControl 部署到机器狗或 Ubuntu 目标机。

## 1. 推荐部署形态

当前项目保留两种可用形态：

- `python`：目标机本地启动 Python ASR/NLU 服务，再启动 pipeline。
- `external`：目标机只跑 pipeline，ASR/NLU 在上位机或其他服务器运行。

对 Unitree 真机，如果本地 ONNX Runtime 仍出现 `cpuid_info` 或 `std::vector assertion` 崩溃，使用 `external`。这不是业务代码问题，而是目标机 CPU/系统与 ORT 预编译包不兼容。

## 2. Windows 打包

在项目根目录：

```powershell
scripts\build_robot_deploy_bundle.bat
```

不带模型：

```powershell
scripts\build_robot_deploy_bundle.bat -NoModels
```

外部推理模式：

```powershell
scripts\build_robot_deploy_bundle.bat -Backend external
```

输出目录：

```text
dist/robot-deploy-<timestamp>/
```

## 3. 复制到目标机

```bash
scp -r dist/robot-deploy-<timestamp> unitree@10.10.20.82:~/
```

或手动拷贝目录到目标机。

## 4. 模型放置

如果打包时用了 `-NoModels`，在目标机部署目录补齐：

```text
models/
├── asr/
└── nlu/
```

使用 `external` 后端时，目标机可以不放 ASR/NLU 模型，因为推理在外部机器上完成。

## 5. 安装

目标机执行：

```bash
cd ~/robot-deploy-<timestamp>
chmod +x scripts/install_robot_target.sh
scripts/install_robot_target.sh --mode webrtc
```

可选模式：

- `--mode webrtc`：Go2 WebRTC 音频。
- `--mode onboard`：本机麦克风。
- `--mode hardware_serial`：串口麦克风阵列。

安装脚本会：

- 创建 `.venv`。
- 安装 `requirements-robot.txt`。
- 当 `backend: python` 时额外安装 `requirements-server-py38.txt`。
- 生成 `start_robot.sh`。

## 6. 启动

```bash
./start_robot.sh
```

也可以直接运行：

```bash
python3 run.py --config config.deploy.yaml --webrtc
```

## 7. systemd

安装到 `/opt/voice-control` 并启用 systemd：

```bash
sudo -E scripts/install_robot_target.sh --install-dir /opt/voice-control --mode webrtc --systemd
```

查看状态：

```bash
systemctl status voice-control --no-pager
journalctl -u voice-control -f
```

重启：

```bash
sudo systemctl restart voice-control
```

## 8. external 后端

`config.deploy.yaml`：

```yaml
inference:
  backend: external

services:
  asr_url: http://<inference-host>:8000/asr
  nlu_url: http://<inference-host>:8001/nlu
```

或者启动前设置环境变量：

```bash
export ASR_URL=http://<inference-host>:8000/asr
export NLU_URL=http://<inference-host>:8001/nlu
./start_robot.sh
```

目标机启动时会先检查：

```text
http://<inference-host>:8000/health
http://<inference-host>:8001/health
```

## 9. 常见问题

### ASR/NLU 启动后立刻崩溃

如果日志包含：

```text
onnxruntime cpuid_info warning
std::vector ... Assertion '__n < this->size()' failed
```

说明当前目标机无法使用该 ONNX Runtime 预编译包。改用 `external` 后端，把推理放到上位机。

### 端口被占用

```bash
ss -lntp | grep -E ':8000|:8001'
```

停止旧进程后再启动。

### WebRTC 连接失败

检查 `config.deploy.yaml`：

```yaml
robot:
  ip: 10.10.20.82
  connection_method: LocalSTA
```

并确认电脑/目标机和机器狗在同一网络。
