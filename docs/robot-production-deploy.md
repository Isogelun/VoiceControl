# 真机部署指南

本文档面向 Ubuntu 真机部署，包括机器狗本机、同网段上位机、工控机或 ARM Ubuntu 设备。目标是把 VoiceControl 作为可长期运行的服务部署起来。

## 1. 推荐部署形态

部署形态取决于目标硬件：

### 普通 aarch64 / x86_64 (ORT 正常工作)

```text
Rust voice-infer (单进程)
  ├── ASR HTTP: http://127.0.0.1:8000/asr
  └── NLU HTTP: http://127.0.0.1:8001/nlu

Python pipeline (仅编排，不做推理)
  ├── 麦克风 / Go2 WebRTC / 串口音频
  ├── VAD
  ├── 调用 ASR/NLU HTTP
  └── 动作分发 / 语音反馈
```

配置 `backend: rust`。Python 只需轻量包 (`requirements-robot.txt`)。

### Unitree Go2 / NVIDIA Jetson 平台

Jetson (Orin NX 等) 的 ARM CPU 不被 PyPI 预编译的 ONNX Runtime >= 1.19 识别，标准 Rust ORT 也会崩溃 (cpuid 断言)。已验证方案：

```text
Python 全推理 (backend: python)
  ├── ASR HTTP: onnxruntime 1.18.0 + INT4 模型
  ├── NLU HTTP: onnxruntime 1.18.0 + float32 模型
  └── pipeline + 音频 + 动作分发
```

配置 `backend: python`。需要安装 `requirements-robot.txt` + `requirements-server-py38.txt`。

必须使用 **系统 Python 3.8**（非 uv 安装的 Python 3.10），创建 venv 时加 `--system-site-packages`。关键版本：

| 包 | 版本 | 说明 |
|---|---|---|
| onnxruntime | **1.18.0** | 支持 INT4 + IR10，Jetson 兼容 |
| tokenizers | **0.20.0** | 当前 ASR tokenizer.json 格式需要 |
| transformers | **4.46.3** | 兼容 tokenizers 0.20 |

详见第 17 节。

## 2. 部署拓扑

### 方案 A：全部跑在机器狗/真机上

```text
Ubuntu 22.04 target
├── rust/voice-infer
├── Python pipeline
├── models/
└── audio/
```

优点：
- 网络链路最短。
- ASR/NLU 调用走本机 `127.0.0.1`。
- 部署包整体拷贝即可。

缺点：
- 真机需要有足够 CPU、内存和磁盘。
- 模型较大，首次加载时间较长。

### 方案 B：Rust 推理跑上位机，pipeline 跑机器狗端

```text
Ubuntu/Windows/Linux inference host
└── Rust voice-infer: 8000/8001

Robot-side Ubuntu
└── Python pipeline -> http://<inference-host>:8000/8001
```

优点：
- 机器狗端负载低。
- 推理服务可用更强硬件。

缺点：
- 依赖局域网稳定。
- 需要正确配置 `ASR_URL`、`NLU_URL`。

### 方案 C：systemd 单独托管 Rust，手动/服务化托管 pipeline

适合长期运行。Rust 服务和 pipeline 分别由 systemd 管理，故障时可自动重启。

## 3. 真机硬件和系统要求

系统：
- Ubuntu 22.04
- `aarch64` 或 `x86_64`

确认架构：
```bash
uname -m
```

推荐资源：
- 内存：至少 4 GB，建议 8 GB 以上。
- 磁盘：至少预留 5 GB，模型和 ONNX Runtime 占用较大。
- 网络：如果使用 Go2 WebRTC，需要与机器狗在可连通网络内。

基础依赖：
```bash
sudo apt-get update
sudo apt-get install -y python3 python3-venv python3-pip curl tar ca-certificates
```

可选排查工具：

```bash
sudo apt-get install -y net-tools lsof jq
```

## 4. 在 Windows 上生成真机部署包

Windows 开发机需要：

- Rust
- Zig
- `cargo-zigbuild`

安装：
```powershell
winget install -e --id Rustlang.Rustup
winget install -e --id zig.zig
cargo install cargo-zigbuild
```

验证：
```powershell
cargo --version
zig version
cargo zigbuild --help
```

生成默认 `aarch64` 部署包（纯 Rust 推理）：

```powershell
scripts\build_robot_deploy_bundle.bat
```

输出示例：
```text
dist/robot-deploy-aarch64-20260611-163000/
```

如果真机是 `x86_64`：
```powershell
scripts\build_robot_deploy_bundle.bat -Arch x86_64
```

如果不希望把模型放进部署包：

```powershell
scripts\build_robot_deploy_bundle.bat -NoModels
```

如果需要 `mixed` 后端（Python ASR + Rust NLU）：

```powershell
scripts\build_robot_deploy_bundle.bat -Backend mixed
```

生成的部署包包含：
```text
robot-deploy-<arch>-<timestamp>/
├── asr/
├── nlu/
├── pipeline/
├── unitree_webrtc_connect/
├── scripts/
├── docs/
├── audio/
├── models/
├── rust/
│   ├── voice-infer
│   ├── libonnxruntime.so.1.23.0
│   ├── start.sh
│   └── verify.sh
├── run.py
├── config.yaml
├── config.deploy.yaml
├── requirements-robot.txt
├── DEPLOY.md
└── start_python_managed_rust.sh
```

## 5. 拷贝部署包到真机

使用 `scp`：
```bash
scp -r dist/robot-deploy-aarch64-<timestamp> user@<target-ip>:~/VoiceControlDeploy
```

或者使用 U 盘、rsync、SFTP。

使用 `rsync`：
```bash
rsync -av --progress dist/robot-deploy-aarch64-<timestamp>/ user@<target-ip>:~/VoiceControlDeploy/
```

进入真机：
```bash
ssh user@<target-ip>
cd ~/VoiceControlDeploy
```

## 6. 模型放置

如果打包时没有使用 `-NoModels`，模型应该已经在：

```text
models/asr
models/nlu
models/kws
```

部署包只保留一份模型：根目录 `models/`。不要再复制一份到 `rust/models/`。

如果打包时用了 `-NoModels`，需要手动放置：

```bash
mkdir -p models
scp -r models/asr user@<target-ip>:~/VoiceControlDeploy/models/asr
scp -r models/nlu user@<target-ip>:~/VoiceControlDeploy/models/nlu
```

Rust 包默认读取：

```text
rust/models/asr
rust/models/nlu
```

真机一键安装脚本会自动创建软链接。如果手动处理，可以这样做：

```bash
cd ~/VoiceControlDeploy
mkdir -p rust/models
ln -s ../../models/asr rust/models/asr
ln -s ../../models/nlu rust/models/nlu
```

注意：`ln -s` 的相对路径以 `rust/models/` 为基准。

## 7. 安装 Python 运行环境

在真机部署包根目录：

推荐直接使用真机一键安装脚本：

```bash
cd ~/VoiceControlDeploy
chmod +x scripts/install_robot_target.sh
scripts/install_robot_target.sh --mode onboard
```

该脚本会完成：
- 检查 `rust/voice-infer` 和 ONNX Runtime。
- 检查或链接 `models/asr`、`models/nlu`。
- 创建 `.venv`。
- 安装 `requirements-robot.txt`。
- 生成 `start_robot.sh`。
- 使用 `config.deploy.yaml` 让 Python 自动启动 Rust 推理服务。

如果要安装到 `/opt/voice-control` 并启用 systemd：
```bash
cd ~/VoiceControlDeploy
chmod +x scripts/install_robot_target.sh
sudo -E scripts/install_robot_target.sh --install-dir /opt/voice-control --mode onboard --systemd
```

如果不想用一键脚本，可以手动执行：
```bash
cd ~/VoiceControlDeploy
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-robot.txt
```

如果 pip 下载慢，可以使用镜像源：

```bash
pip install -r requirements-robot.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

纯 Rust 推理模式下，不需要安装 `requirements-server-py38.txt`（里面是 Python 推理引擎的依赖）。

## 8. 配置真机参数

建议复制配置：
```bash
cp config.deploy.yaml config.local.yaml
```

编辑：
```bash
nano config.local.yaml
```

重点字段：
```yaml
inference:
  backend: rust

robot:
  ip: <robot-ip>
  connection_method: LocalSTA

command:
  service_url: http://127.0.0.1:8090/api/v1/local/motion
```

如果 Rust 服务单独由 systemd 管理，则用 `external`：
```yaml
inference:
  backend: external

services:
  asr_url: http://127.0.0.1:8000/asr
  nlu_url: http://127.0.0.1:8001/nlu
```

## 9. 手动启动验证

如果已经运行过 `scripts/install_robot_target.sh`，可以直接启动：

```bash
cd ~/VoiceControlDeploy
./start_robot.sh
```

`start_robot.sh` 会执行：

```bash
python3 run.py --config config.deploy.yaml --<mode>
```

Python 会自动启动 Rust 推理服务并等待健康检查。

### 9.1 启动 Rust 推理服务

```bash
cd ~/VoiceControlDeploy/rust
chmod +x voice-infer start.sh verify.sh
./start.sh
```

另开终端：
```bash
cd ~/VoiceControlDeploy/rust
./verify.sh
```

手动检查：

```bash
curl -fsS http://127.0.0.1:8000/health
curl -fsS http://127.0.0.1:8001/health
```

NLU 测试：
```bash
curl -fsS -X POST http://127.0.0.1:8001/nlu \
  -H 'Content-Type: application/json' \
  -d '{"text":"向前走三步"}'
```

ASR 测试需要准备 wav：
```bash
curl -fsS -X POST http://127.0.0.1:8000/asr \
  -F "audio=@tests/fixtures/test_1s.wav" \
  -F "language=auto" \
  -F "use_itn=true"
```

### 9.2 启动 Python pipeline

```bash
cd ~/VoiceControlDeploy
source .venv/bin/activate
export ASR_URL=http://127.0.0.1:8000/asr
export NLU_URL=http://127.0.0.1:8001/nlu
python3 run.py --config config.local.yaml --pipeline-only --onboard
```

WebRTC：

```bash
python3 run.py --config config.local.yaml --pipeline-only --webrtc
```

串口麦克风阵列：

```bash
python3 run.py --config config.local.yaml --pipeline-only --hardware-serial
```

### 9.3 一条命令启动 Python + Rust + pipeline

部署包提供：

```bash
cd ~/VoiceControlDeploy
chmod +x start_python_managed_rust.sh
./start_python_managed_rust.sh --onboard
```

这个方式会执行 `python3 run.py --config config.deploy.yaml --onboard`。Python 进程会先启动 Rust `voice-infer`，等待 ASR/NLU health check 通过，然后再进入 pipeline。

兼容旧名字：

```bash
./start_rust_and_pipeline.sh --onboard
```

长期生产环境仍建议使用 systemd 分别托管 Rust 和 pipeline，排错和自动重启更清晰。

## 10. systemd 部署

长期运行建议使用 systemd。

假设部署路径：
```text
/opt/voice-control
```

移动目录：
```bash
sudo mkdir -p /opt/voice-control
sudo rsync -av ~/VoiceControlDeploy/ /opt/voice-control/
sudo chown -R $USER:$USER /opt/voice-control
```

### 10.1 Rust 服务

创建：
```bash
sudo nano /etc/systemd/system/voice-infer.service
```

内容：
```ini
[Unit]
Description=VoiceControl Rust ASR/NLU inference service
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=/opt/voice-control/rust
Environment=VOICE_INFER_HOST=0.0.0.0
Environment=VOICE_INFER_ASR_PORT=8000
Environment=VOICE_INFER_NLU_PORT=8001
Environment=VOICE_INFER_ORT_OPT=disable
ExecStart=/opt/voice-control/rust/start.sh
Restart=always
RestartSec=3
KillSignal=SIGTERM

[Install]
WantedBy=multi-user.target
```

启用：
```bash
sudo systemctl daemon-reload
sudo systemctl enable voice-infer
sudo systemctl start voice-infer
```

查看状态：

```bash
systemctl status voice-infer --no-pager
journalctl -u voice-infer -f
```

### 10.2 Pipeline 服务

创建：
```bash
sudo nano /etc/systemd/system/voice-pipeline.service
```

内容（本机麦克风模式）：
```ini
[Unit]
Description=VoiceControl Python pipeline
After=network-online.target voice-infer.service
Wants=network-online.target
Requires=voice-infer.service

[Service]
Type=simple
WorkingDirectory=/opt/voice-control
Environment=ASR_URL=http://127.0.0.1:8000/asr
Environment=NLU_URL=http://127.0.0.1:8001/nlu
ExecStart=/opt/voice-control/.venv/bin/python /opt/voice-control/run.py --config /opt/voice-control/config.local.yaml --pipeline-only --onboard
Restart=always
RestartSec=3
KillSignal=SIGTERM

[Install]
WantedBy=multi-user.target
```

如果使用 WebRTC，把最后参数改为 `--pipeline-only --webrtc`。
如果使用串口麦克风阵列，改为 `--pipeline-only --hardware-serial`。

启用：
```bash
sudo systemctl daemon-reload
sudo systemctl enable voice-pipeline
sudo systemctl start voice-pipeline
```

查看：
```bash
systemctl status voice-pipeline --no-pager
journalctl -u voice-pipeline -f
```

## 11. 日志和排查命令

端口检查：

```bash
ss -lntp | grep -E '8000|8001'
```

进程检查：

```bash
ps aux | grep -E 'voice-infer|run.py'
```

Rust 服务日志：
```bash
journalctl -u voice-infer -n 200 --no-pager
```

Pipeline 日志：
```bash
journalctl -u voice-pipeline -n 200 --no-pager
```

接口检查：

```bash
curl -v http://127.0.0.1:8000/health
curl -v http://127.0.0.1:8001/health
```

网络检查：

```bash
ping <robot-ip>
```

WebRTC 预检：
```bash
cd /opt/voice-control
source .venv/bin/activate
python run.py --config config.local.yaml --preflight-only --webrtc
```

## 12. 升级部署

在 Windows 上重新打包：

```powershell
scripts\build_robot_deploy_bundle.bat
```

上传到真机临时目录：

```bash
rsync -av dist/robot-deploy-aarch64-<timestamp>/ user@<target-ip>:~/VoiceControlDeployNew/
```

停服务：

```bash
sudo systemctl stop voice-pipeline
sudo systemctl stop voice-infer
```

备份旧版本：

```bash
sudo mv /opt/voice-control /opt/voice-control.bak.$(date +%Y%m%d-%H%M%S)
sudo mv ~/VoiceControlDeployNew /opt/voice-control
sudo chown -R $USER:$USER /opt/voice-control
```

重建 Python venv：
```bash
cd /opt/voice-control
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-robot.txt
```

启动：
```bash
sudo systemctl start voice-infer
sudo systemctl start voice-pipeline
```

确认：
```bash
systemctl status voice-infer --no-pager
systemctl status voice-pipeline --no-pager
```

## 13. 回滚

停服务：

```bash
sudo systemctl stop voice-pipeline
sudo systemctl stop voice-infer
```

恢复备份：
```bash
sudo rm -rf /opt/voice-control
sudo mv /opt/voice-control.bak.<timestamp> /opt/voice-control
```

启动：
```bash
sudo systemctl start voice-infer
sudo systemctl start voice-pipeline
```

## 14. 常见故障

### Rust 服务启动后立刻退出

检查：

```bash
journalctl -u voice-infer -n 200 --no-pager
```

常见原因：
- `rust/voice-infer` 没有可执行权限。从 Windows 拷贝到 Linux 后常见，执行 `chmod +x rust/voice-infer`。
- `models/asr` 或 `models/nlu` 不存在。
- `libonnxruntime.so.1.23.0` 不存在。
- `ORT_DYLIB_PATH` 错误。
- 架构不匹配，例如拿了 x86_64 包在 aarch64 真机运行。

检查架构：

```bash
uname -m
file /opt/voice-control/rust/voice-infer
```

### health 接口不通

检查端口：

```bash
ss -lntp | grep -E '8000|8001'
```

检查服务状态：

```bash
systemctl status voice-infer --no-pager
```

### pipeline 连不上 ASR/NLU

确认：
```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8001/health
```

确认环境变量或配置：

```yaml
services:
  asr_url: http://127.0.0.1:8000/asr
  nlu_url: http://127.0.0.1:8001/nlu
```

### WebRTC 连不上机器狗

检查：

```yaml
robot:
  ip: <robot-ip>
  connection_method: LocalSTA
```

确认机器狗和部署机在同一网络。

预检：
```bash
python run.py --config config.local.yaml --preflight-only --webrtc
```

### 麦克风没有声音

检查设备：

```bash
arecord -l
```

如果是 USB/串口麦克风阵列，确认权限和端口：

```bash
ls /dev/tty*
lsusb
```

可能需要把用户加入相关组：

```bash
sudo usermod -aG dialout,audio $USER
```

重新登录后生效。

### 模型太大，拷贝慢

建议：
- 首次部署带模型。
- 后续升级代码时使用 `-NoModels`。
- 真机保留 `models/`，只替换代码和 Rust 二进制。

## 15. 最小验收清单

上线前至少确认：

- `uname -m` 与部署包架构一致。
- `models/asr` 和 `models/nlu` 存在。
- `rust/verify.sh` 通过。
- `curl http://127.0.0.1:8000/health` 通过。
- `curl http://127.0.0.1:8001/health` 通过。
- `python run.py --config config.local.yaml --preflight-only --webrtc` 通过。
- `voice-infer.service` 能自动重启。
- `voice-pipeline.service` 能自动重启。
- 断电重启后两个服务自动启动。

## 16. 建议的生产目录

```text
/opt/voice-control/
├── rust/
├── models/
├── audio/
├── pipeline/
├── unitree_webrtc_connect/
├── run.py
├── config.yaml
├── config.local.yaml
├── requirements-robot.txt
└── .venv/
```

`config.local.yaml` 用于真机私有配置，不建议提交 Git。

## 17. Unitree Go2 / NVIDIA Jetson 部署说明

### 问题

Unitree Go2 EDU 使用 NVIDIA Jetson Orin NX (L4T R35.3.1, Ubuntu 20.04, CUDA 11.4)。

PyPI 预编译的 ONNX Runtime >= 1.14 (aarch64 wheel) 使用 gcc-toolset-14 构建，包含严格 STL 断言。Jetson 的 ARM CPU vendor 不在 ORT 的 cpuid 查找表中，导致 ORT 初始化时崩溃：

```text
onnxruntime cpuid_info warning: Unknown CPU vendor. cpuinfo_vendor value: 0
std::vector ... Assertion '__n < this->size()' failed.
```

此崩溃影响 **Rust ORT** 和 **Python ORT (>=1.14, Python 3.10 wheel)**。`--threads 1` 和 `mixed` 模式都无法避免。

### 解决方案

使用 **系统 Python 3.8** + **onnxruntime 1.18.0**（该版本的 Python 3.8 aarch64 wheel 使用旧编译器构建，无 cpuid 断言）。

配置 `backend: python`，全部推理由 Python 完成，不使用 Rust 二进制。

### Jetson 部署步骤

```bash
cd ~/VoiceControlDeploy

# 1. 安装 python3.8-venv（如果没有）
sudo apt-get install -y python3.8-venv

# 2. 用系统 Python 3.8 创建 venv（继承系统包）
rm -rf .venv
python3.8 -m venv .venv --system-site-packages
source .venv/bin/activate

# 3. 安装 pipeline 依赖
pip install --upgrade pip
pip install -r requirements-robot.txt

# 4. 安装推理依赖（固定 Jetson 兼容版本）
pip install onnxruntime==1.18.0
pip install tokenizers==0.20.0
pip install "transformers>=4.41.0,<5.0"
pip install -r requirements-server-py38.txt

# 5. 修改配置为 python 后端
sed -i 's/backend: rust/backend: python/' config.deploy.yaml
sed -i 's/backend: mixed/backend: python/' config.deploy.yaml

# 6. 启动
export NLU_STRUCTURED_EARLY_STOP=1
python3 run.py --config config.deploy.yaml --onboard --service-timeout 600
```

### Jetson 版本矩阵（已验证）

| 包 | 版本 | 说明 |
|---|---|---|
| Python | **3.8.10** (系统) | 不能用 uv 的 Python 3.10，其 ORT wheel 会崩 |
| onnxruntime | **1.18.0** | 支持 INT4 + ONNX IR 10，Jetson 兼容 |
| tokenizers | **0.20.0** | 当前 ASR tokenizer.json 格式需要 >=0.20 |
| transformers | **4.46.3** | 兼容 tokenizers 0.20 |

**不能使用的版本：**
- onnxruntime >= 1.19 (PyPI aarch64 wheel 含 cpuid 断言)
- onnxruntime 1.17.0 (不支持 ONNX IR version 10，ASR 模型加载失败)
- tokenizers < 0.20 (无法解析当前 ASR tokenizer.json)
- transformers < 4.41 (运行时拒绝 tokenizers >= 0.20)

### systemd (Jetson)

Jetson 上使用单个 systemd 服务即可，因为 `backend: python` 模式下 `run.py` 同时管理 ASR 和 NLU 子进程：

```bash
sudo tee /etc/systemd/system/voice-control.service > /dev/null <<EOF
[Unit]
Description=VoiceControl Python inference + pipeline
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=unitree
WorkingDirectory=/opt/voice-control
Environment=NLU_STRUCTURED_EARLY_STOP=1
ExecStart=/opt/voice-control/.venv/bin/python /opt/voice-control/run.py --config /opt/voice-control/config.deploy.yaml --onboard --service-timeout 600
Restart=always
RestartSec=3
KillSignal=SIGTERM

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable voice-control
sudo systemctl start voice-control
```
