# Unitree Go2 机器狗部署与运行指南

本文档面向 Unitree Go2 EDU 机器狗，覆盖从打包、部署、配置到运行的完整流程。

## 1. 硬件环境

| 项目 | 规格 |
|------|------|
| 主板 | NVIDIA Jetson Orin NX |
| 系统 | L4T R35.3.1 / Ubuntu 20.04 |
| 架构 | aarch64 |
| CUDA | 11.4 |
| Python | 3.8.10 (系统自带) |
| 默认用户 | unitree |
| 默认密码 | 123 |
| 网络 IP | 10.10.20.82 (出厂默认) |

Go2 机身预装了 `uv` 包管理器，安装脚本已适配。

## 2. 推理后端选择

Go2 的 Jetson 芯片**不兼容** PyPI 预编译的 ONNX Runtime >= 1.19 (aarch64 wheel)，Rust ORT 同样会崩溃。原因：ORT 使用 gcc-toolset-14 构建，包含 cpuid 断言，Jetson ARM CPU vendor 不在查找表中。

唯一可用方案：**`backend: python`** — 使用系统 Python 3.8 + ORT 1.18.0 做全 Python 推理。

```text
Python run.py (单进程管理)
  ├── ASR 子进程: onnxruntime 1.18.0 + Qwen3-ASR INT4
  ├── NLU 子进程: onnxruntime 1.18.0 + T5 float32
  ├── VAD + 唤醒词检测
  ├── 麦克风采集
  └── 动作分发 / 语音反馈
```

### 已验证版本矩阵

| 包 | 版本 | 说明 |
|---|---|---|
| Python | 3.8.10 (系统) | 不可用 uv 的 Python 3.10 |
| onnxruntime | 1.18.0 | 支持 INT4 + ONNX IR 10，Jetson 兼容 |
| tokenizers | 0.20.0 | ASR tokenizer.json 格式需要 >= 0.20 |
| transformers | 4.46.3 | 兼容 tokenizers 0.20 |

### 不可用版本

- onnxruntime >= 1.19: cpuid 断言崩溃
- onnxruntime 1.17.x: 不支持 ONNX IR version 10，ASR 模型加载失败
- tokenizers < 0.20: 无法解析 ASR tokenizer.json
- transformers < 4.41: 运行时拒绝 tokenizers >= 0.20

## 3. 在 Windows 上打包

```powershell
scripts\build_robot_deploy_bundle.bat -Backend python
```

输出目录：
```text
dist/robot-deploy-aarch64-<时间戳>/
```

不带模型打包（模型已在机器狗上时）：
```powershell
scripts\build_robot_deploy_bundle.bat -Backend python -NoModels
```

## 4. 传输到机器狗

```bash
scp -r dist/robot-deploy-aarch64-<时间戳> unitree@10.10.20.82:~/voice-control-deploy
```

或使用 rsync：
```bash
rsync -av --progress dist/robot-deploy-aarch64-<时间戳>/ unitree@10.10.20.82:~/voice-control-deploy/
```

Windows scp 会丢失 Linux 可执行权限，后续安装脚本会自动修复。

## 5. 安装

SSH 登录机器狗：
```bash
ssh unitree@10.10.20.82
cd ~/voice-control-deploy
```

### 5.1 使用安装脚本

```bash
chmod +x scripts/install_robot_target.sh
PYTHON_BIN=python3.8 scripts/install_robot_target.sh --mode onboard
```

脚本自动完成：
- 检测 uv / pip 并选择对应安装方式
- 创建 `.venv` 虚拟环境
- 安装 `requirements-robot.txt` 和 `requirements-server-py38.txt`
- 创建 `rust/models/` 符号链接
- 生成 `start_robot.sh` 启动脚本
- 生成 `config.deploy.yaml` 配置文件

### 5.2 手动安装（安装脚本失败时）

```bash
# 用系统 Python 3.8 创建 venv
python3.8 -m venv .venv --system-site-packages
source .venv/bin/activate

# 安装依赖
pip install -r requirements-robot.txt
pip install onnxruntime==1.18.0
pip install tokenizers==0.20.0
pip install "transformers>=4.41.0,<5.0"
pip install -r requirements-server-py38.txt
```

## 6. 配置

编辑 `config.deploy.yaml`：

```yaml
server:
  host: 0.0.0.0
  asr_port: 8000
  nlu_port: 8001
  service_timeout: 120
  gpu: false

inference:
  backend: python

wake:
  backend: asr
  text:
    - 你好曼波
    - 曼波
    - 你好，曼波
  aliases:
    - 慢播
    - 快播
    - 那波
    - 南波
    - 慢波
    - 曼播
  audio: audio/mabo.mp3
  feedback_enabled: true

microphone:
  device: 24       # ReSpeaker USB 麦克风设备号，见第 8 节
  channel: 0

command:
  service_url: http://10.10.20.82:8090/api/v1/local/motion
```

### 关键配置说明

| 配置项 | 值 | 说明 |
|--------|---|------|
| `inference.backend` | `python` | Jetson 必须用 Python 推理 |
| `wake.backend` | `asr` | sherpa-onnx KWS 在 Jetson 上崩溃，用 ASR 做唤醒 |
| `microphone.device` | 设备号 | 指定麦克风，留空则用系统默认 |
| `command.service_url` | Go2 运动服务地址 | 通常是 `http://10.10.20.82:8090/api/v1/local/motion` |

## 7. 唤醒词配置

唤醒词完全由配置文件控制，代码中不硬编码。

```yaml
wake:
  backend: asr           # 唤醒引擎：asr / kws / hardware
  text:                   # 唤醒短语列表（ASR 识别结果需要匹配其中之一）
    - 你好曼波
    - 曼波
    - 你好，曼波
  aliases:                # 同音容错（ASR 常见误识别）
    - 慢播
    - 快播
    - 那波
    - 南波
  keyword: 你好曼波        # KWS 模式的拼音关键词（仅 kws 后端使用）
  audio: audio/mabo.mp3   # 唤醒成功后播放的提示音
  feedback_enabled: true  # 是否播放唤醒提示音
```

### 唤醒流程 (backend: asr)

1. VAD 检测到语音活动（音量超过阈值）
2. 将语音片段发送给 ASR 服务识别为文字
3. 对识别文字做归一化 + 拼音比对
4. 如果匹配 `text` 或 `aliases` 中任意一项，进入指令监听模式
5. 监听后续语音指令，识别并执行

### 修改唤醒词

只需修改 `config.deploy.yaml` 中的 `wake.text` 和 `wake.aliases`，重启服务即可：

```yaml
wake:
  backend: asr
  text:
    - 你好笨笨
    - 笨笨
  aliases:
    - 奔奔
    - 本本
    - 你好奔奔
  audio: audio/wake.mp3
```

`aliases` 应包含 ASR 模型对唤醒词的常见误识别变体，提高唤醒成功率。

## 8. 麦克风配置

### 查看可用设备

在机器狗上执行：
```bash
cd ~/voice-control-deploy
source .venv/bin/activate
python3 -c "import sounddevice; print(sounddevice.query_devices())"
```

找到目标设备的编号，例如：
```text
[24] reSpeaker Flex XVF3800 C16K6Ch: USB Audio (hw:2,0), channels=6, rate=16000.0
```

### 设置设备号

```yaml
microphone:
  device: 24        # 对应上面查到的设备编号
  channel: 0        # 使用第 0 通道
```

### 常见设备

| 设备 | 说明 |
|------|------|
| ReSpeaker Flex XVF3800 | USB 6 通道麦克风阵列，device 编号视系统而定 |
| Jetson APE (默认) | 板载音频，通常无实际麦克风输入 |

如果不设置 `device`，系统会使用默认音频设备（通常是 Jetson APE，无法采集到声音）。**必须指定外接麦克风的设备号**。

### ReSpeaker DOA（声源定向）

如果使用 ReSpeaker 麦克风阵列，可启用 DOA：

```yaml
respeaker:
  doa_enabled: true
  vid: 0x2886
  angle_offset: 0
```

## 9. 启动服务

### 手动启动

```bash
cd ~/voice-control-deploy
./start_robot.sh
```

或直接执行：
```bash
cd ~/voice-control-deploy
source .venv/bin/activate
python3 run.py --config config.deploy.yaml --onboard
```

### 启动日志

正常启动会依次看到：
```text
ASR 进程 PID=xxxxx (port 8000)
NLU 进程 PID=xxxxx (port 8001)
NLU model loaded (provider: CPUExecutionProvider)           # ~4s
Qwen3-ASR model loaded (int4, provider: CPUExecutionProvider) # ~6s
ASR health check passed
NLU health check passed
进入本机麦克风模式
使用音频设备: reSpeaker Flex XVF3800 ...
麦克风已打开 ...
麦克风监听，等待唤醒词...
```

### 验证服务状态

另开终端检查：
```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8001/health
```

NLU 推理测试：
```bash
curl -X POST http://127.0.0.1:8001/nlu \
  -H 'Content-Type: application/json' \
  -d '{"text":"向前走三步"}'
```

## 10. systemd 开机自启

### 安装服务

使用安装脚本：
```bash
cd ~/voice-control-deploy
sudo PYTHON_BIN=python3.8 scripts/install_robot_target.sh --mode onboard --systemd --skip-pip
```

或手动创建：
```bash
sudo tee /etc/systemd/system/voice-control.service > /dev/null <<'EOF'
[Unit]
Description=VoiceControl Python inference + pipeline
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=unitree
WorkingDirectory=/home/unitree/voice-control-deploy
ExecStart=/home/unitree/voice-control-deploy/.venv/bin/python /home/unitree/voice-control-deploy/run.py --config /home/unitree/voice-control-deploy/config.deploy.yaml --onboard
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

### 管理服务

```bash
sudo systemctl status voice-control    # 查看状态
sudo systemctl restart voice-control   # 重启
sudo systemctl stop voice-control      # 停止
journalctl -u voice-control -f         # 实时日志
journalctl -u voice-control -n 200     # 最近 200 行日志
```

## 11. VAD 参数调优

VAD（语音活动检测）决定了什么时候开始/结束采集语音：

```yaml
vad:
  mode: silence                    # 基于静音检测
  silence_rms: 180                 # 环境噪声基线 RMS
  silence_multiplier: 2.0          # 语音判定阈值 = silence_rms * multiplier
  silence_timeout_ms: 300          # 唤醒阶段：静音超时（说完唤醒词后的等待）
  command_silence_timeout_ms: 180  # 指令阶段：静音超时（说完指令后的等待）
  min_speech_ms: 240               # 最短语音时长，低于此值丢弃
  command_listen_timeout_ms: 8000  # 指令最长监听时间
```

如果环境嘈杂导致误触发，调高 `silence_rms`。如果说话不灵敏，调低。

## 12. 运动控制配置

```yaml
command:
  service_url: http://10.10.20.82:8090/api/v1/local/motion
  move_native_enabled: true          # 使用原生步进模式
  move_native_default_steps: 3       # 默认步数
  move_native_linear_speed: 1.0      # 行走速度
  move_native_yaw_speed: 1.0         # 转弯速度
  move_fast_response: true           # 先播报再动
  success_audio: audio/曼波wow.mp3
  failed_audio: audio/曼波我嘞个豆.mp3
  unavailable_audio: audio/曼波哈基米.mp3
```

### 支持的语音指令

| 指令 | 动作 |
|------|------|
| 向前走 / 前进 | 前进 N 步 |
| 向后退 / 后退 | 后退 N 步 |
| 向左转 | 左转 |
| 向右转 | 右转 |
| 向左走 | 向左平移 |
| 向右走 | 向右平移 |
| 站起来 | 站立 |
| 坐下 / 蹲下 | 坐下 |
| 趴下 | 趴下 |
| 打个招呼 | 招手 |
| 摇一摇 / 抖一抖 | 抖身体 |
| 伸个懒腰 | 伸懒腰 |
| 停 / 别动 | 停止当前动作 |

## 13. 日志与排查

### 进程检查

```bash
ps aux | grep run.py
ss -lntp | grep -E '8000|8001'
```

### 常见问题

#### ORT cpuid 崩溃
```text
onnxruntime cpuid_info warning: Unknown CPU vendor. cpuinfo_vendor value: 0
std::vector ... Assertion '__n < this->size()' failed.
```
原因：使用了 ORT >= 1.19 或 Python 3.10 的 ORT wheel。
解决：确保用系统 Python 3.8 + onnxruntime==1.18.0。

#### ASR 模型加载失败 "Unsupported model IR version: 10"
原因：ORT 版本太低（如 1.17.x）。
解决：升级到 onnxruntime==1.18.0。

#### KWS 唤醒崩溃 "GetFrames:155 0 + 45 > 2"
原因：sherpa-onnx KWS 在 Jetson 上 frame buffer 断言失败。
解决：设置 `wake.backend: asr`。

#### 麦克风采集全零
```text
麦克风电平: raw_mean=0.0 raw_peak=0
```
原因：使用了 Jetson APE 默认设备，无实际麦克风。
解决：设置 `microphone.device` 为外接麦克风的设备号。

#### 权限不足
```text
Permission denied: /dev/snd/...
```
解决：
```bash
sudo usermod -aG audio unitree
# 重新登录
```

#### 服务启动后立刻退出
检查日志：
```bash
journalctl -u voice-control -n 100 --no-pager
```
常见原因：配置文件路径错误、Python 环境问题、端口被占用。

#### 端口被占用
```bash
ss -lntp | grep 8000
# 杀掉旧进程
pkill -f 'python3 run.py'
```

## 14. 升级部署

在 Windows 重新打包：
```powershell
scripts\build_robot_deploy_bundle.bat -Backend python -NoModels
```

上传并替换（保留配置和模型）：
```bash
rsync -av --progress --exclude '.venv' --exclude 'models' --exclude 'config.deploy.yaml' \
  dist/robot-deploy-aarch64-<时间戳>/ unitree@10.10.20.82:~/voice-control-deploy/
```

在机器狗上重启：
```bash
sudo systemctl restart voice-control
```

## 15. 验收清单

- [ ] `inference.backend` 设为 `python`
- [ ] `wake.backend` 设为 `asr`
- [ ] `wake.text` 已配置唤醒词
- [ ] `microphone.device` 已设为正确设备号
- [ ] `curl http://127.0.0.1:8000/health` 返回成功
- [ ] `curl http://127.0.0.1:8001/health` 返回成功
- [ ] 对麦克风说唤醒词能进入监听模式
- [ ] 语音指令能控制机器狗动作
- [ ] systemd 服务已 enable，断电重启后自动恢复
