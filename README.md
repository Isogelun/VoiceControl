# VoiceControl

宇树 Go2 EDU 机器狗语音控制系统。

```
唤醒词检测 → VAD 语音分段 → ASR 语音识别 → NLU 意图理解 → 指令 JSON → 音频反馈
```

## 项目结构

```
VoiceControl/
├── run.py                 # 统一入口
├── config.yaml            # 默认运行配置
├── pyproject.toml         # 统一依赖
├── asr/                   # ASR 模块（SenseVoiceSmall ONNX）
│   ├── engine.py          #   推理引擎
│   └── server.py          #   HTTP 服务
├── nlu/                   # NLU 模块（Mengzi-T5 ONNX）
│   ├── engine.py          #   推理引擎
│   ├── server.py          #   HTTP 服务
│   └── tokenizer/         #   分词器配置
├── pipeline/              # 语音处理管道
│   ├── main.py            #   主程序（WebRTC 模式）
│   ├── onboard.py         #   本机麦克风模式
│   ├── asr_client.py      #   ASR HTTP 客户端
│   ├── nlu_client.py      #   NLU HTTP 客户端
│   ├── speaker.py         #   音频播放
│   └── cleaner.py         #   定时清理
├── models/                # 模型文件（gitignored）
│   ├── asr/               #   model_q8.onnx + tokens.txt
│   ├── nlu/               #   encoder.onnx + decoder.onnx
│   └── kws/               #   sherpa-onnx 唤醒词模型
├── audio/                 # 音频资源
└── scripts/
    └── deploy.sh          # 一键部署
```

## 快速开始

### 1. 部署

```bash
bash scripts/deploy.sh
```

### 2. 放置模型文件

```
models/asr/   → model_q8.onnx, tokens.txt
models/nlu/   → encoder.onnx, decoder.onnx
models/kws/   → sherpa-onnx 唤醒词模型文件
```

### 3. 启动

```bash
# 一键启动全部（ASR + NLU + 本机麦克风）
python run.py

# WebRTC/Go2 音频模式
python run.py --webrtc

# 硬件串口唤醒/音频模式
python run.py --hardware-serial

# 本机麦克风 + 开启轻量降噪
python run.py --denoise

# 静音裁切更慢一点，避免一句话被提前截断
python run.py --vad-mode silence --vad-silence-timeout-ms 1800

# GPU 推理
python run.py --gpu
```

### 配置文件

默认读取项目根目录的 `config.yaml`。常用麦克风、VAD、唤醒词、反馈音频都可以直接在里面改：

```yaml
wake:
  text:
    - 你好花花
    - 你好，花花
  audio: audio/xuanxinghuida.mp3

microphone:
  denoise: false
  channel: 0

audio:
  source: onboard

hardware_serial:
  port: COM25
  baudrate: 115200
  audio_channel: 0
  auto_start_audio: true
  set_wake_keyword: false
  wake_keyword: 你好花花
  wake_threshold: "700"

vad:
  mode: silence
  silence_timeout_ms: 1800
```

也可以指定其他配置文件：

```bash
python run.py --onboard --config config.local.yaml
```

命令行参数优先级高于配置文件，适合临时调试：

```bash
python run.py --denoise --vad-silence-timeout-ms 2200
```

### 单独启动服务

```bash
python run.py --serve-asr              # 仅 ASR 服务 (:8000)
python run.py --serve-nlu              # 仅 NLU 服务 (:8001)
python run.py --pipeline-only          # 仅 Pipeline（服务已在其他地方运行）
```

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `UNITREE_ROBOT_IP` | `192.168.8.181` | 机器人 IP |
| `KWS_MODEL_DIR` | `models/kws` | 唤醒词模型目录 |
| `WAKE_KEYWORD` | `你好花花` | 唤醒词（逗号分隔） |
| `WAKE_BACKEND` | Windows: `asr`, 其他: `kws` | 唤醒后端 |
| `WAKE_TEXT` | `你好花花,你好，花花,花花` | ASR 唤醒匹配文本 |
| `WAKE_AUDIO` | `audio/xuanxinghuida.mp3` | 唤醒确认音频 |
| `ASR_URL` | `http://localhost:8000/asr` | ASR 服务地址 |
| `NLU_URL` | `http://localhost:8001/nlu` | NLU 服务地址 |
| `COMMAND_OUTPUT_DIR` | `output` | 指令 JSON 输出目录 |
| `COMMAND_SERVICE_URL` | 空 | 后续动作/机器人服务地址，配置后会 POST 指令 JSON |
| `COMMAND_SUCCESS_AUDIO` | `audio/xuanxinghuida.mp3` | 指令接收/完成反馈音频 |
| `COMMAND_FAILED_AUDIO` | `audio/command_failed.wav` | 无法解析/无法完成反馈音频 |
| `AUDIO_DENOISE` | `0` | 本机麦克风轻量降噪开关，默认关闭；`1` 开启 |
| `VAD_MODE` | `silence` | 裁切模式：`silence` 按静音能量裁切，`webrtc` 使用 WebRTC VAD |
| `VAD_SILENCE_RMS` | `300` | `silence` 模式下静音/语音 RMS 阈值下限 |
| `VAD_SILENCE_MULTIPLIER` | `2.5` | `silence` 模式下噪声底倍数 |
| `VAD_AGGRESSIVENESS` | `2` | `webrtc` 模式 VAD 强度，0-3 |
| `NOISE_CALIBRATION_SECONDS` | `1.0` | 启动后噪声底估计时长 |
| `NOISE_GATE_MULTIPLIER` | `2.5` | 噪声门阈值倍数，越大越抗噪但越不灵敏 |
| `NOISE_GATE_MIN_RMS` | `120` | 噪声门最低 RMS 阈值 |
| `NOISE_GATE_ATTENUATION` | `0.15` | 低于噪声门时的衰减比例 |
| `MIC_GAIN` | `1.0` | 语音通过噪声门后的增益 |
| `VAD_SILENCE_TIMEOUT_MS` | `1200` | 语音结束前需要等待的静音时长 |
| `VAD_MIN_SPEECH_MS` | `240` | 最短有效语音时长 |
| `COMMAND_LISTEN_TIMEOUT_MS` | `8000` | 唤醒后等待命令的最长时长 |
| `UTTERANCE_PAD_MS` | `240` | 送入 ASR 前后补静音，避免裁掉头尾 |

## ASR 同音纠错

`pipeline/text_normalizer.py` 提供三层处理：

- 唤醒词宽松匹配：文本匹配 + 简易拼音匹配，处理“墨/莫/默”等同音字。
- ASR 文本归一化：在进入 NLU 前修正“网前走”“左传”等常见误识别。
- 命令规则兜底：高频动作命令先匹配有限规则，未命中再调用 NLU。

## 性能

| 音频长度 | CPU 推理 | 说明 |
|----------|----------|------|
| ~1s | ~90ms | 实时率 11x |
| ~2s | ~130ms | 实时率 15x |

- 模型：Q8 INT8 量化，234MB
- 引擎：onnxruntime 1.14.0
- Python：3.8 ~ 3.10
