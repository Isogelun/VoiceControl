# VoiceControl

宇树 Go2 EDU 机器狗语音控制系统。

```
唤醒词检测 → VAD 语音分段 → ASR 语音识别 → NLU 意图理解 → 指令 JSON → 音频反馈
```

## 项目结构

```
VoiceControl/
├── run.py                       # 统一入口
├── config.yaml                  # 默认运行配置
├── pyproject.toml               # 统一依赖 (Python ≥3.10)
├── requirements-robot.txt       # 机器狗端依赖 (Python 3.8)
├── requirements-server-py38.txt # 上位机依赖降级版 (Python 3.8)
├── asr/                         # ASR 模块（Qwen3-ASR ONNX）
│   ├── engine.py                #   推理引擎
│   └── server.py                #   HTTP 服务
├── nlu/                         # NLU 模块（Mengzi-T5 ONNX）
│   ├── engine.py                #   推理引擎
│   ├── server.py                #   HTTP 服务
│   └── tokenizer/               #   分词器配置
├── pipeline/                    # 语音处理管道
│   ├── main.py                  #   主程序（状态机 + 三种音频源）
│   ├── command_dispatcher.py    #   指令分发与动作执行
│   ├── text_normalizer.py       #   同音纠错 + 规则兜底
│   ├── onboard.py               #   本机麦克风模式
│   ├── hardware_serial.py       #   6 通道串口麦克风阵列
│   ├── audio_preprocessor.py    #   DC 去除 / 噪声门 / 高通滤波
│   ├── respeaker.py             #   ReSpeaker DoA 读取
│   ├── asr_client.py            #   ASR HTTP 客户端
│   ├── nlu_client.py            #   NLU HTTP 客户端
│   └── speaker.py               #   Go2 扬声器播放
├── voice-infer/                 # Rust 推理服务 (替代 Python ASR+NLU)
│   ├── Cargo.toml
│   └── src/                     #   详见 voice-infer/README.md
├── models/                      # 模型文件（gitignored）
│   ├── asr/                     #   encoder.int4.onnx + decoder + tokenizer
│   ├── nlu/                     #   encoder.onnx + decoder.onnx + tokenizer/
│   └── kws/                     #   sherpa-onnx 唤醒词模型
├── audio/                       # 音频反馈资源
├── scripts/
│   ├── deploy.sh                #   一键部署
│   ├── export_reference.py      #   导出 Rust 重写参考数据
│   └── compare_outputs.py       #   Python/Rust 输出对比
├── docs/
│   ├── rust-rewrite-plan.md     #   Rust 重写方案
│   └── rust-rewrite-checklist.md #  实施清单 (83 checkboxes)
└── tests/
    └── fixtures/                #   测试音频 + 参考数据
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
| `ASR_RETRIES` | `1` | ASR 调用失败时的额外重试次数（总尝试 = 该值 + 1） |
| `ASR_RETRY_DELAY_MS` | `200` | ASR 重试之间的等待时长 |
| `NLU_URL` | `http://localhost:8001/nlu` | NLU 服务地址 |
| `NLU_TIMEOUT` | `10` | NLU 调用超时（秒） |
| `NLU_RETRIES` | `1` | NLU 调用失败时的额外重试次数（总尝试 = 该值 + 1） |
| `NLU_RETRY_DELAY_MS` | `150` | NLU 重试之间的等待时长 |
| `COMMAND_RULES_ENABLED` | `1` | NLU 不可用或返回 unknown 时，是否用规则库兜底高频命令 |
| `COMMAND_RULES_FAST_PATH` | `1` | 高频命令先走规则快路径，命中则跳过 NLU |
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
- 命令规则兜底：高频动作命令先匹配有限规则（快路径），未命中再调用 NLU；当 NLU 服务不可用或返回 unknown 时，同样回落到规则库，避免整句指令被直接丢弃。

## 部署方案

### 方案 A: Python 全栈 (推荐快速上手)

上位机和机器狗都用 Python，适合 Python 3.10 环境：

```bash
pip install -e .
python run.py
```

### 方案 B: Python 3.8 降级部署

部署环境只有 Python 3.8 时，上位机依赖可降级运行，代码无需修改：

```bash
# 上位机 (ASR + NLU 推理服务)
pip install -r requirements-server-py38.txt
python -m asr.server --serve --port 8000 &
python -m nlu.server --serve --port 8001 &

# 机器狗端 (pipeline-only 模式)
pip install -r requirements-robot.txt
python run.py --pipeline-only --onboard
```

关键版本约束：onnxruntime≤1.19.2、transformers≤4.38.2、librosa<0.11、numpy<2.0。

### 方案 C: Rust 推理服务 (零 Python 部署)

用 Rust 单二进制替代 Python ASR+NLU 服务，部署只需一个可执行文件 + 模型：

```bash
# 构建
cd voice-infer && cargo build --release

# 运行 (需要 libonnxruntime.so)
export ORT_DYLIB_PATH=/path/to/libonnxruntime.so.1.19.2
./target/release/voice-infer \
    --asr-model-dir ../models/asr \
    --nlu-model-dir ../models/nlu
```

HTTP 接口与 Python 版 100% 兼容，机器狗端代码无需修改。详见 [voice-infer/README.md](voice-infer/README.md) 和 [docs/rust-rewrite-plan.md](docs/rust-rewrite-plan.md)。

## 性能

| 音频长度 | CPU 推理 | 说明 |
|----------|----------|------|
| ~1s | ~90ms | 实时率 11x |
| ~2s | ~130ms | 实时率 15x |

- ASR 模型：Qwen3-ASR INT4 量化，encoder 711MB + decoder 0.6MB
- NLU 模型：Mengzi-T5，encoder 418MB + decoder 621MB
- 引擎：onnxruntime (Python 1.23+ / Rust 1.19.2)
- 运行环境：Python 3.10 / Python 3.8（降级）/ Rust（无 Python）
