# Rust 重写实施清单

> 相关文档: [重写方案](rust-rewrite-plan.md) | [Rust 项目 README](../voice-infer/README.md) | [主 README](../README.md)
>
> 每个 checkbox 是一个独立可验证的工作单元。完成后打勾，保证不漏步骤。

---

## Phase 0: 准备工作 (0.5 天)

### 0.1 开发环境

- [ ] 安装 Rust toolchain (`rustup`, stable channel)
- [ ] 如果目标平台是 aarch64: `rustup target add aarch64-unknown-linux-gnu` + 安装交叉编译器
- [ ] 下载 onnxruntime 预编译库
  - x86_64: `onnxruntime-linux-x64-1.19.2.tgz` 从 [GitHub Releases](https://github.com/microsoft/onnxruntime/releases/tag/v1.19.2)
  - aarch64: `onnxruntime-linux-aarch64-1.19.2.tgz`
  - 解压得到 `lib/libonnxruntime.so.1.19.2` + `include/`
- [ ] `cargo init voice-infer` 创建项目
- [ ] 写好 Cargo.toml (照方案中的依赖清单)
- [ ] `cargo check` 确保依赖全部拉取成功

### 0.2 导出 Python 参考数据 (用于后续逐步验证)

写一个 Python 脚本 `scripts/export_reference.py`，导出以下 ground truth:

- [ ] **Mel 参考值**: 取一段 1 秒 16kHz 测试音频 → 导出:
  - `ref_mel.npy` — log_mel_fast() 的完整输出 `[1, 128, T]`
  - `ref_hann_window.npy` — Hann 窗 `[400]`
  - `ref_mel_basis.npy` — mel 滤波器组 `[128, 201]`
  - `ref_power_frame0.npy` — 第 0 帧的 power spectrum `[201]` (用于 FFT 验证)
- [ ] **ASR tokenizer 参考值**:
  - `ref_asr_encode.json` — `tokenizer.encode("<|im_start|>system<|im_end|>...").ids`
  - `ref_asr_decode.json` — `tokenizer.decode([token_ids], skip_special_tokens=True)`
- [ ] **NLU tokenizer 参考值**:
  - `ref_nlu_encode.json` — `tokenizer("指令解析: 向前走三步", padding="max_length", max_length=64, truncation=True)` 的 input_ids + attention_mask
  - `ref_nlu_decode.json` — `tokenizer.decode([0, 312, ...], skip_special_tokens=True)`
- [ ] **ASR 端到端参考值**:
  - `ref_asr_audio_features.npy` — encoder 输出 `[1, audio_len, 1024]` (float32)
  - `ref_asr_init_logits_last.npy` — decoder_init 输出 logits 的最后一个位置 `[151936]`
  - `ref_asr_result.json` — 最终 transcribe() 结果 `{"text": "...", ...}`
- [ ] **NLU 端到端参考值**:
  - `ref_nlu_hidden.npy` — encoder 输出 `[1, 64, 768]` 的前 5 个值 (spot check)
  - `ref_nlu_result.json` — predict("向前走三步") 结果
  - `ref_nlu_result2.json` — predict("停止") 结果
  - `ref_nlu_result3.json` — predict("坐下") 结果
- [ ] **NLU 解析参考值**:
  - `ref_parse_cases.json` — 10 组 `(raw_output, expected_parsed)` 覆盖:
    - JSON cmd 格式 (type=cmd, MoveForward)
    - JSON cmd 格式 (type=cmd, Move, 含 vx/vy/vyaw)
    - JSON chat 格式 (type=chat)
    - JSON intent/slots 格式
    - key=value 格式 (intent=stop)
    - 纯文本 fallback (unknown)

### 0.3 创建测试音频

- [ ] 准备 `tests/fixtures/test_1s.wav` — 16kHz mono, 1 秒，包含语音
- [ ] 准备 `tests/fixtures/test_silence.wav` — 16kHz mono, 0.5 秒静音
- [ ] 准备 `tests/fixtures/test_8k_stereo.wav` — 8kHz stereo (测试重采样 + 声道合并)

---

## Phase 1: 音频模块 `src/audio/` (Day 1 上午)

### 1.1 `src/audio/mel.rs` — Mel 频谱前端

#### 1.1.1 Mel 滤波器组生成

- [ ] 实现 `hz_to_mel(f: f32) -> f32` — Slaney 公式: `2595 * log10(1 + f/700)`
- [ ] 实现 `mel_to_hz(m: f32) -> f32` — 逆变换
- [ ] 实现 `create_mel_filterbank(sr, n_fft, n_mels, fmin, fmax) -> Array2<f32>`
  - 在 mel 域等距取 n_mels+2 个中心频率
  - 转回 Hz，映射到 FFT bin index
  - 构造三角滤波器 (左斜 + 右斜)
  - Slaney 归一化: `filter[i] /= (center[i+1] - center[i-1])` (mel 域宽度)
- [ ] **验证**: 与 `ref_mel_basis.npy` 对比，max abs error < 1e-6

#### 1.1.2 Hann 窗

- [ ] 实现 `create_hann_window(n_fft: usize) -> Vec<f32>`
  - 公式: `0.5 - 0.5 * cos(2π * n / N)` (periodic, 即除以 N 而非 N-1)
- [ ] **验证**: 与 `ref_hann_window.npy` 对比，exact match

#### 1.1.3 FFT + Power Spectrum

- [ ] 实现 `power_spectrum(frame: &[f32], fft: &Arc<dyn Fft<f32>>) -> Vec<f32>`
  - 输入: 一帧加窗信号 `[n_fft]`
  - 过程: zero-pad complex → FFT → 取前 n_fft/2+1 → `|c|²`
  - 输出: `[n_fft/2+1]` = `[201]`
- [ ] 创建 `FftPlanner` 并缓存 plan (不要每帧重新 plan)
- [ ] **验证**: 第 0 帧输出与 `ref_power_frame0.npy` 对比，max abs error < 1e-4

#### 1.1.4 MelFrontend 完整实现

- [ ] 定义 `MelConfig` 结构体 (从 ASR config.json 的 `mel` 段解析)
  ```rust
  pub struct MelConfig {
      pub sample_rate: usize,  // 16000
      pub n_fft: usize,        // 400
      pub hop_length: usize,   // 160
      pub n_mels: usize,       // 128
      pub fmin: f32,           // 0.0
      pub fmax: f32,           // 8000.0
  }
  ```
- [ ] 实现 `MelFrontend::new(config)` — 初始化窗、滤波器组、FFT plan
- [ ] 实现 `MelFrontend::log_mel(wav: &[f32]) -> Array3<f32>`
  - zero-pad wav 两端 (各 n_fft/2)
  - 计算帧数: `1 + (padded_len - n_fft) / hop_length`
  - 逐帧: 取窗 → 加窗 → power_spectrum → mel 点积
  - 全局: `log10(max(mel, 1e-10))`
  - 归一化: `max(log_mel, global_max - 8.0)`, 然后 `(log_mel + 4.0) / 4.0`
  - reshape 到 `[1, 128, frames]`
- [ ] 处理空音频边界: wav.len() == 0 时填充 0.1 秒静音
- [ ] **验证**: 完整 mel 输出与 `ref_mel.npy` 对比，max abs error < 1e-5
- [ ] **验证**: 空音频不 panic

### 1.2 `src/audio/wav.rs` — WAV 读取

- [ ] 实现 `load_audio_from_bytes(data: &[u8]) -> Result<Vec<f32>>`
  - `hound::WavReader::new(Cursor::new(data))`
  - 支持 16-bit PCM → f32 (除以 32768.0)
  - 支持 32-bit float 直接读取
  - 多声道 → 取平均
  - 采样率 != 16000 → `rubato::SincFixedIn` 重采样
- [ ] **验证**: 读取 test_1s.wav，输出 len ≈ 16000，值范围 [-1, 1]
- [ ] **验证**: 读取 test_8k_stereo.wav，输出 len ≈ 16000 (重采样后)，mono
- [ ] **验证**: 读取 test_silence.wav，输出全近零

### 1.3 `src/audio/mod.rs`

- [ ] 导出 `MelFrontend`, `MelConfig`, `load_audio_from_bytes`

---

## Phase 2: ASR 引擎 `src/asr/` (Day 1 下午)

### 2.1 `src/config.rs` — 配置加载

- [ ] 定义 `AsrModelConfig` 结构体 (对应 models/asr/config.json)
  ```rust
  pub struct AsrModelConfig {
      pub model_type: String,       // "qwen3_asr"
      pub encoder: EncoderConfig,
      pub decoder: DecoderConfig,
      pub mel: MelConfig,
      pub special_tokens: SpecialTokens,
      pub quantization: Option<QuantConfig>,
      pub embed_tokens_dtype: String,
  }
  pub struct DecoderConfig {
      pub num_layers: usize,        // 28
      pub hidden_size: usize,       // 1024
      pub vocab_size: usize,        // 151936
      // ... 其余字段
  }
  pub struct SpecialTokens {
      pub eos_token_ids: Vec<i64>,  // [151643, 151645]
      pub pad_token_id: i64,
      pub audio_pad_token_id: i64,  // 151676
      // ... 其余字段
  }
  ```
- [ ] 实现 `AsrModelConfig::load(model_dir: &Path) -> Result<Self>`
- [ ] 定义 `NluExportConfig` 结构体 (对应 models/nlu/export_config.json)
  ```rust
  pub struct NluExportConfig {
      pub max_seq_len: usize,               // 64
      pub decoder_start_token_id: i64,      // 0
      pub eos_token_id: i64,                // 1
      pub pad_token_id: i64,                // 0
  }
  ```
- [ ] 实现 `NluExportConfig::load(model_dir: &Path) -> Result<Self>`
- [ ] **验证**: 加载真实 config.json，检查 `model_type == "qwen3_asr"`

### 2.2 `src/asr/engine.rs` — ASR 推理引擎

#### 2.2.1 结构体与初始化

- [ ] 定义 `AsrEngine` 结构体 (3 个 ort::Session + tokenizer + mmap + mel)
- [ ] 实现 `AsrEngine::new(model_dir, num_threads, use_gpu)`
  - [ ] 检测模型后缀 (.int4.onnx / .onnx)
  - [ ] 验证模型目录完整性 (6 个必需文件)
  - [ ] 创建 3 个 ONNX session (encoder, decoder_init, decoder_step)
  - [ ] 加载 tokenizer.json
  - [ ] mmap embed_tokens.bin
  - [ ] 初始化 MelFrontend
  - [ ] 计算 num_threads (默认 min(cpu_count-1, 8))
- [ ] **验证**: new() 成功返回，无 panic

#### 2.2.2 Prompt 构建

- [ ] 实现 `build_prompt(audio_len: usize) -> (Array2<i64>, Array1<i64>)`
  - encode `"<|im_start|>system<|im_end|><|im_start|>user<|audio_start|>"` → prefix_ids
  - encode `"<|audio_end|><|im_end|><|im_start|>assistant"` → suffix_ids
  - 组合: prefix_ids + [audio_pad_id; audio_len] + suffix_ids → input_ids `[1, seq_len]`
  - audio_offset = prefix_ids.len()
- [ ] **验证**: 与 `ref_asr_encode.json` 对比 prefix/suffix token ids

#### 2.2.3 Embedding 查找

- [ ] 实现 `lookup_embedding(token_id: usize) -> Array3<f32>`
  - offset = token_id * hidden_size(1024) * 2 (f16 = 2 bytes)
  - 从 mmap 切片 → bytemuck::cast_slice → &[f16]
  - 逐元素 f16::to_f32()
  - reshape to `[1, 1, 1024]`
- [ ] 添加边界检查: token_id < vocab_size
- [ ] **验证**: 查找 token 0 和 token 100，输出不全为 0

#### 2.2.4 Argmax 工具

- [ ] 实现 `argmax_last(logits: &ArrayViewD<f32>) -> usize`
  - 取 logits 最后一个时间步: `logits.slice(s![0, -1, ..])`
  - 返回最大值的 index
- [ ] **验证**: 手工构造 tensor 验证正确性

#### 2.2.5 文本清理

- [ ] 实现 `clean_text(text: &str) -> String`
  - 去除 `"<asr_text>"`
  - 正则去除 `<|...|>` 标签: `Regex::new(r"<\|[^|]*\|>")`
  - trim
- [ ] **验证**: `clean_text("<asr_text>你好<|im_end|>")` == `"你好"`

#### 2.2.6 transcribe() 主流程

- [ ] 实现 `transcribe(wav: &[f32]) -> Result<AsrResult>`
  - Step 1: `self.mel_frontend.log_mel(wav)` → mel `[1, 128, T]`
  - Step 2: `self.encoder.run(["mel" => mel])` → audio_features `[1, ?, 1024]`
  - Step 3: `self.build_prompt(audio_len)` → (input_ids, audio_offset)
  - Step 4: 构造 position_ids `[0..seq_len]`
  - Step 5: `self.decoder_init.run(...)` → (logits, past_keys, past_values)
  - Step 6: argmax → 自回归循环 (max_new_tokens 次)
    - 每步: eos 检查 → embed lookup → decoder_step.run → argmax
  - Step 7: tokenizer.decode(out_ids) → clean_text
  - Step 8: 计时返回 AsrResult
- [ ] **验证 (encoder)**: encoder 输出与 `ref_asr_audio_features.npy` 前 10 个值对比
- [ ] **验证 (decoder_init)**: init logits 最后位置与 `ref_asr_init_logits_last.npy` 对比
- [ ] **验证 (端到端)**: 最终文本与 `ref_asr_result.json` 的 text 字段完全一致

### 2.3 `src/asr/server.rs` — HTTP 服务

- [ ] 实现 `router(engine: Arc<AsrEngine>) -> Router`
- [ ] 实现 `POST /asr` handler
  - 解析 multipart: `audio` (bytes), `language` (string, 默认 "auto"), `use_itn` (bool, 默认 true)
  - 调用 `load_audio_from_bytes` + `engine.transcribe`
  - 返回 JSON: `{"text", "feat_ms", "infer_ms", "total_ms", "segments"}`
  - 错误时返回 500: `{"text": "", "error": "..."}`
- [ ] 实现 `GET /health` → `{"status": "ok", "model": "qwen3-asr"}`
- [ ] **验证**: `curl -F audio=@test.wav http://localhost:8000/asr` 返回正确 JSON
- [ ] **验证**: `curl http://localhost:8000/health` 返回 200

### 2.4 `src/asr/mod.rs`

- [ ] 导出 `AsrEngine`, `AsrResult`, `router`

---

## Phase 3: NLU 引擎 `src/nlu/` (Day 2 上午)

### 3.1 `src/nlu/engine.rs`

#### 3.1.1 结构体与初始化

- [ ] 定义 `NluEngine` 结构体 (2 个 ort::Session + tokenizer + config)
- [ ] 实现 `NluEngine::new(model_dir, tokenizer_dir, num_threads, use_gpu)`
  - [ ] 加载 encoder.onnx + decoder.onnx (各一个 session)
  - [ ] 加载 tokenizer.json 并配置 padding + truncation:
    ```rust
    tokenizer.with_padding(Some(PaddingParams {
        strategy: PaddingStrategy::Fixed(config.max_seq_len),  // 64
        pad_id: config.pad_token_id as u32,
        pad_token: "</s>".into(),  // 或从 tokenizer_config.json 读取
        direction: PaddingDirection::Right,
        ..Default::default()
    }));
    tokenizer.with_truncation(Some(TruncationParams {
        max_length: config.max_seq_len,
        strategy: TruncationStrategy::LongestFirst,
        ..Default::default()
    }))?;
    ```
  - [ ] 加载 export_config.json
  - [ ] 计算 num_threads (默认 min(cpu_count-1, 4))
- [ ] **验证**: new() 成功返回

#### 3.1.2 Tokenize

- [ ] 实现 `tokenize(text: &str) -> (Array2<i64>, Array2<i64>)`
  - 拼接前缀: `format!("指令解析: {}", text)`
  - encode → get_ids() → 转 i64 → reshape `[1, 64]`
  - get_attention_mask() → 转 i64 → reshape `[1, 64]`
- [ ] **验证**: 与 `ref_nlu_encode.json` 对比 input_ids 和 attention_mask

#### 3.1.3 Structured Early Stop

- [ ] 实现 `should_stop_structured(tokenizer, token_ids: &[u32]) -> bool`
  - decode token_ids → text
  - 检查最后一个字符是否为 `}` 或 `]`
  - 如果是，尝试 `serde_json::from_str` — 成功则 early stop
- [ ] **验证**: `"{"intent":"stop","slots":{}}"` → true
- [ ] **验证**: `"{"intent":"stop","slots":{}"` → false (不完整)
- [ ] **验证**: `"向前走"` → false

#### 3.1.4 predict() 主流程

- [ ] 实现 `predict(text: &str) -> Result<NluOutput>`
  - Step 1: tokenize(text) → (input_ids, attention_mask)
  - Step 2: encoder.run → hidden_states `[1, 64, 768]`
  - Step 3: 初始化 dec_ids = `[decoder_start_token_id]` (= 0)
  - Step 4: 自回归循环 (max_output=128 次)
    - 构造 dec_input `[1, len]`
    - decoder.run(dec_input, hidden_states, attention_mask) → logits `[1, len, 32128]`
    - argmax(logits[0, -1]) → next_id
    - eos 检查 (next_id == 1)
    - structured early stop 检查
    - 追加到 dec_ids
  - Step 5: tokenizer.decode(dec_ids) → raw_output
  - Step 6: parse_nlu_output(raw_output) → NluOutput
- [ ] **验证**: predict("向前走三步") 与 `ref_nlu_result.json` 对比
- [ ] **验证**: predict("停止") 与 `ref_nlu_result2.json` 对比
- [ ] **验证**: predict("坐下") 与 `ref_nlu_result3.json` 对比

### 3.2 `src/nlu/parser.rs` — 输出解析

#### 3.2.1 数据结构

- [ ] 定义 `NluOutput`
  ```rust
  #[derive(Serialize)]
  pub struct NluOutput {
      pub intent: String,
      pub slots: HashMap<String, Value>,
      pub raw: String,
      #[serde(skip_serializing_if = "Option::is_none")]
      pub command: Option<Value>,
      #[serde(skip_serializing_if = "Option::is_none")]
      pub source: Option<String>,
      #[serde(skip_serializing_if = "Option::is_none")]
      pub message: Option<String>,
  }
  ```

#### 3.2.2 command_type → intent 映射表

- [ ] 实现 `command_type_to_intent(cmd_type: &str, payload: &Map) -> String`
  - 构建 20 条映射 (照搬 Python `_command_type_to_intent`):
    ```
    moveforward → move_forward
    movebackward → move_backward
    moveleft → move_left
    moveright → move_right
    turnleft → turn_left
    turnright → turn_right
    sit / sitdown / standdown → sit_down
    stand / standup / risesit / recoverystand / balancestand → stand_up
    liedown → lie_down
    greet → greet
    shakebody → shake_body
    stretch → stretch
    damp / stop / stopmove → stop
    error → unknown
    ```
  - 特殊处理 `"move"` → 调用 `infer_move_intent(payload)`

#### 3.2.3 infer_move_intent

- [ ] 实现 `infer_move_intent(payload: &Map) -> String`
  - 提取 vx, vy, vyaw (默认 0.0)
  - |vyaw| > max(|vx|, |vy|) → turn_left / turn_right
  - |vy| > |vx| → move_left / move_right
  - |vx| > 0 → move_forward / move_backward
  - 否则 → "move"

#### 3.2.4 augment_slots

- [ ] 实现 `augment_slots(slots, intent, command_type)`
  - 根据 intent 补充 direction 字段
  - 补充 command_type 字段

#### 3.2.5 parse_nlu_output 主函数

- [ ] 实现 `parse_nlu_output(raw: &str) -> NluOutput`
  - 路径 A: JSON 解析成功
    - type == "cmd" → 提取 payload.command_type + payload.payload_json → intent + slots
    - type == "chat" → intent=unknown, 提取 message
    - 含 intent/slots 字段 → 直接使用
  - 路径 B: 含 `=` → key=value 解析 (分号/逗号分隔)
  - 路径 C: fallback → intent=unknown
- [ ] **验证**: 用 `ref_parse_cases.json` 的 10 组用例全部通过

### 3.3 `src/nlu/server.rs`

- [ ] 定义请求体 `NluRequest { text: String }`
- [ ] 实现 `router(engine: Arc<NluEngine>) -> Router`
- [ ] 实现 `POST /nlu` handler
  - 解析 JSON body
  - 调用 `engine.predict(&req.text)`
  - 返回 JSON NluOutput
  - 错误时返回 500: `{"intent":"unknown","slots":{},"raw":"...","error":"..."}`
- [ ] 实现 `GET /health` → `{"status": "ok"}`
- [ ] **验证**: `curl -X POST -H 'Content-Type: application/json' -d '{"text":"向前走"}' http://localhost:8001/nlu`
- [ ] **验证**: `curl http://localhost:8001/health`

### 3.4 `src/nlu/mod.rs`

- [ ] 导出 `NluEngine`, `NluOutput`, `router`, `parse_nlu_output`

---

## Phase 4: 入口与集成 (Day 2 下午)

### 4.1 `src/main.rs`

- [ ] 用 clap derive 定义 CLI 参数:
  ```
  --asr-model-dir   (默认 "models/asr")
  --nlu-model-dir   (默认 "models/nlu")
  --nlu-tokenizer-dir (默认 "{nlu-model-dir}/tokenizer")
  --host            (默认 "0.0.0.0")
  --asr-port        (默认 8000)
  --nlu-port        (默认 8001)
  --gpu             (flag)
  --threads         (可选)
  --asr-only        (flag, 只启动 ASR)
  --nlu-only        (flag, 只启动 NLU)
  ```
- [ ] 初始化 tracing subscriber (日志)
- [ ] 初始化 ort runtime (`ort::init().commit()`)
- [ ] 并行加载 ASR + NLU 引擎 (`tokio::join!`)
- [ ] 绑定两个 TCP listener
- [ ] `tokio::select!` 并行运行两个 axum server
- [ ] 注册 Ctrl+C 信号 graceful shutdown
- [ ] **验证**: `./voice-infer --help` 打印帮助
- [ ] **验证**: 启动后两个端口都可访问

### 4.2 日志

- [ ] 引擎加载时输出: 模型路径、线程数、provider、量化类型
- [ ] 每次推理输出: 耗时、结果摘要 (与 Python 版日志格式保持一致)
- [ ] 示例: `ASR total=87ms | 向前走三步`

### 4.3 错误处理

- [ ] 模型文件缺失 → 启动时 panic 并明确报错 (列出缺少的文件)
- [ ] ONNX runtime 加载失败 → 报错提示检查 `ORT_DYLIB_PATH`
- [ ] 音频格式不支持 → 返回 400 (非 500)
- [ ] 推理异常 → 返回 500 + error 字段

---

## Phase 5: 测试 (Day 3 上午)

### 5.1 单元测试 `tests/`

- [ ] `test_mel_filterbank` — 滤波器组 shape 和数值
- [ ] `test_mel_hann_window` — Hann 窗数值
- [ ] `test_mel_power_spectrum` — 单帧 FFT
- [ ] `test_mel_full` — 完整 mel 频谱 vs Python 参考值
- [ ] `test_mel_empty_audio` — 空输入不 panic
- [ ] `test_wav_16k_mono` — 标准 WAV 读取
- [ ] `test_wav_resample` — 8kHz → 16kHz
- [ ] `test_wav_stereo` — 立体声 → mono
- [ ] `test_asr_tokenizer_encode` — prompt 构建 token ids
- [ ] `test_asr_tokenizer_decode` — decode + clean_text
- [ ] `test_asr_embed_lookup` — embedding 非零且 shape 正确
- [ ] `test_nlu_tokenizer_encode` — padding 到 64
- [ ] `test_nlu_tokenizer_decode` — skip_special_tokens
- [ ] `test_nlu_parse_cmd_json` — type=cmd 解析
- [ ] `test_nlu_parse_move_infer` — vx/vy/vyaw 方向推断
- [ ] `test_nlu_parse_chat_json` — type=chat 解析
- [ ] `test_nlu_parse_kv` — key=value 格式
- [ ] `test_nlu_parse_fallback` — 纯文本 → unknown
- [ ] `test_structured_early_stop` — JSON 完整性检测
- [ ] `cargo test` 全部通过

### 5.2 集成测试

- [ ] ASR 端到端: test_1s.wav → transcribe → 文本与 Python 一致
- [ ] NLU 端到端: "向前走三步" → predict → 与 Python 输出一致
- [ ] NLU 端到端: "停止" → intent == "stop"
- [ ] HTTP ASR: multipart 上传 → 正确响应
- [ ] HTTP NLU: JSON 请求 → 正确响应
- [ ] HTTP health: 两个端口都返回 200
- [ ] 并发测试: 10 个并行请求不 panic、不阻塞

### 5.3 Python 对比测试脚本

- [ ] 写 `scripts/compare_outputs.py`:
  - 同时启动 Python 和 Rust 服务 (不同端口)
  - 用 10 条测试音频分别请求
  - 对比 text 字段是否完全一致
  - 输出性能对比表 (ms)

---

## Phase 6: 构建与部署 (Day 3 下午)

### 6.1 Release 构建

- [ ] `cargo build --release` 成功
- [ ] 检查产物大小: `ls -lh target/release/voice-infer` (预期 < 10MB)
- [ ] strip 符号表: `strip target/release/voice-infer` (进一步缩小)

### 6.2 交叉编译 (如需)

- [ ] 确认目标平台架构 (x86_64 / aarch64)
- [ ] 如果 aarch64:
  - [ ] `cargo build --release --target aarch64-unknown-linux-gnu`
  - [ ] 确保链接的 libonnxruntime.so 也是 aarch64 版本

### 6.3 打包部署目录

- [ ] 创建部署包:
  ```
  voice-infer-deploy/
  ├── voice-infer                    # Rust 二进制
  ├── libonnxruntime.so.1.19.2       # ONNX Runtime
  ├── models/                        # 符号链接或拷贝
  │   ├── asr/ (全部 6 文件)
  │   └── nlu/ (encoder.onnx, decoder.onnx, tokenizer/, export_config.json)
  └── start.sh
  ```
- [ ] 写 `start.sh`:
  ```bash
  #!/bin/bash
  DIR="$(cd "$(dirname "$0")" && pwd)"
  export ORT_DYLIB_PATH="$DIR/libonnxruntime.so.1.19.2"
  exec "$DIR/voice-infer" \
      --asr-model-dir "$DIR/models/asr" \
      --nlu-model-dir "$DIR/models/nlu" \
      "$@"
  ```
- [ ] `chmod +x start.sh voice-infer`

### 6.4 目标机器验证

- [ ] scp 部署包到目标机器
- [ ] `./start.sh` 启动成功
- [ ] `curl localhost:8000/health` → ok
- [ ] `curl localhost:8001/health` → ok
- [ ] 用机器狗端 pipeline 实际发送请求验证
- [ ] 检查内存占用: `ps aux | grep voice-infer` (预期比 Python 少 100-200MB)
- [ ] 运行 10 分钟稳定性测试，无内存泄漏

### 6.5 与机器狗端联调

- [ ] 机器狗端 `run.py --pipeline-only` 配置 ASR/NLU 地址指向 Rust 服务
- [ ] 语音唤醒 → ASR → NLU → 命令执行 全链路跑通
- [ ] 确认延迟 ≤ Python 版
- [ ] 连续运行 1 小时无异常

---

## 完成标志

全部 checkbox 打勾后，Rust 重写完成。Python 上位机代码可归档但不再部署使用。

最终部署架构:
```
┌──────────────────────┐     HTTP      ┌──────────────────────┐
│  机器狗 (Python 3.8)  │ ──────────→  │  上位机 (Rust 二进制)  │
│  pipeline-only 模式   │ ←────────── │  ASR :8000 + NLU :8001│
│  run.py --pipeline    │   JSON       │  单进程，零 Python     │
└──────────────────────┘              └──────────────────────┘
```
