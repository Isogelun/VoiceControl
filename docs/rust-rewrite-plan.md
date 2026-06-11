# Rust 重写方案：上位机 ASR + NLU 推理服务

> 相关文档: [实施清单](rust-rewrite-checklist.md) | [Rust 项目 README](../voice-infer/README.md) | [主 README](../README.md)

## 1. 项目结构

```
voice-infer/
├── Cargo.toml
├── src/
│   ├── main.rs              # 入口：CLI 参数解析 + 启动双服务
│   ├── config.rs            # config.json / export_config.json 加载
│   ├── asr/
│   │   ├── mod.rs
│   │   ├── engine.rs        # Qwen3-ASR ONNX 推理引擎
│   │   └── server.rs        # POST /asr + GET /health
│   ├── nlu/
│   │   ├── mod.rs
│   │   ├── engine.rs        # Mengzi-T5 ONNX 推理引擎
│   │   ├── parser.rs        # parse_nlu_output — JSON/key=value 解析
│   │   └── server.rs        # POST /nlu + GET /health
│   └── audio/
│       ├── mod.rs
│       ├── mel.rs           # Mel 频谱计算 (替代 librosa)
│       └── wav.rs           # WAV 读取 + 重采样
├── models/                  # 符号链接或复制，与 Python 版共用同一套模型
│   ├── asr/
│   └── nlu/
└── tests/
    ├── test_mel.rs          # Mel 频谱对比 Python 输出
    └── test_inference.rs    # 端到端推理对比
```

## 2. Cargo.toml

```toml
[package]
name = "voice-infer"
version = "0.1.0"
edition = "2021"

[dependencies]
# ONNX Runtime
ort = { version = "2", features = ["load-dynamic"] }  # 动态链接 libonnxruntime

# Tokenizer (HuggingFace — ASR + NLU 共用)
tokenizers = { version = "0.20", default-features = false, features = ["onig"] }

# HTTP
axum = "0.7"
tokio = { version = "1", features = ["full"] }
tower = "0.4"
tower-http = { version = "0.5", features = ["cors"] }

# 音频
hound = "3.5"                    # WAV 读取
rubato = "0.15"                  # 高质量重采样 (替代 librosa.resample)

# 数学 / FFT
ndarray = "0.16"                 # numpy 等价
rustfft = "6"                    # FFT (mel 频谱)
half = { version = "2", features = ["num-traits"] }  # float16 embed_tokens

# 序列化
serde = { version = "1", features = ["derive"] }
serde_json = "1"

# 工具
clap = { version = "4", features = ["derive"] }  # CLI
tracing = "0.1"
tracing-subscriber = "0.3"
anyhow = "1"
memmap2 = "0.9"                  # embed_tokens.bin 内存映射

# multipart 上传 (ASR 接口)
axum-extra = { version = "0.9", features = ["multipart"] }
```

> **`load-dynamic` 说明**：`ort` crate 的 `load-dynamic` feature 让二进制在运行时加载
> `libonnxruntime.so`，而不是静态编译进去。这样产物只有几 MB，部署时把
> `libonnxruntime.so.1.19.2` 放在同目录或 `LD_LIBRARY_PATH` 即可。

## 3. 逐模块实现细节

---

### 3.1 音频模块 `src/audio/`

#### `wav.rs` — WAV 读取 + 重采样

```rust
/// 读取 WAV/PCM 音频，输出 16kHz mono f32 采样
pub fn load_audio(data: &[u8]) -> Result<Vec<f32>> {
    // 1. hound::WavReader::new(Cursor::new(data))
    // 2. 读取所有 samples → f32
    // 3. 如果多声道 → 取平均
    // 4. 如果采样率 != 16000 → rubato::SincFixedIn 重采样
    // 5. 返回 Vec<f32>
}
```

**对应 Python**: `asr/engine.py:292-310` (`load_audio` 函数)

#### `mel.rs` — Mel 频谱前端 (替代 librosa)

这是最关键的纯数学模块，需要精确复现 Python 的 `_log_mel_fast()` 输出。

```rust
pub struct MelFrontend {
    n_fft: usize,        // 400
    hop_length: usize,   // 160
    n_mels: usize,       // 128
    window: Vec<f32>,    // Hann 窗
    mel_basis: Array2<f32>,  // [n_mels, n_fft/2+1] mel 滤波器组
}

impl MelFrontend {
    /// 初始化：创建 Hann 窗 + Mel 滤波器组
    pub fn new(config: &MelConfig) -> Self { ... }

    /// 计算 log-mel 频谱 → [1, 128, time] f32
    pub fn log_mel(&self, wav: &[f32]) -> Array3<f32> {
        // 1. 两端 zero-pad (n_fft/2)
        // 2. 分帧 (stride = hop_length, width = n_fft)
        // 3. 逐帧：Hann 窗 → rfft(rustfft) → |X|²
        // 4. power @ mel_basis.T → mel[n_mels, frames]
        // 5. log10(max(mel, 1e-10))
        // 6. max(log_mel, max - 8.0)
        // 7. (log_mel + 4.0) / 4.0
        // 8. reshape to [1, 128, frames]
    }
}
```

**Mel 滤波器组生成** (替代 `librosa.filters.mel()`):

```rust
/// Slaney-style mel 滤波器组
fn create_mel_filterbank(sr: usize, n_fft: usize, n_mels: usize,
                         fmin: f32, fmax: f32) -> Array2<f32> {
    // 1. Hz → Mel: mel = 2595 * log10(1 + f/700)
    // 2. 在 mel 域等距取 n_mels+2 个点
    // 3. Mel → Hz 转回
    // 4. Hz → FFT bin index
    // 5. 构造三角滤波器
    // 6. Slaney 归一化: filter[i] /= (upper[i] - lower[i])
}
```

**验证方法**: 用 Python 导出一段音频的 mel 频谱作为 ground truth，Rust 实现后逐元素对比，误差 < 1e-5。

---

### 3.2 ASR 引擎 `src/asr/engine.rs`

```rust
pub struct AsrEngine {
    encoder: ort::Session,          // encoder.int4.onnx
    decoder_init: ort::Session,     // decoder_init.int4.onnx
    decoder_step: ort::Session,     // decoder_step.int4.onnx
    tokenizer: tokenizers::Tokenizer,
    embed_tokens: Mmap,             // memmap2 映射 embed_tokens.bin
    mel_frontend: MelFrontend,
    config: AsrConfig,
    max_new_tokens: usize,          // 默认 32
}
```

#### ONNX 会话创建

```rust
fn create_session(path: &Path, num_threads: usize) -> Result<ort::Session> {
    ort::Session::builder()?
        .with_intra_threads(num_threads)?
        .with_inter_threads(1)?
        .with_optimization_level(ort::GraphOptimizationLevel::Level3)?
        .commit_from_file(path)
}
```

#### 推理流程 — 精确对应 Python `transcribe()`

```rust
pub fn transcribe(&self, wav: &[f32]) -> Result<AsrResult> {
    let t0 = Instant::now();

    // Step 1: Mel 频谱
    let mel = self.mel_frontend.log_mel(wav);            // [1, 128, T]
    let feat_ms = t0.elapsed().as_secs_f64() * 1000.0;

    // Step 2: Encoder
    let ti = Instant::now();
    let audio_features = self.encoder.run(
        ort::inputs!["mel" => mel.view()]?
    )?["audio_features"];                                // [1, audio_len, 1024]

    // Step 3: Build prompt
    let audio_len = audio_features.shape()[1];
    let (input_ids, audio_offset) = self.build_prompt(audio_len);
    let seq_len = input_ids.len();
    let position_ids: Vec<i64> = (0..seq_len as i64).collect();

    // Step 4: Decoder init
    let init_out = self.decoder_init.run(ort::inputs![
        "input_ids"      => input_ids.view(),
        "position_ids"   => position_ids.view(),
        "audio_features" => audio_features.view(),
        "audio_offset"   => audio_offset.view(),
    ]?)?;
    let mut logits = init_out["logits"];
    let mut past_keys = init_out["present_keys"];
    let mut past_values = init_out["present_values"];

    // Step 5: 自回归解码循环
    let mut out_ids = Vec::new();
    let mut next_id = argmax_last(&logits);

    for _ in 0..self.max_new_tokens {
        if self.config.eos_token_ids.contains(&next_id) {
            break;
        }
        out_ids.push(next_id);

        // embed_tokens lookup: f16 → f32
        let embed = self.lookup_embedding(next_id);      // [1, 1, 1024]
        let step_pos = (seq_len + out_ids.len() - 1) as i64;

        let step_out = self.decoder_step.run(ort::inputs![
            "input_embeds" => embed.view(),
            "position_ids" => array![[step_pos]].view(),
            "past_keys"    => past_keys.view(),
            "past_values"  => past_values.view(),
        ]?)?;

        logits = step_out["logits"];
        past_keys = step_out["present_keys"];
        past_values = step_out["present_values"];
        next_id = argmax_last(&logits);
    }

    // Step 6: 解码文本
    let text = self.tokenizer.decode(&out_ids, true)?;   // skip_special_tokens=true
    let text = clean_text(&text);                         // 去除 <asr_text> 等标签

    let infer_ms = ti.elapsed().as_secs_f64() * 1000.0;
    Ok(AsrResult { text, feat_ms, infer_ms, total_ms: ... })
}
```

#### embed_tokens 读取

```rust
/// 从 memmap 的 f16 数组中查找 token embedding
fn lookup_embedding(&self, token_id: usize) -> Array3<f32> {
    let hidden_size = self.config.decoder.hidden_size;  // 1024
    let offset = token_id * hidden_size * 2;             // f16 = 2 bytes
    let bytes = &self.embed_tokens[offset..offset + hidden_size * 2];
    let f16_slice: &[f16] = bytemuck::cast_slice(bytes);
    let f32_vec: Vec<f32> = f16_slice.iter().map(|x| x.to_f32()).collect();
    Array3::from_shape_vec((1, 1, hidden_size), f32_vec).unwrap()
}
```

**对应 Python**: `asr/engine.py:72-285` (整个 `Qwen3ASREngine` 类)

---

### 3.3 NLU 引擎 `src/nlu/engine.rs`

```rust
pub struct NluEngine {
    encoder: ort::Session,           // encoder.onnx
    decoder: ort::Session,           // decoder.onnx
    tokenizer: tokenizers::Tokenizer,
    config: NluExportConfig,
    max_input: usize,    // 64
    max_output: usize,   // 128
}
```

#### 推理流程

```rust
pub fn predict(&self, text: &str) -> Result<NluResult> {
    let input = format!("指令解析: {}", text);

    // Step 1: Tokenize (padding + truncation)
    let encoding = self.tokenizer.encode(input, true)?;
    let mut input_ids = encoding.get_ids().to_vec();
    let mut attention_mask = encoding.get_attention_mask().to_vec();

    // Pad to max_input (64)
    pad_or_truncate(&mut input_ids, &mut attention_mask, self.max_input,
                    self.config.pad_token_id as u32);

    // Cast to i64
    let input_ids: Array2<i64> = /* [1, 64] */;
    let attention_mask: Array2<i64> = /* [1, 64] */;

    // Step 2: Encoder
    let enc_out = self.encoder.run(ort::inputs![
        "input_ids"      => input_ids.view(),
        "attention_mask"  => attention_mask.view(),
    ]?)?;
    let hidden_states = &enc_out["last_hidden_state"];   // [1, 64, 768]

    // Step 3: 自回归解码
    let mut dec_ids = vec![self.config.decoder_start_token_id]; // [0]
    let mut generated = Vec::new();

    for _ in 0..self.max_output {
        let dec_input: Array2<i64> = /* [1, len] from dec_ids */;

        let dec_out = self.decoder.run(ort::inputs![
            "decoder_input_ids"       => dec_input.view(),
            "encoder_hidden_states"   => hidden_states.view(),
            "encoder_attention_mask"  => attention_mask.view(),
        ]?)?;

        let logits = &dec_out["logits"];                 // [1, len, 32128]
        let next_id = argmax_last(logits);

        if next_id == self.config.eos_token_id as usize {
            break;
        }
        generated.push(next_id);
        dec_ids.push(next_id as i64);

        // Structured early stop
        if self.should_stop_structured(&generated) {
            break;
        }
    }

    // Step 4: 解码 + 解析
    let raw_output = self.tokenizer.decode(&dec_ids, true)?;
    let parsed = parse_nlu_output(&raw_output);
    Ok(parsed)
}
```

**对应 Python**: `nlu/engine.py:43-128`

---

### 3.4 NLU 输出解析 `src/nlu/parser.rs`

直译 Python 的 `parse_nlu_output()` + `_parse_json_output()` + `_command_type_to_intent()`：

```rust
pub fn parse_nlu_output(raw: &str) -> NluOutput {
    // 1. 尝试 JSON 解析
    if let Ok(parsed) = serde_json::from_str::<Value>(raw) {
        if let Some(result) = parse_json_output(&parsed, raw) {
            return result;
        }
    }
    // 2. 尝试 key=value 格式
    if raw.contains('=') {
        return parse_kv_output(raw);
    }
    // 3. fallback
    NluOutput { intent: "unknown".into(), slots: HashMap::new(), raw: raw.into() }
}

fn command_type_to_intent(cmd_type: &str, payload: &Map) -> String {
    // 完整搬运 Python 的 mapping 表 (20 条映射)
    // 包含 "Move" 的 vx/vy/vyaw 方向推断
}
```

**对应 Python**: `nlu/engine.py:140-290` (约 150 行，直译即可)

---

### 3.5 HTTP 服务 `src/asr/server.rs` + `src/nlu/server.rs`

#### ASR Server

```rust
pub fn router(engine: Arc<AsrEngine>) -> Router {
    Router::new()
        .route("/asr", post(asr_handler))
        .route("/health", get(|| async { Json(json!({"status":"ok","model":"qwen3-asr"})) }))
        .with_state(engine)
}

async fn asr_handler(
    State(engine): State<Arc<AsrEngine>>,
    mut multipart: Multipart,
) -> Result<Json<AsrResult>, StatusCode> {
    // 解析 multipart: audio 文件 + language + use_itn
    // engine.transcribe(wav)
    // 返回 JSON
}
```

#### NLU Server

```rust
#[derive(Deserialize)]
struct NluRequest { text: String }

pub fn router(engine: Arc<NluEngine>) -> Router {
    Router::new()
        .route("/nlu", post(nlu_handler))
        .route("/health", get(|| async { Json(json!({"status":"ok"})) }))
        .with_state(engine)
}

async fn nlu_handler(
    State(engine): State<Arc<NluEngine>>,
    Json(req): Json<NluRequest>,
) -> Json<NluOutput> {
    // engine.predict(&req.text)
}
```

**对应 Python**: `asr/server.py` (109 行) + `nlu/server.py` (97 行)

---

### 3.6 入口 `src/main.rs`

```rust
#[derive(Parser)]
struct Args {
    #[arg(long, default_value = "models/asr")]
    asr_model_dir: PathBuf,
    #[arg(long, default_value = "models/nlu")]
    nlu_model_dir: PathBuf,
    #[arg(long, default_value = "0.0.0.0")]
    host: String,
    #[arg(long, default_value_t = 8000)]
    asr_port: u16,
    #[arg(long, default_value_t = 8001)]
    nlu_port: u16,
    #[arg(long)]
    gpu: bool,
    #[arg(long)]
    threads: Option<usize>,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    // 并行加载两个引擎
    let (asr_engine, nlu_engine) = tokio::join!(
        load_asr(&args),
        load_nlu(&args),
    );

    // 启动两个 HTTP 服务
    let asr_server = axum::serve(
        TcpListener::bind((args.host.as_str(), args.asr_port)).await?,
        asr::server::router(Arc::new(asr_engine?)),
    );
    let nlu_server = axum::serve(
        TcpListener::bind((args.host.as_str(), args.nlu_port)).await?,
        nlu::server::router(Arc::new(nlu_engine?)),
    );

    // 并行运行
    tokio::select! {
        r = asr_server => r?,
        r = nlu_server => r?,
    }
    Ok(())
}
```

## 4. 关键技术细节

### 4.1 tokenizers crate 用法

ASR 和 NLU 的 tokenizer.json 都是 HuggingFace 标准格式，直接用 Rust 原生 API：

```rust
// ASR — 纯 BPE，不需要 padding
let tokenizer = tokenizers::Tokenizer::from_file("models/asr/tokenizer.json")?;
let encoded = tokenizer.encode("<|im_start|>system...", false)?;
let ids: &[u32] = encoded.get_ids();

// NLU — 需要 padding 到 max_length=64
let mut tokenizer = tokenizers::Tokenizer::from_file("models/nlu/tokenizer/tokenizer.json")?;
tokenizer.with_padding(Some(PaddingParams {
    strategy: PaddingStrategy::Fixed(64),
    pad_id: 0,
    ..Default::default()
}));
tokenizer.with_truncation(Some(TruncationParams {
    max_length: 64,
    strategy: TruncationStrategy::LongestFirst,
    ..Default::default()
}))?;
```

### 4.2 ort crate (onnxruntime) 用法

```rust
// 初始化 — 指定 libonnxruntime.so 路径
ort::init()
    .with_execution_providers([CPUExecutionProvider::default()])
    .commit()?;

// 推理
let outputs = session.run(ort::inputs![
    "input_ids" => CowArray::from(input_ids),  // ndarray → ort tensor
]?)?;
let logits: ArrayViewD<f32> = outputs["logits"].try_extract_tensor()?;
```

### 4.3 embed_tokens.bin 内存映射

```rust
use memmap2::Mmap;
use half::f16;

let file = File::open("models/asr/embed_tokens.bin")?;
let mmap = unsafe { Mmap::map(&file)? };

// 查找 token_id 的 embedding (vocab_size=151936, hidden_size=1024, dtype=f16)
fn lookup(mmap: &Mmap, token_id: usize, hidden_size: usize) -> Vec<f32> {
    let offset = token_id * hidden_size * 2;  // f16 = 2 bytes
    let bytes = &mmap[offset..offset + hidden_size * 2];
    let f16s: &[f16] = bytemuck::cast_slice(bytes);
    f16s.iter().map(|x| x.to_f32()).collect()
}
```

### 4.4 Mel 频谱 FFT

```rust
use rustfft::{FftPlanner, num_complex::Complex};

fn rfft(frame: &[f32], n_fft: usize) -> Vec<f32> {
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n_fft);

    let mut buffer: Vec<Complex<f32>> = frame.iter()
        .map(|&x| Complex::new(x, 0.0))
        .collect();
    fft.process(&mut buffer);

    // 取前 n_fft/2+1 个的模的平方 (power spectrum)
    buffer[..n_fft/2+1].iter().map(|c| c.norm_sqr()).collect()
}
```

## 5. 构建与部署

### 5.1 本机构建 (开发)

```bash
# 前置：安装 Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# 构建
cd voice-infer
cargo build --release

# 运行（需要 libonnxruntime.so 在 LD_LIBRARY_PATH）
export ORT_DYLIB_PATH=/path/to/libonnxruntime.so
./target/release/voice-infer \
    --asr-model-dir ../models/asr \
    --nlu-model-dir ../models/nlu
```

### 5.2 交叉编译 (为目标平台)

如果上位机是 x86_64 Linux:
```bash
# 直接在 x86_64 上构建即可
cargo build --release --target x86_64-unknown-linux-gnu
```

如果上位机是 aarch64 (如 Jetson):
```bash
# 安装交叉编译工具链
rustup target add aarch64-unknown-linux-gnu
sudo apt install gcc-aarch64-linux-gnu

# 交叉编译
cargo build --release --target aarch64-unknown-linux-gnu
```

### 5.3 部署产物

```
部署目录/
├── voice-infer              # 单二进制 (~5MB)
├── libonnxruntime.so.1.19.2 # ONNX Runtime 动态库 (~50MB)
├── models/
│   ├── asr/                 # 与 Python 版完全相同的模型文件
│   └── nlu/
└── start.sh                 # 启动脚本
```

`start.sh`:
```bash
#!/bin/bash
export ORT_DYLIB_PATH=$(dirname $0)/libonnxruntime.so.1.19.2
exec $(dirname $0)/voice-infer \
    --asr-model-dir $(dirname $0)/models/asr \
    --nlu-model-dir $(dirname $0)/models/nlu \
    --host 0.0.0.0 \
    --asr-port 8000 \
    --nlu-port 8001
```

## 6. 接口兼容性

HTTP 接口与 Python 版 100% 兼容，机器狗端 pipeline 代码零修改：

| 接口 | 方法 | 请求 | 响应 |
|------|------|------|------|
| `/asr` | POST | multipart: `audio`(file) + `language`(str) + `use_itn`(bool) | `{"text":"...", "feat_ms":..., "infer_ms":..., "total_ms":..., "segments":1}` |
| `/nlu` | POST | JSON: `{"text":"..."}` | `{"intent":"...", "slots":{...}, "raw":"..."}` |
| `/health` | GET | — | `{"status":"ok"}` |

## 7. 测试策略

### 7.1 单元测试

| 测试项 | 方法 |
|--------|------|
| Mel 频谱 | Python 导出参考值，Rust 逐元素对比 (max abs error < 1e-5) |
| Tokenizer encode/decode | 与 Python tokenizers 输出对比 token_ids |
| NLU 输出解析 | 搬运 Python tests/ 中的 parse_nlu_output 测试用例 |

### 7.2 集成测试

```bash
# 用同一段音频对比 Python 和 Rust 的 ASR 输出
python -m asr.server --serve --port 8000 &
curl -F audio=@test.wav http://localhost:8000/asr > python_result.json

./voice-infer --asr-port 8000 &
curl -F audio=@test.wav http://localhost:8000/asr > rust_result.json

diff python_result.json rust_result.json  # 文本应完全一致
```

### 7.3 性能对比

预期提升点：
- 推理编排开销 (Python GIL → 零开销)
- Mel 频谱计算 (numpy → rustfft，避免 Python 循环)
- HTTP 服务吞吐 (FastAPI/uvicorn → axum/hyper)
- 内存占用 (无 Python 解释器 ~100MB 开销)

## 8. 风险与缓解

| 风险 | 概率 | 缓解 |
|------|------|------|
| Mel 频谱数值精度偏差导致 ASR 输出不同 | 中 | 先验证：导出 Python mel 作为 ground truth，逐帧对比 |
| ort crate 版本与 libonnxruntime.so 版本不匹配 | 低 | `load-dynamic` 模式下只要 ABI 兼容即可；锁定 ort 2.x + libonnxruntime 1.19.x |
| NLU tokenizer.json 格式与 Rust tokenizers crate 不兼容 | 极低 | 该 tokenizer.json 是由 tokenizers crate 生成的，原生支持 |

## 9. 时间估算

| 阶段 | 内容 | 预计耗时 |
|------|------|----------|
| Day 1 | mel.rs + wav.rs + asr/engine.rs + 验证 mel 精度 | 8h |
| Day 2 | nlu/engine.rs + parser.rs + 两个 server.rs + main.rs | 6h |
| Day 3 | 集成测试 + 交叉编译 + 部署验证 + 性能对比 | 4h |
| **合计** | | **~18h (2-3 天)** |
