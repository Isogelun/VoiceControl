# voice-infer

Rust 实现的 ASR + NLU 推理服务，替代 Python `asr/` 和 `nlu/` 模块。

单二进制，部署无需 Python 环境。HTTP 接口与 Python 版 100% 兼容，机器狗端 pipeline 代码零修改。

## 模块结构

```
src/
├── main.rs           # 入口: CLI + 双服务启动
├── config.rs         # config.json / export_config.json 解析
├── audio/
│   ├── mel.rs        # Mel 频谱前端 (替代 librosa)
│   └── wav.rs        # WAV 读取 + 重采样
├── asr/
│   ├── engine.rs     # Qwen3-ASR ONNX 推理 (encoder + decoder_init + decoder_step)
│   └── server.rs     # POST /asr + GET /health
└── nlu/
    ├── engine.rs     # Mengzi-T5 ONNX 推理 (encoder + decoder)
    ├── parser.rs     # NLU 输出解析 (JSON/key-value/fallback)
    └── server.rs     # POST /nlu + GET /health
```

## 依赖

| Crate | 用途 | 对应 Python |
|-------|------|-------------|
| `ort` (load-dynamic) | ONNX Runtime 绑定 | `onnxruntime` |
| `tokenizers` | HuggingFace 分词器 | `tokenizers` + `transformers` |
| `rustfft` | FFT (mel 频谱) | `numpy.fft` |
| `ndarray` | 张量运算 | `numpy` |
| `half` + `memmap2` | f16 embedding 查找 | `numpy.memmap` |
| `hound` + `rubato` | WAV 读取 + 重采样 | `soundfile` + `librosa` |
| `axum` + `tokio` | HTTP 服务 | `fastapi` + `uvicorn` |

## 构建

```bash
# 前置: Rust toolchain + libonnxruntime
cargo build --release
```

产物: `target/release/voice-infer` (~5MB，需运行时链接 `libonnxruntime.so`)

## 运行

```bash
export ORT_DYLIB_PATH=/path/to/libonnxruntime.so.1.19.2

./voice-infer \
    --asr-model-dir ../models/asr \
    --nlu-model-dir ../models/nlu \
    --host 0.0.0.0 \
    --asr-port 8000 \
    --nlu-port 8001
```

选项:

```
--asr-model-dir     ASR 模型目录 (默认 models/asr)
--nlu-model-dir     NLU 模型目录 (默认 models/nlu)
--nlu-tokenizer-dir NLU 分词器目录 (默认 {nlu-model-dir}/tokenizer)
--host              监听地址 (默认 0.0.0.0)
--asr-port          ASR 端口 (默认 8000)
--nlu-port          NLU 端口 (默认 8001)
--gpu               使用 CUDA
--threads           推理线程数 (默认 auto)
--asr-only          只启动 ASR
--nlu-only          只启动 NLU
```

## 接口

与 Python 版完全一致:

| 接口 | 方法 | 请求 | 响应 |
|------|------|------|------|
| `/asr` | POST | multipart: `audio` + `language` + `use_itn` | `{"text", "feat_ms", "infer_ms", "total_ms", "segments"}` |
| `/nlu` | POST | JSON: `{"text": "..."}` | `{"intent", "slots", "raw"}` |
| `/health` | GET | — | `{"status": "ok"}` |

## 测试

```bash
# 单元测试 (mel/wav/parser)
cargo test

# 导出 Python 参考数据
cd .. && python scripts/export_reference.py

# Python vs Rust 端到端对比
python scripts/compare_outputs.py \
    --py-asr http://localhost:8000 --py-nlu http://localhost:8001 \
    --rs-asr http://localhost:9000 --rs-nlu http://localhost:9001
```

## 部署

```
deploy/
├── voice-infer                    # 单二进制
├── libonnxruntime.so.1.19.2       # ONNX Runtime (~50MB)
├── models/                        # 与 Python 版共用模型
│   ├── asr/
│   └── nlu/
└── start.sh                       # 启动脚本
```

## 相关文档

- [重写方案](../docs/rust-rewrite-plan.md) — 技术设计与 Python→Rust 映射
- [实施清单](../docs/rust-rewrite-checklist.md) — 83 个 checkbox 逐步执行
