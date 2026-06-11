# Rust 重写当前状态
> 更新时间: 2026-06-10
> 相关文档: [重写方案](rust-rewrite-plan.md) | [开工文档](rust-rewrite-kickoff.md) | [实施清单](rust-rewrite-checklist.md) | [Rust README](../voice-infer/README.md)

这份文档补充说明 `voice-infer/` 当前已经做到哪里、哪些部分已经有参考对齐、以及接下来最值得继续推进的点。

## 1. 当前结论

截至 2026-06-10，Rust 重写已经达到下面这个状态：

- `voice-infer` 可以正常编译，`cargo check` 通过
- `cargo test` 通过，当前结果是 `33 passed, 0 failed, 6 ignored`
- ASR / NLU / HTTP 服务入口都已经有可运行代码，不再只是骨架
- Python 侧 ground truth 已经成功导出到 `tests/fixtures/reference/`

但这还不是“Rust 重写完成”。

当前更准确的判断是：

- 编译已经打通
- 基础单测已经成体系
- 一部分关键前处理/解析逻辑已经与 Python 参考对齐
- 端到端推理一致性还没有完全验收

## 2. 这轮推进完成了什么

### 2.1 依赖和运行时接口已经打通

前面已经完成了以下适配：

- `ort` 2.x RC API 适配
- `axum` 的 `multipart` feature 打开
- `tracing-subscriber` 的 `env-filter` feature 打开
- `ndarray` 升级到与 `ort` 对齐的版本

同时，因为 `Session::run()` 在当前 `ort` API 下要求可变借用，当前实现已经稳定到下面这个约束：

- `AsrEngine::transcribe()` 和 `NluEngine::predict()` 使用 `&mut self`
- HTTP 层通过 `Arc<Mutex<...>>` 管理 engine
- 每次 ONNX 输出都立刻 `to_owned()`，避免把 `SessionOutputs` 借用带到后续逻辑

### 2.2 Python 参考数据已经接入 Rust 单测

已经补齐并验证的参考对齐包括：

- `audio::mel`
  - Hann window 与 Python 对齐
  - mel filterbank 与 Python 对齐
  - 第一帧 power spectrum 与 Python 对齐
  - 整段 `log_mel_pre_norm` 和 `full_mel` 已有回归测试
- `nlu::parser`
  - `parse_nlu_output()` 与导出的 `cases.json` 对齐
  - Python `predict_*` 导出的模型 raw output 已整理为 `predict_cases.json`，并用于 parser 参考测试
- `asr::engine`
  - prompt prefix token ids 与 Python 参考对齐
  - prompt suffix token ids 与 Python 参考对齐
  - `embed_tokens.bin` 的 token embedding 读取与 Python 参考对齐
  - `clean_text` 特殊标记清理逻辑已锁定
- `nlu::engine`
  - encoder 输入构造的 `input_ids` / `attention_mask` 与 Python 参考对齐
  - encoder `hidden_head` 参考测试已补充，但因为依赖 Rust ORT 动态库，默认标记为 `ignored`
  - Windows 本地使用 ONNX Runtime 1.19.2 已手动跑通 `predict()` 端到端专项测试

这些测试资源当前位于：

- `voice-infer/tests/resources/reference/mel`
- `voice-infer/tests/resources/reference/parse`
- `voice-infer/tests/resources/reference/asr`
- `voice-infer/tests/resources/reference/nlu`

## 3. 当前还没完全收口的地方

### 3.1 Mel 还有小的数值残差

虽然 mel 前端已经非常接近 Python，但严格对齐目标还没有完全达到。

当前保留了两条 `ignored` 测试，原因如下：

- `log_mel_pre_norm` 当前最大绝对误差约为 `8.1e-4`
- 最终 `full_mel` 当前最大绝对误差约为 `1.8e-4`

此外，ORT 驱动的中间张量对齐测试当前也保留为 `ignored`：

- encoder `audio_features_head`
- `decoder_init` 最后一位 logits
- NLU encoder `hidden_head`
- NLU `predict_forward` 端到端

它们不是逻辑未实现，而是当前 Windows 环境下 Rust 侧 ONNX Runtime 动态库没有收口。现有 `.venv` 中的 Python `onnxruntime` 是 `1.23.2`，而 Rust 重写文档约束的是 `libonnxruntime 1.19.2`。用 `.venv` 的 DLL 跑 Rust ORT session 会超时，所以这些测试现在只认显式设置的 `ORT_DYLIB_PATH`，准备好 Rust 目标版本 ORT 动态库后再专项运行。

2026-06-11 更新：Windows 本地已下载并使用 `onnxruntime-win-x64-1.19.2` 完成下面的手动验证：

- NLU `test_encoder_hidden_head_matches_reference` 通过
- NLU `test_predict_forward_matches_reference` 通过
- Rust HTTP `/nlu` 返回 `move_forward`
- ASR `test_encoder_audio_features_head_matches_reference` 通过
- Rust HTTP `/asr` 对 `tests/fixtures/test_1s.wav` 返回 `odicologyThe.`

仍未通过的是 ASR `decoder_init` logits 严格数值参考测试，当前最大误差约 `1.27`。不过该测试音频的 ASR 端到端文本已经与 Python 参考一致。

这说明：

- Hann window 不是问题
- mel filterbank 不是问题
- 单帧 FFT / power 的基本路径也不是问题
- 剩余偏差更像是多帧累计过程中的细小数值差异

所以 mel 现在不再是“大错”，但也还没到清单里更严格的 `1e-5` 级别。

### 3.2 真实模型推理一致性还没有做完

还没有完成的核心验收包括：

- ASR encoder 输出参考对齐
- ASR `decoder_init` logits 对齐
- ASR `decoder_step` / embedding lookup 对齐
- NLU `predict()` 级别的整链路输出对齐

换句话说，现在已经把“模型之前的输入构造”和“模型之后的一部分解析逻辑”锁住了，但“模型中间张量”和“最终端到端输出”还没有全部锁住。

### 3.3 HTTP 联调还没有验收

最终还需要确认：

- Rust 服务可以实际启动
- `/health`、`/asr`、`/nlu` 都能正常响应
- 现有 `pipeline` 客户端接到 Rust 服务时不需要额外改协议

## 4. 当前测试结果

当前在 `voice-infer/` 目录执行：

```powershell
cargo test
```

结果为：

- `33 passed`
- `0 failed`
- `6 ignored`

其中被忽略的 6 条测试包括：

- 2 条 mel 严格 Python 数值对齐测试
- 2 条 ASR ORT 中间张量专项对齐测试
- 1 条 NLU ORT 中间张量专项对齐测试
- 1 条 NLU ORT 端到端专项测试

## 5. 下一步最值得继续做什么

接下来最合适的推进顺序是：

1. 在 Python 导出脚本里继续补 ASR / NLU 中间张量参考值
2. 先补 ASR `audio_features_head`、`init_logits_last`、embedding lookup 的 Rust 对齐测试
3. 再补 NLU `predict()` 级别的参考输出测试
4. 最后启动 Rust HTTP 服务做联调验证

原因很简单：

- 前处理和 tokenizer 这一层已经基本锁住
- 继续追 mel 的 `1e-4` 小残差，收益暂时不如推进 ASR/NLU 中间张量对齐
- 只有把模型推理链路逐层锁住，后面的 HTTP 联调才不会陷入“接口看起来通了，但结果不对”的排查

## 6. 代码位置

Rust 重写主体目前在这里：

- `voice-infer/`：Rust 服务与推理实现
- `docs/rust-rewrite-*.md`：重写方案、清单和状态文档
- `docs/ubuntu2204-rust-deploy.md`：Ubuntu 22.04 机器狗/上位机部署说明
- `docs/windows-rust-e2e.md`：Windows 本地 Rust ONNX 端到端验证说明
- `scripts/package_voice_infer_linux.sh`：Linux 目标机打包脚本，自动下载匹配架构的 ONNX Runtime 1.19.2
- `scripts/start_voice_infer_windows.ps1`：Windows 本地启动 Rust 推理服务
- `scripts/export_reference.py`：从 Python 导出参考数据
- `tests/fixtures/reference/`：Python 导出的原始参考文件
