# Rust 重写开工文档

> 这份文档给“现在就开始做”的开发者使用。
> 不重复总体方案，直接回答三件事：
> 1. 先做什么
> 2. 做到什么程度算完成
> 3. 遇到分歧时以什么为准

---

## 1. 文档关系

- `rust-rewrite-plan.md`
  - 说明整体设计、模块划分、关键实现思路。
- `rust-rewrite-checklist.md`
  - 说明完整实施清单、测试项和部署项。
- `rust-rewrite-kickoff.md`（本文）
  - 说明开工顺序、第一阶段目标、每日里程碑和决策基线。

如果三者冲突，以以下优先级为准：

1. 当前 Python 代码真实行为
2. `rust-rewrite-plan.md`
3. `rust-rewrite-checklist.md`
4. 本文的执行建议

原因很简单：Rust 版目标不是“长得像”，而是“行为与现有服务一致”。

---

## 2. 这次重写的硬目标

Rust 版需要满足以下目标，缺一项都不能算完成：

- 对外仍提供两个 HTTP 服务：
  - `POST /asr`
  - `POST /nlu`
- 返回 JSON 结构与当前 Python 服务兼容
- `pipeline/asr_client.py` 和 `pipeline/nlu_client.py` 无需改动即可接入
- ASR 文本输出与 Python 版对齐
- NLU `intent/slots/raw` 语义与 Python 版对齐
- 部署时只需要：
  - Rust 二进制
  - `libonnxruntime.so`
  - 现有模型目录

明确不在第一阶段处理的内容：

- UI
- 新功能
- 新模型
- 新接口
- 命令词优化
- pipeline 重构

---

## 3. 开工前先冻结“真值来源”

Rust 重写期间，以下 Python 文件视为行为真值：

- `asr/engine.py`
- `asr/server.py`
- `nlu/engine.py`
- `nlu/server.py`
- `pipeline/asr_client.py`
- `pipeline/nlu_client.py`
- `tests/test_asr_engine.py`
- `tests/test_nlu_engine.py`

其中最关键的是：

- ASR 真值：
  - `_log_mel_fast()`
  - `_build_prompt()`
  - `_clean_text()`
  - `transcribe()`
- NLU 真值：
  - `predict()`
  - `parse_nlu_output()`
  - `_command_type_to_intent()`
  - `_infer_move_intent()`

规则：

- 文档写法和 Python 行为不一致时，以 Python 为准。
- Rust 实现过程中不要“顺手优化逻辑”，先做行为等价。
- 先保兼容，再谈性能。

---

## 4. 建议的开工顺序

不要一上来就写完整服务。正确顺序是：

1. 建 Rust 工程骨架
2. 导出 Python 参考数据
3. 先做音频前处理 `mel.rs`
4. 再做 `wav.rs`
5. 再做 ASR engine
6. 再做 NLU engine
7. 最后接 HTTP server 和 `main.rs`

原因：

- `mel.rs` 是 ASR 中最容易“看起来没错但结果不一样”的部分。
- 如果不先把 mel 对齐，后面所有 encoder/decoder 调试都会变成噪声。
- NLU 比 ASR 简单，放在 ASR 跑通后做更稳。

---

## 5. 第一天的完成标准

第一天不要追求服务跑通。第一天的目标只有一个：

**把音频前处理做成“可验证、可对齐、可复用”的 Rust 模块。**

当以下条件全部满足，Day 1 才算完成：

- `voice-infer` Rust 工程已创建
- `cargo check` 能通过
- Python 参考数据已导出
- `src/audio/mel.rs` 已实现
- `src/audio/wav.rs` 已实现基础版本
- Rust 生成的 mel 与 Python 参考值误差 `< 1e-5`
- 至少有一组自动化测试覆盖 mel 对齐

如果 mel 没对齐，不要进入 ASR session 调试。

---

## 6. 立即开工步骤

### Step 0: 创建工程

建议目录：

```text
voice-infer/
├── Cargo.toml
├── src/
│   ├── main.rs
│   ├── config.rs
│   ├── audio/
│   ├── asr/
│   └── nlu/
└── tests/
```

要求：

- Rust 工程与现有 Python 工程并存
- 不要覆盖当前仓库的 Python 目录
- 模型目录继续复用现有 `models/asr` 和 `models/nlu`

### Step 1: 先写参考导出脚本

先在 Python 侧导出参考值，再写 Rust。

最低限度必须导出的文件：

- `ref_mel.npy`
- `ref_hann_window.npy`
- `ref_mel_basis.npy`
- `ref_power_frame0.npy`
- `ref_asr_result.json`
- `ref_nlu_result.json`
- `ref_parse_cases.json`

原因：

- 没有参考值时，Rust 调试只能靠猜。
- ONNX 推理链路很长，必须分层验证。

### Step 2: 实现 `mel.rs`

实现顺序不要乱：

1. Hann window
2. Mel filterbank
3. FFT power spectrum
4. `log_mel()`

每完成一个子步骤就做一次数值对比，不要等全部写完再一起查错。

### Step 3: 实现 `wav.rs`

只做当前 Python 真实需要的能力：

- 读取 WAV
- 多声道转 mono
- 重采样到 16k
- 输出 `Vec<f32>`

先不追求支持所有音频格式。HTTP `multipart` 输入实际上传的是 WAV。

### Step 4: 只做 ASR engine，不接 server

先写一个本地测试：

- 输入：测试 WAV
- 输出：`transcribe()` 结果
- 对比：Python 的 `text`

只有本地引擎对齐后，再写 HTTP 层。

---

## 7. 第二天建议目标

第二天目标是：

**让 Rust 版 ASR 和 NLU 在本地函数级跑通。**

完成标准：

- `AsrEngine::new()` 成功加载模型
- `AsrEngine::transcribe()` 返回正确文本
- `NluEngine::new()` 成功加载模型
- `NluEngine::predict()` 返回兼容的 `intent/slots/raw`
- 至少覆盖：
  - `"向前走三步"`
  - `"停止"`
  - `"坐下"`

第二天仍然不是部署日。不要太早进入交叉编译和打包。

---

## 8. 第三天建议目标

第三天再做服务和部署：

- 接入 `axum`
- 提供 `/asr`、`/nlu`、`/health`
- 用 `curl` 验证接口
- 用现有 `pipeline` 验证兼容性
- 再做 release build、strip、部署目录整理

第三天的最终验收标准：

- `pipeline-only` 模式下，客户端地址指向 Rust 服务，整条链路可跑通

---

## 9. 实现时必须锁死的兼容点

这些行为不能“凭感觉改”：

### 9.1 ASR

- 采样率必须是 `16000`
- 空音频时要填 0.1 秒静音
- Hann 窗必须用 periodic 公式
- mel 滤波器必须是 `slaney`
- prompt 文本必须与 Python 完全一致
- `audio_pad_token_id` 的填充长度必须等于 `audio_features.shape[1]`
- `decode(..., skip_special_tokens=true)` 后还要再做 `clean_text`
- 输出结构必须包含：
  - `text`
  - `feat_ms`
  - `infer_ms`
  - `total_ms`
  - `segments`

### 9.2 NLU

- 输入前缀必须是 `指令解析: `
- `max_input=64`
- `max_output=128`
- 先 `tokenize + padding + truncation`
- `parse_nlu_output()` 的兼容行为必须完整保留
- `Move` 需要通过 `vx/vy/vyaw` 推断真实 intent
- `type=chat` 不能当成可执行命令

### 9.3 HTTP

- ASR 请求格式必须是 `multipart/form-data`
- NLU 请求格式必须是 JSON body
- 异常时仍返回兼容形状的 JSON

---

## 10. 常见错误路线

以下做法看起来省事，实际会拖慢进度：

- 先写完整 server，再回头补 engine
- 直接对着文档写，不对照 Python 真值
- 不导出参考数据，靠端到端结果调试
- mel 还没对齐就去怀疑 ONNX Runtime
- 提前做 GPU、交叉编译、部署脚本
- 一开始就追求抽象优雅，把模块拆太细

建议策略是：

- 先跑通
- 再对齐
- 再整理
- 最后优化

---

## 11. 每个阶段的“停线条件”

出现以下情况，不要继续往后写，要先停下来修正：

### Phase A: mel

- mel 与 Python 误差持续大于 `1e-5`
- 帧数对不上
- 频谱归一化后整体偏移明显

### Phase B: ASR

- encoder 输出 shape 不一致
- `audio_offset` 错误
- 第一步 `decoder_init` logits 与 Python 完全跑偏

### Phase C: NLU

- tokenizer padding 长度不一致
- decode 出来的 raw 文本和 Python 版风格完全不同
- `parse_nlu_output()` 的映射结果不兼容当前 dispatcher

### Phase D: HTTP

- `pipeline/asr_client.py` 无法直接调用
- `pipeline/nlu_client.py` 无法直接调用

停线后先做最小化定位，不要带着未知偏差继续叠代码。

---

## 12. 推荐的最小里程碑

按下面四个里程碑推进最稳：

### M0: 工程可编译

- `cargo check` 通过
- 目录骨架建立

### M1: 音频前处理对齐

- `wav.rs` + `mel.rs` 完成
- mel 数值对齐

### M2: 本地推理对齐

- ASR 本地函数调用成功
- NLU 本地函数调用成功

### M3: 服务兼容

- HTTP 接口跑通
- pipeline 零修改接入

在没有完成 M1 前，不要声称“Rust 重写已经开始稳定推进”。

---

## 13. 建议的提交节奏

建议不要一个大提交做完。按里程碑拆：

1. `chore: scaffold voice-infer crate and configs`
2. `feat: implement audio wav loading and mel frontend`
3. `feat: add qwen3 asr inference engine`
4. `feat: add mengzi t5 nlu inference and parser`
5. `feat: add axum http services and integration tests`

这样后面回滚、对比和 review 都容易。

---

## 14. 一句话执行策略

**先拿 Python 当真值，先把 mel 做准，再把 ASR/NLU 做通，最后再接 HTTP 和部署。**

这条顺序不要改。
