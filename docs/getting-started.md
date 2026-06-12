# Getting Started

这份文档面向第一次拿到代码的人，目标是让对方知道需要准备什么、模型放哪里、如何在 Windows 打包，以及如何把服务部署到 Ubuntu 22.04 真机。

如果你已经准备上线或需要长期运行服务，请继续看 [真机部署指南](robot-production-deploy.md)。

## 1. 你需要准备什么

### Windows 开发/打包机

用于开发、测试、交叉编译和生成部署包。

必需：
- Git
- Python 3.10
- Rust toolchain
- Zig
- `cargo-zigbuild`

安装 Rust：
```powershell
winget install -e --id Rustlang.Rustup
```

安装 Zig：
```powershell
winget install -e --id zig.zig
```

安装 cargo-zigbuild：
```powershell
cargo install cargo-zigbuild
```

验证：
```powershell
cargo --version
zig version
cargo zigbuild --help
```

如果 `zig version` 找不到，但你确认用 winget 安装过 Zig，重新打开 PowerShell。项目脚本也会自动尝试把 WinGet Links 加入当前进程 PATH。

### Ubuntu 22.04 真机

可以是机器狗本机，也可以是同网段上位机。

基础依赖：
```bash
sudo apt-get update
sudo apt-get install -y python3 python3-venv python3-pip curl tar ca-certificates
```

如果需要在真机本地编译 Rust，再安装：
```bash
sudo apt-get install -y build-essential git
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env
```

确认架构：
```bash
uname -m
```

常见结果：
- `aarch64`：机器狗/ARM Ubuntu，Windows 打包时用 `-Arch aarch64`。
- `x86_64`：普通 PC/工控机 Ubuntu，Windows 打包时用 `-Arch x86_64`。

## 2. 拉取代码

```bash
git clone <repo-url> VoiceControl
cd VoiceControl
```

Windows 上同样可以用 Git clone 到本地，例如：
```powershell
git clone <repo-url> E:\HuaJianCode\VoiceControl
cd E:\HuaJianCode\VoiceControl
```

## 3. 放置模型

模型文件不在 Git 仓库里，需要单独放到 `models/`。

默认结构：
```text
models/
├── asr/
│   ├── encoder.int4.onnx
│   ├── decoder*
│   ├── decoder_weights.int4.data
│   ├── embed_tokens.bin
│   └── tokenizer.json
├── nlu/
│   ├── encoder.onnx
│   ├── decoder.onnx
│   └── tokenizer/
└── kws/
    └── sherpa-onnx wake word models
```

至少需要：

- Rust/Python ASR：`models/asr`
- Rust/Python NLU：`models/nlu`
- 如果使用 KWS 唤醒：`models/kws`

`models/` 已被 `.gitignore` 忽略，不要提交。

## 4. 配置文件

默认配置是 `config.yaml`。
建议复制一份本地配置：

```bash
cp config.yaml config.local.yaml
```

重点字段：
```yaml
inference:
  backend: rust     # rust / mixed / python / external

audio:
  source: onboard   # onboard / webrtc / hardware_serial

robot:
  ip: 10.10.20.66
  connection_method: LocalSTA

command:
  service_url: http://127.0.0.1:8090/api/v1/local/motion
```

后端含义：
- `rust`：`run.py` 启动 Rust `voice-infer`，Python 只跑 pipeline。推荐。
- `mixed`：Python ASR + Rust NLU。如果 Rust ASR 在目标设备崩溃，退回此模式。
- `python`：`run.py` 启动 Python ASR/NLU，无需 Rust 二进制。
- `external`：`run.py` 不启动推理服务，只连接 `services.asr_url` 和 `services.nlu_url`。

## 5. Windows 一键生成真机部署包

默认目标是 `aarch64 Ubuntu 22.04`，适合机器狗 ARM 真机：

```powershell
scripts\build_robot_deploy_bundle.bat
```

生成目录：
```text
dist/robot-deploy-aarch64-<timestamp>/
```

如果真机是 x86_64：
```powershell
scripts\build_robot_deploy_bundle.bat -Arch x86_64
```

如果不想把模型打进部署包：
```powershell
scripts\build_robot_deploy_bundle.bat -NoModels
```

如果需要 `mixed` 后端（Rust ASR 在目标设备有问题时的降级方案）：
```powershell
scripts\build_robot_deploy_bundle.bat -Backend mixed
```

脚本会做这些事情：
- 交叉编译 `voice-infer` Linux 二进制。
- 下载匹配架构的 ONNX Runtime 1.23.0。
- 收集 Python pipeline 代码、配置、脚本和文档。
- 可选复制 `models/`。
- 输出一个可以直接拷贝到真机的部署目录。

## 6. 只交叉编译 Rust 服务

如果你只想生成 Rust 推理服务包：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\cross_build_voice_infer_linux.ps1 -Arch aarch64
```

输出：
```text
dist/voice-infer-ubuntu2204-aarch64/
```

这个 Rust 包默认不包含模型。完整部署包使用根目录 `models/` 保存唯一一份模型，真机安装脚本会为 Rust 创建软链接。

真机运行：
```bash
cd voice-infer-ubuntu2204-aarch64
chmod +x voice-infer start.sh verify.sh
./start.sh
```

另开一个终端验证：

```bash
./verify.sh
```

## 7. 拷贝到真机并启动

把 Windows 生成的部署包复制到真机，例如：
```bash
scp -r dist/robot-deploy-aarch64-<timestamp> user@<robot-ip>:~/VoiceControlDeploy
```

在真机上：
```bash
cd ~/VoiceControlDeploy
chmod +x scripts/install_robot_target.sh
scripts/install_robot_target.sh --mode onboard
```

安装完成后启动：

```bash
./start_robot.sh
```

其他音频源：

```bash
scripts/install_robot_target.sh --mode webrtc
scripts/install_robot_target.sh --mode hardware_serial
```

## 8. 手动分开启动

如果你想先单独验证 Rust 服务：
```bash
cd rust
chmod +x voice-infer start.sh verify.sh
./start.sh
```

另开终端：
```bash
cd rust
./verify.sh
```

然后回到部署包根目录，只启动 pipeline：
```bash
export ASR_URL=http://127.0.0.1:8000/asr
export NLU_URL=http://127.0.0.1:8001/nlu
python3 run.py --pipeline-only --onboard
```

## 9. Python 开发模式

如果只是本地开发，不部署 Rust：
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
python run.py --onboard
```

Windows PowerShell：
```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -e .
python run.py --onboard
```

## 10. 测试

Python 测试：
```bash
pytest
```

Rust 测试：
```bash
cd voice-infer
cargo test
```

ONNX Runtime 真实推理测试需要设置 `ORT_DYLIB_PATH`，并显式运行 ignored tests。

## 11. 常见问题

### Windows 找不到 zig

先确认：

```powershell
zig version
```

如果失败，重新打开 PowerShell。若仍失败，检查：

```powershell
where.exe zig
```

winget 通常会把 Zig 链接放在：
```text
%LOCALAPPDATA%\Microsoft\WinGet\Links
```

项目脚本会自动把这个目录加入当前进程 PATH。

### 交叉编译失败

先确认：

```powershell
cargo --version
zig version
cargo zigbuild --help
rustup target list --installed
```

如果缺 target，脚本会自动执行：
```powershell
rustup target add aarch64-unknown-linux-gnu
```

### 真机运行提示模型不存在

确认部署包里存在：
```text
models/asr
models/nlu
```

如果打包时用了 `-NoModels`，需要手动把模型拷过去。

### Rust ASR 在 aarch64 真机崩溃

已知部分 aarch64 设备上 ORT 创建 INT4 ONNX 会话时崩溃（`Unknown CPU vendor` / `std::vector assertion`）。

降级方案：编辑 `config.deploy.yaml`，将 `backend: rust` 改为 `backend: mixed`，这样 ASR 用 Python、NLU 用 Rust。

或者重新打包：
```powershell
scripts\build_robot_deploy_bundle.bat -Backend mixed
```

### Rust 服务启动了，但 pipeline 连不上

检查：

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8001/health
```

如果 Rust 服务跑在另一台机器，把 `ASR_URL` 和 `NLU_URL` 指向那台机器的 IP。

### WebRTC 连不上机器狗

检查 `config.local.yaml`：
```yaml
robot:
  ip: <robot-ip>
  connection_method: LocalSTA
```

再做预检：
```bash
python3 run.py --config config.local.yaml --preflight-only --webrtc
```

## 12. 提交代码前

不要提交：
- `models/`
- `.venv/`
- `third_party/`
- `voice-infer/target/`
- `dist/`
- `output/`
- `__pycache__/`

这些已经在 `.gitignore` 中处理。
