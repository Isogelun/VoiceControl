# VoiceControl

闈㈠悜瀹囨爲 Go2 鏈哄櫒鐙楃殑璇煶鎺у埗绯荤粺銆?
```text
鍞ら啋璇?-> VAD 璇煶鍒囧垎 -> ASR 璇煶璇嗗埆 -> NLU 鎰忓浘鐞嗚В -> 鎸囦护 JSON -> 鏈哄櫒鐙楀姩浣?璇煶鍙嶉
```

椤圭洰鐜板湪淇濈暀涓ゅ鎺ㄧ悊鍚庣锛?
- `python`锛氬師濮?Python ASR/NLU 鏈嶅姟锛岄€傚悎寮€鍙戣皟璇曘€?- `rust`锛歚voice-infer/` Rust ASR/NLU 鏈嶅姟锛岄€傚悎 Ubuntu 22.04 鐪熸満閮ㄧ讲銆?
绗竴娆℃嬁鍒颁唬鐮侊紝璇峰厛鐪嬶細

- [docs/getting-started.md](docs/getting-started.md)锛氱幆澧冨噯澶囥€佹ā鍨嬫斁缃€乄indows 鎵撳寘銆佺湡鏈洪儴缃插畬鏁存祦绋嬨€?- [docs/robot-production-deploy.md](docs/robot-production-deploy.md)锛歎buntu 22.04 鐪熸満鐢熶骇閮ㄧ讲銆乻ystemd銆佹棩蹇楀拰鏁呴殰鎺掓煡銆?- [docs/ubuntu2204-rust-deploy.md](docs/ubuntu2204-rust-deploy.md)锛歎buntu 22.04 Rust 鏈嶅姟閮ㄧ讲缁嗚妭銆?- [voice-infer/README.md](voice-infer/README.md)锛歊ust 鎺ㄧ悊鏈嶅姟鎺ュ彛鍜屾祴璇曡鏄庛€?
## 褰撳墠鐘舵€?
- Python 鍏ㄩ摼璺粛鍙繍琛屻€?- Rust `voice-infer` 榛樿娴嬭瘯閫氳繃锛岀湡瀹?ONNX Runtime 鎺ㄧ悊宸插湪 Windows 鏈湴鍜?aarch64 浜ゅ弶缂栬瘧璺緞楠岃瘉銆?- Windows 鍙€氳繃 `scripts/build_robot_deploy_bundle.bat` 涓€閿敓鎴愮湡鏈洪儴缃插寘銆?- 澶фā鍨嬨€丱NNX Runtime 涓嬭浇鍖呫€佽櫄鎷熺幆澧冨拰缂栬瘧浜х墿涓嶆彁浜ゅ埌 Git銆?
## 鐩綍缁撴瀯

```text
VoiceControl/
鈹溾攢鈹€ run.py                         # 缁熶竴鍚姩鍏ュ彛
鈹溾攢鈹€ config.yaml                    # 榛樿閰嶇疆
鈹溾攢鈹€ asr/                           # Python ASR
鈹溾攢鈹€ nlu/                           # Python NLU
鈹溾攢鈹€ pipeline/                      # 鍞ら啋銆乂AD銆丄SR/NLU 瀹㈡埛绔€佸姩浣滃垎鍙?鈹溾攢鈹€ unitree_webrtc_connect/        # Unitree Go2 WebRTC 鎺ュ叆
鈹溾攢鈹€ voice-infer/                   # Rust ASR/NLU 鎺ㄧ悊鏈嶅姟
鈹溾攢鈹€ scripts/                       # 鎵撳寘銆侀儴缃层€侀獙璇佽剼鏈?鈹溾攢鈹€ docs/                          # 椤圭洰鏂囨。
鈹溾攢鈹€ tests/                         # 娴嬭瘯涓庡弬鑰冩暟鎹?鈹溾攢鈹€ audio/                         # 鍙嶉闊抽
鈹斺攢鈹€ models/                        # 鏈湴妯″瀷鐩綍锛屼笉鍏ュ簱
```

## 蹇€熼儴缃?
Windows 寮€鍙戞満涓婂畨瑁?Rust銆乑ig銆乣cargo-zigbuild` 鍚庯紝鐩存帴杩愯锛?
```powershell
scripts\build_robot_deploy_bundle.bat
```

榛樿鐢熸垚 `aarch64 Ubuntu 22.04` 鐪熸満閮ㄧ讲鍖咃細

```text
dist/robot-deploy-aarch64-<timestamp>/
```

鎶婅鐩綍鎷峰埌鏈哄櫒鐙楁垨 Ubuntu 22.04 閮ㄧ讲鏈哄悗杩愯锛?
```bash
cd robot-deploy-aarch64-*
chmod +x scripts/install_robot_target.sh
scripts/install_robot_target.sh --mode webrtc
```

瀹夎瀹屾垚鍚庝細鐢熸垚 `start_robot.sh`锛?
```bash
./start_robot.sh
```

濡傛灉瑕佸畨瑁呭埌 `/opt/voice-control` 骞舵敞鍐?systemd锛?
```bash
sudo -E scripts/install_robot_target.sh --install-dir /opt/voice-control --mode webrtc --systemd
```

涔熷彲浠ヤ笉瀹夎锛岀洿鎺ヨ繍琛?Python 鎵樼 Rust 鐨勫叆鍙ｏ細

```bash
chmod +x start_python_managed_rust.sh
./start_python_managed_rust.sh --webrtc
```

濡傛灉浣跨敤鏈満楹﹀厠椋庯細

```bash
./start_python_managed_rust.sh --onboard
```

濡傛灉浣跨敤涓插彛楹﹀厠椋庨樀鍒楋細

```bash
./start_python_managed_rust.sh --hardware-serial
```

鏇村鍑嗗姝ラ鍜屾帓閿欒 [docs/getting-started.md](docs/getting-started.md)銆?
## 鎺ㄧ悊鍚庣閫夋嫨

`config.yaml` 涓娇鐢?`inference.backend` 鍒囨崲锛?
```yaml
inference:
  backend: mixed    # python / rust / mixed / external
```

- `python`锛歚run.py` 鍚姩 Python ASR/NLU 鏈嶅姟銆?- `rust`锛歚run.py` 鍚姩 Rust `voice-infer` 鏈嶅姟銆?- `mixed`锛歅ython ASR + Rust NLU銆傚綋鍓?Unitree aarch64 鐪熸満鎺ㄨ崘璇ユā寮忥紝鍥犱负 Rust ASR 鍔犺浇 ASR INT4 ONNX 鏃朵細瑙﹀彂 ONNX Runtime aarch64 宕╂簝銆?- `external`锛歚run.py` 涓嶅惎鍔ㄦ帹鐞嗘湇鍔★紝鍙繛鎺?`services.asr_url` 鍜?`services.nlu_url`锛岄€傚悎 systemd 鍗曠嫭鎵樼 Rust 鏈嶅姟銆?
## 妯″瀷鐩綍

妯″瀷鏂囦欢闇€瑕佹湰鍦版斁缃紝榛樿缁撴瀯锛?
```text
models/
鈹溾攢鈹€ asr/
鈹?  鈹溾攢鈹€ encoder.int4.onnx
鈹?  鈹溾攢鈹€ decoder*
鈹?  鈹溾攢鈹€ decoder_weights.int4.data
鈹?  鈹溾攢鈹€ embed_tokens.bin
鈹?  鈹斺攢鈹€ tokenizer.json
鈹溾攢鈹€ nlu/
鈹?  鈹溾攢鈹€ encoder.onnx
鈹?  鈹溾攢鈹€ decoder.onnx
鈹?  鈹斺攢鈹€ tokenizer/
鈹斺攢鈹€ kws/
    鈹斺攢鈹€ sherpa-onnx wake word models
```

`models/` 浣撶Н寰堝ぇ锛屽凡琚?`.gitignore` 蹇界暐銆?
## 甯哥敤鍛戒护

Python 寮€鍙戠幆澧冿細

```bash
pip install -e .
python run.py --onboard
```

鍙惎鍔?Python 鎺ㄧ悊鏈嶅姟锛?
```bash
python run.py --serve-asr
python run.py --serve-nlu
```

Rust 鏈湴娴嬭瘯锛?
```bash
cd voice-infer
cargo test
```

Windows 鐢熸垚鐪熸満閮ㄧ讲鍖咃細

```powershell
scripts\build_robot_deploy_bundle.bat
```

Windows 浠呬氦鍙夌紪璇?Rust aarch64 鍖咃紝涓嶅寘鍚ā鍨嬶細

```powershell
powershell -ExecutionPolicy Bypass -File scripts\cross_build_voice_infer_linux.ps1 -Arch aarch64
```

## 涓嶈鎻愪氦鐨勫唴瀹?
- `models/`
- `.venv/`
- `third_party/`
- `voice-infer/target/`
- `dist/`
- `output/`
- `__pycache__/`

濡傛灉闇€瑕佸垎鍙戞ā鍨嬶紝浣跨敤 GitHub Releases銆佸璞″瓨鍌ㄦ垨鍗曠嫭涓嬭浇鑴氭湰锛屼笉瑕佺洿鎺ユ斁杩?Git 浠撳簱銆?
