#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo "=== [1/4] 安装系统依赖 ==="
sudo apt-get install -y ffmpeg libsndfile1

echo "=== [2/4] 安装 uv（Python 包管理器）==="
if ! command -v uv &>/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

echo "=== [3/4] 安装 Python 依赖 ==="
uv python find 3.8 2>/dev/null || echo "警告：未找到 Python 3.8，uv 将自动下载"
uv sync

echo "=== [4/4] 验证依赖 ==="
uv run python -c "
import onnxruntime as ort, soundfile
providers = ort.get_available_providers()
print('Python 版本:', __import__('sys').version)
print('onnxruntime 版本:', ort.__version__)
print('可用推理后端:', providers)
print('依赖正常 ✓')
"

echo ""
echo "部署完成！"
echo ""
echo "=== 模型文件放置 ==="
echo "  models/asr/   → model_q8.onnx + tokens.txt"
echo "  models/nlu/   → encoder.onnx + decoder.onnx"
echo "  models/kws/   → sherpa-onnx 唤醒词模型"
echo ""
echo "=== 运行方式 ==="
echo "  一键启动:     python run.py"
echo "  本机麦克风:   python run.py --onboard"
echo "  仅 ASR 服务:  python run.py --serve-asr"
echo "  仅 NLU 服务:  python run.py --serve-nlu"
echo "  GPU 推理:     python run.py --gpu"
