#!/usr/bin/env bash
set -euo pipefail

INSTALL_DIR=""
MODE="webrtc"
SYSTEMD=0
SKIP_PIP=0
PYTHON_BIN="${PYTHON_BIN:-python3}"
SERVICE_USER="${SERVICE_USER:-}"

usage() {
  cat <<'EOF'
Usage: scripts/install_robot_target.sh [options]

Run this on the Ubuntu 22.04 target from the deploy bundle root.

Options:
  --install-dir PATH   Install/copy bundle to PATH, e.g. /opt/voice-control.
                       If omitted, deploy in the current directory.
  --mode MODE          Pipeline mode: webrtc, onboard, hardware_serial. Default: webrtc.
  --systemd            Install and enable systemd services.
  --skip-pip           Do not install Python requirements.
  -h, --help           Show this help.

Environment:
  PYTHON_BIN           Python executable. Default: python3.
  SERVICE_USER         systemd service user. Default: current user.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --install-dir)
      INSTALL_DIR="${2:-}"
      shift 2
      ;;
    --mode)
      MODE="${2:-}"
      shift 2
      ;;
    --systemd)
      SYSTEMD=1
      shift
      ;;
    --skip-pip)
      SKIP_PIP=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

case "$MODE" in
  webrtc|onboard|hardware_serial) ;;
  *)
    echo "--mode must be one of: webrtc, onboard, hardware_serial" >&2
    exit 2
    ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [[ -n "$INSTALL_DIR" ]]; then
  TARGET_ROOT="$INSTALL_DIR"
  mkdir -p "$TARGET_ROOT"
  rsync -a --delete \
    --exclude '.venv' \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    "$SOURCE_ROOT/" "$TARGET_ROOT/"
else
  TARGET_ROOT="$SOURCE_ROOT"
fi

cd "$TARGET_ROOT"

if [[ -z "$SERVICE_USER" ]]; then
  SERVICE_USER="${SUDO_USER:-$(id -un)}"
fi

echo "Deploy root: $TARGET_ROOT"
echo "Pipeline mode: $MODE"
echo "Service user: $SERVICE_USER"

if [[ ! -x rust/voice-infer ]]; then
  echo "Missing executable rust/voice-infer. Build/copy the Rust package first." >&2
  exit 1
fi

if ! compgen -G "rust/libonnxruntime.so.*" >/dev/null; then
  echo "Missing rust/libonnxruntime.so.*." >&2
  exit 1
fi
ORT_LIB="$(basename "$(ls rust/libonnxruntime.so.* | sort -V | tail -n 1)")"

if [[ ! -d models/asr && ! -d rust/models/asr ]]; then
  echo "Missing models/asr or rust/models/asr." >&2
  exit 1
fi

if [[ ! -d models/nlu && ! -d rust/models/nlu ]]; then
  echo "Missing models/nlu or rust/models/nlu." >&2
  exit 1
fi

mkdir -p rust/models
if [[ ! -e rust/models/asr && ! -L rust/models/asr && -d models/asr ]]; then
  ln -s ../../models/asr rust/models/asr
elif [[ -L rust/models/asr && ! -e rust/models/asr && -d models/asr ]]; then
  ln -sf ../../models/asr rust/models/asr
fi
if [[ ! -e rust/models/nlu && ! -L rust/models/nlu && -d models/nlu ]]; then
  ln -s ../../models/nlu rust/models/nlu
elif [[ -L rust/models/nlu && ! -e rust/models/nlu && -d models/nlu ]]; then
  ln -sf ../../models/nlu rust/models/nlu
fi

chmod +x rust/voice-infer rust/start.sh rust/verify.sh 2>/dev/null || true
chmod +x start_python_managed_rust.sh start_rust_and_pipeline.sh 2>/dev/null || true

if [[ ! -f config.deploy.yaml ]]; then
  cp config.yaml config.deploy.yaml
fi

if ! grep -q '^inference:' config.deploy.yaml; then
  cat > config.deploy.yaml.tmp <<EOF
inference:
  backend: rust
  rust_binary: rust/voice-infer
  rust_ort_dylib: rust/$ORT_LIB
  rust_ort_opt: disable
  rust_asr_max_new_tokens: 16
  rust_threads: 1

EOF
  cat config.deploy.yaml >> config.deploy.yaml.tmp
  mv config.deploy.yaml.tmp config.deploy.yaml
fi

USE_UV=0
if command -v uv &>/dev/null; then
  USE_UV=1
  echo "Detected uv, will use it for venv and package install."
else
  echo "uv not found, using pip."
fi

if [[ ! -d .venv ]]; then
  if [[ "$USE_UV" == "1" ]]; then
    uv venv .venv --python "$PYTHON_BIN"
  else
    "$PYTHON_BIN" -m venv .venv
  fi
fi

source .venv/bin/activate
if [[ "$SKIP_PIP" != "1" ]]; then
  if [[ "$USE_UV" == "1" ]]; then
    uv pip install -r requirements-robot.txt
    if grep -q 'backend: mixed\|backend: python' config.deploy.yaml && [[ -f requirements-server-py38.txt ]]; then
      uv pip install -r requirements-server-py38.txt
    fi
  else
    python -m pip install --upgrade pip
    pip install -r requirements-robot.txt
    if grep -q 'backend: mixed\|backend: python' config.deploy.yaml && [[ -f requirements-server-py38.txt ]]; then
      pip install -r requirements-server-py38.txt
    fi
  fi
fi

cat > start_robot.sh <<EOF
#!/usr/bin/env bash
set -euo pipefail
DIR="\$(cd "\$(dirname "\${BASH_SOURCE[0]}")" && pwd)"
cd "\$DIR"
source .venv/bin/activate
exec python3 run.py --config "\$DIR/config.deploy.yaml" --$MODE
EOF
chmod +x start_robot.sh

if [[ "$SYSTEMD" == "1" ]]; then
  if [[ "$(id -u)" -ne 0 ]]; then
    SUDO=sudo
  else
    SUDO=
  fi

  $SUDO tee /etc/systemd/system/voice-control.service >/dev/null <<EOF
[Unit]
Description=VoiceControl Python-managed Rust service
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$SERVICE_USER
WorkingDirectory=$TARGET_ROOT
ExecStart=$TARGET_ROOT/.venv/bin/python $TARGET_ROOT/run.py --config $TARGET_ROOT/config.deploy.yaml --$MODE
Restart=always
RestartSec=3
KillSignal=SIGTERM

[Install]
WantedBy=multi-user.target
EOF

  $SUDO systemctl daemon-reload
  $SUDO systemctl enable voice-control.service
  $SUDO systemctl restart voice-control.service
  echo "systemd service installed: voice-control.service"
  echo "Logs: journalctl -u voice-control -f"
else
  echo "Install complete."
  echo "Run now:"
  echo "  cd $TARGET_ROOT"
  echo "  ./start_robot.sh"
  echo ""
  echo "Or install systemd later:"
  echo "  scripts/install_robot_target.sh --systemd --mode $MODE"
fi
