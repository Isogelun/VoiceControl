#!/usr/bin/env bash
set -euo pipefail

ORT_VERSION="${ORT_VERSION:-1.23.0}"
WITH_MODELS="${WITH_MODELS:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VOICE_INFER_DIR="$PROJECT_ROOT/voice-infer"
THIRD_PARTY_DIR="$PROJECT_ROOT/third_party"
DIST_ROOT="$PROJECT_ROOT/dist"

usage() {
  cat <<EOF
Usage: $(basename "$0") [--with-models]

Build and package voice-infer for Ubuntu 22.04 on the current Linux machine.

Options:
  --with-models   Copy models/asr and models/nlu into the package.

Environment:
  ORT_VERSION     ONNX Runtime version. Default: 1.23.0
  WITH_MODELS     Set to 1 to copy models without passing --with-models.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --with-models)
      WITH_MODELS=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "This packaging script must run on Linux, preferably the Ubuntu 22.04 target machine." >&2
  exit 1
fi

case "$(uname -m)" in
  x86_64|amd64)
    ORT_ARCH="x64"
    PKG_ARCH="x86_64"
    ;;
  aarch64|arm64)
    ORT_ARCH="aarch64"
    PKG_ARCH="aarch64"
    ;;
  *)
    echo "unsupported Linux architecture: $(uname -m)" >&2
    exit 1
    ;;
esac

command -v cargo >/dev/null 2>&1 || {
  echo "cargo is not installed. Install Rust toolchain first: https://rustup.rs" >&2
  exit 1
}

mkdir -p "$THIRD_PARTY_DIR" "$DIST_ROOT"

ORT_NAME="onnxruntime-linux-${ORT_ARCH}-${ORT_VERSION}"
ORT_TGZ="$THIRD_PARTY_DIR/${ORT_NAME}.tgz"
ORT_DIR="$THIRD_PARTY_DIR/$ORT_NAME"
ORT_URL="https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/${ORT_NAME}.tgz"

if [[ ! -f "$ORT_DIR/lib/libonnxruntime.so.${ORT_VERSION}" ]]; then
  if [[ ! -f "$ORT_TGZ" ]]; then
    echo "Downloading $ORT_URL"
    if command -v curl >/dev/null 2>&1; then
      curl -L "$ORT_URL" -o "$ORT_TGZ"
    elif command -v wget >/dev/null 2>&1; then
      wget -O "$ORT_TGZ" "$ORT_URL"
    else
      echo "curl or wget is required to download ONNX Runtime." >&2
      exit 1
    fi
  fi
  tar -xzf "$ORT_TGZ" -C "$THIRD_PARTY_DIR"
fi

ORT_LIB="$ORT_DIR/lib/libonnxruntime.so.${ORT_VERSION}"
if [[ ! -f "$ORT_LIB" ]]; then
  echo "ONNX Runtime library not found after extraction: $ORT_LIB" >&2
  exit 1
fi

echo "Building voice-infer release"
(
  cd "$VOICE_INFER_DIR"
  cargo build --release
)

PKG_DIR="$DIST_ROOT/voice-infer-ubuntu2204-${PKG_ARCH}"
rm -rf "$PKG_DIR"
mkdir -p "$PKG_DIR/models"

cp "$VOICE_INFER_DIR/target/release/voice-infer" "$PKG_DIR/voice-infer"
cp "$ORT_LIB" "$PKG_DIR/libonnxruntime.so.${ORT_VERSION}"
if [[ -f "$ORT_DIR/lib/libonnxruntime_providers_shared.so" ]]; then
  cp "$ORT_DIR/lib/libonnxruntime_providers_shared.so" "$PKG_DIR/libonnxruntime_providers_shared.so"
fi
ln -sf "libonnxruntime.so.${ORT_VERSION}" "$PKG_DIR/libonnxruntime.so"

if [[ "$WITH_MODELS" == "1" ]]; then
  echo "Copying models into package"
  cp -a "$PROJECT_ROOT/models/asr" "$PKG_DIR/models/asr"
  cp -a "$PROJECT_ROOT/models/nlu" "$PKG_DIR/models/nlu"
else
  cat > "$PKG_DIR/models/README.txt" <<EOF
Place or symlink model directories here before running:
  models/asr
  models/nlu

To include models in the package, rerun:
  WITH_MODELS=1 scripts/package_voice_infer_linux.sh
EOF
fi

cat > "$PKG_DIR/start.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail

DIR="\$(cd "\$(dirname "\${BASH_SOURCE[0]}")" && pwd)"
export ORT_DYLIB_PATH="\$DIR/libonnxruntime.so.${ORT_VERSION}"
export LD_LIBRARY_PATH="\$DIR:\${LD_LIBRARY_PATH:-}"

exec "\$DIR/voice-infer" \
  --asr-model-dir "\$DIR/models/asr" \
  --nlu-model-dir "\$DIR/models/nlu" \
  --host "\${VOICE_INFER_HOST:-0.0.0.0}" \
  --asr-port "\${VOICE_INFER_ASR_PORT:-8000}" \
  --nlu-port "\${VOICE_INFER_NLU_PORT:-8001}" \
  "\$@"
EOF

cat > "$PKG_DIR/verify.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

HOST="${VOICE_INFER_HOST:-127.0.0.1}"
ASR_PORT="${VOICE_INFER_ASR_PORT:-8000}"
NLU_PORT="${VOICE_INFER_NLU_PORT:-8001}"

curl -fsS "http://${HOST}:${ASR_PORT}/health"
echo
curl -fsS "http://${HOST}:${NLU_PORT}/health"
echo
curl -fsS -X POST "http://${HOST}:${NLU_PORT}/nlu" \
  -H 'Content-Type: application/json' \
  -d '{"text":"\u5411\u524d\u8d70\u4e09\u6b65"}'
echo
EOF

chmod +x "$PKG_DIR/voice-infer" "$PKG_DIR/start.sh" "$PKG_DIR/verify.sh"

cat <<EOF
Package created:
  $PKG_DIR

Target usage:
  cd $PKG_DIR
  ./start.sh

In another shell:
  ./verify.sh
EOF
