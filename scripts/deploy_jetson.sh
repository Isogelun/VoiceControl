#!/bin/bash
#
# VoiceControl Jetson Orin Nano 部署脚本
#
# 使用方式:
#   1. 先从已部署好的 Jetson 打包 golden image:
#      bash deploy_jetson.sh --pack
#
#   2. 批量部署到新 Jetson:
#      bash deploy_jetson.sh --deploy 10.10.20.83 unitree 123
#      bash deploy_jetson.sh --deploy 10.10.20.84 unitree 123
#
#   3. 一次部署多台:
#      bash deploy_jetson.sh --batch hosts.txt
#      (hosts.txt 每行格式: IP 用户名 密码)
#
# 前置条件:
#   - 部署机已安装 ssh, scp, sshpass
#   - 目标 Jetson 运行 JetPack 5.1.1, Python 3.8
#   - 网络可达目标 Jetson
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
GOLDEN_TAR="${PROJECT_DIR}/deploy_package/voicecontrol_jetson.tar.gz"
PACKAGES_TAR="${PROJECT_DIR}/deploy_package/python_packages.tar.gz"

# ─── 颜色输出 ───────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# ─── 从已部署的 Jetson 打包 golden image ────────────────────────────
pack_golden_image() {
    local src_host="${1:?用法: --pack <已部署的Jetson IP> [用户名] [密码]}"
    local src_user="${2:-unitree}"
    local src_pass="${3:-123}"

    info "从 ${src_user}@${src_host} 打包 golden image..."
    mkdir -p "${PROJECT_DIR}/deploy_package"

    info "打包项目代码和模型..."
    sshpass -p "$src_pass" ssh -o StrictHostKeyChecking=no "${src_user}@${src_host}" \
        "cd ~ && tar czf /tmp/voicecontrol_jetson.tar.gz VoiceControl/"

    info "下载项目包 (~4.3GB, 请耐心等待)..."
    sshpass -p "$src_pass" scp -o StrictHostKeyChecking=no \
        "${src_user}@${src_host}:/tmp/voicecontrol_jetson.tar.gz" "$GOLDEN_TAR"

    info "打包 Python 依赖..."
    sshpass -p "$src_pass" ssh -o StrictHostKeyChecking=no "${src_user}@${src_host}" \
        "cd ~ && tar czf /tmp/python_packages.tar.gz .local/lib/python3.8/site-packages/ .local/bin/"

    info "下载 Python 包 (~800MB)..."
    sshpass -p "$src_pass" scp -o StrictHostKeyChecking=no \
        "${src_user}@${src_host}:/tmp/python_packages.tar.gz" "$PACKAGES_TAR"

    info "清理远端临时文件..."
    sshpass -p "$src_pass" ssh "${src_user}@${src_host}" \
        "rm -f /tmp/voicecontrol_jetson.tar.gz /tmp/python_packages.tar.gz"

    local total_size
    total_size=$(du -sh "${PROJECT_DIR}/deploy_package/" | cut -f1)
    info "Golden image 打包完成, 总大小: ${total_size}"
    info "  项目包: $GOLDEN_TAR"
    info "  依赖包: $PACKAGES_TAR"
}

# ─── 部署到单台 Jetson ──────────────────────────────────────────────
deploy_single() {
    local host="${1:?缺少目标 IP}"
    local user="${2:-unitree}"
    local pass="${3:-123}"
    local ssh_opts="-o StrictHostKeyChecking=no -o ConnectTimeout=10"

    info "=========================================="
    info "部署到 ${user}@${host}"
    info "=========================================="

    # 检查包是否存在
    [ -f "$GOLDEN_TAR" ]   || error "未找到 golden image: $GOLDEN_TAR\n请先运行: bash $0 --pack <已部署的IP>"
    [ -f "$PACKAGES_TAR" ] || error "未找到 Python 包: $PACKAGES_TAR"

    # 1. 测试连通性
    info "[1/7] 测试 SSH 连通性..."
    sshpass -p "$pass" ssh $ssh_opts "${user}@${host}" "echo ok" || error "无法连接 ${host}"

    # 2. 创建 swap
    info "[2/7] 创建 4GB swap 文件..."
    sshpass -p "$pass" ssh $ssh_opts "${user}@${host}" "
        if [ ! -f /swapfile ]; then
            echo '$pass' | sudo -S bash -c '
                fallocate -l 4G /swapfile
                chmod 600 /swapfile
                mkswap /swapfile
                swapon /swapfile
                grep -q swapfile /etc/fstab || echo \"/swapfile none swap sw 0 0\" >> /etc/fstab
            '
            echo 'swap 已创建'
        else
            echo '$pass' | sudo -S swapon /swapfile 2>/dev/null || true
            echo 'swap 已存在'
        fi
    "

    # 3. 上传项目
    info "[3/7] 上传项目代码和模型 (~4.3GB)..."
    sshpass -p "$pass" scp $ssh_opts "$GOLDEN_TAR" "${user}@${host}:/tmp/voicecontrol_jetson.tar.gz"

    # 4. 上传 Python 包
    info "[4/7] 上传 Python 依赖 (~800MB)..."
    sshpass -p "$pass" scp $ssh_opts "$PACKAGES_TAR" "${user}@${host}:/tmp/python_packages.tar.gz"

    # 5. 解压部署
    info "[5/7] 解压部署文件..."
    sshpass -p "$pass" ssh $ssh_opts "${user}@${host}" "
        cd ~
        rm -rf VoiceControl.bak
        [ -d VoiceControl ] && mv VoiceControl VoiceControl.bak
        tar xzf /tmp/voicecontrol_jetson.tar.gz
        tar xzf /tmp/python_packages.tar.gz
        rm -f /tmp/voicecontrol_jetson.tar.gz /tmp/python_packages.tar.gz
        echo '解压完成'
    "

    # 6. 写入配置（可按需修改 command.service_url）
    info "[6/7] 写入配置文件..."
    sshpass -p "$pass" ssh $ssh_opts "${user}@${host}" "
        cat > ~/VoiceControl/config.yaml << 'YAMLEOF'
server:
  host: 0.0.0.0
  asr_port: 8000
  nlu_port: 8001
  service_timeout: 180
  gpu: true
  gpu_encoder_only: true

inference:
  backend: python

asr:
  engine: qwen3

wake:
  backend: asr
  keyword: 你好曼波
  feedback_enabled: true
  text:
    - 你好曼波
    - 曼波
    - 你好，曼波
    - 你好
    - 你好，小曼
    - 你好，小曼波
    - 曼波，你好
  aliases:
    - 你好曼波
    - 曼波
    - 慢播
    - 快播
    - 那波
    - 南波
    - 慢波
    - 曼播
    - 你好慢播
    - 你好快播
    - 你好那波
    - 你好南波
    - 漫波
    - 你好漫波
  audio: audio/mabo.mp3

microphone:
  device: 0

command:
  service_url: http://${host}:8090/api/v1/local/motion
  service_timeout: 5.0
  fast_response: true
  pre_stand_on_wake: true
  move_step_timeout_ms: 600
  move_default_timeout_ms: 1200
  auto_stand_before_move: true
  move_prepare_delay_ms: 1200
  move_linear_speed: 0.25
  move_yaw_speed: 0.5
  move_prime_timeout_ms: 1500
  move_post_move_delay_ms: 1200
  move_native_enabled: true
  move_native_default_steps: 3
  move_native_min_steps: 3
  move_native_timeout_ms: 1000
  move_native_linear_speed: 1.0
  move_native_yaw_speed: 1.0
  move_fast_response: true
  move_fast_native_first: true
  move_fast_followup_move: true
  move_fast_followup_delay_ms: 80
  move_fast_auto_stand: true
  move_stop_after_timeout: true
YAMLEOF
        echo '配置写入完成'
    "

    # 7. 验证
    info "[7/7] 验证部署..."
    sshpass -p "$pass" ssh $ssh_opts "${user}@${host}" "
        echo '--- Python 版本 ---'
        python3 --version
        echo '--- ORT GPU 检查 ---'
        python3 -c 'import onnxruntime as ort; p=ort.get_available_providers(); print(\"providers:\", p); assert \"CUDAExecutionProvider\" in p, \"无 CUDA!\"'
        echo '--- 模型文件检查 ---'
        ls -lh ~/VoiceControl/models/asr/encoder.onnx ~/VoiceControl/models/nlu/encoder.onnx
        echo '--- 内存 ---'
        free -h
        echo '--- 部署验证通过 ---'
    "

    info "✓ ${host} 部署完成!"
}

# ─── 批量部署 ───────────────────────────────────────────────────────
deploy_batch() {
    local hosts_file="${1:?用法: --batch <hosts.txt>}"
    [ -f "$hosts_file" ] || error "hosts 文件不存在: $hosts_file"

    local total=0 success=0 fail=0
    local failed_hosts=""

    while IFS=' ' read -r ip user pass rest; do
        [ -z "$ip" ] && continue
        [[ "$ip" == \#* ]] && continue
        total=$((total + 1))
        user="${user:-unitree}"
        pass="${pass:-123}"

        if deploy_single "$ip" "$user" "$pass"; then
            success=$((success + 1))
        else
            fail=$((fail + 1))
            failed_hosts="${failed_hosts}\n  - ${ip}"
        fi
        echo ""
    done < "$hosts_file"

    info "=========================================="
    info "批量部署完成: 总计 ${total}, 成功 ${success}, 失败 ${fail}"
    if [ $fail -gt 0 ]; then
        warn "失败主机:${failed_hosts}"
    fi
    info "=========================================="
}

# ─── 快速验证（不部署，只检查服务状态）────────────────────────────
verify_single() {
    local host="${1:?缺少目标 IP}"
    local user="${2:-unitree}"
    local pass="${3:-123}"

    info "验证 ${host}..."
    sshpass -p "$pass" ssh -o StrictHostKeyChecking=no "${user}@${host}" "
        echo '=== 进程 ==='
        ps aux | grep 'run.py' | grep -v grep || echo '(未运行)'
        echo '=== 端口 ==='
        ss -tlnp | grep -E '800[01]' || echo '(端口未监听)'
        echo '=== 内存 ==='
        free -h
        echo '=== 健康检查 ==='
        curl -s --connect-timeout 3 http://localhost:8000/health || echo 'ASR: 未响应'
        echo ''
        curl -s --connect-timeout 3 http://localhost:8001/health || echo 'NLU: 未响应'
        echo ''
    "
}

# ─── 入口 ──────────────────────────────────────────────────────────
case "${1:-}" in
    --pack)
        shift
        pack_golden_image "$@"
        ;;
    --deploy)
        shift
        deploy_single "$@"
        ;;
    --batch)
        shift
        deploy_batch "$@"
        ;;
    --verify)
        shift
        verify_single "$@"
        ;;
    *)
        echo "VoiceControl Jetson 部署工具"
        echo ""
        echo "用法:"
        echo "  $0 --pack   <源IP> [用户名] [密码]    从已部署的 Jetson 打包 golden image"
        echo "  $0 --deploy <目标IP> [用户名] [密码]   部署到单台 Jetson"
        echo "  $0 --batch  <hosts.txt>               批量部署 (每行: IP 用户名 密码)"
        echo "  $0 --verify <目标IP> [用户名] [密码]   验证部署状态"
        echo ""
        echo "首次使用流程:"
        echo "  1. bash $0 --pack 10.10.20.82 unitree 123"
        echo "  2. bash $0 --deploy 10.10.20.83 unitree 123"
        echo "  3. ssh unitree@10.10.20.83 'bash ~/VoiceControl/start_voicecontrol.sh --onboard'"
        ;;
esac
