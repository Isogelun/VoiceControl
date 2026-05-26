#!/bin/bash
OUTPUT="record.wav"

# 1. 让用户输入设备号（格式如 0,0）
read -p "请输入设备号（格式如 0,0）: " DEVICE_NUM
DEVICE="plughw:$DEVICE_NUM"

# 2. 安全退出处理（Ctrl+C 保存）
cleanup() {
    echo -e "\n\n✅ 录音已停止，文件已保存为 $OUTPUT"
    pkill -SIGINT -f arecord 2>/dev/null
    exit 0
}
trap cleanup SIGINT

# 3. 核心逻辑：后台运行 arecord，通过进程状态判断
echo "启动中..."
while true; do
    # 后台运行 arecord，所有输出丢黑洞
    arecord -D "$DEVICE" -r 16000 -f S16_LE -c 1 \
    --period-size=1024 --buffer-size=4096 "$OUTPUT" > /dev/null 2>&1 &
    
    # 获取后台进程PID
    REC_PID=$!

    # 等待1秒，看进程是否还在运行（在=连上了，不在=没连上）
    sleep 1
    if ps -p $REC_PID > /dev/null; then
        # 进程还在，说明设备连上了，切换为“录音中...”
        echo -ne "\r录音中..."
        wait $REC_PID  # 等待录音结束（或被Ctrl+C打断）
        break
    else
        # 进程退出了，说明设备没连上，继续重试
        kill $REC_PID 2>/dev/null
        echo -ne "\r启动中..."
        sleep 0.5
    fi
done
