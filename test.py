import json
import urllib.request
import time

BASE = "http://10.10.20.82:8090"

def post(path, body):
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        BASE + path,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
        )
    with urllib.request.urlopen(req, timeout=10) as resp:
        text = resp.read().decode("utf-8")
        print(resp.status, text)
        return text

# 先确保站立
post("/api/v1/local/motion", {
    "command_type": "stand_up",
    "params": {}
})

time.sleep(2)

# 明显往前走一段
post("/api/v1/local/motion", {
    "command_type": "move",
    "params": {
        "vx": 0.25,
        "vy": 0.0,
        "wz": 0.0,
        "timeout_ms": 1500
    }
})

time.sleep(2)

# 再补一次停止
post("/api/v1/local/motion", {
    "command_type": "move_forward",
    "params": {
        "step": 3,
        "vx": 1,
        "timeout_ms": 1000
    }
})