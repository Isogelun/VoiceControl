#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
便捷入口：等价于 python scripts/mic_probe.py

请使用 python3，不要用 python（Jetson 上 python 常为 2.7）。

探测自带助手（默认，不需 sounddevice）:
    sudo python3 mic.py

扫描麦克风（需要 venv 里的 sounddevice）:
    sudo .venv/bin/python mic.py --scan-all
    sudo .venv/bin/python mic.py --list
"""

import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_ROOT, "scripts"))

from mic_probe import main  # noqa: E402

if __name__ == "__main__":
    main()
