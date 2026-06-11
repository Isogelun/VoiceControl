"""
ASR 文本归一化、同音唤醒匹配和高频命令规则兜底。
"""

import re
from typing import Optional


HOMOPHONE_REPLACEMENTS = {
    "小莫同学": "小墨同学",
    "小默同学": "小墨同学",
    "小末同学": "小墨同学",
    "小沫同学": "小墨同学",
    "晓墨同学": "小墨同学",
    "你好小莫同学": "你好小墨同学",
    "你好小默同学": "你好小墨同学",
    "你好画画": "你好花花",
    "你好华华": "你好花花",
    "你好哗哗": "你好花花",
    "画画": "花花",
    "华华": "花花",
    "哗哗": "花花",
    "网前走": "往前走",
    "望前走": "往前走",
    "王前走": "往前走",
    "往钱走": "往前走",
    "向钱走": "向前走",
    "乡前走": "向前走",
    "左传": "左转",
    "做转": "左转",
    "右传": "右转",
    "又转": "右转",
    "停止一下": "停止",
    "停只": "停止",
    "亭止": "停止",
    "齐步走": "向前走",
    "往左走": "向左走",
    "往右走": "向右走",
    "左移": "向左走",
    "右移": "向右走",
    "我立了": "站起来",
    "我蹲了": "坐下",
    "我躺平了": "趴下",
    "我倒了": "趴下",
}


PINYIN_CHAR_MAP = {
    "你": "ni",
    "尼": "ni",
    "好": "hao",
    "号": "hao",
    "浩": "hao",
    "花": "hua",
    "画": "hua",
    "华": "hua",
    "哗": "hua",
    "小": "xiao",
    "晓": "xiao",
    "墨": "mo",
    "莫": "mo",
    "默": "mo",
    "末": "mo",
    "沫": "mo",
    "同": "tong",
    "童": "tong",
    "学": "xue",
    "雪": "xue",
    "往": "wang",
    "网": "wang",
    "望": "wang",
    "王": "wang",
    "向": "xiang",
    "乡": "xiang",
    "前": "qian",
    "钱": "qian",
    "签": "qian",
    "走": "zou",
    "后": "hou",
    "候": "hou",
    "退": "tui",
    "左": "zuo",
    "做": "zuo",
    "右": "you",
    "又": "you",
    "有": "you",
    "转": "zhuan",
    "传": "zhuan",
    "停": "ting",
    "亭": "ting",
    "听": "ting",
    "止": "zhi",
    "只": "zhi",
    "站": "zhan",
    "立": "li",
    "起": "qi",
    "坐": "zuo",
    "下": "xia",
    "趴": "pa",
    "卧": "wo",
    "倒": "dao",
    "步": "bu",
    "齐": "qi",
    "移": "yi",
    "蹲": "dun",
    "躺": "tang",
    "平": "ping",
    "问": "wen",
    "候": "hou",
    "招": "zhao",
    "手": "shou",
    "摇": "yao",
    "晃": "huang",
    "伸": "shen",
    "懒": "lan",
    "腰": "yao",
}


COMMAND_RULES = [
    ("stop", ["停止", "停下", "别动", "不要动", "停住"], {}),
    ("move_forward", ["往前走", "向前走", "前进", "往前", "向前"], {"direction": "forward"}),
    ("move_backward", ["往后走", "向后走", "后退", "往后", "向后", "退后"], {"direction": "backward"}),
    ("move_left", ["向左走", "往左走", "左移", "向左移动"], {"direction": "left"}),
    ("move_right", ["向右走", "往右走", "右移", "向右移动"], {"direction": "right"}),
    ("turn_left", ["左转", "向左转", "往左转"], {"direction": "left"}),
    ("turn_right", ["右转", "向右转", "往右转"], {"direction": "right"}),
    ("stand_up", ["站起来", "起立", "站立"], {}),
    ("sit_down", ["坐下"], {}),
    ("lie_down", ["趴下", "卧倒", "躺下"], {}),
    ("greet", ["打招呼", "问候", "招手"], {}),
    ("shake_body", ["摇身体", "晃身体", "扭一扭"], {}),
    ("stretch", ["伸懒腰", "伸展"], {}),
]


CN_NUMBERS = {
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
    "十": 10,
}


def compact_text(text: str) -> str:
    return "".join(ch for ch in text.lower() if ch.isalnum())


def normalize_asr_text(text: str) -> str:
    text = (text or "").strip()
    for wrong, right in HOMOPHONE_REPLACEMENTS.items():
        text = text.replace(wrong, right)
    return text


def text_to_pinyin_key(text: str) -> str:
    parts = []
    for ch in compact_text(normalize_asr_text(text)):
        if ch.isascii():
            parts.append(ch)
        else:
            parts.append(PINYIN_CHAR_MAP.get(ch, ch))
    return "".join(parts)


def is_wake_phrase(text: str, wake_phrases) -> bool:
    normalized = normalize_asr_text(text)
    compact = compact_text(normalized)
    pinyin = text_to_pinyin_key(normalized)
    for phrase in wake_phrases:
        phrase = normalize_asr_text(phrase)
        if not phrase:
            continue
        phrase_compact = compact_text(phrase)
        phrase_pinyin = text_to_pinyin_key(phrase)
        if phrase_compact and phrase_compact in compact:
            return True
        if phrase_pinyin and phrase_pinyin in pinyin:
            return True
    return False


def parse_command_rule(text: str) -> Optional[dict]:
    normalized = normalize_asr_text(text)
    compact = compact_text(normalized)
    pinyin = text_to_pinyin_key(normalized)

    for intent, phrases, base_slots in COMMAND_RULES:
        if not _matches_any(compact, pinyin, phrases):
            continue
        slots = dict(base_slots)
        if intent.startswith("move_"):
            slots["steps"] = _extract_steps(normalized)
        if intent.startswith("turn_"):
            angle = _extract_angle(normalized)
            if angle is not None:
                slots["angle"] = angle
        return {
            "intent": intent,
            "slots": slots,
            "raw": text,
            "normalized": normalized,
            "source": "rule",
        }
    if _looks_like_bare_forward_walk(normalized, compact, pinyin):
        return {
            "intent": "move_forward",
            "slots": {
                "direction": "forward",
                "steps": _extract_steps(normalized),
            },
            "raw": text,
            "normalized": normalized,
            "source": "rule",
        }
    return None


def _matches_any(compact: str, pinyin: str, phrases) -> bool:
    for phrase in phrases:
        phrase_compact = compact_text(normalize_asr_text(phrase))
        phrase_pinyin = text_to_pinyin_key(phrase)
        if phrase_compact and phrase_compact in compact:
            return True
        if phrase_pinyin and phrase_pinyin in pinyin:
            return True
    return False


def _looks_like_bare_forward_walk(text: str, compact: str, pinyin: str) -> bool:
    has_walk = "走" in compact or "zou" in pinyin
    if not has_walk:
        return False

    blocked_direction_words = ("向后", "往后", "后退", "向左", "往左", "左转", "向右", "往右", "右转")
    if any(word in text for word in blocked_direction_words):
        return False

    if re.search(r"\d+\s*步", text):
        return True
    if any(f"{word}步" in text for word in CN_NUMBERS):
        return True
    return "起来走" in text or compact in {"走", "走走"}


def _extract_steps(text: str) -> int:
    match = re.search(r"(\d+)\s*步", text)
    if match:
        return int(match.group(1))
    for word, value in CN_NUMBERS.items():
        if f"{word}步" in text:
            return value
    return 1


def _extract_angle(text: str) -> Optional[int]:
    match = re.search(r"(\d+)\s*度", text)
    if match:
        return int(match.group(1))
    return None
