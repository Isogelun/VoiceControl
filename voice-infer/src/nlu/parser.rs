use std::collections::HashMap;

use serde::Serialize;
use serde_json::{Map, Value};

/// NLU 输出结构体，JSON 序列化时与 Python 版完全兼容。
#[derive(Debug, Clone, Serialize)]
pub struct NluOutput {
    pub intent: String,
    pub slots: HashMap<String, Value>,
    pub raw: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub command: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<Value>,
}

/// 解析 NLU 模型的原始输出文本，精确复现 Python nlu/engine.py:parse_nlu_output。
pub fn parse_nlu_output(raw: &str) -> NluOutput {
    let raw = raw.trim();

    // 路径 A: JSON
    if let Ok(parsed) = serde_json::from_str::<Value>(raw) {
        if let Value::Object(ref map) = parsed {
            if let Some(result) = parse_json_output(map, raw, &parsed) {
                return result;
            }
        }
    }

    // 路径 B: key=value
    if raw.contains('=') {
        return parse_kv_output(raw);
    }

    // 路径 C: fallback
    NluOutput {
        intent: "unknown".into(),
        slots: HashMap::new(),
        raw: raw.into(),
        command: None,
        source: None,
        message: None,
    }
}

fn parse_json_output(map: &Map<String, Value>, raw: &str, full: &Value) -> Option<NluOutput> {
    let type_str = map.get("type").and_then(|v| v.as_str()).unwrap_or("");

    // type == "cmd"
    if type_str == "cmd" {
        let payload = map.get("payload")
            .and_then(|v| v.as_object())
            .cloned()
            .unwrap_or_default();
        let command_type = payload.get("command_type")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .trim()
            .to_string();
        let payload_json = payload.get("payload_json")
            .and_then(|v| v.as_object())
            .cloned()
            .unwrap_or_default();

        let intent = command_type_to_intent(&command_type, &payload_json);
        let mut slots = map_to_hashmap(&payload_json);
        augment_slots(&mut slots, &intent, &command_type);

        return Some(NluOutput {
            intent,
            slots,
            raw: raw.into(),
            command: Some(full.clone()),
            source: Some("model".into()),
            message: None,
        });
    }

    // type == "chat"
    if type_str == "chat" {
        let message = dig(map, &["payload", "message"]);
        return Some(NluOutput {
            intent: "unknown".into(),
            slots: HashMap::new(),
            raw: raw.into(),
            command: Some(full.clone()),
            source: Some("chat".into()),
            message,
        });
    }

    // 含 intent 或 slots 字段
    if map.contains_key("intent") || map.contains_key("slots") {
        let intent = map.get("intent")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown")
            .to_string();

        let slots = match map.get("slots") {
            Some(Value::Object(s)) => map_to_hashmap(s),
            Some(Value::Null) | None => {
                // slots 为 null 或缺失：取除 intent 外的所有 key
                let mut h = HashMap::new();
                for (k, v) in map {
                    if k != "intent" {
                        h.insert(k.clone(), v.clone());
                    }
                }
                h
            }
            Some(other) => {
                let mut h = HashMap::new();
                h.insert("value".into(), other.clone());
                h
            }
        };

        return Some(NluOutput {
            intent,
            slots,
            raw: raw.into(),
            command: None,
            source: None,
            message: None,
        });
    }

    None
}

fn parse_kv_output(raw: &str) -> NluOutput {
    let normalized = raw.replace(';', ",");
    let parts: Vec<&str> = normalized.split(',').collect();
    let mut kv = HashMap::new();
    for part in parts {
        let part = part.trim();
        if let Some(eq_pos) = part.find('=') {
            let key = part[..eq_pos].trim().to_string();
            let value = part[eq_pos + 1..].trim().to_string();
            kv.insert(key, Value::String(value));
        }
    }
    let intent = kv.remove("intent")
        .and_then(|v| v.as_str().map(String::from))
        .unwrap_or_else(|| "unknown".into());
    NluOutput {
        intent,
        slots: kv,
        raw: raw.into(),
        command: None,
        source: None,
        message: None,
    }
}

/// 精确复现 Python _command_type_to_intent，20 条映射 + Move 方向推断。
fn command_type_to_intent(command_type: &str, payload_json: &Map<String, Value>) -> String {
    let normalized: String = command_type.chars()
        .filter(|c| c.is_alphanumeric())
        .flat_map(|c| c.to_lowercase())
        .collect();

    if normalized == "move" {
        return infer_move_intent(payload_json);
    }

    match normalized.as_str() {
        "moveforward"    => "move_forward",
        "movebackward"   => "move_backward",
        "moveleft"       => "move_left",
        "moveright"      => "move_right",
        "turnleft"       => "turn_left",
        "turnright"      => "turn_right",
        "sit" | "sitdown" | "standdown" => "sit_down",
        "stand" | "standup" | "risesit" | "recoverystand" | "balancestand" => "stand_up",
        "liedown"        => "lie_down",
        "greet"          => "greet",
        "shakebody"      => "shake_body",
        "stretch"        => "stretch",
        "damp" | "stop" | "stopmove" => "stop",
        "error"          => "unknown",
        "" => "unknown",
        other => other,
    }.to_string()
}

/// 精确复现 Python _infer_move_intent: 根据 vx/vy/vyaw 判断具体方向。
fn infer_move_intent(payload_json: &Map<String, Value>) -> String {
    let vx = extract_f64(payload_json, "vx");
    let vy = extract_f64(payload_json, "vy");
    let vyaw = extract_f64(payload_json, "vyaw");
    let eps = 1e-6;

    if vyaw.abs() > vx.abs().max(vy.abs()).max(eps) {
        return if vyaw > 0.0 { "turn_left" } else { "turn_right" }.into();
    }
    if vy.abs() > vx.abs().max(eps) {
        return if vy > 0.0 { "move_left" } else { "move_right" }.into();
    }
    if vx.abs() > eps {
        return if vx > 0.0 { "move_forward" } else { "move_backward" }.into();
    }
    "move".into()
}

/// 精确复现 Python _augment_slots: 根据 intent 补充 direction 和 command_type。
fn augment_slots(slots: &mut HashMap<String, Value>, intent: &str, command_type: &str) {
    let dir = match intent {
        "move_forward"  => Some("forward"),
        "move_backward" => Some("backward"),
        "move_left"     => Some("left"),
        "move_right"    => Some("right"),
        "turn_left"     => Some("left"),
        "turn_right"    => Some("right"),
        _ => None,
    };
    if let Some(d) = dir {
        slots.entry("direction".into()).or_insert(Value::String(d.into()));
    }
    if !command_type.is_empty() {
        slots.entry("command_type".into()).or_insert(Value::String(command_type.into()));
    }
}

/// 精确复现 Python _dig: 按 key 链深入 dict。
fn dig(map: &Map<String, Value>, keys: &[&str]) -> Option<Value> {
    let mut current: &Value = &Value::Object(map.clone());
    for key in keys {
        match current {
            Value::Object(m) => {
                current = m.get(*key)?;
            }
            _ => return None,
        }
    }
    Some(current.clone())
}

fn extract_f64(map: &Map<String, Value>, key: &str) -> f64 {
    map.get(key)
        .and_then(|v| match v {
            Value::Number(n) => n.as_f64(),
            Value::String(s) => s.parse::<f64>().ok(),
            Value::Null => Some(0.0),
            _ => None,
        })
        .unwrap_or(0.0)
}

fn map_to_hashmap(m: &Map<String, Value>) -> HashMap<String, Value> {
    m.iter().map(|(k, v)| (k.clone(), v.clone())).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cmd_move_forward() {
        let raw = r#"{"type":"cmd","payload":{"command_type":"MoveForward","payload_json":{"vx":0.3}}}"#;
        let out = parse_nlu_output(raw);
        assert_eq!(out.intent, "move_forward");
        assert_eq!(out.slots.get("direction").and_then(|v| v.as_str()), Some("forward"));
        assert!(out.source.as_deref() == Some("model"));
    }

    #[test]
    fn test_cmd_move_infer_left() {
        let raw = r#"{"type":"cmd","payload":{"command_type":"Move","payload_json":{"vx":0.0,"vy":0.3,"vyaw":0.0}}}"#;
        let out = parse_nlu_output(raw);
        assert_eq!(out.intent, "move_left");
        assert_eq!(out.slots.get("direction").and_then(|v| v.as_str()), Some("left"));
    }

    #[test]
    fn test_cmd_move_infer_turn_left() {
        let raw = r#"{"type":"cmd","payload":{"command_type":"Move","payload_json":{"vx":0.0,"vy":0.0,"vyaw":0.5}}}"#;
        let out = parse_nlu_output(raw);
        assert_eq!(out.intent, "turn_left");
    }

    #[test]
    fn test_cmd_move_all_zero() {
        let raw = r#"{"type":"cmd","payload":{"command_type":"Move","payload_json":{"vx":0.0,"vy":0.0,"vyaw":0.0}}}"#;
        let out = parse_nlu_output(raw);
        assert_eq!(out.intent, "move");
    }

    #[test]
    fn test_cmd_sit() {
        let raw = r#"{"type":"cmd","payload":{"command_type":"Sit","payload_json":{}}}"#;
        let out = parse_nlu_output(raw);
        assert_eq!(out.intent, "sit_down");
    }

    #[test]
    fn test_cmd_stop_move() {
        let raw = r#"{"type":"cmd","payload":{"command_type":"StopMove","payload_json":{}}}"#;
        let out = parse_nlu_output(raw);
        assert_eq!(out.intent, "stop");
    }

    #[test]
    fn test_cmd_recovery_stand() {
        let raw = r#"{"type":"cmd","payload":{"command_type":"RecoveryStand","payload_json":{}}}"#;
        let out = parse_nlu_output(raw);
        assert_eq!(out.intent, "stand_up");
    }

    #[test]
    fn test_chat() {
        let raw = r#"{"type":"chat","payload":{"message":"你好，我是曼波"}}"#;
        let out = parse_nlu_output(raw);
        assert_eq!(out.intent, "unknown");
        assert!(out.source.as_deref() == Some("chat"));
        assert_eq!(out.message.as_ref().and_then(|v| v.as_str()), Some("你好，我是曼波"));
    }

    #[test]
    fn test_json_intent_slots() {
        let raw = r#"{"intent":"move_forward","slots":{"direction":"forward","steps":3}}"#;
        let out = parse_nlu_output(raw);
        assert_eq!(out.intent, "move_forward");
        assert_eq!(out.slots.get("direction").and_then(|v| v.as_str()), Some("forward"));
    }

    #[test]
    fn test_json_intent_no_slots() {
        let raw = r#"{"intent":"stop"}"#;
        let out = parse_nlu_output(raw);
        assert_eq!(out.intent, "stop");
        assert!(out.slots.is_empty());
    }

    #[test]
    fn test_kv_format() {
        let raw = "intent=stop, direction=none";
        let out = parse_nlu_output(raw);
        assert_eq!(out.intent, "stop");
        assert_eq!(out.slots.get("direction").and_then(|v| v.as_str()), Some("none"));
    }

    #[test]
    fn test_fallback() {
        let out = parse_nlu_output("你好世界");
        assert_eq!(out.intent, "unknown");
        assert!(out.slots.is_empty());
    }

    #[test]
    fn test_empty_string() {
        let out = parse_nlu_output("");
        assert_eq!(out.intent, "unknown");
    }

    #[test]
    fn test_reference_parse_cases() {
        let cases: Vec<serde_json::Value> = serde_json::from_str(include_str!(
            "../../tests/resources/reference/parse/cases.json"
        )).expect("parse cases.json");

        for case in cases {
            let input = case["input"].as_str().expect("case.input should be string");
            let expected = case["expected"].clone();
            let actual = serde_json::to_value(parse_nlu_output(input)).expect("serialize parser output");
            assert_eq!(actual, expected, "parse case mismatch for input: {input}");
        }
    }

    #[test]
    fn test_reference_predict_cases() {
        let cases: Vec<serde_json::Value> = serde_json::from_str(include_str!(
            "../../tests/resources/reference/nlu/predict_cases.json"
        )).expect("parse predict_cases.json");

        for case in cases {
            let name = case["name"].as_str().expect("case.name should be string");
            let raw = case["raw_output"].as_str().expect("case.raw_output should be string");
            let expected = case["parsed"].clone();
            let actual = serde_json::to_value(parse_nlu_output(raw)).expect("serialize parser output");
            assert_eq!(actual, expected, "predict case mismatch: {name}");
        }
    }
}
