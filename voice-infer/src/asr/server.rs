use std::sync::{Arc, Mutex};

use axum::extract::{Multipart, State};
use axum::http::StatusCode;
use axum::response::Json;
use axum::routing::{get, post};
use axum::Router;
use serde_json::{json, Value};

use super::engine::AsrEngine;
use crate::audio::load_audio_from_bytes;

pub fn router(engine: Arc<Mutex<AsrEngine>>) -> Router {
    Router::new()
        .route("/asr", post(asr_handler))
        .route("/health", get(health))
        .with_state(engine)
}

async fn asr_handler(
    State(engine): State<Arc<Mutex<AsrEngine>>>,
    mut multipart: Multipart,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    let mut audio_data: Option<Vec<u8>> = None;
    let mut _language = "auto".to_string();
    let mut _use_itn = true;

    while let Ok(Some(field)) = multipart.next_field().await {
        let name = field.name().unwrap_or("").to_string();
        match name.as_str() {
            "audio" => {
                audio_data = Some(
                    field.bytes().await
                        .map_err(|e| err_500(&format!("reading audio: {e}")))?
                        .to_vec()
                );
            }
            "language" => {
                _language = field.text().await.unwrap_or_else(|_| "auto".into());
            }
            "use_itn" => {
                let val = field.text().await.unwrap_or_else(|_| "true".into());
                _use_itn = val != "false" && val != "0";
            }
            _ => {}
        }
    }

    let audio_data = audio_data.ok_or_else(|| {
        (StatusCode::BAD_REQUEST, Json(json!({"text": "", "error": "missing audio field"})))
    })?;

    let wav = load_audio_from_bytes(&audio_data)
        .map_err(|e| (StatusCode::BAD_REQUEST, Json(json!({"text": "", "error": e.to_string()}))))?;

    let result = tokio::task::spawn_blocking(move || {
        let mut engine = engine.lock()
            .map_err(|e| anyhow::anyhow!("locking ASR engine: {e}"))?;
        engine.transcribe(&wav)
    })
        .await
        .map_err(|e| err_500(&format!("join error: {e}")))?
        .map_err(|e| err_500(&e.to_string()))?;

    tracing::info!(
        total_ms = result.total_ms,
        text = %truncate(&result.text, 60),
        "ASR"
    );

    Ok(Json(serde_json::to_value(result).unwrap()))
}

async fn health() -> Json<Value> {
    Json(json!({"status": "ok", "model": "qwen3-asr"}))
}

fn err_500(msg: &str) -> (StatusCode, Json<Value>) {
    tracing::error!("ASR error: {msg}");
    (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({"text": "", "error": msg})))
}

fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max { s.to_string() }
    else { s.chars().take(max).collect::<String>() + "…" }
}
