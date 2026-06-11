use std::sync::{Arc, Mutex};

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::Json;
use axum::routing::{get, post};
use axum::Router;
use serde::Deserialize;
use serde_json::{json, Value};

use super::engine::NluEngine;

#[derive(Deserialize)]
struct NluRequest {
    text: String,
}

pub fn router(engine: Arc<Mutex<NluEngine>>) -> Router {
    Router::new()
        .route("/nlu", post(nlu_handler))
        .route("/health", get(health))
        .with_state(engine)
}

async fn nlu_handler(
    State(engine): State<Arc<Mutex<NluEngine>>>,
    Json(req): Json<NluRequest>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    let text = req.text.clone();
    let result = tokio::task::spawn_blocking(move || {
        let mut engine = engine.lock()
            .map_err(|e| anyhow::anyhow!("locking NLU engine: {e}"))?;
        engine.predict(&text)
    })
        .await
        .map_err(|e| err_500(&format!("join: {e}")))?
        .map_err(|e| err_500(&e.to_string()))?;

    tracing::info!(
        input = %req.text,
        intent = %result.intent,
        "NLU"
    );

    let val = serde_json::to_value(&result).unwrap();
    Ok(Json(val))
}

async fn health() -> Json<Value> {
    Json(json!({"status": "ok"}))
}

fn err_500(msg: &str) -> (StatusCode, Json<Value>) {
    tracing::error!("NLU error: {msg}");
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(json!({
            "intent": "unknown",
            "slots": {},
            "raw": "",
            "error": msg,
        })),
    )
}
