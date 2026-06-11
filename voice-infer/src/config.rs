use std::collections::HashSet;
use std::path::Path;

use anyhow::{Context, Result};
use serde::Deserialize;

use crate::audio::MelConfig;

// ─── ASR Model Config (models/asr/config.json) ───

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub struct AsrModelConfig {
    pub model_type: String,
    pub encoder: AsrEncoderSection,
    pub decoder: AsrDecoderSection,
    pub mel: MelConfig,
    pub special_tokens: AsrSpecialTokens,
    #[serde(default)]
    pub embed_tokens_dtype: String,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub struct AsrEncoderSection {
    pub output_dim: usize,
}

#[derive(Debug, Deserialize)]
pub struct AsrDecoderSection {
    pub hidden_size: usize,
    pub vocab_size: usize,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub struct AsrSpecialTokens {
    pub eos_token_ids: Vec<i64>,
    #[serde(default)]
    pub pad_token_id: i64,
    #[serde(default = "default_audio_pad")]
    pub audio_pad_token_id: i64,
}

fn default_audio_pad() -> i64 { 151676 }

impl AsrModelConfig {
    pub fn load(model_dir: &Path) -> Result<Self> {
        let path = model_dir.join("config.json");
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("reading {}", path.display()))?;
        let config: Self = serde_json::from_str(&text)
            .with_context(|| format!("parsing {}", path.display()))?;
        anyhow::ensure!(
            config.model_type == "qwen3_asr",
            "unsupported ASR model_type: {}",
            config.model_type
        );
        Ok(config)
    }

    pub fn eos_set(&self) -> HashSet<i64> {
        self.special_tokens.eos_token_ids.iter().copied().collect()
    }
}

// ─── NLU Export Config (models/nlu/export_config.json) ───

#[derive(Debug, Deserialize)]
pub struct NluExportConfig {
    #[serde(default = "default_max_seq")]
    pub max_seq_len: usize,
    #[serde(default)]
    pub max_target_len: Option<usize>,
    #[serde(default)]
    pub decoder_start_token_id: i64,
    #[serde(default)]
    pub pad_token_id: i64,
    #[serde(default = "default_nlu_eos")]
    pub eos_token_id: i64,
}

fn default_max_seq() -> usize { 64 }
fn default_nlu_eos() -> i64 { 1 }

impl NluExportConfig {
    pub fn load(model_dir: &Path) -> Result<Self> {
        let path = model_dir.join("export_config.json");
        if !path.is_file() {
            tracing::warn!("export_config.json not found, using defaults");
            return Ok(Self {
                max_seq_len: 64,
                max_target_len: None,
                decoder_start_token_id: 0,
                pad_token_id: 0,
                eos_token_id: 1,
            });
        }
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("reading {}", path.display()))?;
        serde_json::from_str(&text)
            .with_context(|| format!("parsing {}", path.display()))
    }

    pub fn max_output(&self) -> usize {
        self.max_target_len.unwrap_or(128)
    }
}
