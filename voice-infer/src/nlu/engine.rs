use std::path::Path;

use anyhow::{Context, Result};
use ndarray::Array2;
use ort::{
    session::{Session, builder::GraphOptimizationLevel},
    value::TensorRef,
};
use tokenizers::{PaddingDirection, PaddingParams, PaddingStrategy, Tokenizer, TruncationParams, TruncationStrategy};

use crate::config::NluExportConfig;
use super::parser::{NluOutput, parse_nlu_output};

const PREFIX: &str = "指令解析: ";

pub struct NluEngine {
    encoder: Session,
    decoder: Session,
    tokenizer: Tokenizer,
    config: NluExportConfig,
    structured_early_stop: bool,
}

impl NluEngine {
    pub fn new(
        model_dir: &Path,
        tokenizer_dir: &Path,
        num_threads: Option<usize>,
        _use_gpu: bool,
    ) -> Result<Self> {
        let config = NluExportConfig::load(model_dir)?;
        let threads = resolve_threads(num_threads, 4);

        tracing::info!(
            model_dir = %model_dir.display(),
            tokenizer_dir = %tokenizer_dir.display(),
            threads,
            max_seq_len = config.max_seq_len,
            "loading NLU model"
        );

        tracing::info!("loading NLU encoder session");
        let encoder = create_session(&model_dir.join("encoder.onnx"), threads)
            .context("loading NLU encoder session")?;
        tracing::info!("loading NLU decoder session");
        let decoder = create_session(&model_dir.join("decoder.onnx"), threads)
            .context("loading NLU decoder session")?;
        tracing::info!("NLU ONNX sessions loaded");

        tracing::info!("loading NLU tokenizer");
        let mut tokenizer = Tokenizer::from_file(tokenizer_dir.join("tokenizer.json"))
            .map_err(|e| anyhow::anyhow!("loading NLU tokenizer: {e}"))?;
        tracing::info!("NLU tokenizer loaded");

        // 配置 padding 到 max_seq_len (默认 64)
        tokenizer.with_padding(Some(PaddingParams {
            strategy: PaddingStrategy::Fixed(config.max_seq_len),
            direction: PaddingDirection::Right,
            pad_id: config.pad_token_id as u32,
            pad_type_id: 0,
            pad_token: String::from("</s>"),
            pad_to_multiple_of: None,
        }));

        // 配置 truncation
        tokenizer.with_truncation(Some(TruncationParams {
            max_length: config.max_seq_len,
            strategy: TruncationStrategy::LongestFirst,
            stride: 0,
            direction: tokenizers::TruncationDirection::Right,
        })).map_err(|e| anyhow::anyhow!("setting truncation: {e}"))?;

        let structured_early_stop = std::env::var("NLU_STRUCTURED_EARLY_STOP")
            .map(|v| !matches!(v.as_str(), "0" | "false" | "False" | "no"))
            .unwrap_or(false);

        tracing::info!("NLU model loaded");

        Ok(Self { encoder, decoder, tokenizer, config, structured_early_stop })
    }

    pub fn predict(&mut self, text: &str) -> Result<NluOutput> {
        let input_text = format!("{PREFIX}{text}");

        // ── Step 1: Tokenize ──
        let encoding = self.tokenizer.encode(input_text, true)
            .map_err(|e| anyhow::anyhow!("tokenize: {e}"))?;

        let max_len = self.config.max_seq_len;
        let input_ids: Vec<i64> = encoding.get_ids().iter()
            .take(max_len)
            .map(|&id| id as i64)
            .collect();
        let attention_mask: Vec<i64> = encoding.get_attention_mask().iter()
            .take(max_len)
            .map(|&m| m as i64)
            .collect();

        let input_ids = Array2::from_shape_vec((1, max_len), input_ids)
            .context("reshape input_ids")?;
        let attention_mask = Array2::from_shape_vec((1, max_len), attention_mask)
            .context("reshape attention_mask")?;
        let input_ids_dyn = input_ids.clone().into_dyn();
        let attention_mask_dyn = attention_mask.clone().into_dyn();

        // ── Step 2: Encoder ──
        let hidden_states = {
            let enc_out = self.encoder.run(ort::inputs![
                "input_ids"      => TensorRef::from_array_view(&input_ids_dyn)?,
                "attention_mask" => TensorRef::from_array_view(&attention_mask_dyn)?,
            ])?;
            enc_out["last_hidden_state"]
                .try_extract_array::<f32>()
                .context("extracting hidden_states")?
                .to_owned()
        };

        // ── Step 3: 自回归解码 ──
        let max_output = self.config.max_output();
        let eos_id = self.config.eos_token_id;
        let mut dec_ids: Vec<i64> = vec![self.config.decoder_start_token_id];
        let mut generated_u32: Vec<u32> = Vec::new();

        for _ in 0..max_output {
            let dec_len = dec_ids.len();
            let dec_input = Array2::from_shape_vec(
                (1, dec_len),
                dec_ids.clone(),
            ).context("reshape dec_input")?;
            let dec_input_dyn = dec_input.into_dyn();
            let attention_mask_dyn = attention_mask.clone().into_dyn();

            let next_id = {
                let dec_out = self.decoder.run(ort::inputs![
                    "decoder_input_ids"      => TensorRef::from_array_view(&dec_input_dyn)?,
                    "encoder_hidden_states"  => TensorRef::from_array_view(&hidden_states)?,
                    "encoder_attention_mask" => TensorRef::from_array_view(&attention_mask_dyn)?,
                ])?;

                let logits = dec_out["logits"]
                    .try_extract_array::<f32>()
                    .context("extracting logits")?
                    .to_owned();
                argmax_last(logits.view())
            };

            if next_id == eos_id as usize {
                break;
            }

            generated_u32.push(next_id as u32);
            dec_ids.push(next_id as i64);

            if self.structured_early_stop && self.should_stop_structured(&generated_u32) {
                break;
            }
        }

        // ── Step 4: 解码 ──
        // Python: tokenizer.decode(dec_ids[0], skip_special_tokens=True)
        // 注意这里 decode 的是完整 dec_ids (包含 decoder_start_token_id)
        let all_u32: Vec<u32> = dec_ids.iter().map(|&id| id as u32).collect();
        let raw_output = self.tokenizer.decode(&all_u32, true)
            .map_err(|e| anyhow::anyhow!("decode: {e}"))?;

        Ok(parse_nlu_output(&raw_output))
    }

    /// JSON 完整性检测，精确复现 Python _should_stop_structured_decode。
    fn should_stop_structured(&self, token_ids: &[u32]) -> bool {
        if token_ids.is_empty() {
            return false;
        }
        let text = match self.tokenizer.decode(token_ids, true) {
            Ok(t) => t,
            Err(_) => return false,
        };
        let text = text.trim();
        if text.is_empty() {
            return false;
        }
        let last = text.chars().last().unwrap();
        if last != '}' && last != ']' {
            return false;
        }
        serde_json::from_str::<serde_json::Value>(text).is_ok()
    }
}

fn resolve_threads(explicit: Option<usize>, cap: usize) -> usize {
    explicit.unwrap_or_else(|| {
        let cores = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);
        cores.saturating_sub(1).clamp(2, cap)
    })
}

fn create_session(path: &Path, threads: usize) -> Result<Session> {
    let builder = Session::builder()
        .map_err(|e| anyhow::anyhow!("creating session builder: {e}"))?;
    let builder = builder
        .with_intra_threads(threads)
        .map_err(|e| anyhow::anyhow!("setting intra_threads: {e}"))?;
    builder
        .with_optimization_level(resolve_optimization_level())
        .map_err(|e| anyhow::anyhow!("setting optimization level: {e}"))?
        .commit_from_file(path)
        .with_context(|| format!("loading ONNX: {}", path.display()))
}

fn resolve_optimization_level() -> GraphOptimizationLevel {
    match std::env::var("VOICE_INFER_ORT_OPT")
        .unwrap_or_else(|_| "level3".into())
        .to_ascii_lowercase()
        .as_str()
    {
        "disable" | "none" | "0" => GraphOptimizationLevel::Disable,
        "level1" | "basic" | "1" => GraphOptimizationLevel::Level1,
        "level2" | "extended" | "2" => GraphOptimizationLevel::Level2,
        "all" => GraphOptimizationLevel::All,
        _ => GraphOptimizationLevel::Level3,
    }
}

/// argmax on last timestep of [batch, seq, vocab] tensor.
fn argmax_last(logits: ndarray::ArrayViewD<'_, f32>) -> usize {
    let shape = logits.shape();
    let vocab = shape[shape.len() - 1];
    let flat = logits.as_slice().unwrap();
    let start = flat.len() - vocab;
    let slice = &flat[start..];
    slice.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::PREFIX;
    use crate::config::NluExportConfig;
    use ndarray::{Array2, Array3};
    use ndarray_npy::ReadNpyExt;
    use std::{io::Cursor, path::PathBuf, sync::OnceLock};
    use tokenizers::{
        PaddingDirection, PaddingParams, PaddingStrategy, Tokenizer, TruncationParams,
        TruncationStrategy,
    };

    #[derive(Debug, serde::Deserialize)]
    struct EncodeReference {
        input_ids: Vec<i64>,
        attention_mask: Vec<i64>,
    }

    static ORT_READY: OnceLock<bool> = OnceLock::new();

    fn workspace_root() -> PathBuf {
        if let Some(path) = std::env::var_os("VOICE_CONTROL_ROOT") {
            return PathBuf::from(path);
        }
        std::env::current_dir()
            .expect("resolve voice-infer test cwd")
            .parent()
            .expect("voice-infer cwd has repo root parent")
            .to_path_buf()
    }

    fn ort_dylib_path() -> Option<PathBuf> {
        std::env::var_os("ORT_DYLIB_PATH")
            .map(PathBuf::from)
            .filter(|path| path.is_file())
    }

    fn init_ort_for_tests() -> bool {
        *ORT_READY.get_or_init(|| {
            let Some(path) = ort_dylib_path() else {
                eprintln!("skip ORT-backed NLU tests: set ORT_DYLIB_PATH to the target ONNX Runtime dynamic library");
                return false;
            };
            eprintln!("NLU ORT test: using ORT_DYLIB_PATH={}", path.display());
            std::env::set_var("ORT_DYLIB_PATH", &path);
            eprintln!("NLU ORT test: initializing ORT");
            let _ = ort::init().commit();
            eprintln!("NLU ORT test: ORT initialized");
            true
        })
    }

    fn load_config() -> NluExportConfig {
        serde_json::from_str(include_str!(
            "../../tests/resources/reference/nlu/export_config.json"
        ))
        .expect("parse NLU export config")
    }

    fn load_tokenizer(config: &NluExportConfig) -> Tokenizer {
        let mut tokenizer =
            Tokenizer::from_bytes(include_bytes!("../../tests/resources/reference/nlu/tokenizer.json"))
                .expect("load NLU tokenizer");
        tokenizer.with_padding(Some(PaddingParams {
            strategy: PaddingStrategy::Fixed(config.max_seq_len),
            direction: PaddingDirection::Right,
            pad_id: config.pad_token_id as u32,
            pad_type_id: 0,
            pad_token: String::from("</s>"),
            pad_to_multiple_of: None,
        }));
        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length: config.max_seq_len,
                strategy: TruncationStrategy::LongestFirst,
                stride: 0,
                direction: tokenizers::TruncationDirection::Right,
            }))
            .expect("set NLU truncation");
        tokenizer
    }

    #[test]
    fn test_encoder_inputs_match_reference() {
        let config = load_config();
        let tokenizer = load_tokenizer(&config);
        let encoding = tokenizer
            .encode(format!("{PREFIX}{}", "向前走三步"), true)
            .expect("encode NLU input");
        let actual_ids: Vec<i64> = encoding
            .get_ids()
            .iter()
            .take(config.max_seq_len)
            .map(|&id| id as i64)
            .collect();
        let actual_mask: Vec<i64> = encoding
            .get_attention_mask()
            .iter()
            .take(config.max_seq_len)
            .map(|&id| id as i64)
            .collect();

        let expected: EncodeReference = serde_json::from_str(include_str!(
            "../../tests/resources/reference/nlu/encode_input.json"
        ))
        .expect("parse NLU encode reference");

        assert_eq!(actual_ids, expected.input_ids);
        assert_eq!(actual_mask, expected.attention_mask);
    }

    #[test]
    #[ignore = "ORT-backed parity test; requires a Rust-compatible ONNX Runtime dynamic library."]
    fn test_encoder_hidden_head_matches_reference() {
        if !init_ort_for_tests() {
            return;
        }

        let config = load_config();
        let reference: EncodeReference = serde_json::from_str(include_str!(
            "../../tests/resources/reference/nlu/encode_input.json"
        ))
        .expect("parse NLU encode reference");
        let input_ids = Array2::from_shape_vec((1, config.max_seq_len), reference.input_ids)
            .expect("shape input_ids");
        let attention_mask =
            Array2::from_shape_vec((1, config.max_seq_len), reference.attention_mask)
                .expect("shape attention_mask");
        let input_ids_dyn = input_ids.into_dyn();
        let attention_mask_dyn = attention_mask.into_dyn();

        let encoder_path = workspace_root().join("models/nlu/encoder.onnx");
        eprintln!("NLU ORT test: loading encoder {}", encoder_path.display());
        let mut encoder = super::create_session(&encoder_path, 2).expect("load NLU encoder");
        eprintln!("NLU ORT test: encoder loaded");
        let actual = {
            eprintln!("NLU ORT test: running encoder");
            let out = encoder
                .run(ort::inputs![
                    "input_ids" => ort::value::TensorRef::from_array_view(&input_ids_dyn).unwrap(),
                    "attention_mask" => ort::value::TensorRef::from_array_view(&attention_mask_dyn).unwrap(),
                ])
                .expect("run NLU encoder");
            eprintln!("NLU ORT test: encoder finished");
            out["last_hidden_state"]
                .try_extract_array::<f32>()
                .expect("extract hidden states")
                .to_owned()
        };
        let expected: Array3<f32> = Array3::read_npy(Cursor::new(include_bytes!(
            "../../tests/resources/reference/nlu/hidden_head.npy"
        )))
        .expect("read hidden_head.npy");
        let expected_shape = expected.shape();
        let mut max_diff = 0.0f32;
        for b in 0..expected_shape[0] {
            for t in 0..expected_shape[1] {
                for d in 0..expected_shape[2] {
                    let diff = (actual[[b, t, d]] - expected[[b, t, d]]).abs();
                    max_diff = max_diff.max(diff);
                }
            }
        }
        assert!(max_diff < 1e-4, "NLU hidden_head max abs diff = {max_diff}");
    }

    #[test]
    #[ignore = "ORT-backed end-to-end test; requires ORT_DYLIB_PATH and local models."]
    fn test_predict_forward_matches_reference() {
        if !init_ort_for_tests() {
            return;
        }

        let root = workspace_root();
        eprintln!("NLU predict test: loading engine");
        let mut engine = super::NluEngine::new(
            &root.join("models/nlu"),
            &root.join("models/nlu/tokenizer"),
            Some(2),
            false,
        )
        .expect("load NLU engine");
        eprintln!("NLU predict test: engine loaded");

        let actual = engine.predict("\u{5411}\u{524d}\u{8d70}\u{4e09}\u{6b65}")
            .expect("run NLU predict");
        let actual = serde_json::to_value(actual).expect("serialize NLU output");
        let cases: Vec<serde_json::Value> = serde_json::from_str(include_str!(
            "../../tests/resources/reference/nlu/predict_cases.json"
        ))
        .expect("parse predict cases");
        let expected = cases
            .iter()
            .find(|case| case["name"] == "predict_forward")
            .expect("find predict_forward case")["parsed"]
            .clone();
        assert_eq!(actual["intent"], expected["intent"]);
    }
}
