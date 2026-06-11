use std::collections::HashSet;
use std::path::Path;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use half::f16;
use memmap2::Mmap;
use ndarray::{Array1, Array2, Array3};
use ort::{
    session::{Session, builder::GraphOptimizationLevel},
    value::TensorRef,
};
use regex::Regex;
use tokenizers::Tokenizer;

use crate::audio::MelFrontend;
use crate::config::AsrModelConfig;

#[derive(Debug, serde::Serialize)]
pub struct AsrResult {
    pub text: String,
    pub feat_ms: f64,
    pub infer_ms: f64,
    pub total_ms: f64,
    pub segments: usize,
}

pub struct AsrEngine {
    encoder: Session,
    decoder_init: Session,
    decoder_step: Session,
    tokenizer: Tokenizer,
    embed_mmap: Mmap,
    mel_frontend: MelFrontend,
    config: AsrModelConfig,
    eos_ids: HashSet<i64>,
    max_new_tokens: usize,
    hidden_size: usize,
    vocab_size: usize,
    // 缓存 prompt 片段（不随输入变化）
    prompt_prefix_ids: Vec<i64>,
    prompt_suffix_ids: Vec<i64>,
    clean_re: Regex,
}

impl AsrEngine {
    pub fn new(model_dir: &Path, num_threads: Option<usize>, use_gpu: bool) -> Result<Self> {
        let config = AsrModelConfig::load(model_dir)?;
        let suffix = detect_model_suffix(model_dir)?;
        validate_model_dir(model_dir, &suffix)?;

        let threads = resolve_threads(num_threads, 8);
        tracing::info!(
            model_dir = %model_dir.display(),
            suffix = %suffix,
            threads,
            "loading ASR model"
        );

        let providers = build_providers(use_gpu);
        let encoder = create_session(&model_dir.join(format!("encoder{suffix}")), threads, &providers)?;
        let decoder_init = create_session(&model_dir.join(format!("decoder_init{suffix}")), threads, &providers)?;
        let decoder_step = create_session(&model_dir.join(format!("decoder_step{suffix}")), threads, &providers)?;

        let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json"))
            .map_err(|e| anyhow::anyhow!("loading ASR tokenizer: {e}"))?;

        let file = std::fs::File::open(model_dir.join("embed_tokens.bin"))
            .context("opening embed_tokens.bin")?;
        let embed_mmap = unsafe { Mmap::map(&file) }.context("mmap embed_tokens.bin")?;

        let mel_frontend = MelFrontend::new(config.mel.clone());

        let eos_ids = config.eos_set();
        let hidden_size = config.decoder.hidden_size;
        let vocab_size = config.decoder.vocab_size;

        let max_new_tokens: usize = std::env::var("QWEN_ASR_MAX_NEW_TOKENS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(32);

        // 预编码 prompt 片段
        let prompt_prefix_ids = encode_token_ids(
            &tokenizer,
            "<|im_start|>system<|im_end|><|im_start|>user<|audio_start|>",
        )?;
        let prompt_suffix_ids = encode_token_ids(
            &tokenizer,
            "<|audio_end|><|im_end|><|im_start|>assistant",
        )?;

        let clean_re = Regex::new(r"<\|[^|]*\|>").unwrap();

        tracing::info!(
            quantization = if suffix.contains("int4") { "int4" } else { "fp32" },
            vocab_size,
            hidden_size,
            max_new_tokens,
            "ASR model loaded"
        );

        Ok(Self {
            encoder, decoder_init, decoder_step, tokenizer,
            embed_mmap, mel_frontend, config, eos_ids, max_new_tokens,
            hidden_size, vocab_size, prompt_prefix_ids, prompt_suffix_ids,
            clean_re,
        })
    }

    pub fn transcribe(&mut self, wav: &[f32]) -> Result<AsrResult> {
        let t0 = Instant::now();

        // ── Step 1: Mel 频谱 ──
        let mel = self.mel_frontend.log_mel(wav);
        let feat_ms = t0.elapsed().as_secs_f64() * 1000.0;

        // ── Step 2: Encoder ──
        let ti = Instant::now();
        let mel_dyn = mel.into_dyn();
        let audio_features = {
            let encoder_out = self.encoder.run(
                ort::inputs![
                    "mel" => TensorRef::from_array_view(&mel_dyn)?
                ]
            )?;
            encoder_out["audio_features"]
                .try_extract_array::<f32>()
                .context("extracting audio_features")?
                .to_owned()
        };
        let audio_len = audio_features.shape()[1];

        // ── Step 3: Build prompt ──
        let (input_ids, audio_offset) = self.build_prompt(audio_len);
        let seq_len = input_ids.shape()[1];
        let position_ids = Array2::from_shape_fn((1, seq_len), |(_, j)| j as i64);
        let input_ids_dyn = input_ids.into_dyn();
        let position_ids_dyn = position_ids.into_dyn();
        let af_owned = audio_features.into_dyn();
        let audio_offset_dyn = audio_offset.into_dyn();

        // ── Step 4: Decoder init ──
        let (mut logits, mut past_keys, mut past_values) = {
            let init_out = self.decoder_init.run(ort::inputs![
                "input_ids"      => TensorRef::from_array_view(&input_ids_dyn)?,
                "position_ids"   => TensorRef::from_array_view(&position_ids_dyn)?,
                "audio_features" => TensorRef::from_array_view(&af_owned)?,
                "audio_offset"   => TensorRef::from_array_view(&audio_offset_dyn)?,
            ])?;

            let logits = init_out["logits"]
                .try_extract_array::<f32>().context("init logits")?
                .to_owned();
            let past_keys = init_out["present_keys"]
                .try_extract_array::<f32>().context("init keys")?
                .to_owned();
            let past_values = init_out["present_values"]
                .try_extract_array::<f32>().context("init values")?
                .to_owned();
            (logits, past_keys, past_values)
        };

        // ── Step 5: 自回归解码 ──
        let mut out_ids: Vec<u32> = Vec::with_capacity(self.max_new_tokens);
        let mut next_id = argmax_last_i64(logits.view());

        for _ in 0..self.max_new_tokens {
            if self.eos_ids.contains(&next_id) {
                break;
            }
            let token_id = next_id as usize;
            out_ids.push(token_id as u32);

            let input_embeds = self.lookup_embedding(token_id)?;
            let step_pos = Array2::from_elem((1, 1), (seq_len + out_ids.len() - 1) as i64);
            let input_embeds_dyn = input_embeds.into_dyn();
            let step_pos_dyn = step_pos.into_dyn();

            let (step_logits, next_past_keys, next_past_values) = {
                let step_out = self.decoder_step.run(ort::inputs![
                    "input_embeds" => TensorRef::from_array_view(&input_embeds_dyn)?,
                    "position_ids" => TensorRef::from_array_view(&step_pos_dyn)?,
                    "past_keys"    => TensorRef::from_array_view(&past_keys)?,
                    "past_values"  => TensorRef::from_array_view(&past_values)?,
                ])?;

                let step_logits = step_out["logits"]
                    .try_extract_array::<f32>().context("step logits")?
                    .to_owned();
                let next_past_keys = step_out["present_keys"]
                    .try_extract_array::<f32>().context("step keys")?
                    .to_owned();
                let next_past_values = step_out["present_values"]
                    .try_extract_array::<f32>().context("step values")?
                    .to_owned();
                (step_logits, next_past_keys, next_past_values)
            };

            past_keys = next_past_keys;
            past_values = next_past_values;
            logits = step_logits;
            next_id = argmax_last_i64(logits.view());
        }

        // ── Step 6: 解码文本 ──
        let text = self.tokenizer
            .decode(&out_ids, true)
            .map_err(|e| anyhow::anyhow!("tokenizer decode: {e}"))?;
        let text = self.clean_text(&text);

        let infer_ms = ti.elapsed().as_secs_f64() * 1000.0;
        let total_ms = t0.elapsed().as_secs_f64() * 1000.0;

        Ok(AsrResult {
            text,
            feat_ms: round1(feat_ms),
            infer_ms: round1(infer_ms),
            total_ms: round1(total_ms),
            segments: 1,
        })
    }

    /// 构造 decoder_init 的输入 prompt。
    /// 返回 (input_ids[1, seq_len], audio_offset[1])。
    fn build_prompt(&self, audio_len: usize) -> (Array2<i64>, Array1<i64>) {
        let pad_id = self.config.special_tokens.audio_pad_token_id;
        let mut ids = Vec::with_capacity(
            self.prompt_prefix_ids.len() + audio_len + self.prompt_suffix_ids.len()
        );
        ids.extend_from_slice(&self.prompt_prefix_ids);
        let audio_offset = ids.len() as i64;
        ids.extend(std::iter::repeat(pad_id).take(audio_len));
        ids.extend_from_slice(&self.prompt_suffix_ids);

        let seq_len = ids.len();
        let input_ids = Array2::from_shape_vec((1, seq_len), ids).unwrap();
        let offset = Array1::from_vec(vec![audio_offset]);
        (input_ids, offset)
    }

    /// 从 memmap 的 f16 embed_tokens.bin 中查找一个 token 的 embedding。
    /// 返回 [1, 1, hidden_size] f32。
    fn lookup_embedding(&self, token_id: usize) -> Result<Array3<f32>> {
        anyhow::ensure!(
            token_id < self.vocab_size,
            "token_id {token_id} >= vocab_size {}",
            self.vocab_size
        );
        let byte_offset = token_id * self.hidden_size * 2; // f16 = 2 bytes
        let byte_end = byte_offset + self.hidden_size * 2;
        anyhow::ensure!(
            byte_end <= self.embed_mmap.len(),
            "embed_tokens.bin too small for token_id {token_id}"
        );
        let bytes = &self.embed_mmap[byte_offset..byte_end];

        // 安全地逐元素读取 f16，避免对齐问题
        let f32_vec: Vec<f32> = (0..self.hidden_size)
            .map(|i| {
                let lo = bytes[i * 2];
                let hi = bytes[i * 2 + 1];
                f16::from_le_bytes([lo, hi]).to_f32()
            })
            .collect();

        Ok(Array3::from_shape_vec((1, 1, self.hidden_size), f32_vec).unwrap())
    }

    /// 去除 ASR 输出中的特殊标记。精确复现 Python clean_text。
    fn clean_text(&self, text: &str) -> String {
        clean_text_impl(&self.clean_re, text)
    }
}

// ─── 辅助函数 ───

fn detect_model_suffix(model_dir: &Path) -> Result<String> {
    if model_dir.join("encoder.int4.onnx").is_file() {
        Ok(".int4.onnx".into())
    } else if model_dir.join("encoder.onnx").is_file() {
        Ok(".onnx".into())
    } else {
        bail!("encoder not found in {}", model_dir.display())
    }
}

fn validate_model_dir(model_dir: &Path, suffix: &str) -> Result<()> {
    let required = [
        "config.json",
        "tokenizer.json",
        "embed_tokens.bin",
        &format!("encoder{suffix}"),
        &format!("decoder_init{suffix}"),
        &format!("decoder_step{suffix}"),
    ];
    let missing: Vec<_> = required.iter()
        .filter(|f| !model_dir.join(f).is_file())
        .collect();
    if !missing.is_empty() {
        bail!("ASR model dir incomplete, missing: {:?}", missing);
    }
    Ok(())
}

fn resolve_threads(explicit: Option<usize>, cap: usize) -> usize {
    explicit.unwrap_or_else(|| {
        let cores = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);
        cores.saturating_sub(1).clamp(2, cap)
    })
}

fn build_providers(_use_gpu: bool) -> Vec<String> {
    // ort 2.x 的 provider 通过 builder 设置，这里只返回名字用于日志
    vec!["CPUExecutionProvider".into()]
}

fn create_session(path: &Path, threads: usize, _providers: &[String]) -> Result<Session> {
    let builder = Session::builder()
        .map_err(|e| anyhow::anyhow!("creating session builder: {e}"))?;
    let builder = builder
        .with_intra_threads(threads)
        .map_err(|e| anyhow::anyhow!("setting intra_threads: {e}"))?;
    builder
        .with_optimization_level(resolve_optimization_level())
        .map_err(|e| anyhow::anyhow!("setting optimization level: {e}"))?
        .commit_from_file(path)
        .with_context(|| format!("loading ONNX model: {}", path.display()))
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

fn encode_token_ids(tokenizer: &Tokenizer, text: &str) -> Result<Vec<i64>> {
    let encoding = tokenizer.encode(text, false)
        .map_err(|e| anyhow::anyhow!("tokenizer encode: {e}"))?;
    Ok(encoding.get_ids().iter().map(|&id| id as i64).collect())
}

#[cfg(test)]
fn lookup_embedding_bytes(
    embed_bytes: &[u8],
    token_id: usize,
    hidden_size: usize,
    vocab_size: usize,
) -> Result<Array3<f32>> {
    anyhow::ensure!(
        token_id < vocab_size,
        "token_id {token_id} >= vocab_size {vocab_size}"
    );
    let byte_offset = token_id * hidden_size * 2;
    let byte_end = byte_offset + hidden_size * 2;
    anyhow::ensure!(
        byte_end <= embed_bytes.len(),
        "embed_tokens.bin too small for token_id {token_id}"
    );
    let bytes = &embed_bytes[byte_offset..byte_end];

    let f32_vec: Vec<f32> = (0..hidden_size)
        .map(|i| {
            let lo = bytes[i * 2];
            let hi = bytes[i * 2 + 1];
            f16::from_le_bytes([lo, hi]).to_f32()
        })
        .collect();

    Ok(Array3::from_shape_vec((1, 1, hidden_size), f32_vec).unwrap())
}

fn clean_text_impl(clean_re: &Regex, text: &str) -> String {
    let text = text.replace("<asr_text>", "");
    let text = clean_re.replace_all(&text, "");
    text.trim().to_string()
}

/// 取 logits tensor 最后一个时间步的 argmax，返回 token id (i64)。
/// logits shape: [batch, seq_len, vocab_size]
fn argmax_last_i64(logits: ndarray::ArrayViewD<'_, f32>) -> i64 {
    let shape = logits.shape();
    let vocab = shape[shape.len() - 1];
    let last_step = logits.as_slice().unwrap();
    // 最后一个时间步在末尾 vocab 个元素
    let start = last_step.len() - vocab;
    let slice = &last_step[start..];
    let mut max_idx = 0usize;
    let mut max_val = f32::NEG_INFINITY;
    for (i, &v) in slice.iter().enumerate() {
        if v > max_val {
            max_val = v;
            max_idx = i;
        }
    }
    max_idx as i64
}

fn round1(v: f64) -> f64 {
    (v * 10.0).round() / 10.0
}

#[cfg(test)]
mod tests {
    use super::{clean_text_impl, encode_token_ids, lookup_embedding_bytes};
    use crate::config::AsrModelConfig;
    use ndarray::{Array1, Array3};
    use ndarray_npy::ReadNpyExt;
    use regex::Regex;
    use std::{
        io::Cursor,
        path::PathBuf,
        sync::{Mutex, OnceLock},
    };
    use tokenizers::Tokenizer;
    use ort::session::Session;

    const PREFIX_TEXT: &str = "<|im_start|>system<|im_end|><|im_start|>user<|audio_start|>";
    const SUFFIX_TEXT: &str = "<|audio_end|><|im_end|><|im_start|>assistant";
    static ORT_READY: OnceLock<bool> = OnceLock::new();
    static ASR_HARNESS: OnceLock<Option<Mutex<AsrTestHarness>>> = OnceLock::new();

    struct AsrTestHarness {
        encoder: Session,
        decoder_init: Session,
        config: AsrModelConfig,
        prompt_prefix_ids: Vec<i64>,
        prompt_suffix_ids: Vec<i64>,
        ref_mel: Array3<f32>,
        ref_audio_features_head: Array3<f32>,
        ref_init_logits_last: Array1<f32>,
    }

    impl AsrTestHarness {
        fn build_prompt(&self, audio_len: usize) -> (ndarray::Array2<i64>, Array1<i64>) {
            let pad_id = self.config.special_tokens.audio_pad_token_id;
            let mut ids = Vec::with_capacity(
                self.prompt_prefix_ids.len() + audio_len + self.prompt_suffix_ids.len(),
            );
            ids.extend_from_slice(&self.prompt_prefix_ids);
            let audio_offset = ids.len() as i64;
            ids.extend(std::iter::repeat(pad_id).take(audio_len));
            ids.extend_from_slice(&self.prompt_suffix_ids);

            let input_ids = ndarray::Array2::from_shape_vec((1, ids.len()), ids).unwrap();
            let offset = Array1::from_vec(vec![audio_offset]);
            (input_ids, offset)
        }
    }

    fn repo_root() -> PathBuf {
        if let Some(path) = std::env::var_os("VOICE_CONTROL_ROOT") {
            return PathBuf::from(path);
        }
        std::env::current_dir()
            .expect("resolve voice-infer test cwd")
            .parent()
            .expect("voice-infer cwd has repo root parent")
            .to_path_buf()
    }

    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f32::max)
    }

    fn ort_dylib_path() -> Option<PathBuf> {
        std::env::var_os("ORT_DYLIB_PATH")
            .map(PathBuf::from)
            .filter(|path| path.is_file())
    }

    fn init_ort_for_tests() -> bool {
        *ORT_READY.get_or_init(|| {
            let Some(path) = ort_dylib_path() else {
                eprintln!("skip ORT-backed ASR tests: set ORT_DYLIB_PATH to the target ONNX Runtime dynamic library");
                return false;
            };
            std::env::set_var("ORT_DYLIB_PATH", &path);
            let _ = ort::init().commit();
            true
        })
    }

    fn load_asr_harness() -> Option<&'static Mutex<AsrTestHarness>> {
        ASR_HARNESS
            .get_or_init(|| {
                if !init_ort_for_tests() {
                    return None;
                }
                let model_dir = repo_root().join("models/asr");
                if !model_dir.join("config.json").is_file() {
                    eprintln!("skip ASR tests: models/asr is missing");
                    return None;
                }

                let suffix = super::detect_model_suffix(&model_dir).expect("detect ASR model suffix");
                let config = AsrModelConfig::load(&model_dir).expect("load ASR config");
                let tokenizer = load_tokenizer();
                let prompt_prefix_ids =
                    encode_token_ids(&tokenizer, PREFIX_TEXT).expect("encode ASR prefix");
                let prompt_suffix_ids =
                    encode_token_ids(&tokenizer, SUFFIX_TEXT).expect("encode ASR suffix");
                let providers = super::build_providers(false);
                let encoder =
                    super::create_session(&model_dir.join(format!("encoder{suffix}")), 2, &providers)
                        .expect("load ASR encoder session");
                let decoder_init = super::create_session(
                    &model_dir.join(format!("decoder_init{suffix}")),
                    2,
                    &providers,
                )
                .expect("load ASR decoder_init session");
                let ref_mel: Array3<f32> = Array3::read_npy(Cursor::new(include_bytes!(
                    "../../tests/resources/reference/mel/full_mel.npy"
                )))
                .expect("read full_mel.npy");
                let ref_audio_features_head: Array3<f32> =
                    Array3::read_npy(Cursor::new(include_bytes!(
                        "../../tests/resources/reference/asr/audio_features_head.npy"
                    )))
                    .expect("read audio_features_head.npy");
                let ref_init_logits_last: Array1<f32> =
                    Array1::read_npy(Cursor::new(include_bytes!(
                        "../../tests/resources/reference/asr/init_logits_last.npy"
                    )))
                    .expect("read init_logits_last.npy");
                Some(Mutex::new(AsrTestHarness {
                    encoder,
                    decoder_init,
                    config,
                    prompt_prefix_ids,
                    prompt_suffix_ids,
                    ref_mel,
                    ref_audio_features_head,
                    ref_init_logits_last,
                }))
            })
            .as_ref()
    }

    fn load_tokenizer() -> Tokenizer {
        Tokenizer::from_bytes(include_bytes!("../../tests/resources/reference/asr/tokenizer.json"))
            .expect("load ASR tokenizer")
    }

    #[test]
    fn test_prompt_prefix_ids_match_reference() {
        let tokenizer = load_tokenizer();
        let actual = encode_token_ids(&tokenizer, PREFIX_TEXT).expect("encode prefix");
        let expected: Vec<i64> = serde_json::from_str(include_str!(
            "../../tests/resources/reference/asr/prompt_prefix_ids.json"
        ))
        .expect("parse prefix reference");
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_prompt_suffix_ids_match_reference() {
        let tokenizer = load_tokenizer();
        let actual = encode_token_ids(&tokenizer, SUFFIX_TEXT).expect("encode suffix");
        let expected: Vec<i64> = serde_json::from_str(include_str!(
            "../../tests/resources/reference/asr/prompt_suffix_ids.json"
        ))
        .expect("parse suffix reference");
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_clean_text_removes_special_markers() {
        let clean_re = Regex::new(r"<\|[^|]*\|>").unwrap();
        let actual = clean_text_impl(
            &clean_re,
            "  <asr_text><|im_start|>你好，世界<|im_end|><|endoftext|>  ",
        );
        assert_eq!(actual, "你好，世界");
    }

    #[test]
    fn test_lookup_embedding_matches_reference() {
        let model_dir = repo_root().join("models/asr");
        if !model_dir.join("embed_tokens.bin").is_file() {
            eprintln!("skip embedding reference test: embed_tokens.bin is missing");
            return;
        }
        let config = AsrModelConfig::load(&model_dir).expect("load ASR config");
        let embed_bytes = std::fs::read(model_dir.join("embed_tokens.bin"))
            .expect("read embed_tokens.bin");
        let actual0 = lookup_embedding_bytes(
            &embed_bytes,
            0,
            config.decoder.hidden_size,
            config.decoder.vocab_size,
        )
        .expect("lookup token 0");
        let actual100 = lookup_embedding_bytes(
            &embed_bytes,
            100,
            config.decoder.hidden_size,
            config.decoder.vocab_size,
        )
        .expect("lookup token 100");
        let actual0 = actual0.as_slice().unwrap();
        let actual100 = actual100.as_slice().unwrap();
        let expected0: Array1<f32> = Array1::read_npy(Cursor::new(include_bytes!(
            "../../tests/resources/reference/asr/embed_token0.npy"
        )))
        .expect("read embed_token0.npy");
        let expected100: Array1<f32> = Array1::read_npy(Cursor::new(include_bytes!(
            "../../tests/resources/reference/asr/embed_token100.npy"
        )))
        .expect("read embed_token100.npy");

        assert!(max_abs_diff(actual0, expected0.as_slice().unwrap()) < 1e-6);
        assert!(max_abs_diff(actual100, expected100.as_slice().unwrap()) < 1e-6);
    }

    #[test]
    #[ignore = "ORT-backed parity test; loads large ASR sessions and is intended for manual validation."]
    fn test_encoder_audio_features_head_matches_reference() {
        let Some(harness) = load_asr_harness() else {
            return;
        };
        let mut harness = harness.lock().unwrap();
        let mel_dyn = harness.ref_mel.clone().into_dyn();
        let actual = {
            let out = harness
                .encoder
                .run(ort::inputs!["mel" => ort::value::TensorRef::from_array_view(&mel_dyn).unwrap()])
                .expect("run ASR encoder");
            out["audio_features"]
                .try_extract_array::<f32>()
                .expect("extract audio_features")
                .to_owned()
        };

        let actual_slice = &actual.as_slice().unwrap()[..harness.ref_audio_features_head.len()];
        let expected_slice = harness.ref_audio_features_head.as_slice().unwrap();
        let max_diff = max_abs_diff(actual_slice, expected_slice);
        assert!(max_diff < 1e-4, "audio_features head max abs diff = {max_diff}");
    }

    #[test]
    #[ignore = "ORT-backed parity test; loads large ASR sessions and is intended for manual validation."]
    fn test_decoder_init_logits_last_matches_reference() {
        let Some(harness) = load_asr_harness() else {
            return;
        };
        let mut harness = harness.lock().unwrap();
        let mel_dyn = harness.ref_mel.clone().into_dyn();
        let audio_features = {
            let out = harness
                .encoder
                .run(ort::inputs!["mel" => ort::value::TensorRef::from_array_view(&mel_dyn).unwrap()])
                .expect("run ASR encoder");
            out["audio_features"]
                .try_extract_array::<f32>()
                .expect("extract audio_features")
                .to_owned()
        };

        let audio_len = audio_features.shape()[1];
        let (input_ids, audio_offset) = harness.build_prompt(audio_len);
        let seq_len = input_ids.shape()[1];
        let position_ids = ndarray::Array2::from_shape_fn((1, seq_len), |(_, j)| j as i64);
        let input_ids_dyn = input_ids.into_dyn();
        let position_ids_dyn = position_ids.into_dyn();
        let audio_features_dyn = audio_features.into_dyn();
        let audio_offset_dyn = audio_offset.into_dyn();

        let actual = {
            let out = harness
                .decoder_init
                .run(ort::inputs![
                    "input_ids" => ort::value::TensorRef::from_array_view(&input_ids_dyn).unwrap(),
                    "position_ids" => ort::value::TensorRef::from_array_view(&position_ids_dyn).unwrap(),
                    "audio_features" => ort::value::TensorRef::from_array_view(&audio_features_dyn).unwrap(),
                    "audio_offset" => ort::value::TensorRef::from_array_view(&audio_offset_dyn).unwrap(),
                ])
                .expect("run ASR decoder_init");
            out["logits"]
                .try_extract_array::<f32>()
                .expect("extract init logits")
                .to_owned()
        };

        let vocab = harness.ref_init_logits_last.len();
        let actual_slice = &actual.as_slice().unwrap()[actual.len() - vocab..];
        let expected_slice = harness.ref_init_logits_last.as_slice().unwrap();
        let max_diff = max_abs_diff(actual_slice, expected_slice);
        assert!(max_diff < 1e-4, "decoder_init logits max abs diff = {max_diff}");
    }
}
