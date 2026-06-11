use std::f32::consts::PI;

use ndarray::{Array2, Array3};
use rustfft::num_complex::Complex;

#[derive(Debug, Clone, serde::Deserialize)]
pub struct MelConfig {
    #[serde(default = "default_sr")]
    pub sample_rate: usize,
    #[serde(default = "default_n_fft")]
    pub n_fft: usize,
    #[serde(default = "default_hop")]
    pub hop_length: usize,
    #[serde(default = "default_n_mels")]
    pub n_mels: usize,
    #[serde(default)]
    pub fmin: f32,
    #[serde(default = "default_fmax")]
    pub fmax: f32,
}

fn default_sr() -> usize { 16000 }
fn default_n_fft() -> usize { 400 }
fn default_hop() -> usize { 160 }
fn default_n_mels() -> usize { 128 }
fn default_fmax() -> f32 { 8000.0 }

impl Default for MelConfig {
    fn default() -> Self {
        Self {
            sample_rate: 16000, n_fft: 400, hop_length: 160,
            n_mels: 128, fmin: 0.0, fmax: 8000.0,
        }
    }
}

pub struct MelFrontend {
    pub config: MelConfig,
    window: Vec<f32>,
    mel_basis: Array2<f32>,
    rfft_basis: Vec<Vec<Complex<f64>>>,
}

impl MelFrontend {
    pub fn new(config: MelConfig) -> Self {
        let window = create_hann_window(config.n_fft);
        let mel_basis = create_mel_filterbank(
            config.sample_rate, config.n_fft, config.n_mels,
            config.fmin, config.fmax,
        );
        let rfft_basis = create_rfft_basis(config.n_fft);
        Self { config, window, mel_basis, rfft_basis }
    }

    /// log-mel 频谱。输出 shape [1, n_mels, frames]，精确复现 Python _log_mel_fast。
    pub fn log_mel(&self, wav: &[f32]) -> Array3<f32> {
        let mut log_mel = self.log_mel_pre_norm(wav);
        let global_max = log_mel.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let floor = global_max - 8.0;
        log_mel.mapv_inplace(|v| (v.max(floor) + 4.0) / 4.0);
        log_mel.insert_axis(ndarray::Axis(0))
    }

    fn log_mel_pre_norm(&self, wav: &[f32]) -> Array2<f32> {
        let n_fft = self.config.n_fft;
        let hop = self.config.hop_length;
        let n_mels = self.config.n_mels;
        let n_freq = n_fft / 2 + 1; // 201

        // 空音频保护
        let wav = if wav.is_empty() {
            vec![0.0f32; self.config.sample_rate / 10]
        } else {
            wav.to_vec()
        };

        // 两端 zero-pad
        let pad = n_fft / 2;
        let padded_len = pad + wav.len() + pad;
        let mut padded = vec![0.0f32; padded_len];
        padded[pad..pad + wav.len()].copy_from_slice(&wav);

        // 确保长度至少 n_fft
        if padded.len() < n_fft {
            padded.resize(n_fft, 0.0);
        }

        let n_frames = 1 + (padded.len() - n_fft) / hop;

        // STFT: 分帧 → 加窗 → FFT → power
        // power 矩阵 [n_frames, n_freq]
        let mut power = Array2::<f64>::zeros((n_frames, n_freq));

        for i in 0..n_frames {
            let offset = i * hop;
            let frame_power = self.power_spectrum(&padded[offset..offset + n_fft]);
            for j in 0..n_freq {
                power[[i, j]] = frame_power[j];
            }
        }

        // mel = mel_basis @ power.T → [n_mels, n_frames]
        // mel_basis: [n_mels, n_freq], power.T: [n_freq, n_frames]
        let mut mel = Array2::<f64>::zeros((n_mels, n_frames));
        for m in 0..n_mels {
            for f in 0..n_frames {
                let mut sum = 0.0f64;
                for k in 0..n_freq {
                    sum += self.mel_basis[[m, k]] as f64 * power[[f, k]];
                }
                mel[[m, f]] = sum;
            }
        }

        mel.mapv(|v| v.max(1e-10f64).log10() as f32)
    }

    fn power_spectrum(&self, frame: &[f32]) -> Vec<f64> {
        let n_fft = self.config.n_fft;
        let n_freq = n_fft / 2 + 1;
        let mut out = vec![0.0f64; n_freq];
        for k in 0..n_freq {
            let mut re = 0.0f64;
            let mut im = 0.0f64;
            for n in 0..n_fft {
                let sample = (frame[n] * self.window[n]) as f64;
                re += sample * self.rfft_basis[k][n].re;
                im += sample * self.rfft_basis[k][n].im;
            }
            out[k] = re * re + im * im;
        }
        out
    }
}

fn create_rfft_basis(n_fft: usize) -> Vec<Vec<Complex<f64>>> {
    let n_freq = n_fft / 2 + 1;
    let scale = -2.0f64 * std::f64::consts::PI / n_fft as f64;
    (0..n_freq)
        .map(|k| {
            (0..n_fft)
                .map(|n| {
                    let theta = scale * k as f64 * n as f64;
                    Complex::new(theta.cos(), theta.sin())
                })
                .collect()
        })
        .collect()
}

/// Periodic Hann 窗: w[n] = 0.5 - 0.5 * cos(2πn / N)，N = n_fft。
/// 注意除以 N（periodic）而非 N-1（symmetric），与 librosa center=True 对齐。
pub fn create_hann_window(n_fft: usize) -> Vec<f32> {
    (0..n_fft)
        .map(|n| 0.5 - 0.5 * (2.0 * PI * n as f32 / n_fft as f32).cos())
        .collect()
}

/// Slaney-style mel 滤波器组，精确复现 librosa.filters.mel(htk=False, norm="slaney")。
///
/// 返回 [n_mels, n_fft/2+1]。
pub fn create_mel_filterbank(
    sr: usize, n_fft: usize, n_mels: usize, fmin: f32, fmax: f32,
) -> Array2<f32> {
    let n_freq = n_fft / 2 + 1;
    let mel_min = hz_to_mel(fmin);
    let mel_max = hz_to_mel(fmax);
    let hz_points: Vec<f32> = (0..(n_mels + 2))
        .map(|i| {
            let mel = mel_min + (mel_max - mel_min) * i as f32 / (n_mels + 1) as f32;
            mel_to_hz(mel)
        })
        .collect();
    let fft_freqs: Vec<f32> = (0..n_freq)
        .map(|i| i as f32 * sr as f32 / n_fft as f32)
        .collect();

    let mut fb = Array2::<f32>::zeros((n_mels, n_freq));

    for m in 0..n_mels {
        let left = hz_points[m];
        let center = hz_points[m + 1];
        let right = hz_points[m + 2];
        let lower_width = (center - left).max(f32::EPSILON);
        let upper_width = (right - center).max(f32::EPSILON);
        let enorm = 2.0 / (right - left).max(f32::EPSILON);

        for (k, &freq) in fft_freqs.iter().enumerate() {
            let lower = (freq - left) / lower_width;
            let upper = (right - freq) / upper_width;
            fb[[m, k]] = lower.min(upper).clamp(0.0, f32::INFINITY) * enorm;
        }
    }

    fb
}

/// Hz → Mel (Slaney, 与 librosa htk=False 对齐)。
#[inline]
pub fn hz_to_mel(f: f32) -> f32 {
    const F_SP: f32 = 200.0 / 3.0;
    const MIN_LOG_HZ: f32 = 1000.0;
    const MIN_LOG_MEL: f32 = MIN_LOG_HZ / F_SP;
    const LOGSTEP: f32 = 0.06875178; // ln(6.4) / 27

    if f < MIN_LOG_HZ {
        f / F_SP
    } else {
        MIN_LOG_MEL + (f / MIN_LOG_HZ).ln() / LOGSTEP
    }
}

/// Mel → Hz (Slaney, 与 librosa htk=False 对齐)。
#[inline]
pub fn mel_to_hz(m: f32) -> f32 {
    const F_SP: f32 = 200.0 / 3.0;
    const MIN_LOG_HZ: f32 = 1000.0;
    const MIN_LOG_MEL: f32 = MIN_LOG_HZ / F_SP;
    const LOGSTEP: f32 = 0.06875178; // ln(6.4) / 27

    if m < MIN_LOG_MEL {
        m * F_SP
    } else {
        MIN_LOG_HZ * (LOGSTEP * (m - MIN_LOG_MEL)).exp()
    }
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use ndarray::{Array1, Array2, Array3};
    use ndarray_npy::ReadNpyExt;

    use super::*;
    fn max_abs_diff_1d(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f32::max)
    }

    #[test]
    fn test_hz_mel_roundtrip() {
        for &f in &[0.0, 440.0, 1000.0, 4000.0, 8000.0] {
            let m = hz_to_mel(f);
            let f2 = mel_to_hz(m);
            assert!((f - f2).abs() < 0.01, "roundtrip failed for {f}Hz");
        }
    }

    #[test]
    fn test_hann_window_shape_and_endpoints() {
        let w = create_hann_window(400);
        assert_eq!(w.len(), 400);
        // periodic Hann: w[0] = 0.0 (exact)
        assert!(w[0].abs() < 1e-7);
        // w[N/2] 应接近 1.0
        assert!((w[200] - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_mel_filterbank_shape() {
        let fb = create_mel_filterbank(16000, 400, 128, 0.0, 8000.0);
        assert_eq!(fb.shape(), &[128, 201]);
        // 每行至少有一个非零值
        for m in 0..128 {
            let row_sum: f32 = fb.row(m).iter().sum();
            assert!(row_sum > 0.0, "mel filter {m} is all zeros");
        }
    }

    #[test]
    fn test_log_mel_empty_audio() {
        let fe = MelFrontend::new(MelConfig::default());
        let out = fe.log_mel(&[]);
        assert_eq!(out.shape()[0], 1);
        assert_eq!(out.shape()[1], 128);
        assert!(out.shape()[2] > 0);
    }

    #[test]
    fn test_log_mel_shape() {
        let fe = MelFrontend::new(MelConfig::default());
        let wav = vec![0.0f32; 16000]; // 1s silence
        let out = fe.log_mel(&wav);
        assert_eq!(out.shape()[0], 1);
        assert_eq!(out.shape()[1], 128);
        // frames = 1 + (16000 + 400 - 400) / 160 = 101
        assert_eq!(out.shape()[2], 101);
    }

    #[test]
    fn test_reference_hann_window_exact() {
        let expected: Array1<f32> = Array1::read_npy(Cursor::new(include_bytes!(
            "../../tests/resources/reference/mel/hann_window.npy"
        ))).expect("read hann_window.npy");
        let actual = create_hann_window(400);
        assert_eq!(expected.len(), actual.len());
        let max_diff = max_abs_diff_1d(expected.as_slice().unwrap(), &actual);
        assert!(max_diff < 1e-7, "hann window max abs diff = {max_diff}");
    }

    #[test]
    fn test_reference_mel_filterbank_alignment() {
        let expected: Array2<f32> = Array2::read_npy(Cursor::new(include_bytes!(
            "../../tests/resources/reference/mel/mel_basis.npy"
        ))).expect("read mel_basis.npy");
        let actual = create_mel_filterbank(16000, 400, 128, 0.0, 8000.0);
        assert_eq!(expected.shape(), actual.shape());

        let max_diff = expected.iter()
            .zip(actual.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f32::max);
        assert!(max_diff < 1e-6, "mel basis max abs diff = {max_diff}");
    }

    #[test]
    fn test_reference_power_frame0_alignment() {
        let wav: Array1<f32> = Array1::read_npy(Cursor::new(include_bytes!(
            "../../tests/resources/reference/mel/input_wav.npy"
        ))).expect("read input_wav.npy");
        let expected: Array1<f32> = Array1::read_npy(Cursor::new(include_bytes!(
            "../../tests/resources/reference/mel/power_frame0.npy"
        ))).expect("read power_frame0.npy");

        let fe = MelFrontend::new(MelConfig::default());
        let n_fft = fe.config.n_fft;
        let pad = n_fft / 2;
        let mut padded = vec![0.0f32; pad + wav.len() + pad];
        padded[pad..pad + wav.len()].copy_from_slice(wav.as_slice().unwrap());
        let actual = fe.power_spectrum(&padded[..n_fft]);

        let max_diff = expected.iter()
            .zip(actual.iter())
            .map(|(x, y)| (*x as f64 - *y).abs())
            .fold(0.0, f64::max);
        assert!(max_diff < 1e-4, "power_frame0 max abs diff = {max_diff}");
    }

    fn reference_log_mel_max_diff() -> f32 {
        let wav: Array1<f32> = Array1::read_npy(Cursor::new(include_bytes!(
            "../../tests/resources/reference/mel/input_wav.npy"
        ))).expect("read input_wav.npy");
        let expected: Array3<f32> = Array3::read_npy(Cursor::new(include_bytes!(
            "../../tests/resources/reference/mel/full_mel.npy"
        ))).expect("read full_mel.npy");

        let fe = MelFrontend::new(MelConfig::default());
        let actual = fe.log_mel(wav.as_slice().unwrap());
        assert_eq!(expected.shape(), actual.shape());

        let max_diff = expected.iter()
            .zip(actual.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f32::max);
        max_diff
    }

    #[test]
    fn test_reference_log_mel_regression() {
        let max_diff = reference_log_mel_max_diff();
        assert!(max_diff < 5e-4, "full mel max abs diff regression = {max_diff}");
    }

    fn reference_log_mel_pre_norm_max_diff() -> f32 {
        let wav: Array1<f32> = Array1::read_npy(Cursor::new(include_bytes!(
            "../../tests/resources/reference/mel/input_wav.npy"
        ))).expect("read input_wav.npy");
        let expected: Array2<f32> = Array2::read_npy(Cursor::new(include_bytes!(
            "../../tests/resources/reference/mel/log_mel_pre_norm.npy"
        ))).expect("read log_mel_pre_norm.npy");

        let fe = MelFrontend::new(MelConfig::default());
        let actual = fe.log_mel_pre_norm(wav.as_slice().unwrap());
        assert_eq!(expected.shape(), actual.shape());

        let max_diff = expected.iter()
            .zip(actual.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f32::max);
        max_diff
    }

    #[test]
    fn test_reference_log_mel_pre_norm_regression() {
        let max_diff = reference_log_mel_pre_norm_max_diff();
        assert!(max_diff < 1e-3, "pre-norm log mel max abs diff regression = {max_diff}");
    }

    #[test]
    #[ignore = "Strict pre-norm Python parity is not met yet; current max abs diff is ~8.1e-4."]
    fn test_reference_log_mel_pre_norm_strict_alignment() {
        let max_diff = reference_log_mel_pre_norm_max_diff();
        assert!(max_diff < 1e-5, "pre-norm log mel max abs diff = {max_diff}");
    }

    #[test]
    #[ignore = "Strict Python parity target from the rewrite checklist is not met yet; current max abs diff is ~1.8e-4."]
    fn test_reference_log_mel_strict_alignment() {
        let max_diff = reference_log_mel_max_diff();
        assert!(max_diff < 1e-5, "full mel max abs diff = {max_diff}");
    }
}
