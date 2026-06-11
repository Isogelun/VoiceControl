use anyhow::{Context, Result};
use std::io::Cursor;

const TARGET_SR: u32 = 16000;

/// 从字节读取 WAV 音频，输出 16kHz mono f32 采样。
/// 支持 16-bit PCM 和 32-bit float，自动重采样和声道合并。
pub fn load_audio_from_bytes(data: &[u8]) -> Result<Vec<f32>> {
    let reader = hound::WavReader::new(Cursor::new(data))
        .context("failed to parse WAV header")?;
    let spec = reader.spec();
    let channels = spec.channels as usize;
    let sample_rate = spec.sample_rate;

    let samples_f32 = read_samples_as_f32(reader, spec)?;

    // 多声道 → mono: 逐帧取平均
    let mono = if channels > 1 {
        let n = samples_f32.len() / channels;
        (0..n)
            .map(|i| {
                let mut sum = 0.0f32;
                for ch in 0..channels {
                    sum += samples_f32[i * channels + ch];
                }
                sum / channels as f32
            })
            .collect()
    } else {
        samples_f32
    };

    // 重采样到 16kHz
    if sample_rate == TARGET_SR {
        return Ok(mono);
    }
    resample(&mono, sample_rate, TARGET_SR)
}

fn read_samples_as_f32(
    mut reader: hound::WavReader<Cursor<&[u8]>>,
    spec: hound::WavSpec,
) -> Result<Vec<f32>> {
    match spec.sample_format {
        hound::SampleFormat::Int => {
            let bits = spec.bits_per_sample;
            let max_val = (1u32 << (bits - 1)) as f32;
            reader
                .samples::<i32>()
                .map(|s| {
                    let v = s.context("reading int sample")?;
                    Ok(v as f32 / max_val)
                })
                .collect()
        }
        hound::SampleFormat::Float => {
            reader
                .samples::<f32>()
                .map(|s| s.context("reading float sample"))
                .collect()
        }
    }
}

/// SincFixedIn 重采样 (rubato)。
fn resample(mono: &[f32], from_sr: u32, to_sr: u32) -> Result<Vec<f32>> {
    use rubato::{SincFixedIn, SincInterpolationType, SincInterpolationParameters, WindowFunction, Resampler};

    if mono.is_empty() {
        return Ok(vec![]);
    }

    let params = SincInterpolationParameters {
        sinc_len: 256,
        f_cutoff: 0.95,
        interpolation: SincInterpolationType::Linear,
        oversampling_factor: 256,
        window: WindowFunction::BlackmanHarris2,
    };

    let ratio = to_sr as f64 / from_sr as f64;
    let chunk_size = mono.len().min(1024);
    let mut resampler = SincFixedIn::<f32>::new(
        ratio, 2.0, params, chunk_size, 1,
    ).context("creating resampler")?;

    let mut output = Vec::with_capacity((mono.len() as f64 * ratio) as usize + 1024);
    let mut pos = 0;

    while pos < mono.len() {
        let end = (pos + chunk_size).min(mono.len());
        let mut chunk = mono[pos..end].to_vec();
        // 最后一个 chunk 需要 pad 到 chunk_size
        if chunk.len() < chunk_size {
            chunk.resize(chunk_size, 0.0);
        }
        let result = resampler.process(&[chunk], None)
            .context("resampling chunk")?;
        output.extend_from_slice(&result[0]);
        pos = end;
    }

    // 修剪到期望长度
    let expected = (mono.len() as f64 * ratio).round() as usize;
    output.truncate(expected);

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_wav_bytes(samples: &[f32], sr: u32, channels: u16) -> Vec<u8> {
        let mut buf = Vec::new();
        let mut writer = hound::WavWriter::new(
            Cursor::new(&mut buf),
            hound::WavSpec {
                channels,
                sample_rate: sr,
                bits_per_sample: 32,
                sample_format: hound::SampleFormat::Float,
            },
        ).unwrap();
        for &s in samples {
            writer.write_sample(s).unwrap();
        }
        writer.finalize().unwrap();
        buf
    }

    #[test]
    fn test_16k_mono() {
        let wav = vec![0.5f32; 16000];
        let bytes = make_wav_bytes(&wav, 16000, 1);
        let out = load_audio_from_bytes(&bytes).unwrap();
        assert_eq!(out.len(), 16000);
        assert!((out[0] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_stereo_to_mono() {
        // L=0.2, R=0.8 → mono=0.5
        let mut interleaved = Vec::new();
        for _ in 0..1000 {
            interleaved.push(0.2f32);
            interleaved.push(0.8f32);
        }
        let bytes = make_wav_bytes(&interleaved, 16000, 2);
        let out = load_audio_from_bytes(&bytes).unwrap();
        assert_eq!(out.len(), 1000);
        assert!((out[0] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_resample_8k_to_16k() {
        let wav = vec![0.1f32; 8000]; // 1s at 8kHz
        let bytes = make_wav_bytes(&wav, 8000, 1);
        let out = load_audio_from_bytes(&bytes).unwrap();
        // 8kHz 1s → 16kHz 1s ≈ 16000 samples
        assert!((out.len() as f64 - 16000.0).abs() < 100.0);
    }
}
