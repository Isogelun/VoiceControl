"""
Lightweight real-time microphone preprocessing.

This is not neural noise suppression. It is a conservative front-end that
reduces steady room noise before VAD/ASR: DC removal, high-pass filtering,
startup noise-floor estimation, soft noise gate, and optional gain.
"""

import logging
import os

import numpy as np

log = logging.getLogger(__name__)


class AudioPreprocessor:
    def __init__(
        self,
        sample_rate: int = 16000,
        frame_samples: int = 480,
        enabled: bool = True,
        gain: float = 1.0,
        calibration_seconds: float = 1.0,
        gate_multiplier: float = 2.5,
        gate_min_rms: float = 120.0,
        gate_attenuation: float = 0.15,
        highpass_alpha: float = 0.97,
    ):
        self.sample_rate = sample_rate
        self.frame_samples = frame_samples
        self.enabled = enabled
        self.gain = gain
        self.calibration_frames = max(1, int(calibration_seconds * sample_rate / frame_samples))
        self.gate_multiplier = gate_multiplier
        self.gate_min_rms = gate_min_rms
        self.gate_attenuation = gate_attenuation
        self.highpass_alpha = highpass_alpha

        self._frames_seen = 0
        self._noise_rms = 0.0
        self._prev_x = 0.0
        self._prev_y = 0.0
        self.last_raw_rms = 0.0
        self.last_processed_rms = 0.0
        self.last_threshold = gate_min_rms

    @classmethod
    def from_env(cls, sample_rate: int = 16000, frame_samples: int = 480):
        return cls(
            sample_rate=sample_rate,
            frame_samples=frame_samples,
            enabled=os.environ.get("AUDIO_DENOISE", "0") not in {"0", "false", "False", "no"},
            gain=float(os.environ.get("MIC_GAIN", "1.0")),
            calibration_seconds=float(os.environ.get("NOISE_CALIBRATION_SECONDS", "1.0")),
            gate_multiplier=float(os.environ.get("NOISE_GATE_MULTIPLIER", "2.5")),
            gate_min_rms=float(os.environ.get("NOISE_GATE_MIN_RMS", "120")),
            gate_attenuation=float(os.environ.get("NOISE_GATE_ATTENUATION", "0.15")),
        )

    def process(self, pcm: np.ndarray) -> np.ndarray:
        pcm = np.asarray(pcm, dtype=np.int16).reshape(-1)
        if not self.enabled:
            self.last_raw_rms = _rms(pcm)
            self.last_processed_rms = self.last_raw_rms
            return pcm

        x = pcm.astype(np.float32)
        self.last_raw_rms = _rms(x)

        x = x - float(np.mean(x))
        x = self._highpass(x)

        hp_rms = _rms(x)
        self._update_noise_floor(hp_rms)
        threshold = max(self.gate_min_rms, self._noise_rms * self.gate_multiplier)
        self.last_threshold = threshold

        # During calibration, keep audio flowing but attenuate it. That avoids
        # false wakeups while the system learns the room noise floor.
        if self._frames_seen <= self.calibration_frames:
            x *= self.gate_attenuation
        elif hp_rms < threshold:
            x *= self.gate_attenuation
        else:
            x *= self.gain

        x = np.clip(x, -32768, 32767)
        out = x.astype(np.int16)
        self.last_processed_rms = _rms(out)
        return out

    @property
    def calibrating(self) -> bool:
        return self.enabled and self._frames_seen <= self.calibration_frames

    def _update_noise_floor(self, rms: float):
        self._frames_seen += 1
        if self._frames_seen == 1:
            self._noise_rms = rms
            return

        if self._frames_seen <= self.calibration_frames:
            # Running average during startup.
            n = self._frames_seen
            self._noise_rms = ((self._noise_rms * (n - 1)) + rms) / n
        elif rms < self.last_threshold:
            # Slowly adapt downward/sideways when we are likely seeing noise.
            self._noise_rms = 0.995 * self._noise_rms + 0.005 * rms

    def _highpass(self, x: np.ndarray) -> np.ndarray:
        y = np.empty_like(x)
        prev_x = self._prev_x
        prev_y = self._prev_y
        alpha = self.highpass_alpha
        for i, sample in enumerate(x):
            value = sample - prev_x + alpha * prev_y
            y[i] = value
            prev_x = sample
            prev_y = value
        self._prev_x = float(prev_x)
        self._prev_y = float(prev_y)
        return y


def _rms(x) -> float:
    arr = np.asarray(x, dtype=np.float32)
    if arr.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(arr * arr)))
