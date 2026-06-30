from __future__ import annotations

import tempfile
from array import array
from dataclasses import dataclass
from functools import lru_cache
from math import sqrt
from pathlib import Path
from sys import byteorder

from voice_agent.config import LOCAL_VAD_PROVIDERS


def validate_vad_provider(provider: str) -> str:
    if provider not in LOCAL_VAD_PROVIDERS:
        raise ValueError(f"Unsupported VAD provider: {provider}")
    return provider


@dataclass(frozen=True)
class VadDecision:
    is_speech: bool
    confidence: float
    provider: str


@dataclass(frozen=True)
class EnergyVad:
    threshold: float = 0.018

    def analyze_pcm16(self, frame: bytes) -> VadDecision:
        if not frame:
            return VadDecision(False, 0.0, "energy")
        samples = array("h")
        samples.frombytes(frame)
        if byteorder == "big":
            samples.byteswap()
        if not samples:
            return VadDecision(False, 0.0, "energy")
        rms = sqrt(sum(sample * sample for sample in samples) / len(samples)) / 32768.0
        confidence = min(1.0, rms / max(self.threshold, 0.0001))
        return VadDecision(rms >= self.threshold, confidence, "energy")


@dataclass(frozen=True)
class SileroVad:
    threshold: float = 0.5
    sampling_rate: int = 16000

    async def analyze_audio(self, audio_bytes: bytes, filename: str, media_type: str) -> VadDecision:
        if not audio_bytes:
            return VadDecision(False, 0.0, "silero")
        suffix = Path(filename).suffix or ".webm"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
            handle.write(audio_bytes)
            temp_path = Path(handle.name)
        try:
            model = _get_silero_model()
            get_speech_timestamps = _get_silero_utils()
            waveform = _decode_audio_for_silero(temp_path, self.sampling_rate)
            timestamps = get_speech_timestamps(
                waveform,
                model,
                threshold=self.threshold,
                sampling_rate=self.sampling_rate,
            )
        finally:
            temp_path.unlink(missing_ok=True)
        return VadDecision(bool(timestamps), 1.0 if timestamps else 0.0, "silero")


@lru_cache(maxsize=1)
def _get_silero_model():
    try:
        from silero_vad import load_silero_vad
    except ImportError as exc:
        raise RuntimeError("Install silero-vad to use Silero VAD") from exc
    return load_silero_vad()


@lru_cache(maxsize=1)
def _get_silero_utils():
    try:
        from silero_vad import get_speech_timestamps
    except ImportError as exc:
        raise RuntimeError("Install silero-vad to use Silero VAD") from exc
    return get_speech_timestamps


def _decode_audio_for_silero(path: Path, sampling_rate: int):
    try:
        import av
        import numpy as np
        import torch
    except ImportError as exc:
        raise RuntimeError("Install PyAV, numpy, and torch to decode audio for Silero VAD") from exc

    chunks: list[np.ndarray] = []
    with av.open(str(path)) as container:
        resampler = av.AudioResampler(format="s16", layout="mono", rate=sampling_rate)
        for frame in container.decode(audio=0):
            for resampled in resampler.resample(frame):
                samples = resampled.to_ndarray().reshape(-1).astype(np.float32) / 32768.0
                chunks.append(samples)
        for resampled in resampler.resample(None):
            samples = resampled.to_ndarray().reshape(-1).astype(np.float32) / 32768.0
            chunks.append(samples)
    if not chunks:
        raise RuntimeError("No audio frames decoded for Silero VAD")
    return torch.from_numpy(np.concatenate(chunks))


def warm_silero_vad() -> None:
    _get_silero_model()
    _get_silero_utils()


@dataclass(frozen=True)
class VADProviderPlan:
    primary: str
    fallback: str


def plan_vad_provider(primary: str) -> VADProviderPlan:
    validate_vad_provider(primary)
    fallback = "energy" if primary != "energy" else "webrtcvad"
    return VADProviderPlan(primary=primary, fallback=fallback)
