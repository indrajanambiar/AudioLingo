"""Automatic speech recognition utilities backed by Whisper."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Union

import numpy as np
import torch
import torchaudio
import whisper

AudioSource = Union[str, np.ndarray, torch.Tensor]


@dataclass
class ASRConfig:
    """Configuration for the Whisper transcription backend."""

    model_name: str = "tiny"
    device: str = "cpu"
    language: Optional[str] = None
    sample_rate: int = 16_000


class ASRTranscriber:
    """Simple wrapper around Whisper for CPU-friendly transcription."""

    def __init__(self, config: Optional[ASRConfig] = None) -> None:
        self.config = config or ASRConfig()
        self._model = whisper.load_model(
            self.config.model_name, device=self.config.device
        )

    def _load_audio(self, audio_source: AudioSource) -> np.ndarray:
        if isinstance(audio_source, str):
            waveform, sample_rate = torchaudio.load(audio_source)
        elif isinstance(audio_source, torch.Tensor):
            waveform = audio_source
            sample_rate = self.config.sample_rate
        else:
            waveform = torch.tensor(audio_source)
            sample_rate = self.config.sample_rate

        if waveform.ndim == 2:
            waveform = waveform.mean(dim=0, keepdim=True)

        if sample_rate != self.config.sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, sample_rate, self.config.sample_rate
            )

        return waveform.squeeze(0).numpy()

    def transcribe(
        self, audio_source: AudioSource, language_override: Optional[str] = None
    ) -> Dict[str, Optional[str]]:
        audio = self._load_audio(audio_source)
        result = self._model.transcribe(
            audio=audio, language=language_override or self.config.language
        )
        return {
            "text": result.get("text", "").strip(),
            "detected_language": result.get("language"),
            "segments": result.get("segments", []),
        }


__all__ = ["ASRTranscriber", "ASRConfig"]

