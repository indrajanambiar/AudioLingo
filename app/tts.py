"""Optional text-to-speech support using Coqui TTS."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:  # pragma: no cover - optional dependency
    from TTS.api import TTS
except Exception as exc:  # noqa: BLE001
    TTS = None  # type: ignore[assignment]
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


@dataclass
class TTSConfig:
    model_name: str = "tts_models/en/ljspeech/tacotron2-DDC"
    gpu: bool = False


class TextToSpeechEngine:
    """Wrapper around Coqui's TTS interface."""

    def __init__(self, config: Optional[TTSConfig] = None) -> None:
        self.config = config or TTSConfig()
        if TTS is None:
            raise RuntimeError(
                "Coqui TTS is unavailable in this environment. "
                "Install a compatible version or disable the TTS stage."
            ) from _IMPORT_ERROR

        self._engine = TTS(self.config.model_name, gpu=self.config.gpu)

    def synthesize(self, text: str, output_path: Optional[str] = None) -> str:
        if not text.strip():
            raise ValueError("Cannot synthesize empty text.")

        target_path = Path(output_path or "translated_tts.wav")
        target_path.parent.mkdir(parents=True, exist_ok=True)
        self._engine.tts_to_file(text=text, file_path=str(target_path))
        return str(target_path)


__all__ = ["TextToSpeechEngine", "TTSConfig"]

