"""End-to-end AudioLingo pipeline orchestration."""

from __future__ import annotations

import time
import logging
from dataclasses import dataclass, field

# Initialize logger
logger = logging.getLogger(__name__)
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

from .asr_transcription import ASRConfig, ASRTranscriber, AudioSource
from .language_detection import LanguageDetectionConfig, LanguageDetector
from .translator import TranslationConfig, Translator
from .tts import TTSConfig, TextToSpeechEngine


@dataclass
class PipelineConfig:
    """Aggregate configuration for the pipeline."""

    asr: ASRConfig = ASRConfig()
    language_detection: LanguageDetectionConfig = LanguageDetectionConfig()
    translation: TranslationConfig = TranslationConfig()
    tts: Optional[TTSConfig] = TTSConfig()
    enable_tts: bool = False


class AudioLingoPipeline:
    """High-level helper that chains ASR e detection e translation e TTS."""

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        asr: Optional[ASRTranscriber] = None,
        detector: Optional[LanguageDetector] = None,
        translator: Optional[Translator] = None,
        tts_engine: Optional[TextToSpeechEngine] = None,
    ) -> None:
        self.config = config or PipelineConfig()
        self.asr = asr or ASRTranscriber(self.config.asr)
        self.detector = detector or LanguageDetector(self.config.language_detection)
        self.translator = translator or Translator(self.config.translation)
        # Do not eagerly initialize TTS here. We lazily create the engine
        # the first time a request actually asks for TTS inside run().
        # This avoids long startup time and allows the UI checkbox to
        # genuinely disable TTS.
        self.tts_engine = tts_engine

    @staticmethod
    def _normalize_targets(targets: Union[str, Sequence[str]]) -> List[str]:
        if isinstance(targets, str):
            return [targets]
        return list(targets)

    def run(
        self,
        audio_source: AudioSource,
        target_languages: Union[str, Sequence[str]],
        enable_tts: Optional[bool] = None,
    ) -> Dict[str, object]:
        logger = logging.getLogger(__name__)
        start_time = time.time()
        
        target_langs = self._normalize_targets(target_languages)
        
        # ASR
        asr_start = time.time()
        asr_result = self.asr.transcribe(audio_source)
        logger.info(f"ASR took {time.time() - asr_start:.2f}s")
        
        transcription = asr_result["text"]
        detected_lang = asr_result.get("detected_language")
        
        # Language Detection
        if not detected_lang:
            detection_start = time.time()
            detection_result = self.detector.detect(transcription)
            logger.info(f"Language detection took {time.time() - detection_start:.2f}s")
            detected_lang = detection_result["language"]
        else:
            detection_result = {
                "language": detected_lang,
                "confidence": 1.0,
                "top_k": [{"language": detected_lang, "confidence": 1.0}],
            }

        # For this simple UI we only support English -> target translations.
        # Some direct language pairs (e.g. hi->ml) do not have MarianMT models
        # like "Helsinki-NLP/opus-mt-hi-ml". To avoid errors like the one you
        # saw, we force the source language to English here so that models such
        # as "opus-mt-en-ml" are used. This means the input speech should be in
        # English for best results.
        if detected_lang != "en":
            logger.info(
                "Overriding detected source language '%s' to 'en' for translation models",
                detected_lang,
            )
            detected_lang = "en"

        # Translation
        translation_start = time.time()
        translations = []
        tts_outputs = []

        # Decide whether to use TTS for this run. The explicit enable_tts
        # argument takes precedence over the config flag.
        use_tts = enable_tts if enable_tts is not None else self.config.enable_tts

        # Lazily initialize the TTS engine on first use to avoid long
        # startup times when TTS is not needed.
        if use_tts and self.tts_engine is None and self.config.tts:
            try:
                self.tts_engine = TextToSpeechEngine(self.config.tts)
                logger.info("TTS engine initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize TTS engine: {e}")
                logger.exception("TTS init error details:")
                # Disable TTS for subsequent runs in this process so we don't
                # repeatedly try (and fail) to download / initialize models.
                self.config.tts = None
                use_tts = False

        for tgt_lang in target_langs:
            trans_start = time.time()
            translation = self.translator.translate(
                transcription, src_lang=detected_lang, tgt_lang=tgt_lang
            )
            logger.info(f"Translation to {tgt_lang} took {time.time() - trans_start:.2f}s")
            
            translations.append(
                {
                    "target_language": tgt_lang,
                    "text": translation["translated_text"],
                    "model_name": translation["model_name"],
                }
            )

            if use_tts and self.tts_engine:
                tts_start = time.time()
                try:
                    # Pass the target language to the TTS engine
                    audio_path = self.tts_engine.synthesize(
                        text=translation["translated_text"],
                        language=tgt_lang  # Pass the target language code
                    )
                    tts_outputs.append(
                        {
                            "target_language": tgt_lang,
                            "audio_path": audio_path,
                        }
                    )
                    logger.info(f"TTS for {tgt_lang} took {time.time() - tts_start:.2f}s")
                except Exception as e:
                    logger.error(f"TTS failed for {tgt_lang}: {e}")
                    logger.exception("TTS error details:")
        
        logger.info(f"Translation phase took {time.time() - translation_start:.2f}s")
        logger.info(f"Total pipeline time: {time.time() - start_time:.2f}s")
        
        return {
            "transcription": transcription,
            "detected_language": detected_lang,
            "detection": detection_result,
            "translations": translations,
            "tts_outputs": tts_outputs,
            "segments": asr_result.get("segments", []),
        }


__all__ = ["AudioLingoPipeline", "PipelineConfig"]

