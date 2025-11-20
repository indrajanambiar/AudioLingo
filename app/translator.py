"""Text translation via Helsinki-NLP Marian models."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Optional, Tuple

import torch
from transformers import MarianMTModel, MarianTokenizer


@dataclass
class TranslationConfig:
    default_src: str = "en"
    default_tgt: str = "fr"
    device: str = "cpu"
    max_length: int = 512


@dataclass(frozen=True)
class ModelOverride:
    model_name: str
    target_prefix: Optional[str] = None


class Translator:
    """Translate text between languages using MarianMT."""

    # Some language pairs (e.g. en->ta) do not exist as Helsinki-NLP identifiers.
    # Map them to community or multilingual checkpoints that need language
    # tokens (e.g. >>tam<<). Store both the model name and the token prefix.
    CUSTOM_MODELS: Dict[Tuple[str, str], ModelOverride] = {
        ("en", "ta"): ModelOverride(
            model_name="Helsinki-NLP/opus-mt-en-mul",
            target_prefix=">>tam<<",
        ),
        ("en", "te"): ModelOverride(
            model_name="Helsinki-NLP/opus-mt-en-mul",
            target_prefix=">>tel<<",
        ),
        ("en", "kn"): ModelOverride(
            model_name="Helsinki-NLP/opus-mt-en-mul",
            target_prefix=">>kan<<",
        ),
    }

    def __init__(self, config: Optional[TranslationConfig] = None) -> None:
        self.config = config or TranslationConfig()

    @staticmethod
    def _model_name(src_lang: str, tgt_lang: str) -> str:
        return f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"

    def _resolve_model(self, src_lang: str, tgt_lang: str) -> ModelOverride:
        override = self.CUSTOM_MODELS.get((src_lang, tgt_lang))
        if override:
            return override
        return ModelOverride(model_name=self._model_name(src_lang, tgt_lang))

    @lru_cache(maxsize=8)
    def _load_model(self, model_name: str) -> Tuple[MarianTokenizer, MarianMTModel]:
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
        model.to(self.config.device)
        return tokenizer, model

    def translate(
        self,
        text: str,
        src_lang: Optional[str] = None,
        tgt_lang: Optional[str] = None,
    ) -> Dict[str, str]:
        if not text.strip():
            return {"translated_text": "", "model_name": ""}

        src = src_lang or self.config.default_src
        tgt = tgt_lang or self.config.default_tgt
        model_override = self._resolve_model(src, tgt)
        tokenizer, model = self._load_model(model_override.model_name)

        normalized_text = text
        if model_override.target_prefix:
            normalized_text = f"{model_override.target_prefix} {normalized_text.strip()}"

        inputs = tokenizer(
            normalized_text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=self.config.max_length,
        ).to(self.config.device)

        with torch.no_grad():
            generated_tokens = model.generate(**inputs, max_new_tokens=256)

        translated = tokenizer.batch_decode(
            generated_tokens, skip_special_tokens=True
        )

        return {
            "translated_text": translated[0],
            "model_name": model_override.model_name,
        }


__all__ = ["Translator", "TranslationConfig"]

