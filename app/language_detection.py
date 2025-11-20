"""Language detection using an XLM-R classifier."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


@dataclass
class LanguageDetectionConfig:
    model_name: str = "papluca/xlm-roberta-base-language-detection"
    device: str = "cpu"
    max_length: int = 256
    top_k: int = 3


class LanguageDetector:
    """Detect the language of a text snippet."""

    def __init__(self, config: Optional[LanguageDetectionConfig] = None) -> None:
        self.config = config or LanguageDetectionConfig()
        self._tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model_name
        ).to(self.config.device)

    @staticmethod
    def _normalize_label(label: str) -> str:
        # Model returns labels like "__label__en"
        if "__label__" in label:
            return label.split("__label__")[-1]
        return label

    def detect(self, text: str) -> Dict[str, Optional[str]]:
        if not text.strip():
            return {"language": None, "confidence": 0.0, "top_k": []}

        encoded = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=self.config.max_length,
        ).to(self.config.device)

        with torch.no_grad():
            logits = self._model(**encoded).logits.squeeze(0)
            probs = torch.nn.functional.softmax(logits, dim=-1)

        values, indices = torch.topk(
            probs, k=min(self.config.top_k, probs.shape[-1])
        )

        labels: List[str] = [self._model.config.id2label[i] for i in indices.tolist()]
        top_k = [
            {"language": self._normalize_label(label), "confidence": prob.item()}
            for label, prob in zip(labels, values)
        ]

        return {
            "language": top_k[0]["language"],
            "confidence": top_k[0]["confidence"],
            "top_k": top_k,
        }


__all__ = ["LanguageDetector", "LanguageDetectionConfig"]

