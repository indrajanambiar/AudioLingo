"""
AudioLingo application package.

This module wires together the individual pipeline stages:
- Automatic speech recognition (ASR)
- Language detection
- Machine translation
- Optional text-to-speech (TTS)
"""

from .pipeline import AudioLingoPipeline

__all__ = ["AudioLingoPipeline"]

