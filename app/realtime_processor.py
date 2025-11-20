"""Real-time audio processing utilities."""
from __future__ import annotations

import asyncio
import io
import json
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import soundfile as sf
import torch
import torchaudio

from app.asr_transcription import ASRTranscriber, ASRConfig
from app.language_detection import LanguageDetector, LanguageDetectionConfig
from app.translator import Translator, TranslationConfig


class RealTimeAudioProcessor:
    """Processes audio chunks in real-time for live transcription."""
    
    def __init__(self):
        self.asr = ASRTranscriber(ASRConfig())
        self.detector = LanguageDetector(LanguageDetectionConfig())
        self.translator = Translator(TranslationConfig())
        self.audio_buffer = []
        self.is_processing = False
        
    async def process_audio_chunk(self, audio_chunk: bytes) -> Optional[Dict]:
        """Process a single audio chunk and return partial results."""
        try:
            # Convert webm to wav format (simplified - in production you'd use proper conversion)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                # For now, we'll accumulate chunks and process when we have enough data
                self.audio_buffer.append(audio_chunk)
                
                # Process when we have accumulated enough data (e.g., 5 seconds worth)
                if len(self.audio_buffer) >= 50:  # Adjust based on chunk size
                    combined_audio = b''.join(self.audio_buffer)
                    tmp_file.write(combined_audio)
                    tmp_file.flush()
                    
                    # Transcribe the accumulated audio
                    result = self.asr.transcribe(tmp_file.name)
                    
                    # Clear buffer after processing
                    self.audio_buffer = []
                    Path(tmp_file.name).unlink(missing_ok=True)
                    
                    return {
                        "transcription": result["text"],
                        "detected_language": result.get("detected_language"),
                        "is_final": False
                    }
                    
        except Exception as e:
            print(f"Error processing audio chunk: {e}")
            return None
            
    async def process_final_audio(self, audio_data: bytes, target_languages: List[str], enable_tts: bool = False) -> Dict:
        """Process the complete audio recording."""
        try:
            # Save complete audio
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                tmp_file.write(audio_data)
                tmp_file.flush()
                
                # Use the existing pipeline for complete processing
                from app.pipeline import AudioLingoPipeline
                pipeline = AudioLingoPipeline()
                
                result = pipeline.run(
                    tmp_file.name,
                    target_languages=target_languages,
                    enable_tts=enable_tts
                )
                
                # Clean up
                Path(tmp_file.name).unlink(missing_ok=True)
                
                return result
                
        except Exception as e:
            print(f"Error processing final audio: {e}")
            return {"error": str(e)}

__all__ = ["RealTimeAudioProcessor"]
