"""Text-to-speech support using the system `espeak-ng` binary.

This implementation is intentionally simple and avoids heavyweight
Python TTS libraries (Coqui, transformers, etc.). It shells out to
`espeak-ng` to generate WAV files that are then served by FastAPI.
"""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class TTSConfig:
    """Configuration for the espeak-ng based TTS engine.

    `LANGUAGE_MAP` maps pipeline language codes (what the UI and
    translator use) to `espeak-ng` voice identifiers.

    We also expose basic parameters like pitch and speed to push the
    output towards a higher, slightly slower voice that tends to sound
    more "female-like" and clear.
    """

    # Map our language codes to espeak-ng voices.
    # These were discovered via `espeak-ng --voices`.
    LANGUAGE_MAP = {
        # Indian languages
        # NOTE: For Hindi and Telugu we choose explicit MBROLA female voices
        # where available. These give a clearly female sound compared to the
        # default language voices.
        "ml": "ml",              # Malayalam (no separate female voice available)
        "hi": "hi-mbrola-2",    # Hindi – female MBROLA voice
        "ta": "ta",              # Tamil (no separate female voice available)
        "te": "te-mbrola-1",    # Telugu – female MBROLA voice
        "kn": "kn",              # Kannada (no separate female voice available)
        # A small set of fallbacks
        "en": "en",              # English
        "es": "es",              # Spanish
        "fr": "fr",              # French
        "de": "de",              # German
    }

    # espeak-ng parameters (see `espeak-ng --help`):
    # -p : pitch (0–99, default ~50).
    # -s : speed in words per minute (default ~175).
    # Use a slightly higher pitch and moderately slow speed for clarity.
    pitch: int = 62
    speed_wpm: int = 135


class TextToSpeechEngine:
    """TTS engine that calls `espeak-ng` to synthesize audio.

    Output WAV files are written under `static/audio` and the
    relative URL path (`/static/audio/...`) is returned so the
    UI can play them.
    """

    def __init__(self, config: Optional[TTSConfig] = None) -> None:
        self.config = config or TTSConfig()

    def synthesize(self, text: str, language: str = "en", output_path: Optional[str] = None) -> str:
        import logging

        logger = logging.getLogger(__name__)

        if not text.strip():
            raise ValueError("Cannot synthesize empty text.")

        # Create static/audio directory if it doesn't exist
        static_dir = Path("static/audio")
        static_dir.mkdir(parents=True, exist_ok=True)

        # Use caller-provided path or generate a unique one
        if output_path is not None:
            target_path = Path(output_path)
            static_dir = target_path.parent
            static_dir.mkdir(parents=True, exist_ok=True)
            filename = target_path.name
        else:
            timestamp = int(time.time())
            filename = f"tts_{timestamp}_{hash(text) % 10000}.wav"
            target_path = static_dir / filename

        logger.info(
            "Generating TTS audio for text: %s... (language: %s)",
            text[:100],
            language,
        )
        logger.info("Saving to: %s", target_path.absolute())

        # Resolve language to an espeak-ng voice, default to English
        lang_key = (language or "en").lower()
        voice = self.config.LANGUAGE_MAP.get(lang_key, "en")
        logger.info("Using espeak-ng voice: %s", voice)

        # espeak-ng tuning parameters for a higher, clearer voice
        pitch = max(0, min(99, int(self.config.pitch)))
        speed = max(80, min(300, int(self.config.speed_wpm)))
        logger.info("espeak-ng params: pitch=%d, speed=%d wpm", pitch, speed)

        try:
            # Call espeak-ng to synthesize to a WAV file.
            # Example: espeak-ng -v ml -p 65 -s 150 -w /tmp/out.wav "..."
            subprocess.run(
                [
                    "espeak-ng",
                    "-v",
                    voice,
                    "-p",
                    str(pitch),
                    "-s",
                    str(speed),
                    "-w",
                    str(target_path),
                    text,
                ],
                check=True,
                capture_output=True,
            )

            if not target_path.exists():
                error_msg = f"Failed to create audio file at {target_path}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)

            file_size = target_path.stat().st_size
            logger.info("Audio file generated successfully. Size: %d bytes", file_size)

            if file_size == 0:
                error_msg = f"Generated audio file is empty: {target_path}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)

            # Optional post-processing with ffmpeg to further increase
            # perceived "femaleness" and clarity by slightly raising the
            # pitch and applying gentle filtering/normalization.
            #
            # This assumes `ffmpeg` is installed and available in PATH
            # (already required elsewhere in this project).
            try:
                import tempfile

                processed_path = target_path.with_suffix(".proc.wav")
                logger.info("Running ffmpeg post-processing: %s -> %s", target_path, processed_path)

                # Mild EQ + loudness normalization only – no extra time/pitch
                # tricks here, to avoid speeding up or distorting the audio.
                ffmpeg_filter = (
                    "highpass=f=120,lowpass=f=6500,"
                    "loudnorm=I=-20:LRA=11:TP=-2.0"
                )

                subprocess.run(
                    [
                        "ffmpeg",
                        "-y",
                        "-i",
                        str(target_path),
                        "-af",
                        ffmpeg_filter,
                        str(processed_path),
                    ],
                    check=True,
                    capture_output=True,
                )

                if processed_path.exists() and processed_path.stat().st_size > 0:
                    logger.info("ffmpeg post-processing succeeded; using %s", processed_path)
                    target_path.unlink(missing_ok=True)
                    target_path = processed_path
                else:
                    logger.warning("ffmpeg post-processing produced no output; keeping original file")
            except FileNotFoundError:
                logger.warning("ffmpeg not found; skipping post-processing")
            except subprocess.CalledProcessError as e:
                stderr = e.stderr.decode("utf-8", errors="replace") if e.stderr else ""
                logger.warning("ffmpeg post-processing failed: %s", stderr)
            except Exception as e:  # noqa: BLE001
                logger.warning("Unexpected error during ffmpeg post-processing: %s", e)

            # Return the relative path that can be served by FastAPI
            # (StaticFiles is mounted at `/static`).
            rel_path = target_path.relative_to(Path("static"))
            return_path = f"/static/{rel_path.as_posix()}"
            logger.info("Audio will be served from: %s", return_path)
            return return_path

        except FileNotFoundError as e:
            # espeak-ng not installed / not in PATH
            error_msg = (
                "espeak-ng binary not found. Make sure it is installed and in PATH. "
                "On macOS with Homebrew: `brew install espeak-ng`."
            )
            logger.error("%s (%s)", error_msg, e)
            raise RuntimeError(error_msg) from e
        except subprocess.CalledProcessError as e:
            # Capture stderr for easier debugging
            stderr = e.stderr.decode("utf-8", errors="replace") if e.stderr else ""
            error_msg = f"espeak-ng failed with exit code {e.returncode}: {stderr}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
        except Exception as e:  # noqa: BLE001
            logger.error("Error in TTS synthesis: %s", str(e))
            logger.exception("Full error details:")
            raise


__all__ = ["TextToSpeechEngine", "TTSConfig"]

