"""Gradio interface for AudioLingo."""

from __future__ import annotations

from pathlib import Path
from typing import List

import sys

# Ensure repository root is on sys.path when running as a script
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gradio as gr
import gradio.routes as gr_routes
import gradio_client.utils as gr_utils

# Monkey patch the Gradio API info endpoint to avoid schema inspection crashes
def _skip_api_info(*args, **kwargs):
    return {}


gr_routes.api_info = _skip_api_info

from app.pipeline import AudioLingoPipeline

pipeline = AudioLingoPipeline()

LANGUAGE_CHOICES = {
    "English": "en",
    "French": "fr",
    "Spanish": "es",
    "German": "de",
    "Hindi": "hi",
    "Arabic": "ar",
    "Chinese": "zh",
}


def translate_audio(audio, targets: List[str], enable_tts: bool):
    if audio is None:
        return "Please provide an audio clip.", "", "", []

    audio_path = audio if isinstance(audio, str) else audio.get("name")
    language_codes = [LANGUAGE_CHOICES[name] for name in targets]
    result = pipeline.run(audio_path, language_codes, enable_tts=enable_tts)

    translations = "\n\n".join(
        f"{entry['target_language']}: {entry['text']}"
        for entry in result["translations"]
    )
    tts_files = [entry["audio_path"] for entry in result["tts_outputs"]]

    return (
        result["detected_language"],
        result["transcription"],
        translations,
        tts_files,
    )


with gr.Blocks() as demo:
    gr.Markdown("# AudioLingo – Multilingual Audio Translation")
    with gr.Row():
        audio_input = gr.Audio(sources=["upload", "microphone"], type="filepath")
        target_selector = gr.CheckboxGroup(
            choices=list(LANGUAGE_CHOICES.keys()),
            value=["English"],
            label="Target languages",
        )
    enable_tts = gr.Checkbox(label="Generate TTS audio", value=False)
    submit_btn = gr.Button("Run Pipeline")

    detected_language = gr.Textbox(label="Detected language", interactive=False)
    transcription = gr.Textbox(label="Transcription", lines=4)
    translation_output = gr.Textbox(label="Translations", lines=6)
    tts_gallery = gr.File(label="TTS audio files", file_count="multiple")

    submit_btn.click(
        translate_audio,
        inputs=[audio_input, target_selector, enable_tts],
        outputs=[detected_language, transcription, translation_output, tts_gallery],
    )

if __name__ == "__main__":
    demo.launch(show_api=False, server_name="0.0.0.0", server_port=7860)

