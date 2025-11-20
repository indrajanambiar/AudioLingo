# AudioLingo – Multilingual Audio Translation Pipeline

AudioLingo (aka **PolyVoice**) is a CPU-friendly end-to-end system that converts audio into translated text (and optional speech) using only free, locally cached models.

## Features

- Whisper Tiny transcription (`openai/whisper-tiny`)
- Language detection with XLM-R (`papluca/xlm-roberta-base-language-detection`)
- MarianMT translation (`Helsinki-NLP/opus-mt-<src>-<tgt>`)
- Optional Coqui TTS for translated speech output
- FastAPI backend (`/api/translate`) and Gradio UI
- Works on CPUs without external API keys

## Project Structure

```
polyvoice/
├── app/                 # Core pipeline components
├── api/                 # FastAPI backend
├── ui/                  # Gradio interface
├── samples/             # Put reference audio files here
├── requirements.txt
└── README.md
```

## Getting Started

1. **Install dependencies**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   python3 -m pip install -r requirements.txt
   ```

2. **Download sample audio**
   - Add a short `.wav` file to `samples/sample_audio.wav`. You can record one via any mic tool.

3. **Run the FastAPI backend**
   ```bash
   python3 -m uvicorn api.main:app --reload
   ```

4. **Run the simple HTML UI (FastAPI)**
   ```bash
   python3 -m uvicorn ui.simple_ui:app --reload --port 7860
   ```
   Visit `http://localhost:7860` to upload audio and view results.

5. **(Optional) Run the legacy Gradio UI**
   ```bash
   python3 -m pip install --upgrade "gradio==4.44.1"
   python3 ui/gradio_app.py
   ```

6. **API request example (curl)**
   ```bash
   curl -X POST "http://localhost:8000/api/translate" \
        -F "audio_file=@samples/sample_audio.wav" \
        -F "target_languages=en,fr" \
        -F "enable_tts=false"
   ```

## Pipeline Overview

1. **Audio Input** – Upload or record audio (16 kHz recommended).
2. **ASR** – Whisper Tiny transcribes the speech.
3. **Language Detection** – XLM-R identifies the input language.
4. **Translation** – MarianMT generates target-language text.
5. **Optional TTS** – Coqui TTS converts translated text back to audio.

## Notes & Tips

- All models are downloaded once and cached locally (`~/.cache/huggingface` & `~/.local/share/tts`).
- Keep audio clips under 30 seconds for best CPU performance.
- To add new target languages, extend the `LANGUAGE_CHOICES` mapping in `ui/gradio_app.py`.
- For batch processing or analytics, build on top of `app/pipeline.py`.

## License

MIT – use freely for learning, demos, or portfolio projects.

