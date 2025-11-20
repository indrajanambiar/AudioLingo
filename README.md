# AudioLingo – Multilingual Audio Translation Pipeline

AudioLingo (aka **PolyVoice**) is a CPU-friendly end-to-end system that converts audio into translated text (and optional speech) using only free, locally cached models.

## Features

- Whisper Tiny transcription (`openai/whisper-tiny`)
- Language detection with XLM-R (`papluca/xlm-roberta-base-language-detection`)
- MarianMT translation (`Helsinki-NLP/opus-mt-<src>-<tgt>` with custom overrides for Indian languages)
- Optional TTS via system `espeak-ng` + `ffmpeg` for translated speech output
- FastAPI backend (`/api/translate`), simple HTML UI, live WebSocket UI, and Gradio demo UI
- Works on CPUs without external API keys

## Project Structure

```
audiolingo/
├── app/                 # Core pipeline components (ASR, detection, translation, TTS, pipeline)
├── api/                 # FastAPI backend
├── ui/                  # HTML, WebSocket live UI, and Gradio interfaces
├── static/              # Generated audio files served to clients
├── requirements.txt
├── README.md
└── documentation.md     # Detailed technical documentation, use cases, and interview prep
```

## Getting Started

1. **Install dependencies**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   python3 -m pip install -r requirements.txt
   ```

2. **Prepare audio**
   - Record a short `.wav` file (16 kHz mono recommended) or plan to use your microphone in the UI.

3. **Run the FastAPI backend**
   ```bash
   python3 -m uvicorn api.main:app --reload
   ```

4. **Run the simple HTML UI (FastAPI)**
   ```bash
   python3 -m uvicorn ui.simple_ui:app --reload --port 7860
   ```
   Visit `http://localhost:7860` to upload or record audio and view results.

5. **(Optional) Run the live WebSocket UI**
   ```bash
   python3 -m uvicorn ui.live_ui:app --reload --port 8001
   ```
   Visit `http://localhost:8001` for live, streaming transcription + translation.

6. **(Optional) Run the Gradio demo UI**
   ```bash
   python3 ui/gradio_app.py
   ```

7. **API request example (curl)**
   ```bash
   curl -X POST "http://localhost:8000/api/translate" \
        -F "audio_file=@path/to/audio.wav" \
        -F "target_languages=en,fr" \
        -F "enable_tts=false"
   ```

## Pipeline Overview

1. **Audio Input** – Upload or record audio (16 kHz recommended).
2. **ASR** – Whisper Tiny transcribes the speech.
3. **Language Detection** – XLM-R identifies the input language.
4. **Translation** – MarianMT generates target-language text.
5. **Optional TTS** – A lightweight `espeak-ng` + `ffmpeg` pipeline converts translated text back to audio.

## Architecture

```text
+---------------------+          HTTP / WebSocket          +----------------------+
|  Client UIs         |  --------------------------------> | FastAPI / Gradio     |
|  - Simple HTML UI   |                                    |  (api, ui, live)     |
|  - Live Web UI      |  <-------------------------------- |                      |
|  - Gradio / curl    |       JSON / HTML / audio          +----------+-----------+
+---------------------+                                               |
                                                                     v
                                                          +----------------------+
                                                          |  AudioLingoPipeline  |
                                                          +----------+-----------+
                                                                     |
           +---------------------------+-----------------------------+--------------------+
           |                           |                                                  |
           v                           v                                                  v
   +---------------+          +-----------------------+                          +------------------+
   | ASR (Whisper  |          | Language Detection    |                          | Translation      |
   |   Tiny)       |          | XLM-R classifier      |                          | MarianMT models  |
   +-------+-------+          +-----------+-----------+                          +--------+---------+
           |                              |                                               |
           +------------------------------+-----------------------------------------------+
                                          | text
                                          v
                                  +------------------------+
                                  | TTS (espeak-ng +       |
                                  |       ffmpeg)          |
                                  +-----------+------------+
                                              |
                                       WAV files under
                                       static/audio/
                                              |
                                              v
                                      Served as /static/... URLs
```

## Documentation

For a deeper dive into models, configuration, code review notes, and interview preparation, see `documentation.md`.

## Notes & Tips

- All models are downloaded once and cached locally (e.g. in Hugging Face and system TTS caches).
- Keep audio clips under ~30 seconds for best CPU performance.
- To add new target languages, extend the `LANGUAGE_CHOICES` mappings in `ui/simple_ui.py`, `ui/live_ui.py`, and/or `ui/gradio_app.py`.
- For batch processing or analytics, build on top of `app/pipeline.py` and the `AudioLingoPipeline` class.

## License

MIT – use freely for learning, demos, or portfolio projects.

