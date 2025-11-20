# AudioLingo Documentation

AudioLingo (aka **PolyVoice**) is a CPU‑friendly, end‑to‑end multilingual audio translation system. It takes raw audio, transcribes it, detects the language, translates the text into one or more target languages, and can optionally synthesize translated speech.

This document covers:
- Architecture and data flow
- All models used and why they were chosen
- How to run and extend the system
- Practical use‑cases
- Code review notes (strengths + improvement ideas)
- Interview preparation topics and sample questions

---

## 1. High‑Level Architecture

**Core idea:** chain together lightweight, offline‑capable components:

1. **ASR (Automatic Speech Recognition)** – Transcribe speech to text
   - Implementation: `ASRTranscriber` in `app/asr_transcription.py`
2. **Language Detection** – Infer the language of the transcribed text
   - Implementation: `LanguageDetector` in `app/language_detection.py`
3. **Translation** – Translate text from source to one or more target languages
   - Implementation: `Translator` in `app/translator.py`
4. **Optional TTS (Text‑to‑Speech)** – Convert translated text back into audio
   - Implementation: `TextToSpeechEngine` in `app/tts.py`
5. **Pipeline Orchestration** – Glue layer composing all the above
   - Implementation: `AudioLingoPipeline` in `app/pipeline.py`
6. **APIs & UIs** – Different front‑ends over the same core pipeline
   - REST API: `api/main.py`, `api/routes.py`
   - Simple HTML UI: `ui/simple_ui.py`
   - Live WebSocket UI: `ui/live_ui.py`
   - Gradio demo UI: `ui/gradio_app.py`

Data flow (typical request):

1. Client uploads or records audio.
2. Backend saves audio to a temporary `.wav` file.
3. `AudioLingoPipeline.run()` is invoked with the audio path & target languages.
4. Pipeline runs ASR → language detection → translation → (optional) TTS.
5. Response returns transcription, detected language, translations, and any TTS audio paths.

---

## 2. Models and Why They Are Used

This section documents each model/engine involved, where it lives in the code, and why it was chosen.

### 2.1 ASR – Whisper Tiny

**File:** `app/asr_transcription.py`

**Config dataclass:** `ASRConfig`
- `model_name: str = "tiny"`
- `device: str = "cpu"`
- `language: Optional[str] = None`
- `sample_rate: int = 16_000`

**Runtime class:** `ASRTranscriber`
- Loads model with `whisper.load_model(self.config.model_name, device=self.config.device)`.
- Accepts `AudioSource` (`str` path, `np.ndarray`, or `torch.Tensor`).
- Uses `torchaudio` to:
  - Load audio from files
  - Convert to mono
  - Resample to 16 kHz if needed

**Why Whisper Tiny?**
- **CPU‑friendly:** Tiny is the smallest Whisper variant; it loads quickly and runs on CPUs.
- **No external API keys:** Uses the open‑source `openai-whisper` package.
- **Good enough for short demo clips:** You accept a trade‑off: slightly lower accuracy vs. much faster inference.

**Output format:**
```python
{
  "text": <transcribed_text>,
  "detected_language": <language_code_or_None>,
  "segments": <whisper_segments_list>,
}
```

**Where it is used:**
- Directly by `AudioLingoPipeline` (in `app/pipeline.py`).
- Also by `RealTimeAudioProcessor` for live transcription (`app/realtime_processor.py`).

---

### 2.2 Language Detection – XLM‑R Classifier

**File:** `app/language_detection.py`

**Config dataclass:** `LanguageDetectionConfig`
- `model_name: str = "papluca/xlm-roberta-base-language-detection"`
- `device: str = "cpu"`
- `max_length: int = 256`
- `top_k: int = 3`

**Runtime class:** `LanguageDetector`
- Uses Hugging Face Transformers:
  - `AutoTokenizer.from_pretrained(config.model_name)`
  - `AutoModelForSequenceClassification.from_pretrained(config.model_name)`
- Sends input text through the classifier and applies softmax to obtain probabilities.
- Extracts the top‑`k` labels and normalizes them from `"__label__xx"` → `"xx"`.

**Why this model?**
- **Multilingual coverage:** `papluca/xlm-roberta-base-language-detection` covers many languages out‑of‑the‑box.
- **Text‑level detection:** Detects language from arbitrary text (i.e., the ASR output), independent of acoustics.
- **Simple integration:** Standard sequence‑classification interface from Transformers.

**Output format:**
```python
{
  "language": <best_lang_code_or_None>,
  "confidence": <float_0_to_1>,
  "top_k": [
    {"language": <code>, "confidence": <float>},
    ... up to top_k entries ...
  ],
}
```

**How it interacts with the pipeline:**
- `AudioLingoPipeline.run()` first asks ASR for `detected_language`.
- If Whisper does **not** provide a language, the pipeline falls back to this detector.
- If Whisper does provide a language, the pipeline constructs a "fake" detection result with confidence 1.0 for consistency.
- For translation, the pipeline currently **forces** `detected_lang = "en"` if it is anything other than English, to match available MarianMT models (see below).

---

### 2.3 Translation – MarianMT (Helsinki‑NLP)

**File:** `app/translator.py`

**Config dataclass:** `TranslationConfig`
- `default_src: str = "en"`
- `default_tgt: str = "fr"`
- `device: str = "cpu"`
- `max_length: int = 512`

**Runtime class:** `Translator`

#### 2.3.1 Default model naming

By default, the model name is constructed as:
```python
f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"
```
Example: `en` → `fr` uses `Helsinki-NLP/opus-mt-en-fr`.

Models are loaded with:
- `MarianTokenizer.from_pretrained(model_name)`
- `MarianMTModel.from_pretrained(model_name)`

The pair `(tokenizer, model)` is cached using `functools.lru_cache(maxsize=8)` to avoid repeated downloads and warm‑ups for frequently used language pairs.

#### 2.3.2 Custom overrides for Indian languages

Some language pairs do not exist directly as `opus-mt-<src>-<tgt>`. To handle this, the code defines a `CUSTOM_MODELS` mapping and a small `ModelOverride` dataclass:

- `ModelOverride.model_name`: actual checkpoint to load
- `ModelOverride.target_prefix`: special language token prefix required by certain multilingual models

Current overrides:
- `en → ta` (Tamil)
  - `model_name="Helsinki-NLP/opus-mt-en-mul"`
  - `target_prefix=">>tam<<"`
- `en → te` (Telugu)
  - `model_name="Helsinki-NLP/opus-mt-en-mul"`
  - `target_prefix=">>tel<<"`
- `en → kn` (Kannada)
  - `model_name="Helsinki-NLP/opus-mt-en-mul"`
  - `target_prefix=">>kan<<"`

If a custom override is **not** found, the system falls back to `opus-mt-<src>-<tgt>`.

#### 2.3.3 Why MarianMT?

- **Open‑source and offline:** Helsinki‑NLP models run locally without API keys.
- **Many language pairs:** A large catalog of `opus-mt` checkpoints simplifies adding new directions.
- **Tokenizer & model integration:** MarianMT is well supported in `transformers`.
- **CPU‑friendly models:** Most MarianMT checkpoints are manageable on CPU for short texts.

#### 2.3.4 Translation call and output

`Translator.translate(text, src_lang, tgt_lang)`:
1. Resolves `src` and `tgt` (falling back to defaults if needed).
2. Selects a model via `_resolve_model()` (including custom overrides).
3. Optionally prepends `target_prefix` for multilingual checkpoints.
4. Tokenizes input with padding + truncation up to `max_length`.
5. Runs `model.generate()` (no gradients) to produce translated tokens.
6. Decodes with `tokenizer.batch_decode(skip_special_tokens=True)`.

Returns:
```python
{
  "translated_text": <str>,
  "model_name": <checkpoint_name>,
}
```

**Where it is used:**
- `AudioLingoPipeline.run()` iterates over all requested `target_languages` and produces a list of translations with model names.
- `RealTimeAudioProcessor` reuses `Translator` for live or final processing.

---

### 2.4 TTS – espeak‑ng + ffmpeg

**File:** `app/tts.py`

Instead of a heavyweight Python TTS model, the project uses the **system `espeak-ng` binary** (and optionally `ffmpeg`) to generate WAV files. This keeps the TTS stack:
- Fast to start
- Easy to install on most platforms
- Independent of GPU availability

**Config dataclass:** `TTSConfig`

Key fields:
- `LANGUAGE_MAP`: maps pipeline language codes (used by translator/UI) to `espeak-ng` voice IDs.
  - Examples:
    - `"ml" → "ml"` (Malayalam)
    - `"hi" → "hi-mbrola-2"` (Hindi – MBROLA female voice)
    - `"ta" → "ta"` (Tamil)
    - `"te" → "te-mbrola-1"` (Telugu – MBROLA female voice)
    - `"kn" → "kn"` (Kannada)
    - Fallbacks: `"en"`, `"es"`, `"fr"`, `"de"`
- `pitch: int = 62` – Slightly higher than the default to sound more “female‑like” and clear.
- `speed_wpm: int = 135` – Slower than default to improve intelligibility.

**Runtime class:** `TextToSpeechEngine`

Workflow in `synthesize(text, language, output_path=None)`:
1. Validates that `text` is non‑empty.
2. Ensures `static/audio` directory exists.
3. Chooses an output path (either provided or auto‑generated using a timestamp + hash).
4. Maps the logical `language` code to an `espeak-ng` voice via `LANGUAGE_MAP` (fallback to `en`).
5. Invokes `espeak-ng` via `subprocess.run([...], check=True, capture_output=True)` to write a WAV file.
6. Validates that the file exists and is non‑empty.
7. Attempts optional **post‑processing** with `ffmpeg`:
   - Applies EQ and loudness normalization (`highpass`, `lowpass`, `loudnorm`).
   - If successful, switches to the processed `.proc.wav` file.
8. Returns a URL path like `/static/audio/tts_<timestamp>_<hash>.proc.wav`.

**Why this TTS approach?**
- **Startup time:** No heavy neural TTS models to load.
- **Portability:** Works where `espeak-ng` and `ffmpeg` are available; suitable for demos and low‑resource machines.
- **Fine‑tuning via CLI flags:** Pitch and speed are easy to tweak to get a more pleasant voice.

> Note: The `requirements.txt` still includes `TTS` (Coqui TTS), but the current implementation does **not** use it. The active, production path is `espeak-ng` + `ffmpeg`.

---

### 2.5 Real‑Time Processing

**File:** `app/realtime_processor.py`

Class: `RealTimeAudioProcessor`

- Internally constructs:
  - `ASRTranscriber(ASRConfig())`
  - `LanguageDetector(LanguageDetectionConfig())`
  - `Translator(TranslationConfig())`
- Maintains an `audio_buffer` of incoming chunks.
- `process_audio_chunk()` appends chunks and, once a threshold is reached, writes them to a temporary `.wav` file and runs ASR for partial transcription.
- `process_final_audio()` writes the full recording to a temp `.wav` file, then delegates to `AudioLingoPipeline` for the full ASR → detection → translation → TTS flow.

This component powers the WebSocket‑based live UI (`ui/live_ui.py`).

---

## 3. Orchestration – `AudioLingoPipeline`

**File:** `app/pipeline.py`

### 3.1 Configuration

`PipelineConfig` bundles the sub‑configs:
- `asr: ASRConfig = ASRConfig()`
- `language_detection: LanguageDetectionConfig = LanguageDetectionConfig()`
- `translation: TranslationConfig = TranslationConfig()`
- `tts: Optional[TTSConfig] = TTSConfig()`
- `enable_tts: bool = False` (global default, can be overridden per request)

### 3.2 Runtime behavior

`AudioLingoPipeline.__init__`:
- Accepts optional pre‑constructed components (for testing/injection).
- Otherwise creates default `ASRTranscriber`, `LanguageDetector`, `Translator`.
- Defers TTS engine initialization until the first time TTS is actually requested.

`AudioLingoPipeline.run(audio_source, target_languages, enable_tts=None)`:

1. Normalize `target_languages` to a list.
2. **ASR:**
   - Transcribe via `self.asr.transcribe(audio_source)`.
   - Capture `transcription`, `detected_language` (if any), and `segments`.
3. **Language detection:**
   - If ASR did not provide a language, call `self.detector.detect(transcription)`.
   - Otherwise, synthesize a detection result with confidence 1.0.
4. **Source language normalization for translation:**
   - For simplicity and to avoid non‑existent MarianMT checkpoints, if `detected_lang != "en"`, the pipeline overrides the source language to `"en"` **before** translation.
   - This assumes the spoken input is in English for reliable results.
5. **TTS enable/disable resolution:**
   - `use_tts = enable_tts if enable_tts is not None else self.config.enable_tts`.
   - If `use_tts` and `self.tts_engine is None` and `self.config.tts` is non‑`None`, lazily initialize `TextToSpeechEngine`.
   - If initialization fails, TTS is disabled for the rest of the process lifetime.
6. **Translation loop:**
   - For each `tgt_lang` in targets:
     - Call `self.translator.translate(transcription, src_lang=detected_lang, tgt_lang=tgt_lang)`.
     - Collect `{target_language, text, model_name}`.
     - If `use_tts` and TTS engine is available, call `self.tts_engine.synthesize()` and collect `{target_language, audio_path}`.
7. Return a dictionary containing:
   - `transcription`
   - `detected_language` (post‑normalization)
   - `detection` (full detection metadata)
   - `translations` (list)
   - `tts_outputs` (list)
   - `segments` (ASR segments, if available)

---

## 4. APIs and User Interfaces

### 4.1 FastAPI REST API

**Files:**
- `api/main.py` – creates `FastAPI` app, adds CORS, includes routes.
- `api/routes.py` – defines:
  - `GET /api/health` – simple health probe.
  - `POST /api/translate` – main translation endpoint.

`POST /api/translate` parameters:
- `audio_file: UploadFile` (multipart form field)
- `target_languages: str` – comma‑separated codes, e.g. `"ml,hi,ta"`
- `enable_tts: bool` – optional flag to enable TTS

Workflow:
1. Save uploaded file to a temp path.
2. Parse `target_languages` into a list.
3. Call `pipeline.run()`.
4. Schedule background deletion of the temp file.
5. Return the pipeline result JSON.

### 4.2 Simple HTML UI (Upload or Live Recording)

**File:** `ui/simple_ui.py`

- FastAPI app with:
  - `GET /` – renders a custom HTML form with:
    - File upload for audio.
    - Optional live recording via `navigator.mediaDevices.getUserMedia` + `MediaRecorder`.
    - Checkboxes for target languages (Indian languages: `ml`, `hi`, `ta`, `te`, `kn`).
    - TTS enable/disable checkbox.
  - `POST /submit` – handles both file upload and live recording input.
- Mounts static files at `/static` to serve generated TTS audio.

### 4.3 Live WebSocket UI

**File:** `ui/live_ui.py`

- FastAPI app with:
  - `GET /` – serves a live recording HTML page.
  - `WebSocket /ws` – used by the browser to stream audio chunks.
- Uses `ConnectionManager` to track active WebSocket connections.
- For each connection, orchestrates:
  - Start/stop recording messages
  - Incremental audio chunk processing via `RealTimeAudioProcessor`
  - Final audio processing via `process_complete_audio()`
- Streams live updates back to the client:
  - Status messages (recording/processing/ready)
  - Partial and final transcription
  - Translations and TTS paths

### 4.4 Gradio Demo

**File:** `ui/gradio_app.py`

- Uses `gr.Blocks` to define a simple demo:
  - Audio input (upload or microphone)
  - Target language checkbox group (a small set of languages)
  - Optional TTS checkbox
  - Outputs: detected language, transcription, translations, and TTS file list
- Works against the same `AudioLingoPipeline` core.
- Includes a small monkey‑patch to `gradio.routes.api_info` to avoid schema inspection issues.

---

## 5. Use Cases

You can frame AudioLingo around several realistic scenarios:

### 5.1 Language Learning Assistant

- A learner records English speech.
- System transcribes and translates into Indian languages (Malayalam, Hindi, Tamil, Telugu, Kannada).
- TTS generates spoken translations in target languages.
- Useful for pronunciation practice and bilingual comprehension.

### 5.2 Offline Multilingual Demo / Portfolio Project

- Showcases an end‑to‑end ML pipeline that runs fully offline.
- Demonstrates integration of multiple Hugging Face models with FastAPI and modern UIs.
- Great portfolio piece for backend/ML engineer interviews.

### 5.3 Accessibility & Content Localization

- Convert spoken explanations (e.g., short tutorials or instructions) into multiple languages.
- Provide transcripts, translations, and optional audio for multilingual audiences.

### 5.4 Live Translation Prototype

- The WebSocket UI can be used as a starting point for:
  - Live captioning in a specific target language.
  - Semi‑real‑time translation during a conversation.
- Not production‑grade yet, but good for proof‑of‑concept demos.

---

## 6. Running and Extending the Project

### 6.1 Basic Setup

1. Create and activate a virtual environment.
2. Install dependencies with `pip install -r requirements.txt`.
3. Ensure system packages are installed:
   - `espeak-ng`
   - `ffmpeg`

### 6.2 Starting Services

- **API backend:**
  - `uvicorn api.main:app --reload --port 8000`
- **Simple UI:**
  - `uvicorn ui.simple_ui:app --reload --port 7860`
- **Live UI:**
  - `uvicorn ui.live_ui:app --reload --port 8001` (or similar)
- **Gradio UI:**
  - `python ui/gradio_app.py`

### 6.3 Adding a New Translation Target

1. Decide on a new language code (e.g., `"es"` for Spanish).
2. Ensure a suitable MarianMT model exists:
   - If direct `opus-mt-en-es` exists, no custom override is needed.
   - If not, create a new `ModelOverride` entry in `Translator.CUSTOM_MODELS`.
3. Add the new target to:
   - `LANGUAGE_CHOICES` in `ui/simple_ui.py` and/or `ui/live_ui.py` and/or `ui/gradio_app.py`.
4. (Optional) Add an `espeak-ng` voice mapping in `TTSConfig.LANGUAGE_MAP`.

---

## 7. Code Review Notes

This section is written as if you are reviewing your own code in a professional setting.

### 7.1 Strengths

- **Clear separation of concerns:**
  - ASR, language detection, translation, TTS, pipeline orchestration, and UI are in separate modules.
- **Config‑driven design:**
  - Each component has its own dataclass (`ASRConfig`, `LanguageDetectionConfig`, `TranslationConfig`, `TTSConfig`, `PipelineConfig`).
  - Makes it easy to swap models, change devices, or tweak parameters.
- **Model caching:**
  - Translation models are cached via `lru_cache`, avoiding repeated downloads for common pairs.
- **Lazy initialization of heavy components:**
  - TTS engine is lazily created only when TTS is actually enabled, improving startup speed.
- **Multiple front‑ends reusing the same core:**
  - API, simple HTML UI, live WebSocket UI, and Gradio all wrap `AudioLingoPipeline` instead of duplicating logic.

### 7.2 Areas for Improvement

- **Model/device configuration:**
  - Currently hard‑coded to `cpu`. Could expose environment‑based configuration (e.g., auto‑detect GPU, set model sizes via env vars).
- **Source language handling:**
  - Forcing `detected_lang = "en"` simplifies MarianMT selection but is brittle if you ever want true non‑English inputs.
  - A future improvement is to use direct `src→tgt` pairs where available and only fall back to `en→tgt` when necessary.
- **TTS dependency clarity:**
  - `requirements.txt` still lists `TTS` (Coqui) even though the runtime path uses `espeak-ng`.
  - Consider either removing unused deps or adding an alternative TTS implementation that leverages Coqui.
- **Error handling and user feedback:**
  - Many exceptions are logged but not fully surfaced to the UI.
  - The simple UI does return an error page, but you could standardize error payloads and HTTP status codes across all entrypoints.
- **Temporary file management:**
  - Most endpoints correctly delete temp files, but checking all paths (especially error paths) would improve robustness.
- **Real‑time pipeline:**
  - `RealTimeAudioProcessor` currently uses a simplistic threshold (`len(audio_buffer) >= 50`) to decide when to transcribe.
  - This could be refined to use actual audio duration and more efficient streaming ASR.

### 7.3 Potential Refactors

- **Central configuration module:**
  - Introduce a single config loader (e.g., `pydantic` settings or environment variable mappings) to construct all `*Config` objects.
- **Abstract storage of generated audio:**
  - Instead of always writing to `static/audio`, allow pluggable storage (local, S3, etc.).
- **Unified entrypoint:**
  - Create a single CLI or main script for starting API/UI variants with flags.

---

## 8. Interview Preparation

This section is designed to help you talk about AudioLingo in an interview.

### 8.1 How to Pitch the Project in 30–60 Seconds

> “AudioLingo is an end‑to‑end, CPU‑friendly multilingual audio translation pipeline. Given an audio clip, it uses Whisper Tiny to transcribe speech, an XLM‑R classifier to detect the language, MarianMT models to translate into multiple target languages, and a lightweight `espeak-ng`‑based TTS engine to speak the translations back. I wrapped this pipeline in FastAPI with multiple UIs, including a REST API, a simple HTML interface, and a live WebSocket‑based recording UI. Everything runs offline using open‑source models.”

### 8.2 System Design / Architecture Questions

**Possible questions:**
- How would you scale this system for many concurrent users?
- How would you reduce latency for real‑time translation?
- How would you support GPU acceleration?
- How would you make the system resilient to model download failures?

**Key talking points:**
- Use a separate worker service for model loading and inference; front‑end API nodes forward requests.
- Pre‑warm models and share them across requests.
- Deploy on GPU machines and set `device="cuda"` in configs.
- Cache models on disk and add startup checks for dependencies (`espeak-ng`, `ffmpeg`).

### 8.3 ML / NLP Questions

**Possible questions:**
- Why did you choose Whisper Tiny instead of a larger ASR model?
- Why separate language detection from ASR?
- How do MarianMT models work at a high level?
- What are the trade‑offs of using multilingual vs. pair‑specific translation models?

**Key talking points:**
- Whisper Tiny vs. base/large: trade accuracy for lower memory and CPU latency.
- External language detection provides a second signal and can be reused for non‑Whisper inputs.
- MarianMT is a sequence‑to‑sequence transformer trained on bitext datasets; integrates easily via Hugging Face.
- Multilingual models need language tokens but provide flexibility; pair‑specific models can be more accurate for a given direction.

### 8.4 Backend / API Questions

**Possible questions:**
- How did you design your FastAPI endpoints?
- How do you handle long‑running requests and file uploads?
- How do you manage temporary files and static TTS outputs?
- How would you secure this API in production?

**Key talking points:**
- Explain the `POST /api/translate` flow and background cleanup tasks.
- Discuss timeouts, request limits, and streaming vs. one‑shot processing.
- Talk about storing generated audio safely (permissions, cleanup, quotas).
- Mention auth (JWT/API keys), rate limiting, CORS, and input validation.

### 8.5 Practical “What Would You Improve Next?” Question

Be ready with 2–3 concrete next steps:
- Replace the `en`‑only assumption with true multilingual `src→tgt` support.
- Add a Coqui TTS backend (neural) as an optional higher‑quality mode while keeping `espeak-ng` as the default.
- Introduce configuration via environment variables and a single `settings.py` module.
- Add unit tests/integration tests around `AudioLingoPipeline` and the FastAPI routes.

---

## 9. Summary

AudioLingo demonstrates how to combine multiple open‑source models and tools—Whisper Tiny for ASR, XLM‑R for language detection, MarianMT for translation, and `espeak-ng` + `ffmpeg` for TTS—into a coherent, offline‑capable audio translation system. It’s a strong portfolio project not only for ML knowledge but also for backend engineering, API design, and user‑facing UI integration.
