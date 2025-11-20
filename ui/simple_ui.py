"""Minimal HTML UI for the AudioLingo pipeline (no Gradio)."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Sequence, Union
import base64
import io
import subprocess
import os
import time
import logging

from fastapi import FastAPI, File, Form, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from app.pipeline import AudioLingoPipeline

app = FastAPI(title="AudioLingo Mini UI")
pipeline = None  # Will be initialized on first request

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/tmp/audiolingo_timing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

LANGUAGE_CHOICES = [
    ("Malayalam", "ml"),
    ("Hindi", "hi"),
    ("Tamil", "ta"),
    ("Telugu", "te"),
    ("Kannada", "kn"),
]

FORM_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>AudioLingo – Simple UI</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 2rem; max-width: 720px; }}
    h1 {{ color: #2b3a67; }}
    .input-method {{ background: #f8f9fa; padding: 1rem; border-radius: 6px; margin-bottom: 1rem; }}
    .input-method label {{ font-weight: bold; margin-bottom: 0.5rem; display: block; }}
    .radio-group {{ margin: 0.5rem 0; }}
    .radio-group label {{ font-weight: normal; margin: 0; }}
    .input-section {{ margin: 1rem 0; padding: 1rem; border: 1px solid #dee2e6; border-radius: 6px; }}
    .input-section.hidden {{ display: none; }}
    label {{ display: block; margin-top: 1rem; font-weight: bold; }}
    input[type=text], select, textarea {{ width: 100%; padding: 0.5rem; }}
    .checkbox-group {{ display: flex; flex-wrap: wrap; gap: 0.5rem 1rem; }}
    .checkbox-group label {{ font-weight: normal; margin: 0; }}
    .result {{ border: 1px solid #d8d8d8; padding: 1rem; margin-top: 1.5rem; border-radius: 6px; }}
    .tts-links a {{ display: block; margin-top: 0.25rem; }}
    button {{ margin-top: 1.5rem; padding: 0.75rem 1.5rem; font-size: 1rem; }}
    .record-btn {{ background: #e74c3c; color: white; }}
    .record-btn.recording {{ background: #c0392b; }}
    .stop-btn {{ background: #95a5a6; color: white; }}
    .stop-btn:disabled {{ background: #bdc3c7; cursor: not-allowed; }}
    .audio-level {{ 
      height: 40px; 
      background: #ecf0f1; 
      border-radius: 4px; 
      margin: 0.5rem 0; 
      position: relative;
      overflow: hidden;
    }}
    .audio-level-bar {{ 
      height: 100%; 
      background: #3498db; 
      width: 0%; 
      transition: width 0.1s ease;
    }}
    .status {{ padding: 0.5rem; border-radius: 4px; margin: 0.5rem 0; }}
    .status.recording {{ background: #ffebee; color: #c62828; }}
    .status.processing {{ background: #fff3e0; color: #f57c00; }}
    .status.ready {{ background: #e8f5e8; color: #2e7d32; }}
  </style>
</head>
<body>
  <h1>AudioLingo – Multilingual Audio Translation</h1>
  
  <div class="input-method">
    <label>Choose Input Method:</label>
    <div class="radio-group">
      <label><input type="radio" name="input_method" value="file" checked onchange="toggleInputMethod('file')" /> Upload Audio File</label>
      <label><input type="radio" name="input_method" value="live" onchange="toggleInputMethod('live')" /> Live Recording</label>
    </div>
  </div>

  <!-- File Upload Section -->
  <div id="file-section" class="input-section">
    <label>Audio file (wav/mp3)</label>
    <input type="file" id="audio_file" name="audio_file" accept="audio/*" required />
  </div>

  <!-- Live Recording Section -->
  <div id="live-section" class="input-section hidden">
    <div style="margin: 1rem 0;">
      <button id="recordBtn" class="record-btn" onclick="toggleRecording()">🎤 Start Recording</button>
      <button id="stopBtn" class="stop-btn" onclick="stopRecording()" disabled>⏹️ Stop</button>
    </div>
    
    <div class="audio-level">
      <div class="audio-level-bar" id="audioLevelBar"></div>
    </div>
    
    <div id="status" class="status ready">Ready to record</div>
  </div>

  <form action="/submit" enctype="multipart/form-data" method="post" id="mainForm" onsubmit="showProcessing()">
    <input type="hidden" id="audio_data" name="audio_data" value="" />
    
    <label>Target languages</label>
    <div class="checkbox-group">
      {language_checkboxes}
    </div>

    <label>
      <input type="checkbox" name="enable_tts" value="true" />
      Generate TTS audio (requires Python 3.10+ and Coqui support)
    </label>

    <button type="submit" id="submitBtn">Run Pipeline</button>
  </form>

  <div id="processingMessage" style="display: none; margin-top: 1rem; padding: 1rem; background: #fff3e0; border-radius: 4px; color: #f57c00;">
    Processing audio... This may take a few moments.
  </div>

  <script>
    let mediaRecorder;
    let audioChunks = [];
    let isRecording = false;
    let audioContext;
    let analyser;
    let microphone;
    let javascriptNode;
    let recordingStartTime;
    let recordingTimer;

    function updateRecordingTime() {{
      const elapsed = Math.floor((Date.now() - recordingStartTime) / 1000);
      document.getElementById('status').textContent = `Recording... ${{elapsed}}s`;
      
      // Auto-stop after 30 seconds
      if (elapsed >= 30) {{
        stopRecording();
        alert('Maximum recording time (30s) reached. Click "Process Recording" to continue.');
      }}
    }}

    function showProcessing() {{
      document.getElementById('processingMessage').style.display = 'block';
      document.getElementById('submitBtn').disabled = true;
      document.getElementById('submitBtn').textContent = 'Processing...';
    }}

    function toggleInputMethod(method) {{
      const fileSection = document.getElementById('file-section');
      const liveSection = document.getElementById('live-section');
      const audioFile = document.getElementById('audio_file');
      const submitBtn = document.getElementById('submitBtn');
      
      if (method === 'file') {{
        fileSection.classList.remove('hidden');
        liveSection.classList.add('hidden');
        audioFile.required = true;
        submitBtn.textContent = 'Run Pipeline';
      }} else {{
        fileSection.classList.add('hidden');
        liveSection.classList.remove('hidden');
        audioFile.required = false;
        submitBtn.textContent = 'Process Recording';
      }}
    }}

    async function toggleRecording() {{
      if (!isRecording) {{
        await startRecording();
      }} else {{
        stopRecording();
      }}
    }}

    async function startRecording() {{
      try {{
        const stream = await navigator.mediaDevices.getUserMedia({{ audio: true }});
        
        // Set up audio visualization
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
        analyser = audioContext.createAnalyser();
        microphone = audioContext.createMediaStreamSource(stream);
        javascriptNode = audioContext.createScriptProcessor(2048, 1, 1);
        
        analyser.smoothingTimeConstant = 0.8;
        analyser.fftSize = 1024;
        
        microphone.connect(analyser);
        analyser.connect(javascriptNode);
        javascriptNode.connect(audioContext.destination);
        
        javascriptNode.onaudioprocess = function() {{
          const array = new Uint8Array(analyser.frequencyBinCount);
          analyser.getByteFrequencyData(array);
          const values = array.reduce((a, b) => a + b, 0);
          const average = values / array.length;
          const percentage = Math.min(100, (average / 128) * 100);
          document.getElementById('audioLevelBar').style.width = percentage + '%';
        }};
        
        mediaRecorder = new MediaRecorder(stream);
        audioChunks = [];
        
        mediaRecorder.ondataavailable = function(event) {{
          audioChunks.push(event.data);
        }};
        
        mediaRecorder.onstop = function() {{
          const audioBlob = new Blob(audioChunks, {{ type: 'audio/webm' }});
          const reader = new FileReader();
          reader.onloadend = function() {{
            const base64data = reader.result;
            document.getElementById('audio_data').value = base64data;
          }};
          reader.readAsDataURL(audioBlob);
        }};
        
        mediaRecorder.start(1000);  // Collect chunks every 1 second
                recordingStartTime = Date.now();
                
                // Update timer every second
                recordingTimer = setInterval(updateRecordingTime, 1000);
        isRecording = true;
        
        document.getElementById('recordBtn').textContent = '🎤 Recording...';
        document.getElementById('recordBtn').classList.add('recording');
        document.getElementById('stopBtn').disabled = false;
        document.getElementById('status').textContent = 'Recording...';
        document.getElementById('status').className = 'status recording';
        
      }} catch (err) {{
        console.error('Error accessing microphone:', err);
        document.getElementById('status').textContent = 'Error: Could not access microphone';
        document.getElementById('status').className = 'status error';
      }}
    }}

    function stopRecording() {{
      if (mediaRecorder && isRecording) {{
        mediaRecorder.stop();
        mediaRecorder.stream.getTracks().forEach(track => track.stop());
        
        if (recordingTimer) {{
          clearInterval(recordingTimer);
          recordingTimer = null;
        }}
        
        if (audioContext) {{
          audioContext.close();
        }}
        
        isRecording = false;
        
        document.getElementById('recordBtn').textContent = '🎤 Start Recording';
        document.getElementById('recordBtn').classList.remove('recording');
        document.getElementById('stopBtn').disabled = true;
        document.getElementById('audioLevelBar').style.width = '0%';
        
        document.getElementById('status').textContent = 'Recording complete. Ready to process.';
        document.getElementById('status').className = 'status ready';
      }}
    }}
  </script>
</body>
</html>
""".format(
    language_checkboxes="\n".join(
        f'<label><input type="checkbox" name="targets" value="{code}" '
        f'{"checked" if code=="en" else ""}/> {label}</label>'
        for label, code in LANGUAGE_CHOICES
    )
)


def _save_upload(upload: UploadFile) -> Path:
    suffix = Path(upload.filename or "audio.wav").suffix or ".wav"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    content = upload.file.read()
    tmp.write(content)
    tmp.flush()
    tmp.close()
    return Path(tmp.name)


def _save_base64_audio(base64_data: str) -> Path:
    """Save base64 audio data to a temporary WAV file."""
    # Remove data URL prefix if present
    if ',' in base64_data:
        base64_data = base64_data.split(',')[1]
    
    # Decode base64 data
    audio_bytes = base64.b64decode(base64_data)
    
    # Save WebM to temporary file
    webm_path = tempfile.NamedTemporaryFile(delete=False, suffix=".webm")
    webm_path.write(audio_bytes)
    webm_path.flush()
    webm_path.close()
    
    # Convert WebM to WAV using ffmpeg
    wav_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    wav_path.close()

    try:
        # Use -y so ffmpeg never prompts "File already exists. Overwrite? [y/N]"
        subprocess.run([
            'ffmpeg', '-y', '-i', webm_path.name,
            '-ar', '16000',  # Sample rate expected by Whisper
            '-ac', '1',      # Mono channel
            wav_path.name,
        ], check=True, capture_output=True)
        
        # Clean up WebM file
        os.unlink(webm_path.name)
        
        return Path(wav_path.name)
    except subprocess.CalledProcessError as e:
        # Clean up files on error
        os.unlink(webm_path.name)
        if os.path.exists(wav_path.name):
            os.unlink(wav_path.name)
        raise RuntimeError(f"Audio conversion failed: {e.stderr.decode()}")


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    language_checkboxes = "".join(
        f'<label><input type="checkbox" name="targets" value="{code}"> {name}</label>'
        for name, code in LANGUAGE_CHOICES
    )
    # Avoid str.format on FORM_HTML because it contains CSS braces `{}` which
    # would be interpreted as format placeholders. We only need to substitute
    # the language_checkboxes placeholder, so use a simple replace instead.
    html = FORM_HTML.replace("{language_checkboxes}", language_checkboxes)
    return HTMLResponse(html)


@app.get("/test")
async def test_logging():
    logger.info("Test endpoint called")
    return {"message": "Logging is working", "timestamp": time.time()}


@app.post("/submit", response_class=HTMLResponse)
async def submit_form(
    audio_file: UploadFile = File(None),
    audio_data: Union[str, None] = Form(None),
    targets: Union[str, Sequence[str], None] = Form(None),
    enable_tts: Union[str, None] = Form(None),
) -> HTMLResponse:
    global pipeline
    start_time = time.time()
    
    if pipeline is None:
        logger.info("Loading pipeline models...")
        pipeline_start = time.time()
        pipeline = AudioLingoPipeline()
        logger.info(f"Pipeline loaded in {time.time() - pipeline_start:.2f}s")
    
    # Handle targets - convert string to list if needed
    if isinstance(targets, str):
        target_langs = [targets]
    else:
        target_langs = list(targets) if targets else ["ml"]
    
    # Determine input source and save audio
    if audio_file and audio_file.filename:
        # File upload
        upload_path = _save_upload(audio_file)
    elif audio_data:
        # Live recording
        upload_path = _save_base64_audio(audio_data)
    else:
        return HTMLResponse("""
        <!doctype html>
        <html><head><title>Error</title></head>
        <body><h1>Error: Please provide either an audio file or record audio</h1>
        <a href="/">Go back</a></body></html>
        """)
    
    logger.info(f"Audio saved in {time.time() - start_time:.2f}s")
    processing_start = time.time()
    
    try:
        result = pipeline.run(
            str(upload_path),
            target_languages=target_langs,
            enable_tts=bool(enable_tts),
        )
        logger.info(f"Pipeline processed in {time.time() - processing_start:.2f}s")
    except Exception as e:
        # Clean up file on error
        if upload_path.exists():
            upload_path.unlink()
        
        # Return error page instead of crashing
        return HTMLResponse(f"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>AudioLingo Error</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 2rem; max-width: 720px; }}
    .error {{ border: 1px solid #d32f2f; padding: 1rem; margin-top: 1.5rem; border-radius: 6px; background: #ffebee; color: #d32f2f; }}
    a.button {{ display: inline-block; margin-top: 1rem; text-decoration: none; color: white; background: #2b3a67; padding: 0.5rem 1rem; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>AudioLingo – Processing Error</h1>
  <div class="error">
    <p><strong>Error:</strong> {str(e)}</p>
    <p>Please try again with a shorter audio recording or clearer audio quality.</p>
  </div>
  <a href="/" class="button">Try Again</a>
</body>
</html>
""")
    finally:
        if upload_path.exists():
            upload_path.unlink()
    
    translations_html = "".join(
        f"<li><strong>{entry['target_language']}</strong>: {entry['text']}</li>"
        for entry in result["translations"]
    )
    tts_html = "".join(
        f"<li>{tts['target_language']}: {tts['audio_path']}</li>"
        for tts in result["tts_outputs"]
    ) or "<li><em>TTS disabled.</em></li>"
    
    return HTMLResponse(
        f"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>AudioLingo Results</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 2rem; max-width: 720px; }}
    .result {{ border: 1px solid #d8d8d8; padding: 1rem; margin-top: 1.5rem; border-radius: 6px; }}
    a.button {{ display: inline-block; margin-top: 1rem; text-decoration: none; color: white; background: #2b3a67; padding: 0.5rem 1rem; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>AudioLingo – Results</h1>
  <div class="result">
    <p><strong>Detected language:</strong> {result["detected_language"]}</p>
    <p><strong>Transcription:</strong></p>
    <p>{result["transcription"]}</p>
    <p><strong>Translations:</strong></p>
    <ul>{translations_html}</ul>
    <p><strong>TTS Outputs:</strong></p>
    <ul>{tts_html}</ul>
  </div>
  <a href="/" class="button">Process another audio</a>
</body>
</html>
"""
    )
