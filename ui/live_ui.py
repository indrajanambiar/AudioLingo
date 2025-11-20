"""Live audio recording and real-time translation UI."""
from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import soundfile as sf
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from app.pipeline import AudioLingoPipeline
from app.realtime_processor import RealTimeAudioProcessor

app = FastAPI(title="AudioLingo Live")
pipeline = AudioLingoPipeline()
realtime_processor = RealTimeAudioProcessor()

LANGUAGE_CHOICES = [
    ("English", "en"),
    ("French", "fr"),
    ("Spanish", "es"),
    ("German", "de"),
    ("Hindi", "hi"),
    ("Arabic", "ar"),
    ("Chinese", "zh"),
]

LIVE_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>AudioLingo – Live Translation</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 2rem; max-width: 720px; }}
    h1 {{ color: #2b3a67; }}
    .controls {{ background: #f5f5f5; padding: 1rem; border-radius: 6px; margin-bottom: 1rem; }}
    .checkbox-group {{ display: flex; flex-wrap: wrap; gap: 0.5rem 1rem; }}
    .checkbox-group label {{ font-weight: normal; margin: 0; }}
    button {{ 
      padding: 0.75rem 1.5rem; 
      font-size: 1rem; 
      border: none; 
      border-radius: 4px; 
      cursor: pointer;
      margin-right: 0.5rem;
    }}
    .record-btn {{ background: #e74c3c; color: white; }}
    .record-btn.recording {{ background: #c0392b; }}
    .stop-btn {{ background: #95a5a6; color: white; }}
    .stop-btn:disabled {{ background: #bdc3c7; cursor: not-allowed; }}
    .results {{ border: 1px solid #d8d8d8; padding: 1rem; margin-top: 1rem; border-radius: 6px; }}
    .transcription {{ background: #ecf0f1; padding: 0.5rem; border-radius: 4px; margin: 0.5rem 0; }}
    .translation {{ margin: 0.5rem 0; }}
    .translation strong {{ color: #2c3e50; }}
    .status {{ padding: 0.5rem; border-radius: 4px; margin: 0.5rem 0; }}
    .status.recording {{ background: #ffebee; color: #c62828; }}
    .status.processing {{ background: #fff3e0; color: #f57c00; }}
    .status.ready {{ background: #e8f5e8; color: #2e7d32; }}
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
  </style>
</head>
<body>
  <h1>AudioLingo – Live Audio Translation</h1>
  
  <div class="controls">
    <label>Target languages</label>
    <div class="checkbox-group">
      {language_checkboxes}
    </div>
    
    <label>
      <input type="checkbox" id="enable_tts" />
      Generate TTS audio (requires Python 3.10+ and Coqui support)
    </label>
    
    <div style="margin-top: 1rem;">
      <button id="recordBtn" class="record-btn">🎤 Start Recording</button>
      <button id="stopBtn" class="stop-btn" disabled>⏹️ Stop</button>
    </div>
    
    <div class="audio-level">
      <div class="audio-level-bar" id="audioLevelBar"></div>
    </div>
    
    <div id="status" class="status ready">Ready to record</div>
  </div>

  <div class="results" id="results" style="display: none;">
    <h3>Live Results</h3>
    <div id="liveContent"></div>
  </div>

  <script>
    const ws = new WebSocket('ws://localhost:8000/ws');
    let mediaRecorder;
let audioChunks = [];
let isRecording = false;
let audioContext;
let analyser;
let microphone;
let javascriptNode;

// Language checkboxes
const languageCheckboxes = document.querySelectorAll('input[name="targets"]');
const recordBtn = document.getElementById('recordBtn');
const stopBtn = document.getElementById('stopBtn');
const statusDiv = document.getElementById('status');
const resultsDiv = document.getElementById('results');
const liveContent = document.getElementById('liveContent');
const audioLevelBar = document.getElementById('audioLevelBar');
const enableTtsCheckbox = document.getElementById('enable_tts');

// WebSocket handlers
ws.onmessage = function(event) {{
    const data = JSON.parse(event.data);
    handleWebSocketMessage(data);
}};

ws.onopen = function() {{
    updateStatus('Connected', 'ready');
}};

ws.onclose = function() {{
    updateStatus('Disconnected', 'ready');
}};

function handleWebSocketMessage(data) {{
    switch(data.type) {{
        case 'status':
            updateStatus(data.message, data.status_type);
            break;
        case 'transcription':
            updateTranscription(data.text, data.detected_language);
            break;
        case 'translation':
            updateTranslation(data.target_language, data.text);
            break;
        case 'tts':
            updateTTS(data.target_language, data.audio_path);
            break;
        case 'error':
            updateStatus('Error: ' + data.message, 'error');
            break;
    }}
}}

function updateStatus(message, type) {{
    statusDiv.textContent = message;
    statusDiv.className = `status ${{type}}`;
}}

function updateTranscription(text, detectedLang) {{
    resultsDiv.style.display = 'block';
    const transcriptionDiv = document.getElementById('transcription') || createTranscriptionDiv();
    transcriptionDiv.innerHTML = `
        <strong>Detected Language:</strong> ${{detectedLang}}<br>
        <strong>Live Transcription:</strong> ${{text}}
    `;
}}

function createTranscriptionDiv() {{
    const div = document.createElement('div');
    div.id = 'transcription';
    div.className = 'transcription';
    liveContent.appendChild(div);
    return div;
}}

function updateTranslation(targetLang, text) {{
    let translationDiv = document.getElementById(`translation-${{targetLang}}`);
    if (!translationDiv) {{
        translationDiv = document.createElement('div');
        translationDiv.id = `translation-${{targetLang}}`;
        translationDiv.className = 'translation';
        liveContent.appendChild(translationDiv);
    }}
    translationDiv.innerHTML = `<strong>${{targetLang}}:</strong> ${{text}}`;
}}

function updateTTS(targetLang, audioPath) {{
    const translationDiv = document.getElementById(`translation-${{targetLang}}`);
    if (translationDiv) {{
        translationDiv.innerHTML += ` <a href="/${{audioPath}}" download>🔊 Play Audio</a>`;
    }}
}}

// Recording functions
recordBtn.addEventListener('click', startRecording);
stopBtn.addEventListener('click', stopRecording);

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
            audioLevelBar.style.width = percentage + '%';
        }};
        
        mediaRecorder = new MediaRecorder(stream);
        audioChunks = [];
        
        mediaRecorder.ondataavailable = function(event) {{
            audioChunks.push(event.data);
            
            // Send audio chunk for processing
            if (event.data.size > 0) {{
                const reader = new FileReader();
                reader.onloadend = function() {{
                    const arrayBuffer = reader.result;
                    ws.send(JSON.stringify({{
                        type: 'audio_chunk',
                        data: Array.from(new Uint8Array(arrayBuffer))
                    }}));
                }};
                reader.readAsArrayBuffer(event.data);
            }}
        }};
        
        mediaRecorder.onstop = function() {{
            const audioBlob = new Blob(audioChunks, {{ type: 'audio/webm' }});
            sendFinalAudio(audioBlob);
        }};
        
        mediaRecorder.start(100); // Collect data every 100ms
        isRecording = true;
        
        recordBtn.textContent = '🎤 Recording...';
        recordBtn.classList.add('recording');
        stopBtn.disabled = false;
        recordBtn.disabled = true;
        
        updateStatus('Recording...', 'recording');
        
        // Send target languages
        const targets = Array.from(languageCheckboxes)
            .filter(cb => cb.checked)
            .map(cb => cb.value);
        
        ws.send(JSON.stringify({{
            type: 'start_recording',
            targets: targets,
            enable_tts: enableTtsCheckbox.checked
        }}));
        
    }} catch (err) {{
        console.error('Error accessing microphone:', err);
        updateStatus('Error: Could not access microphone', 'error');
    }}
}}

function stopRecording() {{
    if (mediaRecorder && isRecording) {{
        mediaRecorder.stop();
        mediaRecorder.stream.getTracks().forEach(track => track.stop());
        
        if (audioContext) {{
            audioContext.close();
        }}
        
        isRecording = false;
        
        recordBtn.textContent = '🎤 Start Recording';
        recordBtn.classList.remove('recording');
        stopBtn.disabled = true;
        recordBtn.disabled = false;
        
        audioLevelBar.style.width = '0%';
        
        updateStatus('Processing...', 'processing');
        
        ws.send(JSON.stringify({{ type: 'stop_recording' }}));
    }}
}}

function sendFinalAudio(audioBlob) {{
    const reader = new FileReader();
    reader.onloadend = function() {{
        const arrayBuffer = reader.result;
        ws.send(JSON.stringify({{
            type: 'final_audio',
            data: Array.from(new Uint8Array(arrayBuffer))
        }}));
    }};
    reader.readAsArrayBuffer(audioBlob);
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

class ConnectionManager:
    """Manages WebSocket connections."""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)
    
    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse(LIVE_HTML)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    
    try:
        audio_buffer = []
        recording_config = {}
        processor = RealTimeAudioProcessor()
        
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message["type"] == "start_recording":
                recording_config = {
                    "targets": message["targets"],
                    "enable_tts": message["enable_tts"]
                }
                await manager.send_personal_message(
                    json.dumps({"type": "status", "message": "Recording started", "status_type": "recording"}),
                    websocket
                )
            
            elif message["type"] == "audio_chunk":
                # Process audio chunk in real-time
                audio_chunk = bytes(message["data"])
                audio_buffer.append(audio_chunk)
                
                # Try to get real-time transcription
                result = await processor.process_audio_chunk(audio_chunk)
                if result and result.get("transcription"):
                    await manager.send_personal_message(
                        json.dumps({
                            "type": "transcription",
                            "text": result["transcription"],
                            "detected_language": result.get("detected_language", "unknown"),
                            "is_final": result.get("is_final", False)
                        }),
                        websocket
                    )
            
            elif message["type"] == "stop_recording":
                await manager.send_personal_message(
                    json.dumps({"type": "status", "message": "Processing audio...", "status_type": "processing"}),
                    websocket
                )
            
            elif message["type"] == "final_audio":
                # Process the complete audio
                complete_audio = bytes(message["data"])
                await process_complete_audio(complete_audio, recording_config, websocket, processor)
                audio_buffer = []
    
    except WebSocketDisconnect:
        manager.disconnect(websocket)

async def process_complete_audio(audio_data: bytes, config: Dict, websocket: WebSocket, processor: RealTimeAudioProcessor):
    """Process the complete audio recording and send final results."""
    try:
        await manager.send_personal_message(
            json.dumps({"type": "status", "message": "Processing complete audio...", "status_type": "processing"}),
            websocket
        )
        
        # Use the real-time processor for final processing
        result = await processor.process_final_audio(
            audio_data,
            config.get("targets", ["en"]),
            config.get("enable_tts", False)
        )
        
        if "error" in result:
            await manager.send_personal_message(
                json.dumps({"type": "error", "message": result["error"]}),
                websocket
            )
            return
        
        # Send final transcription
        await manager.send_personal_message(
            json.dumps({
                "type": "transcription",
                "text": result["transcription"],
                "detected_language": result["detected_language"],
                "is_final": True
            }),
            websocket
        )
        
        # Send translations
        for translation in result["translations"]:
            await manager.send_personal_message(
                json.dumps({
                    "type": "translation",
                    "target_language": translation["target_language"],
                    "text": translation["text"]
                }),
                websocket
            )
        
        # Send TTS results if available
        for tts in result["tts_outputs"]:
            await manager.send_personal_message(
                json.dumps({
                    "type": "tts",
                    "target_language": tts["target_language"],
                    "audio_path": tts["audio_path"]
                }),
                websocket
            )
        
        await manager.send_personal_message(
            json.dumps({"type": "status", "message": "Processing complete", "status_type": "ready"}),
            websocket
        )
        
    except Exception as e:
        await manager.send_personal_message(
            json.dumps({"type": "error", "message": str(e)}),
            websocket
        )

@app.get("/uploads/{file_name}")
async def get_file(file_name: str):
    """Serve generated audio files."""
    file_path = Path(file_name)
    if file_path.exists():
        from fastapi.responses import FileResponse
        return FileResponse(file_path)
    return {"error": "File not found"}

__all__ = ["app"]
