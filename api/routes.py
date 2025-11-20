"""API routes for AudioLingo."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import List, Sequence

from fastapi import APIRouter, BackgroundTasks, File, Form, UploadFile

from app.pipeline import AudioLingoPipeline

router = APIRouter(prefix="/api")
pipeline = AudioLingoPipeline()


def _save_upload(upload: UploadFile) -> Path:
    suffix = Path(upload.filename or "audio.wav").suffix or ".wav"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    content = upload.file.read()
    tmp.write(content)
    tmp.flush()
    tmp.close()
    return Path(tmp.name)


@router.get("/health")
def health() -> dict:
    return {"status": "ok"}


@router.post("/translate")
async def translate_audio(
    background_tasks: BackgroundTasks,
    audio_file: UploadFile = File(...),
    target_languages: str = Form(..., description="Comma separated values"),
    enable_tts: bool = Form(False),
) -> dict:
    upload_path = _save_upload(audio_file)
    targets: Sequence[str] = [t.strip() for t in target_languages.split(",") if t.strip()]

    def cleanup(path: Path) -> None:
        if path.exists():
            path.unlink()

    background_tasks.add_task(cleanup, upload_path)
    result = pipeline.run(
        str(upload_path),
        target_languages=targets or ["en"],
        enable_tts=enable_tts,
    )
    return result

