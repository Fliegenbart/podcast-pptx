"""
PPTX-to-Podcast API.
"""
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from datetime import datetime
from enum import Enum
import uuid
import logging
import asyncio
from typing import Optional
from pydantic import BaseModel

from .config import settings
from .extractor import extract_pptx
from .dialog_generator import generate_dialog
from .tts_service import get_tts_service, setup_default_voices
from .audio_assembly import wav_to_mp3, check_ffmpeg

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# App
app = FastAPI(
    title="PPTX-to-Podcast Converter",
    description="Verwandelt PowerPoint-Präsentationen in natürliche Podcast-Gespräche",
    version="0.1.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In Produktion einschränken
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Job-Status
class JobStatus(str, Enum):
    PENDING = "pending"
    EXTRACTING = "extracting"
    GENERATING_DIALOG = "generating_dialog"
    SYNTHESIZING = "synthesizing"
    ASSEMBLING = "assembling"
    COMPLETED = "completed"
    FAILED = "failed"


class JobInfo(BaseModel):
    job_id: str
    status: JobStatus
    progress: int = 0
    message: str = ""
    created_at: datetime
    completed_at: Optional[datetime] = None
    output_file: Optional[str] = None
    duration_seconds: Optional[float] = None


# In-Memory Job-Tracking (in Produktion: Redis oder DB)
jobs: dict[str, JobInfo] = {}


@app.on_event("startup")
async def startup():
    """Initialisierung beim Start."""
    logger.info("Starte PPTX-to-Podcast Service...")
    
    # FFmpeg prüfen
    if not check_ffmpeg():
        logger.warning("FFmpeg nicht gefunden - Audio-Konvertierung wird fehlschlagen!")
    
    # Voice-Samples prüfen
    host_sample = settings.samples_dir / "host.wav"
    cohost_sample = settings.samples_dir / "cohost.wav"
    
    if not host_sample.exists() or not cohost_sample.exists():
        logger.warning(
            f"Voice-Samples fehlen! Bitte WAV-Dateien bereitstellen:\n"
            f"  - {host_sample}\n"
            f"  - {cohost_sample}\n"
            f"Samples sollten 6-30 Sekunden klar gesprochene Sprache enthalten."
        )
    else:
        setup_default_voices()
    
    logger.info("Service bereit")


@app.get("/")
async def root():
    """Health-Check."""
    return {
        "service": "PPTX-to-Podcast",
        "status": "running",
        "ollama_model": settings.ollama_model
    }


@app.get("/health")
async def health():
    """Detaillierter Health-Check."""
    return {
        "ffmpeg": check_ffmpeg(),
        "voice_samples": {
            "host": (settings.samples_dir / "host.wav").exists(),
            "cohost": (settings.samples_dir / "cohost.wav").exists()
        },
        "gpu_available": __import__("torch").cuda.is_available()
    }


@app.post("/convert", response_model=JobInfo)
async def convert_pptx(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """
    Startet die Konvertierung einer PPTX zu einem Podcast.
    
    Gibt sofort eine Job-ID zurück. Status kann über /status/{job_id} abgefragt werden.
    """
    # Validierung
    if not file.filename.lower().endswith('.pptx'):
        raise HTTPException(
            status_code=400,
            detail="Nur PPTX-Dateien werden unterstützt"
        )
    
    # Job erstellen
    job_id = str(uuid.uuid4())
    
    # Datei speichern
    upload_path = settings.upload_dir / f"{job_id}.pptx"
    content = await file.read()
    upload_path.write_bytes(content)
    
    # Job-Info
    job_info = JobInfo(
        job_id=job_id,
        status=JobStatus.PENDING,
        message="Job erstellt",
        created_at=datetime.now()
    )
    jobs[job_id] = job_info
    
    # Verarbeitung im Hintergrund
    background_tasks.add_task(process_conversion, job_id, upload_path)
    
    return job_info


async def process_conversion(job_id: str, pptx_path: Path):
    """Hintergrund-Task für die Konvertierung."""
    job = jobs[job_id]
    
    try:
        # 1. Extraktion
        job.status = JobStatus.EXTRACTING
        job.message = "Extrahiere Präsentationsinhalte..."
        job.progress = 10
        
        content = extract_pptx(pptx_path)
        logger.info(f"[{job_id}] Extrahiert: {content.total_slides} Folien")
        
        # 2. Dialog generieren
        job.status = JobStatus.GENERATING_DIALOG
        job.message = "Generiere Podcast-Dialog mit KI..."
        job.progress = 30
        
        script = generate_dialog(content)
        logger.info(f"[{job_id}] Dialog generiert: {len(script.lines)} Zeilen")
        
        # 3. TTS-Synthese
        job.status = JobStatus.SYNTHESIZING
        job.message = "Synthetisiere Sprache..."
        job.progress = 50
        
        tts = get_tts_service()
        
        def progress_callback(current: int, total: int):
            job.progress = 50 + int((current / total) * 40)
            job.message = f"Synthetisiere Zeile {current}/{total}..."
        
        wav_data = tts.synthesize_script(script, progress_callback)
        logger.info(f"[{job_id}] Audio synthetisiert")
        
        # 4. Audio-Assembly
        job.status = JobStatus.ASSEMBLING
        job.message = "Finalisiere Audio..."
        job.progress = 95
        
        output_path = settings.output_dir / f"{job_id}.mp3"
        wav_to_mp3(wav_data, output_path)
        
        # 5. Fertig
        job.status = JobStatus.COMPLETED
        job.message = "Podcast erfolgreich erstellt"
        job.progress = 100
        job.completed_at = datetime.now()
        job.output_file = str(output_path.name)
        
        # Dauer ermitteln
        from .audio_assembly import get_audio_duration
        job.duration_seconds = get_audio_duration(output_path)
        
        logger.info(f"[{job_id}] Fertig: {output_path}")
        
    except Exception as e:
        logger.exception(f"[{job_id}] Fehler: {e}")
        job.status = JobStatus.FAILED
        job.message = str(e)
    
    finally:
        # Upload-Datei aufräumen
        if pptx_path.exists():
            pptx_path.unlink()


@app.get("/status/{job_id}", response_model=JobInfo)
async def get_status(job_id: str):
    """Gibt den aktuellen Status eines Jobs zurück."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job nicht gefunden")
    
    return jobs[job_id]


@app.get("/download/{job_id}")
async def download_podcast(job_id: str):
    """Lädt den fertigen Podcast herunter."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job nicht gefunden")
    
    job = jobs[job_id]
    
    if job.status != JobStatus.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail=f"Job noch nicht abgeschlossen. Status: {job.status}"
        )
    
    output_path = settings.output_dir / job.output_file
    
    if not output_path.exists():
        raise HTTPException(status_code=404, detail="Audio-Datei nicht gefunden")
    
    return FileResponse(
        output_path,
        media_type="audio/mpeg",
        filename=f"podcast_{job_id[:8]}.mp3"
    )


@app.delete("/job/{job_id}")
async def delete_job(job_id: str):
    """Löscht einen Job und seine Dateien."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job nicht gefunden")
    
    job = jobs[job_id]
    
    # Output-Datei löschen
    if job.output_file:
        output_path = settings.output_dir / job.output_file
        if output_path.exists():
            output_path.unlink()
    
    # Job entfernen
    del jobs[job_id]
    
    return {"message": "Job gelöscht"}


# Modul-Init
from . import config
