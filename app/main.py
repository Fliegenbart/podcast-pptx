"""
PPTX-to-Podcast API.
Accessibility-fokussierte Konvertierung von Präsentationen zu navigierbaren Podcasts.
"""
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from datetime import datetime
from enum import Enum
import uuid
import logging
from typing import Optional
from pydantic import BaseModel

from .config import settings
from .extractor import extract_pptx
from .image_describer import describe_images_in_content
from .dialog_generator import generate_dialog
from .audio_assembly import check_ffmpeg, create_final_audio, get_audio_duration
from .chapter_manager import ChapterTracker

# Hybrider TTS Service (XTTS v2 mit GPU oder Edge TTS als Fallback)
from .tts_service import get_tts_service, setup_default_voices

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# App
app = FastAPI(
    title="PPTX-to-Podcast Converter",
    description="Verwandelt PowerPoint-Präsentationen in natürliche, barrierefreie Podcast-Gespräche mit Kapitelnavigation",
    version="0.2.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In Produktion einschränken
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Statische Dateien
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


# Job-Status
class JobStatus(str, Enum):
    PENDING = "pending"
    EXTRACTING = "extracting"
    DESCRIBING_IMAGES = "describing_images"  # NEU
    GENERATING_DIALOG = "generating_dialog"
    SYNTHESIZING = "synthesizing"
    ASSEMBLING = "assembling"
    COMPLETED = "completed"
    FAILED = "failed"


class ChapterInfo(BaseModel):
    """Kapitelinformation für API-Response."""
    title: str
    start_seconds: float
    slide_number: Optional[int] = None


class JobInfo(BaseModel):
    job_id: str
    status: JobStatus
    progress: int = 0
    message: str = ""
    created_at: datetime
    completed_at: Optional[datetime] = None
    output_file: Optional[str] = None
    output_format: Optional[str] = None
    duration_seconds: Optional[float] = None
    chapters: list[ChapterInfo] = []
    total_slides: Optional[int] = None
    images_described: Optional[int] = None


# In-Memory Job-Tracking (in Produktion: Redis oder DB)
jobs: dict[str, JobInfo] = {}


@app.on_event("startup")
async def startup():
    """Initialisierung beim Start."""
    logger.info("Starte PPTX-to-Podcast Service v0.2.0...")

    # FFmpeg prüfen
    if not check_ffmpeg():
        logger.warning("FFmpeg nicht gefunden - Audio-Konvertierung wird fehlschlagen!")

    # Edge TTS prüfen
    setup_default_voices()

    logger.info(f"Output-Format: {settings.output_format}")
    logger.info(f"Bildbeschreibung: {'aktiviert' if settings.image_description_enabled else 'deaktiviert'}")
    logger.info(f"Audio-Cues: {'aktiviert' if settings.enable_audio_cues else 'deaktiviert'}")
    logger.info("Service bereit")


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve Frontend."""
    html_path = Path(__file__).parent / "static" / "index.html"
    if html_path.exists():
        return html_path.read_text()
    return HTMLResponse("<h1>PPTX-to-Podcast API</h1><p>Frontend nicht gefunden. Nutze /docs für API.</p>")


@app.get("/api")
async def api_info():
    """API-Info."""
    return {
        "service": "PPTX-to-Podcast",
        "version": "0.2.0",
        "status": "running",
        "features": {
            "image_description": settings.image_description_enabled,
            "chapter_markers": True,
            "audio_cues": settings.enable_audio_cues,
            "output_format": settings.output_format
        }
    }


@app.get("/health")
async def health():
    """Detaillierter Health-Check."""
    try:
        import torch
        gpu_available = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if gpu_available else None
    except Exception:
        gpu_available = False
        gpu_name = None

    # TTS Backend Info
    tts = get_tts_service()
    tts_info = tts.get_backend_info()

    return {
        "ffmpeg": check_ffmpeg(),
        "tts": {
            "backend": tts_info["backend"],
            "gpu_accelerated": tts_info["backend"] == "xtts",
            "voice_samples_available": tts_info["voice_samples"]
        },
        "gpu": {
            "available": gpu_available,
            "name": gpu_name
        },
        "models": {
            "dialog": settings.ollama_model,
            "vision": settings.vision_model if settings.image_description_enabled else "disabled",
            "tts": "XTTS v2" if tts_info["backend"] == "xtts" else "Edge TTS"
        }
    }


@app.post("/convert", response_model=JobInfo)
async def convert_pptx(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """
    Startet die Konvertierung einer PPTX zu einem barrierefreien Podcast.

    Gibt sofort eine Job-ID zurück. Status kann über /status/{job_id} abgefragt werden.

    Features:
    - KI-gestützte Bildbeschreibungen
    - Kapitelmarker für Navigation
    - Audio-Cues bei Folienwechseln
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
        job.progress = 5

        content = extract_pptx(pptx_path)
        job.total_slides = content.total_slides
        total_images = sum(len(s.images) for s in content.slides)

        logger.info(f"[{job_id}] Extrahiert: {content.total_slides} Folien, {total_images} Bilder")

        # 2. Bildbeschreibung (NEU)
        if settings.image_description_enabled and total_images > 0:
            job.status = JobStatus.DESCRIBING_IMAGES
            job.message = f"Beschreibe {total_images} Bilder mit KI..."
            job.progress = 15

            content = describe_images_in_content(content)

            described = sum(
                1 for s in content.slides
                for img in s.images
                if img.description
            )
            job.images_described = described
            logger.info(f"[{job_id}] Bilder beschrieben: {described}/{total_images}")

        # 3. Dialog generieren
        job.status = JobStatus.GENERATING_DIALOG
        job.message = "Generiere Podcast-Dialog mit KI..."
        job.progress = 30

        script = generate_dialog(content)
        logger.info(f"[{job_id}] Dialog generiert: {len(script.lines)} Zeilen, {len(script.chapters)} Kapitel")

        # 4. TTS-Synthese mit Kapitel-Tracking
        job.status = JobStatus.SYNTHESIZING
        job.message = "Synthetisiere Sprache..."
        job.progress = 45

        tts = get_tts_service()
        chapter_tracker = ChapterTracker()
        chapter_tracker.start_intro()

        current_slide = 0

        def progress_callback(current: int, total: int):
            job.progress = 45 + int((current / total) * 40)
            job.message = f"Synthetisiere Zeile {current}/{total}..."

        # TTS mit Kapitel-Tracking
        synthesis_result = tts.synthesize_script(script, progress_callback)

        # Kapitel-Timestamps aus Synthese-Ergebnis berechnen
        current_time_ms = 0
        chapter_times = {}

        for i, (line, segment) in enumerate(zip(script.lines, synthesis_result.segments)):
            if line.marks_chapter_start:
                # Finde passendes Kapitel
                for ch in script.chapters:
                    if ch.slide_number == (line.slide_context.slide_number if line.slide_context else None):
                        chapter_times[ch.title] = current_time_ms
                        break

            current_time_ms += segment.duration_ms

        # Kapitel-Zeiten aktualisieren
        for chapter in script.chapters:
            if chapter.title in chapter_times:
                chapter.start_ms = chapter_times[chapter.title]

        logger.info(f"[{job_id}] Audio synthetisiert: {synthesis_result.total_duration_ms}ms")

        # 5. Audio-Assembly
        job.status = JobStatus.ASSEMBLING
        job.message = "Finalisiere Audio mit Kapiteln..."
        job.progress = 90

        # Kapitel-Metadaten erstellen
        from .chapter_manager import ChapterList
        chapter_list = ChapterList(chapters=script.chapters)
        chapter_list.finalize(synthesis_result.total_duration_ms)
        chapters_metadata = chapter_list.to_ffmpeg_metadata()

        # Audio-Cue-Map erstellen
        audio_cue_map = script.get_audio_cue_map()

        # Finales Audio erstellen
        output_path = settings.output_dir / f"{job_id}"
        final_path = create_final_audio(
            synthesis_result=synthesis_result,
            output_path=output_path,
            chapters_metadata=chapters_metadata,
            audio_cues=audio_cue_map
        )

        # 6. Fertig
        job.status = JobStatus.COMPLETED
        job.message = "Podcast erfolgreich erstellt"
        job.progress = 100
        job.completed_at = datetime.now()
        job.output_file = final_path.name
        job.output_format = settings.output_format

        # Dauer ermitteln
        job.duration_seconds = get_audio_duration(final_path)

        # Kapitel-Info für API
        job.chapters = [
            ChapterInfo(
                title=ch.title,
                start_seconds=ch.start_ms / 1000,
                slide_number=ch.slide_number
            )
            for ch in script.chapters
        ]

        logger.info(f"[{job_id}] Fertig: {final_path}")

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
    """Gibt den aktuellen Status eines Jobs zurück, inkl. Kapitelinformationen."""
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

    # MIME-Type basierend auf Format
    mime_type = "audio/mp4" if job.output_format == "m4b" else "audio/mpeg"
    extension = job.output_format or "mp3"

    return FileResponse(
        output_path,
        media_type=mime_type,
        filename=f"podcast_{job_id[:8]}.{extension}"
    )


@app.get("/chapters/{job_id}")
async def get_chapters(job_id: str):
    """Gibt nur die Kapitelinformationen eines Jobs zurück."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job nicht gefunden")

    job = jobs[job_id]

    if job.status != JobStatus.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail=f"Job noch nicht abgeschlossen. Status: {job.status}"
        )

    return {
        "job_id": job_id,
        "total_chapters": len(job.chapters),
        "duration_seconds": job.duration_seconds,
        "chapters": job.chapters
    }


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


@app.get("/settings")
async def get_settings():
    """Gibt aktuelle Accessibility-Einstellungen zurück."""
    return {
        "image_description": {
            "enabled": settings.image_description_enabled,
            "model": settings.vision_model,
            "skip_decorative": settings.skip_decorative_images
        },
        "chapters": {
            "per_slide": settings.chapter_per_slide,
            "per_section": settings.chapter_per_section
        },
        "audio_cues": {
            "enabled": settings.enable_audio_cues,
            "volume_db": settings.cue_volume_db
        },
        "structure_announcements": {
            "slide_position": settings.announce_slide_position,
            "section_changes": settings.announce_section_changes
        },
        "output": {
            "format": settings.output_format,
            "bitrate": settings.audio_bitrate
        }
    }
