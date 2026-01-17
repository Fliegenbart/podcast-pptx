"""
Hybrider Text-to-Speech Service.
Wählt automatisch das beste Backend:
- CosyVoice 3 (GPU + Voice Samples) für beste Qualität
- XTTS v2 (GPU + Voice Samples) als Fallback
- Edge TTS (Fallback) für CPU-only Systeme
"""
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Literal

from .config import settings

logger = logging.getLogger(__name__)


@dataclass
class AudioSegment:
    """Ein Audio-Segment mit Metadaten."""
    audio_data: bytes
    duration_ms: int
    speaker: str
    text: str
    sample_rate: int = 24000


@dataclass
class SynthesisResult:
    """Ergebnis der Sprach-Synthese."""
    segments: list[AudioSegment]
    total_duration_ms: int


# TTS Backend Typen
TTSBackend = Literal["cosyvoice", "xtts", "edge"]

# Globale Variablen
_tts_backend: Optional[TTSBackend] = None
_xtts_instance = None


def _check_gpu_available() -> bool:
    """Prüft ob CUDA GPU verfügbar ist."""
    try:
        import torch
        available = torch.cuda.is_available()
        if available:
            device_name = torch.cuda.get_device_name(0)
            logger.info(f"GPU gefunden: {device_name}")
        return available
    except ImportError:
        return False
    except Exception as e:
        logger.debug(f"GPU-Check fehlgeschlagen: {e}")
        return False


def _check_voice_samples() -> bool:
    """Prüft ob Voice-Samples vorhanden sind."""
    host_sample = settings.samples_dir / "host.wav"
    cohost_sample = settings.samples_dir / "cohost.wav"
    return host_sample.exists() and cohost_sample.exists()


def _check_xtts_available() -> bool:
    """Prüft ob XTTS v2 verwendbar ist."""
    try:
        # TTS-Bibliothek importieren
        from TTS.api import TTS
        return True
    except ImportError:
        return False
    except Exception as e:
        logger.debug(f"XTTS nicht verfügbar: {e}")
        return False


def _check_cosyvoice_available() -> bool:
    """Prüft ob CosyVoice verfügbar ist."""
    try:
        from .cosyvoice_service import check_cosyvoice_available
        return check_cosyvoice_available()
    except ImportError:
        return False
    except Exception as e:
        logger.debug(f"CosyVoice nicht verfügbar: {e}")
        return False


def detect_best_backend() -> TTSBackend:
    """
    Erkennt automatisch das beste TTS-Backend.

    Priorität:
    1. CosyVoice 3 wenn GPU + Voice Samples + CosyVoice verfügbar
    2. XTTS v2 wenn GPU + Voice Samples + TTS-Bibliothek verfügbar
    3. Edge TTS als Fallback
    """
    global _tts_backend

    if _tts_backend is not None:
        return _tts_backend

    # Prüfe Voraussetzungen
    has_gpu = _check_gpu_available()
    has_samples = _check_voice_samples()
    has_cosyvoice = _check_cosyvoice_available()
    has_xtts = _check_xtts_available()

    logger.info(f"TTS-Backend-Erkennung: GPU={has_gpu}, Samples={has_samples}, CosyVoice={has_cosyvoice}, XTTS={has_xtts}")

    # CosyVoice 3 hat höchste Priorität
    if has_gpu and has_samples and has_cosyvoice:
        _tts_backend = "cosyvoice"
        logger.info("TTS-Backend: CosyVoice 3 (GPU + Voice Cloning)")
    elif has_gpu and has_samples and has_xtts:
        _tts_backend = "xtts"
        logger.info("TTS-Backend: XTTS v2 (GPU + Voice Cloning)")
    else:
        _tts_backend = "edge"
        if not has_gpu:
            logger.info("TTS-Backend: Edge TTS (keine GPU gefunden)")
        elif not has_samples:
            logger.info("TTS-Backend: Edge TTS (keine Voice-Samples)")
        else:
            logger.info("TTS-Backend: Edge TTS (kein GPU-TTS verfügbar)")

    return _tts_backend


def _get_xtts_instance():
    """Lädt XTTS v2 Modell (lazy loading)."""
    global _xtts_instance

    if _xtts_instance is not None:
        return _xtts_instance

    from TTS.api import TTS
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Lade XTTS v2 auf {device}...")

    _xtts_instance = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    logger.info("XTTS v2 geladen")

    return _xtts_instance


def synthesize_text_cosyvoice(text: str, speaker: str = "host") -> AudioSegment:
    """Synthetisiert Text mit CosyVoice 3."""
    from .cosyvoice_service import synthesize_text as cosyvoice_synthesize
    result = cosyvoice_synthesize(text, speaker)
    # Konvertiere zu lokalem AudioSegment Format
    return AudioSegment(
        audio_data=result.audio_data,
        duration_ms=result.duration_ms,
        speaker=result.speaker,
        text=result.text,
        sample_rate=result.sample_rate
    )


def synthesize_text_xtts(text: str, speaker: str = "host") -> AudioSegment:
    """Synthetisiert Text mit XTTS v2."""
    import tempfile
    import wave
    import io

    tts = _get_xtts_instance()

    # Voice Sample auswählen
    sample_path = settings.samples_dir / f"{speaker}.wav"
    if not sample_path.exists():
        sample_path = settings.samples_dir / "host.wav"

    # Synthese
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        tts.tts_to_file(
            text=text,
            file_path=str(tmp_path),
            speaker_wav=str(sample_path),
            language="de"
        )

        # WAV lesen
        wav_data = tmp_path.read_bytes()

        # Dauer ermitteln
        buffer = io.BytesIO(wav_data)
        with wave.open(buffer, 'rb') as wav:
            frames = wav.getnframes()
            rate = wav.getframerate()
            duration_ms = int(frames / rate * 1000)

        return AudioSegment(
            audio_data=wav_data,
            duration_ms=duration_ms,
            speaker=speaker,
            text=text,
            sample_rate=24000
        )
    finally:
        tmp_path.unlink(missing_ok=True)


def synthesize_text_edge(text: str, speaker: str = "host") -> AudioSegment:
    """Synthetisiert Text mit Edge TTS."""
    from .edge_tts_service import synthesize_text
    return synthesize_text(text, speaker)


def synthesize_text(text: str, speaker: str = "host") -> AudioSegment:
    """
    Synthetisiert Text zu Audio.
    Wählt automatisch das beste Backend.
    """
    backend = detect_best_backend()

    if backend == "cosyvoice":
        return synthesize_text_cosyvoice(text, speaker)
    elif backend == "xtts":
        return synthesize_text_xtts(text, speaker)
    else:
        return synthesize_text_edge(text, speaker)


def synthesize_script(
    script,  # PodcastScript
    progress_callback: Optional[Callable[[int, int], None]] = None
) -> SynthesisResult:
    """
    Synthetisiert ein komplettes Podcast-Skript.
    """
    backend = detect_best_backend()
    segments = []
    total_duration_ms = 0
    total_lines = len(script.lines)

    logger.info(f"Synthetisiere {total_lines} Zeilen mit {backend.upper()}...")

    for i, line in enumerate(script.lines):
        if progress_callback:
            progress_callback(i + 1, total_lines)

        try:
            segment = synthesize_text(
                text=line.text,
                speaker=line.speaker
            )
            segments.append(segment)
            total_duration_ms += segment.duration_ms

            logger.debug(f"Zeile {i+1}/{total_lines}: {segment.duration_ms}ms")

        except Exception as e:
            logger.error(f"Fehler bei Zeile {i + 1}: {e}")
            # Leeres Segment als Platzhalter
            segments.append(AudioSegment(
                audio_data=b"",
                duration_ms=0,
                speaker=line.speaker,
                text=line.text
            ))

    logger.info(f"Synthese abgeschlossen: {total_duration_ms}ms Gesamtdauer")

    return SynthesisResult(
        segments=segments,
        total_duration_ms=total_duration_ms
    )


class TTSService:
    """
    Hybrider TTS-Service.
    Wählt automatisch XTTS v2 (GPU) oder Edge TTS (CPU).
    """

    def __init__(self):
        self.backend = detect_best_backend()

    def synthesize_text(self, text: str, speaker: str = "host") -> AudioSegment:
        return synthesize_text(text, speaker)

    def synthesize_script(
        self,
        script,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> SynthesisResult:
        return synthesize_script(script, progress_callback)

    def get_backend_info(self) -> dict:
        """Gibt Informationen über das verwendete Backend zurück."""
        return {
            "backend": self.backend,
            "gpu_available": _check_gpu_available(),
            "voice_samples": _check_voice_samples(),
            "cosyvoice_available": _check_cosyvoice_available(),
            "xtts_available": _check_xtts_available()
        }


# Singleton
_service_instance: Optional[TTSService] = None


def get_tts_service() -> TTSService:
    """Gibt die TTS-Service-Instanz zurück."""
    global _service_instance
    if _service_instance is None:
        _service_instance = TTSService()
    return _service_instance


def setup_default_voices() -> bool:
    """Initialisiert das TTS-System und gibt Status zurück."""
    backend = detect_best_backend()

    if backend == "cosyvoice":
        logger.info("TTS: CosyVoice 3 mit Voice Cloning aktiviert")
        logger.info(f"  Host-Stimme: {settings.samples_dir / 'host.wav'}")
        logger.info(f"  Co-Host-Stimme: {settings.samples_dir / 'cohost.wav'}")
    elif backend == "xtts":
        logger.info("TTS: XTTS v2 mit Voice Cloning aktiviert")
        logger.info(f"  Host-Stimme: {settings.samples_dir / 'host.wav'}")
        logger.info(f"  Co-Host-Stimme: {settings.samples_dir / 'cohost.wav'}")
    else:
        from .edge_tts_service import VOICE_HOST, VOICE_COHOST
        logger.info(f"TTS: Edge TTS - {VOICE_HOST} (Host), {VOICE_COHOST} (Co-Host)")

    return True
