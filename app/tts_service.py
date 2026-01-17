"""
Text-to-Speech Service mit XTTS v2.
Unterstützt zwei verschiedene Stimmen für Host und Co-Host.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional
import logging
import io
import wave
import struct

import torch
from TTS.api import TTS

from .config import settings

logger = logging.getLogger(__name__)

# Globale TTS-Instanz (wird beim ersten Aufruf initialisiert)
_tts_instance: Optional[TTS] = None
_voices_loaded: bool = False


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


def get_tts_instance() -> TTS:
    """Gibt die TTS-Instanz zurück, initialisiert sie bei Bedarf."""
    global _tts_instance

    if _tts_instance is None:
        logger.info("Initialisiere XTTS v2...")

        # GPU prüfen
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Verwende Device: {device}")

        # XTTS v2 laden
        _tts_instance = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

        logger.info("XTTS v2 geladen")

    return _tts_instance


def setup_default_voices():
    """Prüft und lädt die Standard-Voice-Samples."""
    host_sample = settings.samples_dir / "host.wav"
    cohost_sample = settings.samples_dir / "cohost.wav"

    if not host_sample.exists():
        logger.warning(f"Host-Sample fehlt: {host_sample}")
    if not cohost_sample.exists():
        logger.warning(f"Co-Host-Sample fehlt: {cohost_sample}")

    global _voices_loaded
    _voices_loaded = host_sample.exists() and cohost_sample.exists()

    return _voices_loaded


def _get_audio_duration_ms(audio_data: bytes, sample_rate: int) -> int:
    """Berechnet die Dauer eines Audio-Segments in Millisekunden."""
    # Annahme: 16-bit Mono WAV
    num_samples = len(audio_data) // 2
    duration_seconds = num_samples / sample_rate
    return int(duration_seconds * 1000)


def _tensor_to_wav_bytes(audio_tensor: torch.Tensor, sample_rate: int) -> bytes:
    """Konvertiert einen Audio-Tensor zu WAV-Bytes."""
    # Normalisieren auf [-1, 1]
    audio = audio_tensor.cpu().numpy()
    if audio.max() > 1.0 or audio.min() < -1.0:
        audio = audio / max(abs(audio.max()), abs(audio.min()))

    # In 16-bit Integer konvertieren
    audio_int16 = (audio * 32767).astype("int16")

    # WAV-Datei in Memory erstellen
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_int16.tobytes())

    return buffer.getvalue()


def synthesize_text(
    text: str,
    speaker: str = "host",
    emotion: str = "neutral"
) -> AudioSegment:
    """
    Synthetisiert einen Text zu Audio.

    Args:
        text: Der zu synthetisierende Text
        speaker: "host" oder "cohost"
        emotion: Emotion (aktuell nicht verwendet)

    Returns:
        AudioSegment mit den Audio-Daten
    """
    tts = get_tts_instance()

    # Voice-Sample auswählen
    sample_path = settings.samples_dir / f"{speaker}.wav"
    if not sample_path.exists():
        raise FileNotFoundError(f"Voice-Sample nicht gefunden: {sample_path}")

    # Text synthetisieren
    audio_tensor = tts.tts(
        text=text,
        speaker_wav=str(sample_path),
        language=settings.tts_language
    )

    # Zu Tensor konvertieren falls nötig
    if not isinstance(audio_tensor, torch.Tensor):
        audio_tensor = torch.tensor(audio_tensor)

    # In WAV konvertieren
    sample_rate = settings.tts_sample_rate
    audio_bytes = _tensor_to_wav_bytes(audio_tensor, sample_rate)

    # Dauer berechnen
    duration_ms = int(len(audio_tensor) / sample_rate * 1000)

    return AudioSegment(
        audio_data=audio_bytes,
        duration_ms=duration_ms,
        speaker=speaker,
        text=text,
        sample_rate=sample_rate
    )


def synthesize_script(
    script,  # PodcastScript - vermeidet zirkulären Import
    progress_callback: Optional[Callable[[int, int], None]] = None
) -> SynthesisResult:
    """
    Synthetisiert ein komplettes Podcast-Skript.

    Args:
        script: PodcastScript mit allen Dialog-Zeilen
        progress_callback: Optional Callback für Fortschrittsanzeige (current, total)

    Returns:
        SynthesisResult mit allen Audio-Segmenten
    """
    segments = []
    total_duration_ms = 0

    total_lines = len(script.lines)

    for i, line in enumerate(script.lines):
        if progress_callback:
            progress_callback(i + 1, total_lines)

        try:
            segment = synthesize_text(
                text=line.text,
                speaker=line.speaker,
                emotion=line.emotion
            )
            segments.append(segment)
            total_duration_ms += segment.duration_ms

        except Exception as e:
            logger.error(f"Fehler bei Zeile {i + 1}: {e}")
            # Leeres Segment als Platzhalter
            segments.append(AudioSegment(
                audio_data=b"",
                duration_ms=0,
                speaker=line.speaker,
                text=line.text
            ))

    return SynthesisResult(
        segments=segments,
        total_duration_ms=total_duration_ms
    )


class TTSService:
    """
    Service-Klasse für TTS-Operationen.
    Kapselt alle TTS-Funktionalität.
    """

    def __init__(self):
        self._tts: Optional[TTS] = None

    def initialize(self):
        """Initialisiert den TTS-Service."""
        self._tts = get_tts_instance()
        setup_default_voices()

    def synthesize_text(self, text: str, speaker: str = "host") -> AudioSegment:
        """Synthetisiert einen Text."""
        return synthesize_text(text, speaker)

    def synthesize_script(
        self,
        script,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> SynthesisResult:
        """Synthetisiert ein komplettes Skript."""
        return synthesize_script(script, progress_callback)


# Singleton-Instanz
_service_instance: Optional[TTSService] = None


def get_tts_service() -> TTSService:
    """Gibt die TTS-Service-Instanz zurück."""
    global _service_instance

    if _service_instance is None:
        _service_instance = TTSService()
        _service_instance.initialize()

    return _service_instance
