"""
Text-to-Speech Service mit Edge TTS (Microsoft).
Kostenlos, kein GPU nötig, gute deutsche Stimmen.
"""
import asyncio
import edge_tts
import io
import wave
import tempfile
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional
import logging

from .config import settings

logger = logging.getLogger(__name__)

# Deutsche Stimmen für Host und Co-Host
VOICE_HOST = "de-DE-ConradNeural"      # Männliche Stimme
VOICE_COHOST = "de-DE-KatjaNeural"     # Weibliche Stimme

# Alternative Stimmen
# VOICE_HOST = "de-DE-FlorianMultilingualNeural"
# VOICE_COHOST = "de-DE-SeraphinaMultilingualNeural"


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


async def _synthesize_text_async(text: str, voice: str) -> bytes:
    """Synthetisiert Text zu MP3 mit Edge TTS."""
    communicate = edge_tts.Communicate(text, voice)

    mp3_data = b""
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            mp3_data += chunk["data"]

    return mp3_data


def _mp3_to_wav(mp3_data: bytes) -> bytes:
    """Konvertiert MP3 zu WAV mit FFmpeg."""
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as mp3_file:
        mp3_file.write(mp3_data)
        mp3_path = Path(mp3_file.name)

    wav_path = mp3_path.with_suffix(".wav")

    try:
        result = subprocess.run([
            "ffmpeg", "-y", "-i", str(mp3_path),
            "-ar", "24000", "-ac", "1", "-f", "wav",
            str(wav_path)
        ], capture_output=True)

        if wav_path.exists():
            wav_data = wav_path.read_bytes()
            return wav_data
        else:
            raise RuntimeError("WAV-Konvertierung fehlgeschlagen")
    finally:
        mp3_path.unlink(missing_ok=True)
        wav_path.unlink(missing_ok=True)


def _get_wav_duration_ms(wav_data: bytes) -> int:
    """Ermittelt die Dauer einer WAV-Datei in Millisekunden."""
    try:
        buffer = io.BytesIO(wav_data)
        with wave.open(buffer, 'rb') as wav:
            frames = wav.getnframes()
            rate = wav.getframerate()
            return int(frames / rate * 1000)
    except Exception:
        return 0


def _run_async_in_thread(coro):
    """Führt eine Coroutine in einem separaten Thread aus."""
    import concurrent.futures

    def run():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(run)
        return future.result()


def synthesize_text(text: str, speaker: str = "host") -> AudioSegment:
    """
    Synthetisiert einen Text zu Audio.

    Args:
        text: Der zu synthetisierende Text
        speaker: "host" oder "cohost"

    Returns:
        AudioSegment mit den Audio-Daten
    """
    voice = VOICE_HOST if speaker == "host" else VOICE_COHOST

    # Async in separatem Thread ausführen (vermeidet Event-Loop-Konflikt)
    mp3_data = _run_async_in_thread(_synthesize_text_async(text, voice))

    # MP3 zu WAV konvertieren
    wav_data = _mp3_to_wav(mp3_data)
    duration_ms = _get_wav_duration_ms(wav_data)

    return AudioSegment(
        audio_data=wav_data,
        duration_ms=duration_ms,
        speaker=speaker,
        text=text,
        sample_rate=24000
    )


def synthesize_script(
    script,  # PodcastScript
    progress_callback: Optional[Callable[[int, int], None]] = None
) -> SynthesisResult:
    """
    Synthetisiert ein komplettes Podcast-Skript.

    Args:
        script: PodcastScript mit allen Dialog-Zeilen
        progress_callback: Optional Callback für Fortschrittsanzeige

    Returns:
        SynthesisResult mit allen Audio-Segmenten
    """
    segments = []
    total_duration_ms = 0
    total_lines = len(script.lines)

    logger.info(f"Synthetisiere {total_lines} Zeilen mit Edge TTS...")

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


class EdgeTTSService:
    """Service-Klasse für Edge TTS."""

    def __init__(self):
        self.host_voice = VOICE_HOST
        self.cohost_voice = VOICE_COHOST

    def synthesize_text(self, text: str, speaker: str = "host") -> AudioSegment:
        return synthesize_text(text, speaker)

    def synthesize_script(
        self,
        script,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> SynthesisResult:
        return synthesize_script(script, progress_callback)


# Singleton
_service_instance: Optional[EdgeTTSService] = None


def get_tts_service() -> EdgeTTSService:
    """Gibt die TTS-Service-Instanz zurück."""
    global _service_instance
    if _service_instance is None:
        _service_instance = EdgeTTSService()
    return _service_instance


def setup_default_voices():
    """Prüft ob Edge TTS verfügbar ist."""
    try:
        # Prüfen ob edge_tts importiert werden kann
        import edge_tts
        logger.info(f"Edge TTS bereit - Stimmen: {VOICE_HOST} (Host), {VOICE_COHOST} (Co-Host)")
        return True
    except ImportError as e:
        logger.warning(f"Edge TTS nicht verfügbar: {e}")
        return False
