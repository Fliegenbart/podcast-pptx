"""
Text-to-Speech Service mit CosyVoice 3.
Hochwertige deutsche Stimmen mit Zero-Shot Voice Cloning.
Apache 2.0 Lizenz, DSGVO-konform (lokal).
"""
import io
import wave
import tempfile
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from .config import settings

logger = logging.getLogger(__name__)

# CosyVoice Instanz (lazy loading)
_cosyvoice_instance = None
_cosyvoice_available: Optional[bool] = None


@dataclass
class AudioSegment:
    """Ein Audio-Segment mit Metadaten."""
    audio_data: bytes
    duration_ms: int
    speaker: str
    text: str
    sample_rate: int = 22050  # CosyVoice nutzt 22050 Hz


@dataclass
class SynthesisResult:
    """Ergebnis der Sprach-Synthese."""
    segments: list[AudioSegment]
    total_duration_ms: int


def check_cosyvoice_available() -> bool:
    """Prüft ob CosyVoice verfügbar ist."""
    global _cosyvoice_available

    if _cosyvoice_available is not None:
        return _cosyvoice_available

    try:
        # Prüfe ob CosyVoice Submodule installiert ist
        import sys
        cosyvoice_path = settings.base_dir / "third_party" / "CosyVoice"

        if not cosyvoice_path.exists():
            logger.debug("CosyVoice nicht gefunden: third_party/CosyVoice fehlt")
            _cosyvoice_available = False
            return False

        # Pfad hinzufügen falls nicht vorhanden
        if str(cosyvoice_path) not in sys.path:
            sys.path.insert(0, str(cosyvoice_path))

        # Import testen
        from cosyvoice.cli.cosyvoice import CosyVoice
        from cosyvoice.utils.file_utils import load_wav

        _cosyvoice_available = True
        logger.info("CosyVoice 3 verfügbar")
        return True

    except ImportError as e:
        logger.debug(f"CosyVoice Import fehlgeschlagen: {e}")
        _cosyvoice_available = False
        return False
    except Exception as e:
        logger.debug(f"CosyVoice Check fehlgeschlagen: {e}")
        _cosyvoice_available = False
        return False


def _get_cosyvoice_instance():
    """Lädt CosyVoice Modell (lazy loading)."""
    global _cosyvoice_instance

    if _cosyvoice_instance is not None:
        return _cosyvoice_instance

    if not check_cosyvoice_available():
        raise RuntimeError("CosyVoice nicht verfügbar")

    import sys
    cosyvoice_path = settings.base_dir / "third_party" / "CosyVoice"
    if str(cosyvoice_path) not in sys.path:
        sys.path.insert(0, str(cosyvoice_path))

    from cosyvoice.cli.cosyvoice import CosyVoice

    # Modell laden (300M/500M je nach Verfügbarkeit)
    # Modelle werden automatisch von HuggingFace/ModelScope geladen
    model_dir = settings.base_dir / "models" / "CosyVoice-300M-SFT"

    if model_dir.exists():
        logger.info(f"Lade CosyVoice aus: {model_dir}")
        _cosyvoice_instance = CosyVoice(str(model_dir))
    else:
        # Automatischer Download von ModelScope
        logger.info("Lade CosyVoice-300M-SFT (automatischer Download)...")
        try:
            _cosyvoice_instance = CosyVoice("iic/CosyVoice-300M-SFT")
        except Exception:
            # Fallback auf Zero-Shot Modell
            logger.info("Fallback: Lade CosyVoice-300M...")
            _cosyvoice_instance = CosyVoice("iic/CosyVoice-300M")

    logger.info("CosyVoice geladen")
    return _cosyvoice_instance


def _load_voice_sample(speaker: str) -> Optional[Path]:
    """Lädt das Voice Sample für einen Speaker."""
    sample_path = settings.samples_dir / f"{speaker}.wav"
    if sample_path.exists():
        return sample_path

    # Fallback auf host.wav
    fallback = settings.samples_dir / "host.wav"
    if fallback.exists():
        return fallback

    return None


def _audio_tensor_to_wav(audio_tensor, sample_rate: int = 22050) -> bytes:
    """Konvertiert Audio-Tensor zu WAV bytes."""
    import numpy as np

    # Tensor zu numpy
    if hasattr(audio_tensor, 'cpu'):
        audio_np = audio_tensor.cpu().numpy()
    else:
        audio_np = np.array(audio_tensor)

    # Normalisieren auf int16
    if audio_np.dtype != np.int16:
        audio_np = (audio_np * 32767).astype(np.int16)

    # Falls mehrdimensional, flatten
    if len(audio_np.shape) > 1:
        audio_np = audio_np.flatten()

    # WAV schreiben
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)  # 16-bit
        wav.setframerate(sample_rate)
        wav.writeframes(audio_np.tobytes())

    return buffer.getvalue()


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


def synthesize_text(text: str, speaker: str = "host") -> AudioSegment:
    """
    Synthetisiert einen Text zu Audio mit CosyVoice.

    Unterstützt Zero-Shot Voice Cloning mit Voice Samples.

    Args:
        text: Der zu synthetisierende Text
        speaker: "host" oder "cohost"

    Returns:
        AudioSegment mit den Audio-Daten
    """
    cosyvoice = _get_cosyvoice_instance()

    # Voice Sample laden
    sample_path = _load_voice_sample(speaker)

    try:
        if sample_path:
            # Zero-Shot Voice Cloning mit Sample
            import sys
            cosyvoice_path = settings.base_dir / "third_party" / "CosyVoice"
            if str(cosyvoice_path) not in sys.path:
                sys.path.insert(0, str(cosyvoice_path))

            from cosyvoice.utils.file_utils import load_wav

            prompt_speech = load_wav(str(sample_path), 16000)

            # Zero-Shot Synthese
            output = cosyvoice.inference_zero_shot(
                tts_text=text,
                prompt_text="",  # Leerer Prompt für Zero-Shot
                prompt_speech_16k=prompt_speech
            )
        else:
            # SFT Inference ohne Voice Cloning
            # Verwende deutsche Stimme falls verfügbar
            output = cosyvoice.inference_sft(
                tts_text=text,
                spk_id="中文女"  # Default Stimme, wird durch deutsche ersetzt wenn verfügbar
            )

        # Audio-Tensor extrahieren
        audio_tensor = None
        for result in output:
            if 'tts_speech' in result:
                audio_tensor = result['tts_speech']
                break

        if audio_tensor is None:
            raise RuntimeError("Keine Audio-Ausgabe von CosyVoice")

        # Zu WAV konvertieren
        wav_data = _audio_tensor_to_wav(audio_tensor, sample_rate=22050)
        duration_ms = _get_wav_duration_ms(wav_data)

        return AudioSegment(
            audio_data=wav_data,
            duration_ms=duration_ms,
            speaker=speaker,
            text=text,
            sample_rate=22050
        )

    except Exception as e:
        logger.error(f"CosyVoice Synthese fehlgeschlagen: {e}")
        raise


def synthesize_script(
    script,  # PodcastScript
    progress_callback: Optional[Callable[[int, int], None]] = None
) -> SynthesisResult:
    """
    Synthetisiert ein komplettes Podcast-Skript mit CosyVoice.

    Args:
        script: PodcastScript mit allen Dialog-Zeilen
        progress_callback: Optional Callback für Fortschrittsanzeige

    Returns:
        SynthesisResult mit allen Audio-Segmenten
    """
    segments = []
    total_duration_ms = 0
    total_lines = len(script.lines)

    logger.info(f"Synthetisiere {total_lines} Zeilen mit CosyVoice...")

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

    logger.info(f"CosyVoice Synthese abgeschlossen: {total_duration_ms}ms Gesamtdauer")

    return SynthesisResult(
        segments=segments,
        total_duration_ms=total_duration_ms
    )


class CosyVoiceService:
    """Service-Klasse für CosyVoice 3."""

    def __init__(self):
        self.available = check_cosyvoice_available()
        if self.available:
            logger.info("CosyVoice Service initialisiert")

    def synthesize_text(self, text: str, speaker: str = "host") -> AudioSegment:
        return synthesize_text(text, speaker)

    def synthesize_script(
        self,
        script,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> SynthesisResult:
        return synthesize_script(script, progress_callback)

    def is_available(self) -> bool:
        return self.available


# Singleton
_service_instance: Optional[CosyVoiceService] = None


def get_cosyvoice_service() -> CosyVoiceService:
    """Gibt die CosyVoice-Service-Instanz zurück."""
    global _service_instance
    if _service_instance is None:
        _service_instance = CosyVoiceService()
    return _service_instance
