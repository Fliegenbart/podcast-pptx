"""
Audio-Assembly und -Verarbeitung.
Kombiniert Audio-Segmente und fügt Audio-Cues ein.
"""
from pathlib import Path
from typing import Optional
import logging
import subprocess
import wave
import io
import struct
import tempfile

from .config import settings
from .tts_service import AudioSegment, SynthesisResult

logger = logging.getLogger(__name__)


def check_ffmpeg() -> bool:
    """Prüft ob FFmpeg verfügbar ist."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            text=True
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


def get_audio_duration(audio_path: Path) -> float:
    """Ermittelt die Dauer einer Audio-Datei in Sekunden."""
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(audio_path)
            ],
            capture_output=True,
            text=True
        )
        return float(result.stdout.strip())
    except Exception:
        return 0.0


def load_audio_cue(cue_name: str) -> Optional[bytes]:
    """
    Lädt einen Audio-Cue aus dem Assets-Verzeichnis.

    Args:
        cue_name: Name des Cues (ohne .wav Extension)

    Returns:
        WAV-Bytes oder None wenn nicht gefunden
    """
    cue_path = settings.assets_dir / "audio_cues" / f"{cue_name}.wav"

    if not cue_path.exists():
        logger.debug(f"Audio-Cue nicht gefunden: {cue_path}")
        return None

    try:
        with open(cue_path, "rb") as f:
            return f.read()
    except Exception as e:
        logger.warning(f"Fehler beim Laden von Audio-Cue {cue_name}: {e}")
        return None


def generate_silence(duration_ms: int, sample_rate: int = 24000) -> bytes:
    """
    Generiert Stille als WAV-Bytes.

    Args:
        duration_ms: Dauer in Millisekunden
        sample_rate: Sample-Rate

    Returns:
        WAV-Bytes mit Stille
    """
    num_samples = int(sample_rate * duration_ms / 1000)
    silence_data = struct.pack(f"<{num_samples}h", *([0] * num_samples))

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(silence_data)

    return buffer.getvalue()


def adjust_volume(audio_bytes: bytes, volume_db: float) -> bytes:
    """
    Passt die Lautstärke eines Audio-Segments an.

    Args:
        audio_bytes: WAV-Bytes
        volume_db: Lautstärkeänderung in dB

    Returns:
        Angepasste WAV-Bytes
    """
    if volume_db == 0:
        return audio_bytes

    # Faktor berechnen
    factor = 10 ** (volume_db / 20)

    # WAV parsen
    buffer = io.BytesIO(audio_bytes)
    with wave.open(buffer, "rb") as wav_in:
        params = wav_in.getparams()
        frames = wav_in.readframes(wav_in.getnframes())

    # Samples anpassen
    samples = struct.unpack(f"<{len(frames) // 2}h", frames)
    adjusted = [int(max(-32768, min(32767, s * factor))) for s in samples]
    adjusted_frames = struct.pack(f"<{len(adjusted)}h", *adjusted)

    # Neues WAV erstellen
    out_buffer = io.BytesIO()
    with wave.open(out_buffer, "wb") as wav_out:
        wav_out.setparams(params)
        wav_out.writeframes(adjusted_frames)

    return out_buffer.getvalue()


def concatenate_wav_segments(segments: list[bytes], sample_rate: int = 24000) -> bytes:
    """
    Verbindet mehrere WAV-Segmente zu einer Datei.

    Args:
        segments: Liste von WAV-Bytes
        sample_rate: Erwartete Sample-Rate

    Returns:
        Kombiniertes WAV
    """
    if not segments:
        return generate_silence(100, sample_rate)

    all_frames = []

    for segment in segments:
        if not segment:
            continue

        try:
            buffer = io.BytesIO(segment)
            with wave.open(buffer, "rb") as wav:
                frames = wav.readframes(wav.getnframes())
                all_frames.append(frames)
        except Exception as e:
            logger.warning(f"Fehler beim Lesen eines Segments: {e}")

    if not all_frames:
        return generate_silence(100, sample_rate)

    # Kombinieren
    combined_frames = b"".join(all_frames)

    out_buffer = io.BytesIO()
    with wave.open(out_buffer, "wb") as wav_out:
        wav_out.setnchannels(1)
        wav_out.setsampwidth(2)
        wav_out.setframerate(sample_rate)
        wav_out.writeframes(combined_frames)

    return out_buffer.getvalue()


def assemble_podcast(
    synthesis_result: SynthesisResult,
    audio_cues: Optional[dict[int, str]] = None,
    pause_between_segments_ms: int = 300
) -> bytes:
    """
    Assembliert den finalen Podcast aus Segmenten.

    Args:
        synthesis_result: Ergebnis der TTS-Synthese
        audio_cues: Dict von Segment-Index zu Cue-Name
        pause_between_segments_ms: Pause zwischen Segmenten in ms

    Returns:
        Komplettes WAV als Bytes
    """
    audio_cues = audio_cues or {}
    all_parts = []

    # Pause generieren
    pause = generate_silence(pause_between_segments_ms)

    for i, segment in enumerate(synthesis_result.segments):
        # Audio-Cue einfügen falls vorhanden
        if settings.enable_audio_cues and i in audio_cues:
            cue = load_audio_cue(audio_cues[i])
            if cue:
                # Lautstärke anpassen
                cue = adjust_volume(cue, settings.cue_volume_db)
                all_parts.append(cue)
                all_parts.append(generate_silence(100))  # Kurze Pause nach Cue

        # Segment hinzufügen
        if segment.audio_data:
            all_parts.append(segment.audio_data)

        # Pause zwischen Segmenten
        if i < len(synthesis_result.segments) - 1:
            all_parts.append(pause)

    return concatenate_wav_segments(all_parts)


def wav_to_mp3(wav_data: bytes, output_path: Path) -> Path:
    """
    Konvertiert WAV zu MP3.

    Args:
        wav_data: WAV-Bytes
        output_path: Ausgabepfad

    Returns:
        Pfad zur MP3-Datei
    """
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(wav_data)
        tmp_path = Path(tmp.name)

    try:
        cmd = [
            "ffmpeg", "-y",
            "-i", str(tmp_path),
            "-c:a", "libmp3lame",
            "-b:a", settings.audio_bitrate,
            str(output_path)
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"MP3-Konvertierung fehlgeschlagen: {result.stderr}")

        return output_path

    finally:
        tmp_path.unlink(missing_ok=True)


def wav_to_m4b(
    wav_data: bytes,
    output_path: Path,
    chapters_metadata: Optional[str] = None
) -> Path:
    """
    Konvertiert WAV zu M4B mit optionalen Kapiteln.

    Args:
        wav_data: WAV-Bytes
        output_path: Ausgabepfad
        chapters_metadata: FFmpeg-Metadaten für Kapitel

    Returns:
        Pfad zur M4B-Datei
    """
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(wav_data)
        wav_path = Path(tmp.name)

    meta_path = None

    try:
        cmd = [
            "ffmpeg", "-y",
            "-i", str(wav_path),
        ]

        # Kapitel-Metadaten hinzufügen
        if chapters_metadata:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", delete=False
            ) as meta_tmp:
                meta_tmp.write(chapters_metadata)
                meta_path = Path(meta_tmp.name)

            cmd.extend([
                "-i", str(meta_path),
                "-map_metadata", "1"
            ])

        cmd.extend([
            "-c:a", "aac",
            "-b:a", settings.audio_bitrate,
            str(output_path)
        ])

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"M4B-Konvertierung fehlgeschlagen: {result.stderr}")

        return output_path

    finally:
        wav_path.unlink(missing_ok=True)
        if meta_path:
            meta_path.unlink(missing_ok=True)


def create_final_audio(
    synthesis_result: SynthesisResult,
    output_path: Path,
    chapters_metadata: Optional[str] = None,
    audio_cues: Optional[dict[int, str]] = None
) -> Path:
    """
    Erstellt das finale Audio-File.

    Args:
        synthesis_result: TTS-Ergebnis
        output_path: Ausgabepfad
        chapters_metadata: Kapitel-Metadaten
        audio_cues: Audio-Cue-Zuordnung

    Returns:
        Pfad zur finalen Datei
    """
    # Podcast assemblieren
    wav_data = assemble_podcast(synthesis_result, audio_cues)

    # In Zielformat konvertieren
    if settings.output_format == "m4b":
        return wav_to_m4b(wav_data, output_path.with_suffix(".m4b"), chapters_metadata)
    else:
        return wav_to_mp3(wav_data, output_path.with_suffix(".mp3"))
