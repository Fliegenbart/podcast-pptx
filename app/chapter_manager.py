"""
Kapitelmarker-Management für navigierbare Podcasts.
Unterstützt M4B (AAC mit Kapiteln) und MP3 mit ID3v2 CHAP.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import logging
import tempfile
import subprocess

from mutagen.mp4 import MP4
from mutagen.id3 import ID3, CHAP, CTOC, TIT2

from .config import settings

logger = logging.getLogger(__name__)


@dataclass
class ChapterMarker:
    """Ein einzelner Kapitelmarker."""
    title: str
    start_ms: int
    end_ms: Optional[int] = None
    slide_number: Optional[int] = None
    section_name: Optional[str] = None

    def to_ffmpeg_format(self) -> str:
        """Formatiert für FFmpeg Metadaten."""
        start_sec = self.start_ms / 1000
        return f"[CHAPTER]\nTIMEBASE=1/1000\nSTART={self.start_ms}\nEND={self.end_ms or self.start_ms + 1000}\ntitle={self.title}\n"


@dataclass
class ChapterList:
    """Liste aller Kapitel eines Podcasts."""
    chapters: list[ChapterMarker] = field(default_factory=list)
    total_duration_ms: int = 0

    def add_chapter(
        self,
        title: str,
        start_ms: int,
        slide_number: Optional[int] = None,
        section_name: Optional[str] = None
    ):
        """Fügt ein neues Kapitel hinzu."""
        # End-Zeit des vorherigen Kapitels setzen
        if self.chapters:
            self.chapters[-1].end_ms = start_ms

        self.chapters.append(ChapterMarker(
            title=title,
            start_ms=start_ms,
            slide_number=slide_number,
            section_name=section_name
        ))

    def finalize(self, total_duration_ms: int):
        """Finalisiert die Kapitelliste mit Gesamtdauer."""
        self.total_duration_ms = total_duration_ms
        if self.chapters:
            self.chapters[-1].end_ms = total_duration_ms

    def to_ffmpeg_metadata(self) -> str:
        """Erstellt FFmpeg-Metadatendatei für Kapitel."""
        lines = [";FFMETADATA1"]

        for chapter in self.chapters:
            lines.append(chapter.to_ffmpeg_format())

        return "\n".join(lines)

    def get_chapter_at(self, position_ms: int) -> Optional[ChapterMarker]:
        """Findet das Kapitel an einer bestimmten Position."""
        for chapter in self.chapters:
            if chapter.start_ms <= position_ms:
                if chapter.end_ms is None or position_ms < chapter.end_ms:
                    return chapter
        return None


def create_m4b_with_chapters(
    audio_path: Path,
    chapters: ChapterList,
    output_path: Path,
    title: Optional[str] = None
) -> Path:
    """
    Erstellt M4B-Datei mit eingebetteten Kapiteln.

    Args:
        audio_path: Pfad zur Audio-Eingabedatei (WAV oder MP3)
        chapters: Kapitelliste
        output_path: Ausgabepfad für M4B
        title: Optionaler Titel für die Datei

    Returns:
        Pfad zur erstellten M4B-Datei
    """
    logger.info(f"Erstelle M4B mit {len(chapters.chapters)} Kapiteln")

    # Metadatendatei erstellen
    metadata_content = chapters.to_ffmpeg_metadata()

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False
    ) as meta_file:
        meta_file.write(metadata_content)
        meta_path = Path(meta_file.name)

    try:
        # FFmpeg-Befehl
        cmd = [
            "ffmpeg", "-y",
            "-i", str(audio_path),
            "-i", str(meta_path),
            "-map_metadata", "1",
            "-c:a", "aac",
            "-b:a", settings.audio_bitrate,
        ]

        # Titel hinzufügen
        if title:
            cmd.extend(["-metadata", f"title={title}"])

        cmd.append(str(output_path))

        logger.debug(f"FFmpeg Befehl: {' '.join(cmd)}")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            logger.error(f"FFmpeg Fehler: {result.stderr}")
            raise RuntimeError(f"M4B-Erstellung fehlgeschlagen: {result.stderr}")

        logger.info(f"M4B erstellt: {output_path}")
        return output_path

    finally:
        # Temporäre Datei aufräumen
        meta_path.unlink(missing_ok=True)


def create_mp3_with_chapters(
    audio_path: Path,
    chapters: ChapterList,
    output_path: Path,
    title: Optional[str] = None
) -> Path:
    """
    Erstellt MP3-Datei mit ID3v2 CHAP-Frames.

    Args:
        audio_path: Pfad zur Audio-Eingabedatei
        chapters: Kapitelliste
        output_path: Ausgabepfad für MP3
        title: Optionaler Titel

    Returns:
        Pfad zur erstellten MP3-Datei
    """
    logger.info(f"Erstelle MP3 mit {len(chapters.chapters)} Kapiteln")

    # Erst zu MP3 konvertieren falls nötig
    if audio_path.suffix.lower() != ".mp3":
        cmd = [
            "ffmpeg", "-y",
            "-i", str(audio_path),
            "-c:a", "libmp3lame",
            "-b:a", settings.audio_bitrate,
            str(output_path)
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"MP3-Konvertierung fehlgeschlagen: {result.stderr}")
    else:
        # Kopieren
        import shutil
        shutil.copy(audio_path, output_path)

    # ID3-Tags hinzufügen
    try:
        audio = ID3(str(output_path))
    except Exception:
        audio = ID3()

    # Titel setzen
    if title:
        audio.add(TIT2(encoding=3, text=title))

    # Kapitel hinzufügen
    chapter_ids = []
    for i, chapter in enumerate(chapters.chapters):
        chapter_id = f"chp{i}"
        chapter_ids.append(chapter_id)

        # CHAP-Frame erstellen
        audio.add(CHAP(
            element_id=chapter_id,
            start_time=chapter.start_ms,
            end_time=chapter.end_ms or chapters.total_duration_ms,
            sub_frames=[TIT2(encoding=3, text=chapter.title)]
        ))

    # Table of Contents
    if chapter_ids:
        audio.add(CTOC(
            element_id="toc",
            flags=3,  # top-level, ordered
            child_element_ids=chapter_ids,
            sub_frames=[TIT2(encoding=3, text="Inhaltsverzeichnis")]
        ))

    audio.save(str(output_path))

    logger.info(f"MP3 mit Kapiteln erstellt: {output_path}")
    return output_path


def embed_chapters(
    audio_path: Path,
    chapters: ChapterList,
    output_path: Optional[Path] = None,
    title: Optional[str] = None
) -> Path:
    """
    Bettet Kapitel in Audio-Datei ein.
    Wählt automatisch M4B oder MP3 basierend auf Konfiguration.

    Args:
        audio_path: Eingabe-Audiodatei
        chapters: Kapitelliste
        output_path: Optionaler Ausgabepfad
        title: Optionaler Titel

    Returns:
        Pfad zur Ausgabedatei
    """
    if output_path is None:
        suffix = ".m4b" if settings.output_format == "m4b" else ".mp3"
        output_path = audio_path.with_suffix(suffix)

    if settings.output_format == "m4b":
        return create_m4b_with_chapters(audio_path, chapters, output_path, title)
    else:
        return create_mp3_with_chapters(audio_path, chapters, output_path, title)


class ChapterTracker:
    """
    Hilfklasse zum Tracken von Kapiteln während der TTS-Synthese.
    """

    def __init__(self):
        self.chapters = ChapterList()
        self.current_position_ms = 0
        self._intro_added = False

    def start_intro(self):
        """Markiert den Start der Einleitung."""
        if not self._intro_added:
            self.chapters.add_chapter(
                title="Einleitung",
                start_ms=0
            )
            self._intro_added = True

    def start_slide(self, slide_number: int, title: Optional[str] = None):
        """Markiert den Start einer neuen Folie."""
        chapter_title = f"Folie {slide_number}"
        if title:
            chapter_title = f"Folie {slide_number}: {title}"

        self.chapters.add_chapter(
            title=chapter_title,
            start_ms=self.current_position_ms,
            slide_number=slide_number
        )

    def start_section(self, section_name: str):
        """Markiert den Start eines neuen Abschnitts."""
        self.chapters.add_chapter(
            title=section_name,
            start_ms=self.current_position_ms,
            section_name=section_name
        )

    def add_duration(self, duration_ms: int):
        """Fügt Dauer zum aktuellen Position hinzu."""
        self.current_position_ms += duration_ms

    def start_summary(self):
        """Markiert den Start der Zusammenfassung."""
        self.chapters.add_chapter(
            title="Zusammenfassung",
            start_ms=self.current_position_ms
        )

    def finalize(self) -> ChapterList:
        """Finalisiert und gibt die Kapitelliste zurück."""
        self.chapters.finalize(self.current_position_ms)
        return self.chapters
