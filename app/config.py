"""
Konfiguration für PPTX-to-Podcast Converter.
"""
from pathlib import Path
from typing import Literal
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Anwendungseinstellungen."""

    # Pfade
    base_dir: Path = Path(__file__).parent.parent
    upload_dir: Path = base_dir / "uploads"
    output_dir: Path = base_dir / "output"
    samples_dir: Path = base_dir / "samples"
    assets_dir: Path = base_dir / "assets"

    # Ollama
    ollama_model: str = "qwen2.5:14b"
    ollama_host: str = "http://localhost:11434"

    # Podcast-Hosts
    host_name: str = "Alex"
    cohost_name: str = "Sam"

    # Accessibility - Bildbeschreibung
    vision_model: str = "llava:latest"
    skip_decorative_images: bool = True
    max_image_description_length: int = 200
    image_description_enabled: bool = True

    # Accessibility - Kapitel
    output_format: Literal["mp3", "m4b"] = "m4b"
    chapter_per_slide: bool = True
    chapter_per_section: bool = True

    # Accessibility - Audio-Cues
    enable_audio_cues: bool = True
    cue_volume_db: float = -6.0

    # Accessibility - Struktur-Ansagen
    announce_slide_position: bool = True
    announce_section_changes: bool = True

    # TTS
    tts_backend: Literal["auto", "cosyvoice", "xtts", "edge"] = "auto"
    tts_language: str = "de"
    tts_sample_rate: int = 24000

    # CosyVoice
    cosyvoice_model: str = "iic/CosyVoice-300M-SFT"  # oder CosyVoice-300M, CosyVoice-500M

    # Audio
    audio_bitrate: str = "128k"

    class Config:
        env_prefix = "PPTX_PODCAST_"
        env_file = ".env"


# Singleton
settings = Settings()

# Verzeichnisse erstellen
settings.upload_dir.mkdir(parents=True, exist_ok=True)
settings.output_dir.mkdir(parents=True, exist_ok=True)
settings.samples_dir.mkdir(parents=True, exist_ok=True)
(settings.assets_dir / "audio_cues").mkdir(parents=True, exist_ok=True)
