# PPTX-to-Podcast Converter

Verwandelt PowerPoint-Präsentationen in natürliche, **barrierefreie** Podcast-Gespräche zwischen zwei Personen.

## Accessibility-Features

- **KI-Bildbeschreibungen**: Charts, Diagramme und Flowcharts werden automatisch für blinde Zuhörer beschrieben
- **Kapitelnavigation**: M4B-Format mit Kapiteln pro Folie zum einfachen Springen
- **Audio-Cues**: Sanfte Töne bei Folienwechseln zur Orientierung
- **Struktur-Ansagen**: Natürliche Orientierung ("Wir sind jetzt bei Folie 3 von 12")

## Architektur

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   PPTX      │────▶│   LLaVA     │────▶│   Ollama    │────▶│   XTTS v2   │
│  Extraktor  │     │   Vision    │     │   Dialog    │     │   2 Stimmen │
└─────────────┘     └─────────────┘     └─────────────┘     └──────┬──────┘
                                                                    │
                                        ┌─────────────┐     ┌──────▼──────┐
                                        │   M4B mit   │◀────│   FFmpeg    │
                                        │   Kapiteln  │     │   Assembly  │
                                        └─────────────┘     └─────────────┘
```

## Voraussetzungen

- Python 3.11+
- Ollama mit llama3.1:8b und llava:13b
- NVIDIA GPU mit CUDA (für XTTS)
- FFmpeg

## Setup

```bash
# Ollama installieren
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.1:8b
ollama pull llava:13b  # Für Bildbeschreibungen

# Python-Umgebung
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Voice-Samples bereitstellen
# Lege 2 WAV-Dateien (6-30 Sekunden) in samples/:
#   - samples/host.wav
#   - samples/cohost.wav

# Server starten
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## API-Endpunkte

| Endpunkt | Methode | Beschreibung |
|----------|---------|--------------|
| `/convert` | POST | PPTX hochladen, Job starten |
| `/status/{job_id}` | GET | Job-Status mit Kapitelinfo |
| `/download/{job_id}` | GET | Fertigen Podcast herunterladen |
| `/chapters/{job_id}` | GET | Nur Kapitelinformationen |
| `/settings` | GET | Aktuelle Accessibility-Einstellungen |
| `/health` | GET | System-Status |

## Beispiel

```bash
# PPTX hochladen
curl -X POST -F "file=@praesentation.pptx" http://localhost:8000/convert

# Response:
# {"job_id": "abc123", "status": "pending", ...}

# Status prüfen
curl http://localhost:8000/status/abc123

# Podcast herunterladen (nach Abschluss)
curl -O http://localhost:8000/download/abc123
```

## Konfiguration

Umgebungsvariablen (oder `.env`):

```bash
# Bildbeschreibung
PPTX_PODCAST_IMAGE_DESCRIPTION_ENABLED=true
PPTX_PODCAST_VISION_MODEL=llava:13b

# Kapitel
PPTX_PODCAST_OUTPUT_FORMAT=m4b  # oder "mp3"
PPTX_PODCAST_CHAPTER_PER_SLIDE=true

# Audio-Cues
PPTX_PODCAST_ENABLE_AUDIO_CUES=true

# Podcast-Hosts
PPTX_PODCAST_HOST_NAME=Alex
PPTX_PODCAST_COHOST_NAME=Sam
```

## Audio-Cues

Lege WAV-Dateien in `assets/audio_cues/`:
- `slide_transition.wav` - Sanfter Chime bei Folienwechsel
- `section_start.wav` - Ton bei neuem Abschnitt

## Projektstruktur

```
app/
├── __init__.py
├── config.py           # Einstellungen
├── main.py             # FastAPI-Endpoints
├── extractor.py        # PPTX-Parsing
├── image_describer.py  # LLaVA-Bildbeschreibung
├── dialog_generator.py # Ollama-Dialogerstellung
├── tts_service.py      # XTTS v2 Integration
├── chapter_manager.py  # M4B-Kapitel
└── audio_assembly.py   # FFmpeg-Verarbeitung
```
