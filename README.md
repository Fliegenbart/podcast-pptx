# PPTX-to-Podcast Converter

Verwandelt PowerPoint-Präsentationen in natürliche Podcast-Gespräche zwischen zwei Personen.

## Architektur

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   PPTX      │────▶│   Ollama    │────▶│   XTTS v2   │────▶│   FFmpeg    │
│  Extraktor  │     │   Dialog    │     │   2 Stimmen │     │   Assembly  │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
```

## Voraussetzungen

- Python 3.11+
- Ollama mit llama3.1:8b
- NVIDIA GPU mit CUDA (für XTTS)
- FFmpeg

## Setup auf Hetzner GEX44

```bash
# Ollama installieren
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.1:8b

# Python-Umgebung
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# XTTS-Modell wird beim ersten Start automatisch geladen

# Server starten
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## API-Endpunkte

- `POST /convert` – PPTX hochladen, Job starten
- `GET /status/{job_id}` – Job-Status abfragen
- `GET /download/{job_id}` – Fertigen Podcast herunterladen

## Konfiguration

Siehe `app/config.py` für Anpassungen:
- Stimmen-Samples
- Ollama-Modell
- Audio-Qualität
