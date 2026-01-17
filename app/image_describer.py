"""
KI-gestützte Bildbeschreibungen für Accessibility.
Verwendet LLaVA über Ollama für lokale Verarbeitung.
"""
import base64
import logging
from enum import Enum
from typing import Optional

import ollama

from .config import settings
from .extractor import PresentationContent, SlideImage

logger = logging.getLogger(__name__)


class ImageType(str, Enum):
    """Klassifizierung von Bildtypen."""
    CHART = "chart"
    FLOWCHART = "flowchart"
    DIAGRAM = "diagram"
    SCREENSHOT = "screenshot"
    PHOTO = "photo"
    ICON = "icon"
    DECORATIVE = "decorative"
    UNKNOWN = "unknown"


# Spezialisierte Prompts für verschiedene Bildtypen
CLASSIFICATION_PROMPT = """Analysiere dieses Bild und klassifiziere es in GENAU EINE dieser Kategorien:
- chart: Datenvisualisierung (Balken-, Linien-, Kreisdiagramm, etc.)
- flowchart: Prozessdiagramm, Ablaufdiagramm, Workflow
- diagram: Schematische Darstellung, Organigramm, Architekturdiagramm
- screenshot: Software-Screenshot, UI-Abbildung
- photo: Fotografie von realen Objekten/Personen
- icon: Kleines Symbol oder Icon
- decorative: Rein dekoratives Element ohne Informationsgehalt

Antworte NUR mit dem Kategorienamen, nichts anderes."""

CHART_PROMPT = """Beschreibe dieses Diagramm für einen blinden Zuhörer auf Deutsch.

Wichtig:
- Nenne den Diagrammtyp (Balken, Linie, Kreis, etc.)
- Beschreibe die Achsen/Kategorien
- Nenne die wichtigsten Datenpunkte oder Trends
- Halte dich kurz (max 2-3 Sätze)

Beginne direkt mit der Beschreibung, ohne Einleitung."""

FLOWCHART_PROMPT = """Beschreibe dieses Prozessdiagramm für einen blinden Zuhörer auf Deutsch.

Wichtig:
- Beschreibe den Ablauf Schritt für Schritt
- Nenne Start- und Endpunkt
- Erwähne Verzweigungen oder Entscheidungen
- Halte dich kurz (max 3-4 Sätze)

Beginne direkt mit der Beschreibung, ohne Einleitung."""

DIAGRAM_PROMPT = """Beschreibe dieses Diagramm für einen blinden Zuhörer auf Deutsch.

Wichtig:
- Beschreibe die Hauptkomponenten
- Erkläre die Beziehungen zwischen den Elementen
- Halte dich kurz (max 2-3 Sätze)

Beginne direkt mit der Beschreibung, ohne Einleitung."""

SCREENSHOT_PROMPT = """Beschreibe diesen Screenshot für einen blinden Zuhörer auf Deutsch.

Wichtig:
- Nenne die gezeigte Anwendung/Website
- Beschreibe die wichtigsten sichtbaren Elemente
- Halte dich kurz (max 2 Sätze)

Beginne direkt mit der Beschreibung, ohne Einleitung."""

GENERIC_PROMPT = """Beschreibe dieses Bild für einen blinden Zuhörer auf Deutsch.

Wichtig:
- Beschreibe was zu sehen ist
- Fokussiere auf informationsrelevante Details
- Halte dich kurz (max 2 Sätze)

Beginne direkt mit der Beschreibung, ohne Einleitung."""


def _encode_image(image_bytes: bytes) -> str:
    """Kodiert Bilddaten als Base64."""
    return base64.b64encode(image_bytes).decode("utf-8")


def _call_vision_model(image_bytes: bytes, prompt: str) -> str:
    """Ruft das Vision-Modell auf."""
    try:
        response = ollama.chat(
            model=settings.vision_model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                    "images": [_encode_image(image_bytes)]
                }
            ],
            options={
                "temperature": 0.3,
                "num_predict": 300
            }
        )
        return response["message"]["content"].strip()
    except Exception as e:
        logger.error(f"Vision-Modell Fehler: {e}")
        raise


def classify_image(image_bytes: bytes) -> ImageType:
    """
    Klassifiziert den Bildtyp.

    Args:
        image_bytes: Rohe Bilddaten

    Returns:
        ImageType-Klassifizierung
    """
    try:
        result = _call_vision_model(image_bytes, CLASSIFICATION_PROMPT)
        result_lower = result.lower().strip()

        # Mapping zu ImageType
        type_map = {
            "chart": ImageType.CHART,
            "flowchart": ImageType.FLOWCHART,
            "diagram": ImageType.DIAGRAM,
            "screenshot": ImageType.SCREENSHOT,
            "photo": ImageType.PHOTO,
            "icon": ImageType.ICON,
            "decorative": ImageType.DECORATIVE,
        }

        for key, img_type in type_map.items():
            if key in result_lower:
                return img_type

        return ImageType.UNKNOWN

    except Exception as e:
        logger.warning(f"Bildklassifizierung fehlgeschlagen: {e}")
        return ImageType.UNKNOWN


def describe_image(image_bytes: bytes, image_type: Optional[ImageType] = None) -> str:
    """
    Erstellt eine barrierefreie Beschreibung für ein Bild.

    Args:
        image_bytes: Rohe Bilddaten
        image_type: Optionaler Bildtyp (wird automatisch ermittelt wenn nicht angegeben)

    Returns:
        Deutsche Beschreibung für Screen Reader
    """
    # Bildtyp ermitteln falls nicht angegeben
    if image_type is None:
        image_type = classify_image(image_bytes)

    # Dekorative Bilder überspringen
    if image_type == ImageType.DECORATIVE:
        return ""

    # Prompt basierend auf Typ auswählen
    prompt_map = {
        ImageType.CHART: CHART_PROMPT,
        ImageType.FLOWCHART: FLOWCHART_PROMPT,
        ImageType.DIAGRAM: DIAGRAM_PROMPT,
        ImageType.SCREENSHOT: SCREENSHOT_PROMPT,
    }

    prompt = prompt_map.get(image_type, GENERIC_PROMPT)

    try:
        description = _call_vision_model(image_bytes, prompt)

        # Länge begrenzen
        if len(description) > settings.max_image_description_length:
            # Am Satzende abschneiden
            cutoff = description[:settings.max_image_description_length].rfind(".")
            if cutoff > 50:
                description = description[: cutoff + 1]
            else:
                description = description[:settings.max_image_description_length] + "..."

        return description

    except Exception as e:
        logger.error(f"Bildbeschreibung fehlgeschlagen: {e}")
        return ""


def describe_images_in_content(content: PresentationContent) -> PresentationContent:
    """
    Fügt Bildbeschreibungen zu allen Bildern in der Präsentation hinzu.

    Args:
        content: Extrahierte Präsentationsinhalte

    Returns:
        PresentationContent mit Bildbeschreibungen
    """
    if not settings.image_description_enabled:
        logger.info("Bildbeschreibung ist deaktiviert")
        return content

    total_images = sum(len(slide.images) for slide in content.slides)
    logger.info(f"Beschreibe {total_images} Bilder...")

    described_count = 0
    skipped_count = 0

    for slide in content.slides:
        for image in slide.images:
            # Alt-Text prüfen
            if image.alt_text and len(image.alt_text) > 20:
                # Vorhandener Alt-Text ist wahrscheinlich ausreichend
                image.description = image.alt_text
                described_count += 1
                continue

            # Kleine Icons überspringen
            if image.width and image.height:
                if image.width < 100 and image.height < 100:
                    skipped_count += 1
                    continue

            # Bildbeschreibung generieren
            try:
                image_type = classify_image(image.image_bytes)

                if image_type == ImageType.DECORATIVE:
                    skipped_count += 1
                    continue

                if settings.skip_decorative_images and image_type == ImageType.ICON:
                    skipped_count += 1
                    continue

                description = describe_image(image.image_bytes, image_type)
                if description:
                    image.description = description
                    described_count += 1
                    logger.debug(
                        f"Folie {slide.slide_number}: {image_type.value} - {description[:50]}..."
                    )
                else:
                    skipped_count += 1

            except Exception as e:
                logger.warning(
                    f"Fehler bei Bild auf Folie {slide.slide_number}: {e}"
                )
                skipped_count += 1

    logger.info(
        f"Bildbeschreibung abgeschlossen: {described_count} beschrieben, "
        f"{skipped_count} übersprungen"
    )

    return content
